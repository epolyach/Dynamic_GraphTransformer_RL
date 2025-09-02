#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from typing import Dict, Any, List, Callable

# Add project root to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config
from src.generator.generator import create_data_generator

# Solvers
from src.benchmarking.solvers.cpu import exact_dp
from src.benchmarking.solvers.cpu import exact_ortools_vrp_fixed as ortools_greedy
from src.benchmarking.solvers.cpu import ortools_gls


Solver = Callable[[Dict[str, Any]], Any]


def _ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _run_with_timeout(func: Callable, args: tuple, timeout: float):
    """Run a function with a hard timeout using threading. Returns (result, elapsed, status).
    status in {"success", "timeout", "failed"}.
    """
    import threading

    result_holder = {"result": None, "exc": None}

    def target():
        try:
            result_holder["result"] = func(*args)
        except Exception as e:
            result_holder["exc"] = e

    t = threading.Thread(target=target, daemon=True)
    start = time.time()
    t.start()
    t.join(timeout=timeout)
    elapsed = time.time() - start

    if t.is_alive():
        return None, elapsed, "timeout"
    if result_holder["exc"] is not None:
        return None, elapsed, "failed"
    return result_holder["result"], elapsed, "success"


def main():
    parser = argparse.ArgumentParser(description="CPU benchmark (exact+heuristic) using unified generator")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--n-start", type=int, default=5)
    parser.add_argument("--n-end", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--csv", type=str, default="benchmark_cpu/results/csv/cpu_benchmark.csv")
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    # Prepare CSV
    _ensure_parent_dir(args.csv)
    new_file = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["n_customers", "solver", "instance_id", "status", "time", "cpc"])  # header expected by plot

        for N in range(int(args.n_start), int(args.n_end) + 1):
            # Prepare per-N config and generator
            cfgN = dict(base_cfg)
            cfgN["problem"] = dict(base_cfg["problem"])  # deep copy minimal
            cfgN["problem"]["num_customers"] = int(N)
            genN = create_data_generator(cfgN)

            for instance_id in range(1, int(args.instances) + 1):
                seed = int(args.seed) + int(N) * 1000 + instance_id
                # Single instance generation (batch_size=1)
                batch = genN(batch_size=1, epoch=N, seed=seed)
                instance = batch[0]
                n_customers = len(instance["coords"]) - 1

                # Run solvers
                jobs = [
                    ("exact_dp", exact_dp.solve),
                    ("ortools_greedy", ortools_greedy.solve),
                    ("ortools_gls", ortools_gls.solve),
                ]

                for solver_name, solver_fn in jobs:
                    # Skip exact_dp for N>8 (brute force too expensive)
                    if solver_name == "exact_dp" and n_customers > 8:
                        continue

                    # Execute with timeout (no fallbacks)
                    result, elapsed, status = _run_with_timeout(
                        solver_fn, (instance, args.time_limit, False), args.time_limit
                    )

                    if status == "success":
                        cpc = float(result.cost) / max(1, n_customers)
                        writer.writerow([n_customers, solver_name, instance_id, "success", f"{elapsed:.6f}", f"{cpc:.6f}"])
                    elif status == "timeout":
                        writer.writerow([n_customers, solver_name, instance_id, "timeout", f"{elapsed:.6f}", ""])
                    else:  # failed
                        writer.writerow([n_customers, solver_name, instance_id, "failed", f"{elapsed:.6f}", ""])

    print(f"✅ CPU benchmark complete. CSV → {args.csv}")


if __name__ == "__main__":
    main()

