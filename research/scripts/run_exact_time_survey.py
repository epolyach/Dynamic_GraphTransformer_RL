#!/usr/bin/env python3
import os
import sys
import time
import csv
import math
import argparse
from statistics import mean, stdev

import numpy as np

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Add repository root to sys.path so both `research.*` and `src.*` imports resolve
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.exact_solver import ExactCVRPSolver
from src.data.enhanced_generator import EnhancedCVRPGenerator, InstanceType


def main():
    parser = argparse.ArgumentParser(description='Exact CVRP solver timing survey')
    parser.add_argument('--instances', type=int, default=100, help='Number of instances per N (default: 100)')
    parser.add_argument('--stop-seconds', type=float, default=60.0, help='Stop if time for an N exceeds this (default: 60s)')
    parser.add_argument('--n-start', type=int, default=5, help='Start N (default: 5)')
    parser.add_argument('--n-end', type=int, default=50, help='End N inclusive (default: 50)')
    parser.add_argument('--out', type=str, default=os.path.join('results', 'production', 'analysis', 'exact_time_survey.csv'), help='Output CSV path')
    args = parser.parse_args()

    # Output CSV path
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = args.out

    # Fixed problem settings
    capacity = 30
    demand_range = [1, 10]
    coord_range = 100  # integer grid scaled to [0,1] in generator

    # Initialize generator (config is unused but kept for consistency)
    gen = EnhancedCVRPGenerator(config={})

    # Exact solver: prefer DP-only to avoid external deps; enforce CPU-friendly verbosity
    solver = ExactCVRPSolver(time_limit=args.stop_seconds, enable_or_tools=False, enable_gurobi=False, verbose=False)

    # Survey settings
    N_start, N_end = args.n_start, args.n_end
    per_N_instances = args.instances
    stop_threshold_s = args.stop_seconds

    rows = []

    for N in range(N_start, N_end + 1):
        print(f"\nN={N}: running up to {per_N_instances} instances (stop if > {stop_threshold_s:.0f}s)")
        t0 = time.time()
        costs_per_customer = []
        solved = 0

        iter_range = range(per_N_instances)
        if tqdm is not None:
            iter_range = tqdm(iter_range, desc=f"N={N}", leave=False)

        for i in iter_range:
            # Early termination within-N if we already exceed threshold
            if time.time() - t0 > stop_threshold_s:
                if tqdm is not None:
                    iter_range.close()
                print(f"Time cap reached for N={N} at instance {i}. Elapsed {time.time() - t0:.2f}s")
                break

            # Seed per instance for reproducibility
            seed = 4242 + N * 1000 + i
            instance = gen.generate_instance(
                num_customers=N,
                capacity=capacity,
                coord_range=coord_range,
                demand_range=demand_range,
                seed=seed,
                instance_type=InstanceType.RANDOM,
                apply_augmentation=False,
            )

            try:
                sol = solver.solve(instance)
                if math.isfinite(sol.cost):
                    costs_per_customer.append(sol.cost / max(1, N))
                    solved += 1
            except Exception:
                # Solver failed for this instance (e.g., size too large for DP). Skip.
                continue

        elapsed = time.time() - t0

        if tqdm is not None:
            # Ensure the bar is closed for this N before printing summary
            try:
                iter_range.close()
            except Exception:
                pass

        # Compute average and std of cost/customer across solved instances (if any)
        if len(costs_per_customer) >= 1:
            avg_cpc = float(mean(costs_per_customer))
        else:
            avg_cpc = float('nan')
        if len(costs_per_customer) >= 2:
            std_cpc = float(stdev(costs_per_customer))
        else:
            std_cpc = float('nan')

        rows.append({
            'N': N,
            'time_s': round(elapsed, 6),
            'cost_per_customer': avg_cpc,
            'std': std_cpc,
        })

        # Write incremental results so progress is saved even if we terminate early
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['N', 'time_s', 'cost_per_customer', 'std'])
            writer.writeheader()
            writer.writerows(rows)

        # Stop once a full-N (or partial due to time) exceeds threshold
        if elapsed > stop_threshold_s:
            print(f"Stopping after N={N} (elapsed {elapsed:.2f}s > {stop_threshold_s:.0f}s).")
            break

        print(f"Done N={N}: elapsed={elapsed:.2f}s, solved={len(costs_per_customer)} / {per_N_instances}, avg_cpc={avg_cpc if avg_cpc==avg_cpc else float('nan'):.4f}")

    print(f"\nSurvey complete. Wrote {len(rows)} rows to {out_csv}")


if __name__ == '__main__':
    main()

