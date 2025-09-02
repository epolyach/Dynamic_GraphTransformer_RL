#!/usr/bin/env python3
"""
Benchmark script to evaluate optimal and heuristic CPC solvers on N=7 CVRP instances.
Evaluates:
1. Exact DP (optimal)
2. Exact OR-Tools (optimal/near-optimal)
3. Nearest Neighbor heuristic (suboptimal)
4. STS_CPC heuristic (suboptimal)
"""

import argparse
import csv
import os
import time
import numpy as np
from typing import Dict, Any, List, Callable, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.generator.generator import create_data_generator

# Import solvers
from src.benchmarking.solvers.cpu import exact_dp
from src.benchmarking.solvers.cpu import exact_ortools_vrp_fixed
from src.benchmarking.solvers.cpu import heuristic_nearest_neighbor
from src.benchmarking.solvers.cpu import heuristic_sts_cpc


def _ensure_parent_dir(path: str):
    """Ensure parent directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _run_with_timeout(func: Callable, args: tuple, timeout: float) -> Tuple[Any, float, str]:
    """
    Run a function with a hard timeout using threading.
    Returns (result, elapsed, status) where status in {'success', 'timeout', 'failed'}.
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
        print(f"  Error: {result_holder['exc']}")
        return None, elapsed, "failed"
    return result_holder["result"], elapsed, "success"


def generate_instances(n_customers: int, n_instances: int, seed: int, config: dict) -> List[Dict[str, Any]]:
    """Generate CVRP instances with fixed parameters."""
    print(f"\nGenerating {n_instances} instances with N={n_customers} customers...")
    
    # Configure generator for N customers
    cfg = dict(config)
    cfg["problem"] = dict(config.get("problem", {}))
    cfg["problem"]["num_customers"] = n_customers
    
    # Create generator
    generator = create_data_generator(cfg)
    
    # Generate all instances at once with consistent seed
    np.random.seed(seed)
    instances = []
    
    # Generate in batches for efficiency
    batch_size = 100
    n_batches = (n_instances + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_instances)
        current_batch_size = batch_end - batch_start
        
        # Generate batch
        batch_seed = seed + batch_idx * 1000
        batch = generator(batch_size=current_batch_size, epoch=0, seed=batch_seed)
        instances.extend(batch)
    
    print(f"  Generated {len(instances)} instances")
    return instances[:n_instances]  # Ensure exactly n_instances


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPC solvers on N=7 CVRP instances"
    )
    parser.add_argument("--config", type=str, default="configs/small.yaml",
                       help="Configuration file")
    parser.add_argument("--n-customers", type=int, default=7,
                       help="Number of customers (default: 7)")
    parser.add_argument("--n-instances", type=int, default=1000,
                       help="Number of instances to evaluate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--timeout-exact", type=float, default=60.0,
                       help="Timeout for exact solvers in seconds (default: 60)")
    parser.add_argument("--timeout-heuristic", type=float, default=1.0,
                       help="Timeout for heuristic solvers in seconds (default: 1)")
    parser.add_argument("--csv", type=str, 
                       default="benchmark_cpu/results/csv/n7_solver_comparison.csv",
                       help="Output CSV file path")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output from solvers")
    args = parser.parse_args()
    
    # Load configuration
    base_cfg = load_config(args.config)
    
    # Generate instances (same for all solvers)
    instances = generate_instances(
        args.n_customers, args.n_instances, args.seed, base_cfg
    )
    
    # Prepare CSV file
    _ensure_parent_dir(args.csv)
    
    # Define solvers to evaluate
    solvers = [
        ("exact_dp", exact_dp.solve, args.timeout_exact, "Exact DP (Optimal)"),
        ("exact_ortools", exact_ortools_vrp_fixed.solve, args.timeout_exact, "Exact OR-Tools"),
        ("heuristic_nn", heuristic_nearest_neighbor.solve, args.timeout_heuristic, "Nearest Neighbor"),
        ("heuristic_sts_cpc", heuristic_sts_cpc.solve, args.timeout_heuristic, "STS-CPC"),
    ]
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKING CPC SOLVERS")
    print(f"{'='*70}")
    print(f"Instances: {args.n_instances} with N={args.n_customers} customers")
    print(f"Seed: {args.seed}")
    print(f"Timeouts: Exact={args.timeout_exact}s, Heuristic={args.timeout_heuristic}s")
    print(f"Output: {args.csv}")
    print(f"{'='*70}\n")
    
    # Open CSV file for writing
    with open(args.csv, "w", newline="") as csvfile:
        fieldnames = [
            "instance_id", "n_customers",
            "exact_dp_cpc", "exact_dp_time", "exact_dp_status",
            "exact_ortools_cpc", "exact_ortools_time", "exact_ortools_status",
            "heuristic_nn_cpc", "heuristic_nn_time", "heuristic_nn_status",
            "heuristic_sts_cpc_cpc", "heuristic_sts_cpc_time", "heuristic_sts_cpc_status",
            "nn_gap_percent", "sts_gap_percent"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each instance
        for instance_id, instance in enumerate(instances, 1):
            n_customers = len(instance["coords"]) - 1
            
            if instance_id % 100 == 0 or instance_id == 1:
                print(f"Processing instance {instance_id}/{args.n_instances}...")
            
            row = {
                "instance_id": instance_id,
                "n_customers": n_customers
            }
            
            optimal_cpc = None  # Will be set by first exact solver that succeeds
            
            # Run each solver
            for solver_name, solver_fn, timeout, description in solvers:
                if args.verbose:
                    print(f"  Running {description}...")
                
                # Run solver with timeout
                result, elapsed, status = _run_with_timeout(
                    solver_fn, (instance, timeout, args.verbose), timeout
                )
                
                # Store results
                if status == "success" and result is not None:
                    cpc = result.cost / max(1, n_customers)
                    row[f"{solver_name}_cpc"] = f"{cpc:.6f}"
                    row[f"{solver_name}_time"] = f"{elapsed:.6f}"
                    row[f"{solver_name}_status"] = "success"
                    
                    # Set optimal CPC from first successful exact solver
                    if optimal_cpc is None and solver_name in ["exact_dp", "exact_ortools"]:
                        optimal_cpc = cpc
                else:
                    row[f"{solver_name}_cpc"] = ""
                    row[f"{solver_name}_time"] = f"{elapsed:.6f}"
                    row[f"{solver_name}_status"] = status
            
            # Calculate optimality gaps for heuristics
            if optimal_cpc is not None:
                # Nearest Neighbor gap
                if row.get("heuristic_nn_cpc"):
                    nn_cpc = float(row["heuristic_nn_cpc"])
                    row["nn_gap_percent"] = f"{100 * (nn_cpc - optimal_cpc) / optimal_cpc:.2f}"
                else:
                    row["nn_gap_percent"] = ""
                
                # STS-CPC gap
                if row.get("heuristic_sts_cpc_cpc"):
                    sts_cpc = float(row["heuristic_sts_cpc_cpc"])
                    row["sts_gap_percent"] = f"{100 * (sts_cpc - optimal_cpc) / optimal_cpc:.2f}"
                else:
                    row["sts_gap_percent"] = ""
            else:
                row["nn_gap_percent"] = ""
                row["sts_gap_percent"] = ""
            
            # Write row to CSV
            writer.writerow(row)
            csvfile.flush()  # Ensure data is written immediately
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {args.csv}")
    
    # Generate summary statistics
    print(f"\nGenerating summary statistics...")
    _generate_summary(args.csv)


def _generate_summary(csv_path: str):
    """Generate and print summary statistics from the results."""
    import pandas as pd
    
    # Read results
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total instances: {len(df)}")
    
    # Success rates
    print(f"\nSuccess Rates:")
    for solver in ["exact_dp", "exact_ortools", "heuristic_nn", "heuristic_sts_cpc"]:
        success_rate = (df[f"{solver}_status"] == "success").mean() * 100
        print(f"  {solver:20s}: {success_rate:6.2f}%")
    
    # Average CPC values (for successful runs only)
    print(f"\nAverage CPC (successful runs only):")
    for solver in ["exact_dp", "exact_ortools", "heuristic_nn", "heuristic_sts_cpc"]:
        cpc_col = f"{solver}_cpc"
        # Convert to numeric, handling empty strings
        cpc_values = pd.to_numeric(df[cpc_col], errors='coerce')
        valid_cpc = cpc_values.dropna()
        if len(valid_cpc) > 0:
            mean_cpc = valid_cpc.mean()
            std_cpc = valid_cpc.std()
            print(f"  {solver:20s}: {mean_cpc:8.4f} ± {std_cpc:6.4f} (n={len(valid_cpc)})")
    
    # Average computation times
    print(f"\nAverage Computation Time (seconds):")
    for solver in ["exact_dp", "exact_ortools", "heuristic_nn", "heuristic_sts_cpc"]:
        time_col = f"{solver}_time"
        time_values = pd.to_numeric(df[time_col], errors='coerce')
        valid_times = time_values.dropna()
        if len(valid_times) > 0:
            mean_time = valid_times.mean()
            max_time = valid_times.max()
            print(f"  {solver:20s}: mean={mean_time:8.4f}s, max={max_time:8.4f}s")
    
    # Optimality gaps
    print(f"\nOptimality Gaps (% above optimal):")
    for heuristic, gap_col in [("Nearest Neighbor", "nn_gap_percent"), 
                               ("STS-CPC", "sts_gap_percent")]:
        gap_values = pd.to_numeric(df[gap_col], errors='coerce')
        valid_gaps = gap_values.dropna()
        if len(valid_gaps) > 0:
            mean_gap = valid_gaps.mean()
            std_gap = valid_gaps.std()
            max_gap = valid_gaps.max()
            print(f"  {heuristic:20s}: mean={mean_gap:6.2f}%, std={std_gap:6.2f}%, max={max_gap:6.2f}%")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
