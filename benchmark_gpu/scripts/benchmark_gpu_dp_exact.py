#!/usr/bin/env python3
"""
GPU DP-Exact Benchmark for N=10, Capacity=20, 10,000 instances
Uses the GPU-based exact CVRP solver with batched dynamic programming.
"""

import sys
import os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.exact_gpu_dp import solve_batch as gpu_exact_solve_batch
from tabulate import tabulate

def benchmark_gpu_exact_dp(n_customers, capacity, num_instances, batch_size=100):
    """
    Benchmark GPU DP-exact solver with specified parameters.
    """
    print("=" * 70)
    print("GPU DP-Exact CVRP Benchmark")
    print("=" * 70)
    print(f"N = {n_customers} customers")
    print(f"Capacity = {capacity}")
    print(f"Instances = {num_instances:,}")
    print(f"Batch size = {batch_size}")
    print("-" * 70)
    
    if n_customers > 16:
        raise ValueError(f"GPU DP-exact solver supports N≤16, got N={n_customers}")
    
    start_time = time.time()
    all_cpcs = []
    all_costs = []
    all_solve_times = []
    
    # Process in batches
    num_batches = (num_instances + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_instances)
        current_batch_size = batch_end - batch_start
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}: instances {batch_start}-{batch_end-1}")
        
        # Generate batch of instances
        gen_start = time.time()
        instances = []
        for i in range(batch_start, batch_end):
            instance = _generate_instance(
                num_customers=n_customers,
                capacity=capacity,
                coord_range=100,
                demand_range=[1, 10],
                seed=42000 + n_customers * 1000 + i
            )
            instances.append(instance)
        gen_time = time.time() - gen_start
        
        # Solve batch with GPU DP-exact
        solve_start = time.time()
        solutions = gpu_exact_solve_batch(instances, verbose=True)
        solve_time = time.time() - solve_start
        
        # Extract results
        batch_costs = [sol.cost for sol in solutions]
        batch_cpcs = [cost / n_customers for cost in batch_costs]
        batch_solve_times = [sol.solve_time for sol in solutions]
        
        all_costs.extend(batch_costs)
        all_cpcs.extend(batch_cpcs)
        all_solve_times.extend(batch_solve_times)
        
        # Batch statistics
        mean_cpc = np.mean(batch_cpcs)
        print(f"  Generation: {gen_time:.2f}s")
        print(f"  GPU solving: {solve_time:.2f}s ({solve_time/current_batch_size:.3f}s per instance)")
        print(f"  Batch mean CPC: {mean_cpc:.6f}")
        
        # Overall progress
        if len(all_cpcs) % 1000 == 0 or len(all_cpcs) == num_instances:
            overall_mean = np.mean(all_cpcs)
            print(f"  Overall progress: {len(all_cpcs)}/{num_instances}, Mean CPC: {overall_mean:.6f}")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    cpcs = np.array(all_cpcs)
    costs = np.array(all_costs)
    
    mean_cpc = cpcs.mean()
    std_cpc = cpcs.std()
    sem = std_cpc / np.sqrt(len(cpcs))
    sem_pct = (2 * sem / mean_cpc) * 100
    
    min_cost = costs.min()
    max_cost = costs.max()
    mean_cost = costs.mean()
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS - GPU DP-Exact Solver")
    print("=" * 70)
    
    # Results table
    table_data = [[
        n_customers,
        capacity,
        f"{len(all_cpcs):,}",
        f"{mean_cpc:.6f}",
        f"{std_cpc:.6f}",
        f"{sem:.6f}",
        f"{sem_pct:.4f}%"
    ]]
    
    headers = ["N", "Capacity", "Instances", "Mean CPC", "Std CPC", "SEM", "2×SEM/Mean(%)"]
    print(tabulate(table_data, headers=headers, tablefmt="pipe", floatfmt=".6f"))
    
    # Additional statistics
    print(f"\nCost Statistics:")
    print(f"  Mean cost: {mean_cost:.6f}")
    print(f"  Min cost:  {min_cost:.6f}")
    print(f"  Max cost:  {max_cost:.6f}")
    print(f"  Std cost:  {costs.std():.6f}")
    
    print(f"\nTiming Statistics:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Time per instance: {total_time/len(all_cpcs):.3f}s")
    print(f"  Average solve time: {np.mean(all_solve_times):.3f}s")
    
    # LaTeX table format
    print(f"\nLaTeX Table Format:")
    print("-" * 40)
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{GPU DP-Exact CVRP Performance ({len(all_cpcs):,} instances)}}
\\label{{tab:gpu-dp-exact}}
\\begin{{tabular}}{{@{{}}c c c S[table-format=1.6] S[table-format=1.6] S[table-format=1.6] c@{{}}}}
\\toprule
\\textbf{{N}} & \\textbf{{Capacity}} & \\textbf{{Instances}} & {{\\textbf{{Mean CPC}}}} & {{\\textbf{{Std CPC}}}} & {{\\textbf{{SEM}}}} & \\textbf{{2×SEM/Mean(\\%)}} \\\\
\\midrule
{n_customers} & {capacity} & {len(all_cpcs):,} & {mean_cpc:.6f} & {std_cpc:.6f} & {sem:.6f} & {sem_pct:.2f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    print(latex_table)
    
    # Save results
    import json
    results = {
        'method': 'GPU_DP_Exact',
        'n_customers': n_customers,
        'capacity': capacity,
        'instances': len(all_cpcs),
        'mean_cpc': float(mean_cpc),
        'std_cpc': float(std_cpc),
        'sem': float(sem),
        'sem_pct': float(sem_pct),
        'mean_cost': float(mean_cost),
        'min_cost': float(min_cost),
        'max_cost': float(max_cost),
        'total_time': total_time,
        'time_per_instance': total_time / len(all_cpcs),
        'all_costs': [float(c) for c in all_costs[:100]],  # Save first 100 for analysis
        'all_cpcs': [float(c) for c in all_cpcs[:100]]
    }
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f'gpu_dp_exact_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU DP-Exact CVRP Benchmark')
    parser.add_argument('--n-customers', type=int, default=10,
                        help='Number of customers (default: 10, max: 16)')
    parser.add_argument('--capacity', type=int, default=20,
                        help='Vehicle capacity (default: 20)')
    parser.add_argument('--instances', type=int, default=10000,
                        help='Number of instances (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for GPU processing (default: 100)')
    
    args = parser.parse_args()
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: No GPU detected! DP-exact solver requires GPU.")
        exit(1)
    
    benchmark_gpu_exact_dp(
        n_customers=args.n_customers,
        capacity=args.capacity,
        num_instances=args.instances,
        batch_size=args.batch_size
    )
