#!/usr/bin/env python3
"""
GPU Heuristic Benchmark using GPU Improved Heuristic
Tests GPU-accelerated heuristic for multiple problem sizes.
"""

import numpy as np
import time
import sys
import os
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.heuristic_gpu_improved import solve_batch

def run_benchmark_config(n_customers, capacity, num_instances, batch_size=100):
    """Run benchmark for a specific configuration."""
    print(f"\n{'='*70}")
    print(f"Configuration: N={n_customers}, Capacity={capacity}")
    print(f"Instances: {num_instances}, Batch size: {batch_size}")
    print(f"{'='*70}")
    
    all_costs = []
    total_start = time.time()
    
    num_batches = (num_instances + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_instances)
        current_batch_size = end_idx - start_idx
        
        if batch_idx % 10 == 0:  # Print every 10th batch for brevity
            print(f"  Processing batch {batch_idx + 1}/{num_batches}...")
        
        # Generate batch
        instances = []
        for i in range(start_idx, end_idx):
            instance = _generate_instance(
                num_customers=n_customers,
                capacity=capacity,
                coord_range=100,
                demand_range=[1, 10],
                seed=42000 + n_customers * 1000 + i
            )
            instances.append(instance)
        
        # Solve batch
        solutions = solve_batch(instances, max_iterations=100, verbose=False)
        batch_costs = [sol.cost for sol in solutions]
        all_costs.extend(batch_costs)
    
    total_time = time.time() - total_start
    
    # Calculate statistics
    cpcs = np.array(all_costs) / n_customers
    mean_cpc = cpcs.mean()
    std_cpc = cpcs.std()
    sem = std_cpc / np.sqrt(len(cpcs))
    sem_pct = (2 * sem / mean_cpc) * 100
    
    # Calculate log-normal statistics
    log_cpcs = np.log(cpcs)
    gm = np.exp(log_cpcs.mean())
    gsd = np.exp(log_cpcs.std(ddof=1))
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s ({total_time/num_instances:.3f}s per instance)")
    print(f"  Mean CPC: {mean_cpc:.6f}")
    print(f"  GM: {gm:.6f}, GSD: {gsd:.6f}")
    
    results = {
        'n_customers': n_customers,
        'capacity': capacity,
        'instances': num_instances,
        'mean_cpc': float(mean_cpc),
        'std_cpc': float(std_cpc),
        'sem': float(sem),
        'sem_pct': float(sem_pct),
        'gm': float(gm),
        'gsd': float(gsd),
        'total_time': total_time,
        'time_per_instance': total_time / num_instances,
        'all_cpcs': [float(c) for c in cpcs]
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='GPU Heuristic Multi-Config Benchmark')
    parser.add_argument('--instances', type=int, default=10000,
                        help='Number of instances per configuration (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for GPU processing (default: 100)')
    parser.add_argument('--configs', type=str, default='all',
                        help='Configurations to run: "all" or comma-separated N values')
    
    args = parser.parse_args()
    
    import torch
    if torch.cuda.is_available():
        print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Define configurations (N, Capacity)
    all_configs = {
        10: 20,
        20: 30,
        50: 40,
        100: 50
    }
    
    if args.configs == 'all':
        configs = all_configs
    else:
        n_values = [int(x.strip()) for x in args.configs.split(',')]
        configs = {n: all_configs[n] for n in n_values if n in all_configs}
    
    print(f"\n{'='*70}")
    print(f"GPU Improved Heuristic CVRP Benchmark")
    print(f"Configurations: {list(configs.keys())}")
    print(f"Instances per config: {args.instances}")
    print(f"{'='*70}")
    
    all_results = {}
    
    for n_customers, capacity in configs.items():
        results = run_benchmark_config(
            n_customers=n_customers,
            capacity=capacity,
            num_instances=args.instances,
            batch_size=args.batch_size
        )
        all_results[f"N{n_customers}_C{capacity}"] = results
    
    # Save all results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f'gpu_heuristic_multi_results_{timestamp}.json'
    
    final_output = {
        'method': 'GPU_Heuristic_Improved',
        'timestamp': timestamp,
        'instances_per_config': args.instances,
        'batch_size': args.batch_size,
        'results': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"All results saved to: {output_file}")
    print(f"{'='*70}")
    
    return output_file

if __name__ == "__main__":
    main()
