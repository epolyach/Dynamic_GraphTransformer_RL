#!/usr/bin/env python3
"""
GPU Heuristic Benchmark using GPU Improved Heuristic
Tests GPU-accelerated heuristic for N=10, C=20, 10k instances.
"""

import numpy as np
import time
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.heuristic_gpu_improved import solve_batch

def run_benchmark(n_customers=10, capacity=20, num_instances=10000, batch_size=100):
    """Run benchmark for GPU improved heuristic."""
    print(f"\n{'='*70}")
    print(f"GPU Improved Heuristic CVRP Benchmark")
    print(f"N={n_customers}, Capacity={capacity}, Instances={num_instances}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*70}")
    
    all_costs = []
    total_start = time.time()
    
    num_batches = (num_instances + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_instances)
        current_batch_size = end_idx - start_idx
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}: instances {start_idx}-{end_idx-1}")
        
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
        solve_start = time.time()
        solutions = solve_batch(instances, max_iterations=100, verbose=False)
        solve_time = time.time() - solve_start
        
        batch_costs = [sol.cost for sol in solutions]
        all_costs.extend(batch_costs)
        
        batch_cpcs = np.array(batch_costs) / n_customers
        overall_cpcs = np.array(all_costs) / n_customers
        
        print(f"  Solving time: {solve_time:.2f}s ({solve_time/current_batch_size:.3f}s per instance)")
        print(f"  Batch mean CPC: {batch_cpcs.mean():.6f}")
        print(f"  Overall mean CPC: {overall_cpcs.mean():.6f}")
    
    total_time = time.time() - total_start
    
    # Calculate statistics
    cpcs = np.array(all_costs) / n_customers
    mean_cpc = cpcs.mean()
    std_cpc = cpcs.std()
    sem = std_cpc / np.sqrt(len(cpcs))
    sem_pct = (2 * sem / mean_cpc) * 100
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.2f}s ({total_time/num_instances:.3f}s per instance)")
    print(f"Mean CPC: {mean_cpc:.6f}")
    print(f"Std CPC: {std_cpc:.6f}")
    print(f"SEM: {sem:.6f}")
    print(f"2×SEM/Mean: {sem_pct:.4f}%")
    
    # Save results
    results = {
        'method': 'GPU_Heuristic_Improved',
        'n_customers': n_customers,
        'capacity': capacity,
        'instances': num_instances,
        'mean_cpc': float(mean_cpc),
        'std_cpc': float(std_cpc),
        'sem': float(sem),
        'sem_pct': float(sem_pct),
        'total_time': total_time,
        'time_per_instance': total_time / num_instances,
        'all_cpcs': [float(c) for c in cpcs]
    }
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f'gpu_heuristic_improved_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    run_benchmark(n_customers=10, capacity=20, num_instances=10000, batch_size=100)
