#!/usr/bin/env python3
"""
Debug CPU vs GPU CVRP solver discrepancies
Detailed analysis of where implementations diverge
"""

import numpy as np
import time
import torch
from solvers.exact_dp import solve as cpu_solve
from solvers.exact_gpu_dp import solve_batch as gpu_solve_batch

def generate_instance(n_customers, seed):
    """Generate a CVRP instance with fixed seed for reproducibility"""
    np.random.seed(seed)
    n = n_customers + 1
    coords = np.random.uniform(0, 1, size=(n, 2))
    coords[0] = [0.5, 0.5]
    demands = np.zeros(n, dtype=np.float32)
    demands[1:] = np.random.uniform(1, 10, size=n_customers)
    capacity = max(demands.sum() / 2, demands.max() * 2)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])
    return {'coords': coords, 'demands': demands, 'distances': distances,
            'capacity': capacity, 'n_customers': n_customers}

def analyze_instance(instance, n_customers):
    """Analyze a single instance with both solvers"""
    print(f"\nAnalyzing instance with N={n_customers}")
    print(f"Capacity: {instance['capacity']:.2f}")
    print(f"Demands: {instance['demands']}")
    
    # CPU solve
    print("\nCPU Exact DP Solver:")
    cpu_start = time.time()
    try:
        cpu_result = cpu_solve(instance, time_limit=60.0, verbose=True)
        cpu_time = time.time() - cpu_start
        print(f"  Cost: {cpu_result.cost:.6f}")
        print(f"  CPC: {cpu_result.cost/n_customers:.6f}")
        print(f"  Time: {cpu_time:.3f}s")
        print(f"  Routes: {cpu_result.vehicle_routes}")
    except Exception as e:
        print(f"  Error: {e}")
        cpu_result = None
        cpu_time = 0
    
    # GPU solve
    print("\nGPU Exact DP Solver:")
    gpu_start = time.time()
    try:
        gpu_results = gpu_solve_batch([instance], verbose=True)
        gpu_result = gpu_results[0]
        gpu_time = time.time() - gpu_start
        print(f"  Cost: {gpu_result.cost:.6f}")
        print(f"  CPC: {gpu_result.cost/n_customers:.6f}")
        print(f"  Time: {gpu_time:.3f}s")
        print(f"  Routes: {gpu_result.vehicle_routes}")
    except Exception as e:
        print(f"  Error: {e}")
        gpu_result = None
        gpu_time = 0
    
    # Compare results
    if cpu_result and gpu_result:
        print(f"\nDiscrepancy Analysis:")
        print(f"  Cost difference: {abs(cpu_result.cost - gpu_result.cost):.6f}")
        print(f"  CPC difference: {abs(cpu_result.cost/n_customers - gpu_result.cost/n_customers):.6f}")
        print(f"  Relative error: {abs(cpu_result.cost - gpu_result.cost) / min(cpu_result.cost, gpu_result.cost) * 100:.2f}%")
    
    return cpu_result, gpu_result

def main():
    print("=" * 60)
    print("CPU vs GPU CVRP Solver Discrepancy Analysis")
    print("=" * 60)
    
    # Test on small instances where we can verify optimality
    test_cases = [
        (4, 5000),  # N=4, seed=5000
        (5, 5000),  # N=5, seed=5000
        (6, 5000),  # N=6, seed=5000 (the one from your benchmark)
    ]
    
    all_cpu_cpcs = []
    all_gpu_cpcs = []
    
    for n_customers, seed in test_cases:
        instance = generate_instance(n_customers, seed)
        cpu_result, gpu_result = analyze_instance(instance, n_customers)
        
        if cpu_result:
            all_cpu_cpcs.append(cpu_result.cost / n_customers)
        if gpu_result:
            all_gpu_cpcs.append(gpu_result.cost / n_customers)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    if all_cpu_cpcs:
        print(f"\nCPU Results:")
        print(f"  Mean CPC: {np.mean(all_cpu_cpcs):.6f}")
        print(f"  Std CPC: {np.std(all_cpu_cpcs):.6f}")
    
    if all_gpu_cpcs:
        print(f"\nGPU Results:")
        print(f"  Mean CPC: {np.mean(all_gpu_cpcs):.6f}")
        print(f"  Std CPC: {np.std(all_gpu_cpcs):.6f}")

if __name__ == "__main__":
    main()
