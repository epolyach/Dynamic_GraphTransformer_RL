#!/usr/bin/env python3
"""
Reproduce the discrepancy in benchmark tables
"""

import numpy as np
import time
from solvers.exact_dp import solve as cpu_solve
from solvers.exact_gpu_dp import solve_batch as gpu_solve_batch

def generate_instance(n_customers, seed):
    """Generate test instance matching benchmark_gpu_exact.py"""
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

def run_benchmark_comparison():
    """Run both CPU and GPU benchmarks with identical instances"""
    print("Testing N=6 with 100 instances (seeds 5000-5099)")
    print("="*60)
    
    # Generate 100 instances with exact same seeds as benchmark_gpu_exact.py
    instances = [generate_instance(6, 5000 + i) for i in range(100)]
    
    # CPU solve
    print("\nCPU Exact DP Solver:")
    cpu_start = time.time()
    cpu_cpcs = []
    for i, instance in enumerate(instances):
        if i % 20 == 0:
            print(f"  Progress: {i}/100")
        try:
            result = cpu_solve(instance, time_limit=60.0, verbose=False)
            cpu_cpcs.append(result.cost / 6)
        except Exception as e:
            print(f"  Error on instance {i}: {e}")
            cpu_cpcs.append(float('nan'))
    cpu_time = time.time() - cpu_start
    
    cpu_cpcs = np.array(cpu_cpcs)
    valid_cpu = cpu_cpcs[~np.isnan(cpu_cpcs)]
    
    # GPU batch solve
    print("\nGPU Exact DP Solver:")
    gpu_start = time.time()
    gpu_results = gpu_solve_batch(instances, verbose=False)
    gpu_time = time.time() - gpu_start
    gpu_cpcs = np.array([r.cost / 6 for r in gpu_results])
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\n| Solver   | Instances | Mean CPC | Std CPC | TPI (sec) |")
    print("|----------|-----------|----------|---------|-----------|")
    print(f"| CPU      | {len(valid_cpu):9d} | {valid_cpu.mean():.6f} | {valid_cpu.std():.6f} | {cpu_time/100:.6f} |")
    print(f"| GPU      | {len(gpu_cpcs):9d} | {gpu_cpcs.mean():.6f} | {gpu_cpcs.std():.6f} | {gpu_time/100:.6f} |")
    
    print(f"\nSpeedup: {(cpu_time/gpu_time):.1f}x")
    print(f"Mean CPC difference: {abs(valid_cpu.mean() - gpu_cpcs.mean()):.6f}")
    print(f"Std CPC difference: {abs(valid_cpu.std() - gpu_cpcs.std()):.6f}")

if __name__ == "__main__":
    run_benchmark_comparison()
