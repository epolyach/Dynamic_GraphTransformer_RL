#!/usr/bin/env python3
import numpy as np
import time
import torch
from solvers.exact_gpu_dp import solve_batch as gpu_solve_batch
from solvers.exact_dp import solve as cpu_exact_solve

def generate_instance(n_customers, seed):
    np.random.seed(seed)
    n = n_customers + 1
    coords = np.random.uniform(0, 1, size=(n, 2))
    demands = np.zeros(n, dtype=np.float32)
    demands[1:] = np.random.uniform(1, 10, size=n_customers)
    capacity = max(demands.sum() / 2, demands.max() * 2)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])
    return {'coords': coords, 'demands': demands, 'distances': distances,
            'capacity': capacity, 'n_customers': n_customers}

# Test N=6
print("Testing N=6 (100 and 1000 instances)")
print("| Method       | Instances | Mean CPC | Std CPC | TPI (sec) | Speedup |")
print("|--------------|-----------|----------|---------|-----------|---------|")

for n_inst in [100, 1000]:
    instances = [generate_instance(6, 5000 + i) for i in range(n_inst)]
    
    # GPU batch solve
    start = time.time()
    gpu_results = gpu_solve_batch(instances, verbose=(n_inst == 100))
    gpu_time = time.time() - start
    gpu_cpcs = np.array([r.cost / 6 for r in gpu_results])
    
    # CPU solve (sample for comparison)
    sample_size = min(20, n_inst)
    start = time.time()
    cpu_cpcs = []
    for i in range(sample_size):
        r = cpu_exact_solve(instances[i], time_limit=60.0, verbose=False)
        cpu_cpcs.append(r.cost / 6)
    cpu_time = (time.time() - start) * n_inst / sample_size
    
    speedup = cpu_time / gpu_time
    print(f"| GPU-Exact-DP | {n_inst:9d} | {gpu_cpcs.mean():8.6f} | {gpu_cpcs.std():7.6f} | {gpu_time/n_inst:9.6f} | {speedup:7.1f}x |")

# Test N=10 if memory allows
if torch.cuda.is_available():
    print("\nTesting N=10 (100 instances)")
    instances = [generate_instance(10, 5000 + i) for i in range(100)]
    
    start = time.time()
    try:
        gpu_results = gpu_solve_batch(instances[:100], verbose=True)
        gpu_time = time.time() - start
        gpu_cpcs = np.array([r.cost / 10 for r in gpu_results])
        print(f"| GPU-Exact-DP |       100 | {gpu_cpcs.mean():8.6f} | {gpu_cpcs.std():7.6f} | {gpu_time/100:9.6f} |     N/A |")
    except Exception as e:
        print(f"N=10 failed: {e}")
