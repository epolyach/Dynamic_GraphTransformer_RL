#!/usr/bin/env python3
import numpy as np
import time
from solvers.exact_gpu_improved import ImprovedGPUCVRPSolver

def generate_instance(n_customers, seed):
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

def run_test(n_instances):
    n_customers = 6
    instances = [generate_instance(n_customers, 5000 + i) for i in range(n_instances)]
    
    solver = ImprovedGPUCVRPSolver()
    start = time.time()
    
    results = []
    batch_size = 10
    for i in range(0, n_instances, batch_size):
        batch = instances[i:min(i + batch_size, n_instances)]
        results.extend(solver.solve_batch(batch, time_limit=30.0, verbose=False))
    
    total_time = time.time() - start
    costs = np.array([r.cost for r in results])
    cpcs = costs / n_customers
    
    return cpcs.mean(), cpcs.std(), total_time / n_instances

# Run for 100 instances
cpc_100, std_100, tpi_100 = run_test(100)

# Run for 1000 instances  
cpc_1000, std_1000, tpi_1000 = run_test(1000)

# Print results
print("\n| Instances | Mean CPC | Std CPC | TPI (sec) |")
print("|-----------|----------|---------|-----------|")
print(f"|       100 | {cpc_100:8.6f} | {std_100:7.6f} | {tpi_100:9.6f} |")
print(f"|      1000 | {cpc_1000:8.6f} | {std_1000:7.6f} | {tpi_1000:9.6f} |")
