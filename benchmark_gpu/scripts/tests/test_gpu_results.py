#!/usr/bin/env python3
"""
Test GPU solver and compare with expected results
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.heuristic_gpu_fixed import solve_batch

def test_configuration(n_customers, capacity, num_instances):
    """Test a specific configuration"""
    print(f"\nTesting N={n_customers}, Capacity={capacity}, Instances={num_instances}")
    print("-" * 50)
    
    # Generate instances
    instances = []
    for i in range(num_instances):
        instance = _generate_instance(
            num_customers=n_customers,
            capacity=capacity,
            coord_range=100,
            demand_range=[1, 10],
            seed=42000 + n_customers * 1000 + i
        )
        instances.append(instance)
    
    # Solve
    solutions = solve_batch(instances, verbose=True)
    
    # Calculate statistics
    cpcs = np.array([sol.cost / n_customers for sol in solutions])
    mean_cpc = cpcs.mean()
    std_cpc = cpcs.std()
    sem = std_cpc / np.sqrt(num_instances)
    sem_pct = (2 * sem / mean_cpc) * 100
    
    print(f"Mean CPC: {mean_cpc:.4f}")
    print(f"Std CPC: {std_cpc:.4f}")
    print(f"SEM: {sem:.4f}")
    print(f"2×SEM/Mean: {sem_pct:.2f}%")
    
    return mean_cpc, std_cpc, sem, sem_pct

# Test with 10 instances
print("=" * 60)
print("GPU Heuristic Test - 10 instances")
print("=" * 60)

configs = [
    (10, 20),
    (20, 30),
    (50, 40),
    (100, 50)
]

results = []
for n_customers, capacity in configs:
    result = test_configuration(n_customers, capacity, 10)
    results.append(result)

print("\n" + "=" * 60)
print("Summary Table:")
print("=" * 60)
print(f"{'N':>4} {'Capacity':>10} {'Mean CPC':>12} {'Std CPC':>10} {'SEM':>10} {'2×SEM/Mean':>12}")
print("-" * 60)
for i, (n, cap) in enumerate(configs):
    mean, std, sem, pct = results[i]
    print(f"{n:4d} {cap:10d} {mean:12.4f} {std:10.4f} {sem:10.4f} {pct:11.2f}%")
