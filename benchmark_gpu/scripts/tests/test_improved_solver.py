#!/usr/bin/env python3
"""
Test improved GPU solver to verify it produces correct results
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.heuristic_gpu_improved import solve_batch

print("=" * 60)
print("Testing Improved GPU Solver (10 instances)")
print("=" * 60)

# Test configurations
configs = [
    (10, 20),
    (20, 30),
    (50, 40),
    (100, 50)
]

print("\nExpected values (from OR-Tools GLS):")
print("N=10:  Mean CPC ≈ 0.4481")
print("N=20:  Mean CPC ≈ 0.3245")
print("N=50:  Mean CPC ≈ 0.2433")
print("N=100: Mean CPC ≈ 0.1776")

print("\nActual results from improved GPU solver:")
print("-" * 50)

for n_customers, capacity in configs:
    # Generate 10 test instances
    instances = []
    for i in range(10):
        instance = _generate_instance(
            num_customers=n_customers,
            capacity=capacity,
            coord_range=100,
            demand_range=[1, 10],
            seed=42000 + n_customers * 1000 + i
        )
        instances.append(instance)
    
    # Solve with improved algorithm
    solutions = solve_batch(instances, max_iterations=50, verbose=True)
    
    # Calculate statistics
    cpcs = [sol.cost / n_customers for sol in solutions]
    mean_cpc = np.mean(cpcs)
    std_cpc = np.std(cpcs)
    
    print(f"N={n_customers:3d}: Mean CPC = {mean_cpc:.4f}, Std = {std_cpc:.4f}")
    print("-" * 50)
