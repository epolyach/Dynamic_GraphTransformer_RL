#!/usr/bin/env python3
"""
Quick test just for N=15 to complete the data
"""
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from gpu_cvrp_solver_truly_optimal_fixed import GPUExactCVRPFixed

# Test N=15
instance = _generate_instance(
    num_customers=15,
    capacity=30,
    coord_range=100,
    demand_range=[1, 10],
    seed=42
)

solver = GPUExactCVRPFixed()

print(f"Testing N=15 (2^15 = 32,768 states)")
print(f"Starting at {time.strftime('%H:%M:%S')}")
print("Estimated time: ~45 minutes based on scaling...")

start_time = time.time()
solutions = solver.solve_batch([instance], verbose=True)
end_time = time.time()

solve_time = end_time - start_time
print(f"\nN=15 completed in {solve_time:.3f} seconds ({solve_time/60:.1f} minutes)")
print(f"Solution cost: {solutions[0].cost:.6f}")

# Calculate exact multiplication factor from N=14
n14_time = 968.814  # From previous run
factor = solve_time / n14_time
print(f"Multiplication factor vs N=14: {factor:.3f}x")
print(f"Theoretical factor: 2.00x")
print(f"Efficiency ratio: {factor/2.0:.3f}")
