#!/usr/bin/env python3
import sys
sys.path.append('../..')
from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.heuristic_gpu_simple import solve_batch as simple_solve
from src.benchmarking.solvers.gpu.heuristic_gpu_gls import solve_batch as gls_solve

# Generate a simple test instance
instance = _generate_instance(
    num_customers=5,
    capacity=20,
    coord_range=100,
    demand_range=[1, 10],
    seed=42
)

print("Testing with N=5 customers")
print("-" * 40)

# Test simple solver
print("\nSimple Greedy Solver:")
simple_solutions = simple_solve([instance], verbose=True)
print(f"Cost: {simple_solutions[0].cost:.4f}")
print(f"Num vehicles: {simple_solutions[0].num_vehicles}")

# Test GLS solver
print("\nAdvanced GLS Solver:")
gls_solutions = gls_solve([instance], time_limit=1.0, verbose=True)
print(f"Cost: {gls_solutions[0].cost:.4f}")
print(f"Num vehicles: {gls_solutions[0].num_vehicles}")
