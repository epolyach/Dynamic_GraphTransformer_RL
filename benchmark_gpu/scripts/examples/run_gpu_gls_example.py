#!/usr/bin/env python3
"""
Example script showing how to run the GPU GLS heuristic solver
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.heuristic_gpu_gls import GPUHeuristicGLS

def main():
    print("=" * 60)
    print("GPU GLS Heuristic Solver Example")
    print("=" * 60)
    
    # Generate test instances
    instances = []
    n_customers = 10
    capacity = 20
    num_instances = 5
    
    print(f"\nGenerating {num_instances} instances with {n_customers} customers")
    for i in range(num_instances):
        instance = _generate_instance(
            num_customers=n_customers,
            capacity=capacity,
            coord_range=100,
            demand_range=[1, 10],
            seed=42 + i
        )
        instances.append(instance)
    
    # Initialize GPU GLS solver
    print("\nInitializing GPU GLS solver...")
    try:
        solver = GPUHeuristicGLS(device='cuda')
        print("✓ GPU solver initialized successfully")
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("This solver requires a CUDA GPU!")
        return
    
    # Solve instances
    print(f"\nSolving {num_instances} instances with GLS...")
    print("Time limit: 2.0 seconds per instance")
    
    solutions = solver.solve_batch(
        instances, 
        time_limit=2.0,  # seconds per instance
        verbose=True      # show progress
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Results:")
    print("-" * 60)
    
    for i, sol in enumerate(solutions):
        print(f"\nInstance {i+1}:")
        print(f"  Cost: {sol.cost:.4f}")
        print(f"  Cost per customer: {sol.cost/n_customers:.4f}")
        print(f"  Number of vehicles: {sol.num_vehicles}")
        print(f"  Algorithm: {sol.algorithm_used}")
        print(f"  Solve time: {sol.solve_time:.2f}s")
        
        # Show routes
        print(f"  Vehicle routes:")
        for v, route in enumerate(sol.vehicle_routes):
            print(f"    Vehicle {v+1}: {route}")
    
    # Summary statistics
    total_cost = sum(s.cost for s in solutions)
    avg_cost = total_cost / len(solutions)
    avg_cpc = avg_cost / n_customers
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Average cost: {avg_cost:.4f}")
    print(f"  Average cost per customer: {avg_cpc:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
