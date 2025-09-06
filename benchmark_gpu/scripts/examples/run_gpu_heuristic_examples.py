#!/usr/bin/env python3
"""
Examples of how to run GPU heuristic solvers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.heuristic_gpu_simple import GPUHeuristicSimple, solve_batch

def example_1_direct_solver():
    """Example 1: Using the solver class directly"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Direct Solver Usage")
    print("="*60)
    
    # Generate instance
    instance = _generate_instance(
        num_customers=10,
        capacity=20,
        coord_range=100,
        demand_range=[1, 10],
        seed=42
    )
    
    # Create solver and solve
    solver = GPUHeuristicSimple()
    solutions = solver.solve_batch([instance], verbose=True)
    
    print(f"Cost: {solutions[0].cost:.4f}")
    print(f"Vehicles used: {solutions[0].num_vehicles}")
    print(f"Routes: {solutions[0].vehicle_routes}")

def example_2_batch_function():
    """Example 2: Using the convenience function for batch solving"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Solving with Convenience Function")
    print("="*60)
    
    # Generate multiple instances
    instances = []
    for i in range(5):
        instance = _generate_instance(
            num_customers=15,
            capacity=30,
            coord_range=100,
            demand_range=[1, 10],
            seed=100 + i
        )
        instances.append(instance)
    
    # Solve batch
    solutions = solve_batch(instances, verbose=True)
    
    # Show results
    for i, sol in enumerate(solutions):
        cpc = sol.cost / 15  # cost per customer
        print(f"Instance {i+1}: Cost={sol.cost:.2f}, CPC={cpc:.4f}, Vehicles={sol.num_vehicles}")

def example_3_benchmark_script():
    """Example 3: Using the benchmark script from command line"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Command Line Benchmark Scripts")
    print("="*60)
    
    print("\nTo run benchmarks from command line:")
    print("-" * 40)
    
    print("\n1. Simple GPU Greedy (working):")
    print("   python3 benchmark_gpu_heuristic_gls.py --instances 100 --batch-size 50 --configs 10,20")
    
    print("\n2. Advanced GPU GLS (has bugs in cost calculation):")
    print("   python3 benchmark_gpu_heuristic_gls_advanced.py --instances 100 --batch-size 50 --configs 10,20 --time-limit 3.0")
    
    print("\n3. Quick test with small instances:")
    print("   python3 benchmark_gpu_heuristic_gls.py --instances 10 --batch-size 10 --configs 10")
    
    print("\n4. Full benchmark:")
    print("   python3 benchmark_gpu_heuristic_gls.py --instances 1000 --batch-size 100 --configs all")

def main():
    print("\n" + "="*70)
    print("GPU HEURISTIC SOLVER EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_direct_solver()
    example_2_batch_function()
    example_3_benchmark_script()
    
    print("\n" + "="*70)
    print("Note: The simple GPU solver is working correctly.")
    print("The advanced GLS solver has a bug in cost calculation (returns inf).")
    print("="*70)

if __name__ == "__main__":
    main()
