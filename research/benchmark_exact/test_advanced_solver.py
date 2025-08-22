#!/usr/bin/env python3
"""
Simple test script for the advanced CVRP solver.
Tests exact vs heuristic distinction.
"""

from advanced_solver import AdvancedCVRPSolver
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
import numpy as np

def main():
    print("Testing Advanced CVRP Solver - Exact vs Heuristic Distinction")
    print("=" * 60)

    # Generate small test instance
    gen = EnhancedCVRPGenerator(config={})
    instance = gen.generate_instance(
        num_customers=5,
        capacity=30,
        coord_range=100,
        demand_range=[1, 10],
        seed=42,
        instance_type=InstanceType.RANDOM
    )

    print(f"Test instance: {len(instance['coords'])-1} customers, capacity={instance['capacity']}")
    print()

    # Test EXACT mode
    print("Testing EXACT MODE (enable_heuristics=False):")
    exact_solver = AdvancedCVRPSolver(
        time_limit=10.0,
        enable_heuristics=False,  # Exact algorithms only
        verbose=False
    )

    try:
        sol_exact = exact_solver.solve(instance)
        print(f"  Exact solution: {sol_exact.algorithm_used}, cost={sol_exact.cost:.3f}, optimal={sol_exact.is_optimal}")
    except Exception as e:
        print(f"  Exact solver failed: {e}")

    print()

    # Test HEURISTIC mode  
    print("Testing HEURISTIC MODE (enable_heuristics=True):")
    heuristic_solver = AdvancedCVRPSolver(
        time_limit=5.0,
        enable_heuristics=True,  # Heuristic algorithms only
        verbose=False
    )

    try:
        sol_heuristic = heuristic_solver.solve(instance)
        print(f"  Heuristic solution: {sol_heuristic.algorithm_used}, cost={sol_heuristic.cost:.3f}, gap={sol_heuristic.gap:.1%}")
    except Exception as e:
        print(f"  Heuristic solver failed: {e}")

    print()
    print("Test completed!")
    print()

    # Test algorithm availability
    solver_test = AdvancedCVRPSolver(verbose=False)
    print("Available Algorithms:")
    print("   Dynamic Programming: YES (built-in)")
    print(f"   OR-Tools: {'YES' if solver_test._has_ortools else 'NO'}")
    print(f"   Gurobi: {'YES' if solver_test._has_gurobi else 'NO'}")
    print(f"   PyVRP (HGS-CVRP): {'YES' if solver_test._has_pyvrp else 'NO'}")
    print("   ALNS: YES (built-in)")

if __name__ == "__main__":
    main()
