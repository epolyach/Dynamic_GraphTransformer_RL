#!/usr/bin/env python3
"""
Debug script to find the bug where exact solutions are worse than heuristic solutions.
This should never happen - exact algorithms should find optimal (minimum) solutions.
"""

import numpy as np
from advanced_solver import AdvancedCVRPSolver
from enhanced_generator import EnhancedCVRPGenerator, InstanceType

def debug_exact_vs_heuristic():
    """Debug exact vs heuristic solvers on the same instance"""
    
    # Generate a single test instance
    gen = EnhancedCVRPGenerator(config={})
    seed = 4242 + 5 * 1000 + 0  # Same seed as used in benchmark for N=5, instance 0
    
    instance = gen.generate_instance(
        num_customers=5,
        capacity=30,
        coord_range=100,
        demand_range=[1, 10],
        seed=seed,
        instance_type=InstanceType.RANDOM,
        apply_augmentation=False,
    )
    
    print("="*60)
    print("DEBUG: EXACT vs HEURISTIC SOLVER COMPARISON")
    print("="*60)
    print(f"Instance: N=5, capacity=30, seed={seed}")
    print(f"Instance keys: {instance.keys()}")
    print(f"Instance: {instance}")
    print()
    
    # Create exact solver
    exact_solver = AdvancedCVRPSolver(
        time_limit=60.0,
        enable_heuristics=False,  # EXACT ONLY
        verbose=True
    )
    
    # Create heuristic solver
    heuristic_solver = AdvancedCVRPSolver(
        time_limit=5.0,
        enable_heuristics=True,  # HEURISTIC ONLY
        verbose=True
    )
    
    print("🎯 SOLVING WITH EXACT ALGORITHM:")
    print("-" * 40)
    sol_exact = exact_solver.solve(instance)
    print(f"Exact solution: cost={sol_exact.cost:.6f}, time={sol_exact.solve_time:.3f}s")
    print(f"Algorithm: {sol_exact.algorithm_used}, Optimal: {sol_exact.is_optimal}")
    print(f"Routes: {sol_exact.route}")
    print()
    
    print("🎯 SOLVING WITH HEURISTIC ALGORITHM:")
    print("-" * 40)
    sol_heuristic = heuristic_solver.solve(instance)
    print(f"Heuristic solution: cost={sol_heuristic.cost:.6f}, time={sol_heuristic.solve_time:.3f}s")
    print(f"Algorithm: {sol_heuristic.algorithm_used}, Gap: {sol_heuristic.gap:.2%}")
    print(f"Routes: {sol_heuristic.route}")
    print()
    
    print("🔍 COMPARISON:")
    print("-" * 40)
    print(f"Exact cost:     {sol_exact.cost:.6f}")
    print(f"Heuristic cost: {sol_heuristic.cost:.6f}")
    print(f"Difference:     {sol_exact.cost - sol_heuristic.cost:.6f}")
    
    if sol_exact.cost > sol_heuristic.cost:
        print("❌ BUG DETECTED: Exact solution is WORSE than heuristic!")
        print("   This should NEVER happen. Exact algorithms find optimal solutions.")
        
        # Let's validate the solutions
        print("\n🔧 VALIDATING SOLUTIONS:")
        print("-" * 40)
        
        def validate_solution(routes, instance, name):
            print(f"\n{name} solution validation:")
            total_cost = 0
            all_customers = set()
            
            for i, route in enumerate(routes):
                if len(route) == 0:
                    continue
                    
                print(f"  Route {i+1}: {route}")
                
                # Check capacity constraint
                route_demand = sum(instance['demands'][c] for c in route)
                print(f"    Demand: {route_demand}/{instance['capacity']}")
                
                if route_demand > instance['capacity']:
                    print(f"    ❌ CAPACITY VIOLATION!")
                
                # Check route cost
                route_cost = 0
                prev = 0  # depot
                for customer in route:
                    if customer in all_customers:
                        print(f"    ❌ DUPLICATE CUSTOMER {customer}!")
                    all_customers.add(customer)
                    
                    dist = np.linalg.norm(np.array(instance['coords'][prev]) - np.array(instance['coords'][customer]))
                    route_cost += dist
                    prev = customer
                
                # Return to depot
                dist = np.linalg.norm(np.array(instance['coords'][prev]) - np.array(instance['coords'][0]))
                route_cost += dist
                total_cost += route_cost
                print(f"    Route cost: {route_cost:.6f}")
            
            # Check all customers served
            expected_customers = set(range(1, instance['num_customers'] + 1))
            if all_customers != expected_customers:
                print(f"    ❌ MISSING CUSTOMERS: {expected_customers - all_customers}")
                print(f"    ❌ EXTRA CUSTOMERS: {all_customers - expected_customers}")
            
            print(f"  Total cost: {total_cost:.6f}")
            return total_cost
        
        exact_validated_cost = validate_solution(sol_exact.route, instance, "EXACT")
        heuristic_validated_cost = validate_solution(sol_heuristic.route, instance, "HEURISTIC")
        
        print(f"\n📊 VALIDATED COSTS:")
        print(f"  Exact (reported):     {sol_exact.cost:.6f}")
        print(f"  Exact (validated):    {exact_validated_cost:.6f}")
        print(f"  Heuristic (reported): {sol_heuristic.cost:.6f}")
        print(f"  Heuristic (validated): {heuristic_validated_cost:.6f}")
        
    else:
        print("✅ CORRECT: Exact solution is better than or equal to heuristic")
        print(f"   Gap: {((sol_heuristic.cost - sol_exact.cost) / sol_exact.cost * 100):.2f}%")

if __name__ == "__main__":
    debug_exact_vs_heuristic()
