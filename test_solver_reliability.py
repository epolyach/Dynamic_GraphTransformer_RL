#!/usr/bin/env python3
"""
Test script to compare different CVRP solver reliability.
Tests the same small instance with multiple solvers to see which give consistent results.
"""

import sys
import numpy as np
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType

# Import all available solvers
import solvers.exact_milp as exact_milp
import solvers.exact_dp as exact_dp
try:
    import solvers.exact_ortools_vrp as exact_ortools_vrp
    HAS_ORTOOLS_VRP = True
except ImportError:
    HAS_ORTOOLS_VRP = False
    exact_ortools_vrp = None
    print("OR-Tools VRP module not available")

try:
    import solvers.exact_pulp as exact_pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    exact_pulp = None
    print("PuLP module not available")


def test_solver_on_instance(solver_module, solver_name, instance, time_limit=60.0):
    """Test a single solver on an instance."""
    try:
        solution = solver_module.solve(instance, time_limit=time_limit, verbose=False)
        return {
            'solver': solver_name,
            'cost': solution.cost,
            'vehicles': solution.num_vehicles,
            'routes': solution.vehicle_routes,
            'time': solution.solve_time,
            'optimal': solution.is_optimal,
            'algorithm': solution.algorithm_used,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'solver': solver_name,
            'cost': float('inf'),
            'vehicles': 0,
            'routes': [],
            'time': time_limit,
            'optimal': False,
            'algorithm': 'Failed',
            'success': False,
            'error': str(e)
        }

def format_routes(routes):
    """Format routes for display."""
    if not routes:
        return "[]"
    route_strings = []
    for route in routes:
        customers = [node for node in route if node != 0]
        route_strings.append(f"({', '.join(map(str, customers))})")
    return f"[{', '.join(route_strings)}]"

def main():
    print("=" * 80)
    print("CVRP SOLVER RELIABILITY TEST")
    print("=" * 80)
    
    # Generate a small test instance
    gen = EnhancedCVRPGenerator(config={})
    instance = gen.generate_instance(
        num_customers=6,  # Small instance for reliable comparison
        capacity=20,
        coord_range=100,
        demand_range=[1, 8],
        seed=12345,
        instance_type=InstanceType.RANDOM,
        apply_augmentation=False,
    )
    
    print(f"Test instance: {instance['num_customers']} customers, capacity={instance['capacity']}")
    print(f"Demands: {instance['demands'].tolist()}")
    print(f"Coordinates:")
    for i, coord in enumerate(instance['coords']):
        print(f"  Node {i}: ({coord[0]:.3f}, {coord[1]:.3f})")
    print()
    
    # Test all available solvers
    solvers_to_test = [
        (exact_milp, "exact_milp"),
        (exact_dp, "exact_dp"),
    ]
    
    if HAS_ORTOOLS_VRP:
        solvers_to_test.append((exact_ortools_vrp, "exact_ortools_vrp"))
    
    if HAS_PULP:
        solvers_to_test.append((exact_pulp, "exact_pulp"))
    
    
    results = []
    
    print("Testing solvers...")
    print("-" * 80)
    
    for solver_module, solver_name in solvers_to_test:
        print(f"Testing {solver_name}...")
        result = test_solver_on_instance(solver_module, solver_name, instance)
        results.append(result)
        
        if result['success']:
            print(f"  ✅ Success: cost={result['cost']:.4f}, vehicles={result['vehicles']}, time={result['time']:.3f}s")
            print(f"     Algorithm: {result['algorithm']}, Optimal: {result['optimal']}")
            print(f"     Routes: {format_routes(result['routes'])}")
        else:
            print(f"  ❌ Failed: {result['error']}")
        print()
    
    # Compare results
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) == 0:
        print("❌ No solvers succeeded!")
        return
    
    if len(successful_results) == 1:
        print("⚠️  Only one solver succeeded - cannot compare")
        result = successful_results[0]
        print(f"Winner: {result['solver']} with cost {result['cost']:.4f}")
        return
    
    # Find best cost
    best_cost = min(r['cost'] for r in successful_results)
    best_results = [r for r in successful_results if abs(r['cost'] - best_cost) < 1e-6]
    
    print(f"Best cost found: {best_cost:.4f}")
    print(f"Solvers achieving best cost: {[r['solver'] for r in best_results]}")
    print()
    
    # Check for disagreements
    costs = [r['cost'] for r in successful_results]
    max_cost = max(costs)
    min_cost = min(costs)
    cost_range = max_cost - min_cost
    
    if cost_range > 1e-4:  # Significant disagreement
        print("🚨 SIGNIFICANT DISAGREEMENT DETECTED!")
        print(f"Cost range: {cost_range:.6f} ({100*cost_range/min_cost:.3f}% difference)")
        print()
        
        print("Detailed comparison:")
        for result in successful_results:
            diff_pct = 100 * (result['cost'] - best_cost) / best_cost if best_cost > 0 else 0
            status = "OPTIMAL" if abs(result['cost'] - best_cost) < 1e-6 else f"+{diff_pct:.3f}%"
            print(f"  {result['solver']:20} | Cost: {result['cost']:.6f} | {status:>10} | Routes: {format_routes(result['routes'])}")
    
    else:
        print("✅ All solvers agree on the optimal cost (within tolerance)")
        print("The solvers are consistent for this instance.")
    
    print()
    print("Summary:")
    for result in results:
        if result['success']:
            reliability = "HIGH" if result['optimal'] else "MEDIUM"
            print(f"  {result['solver']:20} | {reliability:>6} | {result['algorithm']}")
        else:
            print(f"  {result['solver']:20} | FAILED | {result['error'][:50]}...")

if __name__ == '__main__':
    main()
