#!/usr/bin/env python3
"""Compare heuristic solutions with exact solutions to measure optimality gap"""
import sys
import time
import numpy as np
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
import solvers.heuristic_or as heuristic_or
import solvers.exact_ortools_vrp as exact_or

# Check what exact solvers are available
print("=== Testing Exact Solver Feasibility for N=20 ===\n")

gen = EnhancedCVRPGenerator(config={})
n = 20

# Test a single instance with exact solver
print("Testing exact_ortools_vrp solver with reasonable timeout...")
instance = gen.generate_instance(
    num_customers=n, capacity=30, coord_range=100,
    demand_range=[1, 10], seed=7000,
    instance_type=InstanceType.RANDOM, apply_augmentation=False
)

# Try exact solver with 60s timeout
start = time.time()
try:
    exact_solution = exact_or.solve(instance, time_limit=60.0, verbose=False)
    exact_time = time.time() - start
    print(f"✓ Exact solver succeeded in {exact_time:.2f}s")
    print(f"  Exact cost: {exact_solution.cost:.4f}")
    
    # Compare with heuristic
    start = time.time()
    heur_solution = heuristic_or.solve(instance, time_limit=2.0, verbose=False)
    heur_time = time.time() - start
    print(f"  Heuristic cost (2s): {heur_solution.cost:.4f}")
    print(f"  Optimality gap: {((heur_solution.cost - exact_solution.cost) / exact_solution.cost * 100):.2f}%")
    
except Exception as e:
    print(f"✗ Exact solver failed: {e}")
    exact_solution = None

print("\n" + "="*60 + "\n")

# If exact solver works, run comparison on small sample
if exact_solution is not None:
    print("=== Optimality Gap Analysis (10 instances) ===\n")
    
    exact_costs = []
    heur_costs_2s = []
    heur_costs_10s = []
    gaps_2s = []
    gaps_10s = []
    exact_times = []
    
    print("Instance | Exact Cost | Heur(2s) | Gap(2s) | Heur(10s) | Gap(10s) | Exact Time")
    print("---------|------------|----------|---------|-----------|----------|------------")
    
    for i in range(10):
        instance = gen.generate_instance(
            num_customers=n, capacity=30, coord_range=100,
            demand_range=[1, 10], seed=7000+i,
            instance_type=InstanceType.RANDOM, apply_augmentation=False
        )
        
        # Exact solution (with 300s timeout for safety)
        start = time.time()
        try:
            exact_sol = exact_or.solve(instance, time_limit=300.0, verbose=False)
            exact_time = time.time() - start
            exact_costs.append(exact_sol.cost)
            exact_times.append(exact_time)
            
            # Heuristic with 2s
            heur_sol_2s = heuristic_or.solve(instance, time_limit=2.0, verbose=False)
            heur_costs_2s.append(heur_sol_2s.cost)
            gap_2s = (heur_sol_2s.cost - exact_sol.cost) / exact_sol.cost * 100
            gaps_2s.append(gap_2s)
            
            # Heuristic with 10s
            heur_sol_10s = heuristic_or.solve(instance, time_limit=10.0, verbose=False)
            heur_costs_10s.append(heur_sol_10s.cost)
            gap_10s = (heur_sol_10s.cost - exact_sol.cost) / exact_sol.cost * 100
            gaps_10s.append(gap_10s)
            
            print(f"   {i+1:2d}    | {exact_sol.cost:10.4f} | {heur_sol_2s.cost:8.4f} | {gap_2s:6.2f}% | {heur_sol_10s.cost:9.4f} | {gap_10s:7.2f}% | {exact_time:8.2f}s")
            
        except Exception as e:
            print(f"   {i+1:2d}    | Failed: {str(e)[:50]}")
    
    if exact_costs:
        print("\n=== Summary Statistics ===")
        print(f"• Average exact cost: {np.mean(exact_costs):.4f}")
        print(f"• Average exact solve time: {np.mean(exact_times):.2f}s")
        print(f"\nHeuristic (2s timeout):")
        print(f"• Average cost: {np.mean(heur_costs_2s):.4f}")
        print(f"• Average optimality gap: {np.mean(gaps_2s):.2f}% ± {np.std(gaps_2s, ddof=1):.2f}%")
        print(f"• Gap range: {min(gaps_2s):.2f}% to {max(gaps_2s):.2f}%")
        print(f"\nHeuristic (10s timeout):")
        print(f"• Average cost: {np.mean(heur_costs_10s):.4f}")
        print(f"• Average optimality gap: {np.mean(gaps_10s):.2f}% ± {np.std(gaps_10s, ddof=1):.2f}%")
        print(f"• Gap range: {min(gaps_10s):.2f}% to {max(gaps_10s):.2f}%")
        
        # CPC comparison
        print(f"\n=== Cost Per Customer (CPC) Comparison ===")
        exact_cpc = np.mean(exact_costs) / n
        heur_cpc_2s = np.mean(heur_costs_2s) / n
        heur_cpc_10s = np.mean(heur_costs_10s) / n
        print(f"• Exact CPC: {exact_cpc:.6f}")
        print(f"• Heuristic CPC (2s): {heur_cpc_2s:.6f} (+{(heur_cpc_2s - exact_cpc)/exact_cpc*100:.2f}%)")
        print(f"• Heuristic CPC (10s): {heur_cpc_10s:.6f} (+{(heur_cpc_10s - exact_cpc)/exact_cpc*100:.2f}%)")
        
        print(f"\n📊 Key Finding: Heuristic with 2s timeout is on average {np.mean(gaps_2s):.1f}% away from optimal")
        print(f"   This means the heuristic solutions are approximately {100 - np.mean(gaps_2s):.1f}% optimal")

else:
    print("Cannot perform optimality gap analysis - exact solver not working for N=20")
