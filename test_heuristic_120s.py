#!/usr/bin/env python3
"""Test heuristic_or with 120s timeout"""
import sys
import time
import numpy as np
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
import solvers.heuristic_or as heuristic_or

gen = EnhancedCVRPGenerator(config={})
costs = []
cpcs = []
times = []
n = 20

print("Testing heuristic_or on 10 instances of N=20 with 120s timeout...")
for i in range(10):
    instance = gen.generate_instance(
        num_customers=n, capacity=30, coord_range=100,
        demand_range=[1, 10], seed=7000+i,  # Same seeds as before
        instance_type=InstanceType.RANDOM, apply_augmentation=False
    )
    
    start = time.time()
    solution = heuristic_or.solve(instance, time_limit=120.0, verbose=False)  # 120s timeout
    solve_time = time.time() - start
    
    costs.append(solution.cost)
    cpcs.append(solution.cost / n)
    times.append(solve_time)
    print(f"Instance {i+1}: cost={solution.cost:.4f}, cpc={solution.cost/n:.4f}, time={solve_time:.3f}s")

print(f"\nheuristic_or results with 120s timeout:")
print(f"  Average cost: {np.mean(costs):.4f}")
print(f"  Average CPC: {np.mean(cpcs):.4f} ± {np.std(cpcs)/np.sqrt(len(cpcs)):.4f} (SEM)")
print(f"  Average time: {np.mean(times):.3f}s")

# Compare with 60s results
costs_60s = [8.1461, 6.9843, 6.5852, 5.2871, 5.6154, 6.3677, 5.1909, 6.2359, 7.8326, 6.7713]
print(f"\nComparison:")
print(f"  60s timeout:  avg cost = {np.mean(costs_60s):.4f}, avg CPC = {np.mean(costs_60s)/n:.4f}")
print(f"  120s timeout: avg cost = {np.mean(costs):.4f}, avg CPC = {np.mean(cpcs):.4f}")
print(f"  Improvement: {(np.mean(costs_60s) - np.mean(costs))/np.mean(costs_60s)*100:.2f}%")
