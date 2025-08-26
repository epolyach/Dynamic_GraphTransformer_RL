#!/usr/bin/env python3
"""Test heuristic_or with 10s timeout"""
import sys
import time
import numpy as np
from scipy import stats
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
import solvers.heuristic_or as heuristic_or

gen = EnhancedCVRPGenerator(config={})
costs = []
cpcs = []
times = []
n = 20

print("Testing heuristic_or on 10 instances of N=20 with 10s timeout...")
for i in range(10):
    instance = gen.generate_instance(
        num_customers=n, capacity=30, coord_range=100,
        demand_range=[1, 10], seed=7000+i,  # Same seeds as before
        instance_type=InstanceType.RANDOM, apply_augmentation=False
    )
    
    start = time.time()
    solution = heuristic_or.solve(instance, time_limit=10.0, verbose=False)  # 10s timeout
    solve_time = time.time() - start
    
    costs.append(solution.cost)
    cpcs.append(solution.cost / n)
    times.append(solve_time)
    print(f"Instance {i+1}: cost={solution.cost:.4f}, cpc={solution.cost/n:.4f}, time={solve_time:.3f}s")

# Calculate statistics
avg_cpc = np.mean(cpcs)
std_cpc = np.std(cpcs, ddof=1)
sem_cpc = std_cpc / np.sqrt(len(cpcs))
avg_time = np.mean(times)
std_time = np.std(times, ddof=1)
sem_time = std_time / np.sqrt(len(times))

# 95% confidence interval
ci_95 = stats.t.ppf(0.975, len(cpcs)-1) * sem_cpc
ci_lower = avg_cpc - ci_95
ci_upper = avg_cpc + ci_95

print("\n=== heuristic_or Statistics (10s timeout) ===")
print(f"• Average CPC: {avg_cpc:.6f} ± {sem_cpc:.6f} (mean ± SEM)")
print(f"• Standard deviation: {std_cpc:.6f}")
print(f"• 95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"• Average solve time: {avg_time:.6f} ± {sem_time:.6f} seconds")
print(f"• CPC range: {min(cpcs):.6f} to {max(cpcs):.6f}")

# Compare with 60s and 120s results
costs_60s = [8.1461, 6.9843, 6.5852, 5.2871, 5.6154, 6.3677, 5.1909, 6.2359, 7.8326, 6.7713]
print(f"\n=== Timeout Comparison ===")
print(f"  10s timeout:  avg CPC = {avg_cpc:.6f}")
print(f"  60s timeout:  avg CPC = {np.mean(costs_60s)/n:.6f}")
print(f"  120s timeout: avg CPC = {np.mean(costs_60s)/n:.6f} (no change from 60s)")
print(f"  Quality loss (10s vs 60s): {((avg_cpc - np.mean(costs_60s)/n)/(np.mean(costs_60s)/n))*100:.2f}%")
