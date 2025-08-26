#!/usr/bin/env python3
"""Test heuristic_or with 1s timeout"""
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

print("Testing heuristic_or on 10 instances of N=20 with 1s timeout...")
for i in range(10):
    instance = gen.generate_instance(
        num_customers=n, capacity=30, coord_range=100,
        demand_range=[1, 10], seed=7000+i,  # Same seeds as before
        instance_type=InstanceType.RANDOM, apply_augmentation=False
    )
    
    start = time.time()
    solution = heuristic_or.solve(instance, time_limit=1.0, verbose=False)  # 1s timeout
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

print("\n=== heuristic_or Statistics (1s timeout) ===")
print(f"• Average CPC: {avg_cpc:.6f} ± {sem_cpc:.6f} (mean ± SEM)")
print(f"• Standard deviation: {std_cpc:.6f}")
print(f"• 95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"• Average solve time: {avg_time:.6f} ± {sem_time:.6f} seconds")
print(f"• CPC range: {min(cpcs):.6f} to {max(cpcs):.6f}")

# Compare with all timeout results
costs_60s = [8.1461, 6.9843, 6.5852, 5.2871, 5.6154, 6.3677, 5.1909, 6.2359, 7.8326, 6.7713]

print(f"\n=== Complete Timeout Comparison ===")
print(f"Timeout  | Avg CPC    | Quality Loss | Speed vs 60s")
print(f"---------|------------|--------------|-------------")
print(f"  1s     | {avg_cpc:.6f} | {((avg_cpc - np.mean(costs_60s)/n)/(np.mean(costs_60s)/n))*100:+.2f}%      | 60x faster")
print(f"  3s     | 0.325162   | +0.02%       | 20x faster")
print(f"  10s    | 0.325160   | +0.02%       | 6x faster")
print(f"  60s    | 0.325082   | baseline     | 1x")
print(f"  120s   | 0.325082   | +0.00%       | 0.5x slower")

print(f"\n📊 Diminishing returns analysis:")
print(f"  1s → 3s:   {((0.325162 - avg_cpc)/avg_cpc)*100:.3f}% improvement")
print(f"  3s → 10s:  {((0.325160 - 0.325162)/0.325162)*100:.3f}% improvement")
print(f"  10s → 60s: {((0.325082 - 0.325160)/0.325160)*100:.3f}% improvement")
