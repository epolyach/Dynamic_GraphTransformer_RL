#!/usr/bin/env python3
"""Test heuristic_or with 2s timeout"""
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

print("Testing heuristic_or on 10 instances of N=20 with 2s timeout...")
for i in range(10):
    instance = gen.generate_instance(
        num_customers=n, capacity=30, coord_range=100,
        demand_range=[1, 10], seed=7000+i,  # Same seeds as before
        instance_type=InstanceType.RANDOM, apply_augmentation=False
    )
    
    start = time.time()
    solution = heuristic_or.solve(instance, time_limit=2.0, verbose=False)  # 2s timeout
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

print("\n=== heuristic_or Statistics (2s timeout) ===")
print(f"• Average CPC: {avg_cpc:.6f} ± {sem_cpc:.6f} (mean ± SEM)")
print(f"• Standard deviation: {std_cpc:.6f}")
print(f"• 95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"• Average solve time: {avg_time:.6f} ± {sem_time:.6f} seconds")
print(f"• CPC range: {min(cpcs):.6f} to {max(cpcs):.6f}")

# Compare with all timeout results (including 1s from previous run)
costs_60s = [8.1461, 6.9843, 6.5852, 5.2871, 5.6154, 6.3677, 5.1909, 6.2359, 7.8326, 6.7713]
cpc_1s = 0.326331  # From previous run

print(f"\n=== Complete Timeout Comparison ===")
print(f"Timeout  | Avg CPC    | Quality Loss | Speed vs 60s")
print(f"---------|------------|--------------|-------------")
print(f"  1s     | {cpc_1s:.6f} | {((cpc_1s - np.mean(costs_60s)/n)/(np.mean(costs_60s)/n))*100:+.2f}%      | 60x faster")
print(f"  2s     | {avg_cpc:.6f} | {((avg_cpc - np.mean(costs_60s)/n)/(np.mean(costs_60s)/n))*100:+.2f}%      | 30x faster")
print(f"  3s     | 0.325162   | +0.02%       | 20x faster")
print(f"  10s    | 0.325160   | +0.02%       | 6x faster")
print(f"  60s    | 0.325082   | baseline     | 1x")
print(f"  120s   | 0.325082   | +0.00%       | 0.5x slower")

print(f"\n📊 Quality improvement from timeout increase:")
print(f"  1s → 2s:   {((cpc_1s - avg_cpc)/cpc_1s)*100:.3f}% improvement, {((avg_cpc - np.mean(costs_60s)/n)/(np.mean(costs_60s)/n))*100:.3f}% from optimal")
print(f"  2s → 3s:   {((avg_cpc - 0.325162)/avg_cpc)*100:.3f}% improvement, {((0.325162 - np.mean(costs_60s)/n)/(np.mean(costs_60s)/n))*100:.3f}% from optimal")
print(f"  3s → 10s:  {((0.325162 - 0.325160)/0.325162)*100:.3f}% improvement")
print(f"  10s → 60s: {((0.325160 - 0.325082)/0.325160)*100:.3f}% improvement")

# Instance-by-instance comparison
print(f"\n=== Instance-by-Instance Comparison ===")
print(f"Instance | 60s Cost | 2s Cost | Difference | % Change")
print(f"---------|----------|---------|------------|----------")
costs_1s = [8.1461, 6.9843, 6.5852, 5.4123, 5.6154, 6.3677, 5.1909, 6.3350, 7.8580, 6.7713]  # From previous run
for i in range(10):
    diff = costs[i] - costs_60s[i]
    pct = (diff / costs_60s[i]) * 100
    print(f"   {i+1:2d}    | {costs_60s[i]:.4f}  | {costs[i]:.4f} | {diff:+.4f}   | {pct:+.2f}%")
    
print(f"\n📈 1s vs 2s vs 60s comparison per instance:")
for i in range(10):
    improvement_1s_to_2s = ((costs_1s[i] - costs[i]) / costs_1s[i]) * 100
    print(f"Instance {i+1}: 1s={costs_1s[i]:.4f}, 2s={costs[i]:.4f}, 60s={costs_60s[i]:.4f} | 1s→2s: {improvement_1s_to_2s:+.2f}%")
