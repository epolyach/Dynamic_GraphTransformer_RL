#!/usr/bin/env python3
"""Test heuristic_or with 2s timeout on 100 instances"""
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

print("Testing heuristic_or on 100 instances of N=20 with 2s timeout...")
print("Progress: ", end="", flush=True)

for i in range(100):
    instance = gen.generate_instance(
        num_customers=n, capacity=30, coord_range=100,
        demand_range=[1, 10], seed=7000+i,
        instance_type=InstanceType.RANDOM, apply_augmentation=False
    )
    
    start = time.time()
    solution = heuristic_or.solve(instance, time_limit=2.0, verbose=False)
    solve_time = time.time() - start
    
    costs.append(solution.cost)
    cpcs.append(solution.cost / n)
    times.append(solve_time)
    
    # Progress indicator
    if (i + 1) % 10 == 0:
        print(f"{i+1}", end=" ", flush=True)

print("\nCompleted!\n")

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

print("=== heuristic_or Statistics (2s timeout, 100 instances) ===")
print(f"• Number of instances: {len(cpcs)}")
print(f"• Average CPC: {avg_cpc:.6f} ± {sem_cpc:.6f} (mean ± SEM)")
print(f"• Standard deviation: {std_cpc:.6f}")
print(f"• 95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"• Average solve time: {avg_time:.6f} ± {sem_time:.6f} seconds")
print(f"• Total runtime: {sum(times):.1f} seconds ({sum(times)/60:.1f} minutes)")
print(f"• CPC range: {min(cpcs):.6f} to {max(cpcs):.6f}")

# Distribution analysis
print("\n=== CPC Distribution Analysis ===")
percentiles = [10, 25, 50, 75, 90]
for p in percentiles:
    val = np.percentile(cpcs, p)
    print(f"• {p}th percentile: {val:.6f}")

# Compare with previous results
print("\n=== Comparison with Previous Runs ===")
print("Instances | Timeout | Avg CPC    | 95% CI                | SEM")
print("----------|---------|------------|------------------------|----------")
print(f"   10     |   1s    | 0.326331   | [0.291395, 0.361266]  | 0.015444")
print(f"   10     |   2s    | 0.325162   | [0.289759, 0.360564]  | 0.015650")
print(f"  100     |   2s    | {avg_cpc:.6f}   | [{ci_lower:.6f}, {ci_upper:.6f}]  | {sem_cpc:.6f}")

# Save results to CSV
import csv
csv_filename = "results/csv/heuristic_2s_100instances.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['instance_id', 'n', 'cost', 'cpc', 'solve_time', 'seed'])
    for i in range(100):
        writer.writerow([i+1, n, costs[i], cpcs[i], times[i], 7000+i])
print(f"\n✅ Results saved to {csv_filename}")

# Summary statistics for reporting
print("\n=== Summary for Reporting ===")
print(f"• N=20, 100 instances, 2s timeout")
print(f"• Average Cost per Customer: {avg_cpc:.6f} ± {sem_cpc:.6f}")
print(f"• 95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"• Total runtime: {sum(times)/60:.1f} minutes")
print(f"• Average time per instance: {avg_time:.3f}s")
