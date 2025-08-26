#!/usr/bin/env python3
"""Test heuristic_or with 2s timeout on 600 instances"""
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

print("Testing heuristic_or on 600 instances of N=20 with 2s timeout...")
print("Expected runtime: ~20 minutes")
print("Progress: ", end="", flush=True)

start_total = time.time()

for i in range(600):
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
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_total
        eta = (elapsed / (i + 1)) * (600 - i - 1)
        print(f"{i+1}({elapsed/60:.1f}m, ETA:{eta/60:.1f}m)", end=" ", flush=True)

total_time = time.time() - start_total
print(f"\nCompleted in {total_time/60:.1f} minutes!\n")

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

print("=== heuristic_or Statistics (2s timeout, 600 instances) ===")
print(f"• Number of instances: {len(cpcs)}")
print(f"• Average CPC: {avg_cpc:.6f} ± {sem_cpc:.6f} (mean ± SEM)")
print(f"• Standard deviation: {std_cpc:.6f}")
print(f"• 95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"• CI width: {(ci_upper - ci_lower):.6f}")
print(f"• Average solve time: {avg_time:.6f} ± {sem_time:.6f} seconds")
print(f"• Total runtime: {sum(times):.1f} seconds ({sum(times)/60:.1f} minutes)")
print(f"• CPC range: {min(cpcs):.6f} to {max(cpcs):.6f}")

# Distribution analysis
print("\n=== CPC Distribution Analysis ===")
percentiles = [5, 10, 25, 50, 75, 90, 95]
for p in percentiles:
    val = np.percentile(cpcs, p)
    print(f"• {p:2d}th percentile: {val:.6f}")

# Compare with previous results
print("\n=== Comparison with Previous Runs ===")
print("Instances | Timeout | Avg CPC    | 95% CI                 | CI Width  | SEM")
print("----------|---------|------------|------------------------|-----------|----------")
print(f"   10     |   1s    | 0.326331   | [0.291395, 0.361266]  | 0.069871  | 0.015444")
print(f"   10     |   2s    | 0.325162   | [0.289759, 0.360564]  | 0.070805  | 0.015650")
print(f"  100     |   2s    | 0.329422   | [0.320360, 0.338483]  | 0.018123  | 0.004567")
print(f"  600     |   2s    | {avg_cpc:.6f}   | [{ci_lower:.6f}, {ci_upper:.6f}]  | {(ci_upper-ci_lower):.6f}  | {sem_cpc:.6f}")

# Confidence interval convergence
print("\n=== Standard Error Reduction ===")
sem_10 = 0.015650
sem_100 = 0.004567
print(f"• 10→100 instances: SEM reduced by {sem_10/sem_100:.1f}x")
print(f"• 100→600 instances: SEM reduced by {sem_100/sem_cpc:.1f}x")
print(f"• 10→600 instances: SEM reduced by {sem_10/sem_cpc:.1f}x")

# Save results to CSV
import csv
csv_filename = "results/csv/heuristic_2s_600instances.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['instance_id', 'n', 'cost', 'cpc', 'solve_time', 'seed'])
    for i in range(600):
        writer.writerow([i+1, n, costs[i], cpcs[i], times[i], 7000+i])
print(f"\n✅ Results saved to {csv_filename}")

# Summary statistics for reporting
print("\n=== Summary for Reporting ===")
print(f"• N=20, 600 instances, 2s timeout")
print(f"• Average Cost per Customer: {avg_cpc:.6f} ± {sem_cpc:.6f}")
print(f"• 95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"• Total runtime: {sum(times)/60:.1f} minutes")
print(f"• Average time per instance: {avg_time:.3f}s")

# Statistical stability check
print("\n=== Statistical Stability ===")
# Split into 6 batches of 100
batch_avgs = []
for batch in range(6):
    batch_cpcs = cpcs[batch*100:(batch+1)*100]
    batch_avg = np.mean(batch_cpcs)
    batch_avgs.append(batch_avg)
    print(f"• Batch {batch+1} (instances {batch*100+1}-{(batch+1)*100}): {batch_avg:.6f}")

batch_std = np.std(batch_avgs, ddof=1)
print(f"\n• Batch-to-batch std dev: {batch_std:.6f}")
print(f"• Coefficient of variation: {batch_std/np.mean(batch_avgs)*100:.2f}%")
