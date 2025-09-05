#!/usr/bin/env python3
"""
Generate LaTeX tables for GPU heuristic results with 100, 1000, and 10000 instances
Matching exact formatting: 4 digits for CPC/Std/SEM, 2 digits for percentage
"""

import sys
import os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.heuristic_gpu_fixed import GPUHeuristicFixed

def benchmark_configuration(n_customers, capacity, num_instances, batch_size=100):
    """Benchmark a specific configuration"""
    print(f"\nBenchmarking N={n_customers}, Capacity={capacity}, Instances={num_instances}")
    print("-" * 50)
    
    start_time = time.time()
    solver = GPUHeuristicFixed()
    
    all_cpcs = []
    
    # Process in batches for memory efficiency
    for batch_start in range(0, num_instances, batch_size):
        batch_end = min(batch_start + batch_size, num_instances)
        current_batch_size = batch_end - batch_start
        
        # Generate batch
        instances = []
        for i in range(batch_start, batch_end):
            instance = _generate_instance(
                num_customers=n_customers,
                capacity=capacity,
                coord_range=100,
                demand_range=[1, 10],
                seed=42000 + n_customers * 1000 + i
            )
            instances.append(instance)
        
        # Solve batch
        solutions = solver.solve_batch(instances, verbose=False)
        
        # Extract CPCs
        batch_cpcs = [sol.cost / n_customers for sol in solutions]
        all_cpcs.extend(batch_cpcs)
        
        if (batch_end % 1000 == 0) or (batch_end == num_instances):
            print(f"  Processed {batch_end}/{num_instances} instances...")
    
    # Calculate statistics
    cpcs = np.array(all_cpcs)
    mean_cpc = cpcs.mean()
    std_cpc = cpcs.std()
    sem = std_cpc / np.sqrt(num_instances)
    sem_pct = (2 * sem / mean_cpc) * 100
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Mean CPC: {mean_cpc:.4f} ± {sem:.4f} (2×SEM/Mean: {sem_pct:.2f}%)")
    
    return mean_cpc, std_cpc, sem, sem_pct

def generate_latex_table(results, num_instances):
    """Generate LaTeX table for a specific number of instances with exact formatting"""
    latex = f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{GPU Heuristic Greedy Performance ({num_instances:,} instances)}}
\\label{{tab:gpu-greedy-{num_instances}}}
\\begin{{tabular}}{{@{{}}c c S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] c@{{}}}}
\\toprule
\\textbf{{N}} & \\textbf{{Capacity}} & {{\\textbf{{Mean CPC}}}} & {{\\textbf{{Std CPC}}}} & {{\\textbf{{SEM}}}} & \\textbf{{2×SEM/Mean(\\%)}} \\\\
\\midrule
"""
    
    for n, cap, mean, std, sem, pct in results:
        # Format exactly as in example: 4 digits for mean/std/sem, 2 digits + % for percentage
        latex += f"{n:3d} & {cap} & {mean:.4f} & {std:.4f} & {sem:.4f} & {pct:.2f}\\% \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table*}"""
    
    return latex

# Configurations
configs = [
    (10, 20),
    (20, 30),
    (50, 40),
    (100, 50)
]

# Test different instance counts
instance_counts = [100, 1000, 10000]

all_results = {}

for num_instances in instance_counts:
    print("\n" + "=" * 70)
    print(f"Running benchmark with {num_instances} instances")
    print("=" * 70)
    
    results = []
    for n_customers, capacity in configs:
        # Adjust batch size based on problem size
        batch_size = min(100 if n_customers <= 50 else 50, num_instances)
        result = benchmark_configuration(n_customers, capacity, num_instances, batch_size)
        results.append((n_customers, capacity) + result)
    
    all_results[num_instances] = results
    
    # Generate and print LaTeX table
    latex_table = generate_latex_table(results, num_instances)
    
    print(f"\n{'='*70}")
    print(f"LaTeX Table for {num_instances} instances:")
    print('='*70)
    print(latex_table)

# Save all tables to a file
with open('gpu_heuristic_latex_tables.tex', 'w') as f:
    f.write("% GPU Heuristic Benchmark Results\n")
    f.write("% Generated automatically\n")
    f.write("% Format: 4 digits for Mean CPC, Std CPC, SEM; 2 digits for percentage\n\n")
    
    for num_instances in instance_counts:
        latex_table = generate_latex_table(all_results[num_instances], num_instances)
        f.write(f"\n% {num_instances} instances\n")
        f.write(latex_table)
        f.write("\n\n")

print("\n" + "="*70)
print("All LaTeX tables saved to: gpu_heuristic_latex_tables.tex")
print("="*70)

# Print convergence summary
print("\nConvergence Summary (Mean CPC):")
print("-" * 60)
print(f"{'N':>4} {'Capacity':>10} {'100 inst':>12} {'1000 inst':>12} {'10000 inst':>12}")
print("-" * 60)
for i, (n, cap) in enumerate(configs):
    means = [all_results[num][i][2] for num in instance_counts]
    print(f"{n:4d} {cap:10d} {means[0]:12.4f} {means[1]:12.4f} {means[2]:12.4f}")

print("\nConvergence of 2×SEM/Mean(%):")
print("-" * 60)
print(f"{'N':>4} {'Capacity':>10} {'100 inst':>12} {'1000 inst':>12} {'10000 inst':>12}")
print("-" * 60)
for i, (n, cap) in enumerate(configs):
    sems = [all_results[num][i][5] for num in instance_counts]
    print(f"{n:4d} {cap:10d} {sems[0]:11.2f}% {sems[1]:11.2f}% {sems[2]:11.2f}%")
