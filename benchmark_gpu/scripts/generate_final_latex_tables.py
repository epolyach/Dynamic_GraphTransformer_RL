#!/usr/bin/env python3
"""
Generate final LaTeX tables with correct formatting and improved solver
"""

import sys
import os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.gpu.heuristic_gpu_improved import GPUHeuristicImproved

def benchmark_configuration(n_customers, capacity, num_instances, batch_size=100):
    """Benchmark a specific configuration"""
    print(f"\nN={n_customers}, Capacity={capacity}, Instances={num_instances}")
    print("-" * 40)
    
    start_time = time.time()
    solver = GPUHeuristicImproved()
    
    all_cpcs = []
    
    # Process in batches for memory efficiency
    for batch_start in range(0, num_instances, batch_size):
        batch_end = min(batch_start + batch_size, num_instances)
        
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
        
        # Solve batch with improved algorithm
        max_iter = 30 if n_customers <= 20 else (20 if n_customers <= 50 else 10)
        solutions = solver.solve_batch(instances, max_iterations=max_iter, verbose=False)
        
        # Extract CPCs
        batch_cpcs = [sol.cost / n_customers for sol in solutions]
        all_cpcs.extend(batch_cpcs)
        
        if (batch_end % 500 == 0) or (batch_end == num_instances):
            elapsed = time.time() - start_time
            mean_so_far = np.mean(all_cpcs)
            print(f"  {batch_end}/{num_instances}: Mean CPC = {mean_so_far:.4f} ({elapsed:.1f}s)")
    
    # Calculate final statistics
    cpcs = np.array(all_cpcs)
    mean_cpc = cpcs.mean()
    std_cpc = cpcs.std()
    sem = std_cpc / np.sqrt(num_instances)
    sem_pct = (2 * sem / mean_cpc) * 100
    
    elapsed = time.time() - start_time
    print(f"  Final: Mean CPC = {mean_cpc:.4f}, 2×SEM/Mean = {sem_pct:.2f}%")
    print(f"  Time: {elapsed:.1f}s")
    
    return mean_cpc, std_cpc, sem, sem_pct

def generate_latex_table(results, num_instances):
    """Generate LaTeX table with exact formatting"""
    latex = f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{GPU Improved Heuristic Performance ({num_instances:,} instances)}}
\\label{{tab:gpu-heuristic-{num_instances}}}
\\begin{{tabular}}{{@{{}}c c S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] c@{{}}}}
\\toprule
\\textbf{{N}} & \\textbf{{Capacity}} & {{\\textbf{{Mean CPC}}}} & {{\\textbf{{Std CPC}}}} & {{\\textbf{{SEM}}}} & \\textbf{{2×SEM/Mean(\\%)}} \\\\
\\midrule
"""
    
    for n, cap, mean, std, sem, pct in results:
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

# Test with smaller numbers first for verification
print("=" * 60)
print("Generating GPU Heuristic LaTeX Tables")
print("=" * 60)

# Quick test with 10 instances to verify results are correct
print("\nQuick verification with 10 instances:")
print("-" * 40)
test_results = []
for n, cap in configs:
    result = benchmark_configuration(n, cap, 10, batch_size=10)
    test_results.append((n, cap) + result)
    
print("\nVerification Summary:")
for n, cap, mean, _, _, _ in test_results:
    print(f"N={n:3d}: Mean CPC = {mean:.4f}")

# Now generate full tables
instance_counts = [100, 1000, 10000]
all_results = {}

for num_instances in instance_counts:
    print("\n" + "=" * 60)
    print(f"Benchmarking with {num_instances} instances")
    print("=" * 60)
    
    results = []
    for n_customers, capacity in configs:
        batch_size = min(50 if n_customers <= 20 else (20 if n_customers <= 50 else 10), num_instances)
        result = benchmark_configuration(n_customers, capacity, num_instances, batch_size)
        results.append((n_customers, capacity) + result)
    
    all_results[num_instances] = results
    
    # Generate and print LaTeX table
    latex_table = generate_latex_table(results, num_instances)
    print(f"\nLaTeX Table ({num_instances} instances):")
    print(latex_table)

# Save all tables to file
with open('gpu_heuristic_final_tables.tex', 'w') as f:
    f.write("% GPU Improved Heuristic Benchmark Results\n")
    f.write("% Format: 4 digits for Mean CPC, Std CPC, SEM; 2 digits for percentage\n\n")
    
    for num_instances in instance_counts:
        latex_table = generate_latex_table(all_results[num_instances], num_instances)
        f.write(f"% {num_instances} instances\n")
        f.write(latex_table)
        f.write("\n\n")

print("\n" + "="*60)
print("Tables saved to: gpu_heuristic_final_tables.tex")
print("="*60)

# Print convergence summary
print("\nConvergence Analysis:")
print("-" * 70)
print(f"{'N':>4} {'100 inst':>12} {'1000 inst':>12} {'10000 inst':>12} {'Convergence':>15}")
print("-" * 70)
for i, (n, cap) in enumerate(configs):
    if all(num in all_results for num in instance_counts):
        means = [all_results[num][i][2] for num in instance_counts]
        conv = (means[0] - means[2]) / means[0] * 100 if means[0] > 0 else 0
        print(f"{n:4d} {means[0]:12.4f} {means[1]:12.4f} {means[2]:12.4f} {conv:14.2f}%")
