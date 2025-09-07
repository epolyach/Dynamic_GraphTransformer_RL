#!/usr/bin/env python3
"""
CPU benchmark with only exact solvers:
- exact_dp: Dynamic programming exact solver (N ≤ 8)
- ortools_greedy: OR-Tools greedy/exact solver

Testing N=5,6,7 with 10,000 instances each
"""

import sys
import os
sys.path.append('research/benchmark_exact')

import numpy as np
import time
import json
import csv
from datetime import datetime

# Import solvers
import solvers.exact_dp as exact_dp
import solvers.exact.ortools_greedy as ortools_greedy
from enhanced_generator import EnhancedCVRPGenerator, InstanceType

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)
instance_config = config['instance_generation']

CAPACITY = instance_config['capacity']
DEMAND_MIN = instance_config['demand_min']
DEMAND_MAX = instance_config['demand_max']
COORD_RANGE = instance_config['coord_range']

def progress_bar(current, total, prefix='', suffix='', length=50):
    """Display progress bar"""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    sys.stdout.write(f'\r{prefix} |{bar}| {100*percent:.1f}% {suffix}')
    sys.stdout.flush()
    if current == total:
        print()

def generate_instance(n_customers, instance_idx):
    """Generate instance exactly as other benchmarks"""
    gen = EnhancedCVRPGenerator(config={})
    seed = 4242 + n_customers * 1000 + instance_idx * 10 + 0  # attempt=0
    
    instance = gen.generate_instance(
        num_customers=n_customers,
        capacity=CAPACITY,
        coord_range=COORD_RANGE,
        demand_range=[DEMAND_MIN, DEMAND_MAX],
        seed=seed,
        instance_type=InstanceType.RANDOM,
        apply_augmentation=False,
    )
    return instance

def run_solver_on_instance(solver_module, solver_name, instance, timeout=60.0):
    """Run a solver on a single instance"""
    start_time = time.time()
    try:
        solution = solver_module.solve(instance, time_limit=timeout, verbose=False)
        solve_time = time.time() - start_time
        
        if solution is None:
            return None, solve_time, True  # Timed out
        
        return solution, solve_time, False
        
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"\n  {solver_name} error: {str(e)[:50]}")
        return None, solve_time, False

def run_benchmark_for_n(n_customers, num_instances=10000, csv_writer=None):
    """Run benchmark for a specific N value"""
    print(f"\n{'='*70}")
    print(f"Processing N={n_customers} with {num_instances:,} instances")
    print(f"{'='*70}")
    
    # Define solvers
    solvers = []
    
    # Only include exact_dp for N ≤ 8
    if n_customers <= 8:
        solvers.append(('exact_dp', exact_dp))
    
    # Always include ortools_greedy
    solvers.append(('ortools_greedy', ortools_greedy))
    
    print(f"Active solvers: {[name for name, _ in solvers]}")
    
    # Results storage
    results = {solver_name: {'cpcs': [], 'times': [], 'failures': 0} 
               for solver_name, _ in solvers}
    
    # Process instances
    batch_size = 100
    for batch_start in range(0, num_instances, batch_size):
        batch_end = min(batch_start + batch_size, num_instances)
        batch_size_actual = batch_end - batch_start
        
        # Update progress
        progress_bar(
            batch_end,
            num_instances,
            prefix=f'N={n_customers:2d}',
            suffix=f'Batch {batch_start//batch_size + 1}/{(num_instances + batch_size - 1)//batch_size}'
        )
        
        # Process batch
        for i in range(batch_start, batch_end):
            instance = generate_instance(n_customers, i)
            
            for solver_name, solver_module in solvers:
                timeout = 30.0 if solver_name == 'exact_dp' else 10.0
                
                solution, solve_time, timed_out = run_solver_on_instance(
                    solver_module, solver_name, instance, timeout
                )
                
                if solution:
                    results[solver_name]['cpcs'].append(solution.cost / n_customers)
                    results[solver_name]['times'].append(solve_time)
                else:
                    results[solver_name]['failures'] += 1
    
    # Calculate statistics and print results
    print(f"\nResults for N={n_customers}:")
    print("-" * 60)
    
    summary_data = []
    
    for solver_name, _ in solvers:
        cpcs = np.array(results[solver_name]['cpcs'])
        times = np.array(results[solver_name]['times'])
        failures = results[solver_name]['failures']
        success_rate = len(cpcs) / num_instances * 100
        
        if len(cpcs) > 0:
            mean_cpc = cpcs.mean()
            std_cpc = cpcs.std()
            sem = std_cpc / np.sqrt(len(cpcs))
            relative_error = (2 * sem / mean_cpc) * 100
            mean_time = times.mean()
            
            print(f"\n{solver_name}:")
            print(f"  Success rate:  {success_rate:.1f}% ({len(cpcs)}/{num_instances})")
            print(f"  Mean CPC:      {mean_cpc:.6f}")
            print(f"  Std CPC:       {std_cpc:.6f}")
            print(f"  SEM:           {sem:.6f}")
            print(f"  2×SEM/Mean:    {relative_error:.4f}%")
            print(f"  Mean time:     {mean_time:.4f}s")
            
            summary_data.append({
                'solver': solver_name,
                'mean_cpc': mean_cpc,
                'std_cpc': std_cpc,
                'sem': sem,
                'rel_error': relative_error,
                'mean_time': mean_time,
                'success_rate': success_rate
            })
            
            # Save to CSV if writer provided
            if csv_writer:
                csv_writer.writerow({
                    'N': n_customers,
                    'Solver': solver_name,
                    'Instances_Total': num_instances,
                    'Instances_Solved': len(cpcs),
                    'Success_Rate': success_rate,
                    'Mean_CPC': mean_cpc,
                    'Std_CPC': std_cpc,
                    'SEM': sem,
                    'Relative_Error_%': relative_error,
                    'Mean_Time': mean_time,
                    'Total_Time': times.sum()
                })
        else:
            print(f"\n{solver_name}: All instances failed!")
    
    return summary_data

def main():
    """Run CPU benchmark for N=5,6,7 with 10,000 instances each"""
    n_values = [5, 6, 7]
    num_instances = 10000
    
    # Create CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"cpu_exact_benchmark_{timestamp}.csv"
    
    print(f"Starting CPU benchmark with exact solvers only")
    print(f"N values: {n_values}")
    print(f"Instances per N: {num_instances:,}")
    print(f"Results will be saved to: {csv_filename}")
    
    # Open CSV file and write header
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['N', 'Solver', 'Instances_Total', 'Instances_Solved', 
                     'Success_Rate', 'Mean_CPC', 'Std_CPC', 'SEM', 
                     'Relative_Error_%', 'Mean_Time', 'Total_Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()
        
        # Results storage
        all_results = {}
        
        # Run benchmarks
        start_time = time.time()
        for n in n_values:
            summary = run_benchmark_for_n(n, num_instances, writer)
            all_results[n] = summary
            csvfile.flush()
        
        total_time = time.time() - start_time
        
        # Print comparison table
        print(f"\n{'='*80}")
        print("COMPARISON TABLE")
        print(f"{'='*80}")
        
        # For each solver, show results across N values
        solver_names = set()
        for n in n_values:
            for result in all_results[n]:
                solver_names.add(result['solver'])
        
        for solver_name in sorted(solver_names):
            print(f"\n{solver_name}:")
            print("| N | Mean CPC | Std CPC | SEM     | 2×SEM/Mean(%) | Mean Time |")
            print("|---|----------|---------|---------|---------------|-----------|")
            
            for n in n_values:
                for result in all_results[n]:
                    if result['solver'] == solver_name:
                        print(f"| {n} | {result['mean_cpc']:8.6f} | {result['std_cpc']:7.6f} | "
                              f"{result['sem']:7.6f} |      {result['rel_error']:7.4f}% | {result['mean_time']:9.4f}s |")
        
        # Compare solvers at each N
        print(f"\n{'='*80}")
        print("SOLVER COMPARISON")
        print(f"{'='*80}")
        
        for n in n_values:
            print(f"\nN={n}:")
            if len(all_results[n]) >= 2:
                dp_result = next((r for r in all_results[n] if r['solver'] == 'exact_dp'), None)
                or_result = next((r for r in all_results[n] if r['solver'] == 'ortools_greedy'), None)
                
                if dp_result and or_result:
                    diff = abs(dp_result['mean_cpc'] - or_result['mean_cpc'])
                    rel_diff = diff / min(dp_result['mean_cpc'], or_result['mean_cpc']) * 100
                    
                    print(f"  exact_dp CPC:      {dp_result['mean_cpc']:.6f}")
                    print(f"  ortools_greedy CPC: {or_result['mean_cpc']:.6f}")
                    print(f"  Difference:         {diff:.6f} ({rel_diff:.4f}%)")
                    print(f"  Speedup:            {dp_result['mean_time']/or_result['mean_time']:.1f}x (ortools faster)")
        
        print(f"\nTotal processing time: {total_time:.2f}s")
        print(f"Results saved to: {csv_filename}")

if __name__ == "__main__":
    main()
