#!/usr/bin/env python3
"""
GPU benchmark for multiple N values with 10,000 instances each
Saves results to CSV file on the go with progress bar
"""

import numpy as np
import time
import sys
import json
import csv
from datetime import datetime
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
from solvers.exact_gpu_dp import solve_batch as gpu_solve_batch

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)
instance_config = config['instance_generation']

CAPACITY = instance_config['capacity']
DEMAND_MIN = instance_config['demand_min']
DEMAND_MAX = instance_config['demand_max']
COORD_RANGE = instance_config['coord_range']

def generate_instances_batch(n_customers, start_idx, batch_size):
    """Generate a batch of instances"""
    gen = EnhancedCVRPGenerator(config={})
    instances = []
    for i in range(start_idx, start_idx + batch_size):
        seed = 4242 + n_customers * 1000 + i * 10 + 0
        instance = gen.generate_instance(
            num_customers=n_customers,
            capacity=CAPACITY,
            coord_range=COORD_RANGE,
            demand_range=[DEMAND_MIN, DEMAND_MAX],
            seed=seed,
            instance_type=InstanceType.RANDOM,
            apply_augmentation=False,
        )
        instances.append(instance)
    return instances

def progress_bar(current, total, prefix='', suffix='', length=50):
    """Display progress bar"""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    sys.stdout.write(f'\r{prefix} |{bar}| {100*percent:.1f}% {suffix}')
    sys.stdout.flush()
    if current == total:
        print()

def run_benchmark_for_n(n_customers, num_instances=10000, csv_writer=None):
    """Run GPU benchmark for a specific N value"""
    print(f"\n{'='*70}")
    print(f"Processing N={n_customers} with {num_instances} instances")
    print(f"{'='*70}")
    
    batch_size = 1000
    all_cpcs = []
    total_time = 0
    
    # Process in batches
    for batch_idx in range(0, num_instances, batch_size):
        current_batch_size = min(batch_size, num_instances - batch_idx)
        
        # Update progress bar
        progress_bar(
            batch_idx + current_batch_size, 
            num_instances,
            prefix=f'N={n_customers:2d}',
            suffix=f'Batch {batch_idx//batch_size + 1}/{num_instances//batch_size}'
        )
        
        # Generate and solve batch
        instances = generate_instances_batch(n_customers, batch_idx, current_batch_size)
        start_time = time.time()
        gpu_results = gpu_solve_batch(instances, verbose=False)
        batch_time = time.time() - start_time
        total_time += batch_time
        
        # Calculate CPCs
        batch_cpcs = [r.cost / n_customers for r in gpu_results]
        all_cpcs.extend(batch_cpcs)
    
    # Calculate statistics
    cpcs = np.array(all_cpcs)
    mean_cpc = cpcs.mean()
    std_cpc = cpcs.std()
    sem = std_cpc / np.sqrt(num_instances)
    relative_error = (2 * sem / mean_cpc) * 100
    ci_lower = mean_cpc - 1.96 * sem
    ci_upper = mean_cpc + 1.96 * sem
    
    # Print results
    print(f"\nResults for N={n_customers}:")
    print(f"  Mean CPC:      {mean_cpc:.6f}")
    print(f"  Std CPC:       {std_cpc:.6f}")
    print(f"  SEM:           {sem:.6f}")
    print(f"  2×SEM/Mean:    {relative_error:.2f}%")
    print(f"  95% CI:        [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  Total time:    {total_time:.2f}s")
    print(f"  Time/instance: {total_time/num_instances:.6f}s")
    
    # Save to CSV if writer provided
    if csv_writer:
        csv_writer.writerow({
            'N': n_customers,
            'Instances': num_instances,
            'Mean_CPC': mean_cpc,
            'Std_CPC': std_cpc,
            'SEM': sem,
            'Relative_Error_%': relative_error,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Total_Time': total_time,
            'Time_Per_Instance': total_time/num_instances
        })
    
    return mean_cpc, std_cpc, sem, relative_error

def main():
    """Run benchmarks for N=5 through N=10"""
    n_values = [5, 6, 7, 8, 9, 10]
    num_instances = 10000
    
    # Create CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"gpu_benchmark_results_{timestamp}.csv"
    
    print(f"Starting GPU benchmark for N={n_values}")
    print(f"Results will be saved to: {csv_filename}")
    
    # Open CSV file and write header
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['N', 'Instances', 'Mean_CPC', 'Std_CPC', 'SEM', 
                     'Relative_Error_%', 'CI_Lower', 'CI_Upper', 
                     'Total_Time', 'Time_Per_Instance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()  # Ensure header is written immediately
        
        # Results storage for summary table
        results = []
        
        # Run benchmarks
        start_time = time.time()
        for n in n_values:
            mean, std, sem, rel_err = run_benchmark_for_n(n, num_instances, writer)
            results.append((n, mean, std, sem, rel_err))
            csvfile.flush()  # Save results immediately after each N
        
        total_time = time.time() - start_time
        
        # Print summary table
        print(f"\n{'='*70}")
        print("SUMMARY TABLE")
        print(f"{'='*70}")
        print("\n| N  | Mean CPC | Std CPC | SEM     | 2×SEM/Mean(%) |")
        print("|----|----------|---------|---------|---------------|")
        for n, mean, std, sem, rel_err in results:
            print(f"| {n:2d} | {mean:8.6f} | {std:7.6f} | {sem:7.6f} |        {rel_err:5.2f}% |")
        
        print(f"\nTotal processing time: {total_time:.2f}s")
        print(f"Results saved to: {csv_filename}")

if __name__ == "__main__":
    main()
