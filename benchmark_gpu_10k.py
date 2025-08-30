#!/usr/bin/env python3
"""
GPU benchmark with 10,000 instances for high precision CPC estimation
Uses exact same configuration as benchmark_exact_cpu_modified.py
"""

import numpy as np
import time
import sys
import json
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
from solvers.exact_gpu_dp import solve_batch as gpu_solve_batch

# Load config.json exactly as CPU benchmark does
with open('config.json', 'r') as f:
    config = json.load(f)

instance_config = config['instance_generation']
CAPACITY = instance_config['capacity']  # 30
DEMAND_MIN = instance_config['demand_min']  # 1
DEMAND_MAX = instance_config['demand_max']  # 10
COORD_RANGE = instance_config['coord_range']  # 100

def generate_instances_batch(n_customers, start_idx, batch_size):
    """Generate a batch of instances"""
    gen = EnhancedCVRPGenerator(config={})
    instances = []
    
    for i in range(start_idx, start_idx + batch_size):
        # EXACT same seed formula as CPU benchmark
        seed = 4242 + n_customers * 1000 + i * 10 + 0  # attempt=0
        
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

def run_benchmark_10k(n_customers=6):
    """Run GPU benchmark on 10,000 instances"""
    num_instances = 10000
    batch_size = 1000  # Process in batches to manage memory
    
    print(f"\nGPU Benchmark: N={n_customers}, {num_instances} instances")
    print("=" * 70)
    
    print(f"Configuration (from config.json):")
    print(f"  Capacity: {CAPACITY}")
    print(f"  Demands: [{DEMAND_MIN}, {DEMAND_MAX}]")
    print(f"  Coord range: {COORD_RANGE}")
    print(f"  Processing in batches of {batch_size}")
    
    all_cpcs = []
    total_time = 0
    
    # Process in batches
    for batch_idx in range(0, num_instances, batch_size):
        current_batch_size = min(batch_size, num_instances - batch_idx)
        print(f"\nProcessing batch {batch_idx//batch_size + 1}/{num_instances//batch_size}: instances {batch_idx}-{batch_idx+current_batch_size-1}")
        
        # Generate batch
        instances = generate_instances_batch(n_customers, batch_idx, current_batch_size)
        
        # Solve batch
        start_time = time.time()
        gpu_results = gpu_solve_batch(instances, verbose=False)
        batch_time = time.time() - start_time
        total_time += batch_time
        
        # Calculate CPCs for this batch
        batch_cpcs = [r.cost / n_customers for r in gpu_results]
        all_cpcs.extend(batch_cpcs)
        
        print(f"  Batch time: {batch_time:.2f}s, Mean CPC: {np.mean(batch_cpcs):.6f}")
    
    # Overall statistics
    cpcs = np.array(all_cpcs)
    mean_cpc = cpcs.mean()
    std_cpc = cpcs.std()
    sem = std_cpc / np.sqrt(num_instances)
    
    # 95% confidence interval
    ci_lower = mean_cpc - 1.96 * sem
    ci_upper = mean_cpc + 1.96 * sem
    
    # 99% confidence interval
    ci99_lower = mean_cpc - 2.576 * sem
    ci99_upper = mean_cpc + 2.576 * sem
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS (10,000 instances)")
    print("=" * 70)
    
    print(f"\nStatistics:")
    print(f"  Mean CPC:      {mean_cpc:.6f}")
    print(f"  Std CPC:       {std_cpc:.6f}")
    print(f"  SEM:           {sem:.6f}")
    print(f"  95% CI:        [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  99% CI:        [{ci99_lower:.6f}, {ci99_upper:.6f}]")
    print(f"  CI width (95%): {ci_upper - ci_lower:.6f}")
    print(f"  CI width (99%): {ci99_upper - ci99_lower:.6f}")
    
    print(f"\nPerformance:")
    print(f"  Total time:    {total_time:.2f}s")
    print(f"  Time/instance: {total_time/num_instances:.6f}s")
    print(f"  Instances/sec: {num_instances/total_time:.1f}")
    
    # Compare with CPU benchmark (1000 instances)
    cpu_mean = 0.464466
    cpu_std = 0.090135
    cpu_n = 1000
    cpu_sem = cpu_std / np.sqrt(cpu_n)
    cpu_ci = (cpu_mean - 1.96 * cpu_sem, cpu_mean + 1.96 * cpu_sem)
    
    print("\n" + "=" * 70)
    print("COMPARISON WITH CPU BENCHMARK")
    print("=" * 70)
    
    print(f"\nCPU (N=6, 1000 instances):")
    print(f"  Mean CPC: {cpu_mean:.6f}")
    print(f"  95% CI:   [{cpu_ci[0]:.6f}, {cpu_ci[1]:.6f}]")
    print(f"  CI width: {cpu_ci[1] - cpu_ci[0]:.6f}")
    
    print(f"\nGPU (N=6, 10,000 instances):")
    print(f"  Mean CPC: {mean_cpc:.6f}")
    print(f"  95% CI:   [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  CI width: {ci_upper - ci_lower:.6f}")
    
    # Check overlap
    overlap = not (cpu_ci[1] < ci_lower or ci_upper < cpu_ci[0])
    
    print(f"\nInterval overlap: {overlap}")
    if overlap:
        overlap_start = max(cpu_ci[0], ci_lower)
        overlap_end = min(cpu_ci[1], ci_upper)
        overlap_width = overlap_end - overlap_start
        print(f"  Overlap region: [{overlap_start:.6f}, {overlap_end:.6f}]")
        print(f"  Overlap width:  {overlap_width:.6f}")
    
    print(f"\nDifference in means: {abs(cpu_mean - mean_cpc):.6f}")
    print(f"Relative to GPU SEM: {abs(cpu_mean - mean_cpc) / sem:.2f} × SEM")
    
    # With 10k samples, we have much higher precision
    print(f"\nPrecision improvement:")
    print(f"  SEM reduction:    {cpu_sem/sem:.1f}× smaller with 10k samples")
    print(f"  CI width reduction: {(cpu_ci[1]-cpu_ci[0])/(ci_upper-ci_lower):.1f}× narrower")
    
    return mean_cpc, std_cpc, sem

if __name__ == "__main__":
    mean, std, sem = run_benchmark_10k(n_customers=6)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nGPU with 10,000 instances provides high-precision estimate:")
    print(f"  CPC = {mean:.6f} ± {1.96*sem:.6f} (95% CI)")
    print(f"  Relative uncertainty: ±{1.96*sem/mean*100:.2f}%")
