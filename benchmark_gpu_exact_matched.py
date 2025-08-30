#!/usr/bin/env python3
"""
GPU benchmark using EXACT same configuration as benchmark_exact_cpu_modified.py
Loads config.json and uses identical instance generation
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

def generate_instances_for_n(n_customers, num_instances):
    """
    Generate instances EXACTLY as benchmark_exact_cpu_modified.py does
    Uses EnhancedCVRPGenerator with same seeds and parameters
    """
    gen = EnhancedCVRPGenerator(config={})
    instances = []
    
    for i in range(num_instances):
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

def run_benchmark(n_customers, num_instances):
    """Run GPU benchmark on exact same instances as CPU"""
    print(f"\nGPU Benchmark: N={n_customers}, {num_instances} instances")
    print("=" * 60)
    
    # Generate instances with exact same method as CPU
    instances = generate_instances_for_n(n_customers, num_instances)
    
    # Verify configuration
    print(f"Configuration (from config.json):")
    print(f"  Capacity: {CAPACITY}")
    print(f"  Demands: [{DEMAND_MIN}, {DEMAND_MAX}]")
    print(f"  Coord range: {COORD_RANGE}")
    print(f"  First instance seed: {4242 + n_customers * 1000}")
    
    # Run GPU solver
    start_time = time.time()
    gpu_results = gpu_solve_batch(instances, verbose=False)
    gpu_time = time.time() - start_time
    
    # Calculate CPCs
    cpcs = np.array([r.cost / n_customers for r in gpu_results])
    
    # Statistics
    mean_cpc = cpcs.mean()
    std_cpc = cpcs.std()
    sem = std_cpc / np.sqrt(num_instances)
    
    # 95% confidence interval
    ci_lower = mean_cpc - 1.96 * sem
    ci_upper = mean_cpc + 1.96 * sem
    
    print(f"\nResults:")
    print(f"  Mean CPC: {mean_cpc:.6f}")
    print(f"  Std CPC:  {std_cpc:.6f}")
    print(f"  SEM:      {sem:.6f}")
    print(f"  95% CI:   [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  Total time: {gpu_time:.2f}s")
    print(f"  Time per instance: {gpu_time/num_instances:.6f}s")
    
    return mean_cpc, std_cpc, sem, (ci_lower, ci_upper)

def compare_with_cpu(cpu_mean, cpu_std, cpu_n, gpu_mean, gpu_std, gpu_n):
    """Compare CPU and GPU results with confidence intervals"""
    
    # Calculate SEMs
    cpu_sem = cpu_std / np.sqrt(cpu_n)
    gpu_sem = gpu_std / np.sqrt(gpu_n)
    
    # 95% confidence intervals
    cpu_ci = (cpu_mean - 1.96 * cpu_sem, cpu_mean + 1.96 * cpu_sem)
    gpu_ci = (gpu_mean - 1.96 * gpu_sem, gpu_mean + 1.96 * gpu_sem)
    
    print("\n" + "=" * 60)
    print("COMPARISON WITH CPU BENCHMARK")
    print("=" * 60)
    
    print(f"\nCPU (benchmark_exact_cpu_modified.py):")
    print(f"  Mean CPC: {cpu_mean:.6f}")
    print(f"  Std CPC:  {cpu_std:.6f}")
    print(f"  SEM:      {cpu_sem:.6f}")
    print(f"  95% CI:   [{cpu_ci[0]:.6f}, {cpu_ci[1]:.6f}]")
    
    print(f"\nGPU (this benchmark):")
    print(f"  Mean CPC: {gpu_mean:.6f}")
    print(f"  Std CPC:  {gpu_std:.6f}")
    print(f"  SEM:      {gpu_sem:.6f}")
    print(f"  95% CI:   [{gpu_ci[0]:.6f}, {gpu_ci[1]:.6f}]")
    
    # Check overlap
    overlap = not (cpu_ci[1] < gpu_ci[0] or gpu_ci[1] < cpu_ci[0])
    
    print(f"\nInterval overlap: {overlap}")
    if overlap:
        # Calculate overlap region
        overlap_start = max(cpu_ci[0], gpu_ci[0])
        overlap_end = min(cpu_ci[1], gpu_ci[1])
        overlap_width = overlap_end - overlap_start
        print(f"  Overlap region: [{overlap_start:.6f}, {overlap_end:.6f}]")
        print(f"  Overlap width: {overlap_width:.6f}")
    else:
        gap = max(gpu_ci[0] - cpu_ci[1], cpu_ci[0] - gpu_ci[1])
        print(f"  Gap between intervals: {gap:.6f}")
    
    print(f"\nDifference in means: {abs(cpu_mean - gpu_mean):.6f}")
    print(f"Relative to SEM: {abs(cpu_mean - gpu_mean) / max(cpu_sem, gpu_sem):.2f} × SEM")
    
    # Statistical test
    pooled_sem = np.sqrt(cpu_sem**2 + gpu_sem**2)
    z_stat = abs(cpu_mean - gpu_mean) / pooled_sem
    
    print(f"\nStatistical significance:")
    print(f"  Z-statistic: {z_stat:.4f}")
    print(f"  Significant at α=0.05? {'No' if z_stat < 1.96 else 'Yes'}")
    print(f"  Significant at α=0.01? {'No' if z_stat < 2.576 else 'Yes'}")

def main():
    # Test with N=6, 1000 instances to match your CPU benchmark
    gpu_mean, gpu_std, gpu_sem, gpu_ci = run_benchmark(6, 1000)
    
    # Your reported CPU results
    cpu_mean = 0.464466
    cpu_std = 0.090135
    cpu_n = 1000
    
    # Compare
    compare_with_cpu(cpu_mean, cpu_std, cpu_n, gpu_mean, gpu_std, 1000)
    
    # Also test with smaller sample to verify consistency
    print("\n" + "=" * 60)
    print("VERIFICATION WITH SMALLER SAMPLE (N=6, 100 instances)")
    print("=" * 60)
    gpu_mean_100, gpu_std_100, _, _ = run_benchmark(6, 100)
    
    print(f"\nConsistency check:")
    print(f"  1000 instances: Mean={gpu_mean:.6f}, Std={gpu_std:.6f}")
    print(f"  100 instances:  Mean={gpu_mean_100:.6f}, Std={gpu_std_100:.6f}")
    print(f"  Difference in means: {abs(gpu_mean - gpu_mean_100):.6f}")

if __name__ == "__main__":
    main()
