#!/usr/bin/env python3
"""
Generate comparison table for CPU and GPU benchmarks at different sample sizes
"""

import numpy as np
import time
import sys
import json
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
from solvers.exact_gpu_dp import solve_batch as gpu_solve_batch

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)
instance_config = config['instance_generation']

def generate_instances_batch(n_customers, start_idx, batch_size):
    """Generate a batch of instances"""
    gen = EnhancedCVRPGenerator(config={})
    instances = []
    for i in range(start_idx, start_idx + batch_size):
        seed = 4242 + n_customers * 1000 + i * 10 + 0
        instance = gen.generate_instance(
            num_customers=n_customers,
            capacity=instance_config['capacity'],
            coord_range=instance_config['coord_range'],
            demand_range=[instance_config['demand_min'], instance_config['demand_max']],
            seed=seed,
            instance_type=InstanceType.RANDOM,
            apply_augmentation=False,
        )
        instances.append(instance)
    return instances

def run_gpu_benchmark(n_customers, num_instances, verbose=False):
    """Run GPU benchmark for specified number of instances"""
    batch_size = min(1000, num_instances)
    all_cpcs = []
    
    for batch_idx in range(0, num_instances, batch_size):
        current_batch_size = min(batch_size, num_instances - batch_idx)
        instances = generate_instances_batch(n_customers, batch_idx, current_batch_size)
        gpu_results = gpu_solve_batch(instances, verbose=False)
        batch_cpcs = [r.cost / n_customers for r in gpu_results]
        all_cpcs.extend(batch_cpcs)
    
    cpcs = np.array(all_cpcs)
    mean = cpcs.mean()
    std = cpcs.std()
    sem = std / np.sqrt(num_instances)
    ci_lower = mean - 1.96 * sem
    ci_upper = mean + 1.96 * sem
    
    return mean, std, sem, (ci_lower, ci_upper)

def main():
    n_customers = 6
    
    # CPU result (from your benchmark)
    cpu_mean = 0.464466
    cpu_std = 0.090135
    cpu_n = 1000
    cpu_sem = cpu_std / np.sqrt(cpu_n)
    cpu_ci = (cpu_mean - 1.96 * cpu_sem, cpu_mean + 1.96 * cpu_sem)
    
    # GPU 1000 (already computed)
    gpu_1k_mean = 0.460799
    gpu_1k_std = 0.091148
    gpu_1k_sem = gpu_1k_std / np.sqrt(1000)
    gpu_1k_ci = (gpu_1k_mean - 1.96 * gpu_1k_sem, gpu_1k_mean + 1.96 * gpu_1k_sem)
    
    # GPU 10k (already computed)
    gpu_10k_mean = 0.466432
    gpu_10k_std = 0.089185
    gpu_10k_sem = gpu_10k_std / np.sqrt(10000)
    gpu_10k_ci = (gpu_10k_mean - 1.96 * gpu_10k_sem, gpu_10k_mean + 1.96 * gpu_10k_sem)
    
    # GPU 100k
    print("Computing GPU 100k instances...")
    gpu_100k_mean, gpu_100k_std, gpu_100k_sem, gpu_100k_ci = run_gpu_benchmark(n_customers, 100000)
    
    # Generate table
    print("\n| Solver       | Instances | Mean CPC | Std CPC | SEM     | 95% CI              |")
    print("|--------------|-----------|----------|---------|---------|---------------------|")
    print(f"| CPU          |      1000 | {cpu_mean:8.6f} | {cpu_std:7.6f} | {cpu_sem:7.6f} | [{cpu_ci[0]:7.6f}, {cpu_ci[1]:7.6f}] |")
    print(f"| GPU          |      1000 | {gpu_1k_mean:8.6f} | {gpu_1k_std:7.6f} | {gpu_1k_sem:7.6f} | [{gpu_1k_ci[0]:7.6f}, {gpu_1k_ci[1]:7.6f}] |")
    print(f"| GPU          |     10000 | {gpu_10k_mean:8.6f} | {gpu_10k_std:7.6f} | {gpu_10k_sem:7.6f} | [{gpu_10k_ci[0]:7.6f}, {gpu_10k_ci[1]:7.6f}] |")
    print(f"| GPU          |    100000 | {gpu_100k_mean:8.6f} | {gpu_100k_std:7.6f} | {gpu_100k_sem:7.6f} | [{gpu_100k_ci[0]:7.6f}, {gpu_100k_ci[1]:7.6f}] |")

if __name__ == "__main__":
    main()
