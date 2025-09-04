#!/usr/bin/env python3
"""
GPU Heuristic Benchmark using PyTorch-based GLS
Tests GPU-accelerated Guided Local Search for various CVRP problem sizes.
"""

import numpy as np
import time
import sys
import os
import argparse
from typing import Dict, Any, List, Tuple
from tabulate import tabulate

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the canonical generator
from src.generator.generator import _generate_instance

# Import GPU heuristic solver
from src.benchmarking.solvers.gpu.heuristic_gpu_simple import GPUHeuristicSimple, solve_batch


def generate_instances_batch(n_customers: int, capacity: int, batch_size: int,
                            start_idx: int = 0, coord_range: int = 100,
                            demand_range: List[int] = [1, 10]) -> List[Dict]:
    """Generate a batch of CVRP instances."""
    instances = []
    for i in range(start_idx, start_idx + batch_size):
        seed = 42000 + n_customers * 1000 + i
        instance = _generate_instance(
            num_customers=n_customers,
            capacity=capacity,
            coord_range=coord_range,
            demand_range=demand_range,
            seed=seed
        )
        instances.append(instance)
    return instances


def run_benchmark_configuration(n_customers: int, capacity: int, num_instances: int,
                               time_limit: float, batch_size: int = 100,
                               coord_range: int = 100, demand_range: List[int] = [1, 10],
                               verbose: bool = False) -> Dict:
    """
    Run GPU heuristic benchmark for a specific configuration.
    """
    print(f"\n{'='*70}")
    print(f"Configuration: N={n_customers}, Capacity={capacity}, Instances={num_instances}")
    print(f"GLS Time limit: {time_limit}s per instance")
    print(f"Batch size: {batch_size} instances processed in parallel on GPU")
    print(f"{'='*70}")
    
    all_costs = []
    total_start_time = time.time()
    
    # Process in batches
    num_batches = (num_instances + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        current_batch_size = min(batch_size, num_instances - start_idx)
        
        print(f"\n  Processing batch {batch_idx + 1}/{num_batches}: instances {start_idx}-{start_idx + current_batch_size - 1}")
        
        # Generate batch
        gen_start = time.time()
        instances = generate_instances_batch(
            n_customers=n_customers,
            capacity=capacity,
            batch_size=current_batch_size,
            start_idx=start_idx,
            coord_range=coord_range,
            demand_range=demand_range
        )
        gen_time = time.time() - gen_start
        print(f"    Generation time: {gen_time:.2f}s")
        
        # Solve batch on GPU
        solve_start = time.time()
        solutions = solve_batch(instances, verbose=verbose)
        solve_time = time.time() - solve_start
        
        # Extract costs
        batch_costs = [sol.cost for sol in solutions]
        all_costs.extend(batch_costs)
        
        # Progress stats
        batch_cpcs = np.array(batch_costs) / n_customers
        overall_cpcs = np.array(all_costs) / n_customers
        
        print(f"    GPU solving time: {solve_time:.2f}s ({solve_time/current_batch_size:.3f}s per instance)")
        print(f"    Batch mean CPC: {batch_cpcs.mean():.6f}, Overall mean CPC: {overall_cpcs.mean():.6f}")
    
    total_time = time.time() - total_start_time
    
    # Calculate statistics
    cpcs = np.array(all_costs) / n_customers
    results = {
        'n_customers': n_customers,
        'capacity': capacity,
        'instances': len(all_costs),
        'mean_cpc': cpcs.mean(),
        'std_cpc': cpcs.std(),
        'sem': cpcs.std() / np.sqrt(len(cpcs)),
        'total_time': total_time,
        'time_per_instance': total_time / num_instances
    }
    results['2sem_mean_pct'] = (2 * results['sem'] / results['mean_cpc']) * 100
    
    print(f"\n  Total time: {total_time:.2f}s ({total_time/num_instances:.3f}s per instance)")
    print(f"  Solved {len(all_costs)}/{num_instances} instances")
    print(f"  Final mean CPC: {results['mean_cpc']:.6f} ± {results['sem']:.6f}")
    
    return results


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='GPU Heuristic CVRP Benchmark (PyTorch GLS)')
    parser.add_argument('--instances', type=int, default=1000,
                        help='Number of instances to test per configuration (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for GPU processing (default: 100)')
    parser.add_argument('--time-limit', type=float, default=5.0,
                        help='Time limit in seconds for GLS per instance (default: 5.0)')
    parser.add_argument('--configs', type=str, default='all',
                        help='Which configurations to run: "all", "small" (N≤20), "large" (N>20), or comma-separated N values (default: all)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed solver progress')
    
    args = parser.parse_args()
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"{'='*80}")
    else:
        print("\nWARNING: No GPU detected! Will run on CPU (much slower)")
    
    print("\n" + "="*80)
    print("GPU Heuristic CVRP Benchmark (PyTorch-based GLS)")
    print(f"Instances per configuration: {args.instances}")
    print(f"Batch size: {args.batch_size}")
    print(f"Time limit per instance: {args.time_limit} seconds")
    print("="*80)
    
    # Define all possible configurations: (n_customers, capacity)
    all_configs = [
        (10, 20),
        (20, 30),
        (50, 40),
        (100, 50)
    ]
    
    # Select configurations based on args
    if args.configs == 'all':
        configurations = all_configs
    elif args.configs == 'small':
        configurations = [(n, c) for n, c in all_configs if n <= 20]
    elif args.configs == 'large':
        configurations = [(n, c) for n, c in all_configs if n > 20]
    else:
        # Parse comma-separated N values
        try:
            n_values = [int(n.strip()) for n in args.configs.split(',')]
            configurations = [(n, c) for n, c in all_configs if n in n_values]
            if not configurations:
                print(f"Error: No valid configurations found for N values: {n_values}")
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid config specification: {args.configs}")
            sys.exit(1)
    
    print(f"Running configurations: {configurations}")
    
    # Storage for results
    results_all = []
    
    # Run benchmarks
    total_start_time = time.time()
    
    for n_customers, capacity in configurations:
        result = run_benchmark_configuration(
            n_customers=n_customers,
            capacity=capacity,
            num_instances=args.instances,
            time_limit=args.time_limit,
            batch_size=args.batch_size,
            coord_range=100,
            demand_range=[1, 10],
            verbose=args.verbose
        )
        results_all.append(result)
    
    total_time = time.time() - total_start_time
    
    # Print final results table
    print("\n" + "="*80)
    print("FINAL RESULTS - GPU Heuristic GLS (PyTorch)")
    print("="*80)
    
    if results_all:
        table_data = []
        for res in results_all:
            table_data.append([
                res['n_customers'],
                res['capacity'],
                f"{res['instances']:,}",
                f"{res['mean_cpc']:.6f}",
                f"{res['std_cpc']:.6f}",
                f"{res['sem']:.6f}",
                f"{res['2sem_mean_pct']:.4f}%"
            ])
        
        headers = ["N", "Capacity", "Instances", "Mean CPC", "Std CPC", "SEM", "2×SEM/Mean(%)"]
        print(tabulate(table_data, headers=headers, tablefmt="pipe", floatfmt=".6f"))
    
    # Summary
    print("\n" + "="*80)
    print(f"Total benchmark time: {total_time:.2f} seconds")
    if len(configurations) > 0:
        print(f"Average time per configuration: {total_time/len(configurations):.2f} seconds")
    
    # GPU memory stats if available
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print("="*80)
    
    # Save results to file
    import json
    results = {
        'method': 'GPU_Heuristic_GLS',
        'results': results_all,
        'total_time': total_time,
        'configurations': [(n, c) for n, c in configurations],
        'instances_per_config': args.instances,
        'batch_size': args.batch_size,
        'time_limit': args.time_limit,
        'gpu_available': torch.cuda.is_available(),
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f'gpu_heuristic_gls_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
