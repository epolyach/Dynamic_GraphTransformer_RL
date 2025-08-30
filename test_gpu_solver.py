#!/usr/bin/env python3
"""
Test GPU-based CVRP solver and compare with OR-Tools greedy solver.
"""

import numpy as np
import torch
import time
import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

# Import solvers
from solvers.exact_gpu_improved import ImprovedGPUCVRPSolver as GPUCVRPSolver
from solvers.exact.ortools_greedy import solve as ortools_solve
from solvers.types import CVRPSolution


def generate_cvrp_instance(n_customers: int, seed: int = None) -> Dict[str, Any]:
    """Generate a CVRP instance with n customers."""
    if seed is not None:
        np.random.seed(seed)
    
    n = n_customers + 1  # Including depot
    
    # Generate random coordinates in [0, 1]
    coords = np.random.uniform(0, 1, size=(n, 2))
    coords[0] = [0.5, 0.5]  # Depot at center
    
    # Generate demands (depot has 0 demand)
    demands = np.zeros(n, dtype=np.float32)
    demands[1:] = np.random.uniform(1, 10, size=n_customers)
    
    # Vehicle capacity (ensure feasibility)
    capacity = max(demands.sum() / 2, demands.max() * 2)  # Heuristic for capacity
    
    # Calculate distance matrix (Euclidean distances)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    return {
        'coords': coords,
        'demands': demands,
        'distances': distances,
        'capacity': capacity,
        'n_customers': n_customers
    }


def test_gpu_vs_ortools(n_customers: int = 6, n_instances: int = 1000, batch_size: int = 100):
    """
    Test GPU solver against OR-Tools on multiple instances.
    
    Args:
        n_customers: Number of customers in each instance
        n_instances: Total number of instances to test
        batch_size: Number of instances to solve simultaneously on GPU
    """
    print(f"\n{'='*80}")
    print(f"TESTING GPU SOLVER VS OR-TOOLS GREEDY")
    print(f"{'='*80}")
    print(f"Problem size: {n_customers} customers")
    print(f"Number of instances: {n_instances}")
    print(f"GPU batch size: {batch_size}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Generate instances
    print("Generating instances...")
    instances = []
    for i in range(n_instances):
        instances.append(generate_cvrp_instance(n_customers, seed=5000 + i))
    
    # Test GPU solver
    print("\n" + "-"*40)
    print("Testing GPU Solver")
    print("-"*40)
    
    gpu_solver = GPUCVRPSolver()
    gpu_results = []
    gpu_start = time.time()
    
    # Process in batches
    for batch_start in range(0, n_instances, batch_size):
        batch_end = min(batch_start + batch_size, n_instances)
        batch = instances[batch_start:batch_end]
        
        if batch_start == 0:  # Verbose for first batch
            print(f"Processing batch 1/{(n_instances + batch_size - 1) // batch_size}...")
            batch_results = gpu_solver.solve_batch(batch, verbose=True)
        else:
            if batch_start % (batch_size * 10) == 0:
                print(f"  Progress: {batch_start}/{n_instances} instances")
            batch_results = gpu_solver.solve_batch(batch, verbose=False)
        
        gpu_results.extend(batch_results)
    
    gpu_time = time.time() - gpu_start
    
    # Calculate GPU statistics
    gpu_costs = np.array([r.cost for r in gpu_results])
    gpu_cpcs = np.array([r.cost / n_customers for r in gpu_results])
    
    print(f"\nGPU Results:")
    print(f"  Total time: {gpu_time:.3f}s")
    print(f"  Time per instance: {gpu_time/n_instances*1000:.3f}ms")
    print(f"  Mean cost: {gpu_costs.mean():.4f}")
    print(f"  Std cost: {gpu_costs.std():.4f}")
    print(f"  Mean CPC: {gpu_cpcs.mean():.6f}")
    print(f"  Std CPC: {gpu_cpcs.std():.6f}")
    
    # Test OR-Tools solver (sample for comparison)
    print("\n" + "-"*40)
    print("Testing OR-Tools Greedy Solver")
    print("-"*40)
    
    # Test on a sample (OR-Tools is slower, so test fewer instances)
    sample_size = min(100, n_instances)
    sample_indices = np.random.choice(n_instances, sample_size, replace=False)
    
    ortools_results = []
    ortools_start = time.time()
    
    for idx, i in enumerate(sample_indices):
        if idx == 0:
            print(f"Processing instance 1/{sample_size}...")
        elif idx % 10 == 0:
            print(f"  Progress: {idx}/{sample_size} instances")
        
        result = ortools_solve(instances[i], verbose=False)
        ortools_results.append(result)
    
    ortools_time = time.time() - ortools_start
    
    # Calculate OR-Tools statistics
    ortools_costs = np.array([r.cost for r in ortools_results])
    ortools_cpcs = np.array([r.cost / n_customers for r in ortools_results])
    
    print(f"\nOR-Tools Results (sample of {sample_size}):")
    print(f"  Total time: {ortools_time:.3f}s")
    print(f"  Time per instance: {ortools_time/sample_size*1000:.3f}ms")
    print(f"  Mean cost: {ortools_costs.mean():.4f}")
    print(f"  Std cost: {ortools_costs.std():.4f}")
    print(f"  Mean CPC: {ortools_cpcs.mean():.6f}")
    print(f"  Std CPC: {ortools_cpcs.std():.6f}")
    
    # Direct comparison on same instances
    print("\n" + "-"*40)
    print("Direct Comparison (same instances)")
    print("-"*40)
    
    comparison_results = []
    for i in sample_indices[:10]:  # Compare first 10 sampled instances
        gpu_cost = gpu_results[i].cost
        ortools_cost = ortools_results[sample_indices.tolist().index(i)].cost
        diff = abs(gpu_cost - ortools_cost)
        comparison_results.append({
            'instance': i,
            'gpu_cost': gpu_cost,
            'ortools_cost': ortools_cost,
            'difference': diff,
            'relative_diff': diff / ortools_cost if ortools_cost > 0 else 0
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    print(comparison_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nPerformance comparison:")
    print(f"  GPU speedup: {(ortools_time/sample_size) / (gpu_time/n_instances):.1f}x faster per instance")
    print(f"  GPU throughput: {n_instances/gpu_time:.1f} instances/second")
    print(f"  OR-Tools throughput: {sample_size/ortools_time:.1f} instances/second")
    
    print(f"\nSolution quality comparison (on sample):")
    avg_diff = comparison_df['relative_diff'].mean()
    print(f"  Average relative difference: {avg_diff:.2%}")
    print(f"  Max relative difference: {comparison_df['relative_diff'].max():.2%}")
    
    if avg_diff < 0.01:
        print("  ✅ GPU solver produces nearly identical results to OR-Tools!")
    elif avg_diff < 0.05:
        print("  ✅ GPU solver produces comparable results to OR-Tools (within 5%)")
    else:
        print(f"  ⚠️  GPU solver differs from OR-Tools by {avg_diff:.1%} on average")
    
    # Save results
    results_dir = Path("results/gpu_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        'n_customers': n_customers,
        'n_instances': n_instances,
        'batch_size': batch_size,
        'gpu': {
            'total_time': gpu_time,
            'time_per_instance': gpu_time / n_instances,
            'mean_cost': float(gpu_costs.mean()),
            'std_cost': float(gpu_costs.std()),
            'mean_cpc': float(gpu_cpcs.mean()),
            'std_cpc': float(gpu_cpcs.std()),
        },
        'ortools_sample': {
            'sample_size': sample_size,
            'total_time': ortools_time,
            'time_per_instance': ortools_time / sample_size,
            'mean_cost': float(ortools_costs.mean()),
            'std_cost': float(ortools_costs.std()),
            'mean_cpc': float(ortools_cpcs.mean()),
            'std_cpc': float(ortools_cpcs.std()),
        },
        'speedup': (ortools_time/sample_size) / (gpu_time/n_instances),
        'avg_relative_diff': float(avg_diff)
    }
    
    summary_file = results_dir / f"gpu_test_n{n_customers}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📁 Results saved to {summary_file}")
    
    return summary


if __name__ == "__main__":
    # Run test with N=6, 1000 instances
    results = test_gpu_vs_ortools(n_customers=6, n_instances=1000, batch_size=100)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
