#!/usr/bin/env python3
"""
Test GPU solver on N=10 with 1000 instances to compare with existing benchmarks.
"""

import numpy as np
import torch
import time
import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

from solvers.exact_gpu_improved import ImprovedGPUCVRPSolver
from solvers.exact.ortools_greedy import solve as ortools_solve

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
    capacity = max(demands.sum() / 3, demands.max() * 2)  # Adjusted for N=10
    
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

def test_n10():
    """Test GPU solver on N=10 instances."""
    n_customers = 10
    n_instances = 1000
    batch_size = 50  # Smaller batch for N=10 due to complexity
    
    print(f"\n{'='*80}")
    print(f"GPU SOLVER TEST: N={n_customers}")
    print(f"{'='*80}")
    print(f"Number of instances: {n_instances}")
    print(f"Batch size: {batch_size}")
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
    print("\nTesting GPU Solver...")
    gpu_solver = ImprovedGPUCVRPSolver()
    gpu_results = []
    gpu_start = time.time()
    
    # Process in batches
    for batch_start in range(0, n_instances, batch_size):
        batch_end = min(batch_start + batch_size, n_instances)
        batch = instances[batch_start:batch_end]
        
        if batch_start % 100 == 0:
            print(f"  Progress: {batch_start}/{n_instances} instances")
        
        batch_results = gpu_solver.solve_batch(batch, time_limit=60.0, verbose=False)
        gpu_results.extend(batch_results)
    
    gpu_time = time.time() - gpu_start
    
    # Calculate statistics
    gpu_costs = np.array([r.cost for r in gpu_results])
    gpu_cpcs = np.array([r.cost / n_customers for r in gpu_results])
    gpu_optimal = np.array([r.is_optimal for r in gpu_results])
    
    print(f"\n{'='*80}")
    print(f"RESULTS FOR N={n_customers}")
    print(f"{'='*80}")
    print(f"Total time: {gpu_time:.3f}s")
    print(f"Time per instance (TPI): {gpu_time/n_instances:.6f}s")
    print(f"Mean cost: {gpu_costs.mean():.4f}")
    print(f"Std cost: {gpu_costs.std():.4f}")
    print(f"Mean CPC: {gpu_cpcs.mean():.6f}")
    print(f"Std CPC: {gpu_cpcs.std():.6f}")
    print(f"Optimal solutions: {gpu_optimal.sum()}/{len(gpu_optimal)} ({gpu_optimal.mean():.1%})")
    
    # Compare with known benchmarks
    print(f"\n{'='*80}")
    print("COMPARISON WITH BENCHMARKS")
    print(f"{'='*80}")
    
    # From your data: N=10, ortools_gls: cpc=0.392007, std=0.061565
    print(f"OR-Tools GLS benchmark (from your data):")
    print(f"  CPC: 0.392007 ± 0.061565")
    print(f"  TPI: 2.002470s")
    
    print(f"\nGPU Solver (current test):")
    print(f"  CPC: {gpu_cpcs.mean():.6f} ± {gpu_cpcs.std():.6f}")
    print(f"  TPI: {gpu_time/n_instances:.6f}s")
    
    diff_cpc = abs(gpu_cpcs.mean() - 0.392007) / 0.392007
    speedup = 2.002470 / (gpu_time/n_instances)
    
    print(f"\nRelative difference in CPC: {diff_cpc:.2%}")
    print(f"Speedup vs OR-Tools GLS: {speedup:.1f}x")
    
    # Save results
    results_dir = Path("results/gpu_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"gpu_n{n_customers}_results.csv"
    df = pd.DataFrame({
        'instance_id': range(n_instances),
        'cost': gpu_costs,
        'cpc': gpu_cpcs,
        'is_optimal': gpu_optimal,
        'solve_time': [r.solve_time for r in gpu_results]
    })
    df.to_csv(results_file, index=False)
    print(f"\n📁 Detailed results saved to {results_file}")
    
    # Save summary
    summary = {
        'n_customers': n_customers,
        'n_instances': n_instances,
        'total_time': gpu_time,
        'tpi': gpu_time / n_instances,
        'mean_cpc': float(gpu_cpcs.mean()),
        'std_cpc': float(gpu_cpcs.std()),
        'mean_cost': float(gpu_costs.mean()),
        'std_cost': float(gpu_costs.std()),
        'optimality_rate': float(gpu_optimal.mean()),
        'comparison': {
            'ortools_gls_cpc': 0.392007,
            'ortools_gls_std': 0.061565,
            'ortools_gls_tpi': 2.002470,
            'cpc_difference': float(diff_cpc),
            'speedup': float(speedup)
        }
    }
    
    summary_file = results_dir / f"gpu_n{n_customers}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📁 Summary saved to {summary_file}")
    
    return summary

if __name__ == "__main__":
    test_n10()
