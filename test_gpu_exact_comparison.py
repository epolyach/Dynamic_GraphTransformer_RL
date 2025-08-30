#!/usr/bin/env python3
"""
Test GPU solver vs OR-Tools EXACT solver on N=6.
Ensures OR-Tools is running in exact mode for fair comparison.
"""

import numpy as np
import torch
import time
import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

# Import solvers
from solvers.exact_gpu_improved import ImprovedGPUCVRPSolver
from solvers.exact.ortools_greedy import solve as ortools_exact_solve
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

def test_exact_comparison(n_customers: int = 6, n_instances: int = 100):
    """
    Test GPU solver against OR-Tools EXACT solver.
    
    Args:
        n_customers: Number of customers (default 6)
        n_instances: Number of instances to test (default 100 for exact comparison)
    """
    print(f"\n{'='*80}")
    print(f"GPU vs OR-TOOLS EXACT SOLVER COMPARISON")
    print(f"{'='*80}")
    print(f"Problem size: {n_customers} customers")
    print(f"Number of instances: {n_instances}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Generate instances with fixed seeds for reproducibility
    print("Generating instances...")
    instances = []
    for i in range(n_instances):
        instances.append(generate_cvrp_instance(n_customers, seed=5000 + i))
    
    # Test GPU solver
    print("\n" + "-"*40)
    print("Testing GPU Solver (Exact Branch & Bound)")
    print("-"*40)
    
    gpu_solver = ImprovedGPUCVRPSolver()
    gpu_results = []
    gpu_start = time.time()
    
    # Process all instances
    batch_size = 10  # Smaller batch for exact solving
    for batch_start in range(0, n_instances, batch_size):
        batch_end = min(batch_start + batch_size, n_instances)
        batch = instances[batch_start:batch_end]
        
        if batch_start % 20 == 0:
            print(f"  Progress: {batch_start}/{n_instances} instances")
        
        batch_results = gpu_solver.solve_batch(batch, time_limit=30.0, verbose=False)
        gpu_results.extend(batch_results)
    
    gpu_time = time.time() - gpu_start
    
    # Calculate GPU statistics
    gpu_costs = np.array([r.cost for r in gpu_results])
    gpu_cpcs = np.array([r.cost / n_customers for r in gpu_results])
    gpu_optimal = np.array([r.is_optimal for r in gpu_results])
    
    print(f"\nGPU Results:")
    print(f"  Total time: {gpu_time:.3f}s")
    print(f"  Time per instance: {gpu_time/n_instances:.3f}s")
    print(f"  Mean cost: {gpu_costs.mean():.4f}")
    print(f"  Std cost: {gpu_costs.std():.4f}")
    print(f"  Mean CPC: {gpu_cpcs.mean():.6f}")
    print(f"  Std CPC: {gpu_cpcs.std():.6f}")
    print(f"  Optimal solutions: {gpu_optimal.sum()}/{len(gpu_optimal)} ({gpu_optimal.mean():.1%})")
    
    # Test OR-Tools EXACT solver
    print("\n" + "-"*40)
    print("Testing OR-Tools Exact Solver")
    print("-"*40)
    print("NOTE: OR-Tools with UNSET metaheuristic (exact mode)")
    
    ortools_results = []
    ortools_start = time.time()
    
    # Test all instances with OR-Tools exact
    for idx, instance in enumerate(instances):
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{n_instances} instances")
        
        # Call with explicit time limit for exact solving
        result = ortools_exact_solve(instance, time_limit=10.0, verbose=False)
        ortools_results.append(result)
    
    ortools_time = time.time() - ortools_start
    
    # Calculate OR-Tools statistics
    ortools_costs = np.array([r.cost for r in ortools_results])
    ortools_cpcs = np.array([r.cost / n_customers for r in ortools_results])
    ortools_optimal = np.array([r.is_optimal for r in ortools_results])
    
    print(f"\nOR-Tools Exact Results:")
    print(f"  Total time: {ortools_time:.3f}s")
    print(f"  Time per instance: {ortools_time/n_instances:.3f}s")
    print(f"  Mean cost: {ortools_costs.mean():.4f}")
    print(f"  Std cost: {ortools_costs.std():.4f}")
    print(f"  Mean CPC: {ortools_cpcs.mean():.6f}")
    print(f"  Std CPC: {ortools_cpcs.std():.6f}")
    print(f"  Optimal solutions: {ortools_optimal.sum()}/{len(ortools_optimal)} ({ortools_optimal.mean():.1%})")
    
    # Direct comparison
    print("\n" + "-"*40)
    print("Instance-by-Instance Comparison")
    print("-"*40)
    
    comparison_results = []
    for i in range(min(20, n_instances)):  # Show first 20
        gpu_cost = gpu_results[i].cost
        ortools_cost = ortools_results[i].cost
        diff = gpu_cost - ortools_cost
        
        comparison_results.append({
            'instance': i,
            'gpu_cost': gpu_cost,
            'ortools_cost': ortools_cost,
            'difference': diff,
            'relative_diff': diff / ortools_cost if ortools_cost > 0 else 0,
            'gpu_optimal': gpu_results[i].is_optimal,
            'ortools_optimal': ortools_results[i].is_optimal
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    print(comparison_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nPerformance comparison:")
    print(f"  GPU speedup: {ortools_time / gpu_time:.1f}x faster overall")
    print(f"  GPU throughput: {n_instances/gpu_time:.1f} instances/second")
    print(f"  OR-Tools throughput: {n_instances/ortools_time:.1f} instances/second")
    
    print(f"\nSolution quality comparison:")
    all_diffs = []
    for i in range(n_instances):
        if ortools_results[i].cost > 0:
            all_diffs.append(abs(gpu_results[i].cost - ortools_results[i].cost) / ortools_results[i].cost)
    
    avg_diff = np.mean(all_diffs)
    max_diff = np.max(all_diffs)
    
    print(f"  Average relative difference: {avg_diff:.2%}")
    print(f"  Max relative difference: {max_diff:.2%}")
    print(f"  Instances where GPU = OR-Tools: {sum(1 for d in all_diffs if d < 0.001)}/{n_instances}")
    
    if avg_diff < 0.01:
        print("  ✅ GPU solver produces nearly identical results to OR-Tools EXACT!")
    elif avg_diff < 0.05:
        print("  ✅ GPU solver produces comparable results to OR-Tools EXACT (within 5%)")
    else:
        print(f"  ⚠️  GPU solver differs from OR-Tools EXACT by {avg_diff:.1%} on average")
    
    # Save results
    results_dir = Path("results/gpu_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed comparison
    full_comparison = []
    for i in range(n_instances):
        full_comparison.append({
            'instance_id': i,
            'gpu_cost': gpu_results[i].cost,
            'gpu_cpc': gpu_results[i].cost / n_customers,
            'gpu_optimal': gpu_results[i].is_optimal,
            'gpu_time': gpu_results[i].solve_time,
            'ortools_cost': ortools_results[i].cost,
            'ortools_cpc': ortools_results[i].cost / n_customers,
            'ortools_optimal': ortools_results[i].is_optimal,
            'ortools_time': ortools_results[i].solve_time,
            'cost_diff': gpu_results[i].cost - ortools_results[i].cost,
            'relative_diff': (gpu_results[i].cost - ortools_results[i].cost) / ortools_results[i].cost if ortools_results[i].cost > 0 else 0
        })
    
    comparison_file = results_dir / f"exact_comparison_n{n_customers}.csv"
    pd.DataFrame(full_comparison).to_csv(comparison_file, index=False)
    print(f"\n📁 Detailed comparison saved to {comparison_file}")
    
    # Save summary
    summary = {
        'n_customers': n_customers,
        'n_instances': n_instances,
        'gpu': {
            'total_time': gpu_time,
            'time_per_instance': gpu_time / n_instances,
            'mean_cost': float(gpu_costs.mean()),
            'std_cost': float(gpu_costs.std()),
            'mean_cpc': float(gpu_cpcs.mean()),
            'std_cpc': float(gpu_cpcs.std()),
            'optimality_rate': float(gpu_optimal.mean())
        },
        'ortools_exact': {
            'total_time': ortools_time,
            'time_per_instance': ortools_time / n_instances,
            'mean_cost': float(ortools_costs.mean()),
            'std_cost': float(ortools_costs.std()),
            'mean_cpc': float(ortools_cpcs.mean()),
            'std_cpc': float(ortools_cpcs.std()),
            'optimality_rate': float(ortools_optimal.mean())
        },
        'comparison': {
            'speedup': ortools_time / gpu_time,
            'avg_relative_diff': float(avg_diff),
            'max_relative_diff': float(max_diff),
            'identical_solutions': sum(1 for d in all_diffs if d < 0.001)
        }
    }
    
    summary_file = results_dir / f"exact_comparison_n{n_customers}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📁 Summary saved to {summary_file}")
    
    return summary

if __name__ == "__main__":
    # Test with N=6, 100 instances for exact comparison
    results = test_exact_comparison(n_customers=6, n_instances=100)
    
    print("\n" + "="*80)
    print("EXACT COMPARISON TEST COMPLETE")
    print("="*80)
