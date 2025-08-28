#!/usr/bin/env python3
"""
Comprehensive optimality verification for GPU solver
Tests against exhaustive search and multiple verification methods
"""

import numpy as np
import sys
import time
from itertools import permutations, combinations
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
from solvers.exact_gpu_dp import solve_batch as gpu_solve_batch
from solvers.exact_dp import solve as cpu_exact_solve

def exhaustive_search(instance):
    """
    TRUE exhaustive search - tries ALL possible partitions and orderings
    Guaranteed to find the optimal solution
    """
    n = len(instance['coords'])
    customers = list(range(1, n))
    demands = instance['demands']
    capacity = instance['capacity']
    distances = instance['distances']
    
    best_cost = float('inf')
    best_solution = None
    solutions_checked = 0
    
    # Generate all possible partitions of customers into routes
    def generate_partitions(items):
        if len(items) == 0:
            yield []
        else:
            first = items[0]
            rest = items[1:]
            for partition in generate_partitions(rest):
                # Add first to each existing group
                for i in range(len(partition)):
                    new_partition = [g[:] for g in partition]
                    new_partition[i].append(first)
                    yield new_partition
                # Or create a new group with first
                yield partition + [[first]]
    
    for partition in generate_partitions(customers):
        # Check capacity constraints
        feasible = True
        for route in partition:
            if sum(demands[c] for c in route) > capacity:
                feasible = False
                break
        
        if not feasible:
            continue
        
        # Try all permutations within each route
        total_cost = 0
        for route in partition:
            if len(route) == 0:
                continue
            
            # Find best ordering for this route
            best_route_cost = float('inf')
            for perm in permutations(route):
                # Calculate route cost: depot -> customers -> depot
                cost = distances[0][perm[0]]
                for i in range(len(perm) - 1):
                    cost += distances[perm[i]][perm[i+1]]
                cost += distances[perm[-1]][0]
                
                if cost < best_route_cost:
                    best_route_cost = cost
            
            total_cost += best_route_cost
        
        solutions_checked += 1
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_solution = partition
    
    return best_cost, best_solution, solutions_checked

def verify_gpu_optimality():
    """Test GPU solver against exhaustive search"""
    print("=" * 70)
    print("GPU SOLVER OPTIMALITY VERIFICATION")
    print("=" * 70)
    
    # Test on small instances where exhaustive search is feasible
    test_cases = [
        (4, [5000, 5001, 5002]),  # N=4, 3 different seeds
        (5, [5000, 5001]),         # N=5, 2 different seeds
        (6, [5000]),               # N=6, 1 seed
    ]
    
    gen = EnhancedCVRPGenerator(config={})
    
    all_optimal = True
    results = []
    
    for n_customers, seeds in test_cases:
        print(f"\nTesting N={n_customers}:")
        print("-" * 50)
        
        for seed in seeds:
            # Generate instance using same method as benchmarks
            full_seed = 4242 + n_customers * 1000 + (seed - 5000) * 10
            instance = gen.generate_instance(
                num_customers=n_customers,
                capacity=30,
                coord_range=100,
                demand_range=[1, 10],
                seed=full_seed,
                instance_type=InstanceType.RANDOM,
                apply_augmentation=False,
            )
            
            # GPU solve
            gpu_start = time.time()
            gpu_results = gpu_solve_batch([instance], verbose=False)
            gpu_time = time.time() - gpu_start
            gpu_cost = gpu_results[0].cost
            
            # CPU exact solve for comparison
            cpu_start = time.time()
            cpu_result = cpu_exact_solve(instance, time_limit=60.0, verbose=False)
            cpu_time = time.time() - cpu_start
            cpu_cost = cpu_result.cost
            
            # Exhaustive search (ground truth)
            if n_customers <= 5:  # Only feasible for very small instances
                exh_start = time.time()
                optimal_cost, optimal_solution, solutions_checked = exhaustive_search(instance)
                exh_time = time.time() - exh_start
                
                # Compare
                gpu_gap = abs(gpu_cost - optimal_cost) / optimal_cost * 100 if optimal_cost > 0 else 0
                cpu_gap = abs(cpu_cost - optimal_cost) / optimal_cost * 100 if optimal_cost > 0 else 0
                
                is_gpu_optimal = abs(gpu_cost - optimal_cost) < 0.0001
                is_cpu_optimal = abs(cpu_cost - optimal_cost) < 0.0001
                
                print(f"\n  Seed {seed} (actual: {full_seed}):")
                print(f"    Exhaustive: {optimal_cost:.6f} (checked {solutions_checked} solutions in {exh_time:.3f}s)")
                print(f"    GPU:        {gpu_cost:.6f} (gap: {gpu_gap:.4f}%) in {gpu_time:.3f}s {'✓' if is_gpu_optimal else '✗'}")
                print(f"    CPU:        {cpu_cost:.6f} (gap: {cpu_gap:.4f}%) in {cpu_time:.3f}s {'✓' if is_cpu_optimal else '✗'}")
                
                results.append({
                    'n': n_customers,
                    'seed': seed,
                    'optimal': optimal_cost,
                    'gpu': gpu_cost,
                    'cpu': cpu_cost,
                    'gpu_optimal': is_gpu_optimal,
                    'cpu_optimal': is_cpu_optimal
                })
                
                if not is_gpu_optimal:
                    all_optimal = False
                    print(f"    ⚠️ GPU NOT OPTIMAL! Difference: {gpu_cost - optimal_cost:.6f}")
            else:
                # For larger instances, just compare GPU vs CPU
                print(f"\n  Seed {seed} (actual: {full_seed}):")
                print(f"    GPU: {gpu_cost:.6f} in {gpu_time:.3f}s")
                print(f"    CPU: {cpu_cost:.6f} in {cpu_time:.3f}s")
                
                is_same = abs(gpu_cost - cpu_cost) < 0.0001
                print(f"    Agreement: {'✓' if is_same else '✗'} (diff: {abs(gpu_cost - cpu_cost):.6f})")
                
                results.append({
                    'n': n_customers,
                    'seed': seed,
                    'gpu': gpu_cost,
                    'cpu': cpu_cost,
                    'agreement': is_same
                })
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    # Count optimal solutions
    optimal_count = sum(1 for r in results if 'gpu_optimal' in r and r['gpu_optimal'])
    total_verified = sum(1 for r in results if 'gpu_optimal' in r)
    
    if total_verified > 0:
        print(f"\nExhaustive verification (N ≤ 5):")
        print(f"  GPU optimal: {optimal_count}/{total_verified} instances")
        print(f"  Success rate: {optimal_count/total_verified*100:.1f}%")
    
    # Count agreements
    agreement_count = sum(1 for r in results if 'agreement' in r and r['agreement'])
    total_compared = sum(1 for r in results if 'agreement' in r)
    
    if total_compared > 0:
        print(f"\nGPU vs CPU agreement (all N):")
        print(f"  Matching: {agreement_count}/{total_compared} instances")
        print(f"  Agreement rate: {agreement_count/total_compared*100:.1f}%")
    
    if all_optimal and total_verified > 0:
        print("\n✅ GPU SOLVER IS PROVEN OPTIMAL for all tested instances!")
    elif optimal_count == total_verified and total_verified > 0:
        print("\n✅ GPU SOLVER IS OPTIMAL for all verified instances!")
    else:
        print("\n⚠️ GPU solver may not be finding optimal solutions in all cases")
    
    return all_optimal, results

if __name__ == "__main__":
    all_optimal, results = verify_gpu_optimality()
    
    # Additional statistical test
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    if results:
        gaps = []
        for r in results:
            if 'optimal' in r:
                gap = (r['gpu'] - r['optimal']) / r['optimal'] * 100 if r['optimal'] > 0 else 0
                gaps.append(gap)
        
        if gaps:
            print(f"\nOptimality gaps (GPU vs Exhaustive):")
            print(f"  Mean gap: {np.mean(gaps):.6f}%")
            print(f"  Max gap:  {np.max(gaps):.6f}%")
            print(f"  All gaps < 0.01%: {all(g < 0.01 for g in gaps)}")
