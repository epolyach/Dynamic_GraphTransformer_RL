#!/usr/bin/env python3
"""
Specialized benchmark for N=20 using only heuristic solver with more vehicle flexibility
"""
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
import solvers.heuristic_or as heuristic_or

def run_benchmark(n_instances=100, output_file='results/csv/cpu_20.csv'):
    """Run benchmark for N=20 with specified number of instances"""
    
    print(f"=== CVRP Benchmark for N=20 ===")
    print(f"Instances: {n_instances}")
    print(f"Output: {output_file}")
    print(f"Solver: heuristic_or only")
    print()
    
    # Generator
    gen = EnhancedCVRPGenerator(config={})
    
    # Results storage
    results = []
    
    # Parameters
    n = 20
    capacity = 30
    coord_range = 100
    demand_range = [1, 10]
    
    successful = 0
    failed = 0
    
    print("Progress: ", end="", flush=True)
    
    for i in range(n_instances):
        if i % 10 == 0:
            print(f"{i}", end=".", flush=True)
        
        # Generate instance with unique seed
        seed = 5000 + i  # Different seed range to avoid conflicts
        
        # Keep trying different seeds if we get duplicate coordinates
        max_attempts = 10
        for attempt in range(max_attempts):
            instance = gen.generate_instance(
                num_customers=n,
                capacity=capacity,
                coord_range=coord_range,
                demand_range=demand_range,
                seed=seed + attempt * 1000,
                instance_type=InstanceType.RANDOM,
                apply_augmentation=False,
            )
            
            # Check for duplicate coordinates
            coords = instance['coords']
            has_duplicates = False
            for idx1 in range(len(coords)):
                for idx2 in range(idx1 + 1, len(coords)):
                    if np.allclose(coords[idx1], coords[idx2], rtol=1e-9, atol=1e-9):
                        has_duplicates = True
                        break
                if has_duplicates:
                    break
            
            if not has_duplicates:
                break
        
        # Solve with heuristic
        try:
            start_time = time.time()
            solution = heuristic_or.solve(instance, time_limit=60.0, verbose=False)
            solve_time = time.time() - start_time
            
            results.append({
                'n_customers': n,
                'solver': 'heuristic_or',
                'instance_id': i + 1,
                'status': 'success',
                'time': solve_time,
                'cpc': solution.cost
            })
            successful += 1
            
        except Exception as e:
            results.append({
                'n_customers': n,
                'solver': 'heuristic_or', 
                'instance_id': i + 1,
                'status': 'failed',
                'time': 0.0,
                'cpc': None
            })
            failed += 1
    
    print(f" Done!")
    print(f"\nResults: {successful} successful, {failed} failed")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    if successful > 0:
        avg_time = df[df['status'] == 'success']['time'].mean()
        avg_cost = df[df['status'] == 'success']['cpc'].mean()
        print(f"Average solve time: {avg_time:.4f}s")
        print(f"Average cost: {avg_cost:.4f}")

if __name__ == "__main__":
    # Test with 3 instances first
    print("Testing with 3 instances first...\n")
    run_benchmark(n_instances=3, output_file='results/csv/cpu_20_test.csv')
