#!/usr/bin/env python3
"""
GPU Exact Benchmark for N=10 with truly optimal solutions.
WARNING: This will be MUCH slower than the heuristic version!
"""

import sys
import os
sys.path.append('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

from src.generator.generator import _generate_instance
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import gpu_cvrp_solver_truly_optimal_fixed as gpu_exact
import numpy as np
import time
import pandas as pd
from datetime import datetime
import torch

def benchmark_gpu_exact_n10(num_instances=10000, batch_size=10, capacity=20):
    """
    Benchmark truly exact GPU solver for N=10.
    Note: Smaller batch size due to memory constraints with parent tracking.
    """
    n_customers = 10
    
    print("=" * 70)
    print("GPU TRULY EXACT CVRP Benchmark")
    print("=" * 70)
    print(f"N = {n_customers} customers")
    print(f"Capacity = {capacity}")
    print(f"Instances = {num_instances:,}")
    print(f"Batch size = {batch_size}")
    print("-" * 70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: GPU not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Memory estimate
    states = 2 ** n_customers  # 1024 for N=10
    memory_per_instance = states * (n_customers + 1) * 4 * 3  # dp, parent, costs
    total_memory = batch_size * memory_per_instance / (1024**3)  # GB
    print(f"Estimated GPU memory per batch: {total_memory:.2f} GB")
    
    if total_memory > 8:
        print(f"WARNING: May exceed GPU memory! Consider reducing batch_size.")
    
    print("\nStarting benchmark...")
    print("-" * 70)
    
    all_results = []
    start_total = time.time()
    
    num_batches = (num_instances + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_instances)
        current_batch_size = batch_end - batch_start
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"\nBatch {batch_idx + 1}/{num_batches}: instances {batch_start}-{batch_end-1}")
        
        # Generate batch
        instances = []
        for i in range(batch_start, batch_end):
            instance = _generate_instance(
                num_customers=n_customers,
                capacity=capacity,
                coord_range=100,
                demand_range=[1, 10],
                seed=42000 + n_customers * 1000 + i
            )
            instances.append(instance)
        
        # Solve with GPU exact (fixed version)
        try:
            solve_start = time.time()
            solutions = gpu_exact.solve_batch(instances, verbose=False)
            solve_time = time.time() - solve_start
            
            # Store results
            for i, sol in enumerate(solutions):
                all_results.append({
                    'instance_id': batch_start + i,
                    'cost': sol.cost,
                    'cpc': sol.cost / n_customers,
                    'num_vehicles': sol.num_vehicles,
                    'solve_time': sol.solve_time,
                    'is_optimal': sol.is_optimal
                })
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                avg_cpc = np.mean([r['cpc'] for r in all_results[-current_batch_size:]])
                print(f"  Batch time: {solve_time:.2f}s ({solve_time/current_batch_size:.3f}s per instance)")
                print(f"  Batch avg CPC: {avg_cpc:.6f}")
                
        except torch.cuda.OutOfMemoryError:
            print(f"\nERROR: GPU out of memory at batch {batch_idx}")
            print(f"Reduce batch_size below {batch_size}")
            break
        except Exception as e:
            print(f"\nERROR in batch {batch_idx}: {e}")
            break
        
        # Progress update
        if len(all_results) % 100 == 0:
            elapsed = time.time() - start_total
            rate = len(all_results) / elapsed
            eta = (num_instances - len(all_results)) / rate
            print(f"  Progress: {len(all_results)}/{num_instances} ({rate:.1f} inst/s, ETA: {eta/60:.1f} min)")
    
    # Final statistics
    total_time = time.time() - start_total
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS - GPU TRULY EXACT Solver")
        print("=" * 70)
        print(f"Instances solved: {len(df):,}/{num_instances:,}")
        print(f"Mean CPC: {df['cpc'].mean():.6f}")
        print(f"Std CPC: {df['cpc'].std():.6f}")
        print(f"Min cost: {df['cost'].min():.6f}")
        print(f"Max cost: {df['cost'].max():.6f}")
        print(f"Avg vehicles: {df['num_vehicles'].mean():.2f}")
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Time per instance: {total_time/len(df):.3f}s")
        
        # Check optimality
        optimal_count = df['is_optimal'].sum()
        print(f"\nOptimality verification: {optimal_count}/{len(df)} optimal")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f'gpu_exact_n10_results_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")
        
        return df
    else:
        print("\nNo results obtained!")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU Truly Exact CVRP Benchmark for N=10')
    parser.add_argument('--instances', type=int, default=100,
                        help='Number of instances (default: 100, for testing)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for GPU (default: 10, limited by memory)')
    parser.add_argument('--capacity', type=int, default=20,
                        help='Vehicle capacity (default: 20)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("WARNING: This is MUCH slower than the heuristic version!")
    print("Estimated time for 10,000 instances: 3-5 hours")
    print("Estimated time for 100 instances: 2-3 minutes")
    print("="*70 + "\n")
    
    if args.instances > 100000:
        response = input(f"Run {args.instances} instances? This may take hours! (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    benchmark_gpu_exact_n10(
        num_instances=args.instances,
        batch_size=args.batch_size,
        capacity=args.capacity
    )
