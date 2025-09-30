#!/usr/bin/env python3
"""
Benchmark script to measure speedup from parallel data generation.

This script compares sequential vs parallel data generation performance
and verifies that they produce identical results.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.generator import (
    create_data_generator, 
    create_parallel_data_generator,
    ParallelDataGeneratorPool
)
from src.utils.config import load_config


def verify_identical_generation(config: Dict[str, Any], 
                                batch_size: int = 100,
                                num_batches: int = 3) -> bool:
    """Verify that sequential and parallel generators produce identical results."""
    print("\n" + "="*60)
    print("VERIFYING GENERATOR CONSISTENCY")
    print("="*60)
    
    seq_gen = create_data_generator(config)
    par_gen = create_parallel_data_generator(config, num_workers=4)
    
    all_match = True
    for batch_idx in range(num_batches):
        seed = 12345 + batch_idx * 1000
        
        # Generate with both methods
        seq_instances = seq_gen(batch_size, seed=seed)
        par_instances = par_gen(batch_size, seed=seed)
        
        # Compare each instance
        for i in range(batch_size):
            seq_inst = seq_instances[i]
            par_inst = par_instances[i]
            
            # Check coords
            if not np.allclose(seq_inst['coords'], par_inst['coords']):
                print(f"❌ Batch {batch_idx}, Instance {i}: Coordinates mismatch")
                all_match = False
                
            # Check demands
            if not np.array_equal(seq_inst['demands'], par_inst['demands']):
                print(f"❌ Batch {batch_idx}, Instance {i}: Demands mismatch")
                all_match = False
                
            # Check distances
            if not np.allclose(seq_inst['distances'], par_inst['distances']):
                print(f"❌ Batch {batch_idx}, Instance {i}: Distances mismatch")
                all_match = False
                
            # Check capacity
            if seq_inst['capacity'] != par_inst['capacity']:
                print(f"❌ Batch {batch_idx}, Instance {i}: Capacity mismatch")
                all_match = False
    
    if all_match:
        print(f"✓ All {num_batches * batch_size} instances match perfectly")
        print("✓ Parallel generator produces identical results to sequential")
    else:
        print("❌ Some instances don't match - there may be a bug!")
    
    return all_match


def benchmark_generation(config: Dict[str, Any],
                        batch_size: int = 512,
                        num_batches: int = 10,
                        worker_counts: List[int] = [0, 4, 6, 8]) -> Dict[str, Any]:
    """Benchmark data generation with different worker counts."""
    print("\n" + "="*60)
    print("BENCHMARKING DATA GENERATION")
    print("="*60)
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Total instances: {batch_size * num_batches}")
    print(f"Problem size: {config['problem']['num_customers']} customers")
    print()
    
    results = {}
    
    for num_workers in worker_counts:
        print(f"\n{'='*60}")
        if num_workers == 0:
            print("Testing SEQUENTIAL generation (baseline)")
            generator = create_data_generator(config)
            use_pool = False
        else:
            print(f"Testing PARALLEL generation with {num_workers} workers")
            pool = ParallelDataGeneratorPool(config, num_workers=num_workers)
            generator = pool.generate_batch
            use_pool = True
        
        print("="*60)
        
        # Warmup
        _ = generator(batch_size, seed=99999)
        
        # Benchmark
        start_time = time.time()
        for batch_idx in range(num_batches):
            seed = batch_idx * 1000
            instances = generator(batch_size, seed=seed)
        end_time = time.time()
        
        if use_pool:
            pool.close()
        
        elapsed = end_time - start_time
        instances_per_sec = (batch_size * num_batches) / elapsed
        time_per_batch = elapsed / num_batches
        
        results[num_workers] = {
            'elapsed': elapsed,
            'instances_per_sec': instances_per_sec,
            'time_per_batch': time_per_batch
        }
        
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Time per batch: {time_per_batch:.3f} seconds")
        print(f"Throughput: {instances_per_sec:.1f} instances/second")
        
        # Calculate speedup vs sequential
        if num_workers == 0:
            baseline_time = elapsed
        else:
            speedup = baseline_time / elapsed
            efficiency = (speedup / num_workers) * 100
            print(f"Speedup vs sequential: {speedup:.2f}x")
            print(f"Parallel efficiency: {efficiency:.1f}%")
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary table of results."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Throughput':<20} {'Speedup':<10} {'Efficiency':<12}")
    print("-"*60)
    
    baseline_time = results[0]['elapsed']
    
    for num_workers in sorted(results.keys()):
        r = results[num_workers]
        if num_workers == 0:
            speedup_str = "1.00x"
            efficiency_str = "-"
            workers_str = "Sequential"
        else:
            speedup = baseline_time / r['elapsed']
            efficiency = (speedup / num_workers) * 100
            speedup_str = f"{speedup:.2f}x"
            efficiency_str = f"{efficiency:.1f}%"
            workers_str = str(num_workers)
        
        print(f"{workers_str:<10} {r['elapsed']:<12.2f} {r['instances_per_sec']:<20.1f} "
              f"{speedup_str:<10} {efficiency_str:<12}")
    
    print("="*60)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    best_speedup = 0
    best_workers = 0
    for num_workers in sorted(results.keys()):
        if num_workers == 0:
            continue
        speedup = baseline_time / results[num_workers]['elapsed']
        if speedup > best_speedup:
            best_speedup = speedup
            best_workers = num_workers
    
    print(f"✓ Best speedup: {best_speedup:.2f}x with {best_workers} workers")
    print(f"✓ Recommended for production: {best_workers} workers")
    
    if best_speedup >= 4:
        print(f"✓ Achieved {best_speedup:.1f}x speedup - excellent parallelization!")
    elif best_speedup >= 3:
        print(f"✓ Achieved {best_speedup:.1f}x speedup - good parallelization")
    elif best_speedup >= 2:
        print(f"⚠ Only {best_speedup:.1f}x speedup - consider profiling bottlenecks")
    else:
        print(f"⚠ Limited speedup ({best_speedup:.1f}x) - check for overhead issues")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark parallel data generation performance'
    )
    parser.add_argument('--config', type=str, default='configs/tiny_gpu_512.yaml',
                       help='Configuration file')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for generation')
    parser.add_argument('--num-batches', type=int, default=10,
                       help='Number of batches to generate')
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 4, 6, 8],
                       help='Worker counts to test (0 = sequential)')
    parser.add_argument('--skip-verification', action='store_true',
                       help='Skip verification step')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    config = load_config(str(config_path))
    
    print("\n" + "="*60)
    print("PARALLEL DATA GENERATION BENCHMARK")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Problem size: {config['problem']['num_customers']} customers")
    print(f"Vehicle capacity: {config['problem']['vehicle_capacity']}")
    print()
    
    # Verify consistency
    if not args.skip_verification:
        if not verify_identical_generation(config):
            print("\n⚠ Warning: Generators produce different results!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    # Run benchmarks
    results = benchmark_generation(
        config,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        worker_counts=args.workers
    )
    
    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
