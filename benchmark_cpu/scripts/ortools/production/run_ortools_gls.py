#!/usr/bin/env python3
"""
Production OR-Tools GLS benchmark runner with parallel processing.
Integrates instance generation and solving to ensure proper seed management.
Includes retry logic with exponentially increasing timeouts for failed instances.
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.cpu.ortools_gls import solve


def setup_logging(log_path):
    """Setup logging to both file and console"""
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def solve_instance_with_retry(instance, base_timeout, max_retries=3):
    """
    Attempt to solve an instance with retry logic.
    Each retry doubles the timeout: base_timeout, 2*base_timeout, 4*base_timeout, 8*base_timeout
    
    Returns:
        tuple: (solution, actual_timeout_used, retry_count)
    """
    current_timeout = base_timeout
    last_error = None
    
    for retry in range(max_retries + 1):
        try:
            solution = solve(instance, time_limit=current_timeout, verbose=False)
            # Success! Return solution with metadata
            return solution, current_timeout, retry
        
        except Exception as e:
            last_error = e
            if retry < max_retries:
                # Double the timeout for next attempt
                current_timeout *= 2
                print(f"    Retry {retry + 1}: increasing timeout to {current_timeout}s")
            else:
                # Final failure after all retries
                raise RuntimeError(
                    f"Failed after {max_retries + 1} attempts with max timeout {current_timeout}s: {str(last_error)}"
                )


def solve_instances_batch(args):
    """Solve a batch of instances for a single thread"""
    n, capacity, timeout, thread_id, total_threads, num_instances, output_dir, base_seed, verbose = args
    
    # Calculate which instances this thread handles (striped allocation)
    my_instance_indices = list(range(thread_id, num_instances, total_threads))
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id}: processing {len(my_instance_indices)} instances for N={n}")
    
    results = []
    cpc_values = []  # Collect CPC values for simple format
    failed_count = 0
    retry_count = 0
    
    for instance_idx in my_instance_indices:
        # Use deterministic seed based on base_seed and instance index
        # This ensures same instances regardless of thread count
        instance_seed = base_seed + instance_idx
        
        try:
            # Generate instance with specific seed
            instance = _generate_instance(
                num_customers=n,
                capacity=capacity,
                coord_range=100,
                demand_range=[1, 10],
                seed=instance_seed
            )
            
            # Solve with OR-Tools GLS with retry logic
            solution, actual_timeout, retries = solve_instance_with_retry(
                instance, timeout, max_retries=3
            )
            
            if retries > 0:
                retry_count += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id}: "
                      f"Instance {instance_idx} succeeded after {retries} retries (timeout: {actual_timeout}s)")
            
            # Calculate CPC
            cpc = solution.cost / n
            cpc_values.append(cpc)
            
            # Store detailed result if verbose mode
            if verbose:
                result = {
                    'instance_id': instance_idx,
                    'seed': instance_seed,
                    'n': n,
                    'capacity': capacity,
                    'coords': instance['coords'].tolist(),
                    'demands': instance['demands'].tolist(),
                    'route': solution.route,
                    'cost': solution.cost,
                    'cpc': cpc,
                    'num_vehicles': solution.num_vehicles,
                    'vehicle_routes': solution.vehicle_routes,
                    'solve_time': solution.solve_time,
                    'algorithm': solution.algorithm_used,
                    'is_optimal': solution.is_optimal,
                    'timeout_used': actual_timeout,
                    'retry_count': retries,
                    'failed': False
                }
                results.append(result)
            
        except Exception as e:
            failed_count += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id}: Failed instance {instance_idx} - {str(e)}")
            
            if verbose:
                # Store failed instance info with max timeout attempted
                results.append({
                    'instance_id': instance_idx,
                    'seed': instance_seed,
                    'n': n,
                    'capacity': capacity,
                    'error': str(e),
                    'failed': True,
                    'timeout_attempted': timeout * (2 ** 3),  # Max timeout after 3 retries
                    'retry_count': 3  # All retries were attempted
                })
    
    # Return results
    return {
        'thread_id': thread_id,
        'cpc_values': cpc_values,
        'detailed_results': results if verbose else None,
        'failed_count': failed_count,
        'retry_count': retry_count,
        'instances_processed': len(my_instance_indices)
    }


def main():
    parser = argparse.ArgumentParser(
        description='OR-Tools GLS Parallel Benchmark Runner with Retry Logic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--subfolder', type=str, required=True,
                        help='Subfolder name in benchmark_cpu/results/')
    parser.add_argument('--n', type=int, required=True,
                        help='Number of customer nodes (problem size)')
    parser.add_argument('--instances', type=int, required=True,
                        help='Total number of instances to run')
    parser.add_argument('--timeout', type=float, required=True,
                        help='Base timeout in seconds per instance (will be increased on retries)')
    parser.add_argument('--threads', type=int, required=True,
                        help='Number of parallel threads')
    parser.add_argument('--capacity', type=int, default=None,
                        help='Vehicle capacity (default: auto-calculate based on N)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for reproducible instance generation')
    parser.add_argument('--verbose', action='store_true',
                        help='Save detailed results per thread (default: only save CPC values)')
    
    args = parser.parse_args()
    
    # Auto-calculate capacity if not provided
    if args.capacity is None:
        # Standard capacity formula based on problem size
        capacity_map = {10: 20, 20: 30, 50: 40, 100: 50}
        args.capacity = capacity_map.get(args.n, int(args.n * 0.6))
    
    # Setup output directory
    output_dir = Path('benchmark_cpu/results') / args.subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = output_dir / 'benchmark_log.txt'
    logger = setup_logging(log_path)
    
    # Log configuration
    logger.info("=" * 70)
    logger.info("OR-Tools GLS Parallel Benchmark Runner")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Problem size (N): {args.n}")
    logger.info(f"  Vehicle capacity: {args.capacity}")
    logger.info(f"  Total instances: {args.instances}")
    logger.info(f"  Base timeout per instance: {args.timeout}s")
    logger.info(f"  Retry strategy: up to 4 attempts with timeouts:")
    logger.info(f"    - Attempt 1: {args.timeout}s")
    logger.info(f"    - Attempt 2: {args.timeout * 2}s")
    logger.info(f"    - Attempt 3: {args.timeout * 4}s")
    logger.info(f"    - Attempt 4: {args.timeout * 8}s")
    logger.info(f"  Parallel threads: {args.threads}")
    logger.info(f"  Base seed: {args.seed}")
    logger.info(f"  Output mode: {'Detailed (verbose)' if args.verbose else 'CPC only'}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Log file: {log_path}")
    logger.info("")
    
    # Prepare tasks for parallel execution
    tasks = []
    for thread_id in range(args.threads):
        tasks.append((
            args.n,
            args.capacity,
            args.timeout,
            thread_id,
            args.threads,
            args.instances,
            str(output_dir),
            args.seed,
            args.verbose  # Pass verbose flag to workers
        ))
    
    # Log expected distribution
    instances_per_thread = [len(list(range(i, args.instances, args.threads))) 
                           for i in range(args.threads)]
    logger.info(f"Instance distribution across threads: {instances_per_thread}")
    logger.info(f"Total instances to process: {sum(instances_per_thread)}")
    
    # Log seed information
    logger.info(f"Seed range: {args.seed} to {args.seed + args.instances - 1}")
    logger.info("Each instance has a unique seed: base_seed + instance_id")
    logger.info("This ensures reproducibility regardless of thread count")
    
    # Estimate completion time
    max_instances = max(instances_per_thread)
    typical_time = max_instances * args.timeout * 1.1  # Assume 10% need retries
    worst_case_time = max_instances * (args.timeout + args.timeout*2 + args.timeout*4 + args.timeout*8)
    logger.info(f"Estimated completion time:")
    logger.info(f"  Typical case: {typical_time:.1f}s ({typical_time/60:.1f} minutes)")
    logger.info(f"  Worst case: {worst_case_time:.1f}s ({worst_case_time/60:.1f} minutes)")
    logger.info("")
    logger.info("Starting parallel execution...")
    logger.info("-" * 70)
    
    # Run all tasks in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(solve_instances_batch, task) for task in tasks]
        
        # Collect results from all threads
        all_cpc_values = []
        all_detailed_results = []
        total_failed = 0
        total_retried = 0
        
        for i, future in enumerate(futures):
            try:
                result = future.result()
                all_cpc_values.extend(result['cpc_values'])
                total_failed += result['failed_count']
                total_retried += result['retry_count']
                
                if args.verbose and result['detailed_results']:
                    all_detailed_results.extend(result['detailed_results'])
                    
                    # Save detailed results per thread
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    thread_file = output_dir / f'thread_{result["thread_id"]:02d}_n{args.n}_{timestamp}.json'
                    
                    with open(thread_file, 'w') as f:
                        json.dump({
                            'thread_id': result['thread_id'],
                            'n': args.n,
                            'capacity': args.capacity,
                            'base_timeout': args.timeout,
                            'total_instances': result['instances_processed'],
                            'successful': result['instances_processed'] - result['failed_count'],
                            'failed': result['failed_count'],
                            'instances_requiring_retry': result['retry_count'],
                            'results': result['detailed_results']
                        }, f, indent=2)
                    
                    logger.info(f"Thread {i}: ✓ Saved detailed results to {thread_file.name}")
                
                logger.info(f"Thread {i}: Processed {result['instances_processed']} instances "
                          f"({result['failed_count']} failed, {result['retry_count']} needed retries)")
                
            except Exception as e:
                logger.error(f"Thread {i}: ✗ Exception - {str(e)}")
    
    elapsed = time.time() - start_time
    
    # Save CPC values in simple format (always, regardless of verbose)
    cpc_file = output_dir / f'ortools_n{args.n}.json'
    with open(cpc_file, 'w') as f:
        json.dump({
            'n': args.n,
            'capacity': args.capacity, 
            'instances': len(all_cpc_values),
            'cpc': all_cpc_values
        }, f, indent=2)
    
    logger.info(f"✓ Saved CPC values to {cpc_file.name}")
    
    # Summary
    logger.info("-" * 70)
    logger.info("EXECUTION SUMMARY")
    logger.info("-" * 70)
    logger.info(f"Total execution time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    logger.info(f"Average time per instance: {elapsed/args.instances:.3f}s")
    
    total_successful = len(all_cpc_values)
    logger.info(f"Total instances successfully processed: {total_successful}")
    logger.info(f"Total instances that needed retries: {total_retried}")
    logger.info(f"Total instances failed (after all retries): {total_failed}")
    
    if total_successful > 0:
        cpc_array = np.array(all_cpc_values)
        logger.info(f"\nCPC Statistics:")
        logger.info(f"  Mean: {np.mean(cpc_array):.6f}")
        logger.info(f"  Std:  {np.std(cpc_array):.6f}")
        logger.info(f"  Min:  {np.min(cpc_array):.6f}")
        logger.info(f"  Max:  {np.max(cpc_array):.6f}")
        
        retry_rate = (total_retried / args.instances) * 100 if args.instances > 0 else 0
        failure_rate = (total_failed / args.instances) * 100 if args.instances > 0 else 0
        logger.info(f"\nRetry rate: {retry_rate:.1f}%")
        logger.info(f"Final failure rate: {failure_rate:.1f}%")
    
    # List output files
    logger.info("\nOutput files:")
    if args.verbose:
        json_files = list(output_dir.glob('thread_*.json'))
        logger.info(f"  - {len(json_files)} detailed thread JSON files")
    logger.info(f"  - {cpc_file.name} (CPC values)")
    logger.info(f"  - {log_path.name} (execution log)")
    
    logger.info("")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("=" * 70)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
