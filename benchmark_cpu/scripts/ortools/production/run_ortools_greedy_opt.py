#!/usr/bin/env python3
"""
Production OR-Tools Greedy (optimal mode) benchmark runner with parallel processing.
Mirrors the GLS runner:
- Deterministic per-instance seeds (base_seed + instance_id)
- Striped ProcessPool allocation across threads
- Retry logic with exponentially increasing timeouts
- CPC-only output by default, optional detailed JSON per thread (--verbose)
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
from src.benchmarking.solvers.cpu.exact_ortools_vrp_fixed import solve as solve_greedy_opt


def setup_logging(log_path):
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def solve_instance_with_retry_old(instance, base_timeout, max_retries=3):
    current_timeout = base_timeout
    last_error = None
    for retry in range(max_retries + 1):
        try:
            solution = solve_greedy_opt(instance, time_limit=current_timeout, verbose=False)
            return solution, current_timeout, retry
        except Exception as e:
            last_error = e
            if retry < max_retries:
                current_timeout *= 2
                print(f"    Retry {retry + 1}: increasing timeout to {current_timeout}s")
            else:
                raise RuntimeError(
                    f"Failed after {max_retries + 1} attempts with max timeout {current_timeout}s: {str(last_error)}"
                )


def solve_instances_batch(args):
    n, capacity, thread_id, total_threads, num_instances, output_dir, base_seed, verbose = args

    my_instance_indices = list(range(thread_id, num_instances, total_threads))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id}: processing {len(my_instance_indices)} instances for N={n}")

    results = []
    cpc_values = []
    failed_count = 0
    retry_count = 0

    for instance_idx in my_instance_indices:
        instance_seed = base_seed + instance_idx
        try:
            instance = _generate_instance(
                num_customers=n,
                capacity=capacity,
                coord_range=100,
                demand_range=[1, 10],
                seed=instance_seed
            )

            solution = solve_greedy_opt(instance, time_limit=None, verbose=False)
            retries = 0
            actual_timeout = None
            if retries > 0:
                retry_count += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id}: "
                      f"Instance {instance_idx} succeeded after {retries} retries (timeout: {actual_timeout}s)")

            cpc = solution.cost / n
            cpc_values.append(cpc)

            if verbose:
                results.append({
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
                })

        except Exception as e:
            failed_count += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Thread {thread_id}: Failed instance {instance_idx} - {str(e)}")
            if verbose:
                results.append({
                    'instance_id': instance_idx,
                    'seed': instance_seed,
                    'n': n,
                    'capacity': capacity,
                    'error': str(e),
                    'failed': True,
                    'timeout_attempted': None,
                    'retry_count': 0
                })

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
        description='OR-Tools Greedy (optimal mode) Parallel Benchmark Runner with Retry Logic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--subfolder', type=str, required=True,
                        help='Subfolder name in benchmark_cpu/results/')
    parser.add_argument('--n', type=int, required=True,
                        help='Number of customer nodes (problem size)')
    parser.add_argument('--instances', type=int, required=True,
                        help='Total number of instances to run')
    # --timeout removed: running without per-instance time limits
    parser.add_argument('--threads', type=int, required=True,
                        help='Number of parallel threads')
    parser.add_argument('--capacity', type=int, default=None,
                        help='Vehicle capacity (default: auto-calculate based on N)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for reproducible instance generation')
    parser.add_argument('--verbose', action='store_true',
                        help='Save detailed results per thread (default: only save CPC values)')

    args = parser.parse_args()

    # Auto capacity if not provided (match GLS runner conventions)
    if args.capacity is None:
        capacity_map = {10: 20, 20: 30, 50: 40, 100: 50}
        args.capacity = capacity_map.get(args.n, int(args.n * 0.6))

    output_dir = Path('benchmark_cpu/results') / args.subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / 'benchmark_log.txt'
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("OR-Tools Greedy (Optimal Mode) Parallel Benchmark Runner")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Problem size (N): {args.n}")
    logger.info(f"  Vehicle capacity: {args.capacity}")
    logger.info(f"  Total instances: {args.instances}")
    logger.info("  Per-instance timeout: disabled (no time limit)")
    logger.info("  Retry strategy: disabled")
    logger.info(f"  Parallel threads: {args.threads}")
    logger.info(f"  Base seed: {args.seed}")
    logger.info(f"  Output mode: {'Detailed (verbose)' if args.verbose else 'CPC only'}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Log file: {log_path}")

    # Prepare tasks
    tasks = []
    for thread_id in range(args.threads):
        tasks.append((args.n, args.capacity, thread_id, args.threads,
                      args.instances, str(output_dir), args.seed, args.verbose))

    instances_per_thread = [len(list(range(i, args.instances, args.threads))) for i in range(args.threads)]
    logger.info(f"Instance distribution across threads: {instances_per_thread}")
    logger.info(f"Total instances to process: {sum(instances_per_thread)}")
    logger.info(f"Seed range: {args.seed} to {args.seed + args.instances - 1}")
    logger.info("Each instance has a unique seed: base_seed + instance_id")

    max_instances = max(instances_per_thread)
    logger.info("No per-instance timeouts; runtime depends on instance hardness distribution")
    logger.info("Starting parallel execution...")

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(solve_instances_batch, task) for task in tasks]

        all_cpc_values = []
        total_failed = 0
        total_retried = 0
        for i, future in enumerate(futures):
            try:
                result = future.result()
                all_cpc_values.extend(result['cpc_values'])
                total_failed += result['failed_count']
                total_retried += result['retry_count']

                if args.verbose and result['detailed_results']:
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

    cpc_file = output_dir / f'ortools_n{args.n}.json'
    with open(cpc_file, 'w') as f:
        json.dump({'n': args.n, 'capacity': args.capacity, 'instances': len(all_cpc_values), 'cpc': all_cpc_values},
                  f, indent=2)
    logger.info(f"✓ Saved CPC values to {cpc_file.name}")

    logger.info("-" * 70)
    logger.info("EXECUTION SUMMARY")
    logger.info("-" * 70)
    logger.info(f"Total execution time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    if args.instances:
        logger.info(f"Average time per instance: {elapsed/args.instances:.3f}s")
    total_successful = len(all_cpc_values)
    logger.info(f"Total instances successfully processed: {total_successful}")
    logger.info(f"Total instances that needed retries: {total_retried}")
    logger.info(f"Total instances failed (after all retries): {total_failed}")

    if total_successful > 0:
        cpc_array = np.array(all_cpc_values)
        logger.info("\nCPC Statistics:")
        logger.info(f"  Mean: {np.mean(cpc_array):.6f}")
        logger.info(f"  Std:  {np.std(cpc_array):.6f}")
        logger.info(f"  Min:  {np.min(cpc_array):.6f}")
        logger.info(f"  Max:  {np.max(cpc_array):.6f}")

    logger.info("\nOutput files:")
    logger.info(f"  - {cpc_file.name} (CPC values)")
    logger.info(f"  - {log_path.name} (execution log)")
    logger.info("")
    logger.info(f"Results saved in: {output_dir}")
    logger.info("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
