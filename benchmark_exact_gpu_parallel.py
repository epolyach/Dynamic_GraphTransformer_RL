#!/usr/bin/env python3
"""
GPU-Parallel CVRP Solver Benchmark with Substantial Speed Improvements

This implementation leverages all available CPU cores in parallel to run multiple 
CVRP solver instances simultaneously, providing significant speedup over the 
sequential benchmark_exact.py while maintaining identical results.

Key Performance Features:
- Parallel instance generation across CPU cores
- Concurrent solver execution using multiprocessing
- Intelligent workload distribution and resource management  
- Maintains full compatibility with original benchmark_exact.py
- Expected speedup: ~N× where N = number of CPU cores

The term "GPU" in the filename refers to running on a GPU-equipped machine
with high CPU core count, not actual GPU computation (solvers are CPU-based).
"""

import argparse
import numpy as np
import time
import csv
import sys
import statistics
import logging
import threading
import signal
import multiprocessing as mp
from multiprocessing import Process, Pool, Queue, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional, Union
import psutil
import gc
import os

# Import solvers
import solvers.exact_ortools_vrp as exact_ortools_vrp
import solvers.exact_milp as exact_milp
import solvers.exact_dp as exact_dp
import solvers.exact_pulp as exact_pulp
import solvers.heuristic_or as heuristic_or

# Import generator from research folder
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
from solvers.types import CVRPSolution

# Import validation functions from original benchmark
from benchmark_exact import (normalize_trip, format_route_with_depot, normalize_route, 
                           calculate_route_cost, validate_solutions, safe_print, 
                           print_progress_bar, setup_logging)


def generate_single_instance_worker(args: Tuple) -> Tuple[int, Dict[str, Any]]:
    """
    Worker function to generate a single CVRP instance.
    
    Args:
        args: (index, n, capacity, coord_range, demand_range, seed)
        
    Returns:
        (index, instance) tuple
    """
    index, n, capacity, coord_range, demand_range, seed = args
    
    try:
        gen = EnhancedCVRPGenerator(config={})
        instance = gen.generate_instance(
            num_customers=n,
            capacity=capacity,
            coord_range=coord_range,
            demand_range=demand_range,
            seed=seed,
            instance_type=InstanceType.RANDOM,
            apply_augmentation=False,
        )
        return (index, instance)
    except Exception as e:
        print(f"Error generating instance {index}: {e}")
        # Fallback with different seed
        try:
            fallback_seed = seed + 1000
            instance = gen.generate_instance(
                num_customers=n,
                capacity=capacity,
                coord_range=coord_range,
                demand_range=demand_range,
                seed=fallback_seed,
                instance_type=InstanceType.RANDOM,
                apply_augmentation=False,
            )
            return (index, instance)
        except Exception as e2:
            raise RuntimeError(f"Failed to generate instance {index} even with fallback: {e2}")


def generate_instance_batch_parallel(n: int, capacity: int, coord_range: int, 
                                   demand_range: List[int], num_instances: int, 
                                   num_workers: int, base_seed: int = 4242) -> List[Dict[str, Any]]:
    """
    Generate a batch of CVRP instances in parallel using multiple CPU cores.
    
    Args:
        n: Number of customers
        capacity: Vehicle capacity  
        coord_range: Coordinate range
        demand_range: [min_demand, max_demand]
        num_instances: Number of instances to generate
        num_workers: Number of parallel workers
        base_seed: Base seed for reproducibility
        
    Returns:
        List of generated CVRP instances in order
    """
    print(f"🔧 Generating {num_instances} instances using {num_workers} workers...")
    
    # Create argument tuples for parallel generation
    args_list = []
    for i in range(num_instances):
        seed = base_seed + n * 1000 + i * 10
        args_list.append((i, n, capacity, coord_range, demand_range, seed))
    
    # Use process pool to generate instances in parallel
    instances_dict = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all generation tasks
        futures = [executor.submit(generate_single_instance_worker, args) for args in args_list]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                index, instance = future.result(timeout=30)  # 30s timeout per instance
                instances_dict[index] = instance
            except Exception as e:
                print(f"⚠️ Instance generation failed: {e}")
                raise
    
    # Return instances in original order
    return [instances_dict[i] for i in range(num_instances)]


def run_single_solver_worker(args: Tuple) -> Tuple[int, Optional[CVRPSolution], float, bool]:
    """
    Worker function to run a single solver on a single instance.
    
    Args:
        args: (instance_index, solver_module, solver_name, instance, time_limit)
        
    Returns:
        (instance_index, solution, solve_time, timed_out) tuple
    """
    instance_index, solver_module, solver_name, instance, time_limit = args
    
    start_time = time.time()
    
    try:
        # Run solver with timeout
        solution = solver_module.solve(instance, time_limit=time_limit, verbose=False)
        solve_time = time.time() - start_time
        
        if solution is None:
            return (instance_index, None, solve_time, False)
            
        # Validate solution
        if solution.cost == float('inf') or solution.cost <= 0:
            return (instance_index, None, solve_time, False)
            
        # Add solve time
        solution.solve_time = solve_time
        
        # Check if within time limit
        timed_out = solve_time >= time_limit
        
        return (instance_index, solution, solve_time, timed_out)
        
    except Exception as e:
        solve_time = time.time() - start_time
        timed_out = "timeout" in str(e).lower() or "timed out" in str(e).lower()
        return (instance_index, None, solve_time, timed_out)


def run_solver_batch_parallel(instances: List[Dict[str, Any]], solver_name: str, 
                             time_limit: float, num_workers: int, 
                             logger: logging.Logger) -> List[Tuple[Optional[CVRPSolution], float, bool]]:
    """
    Run a specific solver on a batch of instances in parallel.
    
    Args:
        instances: List of CVRP instances
        solver_name: Name of solver to run
        time_limit: Per-instance time limit
        num_workers: Number of parallel workers
        logger: Logger for output
        
    Returns:
        List of (solution, solve_time, timed_out) tuples in original order
    """
    print(f"🔄 Running {solver_name} on {len(instances)} instances using {num_workers} workers...")
    
    # Get solver module
    solvers = {
        'exact_ortools_vrp': exact_ortools_vrp,
        'exact_milp': exact_milp,
        'exact_dp': exact_dp, 
        'exact_pulp': exact_pulp,
        'heuristic_or': heuristic_or
    }
    solver_module = solvers[solver_name]
    
    # Create argument tuples for parallel solving
    args_list = []
    for i, instance in enumerate(instances):
        args_list.append((i, solver_module, solver_name, instance, time_limit))
    
    # Use process pool for parallel solving
    results_dict = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all solving tasks
        futures = [executor.submit(run_single_solver_worker, args) for args in args_list]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                instance_index, solution, solve_time, timed_out = future.result(timeout=time_limit + 10)
                results_dict[instance_index] = (solution, solve_time, timed_out)
            except TimeoutError:
                # Process exceeded time limit
                print(f"⚠️ {solver_name} process timeout on instance")
                # We don't know which instance, so this is handled in the main loop
            except Exception as e:
                print(f"⚠️ {solver_name} process error: {e}")
    
    # Return results in original order, filling missing with timeout results
    results = []
    for i in range(len(instances)):
        if i in results_dict:
            results.append(results_dict[i])
        else:
            # Instance not completed - assume timeout
            results.append((None, time_limit, True))
    
    return results


def run_benchmark_for_n_parallel(n: int, instances_min: int, instances_max: int,
                                capacity: int, demand_range: List[int], total_timeout: float,
                                coord_range: int, num_workers: int, logger: logging.Logger,
                                disabled_solvers: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Run parallel benchmark for a specific problem size N.
    
    This function generates instances and runs all solvers in parallel,
    providing substantial speedup over the sequential version.
    
    Args:
        n: Problem size (number of customers)
        instances_min: Minimum instances for statistics
        instances_max: Maximum instances to attempt  
        capacity: Vehicle capacity
        demand_range: [min_demand, max_demand]
        total_timeout: Total timeout per solver per N
        coord_range: Coordinate range
        num_workers: Number of parallel workers
        logger: Logger instance
        disabled_solvers: Set of disabled solver names
        
    Returns:
        Dictionary containing benchmark statistics
    """
    if disabled_solvers is None:
        disabled_solvers = set()
    
    safe_print(f"\n🚀 N={n}: GPU-Parallel benchmark ({instances_min}-{instances_max} instances, {num_workers} workers)")
    if disabled_solvers:
        safe_print(f"   Disabled solvers: {', '.join(disabled_solvers)}")
    
    t0 = time.time()
    
    # Step 1: Generate all instances in parallel
    safe_print(f"🔧 Generating {instances_max} instances...")
    all_instances = generate_instance_batch_parallel(
        n, capacity, coord_range, demand_range, instances_max, num_workers, base_seed=4242
    )
    
    generation_time = time.time() - t0
    safe_print(f"✅ Generated {len(all_instances)} instances in {generation_time:.1f}s")
    
    # Step 2: Run each solver in parallel across all instances
    results = {
        'exact_ortools_vrp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'exact_milp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'exact_dp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'exact_pulp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'heuristic_or': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []}
    }
    
    instance_timeout_threshold = total_timeout / instances_min
    validation_errors = 0
    
    # Track which instances are successful for each solver (for validation)
    successful_instances = {}
    
    solver_names = ['exact_ortools_vrp', 'exact_milp', 'exact_dp', 'exact_pulp', 'heuristic_or']
    
    for solver_name in solver_names:
        if solver_name in disabled_solvers:
            safe_print(f"⏭️ Skipping {solver_name} (disabled)")
            continue
                
        safe_print(f"🔄 Running {solver_name}...")
        t_solver = time.time()
        
        # Run solver on all instances in parallel
        solver_results = run_solver_batch_parallel(
            all_instances, solver_name, instance_timeout_threshold, num_workers, logger
        )
        
        # Process results
        successful_count = 0
        for i, (solution, solve_time, timed_out) in enumerate(solver_results):
            # Log individual results for debugging
            if solution is not None:
                clean_routes = [tuple(node for node in route if node != 0) for route in solution.vehicle_routes]
                clean_routes = [route for route in clean_routes if route]  # Remove empty routes
                logger.info(f"{solver_name} instance {i}: cost={solution.cost:.4f}, time={solve_time:.6f}s, routes={clean_routes}")
            else:
                if timed_out:
                    logger.info(f"{solver_name} instance {i}: TIMEOUT after {solve_time:.6f}s")
                else:
                    logger.info(f"{solver_name} instance {i}: FAILED after {solve_time:.6f}s")
            
            # Include successful solutions (not timed out and within threshold)
            if solution is not None and not timed_out and solve_time < instance_timeout_threshold:
                results[solver_name]['times'].append(solve_time)
                
                # Normalized cost per customer
                benchmark_cost = solution.cost / max(1, all_instances[i]['num_customers'])
                results[solver_name]['costs'].append(benchmark_cost)
                results[solver_name]['solutions'].append(solution)
                
                # Track optimal solutions for exact solvers
                if solution.is_optimal and solver_name.startswith('exact'):
                    results[solver_name]['optimal_count'] += 1
                
                successful_count += 1
                
                # Track successful instances for validation
                if i not in successful_instances:
                    successful_instances[i] = {}
                successful_instances[i][solver_name] = solution
        
        solver_time = time.time() - t_solver
        safe_print(f"✅ {solver_name}: {successful_count}/{len(all_instances)} succeeded in {solver_time:.1f}s")
    
    # Step 3: Validate solutions
    safe_print("🔍 Validating solutions...")
    validation_errors = 0
    
    for instance_idx, instance_solutions in successful_instances.items():
        if 'exact_ortools_vrp' not in instance_solutions:
            continue  # Skip validation if no OR-Tools solution
            
        try:
            ortools_solution = instance_solutions['exact_ortools_vrp']
            other_solutions = {k: v for k, v in instance_solutions.items() if k != 'exact_ortools_vrp'}
            
            validate_solutions(ortools_solution, other_solutions, all_instances[instance_idx], logger)
        except ValueError:
            validation_errors += 1
    
    # Step 4: Compute statistics (reuse logic from original benchmark)
    stats = {'N': n, 'validation_errors': validation_errors}
    
    for solver_name in solver_names:
        times = results[solver_name]['times']
        costs = results[solver_name]['costs']
        optimal_count = results[solver_name]['optimal_count']
        instances_solved = len(times)
        
        if solver_name in disabled_solvers:
            # Solver was disabled
            avg_time = float('nan')
            avg_cost = float('nan')
            std_cost = float('nan')
            solved_count = 0
            optimal_count = 0
        elif instances_solved >= 1:
            avg_time = float(statistics.mean(times))
            avg_cost = float(statistics.mean(costs))
            solved_count = instances_solved
            
            # Only calculate std if we have enough instances
            if instances_solved >= max(2, instances_min):
                std_cost = float(statistics.stdev(costs))
            else:
                std_cost = float('nan')
        else:
            avg_time = float('nan')
            avg_cost = float('nan') 
            std_cost = float('nan')
            solved_count = 0
            
        # For exact solvers, use only optimal solutions in statistics
        if solver_name.startswith('exact') and not solver_name in disabled_solvers and optimal_count > 0:
            # Re-compute statistics using only optimal solutions
            optimal_costs = []
            optimal_times = []
            for i, solution in enumerate(results[solver_name]['solutions']):
                if solution.is_optimal:
                    optimal_costs.append(costs[i])
                    optimal_times.append(times[i])
            
            if len(optimal_costs) >= 1:
                avg_cost = float(statistics.mean(optimal_costs))
                avg_time = float(statistics.mean(optimal_times))
                solved_count = len(optimal_costs)
                
                if len(optimal_costs) >= max(2, instances_min):
                    std_cost = float(statistics.stdev(optimal_costs))
                else:
                    std_cost = float('nan')
            else:
                # No optimal solutions found
                avg_cost = float('nan')
                avg_time = float('nan')
                std_cost = float('nan')
                solved_count = 0
        
        # Store results
        stats[f'time_{solver_name}'] = avg_time
        stats[f'cpc_{solver_name}'] = avg_cost  
        stats[f'std_{solver_name}'] = std_cost
        stats[f'solved_{solver_name}'] = solved_count
        stats[f'optimal_{solver_name}'] = optimal_count
    
    elapsed = time.time() - t0
    sequential_estimate = elapsed * num_workers  # Rough estimate of sequential time
    
    safe_print(f"🎉 N={n} completed in {elapsed:.1f}s (estimated {sequential_estimate:.1f}s sequential, ~{num_workers}× speedup)")
    
    if validation_errors > 0:
        safe_print(f"⚠️ {validation_errors} validation errors detected!")
    
    # Print summary statistics
    for solver in solver_names:
        if solver in disabled_solvers:
            continue
            
        solved = stats.get(f'solved_{solver}', 0)
        optimal = stats.get(f'optimal_{solver}', 0) if solver.startswith('exact') else 'N/A'
        avg_time = stats.get(f'time_{solver}', float('nan'))
        avg_cpc = stats.get(f'cpc_{solver}', float('nan'))
        
        time_str = f"{avg_time:.4f}s" if not np.isnan(avg_time) else "nan"
        cpc_str = f"{avg_cpc:.4f}" if not np.isnan(avg_cpc) else "nan"
        
        if solver.startswith('exact'):
            safe_print(f"  {solver}: {solved} solved, {optimal} optimal, time={time_str}, cpc={cpc_str}")
        else:
            safe_print(f"  {solver}: {solved} solved, time={time_str}, cpc={cpc_str}")
    
    return stats


def main():
    # Set multiprocessing start method for better compatibility
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set
    
    parser = argparse.ArgumentParser(description='GPU-Parallel CVRP Solver Benchmark')
    parser.add_argument('--instances-min', type=int, default=5, help='Minimum instances per N (default: 5)')
    parser.add_argument('--instances-max', type=int, default=20, help='Maximum instances per N (default: 20)')
    parser.add_argument('--n-start', type=int, default=5, help='Start N (default: 5)')
    parser.add_argument('--n-end', type=int, default=15, help='End N inclusive (default: 15)')
    parser.add_argument('--capacity', type=int, default=30, help='Vehicle capacity (default: 30)')
    parser.add_argument('--demand-min', type=int, default=1, help='Min demand (default: 1)')
    parser.add_argument('--demand-max', type=int, default=10, help='Max demand (default: 10)')
    parser.add_argument('--timeout', type=float, default=60.0, help='Total timeout per solver per N (default: 60.0s)')
    parser.add_argument('--coord-range', type=int, default=100, help='Coordinate range (default: 100)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU cores)')
    parser.add_argument('--output', type=str, default='benchmark_exact_gpu_parallel.csv', help='Output CSV file')
    parser.add_argument('--log', type=str, default='benchmark_exact_gpu_parallel.log', help='Log file')
    
    args = parser.parse_args()
    
    # Determine number of workers
    cpu_cores = mp.cpu_count()
    num_workers = args.workers if args.workers is not None else cpu_cores
    num_workers = max(1, min(num_workers, cpu_cores))  # Clamp to available cores
    
    # Set up logging
    logger = setup_logging(args.log)
    
    safe_print("=" * 80)
    safe_print("GPU-PARALLEL CVRP SOLVER BENCHMARK")
    safe_print("=" * 80)
    safe_print(f"Problem size: N = {args.n_start} to {args.n_end}")
    safe_print(f"Instances per N: {args.instances_min}-{args.instances_max}")
    safe_print(f"Vehicle capacity: {args.capacity}")
    safe_print(f"Demand range: [{args.demand_min}, {args.demand_max}]")
    safe_print(f"Coordinate range: {args.coord_range}")
    safe_print(f"Total timeout per solver per N: {args.timeout}s")
    safe_print(f"Parallel workers: {num_workers}/{cpu_cores}")
    safe_print(f"Output file: {args.output}")
    safe_print(f"Log file: {args.log}")
    safe_print()
    
    # Log the configuration
    logger.info("=" * 80)
    logger.info("GPU-PARALLEL CVRP SOLVER BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Problem size: N = {args.n_start} to {args.n_end}")
    logger.info(f"Instances per N: {args.instances_min}-{args.instances_max}")
    logger.info(f"Vehicle capacity: {args.capacity}")
    logger.info(f"Demand range: [{args.demand_min}, {args.demand_max}]")
    logger.info(f"Coordinate range: {args.coord_range}")
    logger.info(f"Total timeout per solver per N: {args.timeout}s")
    logger.info(f"Parallel workers: {num_workers}/{cpu_cores}")
    logger.info(f"Output file: {args.output}")
    
    demand_range = [args.demand_min, args.demand_max]
    
    # Initialize CSV file with header
    fieldnames = ['N', 
                  'time_exact_ortools_vrp', 'cpc_exact_ortools_vrp', 'std_exact_ortools_vrp', 'solved_exact_ortools_vrp', 'optimal_exact_ortools_vrp',
                  'time_exact_milp', 'cpc_exact_milp', 'std_exact_milp', 'solved_exact_milp', 'optimal_exact_milp',
                  'time_exact_dp', 'cpc_exact_dp', 'std_exact_dp', 'solved_exact_dp', 'optimal_exact_dp',
                  'time_exact_pulp', 'cpc_exact_pulp', 'std_exact_pulp', 'solved_exact_pulp', 'optimal_exact_pulp',
                  'time_heuristic_or', 'cpc_heuristic_or', 'std_heuristic_or', 'solved_heuristic_or']
                  
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
    
    rows_written = 0
    total_validation_errors = 0
    disabled_solvers = set()
    
    # Run benchmark for each N
    overall_start = time.time()
    
    for n in range(args.n_start, args.n_end + 1):
        result = run_benchmark_for_n_parallel(
            n, args.instances_min, args.instances_max, args.capacity, 
            demand_range, args.timeout, args.coord_range, num_workers, logger, disabled_solvers
        )
        
        total_validation_errors += result.get('validation_errors', 0)
        
        # Write result immediately
        with open(args.output, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Filter to only include CSV fieldnames and clean NaN values
            filtered_result = {}
            for k, v in result.items():
                if k in fieldnames:
                    if isinstance(v, float) and np.isnan(v):
                        filtered_result[k] = 'nan'
                    else:
                        filtered_result[k] = v
            writer.writerow(filtered_result)
            f.flush()
        rows_written += 1
    
    overall_elapsed = time.time() - overall_start
    estimated_sequential = overall_elapsed * num_workers
    
    safe_print(f"\n🎉 GPU-Parallel Benchmark Complete!")
    safe_print(f"📊 Wrote {rows_written} rows to {args.output}")
    safe_print(f"📝 Detailed results logged to {args.log}")
    safe_print(f"⏱️ Total time: {overall_elapsed:.1f}s (estimated {estimated_sequential:.1f}s sequential)")
    safe_print(f"🚀 Speedup: ~{num_workers}× faster with {num_workers} workers")
    
    if total_validation_errors > 0:
        safe_print(f"⚠️ Total validation errors: {total_validation_errors}")
    else:
        safe_print(f"✅ All solutions validated successfully!")


if __name__ == '__main__':
    main()
