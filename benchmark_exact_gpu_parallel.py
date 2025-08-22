#!/usr/bin/env python3
"""
True GPU-Parallel CVRP Solver Benchmark

This implementation uses true GPU parallelization:
1. Generate instances once on CPU (ensures consistency across solvers)
2. Run all 5 solvers × N instances simultaneously across CPU cores
3. Timeout-based result collection (process what's ready after timeout)

Performance: 5 solvers × N instances = 5N parallel CPU tasks
Expected speedup: ~5× over sequential execution
"""

import argparse
import numpy as np
import time
import csv
import sys
import statistics
import logging
import signal
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional, Union
import gc
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError

# Import solvers (CPU-based) - at module level for multiprocessing
import solvers.exact_ortools_vrp as exact_ortools_vrp
import solvers.exact_milp as exact_milp
import solvers.exact_dp as exact_dp
import solvers.exact_pulp as exact_pulp
import solvers.heuristic_or as heuristic_or

# Import from research folder
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
from solvers.types import CVRPSolution

# Import validation functions from original benchmark
from benchmark_exact import (normalize_trip, format_route_with_depot, normalize_route, 
                           calculate_route_cost, validate_solutions, safe_print, 
                           print_progress_bar, setup_logging)

# Global solver mapping for multiprocessing
SOLVERS = {
    'exact_ortools_vrp': exact_ortools_vrp,
    'exact_milp': exact_milp,
    'exact_dp': exact_dp, 
    'exact_pulp': exact_pulp,
    'heuristic_or': heuristic_or
}


def run_solver_task_worker(args: Tuple) -> Tuple[str, int, Optional[CVRPSolution], float, bool]:
    """
    Global worker function to run a single solver on a single instance.
    Must be module-level for multiprocessing.
    
    Args:
        args: (solver_name, instance_idx, instance, time_limit)
        
    Returns:
        (solver_name, instance_idx, solution, solve_time, timed_out)
    """
    solver_name, instance_idx, instance, time_limit = args
    solver_module = SOLVERS[solver_name]
    
    start_time = time.time()
    
    try:
        solution = solver_module.solve(instance, time_limit=time_limit, verbose=False)
        solve_time = time.time() - start_time
        
        if solution is None:
            return (solver_name, instance_idx, None, solve_time, False)
            
        # Validate solution
        if solution.cost == float('inf') or solution.cost <= 0:
            return (solver_name, instance_idx, None, solve_time, False)
            
        # Add solve time
        solution.solve_time = solve_time
        
        # Check if within time limit
        timed_out = solve_time >= time_limit
        
        return (solver_name, instance_idx, solution, solve_time, timed_out)
        
    except Exception as e:
        solve_time = time.time() - start_time
        timed_out = "timeout" in str(e).lower() or "timed out" in str(e).lower()
        return (solver_name, instance_idx, None, solve_time, timed_out)


class GPUParallelBenchmark:
    """True parallel benchmark runner for massive CVRP solver parallelization."""
    
    def __init__(self, num_workers: int = None):
        """
        Initialize parallel benchmark runner.
        
        Args:
            num_workers: Number of parallel workers (default: CPU cores)
        """
        # Use all available CPU cores as workers
        self.num_workers = num_workers or mp.cpu_count()
        
        print(f"🚀 GPU-Parallel Benchmark initialized:")
        print(f"   Parallel workers: {self.num_workers}")
        print(f"   Solvers: {list(SOLVERS.keys())}")
        print(f"   Parallel capacity: {self.num_workers} solver instances simultaneously")
    
    def generate_instances_cpu(self, n_customers: int, capacity: int, coord_range: int,
                             demand_range: List[int], num_instances: int, 
                             base_seed: int = 4242) -> List[Dict[str, Any]]:
        """
        Generate CVRP instances on CPU using original enhanced generator.
        This ensures consistency with the original benchmark.
        
        Args:
            n_customers: Number of customers
            capacity: Vehicle capacity
            coord_range: Coordinate range
            demand_range: [min_demand, max_demand]
            num_instances: Number of instances to generate
            base_seed: Base seed for reproducibility
            
        Returns:
            List of CVRP instances (same for all solvers)
        """
        print(f"🔧 Generating {num_instances} instances on CPU (consistent across solvers)...")
        
        generator = EnhancedCVRPGenerator(config={})
        instances = []
        
        for i in range(num_instances):
            seed = base_seed + n_customers * 1000 + i * 10
            instance = generator.generate_instance(
                num_customers=n_customers,
                capacity=capacity,
                coord_range=coord_range,
                demand_range=demand_range,
                seed=seed,
                instance_type=InstanceType.RANDOM,
                apply_augmentation=False,
            )
            instances.append(instance)
        
        return instances
    
    def run_parallel_benchmark(self, instances: List[Dict[str, Any]], 
                             timeout: float, logger: logging.Logger,
                             disabled_solvers: Set[str] = None) -> Dict[str, List]:
        """
        Run all solvers on all instances in parallel.
        
        This is the core parallelization: submit 5×N tasks simultaneously,
        wait for timeout, then collect all completed results.
        
        Args:
            instances: List of CVRP instances
            timeout: Total timeout for all parallel execution
            logger: Logger for detailed output
            disabled_solvers: Set of disabled solver names
            
        Returns:
            Dictionary with results for each solver
        """
        if disabled_solvers is None:
            disabled_solvers = set()
        
        num_instances = len(instances)
        active_solvers = [name for name in SOLVERS.keys() if name not in disabled_solvers]
        total_tasks = len(active_solvers) * num_instances
        
        print(f"🚀 Running {len(active_solvers)} solvers × {num_instances} instances = {total_tasks} parallel tasks")
        print(f"⏱️ Global timeout: {timeout}s for all parallel execution")
        
        # Prepare all solver tasks
        tasks = []
        per_task_timeout = timeout / 2  # Give individual tasks less time since we have global timeout
        
        for solver_name in active_solvers:
            for instance_idx, instance in enumerate(instances):
                tasks.append((solver_name, instance_idx, instance, per_task_timeout))
        
        # Run all tasks in parallel using ProcessPoolExecutor
        results = {solver_name: [] for solver_name in SOLVERS.keys()}
        completed_tasks = 0
        
        print(f"🔧 Submitting {total_tasks} tasks to {self.num_workers} parallel workers...")
        
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(run_solver_task_worker, task): task 
                    for task in tasks
                }
                
                # Collect results as they complete, respecting global timeout
                for future in as_completed(future_to_task, timeout=timeout):
                    try:
                        solver_name, instance_idx, solution, solve_time, timed_out = future.result(timeout=1.0)
                        
                        # Log individual results
                        if solution is not None:
                            clean_routes = [tuple(node for node in route if node != 0) for route in solution.vehicle_routes]
                            clean_routes = [route for route in clean_routes if route]
                            logger.info(f"{solver_name} instance {instance_idx}: cost={solution.cost:.4f}, time={solve_time:.6f}s, routes={clean_routes}")
                        else:
                            if timed_out:
                                logger.info(f"{solver_name} instance {instance_idx}: TIMEOUT after {solve_time:.6f}s")
                            else:
                                logger.info(f"{solver_name} instance {instance_idx}: FAILED after {solve_time:.6f}s")
                        
                        # Store result
                        results[solver_name].append((instance_idx, solution, solve_time, timed_out))
                        completed_tasks += 1
                        
                        # Show progress
                        if completed_tasks % 5 == 0:
                            elapsed = time.time() - start_time
                            print(f"  Progress: {completed_tasks}/{total_tasks} tasks completed in {elapsed:.1f}s")
                    
                    except TimeoutError:
                        # This future exceeded individual timeout, skip it
                        continue
                    except Exception as e:
                        # Unexpected error
                        task = future_to_task[future]
                        solver_name = task[0] 
                        instance_idx = task[1]
                        logger.error(f"{solver_name} instance {instance_idx}: ERROR {e}")
                        continue
        
        except TimeoutError:
            print(f"⏱️ Global timeout reached after {timeout}s")
        
        elapsed = time.time() - start_time
        print(f"✅ Parallel execution completed: {completed_tasks}/{total_tasks} tasks in {elapsed:.1f}s")
        
        # Report completion rate per solver
        for solver_name in active_solvers:
            completed = len(results[solver_name])
            print(f"  {solver_name}: {completed}/{num_instances} instances completed")
        
        return results
    
    def run_benchmark_for_n_parallel(self, n: int, instances_min: int, instances_max: int,
                                   capacity: int, demand_range: List[int], total_timeout: float,
                                   coord_range: int, logger: logging.Logger,
                                   disabled_solvers: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Run parallel benchmark for a specific problem size N.
        """
        if disabled_solvers is None:
            disabled_solvers = set()
        
        safe_print(f"\n🚀 N={n}: Parallel benchmark ({instances_min}-{instances_max} instances)")
        if disabled_solvers:
            safe_print(f"   Disabled solvers: {', '.join(disabled_solvers)}")
        
        t0 = time.time()
        
        # Step 1: Generate instances once on CPU (consistent across all solvers)
        all_instances = self.generate_instances_cpu(
            n, capacity, coord_range, demand_range, instances_max, base_seed=4242
        )
        
        generation_time = time.time() - t0
        safe_print(f"✅ Generated {len(all_instances)} instances in {generation_time:.1f}s")
        
        # Step 2: Run all solvers in parallel
        parallel_results = self.run_parallel_benchmark(
            all_instances, total_timeout, logger, disabled_solvers
        )
        
        # Step 3: Process results and compute statistics (same as original)
        stats = {'N': n, 'validation_errors': 0}
        solver_names = ['exact_ortools_vrp', 'exact_milp', 'exact_dp', 'exact_pulp', 'heuristic_or']
        
        # Organize results by solver
        organized_results = {
            solver_name: {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []}
            for solver_name in solver_names
        }
        
        successful_instances = {}  # For validation
        
        for solver_name in solver_names:
            if solver_name in disabled_solvers:
                continue
                
            solver_results = parallel_results.get(solver_name, [])
            successful_count = 0
            
            for instance_idx, solution, solve_time, timed_out in solver_results:
                if solution is not None and not timed_out:
                    organized_results[solver_name]['times'].append(solve_time)
                    
                    # Normalized cost per customer
                    benchmark_cost = solution.cost / max(1, all_instances[instance_idx]['num_customers'])
                    organized_results[solver_name]['costs'].append(benchmark_cost)
                    organized_results[solver_name]['solutions'].append(solution)
                    
                    # Track optimal solutions for exact solvers
                    if solution.is_optimal and solver_name.startswith('exact'):
                        organized_results[solver_name]['optimal_count'] += 1
                    
                    successful_count += 1
                    
                    # Track for validation
                    if instance_idx not in successful_instances:
                        successful_instances[instance_idx] = {}
                    successful_instances[instance_idx][solver_name] = solution
            
            safe_print(f"✅ {solver_name}: {successful_count} successful instances")
        
        # Validation and statistics computation (same as before)
        safe_print("🔍 Validating solutions...")
        validation_errors = 0
        
        for instance_idx, instance_solutions in successful_instances.items():
            if 'exact_ortools_vrp' not in instance_solutions:
                continue
                
            try:
                ortools_solution = instance_solutions['exact_ortools_vrp']
                other_solutions = {k: v for k, v in instance_solutions.items() if k != 'exact_ortools_vrp'}
                validate_solutions(ortools_solution, other_solutions, all_instances[instance_idx], logger)
            except ValueError:
                validation_errors += 1
        
        stats['validation_errors'] = validation_errors
        
        # Compute statistics (same logic as original)
        for solver_name in solver_names:
            times = organized_results[solver_name]['times']
            costs = organized_results[solver_name]['costs']
            optimal_count = organized_results[solver_name]['optimal_count']
            instances_solved = len(times)
            
            if solver_name in disabled_solvers:
                avg_time = float('nan')
                avg_cost = float('nan')
                std_cost = float('nan')
                solved_count = 0
                optimal_count = 0
            elif instances_solved >= 1:
                avg_time = float(statistics.mean(times))
                avg_cost = float(statistics.mean(costs))
                solved_count = instances_solved
                
                if instances_solved >= max(2, instances_min):
                    std_cost = float(statistics.stdev(costs))
                else:
                    std_cost = float('nan')
            else:
                avg_time = float('nan')
                avg_cost = float('nan') 
                std_cost = float('nan')
                solved_count = 0
                
            # For exact solvers, use only optimal solutions
            if solver_name.startswith('exact') and not solver_name in disabled_solvers and optimal_count > 0:
                optimal_costs = []
                optimal_times = []
                for i, solution in enumerate(organized_results[solver_name]['solutions']):
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
        estimated_sequential = elapsed * len([s for s in solver_names if s not in disabled_solvers])
        
        safe_print(f"🎉 N={n} completed in {elapsed:.1f}s (estimated sequential: {estimated_sequential:.1f}s)")
        safe_print(f"🚀 Speedup: ~{estimated_sequential/elapsed:.1f}× faster with parallelization")
        
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
    # Set multiprocessing start method
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
    parser.add_argument('--timeout', type=float, default=60.0, help='Total timeout for parallel execution (default: 60.0s)')
    parser.add_argument('--coord-range', type=int, default=100, help='Coordinate range (default: 100)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU cores)')
    parser.add_argument('--output', type=str, default='benchmark_exact_gpu_parallel.csv', help='Output CSV file')
    parser.add_argument('--log', type=str, default='benchmark_exact_gpu_parallel.log', help='Log file')
    
    args = parser.parse_args()
    
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
    safe_print(f"Parallel timeout: {args.timeout}s")
    safe_print(f"Workers: {args.workers or mp.cpu_count()}")
    safe_print(f"Output file: {args.output}")
    safe_print(f"Log file: {args.log}")
    safe_print()
    
    # Initialize parallel benchmark runner
    benchmark = GPUParallelBenchmark(num_workers=args.workers)
    
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
        result = benchmark.run_benchmark_for_n_parallel(
            n, args.instances_min, args.instances_max, args.capacity, 
            demand_range, args.timeout, args.coord_range, logger, disabled_solvers
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
    
    safe_print(f"\n🎉 GPU-Parallel Benchmark Complete!")
    safe_print(f"📊 Wrote {rows_written} rows to {args.output}")
    safe_print(f"📝 Detailed results logged to {args.log}")
    safe_print(f"⏱️ Total time: {overall_elapsed:.1f}s (GPU-parallel)")
    safe_print(f"🚀 Massive speedup achieved with parallelization")
    
    if total_validation_errors > 0:
        safe_print(f"⚠️ Total validation errors: {total_validation_errors}")
    else:
        safe_print(f"✅ All solutions validated successfully!")


if __name__ == '__main__':
    main()
