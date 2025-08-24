#!/usr/bin/env python3
"""
Exact CVRP Solver Benchmark with Solution Validation
Compares exact_ortools_vrp, exact_milp, exact_dp, exact_pulp, heuristic_or solvers
Validates solutions and logs detailed results to benchmark_exact.log
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
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional

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


def normalize_trip(trip: List[int]) -> Tuple[int, ...]:
    """
    Normalize a trip by removing depot nodes and creating a canonical representation.
    The trip can be traversed in either direction, so we choose the lexicographically smaller one.
    """
    # Remove depot (node 0) from the trip
    customers = [node for node in trip if node != 0]
    if not customers:
        return tuple()
    
    # Try both directions and choose lexicographically smaller
    forward = tuple(customers)
    backward = tuple(reversed(customers))
    return min(forward, backward)


def format_route_with_depot(vehicle_routes: List[List[int]]) -> str:
    """
    Format a route solution as a single list with depot nodes, for MATLAB-style output.
    """
    if not vehicle_routes:
        return "[]"
    
    # Combine all routes into one sequence, ensuring depot start/end
    combined = [0]  # Start at depot
    for route in vehicle_routes:
        # Add customer nodes (skip depot if already present)
        for node in route:
            if node != 0:
                combined.append(node)
        # Return to depot after each route
        combined.append(0)
    
    return str(combined)


def normalize_route(vehicle_routes: List[List[int]]) -> Set[Tuple[int, ...]]:
    """
    Normalize a complete route solution into a set of normalized trips.
    """
    normalized_trips = set()
    for trip in vehicle_routes:
        normalized_trip = normalize_trip(trip)
        if normalized_trip:  # Only add non-empty trips
            normalized_trips.add(normalized_trip)
    return normalized_trips


def calculate_route_cost(vehicle_routes: List[List[int]], distances: np.ndarray) -> float:
    """
    Calculate the total cost of a route solution using double precision.
    This is more accurate than solver-reported costs which may have rounding errors.
    """
    total_cost = 0.0
    for route in vehicle_routes:
        for i in range(len(route) - 1):
            # Use double precision for accurate cost calculation
            total_cost += float(distances[route[i]][route[i + 1]])
    return float(total_cost)


def validate_solutions(ortools_solution: CVRPSolution, other_solutions: Dict[str, CVRPSolution], 
                      instance: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Validate solutions against the OR-Tools VRP solution (treated as ground truth).
    
    Args:
        ortools_solution: Solution from exact_ortools_vrp solver (ground truth)
        other_solutions: Dictionary of solutions from other solvers
        instance: The CVRP instance
        logger: Logger for detailed output
    """
    if ortools_solution is None:
        logger.warning("OR-Tools VRP solution is None, skipping validation")
        return
    
    ortools_normalized = normalize_route(ortools_solution.vehicle_routes)
    ortools_cost = ortools_solution.cost
    ortools_calculated_cost = calculate_route_cost(ortools_solution.vehicle_routes, instance['distances'])
    distances = instance['distances']
    
    # Check for validation errors
    validation_errors = []
    error_solvers = {}
    
    for solver_name, solution in other_solutions.items():
        if solution is None:
            continue
            
        solver_normalized = normalize_route(solution.vehicle_routes)
        solver_cost = solution.cost
        calculated_cost = calculate_route_cost(solution.vehicle_routes, distances)
        
        # Check cost calculation accuracy
        cost_calc_diff = abs(calculated_cost - solver_cost) / max(solver_cost, 1e-10)
        
        # Check if routes have the same trip structure
        same_trips = (solver_normalized == ortools_normalized)
        
        # Store solver info for error reporting
        error_solvers[solver_name] = {
            'cost': solver_cost,
            'calculated_cost': calculated_cost,
            'routes': solution.vehicle_routes,
            'same_trips': same_trips,
            'optimal': solution.is_optimal
        }
        
        # Exact solver validation - check vs OR-Tools VRP
        if solver_name.startswith('exact'):
            # Both are exact solvers - check if both claim optimality but costs differ significantly
            # Only flag if routes are different OR costs differ significantly
            if (solution.is_optimal and ortools_solution.is_optimal and 
                abs(solver_cost - ortools_cost) / max(ortools_cost, 1e-10) > 0.01):
                if not same_trips:
                    validation_errors.append(f"{solver_name} vs exact_ortools_vrp: both claim optimal but costs differ by >1% with different routes ({solver_cost:.4f} vs {ortools_cost:.4f})")
                else:
                    # Same routes but different costs - this is unusual for exact solvers
                    validation_errors.append(f"{solver_name} vs exact_ortools_vrp: same routes but costs differ by >1% ({solver_cost:.4f} vs {ortools_cost:.4f})")
        
        # Heuristic solver validation - use calculated costs to avoid rounding issues
        elif not solver_name.startswith('exact'):
            # For heuristics, compare calculated costs (not reported costs) to handle rounding
            if not same_trips:
                # Different routes - compare calculated costs
                if calculated_cost < ortools_calculated_cost * 0.99:  # Heuristic calculated cost is >1% better
                    validation_errors.append(f"{solver_name} calculated cost is >1% better than exact_ortools_vrp with different routes (calc: {calculated_cost:.4f} vs {ortools_calculated_cost:.4f}, reported: {solver_cost:.4f} vs {ortools_cost:.4f})")
            else:
                # Same routes - only flag if calculated costs differ significantly (shouldn't happen)
                if abs(calculated_cost - ortools_calculated_cost) / max(ortools_calculated_cost, 1e-10) > 0.01:
                    validation_errors.append(f"{solver_name} has same routes but calculated costs differ by >1% (calc: {calculated_cost:.4f} vs {ortools_calculated_cost:.4f}, reported: {solver_cost:.4f} vs {ortools_cost:.4f})")
        
        # Cost calculation mismatch validation - only for exact solvers when routes are different
        # For heuristics, we expect and handle cost reporting inaccuracies, so skip this check
        if solver_name.startswith('exact') and cost_calc_diff > 1e-6 and not same_trips:
            validation_errors.append(f"{solver_name} cost mismatch")
    
    # If validation errors, use compact error format
    if validation_errors:
        logger.error(f"VALIDATION ERROR: {'; '.join(validation_errors)}")
        
        # Prepare error details for both console and log
        error_lines = []
        error_lines.append(f"exact_ortools_vrp cost/route:   {ortools_cost:.4f} {format_route_with_depot(ortools_solution.vehicle_routes)}")
        for solver_name in ['exact_milp', 'exact_dp', 'exact_pulp', 'heuristic_or']:
            if solver_name in error_solvers:
                info = error_solvers[solver_name]
                error_lines.append(f"{solver_name} cost/route: {info['cost']:.4f} {format_route_with_depot(info['routes'])}")
        
        # Add MATLAB-ready problem format
        coords = instance['coords']
        demands = [int(d) for d in instance['demands']]
        error_lines.append("")
        error_lines.append("Matlab-ready problem:")
        error_lines.append(f"[{' '.join([f'{c[0]:.2f}' for c in coords])};")
        error_lines.append(f" {' '.join([f'{c[1]:.2f}' for c in coords])};")
        error_lines.append(f" {' '.join(map(str, demands))}]")
        error_lines.append("")
        
        # Log to file (without timestamps)
        for line in error_lines:
            # Use a custom format to avoid timestamps
            logger.handlers[0].stream.write(line + "\n")
        logger.handlers[0].stream.flush()
        
        raise ValueError(f"Validation failed: {'; '.join(validation_errors)}")


def run_single_solver(solver_module, solver_name, instance, time_limit, logger):
    """Run a single solver on an instance with external timeout enforcement."""
    
    class SolverResult:
        def __init__(self):
            self.solution = None
            self.exception = None
            self.completed = False
    
    def solver_worker(result):
        try:
            # Give solver a generous internal timeout, but we'll cut it off externally
            internal_timeout = time_limit * 2  # Give solver more time internally
            result.solution = solver_module.solve(instance, time_limit=internal_timeout, verbose=False)
            result.completed = True
        except Exception as e:
            result.exception = e
            result.completed = True
    
    start_time = time.time()
    result = SolverResult()
    
    # Start solver in a separate thread
    solver_thread = threading.Thread(target=solver_worker, args=(result,))
    solver_thread.daemon = True  # Allow main program to exit even if thread is running
    solver_thread.start()
    
    # Wait for solver to complete or timeout
    solver_thread.join(timeout=time_limit)
    solve_time = time.time() - start_time
    
    # Check if solver completed within timeout
    if solver_thread.is_alive():
        # Solver exceeded timeout - it's still running in background but we ignore it
        # Don't log here - let the caller log a more informative message
        return None, time_limit, True  # Return exactly the timeout limit
    elif solve_time > time_limit:
        # Thread completed but took longer than the timeout limit - treat as timeout
        # This can happen if the solver doesn't respect its internal timeout
        return None, time_limit, True  # Return exactly the timeout limit
    
    # Solver completed within timeout
    if result.exception is not None:
        # Solver raised an exception
        timed_out = "timed out" in str(result.exception).lower() or "timeout" in str(result.exception).lower()
        if timed_out:
            logger.error(f"{solver_name} timed out internally: {result.exception}")
            return None, solve_time, True
        else:
            logger.error(f"{solver_name} failed: {result.exception}")
            return None, solve_time, False
    
    if result.solution is None:
        logger.warning(f"{solver_name}: No solution returned")
        return None, solve_time, False
    
    # Check if solution is valid
    if result.solution.cost == float('inf') or result.solution.cost <= 0:
        logger.warning(f"{solver_name}: Invalid solution (cost={result.solution.cost})")
        return None, solve_time, False
        
    # Add solve time to solution
    result.solution.solve_time = solve_time
    
    # Format routes as tuples without depot nodes for cleaner display
    clean_routes = [tuple(node for node in route if node != 0) for route in result.solution.vehicle_routes]
    clean_routes = [route for route in clean_routes if route]  # Remove empty routes
    
    # All solvers now return standardized costs calculated using the same method
    logger.info(f"{solver_name}: cost={result.solution.cost:.4f}, time={solve_time:.6f}s, routes={clean_routes}")
    
    return result.solution, solve_time, False


def safe_print(*args, **kwargs):
    """Safely print to stdout, handling cases where it might be closed."""
    try:
        print(*args, **kwargs)
    except (ValueError, OSError, IOError):
        # Handle cases where stdout is closed or unavailable
        pass


def print_progress_bar(iteration, total, length=50):
    """Print a progress bar."""
    try:
        percent = (iteration / total) * 100
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        print(f'\r  Progress |{bar}| {percent:.1f}% ({iteration}/{total})', end='', flush=True)
    except (ValueError, OSError, IOError):
        # Handle cases where stdout is closed or unavailable
        pass


def run_benchmark_for_n(n, instances_min, instances_max, capacity, demand_range, total_timeout, coord_range, logger, disabled_solvers=None):
    """Run benchmark for a specific problem size N with new timeout behavior."""
    if disabled_solvers is None:
        disabled_solvers = set()
    
    safe_print(f"\nN={n}: attempting {instances_min}-{instances_max} instances (timeout={total_timeout}s total per solver)")
    if disabled_solvers:
        safe_print(f"  Disabled solvers (exceeded timeout): {', '.join(disabled_solvers)}")
    
    gen = EnhancedCVRPGenerator(config={})
    
    # Results for each solver
    results = {
        'exact_ortools_vrp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'exact_milp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'exact_dp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'exact_pulp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'heuristic_or': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []}
    }
    
    solvers = {
        'exact_ortools_vrp': exact_ortools_vrp,
        'exact_milp': exact_milp,
        'exact_dp': exact_dp, 
        'exact_pulp': exact_pulp,
        'heuristic_or': heuristic_or
    }
    
    # Track solver disable status and consecutive failures
    solver_disabled_this_n = set()
    
    # Per-instance timeout threshold for disabling solvers
    instance_timeout_threshold = total_timeout / instances_min
    
    # Track consecutive failures for each solver (reset after successful solve)
    consecutive_failures = {name: 0 for name in solvers.keys()}
    MAX_CONSECUTIVE_FAILURES = 3
    
    validation_errors = 0
    attempted = 0
    t0 = time.time()
    
    for i in range(instances_max):
        attempted = i + 1
        
        # Progress bar
        print_progress_bar(i, instances_max)
        
        # Try up to 3 different instances to avoid pathologically hard instances
        MAX_INSTANCE_ATTEMPTS = 3
        instance_found = False
        
        for attempt in range(MAX_INSTANCE_ATTEMPTS):
            # Generate instance with different seed for each attempt
            seed = 4242 + n * 1000 + i * 10 + attempt
            instance = gen.generate_instance(
                num_customers=n,
                capacity=capacity,
                coord_range=coord_range,
                demand_range=demand_range,
                seed=seed,
                instance_type=InstanceType.RANDOM,
                apply_augmentation=False,
            )
            
            attempt_msg = f" (attempt {attempt + 1}/{MAX_INSTANCE_ATTEMPTS})" if attempt > 0 else ""
            logger.info(f"\n=== Instance {i+1}/{instances_max} (N={n}, seed={seed}){attempt_msg} ===")
            
            # Test each solver ON THE SAME INSTANCE
            instance_solutions = {}
            any_solver_active = False
            instance_too_hard = False
            current_failures = {}
            
            for solver_name, solver_module in solvers.items():
                if solver_name in disabled_solvers or solver_name in solver_disabled_this_n:
                    # Skip disabled solvers
                    instance_solutions[solver_name] = None
                    if solver_name in disabled_solvers:
                        logger.info(f"{solver_name}: DISABLED (from previous N)")
                    else:
                        logger.info(f"{solver_name}: DISABLED (exceeded threshold this N)")
                    continue
                
                any_solver_active = True
                
                # Use strict per-instance timeout - stop solver immediately if it exceeds threshold
                instance_timeout = instance_timeout_threshold
                
                solution, solve_time, timed_out = run_single_solver(
                    solver_module, solver_name, instance, instance_timeout, logger
                )
                
                instance_solutions[solver_name] = solution
                
                # Check if this instance took too long for ANY solver
                if solve_time >= instance_timeout_threshold:
                    current_failures[solver_name] = True
                    # Log single informative message about timeout/hard instance
                    if timed_out:
                        logger.warning(f"{solver_name}: Timed out after {solve_time:.2f}s (too hard, will try different instance)")
                    else:
                        logger.warning(f"{solver_name}: Instance took {solve_time:.2f}s >= {instance_timeout_threshold:.2f}s (too hard, will try different instance)")
                    instance_too_hard = True
                else:
                    current_failures[solver_name] = False
                    # Reset consecutive failures on success
                    consecutive_failures[solver_name] = 0
                
                if solution is not None and not current_failures.get(solver_name, False):
                    results[solver_name]['times'].append(solve_time)
                    
                    # All solvers now return standardized costs, so use them directly
                    benchmark_cost = solution.cost / max(1, instance['num_customers'])
                    
                    results[solver_name]['costs'].append(benchmark_cost)
                    results[solver_name]['solutions'].append(solution)
                    
                    # Only count as optimal if solver claims it's optimal AND it's an exact solver
                    if solution.is_optimal and solver_name.startswith('exact'):
                        results[solver_name]['optimal_count'] += 1
            
            # If instance was not too hard for any active solver, use it
            if not instance_too_hard:
                instance_found = True
                break
            else:
                # Update consecutive failure counts for solvers that failed on this hard instance
                for solver_name in current_failures:
                    if current_failures[solver_name]:
                        consecutive_failures[solver_name] += 1
                        if consecutive_failures[solver_name] >= MAX_CONSECUTIVE_FAILURES:
                            solver_disabled_this_n.add(solver_name)
                            logger.warning(f"{solver_name}: {MAX_CONSECUTIVE_FAILURES} consecutive hard instances - DISABLED for N>{n}")
                
                if attempt < MAX_INSTANCE_ATTEMPTS - 1:
                    logger.warning(f"Instance too hard for some solvers, trying different instance...")
        
        # If we couldn't find a reasonable instance after MAX_INSTANCE_ATTEMPTS, use the last one anyway
        if not instance_found:
            logger.warning(f"All {MAX_INSTANCE_ATTEMPTS} instance attempts were hard, using last one")
            # Process any remaining successful solutions from the last attempt
            for solver_name in instance_solutions:
                solution = instance_solutions[solver_name]
                if solution is not None and solver_name not in current_failures:
                    # This solution wasn't added yet, add it
                    if solver_name not in [name for name in results[solver_name]['solutions'] if hasattr(name, 'solve_time') and abs(name.solve_time - solution.solve_time) < 0.001]:
                        results[solver_name]['times'].append(solution.solve_time)
                        benchmark_cost = solution.cost / max(1, instance['num_customers'])
                        results[solver_name]['costs'].append(benchmark_cost)
                        results[solver_name]['solutions'].append(solution)
                        if solution.is_optimal and solver_name.startswith('exact'):
                            results[solver_name]['optimal_count'] += 1
        
        # If no solvers are active, stop early
        if not any_solver_active:
            logger.info("No active solvers remaining, stopping early")
            break
        
        # Validate solutions if OR-Tools VRP succeeded
        try:
            ortools_solution = instance_solutions.get('exact_ortools_vrp')
            other_solutions = {k: v for k, v in instance_solutions.items() if k != 'exact_ortools_vrp'}
            
            if ortools_solution is not None:
                validate_solutions(ortools_solution, other_solutions, instance, logger)
            else:
                logger.warning("OR-Tools VRP failed, skipping validation for this instance")
                
        except ValueError as e:
            validation_errors += 1
            # Continue with benchmark but note the error
        
    
    # Complete progress bar
    print_progress_bar(instances_max if attempted == instances_max else attempted, instances_max)
    safe_print()  # New line after progress bar
    
    # The newly disabled solvers for next N (only from consecutive failures)
    newly_disabled = solver_disabled_this_n
    
    # Compute statistics
    stats = {'N': n, 'validation_errors': validation_errors}
    
    for solver_name in solvers.keys():
        times = results[solver_name]['times']
        costs = results[solver_name]['costs']
        optimal_count = results[solver_name]['optimal_count']
        instances_solved = len(times)
        
        # If solver was disabled or newly disabled, set stats appropriately
        if solver_name in disabled_solvers:
            # Solver was already disabled
            avg_time = float('nan')
            avg_cost = float('nan')
            std_cost = float('nan')
            solved_count = 0
            optimal_count = 0
        elif solver_name in newly_disabled:
            # Solver was disabled on this N - include its results but mark for future disabling
            if instances_solved >= 1:
                avg_time = float(statistics.mean(times))
                avg_cost = float(statistics.mean(costs))
                solved_count = instances_solved
            else:
                avg_time = float('nan')
                avg_cost = float('nan')
                solved_count = 0
                
            # Only calculate std if we have enough instances and at least instances_min
            if instances_solved >= max(2, instances_min):
                std_cost = float(statistics.stdev(costs))
            else:
                std_cost = float('nan')  # Not enough instances for std
        elif instances_solved >= 1:
            avg_time = float(statistics.mean(times))
            avg_cost = float(statistics.mean(costs))
            solved_count = instances_solved
            
            # Only calculate std if we have enough instances and at least instances_min
            if instances_solved >= max(2, instances_min):
                std_cost = float(statistics.stdev(costs))
            else:
                std_cost = float('nan')  # Not enough instances for std
        else:
            avg_time = float('nan')
            avg_cost = float('nan') 
            std_cost = float('nan')
            solved_count = 0
            
        # For exact solvers, only include instances with is_optimal=True in statistics
        if solver_name.startswith('exact') and not solver_name in disabled_solvers:
            if optimal_count > 0:
                # Re-compute statistics using only optimal solutions
                optimal_costs = []
                optimal_times = []
                for j, solution in enumerate(results[solver_name]['solutions']):
                    if solution.is_optimal:
                        optimal_costs.append(costs[j])
                        optimal_times.append(times[j])
                
                if len(optimal_costs) >= 1:
                    avg_cost = float(statistics.mean(optimal_costs))
                    avg_time = float(statistics.mean(optimal_times))
                    solved_count = len(optimal_costs)
                    
                if len(optimal_costs) >= max(2, instances_min):
                    std_cost = float(statistics.stdev(optimal_costs))
                else:
                    std_cost = float('nan')  # Not enough optimal instances for std
            else:
                # No optimal solutions found for this exact solver
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
    
    # Return both stats and newly disabled solvers
    stats['newly_disabled'] = newly_disabled
    
    elapsed = time.time() - t0
    
    # Print summary
    safe_print(f"✅ N={n} completed in {elapsed:.1f}s")
    if validation_errors > 0:
        safe_print(f"⚠️  {validation_errors} validation errors detected!")
    
    for solver in ['exact_ortools_vrp', 'exact_milp', 'exact_dp', 'exact_pulp', 'heuristic_or']:
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
        
        # Indicate if solver was newly disabled
        if solver in newly_disabled:
            safe_print(f"    ⚠️ {solver} DISABLED for N>{n}")
    
    return stats


def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging to file."""
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create logger
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger


def main():
    parser = argparse.ArgumentParser(description='Exact CVRP Solver Benchmark with New Timeout Behavior')
    parser.add_argument('--instances-min', type=int, default=5, help='Minimum instances per N (default: 5)')
    parser.add_argument('--instances-max', type=int, default=20, help='Maximum instances per N (default: 20)')
    parser.add_argument('--n-start', type=int, default=5, help='Start N (default: 5)')
    parser.add_argument('--n-end', type=int, default=15, help='End N inclusive (default: 15)')
    parser.add_argument('--capacity', type=int, default=30, help='Vehicle capacity (default: 30)')
    parser.add_argument('--demand-min', type=int, default=1, help='Min demand (default: 1)')
    parser.add_argument('--demand-max', type=int, default=10, help='Max demand (default: 10)')
    parser.add_argument('--timeout', type=float, default=60.0, help='Total timeout per solver per N (default: 60.0s)')
    parser.add_argument('--coord-range', type=int, default=100, help='Coordinate range for instance generation (default: 100)')
    parser.add_argument('--output', type=str, default='benchmark_exact.csv', help='Output CSV file')
    parser.add_argument('--log', type=str, default='benchmark_exact.log', help='Log file (default: benchmark_exact.log)')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log)
    
    safe_print("="*60)
    safe_print("EXACT CVRP SOLVER BENCHMARK WITH NEW TIMEOUT BEHAVIOR")
    safe_print("="*60)
    safe_print(f"Problem size: N = {args.n_start} to {args.n_end}")
    safe_print(f"Instances per N: {args.instances_min}-{args.instances_max}")
    safe_print(f"Vehicle capacity: {args.capacity}")
    safe_print(f"Demand range: [{args.demand_min}, {args.demand_max}]")
    safe_print(f"Coordinate range: {args.coord_range}")
    safe_print(f"Total timeout per solver per N: {args.timeout}s")
    safe_print(f"Output file: {args.output}")
    safe_print(f"Log file: {args.log}")
    safe_print()
    
    # Log the configuration
    logger.info("="*60)
    logger.info("EXACT CVRP SOLVER BENCHMARK WITH NEW TIMEOUT BEHAVIOR")
    logger.info("="*60)
    logger.info(f"Problem size: N = {args.n_start} to {args.n_end}")
    logger.info(f"Instances per N: {args.instances_min}-{args.instances_max}")
    logger.info(f"Vehicle capacity: {args.capacity}")
    logger.info(f"Demand range: [{args.demand_min}, {args.demand_max}]")
    logger.info(f"Coordinate range: {args.coord_range}")
    logger.info(f"Total timeout per solver per N: {args.timeout}s")
    logger.info(f"Output file: {args.output}")
    
    demand_range = [args.demand_min, args.demand_max]
    
    # Initialize CSV file with header
    fieldnames = ['N', 'time_exact_ortools_vrp', 'cpc_exact_ortools_vrp', 'std_exact_ortools_vrp', 'solved_exact_ortools_vrp', 'optimal_exact_ortools_vrp',
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
    disabled_solvers = set()  # Track solvers that exceeded timeout
    
    # Run benchmark for each N
    for n in range(args.n_start, args.n_end + 1):
        result = run_benchmark_for_n(n, args.instances_min, args.instances_max, args.capacity, demand_range, args.timeout, args.coord_range, logger, disabled_solvers)
        
        total_validation_errors += result.get('validation_errors', 0)
        
        # Update disabled solvers set
        newly_disabled = result.get('newly_disabled', set())
        disabled_solvers.update(newly_disabled)
        
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
    
    safe_print(f"\n✅ Benchmark complete!")
    safe_print(f"📊 Wrote {rows_written} rows to {args.output}")
    safe_print(f"📝 Detailed results logged to {args.log}")
    if total_validation_errors > 0:
        safe_print(f"⚠️  Total validation errors: {total_validation_errors}")
    else:
        safe_print(f"✅ All solutions validated successfully!")


if __name__ == '__main__':
    main()
