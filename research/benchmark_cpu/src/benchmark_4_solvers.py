#!/usr/bin/env python3
"""
4-Solver CVRP Benchmark with Solution Validation
Compares exact_milp, exact_dp, heuristic_or, heuristic_dp solvers
Validates solutions and logs detailed results to benchmark_4.log
"""

import argparse
import numpy as np
import time
import csv
import sys
import statistics
import logging
import os
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional

# Import solvers
import solvers.exact_milp as exact_milp
import solvers.exact_dp as exact_dp
import solvers.heuristic_or as heuristic_or
import solvers.heuristic_dp as heuristic_dp

# Import generator from research folder (resolve relative to this script)
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parents[3]
sys.path.insert(0, str(_repo_root))  # Ensure repo root is importable (for solvers/*)
sys.path.append(str(_script_dir.parents[2] / 'benchmark_exact'))
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
    Calculate the total cost of a route solution.
    """
    total_cost = 0.0
    for route in vehicle_routes:
        for i in range(len(route) - 1):
            total_cost += distances[route[i]][route[i + 1]]
    return total_cost


def validate_solutions(milp_solution: CVRPSolution, other_solutions: Dict[str, CVRPSolution], 
                      instance: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Validate solutions against the MILP solution (treated as ground truth).
    
    Args:
        milp_solution: Solution from exact_milp solver
        other_solutions: Dictionary of solutions from other solvers
        instance: The CVRP instance
        logger: Logger for detailed output
    """
    if milp_solution is None:
        logger.warning("MILP solution is None, skipping validation")
        return
    
    milp_normalized = normalize_route(milp_solution.vehicle_routes)
    milp_cost = milp_solution.cost
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
        same_trips = (solver_normalized == milp_normalized)
        
        # Store solver info for error reporting
        error_solvers[solver_name] = {
            'cost': solver_cost,
            'routes': solution.vehicle_routes,
            'same_trips': same_trips,
            'optimal': solution.is_optimal
        }
        
        # Exact solver validation - check exact_dp vs exact_milp discrepancies
        if solver_name == 'exact_dp':
            # Both are exact solvers - check if both claim optimality but costs differ significantly
            if (solution.is_optimal and milp_solution.is_optimal and 
                abs(solver_cost - milp_cost) / max(milp_cost, 1e-10) > 0.01):
                validation_errors.append(f"exact_dp vs exact_milp: both claim optimal but costs differ by >1% ({solver_cost:.4f} vs {milp_cost:.4f})")
        
        # Heuristic solver validation - only flag if heuristic is >1% better AND routes differ
        elif not solver_name.startswith('exact'):
            if (not same_trips and 
                solver_cost < milp_cost * 0.99):  # Heuristic is more than 1% better
                validation_errors.append(f"{solver_name} is >1% better than exact_milp with different routes ({solver_cost:.4f} vs {milp_cost:.4f})")
        
        # Cost mismatch validation
        if cost_calc_diff > 1e-6 and not same_trips:
            validation_errors.append(f"{solver_name} cost mismatch")
    
    # If validation errors, use new compact error format
    if validation_errors:
        logger.error(f"VALIDATION ERROR: {'; '.join(validation_errors)}")
        
        # Prepare error details for both console and log
        error_lines = []
        error_lines.append(f"exact_milp cost/route:   {milp_cost:.4f} {format_route_with_depot(milp_solution.vehicle_routes)}")
        for solver_name in ['exact_dp', 'heuristic_or', 'heuristic_dp']:
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
    """Run a single solver on an instance and return the full solution and timeout status."""
    try:
        start_time = time.time()
        # Use individual instance timeout - let solvers handle their own timeouts
        solution = solver_module.solve(instance, time_limit=time_limit, verbose=False)
        solve_time = time.time() - start_time
        
        # Check if solution is valid
        if solution.cost == float('inf') or solution.cost <= 0:
            logger.warning(f"{solver_name}: Invalid solution (cost={solution.cost})")
            return None, solve_time, False
            
        # Add solve time to solution
        solution.solve_time = solve_time
        
        # Format routes as tuples without depot nodes for cleaner display
        clean_routes = [tuple(node for node in route if node != 0) for route in solution.vehicle_routes]
        clean_routes = [route for route in clean_routes if route]  # Remove empty routes
        logger.info(f"{solver_name}: cost={solution.cost:.4f}, time={solve_time:.6f}s, routes={clean_routes}")
        
        return solution, solve_time, False
        
    except Exception as e:
        solve_time = time.time() - start_time
        # Check if it's a timeout exception
        timed_out = "timed out" in str(e).lower() or "timeout" in str(e).lower()
        if timed_out:
            logger.error(f"{solver_name} timed out: {e}")
            return None, solve_time, True
        else:
            logger.error(f"{solver_name} failed: {e}")
            return None, solve_time, False


def print_progress_bar(iteration, total, length=50):
    """Print a progress bar."""
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r  Progress |{bar}| {percent:.1f}% ({iteration}/{total})', end='', flush=True)


def run_benchmark_for_n(n, instances_min, instances_max, capacity, demand_range, total_timeout, coord_range, logger, disabled_solvers=None):
    """Run benchmark for a specific problem size N with new timeout behavior.
    
    New timeout behavior:
    - total_timeout is the total time limit for one solver for all instances at this N
    - Generate one instance, test all 4 solvers on it, then move to next instance
    - If any solver takes more than timeout/instances_min on an instance, disable it for future N
    - Continue until budget is exhausted or instances_max is reached
    """
    if disabled_solvers is None:
        disabled_solvers = set()
    
    print(f"\nN={n}: attempting {instances_min}-{instances_max} instances (timeout={total_timeout}s total per solver)")
    if disabled_solvers:
        print(f"  Disabled solvers (exceeded timeout): {', '.join(disabled_solvers)}")
    
    gen = EnhancedCVRPGenerator(config={})
    
    # Results for each solver
    results = {
        'exact_milp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'exact_dp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'heuristic_or': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []},
        'heuristic_dp': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []}
    }
    
    solvers = {
        'exact_milp': exact_milp,
        'exact_dp': exact_dp, 
        'heuristic_or': heuristic_or,
        'heuristic_dp': heuristic_dp
    }
    
    # Track solver time budgets and disable status
    solver_budgets = {name: total_timeout for name in solvers.keys()}
    solver_disabled_this_n = set()
    
    # Per-instance timeout threshold for disabling solvers
    instance_timeout_threshold = total_timeout / instances_min
    
    validation_errors = 0
    attempted = 0
    t0 = time.time()
    
    for i in range(instances_max):
        attempted = i + 1
        
        # Progress bar
        print_progress_bar(i, instances_max)
        
        # Generate instance ONCE per iteration
        seed = 4242 + n * 1000 + i
        instance = gen.generate_instance(
            num_customers=n,
            capacity=capacity,
            coord_range=coord_range,
            demand_range=demand_range,
            seed=seed,
            instance_type=InstanceType.RANDOM,
            apply_augmentation=False,
        )
        
        logger.info(f"\n=== Instance {i+1}/{instances_max} (N={n}, seed={seed}) ===")
        
        # Test each solver ON THE SAME INSTANCE
        instance_solutions = {}
        any_solver_active = False
        
        for solver_name, solver_module in solvers.items():
            if solver_name in disabled_solvers or solver_name in solver_disabled_this_n:
                # Skip disabled solvers
                instance_solutions[solver_name] = None
                if solver_name in disabled_solvers:
                    logger.info(f"{solver_name}: DISABLED (from previous N)")
                else:
                    logger.info(f"{solver_name}: DISABLED (exceeded threshold this N)")
                continue
            
            # Check if solver has budget remaining
            if solver_budgets[solver_name] <= 0:
                logger.info(f"{solver_name}: Budget exhausted ({solver_budgets[solver_name]:.2f}s remaining)")
                instance_solutions[solver_name] = None
                continue
                
            any_solver_active = True
            
            # Use generous individual instance timeout
            instance_timeout = max(60.0, total_timeout * 0.5)
            
            solution, solve_time, timed_out = run_single_solver(
                solver_module, solver_name, instance, instance_timeout, logger
            )
            
            instance_solutions[solver_name] = solution
            
            # Update solver budget
            solver_budgets[solver_name] -= solve_time
            
            # Check if this instance took too long (disable solver for future N)
            if solve_time > instance_timeout_threshold:
                solver_disabled_this_n.add(solver_name)
                logger.warning(f"{solver_name}: Instance took {solve_time:.2f}s > {instance_timeout_threshold:.2f}s (timeout/instances_min) - DISABLED for N>{n}")
            
            if solution is not None:
                results[solver_name]['times'].append(solve_time)
                results[solver_name]['costs'].append(solution.cost / max(1, instance['num_customers']))
                results[solver_name]['solutions'].append(solution)
                
                # Only count as optimal if solver claims it's optimal AND it's an exact solver
                if solution.is_optimal and solver_name.startswith('exact'):
                    results[solver_name]['optimal_count'] += 1
        
        # If no solvers are active, stop early
        if not any_solver_active:
            logger.info("No active solvers remaining, stopping early")
            break
        
        # Validate solutions if MILP succeeded
        try:
            milp_solution = instance_solutions.get('exact_milp')
            other_solutions = {k: v for k, v in instance_solutions.items() if k != 'exact_milp'}
            
            if milp_solution is not None:
                validate_solutions(milp_solution, other_solutions, instance, logger)
            else:
                logger.warning("MILP failed, skipping validation for this instance")
                
        except ValueError as e:
            validation_errors += 1
            # Continue with benchmark but note the error
        
        # Check if all solvers have exhausted their budgets
        active_budgets = [b for name, b in solver_budgets.items() 
                         if name not in disabled_solvers and name not in solver_disabled_this_n]
        if all(budget <= 0 for budget in active_budgets):
            logger.info("All active solvers have exhausted their budgets")
            break
    
    # Complete progress bar
    print_progress_bar(instances_max if attempted == instances_max else attempted, instances_max)
    print()  # New line after progress bar
    
    # Add solvers that exhausted their budget to newly_disabled
    budget_exhausted = set()
    for solver_name, remaining_budget in solver_budgets.items():
        if remaining_budget <= 0 and solver_name not in disabled_solvers:
            budget_exhausted.add(solver_name)
            logger.warning(f"{solver_name}: Budget exhausted ({remaining_budget:.2f}s remaining) - DISABLED for N>{n}")
    
    # The newly disabled solvers for next N (both timeout threshold and budget exhausted)
    newly_disabled = solver_disabled_this_n | budget_exhausted
    
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
    print(f"✅ N={n} completed in {elapsed:.1f}s")
    if validation_errors > 0:
        print(f"⚠️  {validation_errors} validation errors detected!")
    
    for solver in ['exact_milp', 'exact_dp', 'heuristic_or', 'heuristic_dp']:
        solved = stats.get(f'solved_{solver}', 0)
        optimal = stats.get(f'optimal_{solver}', 0) if solver.startswith('exact') else 'N/A'
        avg_time = stats.get(f'time_{solver}', float('nan'))
        avg_cpc = stats.get(f'cpc_{solver}', float('nan'))
        
        time_str = f"{avg_time:.4f}s" if not np.isnan(avg_time) else "nan"
        cpc_str = f"{avg_cpc:.4f}" if not np.isnan(avg_cpc) else "nan"
        
        if solver.startswith('exact'):
            print(f"  {solver}: {solved} solved, {optimal} optimal, time={time_str}, cpc={cpc_str}")
        else:
            print(f"  {solver}: {solved} solved, time={time_str}, cpc={cpc_str}")
        
        # Indicate if solver was newly disabled
        if solver in newly_disabled:
            print(f"    ⚠️ {solver} DISABLED for N>{n}")
    
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
    parser = argparse.ArgumentParser(description='4-Solver CVRP Benchmark with New Timeout Behavior')
    parser.add_argument('--instances-min', type=int, default=5, help='Minimum instances per N (default: 5)')
    parser.add_argument('--instances-max', type=int, default=20, help='Maximum instances per N (default: 20)')
    parser.add_argument('--n-start', type=int, default=5, help='Start N (default: 5)')
    parser.add_argument('--n-end', type=int, default=15, help='End N inclusive (default: 15)')
    parser.add_argument('--capacity', type=int, default=30, help='Vehicle capacity (default: 30)')
    parser.add_argument('--demand-min', type=int, default=1, help='Min demand (default: 1)')
    parser.add_argument('--demand-max', type=int, default=10, help='Max demand (default: 10)')
    parser.add_argument('--timeout', type=float, default=60.0, help='Total timeout per solver per N (default: 60.0s)')
    parser.add_argument('--coord-range', type=int, default=100, help='Coordinate range for instance generation (default: 100)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: research/benchmark_cpu/csv/benchmark_4.csv)')
    parser.add_argument('--log', type=str, default=None, help='Log file (default: research/benchmark_cpu/log/benchmark_4.log)')
    
    args = parser.parse_args()
    
    # Resolve benchmark CPU base directories relative to this file
    _script_dir = Path(__file__).resolve().parent
    _benchmark_base = _script_dir.parent  # research/benchmark_cpu
    _csv_dir = _benchmark_base / 'csv'
    _log_dir = _benchmark_base / 'log'
    os.makedirs(_csv_dir, exist_ok=True)
    os.makedirs(_log_dir, exist_ok=True)

    # Determine output paths
    output_csv = args.output if args.output else str(_csv_dir / 'benchmark_4.csv')
    log_file = args.log if args.log else str(_log_dir / 'benchmark_4.log')
    
    # Set up logging
    logger = setup_logging(log_file)
    
    print("="*60)
    print("4-SOLVER CVRP BENCHMARK WITH NEW TIMEOUT BEHAVIOR")
    print("="*60)
    print(f"Problem size: N = {args.n_start} to {args.n_end}")
    print(f"Instances per N: {args.instances_min}-{args.instances_max}")
    print(f"Vehicle capacity: {args.capacity}")
    print(f"Demand range: [{args.demand_min}, {args.demand_max}]")
    print(f"Coordinate range: {args.coord_range}")
    print(f"Total timeout per solver per N: {args.timeout}s")
    print(f"Output file: {output_csv}")
    print(f"Log file: {log_file}")
    print()
    
    # Log the configuration
    logger.info("="*60)
    logger.info("4-SOLVER CVRP BENCHMARK WITH NEW TIMEOUT BEHAVIOR")
    logger.info("="*60)
    logger.info(f"Problem size: N = {args.n_start} to {args.n_end}")
    logger.info(f"Instances per N: {args.instances_min}-{args.instances_max}")
    logger.info(f"Vehicle capacity: {args.capacity}")
    logger.info(f"Demand range: [{args.demand_min}, {args.demand_max}]")
    logger.info(f"Coordinate range: {args.coord_range}")
    logger.info(f"Total timeout per solver per N: {args.timeout}s")
    logger.info(f"Output file: {output_csv}")
    
    demand_range = [args.demand_min, args.demand_max]
    
    # Initialize CSV file with header
    fieldnames = ['N', 'time_exact_milp', 'cpc_exact_milp', 'std_exact_milp', 'solved_exact_milp', 'optimal_exact_milp',
                  'time_exact_dp', 'cpc_exact_dp', 'std_exact_dp', 'solved_exact_dp', 'optimal_exact_dp',
                  'time_heuristic_or', 'cpc_heuristic_or', 'std_heuristic_or', 'solved_heuristic_or', 
                  'time_heuristic_dp', 'cpc_heuristic_dp', 'std_heuristic_dp', 'solved_heuristic_dp']
                  
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
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
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
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
    
    print(f"\n✅ Benchmark complete!")
    print(f"📊 Wrote {rows_written} rows to {output_csv}")
    print(f"📝 Detailed results logged to {log_file}")
    if total_validation_errors > 0:
        print(f"⚠️  Total validation errors: {total_validation_errors}")
    else:
        print(f"✅ All solutions validated successfully!")


if __name__ == '__main__':
    main()
