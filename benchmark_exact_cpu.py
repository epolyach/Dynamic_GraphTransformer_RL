#!/usr/bin/env python3


"""
Exact CVRP Solver Benchmark with Solution Validation
Compares exact_ortools_vrp, exact_milp, and heuristic_or solvers
Validates solutions and logs detailed results to benchmark_exact.log
Updated to produce CSV format compatible with plot_cpu_benchmark.py
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


# Inline config loading functionality
import json
import os


# ============================================================================
# MATLAB LOGGING CONFIGURATION AND FUNCTIONS
# ============================================================================

MATLAB_LOG_ENABLED = True  # Set to False to disable MATLAB logging
MATLAB_LOG_MAX_INSTANCES = 5  # Log first N instances per problem size

def format_route_for_matlab(vehicle_routes, solver_name=""):
    """Format vehicle routes for MATLAB log with depot nodes included.
    
    For heuristic solver, routes don't include depot nodes, so we add them.
    For other solvers, routes already include depot nodes.
    """
    if not vehicle_routes:
        return []
    
    # Check if this is from the heuristic solver
    is_heuristic = 'heuristic' in solver_name.lower()
    
    # Flatten multiple vehicle routes into a single route
    merged_route = []
    
    for i, route in enumerate(vehicle_routes):
        if not route:
            continue
        
        if is_heuristic:
            # Heuristic solver: routes don't have depot nodes
            if i == 0 or not merged_route:
                merged_route.append(0)
            merged_route.extend(route)
            merged_route.append(0)
        else:
            # Other solvers: routes already have depot nodes
            if i == 0:
                merged_route.extend(route)
            else:
                # Skip leading depot for subsequent routes
                start_idx = 1 if route and route[0] == 0 else 0
                merged_route.extend(route[start_idx:])
    
    # Clean up duplicate consecutive depot nodes
    if merged_route:
        cleaned = [merged_route[0]]
        for j in range(1, len(merged_route)):
            if not (merged_route[j] == 0 and merged_route[j-1] == 0):
                cleaned.append(merged_route[j])
        merged_route = cleaned
    
    # Ensure it starts and ends with depot
    if merged_route and merged_route[0] != 0:
        merged_route = [0] + merged_route
    if merged_route and merged_route[-1] != 0:
        merged_route = merged_route + [0]
    
    return merged_route


class MatlabLogger:
    """Logger for MATLAB-formatted output."""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.instance_counts = {}  # Track instances logged per N
        self.file_handle = None
        
        if log_file and MATLAB_LOG_ENABLED:
            # Create directory if needed
            from pathlib import Path
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            self.file_handle = open(log_file, 'w')
            self._write_header()
    
    def _write_header(self):
        """Write MATLAB log header."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file_handle.write("CPU CVRP Benchmark - MATLAB Format Log\n")
        self.file_handle.write(f"Date: {timestamp}\n")
        self.file_handle.write("Config: capacity=30, demand=[1,10]\n")
        self.file_handle.write("\n")
        self.file_handle.flush()

    def should_log_instance(self, n):
        """Check if we should log this instance."""
        if not MATLAB_LOG_ENABLED or not self.file_handle:
            return False
        
        if n not in self.instance_counts:
            self.instance_counts[n] = 0
        
        return self.instance_counts[n] < MATLAB_LOG_MAX_INSTANCES
    
    def log_instance_start(self, n, instance, instance_num):
        """Log the start of a new instance."""
        if not MATLAB_LOG_ENABLED or not self.file_handle:
            return
            
        if n not in self.instance_counts:
            self.instance_counts[n] = 0
            
        # Check if we should log this instance
        if self.instance_counts[n] >= MATLAB_LOG_MAX_INSTANCES:
            return
            
        # Increment counter only if we're actually logging
        self.instance_counts[n] += 1
        
        # Format problem data
        coords = instance["coords"]
        demands = instance["demands"]
        
        x_coords = " ".join([f"{c[0]:.2f}" for c in coords])
        y_coords = " ".join([f"{c[1]:.2f}" for c in coords])
        demand_str = " ".join([f"{int(d)}" for d in demands])
        
        # Write instance header (no % prefix, no capacity repeat)
        # Add empty line before instance (except first)
        if self.instance_counts[n] > 1:
            self.file_handle.write("\n")
        
        self.file_handle.write(f"N={n}, Instance {self.instance_counts[n]}:\n")
        self.file_handle.write(f"problem_matrix = [{x_coords};\n")
        self.file_handle.write(f"                  {y_coords};\n")
        self.file_handle.write(f"                  {demand_str}];\n")
        self.file_handle.flush()


    def log_solver_result(self, n, solver_name, solution):
        """Log a solver's result."""
        if not self.file_handle or not self.should_log_instance(n) or not solution:
            return
        
        # Calculate CPC
        cpc = solution.cost / n
        
        # Format solver name for display
        if 'exact_milp' in solver_name.lower():
            solver_display = "Exact (MILP)"
        elif 'exact_ortools' in solver_name.lower():
            solver_display = "Metaheuristic (OR-Tools)"
        elif 'heuristic' in solver_name.lower():
            solver_display = "Heuristic (OR-Tools)"
        else:
            solver_display = solver_name
        
        # Format to consistent width
        solver_display = f"{solver_display:25s}"
        
        # Format route with depot nodes
        route = format_route_for_matlab(solution.vehicle_routes, solver_name)
        
        # Write result (no % prefix)
        self.file_handle.write(f"{solver_display} {cpc:.4f} {route}\n")
        self.file_handle.flush()

    
    def end_instance(self):
        """Mark the end of an instance with an empty line."""
        if self.file_handle:
            self.file_handle.write("\n")
            self.file_handle.flush()

    def close(self):
        """Close the log file."""
        if self.file_handle:
            self.file_handle.close()

# ============================================================================
# END OF MATLAB LOGGING
# ============================================================================

def load_config(config_path: str = "config.json"):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def get_instance_params(config):
    """Extract instance generation parameters from config."""
    instance_config = config["instance_generation"]
    return {
        "capacity": instance_config["capacity"],
        "demand_range": [instance_config["demand_min"], instance_config["demand_max"]],
        "coord_range": instance_config["coord_range"]
    }

def validate_config(config):
    """Validate that config contains required parameters."""
    required_sections = ["instance_generation", "benchmark_settings", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    instance_params = ["capacity", "demand_min", "demand_max", "coord_range"]
    for param in instance_params:
        if param not in config["instance_generation"]:
            raise ValueError(f"Missing required instance parameter: {param}")
    
    print(f"✅ Config validation passed")
    print(f"   - Capacity: {config['instance_generation']['capacity']}")
    print(f"   - Demand range: [{config['instance_generation']['demand_min']}, {config['instance_generation']['demand_max']}]")
    print(f"   - Coordinate range: [0, {config['instance_generation']['coord_range']}] normalized to [0, 1]")



# Import solvers
import solvers.exact_ortools_vrp as exact_ortools_vrp
import solvers.exact_milp as exact_milp
import solvers.heuristic_or as heuristic_or

# Import generator from research folder
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
from solvers.types import CVRPSolution
from datetime import datetime


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


def normalize_routes_to_depot_free(routes):
    """
    Normalize routes to depot-free format.
    Removes depot node (0) from routes and handles empty routes.
    """
    normalized = []
    for route in routes:
        # Remove depot nodes (0) and keep non-empty routes
        depot_free = [node for node in route if node != 0]
        if depot_free:  # Only add non-empty routes
            normalized.append(depot_free)
    return normalized


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
    # Normalize routes to depot-free format for consistent cost calculation
    ortools_depot_free = normalize_routes_to_depot_free(ortools_solution.vehicle_routes)
    ortools_calculated_cost = calculate_route_cost(ortools_depot_free, instance['distances'])
    distances = instance['distances']
    
    # Check for validation errors
    validation_errors = []
    error_solvers = {}
    
    for solver_name, solution in other_solutions.items():
        if solution is None:
            continue
            
        solver_normalized = normalize_route(solution.vehicle_routes)
        solver_cost = solution.cost
        # Normalize routes to depot-free format for consistent cost calculation
        solver_depot_free = normalize_routes_to_depot_free(solution.vehicle_routes)
        calculated_cost = calculate_route_cost(solver_depot_free, distances)
        
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
        for solver_name in ['exact_milp', 'heuristic_or']:
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


def format_instance_matlab_cpu(instance: Dict[str, Any]) -> str:
    """Format instance in MATLAB-ready format (CPU version)"""
    coords = instance["coords"]
    demands = instance["demands"]
    
    # Extract x and y coordinates
    x_coords = [f"{coord[0]:.2f}" for coord in coords]
    y_coords = [f"{coord[1]:.2f}" for coord in coords]
    demands_list = [f"{int(demand)}" for demand in demands]
    
    # Format as MATLAB matrix
    x_row = " ".join(x_coords)
    y_row = " ".join(y_coords)
    d_row = " ".join(demands_list)
    
    return f"[{x_row};\n {y_row};\n {d_row}]"

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


def run_benchmark_for_n(n, instances_min, instances_max, capacity, demand_range, total_timeout, coord_range, logger, matlab_logger, csv_writer, disabled_solvers=None, debug=False):
    """Run benchmark for a specific problem size N with new timeout behavior.
    
    Updated to write individual instance results to CSV in format expected by plot_cpu_benchmark.py
    """
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
        'heuristic_or': {'times': [], 'costs': [], 'optimal_count': 0, 'solutions': []}
    }
    
    solvers = {
        'exact_milp': exact_milp,
        'exact_ortools_vrp': exact_ortools_vrp,
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
    
    # List to collect CSV rows for this N
    csv_rows = []
    
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

            # Log instance to MATLAB format
            matlab_logger.log_instance_start(n, instance, i+1)
            
            
            attempt_msg = f" (attempt {attempt + 1}/{MAX_INSTANCE_ATTEMPTS})" if attempt > 0 else ""
            logger.info(f"\n=== Instance {i+1}/{instances_max} (N={n}, seed={seed}){attempt_msg} ===")
            
            # Debug: print instance in MATLAB format (once per instance)
            if debug:
                matlab_format = format_instance_matlab_cpu(instance)
                print(f"\n📊 MATLAB Instance {i+1}/{instances_max}, N={n}, Seed={seed}:")
                print(matlab_format)
            
            # Test each solver ON THE SAME INSTANCE
            instance_solutions = {}
            any_solver_active = False
            instance_too_hard = False
            current_failures = {}
            
            # Generate a unique instance ID for this instance
            instance_id = f"n{n}_s{seed}"
            
            for solver_name, solver_module in solvers.items():
                if solver_name in disabled_solvers or solver_name in solver_disabled_this_n:
                    # Skip disabled solvers but record them as failed in CSV
                    csv_rows.append({
                        'n_customers': n,
                        'solver': solver_name,
                        'instance_id': instance_id,
                        'status': 'disabled',
                        'time': np.nan,
                        'cpc': np.nan
                    })
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
                
                # Log to MATLAB format
                if solution:
                    matlab_logger.log_solver_result(n, solver_name, solution)
                
                instance_solutions[solver_name] = solution
                
                # Debug output when --debug flag is set
                if debug and solution is not None:
                    cpc = solution.cost / max(1, instance['num_customers'])
                    routes_str = ', '.join([str(route) for route in solution.vehicle_routes])
                    route_formatted = format_route_with_depot(solution.vehicle_routes)
                    print(f"{solver_name} cost/route/cpc: {solution.cost:.4f} {route_formatted} CPC={cpc:.4f}")
                
                # Check if this instance took too long for ANY solver
                if solve_time >= instance_timeout_threshold:
                    current_failures[solver_name] = True
                    # Log single informative message about timeout/hard instance
                    if timed_out:
                        logger.warning(f"{solver_name}: Timed out after {solve_time:.2f}s (too hard, will try different instance)")
                        csv_rows.append({
                            'n_customers': n,
                            'solver': solver_name,
                            'instance_id': instance_id,
                            'status': 'timeout',
                            'time': solve_time,
                            'cpc': np.nan
                        })
                    else:
                        logger.warning(f"{solver_name}: Instance took {solve_time:.2f}s >= {instance_timeout_threshold:.2f}s (too hard, will try different instance)")
                        # Even though solver finished, mark as timeout if it took too long
                        csv_rows.append({
                            'n_customers': n,
                            'solver': solver_name,
                            'instance_id': instance_id,
                            'status': 'timeout',
                            'time': solve_time,
                            'cpc': np.nan
                        })
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
                    
                    # Write successful result to CSV
                    csv_rows.append({
                        'n_customers': n,
                        'solver': solver_name,
                        'instance_id': instance_id,
                        'status': 'success',
                        'time': solve_time,
                        'cpc': benchmark_cost
                    })
                    
                    # Only count as optimal if solver claims it's optimal AND it's an exact solver
                    if solution.is_optimal and solver_name.startswith('exact'):
                        results[solver_name]['optimal_count'] += 1
                elif solution is None and not current_failures.get(solver_name, False):
                    # Failed but not due to timeout
                    csv_rows.append({
                        'n_customers': n,
                        'solver': solver_name,
                        'instance_id': instance_id,
                        'status': 'failed',
                        'time': solve_time,
                        'cpc': np.nan
                    })
            
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
        # End MATLAB instance logging
        matlab_logger.end_instance()
            
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
        
        # Write CSV rows periodically (after each instance)
        if csv_writer:
            for row in csv_rows:
                csv_writer.writerow(row)
            csv_rows = []  # Clear the buffer
    
    # Complete progress bar
    print_progress_bar(instances_max if attempted == instances_max else attempted, instances_max)
    safe_print()  # New line after progress bar
    
    # The newly disabled solvers for next N (only from consecutive failures)
    newly_disabled = solver_disabled_this_n
    
    # Compute statistics (for logging/display purposes)
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
    
    for solver in ['exact_milp', 'exact_ortools_vrp', 'heuristic_or']:
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
    parser.add_argument('--output', type=str, default='results/csv/cpu_benchmark.csv', help='Output CSV file')
    parser.add_argument('--log', type=str, default='results/logs/benchmark_cpu.log', help='Log file (default: benchmark_exact.log)')
    parser.add_argument("--debug", action="store_true", help="Enable debug output showing CPC and routes for each solver")
    
    args = parser.parse_args()
    
    # Load configuration file
    print('📋 Loading configuration from config.json...')
    config = load_config()
    validate_config(config)
    instance_params = get_instance_params(config)
    
    # Use config values as defaults, allow command line overrides
    if hasattr(args, 'capacity') and args.capacity == 30:  # default value
        args.capacity = instance_params['capacity']
    if hasattr(args, 'demand_min') and args.demand_min == 1:  # default value  
        args.demand_min = instance_params['demand_range'][0]
    if hasattr(args, 'demand_max') and args.demand_max == 10:  # default value
        args.demand_max = instance_params['demand_range'][1] 
    if hasattr(args, 'coord_range') and args.coord_range == 100:  # default value
        args.coord_range = instance_params['coord_range']
        
    print(f'🔧 Using parameters: capacity={args.capacity}, demand=[{args.demand_min},{args.demand_max}], coord_range={args.coord_range}')
    
    # Set up logging
    logger = setup_logging(args.log)
    # Setup MATLAB logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    matlab_log_file = f"results/logs/cpu_benchmark_{timestamp}.log"
    matlab_logger = MatlabLogger(matlab_log_file)
    
    
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
    
    # Initialize CSV file with header format expected by plot_cpu_benchmark.py
    fieldnames = ['n_customers', 'solver', 'instance_id', 'status', 'time', 'cpc']
                  
    # Create parent directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Open CSV file for writing
    csv_file = open(args.output, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()
    
    rows_written = 0
    total_validation_errors = 0
    disabled_solvers = set()  # Track solvers that exceeded timeout
    
    try:
        # Run benchmark for each N
        for n in range(args.n_start, args.n_end + 1):
            result = run_benchmark_for_n(
                n, args.instances_min, args.instances_max, args.capacity, 
                demand_range, args.timeout, args.coord_range, logger, 
                matlab_logger, csv_writer, disabled_solvers, debug=args.debug
            )
            
            total_validation_errors += result.get('validation_errors', 0)
            
            # Update disabled solvers set
            newly_disabled = result.get('newly_disabled', set())
            disabled_solvers.update(newly_disabled)
            
            # Flush CSV file after each N
            csv_file.flush()
        
        safe_print(f"\n✅ Benchmark complete!")
        safe_print(f"📊 Results written to {args.output}")
        safe_print(f"📝 Detailed results logged to {args.log}")
        if total_validation_errors > 0:
            safe_print(f"⚠️  Total validation errors: {total_validation_errors}")
        else:
            safe_print(f"✅ All solutions validated successfully!")
    
    finally:
        # Close CSV file
        csv_file.close()
        # Close MATLAB logger
        matlab_logger.close()


if __name__ == '__main__':
    main()
