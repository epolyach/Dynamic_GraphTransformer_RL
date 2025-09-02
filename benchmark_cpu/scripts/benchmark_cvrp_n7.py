#!/usr/bin/env python3
"""
CVRP Benchmark Script for N=7 Customers
Evaluates optimal and heuristic solvers on 1000 instances

Instance Generation Parameters:
- Customers: 7 (excluding depot)
- Coordinates: Sampled on integer grid [0, 100] then normalized to [0, 1]
- Demands: Uniform integers in [1, 10]
- Vehicle Capacity: 30
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import solvers
from src.benchmarking.solvers.cpu import exact_dp
from src.benchmarking.solvers.cpu import exact_ortools_vrp_fixed
from src.benchmarking.solvers.cpu import ortools_gls


def generate_cvrp_instance(num_customers: int, capacity: int, seed: int) -> Dict[str, Any]:
    """Generate a single CVRP instance with specified parameters."""
    np.random.seed(seed)
    
    # Coordinates: (N+1, 2), depot at index 0
    # Sample on integer grid [0, 100] then normalize to [0, 1]
    coord_range = 100
    coords = np.zeros((num_customers + 1, 2), dtype=np.float64)
    for i in range(num_customers + 1):
        coords[i] = np.random.randint(0, coord_range + 1, size=2) / coord_range
    
    # Demands: uniform integers in [1, 10], depot has 0
    demands = np.zeros(num_customers + 1, dtype=np.int32)
    for i in range(1, num_customers + 1):
        demands[i] = np.random.randint(1, 11)  # [1, 10]
    
    # Calculate distances
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    return {
        'coords': coords,
        'demands': demands,
        'distances': distances,
        'capacity': capacity
    }


def solve_ortools_suboptimal(instance: Dict[str, Any], time_limit: float = 10.0, verbose: bool = False):
    """Modified OR-Tools solver for sub-optimal (heuristic) solutions."""
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
    except ImportError:
        raise ImportError("OR-Tools constraint solver not available. Install ortools.")
    
    from src.benchmarking.solvers.utils import calculate_route_cost
    from src.benchmarking.solvers.types import CVRPSolution
    
    start_time = time.time()
    coords = instance['coords']
    demands = instance['demands']
    distances = instance['distances']
    capacity = instance['capacity']
    n = len(coords)
    n_customers = n - 1
    
    if verbose:
        print(f"OR-Tools VRP (sub-optimal mode) solving {n_customers} customers, capacity={capacity}")
    
    # Scale distances to integers
    scale = 10000
    scaled_distances = (distances * scale).astype(int)
    scaled_demands = demands.astype(int)
    scaled_capacity = int(capacity)
    
    # Create routing model
    manager = pywrapcp.RoutingIndexManager(n, n_customers, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return scaled_distances[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return scaled_demands[from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0,
        [scaled_capacity] * n_customers,
        True, 'Capacity'
    )
    
    # Configure for HEURISTIC solving (sub-optimal)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    # Enable metaheuristics for faster, sub-optimal solutions
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    
    # Set shorter time limit for heuristic mode
    search_parameters.time_limit.seconds = int(min(time_limit, 5))
    
    if verbose:
        print("  Using HEURISTIC configuration (with metaheuristics)")
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        solve_time = time.time() - start_time
        
        # Extract routes
        vehicle_routes = []
        route = [0]
        
        for vehicle_id in range(n_customers):
            index = routing.Start(vehicle_id)
            vehicle_route = []
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    vehicle_route.append(node)
                index = solution.Value(routing.NextVar(index))
            
            if vehicle_route:
                vehicle_routes.append(vehicle_route)
                route.extend(vehicle_route)
        
        route.append(0)
        
        # Calculate cost
        standardized_cost = calculate_route_cost(vehicle_routes, distances)
        
        if verbose:
            print(f"OR-Tools VRP (sub-optimal) found solution: cost={standardized_cost:.4f}, "
                  f"vehicles={len(vehicle_routes)}, time={solve_time:.3f}s")
        
        return CVRPSolution(
            route=route,
            cost=standardized_cost,
            num_vehicles=len(vehicle_routes),
            vehicle_routes=vehicle_routes,
            solve_time=solve_time,
            algorithm_used='OR-Tools-VRP-Suboptimal',
            is_optimal=False
        )
    else:
        raise RuntimeError(f"OR-Tools VRP (sub-optimal) failed to find solution within {time_limit}s")


def run_benchmark(n_instances: int = 1000, n_customers: int = 7, capacity: int = 30, verbose: bool = True):
    """Run benchmark on multiple CVRP instances."""
    
    print("=" * 80)
    print("CVRP Benchmark Configuration")
    print("=" * 80)
    print(f"Number of instances: {n_instances}")
    print(f"Number of customers (N): {n_customers}")
    print(f"Vehicle capacity: {capacity}")
    print(f"Demand distribution: Uniform integers [1, 10]")
    print(f"Coordinate distribution: Integer grid [0, 100] normalized to [0, 1]")
    print(f"Depot location: Randomly placed on same grid as customers")
    print("=" * 80)
    
    # Generate instances
    print("\nGenerating instances...")
    instances = []
    for i in range(n_instances):
        instances.append(generate_cvrp_instance(n_customers, capacity, seed=42 + i))
    print(f"Generated {n_instances} instances")
    
    # Define solvers to test
    solvers = [
        ("Exact DP", exact_dp.solve, {"time_limit": 60.0, "verbose": False}),
        ("OR-Tools Exact", exact_ortools_vrp_fixed.solve, {"time_limit": 60.0, "verbose": False}),
        ("OR-Tools Sub-optimal", solve_ortools_suboptimal, {"time_limit": 10.0, "verbose": False}),
        ("OR-Tools GLS", ortools_gls.solve, {"time_limit": 10.0, "verbose": False})
    ]
    
    # Results storage
    results = {name: {"costs": [], "times": [], "failures": 0} for name, _, _ in solvers}
    
    # Run benchmark
    print("\nRunning benchmark...")
    for idx, instance in enumerate(instances):
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Processing instance {idx + 1}/{n_instances}...")
        
        for solver_name, solver_func, solver_params in solvers:
            try:
                solution = solver_func(instance, **solver_params)
                results[solver_name]["costs"].append(solution.cost)
                results[solver_name]["times"].append(solution.solve_time)
            except Exception as e:
                results[solver_name]["failures"] += 1
                if verbose and results[solver_name]["failures"] <= 5:
                    print(f"    Warning: {solver_name} failed on instance {idx + 1}: {str(e)[:50]}")
    
    print("\nBenchmark completed!")
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Prepare table data
    table_data = []
    headers = ["Solver", "Avg Cost", "CPC", "Std CPC", "Avg Time (s)", "Failed"]
    
    for solver_name, _, _ in solvers:
        costs = results[solver_name]["costs"]
        times = results[solver_name]["times"]
        failures = results[solver_name]["failures"]
        
        if costs:
            avg_cost = np.mean(costs)
            # Cost per customer
            cpc_values = [cost / n_customers for cost in costs]
            avg_cpc = np.mean(cpc_values)
            std_cpc = np.std(cpc_values)
            avg_time = np.mean(times)
            
            table_data.append([
                solver_name,
                f"{avg_cost:.4f}",
                f"{avg_cpc:.4f}",
                f"{std_cpc:.4f}",
                f"{avg_time:.3f}",
                f"{failures}/{n_instances}"
            ])
        else:
            table_data.append([
                solver_name,
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                f"{failures}/{n_instances}"
            ])
    
    # Print table
    print("\nPerformance Summary:")
    print("-" * 80)
    
    # Format as table
    col_widths = [max(len(str(row[i])) for row in [headers] + table_data) for i in range(len(headers))]
    
    # Print headers
    header_line = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))
    
    # Print data
    for row in table_data:
        row_line = " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
        print(row_line)
    
    print("-" * 80)
    print("\nLegend:")
    print("  - CPC: Cost Per Customer (average route cost divided by number of customers)")
    print("  - Std CPC: Standard deviation of Cost Per Customer across instances")
    print("  - Failed: Number of instances where solver failed or timed out")
    
    # Save results to CSV
    csv_filename = f"benchmark_results_n{n_customers}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w') as f:
        f.write(",".join(headers) + "\n")
        for row in table_data:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nResults saved to: {csv_filename}")
    
    return results


if __name__ == "__main__":
    # Run the benchmark
    results = run_benchmark(n_instances=1000, n_customers=7, capacity=30, verbose=True)
