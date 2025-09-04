#!/usr/bin/env python3
"""
OR-Tools Heuristic Benchmark Script
Tests OR-Tools Greedy (PATH_CHEAPEST_ARC) and GLS (GUIDED_LOCAL_SEARCH) methods
for various CVRP problem sizes and capacities.
"""

import numpy as np
import time
import sys
import os
import argparse
from typing import Dict, Any, List, Tuple
from tabulate import tabulate

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the canonical generator
from src.generator.generator import _generate_instance

# Import OR-Tools
try:
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2
except ImportError as e:
    print(f"Error: OR-Tools not installed. Please install with: pip install ortools")
    sys.exit(1)


def solve_cvrp_greedy(instance: Dict[str, Any], time_limit: float = 10.0) -> float:
    """
    Solve CVRP using OR-Tools with PATH_CHEAPEST_ARC (greedy) strategy.
    Returns the total cost.
    """
    coords = instance['coords']
    demands = instance['demands']
    distances = instance['distances']
    capacity = instance['capacity']
    n_customers = len(coords) - 1
    
    # Scale distances for integer representation
    scale = 10000
    scaled_distances = (distances * scale).astype(int)
    
    # Determine number of vehicles based on demand
    total_demand = int(sum(demands[1:]))
    min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    max_vehicles = min(n_customers, min_vehicles + 3)  # Try a few more vehicles
    
    best_cost = float('inf')
    
    for n_vehicles in range(min_vehicles, max_vehicles + 1):
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(len(coords), n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return scaled_distances[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Demand callback
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(demands[from_node])
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [capacity] * n_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Set search parameters for greedy solution
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        # No local search for pure greedy
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.UNSET
        )
        search_parameters.time_limit.seconds = int(time_limit)
        search_parameters.log_search = False
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            total_cost = solution.ObjectiveValue() / scale
            best_cost = min(best_cost, total_cost)
    
    return best_cost if best_cost < float('inf') else -1.0


def solve_cvrp_gls(instance: Dict[str, Any], time_limit: float = 10.0) -> float:
    """
    Solve CVRP using OR-Tools with GUIDED_LOCAL_SEARCH metaheuristic.
    Returns the total cost.
    """
    coords = instance['coords']
    demands = instance['demands']
    distances = instance['distances']
    capacity = instance['capacity']
    n_customers = len(coords) - 1
    
    # Scale distances for integer representation
    scale = 10000
    scaled_distances = (distances * scale).astype(int)
    
    # Determine number of vehicles based on demand
    total_demand = int(sum(demands[1:]))
    min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    max_vehicles = min(n_customers, min_vehicles + 3)
    
    best_cost = float('inf')
    
    for n_vehicles in range(min_vehicles, max_vehicles + 1):
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(len(coords), n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return scaled_distances[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Demand callback
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(demands[from_node])
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            [capacity] * n_vehicles,
            True,
            'Capacity'
        )
        
        # Set search parameters for GLS
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = int(time_limit)
        search_parameters.log_search = False
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            total_cost = solution.ObjectiveValue() / scale
            best_cost = min(best_cost, total_cost)
    
    return best_cost if best_cost < float('inf') else -1.0


def run_benchmark_configuration(n_customers: int, capacity: int, num_instances: int,
                               gls_timeout: float, coord_range: int = 100, 
                               demand_range: List[int] = [1, 10]) -> Tuple[Dict, Dict]:
    """
    Run benchmark for a specific configuration.
    Returns results for both Greedy and GLS methods.
    """
    print(f"\n{'='*70}")
    print(f"Configuration: N={n_customers}, Capacity={capacity}, Instances={num_instances}")
    print(f"GLS Timeout: {gls_timeout}s per instance")
    print(f"{'='*70}")
    
    greedy_costs = []
    gls_costs = []
    
    # Time tracking
    greedy_total_time = 0.0
    gls_total_time = 0.0
    
    # Progress tracking
    print_interval = max(1, num_instances // 10)
    
    for i in range(num_instances):
        # Generate instance with unique seed
        seed = 42000 + n_customers * 1000 + i
        instance = _generate_instance(
            num_customers=n_customers,
            capacity=capacity,
            coord_range=coord_range,
            demand_range=demand_range,
            seed=seed
        )
        
        # Solve with Greedy (shorter timeout since it's just the initial solution)
        start_time = time.time()
        greedy_cost = solve_cvrp_greedy(instance, time_limit=2.0)
        greedy_time = time.time() - start_time
        greedy_total_time += greedy_time
        
        if greedy_cost > 0:
            greedy_costs.append(greedy_cost)
        
        # Solve with GLS (use provided timeout)
        start_time = time.time()
        gls_cost = solve_cvrp_gls(instance, time_limit=gls_timeout)
        gls_time = time.time() - start_time
        gls_total_time += gls_time
        
        if gls_cost > 0:
            gls_costs.append(gls_cost)
        
        # Progress update
        if (i + 1) % print_interval == 0 or i == num_instances - 1:
            print(f"  Progress: {i+1}/{num_instances} instances completed")
            if len(greedy_costs) > 0:
                print(f"    Greedy - Current mean CPC: {np.mean(greedy_costs)/n_customers:.6f}")
            if len(gls_costs) > 0:
                print(f"    GLS    - Current mean CPC: {np.mean(gls_costs)/n_customers:.6f}")
    
    # Calculate statistics for Greedy
    greedy_results = {}
    if len(greedy_costs) > 0:
        greedy_cpcs = np.array(greedy_costs) / n_customers
        greedy_results = {
            'n_customers': n_customers,
            'capacity': capacity,
            'instances': len(greedy_costs),
            'mean_cpc': np.mean(greedy_cpcs),
            'std_cpc': np.std(greedy_cpcs),
            'sem': np.std(greedy_cpcs) / np.sqrt(len(greedy_cpcs)),
            'total_time': greedy_total_time,
            'time_per_instance': greedy_total_time / num_instances
        }
        greedy_results['2sem_mean_pct'] = (2 * greedy_results['sem'] / greedy_results['mean_cpc']) * 100
    
    # Calculate statistics for GLS
    gls_results = {}
    if len(gls_costs) > 0:
        gls_cpcs = np.array(gls_costs) / n_customers
        gls_results = {
            'n_customers': n_customers,
            'capacity': capacity,
            'instances': len(gls_costs),
            'mean_cpc': np.mean(gls_cpcs),
            'std_cpc': np.std(gls_cpcs),
            'sem': np.std(gls_cpcs) / np.sqrt(len(gls_cpcs)),
            'total_time': gls_total_time,
            'time_per_instance': gls_total_time / num_instances
        }
        gls_results['2sem_mean_pct'] = (2 * gls_results['sem'] / gls_results['mean_cpc']) * 100
    
    print(f"\n  Greedy: Solved {len(greedy_costs)}/{num_instances} instances in {greedy_total_time:.2f}s")
    print(f"  GLS:    Solved {len(gls_costs)}/{num_instances} instances in {gls_total_time:.2f}s")
    
    return greedy_results, gls_results


def main():
    """Main benchmark execution."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='OR-Tools Heuristic CVRP Benchmark')
    parser.add_argument('--instances', type=int, default=1000,
                        help='Number of instances to test per configuration (default: 1000)')
    parser.add_argument('--gls-timeout', type=float, default=5.0,
                        help='Timeout in seconds for GLS solver per instance (default: 5.0)')
    parser.add_argument('--configs', type=str, default='all',
                        help='Which configurations to run: "all", "small" (N≤20), "large" (N>20), or comma-separated N values like "10,20" (default: all)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OR-Tools Heuristic CVRP Benchmark")
    print("Methods: Greedy (PATH_CHEAPEST_ARC) and GLS (GUIDED_LOCAL_SEARCH)")
    print(f"Instances per configuration: {args.instances}")
    print(f"GLS timeout per instance: {args.gls_timeout} seconds")
    print("="*80)
    
    # Define all possible configurations: (n_customers, capacity)
    all_configs = [
        (10, 20),
        (20, 30),
        (50, 40),
        (100, 50)
    ]
    
    # Select configurations based on args
    if args.configs == 'all':
        configurations = all_configs
    elif args.configs == 'small':
        configurations = [(n, c) for n, c in all_configs if n <= 20]
    elif args.configs == 'large':
        configurations = [(n, c) for n, c in all_configs if n > 20]
    else:
        # Parse comma-separated N values
        try:
            n_values = [int(n.strip()) for n in args.configs.split(',')]
            configurations = [(n, c) for n, c in all_configs if n in n_values]
            if not configurations:
                print(f"Error: No valid configurations found for N values: {n_values}")
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid config specification: {args.configs}")
            print("Use 'all', 'small', 'large', or comma-separated N values like '10,20'")
            sys.exit(1)
    
    print(f"Running configurations: {configurations}")
    
    # Storage for results
    greedy_results_all = []
    gls_results_all = []
    
    # Run benchmarks
    total_start_time = time.time()
    
    for n_customers, capacity in configurations:
        greedy_res, gls_res = run_benchmark_configuration(
            n_customers=n_customers,
            capacity=capacity,
            num_instances=args.instances,
            gls_timeout=args.gls_timeout,
            coord_range=100,
            demand_range=[1, 10]
        )
        
        if greedy_res:
            greedy_results_all.append(greedy_res)
        if gls_res:
            gls_results_all.append(gls_res)
    
    total_time = time.time() - total_start_time
    
    # Print final results tables
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # Greedy Results Table
    if greedy_results_all:
        print("\nMethod: OR-Tools Greedy (PATH_CHEAPEST_ARC)")
        print("-" * 80)
        
        table_data = []
        for res in greedy_results_all:
            table_data.append([
                res['n_customers'],
                res['capacity'],
                f"{res['instances']:,}",
                f"{res['mean_cpc']:.6f}",
                f"{res['std_cpc']:.6f}",
                f"{res['sem']:.6f}",
                f"{res['2sem_mean_pct']:.4f}%"
            ])
        
        headers = ["N", "Capacity", "Instances", "Mean CPC", "Std CPC", "SEM", "2×SEM/Mean(%)"]
        print(tabulate(table_data, headers=headers, tablefmt="pipe", floatfmt=".6f"))
    
    # GLS Results Table
    if gls_results_all:
        print("\nMethod: OR-Tools GLS (GUIDED_LOCAL_SEARCH)")
        print("-" * 80)
        
        table_data = []
        for res in gls_results_all:
            table_data.append([
                res['n_customers'],
                res['capacity'],
                f"{res['instances']:,}",
                f"{res['mean_cpc']:.6f}",
                f"{res['std_cpc']:.6f}",
                f"{res['sem']:.6f}",
                f"{res['2sem_mean_pct']:.4f}%"
            ])
        
        headers = ["N", "Capacity", "Instances", "Mean CPC", "Std CPC", "SEM", "2×SEM/Mean(%)"]
        print(tabulate(table_data, headers=headers, tablefmt="pipe", floatfmt=".6f"))
    
    # Summary
    print("\n" + "="*80)
    print(f"Total benchmark time: {total_time:.2f} seconds")
    print(f"Average time per configuration: {total_time/len(configurations):.2f} seconds")
    print("="*80)
    
    # Save results to file
    import json
    results = {
        'greedy': greedy_results_all,
        'gls': gls_results_all,
        'total_time': total_time,
        'configurations': [(n, c) for n, c in configurations],
        'instances_per_config': args.instances,
        'gls_timeout': args.gls_timeout
    }
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f'ortools_heuristics_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
