#!/usr/bin/env python3
"""
OR-Tools Heuristic Benchmark Script (Batch Processing Version)
Tests OR-Tools Greedy (PATH_CHEAPEST_ARC) and GLS (GUIDED_LOCAL_SEARCH) methods
Processes instances in batches for efficiency.
"""

import numpy as np
import time
import sys
import os
import argparse
from typing import Dict, Any, List, Tuple
from tabulate import tabulate
from multiprocessing import Pool, cpu_count

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


def solve_single_instance(args):
    """Solve a single CVRP instance with both Greedy and GLS methods.
    Designed for parallel processing with multiprocessing.Pool."""
    instance, gls_timeout = args
    
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
    
    greedy_cost = float('inf')
    gls_cost = float('inf')
    
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
        
        # Solve with Greedy (PATH_CHEAPEST_ARC)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.UNSET
        search_parameters.time_limit.seconds = 2  # Short timeout for greedy
        search_parameters.log_search = False
        
        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            greedy_cost = min(greedy_cost, solution.ObjectiveValue() / scale)
        
        # Solve with GLS
        search_parameters_gls = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters_gls.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters_gls.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters_gls.time_limit.seconds = int(gls_timeout)
        search_parameters_gls.log_search = False
        
        solution_gls = routing.SolveWithParameters(search_parameters_gls)
        if solution_gls:
            gls_cost = min(gls_cost, solution_gls.ObjectiveValue() / scale)
    
    return {
        'greedy': greedy_cost if greedy_cost < float('inf') else -1.0,
        'gls': gls_cost if gls_cost < float('inf') else -1.0
    }


def solve_batch(instances: List[Dict], gls_timeout: float, n_workers: int = None) -> Tuple[List[float], List[float]]:
    """Solve a batch of CVRP instances in parallel using multiprocessing."""
    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # Use up to 8 cores
    
    # Prepare arguments for parallel processing
    args = [(instance, gls_timeout) for instance in instances]
    
    # Process in parallel
    with Pool(n_workers) as pool:
        results = pool.map(solve_single_instance, args)
    
    # Separate greedy and GLS results
    greedy_costs = [r['greedy'] for r in results if r['greedy'] > 0]
    gls_costs = [r['gls'] for r in results if r['gls'] > 0]
    
    return greedy_costs, gls_costs


def generate_instances_batch(n_customers: int, capacity: int, batch_size: int, 
                           start_idx: int = 0, coord_range: int = 100, 
                           demand_range: List[int] = [1, 10]) -> List[Dict]:
    """Generate a batch of CVRP instances."""
    instances = []
    for i in range(start_idx, start_idx + batch_size):
        seed = 42000 + n_customers * 1000 + i
        instance = _generate_instance(
            num_customers=n_customers,
            capacity=capacity,
            coord_range=coord_range,
            demand_range=demand_range,
            seed=seed
        )
        instances.append(instance)
    return instances


def run_benchmark_configuration(n_customers: int, capacity: int, num_instances: int,
                               gls_timeout: float, batch_size: int = 100,
                               coord_range: int = 100, demand_range: List[int] = [1, 10]) -> Tuple[Dict, Dict]:
    """Run benchmark for a specific configuration using batch processing."""
    print(f"\n{'='*70}")
    print(f"Configuration: N={n_customers}, Capacity={capacity}, Instances={num_instances}")
    print(f"GLS Timeout: {gls_timeout}s per instance")
    print(f"Batch size: {batch_size}, Using {min(cpu_count(), 8)} CPU cores")
    print(f"{'='*70}")
    
    all_greedy_costs = []
    all_gls_costs = []
    
    # Time tracking
    total_start_time = time.time()
    
    # Process in batches
    num_batches = (num_instances + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        current_batch_size = min(batch_size, num_instances - start_idx)
        
        print(f"\n  Processing batch {batch_idx + 1}/{num_batches}: instances {start_idx}-{start_idx + current_batch_size - 1}")
        
        # Generate batch
        batch_start = time.time()
        instances = generate_instances_batch(
            n_customers=n_customers,
            capacity=capacity,
            batch_size=current_batch_size,
            start_idx=start_idx,
            coord_range=coord_range,
            demand_range=demand_range
        )
        gen_time = time.time() - batch_start
        print(f"    Generation time: {gen_time:.2f}s")
        
        # Solve batch
        solve_start = time.time()
        greedy_costs, gls_costs = solve_batch(instances, gls_timeout)
        solve_time = time.time() - solve_start
        print(f"    Solving time: {solve_time:.2f}s")
        
        # Accumulate results
        all_greedy_costs.extend(greedy_costs)
        all_gls_costs.extend(gls_costs)
        
        # Progress stats
        if len(all_greedy_costs) > 0:
            print(f"    Greedy - Batch mean CPC: {np.mean(greedy_costs)/n_customers:.6f}, Overall: {np.mean(all_greedy_costs)/n_customers:.6f}")
        if len(all_gls_costs) > 0:
            print(f"    GLS    - Batch mean CPC: {np.mean(gls_costs)/n_customers:.6f}, Overall: {np.mean(all_gls_costs)/n_customers:.6f}")
    
    total_time = time.time() - total_start_time
    
    # Calculate statistics for Greedy
    greedy_results = {}
    if len(all_greedy_costs) > 0:
        greedy_cpcs = np.array(all_greedy_costs) / n_customers
        greedy_results = {
            'n_customers': n_customers,
            'capacity': capacity,
            'instances': len(all_greedy_costs),
            'mean_cpc': np.mean(greedy_cpcs),
            'std_cpc': np.std(greedy_cpcs),
            'sem': np.std(greedy_cpcs) / np.sqrt(len(greedy_cpcs)),
            'total_time': total_time,
            'time_per_instance': total_time / num_instances
        }
        greedy_results['2sem_mean_pct'] = (2 * greedy_results['sem'] / greedy_results['mean_cpc']) * 100
    
    # Calculate statistics for GLS
    gls_results = {}
    if len(all_gls_costs) > 0:
        gls_cpcs = np.array(all_gls_costs) / n_customers
        gls_results = {
            'n_customers': n_customers,
            'capacity': capacity,
            'instances': len(all_gls_costs),
            'mean_cpc': np.mean(gls_cpcs),
            'std_cpc': np.std(gls_cpcs),
            'sem': np.std(gls_cpcs) / np.sqrt(len(gls_cpcs)),
            'total_time': total_time,
            'time_per_instance': total_time / num_instances
        }
        gls_results['2sem_mean_pct'] = (2 * gls_results['sem'] / gls_results['mean_cpc']) * 100
    
    print(f"\n  Total time: {total_time:.2f}s ({total_time/num_instances:.3f}s per instance)")
    print(f"  Greedy: Solved {len(all_greedy_costs)}/{num_instances} instances")
    print(f"  GLS:    Solved {len(all_gls_costs)}/{num_instances} instances")
    
    return greedy_results, gls_results


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='OR-Tools Heuristic CVRP Benchmark (Batch Processing)')
    parser.add_argument('--instances', type=int, default=1000,
                        help='Number of instances to test per configuration (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for parallel processing (default: 100)')
    parser.add_argument('--gls-timeout', type=float, default=5.0,
                        help='Timeout in seconds for GLS solver per instance (default: 5.0)')
    parser.add_argument('--configs', type=str, default='all',
                        help='Which configurations to run: "all", "small" (N≤20), "large" (N>20), or comma-separated N values like "10,20" (default: all)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OR-Tools Heuristic CVRP Benchmark (Batch Processing)")
    print("Methods: Greedy (PATH_CHEAPEST_ARC) and GLS (GUIDED_LOCAL_SEARCH)")
    print(f"Instances per configuration: {args.instances}")
    print(f"Batch size: {args.batch_size}")
    print(f"GLS timeout per instance: {args.gls_timeout} seconds")
    print(f"Available CPU cores: {cpu_count()}, using up to 8")
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
            batch_size=args.batch_size,
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
        'batch_size': args.batch_size,
        'gls_timeout': args.gls_timeout
    }
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f'ortools_heuristics_batch_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
