#!/usr/bin/env python3
"""
Benchmark OR-Tools GLS for multiple N values (10, 20, 50) on CPU
"""

import numpy as np
import time
import json
import argparse
from datetime import datetime
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def generate_cvrp_instance(n_customers=20, seed=None):
    """Generate a random CVRP instance"""
    if seed is not None:
        np.random.seed(seed)
    
    # Depot at center, customers uniformly distributed
    depot = np.array([[0.5, 0.5]])
    customers = np.random.rand(n_customers, 2)
    locations = np.vstack([depot, customers])
    
    # Uniform demands between 1 and 10
    demands = np.random.randint(1, 11, size=n_customers + 1)
    demands[0] = 0  # depot has no demand
    
    # Vehicle capacity
    capacity = max(10, int(np.sum(demands) / 3))
    
    return locations, demands, capacity

def compute_distance_matrix(locations):
    """Compute Euclidean distance matrix"""
    n = len(locations)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])
    # Scale and round
    return (dist_matrix * 10000).astype(int)

def solve_with_ortools_gls(locations, demands, capacity, timeout_sec=1):
    """Solve using OR-Tools with Guided Local Search"""
    n = len(locations)
    dist_matrix = compute_distance_matrix(locations)
    
    # Create routing model
    manager = pywrapcp.RoutingIndexManager(n, 10, 0)  # 10 vehicles max
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(demands[from_node])
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [int(capacity)] * 10,  # vehicle capacities
        True,  # start cumul to zero
        'Capacity'
    )
    
    # Search parameters with GLS
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = int(timeout_sec)  # Convert to int
    search_parameters.log_search = False
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        total_distance = solution.ObjectiveValue() / 10000.0
        return total_distance
    return None

def run_benchmark(n_customers, n_instances, timeout_sec, log_file):
    """Run benchmark for specific N"""
    print(f"\nStarting OR-Tools GLS benchmark for N={n_customers}, {n_instances} instances")
    print(f"Timeout: {timeout_sec}s per instance")
    print(f"Log file: {log_file}")
    
    all_cpcs = []
    failed = 0
    
    start_time = time.time()
    
    with open(log_file, 'w') as f:
        f.write(f"OR-Tools GLS Benchmark N={n_customers}\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Instances: {n_instances}, Timeout: {timeout_sec}s\n")
        f.write("="*60 + "\n\n")
    
    for i in range(n_instances):
        # Generate instance
        locations, demands, capacity = generate_cvrp_instance(n_customers, seed=i)
        
        # Solve with OR-Tools
        ortools_cost = solve_with_ortools_gls(locations, demands, capacity, timeout_sec)
        
        if ortools_cost is None:
            failed += 1
            with open(log_file, 'a') as f:
                f.write(f"Instance {i+1}: FAILED\n")
            continue
        
        # Generate initial solution using nearest neighbor for baseline
        n = len(locations)
        dist_matrix = compute_distance_matrix(locations) / 10000.0
        
        unvisited = set(range(1, n))
        routes = []
        
        while unvisited:
            route = [0]
            current_load = 0
            current = 0
            
            while unvisited:
                nearest = None
                nearest_dist = float('inf')
                
                for next_node in unvisited:
                    if current_load + demands[next_node] <= capacity:
                        if dist_matrix[current][next_node] < nearest_dist:
                            nearest = next_node
                            nearest_dist = dist_matrix[current][next_node]
                
                if nearest is None:
                    break
                
                route.append(nearest)
                current = nearest
                current_load += demands[nearest]
                unvisited.remove(nearest)
            
            route.append(0)
            routes.append(route)
        
        # Compute baseline cost
        baseline_cost = sum(
            sum(dist_matrix[routes[r][i]][routes[r][i+1]] 
                for i in range(len(routes[r])-1))
            for r in range(len(routes))
        )
        
        # CPC ratio
        cpc = ortools_cost / baseline_cost if baseline_cost > 0 else 1.0
        all_cpcs.append(cpc)
        
        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * n_instances - elapsed
            current_gm = np.exp(np.mean(np.log(all_cpcs)))
            
            progress_msg = (f"Progress: {i+1}/{n_instances} "
                          f"({100*(i+1)/n_instances:.1f}%) "
                          f"- Current GM: {current_gm:.6f} "
                          f"- ETA: {eta/60:.1f} min")
            
            print(progress_msg)
            with open(log_file, 'a') as f:
                f.write(progress_msg + "\n")
    
    # Final statistics
    total_time = time.time() - start_time
    cpcs_array = np.array(all_cpcs)
    log_cpcs = np.log(cpcs_array)
    
    gm = np.exp(np.mean(log_cpcs))
    gsd = np.exp(np.std(log_cpcs, ddof=1))
    
    results = {
        'n_customers': n_customers,
        'instances': n_instances,
        'timeout_sec': timeout_sec,
        'failed': failed,
        'total_time_sec': total_time,
        'gm': float(gm),
        'gsd': float(gsd),
        'all_cpcs': [float(x) for x in all_cpcs]
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'ortools_gls_n{n_customers}_{n_instances}inst_{timestamp}.json'
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = f"""
    ========================================
    BENCHMARK COMPLETED: N={n_customers}
    ========================================
    Total instances:    {n_instances}
    Failed instances:   {failed}
    Total time:         {total_time/60:.1f} minutes
    
    Geometric Mean:     {gm:.6f}
    Geometric Std Dev:  {gsd:.6f}
    95% Range:          [{gm*(gsd**(-1.96)):.4f}, {gm*(gsd**(1.96)):.4f}]
    
    Results saved to:   {result_file}
    ========================================
    """
    
    print(summary)
    with open(log_file, 'a') as f:
        f.write(summary)
    
    return result_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True, help='Number of customers')
    parser.add_argument('--instances', type=int, default=10000)
    parser.add_argument('--timeout', type=int, default=2, help='Timeout in seconds (integer)')
    args = parser.parse_args()
    
    log_file = f'ortools_n{args.n}_{args.instances}inst.log'
    run_benchmark(args.n, args.instances, args.timeout, log_file)
