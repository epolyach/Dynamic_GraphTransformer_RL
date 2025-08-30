#!/usr/bin/env python3
"""
Verify if solutions are truly optimal for small CVRP instances
"""

import numpy as np
from itertools import permutations, combinations
import sys

def generate_instance(n_customers, seed):
    """Generate test instance"""
    np.random.seed(seed)
    n = n_customers + 1
    coords = np.random.uniform(0, 1, size=(n, 2))
    coords[0] = [0.5, 0.5]
    demands = np.zeros(n, dtype=np.float32)
    demands[1:] = np.random.uniform(1, 10, size=n_customers)
    capacity = max(demands.sum() / 2, demands.max() * 2)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])
    return {'coords': coords, 'demands': demands, 'distances': distances,
            'capacity': capacity, 'n_customers': n_customers}

def calculate_route_cost(route, distances):
    """Calculate cost of a single route including depot"""
    cost = 0
    prev = 0  # Start from depot
    for node in route:
        cost += distances[prev][node]
        prev = node
    cost += distances[prev][0]  # Return to depot
    return cost

def is_feasible(partition, demands, capacity):
    """Check if a partition is feasible under capacity constraints"""
    for route in partition:
        route_demand = sum(demands[node] for node in route)
        if route_demand > capacity:
            return False
    return True

def generate_all_partitions(customers):
    """Generate all possible partitions of customers into groups"""
    if len(customers) == 0:
        yield []
    elif len(customers) == 1:
        yield [customers]
    else:
        first = customers[0]
        rest = customers[1:]
        for partition in generate_all_partitions(rest):
            # Add first to each group in partition
            for i in range(len(partition)):
                new_partition = []
                for j, group in enumerate(partition):
                    if j == i:
                        new_partition.append([first] + group)
                    else:
                        new_partition.append(group)
                yield new_partition
            # Or create a new group with first
            yield [[first]] + partition

def find_optimal_solution(instance):
    """Find truly optimal solution by exhaustive search"""
    n = instance['n_customers'] + 1
    customers = list(range(1, n))
    demands = instance['demands']
    capacity = instance['capacity']
    distances = instance['distances']
    
    best_cost = float('inf')
    best_solution = None
    solutions_checked = 0
    
    # Try all partitions
    for partition in generate_all_partitions(customers):
        if not is_feasible(partition, demands, capacity):
            continue
        
        # For each route in partition, try all orderings
        total_cost = 0
        best_routes = []
        
        for route in partition:
            if len(route) == 0:
                continue
            
            # Try all permutations of this route
            best_route_cost = float('inf')
            best_route_order = None
            
            for perm in permutations(route):
                route_cost = calculate_route_cost(perm, distances)
                if route_cost < best_route_cost:
                    best_route_cost = route_cost
                    best_route_order = list(perm)
            
            total_cost += best_route_cost
            best_routes.append(best_route_order)
        
        solutions_checked += 1
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_solution = best_routes
    
    return best_cost, best_solution, solutions_checked

def main():
    # Test N=4 instance
    print("Testing N=4 instance (seed 5000)")
    print("="*50)
    instance = generate_instance(4, 5000)
    
    print(f"Capacity: {instance['capacity']:.2f}")
    print(f"Demands: {instance['demands']}")
    
    optimal_cost, optimal_routes, solutions_checked = find_optimal_solution(instance)
    
    print(f"\nOptimal solution found after checking {solutions_checked} solutions:")
    print(f"  Cost: {optimal_cost:.6f}")
    print(f"  CPC: {optimal_cost/4:.6f}")
    print(f"  Routes: {optimal_routes}")
    
    # Compare with solver results
    print("\nComparison with solver results:")
    print("  CPU Exact DP: Cost=2.832457, CPC=0.708114")
    print("  GPU Exact DP: Cost=2.832420, CPC=0.708105")
    print(f"  Difference from optimal: {abs(2.832457 - optimal_cost):.6f}")
    
    # Test N=5 instance
    print("\n" + "="*50)
    print("Testing N=5 instance (seed 5000)")
    print("="*50)
    instance = generate_instance(5, 5000)
    
    print(f"Capacity: {instance['capacity']:.2f}")
    
    optimal_cost, optimal_routes, solutions_checked = find_optimal_solution(instance)
    
    print(f"\nOptimal solution found after checking {solutions_checked} solutions:")
    print(f"  Cost: {optimal_cost:.6f}")
    print(f"  CPC: {optimal_cost/5:.6f}")
    print(f"  Routes: {optimal_routes}")
    
    print("\nComparison with solver results:")
    print("  CPU Exact DP: Cost=2.138542, CPC=0.427708")
    print("  GPU Exact DP: Cost=2.138500, CPC=0.427700")
    print(f"  Difference from optimal: {abs(2.138542 - optimal_cost):.6f}")

if __name__ == "__main__":
    main()
