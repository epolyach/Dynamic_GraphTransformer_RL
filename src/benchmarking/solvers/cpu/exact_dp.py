#!/usr/bin/env python3
import time
import numpy as np
from typing import Dict, Any, List, Tuple
from itertools import combinations
from src.benchmarking.solvers.types import CVRPSolution
from src.benchmarking.solvers.utils import calculate_route_cost


def solve(instance: Dict[str, Any], time_limit: float = 300.0, verbose: bool = False) -> CVRPSolution:
    """
    Exact DP solver using brute force enumeration for small instances.
    Exponential complexity O(n!) - practical only for very small n.
    NO FALLBACKS - either finds exact solution or raises exception.
    """
    start_time = time.time()
    
    coords = instance['coords']
    demands = instance['demands']
    distances = instance['distances']
    capacity = instance['capacity']
    n = len(coords)  # Including depot
    n_customers = n - 1
    
    if verbose:
        print(f"Exact DP solving {n_customers} customers, capacity={capacity}")
    
    # Check if instance is feasible
    if max(demands[1:]) > capacity:
        raise ValueError("Instance infeasible: customer demand exceeds vehicle capacity")
    
    # Use brute force exact enumeration - no fallbacks
    return _solve_brute_force_exact(instance, time_limit, verbose, start_time)








def _solve_brute_force_exact(instance: Dict[str, Any], time_limit: float, verbose: bool, start_time: float) -> CVRPSolution:
    """
    TRUE brute force exact solver for very small instances.
    Tries all possible partitions of customers into vehicles and all orderings within each partition.
    """
    from itertools import permutations
    
    coords = instance['coords']
    demands = instance['demands']
    distances = instance['distances']
    capacity = instance['capacity']
    n = len(coords)
    n_customers = n - 1
    
    customers = list(range(1, n))
    best_cost = float('inf')
    best_solution = None
    
    solutions_tested = 0
    
    # Generate all possible vehicle route partitions
    for vehicle_partition in _generate_all_partitions(customers, capacity, demands):
        
        # For each partition, try all permutations within each vehicle route
        partition_solutions = []
        for vehicle_customers in vehicle_partition:
            if len(vehicle_customers) == 0:
                continue
            elif len(vehicle_customers) == 1:
                partition_solutions.append([vehicle_customers])
            else:
                # Try all permutations for this vehicle
                vehicle_perms = list(permutations(vehicle_customers))
                partition_solutions.append(vehicle_perms)
        
        # Generate all combinations of permutations across vehicles
        if partition_solutions:
            for vehicle_routes_combo in _cartesian_product(partition_solutions):
                solutions_tested += 1
                
                # Build complete vehicle routes with depot
                vehicle_routes = []
                for vehicle_customers in vehicle_routes_combo:
                    if vehicle_customers:  # Skip empty vehicles
                        route = [0] + list(vehicle_customers) + [0]
                        vehicle_routes.append(route)
                
                # Calculate total cost
                total_cost = 0.0
                for route in vehicle_routes:
                    for i in range(len(route) - 1):
                        total_cost += distances[route[i]][route[i + 1]]
                
                # Update best solution
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_solution = vehicle_routes[:]
                    
                if verbose and solutions_tested % 10000 == 0:
                    print(f"  Tested {solutions_tested} solutions, best cost: {best_cost:.4f}")
    
    if best_solution is None:
        raise RuntimeError("Brute force failed to find any solution")
    
    # Convert to single route format
    route = [0]
    for vr in best_solution:
        route.extend(vr[1:])  # Skip depot at start of each route
    
    solve_time = time.time() - start_time
    
    # Use standardized cost calculation for consistency across all solvers
    # Remove depot from routes before passing to calculate_route_cost
    clean_routes = []
    for vr in best_solution:
        clean_route = [node for node in vr if node != 0]
        if clean_route:
            clean_routes.append(clean_route)
    standardized_cost = calculate_route_cost(clean_routes, distances)
    
    if verbose:
        print(f"Brute force completed: tested {solutions_tested} solutions in {solve_time:.3f}s")
        print(f"Optimal cost: {standardized_cost:.4f} with {len(best_solution)} vehicles")
    
    return CVRPSolution(
        route=route,
        cost=standardized_cost,
        num_vehicles=len(clean_routes),
        vehicle_routes=clean_routes,  # Return clean routes without depot
        solve_time=solve_time,
        algorithm_used='Exact-BruteForce',
        is_optimal=True  # This is now truly optimal
    )


def _generate_all_partitions(customers: List[int], capacity: int, demands: np.ndarray):
    """
    Generate all possible partitions of customers into feasible vehicle loads.
    """
    from itertools import combinations
    
    n = len(customers)
    
    # Try all possible numbers of vehicles (1 to n)
    for num_vehicles in range(1, n + 1):
        # Generate all ways to partition customers into num_vehicles groups
        for partition in _partition_customers(customers, num_vehicles, capacity, demands):
            yield partition


def _partition_customers(customers: List[int], num_vehicles: int, capacity: int, demands: np.ndarray):
    """
    Generate all ways to partition customers into exactly num_vehicles feasible groups.
    """
    if num_vehicles == 1:
        # All customers in one vehicle - check capacity
        total_demand = sum(demands[c] for c in customers)
        if total_demand <= capacity:
            yield [customers]
        return
    
    if not customers or num_vehicles <= 0:
        return
    
    # Try all possible first vehicle compositions
    for vehicle_size in range(1, len(customers) + 1):
        for first_vehicle in combinations(customers, vehicle_size):
            # Check capacity constraint for first vehicle
            first_demand = sum(demands[c] for c in first_vehicle)
            if first_demand <= capacity:
                # Recursively partition remaining customers
                remaining = [c for c in customers if c not in first_vehicle]
                for rest_partition in _partition_customers(remaining, num_vehicles - 1, capacity, demands):
                    yield [list(first_vehicle)] + rest_partition


def _cartesian_product(lists_of_lists):
    """
    Generate cartesian product of lists of lists.
    """
    if not lists_of_lists:
        yield []
        return
    
    for item in lists_of_lists[0]:
        for rest in _cartesian_product(lists_of_lists[1:]):
            yield [item] + rest


