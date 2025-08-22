#!/usr/bin/env python3
import time
import numpy as np
import sys
import os
from contextlib import contextmanager
from typing import Dict, Any, List
from solvers.types import CVRPSolution
from solvers.utils import calculate_route_cost

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    pulp = None


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def solve(instance: Dict[str, Any], time_limit: float = 300.0, verbose: bool = False) -> CVRPSolution:
    """
    Exact CVRP solver using PuLP with proper MILP formulation.
    Uses the standard two-index vehicle flow formulation with MTZ subtour elimination.
    """
    if not HAS_PULP:
        raise ImportError("PuLP not available. Install with: pip install pulp")
    
    start_time = time.time()
    coords = instance['coords']
    demands = instance['demands']
    distances = instance['distances']
    capacity = instance['capacity']
    n = len(coords)  # Including depot
    n_customers = n - 1
    
    if verbose:
        print(f"PuLP MILP solving {n_customers} customers, capacity={capacity}")
    
    # Estimate number of vehicles needed
    total_demand = sum(demands[1:])
    min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    max_vehicles = min(n_customers, min_vehicles + 2)
    
    # Try different numbers of vehicles, starting from minimum
    for K in range(min_vehicles, max_vehicles + 1):
        if time.time() - start_time > time_limit:
            break
            
        try:
            solution = _solve_pulp_with_k_vehicles(
                distances, demands, capacity, n, K, 
                time_limit - (time.time() - start_time), verbose
            )
            if solution is not None:
                route, cost, vehicle_routes = solution
                total_time = time.time() - start_time
                
                # Use standardized cost calculation for consistency across all solvers
                standardized_cost = calculate_route_cost(vehicle_routes, distances)
                
                if verbose:
                    print(f"PuLP found optimal solution with {K} vehicles: cost={standardized_cost:.4f}, time={total_time:.3f}s")
                
                return CVRPSolution(
                    route=route,
                    cost=standardized_cost,
                    num_vehicles=K,
                    vehicle_routes=vehicle_routes,
                    solve_time=total_time,
                    algorithm_used='PuLP-MILP',
                    is_optimal=True
                )
        except Exception as e:
            if verbose:
                print(f"PuLP with {K} vehicles failed: {e}")
            continue
    
    raise TimeoutError(f"PuLP solver failed to find solution within {time_limit}s")


def _solve_pulp_with_k_vehicles(distances, demands, capacity, n, K, remaining_time, verbose):
    """Solve CVRP MILP with exactly K vehicles using PuLP."""
    
    if remaining_time <= 0:
        return None
    
    # Create problem
    prob = pulp.LpProblem("CVRP", pulp.LpMinimize)
    
    # Decision variables: x[i,j,k] = 1 if vehicle k goes from i to j
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:  # No self-loops
                for k in range(K):
                    x[i, j, k] = pulp.LpVariable(f'x_{i}_{j}_{k}', cat='Binary')
    
    # Load variables: u[i,k] = cumulative demand when vehicle k leaves node i
    u = {}
    for i in range(1, n):  # Exclude depot
        for k in range(K):
            u[i, k] = pulp.LpVariable(f'u_{i}_{k}', lowBound=0, upBound=capacity, cat='Continuous')
    
    # Objective: minimize total distance
    prob += pulp.lpSum([
        distances[i][j] * x[i, j, k]
        for i in range(n)
        for j in range(n)
        if i != j
        for k in range(K)
    ])
    
    # Constraints
    
    # 1. Each customer visited exactly once
    for j in range(1, n):  # For each customer
        prob += pulp.lpSum([
            x[i, j, k]
            for i in range(n)
            if i != j
            for k in range(K)
        ]) == 1
    
    # 2. Flow conservation for each vehicle at each node
    for k in range(K):
        for i in range(n):
            prob += pulp.lpSum([
                x[j, i, k] for j in range(n) if j != i
            ]) == pulp.lpSum([
                x[i, j, k] for j in range(n) if j != i
            ])
    
    # 3. Each vehicle starts and ends at depot
    for k in range(K):
        # At most one departure from depot
        prob += pulp.lpSum([
            x[0, j, k] for j in range(1, n)
        ]) <= 1
        
        # At most one arrival to depot
        prob += pulp.lpSum([
            x[i, 0, k] for i in range(1, n)
        ]) <= 1
    
    # 4. Miller-Tucker-Zemlin (MTZ) subtour elimination and capacity constraints
    for k in range(K):
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    # u[i] - u[j] + demands[j] * x[i,j,k] <= capacity * (1 - x[i,j,k])
                    # Rearranged: u[i] - u[j] + (demands[j] + capacity) * x[i,j,k] <= capacity
                    prob += u[i, k] - u[j, k] + (demands[j] + capacity) * x[i, j, k] <= capacity
    
    # 5. Load bounds: if customer i is visited by vehicle k, load must be at least demand[i]
    for i in range(1, n):
        for k in range(K):
            # u[i][k] >= demand[i] * sum(x[j,i,k] for j != i)
            prob += u[i, k] >= demands[i] * pulp.lpSum([
                x[j, i, k] for j in range(n) if j != i
            ])
    
    # Set time limit and quiet mode to suppress CBC output
    if hasattr(pulp, 'PULP_CBC_CMD'):
        # Multiple options to suppress CBC output
        solver = pulp.PULP_CBC_CMD(
            timeLimit=int(remaining_time), 
            msg=False,
            logPath=None,
            options=["-printingOptions", "normal"]
        )
    else:
        solver = pulp.getSolver(
            'PULP_CBC_CMD', 
            timeLimit=int(remaining_time), 
            msg=False,
            logPath=None
        )
    
    # Solve
    if verbose:
        print(f"  Solving PuLP MILP with {K} vehicles...")
    
    # Solve while suppressing stdout to prevent CBC verbose output
    with suppress_stdout():
        prob.solve(solver)
    
    if prob.status == pulp.LpStatusOptimal:
        # Extract solution
        if verbose:
            print(f"  PuLP optimal solution found: cost={pulp.value(prob.objective):.4f}")
        
        # Build routes
        vehicle_routes = []
        for k in range(K):
            route = [0]  # Start at depot
            current = 0
            visited = set([0])
            
            # Follow the path for this vehicle
            while True:
                next_node = None
                
                # Look for outgoing arc from current node for vehicle k
                for j in range(n):
                    if j != current and (current, j, k) in x:
                        if pulp.value(x[current, j, k]) > 0.5:
                            if j == 0:  # Return to depot
                                if current != 0:  # Only add depot if we've visited customers
                                    route.append(0)
                                next_node = None  # End this route
                                break
                            else:
                                next_node = j
                                break
                
                if next_node is None:
                    break  # End of route
                
                if next_node in visited:
                    if verbose:
                        print(f"    Warning: cycle detected in vehicle {k} route")
                    break
                    
                route.append(next_node)
                visited.add(next_node)
                current = next_node
                
                # Safety check
                if len(route) > n:
                    if verbose:
                        print(f"    Warning: route too long for vehicle {k}, breaking")
                    break
            
            # Only include routes that visit customers
            if len(route) > 1:  # More than just [0]
                if route[-1] != 0:  # Ensure route ends at depot
                    route.append(0)
                vehicle_routes.append(route)
        
        # Create full route
        full_route = [0]
        for vr in vehicle_routes:
            full_route.extend(vr[1:])  # Skip depot at start of each route
        
        return full_route, pulp.value(prob.objective), vehicle_routes
    
    else:
        if verbose:
            print(f"  PuLP solver status: {pulp.LpStatus[prob.status]}")
        return None
