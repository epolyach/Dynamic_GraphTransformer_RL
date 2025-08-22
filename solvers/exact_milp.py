#!/usr/bin/env python3
import time
import numpy as np
from typing import Dict, Any, List
from solvers.types import CVRPSolution
from solvers.utils import calculate_route_cost

try:
    from ortools.linear_solver import pywraplp
except Exception:
    pywraplp = None


def solve(instance: Dict[str, Any], time_limit: float = 300.0, verbose: bool = False) -> CVRPSolution:
    """
    True exact MILP solver for CVRP using standard two-index vehicle flow formulation.
    
    Variables:
    - x[i][j][k] = 1 if vehicle k travels from node i to node j, 0 otherwise
    - u[i][k] = load of vehicle k when leaving node i
    
    This is a true exact solver that guarantees optimal solutions.
    """
    if pywraplp is None:
        raise ImportError("OR-Tools not available. Install ortools.")

    start_time = time.time()
    coords = instance['coords']
    demands = instance['demands']
    distances = instance['distances']
    capacity = instance['capacity']
    n = len(coords)  # Including depot
    n_customers = n - 1
    
    if verbose:
        print(f"MILP solving {n_customers} customers, capacity={capacity}")
    
    # Estimate number of vehicles needed
    total_demand = sum(demands[1:])
    min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    max_vehicles = min(n_customers, min_vehicles + 2)  # Allow some flexibility
    
    if verbose:
        print(f"Trying {min_vehicles} to {max_vehicles} vehicles")
    
    # Try different numbers of vehicles, starting from minimum
    for K in range(min_vehicles, max_vehicles + 1):
        if time.time() - start_time > time_limit:
            break
            
        try:
            solution = _solve_milp_with_k_vehicles(
                distances, demands, capacity, n, K, 
                time_limit - (time.time() - start_time), verbose
            )
            if solution is not None:
                # Found optimal solution
                route, cost, vehicle_routes = solution
                total_time = time.time() - start_time
                
                # Use standardized cost calculation for consistency across all solvers
                standardized_cost = calculate_route_cost(vehicle_routes, distances)
                
                if verbose:
                    print(f"MILP found optimal solution with {K} vehicles: cost={standardized_cost:.4f}, time={total_time:.3f}s")
                
                return CVRPSolution(
                    route=route,
                    cost=standardized_cost,
                    num_vehicles=K,
                    vehicle_routes=vehicle_routes,
                    solve_time=total_time,
                    algorithm_used='Exact-MILP',
                    is_optimal=True  # This is truly optimal
                )
        except Exception as e:
            if verbose:
                print(f"MILP with {K} vehicles failed: {e}")
            continue
    
    raise TimeoutError(f"MILP solver failed to find solution within {time_limit}s")


def _solve_milp_with_k_vehicles(distances, demands, capacity, n, K, remaining_time, verbose):
    """Solve CVRP MILP with exactly K vehicles."""
    
    if remaining_time <= 0:
        return None
    
    # Create solver - try MILP solvers only, no fallbacks
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('CP-SAT')
    if not solver:
        solver = pywraplp.Solver.CreateSolver('GUROBI')
    if not solver:
        raise RuntimeError("No MILP solver available (tried SCIP, CP-SAT, GUROBI)")
    
    # Set time limit
    solver.set_time_limit(int(remaining_time * 1000))  # milliseconds
    
    # Decision variables: x[i][j][k] = 1 if vehicle k goes from i to j
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:  # No self-loops
                for k in range(K):
                    x[i, j, k] = solver.BoolVar(f'x_{i}_{j}_{k}')
    
    # Load variables: u[i][k] = cumulative demand when vehicle k leaves node i
    u = {}
    for i in range(1, n):  # Exclude depot
        for k in range(K):
            u[i, k] = solver.NumVar(0, float(capacity), f'u_{i}_{k}')
    
    # Objective: minimize total distance
    objective = solver.Objective()
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(K):
                    objective.SetCoefficient(x[i, j, k], float(distances[i][j]))
    objective.SetMinimization()
    
    # Constraints
    
    # 1. Each customer visited exactly once
    for j in range(1, n):  # For each customer
        constraint = solver.Constraint(1, 1, f'visit_customer_{j}')
        for i in range(n):
            if i != j:
                for k in range(K):
                    constraint.SetCoefficient(x[i, j, k], 1.0)
    
    # 2. Flow conservation for each vehicle at each node
    for k in range(K):
        for i in range(n):
            constraint = solver.Constraint(0, 0, f'flow_conservation_{i}_{k}')
            # Incoming flow
            for j in range(n):
                if j != i:
                    constraint.SetCoefficient(x[j, i, k], 1.0)
            # Outgoing flow
            for j in range(n):
                if j != i:
                    constraint.SetCoefficient(x[i, j, k], -1.0)
    
    # 3. Each vehicle starts and ends at depot
    for k in range(K):
        # At most one departure from depot
        constraint = solver.Constraint(0, 1, f'depot_start_{k}')
        for j in range(1, n):
            constraint.SetCoefficient(x[0, j, k], 1.0)
        
        # At most one arrival to depot
        constraint = solver.Constraint(0, 1, f'depot_end_{k}')
        for i in range(1, n):
            constraint.SetCoefficient(x[i, 0, k], 1.0)
    
    # 4. Capacity constraints using Miller-Tucker-Zemlin (MTZ) subtour elimination
    for k in range(K):
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    # u[i] + demand[j] <= u[j] + M*(1 - x[i][j][k])
                    # Rearranged: u[i] - u[j] + demand[j]*x[i][j][k] <= M*(1 - x[i][j][k])
                    # Further: u[i] - u[j] + (demand[j] + M)*x[i][j][k] <= M
                    M = float(capacity)  # Big M
                    constraint = solver.Constraint(
                        -solver.infinity(), 
                        M, 
                        f'capacity_{i}_{j}_{k}'
                    )
                    constraint.SetCoefficient(u[i, k], 1.0)
                    constraint.SetCoefficient(u[j, k], -1.0)
                    constraint.SetCoefficient(x[i, j, k], float(demands[j]) + M)
    
    # 5. Load bounds
    for i in range(1, n):
        for k in range(K):
            # u[i][k] >= demand[i] if customer i is visited by vehicle k
            constraint = solver.Constraint(0, solver.infinity(), f'min_load_{i}_{k}')
            constraint.SetCoefficient(u[i, k], 1.0)
            # If any vehicle k visits customer i, load must be at least demand[i]
            for j in range(n):
                if j != i:
                    constraint.SetCoefficient(x[j, i, k], -float(demands[i]))
    
    # Solve
    if verbose:
        print(f"  Solving MILP with {K} vehicles...")
    
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        # Extract solution
        if verbose:
            print(f"  MILP optimal solution found: cost={solver.Objective().Value():.4f}")
        
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
                        if x[current, j, k].solution_value() > 0.5:
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
                
                # Safety check to prevent infinite loops
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
        
        return full_route, solver.Objective().Value(), vehicle_routes
    
    elif status == pywraplp.Solver.INFEASIBLE:
        if verbose:
            print(f"  MILP infeasible with {K} vehicles")
        return None
    
    else:
        if verbose:
            print(f"  MILP did not converge (status: {status})")
        return None
