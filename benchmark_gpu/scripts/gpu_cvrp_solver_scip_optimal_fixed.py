#!/usr/bin/env python3
"""
SCIP-based CVRP Solver for guaranteed optimal solutions
Requires: pip install pyscipopt
"""

import numpy as np
from typing import Dict, Any, List, Optional
import time
import sys
from dataclasses import dataclass

sys.path.append('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
from src.benchmarking.solvers.types import CVRPSolution

try:
    from pyscipopt import Model, quicksum
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False
    print("WARNING: PySCIPOpt not installed. Install with: pip install pyscipopt")

@dataclass
class SCIPSolution:
    """Intermediate solution format"""
    routes: List[List[int]]
    cost: float
    is_optimal: bool
    solve_time: float

class SCIPCVRPSolver:
    """
    SCIP-based exact CVRP solver using Miller-Tucker-Zemlin (MTZ) formulation
    """
    
    def __init__(self, time_limit: float = 3600, gap: float = 0.0):
        """
        Initialize SCIP solver
        
        Args:
            time_limit: Maximum solve time in seconds
            gap: Optimality gap (0.0 for exact solutions)
        """
        if not SCIP_AVAILABLE:
            raise ImportError("PySCIPOpt is required. Install with: pip install pyscipopt")
        
        self.time_limit = time_limit
        self.gap = gap
    
    def solve(self, instance: Dict[str, Any], verbose: bool = False) -> SCIPSolution:
        """
        Solve a single CVRP instance using SCIP
        Returns SCIPSolution instead of CVRPSolution for compatibility
        """
        start_time = time.time()
        
        # Extract instance data
        distances = instance['distances']
        demands = instance['demands']
        capacity = instance['capacity']
        n = len(demands)
        n_customers = n - 1
        
        # Maximum number of vehicles (worst case: one per customer)
        max_vehicles = n_customers
        
        # Create SCIP model
        model = Model("CVRP")
        
        if not verbose:
            model.hideOutput()
        
        # Decision variables
        # x[i,j,k] = 1 if vehicle k travels from i to j
        x = {}
        for k in range(max_vehicles):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x[i,j,k] = model.addVar(vtype="B", name=f"x_{i}_{j}_{k}")
        
        # Vehicle usage variables
        y = {}
        for k in range(max_vehicles):
            y[k] = model.addVar(vtype="B", name=f"y_{k}")
        
        # MTZ subtour elimination variables
        u = {}
        for i in range(1, n):
            for k in range(max_vehicles):
                u[i,k] = model.addVar(lb=demands[i], ub=capacity, name=f"u_{i}_{k}")
        
        # Objective: minimize total distance
        obj = quicksum(distances[i][j] * x[i,j,k] 
                      for k in range(max_vehicles)
                      for i in range(n)
                      for j in range(n)
                      if i != j)
        model.setObjective(obj, "minimize")
        
        # Constraints
        
        # 1. Each customer must be visited exactly once
        for i in range(1, n):
            model.addCons(
                quicksum(x[i,j,k] for k in range(max_vehicles) 
                        for j in range(n) if i != j) == 1,
                name=f"visit_{i}"
            )
        
        # 2. Flow conservation
        for k in range(max_vehicles):
            for i in range(n):
                model.addCons(
                    quicksum(x[i,j,k] for j in range(n) if i != j) ==
                    quicksum(x[j,i,k] for j in range(n) if i != j),
                    name=f"flow_{i}_{k}"
                )
        
        # 3. Vehicle leaves depot if used
        for k in range(max_vehicles):
            model.addCons(
                quicksum(x[0,j,k] for j in range(1, n)) == y[k],
                name=f"depot_out_{k}"
            )
        
        # 4. Vehicle returns to depot if used
        for k in range(max_vehicles):
            model.addCons(
                quicksum(x[j,0,k] for j in range(1, n)) == y[k],
                name=f"depot_in_{k}"
            )
        
        # 5. Capacity constraints with MTZ formulation
        for k in range(max_vehicles):
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        model.addCons(
                            u[i,k] + demands[j] - capacity * (1 - x[i,j,k]) <= u[j,k],
                            name=f"mtz_{i}_{j}_{k}"
                        )
        
        # 6. Link capacity variables to routes
        for k in range(max_vehicles):
            for i in range(1, n):
                model.addCons(
                    demands[i] <= u[i,k],
                    name=f"cap_lb_{i}_{k}"
                )
                model.addCons(
                    u[i,k] <= capacity * quicksum(x[j,i,k] for j in range(n) if j != i),
                    name=f"cap_ub_{i}_{k}"
                )
        
        # 7. Symmetry breaking: order vehicles by usage
        for k in range(max_vehicles - 1):
            model.addCons(y[k] >= y[k+1], name=f"sym_{k}")
        
        # Set solver parameters
        model.setParam('limits/time', self.time_limit)
        model.setParam('limits/gap', self.gap)
        
        # Solve
        model.optimize()
        
        solve_time = time.time() - start_time
        
        # Extract solution
        if model.getStatus() == "optimal" or model.getStatus() == "bestsollimit":
            routes = self._extract_routes(x, n, max_vehicles)
            cost = model.getObjVal()
            is_optimal = (model.getStatus() == "optimal")
        else:
            # No solution found
            routes = []
            cost = float('inf')
            is_optimal = False
        
        if verbose:
            print(f"SCIP solve time: {solve_time:.2f}s")
            print(f"Status: {model.getStatus()}")
            print(f"Cost: {cost:.4f}")
            print(f"Routes: {routes}")
        
        return SCIPSolution(
            routes=routes,
            cost=cost,
            is_optimal=is_optimal,
            solve_time=solve_time
        )
    
    def solve_batch(self, instances: List[Dict[str, Any]], 
                   verbose: bool = False) -> List[SCIPSolution]:
        """
        Solve multiple CVRP instances sequentially
        """
        solutions = []
        total_start = time.time()
        
        for i, instance in enumerate(instances):
            if verbose and i % 10 == 0:
                print(f"Solving instance {i+1}/{len(instances)}")
            
            solution = self.solve(instance, verbose=(verbose and i == 0))
            solutions.append(solution)
        
        total_time = time.time() - total_start
        
        if verbose:
            print(f"\nTotal batch solve time: {total_time:.2f}s")
            print(f"Average time per instance: {total_time/len(instances):.2f}s")
        
        return solutions
    
    def _extract_routes(self, x, n, max_vehicles):
        """Extract routes from solution"""
        routes = []
        
        for k in range(max_vehicles):
            route = []
            current = 0  # Start at depot
            visited = set([0])
            
            while True:
                next_node = None
                for j in range(n):
                    if j not in visited and x.get((current, j, k)):
                        if x[current, j, k].getVal() > 0.5:
                            next_node = j
                            break
                
                if next_node is None or next_node == 0:
                    break
                
                route.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if route:
                routes.append([0] + route + [0])
        
        return routes
