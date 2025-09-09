#!/usr/bin/env python3
"""
SCIP-based CVRP Solver for guaranteed optimal solutions
Requires: pip install pyscipopt
"""

import numpy as np
from typing import Dict, Any, List, Optional
import time
import sys

sys.path.append('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
from src.benchmarking.solvers.types import CVRPSolution

try:
    from pyscipopt import Model, quicksum
    SCIP_AVAILABLE = True
except ImportError:
    SCIP_AVAILABLE = False
    print("WARNING: PySCIPOpt not installed. Install with: pip install pyscipopt")

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
    
    def solve(self, instance: Dict[str, Any], verbose: bool = False) -> CVRPSolution:
        """
        Solve a single CVRP instance using SCIP
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
        
        return CVRPSolution(
            route=[],
            vehicle_routes=
            vehicle_routes=routes,
            cost=cost,
            num_vehicles=len(routes) if routes else 0,
            solve_time=0,
            is_feasible=True
        )
    
    def solve_batch(self, instances: List[Dict[str, Any]], 
                   verbose: bool = False) -> List[CVRPSolution]:
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


def benchmark_scip_solver():
    """
    Benchmark SCIP solver for N=10, C=20,30
    """
    import pandas as pd
    from datetime import datetime
    from src.generator.generator import _generate_instance
    
    if not SCIP_AVAILABLE:
        print("ERROR: PySCIPOpt not installed!")
        print("Install with: pip install pyscipopt")
        return None
    
    print("\n" + "="*70)
    print("SCIP CVRP SOLVER - BENCHMARK")
    print("="*70)
    
    # Test parameters
    n_customers = 10
    capacities = [20, 30]
    num_instances_per_capacity = 10  # Reduced for SCIP (slower than GPU)
    
    solver = SCIPCVRPSolver(time_limit=60, gap=0.0)  # 60s limit per instance
    
    results = []
    
    for capacity in capacities:
        print(f"\n{'='*50}")
        print(f"Testing N={n_customers}, C={capacity}")
        print(f"{'='*50}")
        
        # Generate instances
        instances = []
        for i in range(num_instances_per_capacity):
            inst = _generate_instance(
                n_customers=n_customers,
                capacity=capacity,
                min_demand=1,
                max_demand=10,
                grid_size=1
            )
            instances.append(inst)
        
        # Solve
        start_time = time.time()
        solutions = solver.solve_batch(instances, verbose=True)
        total_time = time.time() - start_time
        
        # Collect results
        valid_solutions = [s for s in solutions if s.cost < float('inf')]
        if valid_solutions:
            avg_cost = np.mean([s.cost for s in valid_solutions])
            avg_vehicles = np.mean([len(s.routes) for s in valid_solutions])
        else:
            avg_cost = float('inf')
            avg_vehicles = 0
        
        success_rate = len(valid_solutions) / len(solutions) * 100
        throughput = num_instances_per_capacity / total_time
        
        print(f"\nResults for C={capacity}:")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average cost: {avg_cost:.4f}")
        print(f"  Average vehicles: {avg_vehicles:.2f}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.3f} instances/second")
        
        # Estimate time for larger batches
        time_per_instance = total_time / num_instances_per_capacity
        est_time_1k = time_per_instance * 1000 / 3600  # hours
        est_time_10k = time_per_instance * 10000 / 3600  # hours
        
        print(f"\nProjected times (if all succeed):")
        print(f"  1,000 instances: {est_time_1k:.1f} hours")
        print(f"  10,000 instances: {est_time_10k:.1f} hours ({est_time_10k/24:.1f} days)")
        
        results.append({
            'n_customers': n_customers,
            'capacity': capacity,
            'success_rate': success_rate,
            'avg_cost': avg_cost,
            'avg_vehicles': avg_vehicles,
            'throughput': throughput,
            'est_hours_1k': est_time_1k,
            'est_days_10k': est_time_10k/24
        })
    
    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"scip_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SCIP CVRP Solver")
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark for N=10, C=20,30')
    parser.add_argument('--time-limit', type=float, default=60,
                       help='Time limit per instance in seconds')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_scip_solver()
    else:
        print("Use --benchmark to run performance tests")
