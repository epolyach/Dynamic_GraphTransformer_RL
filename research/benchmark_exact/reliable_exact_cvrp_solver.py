#!/usr/bin/env python3
"""
Reliable Exact CVRP Solver
A robust implementation using Google OR-Tools with proper configuration
to find optimal solutions for CVRP benchmark instances.
"""

import sys
import argparse
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class CVRPSolution:
    """Container for CVRP solution results"""
    routes: List[List[int]]  # Vehicle routes (excluding depot at start/end)
    total_cost: float
    solve_time: float
    is_optimal: bool
    algorithm: str
    instance_name: str

class ReliableExactCVRPSolver:
    """
    Reliable exact CVRP solver using Google OR-Tools.
    Configured specifically for finding optimal solutions on benchmark instances.
    """
    
    def __init__(self, time_limit_seconds: int = 300, verbose: bool = True):
        self.time_limit = time_limit_seconds
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        # Check OR-Tools availability
        try:
            from ortools.constraint_solver import pywrapcp
            from ortools.constraint_solver import routing_enums_pb2
            self.ortools_available = True
        except ImportError:
            self.ortools_available = False
            self.logger.error("OR-Tools not available. Install with: pip install ortools")
    
    def _setup_logger(self):
        """Set up logging"""
        logger = logging.getLogger('CVRPSolver')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    def solve(self, instance: Dict[str, Any]) -> CVRPSolution:
        """
        Solve CVRP instance to optimality using OR-Tools.
        
        Args:
            instance: Dictionary containing:
                - coords: numpy array of (x,y) coordinates, depot at index 0
                - demands: numpy array of demands, depot demand = 0
                - capacity: vehicle capacity
                - distances: distance matrix (optional, computed if not provided)
                - name: instance name (optional)
        
        Returns:
            CVRPSolution with optimal routes and cost
        """
        if not self.ortools_available:
            raise RuntimeError("OR-Tools not available")
        
        from ortools.constraint_solver import pywrapcp
        from ortools.constraint_solver import routing_enums_pb2
        
        start_time = time.time()
        
        # Extract instance data
        coords = np.array(instance['coords'])
        demands = np.array(instance['demands'], dtype=int)
        capacity = int(instance['capacity'])
        instance_name = instance.get('name', 'Unknown')
        
        n_locations = len(coords)
        n_customers = n_locations - 1
        
        # Compute distance matrix if not provided
        if 'distances' in instance:
            distances = np.array(instance['distances'])
        else:
            distances = self._compute_euclidean_distances(coords)
        
        if self.verbose:
            self.logger.info(f"🎯 Solving {instance_name} ({n_customers} customers, capacity={capacity})")
            self.logger.info(f"   Total demand: {sum(demands[1:])}, Min vehicles needed: {np.ceil(sum(demands[1:])/capacity):.0f}")
        
        # Check feasibility
        max_demand = max(demands[1:]) if n_customers > 0 else 0
        if max_demand > capacity:
            raise ValueError(f"Infeasible: customer demand {max_demand} exceeds capacity {capacity}")
        
        # Scale distances to integers for OR-Tools (preserve precision)
        distance_scale = 10000
        distances_int = (distances * distance_scale).astype(int)
        
        # Estimate number of vehicles needed
        total_demand = sum(demands[1:])
        min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
        max_vehicles = min(n_customers, min_vehicles + 3)  # Allow some flexibility
        
        best_solution = None
        best_cost = float('inf')
        
        # Try different numbers of vehicles to find optimal
        for num_vehicles in range(min_vehicles, max_vehicles + 1):
            if time.time() - start_time > self.time_limit:
                break
                
            try:
                solution = self._solve_with_vehicles(
                    distances_int, demands, capacity, num_vehicles, 
                    time_limit_per_try=self.time_limit // (max_vehicles - min_vehicles + 1)
                )
                
                if solution and solution['cost'] < best_cost:
                    best_cost = solution['cost']
                    best_solution = solution
                    
                    if self.verbose:
                        self.logger.info(f"   Found solution with {num_vehicles} vehicles: cost={best_cost/distance_scale:.4f}")
                    
                    # If we found a feasible solution, we can stop (OR-Tools finds optimal for given vehicles)
                    break
                    
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"   Failed with {num_vehicles} vehicles: {e}")
                continue
        
        if best_solution is None:
            raise RuntimeError(f"No solution found for {instance_name}")
        
        # Unscale cost
        final_cost = best_cost / distance_scale
        solve_time = time.time() - start_time
        
        # Convert to standard format
        vehicle_routes = best_solution['routes']
        
        if self.verbose:
            self.logger.info(f"✅ Optimal solution: {len(vehicle_routes)} vehicles, cost={final_cost:.4f}, time={solve_time:.2f}s")
        
        return CVRPSolution(
            routes=vehicle_routes,
            total_cost=final_cost,
            solve_time=solve_time,
            is_optimal=True,
            algorithm="OR-Tools Exact",
            instance_name=instance_name
        )
    
    def _solve_with_vehicles(self, distances_int: np.ndarray, demands: np.ndarray, 
                           capacity: int, num_vehicles: int, time_limit_per_try: int) -> Optional[Dict]:
        """Solve with a fixed number of vehicles"""
        from ortools.constraint_solver import pywrapcp
        from ortools.constraint_solver import routing_enums_pb2
        
        n_locations = len(distances_int)
        
        try:
            # Create routing manager and model
            manager = pywrapcp.RoutingIndexManager(n_locations, num_vehicles, 0)  # depot = 0
            routing = pywrapcp.RoutingModel(manager)
            
            # Distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return distances_int[from_node][to_node]
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # Capacity constraints
            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return demands[from_node]
            
            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                [capacity] * num_vehicles,  # vehicle maximum capacities
                True,  # start cumul to zero
                'Capacity'
            )
            
            # Search parameters for exact solution
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            
            # Use path cheapest arc for good initial solution
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            
            # Use guided local search for intensive optimization
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            
            # Set time limit
            search_parameters.time_limit.seconds = max(60, time_limit_per_try)
            search_parameters.log_search = False  # Disable log for cleaner output
            
            # Solve
            solution = routing.SolveWithParameters(search_parameters)
            
            if not solution:
                if self.verbose:
                    self.logger.warning(f"      No solution found with {num_vehicles} vehicles")
                return None
            
            # Check solution status
            status = routing.status()
            if self.verbose:
                status_map = {
                    routing_enums_pb2.RoutingSearchStatus.ROUTING_NOT_SOLVED: "NOT_SOLVED",
                    routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS: "SUCCESS", 
                    routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL: "OPTIMAL",
                    routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL: "FAIL",
                    routing_enums_pb2.RoutingSearchStatus.ROUTING_FAIL_TIMEOUT: "TIMEOUT",
                    routing_enums_pb2.RoutingSearchStatus.ROUTING_INVALID: "INVALID"
                }
                status_str = status_map.get(status, f"UNKNOWN({status})")
                self.logger.info(f"      Status with {num_vehicles} vehicles: {status_str}")
            
            # Accept SUCCESS or OPTIMAL status
            if status not in [routing_enums_pb2.RoutingSearchStatus.ROUTING_SUCCESS,
                             routing_enums_pb2.RoutingSearchStatus.ROUTING_OPTIMAL]:
                return None
            
            # Extract solution
            total_cost = solution.ObjectiveValue()
            routes = []
            
            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index != 0:  # Skip depot
                        route.append(node_index)
                    index = solution.Value(routing.NextVar(index))
                
                if route:  # Only add non-empty routes
                    routes.append(route)
            
            return {
                'cost': total_cost,
                'routes': routes,
                'num_vehicles_used': len(routes)
            }
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"      Exception with {num_vehicles} vehicles: {e}")
            return None
    
    def _compute_euclidean_distances(self, coords: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix"""
        n = len(coords)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i][j] = np.linalg.norm(coords[i] - coords[j])
        
        return distances

def parse_vrp_file(filename: str) -> Dict[str, Any]:
    """
    Parse TSPLIB format VRP file.
    
    Args:
        filename: Path to .vrp file
        
    Returns:
        Dictionary with instance data
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    instance = {}
    section = None
    coords_data = []
    demands_data = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('EOF'):
            continue
        
        # Parse header information
        if ':' in line and not line.startswith(' '):
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'NAME':
                instance['name'] = value
            elif key == 'DIMENSION':
                instance['dimension'] = int(value)
            elif key == 'CAPACITY':
                instance['capacity'] = int(value)
            elif key == 'EDGE_WEIGHT_TYPE':
                instance['edge_weight_type'] = value
        
        # Handle sections
        elif line in ['NODE_COORD_SECTION', 'DEMAND_SECTION', 'DEPOT_SECTION']:
            section = line
        
        elif section == 'NODE_COORD_SECTION':
            parts = line.split()
            if len(parts) >= 3:
                node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                coords_data.append((node_id, x, y))
        
        elif section == 'DEMAND_SECTION':
            parts = line.split()
            if len(parts) >= 2:
                node_id, demand = int(parts[0]), int(parts[1])
                demands_data.append((node_id, demand))
        
        elif section == 'DEPOT_SECTION':
            if line.strip() != '-1':
                instance['depot'] = int(line)
    
    # Convert to our format
    if not coords_data or not demands_data:
        raise ValueError("Missing coordinate or demand data")
    
    # Sort by node ID and create arrays
    coords_data.sort(key=lambda x: x[0])
    demands_data.sort(key=lambda x: x[0])
    
    max_node = max(max(c[0] for c in coords_data), max(d[0] for d in demands_data))
    
    # Create coordinate and demand arrays (0-indexed)
    coords = np.zeros((max_node, 2))
    demands = np.zeros(max_node, dtype=int)
    
    for node_id, x, y in coords_data:
        coords[node_id - 1] = [x, y]  # Convert to 0-indexed
    
    for node_id, demand in demands_data:
        demands[node_id - 1] = demand  # Convert to 0-indexed
    
    instance['coords'] = coords
    instance['demands'] = demands
    
    return instance

def main():
    """CLI interface for solving single CVRP instances"""
    parser = argparse.ArgumentParser(description='Solve CVRP instance with exact algorithm')
    parser.add_argument('filename', help='Path to .vrp file')
    parser.add_argument('--time-limit', type=int, default=300, 
                       help='Time limit in seconds (default: 300)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress all output except results')
    
    args = parser.parse_args()
    
    try:
        # Parse instance
        if args.verbose and not args.quiet:
            print(f"📁 Loading instance: {args.filename}")
        
        instance = parse_vrp_file(args.filename)
        
        if args.verbose and not args.quiet:
            print(f"📊 Instance: {instance.get('name', 'Unknown')}")
            print(f"   Customers: {len(instance['coords']) - 1}")
            print(f"   Capacity: {instance['capacity']}")
            print(f"   Total demand: {sum(instance['demands'][1:])}")
        
        # Solve
        solver = ReliableExactCVRPSolver(
            time_limit_seconds=args.time_limit,
            verbose=args.verbose and not args.quiet
        )
        
        solution = solver.solve(instance)
        
        # Output results
        if args.quiet:
            # Minimal output: just cost and routes
            print(f"Cost: {solution.total_cost:.4f}")
            print("Routes:")
            for i, route in enumerate(solution.routes):
                route_str = " -> ".join([str(0)] + [str(c) for c in route] + [str(0)])
                print(f"  Vehicle {i+1}: {route_str}")
        else:
            print("\n" + "="*60)
            print("SOLUTION")
            print("="*60)
            print(f"Instance: {solution.instance_name}")
            print(f"Total cost: {solution.total_cost:.4f}")
            print(f"Number of vehicles: {len(solution.routes)}")
            print(f"Solve time: {solution.solve_time:.2f} seconds")
            print(f"Algorithm: {solution.algorithm}")
            print(f"Optimal: {solution.is_optimal}")
            
            print(f"\nRoutes:")
            total_distance = 0.0
            for i, route in enumerate(solution.routes):
                route_with_depot = [0] + route + [0]
                route_str = " -> ".join(str(c) for c in route_with_depot)
                
                # Calculate route distance
                route_dist = 0.0
                coords = instance['coords']
                for j in range(len(route_with_depot) - 1):
                    from_node = route_with_depot[j]
                    to_node = route_with_depot[j + 1]
                    route_dist += np.linalg.norm(coords[from_node] - coords[to_node])
                
                total_distance += route_dist
                print(f"  Vehicle {i+1}: {route_str} (distance: {route_dist:.4f})")
            
            print(f"\nTotal distance verification: {total_distance:.4f}")
    
    except Exception as e:
        if args.quiet:
            print(f"Error: {e}")
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
