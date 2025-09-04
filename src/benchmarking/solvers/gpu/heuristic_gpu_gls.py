#!/usr/bin/env python3
"""
GPU-accelerated heuristic CVRP solver using PyTorch.
Implements a parallel Guided Local Search (GLS) inspired algorithm on GPU.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from src.benchmarking.solvers.types import CVRPSolution
import time


class GPUHeuristicGLS:
    """GPU-accelerated Guided Local Search for CVRP."""
    
    def __init__(self, device='cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. No CPU fallback.")
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
    
    def solve_batch(self, instances: List[Dict[str, Any]], 
                   time_limit: float = 5.0,
                   verbose: bool = False) -> List[CVRPSolution]:
        """
        Solve multiple CVRP instances using GPU-accelerated GLS heuristic.
        
        Args:
            instances: List of CVRP instances
            time_limit: Time limit per instance in seconds
            verbose: Print progress information
        
        Returns:
            List of CVRPSolution objects
        """
        start_time = time.time()
        batch_size = len(instances)
        n = len(instances[0]['coords'])
        n_customers = n - 1
        
        if verbose:
            print(f"GPU GLS solving {batch_size} instances with {n_customers} customers")
            print(f"Time limit: {time_limit}s per instance")
        
        # Convert to GPU tensors
        distances = torch.tensor(
            np.array([inst['distances'] for inst in instances]),
            dtype=torch.float32, device=self.device
        )
        demands = torch.tensor(
            np.array([inst['demands'] for inst in instances]),
            dtype=torch.float32, device=self.device
        )
        capacities = torch.tensor(
            [inst['capacity'] for inst in instances],
            dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        
        # Initialize with greedy solution
        routes = self._greedy_initialize(distances, demands, capacities, n_customers)
        best_costs = self._evaluate_routes(routes, distances)
        best_routes = routes.clone()
        
        # GLS parameters
        alpha = 0.3  # Penalty weight
        penalties = torch.zeros_like(distances)
        
        # Local search with guided penalties
        iterations = 0
        max_iterations = int(time_limit * 100)  # Adjust based on performance
        
        improvement_found = True
        while improvement_found and iterations < max_iterations:
            improvement_found = False
            
            # Apply multiple neighborhood operators in parallel
            new_routes_list = []
            
            # 1. Two-opt within routes
            two_opt_routes = self._batch_two_opt(routes, distances + alpha * penalties)
            new_routes_list.append(two_opt_routes)
            
            # 2. Customer relocation between routes
            relocate_routes = self._batch_relocate(routes, distances + alpha * penalties, 
                                                   demands, capacities)
            new_routes_list.append(relocate_routes)
            
            # 3. Customer exchange between routes
            exchange_routes = self._batch_exchange(routes, distances + alpha * penalties,
                                                   demands, capacities)
            new_routes_list.append(exchange_routes)
            
            # Evaluate all candidates
            for new_routes in new_routes_list:
                new_costs = self._evaluate_routes(new_routes, distances)
                
                # Update best solutions
                improved = new_costs < best_costs
                if improved.any():
                    improvement_found = True
                    best_routes[improved] = new_routes[improved]
                    best_costs[improved] = new_costs[improved]
                    routes[improved] = new_routes[improved]
            
            # Update penalties (GLS mechanism)
            if iterations % 10 == 0:
                penalties = self._update_penalties(best_routes, distances, penalties, alpha)
            
            iterations += 1
            
            # Check time limit
            if time.time() - start_time > time_limit * batch_size:
                break
        
        if verbose:
            print(f"Completed {iterations} iterations in {time.time() - start_time:.2f}s")
            print(f"Average cost: {best_costs.mean().item():.4f}")
        
        # Convert back to CVRPSolution format
        solutions = []
        for i in range(batch_size):
            route_list = self._tensor_to_route_list(best_routes[i], n_customers)
            solution = CVRPSolution(
                route=[0] + [c for r in route_list for c in r] + [0],
                vehicle_routes=route_list,
                cost=best_costs[i].item(),
                solve_time=time.time() - start_time,
                algorithm_used="GPU_GLS",
                num_vehicles=len(route_list),
                is_optimal=False
            )
            solutions.append(solution)
        
        return solutions
    
    def _greedy_initialize(self, distances: torch.Tensor, demands: torch.Tensor,
                          capacities: torch.Tensor, n_customers: int) -> torch.Tensor:
        """
        Initialize routes using parallel nearest neighbor heuristic.
        Returns tensor of shape (batch_size, max_vehicles, max_route_length)
        """
        batch_size = distances.shape[0]
        max_vehicles = n_customers  # Worst case: one customer per vehicle
        max_route_length = n_customers + 2  # Depot at start and end
        
        # Initialize routes tensor (-1 for empty slots)
        routes = torch.full((batch_size, max_vehicles, max_route_length), 
                           -1, dtype=torch.long, device=self.device)
        
        # Track unvisited customers
        unvisited = torch.ones((batch_size, n_customers), dtype=torch.bool, device=self.device)
        
        # Build routes greedily
        for b in range(batch_size):
            vehicle_idx = 0
            route_pos = 1  # Position 0 is depot
            current_capacity = capacities[b].item()
            routes[b, vehicle_idx, 0] = 0  # Start at depot
            
            while unvisited[b].any():
                # Find nearest unvisited customer that fits
                current_node = routes[b, vehicle_idx, route_pos - 1]
                if current_node == -1:
                    current_node = 0
                
                valid_mask = unvisited[b].clone()
                valid_demands = demands[b, 1:n_customers+1]
                valid_mask[valid_demands > current_capacity] = False
                
                if not valid_mask.any():
                    # Start new vehicle
                    routes[b, vehicle_idx, route_pos] = 0  # Return to depot
                    vehicle_idx += 1
                    if vehicle_idx >= max_vehicles:
                        break
                    route_pos = 1
                    current_capacity = capacities[b].item()
                    routes[b, vehicle_idx, 0] = 0  # Start at depot
                else:
                    # Select nearest valid customer
                    dist_to_customers = distances[b, current_node, 1:n_customers+1]
                    dist_to_customers[~valid_mask] = float('inf')
                    nearest = dist_to_customers.argmin()
                    
                    # Add to route
                    routes[b, vehicle_idx, route_pos] = nearest + 1
                    unvisited[b, nearest] = False
                    current_capacity -= demands[b, nearest + 1].item()
                    route_pos += 1
        
        return routes
    
    def _batch_two_opt(self, routes: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """Apply 2-opt improvements within routes in parallel."""
        batch_size, max_vehicles, max_route_length = routes.shape
        improved_routes = routes.clone()
        
        # For each vehicle route, try 2-opt swaps
        for v in range(max_vehicles):
            for i in range(1, max_route_length - 2):
                for j in range(i + 1, max_route_length - 1):
                    # Get route segments
                    node_i = routes[:, v, i]
                    node_i_prev = routes[:, v, i - 1]
                    node_j = routes[:, v, j]
                    node_j_next = routes[:, v, j + 1]
                    
                    # Skip if any node is invalid
                    valid = (node_i >= 0) & (node_j >= 0) & (node_j_next >= 0)
                    if not valid.any():
                        continue
                    
                    # Calculate cost difference
                    current_cost = (
                        self._get_edge_cost(distances, node_i_prev, node_i) +
                        self._get_edge_cost(distances, node_j, node_j_next)
                    )
                    new_cost = (
                        self._get_edge_cost(distances, node_i_prev, node_j) +
                        self._get_edge_cost(distances, node_i, node_j_next)
                    )
                    
                    # Apply improvement where beneficial
                    improve = valid & (new_cost < current_cost)
                    if improve.any():
                        # Reverse segment [i, j] for improved instances
                        for b in torch.where(improve)[0]:
                            segment = improved_routes[b, v, i:j+1].clone()
                            improved_routes[b, v, i:j+1] = segment.flip(0)
        
        return improved_routes
    
    def _batch_relocate(self, routes: torch.Tensor, distances: torch.Tensor,
                       demands: torch.Tensor, capacities: torch.Tensor) -> torch.Tensor:
        """Relocate customers between routes in parallel."""
        batch_size, max_vehicles, max_route_length = routes.shape
        improved_routes = routes.clone()
        
        # Try relocating each customer to different vehicles
        for src_v in range(max_vehicles):
            for src_pos in range(1, max_route_length - 1):
                customer = routes[:, src_v, src_pos]
                valid = customer > 0
                if not valid.any():
                    continue
                
                # Calculate current route loads
                route_demands = self._calculate_route_demands(routes, demands)
                
                for dst_v in range(max_vehicles):
                    if src_v == dst_v:
                        continue
                    
                    # Check capacity constraints
                    customer_demand = torch.where(
                        customer > 0,
                        demands[torch.arange(batch_size), customer],
                        torch.zeros(batch_size, device=self.device)
                    )
                    
                    new_load = route_demands[:, dst_v] + customer_demand
                    feasible = valid & (new_load <= capacities.squeeze())
                    
                    if feasible.any():
                        # Find best insertion position
                        best_pos = self._find_best_insertion(
                            improved_routes[:, dst_v], customer, distances
                        )
                        
                        # Calculate cost improvement
                        removal_gain = self._calculate_removal_cost(
                            improved_routes[:, src_v], src_pos, distances
                        )
                        insertion_cost = self._calculate_insertion_cost(
                            improved_routes[:, dst_v], best_pos, customer, distances
                        )
                        
                        improve = feasible & (removal_gain > insertion_cost)
                        
                        # Apply improvements
                        for b in torch.where(improve)[0]:
                            # Remove from source
                            improved_routes[b, src_v, src_pos:-1] = improved_routes[b, src_v, src_pos+1:].clone()
                            improved_routes[b, src_v, -1] = -1
                            
                            # Insert to destination
                            pos = best_pos[b].item()
                            improved_routes[b, dst_v, pos+1:] = improved_routes[b, dst_v, pos:-1].clone()
                            improved_routes[b, dst_v, pos] = customer[b]
        
        return improved_routes
    
    def _batch_exchange(self, routes: torch.Tensor, distances: torch.Tensor,
                       demands: torch.Tensor, capacities: torch.Tensor) -> torch.Tensor:
        """Exchange customers between routes in parallel."""
        batch_size, max_vehicles, max_route_length = routes.shape
        improved_routes = routes.clone()
        
        # Try exchanging customers between vehicle pairs
        for v1 in range(max_vehicles - 1):
            for v2 in range(v1 + 1, max_vehicles):
                for pos1 in range(1, max_route_length - 1):
                    for pos2 in range(1, max_route_length - 1):
                        customer1 = routes[:, v1, pos1]
                        customer2 = routes[:, v2, pos2]
                        
                        valid = (customer1 > 0) & (customer2 > 0)
                        if not valid.any():
                            continue
                        
                        # Check capacity constraints after exchange
                        route1_demands = self._calculate_single_route_demand(
                            routes[:, v1], demands
                        )
                        route2_demands = self._calculate_single_route_demand(
                            routes[:, v2], demands
                        )
                        
                        demand1 = torch.where(
                            customer1 > 0,
                            demands[torch.arange(batch_size), customer1],
                            torch.zeros(batch_size, device=self.device)
                        )
                        demand2 = torch.where(
                            customer2 > 0,
                            demands[torch.arange(batch_size), customer2],
                            torch.zeros(batch_size, device=self.device)
                        )
                        
                        new_route1_demand = route1_demands - demand1 + demand2
                        new_route2_demand = route2_demands - demand2 + demand1
                        
                        feasible = valid & (new_route1_demand <= capacities.squeeze()) & \
                                 (new_route2_demand <= capacities.squeeze())
                        
                        if feasible.any():
                            # Calculate cost difference
                            current_cost = (
                                self._calculate_customer_cost(routes[:, v1], pos1, distances) +
                                self._calculate_customer_cost(routes[:, v2], pos2, distances)
                            )
                            
                            # Cost after exchange
                            improved_temp = improved_routes.clone()
                            improved_temp[:, v1, pos1] = customer2
                            improved_temp[:, v2, pos2] = customer1
                            
                            new_cost = (
                                self._calculate_customer_cost(improved_temp[:, v1], pos1, distances) +
                                self._calculate_customer_cost(improved_temp[:, v2], pos2, distances)
                            )
                            
                            improve = feasible & (new_cost < current_cost)
                            
                            # Apply improvements
                            if improve.any():
                                improved_routes[improve, v1, pos1] = customer2[improve]
                                improved_routes[improve, v2, pos2] = customer1[improve]
        
        return improved_routes
    
    def _update_penalties(self, routes: torch.Tensor, distances: torch.Tensor,
                         penalties: torch.Tensor, alpha: float) -> torch.Tensor:
        """Update penalties for Guided Local Search."""
        batch_size = routes.shape[0]
        
        # Find edges with highest utility (cost / (1 + penalty))
        utilities = distances / (1.0 + penalties)
        
        # Penalize edges in current solution with high utility
        for b in range(batch_size):
            for v in range(routes.shape[1]):
                for i in range(routes.shape[2] - 1):
                    node1 = routes[b, v, i]
                    node2 = routes[b, v, i + 1]
                    if node1 >= 0 and node2 >= 0:
                        # Increase penalty on this edge
                        penalties[b, node1, node2] += 1.0
                        penalties[b, node2, node1] += 1.0
        
        return penalties
    
    def _evaluate_routes(self, routes: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """Calculate total cost for each instance's routes."""
        batch_size = routes.shape[0]
        costs = torch.zeros(batch_size, device=self.device)
        
        for b in range(batch_size):
            for v in range(routes.shape[1]):
                for i in range(routes.shape[2] - 1):
                    node1 = routes[b, v, i]
                    node2 = routes[b, v, i + 1]
                    if node1 >= 0 and node2 >= 0:
                        costs[b] += distances[b, node1, node2]
        
        return costs
    
    def _get_edge_cost(self, distances: torch.Tensor, node1: torch.Tensor, 
                      node2: torch.Tensor) -> torch.Tensor:
        """Get edge costs for batched node pairs."""
        batch_size = distances.shape[0]
        costs = torch.zeros(batch_size, device=self.device)
        
        valid = (node1 >= 0) & (node2 >= 0)
        if valid.any():
            batch_idx = torch.arange(batch_size, device=self.device)[valid]
            costs[valid] = distances[batch_idx, node1[valid], node2[valid]]
        
        return costs
    
    def _calculate_route_demands(self, routes: torch.Tensor, 
                                demands: torch.Tensor) -> torch.Tensor:
        """Calculate total demand for each vehicle route."""
        batch_size, max_vehicles, _ = routes.shape
        route_demands = torch.zeros((batch_size, max_vehicles), device=self.device)
        
        for b in range(batch_size):
            for v in range(max_vehicles):
                for node in routes[b, v]:
                    if node > 0:  # Skip depot (0) and invalid (-1)
                        route_demands[b, v] += demands[b, node]
        
        return route_demands
    
    def _calculate_single_route_demand(self, route: torch.Tensor, 
                                      demands: torch.Tensor) -> torch.Tensor:
        """Calculate total demand for a single route (batched)."""
        batch_size = route.shape[0]
        total_demand = torch.zeros(batch_size, device=self.device)
        
        for b in range(batch_size):
            for node in route[b]:
                if node > 0:
                    total_demand[b] += demands[b, node]
        
        return total_demand
    
    def _find_best_insertion(self, route: torch.Tensor, customer: torch.Tensor,
                            distances: torch.Tensor) -> torch.Tensor:
        """Find best position to insert customer into route."""
        batch_size = route.shape[0]
        best_pos = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        best_cost = torch.full((batch_size,), float('inf'), device=self.device)
        
        for pos in range(1, route.shape[1]):
            prev_node = route[:, pos - 1]
            next_node = route[:, pos]
            
            valid = (prev_node >= 0) & ((next_node >= 0) | (pos == route.shape[1] - 1))
            if not valid.any():
                continue
            
            # If next_node is -1, treat as depot (0)
            next_node = torch.where(next_node >= 0, next_node, 
                                   torch.zeros_like(next_node))
            
            insertion_cost = (
                self._get_edge_cost(distances, prev_node, customer) +
                self._get_edge_cost(distances, customer, next_node) -
                self._get_edge_cost(distances, prev_node, next_node)
            )
            
            better = valid & (insertion_cost < best_cost)
            best_cost[better] = insertion_cost[better]
            best_pos[better] = pos
        
        return best_pos
    
    def _calculate_removal_cost(self, route: torch.Tensor, pos: int,
                               distances: torch.Tensor) -> torch.Tensor:
        """Calculate cost reduction from removing customer at position."""
        batch_size = route.shape[0]
        
        if pos == 0 or pos >= route.shape[1] - 1:
            return torch.zeros(batch_size, device=self.device)
        
        prev_node = route[:, pos - 1]
        curr_node = route[:, pos]
        next_node = route[:, pos + 1]
        
        # Handle invalid nodes
        next_node = torch.where(next_node >= 0, next_node, 
                               torch.zeros_like(next_node))
        
        valid = (prev_node >= 0) & (curr_node >= 0)
        
        removal_gain = torch.zeros(batch_size, device=self.device)
        if valid.any():
            removal_gain[valid] = (
                self._get_edge_cost(distances, prev_node, curr_node)[valid] +
                self._get_edge_cost(distances, curr_node, next_node)[valid] -
                self._get_edge_cost(distances, prev_node, next_node)[valid]
            )
        
        return removal_gain
    
    def _calculate_insertion_cost(self, route: torch.Tensor, pos: torch.Tensor,
                                 customer: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """Calculate cost of inserting customer at position."""
        batch_size = route.shape[0]
        costs = torch.zeros(batch_size, device=self.device)
        
        for b in range(batch_size):
            if pos[b] > 0 and pos[b] < route.shape[1]:
                prev_node = route[b, pos[b] - 1]
                next_node = route[b, pos[b]]
                
                if next_node < 0:
                    next_node = torch.tensor(0, device=self.device)
                
                if prev_node >= 0 and customer[b] > 0:
                    costs[b] = (
                        distances[b, prev_node, customer[b]] +
                        distances[b, customer[b], next_node] -
                        distances[b, prev_node, next_node]
                    )
        
        return costs
    
    def _calculate_customer_cost(self, route: torch.Tensor, pos: int,
                                distances: torch.Tensor) -> torch.Tensor:
        """Calculate cost contribution of customer at position."""
        batch_size = route.shape[0]
        costs = torch.zeros(batch_size, device=self.device)
        
        if pos > 0 and pos < route.shape[1] - 1:
            prev_node = route[:, pos - 1]
            curr_node = route[:, pos]
            next_node = route[:, pos + 1]
            
            # Handle invalid nodes
            next_node = torch.where(next_node >= 0, next_node,
                                   torch.zeros_like(next_node))
            
            valid = (prev_node >= 0) & (curr_node >= 0)
            if valid.any():
                costs[valid] = (
                    self._get_edge_cost(distances, prev_node, curr_node)[valid] +
                    self._get_edge_cost(distances, curr_node, next_node)[valid]
                )
        
        return costs
    
    def _tensor_to_route_list(self, routes_tensor: torch.Tensor, 
                             n_customers: int) -> List[List[int]]:
        """Convert route tensor to list of routes."""
        route_list = []
        
        for v in range(routes_tensor.shape[0]):
            route = []
            for node in routes_tensor[v]:
                if node > 0:  # Skip depot and invalid nodes
                    route.append(int(node.item()))
                elif node == 0 and len(route) > 0:
                    # Depot marks end of route
                    route_list.append(route)
                    route = []
            
            if len(route) > 0:
                route_list.append(route)
        
        return route_list


def solve_batch(instances: List[Dict[str, Any]], 
                time_limit: float = 5.0,
                verbose: bool = False) -> List[CVRPSolution]:
    """Convenience function to solve batch of instances."""
    solver = GPUHeuristicGLS()
    return solver.solve_batch(instances, time_limit=time_limit, verbose=verbose)
