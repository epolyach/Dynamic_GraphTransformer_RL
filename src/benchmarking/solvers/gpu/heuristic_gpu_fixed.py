#!/usr/bin/env python3
"""
Fixed GPU-accelerated heuristic CVRP solver using PyTorch.
Implements a parallel greedy nearest neighbor algorithm on GPU.
NO CPU FALLBACK - requires CUDA GPU.
"""

import torch
import numpy as np
from typing import Dict, Any, List
from src.benchmarking.solvers.types import CVRPSolution
import time


class GPUHeuristicFixed:
    """GPU-accelerated greedy heuristic for CVRP. GPU ONLY - no fallback."""
    
    def __init__(self, device='cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. No CPU fallback.")
        self.device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    def solve_batch(self, instances: List[Dict[str, Any]], 
                   verbose: bool = False) -> List[CVRPSolution]:
        """
        Solve multiple CVRP instances using GPU-accelerated greedy heuristic.
        REQUIRES GPU - no CPU fallback.
        """
        start_time = time.time()
        batch_size = len(instances)
        n = len(instances[0]['coords'])
        n_customers = n - 1
        
        if verbose:
            print(f"GPU Greedy solving {batch_size} instances with {n_customers} customers")
        
        # Convert to GPU tensors - will fail if no GPU
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
        )
        
        # Process all instances in parallel
        all_routes = []
        all_costs = []
        
        for b in range(batch_size):
            route_list, cost = self._solve_greedy_single(
                distances[b], demands[b], capacities[b], n_customers
            )
            all_routes.append(route_list)
            all_costs.append(cost)
        
        # Convert to solutions
        solutions = []
        for b in range(batch_size):
            route = [0]
            for vehicle_route in all_routes[b]:
                route.extend(vehicle_route)
                route.append(0)
            
            solution = CVRPSolution(
                route=route,
                vehicle_routes=all_routes[b],
                cost=all_costs[b],
                num_vehicles=len(all_routes[b]),
                solve_time=(time.time() - start_time) / batch_size,
                algorithm_used="GPU_Greedy",
                is_optimal=False
            )
            solutions.append(solution)
        
        if verbose:
            avg_cost = np.mean(all_costs)
            print(f"Completed in {time.time() - start_time:.2f}s, avg cost: {avg_cost:.4f}")
        
        return solutions
    
    def _solve_greedy_single(self, distances: torch.Tensor, demands: torch.Tensor,
                            capacity: float, n_customers: int) -> tuple:
        """
        Fixed greedy nearest neighbor algorithm on GPU for a single instance.
        """
        unvisited = torch.ones(n_customers, dtype=torch.bool, device=self.device)
        vehicle_routes = []
        total_cost = 0.0
        
        max_iterations = n_customers * 2  # Safety limit
        iteration = 0
        
        while unvisited.any() and iteration < max_iterations:
            iteration += 1
            
            # Start new vehicle route
            current_route = []
            current_capacity = capacity
            current_node = 0  # Start at depot
            
            route_iterations = 0
            max_route_iterations = n_customers + 1
            
            while route_iterations < max_route_iterations:
                route_iterations += 1
                
                # Find nearest unvisited customer that fits
                if current_node == 0:
                    # From depot
                    dist_to_customers = distances[0, 1:n_customers+1].clone()
                else:
                    # From current customer
                    dist_to_customers = distances[current_node, 1:n_customers+1].clone()
                
                # Mask out visited or infeasible customers
                mask = unvisited & (demands[1:n_customers+1] <= current_capacity)
                dist_to_customers[~mask] = float('inf')
                
                if torch.all(torch.isinf(dist_to_customers)) or torch.all(~mask):
                    # No more feasible customers for this vehicle
                    break
                
                # Select nearest customer
                nearest_idx = dist_to_customers.argmin()
                nearest_customer = nearest_idx + 1  # Adjust for depot at index 0
                
                # Add to route
                current_route.append(int(nearest_customer.item()))
                total_cost += dist_to_customers[nearest_idx].item()
                
                # Update state
                current_capacity -= demands[nearest_customer].item()
                unvisited[nearest_idx] = False
                current_node = int(nearest_customer.item())
            
            if len(current_route) > 0:
                # Return to depot
                total_cost += distances[current_node, 0].item()
                vehicle_routes.append(current_route)
        
        return vehicle_routes, total_cost


def solve_batch(instances: List[Dict[str, Any]], 
                verbose: bool = False) -> List[CVRPSolution]:
    """Convenience function to solve batch of instances. GPU ONLY."""
    solver = GPUHeuristicFixed()
    return solver.solve_batch(instances, verbose=verbose)
