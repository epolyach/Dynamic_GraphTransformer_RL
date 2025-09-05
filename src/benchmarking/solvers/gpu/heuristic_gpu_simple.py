#!/usr/bin/env python3
"""
GPU-accelerated heuristic CVRP solver using PyTorch.
Implements a parallel greedy nearest neighbor algorithm on GPU.
NO CPU FALLBACK - requires CUDA GPU.
"""

import torch
import numpy as np
from typing import Dict, Any, List
from src.benchmarking.solvers.types import CVRPSolution
import time


class GPUHeuristicSimple:
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
        
        solutions = []
        for b in range(batch_size):
            # Solve each instance with greedy nearest neighbor
            route_list, cost = self._solve_greedy(
                distances[b], demands[b], capacities[b], n_customers
            )
            
            # Convert to CVRPSolution
            route = [0]
            for vehicle_route in route_list:
                route.extend(vehicle_route)
                route.append(0)
            
            solution = CVRPSolution(
                route=route,
                vehicle_routes=route_list,
                cost=cost,
                num_vehicles=len(route_list),
                solve_time=time.time() - start_time,
                algorithm_used="GPU_Greedy",
                is_optimal=False
            )
            solutions.append(solution)
        
        if verbose:
            avg_cost = np.mean([s.cost for s in solutions])
            print(f"Completed in {time.time() - start_time:.2f}s, avg cost: {avg_cost:.4f}")
        
        return solutions
    
    def _solve_greedy(self, distances: torch.Tensor, demands: torch.Tensor,
                     capacity: float, n_customers: int) -> tuple:
        """
        Simple greedy nearest neighbor algorithm on GPU.
        All operations performed on CUDA tensors.
        """
        unvisited = torch.ones(n_customers, dtype=torch.bool, device=self.device)
        vehicle_routes = []
        total_cost = 0.0
        
        while unvisited.any():
            # Start new vehicle route
            current_route = []
            current_capacity = capacity
            current_node = 0  # Start at depot
            
            while True:
                # Find nearest unvisited customer that fits
                dist_to_customers = distances[current_node, 1:n_customers+1].clone()
                
                # Mask out visited or infeasible customers
                mask = unvisited & (demands[1:n_customers+1] <= current_capacity)
                dist_to_customers[~mask] = float('inf')
                
                if torch.all(torch.isinf(dist_to_customers)):
                    # No more feasible customers
                    break
                
                # Select nearest customer
                nearest_idx = dist_to_customers.argmin()
                nearest_customer = nearest_idx + 1  # Adjust for depot at index 0
                
                # Add to route
                current_route.append(int(nearest_customer.item()))
                total_cost += distances[current_node, nearest_customer].item()
                
                # Update state
                current_capacity -= demands[nearest_customer].item()
                unvisited[nearest_idx] = False
                current_node = nearest_customer
            
            if len(current_route) > 0:
                # Return to depot
                total_cost += distances[current_node, 0].item()
                vehicle_routes.append(current_route)
        
        return vehicle_routes, total_cost


def solve_batch(instances: List[Dict[str, Any]], 
                verbose: bool = False) -> List[CVRPSolution]:
    """Convenience function to solve batch of instances. GPU ONLY."""
    solver = GPUHeuristicSimple()
    return solver.solve_batch(instances, verbose=verbose)
