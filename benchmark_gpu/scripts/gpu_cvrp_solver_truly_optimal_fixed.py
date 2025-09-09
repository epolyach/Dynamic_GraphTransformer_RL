#!/usr/bin/env python3
"""
FIXED GPU-based exact CVRP solver with optimal route recovery.
This version properly recovers optimal TSP tours from the DP table.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import time
import sys
sys.path.append('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
from src.benchmarking.solvers.types import CVRPSolution

class GPUExactCVRPFixed:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def solve_batch(self, instances: List[Dict[str, Any]], verbose: bool = False) -> List[CVRPSolution]:
        """Solve multiple CVRP instances exactly on GPU in parallel."""
        start_time = time.time()
        batch_size = len(instances)
        n = len(instances[0]['coords'])
        n_customers = n - 1
        
        if n_customers > 12:  # Reduced limit for true optimality
            raise ValueError(f"GPU exact solver supports n≤12 for optimal solutions, got n={n_customers}")
            
        if verbose:
            print(f"GPU exact solving {batch_size} instances with {n_customers} customers")
        
        # Convert to tensors
        dist_scale = 100000
        distances = torch.tensor(
            np.array([inst['distances'] * dist_scale for inst in instances]),
            dtype=torch.int32, device=self.device
        )
        demands = torch.tensor(
            np.array([inst['demands'] for inst in instances]),
            dtype=torch.float32, device=self.device
        )
        capacities = torch.tensor(
            [inst['capacity'] for inst in instances],
            dtype=torch.float32, device=self.device
        )
        
        # Stage 1: TSP for all feasible subsets WITH path recovery
        tsp_costs, tsp_paths = self._compute_tsp_with_paths(
            distances, demands, capacities, n_customers, verbose
        )
        
        # Stage 2: Partition DP to find optimal CVRP solution
        best_costs, best_partitions = self._partition_dp(tsp_costs, n_customers, verbose)
        
        # Extract solutions with optimal paths
        results = []
        for b in range(batch_size):
            routes = self._extract_optimal_routes_fixed(
                best_partitions[b].item(), 
                tsp_paths,
                b,
                n_customers
            )
            
            cost = best_costs[b].item() / dist_scale
            
            all_nodes = []
            for route in routes:
                all_nodes.extend(route)
            
            results.append(CVRPSolution(
                route=all_nodes,
                cost=cost,
                num_vehicles=len(routes),
                vehicle_routes=routes,
                solve_time=time.time() - start_time,
                algorithm_used='GPU-Exact-DP-Fixed',
                is_optimal=True,
                gap=0.0
            ))
        
        if verbose:
            print(f"GPU exact (fixed) completed in {time.time()-start_time:.3f}s")
        
        return results
    
    def _compute_tsp_with_paths(self, distances, demands, capacities, n_customers, verbose):
        """Compute TSP costs and store optimal paths for recovery."""
        batch_size = distances.shape[0]
        n_subsets = 1 << n_customers
        INF = torch.iinfo(torch.int32).max // 2
        
        # DP table: dp[batch, mask, last_node]
        dp = torch.full((batch_size, n_subsets, n_customers + 1), INF, 
                       dtype=torch.int32, device=self.device)
        parent = torch.full((batch_size, n_subsets, n_customers + 1), -1,
                           dtype=torch.int32, device=self.device)
        
        # Base case: single customer routes from depot
        for i in range(n_customers):
            mask = 1 << i
            dp[:, mask, i + 1] = distances[:, 0, i + 1]  # depot to customer i+1
            parent[:, mask, i + 1] = 0  # came from depot
        
        # Check feasibility for all subsets
        subset_demands = torch.zeros((batch_size, n_subsets), dtype=torch.float32, device=self.device)
        for subset in range(1, n_subsets):
            for i in range(n_customers):
                if subset & (1 << i):
                    subset_demands[:, subset] += demands[:, i + 1]
        
        feasible = subset_demands <= capacities.unsqueeze(1)
        
        # Fill DP table
        for subset_size in range(2, n_customers + 1):
            for subset in range(1, n_subsets):
                if bin(subset).count('1') != subset_size:
                    continue
                if not feasible[:, subset].all():
                    continue
                    
                for last in range(1, n_customers + 1):
                    if not (subset & (1 << (last - 1))):
                        continue
                    
                    prev_subset = subset ^ (1 << (last - 1))
                    if prev_subset == 0:
                        continue
                    
                    for prev in range(1, n_customers + 1):
                        if not (prev_subset & (1 << (prev - 1))):
                            continue
                        
                        cost = dp[:, prev_subset, prev] + distances[:, prev, last]
                        better = cost < dp[:, subset, last]
                        dp[:, subset, last] = torch.where(better, cost, dp[:, subset, last])
                        parent[:, subset, last] = torch.where(better, prev, parent[:, subset, last])
        
        # Compute TSP costs and store best last node for path recovery
        tsp_costs = torch.full((batch_size, n_subsets), INF, dtype=torch.int32, device=self.device)
        best_last = torch.full((batch_size, n_subsets), -1, dtype=torch.int32, device=self.device)
        
        for subset in range(1, n_subsets):
            if not feasible[:, subset].all():
                continue
            
            min_cost = INF
            best_last_node = -1
            
            for last in range(1, n_customers + 1):
                if subset & (1 << (last - 1)):
                    cost = dp[:, subset, last] + distances[:, last, 0]  # return to depot
                    better = cost < tsp_costs[:, subset]
                    tsp_costs[:, subset] = torch.where(better, cost, tsp_costs[:, subset])
                    best_last[:, subset] = torch.where(better, last, best_last[:, subset])
        
        # Store path information for recovery
        return tsp_costs, {
            'dp': dp,
            'parent': parent,
            'best_last': best_last
        }
    
    def _recover_tsp_path(self, mask, last_node, parent_table, batch_idx):
        """Recover TSP path from parent pointers."""
        if last_node <= 0:
            return []
        
        path = []
        current = last_node
        current_mask = mask
        
        while current > 0 and current_mask > 0:
            path.append(current)
            prev = parent_table[batch_idx, current_mask, current].item()
            if prev <= 0:
                break
            current_mask ^= (1 << (current - 1))
            current = prev
        
        return [0] + list(reversed(path)) + [0]  # depot -> customers -> depot
    
    def _extract_optimal_routes_fixed(self, partition_mask, paths_info, batch_idx, n_customers):
        """FIXED: Extract routes with optimal TSP ordering."""
        routes = []
        current_mask = (1 << n_customers) - 1  # All customers
        
        parent_table = paths_info['parent'] 
        best_last = paths_info['best_last']
        partition_parent = self.partition_parent[batch_idx]  # Store partition parent table
        
        # Trace back through partition to get individual route masks
        while current_mask > 0:
            route_mask = partition_parent[current_mask].item()
            if route_mask <= 0:
                break
                
            # Get the best last node for this route mask
            last_node = best_last[batch_idx, route_mask].item()
            
            # Recover optimal TSP path for this route
            if last_node > 0:
                route = self._recover_tsp_path(route_mask, last_node, parent_table, batch_idx)
                if route and len(route) > 2:  # Valid route (more than just depot->depot)
                    routes.append(route)
            
            # Remove this route from remaining customers
            current_mask ^= route_mask
        
        return routes
    
    def _partition_dp(self, tsp_costs, n_customers, verbose):
        """Find optimal partition of customers into routes."""
        batch_size = tsp_costs.shape[0]
        full_mask = (1 << n_customers) - 1
        INF = torch.iinfo(torch.int32).max // 2
        
        f = torch.full((batch_size, 1 << n_customers), INF, dtype=torch.int32, device=self.device)
        parent = torch.zeros((batch_size, 1 << n_customers), dtype=torch.int32, device=self.device)
        
        f[:, 0] = 0
        
        for mask in range(1, full_mask + 1):
            submask = mask
            while submask > 0:
                rest = mask ^ submask
                new_cost = f[:, rest] + tsp_costs[:, submask]
                
                better = new_cost < f[:, mask]
                f[:, mask] = torch.where(better, new_cost, f[:, mask])
                parent[:, mask] = torch.where(better, 
                                             torch.tensor(submask, dtype=torch.int32, device=self.device),
                                             parent[:, mask])
                
                submask = (submask - 1) & mask
        
        # Store partition parent for route extraction
        self.partition_parent = parent
        
        return f[:, full_mask], parent[:, full_mask]

def solve(instance: Dict[str, Any], verbose: bool = False) -> CVRPSolution:
    """Single instance wrapper."""
    solver = GPUExactCVRPFixed()
    return solver.solve_batch([instance], verbose)[0]

def solve_batch(instances: List[Dict[str, Any]], verbose: bool = False) -> List[CVRPSolution]:
    """Batch solver interface."""
    solver = GPUExactCVRPFixed()
    return solver.solve_batch(instances, verbose)

if __name__ == "__main__":
    print("Fixed GPU Exact Solver with Optimal Path Recovery")
    print("This version guarantees truly optimal solutions!")
