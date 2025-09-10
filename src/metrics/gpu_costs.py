"""GPU-optimized cost computation functions for CVRP."""
import torch
from typing import List, Optional, Union
import numpy as np


def compute_route_cost_gpu(route: Union[List[int], torch.Tensor], 
                           distances: Union[torch.Tensor, np.ndarray]) -> Union[float, torch.Tensor]:
    """
    Compute route cost on GPU without data transfer.
    
    Args:
        route: List of node indices or tensor of indices
        distances: Distance matrix (torch.Tensor on GPU or numpy array)
    
    Returns:
        Route cost (float if single route, tensor if batch)
    """
    # Handle single route case
    if isinstance(route, list):
        if len(route) <= 1:
            return 0.0
        
        # Convert to tensor if needed
        if isinstance(distances, np.ndarray):
            # This is for CPU fallback - shouldn't happen in GPU training
            cost = 0.0
            for i in range(len(route) - 1):
                cost += float(distances[route[i], route[i + 1]])
            return cost
        else:
            # GPU path - keep on GPU
            route_tensor = torch.tensor(route, dtype=torch.long, device=distances.device)
            cost = torch.tensor(0.0, device=distances.device)
            for i in range(len(route) - 1):
                cost += distances[route_tensor[i], route_tensor[i + 1]]
            return cost.item()
    
    # Handle batch case (tensor input)
    else:
        # Assuming route is a 2D tensor [batch_size, seq_len]
        batch_size, seq_len = route.shape
        
        if seq_len <= 1:
            return torch.zeros(batch_size, device=route.device)
        
        # Gather distances for consecutive pairs
        idx1 = route[:, :-1]  # All but last
        idx2 = route[:, 1:]   # All but first
        
        # Use advanced indexing to get distances
        batch_idx = torch.arange(batch_size, device=route.device).unsqueeze(1).expand(-1, seq_len - 1)
        costs = distances[batch_idx, idx1, idx2].sum(dim=1)
        
        return costs


def compute_batch_costs_gpu(routes_list: List[List[int]], 
                           distances_batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute costs for a batch of routes efficiently on GPU.
    
    Args:
        routes_list: List of routes (each route is a list of node indices)
        distances_batch: List of distance matrices (tensors on GPU)
    
    Returns:
        Tensor of costs for each route
    """
    device = distances_batch[0].device if distances_batch else 'cpu'
    costs = []
    
    for route, distances in zip(routes_list, distances_batch):
        if len(route) <= 1:
            costs.append(0.0)
        else:
            # Create tensor for route indices
            route_tensor = torch.tensor(route, dtype=torch.long, device=device)
            
            # Compute cost using tensor operations
            cost = torch.sum(distances[route_tensor[:-1], route_tensor[1:]])
            costs.append(cost)
    
    # Stack all costs into a single tensor
    return torch.stack(costs) if costs else torch.tensor([], device=device)


def compute_route_cost_vectorized(routes: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized route cost computation for batched routes.
    
    Args:
        routes: Tensor of shape [batch_size, max_seq_len] with padding
        distances: Tensor of shape [batch_size, num_nodes, num_nodes]
    
    Returns:
        Tensor of shape [batch_size] with total costs
    """
    batch_size, seq_len = routes.shape
    
    # Create mask for valid transitions (non-padding)
    # Assuming -1 or a specific value indicates padding
    valid_mask = (routes[:, 1:] >= 0) & (routes[:, :-1] >= 0)
    
    # Get indices for gathering
    batch_idx = torch.arange(batch_size, device=routes.device).unsqueeze(1).expand(-1, seq_len - 1)
    from_idx = routes[:, :-1]
    to_idx = routes[:, 1:]
    
    # Gather distances
    # Handle 3D distances (batch_size, num_nodes, num_nodes)
    if distances.dim() == 3:
        edge_costs = distances[batch_idx, from_idx, to_idx]
    else:
        # 2D distances (shared across batch)
        edge_costs = distances[from_idx, to_idx]
    
    # Apply mask and sum
    masked_costs = edge_costs * valid_mask.float()
    total_costs = masked_costs.sum(dim=1)
    
    return total_costs
