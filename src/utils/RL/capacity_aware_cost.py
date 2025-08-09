import torch
import numpy as np
from typing import Tuple

def capacity_aware_euclidean_cost(static, actions, batch, capacity_penalty: float = 10.0):
    """
    Compute route cost with heavy penalty for capacity violations.
    
    Args:
        static: torch.Tensor, coordinates of nodes
        actions: torch.Tensor, node indices of the route
        batch: PyG Data object with demands and capacity
        capacity_penalty: Penalty multiplier for capacity violations
        
    Returns:
        total_cost: torch.Tensor, cost including capacity violation penalties
    """
    # Get basic euclidean cost first
    from .euclidean_cost import euclidean_cost
    base_cost = euclidean_cost(static, actions, batch)
    
    # Extract batch info
    if hasattr(batch, 'num_graphs') and batch.num_graphs is not None:
        batch_size = int(batch.num_graphs)
    elif hasattr(batch, 'batch') and batch.batch is not None:
        batch_size = int(batch.batch.max().item() + 1)
    else:
        batch_size = 1
    
    total_nodes = batch.x.size(0)
    num_nodes = int(total_nodes // batch_size)
    
    # Get demands and capacity
    demands = batch.demand.view(batch_size, num_nodes)  # [B, N]
    if hasattr(batch, 'capacity'):
        capacity = batch.capacity[0].item() if batch.capacity.dim() > 0 else batch.capacity.item()
    else:
        capacity = 3.0  # Default capacity
    
    # Normalize actions to [B, S]
    if actions.dim() == 3 and actions.size(-1) == 1:
        actions = actions.squeeze(-1)
    
    # Calculate capacity violations for each batch
    capacity_violations = torch.zeros(batch_size, device=actions.device)
    
    for b in range(batch_size):
        route = actions[b].tolist()
        demand_vals = demands[b].detach().cpu().numpy()
        
        # Split route into tours
        tours = []
        current_tour = []
        
        for node in route:
            if node == 0 and current_tour:  # End of tour
                tours.append(current_tour)
                current_tour = []
            elif node != 0:  # Customer
                current_tour.append(node)
        
        if current_tour:  # Handle incomplete tour
            tours.append(current_tour)
        
        # Calculate violations for each tour
        violation_penalty = 0.0
        for tour in tours:
            tour_demand = sum(demand_vals[customer] for customer in tour if customer < len(demand_vals))
            if tour_demand > capacity:
                violation_penalty += (tour_demand - capacity) ** 2  # Quadratic penalty
        
        capacity_violations[b] = violation_penalty
    
    # Add capacity penalty to base cost
    total_cost = base_cost + capacity_penalty * capacity_violations
    
    return total_cost

def validate_route_batch(actions, demands, capacity, batch_size, num_nodes):
    """
    Validate a batch of routes for capacity constraints.
    
    Returns:
        violations: torch.Tensor [B], number of capacity violations per route
        total_violation_amount: torch.Tensor [B], total excess demand per route
    """
    violations = torch.zeros(batch_size, device=actions.device)
    total_violation_amount = torch.zeros(batch_size, device=actions.device)
    
    for b in range(batch_size):
        route = actions[b].tolist() if hasattr(actions[b], 'tolist') else actions[b]
        demand_vals = demands[b].detach().cpu().numpy()
        
        # Parse into tours
        tours = []
        current_tour = []
        
        for node in route:
            if node == 0 and current_tour:
                tours.append(current_tour)
                current_tour = []
            elif node != 0:
                current_tour.append(node)
        
        if current_tour:
            tours.append(current_tour)
        
        # Check each tour
        tour_violations = 0
        total_excess = 0.0
        
        for tour in tours:
            tour_demand = sum(demand_vals[customer] for customer in tour if customer < len(demand_vals))
            if tour_demand > capacity:
                tour_violations += 1
                total_excess += tour_demand - capacity
        
        violations[b] = tour_violations
        total_violation_amount[b] = total_excess
    
    return violations, total_violation_amount
