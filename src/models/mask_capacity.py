import torch

def update_mask(demand, capacity, selected, mask, i):
    """
    Update mask for CVRP with proper customer completion enforcement
    """
    go_depot = selected.squeeze(-1).eq(0)
    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)

    # If not going to depot, prevent depot visit temporarily
    if (~go_depot).any():
        mask1[(~go_depot).nonzero(),0] = 0

    # CRITICAL FIX: Only allow termination if ALL customers are visited
    # Count customers visited (exclude depot at index 0)
    customers_visited = mask1[:, 1:].sum(dim=1)  # [batch_size]
    total_customers = demand.size(1) - 1  # Exclude depot
    all_customers_visited = (customers_visited >= total_customers)
    
    # If not all customers visited, force depot to be masked (prevent early termination)
    depot_mask = (~all_customers_visited).float()
    mask1[:, 0] = mask1[:, 0] + depot_mask
    
    # Apply capacity constraints: mask customers that exceed remaining capacity
    # But NEVER mask ALL remaining customers - always keep at least depot available
    remaining_capacity = capacity.squeeze(-1).unsqueeze(1)  # [batch_size, 1]
    customer_demands = demand[:, 1:]  # [batch_size, n_customers]
    
    # Capacity mask for customers only
    capacity_exceeded = (customer_demands > remaining_capacity).float()
    
    # Apply capacity mask to customers (indices 1:)
    mask1[:, 1:] = torch.clamp(mask1[:, 1:] + capacity_exceeded, 0, 1)
    
    # Final mask
    mask = mask1.clone()
    
    return mask.detach(), mask1.detach()

def update_state(demand, dynamic_capacity, selected, base_capacity):
    """Update vehicle capacity after serving a customer or returning to depot"""
    depot = selected.squeeze(-1).eq(0)
    current_demand = torch.gather(demand, 1, selected)
    dynamic_capacity = dynamic_capacity - current_demand
    
    # Reset capacity when visiting depot
    if depot.any():
        dynamic_capacity[depot.nonzero().squeeze()] = base_capacity

    return dynamic_capacity.detach()
