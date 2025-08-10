import torch

def update_mask(demand, capacity, selected, mask, i):
    """
    Final fix for CVRP masking logic
    """
    # Mark selected node as visited  
    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)

    # Apply capacity constraints
    remaining_capacity = capacity.squeeze(-1).unsqueeze(1)
    customer_demands = demand[:, 1:]
    capacity_exceeded = (customer_demands > remaining_capacity).float()
    mask1[:, 1:] = torch.clamp(mask1[:, 1:] + capacity_exceeded, 0, 1)
    
    # Check completion status
    customers_visited = mask1[:, 1:].sum(dim=1)
    total_customers = demand.size(1) - 1
    all_customers_visited = (customers_visited >= total_customers)
    
    # Check if any unvisited customers can be served
    unvisited_mask = (1 - mask1[:, 1:])  # 1 = unvisited customer
    servable_mask = (1 - capacity_exceeded)  # 1 = within capacity
    unvisited_servable = (unvisited_mask * servable_mask).sum(dim=1) > 0
    
    # Depot masking rules (except first step):
    # 1. If all customers visited -> ALLOW depot (route complete)
    # 2. If unvisited customers can be served -> BLOCK depot (serve them first)  
    # 3. If unvisited customers exist but none servable -> ALLOW depot (need capacity reset)
    
    if i > 0:
        block_depot = (~all_customers_visited) & unvisited_servable
        mask1[:, 0] = mask1[:, 0] + block_depot.float()
    
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
