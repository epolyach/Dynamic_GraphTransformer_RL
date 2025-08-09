import torch

def update_mask(demand, capacity, selected, mask, i):
    """
    Simple CVRP masking: visit all customers, use depot to reset capacity when needed
    """
    # Mark selected node as visited
    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)

    # Apply capacity constraints - mask customers that exceed remaining capacity  
    remaining_capacity = capacity.squeeze(-1).unsqueeze(1)  # [batch_size, 1]
    customer_demands = demand[:, 1:]  # [batch_size, n_customers]
    capacity_exceeded = (customer_demands > remaining_capacity).float()
    
    # Mask customers that exceed capacity
    mask1[:, 1:] = torch.clamp(mask1[:, 1:] + capacity_exceeded, 0, 1)
    
    # Count customers visited and remaining
    customers_visited = mask1[:, 1:].sum(dim=1)  # [batch_size]
    total_customers = demand.size(1) - 1  # Exclude depot
    all_customers_visited = (customers_visited >= total_customers)
    
    # Check if any unvisited customers can be served with current capacity
    unvisited_mask = (1 - mask1[:, 1:])  # 1 for unvisited customers
    servable_mask = (1 - capacity_exceeded)  # 1 for customers within capacity
    unvisited_and_servable = (unvisited_mask * servable_mask).sum(dim=1) > 0
    
    # Depot masking rule:
    # - Allow depot if: all customers visited OR no more customers can be served
    # - Block depot if: customers remain and some can be served
    should_block_depot = ~all_customers_visited & unvisited_and_servable
    
    # Apply depot blocking (but never block depot on first step)
    if i > 0:
        mask1[:, 0] = mask1[:, 0] + should_block_depot.float()
    
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
