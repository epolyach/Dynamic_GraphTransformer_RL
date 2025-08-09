"""
Route validation utilities for training
"""

def validate_training_route(actions, demands, capacity, batch_idx=0, step_info=""):
    """
    Strict route validation during training - raises error if invalid.
    
    Args:
        actions: torch.Tensor [B, T] or [B, T, 1] - generated actions
        demands: torch.Tensor [B, N] - node demands  
        capacity: float - vehicle capacity
        batch_idx: int - batch index for error reporting
        step_info: str - additional context for error reporting
    """
    # Handle action tensor shapes
    if actions.dim() == 3 and actions.size(-1) == 1:
        actions = actions.squeeze(-1)  # [B, T]
    
    batch_size = actions.size(0)
    n_nodes = demands.size(1)
    
    for b in range(batch_size):
        route = actions[b].detach().cpu().tolist()
        demands_np = demands[b].detach().cpu().numpy()
        
        # Remove padding/invalid indices
        route = [int(idx) for idx in route if 0 <= int(idx) < n_nodes]
        
        if not route:
            raise ValueError(f"TRAINING ERROR {step_info}: Empty route for batch {b}")
        
        # Check all customers served exactly once
        customers_in_route = [node for node in route if node != 0]
        expected_customers = set(range(1, n_nodes))
        actual_customers = set(customers_in_route)
        
        missing = expected_customers - actual_customers
        if missing:
            raise ValueError(f"TRAINING ERROR {step_info}: Missing customers {missing} in batch {b}. Route: {route}")
        
        duplicates = [x for x in customers_in_route if customers_in_route.count(x) > 1]
        if duplicates:
            raise ValueError(f"TRAINING ERROR {step_info}: Duplicate customers {set(duplicates)} in batch {b}. Route: {route}")
        
        # Capacity validation - split into tours
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
        
        # Check capacity constraints for each tour
        for tour_idx, tour in enumerate(tours):
            tour_demand = int(sum(int(demands_np[customer]) for customer in tour))
            if tour_demand > int(capacity):
                violation = tour_demand - int(capacity)
                raise ValueError(f"TRAINING ERROR {step_info}: Capacity violation in batch {b}, tour {tour_idx+1}: "
                               f"demand {tour_demand} > capacity {int(capacity)} (violation: {violation}). "
                               f"Tour: {tour}, Full route: {route}")
        
        # Success - log for monitoring
        if batch_idx == 0:  # Only log for first batch to avoid spam
            print(f"[validation] {step_info} batch {b}: ✓ Valid route with {len(tours)} tours")
