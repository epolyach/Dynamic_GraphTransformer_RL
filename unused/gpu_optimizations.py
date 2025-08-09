import torch
import torch.nn.functional as F

def compute_batched_route_costs_gpu(routes, distance_matrices, device):
    """
    Efficiently compute route costs on GPU using batched operations
    
    Args:
        routes: List of routes (each route is a list of node indices)
        distance_matrices: List of distance matrices for each instance
        device: GPU device
    
    Returns:
        torch.Tensor: Batched costs computed on GPU
    """
    batch_size = len(routes)
    max_route_length = max(len(route) for route in routes)
    
    # Create padded route tensor
    route_tensor = torch.zeros(batch_size, max_route_length, dtype=torch.long, device=device)
    route_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i, route in enumerate(routes):
        route_len = len(route)
        route_tensor[i, :route_len] = torch.tensor(route, dtype=torch.long, device=device)
        route_lengths[i] = route_len
    
    # Stack distance matrices
    max_nodes = max(dm.shape[0] for dm in distance_matrices)
    dist_tensor = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32, device=device)
    
    for i, dm in enumerate(distance_matrices):
        n = dm.shape[0]
        dist_tensor[i, :n, :n] = torch.tensor(dm, dtype=torch.float32, device=device)
    
    # Compute costs using advanced indexing
    costs = torch.zeros(batch_size, device=device)
    
    for i in range(batch_size):
        route = route_tensor[i, :route_lengths[i]]
        if len(route) > 1:
            # Get consecutive pairs
            from_nodes = route[:-1]
            to_nodes = route[1:]
            # Sum distances
            costs[i] = dist_tensor[i, from_nodes, to_nodes].sum()
    
    return costs

def validate_routes_gpu(routes, n_customers_list):
    """
    Fast route validation that can be partially parallelized
    Only validates structure, not cost computation
    """
    valid_routes = []
    for i, (route, n_customers) in enumerate(zip(routes, n_customers_list)):
        # Quick validation - just check basic structure
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            valid_routes.append(False)
        else:
            # Check if all customers visited (simplified)
            customers_in_route = set(route[1:-1])  # Exclude depot visits
            expected_customers = set(range(1, n_customers + 1))
            valid_routes.append(customers_in_route == expected_customers)
    
    return valid_routes

def create_gpu_optimized_batches(instances, batch_size, device):
    """
    Pre-create optimized batches with tensors already on GPU
    """
    batches = []
    for i in range(0, len(instances), batch_size):
        batch_instances = instances[i:i + batch_size]
        
        # Pre-compute batch tensors
        max_nodes = max(len(inst['coords']) for inst in batch_instances)
        batch_coords = torch.zeros(len(batch_instances), max_nodes, 2, device=device)
        batch_demands = torch.zeros(len(batch_instances), max_nodes, device=device)
        batch_capacities = torch.zeros(len(batch_instances), device=device)
        
        for j, inst in enumerate(batch_instances):
            n_nodes = len(inst['coords'])
            batch_coords[j, :n_nodes] = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            batch_demands[j, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            batch_capacities[j] = inst['capacity']
        
        batches.append({
            'instances': batch_instances,
            'coords': batch_coords,
            'demands': batch_demands,
            'capacities': batch_capacities,
            'max_nodes': max_nodes
        })
    
    return batches
