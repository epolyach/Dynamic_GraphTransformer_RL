#!/usr/bin/env python3
"""
Implement GPU-based cost computation for better performance
"""

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# Add GPU cost computation function after the move_to_gpu function
gpu_cost_function = '''
def compute_costs_gpu(routes, distances_gpu, device):
    """
    Compute route costs on GPU using vectorized operations.
    
    Args:
        routes: List of routes (Python lists)
        distances_gpu: Distance matrices already on GPU (batch_size, n, n)
        device: GPU device
    
    Returns:
        Tensor of costs on GPU
    """
    batch_size = len(routes)
    max_len = max(len(r) for r in routes)
    
    # Pad routes and convert to tensor
    routes_padded = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
    for i, route in enumerate(routes):
        routes_padded[i, :len(route)] = torch.tensor(route, dtype=torch.long, device=device)
    
    # Compute edge costs
    from_idx = routes_padded[:, :-1]
    to_idx = routes_padded[:, 1:]
    valid_mask = (from_idx >= 0) & (to_idx >= 0)
    
    # Batch indexing
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_len-1)
    
    # Get edge costs
    edge_costs = torch.zeros_like(from_idx, dtype=torch.float32)
    valid_indices = valid_mask.nonzero(as_tuple=False)
    
    if len(valid_indices) > 0:
        b_idx = batch_idx[valid_mask]
        f_idx = from_idx[valid_mask]
        t_idx = to_idx[valid_mask]
        edge_costs[valid_mask] = distances_gpu[b_idx, f_idx, t_idx]
    
    # Sum to get total costs
    return edge_costs.sum(dim=1)

'''

# Find where to insert the function (after move_to_gpu_except_distances)
import_marker = "def move_to_gpu_except_distances"
if import_marker in content:
    # Find the end of move_to_gpu_except_distances function
    start_idx = content.find(import_marker)
    # Find the next function definition
    next_func = content.find("\ndef ", start_idx + 1)
    if next_func == -1:
        next_func = content.find("\nclass ", start_idx + 1)
    
    # Insert the GPU cost function
    content = content[:next_func] + "\n" + gpu_cost_function + content[next_func:]
else:
    # If move_to_gpu function doesn't exist, add both
    import_section_end = content.find("logger = logging.getLogger")
    if import_section_end != -1:
        content = content[:import_section_end] + gpu_cost_function + "\n" + content[import_section_end:]

# Now modify the cost computation in the training loop
# Replace CPU cost computation with GPU computation
old_pattern = '''                # Compute per-instance route costs and CPC (log or arithmetic)
                rcosts = []  # actual route costs per instance
                cpc_vals = []  # arithmetic CPC values
                cpc_logs = []  # log-CPC values for geometric mean
                for b in range(len(instances)):
                    distances = instances[b]["distances"]
                    route = routes[b]
                    # Use CPU cost computation to match CPU trainer exactly
                    distances_cpu = distances.cpu().numpy() if isinstance(distances, torch.Tensor) else distances
                    rc = compute_route_cost(route, distances_cpu)
                    # Convert to tensor on GPU
                    if not isinstance(rc, torch.Tensor):
                        rc = torch.tensor(rc, device=gpu_manager.device, dtype=torch.float32)
                    rcosts.append(rc)
                    n_customers = (len(instances[b]["coords"]) - 1)
                    if use_geometric_mean:
                        cpc_logs.append(torch.log(rc + 1e-10) - torch.log(torch.tensor(float(n_customers), device=gpu_manager.device)))
                    else:
                        cpc_vals.append(rc / float(n_customers))'''

new_pattern = '''                # Compute per-instance route costs using GPU
                # First ensure distances are on GPU
                distances_list = []
                for inst in instances:
                    dist = inst["distances"]
                    if isinstance(dist, np.ndarray):
                        dist = torch.tensor(dist, dtype=torch.float32, device=gpu_manager.device)
                    elif isinstance(dist, torch.Tensor) and dist.device != gpu_manager.device:
                        dist = dist.to(gpu_manager.device)
                    distances_list.append(dist)
                
                distances_gpu = torch.stack(distances_list)
                
                # Compute costs on GPU
                rcosts = compute_costs_gpu(routes, distances_gpu, gpu_manager.device)
                
                # Compute CPC values
                n_customers_tensor = torch.tensor([len(inst["coords"]) - 1 for inst in instances],
                                                 device=gpu_manager.device, dtype=torch.float32)
                
                if use_geometric_mean:
                    cpc_logs = torch.log(rcosts + 1e-10) - torch.log(n_customers_tensor)
                    cpc_vals = []
                else:
                    cpc_vals = rcosts / n_customers_tensor
                    cpc_logs = []'''

content = content.replace(old_pattern, new_pattern)

# Fix the aggregation part
old_agg = '''                # Aggregated CPC for this batch (to track train_cost_epoch)
                if use_geometric_mean:
                    batch_cost = torch.exp(torch.stack(cpc_logs).mean())
                else:
                    batch_cost = torch.stack(cpc_vals).mean()
                
                # Build actual costs tensor for RL (match CPU: use actual costs, not CPC)
                costs_tensor = torch.stack(rcosts).to(dtype=torch.float32)'''

new_agg = '''                # Aggregated CPC for this batch (to track train_cost_epoch)
                if use_geometric_mean:
                    batch_cost = torch.exp(cpc_logs.mean())
                else:
                    batch_cost = cpc_vals.mean()
                
                # Build actual costs tensor for RL (use actual costs, not CPC)
                costs_tensor = rcosts.to(dtype=torch.float32)'''

content = content.replace(old_agg, new_agg)

# Save the modified trainer
with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("Implemented GPU-based cost computation")
print("Changes made:")
print("1. Added compute_costs_gpu() function for vectorized GPU computation")
print("2. Modified training loop to use GPU for cost calculations")
print("3. Distances now stay on GPU throughout")
