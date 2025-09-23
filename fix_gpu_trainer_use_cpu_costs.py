#!/usr/bin/env python3
"""Replace GPU cost computation with CPU version in GPU trainer"""

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# Replace the import
content = content.replace(
    'from src.metrics.gpu_costs import compute_route_cost_gpu',
    '# from src.metrics.gpu_costs import compute_route_cost_gpu  # Too slow for small problems'
)

# Replace the GPU cost computations with CPU versions
# Pattern 1: Training loop cost computation
old_pattern1 = '''                for b in range(len(instances)):
                    distances = instances[b]["distances"]
                    route = routes[b]
                    # Use GPU cost computation but simpler approach
                    rc = compute_route_cost_gpu(route, distances)
                    # Convert to tensor on GPU if needed
                    if not isinstance(rc, torch.Tensor):
                        rc = torch.tensor(rc, device=gpu_manager.device, dtype=torch.float32)
                    rcosts.append(rc)'''

new_pattern1 = '''                for b in range(len(instances)):
                    distances = instances[b]["distances"]
                    route = routes[b]
                    # Use CPU cost computation - much faster for small problems
                    distances_cpu = distances.cpu().numpy() if isinstance(distances, torch.Tensor) else distances
                    rc = compute_route_cost(route, distances_cpu)
                    # Convert result to GPU tensor
                    rc = torch.tensor(rc, device=gpu_manager.device, dtype=torch.float32)
                    rcosts.append(rc)'''

content = content.replace(old_pattern1, new_pattern1)

# Pattern 2: Validation cost computation (geometric mean)
old_pattern2 = '''                            rc = compute_route_cost_gpu(route, distances)
                            if not isinstance(rc, torch.Tensor):
                                rc = torch.tensor(rc, device=gpu_manager.device, dtype=torch.float32)'''

new_pattern2 = '''                            distances_cpu = distances.cpu().numpy() if isinstance(distances, torch.Tensor) else distances
                            rc = compute_route_cost(route, distances_cpu)
                            rc = torch.tensor(rc, device=gpu_manager.device, dtype=torch.float32)'''

content = content.replace(old_pattern2, new_pattern2)

# Pattern 3: Validation cost computation (arithmetic mean)
old_pattern3 = '''                            rc = compute_route_cost_gpu(route, distances)
                            if not isinstance(rc, torch.Tensor):
                                rc = torch.tensor(rc, device=gpu_manager.device, dtype=torch.float32)
                            n_customers = (len(val_instances[b]["coords"]) - 1)
                            val_cpcs.append((rc / float(n_customers)).item())'''

new_pattern3 = '''                            distances_cpu = distances.cpu().numpy() if isinstance(distances, torch.Tensor) else distances
                            rc = compute_route_cost(route, distances_cpu)
                            n_customers = (len(val_instances[b]["coords"]) - 1)
                            val_cpcs.append(rc / float(n_customers))'''

content = content.replace(old_pattern3, new_pattern3)

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("✅ Fixed GPU trainer to use CPU cost computation")
print("   This should dramatically improve performance by avoiding GPU overhead")
