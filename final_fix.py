#!/usr/bin/env python3
"""Final fix - ensure ALL cost computations use CPU"""

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    lines = f.readlines()

# Replace all compute_route_cost_gpu with compute_route_cost
new_lines = []
for i, line in enumerate(lines):
    # Replace GPU cost computation in validation
    if 'compute_route_cost_gpu(route, distances)' in line:
        # Get the indent
        indent = len(line) - len(line.lstrip())
        # Add CPU conversion before
        new_lines.append(' ' * indent + 'distances_cpu = distances.cpu().numpy() if isinstance(distances, torch.Tensor) else distances\n')
        # Replace the function call
        new_line = line.replace('compute_route_cost_gpu(route, distances)', 'compute_route_cost(route, distances_cpu)')
        new_lines.append(new_line)
    else:
        new_lines.append(line)

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed ALL cost computations to use CPU version")
