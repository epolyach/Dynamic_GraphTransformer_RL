#!/usr/bin/env python3
"""Revert to the original CPU-based cost computation which was faster"""

# Read the current file
with open('training_gpu/lib/advanced_trainer_gpu.py.broken', 'r') as f:
    lines = f.readlines()

# Find and replace the vectorized GPU computation section (lines ~375-417)
# with the original CPU-based computation

new_lines = []
i = 0
while i < len(lines):
    # Look for the start of the vectorized GPU computation
    if i >= 375 and "# Vectorized GPU cost computation" in lines[i]:
        # Skip until we find the first duplicate block
        while i < len(lines) and "# Aggregated CPC for this batch" not in lines[i]:
            i += 1
        
        # Insert the original CPU-based computation
        new_lines.append("                # Compute per-instance route costs and CPC (log or arithmetic)\n")
        new_lines.append("                rcosts = []  # actual route costs per instance\n")
        new_lines.append("                cpc_vals = []  # arithmetic CPC values\n")
        new_lines.append("                cpc_logs = []  # log-CPC values for geometric mean\n")
        new_lines.append("                for b in range(len(instances)):\n")
        new_lines.append("                    distances = instances[b][\"distances\"]\n")
        new_lines.append("                    route = routes[b]\n")
        new_lines.append("                    # Use GPU cost computation but simpler approach\n")
        new_lines.append("                    rc = compute_route_cost_gpu(route, distances)\n")
        new_lines.append("                    # Convert to tensor on GPU if needed\n")
        new_lines.append("                    if not isinstance(rc, torch.Tensor):\n")
        new_lines.append("                        rc = torch.tensor(rc, device=gpu_manager.device, dtype=torch.float32)\n")
        new_lines.append("                    rcosts.append(rc)\n")
        new_lines.append("                    n_customers = (len(instances[b][\"coords\"]) - 1)\n")
        new_lines.append("                    if use_geometric_mean:\n")
        new_lines.append("                        cpc_logs.append(torch.log(rc + 1e-10) - torch.log(torch.tensor(float(n_customers), device=gpu_manager.device)))\n")
        new_lines.append("                    else:\n")
        new_lines.append("                        cpc_vals.append(rc / float(n_customers))\n")
        new_lines.append("                \n")
        new_lines.append("                # Aggregated CPC for this batch (to track train_cost_epoch)\n")
        new_lines.append("                if use_geometric_mean:\n")
        new_lines.append("                    batch_cost = torch.exp(torch.stack(cpc_logs).mean())\n")
        new_lines.append("                else:\n")
        new_lines.append("                    batch_cost = torch.stack(cpc_vals).mean()\n")
        new_lines.append("                \n")
        new_lines.append("                # Build actual costs tensor for RL (match CPU: use actual costs, not CPC)\n")
        new_lines.append("                costs_tensor = torch.stack(rcosts).to(dtype=torch.float32)\n")
        
        # Skip all the duplicate blocks
        while i < len(lines) and "# Compute baseline" not in lines[i]:
            i += 1
        # Continue from baseline computation
    else:
        new_lines.append(lines[i])
        i += 1

# Write the fixed file
with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.writelines(new_lines)

print("Reverted to simpler cost computation approach")
print(f"Original lines: {len(lines)}")
print(f"New lines: {len(new_lines)}")
