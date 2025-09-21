# Read the file
with open('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL/training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    lines = f.readlines()

# Fix lines 441-444 (0-indexed, so 440-443)
# The issue is that cpc_vals and cpc_logs are already tensors, not lists of tensors
# So we don't need torch.stack(), just use the values directly

for i in range(len(lines)):
    # Line 442 (0-indexed 441)
    if i == 441 and 'torch.stack(cpc_logs)' in lines[i]:
        lines[i] = lines[i].replace('torch.stack(cpc_logs).mean()', 'cpc_logs.mean()')
    # Line 444 (0-indexed 443)
    elif i == 443 and 'torch.stack(cpc_vals)' in lines[i]:
        lines[i] = lines[i].replace('torch.stack(cpc_vals).mean()', 'cpc_vals.mean()')

# Write back
with open('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL/training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.writelines(lines)

print("Fixed torch.stack() type errors")
