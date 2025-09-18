import re

# Read the file
with open('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL/training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# The fix: remove the redundant batch_cost computation that tries to stack already-computed tensors
# Lines 440-444 contain the problematic code
lines = content.split('\n')

# Find and comment out the redundant batch_cost computation
new_lines = []
for i, line in enumerate(lines):
    if i >= 440 and i <= 444 and 'batch_cost = torch' in line:
        # Comment out the redundant batch_cost computation
        new_lines.append(line.replace('batch_cost = torch', '# batch_cost = torch  # Already computed above'))
    else:
        new_lines.append(line)

# Write back the fixed content
with open('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL/training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("Fixed the type error by commenting out redundant batch_cost computation")
