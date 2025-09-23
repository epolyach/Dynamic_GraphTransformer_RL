#!/usr/bin/env python3
"""Analyze where time is spent in training"""

# Look at the training loop to identify all operations
import re

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# Find the training loop section
start_marker = "for batch_idx in range(n_batches):"
end_marker = "train_cost_epoch += batch_cost"

start_idx = content.find(start_marker)
if start_idx == -1:
    print("Could not find training loop")
    exit(1)

end_idx = content.find(end_marker, start_idx) + len(end_marker)
training_loop = content[start_idx:end_idx]

# Count operations in the loop
operations = {
    'data_generation': len(re.findall(r'data_generator\(', training_loop)),
    'move_to_gpu': len(re.findall(r'move_to_gpu_except_distances', training_loop)),
    'model_forward': len(re.findall(r'model\(', training_loop)),
    'cost_computation': len(re.findall(r'compute_route_cost', training_loop)),
    'baseline_eval': len(re.findall(r'baseline\.eval_batch', training_loop)),
    'optimizer_step': len(re.findall(r'optimizer\.step', training_loop)),
    'backward': len(re.findall(r'\.backward\(', training_loop)),
}

print("Operations per batch in training loop:")
for op, count in operations.items():
    print(f"  {op}: {count}")

# Check if there's validation in every epoch
val_section = re.search(r'# Validation.*?val_cost_epoch', content, re.DOTALL)
if val_section:
    print("\n✓ Validation runs every epoch")
else:
    print("\n✗ No validation found")

# Check for other potential bottlenecks
if 'torch.cuda.synchronize()' in training_loop:
    print("\n⚠ Found cuda.synchronize() in training loop - potential bottleneck")

if 'cpu()' in training_loop and 'numpy()' in training_loop:
    print("\n⚠ Found CPU transfers in training loop - potential bottleneck")
