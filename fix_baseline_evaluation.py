#!/usr/bin/env python3
"""
Fix the baseline evaluation to not run on every batch
The baseline should provide pre-computed values, not re-evaluate every batch
"""

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# The problematic code evaluates baseline on every batch during training
# This is incorrect - baseline should be evaluated once and provide stored values

# Find and comment out the per-batch baseline evaluation in training loop
import re

# Find the section with baseline.eval_batch in the training loop
pattern = r'(# Compute baseline.*?)(bl_vals = baseline\.eval_batch\(instances\).*?advantages = bl_vals\.detach\(\) - costs_tensor.*?else:)'
replacement = r'''\1# FIXED: Don't evaluate baseline on every batch - use pre-computed mean
                    # The baseline should only be updated periodically, not evaluated per batch
                    bl_val = torch.tensor(baseline.mean, device=costs_tensor.device, dtype=costs_tensor.dtype)
                    advantages = bl_val - costs_tensor  # Lower cost -> positive advantage
                else:'''

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("Fixed baseline evaluation - now using pre-computed baseline mean instead of evaluating on every batch")
print("This should significantly improve performance!")
