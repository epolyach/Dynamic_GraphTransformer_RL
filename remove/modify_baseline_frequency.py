#!/usr/bin/env python3
"""
Modify the trainer to evaluate baseline every 3 epochs instead of every epoch
"""

# Read the current trainer
with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    lines = f.readlines()

# Find and modify the baseline update frequency
modified = False
new_lines = []

for i, line in enumerate(lines):
    # Look for baseline_update_frequency setting
    if 'baseline_update_frequency = int(baseline_config.get' in line:
        # Change default from 1 to 3
        new_line = line.replace("'frequency', 1", "'frequency', 3")
        new_lines.append(new_line)
        print(f"Modified line {i+1}: {line.strip()} -> {new_line.strip()}")
        modified = True
    # Also reduce eval_batches from 5 to 2
    elif "eval_batches = baseline_config.get('eval_batches', 5)" in line:
        new_line = line.replace("'eval_batches', 5", "'eval_batches', 2")
        new_lines.append(new_line)
        print(f"Modified line {i+1}: {line.strip()} -> {new_line.strip()}")
    else:
        new_lines.append(line)

# Write the modified trainer
with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.writelines(new_lines)

if modified:
    print("\nSuccessfully modified baseline update frequency to 3 epochs")
    print("Also reduced eval_batches from 5 to 2")
else:
    print("Warning: Could not find baseline_update_frequency line")
