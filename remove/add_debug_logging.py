# Add debug logging to find where it's stuck
with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    lines = f.readlines()

# Find the line with "Using oscillating temperature"
for i, line in enumerate(lines):
    if 'Using oscillating temperature' in line:
        # Add debug print after this line
        lines.insert(i + 1, '            print("[DEBUG] After oscillating temp, before cyclic LR")\n')
        break

# Find and add debug after cyclic LR
for i, line in enumerate(lines):
    if 'Cyclic LR =' in line:
        lines.insert(i + 1, '            print("[DEBUG] After cyclic LR, before model.train()")\n')
        break

# Find model.train() and add debug
for i, line in enumerate(lines):
    if 'model.train()' in line and 'Update learning rate' in lines[i-5]:
        lines.insert(i + 1, '        print(f"[DEBUG] Model in training mode, starting epoch {epoch}")\n')
        break

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.writelines(lines)

print("Added debug logging")
