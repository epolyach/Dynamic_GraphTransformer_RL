#!/usr/bin/env python3
"""The real fix - don't move distances to GPU since we compute costs on CPU"""

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# Find and replace the to_device_dict calls to exclude distances
# We need to modify how instances are moved to GPU

# Add a helper function after imports
helper_code = '''
def move_to_gpu_except_distances(instance, gpu_manager):
    """Move instance to GPU but keep distances on CPU for cost computation."""
    gpu_inst = {}
    for key, value in instance.items():
        if key == 'distances':
            # Keep distances on CPU since we need them for cost computation
            gpu_inst[key] = value
        elif isinstance(value, (np.ndarray, list)):
            # Move other arrays to GPU
            gpu_inst[key] = torch.tensor(value, dtype=torch.float32, device=gpu_manager.device)
        else:
            gpu_inst[key] = value
    return gpu_inst

'''

# Insert helper after imports (find the first function definition)
import_end = content.find('class AdaptiveTemperatureScheduler')
content = content[:import_end] + helper_code + content[import_end:]

# Replace all gpu_manager.to_device_dict with our custom function
content = content.replace(
    'instances = [gpu_manager.to_device_dict(inst, non_blocking=True) for inst in instances]',
    'instances = [move_to_gpu_except_distances(inst, gpu_manager) for inst in instances]'
)

content = content.replace(
    'val_instances = [gpu_manager.to_device_dict(inst, non_blocking=True) for inst in val_instances]',
    'val_instances = [move_to_gpu_except_distances(inst, gpu_manager) for inst in val_instances]'
)

content = content.replace(
    'test_instances = [gpu_manager.to_device_dict(inst, non_blocking=True) for inst in test_instances]',
    'test_instances = [move_to_gpu_except_distances(inst, gpu_manager) for inst in test_instances]'
)

content = content.replace(
    'gpu_batch = [gpu_manager.to_device_dict(inst, non_blocking=True) for inst in batch_data]',
    'gpu_batch = [move_to_gpu_except_distances(inst, gpu_manager) for inst in batch_data]'
)

# Now fix the cost computation - distances are already CPU numpy arrays
content = content.replace(
    'distances_cpu = distances.cpu().numpy() if isinstance(distances, torch.Tensor) else distances',
    'distances_cpu = distances  # Already on CPU'
)

# Add missing numpy import if needed
if 'import numpy as np' not in content:
    content = content.replace('import torch', 'import numpy as np\nimport torch')

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("✅ ULTIMATE FIX APPLIED:")
print("   - Distances stay on CPU (no transfer overhead)")
print("   - Only coords/demands moved to GPU (needed for model)")
print("   - Cost computation uses CPU distances directly")
print("   - Eliminated unnecessary GPU↔CPU transfers")
print("\nThis should restore performance to ~22 seconds per epoch!")
