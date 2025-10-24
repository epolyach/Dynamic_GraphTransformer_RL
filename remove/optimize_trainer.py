"""
Optimize the trainer to keep distances on CPU and achieve 22-second epoch times.
"""

import sys
import re

# Read the current trainer
with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# Add a function to move data to GPU except distances
move_to_gpu_except_distances = '''
def move_to_gpu_except_distances(instance, gpu_manager):
    """Move instance to GPU but keep distances on CPU for cost computation."""
    gpu_inst = {}
    for key, value in instance.items():
        if key == 'distances':
            # Keep distances on CPU - they're only needed for cost computation
            gpu_inst[key] = value
        elif isinstance(value, np.ndarray):
            # Move other numpy arrays to GPU
            if key == 'demands':
                gpu_inst[key] = torch.tensor(value, dtype=torch.long, device=gpu_manager.device)
            else:
                gpu_inst[key] = torch.tensor(value, dtype=torch.float32, device=gpu_manager.device)
        elif isinstance(value, torch.Tensor) and key != 'distances':
            # Move existing tensors to GPU if not distances
            gpu_inst[key] = value.to(gpu_manager.device)
        else:
            gpu_inst[key] = value
    return gpu_inst

'''

# Insert the function after imports
import_end = content.find('logger = logging.getLogger(__name__)')
if import_end != -1:
    content = content[:import_end] + move_to_gpu_except_distances + '\n' + content[import_end:]

# Replace all occurrences of gpu_manager.to_device_dict with move_to_gpu_except_distances
content = re.sub(r'gpu_manager\.to_device_dict\((.*?), non_blocking=True\)', 
                 r'move_to_gpu_except_distances(\1, gpu_manager)', 
                 content)

# Write the optimized version
with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("Trainer optimized to keep distances on CPU")
