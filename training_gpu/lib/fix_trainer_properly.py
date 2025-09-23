"""
Properly fix the trainer to achieve 22-second epoch times by keeping distances on CPU.
"""

# Read the original fast trainer
with open('training_gpu/lib/advanced_trainer_gpu.py.original_fast', 'r') as f:
    content = f.read()

# Find where to insert the function (after imports, before the class definitions)
import_section_end = content.find('class AdaptiveTemperatureScheduler:')

# Create the optimized function
move_to_gpu_function = '''

def move_to_gpu_except_distances(instance, gpu_manager):
    """Move instance to GPU but keep distances on CPU for cost computation."""
    import numpy as np
    import torch
    
    gpu_inst = {}
    for key, value in instance.items():
        if key == 'distances':
            # Keep distances on CPU as numpy array for cost computation
            if isinstance(value, torch.Tensor):
                gpu_inst[key] = value.cpu().numpy()
            else:
                gpu_inst[key] = value
        elif isinstance(value, np.ndarray):
            # Move other numpy arrays to GPU
            if key == 'demands':
                gpu_inst[key] = torch.tensor(value, dtype=torch.long, device=gpu_manager.device)
            else:
                gpu_inst[key] = torch.tensor(value, dtype=torch.float32, device=gpu_manager.device)
        elif isinstance(value, torch.Tensor):
            # Move existing tensors to GPU
            gpu_inst[key] = value.to(gpu_manager.device)
        else:
            gpu_inst[key] = value
    return gpu_inst

'''

# Insert the function
content = content[:import_section_end] + move_to_gpu_function + content[import_section_end:]

# Replace all gpu_manager.to_device_dict calls with move_to_gpu_except_distances
import re
content = re.sub(
    r'gpu_manager\.to_device_dict\((.*?), non_blocking=True\)',
    r'move_to_gpu_except_distances(\1, gpu_manager)',
    content
)

# Save the fixed version
with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("Trainer fixed - distances will stay on CPU for fast cost computation")
