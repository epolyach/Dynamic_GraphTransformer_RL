"""GPU-optimized data generator wrapper."""
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable


def create_gpu_data_generator(base_generator: Callable, device: str = 'cuda') -> Callable:
    """
    Wrap a data generator to pre-convert data to GPU tensors.
    
    Args:
        base_generator: Original data generator function
        device: Target device for tensors
    
    Returns:
        GPU-optimized generator function
    """
    def gpu_generator(batch_size: int, epoch: int = 1, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        # Generate base data
        instances = base_generator(batch_size, epoch, seed)
        
        # Convert to GPU tensors
        gpu_instances = []
        for inst in instances:
            gpu_inst = {}
            for key, value in inst.items():
                if isinstance(value, np.ndarray):
                    # Convert numpy arrays to tensors
                    if key == 'demands':
                        gpu_inst[key] = torch.from_numpy(value).long().to(device, non_blocking=True)
                    else:
                        gpu_inst[key] = torch.from_numpy(value).float().to(device, non_blocking=True)
                else:
                    # Keep scalar values as-is
                    gpu_inst[key] = value
            gpu_instances.append(gpu_inst)
        
        return gpu_instances
    
    return gpu_generator


def batch_to_gpu(instances: List[Dict[str, Any]], device: str = 'cuda') -> List[Dict[str, Any]]:
    """
    Convert a batch of instances to GPU tensors.
    
    Args:
        instances: List of problem instances
        device: Target device
    
    Returns:
        List of instances with tensors on GPU
    """
    gpu_instances = []
    for inst in instances:
        gpu_inst = {}
        for key, value in inst.items():
            if isinstance(value, np.ndarray):
                if key == 'demands':
                    gpu_inst[key] = torch.from_numpy(value).long().to(device, non_blocking=True)
                else:
                    gpu_inst[key] = torch.from_numpy(value).float().to(device, non_blocking=True)
            elif isinstance(value, torch.Tensor):
                gpu_inst[key] = value.to(device, non_blocking=True)
            else:
                gpu_inst[key] = value
        gpu_instances.append(gpu_inst)
    
    return gpu_instances
