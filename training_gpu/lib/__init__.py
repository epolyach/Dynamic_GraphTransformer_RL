"""
GPU Training Library for CVRP Models

This library provides GPU-optimized training components for:
- GAT (Graph Attention Network)
- GT (Graph Transformer) 
- DGT (Dynamic Graph Transformer)
"""

from .gpu_utils import (
    GPUManager,
    DataLoaderGPU,
    estimate_batch_size,
    profile_memory_usage
)

# These will be imported after creation
__all__ = [
    'GPUManager',
    'DataLoaderGPU', 
    'estimate_batch_size',
    'profile_memory_usage',
    'advanced_train_model_gpu',
    'RolloutBaselineGPU'
]

def advanced_train_model_gpu(*args, **kwargs):
    """Wrapper for GPU-optimized training."""
    from .advanced_trainer_gpu import advanced_train_model_gpu as _train
    return _train(*args, **kwargs)

def RolloutBaselineGPU(*args, **kwargs):
    """Wrapper for GPU-optimized rollout baseline."""
    from .rollout_baseline_gpu import RolloutBaselineGPU as _baseline
    return _baseline(*args, **kwargs)
