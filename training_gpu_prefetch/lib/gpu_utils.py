"""
GPU Utilities for CVRP Model Training

This module provides GPU-specific utilities for efficient training including:
- Device management and selection
- Memory management and optimization
- Tensor transfer utilities
- CUDA synchronization helpers
- Mixed precision support utilities
"""

import os
import gc
import logging
from typing import Optional, Union, List, Dict, Any, Tuple
import warnings

import torch
import torch.cuda as cuda
from torch.amp import autocast
from torch.amp import GradScaler

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU resources and provides utilities for efficient GPU usage."""
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None,
                 memory_fraction: float = 0.95,
                 enable_mixed_precision: bool = True):
        """
        Initialize GPU Manager.
        
        Args:
            device: Target device (e.g., 'cuda:0', 'cuda', or torch.device)
            memory_fraction: Fraction of GPU memory to allocate (0.0-1.0)
            enable_mixed_precision: Whether to enable automatic mixed precision
        """
        self.device = self._init_device(device)
        self.memory_fraction = memory_fraction
        self.enable_mixed_precision = enable_mixed_precision and cuda.is_available()

        # Configure math precision for Ampere+ (A6000 is SM 8.6)
        if self.device.type == 'cuda':
            try:
                # Enable TF32 for remaining float32 ops
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Prefer higher throughput matmul kernels
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision('high')
                # Let cuDNN autotune best algorithms for fixed shapes
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        # Choose AMP dtype: BF16 on Ampere+ (no scaler overhead), else FP16
        if self.device.type == 'cuda':
            try:
                major, minor = torch.cuda.get_device_capability(self.device)
            except Exception:
                major, minor = (0, 0)
            self.amp_dtype = torch.bfloat16 if getattr(torch.cuda, 'is_bf16_supported', lambda: False)() else torch.float16
        else:
            self.amp_dtype = torch.float32

        # GradScaler only needed for float16 AMP; skip for bfloat16
        if self.enable_mixed_precision and self.device.type == 'cuda' and self.amp_dtype == torch.float16:
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
            
        self._setup_memory_management()
        self._log_device_info()
    
    def _init_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Initialize and validate the target device."""
        if device is None:
            if cuda.is_available():
                device = torch.device('cuda:0')
                logger.info(f"CUDA available. Using default device: {device}")
            else:
                device = torch.device('cpu')
                logger.warning("CUDA not available. Falling back to CPU.")
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Validate device
        if device.type == 'cuda':
            if not cuda.is_available():
                logger.warning("CUDA device requested but not available. Falling back to CPU.")
                device = torch.device('cpu')
            elif device.index is not None and device.index >= cuda.device_count():
                logger.warning(f"Invalid CUDA device index {device.index}. Using cuda:0")
                device = torch.device('cuda:0')
        
        return device
    
    def _setup_memory_management(self):
        """Setup memory management settings for optimal performance."""
        if self.device.type == 'cuda':
            # Set memory fraction if requested
            if self.memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.memory_fraction, 
                    device=self.device.index
                )
            
            # Enable memory caching for better performance
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            # Clear any existing cache
            self.clear_cache()
    
    def _log_device_info(self):
        """Log information about the selected device."""
        if self.device.type == 'cuda':
            props = cuda.get_device_properties(self.device.index)
            logger.info(f"GPU Device: {props.name}")
            logger.info(f"  - Compute Capability: {props.major}.{props.minor}")
            logger.info(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"  - Memory Fraction: {self.memory_fraction:.2%}")
            logger.info(f"  - Mixed Precision: {'Enabled' if self.enable_mixed_precision else 'Disabled'}")
            
            # Check current memory usage
            allocated = cuda.memory_allocated(self.device) / 1024**3
            reserved = cuda.memory_reserved(self.device) / 1024**3
            logger.info(f"  - Current Memory: {allocated:.2f}/{reserved:.2f} GB (allocated/reserved)")
        else:
            logger.info(f"Using CPU device")
    
    def to_device(self, tensor: torch.Tensor, non_blocking: bool = True) -> torch.Tensor:
        """
        Transfer tensor to the managed device.
        
        Args:
            tensor: Input tensor
            non_blocking: Whether to use non-blocking transfer
            
        Returns:
            Tensor on the target device
        """
        if tensor.device != self.device:
            return tensor.to(self.device, non_blocking=non_blocking)
        return tensor
    
    def to_device_dict(self, data: Dict[str, Any], non_blocking: bool = True) -> Dict[str, Any]:
        """
        Transfer all tensors in a dictionary to the managed device.
        
        Args:
            data: Dictionary potentially containing tensors
            non_blocking: Whether to use non-blocking transfer
            
        Returns:
            Dictionary with tensors moved to device
        """
        result = {}
        import numpy as np
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                result[key] = self.to_device(value, non_blocking)
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays directly to GPU tensors
                if key == "demands":
                    result[key] = torch.from_numpy(value).long().to(self.device, non_blocking=non_blocking)
                else:
                    result[key] = torch.from_numpy(value).float().to(self.device, non_blocking=non_blocking)
            elif isinstance(value, dict):
                result[key] = self.to_device_dict(value, non_blocking)
            elif isinstance(value, (list, tuple)):
                result[key] = type(value)(
                    self.to_device(v, non_blocking) if isinstance(v, torch.Tensor) else v
                    for v in value
                )
            else:
                result[key] = value
        return result
    
    def clear_cache(self):
        """Clear GPU cache to free up memory."""
        if self.device.type == 'cuda':
            cuda.empty_cache()
            gc.collect()
    
    def synchronize(self):
        """Synchronize GPU operations (wait for all kernels to complete)."""
        if self.device.type == 'cuda':
            cuda.synchronize(self.device)
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory usage information.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        if self.device.type == 'cuda':
            return {
                'allocated': cuda.memory_allocated(self.device) / 1024**3,
                'reserved': cuda.memory_reserved(self.device) / 1024**3,
                'free': (cuda.get_device_properties(self.device.index).total_memory - 
                        cuda.memory_allocated(self.device)) / 1024**3,
                'total': cuda.get_device_properties(self.device.index).total_memory / 1024**3
            }
        return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}
    
    def check_memory_usage(self, threshold: float = 0.9):
        """
        Check if memory usage exceeds threshold and log warning.
        
        Args:
            threshold: Memory usage threshold (0.0-1.0)
        """
        if self.device.type == 'cuda':
            mem_info = self.get_memory_info()
            usage = mem_info['allocated'] / mem_info['total']
            if usage > threshold:
                logger.warning(f"High GPU memory usage: {usage:.1%} "
                             f"({mem_info['allocated']:.2f}/{mem_info['total']:.2f} GB)")
    
    def autocast_context(self):
        """
        Get autocast context for mixed precision training.
        
        Returns:
            Autocast context manager or nullcontext
        """
        if self.enable_mixed_precision and self.device.type == 'cuda':
            return autocast('cuda', dtype=self.amp_dtype)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision training.
        
        Args:
            loss: Unscaled loss tensor
            
        Returns:
            Scaled loss tensor if using mixed precision, otherwise unchanged
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer, 
                       clip_grad_norm: Optional[float] = None,
                       parameters: Optional[List[torch.nn.Parameter]] = None):
        """
        Perform optimizer step with mixed precision support.
        
        Args:
            optimizer: PyTorch optimizer
            clip_grad_norm: Optional gradient clipping value
            parameters: Model parameters for gradient clipping
        """
        if self.scaler is not None:
            # Unscale gradients before clipping
            self.scaler.unscale_(optimizer)
            
            # Gradient clipping
            if clip_grad_norm is not None and parameters is not None:
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
            
            # Step optimizer with scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Regular gradient clipping
            if clip_grad_norm is not None and parameters is not None:
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
            
            # Regular optimizer step
            optimizer.step()
    
    def get_device_name(self) -> str:
        """Get a readable name for the current device."""
        if self.device.type == 'cuda':
            return f"GPU:{cuda.get_device_name(self.device.index)}"
        return "CPU"


class DataLoaderGPU:
    """GPU-optimized data loader wrapper."""
    
    def __init__(self, gpu_manager: GPUManager, pin_memory: bool = True, 
                 num_workers: int = 4, prefetch_factor: int = 2):
        """
        Initialize GPU-optimized data loader.
        
        Args:
            gpu_manager: GPUManager instance
            pin_memory: Whether to use pinned memory for faster transfers
            num_workers: Number of data loading workers
            prefetch_factor: Number of batches to prefetch
        """
        self.gpu_manager = gpu_manager
        self.pin_memory = pin_memory and gpu_manager.device.type == 'cuda'
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
    
    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for PyTorch DataLoader."""
        kwargs = {
            'pin_memory': self.pin_memory,
            'num_workers': self.num_workers,
        }
        
        # Add prefetch_factor only if using workers
        if self.num_workers > 0:
            kwargs['prefetch_factor'] = self.prefetch_factor
            kwargs['persistent_workers'] = True
        
        return kwargs


def estimate_batch_size(model: torch.nn.Module, 
                        sample_input: Dict[str, torch.Tensor],
                        device: torch.device,
                        target_memory_usage: float = 0.8,
                        max_batch_size: int = 512) -> int:
    """
    Estimate optimal batch size for GPU training.
    
    Args:
        model: The model to train
        sample_input: Sample input batch with batch_size=1
        device: Target device
        target_memory_usage: Target GPU memory usage fraction
        max_batch_size: Maximum allowed batch size
        
    Returns:
        Estimated optimal batch size
    """
    if device.type != 'cuda':
        return 32  # Default for CPU
    
    model = model.to(device)
    model.train()
    
    # Clear cache
    cuda.empty_cache()
    
    # Get available memory
    total_memory = cuda.get_device_properties(device.index).total_memory
    target_memory = total_memory * target_memory_usage
    
    # Test with batch size 1
    batch_size = 1
    try:
        # Move sample to device
        sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in sample_input.items()}
        
        # Forward and backward pass
        with autocast(device_type='cuda', enabled=True):
            output = model(**sample)
            if hasattr(output, 'loss'):
                loss = output.loss
            elif isinstance(output, dict) and 'loss' in output:
                loss = output['loss']
            else:
                loss = output.mean()  # Fallback
        
        loss.backward()
        
        # Measure memory usage
        memory_used = cuda.memory_allocated(device)
        
        # Estimate batch size
        estimated_batch_size = int((target_memory / memory_used) * batch_size * 0.9)  # 90% safety margin
        estimated_batch_size = min(estimated_batch_size, max_batch_size)
        estimated_batch_size = max(estimated_batch_size, 1)
        
        logger.info(f"Estimated optimal batch size: {estimated_batch_size}")
        
    except RuntimeError as e:
        logger.warning(f"Error estimating batch size: {e}")
        estimated_batch_size = 32  # Fallback
    
    finally:
        # Cleanup
        model.zero_grad()
        cuda.empty_cache()
    
    return estimated_batch_size


def profile_memory_usage(func, *args, **kwargs):
    """
    Profile GPU memory usage of a function.
    
    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Tuple of (result, memory_stats)
    """
    if not cuda.is_available():
        result = func(*args, **kwargs)
        return result, {'peak_memory': 0, 'allocated_memory': 0}
    
    cuda.reset_peak_memory_stats()
    cuda.empty_cache()
    
    start_memory = cuda.memory_allocated()
    
    result = func(*args, **kwargs)
    
    cuda.synchronize()
    
    peak_memory = cuda.max_memory_allocated()
    current_memory = cuda.memory_allocated()
    
    memory_stats = {
        'peak_memory_gb': (peak_memory - start_memory) / 1024**3,
        'allocated_memory_gb': (current_memory - start_memory) / 1024**3,
    }
    
    return result, memory_stats
