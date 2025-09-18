"""
GPU Memory Optimization utilities for CVRP training
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import gc

class MemoryOptimizer:
    """Manages GPU memory efficiently during training."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.cache_enabled = True
        
    def optimize_batch_processing(self, batch_data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Optimizes batch data for GPU processing with minimal memory overhead.
        """
        # Pre-allocate tensors
        batch_size = len(batch_data)
        max_nodes = max(len(inst['coords']) for inst in batch_data)
        
        # Use pinned memory for faster transfers
        coords_batch = torch.zeros((batch_size, max_nodes, 2), 
                                  dtype=torch.float32, 
                                  device='cpu', 
                                  pin_memory=True)
        demands_batch = torch.zeros((batch_size, max_nodes), 
                                   dtype=torch.float32, 
                                   device='cpu', 
                                   pin_memory=True)
        capacities_batch = torch.zeros((batch_size,), 
                                      dtype=torch.float32, 
                                      device='cpu', 
                                      pin_memory=True)
        
        # Fill tensors
        for i, inst in enumerate(batch_data):
            n_nodes = len(inst['coords'])
            coords_batch[i, :n_nodes] = torch.tensor(inst['coords'], dtype=torch.float32)
            demands_batch[i, :n_nodes] = torch.tensor(inst['demands'], dtype=torch.float32)
            capacities_batch[i] = inst['capacity']
        
        # Transfer to GPU in one operation
        return coords_batch.to(self.device, non_blocking=True), \
               demands_batch.to(self.device, non_blocking=True), \
               capacities_batch.to(self.device, non_blocking=True)
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            gc.collect()
    
    def enable_memory_efficient_attention(self, model: nn.Module):
        """Enable memory-efficient attention mechanisms."""
        for module in model.modules():
            if hasattr(module, 'attention'):
                # Use Flash Attention or similar optimizations
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    module.use_memory_efficient_attention = True
    
    @staticmethod
    def gradient_checkpointing(module: nn.Module):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(module, 'gradient_checkpointing_enable'):
            module.gradient_checkpointing_enable()
        return module


class BatchDataLoader:
    """Efficient data loader for GPU training."""
    
    def __init__(self, data_generator, batch_size: int, device: torch.device,
                 num_workers: int = 2, prefetch_factor: int = 2):
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.memory_optimizer = MemoryOptimizer(device)
    
    def get_batch(self, epoch: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Generate and optimize a batch of data."""
        # Generate instances
        seed = epoch * 1000 + batch_idx * self.batch_size
        instances = self.data_generator(self.batch_size, epoch=epoch, seed=seed)
        
        # Optimize memory layout
        coords, demands, capacities = self.memory_optimizer.optimize_batch_processing(instances)
        
        # Pre-compute distances on GPU
        distances = self._compute_distances_batch(coords)
        
        return {
            'coords': coords,
            'demands': demands,
            'capacities': capacities,
            'distances': distances,
            'batch_size': self.batch_size
        }
    
    @staticmethod
    @torch.jit.script
    def _compute_distances_batch(coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances for batch of coordinates."""
        # coords: (batch_size, n_nodes, 2)
        # Efficient batched distance computation
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 2)
        return torch.norm(diff, dim=-1)  # (B, N, N)
