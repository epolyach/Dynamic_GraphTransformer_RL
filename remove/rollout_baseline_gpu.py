"""
GPU-Optimized Rollout Baseline for REINFORCE Algorithm

This module provides GPU-optimized rollout baseline computation for:
- Efficient batch rollout generation
- Parallel trajectory sampling
- GPU-accelerated greedy decoding
"""

import copy
import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from .gpu_utils import GPUManager

logger = logging.getLogger(__name__)


class RolloutBaselineGPU:
    """GPU-optimized rollout baseline for REINFORCE algorithm."""
    
    def __init__(self, model: nn.Module, 
                 gpu_manager: GPUManager,
                 n_rollouts: int = 10,
                 warmup_epochs: int = 1,
                 update_epochs: int = 1,
                 use_mixed_precision: bool = True):
        """
        Initialize GPU-optimized rollout baseline.
        
        Args:
            model: The model to use for rollouts
            gpu_manager: GPUManager instance for device management
            n_rollouts: Number of rollout trajectories
            warmup_epochs: Number of warmup epochs
            update_epochs: Epochs between baseline updates
            use_mixed_precision: Whether to use mixed precision for rollouts
        """
        self.model = model
        self.gpu_manager = gpu_manager
        self.n_rollouts = n_rollouts
        self.warmup_epochs = warmup_epochs
        self.update_epochs = update_epochs
        self.use_mixed_precision = use_mixed_precision
        
        # Baseline model (copy of the main model)
        self.baseline_model = None
        self.epoch = 0
        self.baseline_values = []
        
        # Move model to device
        self.model = self.model.to(gpu_manager.device)
        
        logger.info(f"Initialized RolloutBaselineGPU with {n_rollouts} rollouts on {gpu_manager.get_device_name()}")
    
    def setup(self):
        """Setup baseline model as a copy of the main model."""
        self.baseline_model = copy.deepcopy(self.model).to(self.gpu_manager.device)
        self.baseline_model.eval()
        for param in self.baseline_model.parameters():
            param.requires_grad = False
        logger.info("Baseline model initialized")
    
    def update(self, epoch: int):
        """
        Update baseline model if needed.
        
        Args:
            epoch: Current training epoch
        """
        self.epoch = epoch
        
        if self.baseline_model is None:
            self.setup()
        elif epoch > self.warmup_epochs and epoch % self.update_epochs == 0:
            # Update baseline model with current model weights
            self.baseline_model.load_state_dict(self.model.state_dict())
            logger.info(f"Updated baseline model at epoch {epoch}")
            
            # Clear GPU cache after update
            self.gpu_manager.clear_cache()
    
    @torch.no_grad()
    def compute_baseline(self, instances: Dict[str, torch.Tensor],
                        return_costs: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute baseline value using rollout trajectories on GPU.
        
        Args:
            instances: Batch of problem instances
            return_costs: Whether to return individual rollout costs
            
        Returns:
            Tuple of (baseline_value, optional_costs)
        """
        if self.baseline_model is None:
            self.setup()
        
        batch_size = next(iter(instances.values())).shape[0]
        device = self.gpu_manager.device
        
        # Ensure instances are on GPU
        instances = self.gpu_manager.to_device_dict(instances, non_blocking=True)
        
        all_costs = []
        
        # Use mixed precision if enabled
        autocast_context = self.gpu_manager.autocast_context() if self.use_mixed_precision else None
        
        # Perform rollouts in batches to manage memory
        rollouts_per_batch = min(self.n_rollouts, 5)  # Adjust based on GPU memory
        n_batches = (self.n_rollouts + rollouts_per_batch - 1) // rollouts_per_batch
        
        for batch_idx in range(n_batches):
            batch_rollouts = min(rollouts_per_batch, self.n_rollouts - batch_idx * rollouts_per_batch)
            
            # Expand instances for parallel rollouts
            expanded_instances = {}
            for key, value in instances.items():
                # Repeat each instance for multiple rollouts
                expanded = value.unsqueeze(1).expand(-1, batch_rollouts, *value.shape[1:])
                expanded = expanded.reshape(-1, *value.shape[1:])
                expanded_instances[key] = expanded
            
            # Generate rollout trajectories
            if autocast_context:
                with autocast_context:
                    rollout_output = self.baseline_model(
                        expanded_instances,
                        decode_type='sampling' if batch_idx > 0 else 'greedy'
                    )
            else:
                rollout_output = self.baseline_model(
                    expanded_instances,
                    decode_type='sampling' if batch_idx > 0 else 'greedy'
                )
            
            # Extract costs
            if hasattr(rollout_output, 'costs'):
                costs = rollout_output.costs
            elif isinstance(rollout_output, dict) and 'costs' in rollout_output:
                costs = rollout_output['costs']
            else:
                # Compute costs from routes if not directly available
                routes = rollout_output.routes if hasattr(rollout_output, 'routes') else rollout_output['routes']
                costs = self._compute_route_costs(expanded_instances, routes)
            
            # Reshape costs back to (batch_size, n_rollouts)
            costs = costs.reshape(batch_size, batch_rollouts)
            all_costs.append(costs)
        
        # Concatenate all rollout costs
        all_costs = torch.cat(all_costs, dim=1)  # Shape: (batch_size, n_rollouts)
        
        # Compute baseline as mean of rollout costs
        baseline_value = all_costs.mean(dim=1)  # Shape: (batch_size,)
        
        # Store for statistics
        self.baseline_values.append(baseline_value.mean().item())
        
        if return_costs:
            return baseline_value, all_costs
        return baseline_value, None
    
    @torch.no_grad()
    def compute_baseline_greedy(self, instances: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute baseline using greedy decoding only (faster, single rollout).
        
        Args:
            instances: Batch of problem instances
            
        Returns:
            Baseline values
        """
        if self.baseline_model is None:
            self.setup()
        
        # Ensure instances are on GPU
        instances = self.gpu_manager.to_device_dict(instances, non_blocking=True)
        
        # Use mixed precision if enabled
        with self.gpu_manager.autocast_context() if self.use_mixed_precision else torch.no_grad():
            output = self.baseline_model(instances, decode_type='greedy')
        
        # Extract costs
        if hasattr(output, 'costs'):
            baseline_value = output.costs
        elif isinstance(output, dict) and 'costs' in output:
            baseline_value = output['costs']
        else:
            routes = output.routes if hasattr(output, 'routes') else output['routes']
            baseline_value = self._compute_route_costs(instances, routes)
        
        return baseline_value
    
    def _compute_route_costs(self, instances: Dict[str, torch.Tensor], 
                            routes: torch.Tensor) -> torch.Tensor:
        """
        Compute costs for given routes on GPU.
        
        Args:
            instances: Problem instances
            routes: Solution routes
            
        Returns:
            Route costs
        """
        # Extract coordinates
        if 'coordinates' in instances:
            coords = instances['coordinates']
        elif 'depot' in instances and 'customers' in instances:
            depot = instances['depot']
            customers = instances['customers']
            coords = torch.cat([depot.unsqueeze(1), customers], dim=1)
        else:
            raise ValueError("Cannot extract coordinates from instances")
        
        batch_size, n_nodes, _ = coords.shape
        device = coords.device
        
        # Create distance matrix
        distances = torch.cdist(coords, coords, p=2)
        
        # Compute route costs
        costs = torch.zeros(batch_size, device=device)
        
        for b in range(batch_size):
            route = routes[b]
            # Remove padding (-1 values)
            valid_mask = route >= 0
            valid_route = route[valid_mask]
            
            if len(valid_route) > 0:
                # Add depot at start and end if not present
                if valid_route[0] != 0:
                    valid_route = torch.cat([torch.tensor([0], device=device), valid_route])
                if valid_route[-1] != 0:
                    valid_route = torch.cat([valid_route, torch.tensor([0], device=device)])
                
                # Compute total distance
                for i in range(len(valid_route) - 1):
                    costs[b] += distances[b, valid_route[i], valid_route[i+1]]
        
        return costs
    
    def get_stats(self) -> Dict[str, float]:
        """Get baseline statistics."""
        if not self.baseline_values:
            return {'baseline_mean': 0.0, 'baseline_std': 0.0}
        
        values = torch.tensor(self.baseline_values[-100:])  # Last 100 values
        return {
            'baseline_mean': values.mean().item(),
            'baseline_std': values.std().item() if len(values) > 1 else 0.0,
            'baseline_min': values.min().item(),
            'baseline_max': values.max().item()
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        return {
            'baseline_model': self.baseline_model.state_dict() if self.baseline_model else None,
            'epoch': self.epoch,
            'baseline_values': self.baseline_values[-1000:],  # Keep last 1000 values
            'n_rollouts': self.n_rollouts,
            'warmup_epochs': self.warmup_epochs,
            'update_epochs': self.update_epochs
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from checkpoint."""
        if state_dict['baseline_model'] is not None:
            if self.baseline_model is None:
                self.setup()
            self.baseline_model.load_state_dict(state_dict['baseline_model'])
        
        self.epoch = state_dict.get('epoch', 0)
        self.baseline_values = state_dict.get('baseline_values', [])
        self.n_rollouts = state_dict.get('n_rollouts', self.n_rollouts)
        self.warmup_epochs = state_dict.get('warmup_epochs', self.warmup_epochs)
        self.update_epochs = state_dict.get('update_epochs', self.update_epochs)


class ExponentialBaseline:
    """Exponential moving average baseline (simpler, no rollouts needed)."""
    
    def __init__(self, alpha: float = 0.99):
        """
        Initialize exponential baseline.
        
        Args:
            alpha: Decay factor for exponential moving average
        """
        self.alpha = alpha
        self.value = None
        self.n_updates = 0
    
    def update(self, costs: torch.Tensor):
        """
        Update baseline with new costs.
        
        Args:
            costs: Batch of costs
        """
        mean_cost = costs.mean().item()
        
        if self.value is None:
            self.value = mean_cost
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * mean_cost
        
        self.n_updates += 1
    
    def get(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get baseline values.
        
        Args:
            batch_size: Size of the batch
            device: Target device
            
        Returns:
            Baseline values
        """
        if self.value is None:
            return torch.zeros(batch_size, device=device)
        return torch.full((batch_size,), self.value, device=device)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        return {
            'value': self.value,
            'n_updates': self.n_updates,
            'alpha': self.alpha
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary."""
        self.value = state_dict.get('value', None)
        self.n_updates = state_dict.get('n_updates', 0)
        self.alpha = state_dict.get('alpha', self.alpha)
