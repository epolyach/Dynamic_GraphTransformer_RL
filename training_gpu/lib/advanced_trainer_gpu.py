"""
GPU-Optimized Advanced Trainer for CVRP Models

This module provides GPU-optimized training functionality for:
- GAT (Graph Attention Network)
- GT (Graph Transformer)
- DGT (Dynamic Graph Transformer)

Key optimizations:
- Mixed precision training (FP16/FP32)
- Efficient batch processing on GPU
- Non-blocking data transfers
- Gradient accumulation
- Memory management
"""

import os
import time
import copy
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.metrics.costs import compute_route_cost
from src.metrics.gpu_costs import compute_route_cost_gpu
from .validation_gpu import validate_route
from .rollout_baseline_gpu_fixed import RolloutBaselineGPU
from .gpu_utils import GPUManager, DataLoaderGPU

logger = logging.getLogger(__name__)

def get_current_batch_size(epoch, train_config):
    """
    Get the current batch size based on epoch and curriculum schedule.
    
    Args:
        epoch: Current training epoch
        train_config: Training configuration dict
        
    Returns:
        int: Current batch size for this epoch
    """
    # Check if curriculum is enabled
    curriculum = train_config.get('curriculum', {})
    if not curriculum.get('enabled', False):
        return train_config.get('batch_size', 32)
    
    # Get batch size schedule
    batch_size_schedule = curriculum.get('batch_size_schedule', [])
    if not batch_size_schedule:
        return train_config.get('batch_size', 32)
    
    # Sort schedule by epoch (in case it's not sorted)
    sorted_schedule = sorted(batch_size_schedule, key=lambda x: x['epoch'])
    
    # Find the appropriate batch size for current epoch
    current_batch_size = train_config.get('batch_size', 32)  # fallback
    
    for schedule_entry in sorted_schedule:
        if epoch >= schedule_entry['epoch']:
            current_batch_size = schedule_entry['batch_size']
        else:
            break
    
    return current_batch_size


def get_oscillating_temperature(epoch, adv_config):
    """Get temperature for oscillating schedule."""
    if not adv_config.get('use_oscillating_temperature', False):
        return None
    
    period = adv_config.get('temp_oscillation_period', 20)
    temp_high = adv_config.get('temp_high', 2.5)
    temp_low = adv_config.get('temp_low', 1.5)
    
    # Oscillate between high and low
    phase = (epoch % period) / period
    import math
    # Use cosine for smooth transition
    temp = temp_low + (temp_high - temp_low) * (1 + math.cos(2 * math.pi * phase)) / 2
    return temp

def get_cyclic_lr(epoch, base_lr, adv_config):
    """Get learning rate for cyclic schedule."""
    if adv_config.get('scheduler_type') != 'cyclic':
        return base_lr
    
    lr_base = float(adv_config.get('lr_base', base_lr))
    lr_max = float(adv_config.get('lr_max', base_lr * 4))
    cycle_epochs = int(adv_config.get('lr_cycle_epochs', 30))
    
    # Triangular cyclic schedule
    cycle_pos = epoch % cycle_epochs
    if cycle_pos < cycle_epochs / 2:
        # Increasing phase
        progress = cycle_pos / (cycle_epochs / 2)
        lr = lr_base + (lr_max - lr_base) * progress
    else:
        # Decreasing phase
        progress = (cycle_pos - cycle_epochs / 2) / (cycle_epochs / 2)
        lr = lr_max - (lr_max - lr_base) * progress
    
    return lr

def get_adaptive_entropy_coef(epoch, recent_losses, adv_config):
    """Adjust entropy coefficient based on plateau detection."""
    base_entropy = adv_config.get('entropy_coef', 0.01)
    
    if not adv_config.get('use_adaptive_entropy', False):
        return base_entropy
        
    window = adv_config.get('plateau_detection_window', 10)
    threshold = adv_config.get('plateau_threshold', 0.001)
    boost = adv_config.get('entropy_boost_on_plateau', 0.02)
    min_entropy = adv_config.get('entropy_min', 0.001)
    
    # Need at least window epochs of history
    if len(recent_losses) < window:
        return base_entropy
    
    # Check for plateau
    recent = recent_losses[-window:]
    improvement = max(recent[:-1]) - recent[-1]
    
    if improvement < threshold:
        # Plateau detected, boost entropy
        return min(base_entropy + boost, 0.1)  # Cap at 0.1
    else:
        # Making progress, use base entropy
        return max(base_entropy, min_entropy)

def should_use_critic_baseline(epoch, config):
    """Determine if should use critic baseline based on hybrid strategy."""
    adv_config = config.get('training_advanced', {})
    
    if not adv_config.get('use_hybrid_baseline', False):
        # Check baseline type directly
        baseline_type = config.get('baseline', {}).get('type', 'rollout')
        return baseline_type == 'critic'
    
    # Hybrid: switch at specified epoch
    switch_epoch = adv_config.get('baseline_switch_epoch', 50)
    return epoch >= switch_epoch

def get_dropout_rate(epoch, config):
    """Get dropout rate based on curriculum schedule."""
    curriculum = config.get('curriculum_learning', {})
    dropout_schedule = curriculum.get('dropout_schedule', [])
    
    if not dropout_schedule:
        return 0.0  # No dropout by default
    
    # Sort schedule by epoch
    sorted_schedule = sorted(dropout_schedule, key=lambda x: x['epoch'])
    
    # Find appropriate dropout for current epoch
    current_dropout = 0.0
    for entry in sorted_schedule:
        if epoch >= entry['epoch']:
            current_dropout = entry['dropout']
        else:
            break
    
    return current_dropout


# Speed knobs for Ampere+ GPUs (e.g., A6000):
# - TF32 can accelerate large matmul-heavy models with minimal accuracy loss
# - High matmul precision hints PyTorch to use TF32 where possible
try:
    import torch.backends.cuda
    torch.backends.cuda.matmul.allow_tf32 = True  # enable TF32 matmul
    # cudnn TF32 mainly affects convs; harmless to enable
    import torch.backends.cudnn as cudnn
    cudnn.allow_tf32 = True
except Exception:
    pass

try:
    # PyTorch 2.0+: set float32 matmul precision policy
    torch.set_float32_matmul_precision('high')
except Exception:
    pass





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
        elif key == 'capacity':
            # Coerce capacity to a plain Python int to avoid type issues in validation
            try:
                if hasattr(value, 'item'):
                    gpu_inst[key] = int(value.item())
                else:
                    import numpy as _np
                    gpu_inst[key] = int(_np.array(value).item())
            except Exception:
                try:
                    gpu_inst[key] = int(value)
                except Exception:
                    gpu_inst[key] = 0
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

class AdaptiveTemperatureScheduler:
    """Adaptive temperature scheduling for exploration-exploitation balance."""
    
    def __init__(self, temp_start: float = 2.0, temp_min: float = 0.5, 
                 adaptation_rate: float = 0.1, performance_window: int = 5):
        self.temp_start = temp_start
        self.temp_min = temp_min
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        self.current_temp = temp_start
        self.performance_history = []
    
    def update(self, performance: float) -> float:
        """Update temperature based on recent performance."""
        self.performance_history.append(performance)
        
        if len(self.performance_history) >= self.performance_window:
            # Keep only recent history
            self.performance_history = self.performance_history[-self.performance_window:]
            
            # Calculate performance trend
            recent_mean = np.mean(self.performance_history[-self.performance_window//2:])
            older_mean = np.mean(self.performance_history[:self.performance_window//2])
            
            # If performance is improving, reduce temperature (more exploitation)
            # If performance is stagnating, increase temperature (more exploration)
            if recent_mean < older_mean:  # Lower cost is better
                self.current_temp = max(self.temp_min, 
                                      self.current_temp * (1 - self.adaptation_rate))
            else:
                self.current_temp = min(self.temp_start,
                                      self.current_temp * (1 + self.adaptation_rate))
        
        return self.current_temp

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        elif val_score < self.best_score - self.min_delta:
            self.best_score = val_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_best_model(self, model: nn.Module):
        """Restore the best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class AdvancedMetricsGPU:
    """GPU-optimized metrics tracking for training analysis."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.metrics[key].append(value)
    
    def epoch_summary(self, prefix: str = ""):
        """Compute epoch summary statistics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{prefix}{key}_mean"] = np.mean(values)
                summary[f"{prefix}{key}_std"] = np.std(values)
                summary[f"{prefix}{key}_min"] = np.min(values)
                summary[f"{prefix}{key}_max"] = np.max(values)
        
        # Store epoch summaries
        for key, value in summary.items():
            self.epoch_metrics[key].append(value)
        
        self.reset()  # Reset for next epoch
        return summary
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get the full history of epoch metrics."""
        return dict(self.epoch_metrics)



def compute_costs_gpu(routes, distances_gpu, device):
    """
    Compute route costs on GPU using vectorized operations.
    
    Args:
        routes: List of routes (Python lists)
        distances_gpu: Distance matrices already on GPU (batch_size, n, n)
        device: GPU device
    
    Returns:
        Tensor of costs on GPU
    """
    batch_size = len(routes)
    max_len = max(len(r) for r in routes)
    
    # Pad routes and convert to tensor
    routes_padded = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
    for i, route in enumerate(routes):
        routes_padded[i, :len(route)] = torch.tensor(route, dtype=torch.long, device=device)
    
    # Compute edge costs
    from_idx = routes_padded[:, :-1]
    to_idx = routes_padded[:, 1:]
    valid_mask = (from_idx >= 0) & (to_idx >= 0)
    
    # Batch indexing
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_len-1)
    
    # Get edge costs
    edge_costs = torch.zeros_like(from_idx, dtype=torch.float32)
    valid_indices = valid_mask.nonzero(as_tuple=False)
    
    if len(valid_indices) > 0:
        b_idx = batch_idx[valid_mask]
        f_idx = from_idx[valid_mask]
        t_idx = to_idx[valid_mask]
        edge_costs[valid_mask] = distances_gpu[b_idx, f_idx, t_idx]
    
    # Sum to get total costs
    return edge_costs.sum(dim=1)


def advanced_train_model_gpu(
    model: nn.Module,
    model_name: str,
    data_generator: Any,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Path] = None,
    callbacks: Optional[List[Callable]] = None,
    use_wandb: bool = False
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    GPU-optimized training function for CVRP models.
    
    Args:
        model: The model to train
        data_generator: Data generator for creating problem instances
        config: Training configuration
        device: Target device (if None, will auto-select GPU)
        checkpoint_dir: Directory for saving checkpoints
        callbacks: Optional training callbacks
        use_wandb: Whether to use Weights & Biases logging
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Initialize GPU manager
    gpu_config = config.get('gpu', {})
    print("[INIT] Initializing GPU Manager...")
    gpu_manager = GPUManager(
        device=device or gpu_config.get('device'),
        memory_fraction=gpu_config.get('memory_fraction', 0.95),
        enable_mixed_precision=gpu_config.get('mixed_precision', True)
    )
    
    # Move model to GPU
    model = model.to(gpu_manager.device)
    logger.info(f"Model moved to {gpu_manager.get_device_name()}")
    
    # Training parameters
    train_config = config.get('training', {})
    adv_config = config.get('training_advanced', {})
    n_epochs = train_config.get('num_epochs', 100)
    # Initialize batch size (support curriculum learning)
    initial_batch_size = train_config.get('batch_size', 32)
    batch_size = get_current_batch_size(0, train_config)  # Start with epoch 0 batch size
    if batch_size != initial_batch_size:
        logger.info(f"Curriculum learning: Using batch_size={batch_size} instead of configured {initial_batch_size} for epoch 0")
    base_lr = train_config.get('learning_rate', 1e-4)
    validation_frequency = train_config.get('validation_frequency', 1)
    gradient_accumulation_steps = gpu_config.get('gradient_accumulation_steps', 1)

    # Feature flags (match CPU)
    use_lr_scheduling = adv_config.get('use_lr_scheduling', True)
    use_adaptive_temp = adv_config.get('use_adaptive_temperature', True)
    strict_validation = config.get('experiment', {}).get('strict_validation', True)
    
    # Adjust effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch_size} "
                f"(batch_size={batch_size}, accumulation_steps={gradient_accumulation_steps})")
    
    # Data loader configuration
    dataloader_gpu = DataLoaderGPU(
        gpu_manager=gpu_manager,
        pin_memory=gpu_config.get('pin_memory', True),
        num_workers=gpu_config.get('num_workers', 4),
        prefetch_factor=gpu_config.get('prefetch_factor', 2)
    )
    dataloader_kwargs = dataloader_gpu.get_dataloader_kwargs()
    
    # Optimizer
    beta1 = adv_config.get('adam_beta1', 0.9)
    beta2 = adv_config.get('adam_beta2', 0.999)
    adam_eps = adv_config.get('adam_eps', 1e-8)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=adv_config.get('weight_decay', 1e-4),
        betas=(beta1, beta2),
        eps=adam_eps
    )

    # Initialize tracking for adaptive features
    recent_losses = []  # Track recent losses for plateau detection
    
    # Learning rate scheduler
    scheduler = None
    if use_lr_scheduling:
        if adv_config.get('scheduler_type', 'cosine') == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=n_epochs,
                eta_min=adv_config.get('min_lr', base_lr * 0.01)
            )
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=adv_config.get('lr_factor', 0.5),
                patience=adv_config.get('lr_patience', 5),
                min_lr=adv_config.get('min_lr', base_lr * 0.01)
            )
    
    # Initialize baseline
    baseline = None
    baseline_config = config.get('baseline', {})
    baseline_update_warmup_epochs = int(baseline_config.get('update', {}).get('warmup_epochs', 0))
    
    baseline_update_frequency = int(baseline_config.get('update', {}).get('frequency', 3))
    # Only initialize RolloutBaseline for RL training (matching CPU)
    print("[INIT] Checking if baseline needed for RL training...")
    # Check if baseline is actually wanted (not just if RL is in name)
    baseline_type = baseline_config.get('type', 'rollout').lower()
    use_baseline = 'RL' in model_name and baseline_type != 'none'
    
    if use_baseline:
        # Check if hybrid baseline requested
        if baseline_type == 'hybrid':
            print("[INIT] Creating hybrid baseline")
            from .critic_baseline import HybridBaseline
            baseline = HybridBaseline(
                gpu_manager=gpu_manager,
                model=model,
                config=config,
                data_generator=data_generator,
                batch_size=batch_size,
                move_to_gpu_except_distances=move_to_gpu_except_distances,
                logger_print=print
            )
            print("[INIT] Hybrid baseline initialized")
        else:
            # Original rollout baseline code follows
            # Create eval dataset (same as CPU version)
            eval_batches = baseline_config.get('eval_batches', 1)
            # For large batch sizes, use fewer eval batches to prevent hangs
            if batch_size >= 2048:
                eval_batches = min(eval_batches, 1)
                logger.info(f"Large batch size ({batch_size}) detected: reducing eval_batches to {eval_batches}")
            elif batch_size >= 1024:
                eval_batches = min(eval_batches, 2)
                logger.info(f"Medium-large batch size ({batch_size}) detected: limiting eval_batches to {eval_batches}")
            print(f"[INIT] Building eval dataset: eval_batches={eval_batches}, batch_size={batch_size}")
            eval_dataset = []
            # Use only 1 batch for initialization to speed up
            init_eval_batches = min(1, eval_batches)
            for i in range(init_eval_batches):
                # Use fixed seeds to keep eval set stable
                seed_val = 123456 + i
                batch_data = data_generator(batch_size, seed=seed_val)
            # Pre-convert batch to GPU tensors to minimize transfers during baseline evaluation
                gpu_batch = [move_to_gpu_except_distances(inst, gpu_manager) for inst in batch_data]
                eval_dataset.append(gpu_batch)
            print(f"[INIT] Eval dataset built: {len(eval_dataset)} batches")
        
            # Create baseline with CPU-identical parameters
            baseline = RolloutBaselineGPU(
                gpu_manager=gpu_manager,
                model=model,
                eval_dataset=eval_dataset,
                config=config,
                logger_print=print  # Use print for identical output
            )
        print("[INIT] Baseline initialized")
    
    # Adaptive temperature scheduler (matching CPU)
    temp_scheduler = None
    if use_adaptive_temp:
        temp_scheduler = AdaptiveTemperatureScheduler(
            temp_start=adv_config.get('temp_start', 2.0),
            temp_min=adv_config.get('temp_min', 0.5),
            adaptation_rate=adv_config.get('temp_adaptation_rate', 0.1)
        )
    
    # Early stopping
    early_stop_config = train_config.get('early_stopping', {})
    early_stopping = EarlyStopping(
        patience=early_stop_config.get('patience', 20),
        min_delta=early_stop_config.get('min_delta', 1e-4),
        restore_best_weights=early_stop_config.get('restore_best', True)
    ) if early_stop_config.get('enabled', True) else None
    
    # Metrics tracking
    train_metrics = AdvancedMetricsGPU(device=gpu_manager.device)
    val_metrics = AdvancedMetricsGPU(device=gpu_manager.device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_cost_arithmetic': [],
        'val_cost_arithmetic': [],
        'learning_rate': [],
        'epoch_time': [],
        'gpu_memory': [],
        'baseline_type': [],
        'baseline_value': [],
        'mean_type': []
    }
    
    # Initialize wandb if requested
    if use_wandb:
        import wandb
        wandb.init(
            project="cvrp-training-gpu",
            config=config,
            tags=[model.__class__.__name__, "gpu"]
        )
        wandb.watch(model)
    
    # Training loop
    num_batches_per_epoch = train_config.get('num_batches_per_epoch', 150)
    # Match CPU semantics: include epoch 0, so epochs are (num_epochs + 1)
    total_epochs = n_epochs + 1
    total_instances = train_config.get('num_instances', total_epochs * num_batches_per_epoch * batch_size)
    print(f"[{model_name}] Training with {total_instances} total instances over {total_epochs} epochs")
    print(f"[{model_name}] Batches per epoch: {num_batches_per_epoch}, batch size: {batch_size}")
    # Match CPU-side informational print about CPC aggregation
    use_geometric_mean = config.get('training', {}).get('use_geometric_mean', True)
    print(f"[{model_name}] Using {'geometric' if use_geometric_mean else 'arithmetic'} mean for CPC aggregation")
    
    for epoch in range(0, total_epochs):        
        epoch_start = time.time()
        
        # Check if batch size should change for this epoch (curriculum learning)
        new_batch_size = get_current_batch_size(epoch, train_config)
        if new_batch_size != batch_size:
            old_batch_size = batch_size
            batch_size = new_batch_size
            effective_batch_size = batch_size * gradient_accumulation_steps
            logger.info(f"Curriculum learning: Batch size changed from {old_batch_size} to {batch_size} at epoch {epoch}")
            logger.info(f"New effective batch size: {effective_batch_size}")

        # Update dropout if scheduled
        current_dropout = get_dropout_rate(epoch, config)
        if current_dropout > 0:
            logger.info(f"Epoch {epoch}: Dropout rate = {current_dropout:.2f}")
            # Note: Model needs to be modified to support dynamic dropout
                
        # Current temperature (adaptive or scheduled) - matching CPU exactly
        
        # Oscillating temperature (takes priority)
        oscillating_temp = get_oscillating_temperature(epoch, adv_config)
        if oscillating_temp is not None:
            current_temp = oscillating_temp
            # print(f"Epoch {epoch}: Using oscillating temperature: {current_temp:.3f}")
        elif temp_scheduler:
            current_temp = temp_scheduler.current_temp
            current_temp = temp_scheduler.current_temp
        else:
            # Original temperature scheduling
            temp_progress = (epoch - 1) / max(1, n_epochs - 1)
            temp_start = adv_config.get("temp_start", 1.5)
            temp_min = adv_config.get("temp_min", 0.2)
            current_temp = temp_min + (temp_start - temp_min) * (0.5 * (1 + np.cos(np.pi * temp_progress)))
        
        epoch_start = time.time()
        
        # Training phase
        # Update learning rate (cyclic or standard)
        current_lr = get_cyclic_lr(epoch, base_lr, adv_config)
        if current_lr != base_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            # logger.info(f"Epoch {epoch}: Cyclic LR = {current_lr:.6f}")
        
        model.train()
        train_loss_epoch = []
        train_cost_epoch = []
        
        # Generate training batches
        n_batches = num_batches_per_epoch
        
        for batch_idx in range(n_batches):
            # Generate problem instances (align seeds with CPU)
            
            # batch_seed = epoch * n_batches * 1000 + batch_idx * 1000 if epoch > 0 else batch_idx * 1000
            batch_seed = epoch * n_batches * batch_size + batch_idx * batch_size 
            instances = data_generator(batch_size, epoch=epoch, seed=batch_seed)
            
            # Move to GPU
            instances = [move_to_gpu_except_distances(inst, gpu_manager) for inst in instances]
            

            # Forward pass with mixed precision
            with gpu_manager.autocast_context():
                routes, log_probs, entropy = model(
                    instances,
                    max_steps=len(instances[0]['coords']) * config.get('inference', {}).get('max_steps_multiplier', 10),
                    temperature=current_temp,
                    greedy=False,
                    config=config
                )
                
                # Compute per-instance route costs using GPU
                # First ensure distances are on GPU
                distances_list = []
                for inst in instances:
                    dist = inst["distances"]
                    if isinstance(dist, np.ndarray):
                        dist = torch.tensor(dist, dtype=torch.float32, device=gpu_manager.device)
                    elif isinstance(dist, torch.Tensor) and dist.device != gpu_manager.device:
                        dist = dist.to(gpu_manager.device)
                    distances_list.append(dist)
                
                distances_gpu = torch.stack(distances_list)
                
                # Compute costs on GPU
                rcosts = compute_costs_gpu(routes, distances_gpu, gpu_manager.device)
                
                # Compute CPC values
                n_customers_tensor = torch.tensor([len(inst["coords"]) - 1 for inst in instances],
                                                 device=gpu_manager.device, dtype=torch.float32)
                
                if use_geometric_mean:
                    cpc_logs = torch.log(rcosts + 1e-10) - torch.log(n_customers_tensor)
                    cpc_vals = []
                else:
                    cpc_vals = rcosts / n_customers_tensor
                    cpc_logs = []
                
                # Aggregated CPC for this batch (to track train_cost_epoch)
                if use_geometric_mean:
                    batch_cost = torch.exp(cpc_logs.mean())
                else:
                    batch_cost = cpc_vals.mean()
                
                # Build actual costs tensor for RL (use actual costs, not CPC)
                costs_tensor = rcosts.to(dtype=torch.float32)
                
                # Compute baseline (matching CPU)
                if baseline is not None:
                    # FIXED: Don't evaluate baseline on every batch - use pre-computed mean
                    # The baseline should only be updated periodically, not evaluated per batch
                    bl_val = torch.tensor(baseline.mean, device=costs_tensor.device, dtype=costs_tensor.dtype)
                    advantages = bl_val - costs_tensor  # Lower cost -> positive advantage
                else:
                    base_scalar = costs_tensor.mean().detach()
                    advantages = base_scalar - costs_tensor
                
                # REINFORCE loss with optional entropy regularization
                adv = advantages
                adv_mean = adv.mean()
                adv_std = adv.std()
                adv_norm = (adv - adv_mean) / (adv_std + 1e-8)
                # Fix: log_probs is already the total log probability for the sequence
                # Don't sum it - use it directly like CPU trainer
                # Policy loss (matching CPU exactly)
                policy_loss = -(adv_norm * log_probs).mean()
                
                # Entropy regularization (matching CPU)
                # Adaptive entropy coefficient
                entropy_coef = get_adaptive_entropy_coef(epoch, recent_losses, adv_config)
                if adv_config.get('use_adaptive_entropy', False) and epoch > 0:
                    logger.debug(f"Adaptive entropy coef: {entropy_coef:.4f}")
                entropy_min = adv_config.get("entropy_min", 0.0)
                if n_epochs > 1:
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * (epoch - 1) / (n_epochs - 1)))
                    entropy_coef = entropy_min + (entropy_coef - entropy_min) * cosine_factor
                
                entropy_loss = -entropy_coef * entropy.mean()
                total_loss = policy_loss + entropy_loss
                
                # Scale loss for gradient accumulation
                # Only scale for gradient accumulation if steps > 1
                loss = total_loss if gradient_accumulation_steps == 1 else total_loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            if gpu_manager.scaler is not None:
                gpu_manager.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation and optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping and optimizer step
                gpu_manager.step_optimizer(
                    optimizer,
                    clip_grad_norm=train_config.get('clip_grad_norm', 1.0),
                    parameters=model.parameters()
                )
                optimizer.zero_grad()
            
            # Track metrics
            train_loss_epoch.append(total_loss.item())  # Track unscaled total loss
            train_cost_epoch.append(batch_cost.item())
            train_metrics.update(
                loss=(loss.item() * gradient_accumulation_steps),
                cost=batch_cost.item(),
                advantage=adv_norm.mean().item()
            )
            
            # Check memory usage periodically
            if batch_idx % 50 == 0:
                gpu_manager.check_memory_usage(threshold=0.9)
        
        # Clear gradients if not already done
        if n_batches % gradient_accumulation_steps != 0:
            optimizer.zero_grad()
        
        # Validation phase (align with CPU validation_frequency)
        model.eval()
        val_cost_epoch = []
        do_validate = (epoch % validation_frequency) == 0 or epoch == n_epochs
        if do_validate:
            with torch.no_grad():
                val_seed = 1000000 + epoch * batch_size
                val_instances = data_generator(batch_size, seed=val_seed)
                val_instances = [move_to_gpu_except_distances(inst, gpu_manager) for inst in val_instances]
                with gpu_manager.autocast_context():
                    routes_val, _, _ = model(
                        val_instances,
                        max_steps=len(val_instances[0]['coords']) * config.get('inference', {}).get('max_steps_multiplier', 10),
                        temperature=current_temp,
                        greedy=False,
                        config=config
                    )
                    # Compute validation CPC using requested mean
                    if use_geometric_mean:
                        val_logs = []
                        for b in range(len(val_instances)):
                            distances = val_instances[b]["distances"]
                            route = routes_val[b]
                            rc = compute_route_cost_gpu(route, distances)
                            if not isinstance(rc, torch.Tensor):
                                rc = torch.tensor(rc, device=gpu_manager.device, dtype=torch.float32)
                            n_customers = (len(val_instances[b]["coords"]) - 1)
                            val_logs.append(torch.log(rc + 1e-10) - torch.log(torch.tensor(float(n_customers), device=gpu_manager.device)))
                        batch_val_cost = float(torch.exp(torch.stack(val_logs).mean()).item())
                    else:
                        val_cpcs = []
                        for b in range(len(val_instances)):
                            distances = val_instances[b]["distances"]
                            route = routes_val[b]
                            rc = compute_route_cost_gpu(route, distances)
                            if not isinstance(rc, torch.Tensor):
                                rc = torch.tensor(rc, device=gpu_manager.device, dtype=torch.float32)
                            n_customers = (len(val_instances[b]["coords"]) - 1)
                            val_cpcs.append((rc / float(n_customers)).item())
                        batch_val_cost = float(np.mean(val_cpcs))
                val_cost_epoch.append(batch_val_cost)
                val_metrics.update(cost=batch_val_cost)
                
                # Update adaptive temperature based on validation performance
                if temp_scheduler and 'batch_val_cost' in locals():
                    temp_scheduler.update(batch_val_cost)
        # Epoch statistics
        epoch_time = time.time() - epoch_start
        avg_train_loss = np.mean(train_loss_epoch)
        # Track for adaptive entropy
        recent_losses.append(avg_train_loss)
        avg_train_cost = np.mean(train_cost_epoch)
        avg_val_cost = np.mean(val_cost_epoch) if val_cost_epoch else float('nan')
        current_lr = optimizer.param_groups[0]['lr']
        
        # GPU memory tracking
        mem_info = gpu_manager.get_memory_info()
        history['gpu_memory'].append(mem_info['allocated'])
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_cost)
            else:
                scheduler.step()
        

        # Rollout baseline update (exactly like CPU)
        if baseline is not None:
            try:
                # Only allow baseline updates after warmup epochs
                if epoch >= baseline_update_warmup_epochs:
                    # Log baseline state before callback (which might change it)
                    if hasattr(baseline, 'current_mode'):
                        old_mode = baseline.current_mode
                    baseline.epoch_callback(model, int(epoch))
                    # Check if mode changed
                    if hasattr(baseline, 'current_mode') and baseline.current_mode != old_mode:
                        print(f"[{model_name}] Baseline mode changed: {old_mode} → {baseline.current_mode}")
            except Exception as e:
                print(f"[RolloutBaseline] Update failed at epoch {epoch}: {e}")
                # Compute baseline value for CSV logging (match CPU behavior)
        # Detect baseline type
        if baseline is None:
            baseline_type = "none"
        elif hasattr(baseline, "current_mode"):
            # HybridBaseline
            baseline_type = f"hybrid({baseline.current_mode})"
        else:
            # RolloutBaseline or other
            baseline_type = "rollout"
        baseline_value = None
        if train_cost_epoch:  # if epoch_costs
            if baseline is not None:
                # Only compute when the baseline update is scheduled
                if (epoch % baseline_update_frequency) == 0:
                    try:
                        # Get a representative baseline value from the rollout
                        test_instances = data_generator(batch_size, seed=999999 + epoch)
                        # Move to GPU
                        test_instances = [move_to_gpu_except_distances(inst, gpu_manager) for inst in test_instances]
                        with torch.no_grad():
                            baseline_costs = baseline.eval_batch(test_instances)
                        baseline_value = float(baseline_costs.mean())
                    except:
                        baseline_value = float(avg_train_cost)
                else:
                    baseline_value = None  # Skip computing on non-update epochs to save time
            else:
                # Mean baseline (using same aggregation as training)
                baseline_value = float(avg_train_cost)  # Already aggregated costs
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_cost_arithmetic'].append(avg_train_cost)
        history['val_cost_arithmetic'].append(avg_val_cost)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        history.setdefault('baseline_type', []).append(baseline_type)
        history.setdefault('baseline_value', []).append(baseline_value)
        history.setdefault('mean_type', []).append('geometric' if use_geometric_mean else 'arithmetic')
        history.setdefault('temperature', []).append(float(current_temp))
                
        # Get current learning rate for logging
        current_lr = scheduler.get_last_lr()[0] if (scheduler and not isinstance(scheduler, ReduceLROnPlateau)) else optimizer.param_groups[0]['lr']
        
        # Logging
        if val_cost_epoch:
            print(
                f"[{model_name}] Epoch {epoch:03d}: "
                f"train={np.mean(train_cost_epoch):.4f}, val={np.mean(val_cost_epoch):.4f}, "
                f"lr={current_lr:.2e}, temp={current_temp:.3f}, baseline={baseline_type}, time={epoch_time:.1f}s"
            )
        else:
            print(
                f"[{model_name}] Epoch {epoch:03d}: "
                f"train={np.mean(train_cost_epoch):.4f}, "
                f"lr={current_lr:.2e}, temp={current_temp:.3f}, baseline={baseline_type}, time={epoch_time:.1f}s"
            )
        
        # Wandb logging
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_cost_arithmetic': avg_train_cost,
                'val_cost_arithmetic': avg_val_cost,
                'baseline_type': baseline_type,
                'baseline_value': baseline_value,
                'mean_type': 'geometric' if use_geometric_mean else 'arithmetic',
                'learning_rate': current_lr,
                'epoch_time': epoch_time,
                'gpu_memory_gb': mem_info['allocated'],
                'gpu_utilization': mem_info['allocated'] / mem_info['total']
            })
        
        # Checkpointing
        if checkpoint_dir and (epoch + 1) % train_config.get('checkpoint_interval', 10) == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'baseline_state_dict': baseline.state_dict() if hasattr(baseline, 'state_dict') else None,
                'history': history,
                'config': config
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Early stopping
        if early_stopping and early_stopping(avg_val_cost, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            early_stopping.restore_best_model(model)
            break
        
        # Callbacks
        if callbacks:
            for callback in callbacks:
                callback(epoch, model, history)
        
        # Clear GPU cache periodically
        if (epoch + 1) % 5 == 0:
            gpu_manager.clear_cache()
    
    # Final cleanup
    gpu_manager.clear_cache()
    
    # Close wandb
    if use_wandb:
        wandb.finish()
    
    # Compile final metrics
    final_metrics = {
        'final_train_cost': history['train_cost_arithmetic'][-1] if history.get('train_cost_arithmetic') else None,
        'final_val_cost': history['val_cost_arithmetic'][-1] if history.get('val_cost_arithmetic') else None,
        'best_val_cost': min(history['val_cost_arithmetic']) if history.get('val_cost_arithmetic') else None,
        'total_time': float(sum(history['epoch_time'])) if history.get('epoch_time') else 0.0,
        'avg_epoch_time': float(np.mean(history['epoch_time'])) if history.get('epoch_time') else 0.0,
        'peak_gpu_memory': float(max(history['gpu_memory'])) if history.get('gpu_memory') else 0.0,
        'device': gpu_manager.get_device_name()
    }
    
    logger.info(f"Training completed. Final metrics: {final_metrics}")
    
    return model, {'history': history, 'metrics': final_metrics}
