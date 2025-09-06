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
from src.eval.validation import validate_route
from .rollout_baseline_gpu_fixed import RolloutBaselineGPU
from .gpu_utils import GPUManager, DataLoaderGPU

logger = logging.getLogger(__name__)


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
    n_epochs = train_config.get('n_epochs', 100)
    batch_size = train_config.get('batch_size', 32)
    learning_rate = train_config.get('learning_rate', 1e-4)
    gradient_accumulation_steps = gpu_config.get('gradient_accumulation_steps', 1)
    
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler_config = train_config.get('scheduler', {})
    if scheduler_config.get('type') == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=n_epochs,
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_config.get('type') == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None
    
    # Initialize baseline
    baseline = None
    baseline_config = train_config.get('baseline', {})
    
    # Only initialize RolloutBaseline for RL training (matching CPU)
    print("[INIT] Checking if baseline needed for RL training...")
    if 'RL' in model_name:
        # Create eval dataset (same as CPU version)
        eval_batches = baseline_config.get('eval_batches', 5)
        print(f"[INIT] Building eval dataset: eval_batches={eval_batches}, batch_size={batch_size}")
        eval_dataset = []
        for i in range(eval_batches):
            # Use fixed seeds to keep eval set stable
            seed_val = 123456 + i
            eval_dataset.append(data_generator(batch_size, seed=seed_val))
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
        'train_cost': [],
        'val_cost': [],
        'learning_rate': [],
        'epoch_time': [],
        'gpu_memory': []
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
    print(f"[{model_name}] Training with {n_epochs * train_config.get('batches_per_epoch', 100) * batch_size} total instances over {n_epochs} epochs")
    print(f"[{model_name}] Batches per epoch: {train_config.get('batches_per_epoch', 100)}, batch size: {batch_size}")
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Rollout baseline update (exactly like CPU)
        if baseline is not None:
            try:
                # Only allow baseline updates after warmup epochs
                warmup = train_config.get('baseline', {}).get('warmup_epochs', 0)
                if epoch >= warmup:
                    baseline.epoch_callback(model, epoch)
            except Exception as e:
                print(f"[RolloutBaseline] Update failed at epoch {epoch}: {e}")
        
        # Training phase
        model.train()
        train_loss_epoch = []
        train_cost_epoch = []
        
        # Generate training batches
        n_batches = train_config.get('batches_per_epoch', 100)
        
        for batch_idx in range(n_batches):
            # Generate problem instances
            instances = data_generator(batch_size, epoch=epoch)
            
            # Move to GPU
            instances = [gpu_manager.to_device_dict(inst, non_blocking=True) for inst in instances]
            
            # Forward pass with mixed precision
            with gpu_manager.autocast_context():
                routes, log_probs, entropy = model(instances)
                
                # Compute costs
                batch_costs = []
                for b in range(len(instances)):
                    c = compute_route_cost(routes[b], instances[b]['distances']) / (len(instances[b]['coords']) - 1)
                    batch_costs.append(c)
                costs = torch.tensor(batch_costs, device=gpu_manager.device)
                
                
                # Compute baseline (matching CPU)
                if baseline is not None:
                    baseline_value = baseline.eval_batch(instances)
                    if baseline_value.device != costs.device:
                        baseline_value = baseline_value.to(costs.device)
                else:
                    baseline_value = torch.zeros_like(costs)
                
                # Compute advantage
                advantage = costs - baseline_value
                
                # REINFORCE loss
                reinforce_loss = (advantage * log_probs.sum(dim=-1)).mean()
                
                # Add entropy regularization if configured
                if train_config.get('entropy_weight', 0) > 0:
                    entropy = -(log_probs * log_probs.exp()).sum(dim=-1).mean()
                    reinforce_loss = reinforce_loss - train_config['entropy_weight'] * entropy
                
                # Scale loss for gradient accumulation
                loss = reinforce_loss / gradient_accumulation_steps
            
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
            train_loss_epoch.append(reinforce_loss.item())
            train_cost_epoch.append(costs.mean().item())
            train_metrics.update(
                loss=reinforce_loss.item(),
                cost=costs.mean().item(),
                advantage=advantage.mean().item()
            )
            
            # Check memory usage periodically
            if batch_idx % 50 == 0:
                gpu_manager.check_memory_usage(threshold=0.9)
        
        # Clear gradients if not already done
        if n_batches % gradient_accumulation_steps != 0:
            optimizer.zero_grad()
        
        # Validation phase
        model.eval()
        val_cost_epoch = []
        
        with torch.no_grad():
            for _ in range(train_config.get('val_batches', 10)):
                val_instances = data_generator(batch_size, epoch=epoch)
                val_instances = [gpu_manager.to_device_dict(inst, non_blocking=True) for inst in val_instances]
                
                with gpu_manager.autocast_context():
                    routes_val, log_probs_val, entropy_val = model(val_instances)
                    
                    # Compute validation costs
                    val_batch_costs = []
                    for b in range(len(val_instances)):
                        c = compute_route_cost(routes_val[b], val_instances[b]['distances']) / (len(val_instances[b]['coords']) - 1)
                        val_batch_costs.append(c)
                    val_costs = torch.tensor(val_batch_costs, device=gpu_manager.device)
                    
                    # val_costs already computed above
                
                val_cost_epoch.append(val_costs.mean().item())
                val_metrics.update(cost=val_costs.mean().item())
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start
        avg_train_loss = np.mean(train_loss_epoch)
        avg_train_cost = np.mean(train_cost_epoch)
        avg_val_cost = np.mean(val_cost_epoch)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_cost'].append(avg_train_cost)
        history['val_cost'].append(avg_val_cost)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        # GPU memory tracking
        mem_info = gpu_manager.get_memory_info()
        history['gpu_memory'].append(mem_info['allocated'])
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_cost)
            else:
                scheduler.step()
        
        # Get current learning rate for logging
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        
        # Logging
        print(
            f"[{model_name}] Epoch {epoch:03d}: "
            f"train={np.mean(train_cost_epoch):.4f}, "
            f"val={np.mean(val_cost_epoch) if val_cost_epoch else 0:.4f}, "
            f"lr={current_lr:.2e}, temp=2.500, "
            f"time={epoch_time:.1f}s"
        ) if val_cost_epoch else print(
            f"[{model_name}] Epoch {epoch:03d}: "
            f"train={np.mean(train_cost_epoch):.4f}, "
            f"lr={current_lr:.2e}, temp=2.500, "
            f"time={epoch_time:.1f}s"
        )
        
        # Wandb logging
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_cost': avg_train_cost,
                'val_cost': avg_val_cost,
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
        'final_train_cost': history['train_cost'][-1],
        'final_val_cost': history['val_cost'][-1],
        'best_val_cost': min(history['val_cost']),
        'total_time': sum(history['epoch_time']),
        'avg_epoch_time': np.mean(history['epoch_time']),
        'peak_gpu_memory': max(history['gpu_memory']),
        'device': gpu_manager.get_device_name()
    }
    
    logger.info(f"Training completed. Final metrics: {final_metrics}")
    
    return model, {'history': history, 'metrics': final_metrics}
