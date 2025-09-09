import os
import time
import copy
from typing import Dict, Any, List, Tuple, Optional, Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.metrics.costs import compute_route_cost
from src.eval.validation import validate_route
from .rollout_baseline import RolloutBaseline


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


class AdvancedMetrics:
    """Advanced metrics tracking for training analysis."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
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


def advanced_train_model(
    model: nn.Module, 
    model_name: str, 
    config: Dict[str, Any],
    data_generator: Callable,
    logger_print: Callable = print,
    use_advanced_features: bool = True,
    epoch_callback: Optional[Callable] = None
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    Advanced training with modern techniques:
    - Learning rate scheduling
    - Early stopping
    - Adaptive temperature
    - Advanced metrics tracking
    - Gradient clipping with adaptive norms
    - Optional epoch callback for incremental logging
    """
    device = torch.device('cpu')
    model.to(device)
    model.train()

    # Training hyperparameters
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    base_lr = config['training']['learning_rate']
    use_geometric_mean = config['training'].get('use_geometric_mean', True)
    
    # Advanced training parameters
    adv_config = config.get('training_advanced', {})
    use_early_stopping = adv_config.get('use_early_stopping', use_advanced_features)
    use_lr_scheduling = adv_config.get('use_lr_scheduling', use_advanced_features)
    use_adaptive_temp = adv_config.get('use_adaptive_temperature', use_advanced_features)
    
    # Validation settings
    strict_validation = config.get('experiment', {}).get('strict_validation', True)
    
    # Initialize optimizer with weight decay for regularization
    if model_name != 'GT-Greedy':
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
    else:
        optimizer = None

    # Learning rate scheduler
    scheduler = None
    if optimizer is not None and use_lr_scheduling:
        if adv_config.get('scheduler_type', 'cosine') == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=num_epochs,
                eta_min=adv_config.get('min_lr', base_lr * 0.01)
            )
        else:  # ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=adv_config.get('lr_factor', 0.5), 
                patience=adv_config.get('lr_patience', 5),
                min_lr=adv_config.get('min_lr', base_lr * 0.01)
            )
    
    # Early stopping
    early_stopping = None
    if use_early_stopping and model_name != 'GT-Greedy':
        early_stopping = EarlyStopping(
            patience=adv_config.get('early_stopping_patience', 15),
            min_delta=adv_config.get('early_stopping_delta', 1e-4),
            restore_best_weights=True
        )
    
    # Adaptive temperature scheduler
    temp_scheduler = None
    if use_adaptive_temp:
        temp_scheduler = AdaptiveTemperatureScheduler(
            temp_start=adv_config.get('temp_start', 2.0),
            temp_min=adv_config.get('temp_min', 0.5),
            adaptation_rate=adv_config.get('temp_adaptation_rate', 0.1)
        )
    
    # Metrics tracking
    metrics = AdvancedMetrics()
    
    # Training history
    train_losses: List[float] = []
    train_costs: List[float] = []
    val_costs: List[float] = []
    learning_rates: List[float] = []
    temperatures: List[float] = []
    
    # Calculate total number of batches to process all instances
    num_instances = config['training'].get('num_instances', batch_size * num_epochs)
    batches_per_epoch = max(1, num_instances // (batch_size * (num_epochs + 1)))  # +1 because we have epoch 0
    logger_print(f"[{model_name}] Training with {num_instances} total instances over {num_epochs + 1} epochs")
    logger_print(f"[{model_name}] Batches per epoch: {batches_per_epoch}, batch size: {batch_size}")
    logger_print(f"[{model_name}] Using {'geometric' if use_geometric_mean else 'arithmetic'} mean for CPC aggregation")
    
    start_time = time.time()
    
    # Set up optional rollout baseline
    baseline_cfg = config.get('baseline', {})
    use_rollout_baseline = str(baseline_cfg.get('type', 'mean')).lower() == 'rollout'
    baseline_update_frequency = int(baseline_cfg.get('update', {}).get('frequency', 1))
    baseline_update_warmup_epochs = int(baseline_cfg.get('update', {}).get('warmup_epochs', 0))
    baseline: Optional[RolloutBaseline] = None

    if use_rollout_baseline and model_name != 'GT-Greedy':
        # Build a fixed evaluation dataset (list of instance-batches)
        eval_batches = int(baseline_cfg.get('eval_batches', 2))
        eval_dataset: List[List[Dict[str, Any]]] = []
        for i in range(eval_batches):
            # Use fixed seeds to keep the eval set stable across epochs
            seed_val = 123456 + i
            eval_dataset.append(data_generator(batch_size, seed=seed_val))
        baseline = RolloutBaseline(model, eval_dataset, config, logger_print)

    for epoch in range(0, num_epochs + 1):
        epoch_start = time.time()
        epoch_losses = []
        epoch_costs = []
        
        # Current temperature (adaptive or scheduled)
        if temp_scheduler:
            current_temp = temp_scheduler.current_temp
        else:
            # Original temperature scheduling
            temp_progress = (epoch - 1) / max(1, num_epochs - 1)
            temp_start = adv_config.get('temp_start', 1.5)
            temp_min = adv_config.get('temp_min', 0.2)
            current_temp = temp_min + (temp_start - temp_min) * (0.5 * (1 + np.cos(np.pi * temp_progress)))
        
        temperatures.append(current_temp)
        
        # Process multiple batches per epoch
        for batch_idx in range(batches_per_epoch):
            # Generate batch with unique seed
            batch_seed = epoch * batches_per_epoch * 1000 + batch_idx * 1000 if epoch > 0 else batch_idx * 1000
            instances = data_generator(batch_size, epoch=epoch, seed=batch_seed)
            
            # Forward pass
            routes, logp, ent = model(
                instances,
                max_steps=len(instances[0]['coords']) * config['inference']['max_steps_multiplier'],
                temperature=current_temp,
                greedy=False,
                config=config
            )
            
            # Compute costs and validate routes
            batch_costs = []
            n_customers = len(instances[0]['coords']) - 1  # Number of customers (excluding depot)
            for b in range(len(instances)):
                r = routes[b]
                if strict_validation:
                    validate_route(r, n_customers, f"{model_name}-TRAIN", instances[b])
                route_cost = compute_route_cost(r, instances[b]['distances'])
                
                # Compute CPC in appropriate space
                if use_geometric_mean:
                    # In log space: ln(cost/N) = ln(cost) - ln(N)
                    cpc = np.log(route_cost + 1e-10) - np.log(n_customers)
                else:
                    # Regular arithmetic CPC
                    cpc = route_cost / n_customers
                batch_costs.append(cpc)
            
            # Aggregate using appropriate mean
            if use_geometric_mean:
                # Geometric mean: exp(mean of log values)
                batch_cost = float(np.exp(np.mean(batch_costs)))
            else:
                # Arithmetic mean
                batch_cost = float(np.mean(batch_costs))
            epoch_costs.append(batch_cost)
            
            # Update metrics
            metrics.update(
                cost=batch_cost,
                cost_std=np.std(batch_costs),
                route_length=np.mean([len(r) for r in routes]),
                entropy=ent.mean().item() if hasattr(ent, 'mean') else float(ent.mean() if torch.is_tensor(ent) else 0.0)
            )
            
            # REINFORCE update for RL models
            if optimizer is not None and model_name != 'GT-Greedy':
                optimizer.zero_grad()
                
                # For REINFORCE, we need actual costs (not log-transformed)
                if use_geometric_mean:
                    # Convert back from log space for loss computation
                    actual_costs = [np.exp(log_cpc) * n_customers for log_cpc in batch_costs]
                    costs_tensor = torch.tensor(actual_costs, dtype=torch.float32)
                else:
                    # Already in regular space, just scale back
                    actual_costs = [cpc * n_customers for cpc in batch_costs]
                    costs_tensor = torch.tensor(actual_costs, dtype=torch.float32)

                # Compute baseline (rollout or mean)
                if baseline is not None:
                    with torch.no_grad():
                        bl_vals = baseline.eval_batch(instances)  # per-instance costs (CPU tensor)
                    # Ensure dtype/device alignment
                    bl_vals = bl_vals.to(dtype=costs_tensor.dtype)
                    if bl_vals.numel() != costs_tensor.numel():
                        # Fallback to mean if mismatch occurs
                        base_scalar = costs_tensor.mean().detach()
                        advantages = base_scalar - costs_tensor
                    else:
                        advantages = bl_vals.detach() - costs_tensor  # Lower cost -> positive advantage
                else:
                    base_scalar = costs_tensor.mean().detach()
                    advantages = base_scalar - costs_tensor  # Lower cost -> positive advantage
                
                # Entropy regularization with adaptive coefficient
                entropy_coef = adv_config.get('entropy_coef', 0.01)
                entropy_min = adv_config.get('entropy_min', 0.0)
                if num_epochs > 1:
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * (epoch - 1) / (num_epochs - 1)))
                    entropy_coef = entropy_min + (entropy_coef - entropy_min) * cosine_factor
                
                # Normalize advantages to reduce variance and stabilize updates
                adv = advantages
                adv_mean = adv.mean()
                adv_std = adv.std()
                adv_norm = (adv - adv_mean) / (adv_std + 1e-8)

                # Policy loss with entropy regularization (use normalized advantages)
                policy_loss = -(adv_norm * logp).mean()
                entropy_loss = -entropy_coef * ent.mean()
                total_loss = policy_loss + entropy_loss
                
                total_loss.backward()
                
                # Adaptive gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    adv_config.get('gradient_clip_norm', 2.0)
                )
                
                optimizer.step()
                
                epoch_losses.append(float(total_loss.detach().cpu()))
                metrics.update(
                    loss=total_loss.item(),
                    policy_loss=policy_loss.item(),
                    entropy_loss=entropy_loss.item(),
                    grad_norm=grad_norm.item(),
                    entropy_coef=entropy_coef
                )
            else:
                epoch_losses.append(float('nan'))
        
        # Average costs and losses over all batches in epoch
        train_cost = float(np.mean(epoch_costs)) if epoch_costs else 0.0
        train_loss = float(np.nanmean(epoch_losses)) if epoch_losses else float('nan')
        train_costs.append(train_cost)
        train_losses.append(train_loss)
        
        # Learning rate tracking
        current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0
        learning_rates.append(current_lr)
        
        # Validation
        val_cost = None
        if (epoch % config['training']['validation_frequency']) == 0 or epoch == num_epochs:
            with torch.no_grad():
                # Use different seed range for validation (1M offset from training)
                val_seed = 1000000 + epoch * batch_size  # Ensures no overlap with training seeds
                val_instances = data_generator(batch_size, seed=val_seed)
                routes_g, _, _ = model(
                    val_instances,
                    max_steps=len(val_instances[0]['coords']) * config['inference']['max_steps_multiplier'],
                    # Use same temperature as training for consistent validation
                    # Research shows "validate what you train" principle is important
                    temperature=current_temp,  # Match training temperature
                    greedy=False,  # Sample from distribution, don't use greedy
                    config=config
                )
                val_batch_costs = []
                n_customers = len(val_instances[0]['coords']) - 1
                for b in range(len(val_instances)):
                    r = routes_g[b]
                    if strict_validation:
                        validate_route(r, n_customers, f"{model_name}-VAL", val_instances[b])
                    route_cost = compute_route_cost(r, val_instances[b]['distances'])
                    
                    # Compute CPC in appropriate space
                    if use_geometric_mean:
                        cpc = np.log(route_cost + 1e-10) - np.log(n_customers)
                    else:
                        cpc = route_cost / n_customers
                    val_batch_costs.append(cpc)
                
                # Aggregate using appropriate mean
                if use_geometric_mean:
                    val_cost = float(np.exp(np.mean(val_batch_costs)))
                else:
                    val_cost = float(np.mean(val_batch_costs))
                val_costs.append(val_cost)
                
                # Update adaptive temperature based on validation performance
                if temp_scheduler:
                    temp_scheduler.update(val_cost)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                if val_cost is not None:
                    scheduler.step(val_cost)
            else:
                scheduler.step()
        
        # Rollout baseline update (if enabled)
        if baseline is not None:
            try:
                # Only allow baseline updates after warmup epochs
                if epoch >= baseline_update_warmup_epochs:
                    baseline.epoch_callback(model, epoch)
            except Exception as e:
                logger_print(f"[RolloutBaseline] Update failed at epoch {epoch}: {e}")
        
        # Early stopping check
        if early_stopping is not None and val_cost is not None:
            if early_stopping(val_cost, model):
                logger_print(f"[{model_name}] Early stopping triggered at epoch {epoch}")
                early_stopping.restore_best_model(model)
                break
        
        # Compute baseline value for this epoch (CSV logging only).
        # To reduce runtime, only compute rollout-baseline value on scheduled update epochs.
        baseline_value = None
        if epoch_costs:
            if use_rollout_baseline and baseline is not None:
                # Only compute when the baseline update is scheduled (every baseline_update_frequency epochs)
                if (epoch % baseline_update_frequency) == 0:
                    try:
                        # Get a representative baseline value from the rollout
                        test_instances = data_generator(batch_size, seed=999999 + epoch)
                        with torch.no_grad():
                            baseline_costs = baseline.eval_batch(test_instances)
                        baseline_value = float(baseline_costs.mean())
                    except:
                        baseline_value = float(np.mean(epoch_costs))
                else:
                    baseline_value = None  # Skip computing on non-update epochs to save time
            else:
                # Mean baseline (using same aggregation as training)
                baseline_value = float(np.mean(epoch_costs))  # Already aggregated costs
        
        # Call epoch callback for incremental CSV writing
        if epoch_callback is not None:
            epoch_callback(
                epoch=epoch,
                train_loss=train_loss,
                train_cost=train_cost,
                val_cost=val_cost,
                learning_rate=current_lr,
                temperature=current_temp,
                baseline_value=baseline_value
            )
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_summary = metrics.epoch_summary("train_")
        
        if val_cost is not None:
            logger_print(
                f"[{model_name}] Epoch {epoch:03d}: "
                f"train={train_cost:.4f}, val={val_cost:.4f}, "
                f"lr={current_lr:.2e}, temp={current_temp:.3f}, "
                f"time={epoch_time:.1f}s"
            )
        else:
            logger_print(
                f"[{model_name}] Epoch {epoch:03d}: "
                f"train={train_cost:.4f}, "
                f"lr={current_lr:.2e}, temp={current_temp:.3f}, "
                f"time={epoch_time:.1f}s"
            )
    
    training_time = time.time() - start_time
    
    # Enhanced history with advanced metrics
    history = {
        'train_losses': train_losses,
        'train_costs': train_costs,
        'val_costs': val_costs,
        'final_val_cost': val_costs[-1] if val_costs else (train_costs[-1] if train_costs else 0.0),
        'learning_rates': learning_rates,
        'temperatures': temperatures,
        'epoch_metrics': dict(metrics.epoch_metrics),
    }
    
    # Additional artifacts
    artifacts = {
        'final_temperature': temperatures[-1] if temperatures else config['inference']['default_temperature'],
        'best_val_cost': min(val_costs) if val_costs else float('inf'),
        'convergence_epoch': np.argmin(val_costs) + 1 if val_costs else num_epochs,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    return history, training_time, artifacts
