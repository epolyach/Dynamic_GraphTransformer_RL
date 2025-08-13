import os
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.metrics.costs import compute_route_cost, compute_naive_baseline_cost
from src.eval.validation import validate_route


def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_cvrp_instance(num_customers: int, capacity: int, coord_range: int, demand_range: List[int], seed: int | None = None) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)
    coords = np.zeros((num_customers + 1, 2), dtype=np.float64)
    for i in range(num_customers + 1):
        coords[i] = np.random.randint(0, coord_range + 1, size=2) / coord_range
    demands = np.zeros(num_customers + 1, dtype=np.int32)
    for i in range(1, num_customers + 1):
        demands[i] = np.random.randint(demand_range[0], demand_range[1] + 1)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    return {
        'coords': coords,
        'demands': demands.astype(np.int32),
        'distances': distances,
        'capacity': int(capacity)
    }


def train_one_model(model: nn.Module, model_name: str, config: Dict[str, Any], logger_print=print) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """Train a single model and return (history, training_time, artifacts).
    History keys: train_losses, train_costs, val_costs, final_val_cost.
    """
    device = torch.device('cpu')
    model.to(device)
    model.train()

    # Optimizer for RL models; greedy baseline will ignore
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate']) if hasattr(model, 'parameters') else None

    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']

    n_customers = config['problem']['num_customers']
    capacity = config['problem']['vehicle_capacity']
    coord_range = config['problem']['coord_range']
    demand_range = config['problem']['demand_range']

    train_losses: List[float] = []
    train_costs: List[float] = []
    val_costs: List[float] = []

    start_time = time.time()

    for epoch in range(0, num_epochs + 1):
        # Generate a batch of instances (simple iid generation by seed)
        instances: List[Dict[str, Any]] = []
        for i in range(batch_size):
            instance = generate_cvrp_instance(n_customers, capacity, coord_range, demand_range, seed=epoch * 1000 + i)
            instances.append(instance)

        # Forward (sample) with gradients enabled for policy (logp, ent)
        routes, logp, ent = model(instances, max_steps=n_customers * config['inference']['max_steps_multiplier'],
                                  temperature=config['inference']['default_temperature'], greedy=False, config=config)
        # Compute average cost per customer over batch
        batch_costs = []
        for b in range(len(instances)):
            r = routes[b]
            validate_route(r, n_customers, f"{model_name}-TRAIN", instances[b])
            c = compute_route_cost(r, instances[b]['distances']) / n_customers
            batch_costs.append(c)
        train_cost = float(np.mean(batch_costs)) if batch_costs else 0.0
        train_costs.append(train_cost)

        # Real REINFORCE-style update (aligned with legacy snapshot)
        if optimizer is not None and model_name != 'GT-Greedy':
            optimizer.zero_grad()
            # Build per-sample cost tensor (per-customer already)
            costs_tensor = torch.tensor(batch_costs, dtype=torch.float32)
            baseline = costs_tensor.mean().detach()
            advantages = baseline - costs_tensor  # lower cost -> positive advantage
            # Entropy regularization schedule (cosine decay)
            start_c = float(config.get('training_advanced', {}).get('entropy_coef', 0.0))
            end_c = float(config.get('training_advanced', {}).get('entropy_min', 0.0))
            if num_epochs > 1:
                cosine_factor = 0.5 * (1 + np.cos(np.pi * (epoch - 1) / (num_epochs - 1)))
            else:
                cosine_factor = 0.0
            entropy_coef = end_c + (start_c - end_c) * cosine_factor
            # logp and ent come per-sample from model forward
            loss = (-(advantages) * logp).mean() - entropy_coef * ent.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get('training_advanced', {}).get('gradient_clip_norm', 1.0)))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().numpy()))
        else:
            train_losses.append(float('nan'))

        # Validation every N epochs
        if (epoch % config['training']['validation_frequency']) == 0 or epoch == num_epochs:
            with torch.no_grad():
                val_instances = [generate_cvrp_instance(n_customers, capacity, coord_range, demand_range, seed=42 + j)
                                 for j in range(batch_size)]
                routes_g, _, _ = model(val_instances, max_steps=n_customers * config['inference']['max_steps_multiplier'],
                                        temperature=config['inference']['default_temperature'], greedy=True, config=config)
                costs = []
                for b in range(len(val_instances)):
                    r = routes_g[b]
                    validate_route(r, n_customers, f"{model_name}-VAL", val_instances[b])
                    c = compute_route_cost(r, val_instances[b]['distances']) / n_customers
                    costs.append(c)
                val_cost = float(np.mean(costs)) if costs else train_cost
                val_costs.append(val_cost)
                logger_print(f"[{model_name}] Epoch {epoch:03d}: train={train_cost:.4f}, val={val_cost:.4f}")

    training_time = time.time() - start_time

    history = {
        'train_losses': train_losses,
        'train_costs': train_costs,
        'val_costs': val_costs,
        'final_val_cost': val_costs[-1] if val_costs else (train_costs[-1] if train_costs else 0.0),
    }
    artifacts = {}
    return history, training_time, artifacts

def train_all_models(config: Dict[str, Any], model_builder, logger_print=print) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, nn.Module]]:
    """Train all configured models using provided model_builder(name)->nn.Module.
    Returns (results, training_times, models) where results[name] has 'history'.
    """
    models_to_train = [
        'Pointer+RL',
        'GT-Greedy',
        'GT+RL',
        'DGT+RL',
        'GAT+RL',
    ]
    results: Dict[str, Any] = {}
    training_times: Dict[str, float] = {}
    models: Dict[str, nn.Module] = {}
    for name in models_to_train:
        model = model_builder(name)
        hist, t, art = train_one_model(model, name, config, logger_print)
        results[name] = {'history': hist}
        training_times[name] = t
        models[name] = model
    return results, training_times, models

