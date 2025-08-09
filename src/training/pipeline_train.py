#!/usr/bin/env python3
"""
Generic training pipeline for CVRP models (PointerRL, StaticRL, DynamicGT) supporting:
- CPU, GPU, GPU+AMP training
- Deterministic instance generation (shared across pipelines via --data_seed)
- Per-epoch validation and final summary CSV
- Saving the minimal-cost validation route with a plot for manual inspection

This trainer uses the existing PyG-based models from src.models and the euclidean_cost.
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.data import Data

from src.models import (
    GreedyGraphTransformerBaseline,
    PointerRLModel,
    StaticRLGraphTransformer,
    DynamicGraphTransformerModel,
)
from src.utils.RL.euclidean_cost import euclidean_cost
from .validate_routes import validate_training_route

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed: int, deterministic: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_instance(num_nodes: int, capacity: float, device: torch.device, seed: int) -> Data:
    # Deterministic per-instance generation based on provided seed
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    coords = (torch.randint(1, 101, (num_nodes, 2), generator=g, device=device, dtype=torch.int64).float() / 100.0)
    demands = (torch.randint(1, 11, (num_nodes, 1), generator=g, device=device, dtype=torch.int64).float())
    demands[0] = 0.0

    edge_idx = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            edge_idx.append([i, j])
            d = torch.norm(coords[i] - coords[j], dim=0, keepdim=True)
            edge_attr.append(d)
    edge_index = torch.tensor(edge_idx, device=device).t().long() if edge_idx else torch.empty((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.stack(edge_attr) if edge_attr else torch.empty((0, 1), device=device)

    data = Data(
        x=coords,
        edge_index=edge_index,
        edge_attr=edge_attr,
        demand=demands,
        capacity=torch.full((num_nodes,), capacity, device=device),
        batch=torch.zeros(num_nodes, dtype=torch.long, device=device),
    )
    data.num_graphs = 1
    return data


def make_batch(instances: List[Data]) -> Data:
    device = instances[0].x.device
    x_list, edge_index_list, edge_attr_list, demand_list, capacity_list, batch_vec = [], [], [], [], [], []
    offset = 0
    for b, d in enumerate(instances):
        n = d.x.size(0)
        x_list.append(d.x)
        demand_list.append(d.demand)
        capacity_list.append(d.capacity)
        batch_vec.append(torch.full((n,), b, dtype=torch.long, device=device))
        if d.edge_index.numel() > 0:
            edge_index_list.append(d.edge_index + offset)
            edge_attr_list.append(d.edge_attr)
        offset += n
    x = torch.cat(x_list, dim=0)
    demand = torch.cat(demand_list, dim=0)
    capacity = torch.cat(capacity_list, dim=0)
    batch = torch.cat(batch_vec, dim=0)
    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), device=device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, demand=demand, capacity=capacity, batch=batch)
    data.num_graphs = len(instances)
    return data


def get_model(model_name: str, device: torch.device):
    if model_name == 'greedy_baseline':
        return GreedyGraphTransformerBaseline().to(device)
    if model_name == 'pointer_rl':
        return PointerRLModel().to(device)
    if model_name == 'static_rl':
        return StaticRLGraphTransformer().to(device)
    if model_name == 'dynamic_gt_rl':
        return DynamicGraphTransformerModel().to(device)
    raise ValueError(f'Unknown model: {model_name}')


def evaluate_batch(model, batch_data: Data, n_steps: int, device: torch.device, use_amp: bool, greedy: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns actions, log_probs, costs (per instance)
    model.eval() if greedy else model.train()
    if use_amp and device.type == 'cuda' and getattr(model, 'training', False):
        from torch.cuda.amp import autocast
        with autocast():
            actions, logp = model(batch_data, n_steps=n_steps, greedy=greedy)
    else:
        actions, logp = model(batch_data, n_steps=n_steps, greedy=greedy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    costs = euclidean_cost(batch_data.x, actions.detach(), batch_data)  # [batch]
    return actions, logp, costs


def save_best_route_plot(coords: np.ndarray, route: List[int], out_png: Path, out_json: Path):
    # Save JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open('w') as f:
        json.dump({'route': route, 'coords': coords.tolist()}, f)

    # Plot
    plt.figure(figsize=(5, 5))
    xs, ys = coords[:, 0], coords[:, 1]
    plt.scatter(xs[1:], ys[1:], c='blue', s=20, label='Customers')
    plt.scatter(xs[0:1], ys[0:1], c='red', s=50, label='Depot')
    # Draw route
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        plt.plot([xs[a], xs[b]], [ys[a], ys[b]], 'g-', alpha=0.8)
    plt.title('Best Validation Route')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def train_pipeline(
    model_name: str,
    device: torch.device,
    use_amp: bool,
    customers: int,
    instances: int,
    epochs: int,
    batch_size: int,
    capacity: float,
    lr: float,
    data_seed: int,
    out_dir: Path,
) -> Dict[str, float]:
    set_seed(data_seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(model_name, device)
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')

    # Prepare deterministic per-instance seeds list
    inst_seeds = [data_seed + i for i in range(instances)]

    n_steps = customers + 6  # slightly above nodes to ensure depot closure

    best_val_cost = float('inf')
    best_val_route: List[int] = []
    best_val_coords: np.ndarray = np.empty((customers + 1, 2), dtype=float)

    # We'll use a fixed validation set of size min(128, instances//5)
    val_count = max(32, min(128, instances // 5))
    val_seeds = inst_seeds[:val_count]

    # Training loop
    for epoch in range(epochs):
        model.train()
        start_time = time.perf_counter()
        # Iterate over remaining seeds for training (skip val seeds)
        train_seeds = inst_seeds[val_count:]
        # Batch over seeds
        for i in range(0, len(train_seeds), batch_size):
            batch_seeds = train_seeds[i:i + batch_size]
            inst_list = [make_instance(customers + 1, capacity, device, s) for s in batch_seeds]
            batch_data = make_batch(inst_list)

            optimizer.zero_grad(set_to_none=True)
            if use_amp and device.type == 'cuda':
                from torch.cuda.amp import autocast
                with autocast():
                    actions, logp = model(batch_data, n_steps=n_steps, greedy=False)
                    costs = euclidean_cost(batch_data.x, actions.detach(), batch_data)  # [B]
                    # STRICT VALIDATION: Check training routes
                    validate_training_route(actions, batch_data.demand.view(batch_data.num_graphs, -1), capacity, i, f"Training epoch {epoch+1} batch {i+1} (AMP)")
                    baseline = costs.mean().detach()
                    advantages = (costs - baseline).detach()  # [B]
                    # Aggregate log-probs across steps to [B]
                    logp_sum = logp.sum(dim=1) if logp.dim() > 1 else logp.view(-1)
                    loss = (advantages * logp_sum).mean()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                actions, logp = model(batch_data, n_steps=n_steps, greedy=False)
                costs = euclidean_cost(batch_data.x, actions.detach(), batch_data)  # [B]
                baseline = costs.mean().detach()
                # STRICT VALIDATION: Check training routes
                validate_training_route(actions, batch_data.demand.view(batch_data.num_graphs, -1), capacity, i, f"Training epoch {epoch+1} batch {i+1}")
                advantages = (costs - baseline).detach()  # [B]
                logp_sum = logp.sum(dim=1) if logp.dim() > 1 else logp.view(-1)
                loss = (advantages * logp_sum).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        train_time = time.perf_counter() - start_time

        # Validation (greedy)
        model.eval()
        val_costs = []
        with torch.no_grad():
            for i in range(0, len(val_seeds), batch_size):
                batch_seeds = val_seeds[i:i + batch_size]
                inst_list = [make_instance(customers + 1, capacity, device, s) for s in batch_seeds]
                batch_data = make_batch(inst_list)
                actions, logp = model(batch_data, n_steps=n_steps, greedy=True)
                costs = euclidean_cost(batch_data.x, actions, batch_data)
                val_costs.append(costs.detach().cpu())
                # Track best route in this batch
                # Reconstruct route with depot at start and end
                # Ensure actions has shape [B, T] of long indices
                act = actions
                if act.dim() == 3 and act.size(-1) == 1:
                    act = act.squeeze(-1)
                act = act.long()
                depot = torch.zeros(act.size(0), 1, dtype=torch.long, device=act.device)
                with_depot = torch.cat([depot, act, depot], dim=1)
                for b in range(with_depot.size(0)):
                    route = with_depot[b].tolist()
                    cost_val = float(costs[b].item())
                    if cost_val < best_val_cost:
                        best_val_cost = cost_val
                        best_val_route = route
                        coords_np = batch_data.x.view(-1, 2).detach().cpu().numpy()
                        # For batched data, slice current graph
                        if hasattr(batch_data, 'ptr') and batch_data.ptr is not None:
                            start = int(batch_data.ptr[b].item())
                            end = int(batch_data.ptr[b + 1].item())
                            best_val_coords = coords_np[start:end]
                        else:
                            best_val_coords = coords_np

        val_cost_tensor = torch.cat(val_costs) if val_costs else torch.tensor([], dtype=torch.float32)
        val_cost_per_cust = float(val_cost_tensor.mean().item()) / max(1, customers)

        # Append to CSV
        csv_path = out_dir / 'train_history.csv'
        header_needed = not csv_path.exists()
        with csv_path.open('a') as f:
            if header_needed:
                f.write('epoch,train_time_s,val_cost_per_customer\n')
            f.write(f'{epoch},{train_time:.6f},{val_cost_per_cust:.6f}\n')

    # Save best route plot and json
    save_best_route_plot(best_val_coords, best_val_route, out_dir / 'best_route.png', out_dir / 'best_route.json')

    # Return summary
    return {
        'best_val_cost_per_customer': best_val_cost / max(1, customers),
    }


def parse_args():
    p = argparse.ArgumentParser(description='Unified CVRP training pipeline (CPU/GPU/AMP)')
    p.add_argument('--pipeline', choices=['cpu', 'gpu', 'gpu_amp'], default='cpu')
    p.add_argument('--model', choices=['pointer_rl', 'static_rl', 'dynamic_gt_rl'], default='dynamic_gt_rl')
    p.add_argument('--customers', type=int, default=20)
    p.add_argument('--instances', type=int, default=800)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--capacity', type=float, default=3.0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--data_seed', type=int, default=12345, help='Base seed to generate deterministic instances shared across pipelines')
    p.add_argument('--out_dir', type=str, default='results_train')
    return p.parse_args()


def main():
    args = parse_args()
    # Device and AMP selection
    if args.pipeline == 'cpu':
        device = torch.device('cpu')
        use_amp = False
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for gpu/gpu_amp pipelines')
        device = torch.device('cuda')
        use_amp = (args.pipeline == 'gpu_amp')

    out_dir = Path(args.out_dir) / f"{args.pipeline}_{args.model}_C{args.customers}_I{args.instances}_E{args.epochs}_B{args.batch}"

    summary = train_pipeline(
        model_name=args.model,
        device=device,
        use_amp=use_amp,
        customers=args.customers,
        instances=args.instances,
        epochs=args.epochs,
        batch_size=args.batch,
        capacity=args.capacity,
        lr=args.lr,
        data_seed=args.data_seed,
        out_dir=out_dir,
    )

    print(json.dumps({
        'pipeline': args.pipeline,
        'model': args.model,
        'customers': args.customers,
        'instances': args.instances,
        'epochs': args.epochs,
        'batch': args.batch,
        'best_val_cost_per_customer': summary['best_val_cost_per_customer'],
        'out_dir': str(out_dir),
    }, indent=2))


if __name__ == '__main__':
    main()
