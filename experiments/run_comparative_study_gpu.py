#!/usr/bin/env python3
"""
GPU Comparative Study for 4 Models (CVRP)
- Greedy baseline (Transformer greedy)
- PointerRL (Transformer encoder + Pointer decoder)
- StaticRL (Transformer + RL decoder)
- DynamicGT RL (Transformer + RL + dynamic updates)
Includes a naive baseline in the top-right plot for reference.
"""
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data

# Ensure project root on path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models import (
    GreedyGraphTransformerBaseline,
    PointerRLModel,
    StaticRLGraphTransformer,
    DynamicGraphTransformerModel,
)
from src.utils.RL.euclidean_cost import euclidean_cost


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_instance(num_nodes: int, capacity: float, device: torch.device) -> Data:
    # Coordinates in unit square on a discrete 1..100 grid, divided by 100
    coords = (torch.randint(1, 101, (num_nodes, 2), device=device, dtype=torch.int64).float() / 100.0)
    # Customer demands sampled from {0.1, 0.2, ..., 1.0}; depot demand = 0
    demands = (torch.randint(1, 11, (num_nodes, 1), device=device, dtype=torch.int64).float() / 10.0)
    demands[0] = 0.0  # depot demand = 0

    # fully connected directed edges
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
    # Concatenate instances into a single batch manually
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


def naive_roundtrip_cost(coords_np: np.ndarray) -> float:
    """Naive baseline: serve each customer individually.
    Route: depot -> customer i -> depot, for all i >= 1.
    Returns total distance across the full route set.
    """
    depot = coords_np[0]
    # Sum of 2 * distance(depot, customer) for each customer
    dists = np.linalg.norm(coords_np[1:] - depot, axis=1)
    return float(2.0 * dists.sum())


def evaluate_model(model_name: str, data: Data, n_steps: int, device: torch.device, use_amp: bool = False) -> Dict[str, float]:
    t0 = time.perf_counter()
    if model_name == 'greedy_baseline':
        model = GreedyGraphTransformerBaseline().to(device)
    elif model_name == 'pointer_rl':
        model = PointerRLModel().to(device)
    elif model_name == 'static_rl':
        model = StaticRLGraphTransformer().to(device)
    elif model_name == 'dynamic_gt_rl':
        model = DynamicGraphTransformerModel().to(device)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model.eval()
    with torch.no_grad():
        # Use AMP only for heavy models; force AMP disabled for greedy_baseline
        if use_amp and device.type == 'cuda' and model_name != 'greedy_baseline':
            from torch.cuda.amp import autocast
            with autocast():
                actions, logp = model(data, n_steps=n_steps, greedy=True)
        elif model_name == 'greedy_baseline' and device.type == 'cuda':
            # Explicitly disable autocast even if a higher context is active
            try:
                from torch.amp import autocast as amp_autocast
                with amp_autocast('cuda', enabled=False):
                    actions, logp = model(data, n_steps=n_steps, greedy=True)
            except Exception:
                actions, logp = model(data, n_steps=n_steps, greedy=True)
        else:
            actions, logp = model(data, n_steps=n_steps, greedy=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Compute cost per batch instance and average per customer
    cost = euclidean_cost(data.x, actions, data)  # shape: [batch]
    # Derive customers per instance (nodes minus depot)
    num_nodes = int(data.x.size(0) // data.num_graphs)
    customers = max(num_nodes - 1, 1)
    cost_per_customer = cost / customers
    avg_cost_per_customer = float(cost_per_customer.mean().item())
    return {"avg_cost_per_customer": avg_cost_per_customer, "time": t1 - t0}


def run_experiment(device: torch.device, problem_sizes: List[int], instances: int, runs: int, seed: int, capacity: float = 50.0, use_amp: bool = False):
    set_seed(seed)
    results_rows = []

    models = ['greedy_baseline', 'pointer_rl', 'static_rl', 'dynamic_gt_rl']

    for n_nodes in problem_sizes:
        print(f"\n=== Problem size: {n_nodes} ===")
        # Generate instances (on device) and batch them for throughput
        per_batch = min(32, instances)  # batch size cap for memory safety
        generated = 0
        model_aggregates = {m: {"costs": [], "times": []} for m in models}
        naive_costs = []

        while generated < instances:
            b = min(per_batch, instances - generated)
            inst_list = [make_instance(n_nodes, capacity, device) for _ in range(b)]
            batch_data = make_batch(inst_list)
            n_steps = n_nodes + 5

            # Evaluate each model on the batch
            for m in models:
                out = evaluate_model(m, batch_data, n_steps, device, use_amp=use_amp)
                model_aggregates[m]["costs"].append(out["avg_cost_per_customer"])
                model_aggregates[m]["times"].append(out["time"])

            # Naive baseline cost computed per instance (CPU numpy) for top-right plot
            for d in inst_list:
                coords = d.x.detach().cpu().numpy()
                total_cost = naive_roundtrip_cost(coords)
                customers = max(coords.shape[0] - 1, 1)
                naive_costs.append(total_cost / customers)

            generated += b

        # Aggregate across all batches
        for m in models:
            avg_cost = float(np.mean(model_aggregates[m]["costs"]))
            avg_time = float(np.mean(model_aggregates[m]["times"]))
            results_rows.append({
                "variant": m,
                "problem_size": n_nodes,
                "avg_solution_cost_per_customer": avg_cost,
                "avg_computation_time": avg_time,
            })
        # Add naive baseline as separate variant for plotting
        results_rows.append({
            "variant": "naive_baseline",
            "problem_size": n_nodes,
            "avg_solution_cost_per_customer": float(np.mean(naive_costs)) if naive_costs else np.nan,
            "avg_computation_time": np.nan,
        })

    df = pd.DataFrame(results_rows)
    return df


def generate_plots(df: pd.DataFrame, out_dir: Path, problem_sizes: List[int]):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8')

    # 2x2 figure; top-right includes naive baseline
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPU Comparative Study (4 Models)')

    # Top-left: Average solution cost per customer vs problem size (4 models only)
    ax = axes[0, 0]
    plot_df = df[df['variant'].isin(['greedy_baseline', 'pointer_rl', 'static_rl', 'dynamic_gt_rl'])]
    for v, label in [
        ('greedy_baseline', 'Greedy Baseline'),
        ('pointer_rl', 'Pointer+RL'),
        ('static_rl', 'Transformer+RL'),
        ('dynamic_gt_rl', 'DynamicGT+RL')
    ]:
        vd = plot_df[plot_df['variant'] == v]
        ax.plot(vd['problem_size'], vd['avg_solution_cost_per_customer'], marker='o', label=label)
    ax.set_title('Avg Solution Cost per Customer vs Problem Size')
    ax.set_xlabel('Problem Size (nodes)')
    ax.set_ylabel('Avg Solution Cost per Customer')
    ax.legend()

    # Top-right: Include naive baseline for reference (per customer)
    ax = axes[0, 1]
    plot_df2 = df[df['variant'].isin(['greedy_baseline', 'pointer_rl', 'static_rl', 'dynamic_gt_rl', 'naive_baseline'])]
    for v, label in [
        ('naive_baseline', 'Naive (NN) Baseline'),
        ('greedy_baseline', 'Greedy Baseline'),
        ('pointer_rl', 'Pointer+RL'),
        ('static_rl', 'Transformer+RL'),
        ('dynamic_gt_rl', 'DynamicGT+RL')
    ]:
        vd = plot_df2[plot_df2['variant'] == v]
        ax.plot(vd['problem_size'], vd['avg_solution_cost_per_customer'], marker='o', label=label)
    ax.set_title('Avg Cost per Customer with Naive Baseline (Top-Right)')
    ax.set_xlabel('Problem Size (nodes)')
    ax.set_ylabel('Avg Solution Cost per Customer')
    ax.legend()

    # Bottom-left: Computation time vs problem size (models only)
    ax = axes[1, 0]
    for v, label in [
        ('greedy_baseline', 'Greedy Baseline'),
        ('pointer_rl', 'Pointer+RL'),
        ('static_rl', 'Transformer+RL'),
        ('dynamic_gt_rl', 'DynamicGT+RL')
    ]:
        vd = plot_df[plot_df['variant'] == v]
        ax.plot(vd['problem_size'], vd['avg_computation_time'], marker='s', label=label)
    ax.set_title('Avg Computation Time vs Problem Size')
    ax.set_xlabel('Problem Size (nodes)')
    ax.set_ylabel('Avg Time (s)')
    ax.legend()

    # Bottom-right: Bar chart of cost per customer at largest problem size with naive baseline
    ax = axes[1, 1]
    max_size = max(problem_sizes)
    br = df[df['problem_size'] == max_size]
    br = br[br['variant'].isin(['naive_baseline', 'greedy_baseline', 'pointer_rl', 'static_rl', 'dynamic_gt_rl'])]
    ax.bar(br['variant'], br['avg_solution_cost_per_customer'])
    ax.set_title(f'Cost per Customer at {max_size} Nodes (incl. Naive)')
    ax.set_xticklabels(br['variant'], rotation=45, ha='right')

    plt.tight_layout()
    fig_path = out_dir / 'comparative_study_gpu.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='GPU Comparative Study for 4 Models (CVRP)')
    # Enforce CUDA-only execution; no device flag is accepted here
    parser.add_argument('--problem_sizes', type=int, nargs='+', default=[20, 50], help='List of node counts to test')
    parser.add_argument('--instances', type=int, default=50, help='Number of instances per problem size')
    parser.add_argument('--runs', type=int, default=1, help='Runs per instance (kept 1 for speed)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--capacity', type=float, default=3.0)
    parser.add_argument('--out_dir', type=str, default='results_gpu')
    parser.add_argument('--amp', action='store_true', help='Enable AMP (autocast) for GPU inference')
    args = parser.parse_args()

    # Enforce CUDA GPU; no fallbacks
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for experiments/run_comparative_study_gpu.py; no CPU/MPS fallback.')
    device = torch.device('cuda')

    print(f"Device: {device}")
    exp_t0 = time.perf_counter()
    df = run_experiment(device, args.problem_sizes, args.instances, args.runs, args.seed, args.capacity, use_amp=args.amp)
    torch.cuda.synchronize()
    exp_t1 = time.perf_counter()
    print(f"Total experiment time: {exp_t1 - exp_t0:.2f}s")

    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'comparative_study_gpu.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    generate_plots(df, out_dir, args.problem_sizes)


if __name__ == '__main__':
    main()

