#!/usr/bin/env python3
"""
Minimal CPU Comparative Study (quick and simple) with batching support
- Trains 4 models briefly on synthetic CVRP instances
- Computes naive baseline
- Produces a simple plot and a compact results table

Outputs:
- utils/plots/comparative_study_results_cpu_min.png
- results/cpu_comparative_study_results_min.csv
"""

import argparse
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

# Make local src importable
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from models import (
    GreedyGraphTransformerBaseline,
    PointerRLModel,
    StaticRLGraphTransformer,
    DynamicGraphTransformerModel,
)
from utils.RL.euclidean_cost import euclidean_cost


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_instance(num_customers: int, capacity: float, device: torch.device, seed: int = None) -> Data:
    if seed is not None:
        np.random.seed(seed)
    n = num_customers + 1  # include depot

    # coords in [0, 20]/100 like Ver1
    coords = torch.randint(0, 21, (n, 2), device=device, dtype=torch.float32) / 100.0
    # demands: {0.1..1.0}, depot 0
    demand = torch.randint(1, 11, (n, 1), device=device, dtype=torch.float32) / 10.0
    demand[0] = 0.0

    # fully connected edges with euclidean distance as attr
    edge_idx = []
    edge_attr = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge_idx.append([i, j])
            edge_attr.append(torch.norm(coords[i] - coords[j]).unsqueeze(0))
    edge_index = torch.tensor(edge_idx, device=device).t().long() if edge_idx else torch.empty((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.stack(edge_attr) if edge_attr else torch.empty((0, 1), device=device)

    # IMPORTANT: graph_transformer concatenates x and demand internally, so keep x=coords
    return Data(
        x=coords,
        edge_index=edge_index,
        edge_attr=edge_attr,
        demand=demand,
        capacity=torch.full((n,), capacity, device=device),
        batch=torch.zeros(n, dtype=torch.long, device=device),
        num_graphs=1,
    )


def make_batch(graphs: List[Data]) -> Data:
    """Concatenate a list of PyG Data graphs into a single batch with proper batch vector and offset edge_index."""
    if not graphs:
        raise ValueError("Empty batch")
    if len(graphs) == 1:
        return graphs[0]

    device = graphs[0].x.device
    x_list, demand_list, capacity_list, batch_vec = [], [], [], []
    edge_index_list, edge_attr_list = [], []
    offset = 0
    for i, g in enumerate(graphs):
        n = g.x.size(0)
        x_list.append(g.x)
        demand_list.append(g.demand)
        capacity_list.append(g.capacity)
        batch_vec.append(torch.full((n,), i, dtype=torch.long, device=device))
        if g.edge_index.numel() > 0:
            edge_index_list.append(g.edge_index + offset)
            edge_attr_list.append(g.edge_attr)
        offset += n

    return Data(
        x=torch.cat(x_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2, 0), dtype=torch.long, device=device),
        edge_attr=torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.empty((0, 1), device=device),
        demand=torch.cat(demand_list, dim=0),
        capacity=torch.cat(capacity_list, dim=0),
        batch=torch.cat(batch_vec, dim=0),
        num_graphs=len(graphs),
    )


def actions_to_index_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Normalize a sequence to 1D LongTensor of node indices per step.
    Acceptable shapes:
      - [S] (already indices)
      - [S, N] (per-step scores or one-hot) -> argmax(-1)
    """
    if not isinstance(seq, torch.Tensor):
        seq = torch.tensor(seq)
    if seq.dim() == 1:
        return seq.long()
    elif seq.dim() == 2:
        return seq.argmax(dim=-1).long()
    else:
        # Fallback: flatten last dim if extra dims exist, then argmax
        return seq.view(seq.size(0), -1).argmax(dim=-1).long()


def split_actions_to_graph_seqs(actions: Any, batch_size: int) -> List[torch.Tensor]:
    """Convert model actions output to a list of per-graph 1D index sequences.
    Accepts:
      - list of length B: each element [S] or [S, N]
      - tensor [B, S] (indices)
      - tensor [B, S, N] (scores) -> argmax(-1)
      - tensor [S] or [S, N] (single graph)
    """
    seqs: List[torch.Tensor] = []
    if isinstance(actions, list):
        for a in actions:
            seqs.append(actions_to_index_sequence(a))
        return seqs
    if isinstance(actions, torch.Tensor):
        if actions.dim() == 3:
            idx = actions.argmax(dim=-1)  # [B, S]
            return [idx[i] for i in range(idx.size(0))]
        elif actions.dim() == 2:
            return [actions[i] for i in range(actions.size(0))]
        elif actions.dim() == 1:
            return [actions]
    # Unknown structure, best-effort fallback: wrap as single sequence
    return [actions_to_index_sequence(actions)]


def tour_cost_from_indices(idx_seq: torch.Tensor, coords_np: np.ndarray) -> float:
    tour = [0]
    for a in idx_seq:
        ai = int(a.item())
        if ai != 0 and ai not in tour:
            tour.append(ai)
    if tour[-1] != 0:
        tour.append(0)
    return float(euclidean_cost(tour, coords_np))


def eval_instance_cost(model, instance: Data, steps: int) -> float:
    model.eval()
    with torch.no_grad():
        actions, _ = model(instance, steps, greedy=True)
        seqs = split_actions_to_graph_seqs(actions, batch_size=1)
        seq = actions_to_index_sequence(seqs[0]).unsqueeze(0)
        costs = euclidean_cost(instance.x, seq, instance)
        return float(costs[0].item())


def train_quick(model, train_set: List[Data], val_set: List[Data], num_customers: int, epochs: int, lr: float, batch_size: int = 1) -> Dict[str, Any]:
    # Minimal trainer supporting batching; uses a simple REINFORCE-like signal if log_probs are returned.
    device = train_set[0].x.device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"train_costs": [], "val_costs": [], "final_val_cost": None}
    steps = num_customers * 2

    for _ep in range(epochs):
        model.train()
        epoch_costs = []
        for start in range(0, len(train_set), batch_size):
            batch_graphs = train_set[start:start + batch_size]
            batch_data = make_batch(batch_graphs)

            optimizer.zero_grad()
            actions, log_probs = model(batch_data, steps, greedy=False)

            # Normalize to per-graph index sequences
            seqs = split_actions_to_graph_seqs(actions, batch_size=len(batch_graphs))

            # Compute batch costs via euclidean_cost
            seq_tensor = torch.stack([actions_to_index_sequence(s) for s in seqs], dim=0).to(device)
            costs_t = euclidean_cost(batch_data.x, seq_tensor, batch_data)
            epoch_costs.append(float(costs_t.mean().item()))

            # Simple loss from log_probs and costs (only if differentiable)
            if isinstance(log_probs, torch.Tensor) and log_probs.requires_grad:
                if log_probs.dim() == 2 and log_probs.size(0) == len(seqs):  # [B, S]
                    lp_per_graph = log_probs.sum(dim=1)
                    loss = (lp_per_graph * costs_t).mean()
                else:
                    # assume single sequence
                    loss = log_probs.sum() * costs_t.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                # Greedy/non-differentiable path: skip backprop
                pass

        hist["train_costs"].append(float(np.mean(epoch_costs)))

        # Validation
        val_costs = [eval_instance_cost(model, v, steps) for v in val_set]
        hist["val_costs"].append(float(np.mean(val_costs)))

    hist["final_val_cost"] = hist["val_costs"][-1]
    return hist


def compute_naive_baseline(val_set: List[Data]) -> float:
    costs = []
    for inst in val_set:
        coords = inst.x.detach().cpu().numpy()
        depot = coords[0]
        customers = coords[1:]
        c = 0.0
        for cust in customers:
            c += 2 * np.linalg.norm(depot - cust)
        costs.append(c)
    return float(np.mean(costs))


def main():
    ap = argparse.ArgumentParser(description="Minimal CPU comparative study")
    ap.add_argument("--num_customers", type=int, default=20)
    ap.add_argument("--capacity", type=float, default=3.0)
    ap.add_argument("--train_instances", type=int, default=20)
    ap.add_argument("--val_instances", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cpu")
    set_seed(args.seed)

    # data
    train_set = [make_instance(args.num_customers, args.capacity, device, seed=i) for i in range(args.train_instances)]
    val_set = [make_instance(args.num_customers, args.capacity, device, seed=10_000 + i) for i in range(args.val_instances)]

    # naive baseline (validation)
    naive_cost = compute_naive_baseline(val_set)
    naive_per_cust = naive_cost / args.num_customers

    # models
    models = {
        "greedy_baseline": GreedyGraphTransformerBaseline(node_input_dim=3, edge_input_dim=1, hidden_dim=64).to(device),
        "pointer_rl": PointerRLModel(node_input_dim=3, edge_input_dim=1, hidden_dim=64).to(device),
        "static_rl": StaticRLGraphTransformer(node_input_dim=3, edge_input_dim=1, hidden_dim=64).to(device),
        "dynamic_gt_rl": DynamicGraphTransformerModel(node_input_dim=3, edge_input_dim=1, hidden_dim=64).to(device),
    }

    results: Dict[str, Dict[str, Any]] = {}
    train_times: Dict[str, float] = {}

    for name, model in models.items():
        t0 = time.time()
        hist = train_quick(model, train_set, val_set, args.num_customers, args.epochs, args.lr, batch_size=args.batch_size)
        t1 = time.time()
        results[name] = hist
        train_times[name] = t1 - t0

    # make table
    rows = []
    for name in models.keys():
        train_cust = results[name]["train_costs"][-1] / args.num_customers
        val_cust = results[name]["final_val_cost"] / args.num_customers
        rows.append({
            "Model": name,
            "CPU Train/Cust": f"{train_cust:.3f}",
            "CPU Time (s)": f"{train_times[name]:.3f}",
            "Val/Cust": f"{val_cust:.3f}",
        })
    rows.append({
        "Model": "naive_baseline",
        "CPU Train/Cust": f"{naive_per_cust:.3f}",
        "CPU Time (s)": "—",
        "Val/Cust": f"{naive_per_cust:.3f}",
    })

    df = pd.DataFrame(rows)
    os.makedirs(project_root / 'results', exist_ok=True)
    csv_path = project_root / 'results' / 'cpu_comparative_study_results_min.csv'
    df.to_csv(csv_path, index=False)
    print("\nRESULTS TABLE")
    print(df.to_string(index=False))
    print(f"\nSaved table to {csv_path}")

    # simple plot: bars for Val/Cust and a secondary bar for CPU Time
    os.makedirs(project_root / 'utils' / 'plots', exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    names = [r["Model"] for r in rows]
    val_vals = [float(r["Val/Cust"]) for r in rows]
    time_vals = [0.0 if r["CPU Time (s)"] == "—" else float(r["CPU Time (s)"]) for r in rows]

    x = np.arange(len(names))
    w = 0.35
    ax1.bar(x - w/2, val_vals, width=w, label='Val/Cust')
    ax1.set_ylabel('Average Cost per Customer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right')

    ax2 = ax1.twinx()
    ax2.bar(x + w/2, time_vals, width=w, color='orange', label='CPU Time (s)')
    ax2.set_ylabel('CPU Time (s)')

    ax1.set_title('CPU Comparative Study (Minimal)')
    # Build a joint legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper center')

    out_path = project_root / 'utils' / 'plots' / 'comparative_study_results_cpu_min.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
