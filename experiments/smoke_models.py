#!/usr/bin/env python3
"""
Smoke test for model modules: runs each model on synthetic data to ensure basic execution.
"""
import torch
from torch_geometric.data import Data
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models import (
    PointerRLModel,
    StaticRLGraphTransformer,
    DynamicGraphTransformerModel,
    GreedyGraphTransformerBaseline,
)


def make_data(batch_size=2, num_nodes=10, device="cpu"):
    coords = torch.rand(batch_size * num_nodes, 2, device=device) * 100
    demands = torch.rand(batch_size * num_nodes, 1, device=device) * 10

    edge_indices = []
    edge_attrs = []
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    s = b * num_nodes + i
                    t = b * num_nodes + j
                    edge_indices.append([s, t])
                    dist = torch.norm(coords[s] - coords[t], dim=0, keepdim=True)
                    edge_attrs.append(dist)
    edge_index = torch.tensor(edge_indices, device=device).t().long()
    edge_attr = torch.stack(edge_attrs).to(device)
    batch_vec = torch.repeat_interleave(torch.arange(batch_size, device=device), num_nodes)

    data = Data(x=coords, edge_index=edge_index, edge_attr=edge_attr, demand=demands, batch=batch_vec,
                capacity=torch.full((batch_size * num_nodes,), 50.0, device=device))
    data.num_graphs = batch_size
    return data


def run_smoke(device="cpu"):
    data = make_data(device=device)
    n_steps = 5

    models = {
        "pointer_rl": PointerRLModel(),
        "transformer_rl": StaticRLGraphTransformer(),
        "dynamic_gt_rl": DynamicGraphTransformerModel(),
        "greedy_baseline": GreedyGraphTransformerBaseline(),
    }

    for name, model in models.items():
        model = model.to(device)
        with torch.no_grad():
            actions, logp = model(data, n_steps=n_steps, greedy=True)
        print(name, actions.shape, logp.shape)


if __name__ == "__main__":
    run_smoke(device="cuda" if torch.cuda.is_available() else "cpu")

