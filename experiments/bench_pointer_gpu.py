#!/usr/bin/env python3
"""
Microbenchmarks for Pointer+RL GPU acceleration.
Measures per-block speed on CPU vs GPU (if available).
"""
import time
import math
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.models.graph_transformer import GraphTransformerEncoder
from src.models.GAT_Decoder import GAT_Decoder
from src.utils.RL.euclidean_cost import euclidean_cost


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timer(fn, *args, repeat=5, warmup=2, **kwargs):
    # Warmup
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            out = out.sum().item()
    # Timed runs
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        sync_cuda()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


def make_synthetic_batch(batch_size=32, num_nodes=50, device="cpu"):
    # coordinates and demand
    coords = torch.rand(batch_size * num_nodes, 2, device=device) * 100
    demands = torch.rand(batch_size * num_nodes, 1, device=device)

    # complete directed graph edges
    edge_indices = []
    edge_attrs = []
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                s = b * num_nodes + i
                t = b * num_nodes + j
                edge_indices.append([s, t])
                distance = torch.norm(coords[s] - coords[t], dim=0, keepdim=True)
                edge_attrs.append(distance)
    edge_index = torch.tensor(edge_indices, device=device).t().long()
    edge_attr = torch.stack(edge_attrs).to(device)
    batch_vec = torch.repeat_interleave(torch.arange(batch_size, device=device), num_nodes)

    data = Data(x=coords, edge_index=edge_index, edge_attr=edge_attr, demand=demands, batch=batch_vec)
    data.num_graphs = batch_size

    # capacity per instance
    capacity = torch.full((batch_size, 1), 20.0, device=device)
    demand_per_node = demands.view(batch_size, num_nodes)

    return data, capacity, demand_per_node


def bench_pointer_rl(device="cpu", hidden_dim=128, n_layers=3, n_heads=8, n_steps=50, batch_size=32, num_nodes=50):
    data, capacity, demand = make_synthetic_batch(batch_size=batch_size, num_nodes=num_nodes, device=device)

    encoder = GraphTransformerEncoder(node_input_dim=3, edge_input_dim=1, hidden_dim=hidden_dim, num_heads=n_heads, num_layers=n_layers).to(device)
    decoder = GAT_Decoder(hidden_dim, hidden_dim).to(device)

    # B1: Encoder forward
    def run_encoder():
        return encoder(data)

    t_enc = timer(run_encoder)
    x = encoder(data)
    pool = x.mean(dim=1)

    # B2+B3: Decoder full episode (includes pointer attention per step)
    T = 1.0
    def run_decoder():
        actions, logp = decoder(x, pool, capacity, demand, n_steps=n_steps, T=T, greedy=False)
        return logp

    t_dec = timer(run_decoder)
    actions, logp = decoder(x, pool, capacity, demand, n_steps=n_steps, T=T, greedy=True)

    # B4: Euclidean cost
    def run_cost():
        return euclidean_cost(data.x, actions, data)

    t_cost = timer(run_cost)

    # Aggregate
    return {
        "device": device,
        "encoder_s": t_enc,
        "decoder_s": t_dec,
        "cost_s": t_cost,
        "total_s": t_enc + t_dec + t_cost,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    results = []

    # CPU
    results.append(bench_pointer_rl(device="cpu"))

    # GPU (if available)
    if torch.cuda.is_available():
        results.append(bench_pointer_rl(device="cuda"))

    # Print summary
    for r in results:
        print(f"Device={r['device']}: encoder={r['encoder_s']:.4f}s, decoder={r['decoder_s']:.4f}s, cost={r['cost_s']:.4f}s, total={r['total_s']:.4f}s")

    if len(results) == 2:
        cpu, gpu = results
        speedup = cpu['total_s'] / gpu['total_s']
        print(f"Overall speedup ~ {speedup:.2f}x")

