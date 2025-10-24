#!/usr/bin/env python3
"""Benchmark GPU vs CPU cost computation"""
import time
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
from src.metrics.costs import compute_route_cost

device = torch.device('cuda:0')
n_customers = 20
batch_size = 128
n_iterations = 100

print("="*60)
print("GPU vs CPU Cost Computation Benchmark")
print(f"Config: n={n_customers}, batch_size={batch_size}, iterations={n_iterations}")
print("="*60)

# Generate test data
np.random.seed(42)
distances_np = np.random.rand(batch_size, n_customers+1, n_customers+1).astype(np.float32)
routes = [[0] + list(np.random.permutation(range(1, n_customers+1))) + [0] for _ in range(batch_size)]

# CPU approach (current)
print("\n1. CPU Cost Computation (current implementation):")
start = time.time()
for _ in range(n_iterations):
    costs_cpu = []
    for i in range(batch_size):
        cost = compute_route_cost(routes[i], distances_np[i])
        costs_cpu.append(cost)
cpu_time = time.time() - start
print(f"   Time: {cpu_time:.3f}s")

# GPU approach - vectorized
print("\n2. GPU Cost Computation (vectorized):")
distances_gpu = torch.tensor(distances_np, device=device)
routes_tensor = torch.tensor(routes, device=device, dtype=torch.long)

torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    # Vectorized GPU computation
    from_idx = routes_tensor[:, :-1]
    to_idx = routes_tensor[:, 1:]
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, len(routes[0])-1)
    edge_costs = distances_gpu[batch_idx, from_idx, to_idx]
    costs_gpu = edge_costs.sum(dim=1)
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"   Time: {gpu_time:.3f}s")

# Hybrid: distances on GPU, routes on CPU, transfer costs back
print("\n3. Hybrid (distances GPU, compute on GPU, transfer back):")
torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    routes_gpu = torch.tensor(routes, device=device, dtype=torch.long)
    from_idx = routes_gpu[:, :-1]
    to_idx = routes_gpu[:, 1:]
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, len(routes[0])-1)
    edge_costs = distances_gpu[batch_idx, from_idx, to_idx]
    costs = edge_costs.sum(dim=1)
    costs_cpu = costs.cpu().numpy()  # Transfer back for use in training
torch.cuda.synchronize()
hybrid_time = time.time() - start
print(f"   Time: {hybrid_time:.3f}s")

print("\n" + "="*60)
print("RESULTS:")
print(f"  CPU:    {cpu_time:.3f}s (baseline)")
print(f"  GPU:    {gpu_time:.3f}s ({cpu_time/gpu_time:.1f}x faster)")
print(f"  Hybrid: {hybrid_time:.3f}s ({cpu_time/hybrid_time:.1f}x faster)")
print("\nConclusion: ", end="")
if gpu_time < cpu_time:
    print("GPU is faster! Should move cost computation to GPU.")
else:
    print("CPU is faster for this problem size.")
print("="*60)
