#!/usr/bin/env python3
"""Diagnose where the time is actually spent - simplified"""
import time
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

def simple_benchmark():
    """Just benchmark the key operations"""
    device = torch.device("cuda:0")
    batch_size = 128
    n_nodes = 11  # 10 customers + depot
    
    print("Simple Component Timing (batch_size=128, N=10):")
    print("=" * 60)
    
    # 1. Data generation simulation
    start = time.time()
    distances = []
    for _ in range(batch_size):
        dist = np.random.rand(n_nodes, n_nodes)
        distances.append(dist)
    gen_time = time.time() - start
    print(f"1. Generate {batch_size} distance matrices:    {gen_time*1000:.2f} ms")
    
    # 2. Transfer to GPU
    start = time.time()
    distances_gpu = []
    for dist in distances:
        distances_gpu.append(torch.tensor(dist, dtype=torch.float32, device=device))
    torch.cuda.synchronize()
    transfer_time = time.time() - start
    print(f"2. Transfer to GPU:                    {transfer_time*1000:.2f} ms")
    
    # 3. Transfer back to CPU (for cost computation)
    start = time.time()
    for dist_gpu in distances_gpu:
        _ = dist_gpu.cpu().numpy()
    back_time = time.time() - start
    print(f"3. Transfer back to CPU:               {back_time*1000:.2f} ms")
    
    # 4. Simple cost computation on CPU
    from src.metrics.costs import compute_route_cost
    routes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0] for _ in range(batch_size)]
    
    start = time.time()
    for i in range(batch_size):
        cost = compute_route_cost(routes[i], distances[i])
    cpu_cost_time = time.time() - start
    print(f"4. CPU cost computation:               {cpu_cost_time*1000:.2f} ms")
    
    # Total overhead
    overhead = transfer_time + back_time
    print(f"\nGPU transfer overhead:                 {overhead*1000:.2f} ms")
    print(f"For 25 batches:                        {overhead*25:.2f} seconds")
    
    if overhead > 0.5:
        print("\n⚠️  GPU transfer overhead is significant!")
        print("    This could explain the performance degradation.")

if __name__ == "__main__":
    simple_benchmark()
