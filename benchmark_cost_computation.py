#!/usr/bin/env python3
"""Benchmark CPU vs GPU cost computation to find the performance leak"""
import time
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

from src.metrics.costs import compute_route_cost
from src.metrics.gpu_costs import compute_route_cost_gpu

def benchmark_cost_computation():
    device = torch.device("cuda:0")
    
    # Generate test data (similar to tiny_1 config)
    batch_size = 128
    n_customers = 10
    n_total = n_customers + 1  # Including depot
    
    routes = []
    distances_cpu = []
    distances_gpu = []
    
    # Generate test routes and distance matrices
    for _ in range(batch_size):
        # Random route
        route = [0] + list(np.random.permutation(range(1, n_total))) + [0]
        routes.append(route)
        
        # Random distance matrix
        dist = np.random.rand(n_total, n_total) * 100
        dist = (dist + dist.T) / 2  # Make symmetric
        np.fill_diagonal(dist, 0)
        
        distances_cpu.append(dist)
        distances_gpu.append(torch.tensor(dist, dtype=torch.float32, device=device))
    
    # Benchmark CPU version
    times_cpu = []
    for run in range(5):
        start = time.time()
        for i in range(batch_size):
            cost = compute_route_cost(routes[i], distances_cpu[i])
        end = time.time()
        times_cpu.append(end - start)
    
    # Benchmark GPU version
    torch.cuda.synchronize()
    times_gpu = []
    for run in range(5):
        start = time.time()
        for i in range(batch_size):
            cost = compute_route_cost_gpu(routes[i], distances_gpu[i])
        torch.cuda.synchronize()
        end = time.time()
        times_gpu.append(end - start)
    
    # Benchmark GPU version with tensor conversions (like in trainer)
    torch.cuda.synchronize()
    times_gpu_with_conversions = []
    for run in range(5):
        start = time.time()
        costs = []
        for i in range(batch_size):
            rc = compute_route_cost_gpu(routes[i], distances_gpu[i])
            if not isinstance(rc, torch.Tensor):
                rc = torch.tensor(rc, device=device, dtype=torch.float32)
            costs.append(rc)
            # Simulate the CPC computation like in trainer
            n_customers = 10
            cpc = rc / float(n_customers)
            # Extract value like trainer does
            cpc_value = cpc if isinstance(cpc, float) else cpc.item()
        torch.cuda.synchronize()
        end = time.time()
        times_gpu_with_conversions.append(end - start)
    
    # Report results
    cpu_avg = np.mean(times_cpu[1:]) * 1000  # Skip first run, convert to ms
    gpu_avg = np.mean(times_gpu[1:]) * 1000
    gpu_conv_avg = np.mean(times_gpu_with_conversions[1:]) * 1000
    
    print("Cost Computation Benchmark (batch_size=128)")
    print("=" * 50)
    print(f"CPU version:                {cpu_avg:.2f} ms")
    print(f"GPU version (simple):       {gpu_avg:.2f} ms")
    print(f"GPU version (with .item()): {gpu_conv_avg:.2f} ms")
    print()
    print(f"GPU speedup over CPU:       {cpu_avg/gpu_avg:.2f}x")
    print(f"GPU+conversions vs CPU:     {gpu_conv_avg/cpu_avg:.2f}x")
    
    if gpu_conv_avg > cpu_avg:
        print(f"❌ GPU with conversions is {gpu_conv_avg/cpu_avg:.2f}x SLOWER than CPU!")
        print("   The .item() calls and tensor conversions are killing performance!")
    else:
        print("✅ GPU version is faster")

if __name__ == "__main__":
    benchmark_cost_computation()
