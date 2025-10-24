#!/usr/bin/env python3
"""Diagnose where the time is actually spent"""
import time
import torch
import sys
import os
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

from src.generator.generator import create_data_generator
from src.models.gt import GraphTransformer

def time_components():
    device = torch.device("cuda:0")
    config = {
        'problem': {
            'num_customers': 10,
            'vehicle_capacity': 20,
            'coord_range': 100,
            'demand_min': 1,
            'demand_max': 10
        }
    }
    
    # Model
    model = GraphTransformer(hidden_dim=128, num_heads=4, num_layers=3).to(device)
    model.eval()
    
    # Generator
    data_generator = create_data_generator(config)
    
    batch_size = 128
    print("Timing individual components (batch_size=128, N=10):")
    print("=" * 60)
    
    # 1. Data generation
    start = time.time()
    instances = data_generator(batch_size, epoch=0, seed=42)
    gen_time = time.time() - start
    print(f"1. Data generation:          {gen_time*1000:.2f} ms")
    
    # 2. Move to GPU
    start = time.time()
    for inst in instances:
        for key in ['coords', 'demands', 'distances']:
            if key in inst:
                inst[key] = torch.tensor(inst[key], dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    gpu_transfer_time = time.time() - start
    print(f"2. Transfer to GPU:          {gpu_transfer_time*1000:.2f} ms")
    
    # 3. Model forward pass
    start = time.time()
    with torch.no_grad():
        routes, log_probs, entropy = model(
            instances,
            max_steps=100,
            temperature=1.0,
            greedy=False,
            config={'inference': {'max_steps_multiplier': 10}}
        )
    torch.cuda.synchronize()
    forward_time = time.time() - start
    print(f"3. Model forward pass:       {forward_time*1000:.2f} ms")
    
    # 4. Cost computation (CPU)
    from src.metrics.costs import compute_route_cost
    start = time.time()
    costs = []
    for i in range(batch_size):
        distances_cpu = instances[i]['distances'].cpu().numpy()
        cost = compute_route_cost(routes[i], distances_cpu)
        costs.append(cost)
    cost_time = time.time() - start
    print(f"4. Cost computation (CPU):   {cost_time*1000:.2f} ms")
    
    # Total
    total = gen_time + gpu_transfer_time + forward_time + cost_time
    print(f"\nTotal time:                  {total*1000:.2f} ms")
    print(f"Expected time for 25 batches: {total*25:.2f} seconds")
    
    print("\nBreakdown:")
    print(f"  Data generation:  {gen_time/total*100:.1f}%")
    print(f"  GPU transfer:     {gpu_transfer_time/total*100:.1f}%")
    print(f"  Model forward:    {forward_time/total*100:.1f}%")
    print(f"  Cost computation: {cost_time/total*100:.1f}%")

if __name__ == "__main__":
    time_components()
