#!/usr/bin/env python3
"""Direct benchmark to identify the bottleneck"""
import time
import sys
import os
import torch
import numpy as np
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

from src.generator.generator import create_data_generator
from src.models.model_factory import ModelFactory
from src.metrics.costs import compute_route_cost

print("="*60)
print("DIRECT PERFORMANCE BENCHMARK")
print("Testing raw performance without training framework")
print("="*60)

# Setup
device = torch.device('cuda:0')
config = {'problem': {'num_customers': 10, 'vehicle_capacity': 30}}

# Create generator and model
generator = create_data_generator(config)
model = ModelFactory.create_model(
    'GT+RL', 
    config={'problem': {'num_customers': 10, 'vehicle_capacity': 30},
            'model': {'hidden_dim': 128, 'num_heads': 4, 'num_layers': 3}}
).to(device)

print(f"Model on {device}, parameters: {sum(p.numel() for p in model.parameters()):,}")

# Simple instance preparation
def prep_batch(instances):
    batch = []
    for inst in instances:
        gpu_inst = {}
        for k, v in inst.items():
            if k == 'distances':
                gpu_inst[k] = v  # Keep on CPU
            elif isinstance(v, np.ndarray):
                gpu_inst[k] = torch.tensor(
                    v, 
                    dtype=torch.long if k == 'demands' else torch.float32,
                    device=device
                )
            else:
                gpu_inst[k] = v
        batch.append(gpu_inst)
    return batch

# Warmup
print("\nWarming up...")
for _ in range(2):
    data = generator(32, epoch=0, seed=0)
    gpu_data = prep_batch(data)
    with torch.no_grad():
        model(gpu_data, decode_type='greedy')
torch.cuda.synchronize()

# Time 25 batches (1 epoch worth)
print("\nTiming 25 batches of 128 instances...")
torch.cuda.synchronize()
start = time.time()

for i in range(25):
    # Generate
    instances = generator(128, epoch=0, seed=i*1000)
    
    # Prep (distances stay on CPU)
    gpu_instances = prep_batch(instances)
    
    # Inference
    with torch.no_grad():
        routes, _ = model(gpu_instances, decode_type='sampling', temperature=2.5)
    
    # Cost computation on CPU
    costs = [compute_route_cost(route, instances[j]['distances']) 
             for j, route in enumerate(routes)]
    
    if (i+1) % 5 == 0:
        elapsed = time.time() - start
        print(f"  Batch {i+1}/25: {elapsed:.1f}s")

torch.cuda.synchronize()
total = time.time() - start

print("\n" + "="*60)
print(f"RESULT: {total:.2f} seconds for 25 batches")
print(f"Target: 22-23 seconds")
if total <= 23:
    print("✓ RAW PERFORMANCE IS GOOD - issue is in training framework")
else:
    print(f"✗ STILL SLOW ({total:.1f}s) - fundamental performance issue")
print("="*60)
