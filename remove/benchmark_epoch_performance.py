#!/usr/bin/env python3
"""Benchmark actual epoch performance to find bottlenecks"""
import time
import sys
import os
import torch
import numpy as np

sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
os.chdir('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

from src.generator.generator import create_data_generator
from src.models.model_factory import ModelFactory
from src.metrics.costs import compute_route_cost

print("="*60)
print("PERFORMANCE BENCHMARK")
print("Configuration: n=10, capacity=30, 25 batches x 128")
print("="*60)

# Configuration
config = {
    'problem': {'num_customers': 10, 'vehicle_capacity': 30},
    'model': {'hidden_dim': 128, 'num_heads': 4, 'num_layers': 3},
    'training': {'batch_size': 128, 'num_batches_per_epoch': 25}
}

# Setup
device = torch.device('cuda:0')
print(f"Using device: {device}")

# Create data generator
data_generator = create_data_generator(config)

# Create model
model = ModelFactory.create_model(
    'GT+RL',
    n_customers=10,
    model_config=config['model']
).to(device)
model.eval()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test function to prepare instances
def prep_instances(instances, device):
    prepped = []
    for inst in instances:
        new_inst = {}
        for key, val in inst.items():
            if key == 'distances':
                # Keep distances on CPU for cost computation
                new_inst[key] = val
            elif isinstance(val, np.ndarray):
                if key == 'demands':
                    new_inst[key] = torch.tensor(val, dtype=torch.long, device=device)
                else:
                    new_inst[key] = torch.tensor(val, dtype=torch.float32, device=device)
            else:
                new_inst[key] = val
        prepped.append(new_inst)
    return prepped

print("\nWarming up...")
# Warmup
for i in range(2):
    warmup_data = data_generator(32, epoch=0, seed=i)
    gpu_data = prep_instances(warmup_data, device)
    with torch.no_grad():
        model(gpu_data, decode_type='greedy')

print("\nRunning benchmark...")
torch.cuda.synchronize()
epoch_start = time.time()

n_batches = 25
batch_size = 128
total_cost = 0.0

for batch_idx in range(n_batches):
    batch_start = time.time()
    
    # Generate data
    instances = data_generator(batch_size, epoch=0, seed=batch_idx * 1000)
    
    # Prep for GPU (distances stay on CPU)
    gpu_instances = prep_instances(instances, device)
    
    # Model forward pass
    with torch.no_grad():
        routes, log_probs = model(gpu_instances, decode_type='sampling', temperature=2.5)
    
    # Compute costs on CPU
    costs = []
    for i, route in enumerate(routes):
        cost = compute_route_cost(route, instances[i]['distances'])
        costs.append(cost)
    
    batch_cost = np.mean(costs)
    total_cost += batch_cost
    
    batch_time = time.time() - batch_start
    
    if (batch_idx + 1) % 5 == 0:
        print(f"  Batch {batch_idx+1:2d}/{n_batches}: {batch_time:.2f}s, avg_cost={batch_cost:.4f}")

torch.cuda.synchronize()
total_time = time.time() - epoch_start

print("\n" + "="*60)
print("RESULTS:")
print(f"  Total time: {total_time:.2f} seconds")
print(f"  Instances: {n_batches * batch_size}")
print(f"  Throughput: {(n_batches * batch_size)/total_time:.1f} instances/second")
print(f"  Average cost: {total_cost/n_batches:.4f}")
print(f"\n  Target: 22-23 seconds")
if total_time <= 23:
    print(f"  ✓ SUCCESS - Target achieved!")
else:
    print(f"  ✗ SLOW - {total_time-23:.1f}s over target")
print("="*60)
