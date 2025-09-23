#!/usr/bin/env python3
"""Simple timing test to identify bottlenecks"""
import time
import sys
import os
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
os.chdir('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

# Import required modules
from src.generator.generator import DataGenerator
from src.models.model_factory import ModelFactory
from training_gpu.lib.gpu_utils import GPUManager
import torch
import numpy as np
from src.metrics.costs import compute_route_cost

print("="*60)
print("Simple timing test - GT model inference only")
print("n=10, capacity=30, 25 batches x 128")
print("="*60)

# Initialize
gpu_manager = GPUManager(device='cuda:0')
generator = DataGenerator()

# Create model
model = ModelFactory.create_model(
    'GT+RL', 
    n_customers=10,
    model_config={'hidden_dim': 128, 'num_heads': 4, 'num_layers': 3}
).to(gpu_manager.device)

print(f"Model on device: {next(model.parameters()).device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Simple data prep function
def prep_for_gpu(instances, device):
    gpu_batch = []
    for inst in instances:
        gpu_inst = {}
        for key, val in inst.items():
            if key == 'distances':
                gpu_inst[key] = val  # Keep on CPU
            elif isinstance(val, np.ndarray):
                if key == 'demands':
                    gpu_inst[key] = torch.tensor(val, dtype=torch.long, device=device)
                else:
                    gpu_inst[key] = torch.tensor(val, dtype=torch.float32, device=device)
            else:
                gpu_inst[key] = val
        gpu_batch.append(gpu_inst)
    return gpu_batch

# Test different phases
print("\n1. Testing data generation speed...")
start = time.time()
for i in range(25):
    data = generator.generate(128, n_customers=10, capacity=30, seed=i*1000)
gen_time = time.time() - start
print(f"   Data generation: {gen_time:.2f}s for 3200 instances")

print("\n2. Testing data transfer to GPU...")
test_data = generator.generate(128, n_customers=10, capacity=30, seed=0)
start = time.time()
for i in range(25):
    gpu_data = prep_for_gpu(test_data, gpu_manager.device)
transfer_time = time.time() - start
print(f"   Data transfer: {transfer_time:.2f}s for 25 batches")

print("\n3. Testing model inference...")
gpu_data = prep_for_gpu(test_data, gpu_manager.device)
model.eval()
with torch.no_grad():
    start = time.time()
    for i in range(25):
        routes, _ = model(gpu_data, decode_type='sampling', temperature=2.5)
    torch.cuda.synchronize()
    inference_time = time.time() - start
print(f"   Model inference: {inference_time:.2f}s for 25 batches")

print("\n4. Testing cost computation...")
routes_list = []
with torch.no_grad():
    routes, _ = model(gpu_data, decode_type='sampling')
    routes_list = routes
start = time.time()
for _ in range(25):
    costs = []
    for i, route in enumerate(routes_list):
        cost = compute_route_cost(route, test_data[i]['distances'])
        costs.append(cost)
cost_time = time.time() - start
print(f"   Cost computation: {cost_time:.2f}s for 25 batches")

print("\n" + "="*60)
total = gen_time + transfer_time + inference_time + cost_time
print(f"BREAKDOWN:")
print(f"  Data generation:  {gen_time:.2f}s ({gen_time/total*100:.1f}%)")
print(f"  Data transfer:    {transfer_time:.2f}s ({transfer_time/total*100:.1f}%)")
print(f"  Model inference:  {inference_time:.2f}s ({inference_time/total*100:.1f}%)")
print(f"  Cost computation: {cost_time:.2f}s ({cost_time/total*100:.1f}%)")
print(f"  TOTAL:           {total:.2f}s")
print(f"\n  Target: ~22-23 seconds")
if total < 25:
    print(f"  Status: ✓ Can achieve target")
else:
    print(f"  Status: ✗ Too slow - main bottleneck: ", end="")
    times = {'generation': gen_time, 'transfer': transfer_time, 'inference': inference_time, 'cost': cost_time}
    print(max(times, key=times.get))
print("="*60)
