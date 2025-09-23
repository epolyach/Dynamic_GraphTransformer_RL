#!/usr/bin/env python3
"""Test trainer performance without baseline to isolate the issue"""
import time
import sys
import os
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
os.chdir('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

# Import required modules
from src.utils.config import load_config
from src.generator.generator import create_data_generator
from src.models.model_factory import ModelFactory
from training_gpu.lib.gpu_utils import GPUManager
import torch
import numpy as np
from src.metrics.costs import compute_route_cost

print("="*60)
print("Direct timing test - GT+RL model, n=10, capacity=30")
print("Testing 25 batches x 128 = 3200 instances")
print("="*60)

# Initialize
config = {
    'problem': {'num_customers': 10, 'vehicle_capacity': 30},
    'model': {'hidden_dim': 128, 'num_heads': 4, 'num_layers': 3},
    'training': {'batch_size': 128, 'num_batches_per_epoch': 25}
}

# Create components
gpu_manager = GPUManager(device='cuda:0')
data_generator = create_data_generator(
    mode='train',
    n_customers=10,
    capacity=30
)

# Create model
model = ModelFactory.create_model(
    'GT+RL', 
    n_customers=10,
    model_config=config['model']
).to(gpu_manager.device)

# Function to move data to GPU except distances
def move_to_gpu_except_distances(instance, device):
    gpu_inst = {}
    for key, value in instance.items():
        if key == 'distances':
            # Keep distances on CPU
            gpu_inst[key] = value
        elif isinstance(value, np.ndarray):
            if key == 'demands':
                gpu_inst[key] = torch.tensor(value, dtype=torch.long, device=device)
            else:
                gpu_inst[key] = torch.tensor(value, dtype=torch.float32, device=device)
        else:
            gpu_inst[key] = value
    return gpu_inst

# Time one epoch
print("\nWarming up GPU...")
# Warmup
for i in range(2):
    instances = data_generator(32, seed=i)
    gpu_instances = [move_to_gpu_except_distances(inst, gpu_manager.device) for inst in instances]
    with torch.no_grad():
        model(gpu_instances, decode_type='greedy')
torch.cuda.synchronize()

print("Starting timing test...")
start_time = time.time()

total_instances = 0
total_cost = 0.0
n_batches = 25
batch_size = 128

for batch_idx in range(n_batches):
    # Generate instances
    instances = data_generator(batch_size, seed=batch_idx * 1000)
    
    # Move to GPU (except distances)
    gpu_instances = [move_to_gpu_except_distances(inst, gpu_manager.device) for inst in instances]
    
    # Forward pass
    with torch.no_grad():
        routes, log_probs = model(gpu_instances, decode_type='sampling', temperature=2.5)
    
    # Compute costs on CPU
    batch_costs = []
    for i, route in enumerate(routes):
        distances = instances[i]['distances']  # Already on CPU
        cost = compute_route_cost(route, distances)
        batch_costs.append(cost)
    
    total_cost += np.mean(batch_costs)
    total_instances += batch_size
    
    if (batch_idx + 1) % 5 == 0:
        elapsed = time.time() - start_time
        print(f"  Batch {batch_idx+1}/{n_batches}: {elapsed:.1f}s elapsed, "
              f"{total_instances} instances, "
              f"avg cost: {total_cost/(batch_idx+1):.4f}")

torch.cuda.synchronize()
total_time = time.time() - start_time

print("\n" + "="*60)
print(f"RESULTS:")
print(f"  Total instances: {total_instances}")
print(f"  Total time: {total_time:.2f} seconds")
print(f"  Instances/second: {total_instances/total_time:.1f}")
print(f"  Average cost: {total_cost/n_batches:.4f}")
print(f"  Target time: ~22-23 seconds")
if total_time < 25:
    print(f"  Status: ✓ SUCCESS - Performance target achieved!")
else:
    print(f"  Status: ✗ SLOW ({total_time:.1f}s) - Need optimization")
print("="*60)
