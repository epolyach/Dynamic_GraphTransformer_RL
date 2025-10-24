import time
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

# Simple profiling to identify bottlenecks
def profile_training_step():
    from src.generator.generator import CVRPInstanceGenerator
    from src.models.gt import GraphTransformer
    from src.metrics.gpu_costs import compute_route_cost_gpu
    
    # Initialize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = GraphTransformer(hidden_dim=128, num_heads=4, num_layers=3).to(device)
    model.eval()
    
    # Generator
    generator = CVRPInstanceGenerator({
        'num_customers': 10,
        'vehicle_capacity': 20,
        'coord_range': 100,
        'demand_min': 1,
        'demand_max': 10
    })
    
    batch_size = 128
    n_tests = 5
    
    timings = {
        'data_generation': [],
        'to_gpu': [],
        'forward_pass': [],
        'cost_computation': [],
        'total': []
    }
    
    for test in range(n_tests):
        total_start = time.time()
        
        # Data generation
        t0 = time.time()
        instances = [generator.generate() for _ in range(batch_size)]
        timings['data_generation'].append(time.time() - t0)
        
        # Move to GPU
        t0 = time.time()
        for inst in instances:
            inst['coords'] = torch.tensor(inst['coords'], dtype=torch.float32, device=device)
            inst['demands'] = torch.tensor(inst['demands'], dtype=torch.float32, device=device)
            inst['distances'] = torch.tensor(inst['distances'], dtype=torch.float32, device=device)
        timings['to_gpu'].append(time.time() - t0)
        
        # Forward pass
        t0 = time.time()
        with torch.no_grad():
            routes, log_probs, entropy = model(
                instances,
                max_steps=100,
                temperature=1.0,
                greedy=False,
                config={'inference': {'max_steps_multiplier': 10}}
            )
        torch.cuda.synchronize()
        timings['forward_pass'].append(time.time() - t0)
        
        # Cost computation
        t0 = time.time()
        costs = []
        for route, inst in zip(routes, instances):
            cost = compute_route_cost_gpu(route, inst['distances'])
            costs.append(cost)
        torch.cuda.synchronize()
        timings['cost_computation'].append(time.time() - t0)
        
        timings['total'].append(time.time() - total_start)
    
    # Report
    print("\nProfiling Results (seconds):")
    print("-" * 50)
    for key, values in timings.items():
        avg = np.mean(values[1:])  # Skip first (warmup)
        std = np.std(values[1:])
        pct = (avg / np.mean(timings['total'][1:])) * 100 if key != 'total' else 100
        print(f"{key:20s}: {avg:.4f} ± {std:.4f} ({pct:.1f}%)")
    
    print("\nExtrapolated for full epoch (25 batches):")
    print(f"Expected time: {np.mean(timings['total'][1:]) * 25:.2f} seconds")

if __name__ == "__main__":
    profile_training_step()
