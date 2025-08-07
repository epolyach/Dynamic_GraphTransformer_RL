#!/usr/bin/env python3
"""
Debug floating-point precision issues in cost calculations
"""

import torch
import numpy as np
from run_comparative_study import (
    BaselinePointerNetwork, generate_cvrp_instance, 
    compute_route_cost, compute_naive_baseline_cost,
    validate_route, set_seeds, train_model
)

def debug_precision():
    """Debug floating-point precision in validation cost calculation"""
    print("🔍 DEBUGGING FLOATING-POINT PRECISION")
    print("=" * 50)
    
    set_seeds(42)
    
    # Load the exact configuration from the main script
    config = {
        'num_customers': 6,
        'capacity': 3,
        'coord_range': 50,
        'demand_range': (1, 3),
        'hidden_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'batch_size': 8,
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'grad_clip': 1.0,
        'temperature': 1.0,
        'num_instances': 800
    }
    
    # Generate ALL validation instances (exactly as in main script)
    print("Generating validation instances...")
    val_instances = []
    naive_costs = []
    
    split_idx = int(0.8 * config['num_instances'])  # 640
    
    for i in range(split_idx, config['num_instances']):  # 640-799
        instance = generate_cvrp_instance(
            config['num_customers'], config['capacity'],
            config['coord_range'], config['demand_range'], seed=i
        )
        val_instances.append(instance)
        naive_costs.append(compute_naive_baseline_cost(instance))
    
    naive_avg = np.mean(naive_costs)
    naive_normalized = naive_avg / config['num_customers']
    
    print(f"Validation instances: {len(val_instances)}")
    print(f"Naive average cost: {naive_avg:.6f}")
    print(f"Naive normalized: {naive_normalized:.6f}")
    
    # Load trained model
    try:
        checkpoint = torch.load('model_baseline_pointer.pt', map_location='cpu', weights_only=False)
        model = BaselinePointerNetwork(input_dim=3, hidden_dim=64)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded trained model")
    except FileNotFoundError:
        print("❌ No trained model found")
        return
    
    # Compute validation cost exactly as in main script
    print("\n🔍 COMPUTING VALIDATION COST (same as main script):")
    batch_size = config['batch_size']
    model.eval()
    val_batch_costs = []
    
    with torch.no_grad():
        for i in range(0, len(val_instances), batch_size):
            batch_val = val_instances[i:i + batch_size]
            routes, _ = model(batch_val, greedy=True)
            
            for j, (route, instance) in enumerate(zip(routes, batch_val)):
                n_customers = len(instance['coords']) - 1
                validate_route(route, n_customers, f"PRECISION-VAL")
                total_cost = compute_route_cost(route, instance['distances'])
                val_batch_costs.append(total_cost)
    
    val_cost = np.mean(val_batch_costs)
    val_normalized = val_cost / config['num_customers']
    
    print(f"Model validation cost: {val_cost:.6f}")  
    print(f"Model normalized: {val_normalized:.6f}")
    
    # Compare with exact precision
    diff = val_normalized - naive_normalized
    print(f"\nPRECISION ANALYSIS:")
    print(f"Difference: {diff:.10f}")
    print(f"Absolute difference: {abs(diff):.10f}")
    print(f"Relative difference: {(diff / naive_normalized * 100):.6f}%")
    
    if diff > 0:
        print(f"🚨 Model cost ({val_normalized:.6f}) > Naive cost ({naive_normalized:.6f})")
        print(f"   This is a {diff:.6f} excess per customer")
        
        # Check if this is within floating-point tolerance
        eps = 1e-10
        if abs(diff) < eps:
            print(f"✅ Within floating-point tolerance ({eps})")
        else:
            print(f"❌ Exceeds floating-point tolerance - this is a real violation")
    
    # Test a few individual routes for precision
    print(f"\n🔍 TESTING INDIVIDUAL ROUTES:")
    model.eval()
    for i in range(5):  # Test first 5 validation instances
        instance = val_instances[i]
        naive_cost = compute_naive_baseline_cost(instance)
        
        routes, _ = model([instance], greedy=True)
        route = routes[0]
        model_cost = compute_route_cost(route, instance['distances'])
        
        diff_individual = model_cost - naive_cost
        print(f"Instance {i}: naive={naive_cost:.6f}, model={model_cost:.6f}, diff={diff_individual:.10f}")

if __name__ == "__main__":
    debug_precision()
