#!/usr/bin/env python3
"""
Debug script to trace GAT decoder decisions step by step.
"""

import torch
import sys
sys.path.append('.')

from src.models.legacy_gat import LegacyGATModel

def debug_decoder():
    """Debug decoder's capacity tracking and masking."""
    
    # Simple instance that should require depot visit
    instance = {
        'coords': [(0.5, 0.5), (0.2, 0.3), (0.8, 0.7), (0.3, 0.9)],
        'demands': [0, 18, 17, 16],  # Can't fit 18+17 in capacity 30
        'capacity': 30
    }
    
    print(f"Instance: capacity={instance['capacity']}")
    print(f"Customer demands: {instance['demands'][1:]}")
    print(f"Expected: Should return to depot after customer 1 (demand=18)")
    print()
    
    # Initialize model
    model = LegacyGATModel(
        node_input_dim=3,
        edge_input_dim=1,
        hidden_dim=128,
        edge_dim=16,
        layers=2,
        config={'inference': {'default_temperature': 1.0, 'max_steps_multiplier': 3}}
    )
    model.eval()
    
    # Monkey-patch decoder to add debug output
    original_update_mask = model.decoder.update_mask
    original_update_state = model.decoder.update_state
    
    def debug_update_mask(demands, dynamic_capacity, index, mask1, step):
        print(f"\nStep {step}: update_mask called")
        print(f"  Current index: {index}")
        print(f"  Dynamic capacity: {dynamic_capacity}")
        print(f"  Demands: {demands}")
        
        mask, mask1 = original_update_mask(demands, dynamic_capacity, index, mask1, step)
        
        print(f"  Resulting mask: {mask}")
        print(f"  Mask1 (visited): {mask1}")
        
        # Check what's feasible
        for b in range(mask.size(0)):
            feasible = []
            for n in range(mask.size(1)):
                if mask[b, n] == 0:
                    feasible.append(n)
            print(f"  Feasible nodes for batch {b}: {feasible}")
        
        return mask, mask1
    
    def debug_update_state(demands, dynamic_capacity, index, max_capacity):
        print(f"\nupdate_state called")
        print(f"  Selected node: {index}")
        print(f"  Node value: {index[0].item()}")
        print(f"  Is depot: {index[0].item() == 0}")
        print(f"  Max capacity: {max_capacity}")
        print(f"  Dynamic capacity before: {dynamic_capacity}")
        
        result = original_update_state(demands, dynamic_capacity, index, max_capacity)
        
        print(f"  Dynamic capacity after: {result}")
        if index[0].item() == 0:
            print(f"  WARNING: Depot visit should reset to {max_capacity}!")
        
        return result
    
    model.decoder.update_mask = debug_update_mask
    model.decoder.update_state = debug_update_state
    
    # Run model
    with torch.no_grad():
        routes, log_probs, entropy = model(
            [instance],
            max_steps=10,  # Limit steps for debugging
            temperature=1.0,
            greedy=True
        )
    
    route = routes[0]
    print(f"\nFinal route: {route}")
    
    # Check capacity
    current_load = 0
    print("\nCapacity check:")
    for i, node in enumerate(route):
        if node == 0:
            if i > 0:
                print(f"  Step {i}: Return to depot, reset load from {current_load} to 0")
            current_load = 0
        else:
            current_load += instance['demands'][node]
            print(f"  Step {i}: Visit customer {node} (demand={instance['demands'][node]}), load={current_load}/{instance['capacity']}")
            if current_load > instance['capacity']:
                print(f"  ERROR: Capacity exceeded!")
                break

if __name__ == "__main__":
    debug_decoder()
