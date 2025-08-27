#!/usr/bin/env python3
"""
Test script to verify GAT+RL model generates valid routes without capacity violations.
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from src.models.legacy_gat import LegacyGATModel

def test_gat_capacity():
    """Test that GAT model respects capacity constraints."""
    
    # Create a simple test instance
    instance = {
        'coords': [(0.5, 0.5), (0.2, 0.3), (0.8, 0.7), (0.3, 0.9)],  # depot + 3 customers
        'demands': [0, 15, 10, 12],  # depot has 0 demand
        'capacity': 30
    }
    
    print(f"Instance: {len(instance['coords']) - 1} customers, capacity={instance['capacity']}")
    print(f"Customer demands: {instance['demands'][1:]}")
    
    # Initialize model
    model = LegacyGATModel(
        node_input_dim=3,
        edge_input_dim=1,
        hidden_dim=128,
        edge_dim=16,
        layers=2,
        config={'inference': {'default_temperature': 1.0, 'max_steps_multiplier': 3}}
    )
    
    # Move to eval mode
    model.eval()
    
    # Test with multiple runs
    print("\nTesting GAT+RL model capacity constraints...")
    all_valid = True
    
    for run in range(5):
        with torch.no_grad():
            routes, log_probs, entropy = model(
                [instance],
                max_steps=20,
                temperature=1.0,
                greedy=True  # Use greedy for deterministic results
            )
        
        route = routes[0]
        print(f"\nRun {run + 1}:")
        print(f"  Generated route: {route}")
        
        # Manual validation
        is_valid = True
        error_msg = ""
        
        # Check start and end at depot
        if route[0] != 0 or route[-1] != 0:
            is_valid = False
            error_msg = f"Route must start and end at depot (got start={route[0]}, end={route[-1]})"
        
        # Check no consecutive depots
        for i in range(len(route) - 1):
            if route[i] == 0 and route[i+1] == 0:
                is_valid = False
                error_msg = f"Consecutive depot visits at positions {i}-{i+1}"
                break
        
        # Check all customers visited
        customers = set(range(1, len(instance['coords'])))
        visited = set(n for n in route if n != 0)
        if visited != customers:
            is_valid = False
            missing = customers - visited
            extra = visited - customers
            error_msg = f"Customer coverage issue - missing: {missing}, extra: {extra}"
        
        # Check capacity constraints
        if is_valid:
            current_load = 0
            print(f"  Capacity tracking:")
            for i, node in enumerate(route):
                if node == 0:
                    if i > 0:  # Not the starting depot
                        print(f"    Step {i}: Return to depot, reset load from {current_load} to 0")
                    current_load = 0
                else:
                    current_load += instance['demands'][node]
                    print(f"    Step {i}: Visit customer {node} (demand={instance['demands'][node]}), " +
                          f"load={current_load}/{instance['capacity']}")
                    if current_load > instance['capacity']:
                        is_valid = False
                        error_msg = f"Capacity exceeded at step {i}: load={current_load} > capacity={instance['capacity']}"
                        print(f"    ERROR: {error_msg}")
                        break
        
        if is_valid:
            print(f"  Route is VALID")
        else:
            print(f"  Route is INVALID: {error_msg}")
            all_valid = False
    
    if all_valid:
        print("\n✓ All tests passed! GAT model generates valid routes.")
    else:
        print("\n✗ Some tests failed! GAT model has issues with route generation.")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_gat_capacity()
