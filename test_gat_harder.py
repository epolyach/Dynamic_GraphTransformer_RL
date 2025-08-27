#!/usr/bin/env python3
"""
Test script with harder instances to verify GAT+RL respects capacity constraints.
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from src.models.legacy_gat import LegacyGATModel

def test_gat_harder():
    """Test GAT model with challenging capacity scenarios."""
    
    # Test case 1: Tight capacity constraints
    instance1 = {
        'coords': [(0.5, 0.5), (0.2, 0.3), (0.8, 0.7), (0.3, 0.9), (0.7, 0.2), (0.1, 0.6)],
        'demands': [0, 18, 17, 16, 19, 15],  # Very tight - need careful routing
        'capacity': 35
    }
    
    # Test case 2: Mixed demands
    instance2 = {
        'coords': [(0.5, 0.5)] + [(np.random.random(), np.random.random()) for _ in range(10)],
        'demands': [0, 5, 25, 8, 22, 7, 20, 6, 15, 9, 18],  # Mixed small and large
        'capacity': 30
    }
    
    instances = [instance1, instance2]
    names = ["Tight capacity", "Mixed demands"]
    
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
    
    for inst_idx, (instance, name) in enumerate(zip(instances, names)):
        print(f"\n{'='*60}")
        print(f"Test case {inst_idx + 1}: {name}")
        print(f"Customers: {len(instance['coords']) - 1}, Capacity: {instance['capacity']}")
        print(f"Demands: {instance['demands'][1:]}")
        print(f"Total demand: {sum(instance['demands'][1:])}")
        
        # Theoretical minimum trips
        min_trips = 0
        remaining = list(instance['demands'][1:])
        while remaining:
            trip_load = 0
            i = 0
            while i < len(remaining):
                if trip_load + remaining[i] <= instance['capacity']:
                    trip_load += remaining[i]
                    remaining.pop(i)
                else:
                    i += 1
            min_trips += 1
        print(f"Theoretical minimum trips needed: {min_trips}")
        
        # Test model
        all_valid = True
        for run in range(3):
            with torch.no_grad():
                routes, log_probs, entropy = model(
                    [instance],
                    max_steps=50,  # More steps for complex routes
                    temperature=1.0,
                    greedy=True
                )
            
            route = routes[0]
            print(f"\n  Run {run + 1}: {route}")
            
            # Count trips
            trips = 0
            for i in range(len(route) - 1):
                if route[i] != 0 and route[i+1] == 0:
                    trips += 1
            print(f"    Trips taken: {trips}")
            
            # Validate
            is_valid = True
            
            # Check all customers visited exactly once
            customers = set(range(1, len(instance['coords'])))
            visited = set(n for n in route if n != 0)
            if visited != customers:
                is_valid = False
                print(f"    ERROR: Customer coverage issue")
            
            # Check capacity per trip
            current_load = 0
            trip_num = 1
            for i, node in enumerate(route):
                if node == 0:
                    if i > 0 and current_load > 0:  # End of trip
                        print(f"    Trip {trip_num} ended with load {current_load}/{instance['capacity']}")
                        trip_num += 1
                    current_load = 0
                else:
                    current_load += instance['demands'][node]
                    if current_load > instance['capacity']:
                        is_valid = False
                        print(f"    ERROR: Capacity exceeded at position {i}: {current_load} > {instance['capacity']}")
                        break
            
            if is_valid:
                print(f"    ✓ Valid route")
            else:
                print(f"    ✗ Invalid route")
                all_valid = False
        
        if all_valid:
            print(f"\n✓ All runs valid for '{name}'")
        else:
            print(f"\n✗ Some runs failed for '{name}'")

if __name__ == "__main__":
    test_gat_harder()
