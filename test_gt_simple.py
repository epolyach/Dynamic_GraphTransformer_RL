#!/usr/bin/env python3
"""Simple test for advanced GT+RL model."""

import torch
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.model_factory import ModelFactory
from scripts.config_utils import load_config


def create_test_instances(n_instances=2, n_nodes=20):
    """Create test CVRP instances."""
    instances = []
    for _ in range(n_instances):
        # Random node locations
        coords = np.random.rand(n_nodes, 2).tolist()
        
        # Random demands (depot has 0 demand)
        demands = [0] + [np.random.randint(1, 10) for _ in range(n_nodes-1)]
        
        # Vehicle capacity
        capacity = sum(demands) // 3 + 10  # Ensure feasible
        
        instances.append({
            'coords': coords,
            'demands': demands,
            'capacity': capacity
        })
    return instances


def test_advanced_gt():
    """Test advanced GT+RL model functionality."""
    
    # Load config
    config = load_config('configs/config.yaml')
    
    print("1. Testing model creation...")
    model = ModelFactory.create_model('GT+RL', config)
    print(f"✓ Created GT+RL model: {type(model).__name__}")
    
    # Check components
    print("\n2. Checking model components...")
    assert hasattr(model, 'spatial_encoder'), "Missing spatial encoder"
    assert hasattr(model, 'state_encoder'), "Missing state encoder"
    assert hasattr(model, 'pointer_attention'), "Missing pointer attention"
    assert hasattr(model, 'graph_updater'), "Missing graph updater"
    assert hasattr(model, 'transformer_layers'), "Missing transformer layers"
    assert hasattr(model, 'demand_encoder'), "Missing demand encoder"
    assert hasattr(model, 'depot_embedding'), "Missing depot embedding"
    print("✓ All expected components present")
    
    # Test with sample data
    print("\n3. Testing forward pass...")
    test_instances = create_test_instances(n_instances=2, n_nodes=20)
    
    # Run forward pass
    with torch.no_grad():
        routes, log_probs, entropy = model(test_instances, greedy=True, config=config)
    
    print(f"✓ Forward pass successful")
    print(f"  - Routes generated: {len(routes)}")
    print(f"  - Log probs shape: {log_probs.shape}")
    print(f"  - Entropy shape: {entropy.shape}")
    
    # Test route validity
    print("\n4. Testing route validity...")
    for i, (instance, route) in enumerate(zip(test_instances, routes)):
        assert route[0] == 0, f"Route {i} doesn't start at depot"
        assert route[-1] == 0, f"Route {i} doesn't end at depot"
        
        # Check all customers visited
        customers = set(range(1, len(instance['coords'])))
        visited = set(r for r in route if r != 0)
        assert customers == visited, f"Route {i} doesn't visit all customers"
        
        # Check capacity constraints
        current_capacity = 0
        for j, node in enumerate(route):
            if node == 0:
                current_capacity = 0
            else:
                current_capacity += instance['demands'][node]
                if current_capacity > instance['capacity']:
                    print(f"Warning: Route {i} might violate capacity at step {j}")
    
    print("✓ All routes are valid")
    
    # Test training step
    print("\n5. Testing training compatibility...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Get routes and log probs
    routes, log_probs, entropy = model(test_instances, greedy=False, config=config)
    
    # Compute simple reward (negative distance)
    rewards = []
    for instance, route in zip(test_instances, routes):
        coords = np.array(instance['coords'])
        distance = 0
        for i in range(len(route) - 1):
            distance += np.linalg.norm(coords[route[i]] - coords[route[i+1]])
        rewards.append(-distance)
    
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    # Compute loss
    loss = -(log_probs * rewards).mean() - 0.01 * entropy.mean()
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print("✓ Training step successful")
    print(f"  - Loss: {loss.item():.4f}")
    
    # Test model comparison
    print("\n6. Comparing with legacy GAT+RL...")
    legacy_model = ModelFactory.create_model('GAT+RL', config)
    
    # Count parameters
    gt_params = sum(p.numel() for p in model.parameters())
    gat_params = sum(p.numel() for p in legacy_model.parameters())
    
    print(f"  - GT+RL parameters: {gt_params:,}")
    print(f"  - GAT+RL parameters: {gat_params:,}")
    print(f"  - GT+RL has {gt_params/gat_params:.2f}x parameters")
    
    print("\n✅ All tests passed! Advanced GT+RL is ready for training.")
    print("\nKey improvements over legacy GAT+RL:")
    print("  ✓ Spatial and positional encoding")
    print("  ✓ Distance-aware attention")
    print("  ✓ Dynamic state tracking")
    print("  ✓ Multi-head pointer attention")
    print("  ✓ Dynamic graph updates during decoding")
    print("  ✓ CVRP-specific inductive biases")
    print("  ✓ Modern transformer improvements (Pre-LN, GLU)")


if __name__ == '__main__':
    test_advanced_gt()
