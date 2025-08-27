#!/usr/bin/env python3
"""Test that advanced GT+RL works with the training pipeline."""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.model_factory import ModelFactory
from src.data.generate_data import load_instances_from_file
from scripts.config_utils import load_config
import numpy as np


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
    val_instances = load_instances_from_file('data/cvrp20_val.pkl')[:2]
    
    # Run forward pass
    with torch.no_grad():
        routes, log_probs, entropy = model(val_instances, greedy=True, config=config)
    
    print(f"✓ Forward pass successful")
    print(f"  - Routes generated: {len(routes)}")
    print(f"  - Log probs shape: {log_probs.shape}")
    print(f"  - Entropy shape: {entropy.shape}")
    
    # Test route validity
    print("\n4. Testing route validity...")
    for i, (instance, route) in enumerate(zip(val_instances, routes)):
        assert route[0] == 0, f"Route {i} doesn't start at depot"
        assert route[-1] == 0, f"Route {i} doesn't end at depot"
        
        # Check all customers visited
        customers = set(range(1, len(instance['coords'])))
        visited = set(r for r in route if r != 0)
        assert customers == visited, f"Route {i} doesn't visit all customers"
        
        # Check capacity constraints
        total_demand = 0
        for node in route[1:-1]:  # Skip depot
            if node == 0:
                total_demand = 0
            else:
                total_demand += instance['demands'][node]
                assert total_demand <= instance['capacity'], f"Route {i} violates capacity"
    
    print("✓ All routes are valid")
    
    # Test training step
    print("\n5. Testing training compatibility...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Get routes and log probs
    routes, log_probs, entropy = model(val_instances, greedy=False, config=config)
    
    # Compute simple reward (negative distance)
    rewards = []
    for instance, route in zip(val_instances, routes):
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
    
    # Test inference speed
    import time
    
    print("\n7. Testing inference speed...")
    n_runs = 5
    
    # GT+RL
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(val_instances, greedy=True, config=config)
    gt_time = (time.time() - start) / n_runs
    
    # Legacy GAT+RL
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            legacy_model(val_instances, greedy=True, config=config)
    gat_time = (time.time() - start) / n_runs
    
    print(f"  - GT+RL: {gt_time:.3f}s per batch")
    print(f"  - GAT+RL: {gat_time:.3f}s per batch")
    print(f"  - GT+RL is {gat_time/gt_time:.2f}x speed")
    
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
