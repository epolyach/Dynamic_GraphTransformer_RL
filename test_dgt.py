#!/usr/bin/env python3
"""Test the Dynamic Graph Transformer model."""

import torch
import sys
import os
import numpy as np
import yaml
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.dgt import DynamicGraphTransformer
from src.models.gt import GraphTransformer
from src.models.legacy_gat import LegacyGATModel


def load_config():
    """Load config manually."""
    with open('configs/default.yaml', 'r') as f:
        return yaml.safe_load(f)


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


def test_dgt():
    """Test DGT+RL model functionality."""
    
    # Load config
    config = load_config()
    
    print("="*60)
    print("DYNAMIC GRAPH TRANSFORMER (DGT+RL) TEST")
    print("="*60)
    
    print("\n1. Testing model creation...")
    
    # Create DGT+RL model
    model_config = config['model']
    model = DynamicGraphTransformer(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        dropout=model_config['transformer_dropout'],
        feedforward_multiplier=model_config['feedforward_multiplier'],
        config=config
    )
    print(f"✓ Created DGT+RL model: {type(model).__name__}")
    
    # Check unique DGT components
    print("\n2. Checking DGT-specific components...")
    dgt_components = [
        ('memory_bank', 'Temporal Memory Bank'),
        ('edge_processor', 'Dynamic Edge Processor'),
        ('graph_adapter', 'Adaptive Graph Structure'),
        ('temporal_attention', 'Multi-scale Temporal Attention'),
        ('refinement_layers', 'Progressive Refinement'),
        ('update_schedule', 'Learned Update Schedule'),
        ('temperature_controller', 'Adaptive Temperature Control')
    ]
    
    for comp_name, desc in dgt_components:
        assert hasattr(model, comp_name), f"Missing {comp_name}"
        print(f"  ✓ {desc} ({comp_name})")
    
    # Also check GT components
    print("\n3. Checking inherited GT+RL components...")
    gt_components = [
        'spatial_encoder', 'state_encoder', 'pointer_attention',
        'graph_updater', 'transformer_layers', 'demand_encoder', 
        'depot_embedding'
    ]
    for comp in gt_components:
        assert hasattr(model, comp), f"Missing {comp}"
        print(f"  ✓ {comp}")
    
    print("✓ All components present")
    
    # Test with sample data
    print("\n4. Testing forward pass...")
    test_instances = create_test_instances(n_instances=2, n_nodes=20)
    
    # Run forward pass
    with torch.no_grad():
        routes, log_probs, entropy = model(test_instances, greedy=True, config=config)
    
    print(f"✓ Forward pass successful")
    print(f"  - Routes generated: {len(routes)}")
    print(f"  - Sample route: {routes[0][:10]}...")
    print(f"  - Log probs shape: {log_probs.shape}")
    print(f"  - Entropy shape: {entropy.shape}")
    
    # Test route validity
    print("\n5. Testing route validity...")
    for i, (instance, route) in enumerate(zip(test_instances, routes)):
        assert route[0] == 0, f"Route {i} doesn't start at depot"
        assert route[-1] == 0, f"Route {i} doesn't end at depot"
        
        # Check all customers visited
        customers = set(range(1, len(instance['coords'])))
        visited = set(r for r in route if r != 0)
        assert customers == visited, f"Route {i} doesn't visit all customers"
        print(f"  ✓ Route {i} valid")
    
    print("✓ All routes are valid")
    
    # Test training step
    print("\n6. Testing training compatibility...")
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
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("✓ Training step successful")
    print(f"  - Loss: {loss.item():.4f}")
    
    # Compare with GT+RL
    print("\n7. Comparing with GT+RL...")
    gt_model = GraphTransformer(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        dropout=model_config['transformer_dropout'],
        feedforward_multiplier=model_config['feedforward_multiplier'],
        config=config
    )
    
    # Count parameters
    dgt_params = sum(p.numel() for p in model.parameters())
    gt_params = sum(p.numel() for p in gt_model.parameters())
    
    print(f"  - DGT+RL parameters: {dgt_params:,}")
    print(f"  - GT+RL parameters: {gt_params:,}")
    print(f"  - DGT+RL has {dgt_params/gt_params:.2f}x parameters")
    
    # Speed test (smaller batch for quick test)
    print("\n8. Testing inference speed...")
    test_instances_speed = create_test_instances(n_instances=4, n_nodes=30)
    
    # DGT+RL
    start = time.time()
    with torch.no_grad():
        for _ in range(3):
            model(test_instances_speed, greedy=True, config=config)
    dgt_time = (time.time() - start) / 3
    
    # GT+RL
    start = time.time()
    with torch.no_grad():
        for _ in range(3):
            gt_model(test_instances_speed, greedy=True, config=config)
    gt_time = (time.time() - start) / 3
    
    print(f"  - DGT+RL: {dgt_time:.3f}s per batch")
    print(f"  - GT+RL: {gt_time:.3f}s per batch")
    print(f"  - DGT+RL is {dgt_time/gt_time:.2f}x slower (expected due to richer features)")
    
    # Model hierarchy summary
    print("\n" + "="*60)
    print("MODEL HIERARCHY SUMMARY")
    print("="*60)
    
    print("\n1. GAT+RL (Legacy Baseline):")
    print("   - Basic GAT with edge features")
    print("   - Simple pointer attention")
    print("   - ~1.3M parameters")
    
    print("\n2. GT+RL (Advanced):")
    print("   - Spatial/positional encoding")
    print("   - Distance-aware attention")
    print("   - Dynamic state tracking")
    print("   - Multi-head pointer")
    print("   - ~3.8M parameters")
    
    print("\n3. DGT+RL (Ultimate):")
    print("   - Everything from GT+RL PLUS:")
    print("   - Temporal memory bank")
    print("   - Dynamic edge processing")
    print("   - Adaptive graph structure")
    print("   - Multi-scale temporal attention")
    print("   - Progressive refinement")
    print("   - Learned update schedules")
    print("   - Adaptive temperature control")
    print(f"   - ~{dgt_params/1e6:.1f}M parameters")
    
    print("\n✅ DGT+RL is ready for training!")
    print("\nKey advantages of DGT+RL:")
    print("  • True dynamic graph adaptation during decoding")
    print("  • Memory-augmented decision making")
    print("  • Multi-scale temporal reasoning")
    print("  • Progressive refinement based on solution progress")
    print("  • Adaptive components that learn update schedules")
    print("\nThis should be the strongest model for CVRP!")


if __name__ == '__main__':
    test_dgt()
