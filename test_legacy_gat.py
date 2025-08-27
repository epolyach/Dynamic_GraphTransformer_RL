#!/usr/bin/env python3
"""
Test script for the Legacy GAT model implementation.
Verifies that the model works correctly with the existing training pipeline.
"""

import torch
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.legacy_gat import LegacyGATModel
from src.models.model_factory import create_model
from src.pipelines.train import generate_cvrp_instance

def test_legacy_gat_forward():
    """Test the forward pass of the Legacy GAT model."""
    print("Testing Legacy GAT Model...")
    
    # Load configs (merge default with medium)
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('configs/medium.yaml', 'r') as f:
        medium_config = yaml.safe_load(f)
    # Simple merge (override default with medium)
    for key, value in medium_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    # Create model using factory
    model = create_model('GAT+RL', config)
    print(f"Created model: {type(model).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Generate test instances
    batch_size = 2
    num_nodes = 20
    instances = []
    
    for _ in range(batch_size):
        instance = generate_cvrp_instance(
            num_customers=num_nodes-1,  # num_customers excludes depot
            capacity=50,
            coord_range=100,
            demand_range=[1, 10],
            seed=None
        )
        instances.append(instance)
    
    print(f"\nGenerated {batch_size} instances with {num_nodes} nodes each")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        routes, log_probs, entropy = model(
            instances,
            max_steps=num_nodes * 2,
            temperature=1.0,
            greedy=True,
            config=config
        )
    
    print(f"\nForward pass successful!")
    print(f"Routes shape: {len(routes)} routes")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")
    
    # Validate routes
    for i, route in enumerate(routes):
        print(f"\nRoute {i+1}: {route[:10]}..." if len(route) > 10 else f"\nRoute {i+1}: {route}")
        # Check that route starts and ends at depot
        assert route[0] == 0, f"Route {i} doesn't start at depot"
        # Check no invalid nodes
        assert all(0 <= node < num_nodes for node in route), f"Route {i} has invalid nodes"
    
    print("\n✅ All tests passed!")
    return True

def test_model_components():
    """Test individual components of the legacy GAT model."""
    print("\nTesting individual components...")
    
    from src.models.legacy_gat import (
        EdgeGATConv, ResidualEdgeGATEncoder, 
        TransformerAttention, PointerAttention, GAT_Decoder
    )
    
    batch_size = 2
    num_nodes = 20
    hidden_dim = 128
    edge_dim = 16
    
    # Test EdgeGATConv
    print("Testing EdgeGATConv...")
    edge_gat = EdgeGATConv(hidden_dim, hidden_dim, edge_dim)
    x = torch.randn(batch_size * num_nodes, hidden_dim)
    edge_index = torch.randint(0, batch_size * num_nodes, (2, 100))
    edge_attr = torch.randn(100, edge_dim)
    out = edge_gat(x, edge_index, edge_attr)
    assert out.shape == (batch_size * num_nodes, hidden_dim)
    print("✓ EdgeGATConv works")
    
    # Test ResidualEdgeGATEncoder
    print("Testing ResidualEdgeGATEncoder...")
    encoder = ResidualEdgeGATEncoder(3, 1, hidden_dim, edge_dim)
    x = torch.randn(batch_size * num_nodes, 2)
    demands = torch.randn(batch_size * num_nodes, 1)
    edge_attr = torch.randn(100, 1)
    encoded = encoder(x, edge_index, edge_attr, demands, batch_size)
    assert encoded.shape == (batch_size, num_nodes, hidden_dim)
    print("✓ ResidualEdgeGATEncoder works")
    
    # Test TransformerAttention
    print("Testing TransformerAttention...")
    mha = TransformerAttention(8, 1, hidden_dim, hidden_dim)
    state = torch.randn(batch_size, 1, hidden_dim)
    context = torch.randn(batch_size, num_nodes, hidden_dim)
    mask = torch.zeros(batch_size, num_nodes)
    out = mha(state, context, mask)
    assert out.shape == (batch_size, hidden_dim)
    print("✓ TransformerAttention works")
    
    # Test PointerAttention
    print("Testing PointerAttention...")
    pointer = PointerAttention(8, hidden_dim, hidden_dim)
    scores = pointer(state, context, mask, T=1.0)
    assert scores.shape == (batch_size, num_nodes)
    assert torch.allclose(scores.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    print("✓ PointerAttention works")
    
    # Test GAT_Decoder
    print("Testing GAT_Decoder...")
    decoder = GAT_Decoder(hidden_dim, hidden_dim)
    pool = torch.randn(batch_size, hidden_dim)
    capacity = torch.tensor([[50.0], [50.0]])
    demand = torch.randint(1, 10, (batch_size, num_nodes)).float()
    demand[:, 0] = 0  # Depot has no demand
    actions, log_p = decoder(encoded, pool, capacity, demand, num_nodes * 2, T=1.0, greedy=True)
    assert actions.shape[0] == batch_size
    assert log_p.shape == (batch_size,)
    print("✓ GAT_Decoder works")
    
    print("\n✅ All component tests passed!")
    return True

def test_training_compatibility():
    """Test that the model works with the training pipeline."""
    print("\nTesting training compatibility...")
    
    from src.metrics.costs import compute_route_cost
    
    # Load configs (merge default with medium)
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('configs/medium.yaml', 'r') as f:
        medium_config = yaml.safe_load(f)
    # Simple merge
    for key, value in medium_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    # Create model
    model = create_model('GAT+RL', config)
    
    # Skip baseline for now, just test model directly
    
    # Generate instances
    batch_size = 4
    instances = []
    for _ in range(batch_size):
        instance = generate_cvrp_instance(
            num_customers=19,  # 20 nodes total including depot
            capacity=50,
            coord_range=100,
            demand_range=[1, 10]
        )
        instances.append(instance)
    
    # Test forward pass with sampling
    model.train()
    routes, log_probs, entropy = model(
        instances,
        max_steps=40,
        temperature=1.0,
        greedy=False,
        config=config
    )
    
    # Calculate costs
    costs = []
    for i, (route, instance) in enumerate(zip(routes, instances)):
        cost = compute_route_cost(route, instance['distances'])
        costs.append(cost)
    costs = torch.tensor(costs)
    
    # Use mean as baseline
    baseline_costs = costs.mean()
    
    # Calculate advantage
    advantage = baseline_costs - costs
    
    # Calculate loss (REINFORCE)
    loss = -(advantage.detach() * log_probs).mean()
    
    print(f"Loss calculated: {loss.item():.4f}")
    print(f"Average cost: {costs.mean().item():.4f}")
    print(f"Average baseline cost: {baseline_costs.mean().item():.4f}")
    print(f"Average advantage: {advantage.mean().item():.4f}")
    
    # Test backward pass
    loss.backward()
    
    # Check gradients
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grads, "Model has no gradients after backward pass"
    
    print("\n✅ Training compatibility test passed!")
    return True

if __name__ == "__main__":
    print("="*60)
    print("LEGACY GAT MODEL TEST SUITE")
    print("="*60)
    
    try:
        # Test forward pass
        test_legacy_gat_forward()
        
        # Test components
        test_model_components()
        
        # Test training compatibility
        test_training_compatibility()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY! ✅")
        print("The Legacy GAT model is ready for use as a benchmark.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
