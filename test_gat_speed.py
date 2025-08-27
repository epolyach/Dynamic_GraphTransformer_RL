#!/usr/bin/env python3
"""Quick speed test for GAT model."""

import torch
import time
import sys
sys.path.append('.')

from src.models.legacy_gat import LegacyGATModel

def test_speed():
    # Create small instances
    instances = [
        {
            'coords': [(0.5, 0.5), (0.2, 0.3), (0.8, 0.7), (0.3, 0.9), (0.6, 0.4)],
            'demands': [0, 15, 10, 12, 8],
            'capacity': 30
        } for _ in range(4)  # Small batch
    ]
    
    # Initialize model
    model = LegacyGATModel(
        node_input_dim=3,
        edge_input_dim=1,
        hidden_dim=128,
        edge_dim=16,
        layers=2,  # Fewer layers for testing
        config={'inference': {'default_temperature': 1.0, 'max_steps_multiplier': 2}}
    )
    model.eval()
    
    print(f"Testing GAT model with batch size {len(instances)}, {len(instances[0]['coords'])} nodes each")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    print("\nTiming forward pass...")
    with torch.no_grad():
        start = time.time()
        routes, log_probs, entropy = model(instances, max_steps=20, temperature=1.0, greedy=True)
        end = time.time()
    
    print(f"Forward pass took {end - start:.3f} seconds")
    print(f"Generated routes:")
    for i, route in enumerate(routes):
        print(f"  Instance {i}: {route}")
    
    # Test with gradient computation
    print("\nTiming forward pass with gradients...")
    model.train()
    start = time.time()
    routes, log_probs, entropy = model(instances, max_steps=20, temperature=1.0, greedy=False)
    loss = -log_probs.mean()
    loss.backward()
    end = time.time()
    
    print(f"Forward + backward pass took {end - start:.3f} seconds")
    print(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    test_speed()
