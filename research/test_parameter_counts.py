#!/usr/bin/env python3
"""
Test script to validate parameter counts of redesigned DGT variants.

This script creates instances of each model and counts their parameters
to verify they match the target counts.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.dgt_super import DynamicGraphTransformerSuper
from models.dgt_ultra import DynamicGraphTransformerUltra  
from models.dgt_lite import DynamicGraphTransformerLite

def count_parameters(model):
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())

def test_model_parameters():
    """Test parameter counts for all redesigned models."""
    
    # Standard config for model initialization
    input_dim = 3
    hidden_dim = 128  # Will be overridden by model internals
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    feedforward_multiplier = 2
    config = {}  # Empty config for testing
    
    models_to_test = [
        ("DGT-Super", DynamicGraphTransformerSuper, "~50K (smallest)"),
        ("DGT-Ultra", DynamicGraphTransformerUltra, "~80K (middle)"),
        ("DGT-Lite", DynamicGraphTransformerLite, "~120K (largest)"),
    ]
    
    print("🔍 Testing Parameter Counts for Redesigned DGT Models")
    print("=" * 60)
    
    for model_name, model_class, target in models_to_test:
        try:
            # Create model instance
            model = model_class(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                feedforward_multiplier=feedforward_multiplier,
                config=config
            )
            
            # Count parameters
            param_count = count_parameters(model)
            
            # Display results
            print(f"{model_name:12}: {param_count:>8,} parameters (target: {target})")
            
            # Show model architecture details
            print(f"{'':14}Hidden: {model.hidden_dim}, Heads: {model.num_heads}, Layers: {model.num_layers}")
            
        except Exception as e:
            print(f"{model_name:12}: ❌ Error creating model: {e}")
    
    print("=" * 60)
    print("Reference Models:")
    print(f"{'GAT+RL':12}: {'364,801':>8} parameters (baseline)")
    print(f"{'Original DGT':12}: {'630,146':>8} parameters")

def test_model_forward():
    """Test that models can perform forward pass."""
    print("\n🚀 Testing Forward Pass...")
    
    # Create dummy CVRP instance
    dummy_instance = {
        'coords': torch.tensor([[0.5, 0.5], [0.1, 0.2], [0.8, 0.9]], dtype=torch.float32),
        'demands': torch.tensor([0, 5, 3], dtype=torch.int32),
        'capacity': 10
    }
    
    dummy_config = {
        'inference': {
            'max_steps_multiplier': 2,
            'default_temperature': 1.0,
            'masked_score_value': -1e9,
            'log_prob_epsilon': 1e-12
        }
    }
    
    models_to_test = [
        ("DGT-Super", DynamicGraphTransformerSuper),
        ("DGT-Ultra", DynamicGraphTransformerUltra), 
        ("DGT-Lite", DynamicGraphTransformerLite),
    ]
    
    for model_name, model_class in models_to_test:
        try:
            model = model_class(3, 128, 4, 4, 0.1, 2, {})
            model.eval()
            
            with torch.no_grad():
                routes, log_probs, entropy = model(
                    instances=[dummy_instance],
                    max_steps=10,
                    temperature=1.0,
                    greedy=True,
                    config=dummy_config
                )
            
            print(f"{model_name:12}: ✅ Forward pass successful")
            print(f"{'':14}Route: {routes[0]}, Cost: {log_probs[0]:.3f}")
            
        except Exception as e:
            print(f"{model_name:12}: ❌ Forward pass failed: {e}")

if __name__ == "__main__":
    test_model_parameters()
    test_model_forward()
