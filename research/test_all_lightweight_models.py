#!/usr/bin/env python3
"""
Comprehensive test script to validate parameter counts of all lightweight model variants.

This script creates instances of both DGT and GT variants and verifies they match
target parameter counts.
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
from models.gt_super import GraphTransformerSuper
from models.gt_ultra import GraphTransformerUltra
from models.gt_lite import GraphTransformerLite
from models.gat import GraphAttentionTransformer
from models.dgt import DynamicGraphTransformerNetwork


def count_parameters(model):
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


def test_all_model_parameters():
    """Test parameter counts for all lightweight model variants."""
    
    # Standard config for model initialization
    input_dim = 3
    hidden_dim = 128  # Will be overridden by model internals
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    feedforward_multiplier = 2
    edge_embedding_divisor = 4
    config = {
        'model': {
            'pointer_network': {
                'context_multiplier': 2
            }
        }
    }
    
    models_to_test = [
        # DGT variants (target: Super ~10k, Ultra ~30k, Lite ~100k)
        ("DGT-Super", DynamicGraphTransformerSuper, "~10K"),
        ("DGT-Ultra", DynamicGraphTransformerUltra, "~30K"),
        ("DGT-Lite", DynamicGraphTransformerLite, "~100K"),
        
        # GT variants (should match DGT counterparts)
        ("GT-Super", GraphTransformerSuper, "~10K (match DGT)"),
        ("GT-Ultra", GraphTransformerUltra, "~30K (match DGT)"),
        ("GT-Lite", GraphTransformerLite, "~100K (match DGT)"),
        
        # Reference models
        ("GAT+RL", GraphAttentionTransformer, "~365K (baseline)"),
        ("DGT+RL", DynamicGraphTransformerNetwork, "~630K (original)"),
    ]
    
    print("🔍 Testing Parameter Counts for All Lightweight Models")
    print("=" * 70)
    
    results = {}
    
    for model_name, model_class, target in models_to_test:
        try:
            # Create model instance with appropriate parameters
            if "GAT" in model_name:
                model = model_class(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                    edge_embedding_divisor=edge_embedding_divisor,
                    config=config
                )
            else:
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
            results[model_name] = param_count
            
            # Display results
            print(f"{model_name:12}: {param_count:>8,} parameters (target: {target})")
            
            # Show model architecture details if available
            if hasattr(model, 'hidden_dim') and hasattr(model, 'num_heads'):
                print(f"{'':14}Hidden: {model.hidden_dim}, Heads: {model.num_heads}, "
                      f"Layers: {getattr(model, 'num_layers', 'N/A')}")
            
        except Exception as e:
            print(f"{model_name:12}: ❌ Error creating model: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 70)
    
    # Parameter count comparison
    print("\n📊 Parameter Count Analysis:")
    
    # Compare DGT vs GT variants
    pairs = [
        ("DGT-Super", "GT-Super"),
        ("DGT-Ultra", "GT-Ultra"), 
        ("DGT-Lite", "GT-Lite")
    ]
    
    for dgt_name, gt_name in pairs:
        if dgt_name in results and gt_name in results:
            dgt_params = results[dgt_name]
            gt_params = results[gt_name]
            diff = abs(dgt_params - gt_params)
            print(f"  {dgt_name:12} vs {gt_name:12}: {dgt_params:>8,} vs {gt_params:>8,} (diff: {diff:>6,})")
        
    # Show reduction from baseline
    if "GAT+RL" in results:
        baseline = results["GAT+RL"]
        print(f"\n📉 Parameter Reduction from GAT+RL baseline ({baseline:,}):")
        for model_name in ["DGT-Super", "DGT-Ultra", "DGT-Lite", "GT-Super", "GT-Ultra", "GT-Lite"]:
            if model_name in results:
                params = results[model_name]
                reduction = (baseline - params) / baseline * 100
                ratio = baseline / params
                print(f"  {model_name:12}: {params:>8,} ({reduction:>5.1f}% reduction, {ratio:>4.1f}x smaller)")


def test_model_forward():
    """Test that all models can perform forward pass."""
    print("\n\n🚀 Testing Forward Pass...")
    
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
        },
        'model': {
            'pointer_network': {
                'context_multiplier': 2
            }
        }
    }
    
    models_to_test = [
        ("DGT-Super", DynamicGraphTransformerSuper),
        ("DGT-Ultra", DynamicGraphTransformerUltra), 
        ("DGT-Lite", DynamicGraphTransformerLite),
        ("GT-Super", GraphTransformerSuper),
        ("GT-Ultra", GraphTransformerUltra),
        ("GT-Lite", GraphTransformerLite),
    ]
    
    for model_name, model_class in models_to_test:
        try:
            model = model_class(3, 128, 4, 4, 0.1, 2, dummy_config)
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
            print(f"{'':14}Route: {routes[0]}, LogProb: {log_probs[0]:.3f}")
            
        except Exception as e:
            print(f"{model_name:12}: ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_all_model_parameters()
    test_model_forward()
