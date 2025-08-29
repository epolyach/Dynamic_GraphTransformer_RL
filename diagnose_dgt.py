#!/usr/bin/env python3
"""
Diagnose why DGT+RL doesn't learn compared to GT+RL.
"""

import torch
import numpy as np
from src.utils.config import load_config
from src.models.model_factory import ModelFactory

def test_gradient_flow(model, config):
    """Test if gradients flow properly through the model."""
    model.train()
    
    # Create simple test instance
    instances = [{
        'coords': np.array([[0.5, 0.5], [0.2, 0.3], [0.7, 0.8], [0.3, 0.9]]),
        'demands': np.array([0.0, 5.0, 3.0, 4.0]),
        'capacity': 10.0
    }]
    
    # Forward pass
    routes, log_probs, entropy = model(instances, greedy=False, config=config)
    
    # Check outputs
    print(f"Routes: {routes}")
    print(f"Log probs: {log_probs}")
    print(f"Entropy: {entropy}")
    
    # Create fake loss and backward
    loss = -log_probs.mean() - 0.01 * entropy.mean()
    loss.backward()
    
    # Check gradients
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std
            }
    
    return grad_stats

def compare_models():
    """Compare GT+RL and DGT+RL gradient flow."""
    config = load_config('configs/tiny.yaml')
    
    print("="*60)
    print("Testing GT+RL")
    print("="*60)
    gt_model = ModelFactory.create_model("GT+RL", config)
    gt_grads = test_gradient_flow(gt_model, config)
    
    print("\n" + "="*60)
    print("Testing DGT+RL")
    print("="*60)
    dgt_model = ModelFactory.create_model("DGT+RL", config)
    dgt_grads = test_gradient_flow(dgt_model, config)
    
    # Analyze gradient statistics
    print("\n" + "="*60)
    print("Gradient Analysis")
    print("="*60)
    
    # Check for zero gradients
    print("\nZero gradients in GT+RL:")
    zero_grads_gt = [k for k, v in gt_grads.items() if v['norm'] < 1e-8]
    print(f"  {len(zero_grads_gt)} parameters with zero gradients")
    if zero_grads_gt[:5]:
        print(f"  Examples: {zero_grads_gt[:5]}")
    
    print("\nZero gradients in DGT+RL:")
    zero_grads_dgt = [k for k, v in dgt_grads.items() if v['norm'] < 1e-8]
    print(f"  {len(zero_grads_dgt)} parameters with zero gradients")
    if zero_grads_dgt[:5]:
        print(f"  Examples: {zero_grads_dgt[:5]}")
    
    # Check for exploding gradients
    print("\nLarge gradients (norm > 10) in GT+RL:")
    large_grads_gt = [(k, v['norm']) for k, v in gt_grads.items() if v['norm'] > 10]
    print(f"  {len(large_grads_gt)} parameters")
    
    print("\nLarge gradients (norm > 10) in DGT+RL:")
    large_grads_dgt = [(k, v['norm']) for k, v in dgt_grads.items() if v['norm'] > 10]
    print(f"  {len(large_grads_dgt)} parameters")
    
    # Average gradient norms
    avg_norm_gt = np.mean([v['norm'] for v in gt_grads.values()])
    avg_norm_dgt = np.mean([v['norm'] for v in dgt_grads.values()])
    
    print(f"\nAverage gradient norm:")
    print(f"  GT+RL: {avg_norm_gt:.6f}")
    print(f"  DGT+RL: {avg_norm_dgt:.6f}")
    
    # Check specific critical components
    print("\nCritical component gradients (DGT+RL):")
    critical_components = [
        'memory_bank', 'edge_processor', 'graph_adapter', 
        'temporal_attention', 'refinement_layers', 'update_schedule'
    ]
    
    for comp in critical_components:
        comp_grads = {k: v for k, v in dgt_grads.items() if comp in k}
        if comp_grads:
            avg_norm = np.mean([v['norm'] for v in comp_grads.values()])
            print(f"  {comp}: avg norm = {avg_norm:.6f}")
        else:
            print(f"  {comp}: NO GRADIENTS")

if __name__ == "__main__":
    compare_models()
