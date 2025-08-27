#!/usr/bin/env python3
"""
Add GT-Greedy baseline results to existing analysis without training.
GT-Greedy uses a Graph Transformer with greedy decoding (no learning).

Usage:
    python add_gt_greedy_baseline.py --config configs/medium.yaml
"""

import torch
import numpy as np
import os
import sys
import yaml
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def compute_route_cost(route, coords):
    """Compute the total cost of a route given coordinates."""
    if len(route) < 2:
        return 0.0
    
    cost = 0.0
    coords_np = np.array(coords)
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        # Euclidean distance
        dist = np.linalg.norm(coords_np[to_node] - coords_np[from_node])
        cost += dist
    
    # Add return to depot if not already there
    if route[-1] != 0:
        dist = np.linalg.norm(coords_np[0] - coords_np[route[-1]])
        cost += dist
    
    return cost

def main():
    parser = argparse.ArgumentParser(description='Add GT-Greedy baseline to existing results')
    parser.add_argument('--config', type=str, default='configs/medium.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load default config and merge
    with open('configs/default.yaml', 'r') as f:
        default_config = yaml.safe_load(f)
    
    def deep_merge(base, override):
        if override is None:
            return base
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    config = deep_merge(default_config, config)
    
    print(f"📊 Adding GT-Greedy baseline using config: {args.config}")
    print(f"   Working directory: {config['working_dir_path']}")
    
    # Load existing analysis
    analysis_path = os.path.join(config['working_dir_path'], 'analysis', 'enhanced_comparative_study.pt')
    if os.path.exists(analysis_path):
        print(f"✅ Loading existing analysis from {analysis_path}")
        data = torch.load(analysis_path, map_location='cpu', weights_only=False)
        results = data.get('results', {})
        training_times = data.get('training_times', {})
    else:
        print("❌ No existing analysis found.")
        return
    
    # Check if GT-Greedy already exists
    if 'GT-Greedy' in results:
        print("⚠️ GT-Greedy already exists in results. Overwriting...")
    
    print("🔧 Creating GT-Greedy baseline evaluation...")
    
    # Import necessary modules
    from src.models.greedy_gt import GraphTransformerGreedy
    from src.data.enhanced_generator import EnhancedCVRPGenerator
    
    # Create model
    model_config = config['model']
    device = torch.device('cpu')  # Use CPU for consistency
    
    model = GraphTransformerGreedy(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        dropout=model_config.get('transformer_dropout', 0.1),
        feedforward_multiplier=model_config.get('feedforward_multiplier', 2),
        config=config
    ).to(device)
    
    # Initialize with random weights (no training needed for greedy baseline)
    model.eval()
    
    # Create data generator with full config
    generator = EnhancedCVRPGenerator(config)
    
    # Evaluate on validation instances
    num_eval_instances = 1000  # Evaluate on 1000 instances for stable estimate
    batch_size = 100
    num_customers = config['problem']['num_customers']
    
    print(f"   Evaluating on {num_eval_instances} instances with greedy decoding...")
    print(f"   Problem size: {num_customers} customers")
    
    all_costs = []
    
    for batch_start in range(0, num_eval_instances, batch_size):
        batch_end = min(batch_start + batch_size, num_eval_instances)
        actual_batch_size = batch_end - batch_start
        
        # Generate batch using the same method as training
        batch = generator.generate_batch(
            batch_size=actual_batch_size,
            # Use standard uniform instances
            num_customers=num_customers,
            capacity=config['problem']['vehicle_capacity'],
            coord_range=config['problem']['coord_range'],
            demand_range=config['problem']['demand_range']
        )
        
        with torch.no_grad():
            # Forward pass with greedy decoding
            # GT-Greedy returns routes, log_probs (zeros), entropy (zeros)
            routes, log_probs, entropy = model(batch, greedy=True, config=config)
            
            # Compute costs from routes
            batch_costs = []
            for i, route in enumerate(routes):
                cost = compute_route_cost(route, batch[i]['coords'])
                batch_costs.append(cost)
            
            all_costs.extend(batch_costs)
        
        if (batch_start // batch_size + 1) % 2 == 0:
            print(f"      Processed {batch_end}/{num_eval_instances} instances...")
    
    # Calculate statistics
    all_costs = np.array(all_costs)
    avg_cost = np.mean(all_costs)
    std_cost = np.std(all_costs)
    avg_cost_per_customer = avg_cost / num_customers
    std_cost_per_customer = std_cost / num_customers
    
    print(f"\n   🎯 GT-Greedy results:")
    print(f"      Average cost: {avg_cost:.4f} ± {std_cost:.4f}")
    print(f"      Cost per customer: {avg_cost_per_customer:.4f} ± {std_cost_per_customer:.4f}")
    
    # Add to results
    results['GT-Greedy'] = {
        'final_val_cost': avg_cost_per_customer,
        'val_cost_std': std_cost_per_customer,
        'train_costs': [],  # No training for greedy
        'val_costs': [avg_cost_per_customer],  # Single evaluation
        'history': {
            'final_val_cost': avg_cost_per_customer,
            'val_cost_std': std_cost_per_customer,
            'train_costs': [],
            'val_costs': [avg_cost_per_customer]
        }
    }
    
    # No training time for greedy baseline
    training_times['GT-Greedy'] = 0.0
    
    # Save model state for consistency
    pytorch_dir = os.path.join(config['working_dir_path'], 'pytorch')
    os.makedirs(pytorch_dir, exist_ok=True)
    
    model_path = os.path.join(pytorch_dir, 'model_gt_greedy.pt')
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"      Model parameters: {param_count:,}")
    
    torch.save({
        'model_name': 'GT-Greedy',
        'model_state_dict': model.state_dict(),
        'results': results['GT-Greedy'],
        'training_time': 0.0,
        'config': config
    }, model_path)
    print(f"   📁 Saved model to {model_path}")
    
    # Update and save analysis
    data['results'] = results
    data['training_times'] = training_times
    torch.save(data, analysis_path)
    print(f"   ✅ Updated analysis saved to {analysis_path}")
    
    print("\n📊 All models in analysis:")
    print("   " + "-" * 50)
    for name in sorted(results.keys()):
        val_cost = results[name].get('final_val_cost', 
                                     results[name].get('history', {}).get('final_val_cost', 'N/A'))
        if isinstance(val_cost, (int, float)):
            print(f"   {name:20s}: {val_cost:.4f}/customer")
        else:
            print(f"   {name:20s}: {val_cost}")
    print("   " + "-" * 50)

if __name__ == '__main__':
    main()
