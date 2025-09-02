#!/usr/bin/env python3
"""
Regenerate comparative analysis from existing model files and CSV histories.
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def regenerate_analysis(base_dir):
    """Regenerate analysis from existing models and CSV files."""
    
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    csv_dir = os.path.join(base_dir, 'csv')
    analysis_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    results = {}
    training_times = {}
    
    # Process each model
    for model_file in os.listdir(pytorch_dir):
        if not model_file.startswith('model_') or not model_file.endswith('.pt'):
            continue
            
        model_key = model_file[6:-3]  # Remove 'model_' and '.pt'
        # Convert model key to display name
        if model_key == 'gat_rl':
            model_name = 'GAT+RL'
        elif model_key == 'gt_rl':
            model_name = 'GT+RL'
        elif model_key == 'dgt_rl':
            model_name = 'DGT+RL'
        elif model_key == 'gt_greedy':
            model_name = 'GT-Greedy'
        else:
            model_name = model_key.upper()
        
        print(f"Processing {model_name}...")
        
        # Load model data
        model_path = os.path.join(pytorch_dir, model_file)
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get training time
        training_times[model_name] = model_data.get('training_time', 0.0)
        
        # Load CSV history
        csv_path = os.path.join(csv_dir, f'history_{model_key}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Extract costs
            train_costs = df['train_cost'].tolist()
            val_costs = df['val_cost'].dropna().tolist()
            
            # Find best validation cost
            if val_costs:
                best_val = min(val_costs)
                final_val = val_costs[-1]
            else:
                best_val = train_costs[-1] if train_costs else 0.0
                final_val = best_val
            
            results[model_name] = {
                'history': {
                    'train_costs': train_costs,
                    'val_costs': val_costs,
                    'final_val_cost': final_val,
                },
                'final_val_cost': final_val,
                'best_val_cost': best_val,
                'training_time': training_times[model_name]
            }
            
            print(f"  - Final val cost: {final_val:.4f}")
            print(f"  - Best val cost: {best_val:.4f}")
    
    # Load config
    from src.utils.config import load_config
    config_path = 'configs/tiny.yaml'
    config = load_config(config_path)
    
    # Save enhanced analysis
    analysis_path = os.path.join(analysis_dir, 'enhanced_comparative_study.pt')
    torch.save({
        'results': results,
        'training_times': training_times,
        'config': config
    }, analysis_path)
    
    print(f"\n✅ Analysis saved to {analysis_path}")
    print(f"Models processed: {list(results.keys())}")

if __name__ == "__main__":
    base_dir = "training_cpu/results/tiny"
    regenerate_analysis(base_dir)
