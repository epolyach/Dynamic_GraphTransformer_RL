import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_analysis_for_gat_optimized():
    """Create analysis file for GAT optimized training to enable comparative plotting"""
    
    # Paths
    base_dir = Path('/Users/evgeny.polyachenko/WORK/EUROSENDER/Conf_Papers/ICORES/Dynamic_GraphTransformer_RL/training_cpu/results/tiny_gat')
    analysis_dir = base_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    # Load GAT model and CSV data
    gat_model_path = base_dir / 'pytorch' / 'model_gat_rl.pt'
    gat_csv_path = base_dir / 'csv' / 'history_gat_rl.csv'
    
    # Load the model file
    gat_model = torch.load(gat_model_path, map_location='cpu', weights_only=False)
    
    # Load CSV history
    gat_df = pd.read_csv(gat_csv_path)
    
    # Extract training metrics
    train_losses = gat_df['train_loss'].tolist()
    train_costs = gat_df['train_cost'].tolist()
    
    # Extract validation data (only where val_cost is not NaN)
    val_mask = gat_df['val_cost'].notna()
    val_epochs = gat_df.loc[val_mask, 'epoch'].tolist()
    val_costs = gat_df.loc[val_mask, 'val_cost'].tolist()
    
    # Create results structure
    results = {
        'GAT+RL': {
            'history': {
                'train_losses': train_losses,
                'train_costs': train_costs,
                'val_costs': val_costs,
                'val_epochs': val_epochs,
                'final_val_cost': val_costs[-1] if val_costs else None,
                'best_val_cost': min(val_costs) if val_costs else None,
            },
            'final_val_cost': val_costs[-1] if val_costs else None,
            'training_time': gat_model.get('training_time', 106.8),  # From training output
        }
    }
    
    # Create config structure (from tiny_gat_optimized.yaml)
    config = {
        'problem': {
            'num_customers': 7,
            'vehicle_capacity': 30,
            'coord_range': 100,
            'demand_range': [1, 9]
        },
        'training': {
            'num_epochs': 80,
            'num_instances': 51840,
            'batch_size': 128,
            'num_batches_per_epoch': 5
        },
        'working_dir_path': str(base_dir)
    }
    
    # Create the analysis file
    analysis_data = {
        'results': results,
        'config': config,
        'training_times': {'GAT+RL': 106.8}
    }
    
    # Save the analysis file
    analysis_path = analysis_dir / 'enhanced_comparative_study.pt'
    torch.save(analysis_data, analysis_path)
    print(f"Created analysis file: {analysis_path}")
    
    # Also copy other model results from tiny directory if they exist
    tiny_dir = Path('/Users/evgeny.polyachenko/WORK/EUROSENDER/Conf_Papers/ICORES/Dynamic_GraphTransformer_RL/training_cpu/results/tiny')
    
    # Try to load the tiny analysis to get other models
    tiny_analysis_paths = [
        tiny_dir / 'analysis' / 'enhanced_comparative_study.pt',
        tiny_dir / 'analysis' / 'comparative_study_complete.pt'
    ]
    
    for tiny_analysis_path in tiny_analysis_paths:
        if tiny_analysis_path.exists():
            print(f"Loading other models from: {tiny_analysis_path}")
            tiny_data = torch.load(tiny_analysis_path, map_location='cpu', weights_only=False)
            
            # Merge with GAT optimized results
            if 'results' in tiny_data:
                for model_name, model_results in tiny_data['results'].items():
                    if model_name != 'GAT+RL':  # Don't overwrite GAT optimized
                        results[model_name] = model_results
            
            # Update training times
            if 'training_times' in tiny_data:
                for model_name, time in tiny_data['training_times'].items():
                    if model_name != 'GAT+RL':
                        analysis_data['training_times'][model_name] = time
            
            # Save updated analysis
            analysis_data['results'] = results
            torch.save(analysis_data, analysis_path)
            print(f"Updated analysis with {len(results)} models")
            break
    
    # Copy model files from tiny to tiny_gat for comparative plotting
    tiny_pytorch = tiny_dir / 'pytorch'
    gat_pytorch = base_dir / 'pytorch'
    
    if tiny_pytorch.exists():
        for model_file in ['model_gt_rl.pt', 'model_dgt_rl.pt', 'model_gt_greedy.pt']:
            src = tiny_pytorch / model_file
            dst = gat_pytorch / model_file
            if src.exists() and not dst.exists():
                import shutil
                shutil.copy2(src, dst)
                print(f"Copied {model_file} to GAT directory")
    
    # Copy CSV files as well for the comparative plot
    tiny_csv = tiny_dir / 'csv'
    gat_csv = base_dir / 'csv'
    
    if tiny_csv.exists():
        for csv_file in ['history_gt_rl.csv', 'history_dgt_rl.csv', 'history_gt_greedy.csv']:
            src = tiny_csv / csv_file
            dst = gat_csv / csv_file
            if src.exists() and not dst.exists():
                import shutil
                shutil.copy2(src, dst)
                print(f"Copied {csv_file} to GAT directory")
    
    print("Analysis preparation complete!")
    return analysis_path

if __name__ == "__main__":
    create_analysis_for_gat_optimized()
