#!/usr/bin/env python3
"""
Comprehensive CVRP training script for all models including new lightweight variants.
Trains both original models and the new lite/ultra variants with 64 epochs.
"""

import os
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
from src.pipelines.train import train_one_model, set_seeds
from src.models.pointer import BaselinePointerNetwork
from src.models.gt import GraphTransformerNetwork
from src.models.greedy_gt import GraphTransformerGreedy
from src.models.dgt import DynamicGraphTransformerNetwork
from src.models.gat import GraphAttentionTransformer

# Import the lightweight variants
from src.models.gt_lite import GraphTransformerLite
from src.models.dgt_lite import DynamicGraphTransformerLite
from src.models.gt_ultra import GraphTransformerUltra
from src.models.dgt_ultra import DynamicGraphTransformerUltra
from src.models.dgt_super import DynamicGraphTransformerSuper


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)


def model_key(name: str) -> str:
    """Map display names to file keys"""
    mapping = {
        'Pointer+RL': 'pointer_rl',
        'GT+RL': 'gt_rl',
        'DGT+RL': 'dgt_rl',
        'GAT+RL': 'gat_rl',
        'GT-Greedy': 'gt_greedy',
        'GT-Lite+RL': 'gt_lite_rl',
        'DGT-Lite+RL': 'dgt_lite_rl',
        'GT-Ultra+RL': 'gt_ultra_rl',
        'DGT-Ultra+RL': 'dgt_ultra_rl',
        'DGT-Super+RL': 'dgt_super_rl',
    }
    return mapping.get(name, name.lower().replace(' ', '_').replace('+', '_').replace('-', '_'))


def write_history_csv(model_name: str, history: dict, config: dict, base_dir: str, logger) -> None:
    """Write model training history to CSV file."""
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    num_epochs = int(config['training']['num_epochs'])
    val_freq = int(config['training']['validation_frequency'])
    
    train_losses = list(history.get('train_losses', []))
    train_costs = list(history.get('train_costs', []))
    val_costs_seq = list(history.get('val_costs', []))
    
    # Create epoch series (0 to num_epochs inclusive)
    total_rows = num_epochs + 1
    train_loss_series = [float('nan')] * total_rows
    train_cost_series = [float('nan')] * total_rows
    val_cost_series = [float('nan')] * total_rows
    
    # Map training data
    for i, v in enumerate(train_losses):
        if i < num_epochs:
            train_loss_series[i] = v
    for i, v in enumerate(train_costs):
        if i < num_epochs:
            train_cost_series[i] = v
            
    # Carry forward last values to final epoch
    if num_epochs > 0:
        if len(train_losses) > 0:
            train_loss_series[num_epochs] = train_losses[-1]
        if len(train_costs) > 0:
            train_cost_series[num_epochs] = train_costs[-1]
    
    # Map validation costs to correct epochs
    expected_val_epochs = [ep for ep in range(0, num_epochs + 1)
                          if (ep % max(1, val_freq) == 0) or (ep == num_epochs)]
    
    for i, ep in enumerate(expected_val_epochs):
        if i < len(val_costs_seq):
            val_cost_series[ep] = val_costs_seq[i]
    
    # Ensure final epoch gets the last validation cost
    if len(val_costs_seq) > 0:
        val_cost_series[num_epochs] = val_costs_seq[-1]
    
    df = pd.DataFrame({
        'epoch': list(range(0, num_epochs + 1)),
        'train_loss': train_loss_series,
        'train_cost': train_cost_series,
        'val_cost': val_cost_series,
    })
    
    out_path = os.path.join(csv_dir, f'history_{model_key(model_name)}.csv')
    df.to_csv(out_path, index=False)
    logger.info(f'📊 Saved history CSV: {out_path}')


def save_models_and_analysis(results, training_times, models, config, base_dir, logger):
    """Save trained models and comprehensive analysis."""
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    analysis_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(pytorch_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    # Save individual models
    for model_name, model in models.items():
        path = os.path.join(pytorch_dir, f'model_{model_key(model_name)}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': config,
            'results': results[model_name]['history'],
            'training_time': training_times[model_name],
        }, path)
        logger.info(f'💾 Saved model checkpoint: {path}')

    # Save comprehensive analysis
    analysis_data = {
        'results': {k: v['history'] for k, v in results.items()},
        'training_times': training_times,
        'config': config,
    }
    
    analysis_path = os.path.join(analysis_dir, 'comparative_study_complete.pt')
    torch.save(analysis_data, analysis_path)
    logger.info(f'📦 Saved comparative analysis: {analysis_path}')


def build_model(name: str, config: dict):
    """Build model by name with given configuration."""
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    num_heads = config['model']['num_heads']
    num_layers = config['model']['num_layers']
    dropout = config['model']['transformer_dropout']
    ff_mult = config['model']['feedforward_multiplier']
    edge_div = config['model']['edge_embedding_divisor']
    
    if name == 'Pointer+RL':
        return BaselinePointerNetwork(input_dim, hidden_dim, config)
    elif name == 'GT-Greedy':
        return GraphTransformerGreedy(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'GT+RL':
        return GraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'DGT+RL':
        return DynamicGraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'GAT+RL':
        return GraphAttentionTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout, edge_div, config)
    # New lightweight variants
    elif name == 'GT-Lite+RL':
        return GraphTransformerLite(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'DGT-Lite+RL':
        return DynamicGraphTransformerLite(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'GT-Ultra+RL':
        return GraphTransformerUltra(input_dim, 64, num_heads, num_layers, dropout, ff_mult, config)  # 64 hidden_dim
    elif name == 'DGT-Ultra+RL':
        return DynamicGraphTransformerUltra(input_dim, 64, num_heads, num_layers, dropout, ff_mult, config)  # 64 hidden_dim
    elif name == 'DGT-Super+RL':
        return DynamicGraphTransformerSuper(input_dim, 64, num_heads, num_layers, dropout, ff_mult, config)  # 64 hidden_dim
    else:
        raise ValueError(f'Unknown model name: {name}')


def main():
    parser = argparse.ArgumentParser(description='Train all CVRP models including lightweight variants')
    parser.add_argument('--config', type=str, default='configs/small.yaml', 
                       help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger = setup_logging()
    
    # Set up environment
    set_seeds(config.get('experiment', {}).get('random_seed', 42))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    
    base_dir = str(Path(config.get('working_dir_path', 'results')).as_posix())
    os.makedirs(base_dir, exist_ok=True)
    
    logger.info("🚀 Starting Comprehensive CVRP Model Training")
    logger.info(f"📂 Working directory: {base_dir}")
    logger.info(f"🔢 Training epochs: {config['training']['num_epochs']}")
    logger.info(f"👥 Customers: {config['problem']['num_customers']}")
    
    # All models to train (original + lightweight variants)
    all_models = [
        'Pointer+RL',    # Reference baseline
        'GT-Greedy',     # Non-RL baseline
        'GT+RL',         # Original GT
        'DGT+RL',        # Original DGT  
        'GAT+RL',        # Best performing baseline
        # New lightweight variants
        'GT-Lite+RL',    # Lite GT variant
        'DGT-Lite+RL',   # Lite DGT variant
        'GT-Ultra+RL',   # Ultra-light GT (64 hidden_dim)
        'DGT-Ultra+RL',  # Ultra-light DGT (64 hidden_dim)
        'DGT-Super+RL',  # Super-light DGT (64 hidden_dim, minimal params)
    ]
    
    logger.info(f"🎯 Training {len(all_models)} models:")
    for i, name in enumerate(all_models, 1):
        model = build_model(name, config)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"  {i:2d}. {name}: {param_count:,} parameters")
    
    # Prepare output directories
    csv_dir = os.path.join(base_dir, 'csv')
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(pytorch_dir, exist_ok=True)
    
    # Train all models
    results = {}
    training_times = {}
    models = {}
    
    total_start_time = time.time()
    
    for i, name in enumerate(all_models, 1):
        logger.info(f"\n🚀 Training model {i}/{len(all_models)}: {name}")
        
        # Build and train model
        model = build_model(name, config)
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"   📊 Parameters: {param_count:,}")
        
        # Train the model
        history, training_time, trained_model = train_one_model(model, name, config, logger.info)
        
        # Store results
        results[name] = {'history': history}
        training_times[name] = training_time
        models[name] = trained_model
        
        # Write CSV immediately
        write_history_csv(name, history, config, base_dir, logger)
        
        # Log results
        final_val_cost = history.get('final_val_cost', float('nan'))
        logger.info(f"✅ {name} completed:")
        logger.info(f"   ⏱️  Training time: {training_time:.1f}s")
        logger.info(f"   📈 Final validation cost: {final_val_cost:.4f}")
    
    total_time = time.time() - total_start_time
    logger.info(f"\n⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")
    
    # Save models and analysis
    save_models_and_analysis(results, training_times, models, config, base_dir, logger)
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL RESULTS SUMMARY")
    logger.info("="*80)
    
    # Sort by performance
    sorted_results = sorted(
        [(name, results[name]['history'].get('final_val_cost', float('inf')), 
         sum(p.numel() for p in models[name].parameters()))
         for name in all_models],
        key=lambda x: x[1]
    )
    
    logger.info(f"{'Model':<20} {'Parameters':<12} {'Val Cost':<10} {'Time (s)':<10}")
    logger.info("-" * 60)
    
    for name, val_cost, params in sorted_results:
        time_str = f"{training_times[name]:.1f}s"
        logger.info(f"{name:<20} {params:<12,} {val_cost:<10.4f} {time_str:<10}")
    
    best_model, best_cost, best_params = sorted_results[0]
    logger.info("="*80)
    logger.info(f"🏆 Best model: {best_model}")
    logger.info(f"   📈 Validation cost: {best_cost:.4f}")
    logger.info(f"   🔢 Parameters: {best_params:,}")
    logger.info(f"   ⏱️  Training time: {training_times[best_model]:.1f}s")
    logger.info("="*80)
    
    logger.info(f"✅ All results saved to: {base_dir}")


if __name__ == "__main__":
    main()
