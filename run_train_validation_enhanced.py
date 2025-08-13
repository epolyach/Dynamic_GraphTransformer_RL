#!/usr/bin/env python3
"""
MODIFIED VERSION of your original run_train_validation.py with enhanced training
This shows exactly what I added to improve performance while keeping your structure
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
# === ENHANCED IMPORT ===
from src.training.advanced_trainer import advanced_train_model  # NEW: Enhanced trainer
from src.pipelines.train import train_all_models, train_one_model, set_seeds, generate_cvrp_instance
from src.models.pointer import BaselinePointerNetwork
from src.models.gt import GraphTransformerNetwork
from src.models.greedy_gt import GraphTransformerGreedy
from src.models.dgt import DynamicGraphTransformerNetwork
from src.models.gat import GraphAttentionTransformer


def setup_logging(config=None):
    level = logging.INFO
    format_str = '%(message)s'
    if config and 'logging' in config:
        level_str = config['logging'].get('level', 'INFO')
        format_str = config['logging'].get('format', '%(message)s')
        level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger(__name__)


def model_key(name: str) -> str:
    mapping = {
        'Pointer+RL': 'pointer_rl',
        'GT+RL': 'gt_rl',
        'DGT+RL': 'dgt_rl',
        'GAT+RL': 'gat_rl',
        'GT-Greedy': 'gt_greedy',
        'GAT+RL (legacy)': 'gat_rl_legacy',
    }
    return mapping.get(name, name.lower().replace(' ', '_').replace('+', '_').replace('-', '_'))


def write_history_csv(model_name: str, history: dict, config: dict, base_dir: str, logger) -> None:
    """Enhanced CSV writing with new metrics - includes epoch 0"""
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    num_epochs = int(config['training']['num_epochs'])
    val_freq = int(config['training']['validation_frequency'])
    
    # Basic series
    train_losses = list(history.get('train_losses', []))
    train_costs = list(history.get('train_costs', []))
    val_costs_seq = list(history.get('val_costs', []))
    
    # === NEW: Enhanced series ===
    learning_rates = list(history.get('learning_rates', []))
    temperatures = list(history.get('temperatures', []))
    
    # Create DataFrame with enhanced columns - START FROM EPOCH 0
    rows = []
    val_idx = 0
    
    # Add epochs 0 to num_epochs (training data maps directly now)
    for epoch in range(num_epochs + 1):
        row = {
            'epoch': epoch,  # 0-indexed epochs as requested
            'train_loss': train_losses[epoch] if epoch < len(train_losses) else float('nan'),
            'train_cost': train_costs[epoch] if epoch < len(train_costs) else float('nan'),
            # === NEW: Enhanced columns ===
            'learning_rate': learning_rates[epoch] if epoch < len(learning_rates) else float('nan'),
            'temperature': temperatures[epoch] if epoch < len(temperatures) else float('nan'),
        }
        
        # Validation cost mapping (validation happens at specific epochs)
        if (epoch % val_freq) == 0 or epoch == num_epochs:
            if val_idx < len(val_costs_seq):
                row['val_cost'] = val_costs_seq[val_idx]
                val_idx += 1
            else:
                row['val_cost'] = float('nan')
        else:
            row['val_cost'] = float('nan')
            
        rows.append(row)
    
    df = pd.DataFrame(rows)
    out = os.path.join(csv_dir, f'history_{model_key(model_name)}.csv')
    df.to_csv(out, index=False)
    logger.info(f'🧾 Saved enhanced history CSV: {out}')


def save_models_and_analysis(results, training_times, models, config, base_dir, logger):
    """Your original save function with enhanced metadata"""
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    analysis_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(pytorch_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    for model_name, model in models.items():
        # === NEW: Enhanced model info ===
        model_info = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': config,
            'results': results[model_name]['history'],
            'training_time': training_times[model_name],
            # NEW: Additional metadata
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        
        # Add artifacts if available
        if 'artifacts' in results[model_name]:
            model_info.update(results[model_name]['artifacts'])
            
        path = os.path.join(pytorch_dir, f'model_{model_key(model_name)}.pt')
        torch.save(model_info, path)
        logger.info(f'💾 Saved model checkpoint: {path}')

    # Enhanced analysis
    torch.save({
        'results': {k: v['history'] for k, v in results.items()},
        'training_times': training_times,
        'config': config,
        # NEW: Enhanced summary
        'training_summary': {
            'total_training_time': sum(training_times.values()),
            'models_trained': list(models.keys()),
            'best_model': min(results.keys(), key=lambda k: results[k]['history']['final_val_cost']),
        }
    }, os.path.join(analysis_dir, 'comparative_study_complete.pt'))
    logger.info(f'📦 Saved comparative analysis: {os.path.join(analysis_dir, "comparative_study_complete.pt")}')


# === NEW: Enhanced data generator ===
def create_simple_data_generator(config):
    """Simple data generator with optional augmentation"""
    def data_generator(batch_size: int, epoch: int = 1, seed: int = None) -> list:
        n_customers = config['problem']['num_customers']
        capacity = config['problem']['vehicle_capacity']
        coord_range = config['problem']['coord_range']
        demand_range = config['problem']['demand_range']
        
        instances = []
        for i in range(batch_size):
            instance_seed = (seed + i) if seed is not None else (epoch * 1000 + i)
            instance = generate_cvrp_instance(n_customers, capacity, coord_range, demand_range, seed=instance_seed)
            instances.append(instance)
        return instances
    return data_generator


def parse_args():
    p = argparse.ArgumentParser(description='Enhanced CVRP training (automatically uses enhanced features)')
    p.add_argument('--config', type=str, default='configs/small.yaml', help='Path to YAML configuration file')
    p.add_argument('--disable-enhanced', action='store_true', help='Disable enhanced training features (use original training)')
    p.add_argument('--force-retrain', action='store_true', help='Force retraining even if outputs already exist')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging(cfg)

    # Seeds and CPU-only environment
    set_seeds(cfg.get('experiment', {}).get('random_seed', 42))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    base_dir = str(Path(cfg.get('working_dir_path', 'results')).as_posix())
    os.makedirs(base_dir, exist_ok=True)

    # Build models on demand (your original function)
    input_dim = cfg['model']['input_dim']
    hidden_dim = cfg['model']['hidden_dim']
    num_heads = cfg['model']['num_heads']
    num_layers = cfg['model']['num_layers']
    dropout = cfg['model']['transformer_dropout']
    ff_mult = cfg['model']['feedforward_multiplier']
    edge_div = cfg['model']['edge_embedding_divisor']

    def build_model(name: str):
        if name == 'Pointer+RL':
            return BaselinePointerNetwork(input_dim, hidden_dim, cfg)
        if name == 'GT-Greedy':
            return GraphTransformerGreedy(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, cfg)
        if name == 'GT+RL':
            return GraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, cfg)
        if name == 'DGT+RL':
            return DynamicGraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, cfg)
        if name == 'GAT+RL':
            return GraphAttentionTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout, edge_div, cfg)
        raise ValueError(f'Unknown model name: {name}')

    # Determine if we should use enhanced features
    use_enhanced_features = (
        cfg.get('experiment', {}).get('use_advanced_features', False) 
        and not args.disable_enhanced
    )
    
    logger.info('🚀 Training models...')
    if use_enhanced_features:
        logger.info('✅ Using enhanced training features')
    else:
        logger.info('📋 Using original training features')
    
    # Prepare output dirs
    csv_dir = os.path.join(base_dir, 'csv')
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(pytorch_dir, exist_ok=True)

    # Train models
    results = {}
    training_times = {}
    models = {}
    
    # === NEW: Data generator ===
    data_generator = create_simple_data_generator(cfg)
    
    for name in ['Pointer+RL', 'GT-Greedy', 'GT+RL', 'DGT+RL', 'GAT+RL']:
        m = build_model(name)
        
        # === CHOICE: Use enhanced or original trainer ===
        if use_enhanced_features:
            # NEW: Enhanced training with all features
            hist, ttime, artifacts = advanced_train_model(
                m, name, cfg, data_generator, logger.info, use_advanced_features=True
            )
            results[name] = {'history': hist, 'artifacts': artifacts}
        else:
            # ORIGINAL: Your existing training
            hist, ttime, _ = train_one_model(m, name, cfg, logger.info)
            results[name] = {'history': hist}
            
        training_times[name] = ttime
        models[name] = m
        
        # Write enhanced CSV
        write_history_csv(name, results[name]['history'], cfg, base_dir, logger)

    # Save artifacts
    save_models_and_analysis(results, training_times, models, cfg, base_dir, logger)

    # === ENHANCED: Better summary ===
    logger.info('\n📊 SUMMARY')
    sorted_models = sorted(results.keys(), key=lambda x: results[x]['history']['final_val_cost'])
    
    for model_name in sorted_models:
        params = sum(p.numel() for p in models[model_name].parameters())
        final_val = results[model_name]['history']['final_val_cost']
        
        # Show enhanced info if available
        if 'artifacts' in results[model_name]:
            best_val = results[model_name]['artifacts'].get('best_val_cost', final_val)
            if best_val != final_val:
                logger.info(f'- {model_name}: params={params:,}, time={training_times[model_name]:.1f}s, '
                           f'final_val={final_val:.3f}/cust, best_val={best_val:.3f}/cust')
            else:
                logger.info(f'- {model_name}: params={params:,}, time={training_times[model_name]:.1f}s, final_val={final_val:.3f}/cust')
        else:
            logger.info(f'- {model_name}: params={params:,}, time={training_times[model_name]:.1f}s, final_val={final_val:.3f}/cust')


if __name__ == '__main__':
    main()
