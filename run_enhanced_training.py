#!/usr/bin/env python3
"""
Enhanced CVRP training script with advanced features:
- Advanced training loop with learning rate scheduling and early stopping
- Enhanced model architectures 
- Diverse data generation with augmentation
- Comprehensive metrics tracking
"""

import os
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
from src.training.advanced_trainer import advanced_train_model
from src.data.enhanced_generator import create_enhanced_data_generator
from src.models.enhanced_dgt import EnhancedDynamicGraphTransformer
from src.models.dgt import DynamicGraphTransformerNetwork as DynamicGraphTransformer
from src.models.pointer import BaselinePointerNetwork
from src.models.gt import GraphTransformerNetwork
from src.models.greedy_gt import GraphTransformerGreedy
from src.models.gat import GraphAttentionTransformer
from src.pipelines.train import set_seeds


def setup_logging(config=None):
    level = logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if config and 'logging' in config:
        level_str = config['logging'].get('level', 'INFO')
        format_str = config['logging'].get('format', format_str)
        level = getattr(logging, level_str.upper(), logging.INFO)
    
    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger(__name__)


def model_key(name: str) -> str:
    mapping = {
        'Pointer+RL': 'pointer_rl',
        'GT+RL': 'gt_rl',
        'DGT+RL': 'dgt_rl',
        'Enhanced-DGT+RL': 'enhanced_dgt_rl',
        'GAT+RL': 'gat_rl',
        'GT-Greedy': 'gt_greedy',
    }
    return mapping.get(name, name.lower().replace(' ', '_').replace('+', '_').replace('-', '_'))


def write_enhanced_history_csv(model_name: str, history: dict, config: dict, base_dir: str, logger) -> None:
    """Write enhanced history CSV with additional metrics."""
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    num_epochs = len(history.get('train_costs', []))
    val_freq = int(config['training']['validation_frequency'])
    
    # Basic series
    train_losses = history.get('train_losses', [])
    train_costs = history.get('train_costs', [])
    val_costs = history.get('val_costs', [])
    
    # Enhanced series
    learning_rates = history.get('learning_rates', [])
    temperatures = history.get('temperatures', [])
    
    # Create comprehensive DataFrame
    rows = []
    val_idx = 0
    
    for epoch in range(num_epochs):
        row = {
            'epoch': epoch + 1,  # 1-indexed for readability
            'train_loss': train_losses[epoch] if epoch < len(train_losses) else float('nan'),
            'train_cost': train_costs[epoch] if epoch < len(train_costs) else float('nan'),
            'learning_rate': learning_rates[epoch] if epoch < len(learning_rates) else float('nan'),
            'temperature': temperatures[epoch] if epoch < len(temperatures) else float('nan'),
        }
        
        # Add validation cost if this epoch had validation
        if (epoch + 1) % val_freq == 0 or epoch == num_epochs - 1:
            if val_idx < len(val_costs):
                row['val_cost'] = val_costs[val_idx]
                val_idx += 1
            else:
                row['val_cost'] = float('nan')
        else:
            row['val_cost'] = float('nan')
        
        rows.append(row)
    
    # Add epoch metrics if available
    epoch_metrics = history.get('epoch_metrics', {})
    for metric_name, values in epoch_metrics.items():
        metric_col = f'metric_{metric_name}'
        for i, row in enumerate(rows):
            if i < len(values):
                row[metric_col] = values[i]
            else:
                row[metric_col] = float('nan')
    
    df = pd.DataFrame(rows)
    out = os.path.join(csv_dir, f'history_{model_key(model_name)}.csv')
    df.to_csv(out, index=False)
    logger.info(f'📊 Saved enhanced history CSV: {out}')


def save_enhanced_models_and_analysis(results, training_times, models, config, base_dir, logger):
    """Save models and analysis with enhanced metadata."""
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    analysis_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(pytorch_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    enhanced_results = {}
    
    for model_name, model in models.items():
        # Calculate detailed model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Save individual model
        model_info = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': config,
            'results': results[model_name],
            'training_time': training_times[model_name],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_architecture': str(model.__class__.__name__),
        }
        
        # Add model-specific artifacts if available
        if 'artifacts' in results[model_name]:
            model_info.update(results[model_name]['artifacts'])
        
        path = os.path.join(pytorch_dir, f'model_{model_key(model_name)}.pt')
        torch.save(model_info, path)
        logger.info(f'💾 Saved enhanced model: {path}')
        
        # Prepare for analysis
        enhanced_results[model_name] = {
            'history': results[model_name]['history'],
            'training_time': training_times[model_name],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'artifacts': results[model_name].get('artifacts', {}),
        }
    
    # Save comprehensive analysis
    analysis_data = {
        'results': enhanced_results,
        'config': config,
        'training_summary': {
            'total_training_time': sum(training_times.values()),
            'models_trained': list(models.keys()),
            'best_model': min(enhanced_results.keys(), 
                            key=lambda k: enhanced_results[k]['history']['final_val_cost']),
            'config_hash': hash(str(sorted(config.items()))),
        }
    }
    
    analysis_path = os.path.join(analysis_dir, 'enhanced_comparative_study.pt')
    torch.save(analysis_data, analysis_path)
    logger.info(f'📦 Saved enhanced analysis: {analysis_path}')


def build_enhanced_model(name: str, config: dict):
    """Build enhanced models with improved architectures."""
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    num_heads = config['model']['num_heads']
    num_layers = config['model']['num_layers']
    dropout = config['model']['transformer_dropout']
    ff_mult = config['model']['feedforward_multiplier']
    edge_div = config['model']['edge_embedding_divisor']

    if name == 'Enhanced-DGT+RL':
        return EnhancedDynamicGraphTransformer(
            input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config
        )
    elif name == 'DGT+RL':
        return DynamicGraphTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'Pointer+RL':
        return BaselinePointerNetwork(input_dim, hidden_dim, config)
    elif name == 'GT-Greedy':
        return GraphTransformerGreedy(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'GT+RL':
        return GraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'GAT+RL':
        return GraphAttentionTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout, edge_div, config)
    else:
        raise ValueError(f'Unknown model name: {name}')


def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced CVRP training with advanced features')
    parser.add_argument('--config', type=str, default='configs/enhanced.yaml',
                       help='Path to enhanced configuration file')
    parser.add_argument('--models', nargs='+', 
                       choices=['Enhanced-DGT+RL', 'DGT+RL', 'Pointer+RL', 'GT-Greedy', 'GT+RL', 'GAT+RL'],
                       default=['Enhanced-DGT+RL', 'Pointer+RL', 'GT+RL'],
                       help='Models to train')
    parser.add_argument('--use-curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--use-augmentation', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if outputs exist')
    parser.add_argument('--disable-advanced', action='store_true',
                       help='Disable advanced training features for comparison')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config)
    
    logger.info("🚀 Starting Enhanced CVRP Training")
    logger.info(f"📋 Configuration: {args.config}")
    logger.info(f"🏗️  Models to train: {args.models}")
    logger.info(f"🎓 Curriculum learning: {'✅' if args.use_curriculum else '❌'}")
    logger.info(f"🎨 Data augmentation: {'✅' if args.use_augmentation else '❌'}")
    
    # Setup environment
    set_seeds(config.get('experiment', {}).get('random_seed', 42))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    
    # Setup directories
    base_dir = str(Path(config.get('working_dir_path', 'results/enhanced')).as_posix())
    os.makedirs(base_dir, exist_ok=True)
    
    # Create enhanced data generator
    data_generator = create_enhanced_data_generator(
        config, 
        use_curriculum=args.use_curriculum
    )
    
    # Training results
    results = {}
    training_times = {}
    models = {}
    
    # Use advanced features unless disabled
    use_advanced = not args.disable_advanced and config.get('experiment', {}).get('use_advanced_features', True)
    
    logger.info(f"🔧 Advanced features: {'✅' if use_advanced else '❌'}")
    
    # Train each model
    for model_name in args.models:
        logger.info(f"\n🏋️  Training {model_name}...")
        
        try:
            # Build model
            model = build_enhanced_model(model_name, config)
            logger.info(f"📐 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train model
            start_time = time.time()
            history, training_time, artifacts = advanced_train_model(
                model=model,
                model_name=model_name,
                config=config,
                data_generator=data_generator,
                logger_print=logger.info,
                use_advanced_features=use_advanced
            )
            
            # Store results
            results[model_name] = {
                'history': history,
                'artifacts': artifacts
            }
            training_times[model_name] = training_time
            models[model_name] = model
            
            # Save CSV immediately
            write_enhanced_history_csv(model_name, history, config, base_dir, logger)
            
            # Log summary
            final_val = history.get('final_val_cost', float('inf'))
            best_val = artifacts.get('best_val_cost', final_val)
            convergence_epoch = artifacts.get('convergence_epoch', 'N/A')
            
            logger.info(f"✅ {model_name} completed:")
            logger.info(f"   ⏱️  Training time: {training_time:.1f}s")
            logger.info(f"   📊 Final validation cost: {final_val:.4f}")
            logger.info(f"   🎯 Best validation cost: {best_val:.4f}")
            logger.info(f"   🔄 Convergence epoch: {convergence_epoch}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    if results:
        save_enhanced_models_and_analysis(results, training_times, models, config, base_dir, logger)
        
        # Final summary
        logger.info("\n📊 ENHANCED TRAINING SUMMARY")
        logger.info("=" * 80)
        
        # Sort by performance
        sorted_models = sorted(results.keys(), 
                             key=lambda x: results[x]['history']['final_val_cost'])
        
        for i, model_name in enumerate(sorted_models, 1):
            data = results[model_name]
            history = data['history']
            artifacts = data.get('artifacts', {})
            
            params = artifacts.get('total_parameters', sum(p.numel() for p in models[model_name].parameters()))
            final_val = history['final_val_cost']
            best_val = artifacts.get('best_val_cost', final_val)
            time_taken = training_times[model_name]
            
            logger.info(f"#{i} {model_name}:")
            logger.info(f"    📊 Final cost: {final_val:.4f}/customer")
            logger.info(f"    🎯 Best cost: {best_val:.4f}/customer") 
            logger.info(f"    📐 Parameters: {params:,}")
            logger.info(f"    ⏱️  Time: {time_taken:.1f}s")
            
            if best_val != final_val:
                improvement = ((final_val - best_val) / best_val) * 100
                logger.info(f"    📈 Early stopping saved: {improvement:.1f}%")
                
        total_time = sum(training_times.values())
        logger.info(f"\n⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"🎯 Best overall model: {sorted_models[0]} ({results[sorted_models[0]]['history']['final_val_cost']:.4f})")
        
        logger.info("\n🎉 Enhanced training completed successfully!")
    else:
        logger.error("❌ No models were successfully trained")


if __name__ == '__main__':
    main()
