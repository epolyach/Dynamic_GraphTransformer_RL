#!/usr/bin/env python3
"""
CVRP Training Script - Train models individually or in groups.

Supports:
- GAT+RL: Legacy baseline with edge-aware GAT
- GT+RL: Advanced Graph Transformer 
- DGT+RL: Dynamic Graph Transformer (state-of-the-art)
- GT-Greedy: Greedy baseline for comparison
"""

import os
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
from src.training.advanced_trainer import advanced_train_model
from src.data.enhanced_generator import create_enhanced_data_generator
from src.models.model_factory import ModelFactory
from src.pipelines.train import set_seeds


def setup_logging(config=None):
    """Setup logging configuration."""
    level = logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if config and 'logging' in config:
        level_str = config['logging'].get('level', 'INFO')
        format_str = config['logging'].get('format', format_str)
        level = getattr(logging, level_str.upper(), logging.INFO)
    
    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger(__name__)


def model_key(name: str) -> str:
    """Convert model name to file-safe key."""
    return name.lower().replace('+', '_').replace('-', '_').replace(' ', '_')


def write_history_csv(model_name: str, history: dict, config: dict, base_dir: str, logger) -> None:
    """Write training history to CSV file."""
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    num_epochs = len(history.get('train_costs', []))
    val_freq = int(config['training']['validation_frequency'])
    
    # Extract metrics
    train_losses = history.get('train_losses', [])
    train_costs = history.get('train_costs', [])
    val_costs = history.get('val_costs', [])
    learning_rates = history.get('learning_rates', [])
    temperatures = history.get('temperatures', [])
    
    # Build rows for DataFrame
    rows = []
    val_idx = 0
    
    for epoch in range(num_epochs):
        row = {
            'epoch': epoch + 1,
            'train_loss': train_losses[epoch] if epoch < len(train_losses) else float('nan'),
            'train_cost': train_costs[epoch] if epoch < len(train_costs) else float('nan'),
            'learning_rate': learning_rates[epoch] if epoch < len(learning_rates) else float('nan'),
            'temperature': temperatures[epoch] if epoch < len(temperatures) else float('nan'),
        }
        
        # Add validation cost if available
        if (epoch + 1) % val_freq == 0 or epoch == num_epochs - 1:
            if val_idx < len(val_costs):
                row['val_cost'] = val_costs[val_idx]
                val_idx += 1
            else:
                row['val_cost'] = float('nan')
        else:
            row['val_cost'] = float('nan')
        
        rows.append(row)
    
    # Save to CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(csv_dir, f'history_{model_key(model_name)}.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f'📊 Saved training history: {csv_path}')


def save_model(model_name: str, model: torch.nn.Module, results: dict, 
               training_time: float, config: dict, base_dir: str, logger) -> None:
    """Save trained model with metadata."""
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    os.makedirs(pytorch_dir, exist_ok=True)
    
    # Calculate model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Prepare model data
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'config': config,
        'history': results['history'],
        'artifacts': results.get('artifacts', {}),
        'training_time': training_time,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_architecture': str(model.__class__.__name__),
    }
    
    # Save model
    model_path = os.path.join(pytorch_dir, f'model_{model_key(model_name)}.pt')
    torch.save(model_data, model_path)
    logger.info(f'💾 Saved model: {model_path}')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CVRP models individually or in groups',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all three main models
  python3 run_training.py --models GAT+RL GT+RL DGT+RL
  
  # Train only the advanced GT model
  python3 run_training.py --models GT+RL
  
  # Train legacy baseline and best model
  python3 run_training.py --models GAT+RL DGT+RL
  
  # Train with specific config
  python3 run_training.py --config configs/small.yaml --models GT+RL
  
  # Force retrain existing models
  python3 run_training.py --models DGT+RL --force-retrain
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models', nargs='+', 
                       choices=['GAT+RL', 'GT+RL', 'DGT+RL', 'GT-Greedy', 'all'],
                       default=['GT+RL'],
                       help='Models to train (default: GT+RL)')
    parser.add_argument('--use-curriculum', action='store_true',
                       help='Use curriculum learning (gradually increase difficulty)')
    parser.add_argument('--use-augmentation', action='store_true', default=True,
                       help='Use data augmentation (default: True)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if saved models exist')
    parser.add_argument('--disable-advanced', action='store_true',
                       help='Disable advanced training features (for testing)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory from config')
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if 'all' in args.models:
        args.models = ['GAT+RL', 'GT+RL', 'DGT+RL', 'GT-Greedy']
    
    return args


def check_existing_model(model_name: str, base_dir: str) -> bool:
    """Check if a model has already been trained."""
    model_path = os.path.join(base_dir, 'pytorch', f'model_{model_key(model_name)}.pt')
    return os.path.exists(model_path)


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config)
    
    # Print training configuration
    logger.info("="*60)
    logger.info("CVRP MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"📋 Configuration: {args.config}")
    logger.info(f"🏗️  Models to train: {', '.join(args.models)}")
    logger.info(f"🎓 Curriculum learning: {'✅ Enabled' if args.use_curriculum else '❌ Disabled'}")
    logger.info(f"🎨 Data augmentation: {'✅ Enabled' if args.use_augmentation else '❌ Disabled'}")
    logger.info(f"🔧 Advanced features: {'❌ Disabled' if args.disable_advanced else '✅ Enabled'}")
    
    # Setup environment
    set_seeds(config.get('experiment', {}).get('random_seed', 42))
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"💻 Using device: {device}")
    
    # Setup output directory
    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = str(Path(config.get('working_dir_path', 'results/training')).as_posix())
    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"📁 Output directory: {base_dir}")
    
    # Check for existing models
    if not args.force_retrain:
        models_to_skip = []
        for model_name in args.models:
            if check_existing_model(model_name, base_dir):
                models_to_skip.append(model_name)
        
        if models_to_skip:
            logger.info(f"⚠️  Found existing models: {', '.join(models_to_skip)}")
            logger.info("   Use --force-retrain to retrain them")
            
            # Remove from training list
            args.models = [m for m in args.models if m not in models_to_skip]
            
            if not args.models:
                logger.info("✅ All requested models already trained!")
                return
    
    # Create data generator
    logger.info("\n📊 Setting up data generator...")
    data_generator = create_enhanced_data_generator(
        config, 
        use_curriculum=args.use_curriculum
    )
    
    # Training settings
    use_advanced = not args.disable_advanced and config.get('experiment', {}).get('use_advanced_features', True)
    
    # Store results
    all_results = {}
    training_times = {}
    
    # Train each model
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    for i, model_name in enumerate(args.models, 1):
        logger.info(f"\n[{i}/{len(args.models)}] Training {model_name}")
        logger.info("-"*40)
        
        try:
            # Create model
            model = ModelFactory.create_model(model_name, config)
            model = model.to(device)
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"📐 Model architecture: {model.__class__.__name__}")
            logger.info(f"📊 Total parameters: {total_params:,}")
            
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
            results = {
                'history': history,
                'artifacts': artifacts
            }
            all_results[model_name] = results
            training_times[model_name] = training_time
            
            # Save outputs
            write_history_csv(model_name, history, config, base_dir, logger)
            save_model(model_name, model, results, training_time, config, base_dir, logger)
            
            # Log summary
            final_val = history.get('final_val_cost', float('inf'))
            best_val = artifacts.get('best_val_cost', final_val)
            convergence_epoch = artifacts.get('convergence_epoch', 'N/A')
            
            logger.info(f"\n✅ {model_name} Training Complete:")
            logger.info(f"   ⏱️  Training time: {training_time:.1f}s")
            logger.info(f"   📊 Final validation cost: {final_val:.4f}")
            logger.info(f"   🎯 Best validation cost: {best_val:.4f}")
            logger.info(f"   🔄 Convergence epoch: {convergence_epoch}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    if all_results:
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        
        # Sort by performance
        sorted_models = sorted(all_results.keys(), 
                             key=lambda x: all_results[x]['history']['final_val_cost'])
        
        logger.info("\n📊 Model Performance Ranking:")
        for rank, model_name in enumerate(sorted_models, 1):
            history = all_results[model_name]['history']
            artifacts = all_results[model_name].get('artifacts', {})
            
            final_val = history['final_val_cost']
            best_val = artifacts.get('best_val_cost', final_val)
            time_taken = training_times[model_name]
            
            logger.info(f"\n  #{rank} {model_name}:")
            logger.info(f"      Final cost: {final_val:.4f}/customer")
            logger.info(f"      Best cost:  {best_val:.4f}/customer")
            logger.info(f"      Time:       {time_taken:.1f}s")
            
            if best_val < final_val:
                improvement = ((final_val - best_val) / final_val) * 100
                logger.info(f"      Early stop: {improvement:.1f}% better than final")
        
        # Total time
        total_time = sum(training_times.values())
        logger.info(f"\n⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        # Best model
        best_model = sorted_models[0]
        best_cost = all_results[best_model]['history']['final_val_cost']
        logger.info(f"\n🏆 Best model: {best_model} (cost: {best_cost:.4f}/customer)")
        
        # Compare to baseline if available
        if 'GAT+RL' in all_results and best_model != 'GAT+RL':
            baseline_cost = all_results['GAT+RL']['history']['final_val_cost']
            improvement = ((baseline_cost - best_cost) / baseline_cost) * 100
            logger.info(f"📈 Improvement over GAT+RL baseline: {improvement:.2f}%")
        
        logger.info("\n🎉 Training completed successfully!")
    else:
        logger.error("❌ No models were successfully trained")


if __name__ == '__main__':
    main()
