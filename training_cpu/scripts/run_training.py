#!/usr/bin/env python3
"""
CVRP Model Training Script

Supported models:
- GAT+RL: Graph Attention Network with RL
- GT+RL: Graph Transformer with RL
- DGT+RL: Dynamic Graph Transformer with RL
- GT-Greedy: Greedy baseline
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
# Import from training_cpu.lib since we're now in scripts/ subdirectory
training_cpu_path = Path(__file__).parent.parent
sys.path.insert(0, str(training_cpu_path))
from lib import advanced_train_model
from src.generator.generator import create_data_generator
from src.models.model_factory import ModelFactory
from src.utils.seeding import set_seeds


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


class IncrementalCSVWriter:
    """Write training metrics to CSV incrementally during training."""
    
    def __init__(self, model_name: str, base_dir: str, config: dict, logger):
        self.csv_dir = os.path.join(base_dir, 'csv')
        os.makedirs(self.csv_dir, exist_ok=True)
        self.csv_path = os.path.join(self.csv_dir, f'history_{model_key(model_name)}.csv')
        self.val_freq = int(config['training']['validation_frequency'])
        self.logger = logger
        self.rows = []
        self.val_idx = 0
        self.baseline_type = config.get('baseline', {}).get('type', 'mean')
        
        # Initialize file with headers
        headers = ['epoch', 'train_loss', 'train_cost', 'val_cost', 'learning_rate', 
                   'temperature', 'baseline_type', 'baseline_value']
        pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False)
        self.logger.info(f'Created CSV history file: {self.csv_path}')
    
    def write_epoch(self, epoch: int, train_loss: float, train_cost: float, 
                    val_cost: Optional[float], learning_rate: float, 
                    temperature: float, baseline_value: float = None):
        """Write data for a single epoch."""
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_cost': train_cost,
            'val_cost': val_cost if val_cost is not None else float('nan'),
            'learning_rate': learning_rate,
            'temperature': temperature,
            'baseline_type': self.baseline_type,
            'baseline_value': baseline_value if baseline_value is not None else float('nan')
        }
        
        self.rows.append(row)
        
        # Append to CSV file
        pd.DataFrame([row]).to_csv(self.csv_path, mode='a', header=False, index=False)
    
    def finalize(self):
        """Optional: perform any final operations."""
        self.logger.info(f'Training history saved to: {self.csv_path}')


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
    logger.info(f'Saved model: {model_path}')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CVRP models')
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str,
                       choices=['GAT+RL', 'GT+RL', 'DGT+RL', 'GT-Greedy'],
                       default='GT+RL',
                       help='Model to train (default: GT+RL)')
    parser.add_argument('--all', action='store_true',
                       help='Train all available models sequentially')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if saved model exists')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory from config')
    
    return parser.parse_args()


def check_existing_model(model_name: str, base_dir: str) -> bool:
    """Check if a model has already been trained."""
    model_path = os.path.join(base_dir, 'pytorch', f'model_{model_key(model_name)}.pt')
    return os.path.exists(model_path)


def apply_gat_specific_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Apply GAT-specific parameters if training GAT model."""
    if 'GAT' not in model_name.upper():
        return config
    
    # Check if GAT-specific parameters exist in config
    if 'gat_training' not in config:
        return config
    
    # Create a deep copy to avoid modifying original
    import copy
    gat_config = copy.deepcopy(config)
    gat_params = config['gat_training']
    
    # Override training parameters for GAT
    if 'learning_rate' in gat_params:
        gat_config['training']['learning_rate'] = gat_params['learning_rate']
    if 'batch_size' in gat_params:
        gat_config['training']['batch_size'] = gat_params['batch_size']
    
    # Override advanced training parameters for GAT
    if 'training_advanced' not in gat_config:
        gat_config['training_advanced'] = {}
    
    gat_advanced = gat_config['training_advanced']
    
    if 'temp_start' in gat_params:
        gat_advanced['temp_start'] = gat_params['temp_start']
    if 'temp_min' in gat_params:
        gat_advanced['temp_min'] = gat_params['temp_min']
    if 'temp_adaptation_rate' in gat_params:
        gat_advanced['temp_adaptation_rate'] = gat_params['temp_adaptation_rate']
    if 'entropy_coef' in gat_params:
        gat_advanced['entropy_coef'] = gat_params['entropy_coef']
    if 'entropy_min' in gat_params:
        gat_advanced['entropy_min'] = gat_params['entropy_min']
    if 'gradient_clip_norm' in gat_params:
        gat_advanced['gradient_clip_norm'] = gat_params['gradient_clip_norm']
    if 'early_stopping_patience' in gat_params:
        gat_advanced['early_stopping_patience'] = gat_params['early_stopping_patience']
    
    return gat_config


def main():
    """Main training function."""
    args = parse_args()
    
    # Save current directory and change to project root for config loading
    original_cwd = os.getcwd()
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # Convert config path to absolute before changing directory
    config_path = Path(original_cwd) / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(str(config_path))
    
    # Change back to original directory
    os.chdir(original_cwd)
    
    logger = setup_logging(config)
    
    # Print training configuration
    logger.info("="*60)
    logger.info("CVRP MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config}")
    # Determine which models to train
    if args.all:
        models_to_train = ['GAT+RL', 'GT+RL', 'DGT+RL', 'GT-Greedy']
    else:
        models_to_train = [args.model]
    logger.info(f"Models to train: {', '.join(models_to_train)}")
    
    # Setup environment
    set_seeds(config.get('experiment', {}).get('random_seed', 42))
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Setup output directory
    if args.output_dir:
        base_dir = args.output_dir
    elif 'working_dir_path' in config:
        # Use working_dir_path from config
        working_dir = Path(config['working_dir_path'])
        if not working_dir.is_absolute():
            # If relative, it's relative to project root
            base_dir = project_root / working_dir
        else:
            base_dir = working_dir
        base_dir = str(base_dir)
    else:
        # Fallback to local results directory within training_cpu
        base_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"Output directory: {base_dir}")
    
    # Check for existing model
    # We'll check for existing models inside the training loop
    
    # Create data generator
    logger.info("\nSetting up data generator...")
    data_generator = create_data_generator(config)
    
    # Training settings
    use_advanced = config.get('experiment', {}).get('use_advanced_features', True)
    
    # Store results
    all_results = {}
    training_times = {}
    
    # Train model
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    # Train each model
    for model_name in models_to_train:
        # Check for existing model
        if not args.force_retrain and check_existing_model(model_name, base_dir):
            logger.info(f"Model {model_name} already exists. Use --force-retrain to retrain.")
            continue
        
        logger.info("-"*40)
        logger.info(f"Training {model_name}")
        
        try:
            # Apply GAT-specific configuration if needed
            model_config = apply_gat_specific_config(config, model_name)
            
            # Log if GAT-specific parameters are being used
            if 'GAT' in model_name.upper() and 'gat_training' in config:
                logger.info(f"Applying GAT-specific parameters:")
                gat_params = config['gat_training']
                for key, value in gat_params.items():
                    logger.info(f"  {key}: {value}")
            
            # Create model
            model = ModelFactory.create_model(model_name, model_config)
            model = model.to(device)
        
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model architecture: {model.__class__.__name__}")
            logger.info(f"Total parameters: {total_params:,}")
        
            # Create CSV writer for incremental logging
            csv_writer = IncrementalCSVWriter(model_name, base_dir, model_config, logger)
        
            # Train model with CSV callback
            start_time = time.time()
            history, training_time, artifacts = advanced_train_model(
                model=model,
                model_name=model_name,
                config=model_config,
                data_generator=data_generator,
                logger_print=logger.info,
                use_advanced_features=use_advanced,
                epoch_callback=csv_writer.write_epoch
            )
        
            # Finalize CSV
            csv_writer.finalize()
        
            # Store results
            results = {
                'history': history,
                'artifacts': artifacts
            }
        
            # Save model
            save_model(model_name, model, results, training_time, config, base_dir, logger)
        
            # Log summary
            final_val = history.get('final_val_cost', float('inf'))
            best_val = artifacts.get('best_val_cost', final_val)
            convergence_epoch = artifacts.get('convergence_epoch', 'N/A')
            baseline_type = config.get('baseline', {}).get('type', 'mean')
        
            logger.info(f"\nTraining Complete:")
            logger.info(f"  Training time: {training_time:.1f}s")
            logger.info(f"  Final validation cost: {final_val:.4f}")
            logger.info(f"  Best validation cost: {best_val:.4f}")
            logger.info(f"  Convergence epoch: {convergence_epoch}")
            logger.info(f"  Baseline type: {baseline_type}")
        
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
    


if __name__ == '__main__':
    main()
