#!/usr/bin/env python3
"""
GPU-Optimized CVRP Model Training Script

Supported models:
- GAT+RL: Graph Attention Network with RL (GPU-optimized)
- GT+RL: Graph Transformer with RL (GPU-optimized)
- DGT+RL: Dynamic Graph Transformer with RL (GPU-optimized)
- GT-Greedy: Greedy baseline (GPU-optimized)

Key GPU optimizations:
- Mixed precision training (FP16)
- Non-blocking tensor transfers
- Pinned memory for faster data loading
- Gradient accumulation support
- Memory management and monitoring
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import yaml

# Add project root to path to find src module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.cuda as cuda

from src.utils.config import load_config
from src.generator.generator import create_data_generator
from src.models.model_factory import ModelFactory
from src.utils.seeding import set_seeds

# Import GPU training components
training_gpu_path = Path(__file__).parent.parent
sys.path.insert(0, str(training_gpu_path))
from lib import GPUManager, advanced_train_model_gpu
from lib.gpu_utils import estimate_batch_size, profile_memory_usage


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
    """Normalize model name for factory."""
    # Handle both formats: with and without +RL suffix
    name_map = {
        'gat': 'gat',
        'gt': 'gt',
        'dgt': 'dgt',
        'greedy': 'greedy_gt',
        'gat-rl': 'gat',
        'gt-rl': 'gt',
        'dgt-rl': 'dgt',
        'gt-greedy': 'greedy_gt',
        'gat+rl': 'gat',
        'gt+rl': 'gt',
        'dgt+rl': 'dgt'
    }
    return name_map.get(name.lower(), name.lower())


def print_gpu_info():
    """Print GPU information and availability."""
    print("\n" + "="*60)
    print("GPU INFORMATION")
    print("="*60)
    
    if not cuda.is_available():
        print("CUDA is not available. Training will run on CPU.")
        return False
    
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {cuda.device_count()}")
    
    for i in range(cuda.device_count()):
        props = cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - Multiprocessors: {props.multi_processor_count}")
        print(f"  - CUDA Cores: ~{props.multi_processor_count * 64}")  # Approximate
        
        # Current memory usage
        if cuda.is_initialized():
            allocated = cuda.memory_allocated(i) / 1024**3
            reserved = cuda.memory_reserved(i) / 1024**3
            print(f"  - Memory Used: {allocated:.2f}/{reserved:.2f} GB (allocated/reserved)")
    
    print("="*60 + "\n")
    return True


def validate_gpu_config(config: Dict[str, Any]) -> bool:
    """Validate GPU configuration and availability."""
    gpu_config = config.get('gpu', {})
    
    if not gpu_config.get('enabled', True):
        print("GPU training is disabled in configuration.")
        return False
    
    if not cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU training.")
        return False
    
    device_str = gpu_config.get('device', 'cuda:0')
    if device_str.startswith('cuda:'):
        device_idx = int(device_str.split(':')[1])
        if device_idx >= cuda.device_count():
            print(f"ERROR: Requested device {device_str} not available. "
                  f"Only {cuda.device_count()} GPU(s) detected.")
            return False
    
    return True


def estimate_optimal_batch_size(model, config, logger):
    """Estimate optimal batch size for GPU memory."""
    logger.info("Estimating optimal batch size for GPU...")
    
    gpu_config = config.get('gpu', {})
    gpu_manager = GPUManager(
        device=gpu_config.get('device', 'cuda:0'),
        memory_fraction=gpu_config.get('memory_fraction', 0.95),
        enable_mixed_precision=gpu_config.get('mixed_precision', True)
    )
    
    # Create sample input
    problem_config = config.get('problem', {})
    n_customers = problem_config.get('num_customers', 20)
    
    sample_input = {
        'depot': torch.randn(1, 2),
        'customers': torch.randn(1, n_customers, 2),
        'demands': torch.randint(1, 10, (1, n_customers)).float(),
        'vehicle_capacity': torch.tensor([30.0])
    }
    
    optimal_batch_size = estimate_batch_size(
        model,
        sample_input,
        gpu_manager.device,
        target_memory_usage=0.8,
        max_batch_size=config['training'].get('batch_size', 512)
    )
    
    logger.info(f"Recommended batch size: {optimal_batch_size}")
    return optimal_batch_size


def create_training_summary(model, history, config, start_time, end_time):
    """Create a training summary report."""
    summary = {
        'model': model.__class__.__name__,
        'config': {
            'problem_size': config['problem']['num_customers'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'epochs': config['training']['num_epochs'],
            'gpu_device': config['gpu'].get('device', 'N/A'),
            'mixed_precision': config['gpu'].get('mixed_precision', False)
        },
        'results': {
            'final_train_cost': history['metrics']['final_train_cost'],
            'final_val_cost': history['metrics']['final_val_cost'],
            'best_val_cost': history['metrics']['best_val_cost'],
            'total_time': end_time - start_time,
            'time_per_epoch': history['metrics']['avg_epoch_time'],
            'peak_gpu_memory_gb': history['metrics'].get('peak_gpu_memory', 0),
            'device': history['metrics']['device']
        },
        'performance': {
            'throughput': config['training']['batch_size'] / history['metrics']['avg_epoch_time'],
            'gpu_efficiency': None
        }
    }
    
    # Calculate GPU efficiency if available
    if 'gpu_memory' in history['history']:
        gpu_mem_usage = history['history']['gpu_memory']
        if gpu_mem_usage:
            props = cuda.get_device_properties(0)
            total_mem = props.total_memory / 1024**3
            avg_usage = np.mean(gpu_mem_usage) / total_mem
            summary['performance']['gpu_efficiency'] = f"{avg_usage:.1%}"
    
    return summary


def check_existing_model(model_name: str, output_dir: Path, force_retrain: bool) -> Optional[Path]:
    """Check if a trained model already exists."""
    model_path = output_dir / f"{model_name}_final.pt"
    if model_path.exists() and not force_retrain:
        return model_path
    return None




def model_key(name: str) -> str:
    """Convert model name to file-safe key."""
    return name.lower().replace('+', '_').replace('-', '_').replace(' ', '_')


class IncrementalCSVWriter:
    """Write training metrics to CSV incrementally during training."""
    
    def __init__(self, model_name: str, base_dir: Path, logger):
        # Create directory structure
        self.base_dir = base_dir
        self.csv_dir = base_dir / 'csv'
        self.pytorch_dir = base_dir / 'pytorch'
        self.analysis_dir = base_dir / 'analysis'
        self.plots_dir = base_dir / 'plots'
        
        # Create all directories
        for dir_path in [self.csv_dir, self.pytorch_dir, self.analysis_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.csv_dir / f'history_{model_key(model_name)}.csv'
        self.logger = logger
        self.rows = []
        
        # Initialize CSV with headers (overwrite any existing file)
        headers = [
            'epoch',
            'train_loss',
            'train_cost_arithmetic',
            'val_cost_arithmetic',
            'learning_rate',
            'temperature',
            'time_per_epoch',
            'baseline_type',
            'baseline_value',
            'mean_type'
        ]
        pd.DataFrame(columns=headers).to_csv(self.csv_path, index=False, mode='w')
        self.logger.info(f'Created CSV history file: {self.csv_path}')
    
    def write_epoch(self, history: dict, epoch: int):
        """Write data for a single epoch from history dict."""
        if epoch >= len(history.get('train_cost_arithmetic', [])):
            return
            
        row = {
            'epoch': epoch,
            'train_loss': history.get('train_loss', [None]* (epoch+1))[epoch],
            'train_cost_arithmetic': history.get('train_cost_arithmetic', [None]* (epoch+1))[epoch],
            'val_cost_arithmetic': history.get('val_cost_arithmetic', [None]* (epoch+1))[epoch] if epoch < len(history.get('val_cost_arithmetic', [])) else None,
            'learning_rate': history.get('learning_rate', [None]* (epoch+1))[epoch],
            'temperature': (history.get('temperature', [2.5]* (epoch+1))[epoch]),
            'time_per_epoch': history.get('epoch_time', [None]* (epoch+1))[epoch],
            'baseline_type': history.get('baseline_type', [None]* (epoch+1))[epoch] if 'baseline_type' in history else None,
            'baseline_value': history.get('baseline_value', [None]* (epoch+1))[epoch] if 'baseline_value' in history else None,
            'mean_type': history.get('mean_type', [None]* (epoch+1))[epoch] if 'mean_type' in history else None
        }
        
        self.rows.append(row)
        pd.DataFrame([row]).to_csv(self.csv_path, mode='a', header=False, index=False)
    
    def save_all_history(self, history: dict):
        """Save complete history at once."""
        import numpy as np
        
        # Convert history values to regular Python types
        def convert_value(val):
            if isinstance(val, (np.ndarray, np.generic)):
                return float(val)
            elif val is None:
                return float('nan')
            return float(val) if not isinstance(val, float) else val
        
        n_epochs = len(history.get('train_cost_arithmetic', []))
        if n_epochs == 0:
            self.logger.warning(f"No history data to save to CSV")
            return
            
        for epoch in range(n_epochs):
            row = {
                'epoch': epoch,
                'train_loss': convert_value(history['train_loss'][epoch]) if 'train_loss' in history and epoch < len(history['train_loss']) else float('nan'),
                'train_cost_arithmetic': convert_value(history['train_cost_arithmetic'][epoch]) if 'train_cost_arithmetic' in history and epoch < len(history['train_cost_arithmetic']) else float('nan'),
                'val_cost_arithmetic': convert_value(history['val_cost_arithmetic'][epoch]) if 'val_cost_arithmetic' in history and epoch < len(history['val_cost_arithmetic']) else float('nan'),
                'learning_rate': convert_value(history['learning_rate'][epoch]) if 'learning_rate' in history and epoch < len(history['learning_rate']) else 1e-4,
                'temperature': convert_value(history['temperature'][epoch]) if 'temperature' in history and epoch < len(history['temperature']) else 2.5,
                'time_per_epoch': convert_value(history['epoch_time'][epoch]) if 'epoch_time' in history and epoch < len(history['epoch_time']) else 0.0,
                'baseline_type': history['baseline_type'][epoch] if 'baseline_type' in history and epoch < len(history['baseline_type']) else '',
                'baseline_value': convert_value(history['baseline_value'][epoch]) if 'baseline_value' in history and epoch < len(history['baseline_value']) else float('nan'),
                'mean_type': history['mean_type'][epoch] if 'mean_type' in history and epoch < len(history['mean_type']) else 'arithmetic'
            }
            self.rows.append(row)
            pd.DataFrame([row]).to_csv(self.csv_path, mode='a', header=False, index=False)
        
        self.logger.info(f"Saved {n_epochs} epochs of history to {self.csv_path}")
    
    def finalize(self):
        """Save final summary."""
        self.logger.info(f'Training history saved to: {self.csv_path}')


def save_model_with_structure(model_name: str, model: torch.nn.Module, 
                              history: dict, config: dict, base_dir: Path, 
                              logger, training_time: float = None):
    """Save model in pytorch/ directory with metadata."""
    pytorch_dir = base_dir / 'pytorch'
    pytorch_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Prepare complete model data
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'config': config,
        'history': history,
        'training_time': training_time,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_architecture': str(model.__class__.__name__),
    }
    
    # Save in pytorch directory
    model_path = pytorch_dir / f'model_{model_key(model_name)}.pt'
    torch.save(model_data, model_path)
    logger.info(f'Saved model to pytorch/: {model_path}')
    
    return model_path


def train_single_model(model_name: str, config: Dict[str, Any], args, logger):
    """Train a single model."""
    
    # Create output directory for this model
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path(config.get('working_dir_path', 'results/gpu'))
    
    # Use base_output_dir directly, no model-specific subdirectory
    output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize CSV writer for incremental history saving (uses shared directories)
    csv_writer = IncrementalCSVWriter(model_name, output_dir, logger)
    
    # Check if model already exists
    existing_model = check_existing_model(model_name, output_dir, args.force_retrain)
    if existing_model:
        logger.info(f"Model {model_name} already trained. Found at: {existing_model}")
        logger.info("Use --force-retrain to retrain the model.")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    
    # Create checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir) / model_name.replace('+', '_')
    else:
        checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data generator
    logger.info("Creating data generator...")
    data_generator = create_data_generator(config)
    
    # Create model
    logger.info(f"Creating model: {model_name}")
    model = ModelFactory.create_model(model_name, config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print model info (matching CPU format)
    print("-" * 40)
    print(f"Training {model_name}")
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Estimate optimal batch size if requested
    if args.estimate_batch_size:
        optimal_batch_size = estimate_optimal_batch_size(model, config, logger)
        if optimal_batch_size != config['training']['batch_size']:
            logger.info(f"Updating batch size from {config['training']['batch_size']} "
                       f"to {optimal_batch_size}")
            config['training']['batch_size'] = optimal_batch_size
    
    # Save configuration with model name
    config_save_path = output_dir / f"{model_name.replace('+', '_')}_training_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {config_save_path}")
    
    # Training
    logger.info("Starting GPU-optimized training...")
    start_time = time.time()
    
    try:
        trained_model, training_result = advanced_train_model_gpu(
            model=model,
            model_name=model_name,
            data_generator=data_generator,
            config=config,
            checkpoint_dir=checkpoint_dir,
            callbacks=[lambda epoch, _model, hist: csv_writer.write_epoch(hist, epoch)],
            use_wandb=args.wandb
        )
        
        end_time = time.time()
        
        # Extract history from result
        history = training_result.get('history', {}) if isinstance(training_result, dict) else {}
        
        metrics = training_result.get('metrics', {}) if isinstance(training_result, dict) else {}
        # Debug: log what we got
        logger.info(f"History keys: {list(history.keys()) if history else 'None'}")
        if history and 'train_cost' in history:
            logger.info(f"History has {len(history['train_cost'])} epochs of training data")
        
        # Save history to CSV
        # If incremental per-epoch writes already populated the CSV, avoid duplicating rows.
        try:
            with open(csv_writer.csv_path, 'r') as f:
                existing_rows = sum(1 for _ in f) - 1  # exclude header
        except Exception:
            existing_rows = 0
        n_epochs = len(history.get('train_cost_arithmetic', []))
        if existing_rows < n_epochs:
            csv_writer.save_all_history(history)
        csv_writer.finalize()
        
        # Save model with full structure (in pytorch/ directory)
        save_model_with_structure(
            model_name=model_name,
            model=trained_model,
            history=history,
            config=config,
            base_dir=output_dir,  # Use output_dir directly (e.g., training_gpu/results/tiny/)
            logger=logger,
            training_time=end_time - start_time
        )
        
        # Save final model in the base directory with model name
        model_save_path = output_dir / f"{model_name.replace('+', '_')}_final.pt"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': config,
            'history': history
        }, model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        # Create and save training summary
        summary = create_training_summary(
            trained_model, {'history': history, 'metrics': metrics}, config, start_time, end_time
        )
        
        summary_path = output_dir / f"{model_name.replace('+', '_')}_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Training summary saved to {summary_path}")
        
        # Print summary
        print("\n" + "="*60)
        print(f"TRAINING COMPLETED: {model_name}")
        print("="*60)
        print(f"Model: {summary['model']}")
        print(f"Problem Size: {summary['config']['problem_size']} customers")
        print(f"Device: {summary['results']['device']}")
        print(f"Final Train Cost: {summary['results']['final_train_cost']:.4f}")
        print(f"Final Val Cost: {summary['results']['final_val_cost']:.4f}")
        print(f"Best Val Cost: {summary['results']['best_val_cost']:.4f}")
        print(f"Total Time: {summary['results']['total_time']:.2f} seconds")
        print(f"Time per Epoch: {summary['results']['time_per_epoch']:.2f} seconds")
        print(f"Peak GPU Memory: {summary['results']['peak_gpu_memory_gb']:.2f} GB")
        if summary['performance']['gpu_efficiency']:
            print(f"GPU Memory Efficiency: {summary['performance']['gpu_efficiency']}")
        print(f"Throughput: {summary['performance']['throughput']:.1f} instances/second")
        print("="*60)
        
        return summary
        
    except Exception as e:
        logger.error(f"Training failed for {model_name}: {str(e)}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='GPU-Optimized CVRP Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train DGT model with default config
  python %(prog)s --model DGT+RL --config configs/gpu_default.yaml
  
  # Train GAT with mixed precision and auto batch size
  python %(prog)s --model GAT+RL --mixed_precision --estimate_batch_size
  
  # Train all models sequentially
  python %(prog)s --all --config configs/gpu_default.yaml
  
  # Force retrain existing model
  python %(prog)s --model GT+RL --force-retrain
  
  # Train on specific GPU with custom output directory
  python %(prog)s --model DGT+RL --device cuda:1 --output-dir results/my_experiment
        """
    )
    
    # Configuration - FIRST for compatibility with CPU version
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    
    # Model selection - matching CPU version format
    parser.add_argument('--model', type=str,
                       choices=['GAT+RL', 'GT+RL', 'DGT+RL', 'GT-Greedy',
                               'gat', 'gt', 'dgt', 'greedy'],  # Also support lowercase
                       help='Model type to train')
    
    parser.add_argument('--all', action='store_true',
                       help='Train all available models sequentially')
    
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if saved model exists')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory from config')
    
    # Additional training parameters
    parser.add_argument('--problem_size', type=int, default=None,
                       help='Override problem size (number of customers)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    
    # GPU-specific settings
    parser.add_argument('--device', type=str, default=None,
                       help='GPU device (e.g., cuda:0, cuda:1)')
    parser.add_argument('--mixed_precision', action='store_true', default=None,
                       help='Enable mixed precision training')
    parser.add_argument('--no_mixed_precision', action='store_false', 
                       dest='mixed_precision',
                       help='Disable mixed precision training')
    parser.add_argument('--estimate_batch_size', action='store_true',
                       help='Automatically estimate optimal batch size')
    parser.add_argument('--memory_fraction', type=float, default=None,
                       help='GPU memory fraction to use (0.0-1.0)')
    
    # Checkpointing and logging
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory for saving checkpoints')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run in benchmark mode')
    
    # Other options
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--no_gpu_info', action='store_true',
                       help='Skip printing GPU information')
    
    args = parser.parse_args()
    
    # Print GPU information unless disabled
    if not args.no_gpu_info:
        has_gpu = print_gpu_info()
    else:
        has_gpu = cuda.is_available()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = project_root / args.config
    
    config = load_config(str(config_path))
    
    # Fix relative paths to be relative to script location
    if 'working_dir_path' in config and config['working_dir_path'].startswith('..'):
        script_dir = Path(__file__).parent
        config['working_dir_path'] = str((script_dir / config['working_dir_path']).resolve())
    
    # Automatically replace "training_cpu" with "training_gpu" in working_dir_path
    # This ensures GPU training results go to training_gpu/ directory structure
    if 'working_dir_path' in config and config['working_dir_path']:
        original_path = config['working_dir_path']
        # Replace training_cpu with training_gpu in the path
        modified_path = original_path.replace('training_cpu', 'training_gpu')
        if modified_path != original_path:
            config['working_dir_path'] = modified_path
            print(f"Auto-adjusted output path: {original_path} -> {modified_path}")
    
    # Override configuration with command line arguments
    if args.problem_size is not None:
        config['problem']['num_customers'] = args.problem_size
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.device is not None:
        config['gpu']['device'] = args.device
    if args.mixed_precision is not None:
        config['gpu']['mixed_precision'] = args.mixed_precision
    if args.memory_fraction is not None:
        config['gpu']['memory_fraction'] = args.memory_fraction
    if args.seed is not None:
        config['experiment']['random_seed'] = args.seed
    if args.output_dir is not None:
        config['working_dir_path'] = args.output_dir
    
    # Setup logging
    logger = setup_logging(config)
    
    # Validate GPU configuration
    if not validate_gpu_config(config):
        logger.error("GPU configuration validation failed. Exiting.")
        return 1
    
    # Set random seeds
    seed = config['experiment'].get('random_seed', 42)
    set_seeds(seed)
    logger.info(f"Random seed set to {seed}")
    
    # Determine which models to train
    if args.all:
        models_to_train = ['GAT+RL', 'GT+RL', 'DGT+RL', 'GT-Greedy']
        logger.info(f"Training all models: {models_to_train}")
    elif args.model:
        models_to_train = [args.model]
    else:
        logger.error("No model specified. Use --model or --all")
        parser.print_help()
        return 1
    
    # Train models
    results = {}
    for model_name in models_to_train:
        result = train_single_model(model_name, config.copy(), args, logger)
        if result:
            results[model_name] = result
    
    # Print final summary if multiple models were trained
    if len(results) > 1:
        print("\n" + "="*80)
        print("TRAINING SESSION SUMMARY")
        print("="*80)
        for model_name, summary in results.items():
            print(f"\n{model_name}:")
            print(f"  - Final Val Cost: {summary['results']['final_val_cost']:.4f}")
            print(f"  - Best Val Cost: {summary['results']['best_val_cost']:.4f}")
            print(f"  - Training Time: {summary['results']['total_time']:.2f}s")
            print(f"  - Peak GPU Memory: {summary['results']['peak_gpu_memory_gb']:.2f} GB")
        print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
