#!/usr/bin/env python3
"""
Optimizer Experiment Training Script (Isolated)

- Uses the experimental trainer with Adam β2 batch-size scaling and advantage normalization control
- Keeps the standard training pipeline intact (separate script + config)
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
# Make training_cpu/lib importable
training_cpu_path = Path(__file__).parent.parent
sys.path.insert(0, str(training_cpu_path))
from lib.advanced_trainer_opt_experiments import advanced_train_model_opt
from src.generator.generator import create_data_generator
from src.models.model_factory import ModelFactory
from src.utils.seeding import set_seeds


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
    return name.lower().replace('+', '_').replace('-', '_').replace(' ', '_')


class IncrementalCSVWriter:
    def __init__(self, model_name: str, base_dir: str, config: dict, logger):
        self.csv_dir = os.path.join(base_dir, 'csv')
        os.makedirs(self.csv_dir, exist_ok=True)
        self.csv_path = os.path.join(self.csv_dir, f'history_{model_key(model_name)}.csv')
        self.val_freq = int(config['training']['validation_frequency'])
        self.logger = logger
        self.rows = []
        self.val_idx = 0
        self.baseline_type = config.get('baseline', {}).get('type', 'mean')
        pd.DataFrame(columns=['epoch','train_loss','train_cost','val_cost','learning_rate','temperature','baseline_type','baseline_value']).to_csv(self.csv_path, index=False)
        self.logger.info(f'[OPT] Created CSV history file: {self.csv_path}')
    
    def write_epoch(self, epoch: int, train_loss: float, train_cost: float, val_cost: Optional[float], learning_rate: float, temperature: float, baseline_value: float = None):
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
        pd.DataFrame([row]).to_csv(self.csv_path, mode='a', header=False, index=False)

    def finalize(self):
        self.logger.info(f'[OPT] Training history saved to: {self.csv_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Optimizer experiment: Adam β2 batch-size scaling')
    parser.add_argument('--config', type=str, default='../../configs/tiny_opt.yaml', help='Path to experimental config file')
    parser.add_argument('--model', type=str, choices=['GT+RL'], default='GT+RL', help='Model to train (default: GT+RL)')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining even if saved model exists')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory from config')
    return parser.parse_args()


def main():
    args = parse_args()

    # Save current directory and change to project root for config loading
    original_cwd = os.getcwd()
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    config_path = Path(original_cwd) / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(str(config_path))
    os.chdir(original_cwd)

    logger = setup_logging(config)
    logger.info('='*60)
    logger.info('OPTIMIZER EXPERIMENT: Adam β2 batch-size scaling')
    logger.info('='*60)
    logger.info(f"Configuration: {args.config}")

    models_to_train = [args.model]
    logger.info(f"Models to train: {', '.join(models_to_train)}")

    # Setup environment
    set_seeds(config.get('experiment', {}).get('random_seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Setup output directory
    if args.output_dir:
        base_dir = args.output_dir
    elif 'working_dir_path' in config:
        working_dir = Path(config['working_dir_path'])
        base_dir = project_root / working_dir if not working_dir.is_absolute() else working_dir
        base_dir = str(base_dir)
    else:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'results_opt')
    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"Output directory: {base_dir}")

    # Create data generator
    logger.info("\nSetting up data generator (experiment)...")
    data_generator = create_data_generator(config)

    logger.info("\n" + "="*60)
    logger.info("STARTING OPTIMIZER EXPERIMENT TRAINING")
    logger.info("="*60)

    for model_name in models_to_train:
        try:
            # Create model and move to device
            model = ModelFactory.create_model(model_name, config)
            model = model.to(device)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model architecture: {model.__class__.__name__}")
            logger.info(f"Total parameters: {total_params:,}")

            # CSV writer
            csv_writer = IncrementalCSVWriter(model_name, base_dir, config, logger)

            # Train with experimental trainer
            history, training_time, artifacts = advanced_train_model_opt(
                model=model,
                model_name=model_name,
                config=config,
                data_generator=data_generator,
                logger_print=logger.info,
                use_advanced_features=config.get('experiment', {}).get('use_advanced_features', True),
                epoch_callback=csv_writer.write_epoch
            )

            csv_writer.finalize()

            # Save model
            pytorch_dir = os.path.join(base_dir, 'pytorch')
            os.makedirs(pytorch_dir, exist_ok=True)
            model_path = os.path.join(pytorch_dir, f'model_{model_key(model_name)}_opt.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'config': config,
                'history': history,
                'artifacts': artifacts,
                'training_time': training_time,
                'total_parameters': total_params,
            }, model_path)
            logger.info(f"[OPT] Saved model: {model_path}")

            final_val = history.get('final_val_cost', float('inf'))
            best_val = artifacts.get('best_val_cost', final_val)
            convergence_epoch = artifacts.get('convergence_epoch', 'N/A')

            logger.info(f"\nExperiment Complete:")
            logger.info(f"  Training time: {training_time:.1f}s")
            logger.info(f"  Final validation cost: {final_val:.4f}")
            logger.info(f"  Best validation cost: {best_val:.4f}")
            logger.info(f"  Convergence epoch: {convergence_epoch}")

        except Exception as e:
            logger.error(f"[OPT] Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

