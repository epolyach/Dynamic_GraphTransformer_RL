#!/usr/bin/env python3
"""
Enhanced CVRP Model Training Script
Supports training multiple models in sequence or all at once.

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
import subprocess
import json

import numpy as np
import pandas as pd
import torch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.training.advanced_trainer import advanced_train_model
from src.data.enhanced_generator import create_enhanced_data_generator
from src.models.model_factory import ModelFactory
from src.pipelines.train import set_seeds

# Import the original functions from run_training.py
from run_training import setup_logging, model_key, IncrementalCSVWriter


AVAILABLE_MODELS = ['GAT+RL', 'GT+RL', 'DGT+RL', 'GT-Greedy']


def train_single_model(model_name: str, config_path: str, force_retrain: bool = False, 
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Train a single model and return results."""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    
    # Set up output directory
    if output_dir:
        config['working_dir_path'] = output_dir
    working_dir = Path(config['working_dir_path'])
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    model_file = working_dir / 'pytorch' / f'model_{model_key(model_name)}.pt'
    if model_file.exists() and not force_retrain:
        logger.info(f"Model {model_name} already trained. Skipping (use --force-retrain to override)")
        return {'model': model_name, 'status': 'skipped', 'path': str(model_file)}
    
    # Train the model
    start_time = time.time()
    
    try:
        # Set random seeds
        seed = config['train_seed']
        set_seeds(seed, deterministic=True)
        
        # Create data generator
        data_generator = create_enhanced_data_generator(config)
        
        # Create model
        model = ModelFactory.create_model(
            model_type=model_name,
            config=config
        )
        
        # Set up CSV writer for incremental logging
        csv_writer = IncrementalCSVWriter(
            config=config,
            model_name=model_name,
            output_dir=working_dir / 'csv',
        )
        
        # Train with advanced features
        history, best_val_cost, artifacts = advanced_train_model(
            model=model,
            config=config,
            data_generator=data_generator,
            model_name=model_name,
            use_advanced_features=True,
            epoch_callback=csv_writer.write_epoch
        )
        
        # Save model and results
        pytorch_dir = working_dir / 'pytorch'
        pytorch_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history,
            'best_val_cost': best_val_cost,
            'model_name': model_name,
        }, model_file)
        
        # Finalize CSV
        csv_writer.finalize()
        
        training_time = time.time() - start_time
        
        result = {
            'model': model_name,
            'status': 'success',
            'best_val_cost': best_val_cost,
            'final_train_cost': history['train_costs'][-1] if history['train_costs'] else None,
            'final_val_cost': history.get('final_val_cost', None),
            'training_time': training_time,
            'epochs': len(history['train_costs']),
            'path': str(model_file)
        }
        
        logger.info(f"✓ {model_name} training completed in {training_time:.2f}s")
        logger.info(f"  Best validation cost: {best_val_cost:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ {model_name} training failed: {str(e)}")
        return {
            'model': model_name,
            'status': 'failed',
            'error': str(e),
            'training_time': time.time() - start_time
        }


def train_all_models(config_path: str, force_retrain: bool = False,
                    output_dir: Optional[str] = None,
                    models: Optional[List[str]] = None) -> pd.DataFrame:
    """Train all specified models and return comparison results."""
    
    if models is None:
        models = AVAILABLE_MODELS
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"TRAINING MULTIPLE MODELS")
    print(f"{'='*60}")
    print(f"Models to train: {', '.join(models)}")
    print(f"Config: {config_path}")
    print(f"Force retrain: {force_retrain}")
    
    start_time = time.time()
    
    for model_name in models:
        if model_name not in AVAILABLE_MODELS:
            print(f"Warning: Unknown model {model_name}, skipping")
            continue
            
        result = train_single_model(
            model_name=model_name,
            config_path=config_path,
            force_retrain=force_retrain,
            output_dir=output_dir
        )
        results.append(result)
        
        # Small delay between models
        time.sleep(2)
    
    total_time = time.time() - start_time
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Save summary
    config = load_config(config_path)
    if output_dir:
        config['working_dir_path'] = output_dir
    working_dir = Path(config['working_dir_path'])
    
    summary_file = working_dir / 'training_summary.csv'
    df.to_csv(summary_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(df.to_string())
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Summary saved to: {summary_file}")
    
    # Also save as JSON for easy parsing
    json_file = working_dir / 'training_summary.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return df


def compare_results(config_path: str, output_dir: Optional[str] = None):
    """Load and compare results from all trained models."""
    
    config = load_config(config_path)
    if output_dir:
        config['working_dir_path'] = output_dir
    working_dir = Path(config['working_dir_path'])
    
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON")
    print(f"{'='*60}")
    
    comparison_data = []
    
    for model_name in AVAILABLE_MODELS:
        csv_file = working_dir / 'csv' / f'history_{model_key(model_name)}.csv'
        model_file = working_dir / 'pytorch' / f'model_{model_key(model_name)}.pt'
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            
            # Get key metrics
            train_costs = df['train_cost'].dropna()
            val_costs = df['val_cost'].dropna()
            
            comparison_data.append({
                'Model': model_name,
                'Final Train Cost': train_costs.iloc[-1] if len(train_costs) > 0 else None,
                'Best Train Cost': train_costs.min() if len(train_costs) > 0 else None,
                'Final Val Cost': val_costs.iloc[-1] if len(val_costs) > 0 else None,
                'Best Val Cost': val_costs.min() if len(val_costs) > 0 else None,
                'Train-Val Gap (%)': ((val_costs.mean() - train_costs.mean()) / train_costs.mean() * 100) 
                                    if len(train_costs) > 0 and len(val_costs) > 0 else None,
                'Epochs': len(df),
                'Model Exists': model_file.exists()
            })
    
    if comparison_data:
        df_compare = pd.DataFrame(comparison_data)
        print(df_compare.to_string())
        
        # Save comparison
        comparison_file = working_dir / 'model_comparison.csv'
        df_compare.to_csv(comparison_file, index=False)
        print(f"\nComparison saved to: {comparison_file}")
    else:
        print("No trained models found")
    
    return df_compare if comparison_data else None


def main():
    parser = argparse.ArgumentParser(
        description='Train and compare CVRP models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  python run_all_experiments.py --all --config configs/small.yaml
  
  # Train specific models
  python run_all_experiments.py --models GT+RL DGT+RL --config configs/small.yaml
  
  # Train single model
  python run_all_experiments.py --models DGT+RL --config configs/tiny.yaml
  
  # Force retrain all models
  python run_all_experiments.py --all --force-retrain --config configs/small.yaml
  
  # Just compare existing results
  python run_all_experiments.py --compare-only --config configs/small.yaml
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    
    parser.add_argument('--models', nargs='+', type=str,
                       choices=AVAILABLE_MODELS,
                       help='Specific models to train')
    
    parser.add_argument('--all', action='store_true',
                       help='Train all available models')
    
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining even if saved models exist')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory from config')
    
    parser.add_argument('--compare-only', action='store_true',
                       help='Only compare existing results without training')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.compare_only and not args.all and not args.models:
        parser.error("Must specify either --all, --models, or --compare-only")
    
    # Just compare if requested
    if args.compare_only:
        compare_results(args.config, args.output_dir)
        return
    
    # Determine which models to train
    if args.all:
        models_to_train = AVAILABLE_MODELS
    else:
        models_to_train = args.models
    
    # Train models
    results_df = train_all_models(
        config_path=args.config,
        force_retrain=args.force_retrain,
        output_dir=args.output_dir,
        models=models_to_train
    )
    
    # Compare results
    compare_results(args.config, args.output_dir)
    
    print("\n✓ All experiments completed!")


if __name__ == '__main__':
    main()
