#!/usr/bin/env python3
"""
Experimental training script with early stopping for hyperparameter search.

Key improvements:
1. Early stopping with patience parameter (default 36)
2. Minimum epochs before early stopping (default 16)
3. Tracks epochs trained for efficiency analysis
4. Better progress monitoring with CSV output
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
from src.pipelines.train import set_seeds, generate_cvrp_instance
from src.metrics.costs import compute_route_cost, compute_naive_baseline_cost
from src.eval.validation import validate_route
from src.models.pointer import BaselinePointerNetwork
from src.models.gt import GraphTransformerNetwork
from src.models.greedy_gt import GraphTransformerGreedy
from src.models.dgt import DynamicGraphTransformerNetwork
from src.models.gat import GraphAttentionTransformer

# Import our new lightweight variants
from src.models.gt_lite import GraphTransformerLite
from src.models.dgt_lite import DynamicGraphTransformerLite
from src.models.gt_ultra import GraphTransformerUltra
from src.models.dgt_ultra import DynamicGraphTransformerUltra
from src.models.dgt_super import DynamicGraphTransformerSuper


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
        'GAT+RL': 'gat_rl',
        'GT+RL': 'gt_rl',
        'DGT+RL': 'dgt_rl',
        'GT-Lite+RL': 'gt_lite_rl',
        'DGT-Lite+RL': 'dgt_lite_rl',
        'GT-Ultra+RL': 'gt_ultra_rl',
        'DGT-Ultra+RL': 'dgt_ultra_rl',
        'DGT-Super+RL': 'dgt_super_rl',
    }
    return mapping.get(name, name.lower().replace(' ', '_').replace('+', '_').replace('-', '_'))


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_history_csv(model_name: str, history: dict, config: dict, base_dir: str, logger) -> None:
    """Write one model's history to CSV immediately."""
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    num_epochs = int(history.get('epochs_trained', config['training']['num_epochs']))
    val_freq = int(config['training']['validation_frequency'])
    train_losses = list(history.get('train_losses', []))
    train_costs = list(history.get('train_costs', []))
    val_costs_seq = list(history.get('val_costs', []))
    
    # Create series for all epochs
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
    
    # Fill final epoch
    if num_epochs > 0:
        if not np.isnan(train_loss_series[num_epochs-1]):
            train_loss_series[num_epochs] = train_loss_series[num_epochs - 1]
        if not np.isnan(train_cost_series[num_epochs-1]):
            train_cost_series[num_epochs] = train_cost_series[num_epochs - 1]
    
    # Map validation costs
    expected_val_epochs = [ep for ep in range(0, num_epochs + 1)
                           if (ep % max(1, val_freq) == 0) or (ep == num_epochs)]
    for i, ep in enumerate(expected_val_epochs):
        if i < len(val_costs_seq):
            val_cost_series[ep] = val_costs_seq[i]
    
    if len(val_costs_seq) > 0:
        val_cost_series[num_epochs] = val_costs_seq[-1]
    
    df = pd.DataFrame({
        'epoch': list(range(0, num_epochs + 1)),
        'train_loss': train_loss_series,
        'train_cost': train_cost_series,
        'val_cost': val_cost_series,
    })
    out = os.path.join(csv_dir, f'history_{model_key(model_name)}.csv')
    df.to_csv(out, index=False)
    logger.info(f'📊 Saved history CSV: {out}')

    # Also write detailed results to a separate file
    details_file = os.path.join(base_dir, model_name, 'training_details.csv')
    os.makedirs(os.path.dirname(details_file), exist_ok=True)
    
    # Create more detailed DataFrame
    details_df = pd.DataFrame({
        'epoch': list(range(0, num_epochs + 1)),
        'train_loss': train_loss_series,
        'train_cost': train_cost_series,
        'val_cost': val_cost_series,
    })
    details_df.to_csv(details_file, index=False)


def save_models_and_analysis(results, training_times, models, config, base_dir, logger):
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    analysis_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(pytorch_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    for model_name, model in models.items():
        path = os.path.join(pytorch_dir, f'model_{model_key(model_name)}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': config,
            'results': results[model_name]['history'],
            'training_time': training_times[model_name],
            'num_parameters': count_parameters(model),
        }, path)
        logger.info(f'💾 Saved model checkpoint: {path}')

    # Save complete analysis
    torch.save({
        'results': {k: v['history'] for k, v in results.items()},
        'training_times': training_times,
        'config': config,
        'parameter_counts': {name: count_parameters(model) for name, model in models.items()},
    }, os.path.join(analysis_dir, 'experimental_study_complete.pt'))
    logger.info(f'📦 Saved experimental analysis: {os.path.join(analysis_dir, "experimental_study_complete.pt")}')


def build_model(name: str, config: dict):
    """Build model based on name and config."""
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    num_heads = config['model']['num_heads']
    num_layers = config['model']['num_layers']
    dropout = config['model']['transformer_dropout']
    ff_mult = config['model']['feedforward_multiplier']
    edge_div = config['model']['edge_embedding_divisor']
    
    if name == 'GAT+RL':
        return GraphAttentionTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout, edge_div, config)
    elif name == 'GT+RL':
        return GraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'DGT+RL':
        return DynamicGraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'GT-Lite+RL':
        return GraphTransformerLite(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'DGT-Lite+RL':
        return DynamicGraphTransformerLite(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'GT-Ultra+RL':
        return GraphTransformerUltra(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'DGT-Ultra+RL':
        return DynamicGraphTransformerUltra(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    elif name == 'DGT-Super+RL':
        return DynamicGraphTransformerSuper(input_dim, hidden_dim, num_heads, num_layers, dropout, ff_mult, config)
    else:
        raise ValueError(f'Unknown model name: {name}')


def train_with_early_stopping(model, model_name, config, log_func):
    """Modified training function with early stopping."""
    # Get early stopping parameters
    early_stopping_patience = config.get('early_stopping_patience', 36)
    min_epochs = config.get('min_epochs', 16)
    max_epochs = config.get('max_epochs', config['training']['num_epochs'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_func(f"🖥️  Using device: {device}")
    model = model.to(device)
    
    # Set up optimizer
    lr = config['training']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Generate training and validation data
    num_instances = config['training']['num_instances']
    split_ratio = config.get('training', {}).get('train_val_split', 0.8)
    train_size = int(num_instances * split_ratio)
    val_size = num_instances - train_size
    
    log_func(f"📊 Data: {train_size} train, {val_size} validation instances")
    
    # Temperature settings
    temp_config = config.get('training_advanced', {})
    temp_start = temp_config.get('temp_start', 1.0)
    temp_min = temp_config.get('temp_min', 0.1)
    temp_adaptation_rate = temp_config.get('temp_adaptation_rate', 0.0)
    use_adaptive = temp_config.get('use_adaptive_temperature', True)
    
    log_func(f"🌡️  Temperature: start={temp_start}, min={temp_min}, decay={temp_adaptation_rate}")
    
    # Generate problem instances
    problem_config = config['problem']
    coord_range = problem_config.get('coord_range', problem_config.get('coordinate_range', [0, 100]))
    if isinstance(coord_range, list):
        coord_range = coord_range[1]  # Take the max value
    demand_range = problem_config.get('demand_range', [1, 9])
    
    train_instances = [
        generate_cvrp_instance(problem_config['num_customers'], problem_config['vehicle_capacity'], 
                             coord_range, demand_range, seed=i)
        for i in range(train_size)
    ]
    val_instances = [
        generate_cvrp_instance(problem_config['num_customers'], problem_config['vehicle_capacity'],
                             coord_range, demand_range, seed=10000+i)
        for i in range(val_size)
    ]
    
    # Training loop with early stopping
    batch_size = config['training']['batch_size']
    val_freq = config['training']['validation_frequency']
    
    history = {
        'train_losses': [],
        'train_costs': [],
        'val_costs': [],
        'final_val_cost': float('nan'),
        'epochs_trained': 0
    }
    
    best_val_cost = float('inf')
    best_epoch = -1
    no_improvement = 0
    
    temperature = temp_start
    
    # Main training loop
    for epoch in range(max_epochs):
        # Training
        model.train()
        epoch_losses = []
        epoch_costs = []
        
        # Shuffle and batch
        indices = np.random.permutation(len(train_instances))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [train_instances[idx] for idx in batch_indices]
            
            optimizer.zero_grad()
            
            # Forward pass gets routes, log_probs, entropy
            routes, log_probs, entropy = model(batch, 
                                              max_steps=problem_config['num_customers'] * config['inference']['max_steps_multiplier'],
                                              temperature=temperature, 
                                              greedy=False, 
                                              config=config)
            
            # Compute costs for each route in the batch
            batch_costs = []
            for b in range(len(batch)):
                route = routes[b]
                validate_route(route, problem_config['num_customers'], f"{model_name}-TRAIN", batch[b])
                cost = compute_route_cost(route, batch[b]['distances']) / problem_config['num_customers']
                batch_costs.append(cost)
            
            # REINFORCE algorithm
            costs_tensor = torch.tensor(batch_costs, dtype=torch.float32)
            baseline = costs_tensor.mean().detach()
            advantages = baseline - costs_tensor  # lower cost -> positive advantage
            
            # Entropy regularization (simplified version)
            entropy_coef = config.get('training_advanced', {}).get('entropy_coef', 0.01)
            
            # Compute loss: policy gradient + entropy regularization
            loss = (-(advantages) * log_probs).mean() - entropy_coef * entropy.mean()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                         float(config.get('training_advanced', {}).get('gradient_clip_norm', 1.0)))
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_costs.append(float(np.mean(batch_costs)))
        
        # Update history
        avg_loss = np.mean(epoch_losses)
        avg_cost = np.mean(epoch_costs)
        history['train_losses'].append(avg_loss)
        history['train_costs'].append(avg_cost)
        
        # Validation
        if epoch % val_freq == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_costs = []
                for i in range(0, len(val_instances), batch_size):
                    batch = val_instances[i:min(i+batch_size, len(val_instances))]
                    
                    # Validation with greedy decoding
                    routes, _, _ = model(batch, 
                                       max_steps=problem_config['num_customers'] * config['inference']['max_steps_multiplier'],
                                       temperature=temp_min, 
                                       greedy=True, 
                                       config=config)
                    
                    # Compute costs for validation batch
                    for b in range(len(batch)):
                        route = routes[b]
                        validate_route(route, problem_config['num_customers'], f"{model_name}-VAL", batch[b])
                        cost = compute_route_cost(route, batch[b]['distances']) / problem_config['num_customers']
                        val_costs.append(cost)
                
                avg_val_cost = np.mean(val_costs)
                history['val_costs'].append(avg_val_cost)
                
                # Check for improvement
                if avg_val_cost < best_val_cost:
                    improvement = (best_val_cost - avg_val_cost) / best_val_cost * 100 if best_val_cost != float('inf') else 100
                    log_func(f"Epoch {epoch+1}: validation cost = {avg_val_cost:.4f} (↓{improvement:.1f}%)")
                    
                    best_val_cost = avg_val_cost
                    best_epoch = epoch
                    no_improvement = 0
                else:
                    no_improvement += val_freq
                    log_func(f"Epoch {epoch+1}: validation cost = {avg_val_cost:.4f} (no improvement for {no_improvement} epochs)")
        
        # Update temperature
        if use_adaptive and temp_adaptation_rate > 0:
            temperature = max(temp_min, temperature * (1 - temp_adaptation_rate))
        
        # Early stopping check - but only after min_epochs
        if epoch >= min_epochs and no_improvement >= early_stopping_patience:
            log_func(f"🛑 Early stopping after {epoch+1} epochs (no improvement for {no_improvement} epochs)")
            break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_val_costs = []
        for i in range(0, len(val_instances), batch_size):
            batch = val_instances[i:min(i+batch_size, len(val_instances))]
            
            # Final evaluation with greedy decoding
            routes, _, _ = model(batch, 
                               max_steps=problem_config['num_customers'] * config['inference']['max_steps_multiplier'],
                               temperature=temp_min, 
                               greedy=True, 
                               config=config)
            
            # Compute costs for final evaluation
            for b in range(len(batch)):
                route = routes[b]
                validate_route(route, problem_config['num_customers'], f"{model_name}-FINAL", batch[b])
                cost = compute_route_cost(route, batch[b]['distances']) / problem_config['num_customers']
                final_val_costs.append(cost)
        
        final_val_cost = np.mean(final_val_costs)
        history['final_val_cost'] = final_val_cost
        history['epochs_trained'] = epoch + 1
        
        log_func(f"🏁 Final validation cost: {final_val_cost:.4f}")
        log_func(f"⏱️  Epochs trained: {epoch+1}/{max_epochs}")
        
        if best_epoch >= 0:
            log_func(f"🏆 Best epoch: {best_epoch+1} with validation cost {best_val_cost:.4f}")
    
    # Return history, None for time (calculated externally), and None for artifacts
    return history, None, None


def main():
    parser = argparse.ArgumentParser(description='Experimental CVRP model training with early stopping')
    parser.add_argument('--config', type=str, default='configs/small.yaml', help='Path to YAML configuration file')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['GAT+RL', 'GT+RL', 'DGT+RL'],
                       help='Models to train')
    parser.add_argument('--embedding_dim', type=int, default=None, help='Node embedding dimension')
    parser.add_argument('--n_layers', type=int, default=None, help='Number of encoder layers')
    parser.add_argument('--n_heads', type=int, default=None, help='Number of attention heads')
    parser.add_argument('--learning_rate', type=float, default=None, help='Adam optimizer learning rate')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Training batch size')
    parser.add_argument('--temp_start', type=float, default=None, help='Initial temperature for exploration')
    parser.add_argument('--temp_min', type=float, default=None, help='Minimum temperature')
    parser.add_argument('--temp_decay', type=float, default=None, help='Temperature decay rate')
    parser.add_argument('--early_stopping_patience', type=int, default=36, help='Early stopping patience')
    parser.add_argument('--min_epochs', type=int, default=16, help='Minimum epochs before early stopping')
    parser.add_argument('--max_epochs', type=int, default=128, help='Maximum epochs to train')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.embedding_dim is not None:
        config['model']['hidden_dim'] = args.embedding_dim
    if args.n_layers is not None:
        config['model']['num_layers'] = args.n_layers
    if args.n_heads is not None:
        config['model']['num_heads'] = args.n_heads
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.dropout is not None:
        config['model']['transformer_dropout'] = args.dropout
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.temp_start is not None:
        if 'training_advanced' not in config:
            config['training_advanced'] = {}
        config['training_advanced']['temp_start'] = args.temp_start
    if args.temp_min is not None:
        if 'training_advanced' not in config:
            config['training_advanced'] = {}
        config['training_advanced']['temp_min'] = args.temp_min
    if args.temp_decay is not None:
        if 'training_advanced' not in config:
            config['training_advanced'] = {}
        config['training_advanced']['temp_adaptation_rate'] = args.temp_decay
    
    # Add early stopping parameters
    config['early_stopping_patience'] = args.early_stopping_patience
    config['min_epochs'] = args.min_epochs
    config['max_epochs'] = args.max_epochs

    logger = setup_logging(config)
    
    # Ensure we have the right problem parameters (20 customers, capacity 30)
    assert config['problem']['num_customers'] == 20, f"Expected 20 customers, got {config['problem']['num_customers']}"
    assert config['problem']['vehicle_capacity'] == 30, f"Expected capacity 30, got {config['problem']['vehicle_capacity']}"
    
    logger.info(f"🎯 Target: Beat GAT+RL's 0.54 validation cost")
    logger.info(f"📏 Problem: {config['problem']['num_customers']} customers, capacity {config['problem']['vehicle_capacity']}")
    logger.info(f"⏱️  Early stopping: patience={args.early_stopping_patience}, min_epochs={args.min_epochs}, max_epochs={args.max_epochs}")
    
    # Seeds and environment
    set_seeds(config.get('experiment', {}).get('random_seed', 42))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    base_dir = str(Path(config.get('working_dir_path', 'results/experimental')).as_posix())
    os.makedirs(base_dir, exist_ok=True)
    
    logger.info(f"🔬 Training models: {args.models}")

    # Build all models and count parameters
    models = {}
    parameter_counts = {}
    for model_name in args.models:
        model = build_model(model_name, config)
        param_count = count_parameters(model)
        models[model_name] = model
        parameter_counts[model_name] = param_count
        logger.info(f"🏗️  {model_name}: {param_count:,} parameters")

    # Training and results tracking
    results = {}
    training_times = {}

    for model_name in args.models:
        logger.info(f"\n🚀 Training {model_name}...")
        start_time = time.time()
        
        try:
            # Train the model with early stopping
            model = models[model_name]
            history, _, _ = train_with_early_stopping(model, model_name, config, logger.info)
            result = {'history': history}
            
            training_time = time.time() - start_time
            training_times[model_name] = training_time
            results[model_name] = result
            
            # Extract final validation cost
            final_val_cost = result['history'].get('final_val_cost', float('nan'))
            epochs_trained = result['history'].get('epochs_trained', 0)
            
            if not np.isnan(final_val_cost):
                improvement_vs_gat = ((0.54 - final_val_cost) / 0.54) * 100
                logger.info(f"✅ {model_name} final validation cost: {final_val_cost:.4f}")
                logger.info(f"⏱️  Epochs trained: {epochs_trained}")
                logger.info(f"📈 Improvement vs GAT+RL (0.54): {improvement_vs_gat:+.1f}%")
                if final_val_cost < 0.54:
                    logger.info(f"🎉 {model_name} BEATS GAT+RL!")
            else:
                logger.warning(f"⚠️  {model_name} validation cost unavailable")
            
            # Save CSV immediately
            write_history_csv(model_name, result['history'], config, base_dir, logger)
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            training_times[model_name] = time.time() - start_time
            results[model_name] = {'history': {'train_losses': [], 'train_costs': [], 'val_costs': [], 
                                            'final_val_cost': float('nan'), 'epochs_trained': 0}}

    # Save all results
    save_models_and_analysis(results, training_times, models, config, base_dir, logger)

    # Summary
    logger.info(f"\n📊 EXPERIMENTAL RESULTS SUMMARY:")
    logger.info(f"Target to beat: GAT+RL = 0.54")
    logger.info(f"{'Model':<20} {'Parameters':<12} {'Val Cost':<10} {'vs GAT+RL':<10} {'Epochs':<8}")
    logger.info("-" * 70)
    
    for model_name in args.models:
        param_count = parameter_counts[model_name]
        final_val_cost = results[model_name]['history'].get('final_val_cost', float('nan'))
        epochs_trained = results[model_name]['history'].get('epochs_trained', 0)
        
        if not np.isnan(final_val_cost):
            improvement = ((0.54 - final_val_cost) / 0.54) * 100
            status = "BETTER" if final_val_cost < 0.54 else "WORSE"
            logger.info(f"{model_name:<20} {param_count:<12,} {final_val_cost:<10.4f} {improvement:+.1f}% {status} {epochs_trained:<8}")
        else:
            logger.info(f"{model_name:<20} {param_count:<12,} {'N/A':<10} {'N/A':<10} {epochs_trained:<8}")

    # Print final validation cost for hyperparameter search parsing
    best_model = None
    best_cost = float('inf')
    for model_name in args.models:
        final_val_cost = results[model_name]['history'].get('final_val_cost', float('nan'))
        if not np.isnan(final_val_cost) and final_val_cost < best_cost:
            best_cost = final_val_cost
            best_model = model_name
    
    if best_model is not None:
        print(f"final validation cost: {best_cost:.6f}")  # For hyperparameter search parsing
        print(f"epochs trained: {results[best_model]['history'].get('epochs_trained', 0)}")
    
    logger.info(f"\n🎯 Experiment completed! Results saved to: {base_dir}")


if __name__ == '__main__':
    main()
