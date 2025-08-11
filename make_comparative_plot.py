#!/usr/bin/env python3
"""
Standalone Comparative Plot Generator

Loads saved training results and generates comprehensive comparison plots.
Creates plots/comparative_study_results.png with 8 subplots showing:
1. Training Loss Evolution
2. Training Cost Evolution (Per Customer) 
3. Validation Cost vs Naive (Per Customer)
4. Final Performance Bar Chart
5. Training Time Comparison
6. Model Complexity (Parameters)
7. Learning Efficiency (Cost Improvement)
8. Performance vs Complexity Scatter

Usage:
    python make_comparative_plot.py [--config CONFIG_PATH] [--suffix SUFFIX]
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    return logging.getLogger(__name__)

def load_results(scale='small'):
    """Load saved training results and model information"""
    logger = setup_logging()
    
    # Paths to saved data
    analysis_path = f"results/{scale}/analysis/comparative_study_complete.pt"
    pytorch_dir = f"results/{scale}/pytorch"
    
    if not os.path.exists(analysis_path):
        raise FileNotFoundError(f"Analysis results not found at {analysis_path}")
    
    if not os.path.exists(pytorch_dir):
        raise FileNotFoundError(f"PyTorch models directory not found at {pytorch_dir}")
    
    # Load main results
    logger.info(f"📊 Loading results from {analysis_path}")
    data = torch.load(analysis_path, map_location='cpu', weights_only=False)
    
    results = data['results']
    training_times = data['training_times'] 
    config = data['config']
    
    # Load model parameter counts from saved models
    model_params = {}
    model_files = {
        'Pointer+RL': 'model_pointerplusrl.pt',
        'GT-Greedy': 'model_gt-greedy.pt', 
        'GT+RL': 'model_gtplusrl.pt',
        'DGT+RL': 'model_dgtplusrl.pt',
        'GAT+RL': 'model_gatplusrl.pt',
        'GAT+RL (legacy)': 'model_gatplusrl_(legacy).pt'
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(pytorch_dir, filename)
        if os.path.exists(model_path):
            try:
                model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                state_dict = model_data.get('model_state_dict', model_data)
                param_count = sum(tensor.numel() for tensor in state_dict.values() if hasattr(tensor, 'numel'))
                model_params[model_name] = param_count
                logger.info(f"   📋 {model_name}: {param_count:,} parameters")
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to load {model_name} parameters: {e}")
                model_params[model_name] = 0
        else:
            logger.warning(f"   ⚠️ Model file not found: {model_path}")
            model_params[model_name] = 0
    
    # Filter out models with no results or parameters
    available_models = []
    for model_name in results.keys():
        if model_name in model_params and model_params[model_name] > 0:
            available_models.append(model_name)
        else:
            logger.warning(f"   ⚠️ Skipping {model_name} (missing data)")
    
    # Filter results and parameters to only include available models
    filtered_results = {name: results[name] for name in available_models}
    filtered_params = {name: model_params[name] for name in available_models}
    filtered_times = {name: training_times[name] for name in available_models}
    
    logger.info(f"✅ Loaded data for {len(available_models)} models: {list(available_models)}")
    
    return filtered_results, filtered_times, filtered_params, config

def generate_cvrp_instance(num_customers, capacity, coord_range, demand_range, seed=None):
    """Generate CVRP instance (same as training script)"""
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coordinates: random integers 0 to coord_range, then divide by coord_range for normalization to [0,1]
    coords = np.zeros((num_customers + 1, 2), dtype=np.float64)
    for i in range(num_customers + 1):
        coords[i] = np.random.randint(0, coord_range + 1, size=2) / coord_range
    
    # Generate integer demands from demand_range - ensure they are integers
    demands = np.zeros(num_customers + 1, dtype=np.int32)
    for i in range(1, num_customers + 1):  # Skip depot (index 0)
        demands[i] = np.random.randint(demand_range[0], demand_range[1] + 1)
    
    # Compute distance matrix
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    return {
        'coords': coords,
        'demands': demands.astype(np.int32),  # Ensure demands are integers
        'distances': distances,
        'capacity': int(capacity)  # Ensure capacity is integer
    }

def compute_naive_baseline_cost(instance):
    """Compute cost of naive solution: depot->node->depot for each customer (same as training script)"""
    distances = instance['distances']
    n_customers = len(instance['coords']) - 1  # excluding depot
    naive_cost = 0.0
    
    for customer_idx in range(1, n_customers + 1):  # customers are indexed 1 to n
        naive_cost += distances[0, customer_idx] * 2  # depot->customer->depot
    
    return naive_cost

def compute_naive_baseline_cost_per_customer(config):
    """Compute ACTUAL naive baseline cost per customer using the same method as training"""
    import numpy as np
    from tqdm import tqdm
    
    logger = setup_logging()
    logger.info("📊 Computing actual naive baseline from generated instances...")
    
    # Use the same parameters as training
    num_customers = config.get('num_customers', 20)
    capacity = config.get('capacity', 20) 
    coord_range = config.get('coord_range', 100)
    demand_range = config.get('demand_range', [1, 10])
    num_instances = config.get('num_instances')  # Use ALL instances for accuracy
    
    # Generate the same instances as training (using same seeds)
    naive_costs = []
    for i in range(num_instances):
        instance = generate_cvrp_instance(
            num_customers, capacity, coord_range, demand_range, seed=i
        )
        naive_cost = compute_naive_baseline_cost(instance)
        naive_costs.append(naive_cost)
    
    naive_avg_cost = np.mean(naive_costs)
    naive_normalized = naive_avg_cost / num_customers
    
    logger.info(f"   📍 Computed naive baseline: {naive_avg_cost:.3f} ({naive_normalized:.3f}/cust) from {num_instances} instances")
    
    return naive_normalized

def create_comparison_plots(results, training_times, model_params, config, scale='small', suffix=''):
    """Create comprehensive comparison plots with normalized costs (per customer)"""
    logger = setup_logging()
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # Get naive baseline estimate
    naive_normalized = compute_naive_baseline_cost_per_customer(config)
    
    # Build consistent color map for all figures (lines and bars)
    model_names = list(results.keys())
    palette = sns.color_palette("tab10", n_colors=len(model_names))
    color_map = {name: palette[i] for i, name in enumerate(model_names)}

    logger.info(f"🎨 Creating plots for {len(model_names)} models")
    logger.info(f"   Models: {model_names}")

    # 1. Training Loss Comparison (standardized REINFORCE loss for all models)
    plt.subplot(2, 4, 1)
    for model_name, result in results.items():
        train_losses = result.get('train_losses', [])
        if train_losses and not all(np.isnan(train_losses)):
            plt.plot(train_losses, label=model_name, linewidth=2, marker='o', markersize=3, color=color_map[model_name])
    plt.title('Training Loss Evolution\\n(Standardized REINFORCE)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('REINFORCE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Training Cost Comparison (NORMALIZED)
    plt.subplot(2, 4, 2)
    for model_name, result in results.items():
        train_costs = result.get('train_costs', [])
        if train_costs:
            # Normalize training costs by dividing by number of customers
            normalized_train_costs = [cost / config['num_customers'] for cost in train_costs]
            plt.plot(normalized_train_costs, label=model_name, linewidth=2, marker='s', markersize=3, color=color_map[model_name])
    plt.title('Training Cost Evolution (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Average Cost per Customer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Validation Cost vs Naive (NORMALIZED)
    plt.subplot(2, 4, 3)
    # Plot naive baseline as background reference line
    num_epochs = config.get('num_epochs', 25)
    val_epochs_full = list(range(0, num_epochs, 3))
    if len(val_epochs_full) == 0:
        val_epochs_full = [0]
    plt.axhline(y=naive_normalized, color='lightgray', linewidth=3, linestyle='--', label='Naive Baseline')
    
    for model_name, result in results.items():
        val_costs = result.get('val_costs', [])
        if val_costs:
            val_epochs = list(range(0, num_epochs, 3))[:len(val_costs)]
            normalized_val_costs = [cost / config['num_customers'] for cost in val_costs][:len(val_epochs)]
            plt.plot(val_epochs, normalized_val_costs, 'o-', label=model_name, linewidth=2, markersize=5, color=color_map[model_name])
    plt.title('Validation Cost vs Naive (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Average Cost per Customer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Final Performance Bar Chart with Naive Baseline (NORMALIZED)
    plt.subplot(2, 4, 4)
    # Normalize final costs by dividing by number of customers
    final_costs_normalized = [results[name]['final_val_cost'] / config['num_customers'] for name in model_names]
    # Colors consistent with lines
    colors = [color_map[name] for name in model_names]
    
    # Add naive baseline to the comparison (normalized)
    all_names = model_names + ['Naive Baseline']
    all_costs_normalized = final_costs_normalized + [naive_normalized]
    all_colors = colors + [(0.8, 0.2, 0.2)]  # red-like for naive
    
    bars = plt.bar(range(len(all_names)), all_costs_normalized, color=all_colors, alpha=0.8)
    plt.title('Final Performance vs Naive Baseline (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Approach')
    plt.ylabel('Average Cost per Customer')
    plt.xticks(range(len(all_names)), [name.replace(' ', '\\n') for name in all_names], rotation=45, ha='right')
    
    # Add value labels on bars (normalized)
    for bar, cost in zip(bars, all_costs_normalized):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{cost:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Training Time Comparison
    plt.subplot(2, 4, 5)
    times = [training_times[name] for name in model_names]
    bars = plt.bar(range(len(model_names)), times, color=[color_map[n] for n in model_names], alpha=0.8)
    plt.title('Training Time', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(model_names)), [name.replace(' ', '\\n') for name in model_names], rotation=45, ha='right')
    
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Model Complexity (Parameters)
    plt.subplot(2, 4, 6)
    param_counts = [model_params[name] for name in model_names]
    bars = plt.bar(range(len(model_names)), param_counts, color=[color_map[n] for n in model_names], alpha=0.8)
    plt.title('Model Complexity', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Parameters')
    plt.xticks(range(len(model_names)), [name.replace(' ', '\\n') for name in model_names], rotation=45, ha='right')
    
    for bar, params in zip(bars, param_counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(param_counts)*0.01,
                f'{params:,}', ha='center', va='bottom', fontweight='bold', fontsize=8, rotation=0)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 7. Learning Efficiency (Cost Improvement)
    plt.subplot(2, 4, 7)
    improvements = []
    for model_name in model_names:
        result = results[model_name]
        train_costs = result.get('train_costs', [])
        if len(train_costs) >= 2:
            initial_cost = train_costs[0]
            final_cost = train_costs[-1]
            if initial_cost > 0:
                improvement = ((initial_cost - final_cost) / initial_cost) * 100
            else:
                improvement = 0
        else:
            improvement = 0
        improvements.append(improvement)
    
    bars = plt.bar(range(len(model_names)), improvements, color=[color_map[n] for n in model_names], alpha=0.8)
    plt.title('Learning Efficiency', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Cost Improvement (%)')
    plt.xticks(range(len(model_names)), [name.replace(' ', '\\n') for name in model_names], rotation=45, ha='right')
    
    for bar, imp in zip(bars, improvements):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(max(improvements), 1)*0.02,
                f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 8. Performance vs Complexity Scatter (NORMALIZED)
    plt.subplot(2, 4, 8)
    for i, model_name in enumerate(model_names):
        plt.scatter(param_counts[i], final_costs_normalized[i], 
                   s=100, color=color_map[model_name], alpha=0.8, label=model_name)
        plt.annotate(model_name.replace(' ', '\\n'), 
                    (param_counts[i], final_costs_normalized[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Performance vs Complexity (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Parameters')
    plt.ylabel('Validation Cost per Customer')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure plots directory exists
    plots_dir = f"results/{scale}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save plot
    output_filename = f"comparative_study_results{suffix}.png"
    output_path = os.path.join(plots_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Comparison plots saved to {output_path}")
    
    # Create performance summary
    create_performance_summary(results, training_times, model_params, config, naive_normalized, scale, suffix)
    
    # Don't show the plot in non-interactive mode
    # plt.show()

def create_performance_summary(results, training_times, model_params, config, naive_baseline_per_customer, scale, suffix=''):
    """Create a detailed performance summary table and save to CSV"""
    logger = setup_logging()
    
    data = []
    model_names = list(results.keys())
    
    for model_name in model_names:
        result = results[model_name]
        final_val_cost_per_customer = result['final_val_cost'] / config['num_customers']
        improvement_vs_naive = ((naive_baseline_per_customer - final_val_cost_per_customer) / naive_baseline_per_customer) * 100
        
        # Calculate learning efficiency
        train_costs = result.get('train_costs', [])
        if len(train_costs) >= 2:
            initial_cost = train_costs[0]
            final_cost = train_costs[-1]
            learning_efficiency = ((initial_cost - final_cost) / initial_cost) * 100 if initial_cost > 0 else 0
        else:
            learning_efficiency = 0
        
        data.append({
            'Model': model_name,
            'Parameters': model_params[model_name],
            'Training Time (s)': training_times[model_name],
            'Final Train Cost': result['train_costs'][-1] if result.get('train_costs') else 0,
            'Final Val Cost': result['final_val_cost'],
            'Final Val Cost/Customer': final_val_cost_per_customer,
            'Improvement vs Naive (%)': improvement_vs_naive,
            'Learning Efficiency (%)': learning_efficiency,
            'Best Val Cost': min(result.get('val_costs', [result['final_val_cost']])),
            'Val Cost Std': np.std(result.get('val_costs', [result['final_val_cost']]))
        })
    
    df = pd.DataFrame(data)
    
    # Sort by performance (lowest cost per customer first)
    df = df.sort_values('Final Val Cost/Customer')
    
    # Save to CSV
    csv_dir = f"results/{scale}/csv"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f'{csv_dir}/comparative_results{suffix}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"📋 Performance summary saved to {csv_path}")
    
    # Print formatted table to console
    logger.info("\\n📊 DETAILED PERFORMANCE COMPARISON")
    logger.info("=" * 100)
    
    # Create nicely formatted table
    col_widths = {
        'Model': max(len('Model'), max(len(model) for model in df['Model'])),
        'Parameters': 11,
        'Time (s)': 9,
        'Val/Cust': 9,
        'Improv %': 9
    }
    
    # Header
    header = f"| {'Model':<{col_widths['Model']}} | {'Parameters':>{col_widths['Parameters']}} | {'Time (s)':>{col_widths['Time (s)']}} | {'Val/Cust':>{col_widths['Val/Cust']}} | {'Improv %':>{col_widths['Improv %']}} |"
    separator = "|" + "-" * (col_widths['Model'] + 2) + "|" + "-" * (col_widths['Parameters'] + 2) + "|" + "-" * (col_widths['Time (s)'] + 2) + "|" + "-" * (col_widths['Val/Cust'] + 2) + "|" + "-" * (col_widths['Improv %'] + 2) + "|"
    
    print(header)
    print(separator)
    
    for _, row in df.iterrows():
        model_name = row['Model']
        params = f"{row['Parameters']:,}"
        time_s = f"{row['Training Time (s)']:.1f}s"
        val_per_cust = f"{row['Final Val Cost/Customer']:.4f}"
        improvement = f"{row['Improvement vs Naive (%)']:+.1f}%"
        
        row_str = f"| {model_name:<{col_widths['Model']}} | {params:>{col_widths['Parameters']}} | {time_s:>{col_widths['Time (s)']}} | {val_per_cust:>{col_widths['Val/Cust']}} | {improvement:>{col_widths['Improv %']}} |"
        print(row_str)
    
    print("=" * 100)
    print(f"🏆 Best model: {df.iloc[0]['Model']} ({df.iloc[0]['Final Val Cost/Customer']:.4f}/customer, {df.iloc[0]['Improvement vs Naive (%)']:+.1f}% vs naive)")
    
def load_config(config_path):
    """Load configuration from YAML file"""
    import yaml
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def determine_scale_from_config(config):
    """Determine scale from config file"""
    # Extract nested config values if they exist
    if 'problem' in config:
        num_customers = config['problem']['num_customers']
    else:
        num_customers = config.get('num_customers', 15)
    
    # Determine scale from num_customers
    if num_customers <= 20:
        return 'small'
    elif num_customers <= 50:
        return 'medium'
    else:
        return 'production'

def parse_args():
    parser = argparse.ArgumentParser(description='Generate comparative study plots from saved results')
    parser.add_argument('--config', type=str, default='configs/small.yaml',
                       help='Path to YAML configuration file (determines scale from num_customers)')
    parser.add_argument('--suffix', type=str, default='', 
                       help='Suffix to add to output filename')
    return parser.parse_args()

def main():
    """Main function to generate comparative plots"""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("🚀 Starting Comparative Plot Generation")
    logger.info(f"📂 Loading config from: {args.config}")
    
    try:
        # Load config and determine scale
        config = load_config(args.config)
        scale = determine_scale_from_config(config)
        logger.info(f"📏 Determined scale: {scale} (from {config.get('problem', {}).get('num_customers', config.get('num_customers', 'unknown'))} customers)")
        
        # Load saved results
        results, training_times, model_params, loaded_config = load_results(scale)
        
        # Create plots
        suffix = f"_{args.suffix}" if args.suffix else ""
        create_comparison_plots(results, training_times, model_params, loaded_config, scale, suffix)
        
        logger.info("✅ Comparative plots generated successfully!")
        logger.info(f"📊 Output: results/{scale}/plots/comparative_study_results{suffix}.png")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate plots: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
