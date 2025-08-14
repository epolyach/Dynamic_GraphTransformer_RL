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
    python make_comparative_plot.py [--config CONFIG_PATH] [--suffix SUFFIX] [--exact NUM_SAMPLES]
    
    --exact NUM_SAMPLES: Compute exact baseline by solving NUM_SAMPLES random instances
                         with exact algorithms (can be time-consuming)
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

def load_results(base_dir):
    """Load saved training results and model information from working_dir_path.
    If some models are not in the main analysis file, attempt to recover them from
    individual model_*.pt files in the pytorch directory so we can plot all available models.
    """
    logger = setup_logging()
    
    # Paths to saved data
    # Try both potential analysis filenames
    analysis_path_enhanced = os.path.join(base_dir, 'analysis', 'enhanced_comparative_study.pt')
    analysis_path_complete = os.path.join(base_dir, 'analysis', 'comparative_study_complete.pt')
    
    if os.path.exists(analysis_path_enhanced):
        analysis_path = analysis_path_enhanced
    elif os.path.exists(analysis_path_complete):
        analysis_path = analysis_path_complete
    else:
        raise FileNotFoundError(f"Analysis results not found at {analysis_path_enhanced} or {analysis_path_complete}")
    pytorch_dir = os.path.join(base_dir, 'pytorch')
    
    if not os.path.exists(analysis_path):
        raise FileNotFoundError(f"Analysis results not found at {analysis_path}")
    
    if not os.path.exists(pytorch_dir):
        raise FileNotFoundError(f"PyTorch models directory not found at {pytorch_dir}")
    
    # Load main results
    logger.info(f"📊 Loading results from {analysis_path}")
    data = torch.load(analysis_path, map_location='cpu', weights_only=False)
    
    results = dict(data.get('results', {}))
    
    # Handle both old and new formats for training times
    if 'training_times' in data:
        # Old format: direct training_times dict
        training_times = dict(data.get('training_times', {}))
    else:
        # New enhanced format: training_time stored per model
        training_times = {name: result.get('training_time', 0.0) for name, result in results.items()}
    
    config = data.get('config', {})
    
    # Scan pytorch_dir for any saved model files to augment results
    logger.info(f"📁 Scanning models in {pytorch_dir}")
    model_params = {}
    recovered = 0
    for fname in os.listdir(pytorch_dir):
        if not fname.startswith('model_') or not fname.endswith('.pt'):
            continue
        fpath = os.path.join(pytorch_dir, fname)
        try:
            m = torch.load(fpath, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load model file {fname}: {e}")
            continue
        # Determine model name
        mname = m.get('model_name')
        if not mname:
            # Derive from filename: model_<slug>.pt -> slug
            mname = fname[len('model_'):-len('.pt')]
            # Map common slugs back to display names if needed
            mname = mname.replace('plus', '+').replace('_', ' ')
        # Parameter count
        state_dict = m.get('model_state_dict', m)
        try:
            pcount = sum(t.numel() for t in state_dict.values() if hasattr(t, 'numel'))
        except Exception:
            pcount = 0
        model_params[mname] = pcount
        logger.info(f"   📋 {mname}: {pcount:,} parameters from {fname}")
        # Recover results/training_time when missing in main analysis
        if mname not in results and 'results' in m:
            results[mname] = m['results']
            recovered += 1
        if mname not in training_times and 'training_time' in m:
            training_times[mname] = m['training_time']
    if recovered:
        logger.info(f"🔎 Recovered {recovered} model(s) from saved model files to include in plots")
    
    # Keep only models for which we have parameters (i.e., present in pytorch_dir)
    available_models = [name for name in results.keys() if model_params.get(name, 0) > 0]
    missing = [name for name in results.keys() if name not in available_models]
    for name in missing:
        logger.warning(f"   ⚠️ Skipping {name} (no model file / parameters found)")
    
    filtered_results = {name: results[name] for name in available_models}
    filtered_params = {name: model_params[name] for name in available_models}
    filtered_times = {name: training_times.get(name, 0.0) for name in available_models}
    
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

def compute_naive_baseline_cost(instance, depot_penalty_per_visit: float = 0.0):
    """Compute cost of naive solution: depot->customer->depot for each customer.
    Includes optional depot penalty per internal return when configured.
    """
    distances = instance['distances']
    n_customers = len(instance['coords']) - 1  # excluding depot
    naive_cost = 0.0
    
    for customer_idx in range(1, n_customers + 1):  # customers are indexed 1 to n
        naive_cost += distances[0, customer_idx] * 2  # depot->customer->depot
    
    # Internal depot visits in naive route: n_customers - 1 (exclude first start and final end)
    if depot_penalty_per_visit and depot_penalty_per_visit != 0.0 and n_customers > 0:
        naive_cost += depot_penalty_per_visit * (n_customers - 1)
    
    return naive_cost

def compute_naive_baseline_cost_per_customer(config):
    """Compute ACTUAL naive baseline cost per customer using the same method as training"""
    import numpy as np
    from tqdm import tqdm
    
    logger = setup_logging()
    logger.info("📊 Computing actual naive baseline from generated instances...")
    
    # Use the same parameters as training
    if 'problem' in config:
        num_customers = config['problem']['num_customers']
        capacity = config['problem']['vehicle_capacity']
        coord_range = config['problem']['coord_range']
        demand_range = config['problem']['demand_range']
    else:
        num_customers = config['num_customers']
        capacity = config['capacity']
        coord_range = config['coord_range']
        demand_range = config['demand_range']
    
    if 'training' in config:
        num_instances = config['training']['num_instances']  # Use ALL instances for accuracy
    else:
        num_instances = config['num_instances']
    
    depot_penalty = 0.0
    if isinstance(config, dict):
        depot_penalty = config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
    
    # Generate the same instances as training (using same seeds)
    naive_costs = []
    for i in range(num_instances):
        instance = generate_cvrp_instance(
            num_customers, capacity, coord_range, demand_range, seed=i
        )
        naive_cost = compute_naive_baseline_cost(instance, depot_penalty)
        naive_costs.append(naive_cost)
    
    naive_avg_cost = np.mean(naive_costs)
    naive_normalized = naive_avg_cost / num_customers
    
    if depot_penalty:
        logger.info(f"   📍 Computed naive baseline (with depot penalty): {naive_avg_cost:.3f} ({naive_normalized:.3f}/cust) from {num_instances} instances")
    else:
        logger.info(f"   📍 Computed naive baseline: {naive_avg_cost:.3f} ({naive_normalized:.3f}/cust) from {num_instances} instances")
    
    return naive_normalized

def compute_exact_baseline_cost_per_customer(config, num_samples=100):
    """Compute exact baseline cost per customer by solving a random sample of instances"""
    logger = setup_logging()
    logger.info(f"🎯 Computing exact baseline from {num_samples} random instances...")
    
    try:
        from exact_solver import ExactCVRPSolver
    except ImportError:
        logger.error("❌ exact_solver.py not found. Please ensure it's in the project directory.")
        return None
    
    # Use the same parameters as training
    if 'problem' in config:
        num_customers = config['problem']['num_customers']
        capacity = config['problem']['vehicle_capacity']
        coord_range = config['problem']['coord_range']
        demand_range = config['problem']['demand_range']
    else:
        num_customers = config['num_customers']
        capacity = config['capacity']
        coord_range = config['coord_range']
        demand_range = config['demand_range']
    
    # Initialize exact solver
    solver = ExactCVRPSolver(time_limit=60.0, verbose=False)  # 1 min per instance
    
    exact_costs = []
    solve_times = []
    failed_instances = 0
    
    logger.info(f"   📋 Problem size: {num_customers} customers, capacity={capacity}")
    logger.info(f"   🔍 Using random seeds {num_samples} to {num_samples + num_samples - 1}")
    
    from tqdm import tqdm
    
    # Use different random seeds than training to get independent samples
    for i in tqdm(range(num_samples), desc="Solving exact instances"):
        try:
            # Use higher seed range to avoid overlap with training instances
            seed = 10000 + i  
            instance = generate_cvrp_instance(
                num_customers, capacity, coord_range, demand_range, seed=seed
            )
            
            # Solve exactly
            solution = solver.solve(instance)
            
            if solution.cost < float('inf'):
                exact_costs.append(solution.cost)
                solve_times.append(solution.solve_time)
            else:
                failed_instances += 1
                logger.warning(f"   ⚠️ Failed to solve instance {i}")
                
        except Exception as e:
            failed_instances += 1
            logger.warning(f"   ⚠️ Error solving instance {i}: {e}")
    
    if not exact_costs:
        logger.error("❌ No instances solved successfully")
        return None
    
    exact_avg_cost = np.mean(exact_costs)
    exact_normalized = exact_avg_cost / num_customers
    exact_std = np.std(exact_costs) / num_customers
    avg_solve_time = np.mean(solve_times)
    
    success_rate = (num_samples - failed_instances) / num_samples
    
    logger.info(f"   ✅ Exact baseline: {exact_avg_cost:.3f} ({exact_normalized:.3f}±{exact_std:.3f}/cust)")
    logger.info(f"   📊 Success rate: {success_rate:.1%}, avg solve time: {avg_solve_time:.1f}s")
    logger.info(f"   🔬 Solved {len(exact_costs)}/{num_samples} instances successfully")
    
    return {
        'cost_per_customer': exact_normalized,
        'std_per_customer': exact_std,
        'success_rate': success_rate,
        'avg_solve_time': avg_solve_time,
        'num_solved': len(exact_costs),
        'num_requested': num_samples
    }

def create_comparison_plots(results, training_times, model_params, config, scale, suffix='', exact_baseline_stats=None):
    """Create comprehensive comparison plots with normalized costs (per customer)
    Now reads per-epoch series (loss, train_cost, val_cost) directly from CSV history files.
    """
    logger = setup_logging()

    def _safe_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend()

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # Get naive baseline estimate
    naive_normalized = compute_naive_baseline_cost_per_customer(config)
    
    # Check if we have exact baseline
    exact_normalized = None
    if exact_baseline_stats:
        exact_normalized = exact_baseline_stats['cost_per_customer']
        logger.info(f"📍 Using exact baseline: {exact_normalized:.3f}/customer")
    
    # Build consistent color map for all figures (lines and bars)
    model_names = list(results.keys())
    palette = sns.color_palette("tab10", n_colors=len(model_names))
    color_map = {name: palette[i] for i, name in enumerate(model_names)}

    logger.info(f"🎨 Creating plots for {len(model_names)} models")
    logger.info(f"   Models: {model_names}")

    # CSV dir
    csv_dir = os.path.join(config.get('working_dir_path', 'results'), 'csv') if isinstance(config, dict) else os.path.join('results', scale, 'csv')
    
    # Map display names to CSV keys
    name_to_key = {
        'Pointer+RL': 'pointer_rl',
        'GT+RL': 'gt_rl',
        'DGT+RL': 'dgt_rl',
        'Simplified-DGT+RL': 'simplified_dgt_rl',
        'GAT+RL': 'gat_rl',
        'GT-Greedy': 'gt_greedy',
        'GAT+RL (legacy)': 'gat_rl_legacy',
    }

    def load_csv_series_for_model(model_name):
        key = name_to_key.get(model_name)
        if not key:
            return None
        fpath = os.path.join(csv_dir, f"history_{key}.csv")
        if not os.path.exists(fpath):
            return None
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to read CSV for {model_name}: {e}")
            return None
        # Ensure required columns exist
        for col in ['epoch', 'train_loss', 'train_cost', 'val_cost']:
            if col not in df.columns:
                logger.warning(f"   ⚠️ CSV for {model_name} missing column '{col}'")
        # Build series using exact CSV rows
        epochs = df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df)))
        train_loss = df['train_loss'].tolist() if 'train_loss' in df.columns else []
        train_cost = df['train_cost'].tolist() if 'train_cost' in df.columns else []
        val_mask = df['val_cost'].notna() if 'val_cost' in df.columns else pd.Series([False]*len(df))
        val_epochs = df.loc[val_mask, 'epoch'].tolist() if 'epoch' in df.columns else [i for i, m in enumerate(val_mask) if m]
        val_costs = df.loc[val_mask, 'val_cost'].tolist() if 'val_cost' in df.columns else []
        return {
            'epochs': epochs,
            'train_loss': train_loss,
            'train_cost': train_cost,
            'val_epochs': val_epochs,
            'val_costs': val_costs,
        }

    csv_series = {name: load_csv_series_for_model(name) for name in model_names}

# 1. Training Loss Comparison (standardized REINFORCE loss for all models)
    ax1 = plt.subplot(2, 4, 1)
    for model_name in model_names:
        series = csv_series.get(model_name)
        if series and series['train_loss']:
            xs = series['epochs'][:len(series['train_loss'])]
            ys = [v for v in series['train_loss'] if pd.notna(v)]
            xs = [series['epochs'][i] for i, v in enumerate(series['train_loss']) if pd.notna(v)]
            if ys:
                ax1.plot(xs, ys, label=model_name, linewidth=2, marker='o', markersize=3, color=color_map[model_name])
    ax1.set_title('Training Loss Evolution\n(Standardized REINFORCE)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('REINFORCE Loss')
    _safe_legend(ax1)
    ax1.grid(True, alpha=0.3)
    
# 2. Training Cost Comparison (NORMALIZED)
    ax2 = plt.subplot(2, 4, 2)
    for model_name in model_names:
        series = csv_series.get(model_name)
        if series and series['train_cost']:
            xs = [series['epochs'][i] for i, v in enumerate(series['train_cost']) if pd.notna(v)]
            ys = [v for v in series['train_cost'] if pd.notna(v)]
            if ys:
                ax2.plot(xs, ys, label=model_name, linewidth=2, marker='s', markersize=3, color=color_map[model_name])
    ax2.set_title('Training Cost Evolution (Per Customer)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Cost per Customer')
    _safe_legend(ax2)
    ax2.grid(True, alpha=0.3)
    
# 3. Validation Cost vs Naive (NORMALIZED)
    ax3 = plt.subplot(2, 4, 3)
    # Plot naive baseline as background reference line
    num_epochs = config['training']['num_epochs']
    # Annotate if depot penalty is active
    penalty_active = isinstance(config, dict) and config.get('cost', {}).get('depot_penalty_per_visit', 0.0)
    baseline_label = 'Naive Baseline (with penalty)' if penalty_active else 'Naive Baseline'
    ax3.axhline(y=naive_normalized, color='lightgray', linewidth=3, linestyle='--', label=baseline_label)
    
    # Add GT-Greedy baseline (greedy attention without RL training)
    if 'GT-Greedy' in results:
        result = results['GT-Greedy']
        if 'history' in result:
            gg_val = result['history'].get('final_val_cost', None)
        else:
            gg_val = result.get('final_val_cost', None)
        
        if gg_val is not None:
            ax3.axhline(y=gg_val, color='orange', linewidth=3, linestyle='-.', 
                       label='GT-Greedy Baseline (no RL)', alpha=0.8)
    
    # Add exact baseline if available
    if exact_normalized is not None:
        ax3.axhline(y=exact_normalized, color='red', linewidth=3, linestyle=':', 
                   label=f'Exact Baseline ({exact_baseline_stats["num_solved"]} samples)', alpha=0.8)

    # Plot RL model validation series only
    for model_name in model_names:
        if model_name == 'GT-Greedy':
            continue  # GT-Greedy now shown as baseline
        series = csv_series.get(model_name)
        if series and series['val_costs']:
            xs = series['val_epochs']
            ys = series['val_costs']
            ax3.plot(xs, ys, 'o-', label=model_name, linewidth=2, markersize=5, color=color_map[model_name])

    ax3.set_title('Validation Cost vs Naive (Per Customer)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Cost per Customer')
    _safe_legend(ax3)
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Performance Bar Chart with Naive Baseline (NORMALIZED)
    plt.subplot(2, 4, 4)
    # Final costs are already per customer in saved results - handle both formats
    final_costs_normalized = []
    for name in model_names:
        result = results[name]
        if 'history' in result:
            # New enhanced format
            final_cost = result['history'].get('final_val_cost', 0.0)
        else:
            # Old format
            final_cost = result.get('final_val_cost', 0.0)
        final_costs_normalized.append(final_cost)
    
    # Colors consistent with lines
    colors = [color_map[name] for name in model_names]
    
    # Add baselines to the comparison (normalized)
    baseline_names = ['Naive Baseline']
    baseline_costs = [naive_normalized]
    baseline_colors = [(0.8, 0.2, 0.2)]  # red-like for naive
    
    if exact_normalized is not None:
        baseline_names.append(f'Exact Baseline\n({exact_baseline_stats["num_solved"]} samples)')
        baseline_costs.append(exact_normalized)
        baseline_colors.append((0.6, 0.0, 0.0))  # dark red for exact
    
    all_names = model_names + baseline_names
    all_costs_normalized = final_costs_normalized + baseline_costs
    all_colors = colors + baseline_colors
    
    bars = plt.bar(range(len(all_names)), all_costs_normalized, color=all_colors, alpha=0.8)
    title_suffix = ' vs Baselines' if exact_normalized is not None else ' vs Naive Baseline'
    plt.title(f'Final Performance{title_suffix} (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Approach')
    plt.ylabel('Average Cost per Customer')
    plt.xticks(range(len(all_names)), [name.replace(' ', '\\n') for name in all_names], rotation=45, ha='right')
    
    # Add value labels on bars (normalized)
    for bar, cost in zip(bars, all_costs_normalized):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{cost:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Training Time Comparison (exclude GT-Greedy which has no training loop)
    plt.subplot(2, 4, 5)
    time_models = [n for n in model_names if n != 'GT-Greedy']
    times = [training_times[name] for name in time_models]
    bars = plt.bar(range(len(time_models)), times, color=[color_map[n] for n in time_models], alpha=0.8)
    plt.title('Training Time', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(time_models)), [name.replace(' ', '\\n') for name in time_models], rotation=45, ha='right')
    
    if times:
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
    
    # 7. Learning Efficiency (Cost Improvement) - exclude GT-Greedy
    plt.subplot(2, 4, 7)
    eff_models = [n for n in model_names if n != 'GT-Greedy']
    improvements = []
    for model_name in eff_models:
        series = csv_series.get(model_name)
        imp = 0.0
        if series:
            # Prefer training cost series; fallback to validation series if training is unavailable (e.g., legacy)
            train_vals = [v for v in (series.get('train_cost') or []) if pd.notna(v)]
            if len(train_vals) >= 2 and train_vals[0] > 0:
                imp = ((train_vals[0] - train_vals[-1]) / train_vals[0]) * 100
            else:
                val_vals = [v for v in (series.get('val_costs') or []) if pd.notna(v)]
                if len(val_vals) >= 2 and val_vals[0] > 0:
                    imp = ((val_vals[0] - val_vals[-1]) / val_vals[0]) * 100
        improvements.append(imp)
    
    bars = plt.bar(range(len(eff_models)), improvements, color=[color_map[n] for n in eff_models], alpha=0.8)
    plt.title('Learning Efficiency', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Cost Improvement (%)')
    plt.xticks(range(len(eff_models)), [name.replace(' ', '\n') for name in eff_models], rotation=45, ha='right')
    
    if improvements:
        ymax = max(max(improvements), 1)
        for bar, imp in zip(bars, improvements):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + ymax*0.02,
                    f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 8. Performance vs Complexity Scatter (NORMALIZED)
    plt.subplot(2, 4, 8)
    # Use the same final costs we computed above
    # final_costs_normalized is already computed correctly above
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
    plots_dir = f"results/{scale}/plots" if os.path.isabs(scale) else os.path.join(f"results/{scale}", 'plots')
    # If scale is actually a base_dir label, recompute plots_dir using working_dir_path when available
    if isinstance(config, dict) and 'working_dir_path' in config:
        plots_dir = os.path.join(config['working_dir_path'], 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save plot
    output_filename = f"comparative_study_results{suffix}.png"
    output_path = os.path.join(plots_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Comparison plots saved to {output_path}")
    
    # Create performance summary
    create_performance_summary(results, training_times, model_params, config, naive_normalized, scale, suffix, exact_baseline_stats)
    
    # Don't show the plot in non-interactive mode
    # plt.show()

def create_performance_summary(results, training_times, model_params, config, naive_baseline_per_customer, scale, suffix='', exact_baseline_stats=None):
    """Create a detailed performance summary table and save to CSV
    All costs are reported per customer for consistency across models (including legacy).
    """
    logger = setup_logging()
    
    data = []
    model_names = list(results.keys())
    num_customers = config['problem']['num_customers']
    
    for model_name in model_names:
        result = results[model_name]
        
        # Handle both old and new enhanced formats
        if 'history' in result:
            # New enhanced format: data is in 'history' subdictionary
            history = result['history']
            final_val_cost_per_customer = history.get('final_val_cost', 0.0)
            train_costs = history.get('train_costs', [])
            val_costs = history.get('val_costs', [])
        else:
            # Old format: data is directly in result
            final_val_cost_per_customer = result.get('final_val_cost', 0.0)
            train_costs = result.get('train_costs', [])
            val_costs = result.get('val_costs', [])
        
        improvement_vs_naive = ((naive_baseline_per_customer - final_val_cost_per_customer) / naive_baseline_per_customer) * 100
        
        # Calculate improvement vs exact baseline if available
        improvement_vs_exact = None
        if exact_baseline_stats:
            exact_baseline = exact_baseline_stats['cost_per_customer']
            improvement_vs_exact = ((exact_baseline - final_val_cost_per_customer) / exact_baseline) * 100
        
        # Calculate learning efficiency (percent change unaffected by normalization)
        if len(train_costs) >= 2:
            initial_cost = train_costs[0]
            final_cost = train_costs[-1]
            learning_efficiency = ((initial_cost - final_cost) / initial_cost) * 100 if initial_cost > 0 else 0
        else:
            learning_efficiency = 0
        
        # Train and validation costs are already per customer in saved results
        final_train_cost_per_customer = train_costs[-1] if train_costs else 0.0
        best_val_cost_per_customer = min(val_costs) if val_costs else final_val_cost_per_customer
        val_cost_std_per_customer = np.std(val_costs) if val_costs else 0.0
        
        row_data = {
            'Model': model_name,
            'Parameters': model_params[model_name],
            'Training Time (s)': training_times[model_name],
            'Final Train Cost/Customer': final_train_cost_per_customer,
            'Final Val Cost/Customer': final_val_cost_per_customer,
            'Improvement vs Naive (%)': improvement_vs_naive,
            'Learning Efficiency (%)': learning_efficiency,
            'Best Val Cost/Customer': best_val_cost_per_customer,
            'Val Cost Std/Customer': val_cost_std_per_customer
        }
        
        if improvement_vs_exact is not None:
            row_data['Improvement vs Exact (%)'] = improvement_vs_exact
            
        data.append(row_data)
    
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
    logger.info("\n📊 DETAILED PERFORMANCE COMPARISON")
    if exact_baseline_stats:
        logger.info(f"📍 Baselines: Naive={naive_baseline_per_customer:.4f}/cust, Exact={exact_baseline_stats['cost_per_customer']:.4f}±{exact_baseline_stats['std_per_customer']:.4f}/cust")
    logger.info("=" * 120)
    
    # Create nicely formatted table
    has_exact = exact_baseline_stats is not None
    col_widths = {
        'Model': max(len('Model'), max(len(model) for model in df['Model'])),
        'Parameters': 11,
        'Time (s)': 9,
        'Val/Cust': 9,
        'vs Naive': 9,
        'vs Exact': 9 if has_exact else 0
    }
    
    # Header
    if has_exact:
        header = f"| {'Model':<{col_widths['Model']}} | {'Parameters':>{col_widths['Parameters']}} | {'Time (s)':>{col_widths['Time (s)']}} | {'Val/Cust':>{col_widths['Val/Cust']}} | {'vs Naive':>{col_widths['vs Naive']}} | {'vs Exact':>{col_widths['vs Exact']}} |"
        separator = "|" + "-" * (col_widths['Model'] + 2) + "|" + "-" * (col_widths['Parameters'] + 2) + "|" + "-" * (col_widths['Time (s)'] + 2) + "|" + "-" * (col_widths['Val/Cust'] + 2) + "|" + "-" * (col_widths['vs Naive'] + 2) + "|" + "-" * (col_widths['vs Exact'] + 2) + "|"
    else:
        header = f"| {'Model':<{col_widths['Model']}} | {'Parameters':>{col_widths['Parameters']}} | {'Time (s)':>{col_widths['Time (s)']}} | {'Val/Cust':>{col_widths['Val/Cust']}} | {'vs Naive':>{col_widths['vs Naive']}} |"
        separator = "|" + "-" * (col_widths['Model'] + 2) + "|" + "-" * (col_widths['Parameters'] + 2) + "|" + "-" * (col_widths['Time (s)'] + 2) + "|" + "-" * (col_widths['Val/Cust'] + 2) + "|" + "-" * (col_widths['vs Naive'] + 2) + "|"
    
    print(header)
    print(separator)
    
    for _, row in df.iterrows():
        model_name = row['Model']
        params = f"{row['Parameters']:,}"
        time_s = f"{row['Training Time (s)']:.1f}s"
        val_per_cust = f"{row['Final Val Cost/Customer']:.4f}"
        improvement_naive = f"{row['Improvement vs Naive (%)']:+.1f}%"
        
        if has_exact:
            improvement_exact = f"{row.get('Improvement vs Exact (%)', 0):+.1f}%" 
            row_str = f"| {model_name:<{col_widths['Model']}} | {params:>{col_widths['Parameters']}} | {time_s:>{col_widths['Time (s)']}} | {val_per_cust:>{col_widths['Val/Cust']}} | {improvement_naive:>{col_widths['vs Naive']}} | {improvement_exact:>{col_widths['vs Exact']}} |"
        else:
            row_str = f"| {model_name:<{col_widths['Model']}} | {params:>{col_widths['Parameters']}} | {time_s:>{col_widths['Time (s)']}} | {val_per_cust:>{col_widths['Val/Cust']}} | {improvement_naive:>{col_widths['vs Naive']}} |"
        print(row_str)
    
    print("=" * 120)
    best_model = df.iloc[0]
    if has_exact:
        print(f"🏆 Best model: {best_model['Model']} ({best_model['Final Val Cost/Customer']:.4f}/customer, {best_model['Improvement vs Naive (%)']:+.1f}% vs naive, {best_model.get('Improvement vs Exact (%)', 0):+.1f}% vs exact)")
    else:
        print(f"🏆 Best model: {best_model['Model']} ({best_model['Final Val Cost/Customer']:.4f}/customer, {best_model['Improvement vs Naive (%)']:+.1f}% vs naive)")
    
def _deep_merge_dict(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge_dict(a[k], v)
        else:
            a[k] = v
    return a

def load_config(config_path):
    """Unified config loader (shared)"""
    from src.utils.config import load_config as _shared_load
    return _shared_load(config_path)

def determine_scale_from_config_path(config_path):
    """Determine scale directly from config filename"""
    config_filename = Path(config_path).stem  # Get filename without extension
    if config_filename in ['small', 'medium', 'production']:
        return config_filename
    else:
        raise ValueError(f"Unknown config scale '{config_filename}'. Expected 'small', 'medium', or 'production'")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate comparative study plots from saved results')
    parser.add_argument('--config', type=str, default='configs/small.yaml',
                       help='Path to YAML configuration file (reads working_dir_path)')
    parser.add_argument('--suffix', type=str, default='', 
                       help='Suffix to add to output filename')
    parser.add_argument('--exact', type=int, default=0, metavar='NUM_SAMPLES',
                       help='Compute exact baseline by solving NUM_SAMPLES random instances with exact algorithms (0 = disabled)')
    return parser.parse_args()

def main():
    """Main function to generate comparative plots"""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("🚀 Starting Comparative Plot Generation")
    logger.info(f"📂 Loading config from: {args.config}")
    
    try:
        # Load config to get working_dir_path
        cfg = load_config(args.config)
        base_dir = str(Path(cfg.get('working_dir_path', 'results')).as_posix())
        logger.info(f"📁 Working directory: {base_dir}")
        
        # Load saved results
        results, training_times, model_params, loaded_config = load_results(base_dir)
        
        # Compute exact baseline if requested
        exact_baseline_stats = None
        if args.exact > 0:
            logger.info(f"🎯 Computing exact baseline with {args.exact} samples...")
            exact_baseline_stats = compute_exact_baseline_cost_per_customer(loaded_config, args.exact)
            if exact_baseline_stats is None:
                logger.warning("⚠️ Failed to compute exact baseline, proceeding without it")
        
        # Create plots
        suffix = f"_{args.suffix}" if args.suffix else ""
        # Derive a label (for file locations only) from working_dir_path leaf
        label = Path(base_dir).name
        create_comparison_plots(results, training_times, model_params, loaded_config, label, suffix, exact_baseline_stats)
        
        logger.info("✅ Comparative plots generated successfully!")
        logger.info(f"📊 Output: {base_dir}/plots/comparative_study_results{suffix}.png")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate plots: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
