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

# Add the project root to the Python path to allow importing 'src'
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '..', '..'))
sys.path.insert(0, project_root)
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
    
    # Process results to extract final_val_cost from tuple format if needed
    for name in available_models:
        result = filtered_results[name]
        if 'history' in result and 'val_costs' in result['history']:
            val_costs = result['history']['val_costs']
            # Check if val_costs are in tuple format (epoch, cost)
            if val_costs and isinstance(val_costs[0], tuple):
                # Extract just the costs from tuples
                val_costs_only = [cost for epoch, cost in val_costs]
                result['history']['val_costs_raw'] = val_costs  # Keep original for reference
                result['history']['val_costs'] = val_costs_only
                result['history']['val_epochs'] = [epoch for epoch, cost in val_costs]
                # Set final_val_cost from the last validation cost
                if val_costs_only:
                    result['history']['final_val_cost'] = val_costs_only[-1]
            elif val_costs:
                # Already in list format, set final_val_cost
                result['history']['final_val_cost'] = val_costs[-1]
        
        # Process train_costs similarly if in tuple format
        if 'history' in result and 'train_costs' in result['history']:
            train_costs = result['history']['train_costs']
            if train_costs and isinstance(train_costs[0], tuple):
                train_costs_only = [cost for epoch, cost in train_costs]
                result['history']['train_costs'] = train_costs_only
    
    logger.info(f"\n✅ Loaded data for {len(available_models)} models: {list(available_models)}")
    
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

# NOTE: Exact baseline computation removed - solver implementation was incomplete

def create_comparison_plots(results, training_times, model_params, config, scale='small', suffix='', base_dir=None):
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
    
    # Get naive baseline estimate (hardcoded based on previous experiments)
    naive_normalized = 1.053
    logger.info(f"📍 Using fixed naive baseline: {naive_normalized:.3f} per customer")
    
    # Build consistent color map for all figures (lines and bars)
    model_names = list(results.keys())
    palette = sns.color_palette("tab10", n_colors=len(model_names))
    color_map = {name: palette[i] for i, name in enumerate(model_names)}
    # Override GT-Greedy to grey
    if 'GT-Greedy' in color_map:
        color_map['GT-Greedy'] = (0.5, 0.5, 0.5)  # Grey color

    logger.info(f"🎨 Creating plots for {len(model_names)} models")
    logger.info(f"   Models: {model_names}")

    # CSV dir - use base_dir if provided
    if base_dir:
        csv_dir = os.path.join(base_dir, 'csv')
    elif isinstance(config, dict) and 'working_dir_path' in config:
        csv_dir = os.path.join(config['working_dir_path'], 'csv')
    else:
        csv_dir = os.path.join('results', scale, 'csv')
    
    logger.info(f"📂 Looking for CSV files in: {csv_dir}")
    
    # Map display names to CSV keys - matches current model factory
    name_to_key = {
        'GAT+RL': 'gat_rl',      # Legacy GAT model
        'GT-Greedy': 'gt_greedy', # Greedy baseline
        'GT+RL': 'gt_rl',        # Advanced Graph Transformer
        'DGT+RL': 'dgt_rl',      # Dynamic Graph Transformer
    }

    def load_csv_series_for_model(model_name):
        key = name_to_key.get(model_name)
        if not key:
            logger.warning(f"   ⚠️ No CSV key mapping for model: {model_name}")
            return None
        fpath = os.path.join(csv_dir, f"history_{key}.csv")
        if not os.path.exists(fpath):
            logger.warning(f"   ⚠️ CSV file not found: {fpath}")
            return None
        try:
            df = pd.read_csv(fpath)
            logger.info(f"   ✅ Loaded CSV for {model_name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to read CSV for {model_name}: {e}")
            return None
        # Ensure required columns exist (best-effort)
        for col in ['epoch', 'train_loss', 'train_cost', 'val_cost']:
            if col not in df.columns:
                logger.warning(f"   ⚠️ CSV for {model_name} missing column '{col}'")
        # Enhanced metrics (optional)
        ent_col = 'metric_train_entropy_mean'
        pol_loss_col = 'metric_train_policy_loss_mean'
        ent_series = df[ent_col].tolist() if ent_col in df.columns else []
        pol_series = df[pol_loss_col].tolist() if pol_loss_col in df.columns else []
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
            'train_entropy': ent_series,
            'policy_loss': pol_series,
        }

    csv_series = {name: load_csv_series_for_model(name) for name in model_names}

    # 1. Entropy or Policy Loss evolution panel
    ax1 = plt.subplot(2, 4, 1)
    # Decide what to plot: prefer entropy if available, else fall back to policy gradient loss
    have_entropy = any(
        (series is not None) and ('train_entropy' in series) and any(pd.notna(v) for v in (series['train_entropy'] or []))
        for series in csv_series.values()
    )
    if have_entropy:
        for model_name in model_names:
            series = csv_series.get(model_name)
            if series and series.get('train_entropy'):
                ys_raw = series['train_entropy']
                xs = [series['epochs'][i] for i, v in enumerate(ys_raw) if pd.notna(v)]
                ys = [v for v in ys_raw if pd.notna(v)]  # plot entropy directly
                if ys:
                    ax1.plot(xs, ys, label=model_name, linewidth=2, marker='o', markersize=3, color=color_map[model_name])
        ax1.set_title('Entropy Evolution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Entropy (per-epoch mean)')
    else:
        for model_name in model_names:
            series = csv_series.get(model_name)
            if series and series['train_loss']:
                xs = [series['epochs'][i] for i, v in enumerate(series['train_loss']) if pd.notna(v)]
                # Plot the policy gradient loss directly (typically negative for REINFORCE with advantages)
                ys = [v for v in series['train_loss'] if pd.notna(v)]
                if ys:
                    ax1.plot(xs, ys, label=model_name, linewidth=2, marker='o', markersize=3, color=color_map[model_name])
        ax1.set_title('Policy Gradient Loss Evolution\n(REINFORCE with Advantages)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Policy Loss (negative = good)')
    ax1.set_xlabel('Epoch')
    _safe_legend(ax1)
    ax1.grid(True, alpha=0.3)
    
# 2. Training Cost Comparison (NORMALIZED)
    ax2 = plt.subplot(2, 4, 2)
    for model_name in model_names:
        # Skip GT-Greedy since it has no training/RL process
        if model_name == 'GT-Greedy':
            continue
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
    
# 3. Validation Cost vs Baselines (NORMALIZED)
    ax3 = plt.subplot(2, 4, 3)
    num_epochs = config['training']['num_epochs']
    
    # GT-Greedy as baseline: draw a dashed horizontal line at its final validation cost
    if 'GT-Greedy' in results:
        gg_val = results['GT-Greedy'].get('final_val_cost', None)
        gg_std = results['GT-Greedy'].get('val_cost_std', results['GT-Greedy'].get('history', {}).get('val_cost_std', None))
        if gg_val is not None:
            # Add confidence band if std is available
            if gg_std is not None:
                ax3.fill_between([0, num_epochs], 
                                 gg_val - gg_std, 
                                 gg_val + gg_std,
                                 color='grey', alpha=0.15, label='GT-Greedy (±1 std)')
            # Plot middle line
            ax3.axhline(y=gg_val, color='grey', linewidth=2.5, linestyle='--', label='GT-Greedy Baseline')
    
    
    # Add OR-Tools GLS benchmark with N-dependent values
    # OR-Tools GLS baseline lookup table (n -> (avg_cpc, std_cpc))
    ortools_gls_data = {
        5: (0.489926, 0.097803), 6: (0.464466, 0.090135), 7: (0.449266, 0.081022),
        8: (0.424086, 0.070415), 9: (0.409788, 0.062406), 10: (0.392007, 0.061565),
        11: (0.384301, 0.056803), 12: (0.374863, 0.054727), 13: (0.368317, 0.054510),
        14: (0.356064, 0.052286), 15: (0.352531, 0.050019), 16: (0.347448, 0.048779),
        17: (0.339093, 0.048178), 18: (0.335759, 0.046840), 19: (0.331299, 0.048327),
        20: (0.330130, 0.045844)
    }
    
    # Get num_customers from config
    if isinstance(config, dict):
        num_customers = config.get('problem', {}).get('num_customers', 20)
    else:
        num_customers = 20  # default
    
    # Get OR-Tools GLS values for this problem size
    if num_customers in ortools_gls_data:
        ortools_gls_avg, ortools_gls_std = ortools_gls_data[num_customers]
    else:
        # Linear interpolation or use closest value for other sizes
        if num_customers < 5:
            ortools_gls_avg, ortools_gls_std = ortools_gls_data[5]
        elif num_customers > 20:
            ortools_gls_avg, ortools_gls_std = ortools_gls_data[20]
        else:
            # Find nearest neighbors for interpolation
            lower_n = max(k for k in ortools_gls_data.keys() if k <= num_customers)
            upper_n = min(k for k in ortools_gls_data.keys() if k >= num_customers)
            if lower_n == upper_n:
                ortools_gls_avg, ortools_gls_std = ortools_gls_data[lower_n]
            else:
                # Linear interpolation
                lower_avg, lower_std = ortools_gls_data[lower_n]
                upper_avg, upper_std = ortools_gls_data[upper_n]
                alpha = (num_customers - lower_n) / (upper_n - lower_n)
                ortools_gls_avg = lower_avg + alpha * (upper_avg - lower_avg)
                ortools_gls_std = lower_std + alpha * (upper_std - lower_std)
    # Plot confidence band (±1 std)
    ax3.fill_between([0, num_epochs], 
                     ortools_gls_avg - ortools_gls_std, 
                     ortools_gls_avg + ortools_gls_std,
                     color='green', alpha=0.15, label='OR-Tools GLS (±1 std)')
    # Plot middle line
    ax3.axhline(y=ortools_gls_avg, color='green', linewidth=2, linestyle='-', 
                label='OR-Tools GLS', alpha=0.7)

    # Plot model validation series (exclude GT-Greedy since it's now shown as baseline)
    for model_name in model_names:
        if model_name == 'GT-Greedy':
            continue  # GT-Greedy is shown as baseline
        series = csv_series.get(model_name)
        if series and series['val_costs']:
            xs = series['val_epochs']
            ys = series['val_costs']
            ax3.plot(xs, ys, 'o-', label=model_name, linewidth=2, markersize=5, color=color_map[model_name])

    ax3.set_title('Validation Cost vs Baselines (Per Customer)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Cost per Customer')
    _safe_legend(ax3)
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Performance Bar Chart with Naive/Exact Baseline (NORMALIZED)
    ax4 = plt.subplot(2, 4, 4)
    
    # Define the desired order for Panel 4
    panel4_order = ['GT-Greedy', 'GAT+RL', 'GT+RL', 'DGT+RL']
    
    # Filter and reorder models based on desired order
    ordered_names = []
    ordered_costs = []
    ordered_colors = []
    
    for name in panel4_order:
        if name in results:
            result = results[name]
            if 'history' in result:
                # New enhanced format
                final_cost = result['history'].get('final_val_cost', 0.0)
            else:
                # Old format
                final_cost = result.get('final_val_cost', 0.0)
            ordered_names.append(name)
            ordered_costs.append(final_cost)
            ordered_colors.append(color_map[name])
    
    # Add baselines in specific order: OR-Tools GLS, then Naive
    # OR-Tools GLS
    ordered_names.append('OR-Tools GLS')
    ordered_costs.append(ortools_gls_avg)  # Use N-dependent value
    ordered_colors.append((0.0, 0.5, 0.0))  # green for OR-Tools
    
    # Naive Baseline
    ordered_names.append('Worst')
    ordered_costs.append(naive_normalized)
    ordered_colors.append((0.8, 0.2, 0.2))  # red-like for naive
    
    
    all_names = ordered_names
    all_costs_normalized = ordered_costs
    all_colors = ordered_colors
    
    bars = ax4.bar(range(len(all_names)), all_costs_normalized, color=all_colors, alpha=0.85)
    title_suffix = ' vs Baselines'
    ax4.set_title(f'Final Performance{title_suffix} (Per Customer)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Approach')
    ax4.set_ylabel('Average Cost per Customer')
    ax4.set_xticks(range(len(all_names)))
    ax4.set_xticklabels([name.replace(' ', '\\n') for name in all_names], rotation=30, ha='right', fontsize=8)
    
    # Add headroom to reduce label overlap
    y_max = max(all_costs_normalized) if all_costs_normalized else 1.0
    ax4.set_ylim(0.0, y_max * 1.15)
    
    # Add value labels on bars (normalized) with padding and smaller font
    for bar, cost in zip(bars, all_costs_normalized):
        ax4.annotate(f'{cost:.3f}',
                     xy=(bar.get_x() + bar.get_width()/2., bar.get_height()),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8, rotation=0, clip_on=False)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Training Time Comparison
    plt.subplot(2, 4, 5)
    # Define the desired order for Panel 5 (RL models only)
    panel5_order = ['GAT+RL', 'GT+RL', 'DGT+RL']
    
    # Filter and reorder models based on desired order
    time_models = []
    times = []
    time_colors = []
    
    for name in panel5_order:
        if name in results and name != 'GT-Greedy':
            time_models.append(name)
            times.append(training_times.get(name, 0.0))
            time_colors.append(color_map[name])
    
    bars = plt.bar(range(len(time_models)), times, color=time_colors, alpha=0.8)
    plt.title('Training Time', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(time_models)), [name.replace(' ', '\n') for name in time_models], rotation=45, ha='right')
    
    if times:
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(times)*0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Model Complexity (Parameters)
    plt.subplot(2, 4, 6)
    # Define the desired order for Panel 6 (include GT-Greedy)
    panel6_order = ['GT-Greedy', 'GAT+RL', 'GT+RL', 'DGT+RL']
    
    # Filter and reorder models based on desired order
    complexity_models = []
    param_counts = []
    param_colors = []
    
    for name in panel6_order:
        if name in results:
            complexity_models.append(name)
            param_counts.append(model_params[name])
            param_colors.append(color_map[name])
    
    bars = plt.bar(range(len(complexity_models)), param_counts, color=param_colors, alpha=0.8)
    plt.title('Model Complexity', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture')
    plt.ylabel('Parameters')
    plt.xticks(range(len(complexity_models)), [name.replace(' ', '\n') for name in complexity_models], rotation=45, ha='right')
    
    for bar, params in zip(bars, param_counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(param_counts)*0.01,
                f'{params:,}', ha='center', va='bottom', fontweight='bold', fontsize=8, rotation=0)
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # 7. Learning Efficiency (Cost Improvement) - exclude GT-Greedy
    plt.subplot(2, 4, 7)
    # Define the desired order for Panel 7 (RL models only)
    panel7_order = ['GAT+RL', 'GT+RL', 'DGT+RL']
    
    # Filter and reorder models based on desired order
    eff_models = []
    improvements = []
    eff_colors = []
    
    for name in panel7_order:
        if name in results and name != 'GT-Greedy':  # Ensure GT-Greedy is excluded
            series = csv_series.get(name)
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
            eff_models.append(name)
            improvements.append(imp)
            eff_colors.append(color_map[name])
    
    bars = plt.bar(range(len(eff_models)), improvements, color=eff_colors, alpha=0.8)
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

    # 8. Performance vs Complexity Scatter (NORMALIZED)
    plt.subplot(2, 4, 8)
    
    # Prepare data for scatter plot: final costs and parameters for available models
    scatter_costs = []
    scatter_params = []
    scatter_labels = []

    for model_name in model_names: # model_names list comes from model_names = list(results.keys())
        if model_name in results and model_name in model_params:
            result = results[model_name]
            # Handle both old and new enhanced formats for final_val_cost
            if 'history' in result:
                final_cost = result['history'].get('final_val_cost', None)
            else:
                final_cost = result.get('final_val_cost', None)

            if final_cost is not None and model_params[model_name] > 0: # Ensure valid data for plotting
                scatter_costs.append(final_cost)
                scatter_params.append(model_params[model_name])
                scatter_labels.append(model_name)

    for i, model_name in enumerate(scatter_labels):
        plt.scatter(scatter_params[i], scatter_costs[i], 
                   s=100, color=color_map[model_name], alpha=0.8, label=model_name)
        plt.annotate(model_name.replace(' ', '\n'), 
                    (scatter_params[i], scatter_costs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Performance vs Complexity (Per Customer)', fontsize=12, fontweight='bold')
    plt.xlabel('Parameters')
    plt.ylabel('Validation Cost per Customer')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure plots directory exists - use base_dir parameter if provided
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), 'results')
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save plot
    output_filename = f"comparative_study_results{suffix}.png"
    output_path = os.path.join(plots_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Comparison plots saved to {output_path}")
    
    # Create performance summary
    create_performance_summary(results, training_times, model_params, config, naive_normalized, scale, suffix, base_dir)
    
    # Don't show the plot in non-interactive mode
    # plt.show()

def create_performance_summary(results, training_times, model_params, config, naive_baseline_per_customer, scale, suffix='', base_dir=None):
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
        
        data.append(row_data)
    
    df = pd.DataFrame(data)
    
    # Sort by performance (lowest cost per customer first)
    df = df.sort_values('Final Val Cost/Customer')
    
    # Save to CSV - use base_dir parameter if provided, otherwise fallback
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), 'results')
    csv_dir = os.path.join(base_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f'{csv_dir}/comparative_results{suffix}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"📋 Performance summary saved to {csv_path}")
    
    # Print formatted table to console
    logger.info("\n📊 DETAILED PERFORMANCE COMPARISON")
    logger.info(f"📍 Baseline: Naive={naive_baseline_per_customer:.4f}/customer")
    logger.info("=" * 120)
    
    # Create nicely formatted table
    col_widths = {
        'Model': max(len('Model'), max(len(model) for model in df['Model'])),
        'Parameters': 11,
        'Time (s)': 9,
        'Val/Cust': 9,
        'vs Naive': 9
    }
    
    # Header
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
        
        row_str = f"| {model_name:<{col_widths['Model']}} | {params:>{col_widths['Parameters']}} | {time_s:>{col_widths['Time (s)']}} | {val_per_cust:>{col_widths['Val/Cust']}} | {improvement_naive:>{col_widths['vs Naive']}} |"
        print(row_str)
    
    print("=" * 120)
    best_model = df.iloc[0]
    print(f"🏆 Best model: {best_model['Model']} ({best_model['Final Val Cost/Customer']:.4f}/customer, {best_model['Improvement vs Naive (%)']:+.1f}% vs naive)")
    

def _deep_merge_dict(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge_dict(a[k], v)
        else:
            a[k] = v
    return a

def load_config(config_path):
    """Load config with proper path resolution"""
    import yaml
    from pathlib import Path
    import os
    
    # Save the current working directory
    original_cwd = os.getcwd()
    
    # Convert config_path to absolute path first
    config_path = Path(config_path).resolve()
    
    try:
        # Change to project root for config loading
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        # Now load the config using the shared loader with absolute path
        from src.utils.config import load_config as _shared_load
        config = _shared_load(str(config_path))
        
        return config
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

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
    return parser.parse_args()

def main():
    """Main function to generate comparative plots"""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("🚀 Starting Comparative Plot Generation")
    logger.info(f"📂 Loading config from: {args.config}")
    
    try:
        # Load config
        cfg = load_config(args.config)
        
        # Use working_dir_path from config if available
        if 'working_dir_path' in cfg:
            from pathlib import Path
            working_dir = Path(cfg['working_dir_path'])
            if not working_dir.is_absolute():
                # If relative, it's relative to project root
                project_root = Path(__file__).parent.parent
                base_dir = project_root / working_dir
            else:
                base_dir = working_dir
            base_dir = str(base_dir)
        else:
            # Fallback to local results directory within training_cpu
            base_dir = os.path.join(os.path.dirname(__file__), 'results')
        
        logger.info(f"📁 Working directory: {base_dir}")
        
        # Load saved results
        results, training_times, model_params, loaded_config = load_results(base_dir)
        
        # Create plots
        suffix = f"_{args.suffix}" if args.suffix else ""
        # Derive a label (for file locations only) from working_dir_path leaf
        label = Path(base_dir).name
        create_comparison_plots(results, training_times, model_params, loaded_config, label, suffix, base_dir)
        
        logger.info("✅ Comparative plots generated successfully!")
        logger.info(f"📊 Output: {base_dir}/plots/comparative_study_results{suffix}.png")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate plots: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

