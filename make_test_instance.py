#!/usr/bin/env python3
"""
Create and solve a single test instance using trained models.
Exact copy of the test instance pipeline from run_comparative_study.py.
Includes integrated visualization functions to be fully standalone.
"""

import os
import sys
import torch
import numpy as np
import argparse
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Suppress matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)

# Import from run_train_validation.py  
from run_train_validation import (
    generate_cvrp_instance, compute_route_cost, validate_route, naive_baseline_solution,
    BaselinePointerNetwork, GraphTransformerGreedy, GraphTransformerNetwork, 
    DynamicGraphTransformerNetwork, GraphAttentionTransformer, compute_naive_baseline_cost,
    build_pyg_data_from_instance, setup_logging
)

device = torch.device("cpu")

# =============================================================================
# INTEGRATED VISUALIZATION FUNCTIONS (from visualize_test_routes.py)
# =============================================================================

def analyze_route_trips(route, demands, capacity):
    """Analyze route to identify trips and validate demand constraints."""
    trips = []
    current_trip = []
    current_demand = 0
    
    for i, node in enumerate(route):
        current_trip.append(node)
        
        if node == 0 and len(current_trip) > 1:  # End of trip (return to depot)
            # Calculate trip demand (excluding depot nodes)
            trip_customers = [n for n in current_trip if n != 0]
            trip_demand = sum(int(demands[customer]) for customer in trip_customers)
            
            trips.append({
                'nodes': current_trip[:],
                'customers': trip_customers,
                'demand': trip_demand,
                'capacity_used': trip_demand / capacity * 100,
                'valid': trip_demand <= capacity
            })
            
            current_trip = [0]  # Start new trip at depot
    
    return trips

def create_model_route_plot(coords, demands, sizes, route, cost_per_customer, model_name, color, save_dir, capacity, validation_cost=None):
    """Create a single route plot for one model with trip analysis."""
    n = len(coords)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Analyze trips
    trips = analyze_route_trips(route, demands, capacity)
    
    # Print trip analysis
    print(f"\n🚛 {model_name} - Trip Analysis:")
    print(f"   Total trips: {len(trips)}")
    total_demand = 0
    for i, trip in enumerate(trips, 1):
        status = "✅" if trip['valid'] else "❌"
        print(f"   Trip {i}: {' → '.join(map(str, trip['nodes']))}")
        print(f"           Customers: {trip['customers']} | Demand: {trip['demand']}/{capacity} ({trip['capacity_used']:.1f}%) {status}")
        total_demand += trip['demand']
    print(f"   Total demand served: {total_demand}")
    
    # Plot route segments with different styles based on trip type
    trip_colors = plt.cm.Set3(range(len(trips)))  # Different colors for multi-customer trips
    
    # Plot each trip with appropriate styling
    multi_trip_idx = 0  # Counter for multi-customer trips only
    
    for trip_idx, trip in enumerate(trips):
        trip_nodes = trip['nodes']
        is_round_trip = len(trip['customers']) == 1  # Only one customer in this trip
        
        if is_round_trip:
            # Round trip (0->customer->0): light grey solid line
            trip_color = 'lightgrey'
            alpha = 0.7
            linewidth = 1.5
            label = 'Round trip' if trip_idx == 0 and any(len(t['customers']) == 1 for t in trips) else ''
        else:
            # Multi-customer trip: use distinct colors
            if multi_trip_idx < len(trip_colors):
                trip_color = trip_colors[multi_trip_idx]
            else:
                trip_color = color
            alpha = 0.8
            linewidth = 2.0
            label = f'Trip {multi_trip_idx + 1}' if multi_trip_idx == 0 else ''
            multi_trip_idx += 1
        
        # Plot all segments in this trip with the same color/style
        for i in range(len(trip_nodes) - 1):
            a, b = trip_nodes[i], trip_nodes[i + 1]
            ax.plot([coords[a, 0], coords[b, 0]], 
                   [coords[a, 1], coords[b, 1]], 
                   '-', color=trip_color, alpha=alpha, linewidth=linewidth,
                   label=label if i == 0 else '')  # Only label first segment of each trip type
            label = ''  # Clear label after first use
    
    # Plot depot (node 0) with star marker
    ax.scatter(coords[0:1, 0], coords[0:1, 1], 
              marker='*', s=200, c='red', 
              edgecolors='black', linewidths=1.5, 
              label='Depot', zorder=10)
    
    # Plot customers with size proportional to demand
    ax.scatter(coords[1:, 0], coords[1:, 1], 
              s=sizes[1:], c='lightblue', 
              edgecolors='black', linewidths=1.0, 
              label='Customers', zorder=5)
    
    # Annotate customers with (index, demand) - show integer demands
    for i in range(1, n):
        ax.annotate(f'({i}, {int(demands[i])})', 
                   (coords[i, 0], coords[i, 1]), 
                   textcoords='offset points', xytext=(8, 8), 
                   fontsize=8, color='black')
    
    # Create title with test instance cost and validation cost (if available)
    if validation_cost is not None:
        title = f'{model_name}\nCost per Customer: {cost_per_customer:.3f} (Val: {validation_cost:.3f})'
    else:
        title = f'{model_name}\nCost per Customer: {cost_per_customer:.3f}'
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    # ax.legend(loc='upper right', fontsize=10)
    
    # Set reasonable axis limits with some padding
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    padding = 0.05
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    plt.tight_layout()
    
    # Use consistent filename sanitization for both PNG and JSON
    def sanitize_filename(name):
        return name.lower().replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '').replace('/', '_')
    
    # Save plot
    filename = f"test_route_{sanitize_filename(model_name)}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save route as JSON
    json_filename = f"test_route_{sanitize_filename(model_name)}.json"
    with open(os.path.join(save_dir, json_filename), 'w') as f:
        json.dump({
            'model_name': model_name,
            'route': route,
            'cost_per_customer': cost_per_customer,
            'coordinates': coords.tolist(),
            'demands': demands.tolist()
        }, f, indent=2)

def create_comparison_plot(coords, demands, sizes, model_results, naive_baseline, route_colors, save_dir):
    """Create a comparison plot showing all model routes in subplots."""
    # Filter out 'Naive Baseline' from model_results to avoid duplication
    # Since naive_baseline is passed separately
    filtered_model_results = {k: v for k, v in model_results.items() if k != 'Naive Baseline'}
    
    # Determine subplot layout
    n_models = len(filtered_model_results) + 1  # +1 for naive baseline
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot each model (excluding naive baseline)
    for model_name, results in filtered_model_results.items():
        ax = axes[plot_idx]
        route = results['greedy_route']
        cost_per_customer = results['greedy_cost_per_customer']
        color = route_colors.get(model_name, 'gray')
        
        # Plot route
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            ax.plot([coords[a, 0], coords[b, 0]], 
                   [coords[a, 1], coords[b, 1]], 
                   '-', color=color, alpha=0.85, linewidth=1.5)
        
        # Plot nodes
        ax.scatter(coords[0:1, 0], coords[0:1, 1], 
                  marker='*', s=100, c='red', 
                  edgecolors='black', linewidths=1.0)
        ax.scatter(coords[1:, 0], coords[1:, 1], 
                  s=sizes[1:]*0.3, c='lightblue', 
                  edgecolors='black', linewidths=0.5)
        
        ax.set_title(f'{model_name}\n{cost_per_customer:.3f}/cust', 
                    fontsize=10, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Plot naive baseline (always add it once at the end)
    if plot_idx < len(axes):
        ax = axes[plot_idx]
        route = naive_baseline['route']
        cost_per_customer = naive_baseline['cost_per_customer']
        color = route_colors['Naive Baseline']
        
        # Plot route
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            ax.plot([coords[a, 0], coords[b, 0]], 
                   [coords[a, 1], coords[b, 1]], 
                   '-', color=color, alpha=0.85, linewidth=1.5)
        
        # Plot nodes
        ax.scatter(coords[0:1, 0], coords[0:1, 1], 
                  marker='*', s=100, c='red', 
                  edgecolors='black', linewidths=1.0)
        ax.scatter(coords[1:, 0], coords[1:, 1], 
                  s=sizes[1:]*0.3, c='lightblue', 
                  edgecolors='black', linewidths=0.5)
        
        ax.set_title(f'Naive Baseline\n{cost_per_customer:.3f}/cust', 
                    fontsize=10, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Test Instance Route Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison plot
    plt.savefig(os.path.join(save_dir, "test_routes_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_test_instance_routes(test_analysis, config, logger, save_dir, scale=None):
    """
    Plot routes for all models on the test instance with annotated styling.
    
    Args:
        test_analysis: Dictionary containing test instance data and model results
        config: Configuration dictionary containing problem parameters
        logger: Logger for output messages
        save_dir: Directory to save plots
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract test instance data
    test_instance = test_analysis['test_instance']
    coords = np.array(test_instance['coords'])
    demands = np.array(test_instance['demands'], dtype=np.int32)  # Ensure integers
    capacity = int(test_instance['capacity'])  # Ensure integer
    n_customers = config['num_customers']
    
    # Get model results
    model_results = test_analysis['model_results']
    naive_baseline = test_analysis['naive_baseline']
    
    # Load validation costs for comparison (if available)
    validation_costs = {}
    try:
        # Try to load the comparative study results to get validation costs
        # Use working directory from config
        base_dir = config.get('working_dir_path') if isinstance(config, dict) else None
        if base_dir is None:
            raise ValueError('working_dir_path must be set in config')
        comparative_results = torch.load(os.path.join(base_dir, 'analysis', 'comparative_study_complete.pt'), map_location='cpu', weights_only=False)
        results = comparative_results.get('results', {})
        for model_name in model_results.keys():
            if model_name in results:
                # final_val_cost is already per-customer in comparative results; do not divide again
                validation_costs[model_name] = results[model_name]['final_val_cost']
    except Exception as e:
        logger.warning(f"Could not load validation costs: {e}")
    
    logger.info("🎨 Creating route visualization plots...")
    
    # Plot styling parameters
    base_size = 4.0
    scale_size = 30.0
    sizes = base_size + scale_size * demands
    
    # Colors for different models
    route_colors = {
        'Pointer+RL': 'green',
        'GT-Greedy': 'blue',
        'GT+RL': 'purple',
        'DGT+RL': 'orange',
        'GAT+RL': 'brown',
        'GAT+RL (legacy)': 'pink',
        'Naive Baseline': 'red'
    }
    
    # Create plots for each model
    for model_name, results in model_results.items():
        validation_cost = validation_costs.get(model_name, None)
        create_model_route_plot(
            coords, demands, sizes, 
            results['greedy_route'], 
            results['greedy_cost_per_customer'],
            model_name, 
            route_colors.get(model_name, 'gray'),
            save_dir,
            capacity,
            validation_cost
        )
        
        logger.info(f"   📊 Saved route plot for {model_name}")
    
    # Create naive baseline plot (no validation cost for naive baseline)
    create_model_route_plot(
        coords, demands, sizes,
        naive_baseline['route'],
        naive_baseline['cost_per_customer'],
        'Naive Baseline',
        route_colors['Naive Baseline'],
        save_dir,
        capacity,
        None  # No validation cost for naive baseline
    )
    
    logger.info(f"   📊 Saved route plot for Naive Baseline")
    
    # Create comparison subplot with all models
    create_comparison_plot(coords, demands, sizes, model_results, naive_baseline, route_colors, save_dir)
    logger.info(f"   📊 Saved comparison plot with all models")
    
    logger.info(f"🎨 All route plots saved to {save_dir}")

# =============================================================================
# MAIN SCRIPT FUNCTIONS
# =============================================================================

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_config(config_path):
    """Load configuration from YAML file with proper type conversion"""
    import yaml
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Fix type conversions for numerical parameters that might be read as strings
    
    # Convert inference parameters
    if 'inference' in config:
        inf_config = config['inference']
        if 'log_prob_epsilon' in inf_config:
            inf_config['log_prob_epsilon'] = float(inf_config['log_prob_epsilon'])
        if 'masked_score_value' in inf_config:
            inf_config['masked_score_value'] = float(inf_config['masked_score_value'])
        if 'default_temperature' in inf_config:
            inf_config['default_temperature'] = float(inf_config['default_temperature'])
        if 'max_steps_multiplier' in inf_config:
            inf_config['max_steps_multiplier'] = int(inf_config['max_steps_multiplier'])
        if 'attention_temperature_scaling' in inf_config:
            inf_config['attention_temperature_scaling'] = float(inf_config['attention_temperature_scaling'])
    
    # Convert training_advanced parameters
    if 'training_advanced' in config:
        ta_config = config['training_advanced']
        # Convert legacy_gat sub-parameters
        if 'legacy_gat' in ta_config:
            lg_config = ta_config['legacy_gat']
            if 'learning_rate' in lg_config:
                lg_config['learning_rate'] = float(lg_config['learning_rate'])
            if 'temperature' in lg_config:
                lg_config['temperature'] = float(lg_config['temperature'])
    
    # Convert model parameters
    if 'model' in config:
        model_config = config['model']
        # Convert dynamic_graph_transformer sub-parameters
        if 'dynamic_graph_transformer' in model_config:
            dgt_config = model_config['dynamic_graph_transformer']
            if 'residual_gate_init' in dgt_config:
                dgt_config['residual_gate_init'] = float(dgt_config['residual_gate_init'])
    
    return config

def load_trained_models(models_dir, config, logger):
    """Load all trained models from saved files"""
    models = {}
    model_files = {
        'Pointer+RL': os.path.join(models_dir, 'model_pointer_rl.pt'),
        'GT-Greedy': os.path.join(models_dir, 'model_gt_greedy.pt'), 
        'GT+RL': os.path.join(models_dir, 'model_gt_rl.pt'),
        'DGT+RL': os.path.join(models_dir, 'model_dgt_rl.pt'),
        'GAT+RL': os.path.join(models_dir, 'model_gat_rl.pt')
    }
    
    # Try to load legacy GAT model
    try:
        from src_batch.model.Model import Model as LegacyGATModel
        legacy_model = LegacyGATModel(node_input_dim=3, edge_input_dim=1, hidden_dim=128, edge_dim=16, layers=4, negative_slope=0.2, dropout=0.6)
        legacy_path = os.path.join(models_dir, 'model_gat_rl_legacy.pt')
        if os.path.exists(legacy_path):
            state_dict = torch.load(legacy_path, map_location=device, weights_only=False)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                legacy_model.load_state_dict(state_dict['model_state_dict'])
            else:
                legacy_model.load_state_dict(state_dict)
            models['GAT+RL (legacy)'] = legacy_model
            logger.info("✅ Loaded model: GAT+RL (legacy)")
    except Exception as e:
        logger.warning(f"Legacy GAT+RL unavailable: {e}")
    
    # Load regular models with proper parameters
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                state_dict = torch.load(filepath, map_location=device, weights_only=False)
                
                # Extract model config parameters
                input_dim = config['model']['input_dim']
                hidden_dim = config['model']['hidden_dim']
                num_heads = config['model']['num_heads']
                num_layers = config['model']['num_layers']
                dropout = config['model']['transformer_dropout']
                feedforward_multiplier = config['model']['feedforward_multiplier']
                edge_embedding_divisor = config['model']['edge_embedding_divisor']
                
                if name == 'Pointer+RL':
                    model = BaselinePointerNetwork(input_dim, hidden_dim, config)
                elif name == 'GT-Greedy':
                    model = GraphTransformerGreedy(input_dim, hidden_dim, num_heads, num_layers, dropout, feedforward_multiplier, config)
                elif name == 'GT+RL':
                    model = GraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, feedforward_multiplier, config)
                elif name == 'DGT+RL':
                    model = DynamicGraphTransformerNetwork(input_dim, hidden_dim, num_heads, num_layers, dropout, feedforward_multiplier, config)
                elif name == 'GAT+RL':
                    model = GraphAttentionTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout, edge_embedding_divisor, config)
                
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
                
                model.eval()
                models[name] = model
                logger.info(f"✅ Loaded model: {name}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
    
    return models

def create_and_solve_test_instance(models, config, logger, base_dir):
    """Create a test CVRP instance and solve it with each trained model for detailed analysis
    
    This is an exact copy from run_comparative_study.py create_and_solve_test_instance function.
    """
    logger.info("\n🧪 Creating and solving a new test instance...")
    
    # Use working directory from config
    test_dir = os.path.join(base_dir, 'plots')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a representative test instance
    test_instance = generate_cvrp_instance(
        num_customers=config['num_customers'],
        capacity=config['capacity'],
        coord_range=config['coord_range'],
        demand_range=config['demand_range'],
        seed=12345  # Fixed seed for reproducibility
    )
    
    logger.info(f"📍 Test instance (seed=12345): {config['num_customers']} customers, capacity={config['capacity']}")
    
    # Compute naive baseline for reference
    naive_cost = compute_naive_baseline_cost(test_instance)
    naive_route = naive_baseline_solution(test_instance)
    naive_normalized = naive_cost / config['num_customers']
    
    # Test each model on the instance
    test_results = {}
    
    for model_name, model in models.items():
        logger.info(f"\n🔍 Solving with {model_name}...")
        model.eval()
        
        with torch.no_grad():
            # Test both greedy and sampling - handle legacy model differently
            if model_name == 'GAT+RL (legacy)':
                # Legacy model uses different interface and expects batched PyG data
                from torch_geometric.loader import DataLoader
                test_data = build_pyg_data_from_instance(test_instance)
                # Create a DataLoader with batch_size=1 to properly batch the single instance
                test_loader = DataLoader([test_data], batch_size=1, shuffle=False)
                test_batch = next(iter(test_loader))
                
                n_steps = config['num_customers'] * 2
                T = 2.5
                
                # Legacy model returns actions, log_probs (no entropy)
                greedy_actions, greedy_log_probs = model(test_batch, n_steps=n_steps, greedy=True, T=T)
                
                # Convert to route format (add depot at start and end)
                # Handle different possible tensor shapes from legacy model
                if greedy_actions.dim() == 1:
                    # If 1D, add batch dimension
                    greedy_actions = greedy_actions.unsqueeze(0)
                
                depot = torch.zeros(greedy_actions.size(0), 1, dtype=torch.long)
                greedy_routes = [torch.cat([depot, greedy_actions, depot], dim=1)[0].cpu().tolist()]
                
                # Use greedy route for both (legacy doesn't need sampling)
                sample_routes = greedy_routes
                sample_log_probs = greedy_log_probs
                
                # Set dummy entropy values
                greedy_entropy = torch.zeros(1)
                sample_entropy = torch.zeros(1)
            else:
                # Regular models - only use greedy for this standalone script
                greedy_routes, greedy_log_probs, greedy_entropy = model(
                    [test_instance], 
                    max_steps=config['inference']['max_steps_multiplier'] * config['num_customers'],
                    temperature=config['inference']['default_temperature'],
                    greedy=True, 
                    config=config
                )
                sample_routes = greedy_routes  # Use same route for consistency
                sample_log_probs = greedy_log_probs
                sample_entropy = greedy_entropy
            
            greedy_route = greedy_routes[0]
            sample_route = sample_routes[0]
            
            # Validate routes
            validate_route(greedy_route, config['num_customers'], f"{model_name}-TEST", test_instance)
            
            # Compute costs
            greedy_cost = compute_route_cost(greedy_route, test_instance['distances'])
            sample_cost = greedy_cost  # Same as greedy for this script
            
            test_results[model_name] = {
                'greedy_route': greedy_route,
                'sample_route': sample_route,
                'greedy_cost': greedy_cost,
                'sample_cost': sample_cost,
                'greedy_cost_per_customer': greedy_cost / config['num_customers'],
                'sample_cost_per_customer': sample_cost / config['num_customers'],
                'greedy_log_prob': greedy_log_probs[0].item(),
                'sample_log_prob': sample_log_probs[0].item(),
                'greedy_entropy': greedy_entropy[0].item(),
                'sample_entropy': sample_entropy[0].item()
            }
            
            logger.info(f"   Route Cost: {greedy_cost:.4f} ({greedy_cost/config['num_customers']:.4f}/customer)")
    
    # Add naive baseline to results
    test_results['Naive Baseline'] = {
        'greedy_route': naive_route,
        'sample_route': naive_route,
        'greedy_cost': naive_cost,
        'sample_cost': naive_cost,
        'greedy_cost_per_customer': naive_normalized,
        'sample_cost_per_customer': naive_normalized,
        'greedy_log_prob': 0.0,
        'sample_log_prob': 0.0,
        'greedy_entropy': 0.0,
        'sample_entropy': 0.0
    }
    
    # Create detailed test results analysis
    test_analysis = {
        'test_instance': {
            'coords': test_instance['coords'].tolist(),
            'demands': test_instance['demands'].tolist(),
            'capacity': test_instance['capacity'],
            'num_customers': config['num_customers']
        },
        'naive_baseline': {
            'route': naive_route,
            'cost': naive_cost,
            'cost_per_customer': naive_normalized
        },
        'model_results': test_results,
        'config': config
    }
    
    # Convert numpy types to native Python for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    # Save JSON files for each model
    for model_name, result in test_results.items():
        def sanitize_name(name):
            return name.lower().replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '').replace('/', '_')
        
        filename = f"test_route_{sanitize_name(model_name)}.json"
        filepath = os.path.join(test_dir, filename)
        
        model_data = {
            'model': model_name,
            'seed': 12345,
            'instance': {
                'coords': test_instance['coords'].tolist(),
                'demands': test_instance['demands'].tolist(),
                'capacity': int(test_instance['capacity'])
            },
            'route': result['greedy_route'],
            'cost': float(result['greedy_cost']),
            'cost_per_customer': float(result['greedy_cost_per_customer'])
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"   💾 Saved route JSON for {model_name}")
    
    # Create route visualizations - now using integrated functions
    try:
        plots_dir = os.path.join(base_dir, 'plots')
        plot_test_instance_routes(test_analysis, config, logger, plots_dir, Path(base_dir).name)
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to create route visualizations: {e}")
    
    return test_results

def main():
    parser = argparse.ArgumentParser(description='Generate test instance and solve with trained models')
    parser.add_argument('--config', type=str, default='configs/medium.yaml', help='Path to YAML configuration file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for instance generation (optional)')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Set random seed if provided (note: test instance uses fixed seed 12345)
    if args.seed is not None:
        set_seeds(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract nested config values
    if 'problem' in config:
        num_customers = config['problem']['num_customers']
        capacity = config['problem']['vehicle_capacity']
        coord_range = config['problem']['coord_range']
        demand_range = config['problem']['demand_range']
        config.update({
            'num_customers': num_customers,
            'capacity': capacity,
            'coord_range': coord_range,
            'demand_range': demand_range
        })
    
    # Working directory for artifacts
    base_dir = str(Path(config.get('working_dir_path', 'results')).as_posix())
    
    # Load models
    models_dir = os.path.join(base_dir, 'pytorch')
    models = load_trained_models(models_dir, config, logger)
    
    if not models:
        logger.error(f"No trained models found in {models_dir}")
        return
    
    # Create and solve test instance (exact copy from run_comparative_study.py)
    test_results = create_and_solve_test_instance(models, config, logger, base_dir)
    
    logger.info(f"\n💾 Generated plots and JSON files in {base_dir}/plots")

if __name__ == '__main__':
    main()
