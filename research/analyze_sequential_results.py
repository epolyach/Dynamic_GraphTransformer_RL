#!/usr/bin/env python3
"""
Analysis and visualization script for sequential hyperparameter optimization results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import glob
import os

def load_latest_results():
    """Load the most recent results file."""
    results_files = glob.glob("sequential_optimization_results_*.json")
    if not results_files:
        raise FileNotFoundError("No sequential optimization results found")
    
    # Get the most recent file
    latest_file = max(results_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def analyze_results(results):
    """Analyze the optimization results."""
    print("\n" + "="*60)
    print("SEQUENTIAL OPTIMIZATION ANALYSIS")
    print("="*60)
    
    models = ['GAT+RL', 'GT+RL', 'DGT+RL']
    costs = []
    
    for model in models:
        if model in results:
            cost = results[model]['best_cost']
            params = results[model]['best_params']
            starting = results[model]['starting_params']
            
            costs.append(cost)
            print(f"\n{model}:")
            print(f"  Best cost: {cost:.6f}")
            print(f"  Starting params: {starting}")
            print(f"  Best params: {params}")
        else:
            print(f"\n{model}: No results found")
            costs.append(None)
    
    # Check if sequential improvement was achieved
    valid_costs = [c for c in costs if c is not None and c != 10.0]  # Exclude failed runs
    
    if len(valid_costs) >= 3:
        gat_cost, gt_cost, dgt_cost = costs[0], costs[1], costs[2]
        
        print(f"\n" + "-"*40)
        print("SEQUENTIAL IMPROVEMENT ANALYSIS:")
        print(f"GAT+RL:  {gat_cost:.6f}")
        print(f"GT+RL:   {gt_cost:.6f}")
        print(f"DGT+RL:  {dgt_cost:.6f}")
        
        success = (dgt_cost < gt_cost < gat_cost)
        print(f"\nTarget (DGT < GT < GAT): {'✅ ACHIEVED' if success else '❌ NOT ACHIEVED'}")
        
        if success:
            improvement_gt = ((gat_cost - gt_cost) / gat_cost) * 100
            improvement_dgt = ((gt_cost - dgt_cost) / gt_cost) * 100
            total_improvement = ((gat_cost - dgt_cost) / gat_cost) * 100
            
            print(f"\nImprovements:")
            print(f"GAT→GT:   {improvement_gt:.2f}% improvement")
            print(f"GT→DGT:   {improvement_dgt:.2f}% improvement") 
            print(f"GAT→DGT:  {total_improvement:.2f}% total improvement")
    
    return costs

def plot_results(results):
    """Create visualizations of the optimization results."""
    models = ['GAT+RL', 'GT+RL', 'DGT+RL']
    
    # Extract data for plotting
    costs = []
    param_data = {
        'embedding_dim': [],
        'n_layers': [],
        'n_heads': [],
        'learning_rate': [],
        'dropout': [],
        'temperature': [],
        'weight_decay': []
    }
    
    for model in models:
        if model in results and results[model]['best_cost'] != 10.0:
            costs.append(results[model]['best_cost'])
            params = results[model]['best_params']
            
            for param_name in param_data.keys():
                if param_name in params:
                    param_data[param_name].append(params[param_name])
                else:
                    param_data[param_name].append(None)
        else:
            costs.append(None)
            for param_name in param_data.keys():
                param_data[param_name].append(None)
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle('Sequential Hyperparameter Optimization Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Validation costs
    ax = axes[0, 0]
    valid_indices = [i for i, c in enumerate(costs) if c is not None]
    valid_costs = [costs[i] for i in valid_indices]
    valid_models = [models[i] for i in valid_indices]
    
    if valid_costs:
        bars = ax.bar(valid_models, valid_costs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_ylabel('Validation Cost')
        ax.set_title('Final Validation Costs')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, cost in zip(bars, valid_costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{cost:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Embedding dimension evolution
    ax = axes[0, 1]
    embedding_dims = [d for d in param_data['embedding_dim'] if d is not None]
    if embedding_dims:
        ax.plot(range(len(embedding_dims)), embedding_dims, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Embedding Dimension')
        ax.set_title('Embedding Dimension Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(embedding_dims)))
        ax.set_xticklabels([models[i] for i in range(len(embedding_dims))])
    
    # Plot 3: Learning rate evolution  
    ax = axes[0, 2]
    learning_rates = [lr for lr in param_data['learning_rate'] if lr is not None]
    if learning_rates:
        ax.semilogy(range(len(learning_rates)), learning_rates, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Learning Rate (log scale)')
        ax.set_title('Learning Rate Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(learning_rates)))
        ax.set_xticklabels([models[i] for i in range(len(learning_rates))])
    
    # Plot 4: Network depth evolution
    ax = axes[0, 3]
    n_layers = [n for n in param_data['n_layers'] if n is not None]
    if n_layers:
        ax.plot(range(len(n_layers)), n_layers, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Number of Layers')
        ax.set_title('Network Depth Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(n_layers)))
        ax.set_xticklabels([models[i] for i in range(len(n_layers))])
        ax.set_ylim(bottom=0)
    
    # Plot 5: Attention heads evolution
    ax = axes[1, 0]
    n_heads = [n for n in param_data['n_heads'] if n is not None]
    if n_heads:
        ax.plot(range(len(n_heads)), n_heads, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Number of Attention Heads')
        ax.set_title('Attention Heads Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(n_heads)))
        ax.set_xticklabels([models[i] for i in range(len(n_heads))])
    
    # Plot 6: Dropout evolution
    ax = axes[1, 1]
    dropout = [d for d in param_data['dropout'] if d is not None]
    if dropout:
        ax.plot(range(len(dropout)), dropout, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Dropout Rate')
        ax.set_title('Dropout Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(dropout)))
        ax.set_xticklabels([models[i] for i in range(len(dropout))])
    
    # Plot 7: Temperature evolution
    ax = axes[1, 2]
    temperature = [t for t in param_data['temperature'] if t is not None]
    if temperature:
        ax.plot(range(len(temperature)), temperature, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Temperature')
        ax.set_title('RL Temperature Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(temperature)))
        ax.set_xticklabels([models[i] for i in range(len(temperature))])
    
    # Plot 8: Weight decay evolution
    ax = axes[1, 3]
    weight_decay = [wd for wd in param_data['weight_decay'] if wd is not None]
    if weight_decay:
        ax.semilogy(range(len(weight_decay)), weight_decay, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Weight Decay (log scale)')
        ax.set_title('Weight Decay Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(weight_decay)))
        ax.set_xticklabels([models[i] for i in range(len(weight_decay))])
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"sequential_optimization_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved as: {plot_filename}")
    
    plt.show()

def main():
    """Main analysis function."""
    try:
        # Load results
        results = load_latest_results()
        
        # Analyze results
        costs = analyze_results(results)
        
        # Create visualizations
        plot_results(results)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
