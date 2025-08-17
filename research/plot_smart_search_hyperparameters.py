#!/usr/bin/env python3
"""
Smart Search Hyperparameters Analysis: Shows both Training and Validation costs clearly
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def setup_plot_style():
    """Setup publication-ready plot style"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def load_model_data():
    """Load training data for all models"""
    csv_dir = "results/smart_search/csv"
    
    model_info = {
        'GAT+RL': {
            'filename': 'history_gat_rl.csv',
            'color': '#1f77b4',
            'marker': 'o'
        },
        'DGT+RL': {
            'filename': 'history_dgt_rl.csv', 
            'color': '#ff7f0e',
            'marker': 's'
        },
        'GT+RL': {
            'filename': 'history_gt_rl.csv',
            'color': '#2ca02c',
            'marker': '^'
        }
    }
    
    model_data = {}
    for model_name, info in model_info.items():
        filepath = os.path.join(csv_dir, info['filename'])
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            model_data[model_name] = {
                'data': df,
                'info': info
            }
            print(f"Loaded {model_name}: {len(df)} training points")
        else:
            print(f"Could not find {filepath}")
    
    return model_data

def find_minima_both_costs(model_data):
    """Find minima for both training and validation costs"""
    results = {}
    
    for model_name, data in model_data.items():
        df = data['data']
        
        # Training cost minima
        train_costs = df['train_cost'].dropna()
        train_min_idx = np.argmin(train_costs)
        train_min_cost = train_costs.iloc[train_min_idx]
        
        # Validation cost minima
        val_data = df[df['val_cost'].notna()]
        if len(val_data) > 0:
            val_costs = val_data['val_cost']
            val_min_idx = np.argmin(val_costs)
            val_min_cost = val_costs.iloc[val_min_idx]
            val_min_epoch = val_data.iloc[val_min_idx]['epoch']
        else:
            val_min_cost = None
            val_min_epoch = None
        
        results[model_name] = {
            'train_min': (train_min_idx, train_min_cost),
            'val_min': (val_min_epoch, val_min_cost) if val_min_cost is not None else None
        }
        
        print(f"{model_name}:")
        print(f"  Training min: {train_min_cost:.6f} at epoch {train_min_idx}")
        if val_min_cost is not None:
            print(f"  Validation min: {val_min_cost:.6f} at epoch {val_min_epoch}")
        else:
            print("  No validation data")
    
    return results

def plot_corrected_analysis():
    """Create corrected analysis showing both cost types"""
    setup_plot_style()
    
    model_data = load_model_data()
    if not model_data:
        return
    
    minima_results = find_minima_both_costs(model_data)
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(24, 12))
    
    # 1. Training Cost vs Iteration
    ax1 = plt.subplot(2, 3, (1, 2))
    
    for model_name, data in model_data.items():
        df = data['data']
        info = data['info']
        
        # Plot training cost
        train_costs = df['train_cost'].values
        iterations = range(len(train_costs))
        
        ax1.plot(iterations, train_costs, 
                label=f"{model_name} (Training)", 
                color=info['color'], 
                linewidth=3, 
                alpha=0.8,
                linestyle='-')
        
        # Mark training minimum
        train_min_epoch, train_min_cost = minima_results[model_name]['train_min']
        ax1.plot(train_min_epoch, train_min_cost, 
                marker=info['marker'], 
                color=info['color'], 
                markersize=12, 
                markeredgecolor='white', 
                markeredgewidth=2,
                zorder=5)
        
        # Annotation for training minimum
        ax1.annotate(f'Train: {train_min_cost:.4f}',
                    xy=(train_min_epoch, train_min_cost),
                    xytext=(10, 15), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=info['color'], alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
                    fontsize=9, color='white', fontweight='bold')
    
    ax1.set_xlabel('Training Iteration (Epoch)', fontweight='bold')
    ax1.set_ylabel('Training Cost per Customer', fontweight='bold')
    ax1.set_title('Training Cost Evolution\\n(Continuous training data)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Cost vs Iteration
    ax2 = plt.subplot(2, 3, (4, 5))
    
    for model_name, data in model_data.items():
        df = data['data']
        info = data['info']
        
        # Plot validation cost (sparse data)
        val_data = df[df['val_cost'].notna()]
        if len(val_data) > 0:
            ax2.plot(val_data['epoch'], val_data['val_cost'], 
                    label=f"{model_name} (Validation)", 
                    color=info['color'], 
                    linewidth=3, 
                    alpha=0.8,
                    linestyle='--',
                    marker=info['marker'],
                    markersize=6)
            
            # Mark validation minimum
            val_min = minima_results[model_name]['val_min']
            if val_min is not None:
                val_min_epoch, val_min_cost = val_min
                ax2.plot(val_min_epoch, val_min_cost, 
                        marker=info['marker'], 
                        color=info['color'], 
                        markersize=12, 
                        markeredgecolor='white', 
                        markeredgewidth=2,
                        zorder=5)
                
                # Annotation for validation minimum
                ax2.annotate(f'Val: {val_min_cost:.4f}',
                            xy=(val_min_epoch, val_min_cost),
                            xytext=(10, 15), 
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=info['color'], alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
                            fontsize=9, color='white', fontweight='bold')
    
    ax2.set_xlabel('Training Iteration (Epoch)', fontweight='bold')
    ax2.set_ylabel('Validation Cost per Customer', fontweight='bold')
    ax2.set_title('Validation Cost Evolution\\n(Sparse evaluation data)', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Summary comparison - Split into multiple text boxes for better formatting
    ax3 = plt.subplot(2, 3, (3, 6))
    ax3.axis('off')
    
    # Title box
    title_text = "COST COMPARISON SUMMARY"
    ax3.text(0.5, 0.95, title_text,
             transform=ax3.transAxes,
             fontsize=14, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='darkblue', 
                      alpha=0.8,
                      edgecolor='navy',
                      linewidth=2),
             color='white')
    
    # Explanation box - more concise
    explanation_text = ("DISCREPANCY EXPLANATION:\n"
                       "• quick_comparison: VALIDATION cost\n"
                       "• cost_minima_analysis: TRAINING cost")
    ax3.text(0.05, 0.82, explanation_text,
             transform=ax3.transAxes,
             fontsize=9, fontweight='normal',
             ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', 
                      facecolor='lightcoral', 
                      alpha=0.7,
                      edgecolor='red',
                      linewidth=1.5))
    
    # Training minima box
    train_text = "TRAINING COST MINIMA:\n"
    for model_name, minima in minima_results.items():
        train_min_epoch, train_min_cost = minima['train_min']
        train_text += f"{model_name}: {train_min_cost:.4f} @epoch {train_min_epoch}\n"
    
    ax3.text(0.05, 0.62, train_text.strip(),
             transform=ax3.transAxes,
             fontsize=10, fontfamily='monospace',
             ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.4', 
                      facecolor='lightblue', 
                      alpha=0.7,
                      edgecolor='blue',
                      linewidth=1.5))
    
    # Validation minima box
    val_text = "VALIDATION COST MINIMA:\n"
    for model_name, minima in minima_results.items():
        if minima['val_min'] is not None:
            val_min_epoch, val_min_cost = minima['val_min']
            val_text += f"{model_name}: {val_min_cost:.4f} @epoch {val_min_epoch:.0f}\n"
        else:
            val_text += f"{model_name}: No validation data\n"
    
    ax3.text(0.05, 0.40, val_text.strip(),
             transform=ax3.transAxes,
             fontsize=10, fontfamily='monospace',
             ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.4', 
                      facecolor='lightgreen', 
                      alpha=0.7,
                      edgecolor='green',
                      linewidth=1.5))
    
    # Key insight box - more concise
    insight_text = ("KEY INSIGHTS:\n"
                   "• Validation < Training costs\n"
                   "• Good generalization\n"
                   "• Best val: GAT+RL (0.484)\n"
                   "• Best train: DGT+RL (0.636)")
    ax3.text(0.05, 0.18, insight_text,
             transform=ax3.transAxes,
             fontsize=9, fontweight='bold',
             ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', 
                      facecolor='lightyellow', 
                      alpha=0.8,
                      edgecolor='orange',
                      linewidth=2))
    
    # Manual spacing instead of tight_layout to avoid warnings
    plt.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.92, 
                       wspace=0.15, hspace=0.25)
    
    # Save
    plots_dir = "results/smart_search/plots"
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "smart_search_hyperparameters.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\nSmart search hyperparameters analysis saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    print("Creating smart search hyperparameters analysis...")
    plot_corrected_analysis()
    print("Done!")
