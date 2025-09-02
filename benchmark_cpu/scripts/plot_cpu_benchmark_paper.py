#!/usr/bin/env python3
"""
Production-quality CPU benchmark plotting script for paper publication
Creates publication-ready plots with specific formatting requirements
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
from pathlib import Path

def setup_publication_style():
    """Set up matplotlib parameters for publication quality"""
    # Set up publication-quality parameters
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 0.8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'lightgray',
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'patch.linewidth': 0.8,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.constrained_layout.use': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })

def plot_cpu_benchmark_paper(csv_file, output_base='benchmark'):
    """Create production-quality visualization of benchmark results"""
    
    # Set up publication style
    setup_publication_style()
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Handle column name differences
    if 'n' in df.columns:
        df = df.rename(columns={'n': 'n_customers'})
    
    # Create status column from failed/timeout if needed
    if 'status' not in df.columns and 'failed' in df.columns:
        def get_status(row):
            if row['failed']:
                return 'failed'
            elif row.get('timeout', False):
                return 'timeout'
            else:
                return 'success'
        df['status'] = df.apply(get_status, axis=1)
    
    print(f"📊 Loaded CPU benchmark data with {len(df)} data points")
    print(f"Problem sizes: N = {df['n_customers'].min()} to {df['n_customers'].max()}")
    
    # Define solver configurations with publication-quality styling
    solver_configs = {
        'ortools_greedy': {
            'name': 'OR-Tools Greedy',
            'color': '#1f77b4',  # Professional blue
            'marker': 'o',
            'linestyle': '-'
        },
        'ortools_gls': {
            'name': 'OR-Tools GLS (2s)',
            'color': '#ff7f0e',  # Professional orange
            'marker': '^',
            'linestyle': ':'
        },
        'exact_dp': {
            'name': 'Exact DP',
            'color': '#2ca02c',  # Professional green
            'marker': 's',
            'linestyle': '-'
        }
    }
    
    # Filter to successful runs only
    success_df = df[df['status'] == 'success'].copy()
    
    # Aggregate by solver and n_customers (mean and std for error bars)
    agg_df = success_df.groupby(['solver', 'n_customers']).agg({
        'time': ['mean', 'std'],
        'cpc': ['mean', 'std']
    }).reset_index()
    agg_df.columns = ['solver', 'n_customers', 'time_mean', 'time_std', 'cpc_mean', 'cpc_std']
    
    # Convert mm to inches for matplotlib (1 mm = 0.0393701 inches)
    fig_width = 160 * 0.0393701  # 160mm to inches
    fig_height = 120 * 0.0393701  # 120mm to inches
    
    # Create figure with specified size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    # Top panel: Solve Time
    for solver_id, config in solver_configs.items():
        solver_data = agg_df[agg_df['solver'] == solver_id]
        if len(solver_data) == 0:
            continue
            
        x = solver_data['n_customers'].values
        y = solver_data['time_mean'].values
        
        if solver_id == 'ortools_greedy':
            # Split data at N=8 for OR-Tools Greedy
            exact_mask = x <= 8
            heuristic_mask = x >= 9
            
            # Plot exact portion (N≤8)
            if any(exact_mask):
                ax1.plot(x[exact_mask], y[exact_mask],
                        color=config['color'],
                        marker=config['marker'],
                        linestyle='-',  # Solid for exact
                        label=f"{config['name']} (exact)",
                        markerfacecolor='white',
                        markeredgecolor=config['color'],
                        markeredgewidth=1)
            
            # Plot heuristic portion (N≥9)
            if any(heuristic_mask):
                ax1.plot(x[heuristic_mask], y[heuristic_mask],
                        color=config['color'],
                        marker=config['marker'],
                        linestyle='--',  # Dashed for heuristic
                        label=f"{config['name']} (heuristic)",
                        markerfacecolor=config['color'],
                        markeredgecolor=config['color'])
        
        elif solver_id == 'ortools_gls':
            # For OR-Tools GLS, use horizontal line (axhline) with dotted style, no markers
            mean_time = y.mean()  # Average time across all problem sizes
            ax1.axhline(y=mean_time, 
                       color=config['color'],
                       linestyle=':',  # Dotted line
                       linewidth=1.2,
                       label=config['name'])
        else:
            # Normal plotting for other solvers (Exact DP)
            ax1.plot(x, y,
                    color=config['color'],
                    marker=config['marker'],
                    linestyle=config['linestyle'],
                    label=config['name'],
                    markerfacecolor='white',
                    markeredgecolor=config['color'],
                    markeredgewidth=1)
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Solve time, s')
    ax1.tick_params(axis='x', labelbottom=False)  # Remove x-axis labels from top panel
    ax1.set_xticks(range(5, 21, 1))
    ax1.set_xlim(4.5, 20.5)
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Add vertical line at N=8.5 to show transition
    ax1.axvline(x=8.5, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    
    # Bottom panel: Cost per Customer
    for solver_id, config in solver_configs.items():
        solver_data = agg_df[agg_df['solver'] == solver_id]
        if len(solver_data) == 0:
            continue
            
        x = solver_data['n_customers'].values
        y = solver_data['cpc_mean'].values
        y_err = solver_data['cpc_std'].values
        
        if solver_id == 'ortools_greedy':
            # Split data at N=8 for OR-Tools Greedy
            exact_mask = x <= 8
            heuristic_mask = x >= 9
            
            # Plot exact portion (N≤8) with error bars
            if any(exact_mask):
                ax2.errorbar(x[exact_mask], y[exact_mask], yerr=y_err[exact_mask],
                        color=config['color'],
                        marker=config['marker'],
                        linestyle='-',  # Solid for exact
                        label=f"{config['name']} (exact)",
                        markerfacecolor='white',
                        markeredgecolor=config['color'],
                        markeredgewidth=1,
                        capsize=2,
                        capthick=0.8)
            
            # Plot heuristic portion (N≥9) with error bars
            if any(heuristic_mask):
                ax2.errorbar(x[heuristic_mask], y[heuristic_mask], yerr=y_err[heuristic_mask],
                        color=config['color'],
                        marker=config['marker'],
                        linestyle='--',  # Dashed for heuristic
                        label=f"{config['name']} (heuristic)",
                        markerfacecolor=config['color'],
                        markeredgecolor=config['color'],
                        capsize=2,
                        capthick=0.8)
        else:
            # Normal plotting for other solvers with error bars
            ax2.errorbar(x, y, yerr=y_err,
                    color=config['color'],
                    marker=config['marker'],
                    linestyle=config['linestyle'],
                    label=config['name'],
                    markerfacecolor='white',
                    markeredgecolor=config['color'],
                    markeredgewidth=1,
                    capsize=2,
                    capthick=0.8)
    
    ax2.set_xlabel('Number of customers (N)')
    ax2.set_ylabel('CPC')
    ax2.set_xticks(range(5, 21, 1))
    ax2.set_xlim(4.5, 20.5)
    ax2.legend(loc='upper right', framealpha=0.9)
    
    # Set reasonable y-axis range for CPC
    ax2.set_ylim(0.2, 0.7)
    
    # Add vertical line at N=8.5 to show transition
    ax2.axvline(x=8.5, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    
    # Adjust spacing between panels
    plt.subplots_adjust(hspace=0.1)
    
    # Save in both PNG and EPS formats
    png_file = f"{output_base}.png"
    eps_file = f"{output_base}.eps"
    
    # Save PNG with high DPI
    plt.savefig(png_file, dpi=300, format='png', bbox_inches='tight', pad_inches=0.05)
    print(f"✅ PNG plot saved to: {png_file}")
    
    # Save EPS (vector format)
    plt.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=0.05)
    print(f"✅ EPS plot saved to: {eps_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for solver_id, config in solver_configs.items():
        solver_data = agg_df[agg_df['solver'] == solver_id]
        if len(solver_data) > 0:
            print(f"\n{config['name']}:")
            if solver_id == 'ortools_greedy':
                # Separate stats for exact and heuristic modes
                exact_data = solver_data[solver_data['n_customers'] <= 8]
                heuristic_data = solver_data[solver_data['n_customers'] >= 9]
                if len(exact_data) > 0:
                    print(f"  Exact mode (N≤8): Avg CPC: {exact_data['cpc_mean'].mean():.3f}, Avg Time: {exact_data['time_mean'].mean():.2f}s")
                if len(heuristic_data) > 0:
                    print(f"  Heuristic mode (N≥9): Avg CPC: {heuristic_data['cpc_mean'].mean():.3f}, Avg Time: {heuristic_data['time_mean'].mean():.2f}s")
            else:
                print(f"  Average CPC: {solver_data['cpc_mean'].mean():.3f}")
                print(f"  Average Time: {solver_data['time_mean'].mean():.2f}s")
                print(f"  N range: {solver_data['n_customers'].min()} to {solver_data['n_customers'].max()}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create production-quality CPU benchmark plot")
    parser.add_argument('--csv', type=str, required=True, help='CSV file with benchmark results')
    parser.add_argument('--output', type=str, default='benchmark', help='Output file base name (without extension)')
    
    args = parser.parse_args()
    plot_cpu_benchmark_paper(args.csv, args.output)
