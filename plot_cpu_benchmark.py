#!/usr/bin/env python3
"""
Plot CPU benchmark results with 2-panel layout
Updated to handle modified CSV format with 'n', 'failed', 'timeout' columns
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_cpu_benchmark(csv_file, output_file='plots/cpu_benchmark.png'):
    """Create 2-panel visualization of benchmark results"""
    
    # Read data with column name handling
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
    print(f"Problem sizes: N = {df['n_customers'].min()} to {df['n_customers'].max()}\n")
    
    # Define solver configurations with display names
    solver_configs = {
        'ortools_greedy': {
            'name': 'OR-Tools Greedy',
            'color': '#1976D2',  # Blue  
            'marker': 'o',
            'linestyle': '--'
        },
        'ortools_gls': {
            'name': 'OR-Tools GLS (2s)',
            'color': '#F57C00',  # Orange
            'marker': '^',
            'linestyle': ':'
        },
        'exact_dp': {
            'name': 'Exact DP (N≤8)',
            'color': '#2E7D32',  # Green
            'marker': 's',
            'linestyle': '-'
        }
    }
    
    # Filter to successful runs only
    success_df = df[df['status'] == 'success'].copy()
    
    # Aggregate by solver and n_customers
    agg_df = success_df.groupby(['solver', 'n_customers']).agg({
        'time': ['mean', 'std', 'count'],
        'cpc': ['mean', 'std']
    }).reset_index()
    agg_df.columns = ['solver', 'n_customers', 'time_mean', 'time_std', 'count', 'cpc_mean', 'cpc_std']
    
    # Create figure with 2 subplots (vertical layout)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Top panel: Solve Time
    for solver_id, config in solver_configs.items():
        solver_data = agg_df[agg_df['solver'] == solver_id]
        if len(solver_data) > 0:
            x = solver_data['n_customers'].values
            y = solver_data['time_mean'].values
            y_err = solver_data['time_std'].values
            
            # Special handling for ortools_greedy to show the break at N=8/9
            if solver_id == 'ortools_greedy':
                # Split data at N=8
                exact_mask = x <= 8
                heuristic_mask = x >= 9
                
                # Plot exact portion (N≤8)
                if any(exact_mask):
                    ax1.errorbar(x[exact_mask], y[exact_mask], yerr=y_err[exact_mask],
                                label=config['name'] + ' (exact)',
                                color=config['color'],
                                marker=config['marker'],
                                markersize=8,
                                linewidth=2,
                                linestyle='-',  # Solid for exact
                                capsize=5,
                                capthick=1,
                                alpha=0.8)
                
                # Plot heuristic portion (N≥9)
                if any(heuristic_mask):
                    ax1.errorbar(x[heuristic_mask], y[heuristic_mask], yerr=y_err[heuristic_mask],
                                label=config['name'] + ' (heuristic)',
                                color=config['color'],
                                marker=config['marker'],
                                markersize=8,
                                linewidth=2,
                                linestyle='--',  # Dashed for heuristic
                                capsize=5,
                                capthick=1,
                                alpha=0.8)
            else:
                # Normal plotting for other solvers
                ax1.errorbar(x, y, yerr=y_err,
                            label=config['name'],
                            color=config['color'],
                            marker=config['marker'],
                            markersize=8,
                            linewidth=2,
                            linestyle=config['linestyle'],
                            capsize=5,
                            capthick=1,
                            alpha=0.8)
    
    ax1.set_yscale('log')
    # Remove x-label and tick labels from top panel
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Set x-axis ticks (even though labels are hidden, ticks help with alignment)
    ax1.set_xticks(range(5, 21, 1))
    ax1.set_xlim(4.5, 20.5)
    
    # Add minor gridlines
    ax1.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    # Add vertical line at N=8.5 to show the transition
    ax1.axvline(x=8.5, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Bottom panel: Cost per Customer
    for solver_id, config in solver_configs.items():
        solver_data = agg_df[agg_df['solver'] == solver_id]
        if len(solver_data) > 0:
            x = solver_data['n_customers'].values
            y = solver_data['cpc_mean'].values
            y_err = solver_data['cpc_std'].values
            
            # Special handling for ortools_greedy to show the break at N=8/9
            if solver_id == 'ortools_greedy':
                # Split data at N=8
                exact_mask = x <= 8
                heuristic_mask = x >= 9
                
                # Plot exact portion (N≤8)
                if any(exact_mask):
                    ax2.errorbar(x[exact_mask], y[exact_mask], yerr=y_err[exact_mask],
                                label=config['name'] + ' (exact)',
                                color=config['color'],
                                marker=config['marker'],
                                markersize=8,
                                linewidth=2,
                                linestyle='-',  # Solid for exact
                                capsize=5,
                                capthick=1,
                                alpha=0.8)
                
                # Plot heuristic portion (N≥9)
                if any(heuristic_mask):
                    ax2.errorbar(x[heuristic_mask], y[heuristic_mask], yerr=y_err[heuristic_mask],
                                label=config['name'] + ' (heuristic)',
                                color=config['color'],
                                marker=config['marker'],
                                markersize=8,
                                linewidth=2,
                                linestyle='--',  # Dashed for heuristic
                                capsize=5,
                                capthick=1,
                                alpha=0.8)
            else:
                # Normal plotting for other solvers
                ax2.errorbar(x, y, yerr=y_err,
                            label=config['name'],
                            color=config['color'],
                            marker=config['marker'],
                            markersize=8,
                            linewidth=2,
                            linestyle=config['linestyle'],
                            capsize=5,
                            capthick=1,
                            alpha=0.8)
    
    ax2.set_xlabel('Number of Customers (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cost per Customer (CPC)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Set x-axis ticks
    ax2.set_xticks(range(5, 21, 1))
    ax2.set_xlim(4.5, 20.5)
    
    # Set y-axis range for CPC
    ax2.set_ylim(0.2, 0.7)
    
    # Add vertical line at N=8.5 to show the transition
    ax2.axvline(x=8.5, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Single title for the entire figure
    fig.suptitle('CVRP Solver Performance', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout with less space between subplots
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for solver_id, config in solver_configs.items():
        solver_data = agg_df[agg_df['solver'] == solver_id]
        if len(solver_data) > 0:
            print(f"\n{config['name']}:")
            if solver_id == 'ortools_greedy':
                # Separate stats for exact and heuristic modes
                exact_data = solver_data[solver_data['n_customers'] <= 8]
                heuristic_data = solver_data[solver_data['n_customers'] >= 9]
                if len(exact_data) > 0:
                    print(f"  Exact mode (N≤8):")
                    print(f"    Average CPC: {exact_data['cpc_mean'].mean():.3f} ± {exact_data['cpc_mean'].std():.3f}")
                    print(f"    Average Time: {exact_data['time_mean'].mean():.2f}s")
                if len(heuristic_data) > 0:
                    print(f"  Heuristic mode (N≥9):")
                    print(f"    Average CPC: {heuristic_data['cpc_mean'].mean():.3f} ± {heuristic_data['cpc_mean'].std():.3f}")
                    print(f"    Average Time: {heuristic_data['time_mean'].mean():.2f}s")
            else:
                print(f"  Average CPC: {solver_data['cpc_mean'].mean():.3f} ± {solver_data['cpc_mean'].std():.3f}")
                print(f"  Average Time: {solver_data['time_mean'].mean():.2f}s")
                print(f"  N range: {solver_data['n_customers'].min()} to {solver_data['n_customers'].max()}")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CPU benchmark results")
    parser.add_argument('--csv', type=str, required=True, help='CSV file with benchmark results')
    parser.add_argument('--output', type=str, default='plots/cpu_benchmark.png', help='Output plot file')
    
    args = parser.parse_args()
    plot_cpu_benchmark(args.csv, args.output)
