#!/usr/bin/env python3
"""
Plot results from benchmark_exact_cpu_modified.py
Creates a 2x2 subplot figure showing:
1. Cost per Customer (CPC) vs N
2. Solve Time vs N (log scale)
3. Success Rate vs N
4. Solution Quality Comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def load_latest_results():
    """Load the most recent benchmark results"""
    results_dir = Path('results/csv')
    
    # Find the latest benchmark_modified file
    csv_files = list(results_dir.glob('benchmark_modified_*.csv'))
    if not csv_files:
        # Try the one from the quick test
        csv_files = list(results_dir.glob('heuristic_2s_600instances.csv'))
    
    if not csv_files:
        print("No benchmark results found in results/csv/")
        sys.exit(1)
    
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df, latest_file.stem

def aggregate_results(df):
    """Aggregate results by N and solver"""
    agg_dict = {
        'cost': 'mean',
        'cpc': 'mean',
        'time': 'mean',
        'timeout': 'sum',
        'failed': 'sum'
    }
    
    grouped = df.groupby(['n', 'solver']).agg(agg_dict).reset_index()
    
    # Count total instances per N/solver
    counts = df.groupby(['n', 'solver']).size().reset_index(name='count')
    grouped = grouped.merge(counts, on=['n', 'solver'])
    
    # Calculate success rate
    grouped['success_rate'] = ((grouped['count'] - grouped['timeout'] - grouped['failed']) / grouped['count']) * 100
    
    return grouped

def create_plots(agg_df, title_suffix=""):
    """Create 2x2 subplot figure"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'CVRP Benchmark Results{title_suffix}', fontsize=16, fontweight='bold')
    
    # Color scheme for solvers
    colors = {
        'exact_dp': '#1f77b4',  # blue
        'ortools_greedy': '#ff7f0e',  # orange  
        'ortools_gls': '#2ca02c'  # green
    }
    
    # Labels for legend
    labels = {
        'exact_dp': 'Exact DP',
        'ortools_greedy': 'OR-Tools Greedy',
        'ortools_gls': 'OR-Tools GLS (2s)'
    }
    
    # Get unique N values
    n_values = sorted(agg_df['n'].unique())
    
    # Plot 1: Cost per Customer vs N
    ax1 = axes[0, 0]
    for solver in colors.keys():
        solver_data = agg_df[agg_df['solver'] == solver]
        if not solver_data.empty:
            ax1.plot(solver_data['n'], solver_data['cpc'], 
                    marker='o', color=colors[solver], label=labels[solver],
                    linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Customers (N)', fontsize=11)
    ax1.set_ylabel('Cost per Customer (CPC)', fontsize=11)
    ax1.set_title('Solution Quality', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xticks(n_values)
    
    # Plot 2: Solve Time vs N (log scale)
    ax2 = axes[0, 1]
    for solver in colors.keys():
        solver_data = agg_df[agg_df['solver'] == solver]
        if not solver_data.empty:
            # Only plot where success_rate > 0
            valid_data = solver_data[solver_data['success_rate'] > 0]
            if not valid_data.empty:
                ax2.plot(valid_data['n'], valid_data['time'],
                        marker='o', color=colors[solver], label=labels[solver],
                        linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Customers (N)', fontsize=11)
    ax2.set_ylabel('Average Solve Time (seconds)', fontsize=11)
    ax2.set_title('Computational Time', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper left')
    ax2.set_xticks(n_values)
    
    # Add horizontal line at 2s for reference
    ax2.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='2s timeout')
    
    # Plot 3: Success Rate vs N
    ax3 = axes[1, 0]
    for solver in colors.keys():
        solver_data = agg_df[agg_df['solver'] == solver]
        if not solver_data.empty:
            ax3.plot(solver_data['n'], solver_data['success_rate'],
                    marker='o', color=colors[solver], label=labels[solver],
                    linewidth=2, markersize=8)
    
    ax3.set_xlabel('Number of Customers (N)', fontsize=11)
    ax3.set_ylabel('Success Rate (%)', fontsize=11)
    ax3.set_title('Solver Reliability', fontsize=12, fontweight='bold')
    ax3.set_ylim(-5, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower left')
    ax3.set_xticks(n_values)
    ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    
    # Plot 4: Bar chart comparing CPC at different N values
    ax4 = axes[1, 1]
    selected_n = [5, 8, 12, 20]  # Key N values
    bar_width = 0.25
    x_positions = np.arange(len(selected_n))
    
    for i, solver in enumerate(colors.keys()):
        solver_data = agg_df[agg_df['solver'] == solver]
        cpc_values = []
        for n in selected_n:
            n_data = solver_data[solver_data['n'] == n]
            if not n_data.empty and n_data['success_rate'].values[0] > 0:
                cpc_values.append(n_data['cpc'].values[0])
            else:
                cpc_values.append(0)  # No data
        
        ax4.bar(x_positions + i * bar_width, cpc_values, bar_width,
               label=labels[solver], color=colors[solver], alpha=0.8)
    
    ax4.set_xlabel('Number of Customers (N)', fontsize=11)
    ax4.set_ylabel('Cost per Customer (CPC)', fontsize=11)
    ax4.set_title('CPC Comparison at Key Problem Sizes', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_positions + bar_width)
    ax4.set_xticklabels(selected_n)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def main():
    # Load results
    df, filename = load_latest_results()
    
    # Aggregate by N and solver
    agg_df = aggregate_results(df)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for n in sorted(agg_df['n'].unique()):
        print(f"\nN={n}:")
        n_data = agg_df[agg_df['n'] == n]
        for _, row in n_data.iterrows():
            if row['success_rate'] > 0:
                print(f"  {row['solver']:15s}: CPC={row['cpc']:.4f}, time={row['time']:.3f}s, success={row['success_rate']:.0f}%")
            else:
                print(f"  {row['solver']:15s}: failed/skipped")
    
    # Create plots
    title_suffix = f" (N≤8 exact, GLS=2s)"
    fig = create_plots(agg_df, title_suffix)
    
    # Save figure
    output_file = 'plots/benchmark_modified.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to {output_file}")
    
    # Also save as PDF for papers
    pdf_file = 'plots/benchmark_modified.pdf'
    fig.savefig(pdf_file, bbox_inches='tight')
    print(f"✅ PDF version saved to {pdf_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
