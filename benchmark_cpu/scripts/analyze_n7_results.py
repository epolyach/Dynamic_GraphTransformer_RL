#!/usr/bin/env python3
"""
Analysis and visualization script for N=7 CPC benchmark results.
Creates plots and detailed statistical analysis.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(csv_path: str) -> pd.DataFrame:
    """Load and preprocess results from CSV."""
    df = pd.read_csv(csv_path)
    
    # Convert CPC columns to numeric
    for col in ['exact_dp_cpc', 'exact_ortools_cpc', 'heuristic_nn_cpc', 'heuristic_sts_cpc_cpc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert time columns to numeric
    for col in ['exact_dp_time', 'exact_ortools_time', 'heuristic_nn_time', 'heuristic_sts_cpc_time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert gap columns to numeric
    df['nn_gap_percent'] = pd.to_numeric(df['nn_gap_percent'], errors='coerce')
    df['sts_gap_percent'] = pd.to_numeric(df['sts_gap_percent'], errors='coerce')
    
    return df


def create_cpc_distribution_plot(df: pd.DataFrame, output_dir: Path):
    """Create box plots showing CPC distributions for all solvers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for box plot
    solver_data = []
    solver_labels = []
    
    if not df['exact_dp_cpc'].isna().all():
        solver_data.append(df['exact_dp_cpc'].dropna())
        solver_labels.append('Exact DP\n(Optimal)')
    
    if not df['exact_ortools_cpc'].isna().all():
        solver_data.append(df['exact_ortools_cpc'].dropna())
        solver_labels.append('Exact OR-Tools\n(Optimal)')
    
    if not df['heuristic_nn_cpc'].isna().all():
        solver_data.append(df['heuristic_nn_cpc'].dropna())
        solver_labels.append('Nearest\nNeighbor')
    
    if not df['heuristic_sts_cpc_cpc'].isna().all():
        solver_data.append(df['heuristic_sts_cpc_cpc'].dropna())
        solver_labels.append('STS-CPC')
    
    # Box plot of CPC values
    bp = ax1.boxplot(solver_data, labels=solver_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'salmon', 'gold']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_ylabel('Cost Per Customer (CPC)', fontsize=12)
    ax1.set_title('CPC Distribution by Solver', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Violin plot for better distribution visualization
    positions = range(1, len(solver_data) + 1)
    parts = ax2.violinplot(solver_data, positions=positions, showmeans=True, showmedians=True)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(solver_labels)
    ax2.set_ylabel('Cost Per Customer (CPC)', fontsize=12)
    ax2.set_title('CPC Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cpc_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_dir / 'cpc_distributions.png'}")


def create_optimality_gap_plot(df: pd.DataFrame, output_dir: Path):
    """Create histogram of optimality gaps for heuristics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Nearest Neighbor gaps
    nn_gaps = df['nn_gap_percent'].dropna()
    if len(nn_gaps) > 0:
        ax1.hist(nn_gaps, bins=30, color='salmon', alpha=0.7, edgecolor='black')
        ax1.axvline(nn_gaps.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {nn_gaps.mean():.2f}%')
        ax1.set_xlabel('Optimality Gap (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Nearest Neighbor Optimality Gap Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # STS-CPC gaps
    sts_gaps = df['sts_gap_percent'].dropna()
    if len(sts_gaps) > 0:
        ax2.hist(sts_gaps, bins=30, color='gold', alpha=0.7, edgecolor='black')
        ax2.axvline(sts_gaps.mean(), color='orange', linestyle='--', linewidth=2,
                   label=f'Mean: {sts_gaps.mean():.2f}%')
        ax2.set_xlabel('Optimality Gap (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('STS-CPC Optimality Gap Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimality_gaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_dir / 'optimality_gaps.png'}")


def create_time_comparison_plot(df: pd.DataFrame, output_dir: Path):
    """Create comparison of computation times."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare time data
    time_data = {
        'Exact DP': df['exact_dp_time'].dropna(),
        'Exact OR-Tools': df['exact_ortools_time'].dropna(),
        'Nearest Neighbor': df['heuristic_nn_time'].dropna(),
        'STS-CPC': df['heuristic_sts_cpc_time'].dropna()
    }
    
    # Create log-scale box plot
    data_to_plot = []
    labels = []
    for label, data in time_data.items():
        if len(data) > 0:
            data_to_plot.append(data)
            labels.append(label)
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'salmon', 'gold']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax.set_yscale('log')
    ax.set_ylabel('Computation Time (seconds, log scale)', fontsize=12)
    ax.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add mean time annotations
    for i, (label, data) in enumerate(time_data.items(), 1):
        if len(data) > 0:
            mean_time = data.mean()
            ax.text(i, mean_time, f'{mean_time:.4f}s', 
                   horizontalalignment='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_dir / 'time_comparison.png'}")


def create_performance_scatter(df: pd.DataFrame, output_dir: Path):
    """Create scatter plot of solution quality vs computation time."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Get optimal CPC for each instance (from exact_dp)
    optimal_cpc = df['exact_dp_cpc'].fillna(df['exact_ortools_cpc'])
    
    # Plot each solver
    solvers = [
        ('exact_dp', 'Exact DP', 'blue', 'o'),
        ('exact_ortools', 'Exact OR-Tools', 'green', 's'),
        ('heuristic_nn', 'Nearest Neighbor', 'red', '^'),
        ('heuristic_sts_cpc', 'STS-CPC', 'orange', 'D')
    ]
    
    for solver_prefix, label, color, marker in solvers:
        cpc_col = f'{solver_prefix}_cpc' if solver_prefix != 'heuristic_sts_cpc' else 'heuristic_sts_cpc_cpc'
        time_col = f'{solver_prefix}_time'
        
        # Calculate relative error
        solver_cpc = df[cpc_col]
        solver_time = df[time_col]
        relative_error = 100 * (solver_cpc - optimal_cpc) / optimal_cpc
        
        # Filter valid data
        mask = ~(solver_cpc.isna() | solver_time.isna() | relative_error.isna())
        
        if mask.sum() > 0:
            ax.scatter(solver_time[mask], relative_error[mask], 
                      alpha=0.6, color=color, marker=marker, label=label, s=20)
    
    ax.set_xscale('log')
    ax.set_xlabel('Computation Time (seconds, log scale)', fontsize=12)
    ax.set_ylabel('Relative Error (% above optimal)', fontsize=12)
    ax.set_title('Solution Quality vs Computation Time Trade-off', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_dir / 'performance_scatter.png'}")


def print_detailed_statistics(df: pd.DataFrame):
    """Print detailed statistical analysis."""
    print("\n" + "="*70)
    print("DETAILED STATISTICAL ANALYSIS")
    print("="*70)
    
    # Overall statistics
    print(f"\nTotal instances analyzed: {len(df)}")
    print(f"Instances with N=7 customers: {(df['n_customers'] == 7).sum()}")
    
    # CPC statistics by solver
    print("\n" + "-"*50)
    print("CPC STATISTICS BY SOLVER")
    print("-"*50)
    
    solvers = [
        ('exact_dp_cpc', 'Exact DP'),
        ('exact_ortools_cpc', 'Exact OR-Tools'),
        ('heuristic_nn_cpc', 'Nearest Neighbor'),
        ('heuristic_sts_cpc_cpc', 'STS-CPC')
    ]
    
    for col, name in solvers:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"\n{name}:")
            print(f"  Count:     {len(data)}")
            print(f"  Mean CPC:  {data.mean():.6f}")
            print(f"  Std Dev:   {data.std():.6f}")
            print(f"  Minimum:   {data.min():.6f}")
            print(f"  Q1 (25%):  {data.quantile(0.25):.6f}")
            print(f"  Median:    {data.median():.6f}")
            print(f"  Q3 (75%):  {data.quantile(0.75):.6f}")
            print(f"  Maximum:   {data.max():.6f}")
    
    # Optimality gap statistics
    print("\n" + "-"*50)
    print("OPTIMALITY GAP STATISTICS")
    print("-"*50)
    
    for col, name in [('nn_gap_percent', 'Nearest Neighbor'), ('sts_gap_percent', 'STS-CPC')]:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"\n{name} Gap (% above optimal):")
            print(f"  Count:           {len(data)}")
            print(f"  Mean:            {data.mean():.2f}%")
            print(f"  Std Dev:         {data.std():.2f}%")
            print(f"  Minimum:         {data.min():.2f}%")
            print(f"  Q1 (25%):        {data.quantile(0.25):.2f}%")
            print(f"  Median:          {data.median():.2f}%")
            print(f"  Q3 (75%):        {data.quantile(0.75):.2f}%")
            print(f"  Maximum:         {data.max():.2f}%")
            print(f"  % within 10%:    {(data <= 10).mean() * 100:.1f}%")
            print(f"  % within 20%:    {(data <= 20).mean() * 100:.1f}%")
    
    # Time statistics
    print("\n" + "-"*50)
    print("COMPUTATION TIME STATISTICS")
    print("-"*50)
    
    time_cols = [
        ('exact_dp_time', 'Exact DP'),
        ('exact_ortools_time', 'Exact OR-Tools'),
        ('heuristic_nn_time', 'Nearest Neighbor'),
        ('heuristic_sts_cpc_time', 'STS-CPC')
    ]
    
    for col, name in time_cols:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"\n{name} Time (seconds):")
            print(f"  Mean:     {data.mean():.6f}")
            print(f"  Std Dev:  {data.std():.6f}")
            print(f"  Minimum:  {data.min():.6f}")
            print(f"  Median:   {data.median():.6f}")
            print(f"  Maximum:  {data.max():.6f}")
    
    # Correlation analysis
    print("\n" + "-"*50)
    print("CORRELATION ANALYSIS")
    print("-"*50)
    
    # Check if heuristic quality correlates with optimal CPC
    optimal_cpc = df['exact_dp_cpc'].fillna(df['exact_ortools_cpc'])
    
    if not df['nn_gap_percent'].isna().all():
        corr = optimal_cpc.corr(df['nn_gap_percent'])
        print(f"\nCorrelation between optimal CPC and NN gap: {corr:.4f}")
    
    if not df['sts_gap_percent'].isna().all():
        corr = optimal_cpc.corr(df['sts_gap_percent'])
        print(f"Correlation between optimal CPC and STS gap: {corr:.4f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze N=7 CPC benchmark results")
    parser.add_argument("--csv", type=str, 
                       default="benchmark_cpu/results/csv/n7_solver_comparison.csv",
                       help="Input CSV file path")
    parser.add_argument("--output-dir", type=str,
                       default="benchmark_cpu/plots",
                       help="Output directory for plots")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from: {args.csv}")
    df = load_results(args.csv)
    
    # Print detailed statistics
    print_detailed_statistics(df)
    
    # Create visualizations
    if not args.no_plots:
        print("\nGenerating visualizations...")
        create_cpc_distribution_plot(df, output_dir)
        create_optimality_gap_plot(df, output_dir)
        create_time_comparison_plot(df, output_dir)
        create_performance_scatter(df, output_dir)
        print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
