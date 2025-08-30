#!/usr/bin/env python3
"""
Plot CPU benchmark results in the same style as GPU benchmark.
Creates 2-panel plot: Time per instance and Cost per Customer.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_cpu_benchmark(csv_file: str = "results/csv/cpu_benchmark.csv", 
                       output_dir: str = "plots",
                       title: str = None):
    """
    Create benchmark visualization plots from CSV results.
    Matches the style of plot_gpu_benchmark.py with 2 panels.
    """
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded CPU benchmark data with {len(df)} data points")
        print(f"Problem sizes: N = {df['n_customers'].min()} to {df['n_customers'].max()}")
        print()
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_file}")
        return
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics per solver and n
    stats = []
    for solver in df['solver'].unique():
        for n in df['n_customers'].unique():
            solver_df = df[(df['solver'] == solver) & (df['n_customers'] == n)]
            success_df = solver_df[solver_df['status'] == 'success']
            
            if len(success_df) > 0:
                stats.append({
                    'solver': solver,
                    'n_customers': n,
                    'avg_cpc': success_df['cpc'].mean(),
                    'std_cpc': success_df['cpc'].std(),
                    'avg_time': success_df['time'].mean(),
                    'std_time': success_df['time'].std(),
                    'success_count': len(success_df),
                    'total_count': len(solver_df)
                })
    
    stats_df = pd.DataFrame(stats)
    
    # Set up plot style to match GPU benchmark
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    
    # Create 2 vertical subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    else:
        total_instances = len(df['instance_id'].unique()) * len(df['n_customers'].unique())
        fig.suptitle(f'CPU CVRP Solver Performance Comparison\n{total_instances} Tasks • 3 Solver Types', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    # Define solver info matching the style
    solver_info = {
        'exact_dp': {
            'label': 'Exact (DP, N≤10)',
            'color': '#1f77b4',  # Blue
            'marker': 'o'
        },
        'exact_ortools_vrp': {
            'label': 'Metaheuristic (OR-Tools)',
            'color': '#ff7f0e',  # Orange
            'marker': 's'
        },
        'heuristic_or': {
            'label': 'Heuristic (OR-Tools)',
            'color': '#2ca02c',  # Green
            'marker': '^'
        }
    }
    
    # Panel 1: Execution Time vs Problem Size (Log Scale)
    for solver_key, info in solver_info.items():
        solver_stats = stats_df[stats_df['solver'] == solver_key]
        if not solver_stats.empty:
            ax1.semilogy(solver_stats['n_customers'], solver_stats['avg_time'], 
                        info['marker'] + '-', color=info['color'], linewidth=3,
                        markersize=8, label=info['label'], alpha=0.8,
                        markerfacecolor='white', markeredgewidth=2, markeredgecolor=info['color'])
    
    ax1.set_ylabel('Time per Instance (seconds)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper right')
    
    # Panel 2: Cost per Customer vs Problem Size
    for solver_key, info in solver_info.items():
        solver_stats = stats_df[stats_df['solver'] == solver_key]
        if not solver_stats.empty:
            ax2.plot(solver_stats['n_customers'], solver_stats['avg_cpc'],
                    info['marker'] + '-', color=info['color'], linewidth=3,
                    markersize=8, label=info['label'], alpha=0.8,
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=info['color'])
            
            # Add error bars using standard error of the mean
            sems = []
            for _, row in solver_stats.iterrows():
                if row['success_count'] > 0 and not np.isnan(row['std_cpc']):
                    sem = row['std_cpc'] / np.sqrt(row['success_count'])
                    sems.append(sem)
                else:
                    sems.append(0.0)
            
            if sems:
                ax2.errorbar(solver_stats['n_customers'], solver_stats['avg_cpc'], 
                           yerr=sems, fmt='none', color=info['color'], 
                           alpha=0.5, capsize=4, capthick=1)
    
    ax2.set_xlabel('Number of Customers (N)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cost per Customer', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Set x-ticks for both panels
    n_values = sorted(df['n_customers'].unique())
    ax1.set_xticks(n_values)
    ax2.set_xticks(n_values)
    
    # Remove x-axis labels from top panel
    ax1.set_xticklabels([])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot (matching GPU benchmark naming)
    output_path = Path(output_dir) / "cpu_benchmark.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for solver_key, info in solver_info.items():
        solver_stats = stats_df[stats_df['solver'] == solver_key]
        if not solver_stats.empty:
            print(f"\n{info['label']}:")
            print(f"  Average CPC: {solver_stats['avg_cpc'].mean():.3f} ± {solver_stats['avg_cpc'].std():.3f}")
            print(f"  Average Time: {solver_stats['avg_time'].mean():.2f}s")
            print(f"  N range: {solver_stats['n_customers'].min():.0f} to {solver_stats['n_customers'].max():.0f}")


def plot_cpu_benchmark_detailed(csv_file: str = "results/csv/cpu_benchmark.csv",
                                output_dir: str = "plots"):
    """
    Create detailed analysis plots (6 panels) for deeper insights.
    This is the previous comprehensive version.
    """
    import seaborn as sns
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Calculate statistics
    stats = []
    for solver in df['solver'].unique():
        for n in df['n_customers'].unique():
            solver_df = df[(df['solver'] == solver) & (df['n_customers'] == n)]
            success_df = solver_df[solver_df['status'] == 'success']
            
            if len(success_df) > 0:
                stats.append({
                    'solver': solver,
                    'n_customers': n,
                    'avg_cpc': success_df['cpc'].mean(),
                    'std_cpc': success_df['cpc'].std(),
                    'avg_time': success_df['time'].mean(),
                    'std_time': success_df['time'].std(),
                    'success_rate': len(success_df) / len(solver_df) * 100,
                    'num_success': len(success_df)
                })
    
    stats_df = pd.DataFrame(stats)
    
    # Create 6-panel figure
    fig = plt.figure(figsize=(16, 10))
    
    solver_info = {
        'exact_dp': {'label': 'Exact (DP, N≤10)', 'color': '#2E7D32', 'marker': 'o'},
        'exact_ortools_vrp': {'label': 'Metaheuristic', 'color': '#1565C0', 'marker': 's'},
        'heuristic_or': {'label': 'Heuristic', 'color': '#E65100', 'marker': '^'}
    }
    
    # Panel 1: Time complexity
    ax1 = plt.subplot(2, 3, 1)
    for solver, info in solver_info.items():
        solver_stats = stats_df[stats_df['solver'] == solver]
        if not solver_stats.empty:
            ax1.plot(solver_stats['n_customers'], solver_stats['avg_time'],
                    label=info['label'], color=info['color'], marker=info['marker'],
                    linewidth=2, markersize=8)
    ax1.set_yscale('log')
    ax1.set_xlabel('N')
    ax1.set_ylabel('Time (s, log)')
    ax1.set_title('Time Complexity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: CPC evolution
    ax2 = plt.subplot(2, 3, 2)
    for solver, info in solver_info.items():
        solver_stats = stats_df[stats_df['solver'] == solver]
        if not solver_stats.empty:
            ax2.plot(solver_stats['n_customers'], solver_stats['avg_cpc'],
                    label=info['label'], color=info['color'], marker=info['marker'],
                    linewidth=2, markersize=8)
    ax2.set_xlabel('N')
    ax2.set_ylabel('CPC')
    ax2.set_title('Solution Quality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Success rate
    ax3 = plt.subplot(2, 3, 3)
    for solver, info in solver_info.items():
        solver_stats = stats_df[stats_df['solver'] == solver]
        if not solver_stats.empty:
            ax3.plot(solver_stats['n_customers'], solver_stats['success_rate'],
                    label=info['label'], color=info['color'], marker=info['marker'],
                    linewidth=2, markersize=8)
    ax3.set_xlabel('N')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Reliability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 105)
    
    # Panel 4: Time-Quality tradeoff
    ax4 = plt.subplot(2, 3, 4)
    for solver, info in solver_info.items():
        solver_stats = stats_df[stats_df['solver'] == solver]
        if not solver_stats.empty:
            sizes = (solver_stats['n_customers'] - 8) * 20
            ax4.scatter(solver_stats['avg_time'], solver_stats['avg_cpc'],
                       label=info['label'], color=info['color'],
                       marker=info['marker'], s=sizes, alpha=0.7)
    ax4.set_xscale('log')
    ax4.set_xlabel('Time (s, log)')
    ax4.set_ylabel('CPC')
    ax4.set_title('Time-Quality Trade-off')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: CPC distribution
    ax5 = plt.subplot(2, 3, 5)
    success_df = df[df['status'] == 'success'].copy()
    solver_order = ['exact_dp', 'exact_ortools_vrp', 'heuristic_or']
    solver_labels = [solver_info[s]['label'] for s in solver_order if s in success_df['solver'].unique()]
    box_data = []
    colors = []
    for solver in solver_order:
        if solver in success_df['solver'].unique():
            box_data.append(success_df[success_df['solver'] == solver]['cpc'].values)
            colors.append(solver_info[solver]['color'])
    if box_data:
        bp = ax5.boxplot(box_data, labels=solver_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax5.set_ylabel('CPC')
    ax5.set_title('CPC Distribution')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Speedup
    ax6 = plt.subplot(2, 3, 6)
    heuristic_times = stats_df[stats_df['solver'] == 'heuristic_or'].set_index('n_customers')['avg_time']
    for solver in ['exact_dp', 'exact_ortools_vrp']:
        solver_stats = stats_df[stats_df['solver'] == solver]
        if not solver_stats.empty:
            speedup = []
            n_values = []
            for _, row in solver_stats.iterrows():
                n = row['n_customers']
                if n in heuristic_times.index:
                    speedup.append(heuristic_times[n] / row['avg_time'])
                    n_values.append(n)
            if speedup:
                info = solver_info[solver]
                ax6.plot(n_values, speedup, label=info['label'],
                        color=info['color'], marker=info['marker'],
                        linewidth=2, markersize=8)
    ax6.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('N')
    ax6.set_ylabel('Speedup')
    ax6.set_title('Performance vs Heuristic')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    fig.suptitle('CPU Solver Detailed Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save detailed plot
    output_path = Path(output_dir) / "cpu_benchmark_detailed_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Detailed analysis saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot CPU benchmark results')
    parser.add_argument('--csv', default='results/csv/cpu_benchmark.csv',
                       help='Path to CSV file')
    parser.add_argument('--output', default='plots', help='Output directory')
    parser.add_argument('--detailed', action='store_true', 
                       help='Also create detailed 6-panel analysis')
    
    args = parser.parse_args()
    
    # Create main 2-panel plot (matching GPU style)
    plot_cpu_benchmark(args.csv, args.output)
    
    # Optionally create detailed analysis
    if args.detailed:
        plot_cpu_benchmark_detailed(args.csv, args.output)
