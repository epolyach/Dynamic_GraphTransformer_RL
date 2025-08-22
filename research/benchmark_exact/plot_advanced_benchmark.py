#!/usr/bin/env python3
"""
Enhanced Plot Script for Advanced CVRP Benchmark Results
Supports multiple benchmark modes:
- exact-only: Single solver (exact algorithms only)
- compare: Exact vs Heuristic comparison  
- heuristic-only: Single solver (heuristic algorithms only)

Compatible with both original benchmark_cli.py and advanced benchmark_advanced_cli.py
Creates publication-quality plots with algorithm-specific formatting.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path


def detect_benchmark_mode(data):
    """Detect which benchmark mode was used based on CSV columns"""
    columns = set(data[0].keys()) if data else set()
    
    if 'time_exact' in columns and 'time_heuristic' in columns:
        return 'compare'
    elif 'time_exact' in columns:
        return 'exact-only'
    elif 'time_heuristic' in columns:
        return 'heuristic-only'
    elif 'time_dp' in columns and 'time_or' in columns:
        return 'dp-vs-ortools'  # Original benchmark format
    elif 'time_dp' in columns:
        return 'dp-only'  # Original DP-only format
    else:
        return 'unknown'


def create_advanced_benchmark_plots(csv_file: str, output_prefix: str = "advanced_cvrp_benchmark", title: str = None):
    """
    Create advanced benchmark visualization plots from CSV results.
    """
    # Read data
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        print(f"📊 Loaded data with {len(data)} data points")
        if not data:
            print("❌ Error: No data found in CSV file")
            return
            
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_file}")
        return
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Convert strings to numbers
    for row in data:
        for key, value in row.items():
            try:
                if isinstance(value, str) and value.lower() in ['nan', 'inf', '-inf', '']:
                    row[key] = float('nan')
                elif isinstance(value, str):
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass  # Keep as string (e.g., algorithm names)
            except (ValueError, AttributeError):
                pass
    
    # Detect benchmark mode
    mode = detect_benchmark_mode(data)
    print(f"Detected benchmark mode: {mode}")
    
    # Extract data
    n_values = [row['N'] for row in data]
    print(f"Problem sizes: N = {min(n_values)} to {max(n_values)}")
    print()
    
    # Create plots based on mode
    if mode == 'compare':
        create_exact_vs_heuristic_plots(data, n_values, output_prefix, title)
    elif mode == 'exact-only':
        create_exact_only_plots(data, n_values, output_prefix, title)
    elif mode == 'heuristic-only':
        create_heuristic_only_plots(data, n_values, output_prefix, title)
    elif mode == 'dp-vs-ortools':
        create_dp_vs_ortools_plots(data, n_values, output_prefix, title)
    elif mode == 'dp-only':
        create_dp_only_plots(data, n_values, output_prefix, title)
    else:
        print(f"❌ Unknown benchmark mode: {mode}")
        return
    
    print(f"✅ Plots saved with prefix '{output_prefix}'")


def create_exact_vs_heuristic_plots(data, n_values, output_prefix, title):
    """Create plots for exact vs heuristic comparison mode"""
    
    # Extract data
    time_exact = [row['time_exact'] for row in data]
    time_heuristic = [row['time_heuristic'] for row in data]
    cpc_exact = [row['cpc_exact'] for row in data]
    cpc_heuristic = [row['cpc_heuristic'] for row in data]
    std_exact = [row['std_exact'] for row in data]
    std_heuristic = [row['std_heuristic'] for row in data]
    
    # Setup plot style
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    
    # Create 3-panel plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Advanced CVRP Solver Performance: Exact vs Heuristic', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    # Colors
    exact_color = '#2E86AB'    # Blue
    heuristic_color = '#A23B72' # Purple
    
    # Panel 1: Execution Time (Log Scale)
    valid_exact_times = [(n, t) for n, t in zip(n_values, time_exact) if not np.isnan(t)]
    valid_heur_times = [(n, t) for n, t in zip(n_values, time_heuristic) if not np.isnan(t)]
    
    if valid_exact_times:
        n_exact, t_exact = zip(*valid_exact_times)
        ax1.semilogy(n_exact, t_exact, 'o-', color=exact_color, linewidth=3, 
                    markersize=8, label='Exact Algorithms', alpha=0.8, 
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=exact_color)
    
    if valid_heur_times:
        n_heur, t_heur = zip(*valid_heur_times)
        ax1.semilogy(n_heur, t_heur, 's-', color=heuristic_color, linewidth=3, 
                    markersize=8, label='Heuristic Algorithms', alpha=0.8,
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=heuristic_color)
    
    ax1.set_ylabel('Time per Instance (seconds)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.set_xticks(n_values[::max(1, len(n_values)//10)])
    ax1.set_xticklabels([])  # Remove x labels from top panel
    
    # Panel 2: Cost per Customer
    valid_exact_costs = [(n, c, s) for n, c, s in zip(n_values, cpc_exact, std_exact) if not np.isnan(c)]
    valid_heur_costs = [(n, c, s) for n, c, s in zip(n_values, cpc_heuristic, std_heuristic) if not np.isnan(c)]
    
    if valid_exact_costs:
        n_exact, c_exact, s_exact = zip(*valid_exact_costs)
        ax2.plot(n_exact, c_exact, 'o-', color=exact_color, linewidth=3, 
                 markersize=8, label='Exact Algorithms', alpha=0.8,
                 markerfacecolor='white', markeredgewidth=2, markeredgecolor=exact_color)
        # Error bars
        if not all(np.isnan(s) for s in s_exact):
            ax2.errorbar(n_exact, c_exact, yerr=s_exact, fmt='none', 
                        color=exact_color, alpha=0.5, capsize=4, capthick=1)
    
    if valid_heur_costs:
        n_heur, c_heur, s_heur = zip(*valid_heur_costs)
        ax2.plot(n_heur, c_heur, 's-', color=heuristic_color, linewidth=3, 
                 markersize=8, label='Heuristic Algorithms', alpha=0.8,
                 markerfacecolor='white', markeredgewidth=2, markeredgecolor=heuristic_color)
        # Error bars
        if not all(np.isnan(s) for s in s_heur):
            ax2.errorbar(n_heur, c_heur, yerr=s_heur, fmt='none', 
                        color=heuristic_color, alpha=0.5, capsize=4, capthick=1)
    
    ax2.set_ylabel('Cost per Customer', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xticks(n_values[::max(1, len(n_values)//10)])
    ax2.set_xticklabels([])  # Remove x labels from middle panel
    
    # Panel 3: Optimality Gap Analysis (if exact solutions available)
    if valid_exact_costs and valid_heur_costs:
        # Calculate gaps where both solutions exist
        gaps = []
        gap_n = []
        for n in n_values:
            exact_cost = next((c for nv, c, _ in valid_exact_costs if nv == n), None)
            heur_cost = next((c for nv, c, _ in valid_heur_costs if nv == n), None)
            
            if exact_cost is not None and heur_cost is not None and exact_cost > 0:
                gap = ((heur_cost - exact_cost) / exact_cost) * 100
                gaps.append(gap)
                gap_n.append(n)
        
        if gaps:
            ax3.plot(gap_n, gaps, 'o-', color='#F18F01', linewidth=3, 
                     markersize=8, label='Optimality Gap', alpha=0.8,
                     markerfacecolor='white', markeredgewidth=2, markeredgecolor='#F18F01')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Optimality Gap (%)', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=11)
    else:
        # If no gap analysis possible, show algorithm distribution
        exact_algos = [row.get('exact_algorithms', '') for row in data if row.get('exact_algorithms')]
        heur_algos = [row.get('heuristic_algorithms', '') for row in data if row.get('heuristic_algorithms')]
        
        ax3.text(0.5, 0.5, f'Algorithm Usage:\nExact: {", ".join(set(exact_algos)) if exact_algos else "None"}\nHeuristic: {", ".join(set(heur_algos)) if heur_algos else "None"}',
                 transform=ax3.transAxes, ha='center', va='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax3.set_ylabel('Algorithm Info', fontsize=13, fontweight='bold')
    
    ax3.set_xlabel('Number of Customers (N)', fontsize=13, fontweight='bold')
    ax3.set_xticks(n_values[::max(1, len(n_values)//10)])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_exact_only_plots(data, n_values, output_prefix, title):
    """Create plots for exact-only benchmark mode"""
    
    time_exact = [row['time_exact'] for row in data]
    cpc_exact = [row['cpc_exact'] for row in data]
    std_exact = [row['std_exact'] for row in data]
    
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Exact CVRP Solver Performance', fontsize=16, fontweight='bold', y=0.98)
    
    exact_color = '#2E86AB'
    
    # Panel 1: Time
    valid_times = [(n, t) for n, t in zip(n_values, time_exact) if not np.isnan(t)]
    if valid_times:
        n_vals, t_vals = zip(*valid_times)
        ax1.semilogy(n_vals, t_vals, 'o-', color=exact_color, linewidth=3, 
                    markersize=8, label='Exact Algorithms', alpha=0.8,
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=exact_color)
    
    ax1.set_ylabel('Time per Instance (seconds)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xticks(n_values[::max(1, len(n_values)//10)])
    ax1.set_xticklabels([])
    
    # Panel 2: Cost
    valid_costs = [(n, c, s) for n, c, s in zip(n_values, cpc_exact, std_exact) if not np.isnan(c)]
    if valid_costs:
        n_vals, c_vals, s_vals = zip(*valid_costs)
        ax2.plot(n_vals, c_vals, 'o-', color=exact_color, linewidth=3, 
                 markersize=8, label='Exact Algorithms', alpha=0.8,
                 markerfacecolor='white', markeredgewidth=2, markeredgecolor=exact_color)
        if not all(np.isnan(s) for s in s_vals):
            ax2.errorbar(n_vals, c_vals, yerr=s_vals, fmt='none', 
                        color=exact_color, alpha=0.5, capsize=4, capthick=1)
    
    ax2.set_xlabel('Number of Customers (N)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cost per Customer', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xticks(n_values[::max(1, len(n_values)//10)])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_heuristic_only_plots(data, n_values, output_prefix, title):
    """Create plots for heuristic-only benchmark mode"""
    
    time_heuristic = [row['time_heuristic'] for row in data]
    cpc_heuristic = [row['cpc_heuristic'] for row in data]
    std_heuristic = [row['std_heuristic'] for row in data]
    avg_gaps = [row.get('avg_gap', 0) for row in data]
    
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Heuristic CVRP Solver Performance', fontsize=16, fontweight='bold', y=0.98)
    
    heuristic_color = '#A23B72'
    
    # Panel 1: Time
    valid_times = [(n, t) for n, t in zip(n_values, time_heuristic) if not np.isnan(t)]
    if valid_times:
        n_vals, t_vals = zip(*valid_times)
        ax1.semilogy(n_vals, t_vals, 's-', color=heuristic_color, linewidth=3, 
                    markersize=8, label='Heuristic Algorithms', alpha=0.8,
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=heuristic_color)
    
    ax1.set_ylabel('Time per Instance (seconds)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xticks(n_values[::max(1, len(n_values)//10)])
    ax1.set_xticklabels([])
    
    # Panel 2: Cost
    valid_costs = [(n, c, s) for n, c, s in zip(n_values, cpc_heuristic, std_heuristic) if not np.isnan(c)]
    if valid_costs:
        n_vals, c_vals, s_vals = zip(*valid_costs)
        ax2.plot(n_vals, c_vals, 's-', color=heuristic_color, linewidth=3, 
                 markersize=8, label='Heuristic Algorithms', alpha=0.8,
                 markerfacecolor='white', markeredgewidth=2, markeredgecolor=heuristic_color)
        if not all(np.isnan(s) for s in s_vals):
            ax2.errorbar(n_vals, c_vals, yerr=s_vals, fmt='none', 
                        color=heuristic_color, alpha=0.5, capsize=4, capthick=1)
    
    ax2.set_ylabel('Cost per Customer', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xticks(n_values[::max(1, len(n_values)//10)])
    ax2.set_xticklabels([])
    
    # Panel 3: Average optimality gaps
    valid_gaps = [(n, g) for n, g in zip(n_values, avg_gaps) if not np.isnan(g) and g > 0]
    if valid_gaps:
        n_vals, g_vals = zip(*valid_gaps)
        g_percent = [g * 100 for g in g_vals]  # Convert to percentage
        ax3.plot(n_vals, g_percent, 'o-', color='#F18F01', linewidth=3, 
                 markersize=8, label='Average Optimality Gap', alpha=0.8,
                 markerfacecolor='white', markeredgewidth=2, markeredgecolor='#F18F01')
        ax3.set_ylabel('Estimated Gap (%)', fontsize=13, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No gap information available', 
                 transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        ax3.set_ylabel('Gap Info', fontsize=13, fontweight='bold')
    
    ax3.set_xlabel('Number of Customers (N)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    ax3.set_xticks(n_values[::max(1, len(n_values)//10)])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_dp_vs_ortools_plots(data, n_values, output_prefix, title):
    """Create plots for DP vs OR-Tools comparison (original format)"""
    # Use existing plotting logic from plot_benchmark.py
    from plot_benchmark import create_benchmark_plots
    create_benchmark_plots(csv_file, output_prefix, title)


def create_dp_only_plots(data, n_values, output_prefix, title):
    """Create plots for DP-only mode (original format)"""
    # Use existing plotting logic from plot_benchmark.py  
    from plot_benchmark import create_benchmark_plots
    create_benchmark_plots(csv_file, output_prefix, title)


def main():
    parser = argparse.ArgumentParser(description='Enhanced Plot Script for Advanced CVRP Benchmark Results')
    parser.add_argument('csv_file', help='CSV file with benchmark results')
    parser.add_argument('--output', type=str, default=None, help='Output prefix for plot files (default: auto-generate)')
    parser.add_argument('--title', type=str, default=None, help='Custom title for the plot')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"❌ Error: File {args.csv_file} not found")
        sys.exit(1)
    
    # Auto-generate output prefix if not provided
    if args.output is None:
        csv_path = Path(args.csv_file)
        args.output = f"plot_{csv_path.stem}"
    
    print(f"📊 Creating advanced benchmark plots from {args.csv_file}")
    create_advanced_benchmark_plots(args.csv_file, args.output, args.title)
    print("✅ Done!")


if __name__ == '__main__':
    main()
