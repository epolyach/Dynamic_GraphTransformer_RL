#!/usr/bin/env python3
"""
Plot 4-Solver CVRP Benchmark Results
Creates publication-quality plots from 4-solver benchmark CSV files.
Similar format to research/benchmark_exact/benchmark_30cx100ix120s.png
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path


def create_4_solver_plots(csv_file: str, output_prefix: str = "4_solver_benchmark", title: str = None):
    """
    Create benchmark visualization plots from CSV results.
    
    Args:
        csv_file: Path to the CSV file with benchmark results
        output_prefix: Prefix for output plot files  
        title: Custom title for the plot (optional)
    """
    # Read the data using csv module
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        print(f"📊 Loaded data with {len(data)} data points")
        if not data:
            print("❌ Error: No data found in CSV file")
            return
            
        print(f"Problem sizes: N = {data[0]['N']} to {data[-1]['N']}")
        print()
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_file}")
        return
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Convert string data to numbers
    for row in data:
        for key, value in row.items():
            try:
                if isinstance(value, str):
                    if value.strip().lower() in ['nan', 'inf', '-inf', '']:
                        row[key] = float('nan')
                    else:
                        row[key] = float(value.strip())
                elif value is None or value == '':
                    row[key] = float('nan')
                else:
                    row[key] = float(value)
            except (ValueError, AttributeError, TypeError):
                row[key] = float('nan')
    
    # Extract data arrays
    n_values = [row['N'] for row in data]
    
    # Set up the plot style
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    
    # Create 2 vertical subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Set custom title if provided
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('4-Solver CVRP Performance Comparison', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    # Define solver info
    solvers = [
        ('exact_milp', 'Exact-MILP', '#1f77b4', 'o'),     # Blue circle
        ('exact_dp', 'Exact-DP', '#ff7f0e', 's'),            # Orange square  
        ('heuristic_or', 'Heuristic-OR', '#2ca02c', '^'),     # Green triangle
        ('heuristic_dp', 'Heuristic-DP', '#d62728', 'v')      # Red inverted triangle
    ]
    
    # Panel 1: Execution Time vs Problem Size (Log Scale)
    for solver_key, solver_label, color, marker in solvers:
        time_key = f'time_{solver_key}'
        times = [row.get(time_key, float('nan')) for row in data]
        
        # Filter out NaN values for plotting
        valid_indices = [i for i, t in enumerate(times) if not np.isnan(t)]
        if valid_indices:
            valid_n = [n_values[i] for i in valid_indices]
            valid_times = [times[i] for i in valid_indices]
            
            ax1.semilogy(valid_n, valid_times, marker + '-', color=color, linewidth=3, 
                        markersize=8, label=solver_label, alpha=0.8, 
                        markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
    
    ax1.set_ylabel('Time per Instance (seconds)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper left')
    
    # Panel 2: Cost per Customer vs Problem Size
    for solver_key, solver_label, color, marker in solvers:
        cpc_key = f'cpc_{solver_key}'
        std_key = f'std_{solver_key}'
        solved_key = f'solved_{solver_key}'
        costs = [row.get(cpc_key, float('nan')) for row in data]
        stds = [row.get(std_key, float('nan')) for row in data]
        solved_counts = [row.get(solved_key, 0) for row in data]
        
        # Filter out NaN values for plotting
        valid_indices = [i for i, c in enumerate(costs) if not np.isnan(c)]
        if valid_indices:
            valid_n = [n_values[i] for i in valid_indices]
            valid_costs = [costs[i] for i in valid_indices]
            valid_stds = [stds[i] for i in valid_indices if i < len(stds) and not np.isnan(stds[i])]
            valid_solved = [solved_counts[i] for i in valid_indices]
            
            ax2.plot(valid_n, valid_costs, marker + '-', color=color, linewidth=3, 
                     markersize=8, label=solver_label, alpha=0.8,
                     markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
            
            # Add error bars using standard error of the mean (SEM = std/sqrt(n))
            if len(valid_stds) == len(valid_costs) and len(valid_stds) > 0:
                # Calculate standard error of the mean for each data point
                sems = []
                for i, (std_val, n_solved) in enumerate(zip(valid_stds, valid_solved)):
                    if not np.isnan(std_val) and n_solved > 0:
                        sem = std_val / np.sqrt(n_solved)
                        sems.append(sem)
                    else:
                        sems.append(0.0)  # No error bar if no valid std or no solved instances
                
                if sems:
                    ax2.errorbar(valid_n, valid_costs, yerr=sems, fmt='none', 
                                color=color, alpha=0.5, capsize=4, capthick=1)
    
    # Common formatting for panel 2
    ax2.set_xlabel('Number of Customers (N)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cost per Customer', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Set x-ticks for both panels - remove labels from top panel
    ax1.set_xticks(n_values)
    ax1.set_xticklabels([])  # Remove x-tick labels from top panel
    ax2.set_xticks(n_values)
    
    # Adjust layout with minimal room for title
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save the plot
    png_file = f'{output_prefix}.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📈 Plot saved as '{png_file}'")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("🔍 4-SOLVER BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"\n📋 Dataset Summary:")
    print(f"• Problem Size Range: N = {min(n_values)} to {max(n_values)} customers")
    print(f"• Total Data Points: {len(data)}")
    
    print(f"\n⏱️  Performance Summary:")
    for solver_key, solver_label, _, _ in solvers:
        time_key = f'time_{solver_key}'
        cpc_key = f'cpc_{solver_key}'
        solved_key = f'solved_{solver_key}'
        optimal_key = f'optimal_{solver_key}'
        
        # Collect valid data
        times = [row.get(time_key, float('nan')) for row in data]
        costs = [row.get(cpc_key, float('nan')) for row in data]
        solved_counts = [row.get(solved_key, 0) for row in data]
        
        valid_times = [t for t in times if not np.isnan(t)]
        valid_costs = [c for c in costs if not np.isnan(c)]
        total_solved = int(sum(solved_counts))
        
        if valid_times and valid_costs:
            avg_time = sum(valid_times) / len(valid_times)
            avg_cost = sum(valid_costs) / len(valid_costs)
            print(f"• {solver_label:15}: {total_solved:3d} solved, time={avg_time:.4f}s, cpc={avg_cost:.4f}")
            
            # Show optimal count for exact solvers
            if solver_key.startswith('exact'):
                optimal_counts = [row.get(optimal_key, 0) for row in data]
                total_optimal = int(sum(optimal_counts))
                print(f"  {'':15}   {total_optimal:3d} optimal solutions")
        else:
            print(f"• {solver_label:15}: No valid solutions")


def main():
    parser = argparse.ArgumentParser(description='Plot 4-Solver CVRP Benchmark Results')
    parser.add_argument('csv_file', help='CSV file with benchmark results')
    parser.add_argument('--output', default='4_solver_benchmark', help='Output prefix for plot files')
    parser.add_argument('--title', help='Custom title for the plot')
    
    args = parser.parse_args()
    
    create_4_solver_plots(args.csv_file, args.output, args.title)


if __name__ == '__main__':
    main()
