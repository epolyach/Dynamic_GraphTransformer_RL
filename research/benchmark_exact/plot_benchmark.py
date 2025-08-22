#!/usr/bin/env python3
"""
Plot CVRP Solver Benchmark Results
Creates publication-quality plots from benchmark CSV files.
Supports both single-solver (DP-only) and comparison (DP vs OR-Tools) modes.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path

def create_benchmark_plots(csv_file: str, output_prefix: str = "cvrp_benchmark", title: str = None):
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
            
        # Determine if this is comparison mode or single mode
        has_comparison = 'time_or' in data[0] and 'time_dp' in data[0]
        
        print(f"Mode: {'Comparison (DP vs OR-Tools)' if has_comparison else 'Single solver (DP-only)'}")
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
                if value.lower() in ['nan', 'inf', '-inf']:
                    row[key] = float('nan')
                else:
                    row[key] = float(value)
            except (ValueError, AttributeError):
                pass
    
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
    elif has_comparison:
        fig.suptitle('CVRP Exact Solver Performance Comparison', 
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('CVRP DP Solver Performance', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    if has_comparison:
        # Extract comparison data
        time_dp = [row['time_dp'] for row in data]
        time_or = [row['time_or'] for row in data]
        cpc_dp = [row['cpc_dp'] for row in data]
        cpc_or = [row['cpc_or'] for row in data]
        std_dp = [row['std_dp'] for row in data]
        std_or = [row['std_or'] for row in data]
        
        # Colors
        dp_color = '#1f77b4'  # Blue
        or_color = '#ff7f0e'  # Orange
        
        # Panel 1: Execution Time vs Problem Size (Log Scale)
        ax1.semilogy(n_values, time_dp, 'o-', color=dp_color, linewidth=3, 
                    markersize=8, label='DP (Exact)', alpha=0.8, markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=dp_color)
        ax1.semilogy(n_values, time_or, 's-', color=or_color, linewidth=3, 
                    markersize=8, label='OR-Tools', alpha=0.8, 
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=or_color)
        
        ax1.set_ylabel('Time per Instance (seconds)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11, loc='upper left')
        
        # Panel 2: Cost per Customer vs Problem Size
        ax2.plot(n_values, cpc_dp, 'o-', color=dp_color, linewidth=3, 
                 markersize=8, label='DP (Exact)', alpha=0.8, markerfacecolor='white',
                 markeredgewidth=2, markeredgecolor=dp_color)
        ax2.plot(n_values, cpc_or, 's-', color=or_color, linewidth=3, 
                 markersize=8, label='OR-Tools', alpha=0.8,
                 markerfacecolor='white', markeredgewidth=2, markeredgecolor=or_color)
        
        # Add error bars for standard deviation (only for valid data)
        valid_dp_mask = [not np.isnan(s) for s in std_dp]
        valid_or_mask = [not np.isnan(s) for s in std_or]
        
        if any(valid_dp_mask):
            ax2.errorbar(n_values, cpc_dp, yerr=std_dp, fmt='none', 
                        color=dp_color, alpha=0.5, capsize=4, capthick=1)
        if any(valid_or_mask):
            ax2.errorbar(n_values, cpc_or, yerr=std_or, fmt='none', 
                        color=or_color, alpha=0.5, capsize=4, capthick=1)
    
    else:
        # Single solver mode (DP-only)
        # Extract single solver data
        time_s = [row['time_s'] for row in data]
        cpc = [row['cost_per_customer'] for row in data]
        std_cpc = [row['std'] for row in data]
        
        dp_color = '#1f77b4'  # Blue
        
        # Panel 1: Execution Time vs Problem Size (Log Scale)
        ax1.semilogy(n_values, time_s, 'o-', color=dp_color, linewidth=3, 
                    markersize=8, label='DP (Exact)', alpha=0.8, markerfacecolor='white', 
                    markeredgewidth=2, markeredgecolor=dp_color)
        
        ax1.set_ylabel('Time per Instance (seconds)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11, loc='upper left')
        
        # Panel 2: Cost per Customer vs Problem Size
        ax2.plot(n_values, cpc, 'o-', color=dp_color, linewidth=3, 
                 markersize=8, label='DP (Exact)', alpha=0.8, markerfacecolor='white',
                 markeredgewidth=2, markeredgecolor=dp_color)
        
        # Add error bars for standard deviation (only for valid data)
        valid_std_mask = [not np.isnan(s) for s in std_cpc]
        if any(valid_std_mask):
            ax2.errorbar(n_values, cpc, yerr=std_cpc, fmt='none', 
                        color=dp_color, alpha=0.5, capsize=4, capthick=1)
    
    # Common formatting for panel 2
    ax2.set_xlabel('Number of Customers (N)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cost per Customer', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Set x-ticks for both panels - keep ticks but remove labels from top panel
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
    print("🔍 BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"\n📋 Dataset Summary:")
    print(f"• Problem Size Range: N = {min(n_values)} to {max(n_values)} customers")
    print(f"• Total Data Points: {len(data)}")
    
    if has_comparison:
        # Filter out NaN values for statistics
        valid_dp_times = [t for t in time_dp if not np.isnan(t)]
        valid_or_times = [t for t in time_or if not np.isnan(t)]
        valid_dp_costs = [c for c in cpc_dp if not np.isnan(c)]
        valid_or_costs = [c for c in cpc_or if not np.isnan(c)]
        
        print(f"\n⏱️  Performance Summary:")
        if valid_dp_times and valid_or_times:
            avg_time_dp = sum(valid_dp_times) / len(valid_dp_times)
            avg_time_or = sum(valid_or_times) / len(valid_or_times)
            avg_cpc_dp = sum(valid_dp_costs) / len(valid_dp_costs) if valid_dp_costs else float('nan')
            avg_cpc_or = sum(valid_or_costs) / len(valid_or_costs) if valid_or_costs else float('nan')
            
            print(f"• DP Average Time: {avg_time_dp:.4f}s")
            print(f"• OR-Tools Average Time: {avg_time_or:.4f}s")
            print(f"• DP Average Cost/Customer: {avg_cpc_dp:.4f}")
            print(f"• OR-Tools Average Cost/Customer: {avg_cpc_or:.4f}")
            print(f"• Speed Advantage: DP is {avg_time_or/avg_time_dp:.1f}x faster on average")
    else:
        # Single solver summary
        valid_times = [t for t in time_s if not np.isnan(t)]
        valid_costs = [c for c in cpc if not np.isnan(c)]
        
        if valid_times and valid_costs:
            avg_time = sum(valid_times) / len(valid_times)
            avg_cost = sum(valid_costs) / len(valid_costs)
            print(f"\n⏱️  Performance Summary:")
            print(f"• Average Time per Instance: {avg_time:.4f}s")
            print(f"• Average Cost per Customer: {avg_cost:.4f}")
            print(f"• Complexity: Exponential growth O(n²·2ⁿ)")

def main():
    parser = argparse.ArgumentParser(description='Plot CVRP Solver Benchmark Results')
    parser.add_argument('csv_file', help='CSV file with benchmark results')
    parser.add_argument('--output', default='cvrp_benchmark', help='Output prefix for plot files')
    parser.add_argument('--title', help='Custom title for the plot')
    
    args = parser.parse_args()
    
    create_benchmark_plots(args.csv_file, args.output, args.title)

if __name__ == '__main__':
    main()
