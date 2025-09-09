#!/usr/bin/env python3
"""
Plot histogram of CPC values with log-normal distribution fit.
Reads data from ortools_n{N}.json files.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys


def load_cpc_data(json_file):
    """Load CPC values from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if 'cpc' in data:
        cpc_values = data['cpc']
    elif 'all_cpcs' in data:
        cpc_values = data['all_cpcs']
    else:
        raise KeyError("No CPC data found. Expected 'cpc' or 'all_cpcs' field in JSON.")
    
    if 'n' in data:
        n = data['n']
    elif 'n_customers' in data:
        n = data['n_customers']
    else:
        raise KeyError("No customer count found. Expected 'n' or 'n_customers' field in JSON.")
    
    capacity = data.get('capacity', 'N/A')
    
    return cpc_values, n, capacity


def plot_lognormal_histogram(cpc_values, n, capacity, output_file=None, bins=50):
    """
    Plot histogram of CPC values with log-normal distribution fit.
    
    Args:
        cpc_values: List of CPC values
        n: Problem size (number of customers)
        capacity: Vehicle capacity
        output_file: Optional output file path for saving the figure
        bins: Number of histogram bins
    """
    cpc_array = np.array(cpc_values)
    
    # Fit log-normal distribution
    shape, loc, scale = stats.lognorm.fit(cpc_array, floc=0)
    
    # Create single panel figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Histogram (default matplotlib colors - typically blue-ish)
    ax.hist(cpc_array, bins=bins, density=True, alpha=0.7, edgecolor='black')
    
    # Plot fitted log-normal distribution
    x_range = np.linspace(cpc_array.min() * 0.9, cpc_array.max() * 1.1, 1000)
    pdf_fitted = stats.lognorm.pdf(x_range, shape, loc, scale)
    ax.plot(x_range, pdf_fitted, 'r-', linewidth=2, label=f'Log-normal fit\n(σ={shape:.3f}, μ={np.log(scale):.3f})')
    
    ax.set_xlabel('CPC (Cost Per Customer)', fontsize=12)
    ax.set_ylabel('CPC Distribution', fontsize=12)
    ax.set_title(f'CPC Distribution for N={n}, Capacity={capacity}\n({len(cpc_values)} instances)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved figure to {output_file}")
    else:
        plt.show()
    
    return fig


def plot_multiple_histograms(json_files, output_file=None):
    """
    Plot histograms for multiple problem sizes on the same figure.
    
    Args:
        json_files: List of JSON file paths
        output_file: Optional output file path
    """
    fig, axes = plt.subplots(1, len(json_files), figsize=(6*len(json_files), 6))
    
    if len(json_files) == 1:
        axes = [axes]
    
    for idx, json_file in enumerate(json_files):
        cpc_values, n, capacity = load_cpc_data(json_file)
        cpc_array = np.array(cpc_values)
        
        # Fit log-normal
        shape, loc, scale = stats.lognorm.fit(cpc_array, floc=0)
        
        ax = axes[idx]
        ax.hist(cpc_array, bins=30, density=True, alpha=0.7, edgecolor='black')
        
        x_range = np.linspace(cpc_array.min() * 0.9, cpc_array.max() * 1.1, 1000)
        pdf_fitted = stats.lognorm.pdf(x_range, shape, loc, scale)
        ax.plot(x_range, pdf_fitted, 'r-', linewidth=2, label='Log-normal fit')
        
        ax.set_xlabel('CPC', fontsize=10)
        ax.set_ylabel('CPC Distribution', fontsize=10)
        ax.set_title(f'N={n}, Cap={capacity}\n{len(cpc_values)} instances', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('CPC Distributions with Log-Normal Fits', fontsize=14)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved figure to {output_file}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot CPC histogram with log-normal distribution fit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, default='ortools_n10.json',
                        help='Input JSON file (default: ortools_n10.json in current directory)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output figure file (e.g., histogram.png). If not specified, displays plot.')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins')
    parser.add_argument('--multi', nargs='+', 
                        help='Plot multiple JSON files (e.g., --multi ortools_n10.json ortools_n20.json)')
    
    args = parser.parse_args()
    
    try:
        if args.multi:
            # Multiple files comparison
            json_files = []
            for f in args.multi:
                json_path = Path(f)
                if not json_path.exists():
                    # Try in current directory
                    json_path = Path.cwd() / f
                if not json_path.exists():
                    print(f"❌ File not found: {f}", file=sys.stderr)
                    return 1
                json_files.append(json_path)
            
            output_file = args.output or 'cpc_comparison.png'
            plot_multiple_histograms(json_files, output_file)
            
        else:
            # Single file
            json_path = Path(args.input)
            if not json_path.exists():
                # Try in current directory
                json_path = Path.cwd() / args.input
            
            if not json_path.exists():
                print(f"❌ File not found: {args.input}", file=sys.stderr)
                print(f"   Searched in: {Path(args.input).absolute()} and {json_path.absolute()}")
                return 1
            
            # Load data
            cpc_values, n, capacity = load_cpc_data(json_path)
            print(f"📊 Loaded {len(cpc_values)} CPC values from {json_path.name}")
            print(f"   Problem size: N={n}, Capacity={capacity}")
            
            # Generate output filename if not specified
            if args.output is None and not sys.stdout.isatty():
                # If running in non-interactive mode, save to file
                args.output = f'cpc_histogram_n{n}.png'
            
            # Plot
            plot_lognormal_histogram(cpc_values, n, capacity, args.output, args.bins)
            
            # Print statistics
            cpc_array = np.array(cpc_values)
            print(f"\n📈 Statistics:")
            print(f"   Mean CPC: {np.mean(cpc_array):.6f}")
            print(f"   Std CPC:  {np.std(cpc_array):.6f}")
            print(f"   Min CPC:  {np.min(cpc_array):.6f}")
            print(f"   Max CPC:  {np.max(cpc_array):.6f}")
            
            # Test for log-normality
            shape, loc, scale = stats.lognorm.fit(cpc_array, floc=0)
            ks_stat, ks_pvalue = stats.kstest(cpc_array, lambda x: stats.lognorm.cdf(x, shape, loc, scale))
            print(f"\n📊 Log-normal fit:")
            print(f"   Shape (σ): {shape:.4f}")
            print(f"   Scale (exp(μ)): {scale:.4f}")
            print(f"   KS test p-value: {ks_pvalue:.4f}")
            if ks_pvalue > 0.05:
                print(f"   ✅ Data is consistent with log-normal distribution (p > 0.05)")
            else:
                print(f"   ⚠️  Data may not follow log-normal distribution (p < 0.05)")
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
