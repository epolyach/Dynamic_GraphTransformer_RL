#!/usr/bin/env python3
"""
Plot histogram of CPC values with both normal and log-normal distribution fits.
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
    return data['cpc'], data['n'], data.get('capacity', 'N/A')


def plot_distribution_histogram(cpc_values, n, capacity, output_file=None, bins=50):
    """
    Plot histogram of CPC values with both normal and log-normal distribution fits.
    
    Args:
        cpc_values: List of CPC values
        n: Problem size (number of customers)
        capacity: Vehicle capacity
        output_file: Optional output file path for saving the figure
        bins: Number of histogram bins
    """
    cpc_array = np.array(cpc_values)
    
    # Fit log-normal distribution
    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(cpc_array, floc=0)
    
    # Fit normal distribution
    loc_norm, scale_norm = stats.norm.fit(cpc_array)
    
    # Create single panel figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Histogram (default matplotlib colors - typically blue-ish)
    ax.hist(cpc_array, bins=bins, density=True, alpha=0.7, edgecolor='black')
    
    # Plot fitted distributions
    x_range = np.linspace(cpc_array.min() * 0.9, cpc_array.max() * 1.1, 1000)
    
    # Log-normal distribution
    pdf_lognorm = stats.lognorm.pdf(x_range, shape_lognorm, loc_lognorm, scale_lognorm)
    ax.plot(x_range, pdf_lognorm, 'r-', linewidth=2, 
            label=f'Log-normal fit\n(σ={shape_lognorm:.3f}, μ={np.log(scale_lognorm):.3f})')
    
    # Normal distribution
    pdf_norm = stats.norm.pdf(x_range, loc_norm, scale_norm)
    ax.plot(x_range, pdf_norm, 'g--', linewidth=2, 
            label=f'Normal fit\n(μ={loc_norm:.3f}, σ={scale_norm:.3f})')
    
    ax.set_xlabel('CPC (Cost Per Customer)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
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
        
        # Fit both distributions
        shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(cpc_array, floc=0)
        loc_norm, scale_norm = stats.norm.fit(cpc_array)
        
        ax = axes[idx]
        ax.hist(cpc_array, bins=30, density=True, alpha=0.7, edgecolor='black')
        
        x_range = np.linspace(cpc_array.min() * 0.9, cpc_array.max() * 1.1, 1000)
        
        # Plot both fits
        pdf_lognorm = stats.lognorm.pdf(x_range, shape_lognorm, loc_lognorm, scale_lognorm)
        ax.plot(x_range, pdf_lognorm, 'r-', linewidth=2, label='Log-normal fit')
        
        pdf_norm = stats.norm.pdf(x_range, loc_norm, scale_norm)
        ax.plot(x_range, pdf_norm, 'g--', linewidth=2, label='Normal fit')
        
        ax.set_xlabel('CPC', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.set_title(f'N={n}, Cap={capacity}\n{len(cpc_values)} instances', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('CPC Distributions with Normal and Log-Normal Fits', fontsize=14)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved figure to {output_file}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot CPC histogram with normal and log-normal distribution fits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, default='ortools_n6.json',
                        help='Input JSON file (default: ortools_n6.json in current directory)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output figure file (e.g., histogram.png). If not specified, displays plot.')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins')
    parser.add_argument('--multi', nargs='+', 
                        help='Plot multiple JSON files (e.g., --multi ortools_n6.json ortools_n20.json)')
    
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
            plot_distribution_histogram(cpc_values, n, capacity, args.output, args.bins)
            
            # Print statistics
            cpc_array = np.array(cpc_values)
            print(f"\n📈 Statistics:")
            print(f"   Mean CPC: {np.mean(cpc_array):.6f}")
            print(f"   Std CPC:  {np.std(cpc_array):.6f}")
            print(f"   Min CPC:  {np.min(cpc_array):.6f}")
            print(f"   Max CPC:  {np.max(cpc_array):.6f}")
            
            # Fit both distributions for testing
            shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(cpc_array, floc=0)
            loc_norm, scale_norm = stats.norm.fit(cpc_array)
            
            # Test for log-normality
            ks_stat_lognorm, ks_pvalue_lognorm = stats.kstest(
                cpc_array, 
                lambda x: stats.lognorm.cdf(x, shape_lognorm, loc_lognorm, scale_lognorm)
            )
            
            # Test for normality
            ks_stat_norm, ks_pvalue_norm = stats.kstest(cpc_array, 'norm', args=(loc_norm, scale_norm))
            
            print(f"\n📊 Distribution Fits:")
            print(f"\n   Log-normal distribution:")
            print(f"   - Shape (σ): {shape_lognorm:.4f}")
            print(f"   - Scale (exp(μ)): {scale_lognorm:.4f}")
            print(f"   - KS test statistic: {ks_stat_lognorm:.4f}")
            print(f"   - KS test p-value: {ks_pvalue_lognorm:.4f}")
            if ks_pvalue_lognorm > 0.05:
                print(f"   ✅ Data is consistent with log-normal distribution (p > 0.05)")
            else:
                print(f"   ⚠️  Data may not follow log-normal distribution (p < 0.05)")
            
            print(f"\n   Normal distribution:")
            print(f"   - Mean (μ): {loc_norm:.4f}")
            print(f"   - Std dev (σ): {scale_norm:.4f}")
            print(f"   - KS test statistic: {ks_stat_norm:.4f}")
            print(f"   - KS test p-value: {ks_pvalue_norm:.4f}")
            if ks_pvalue_norm > 0.05:
                print(f"   ✅ Data is consistent with normal distribution (p > 0.05)")
            else:
                print(f"   ⚠️  Data may not follow normal distribution (p < 0.05)")
            
            print(f"\n📊 Distribution Comparison:")
            if ks_pvalue_lognorm > ks_pvalue_norm:
                print(f"   → Log-normal distribution appears to be a better fit (higher p-value)")
            elif ks_pvalue_norm > ks_pvalue_lognorm:
                print(f"   → Normal distribution appears to be a better fit (higher p-value)")
            else:
                print(f"   → Both distributions have similar fit quality")
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
