#!/usr/bin/env python3
"""
Plot histogram of CPC values with log-normal distribution fit for paper publication.
Reads data from CSV file with 100k instances and creates paper-ready figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import sys
from pathlib import Path

# Set matplotlib parameters for paper publication
plt.rcParams['font.size'] = 7
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.linewidth'] = 0.3
plt.rcParams['lines.linewidth'] = 0.8


def load_cpc_data():
    """Load CPC values from hardcoded CSV file."""
    csv_file = "gpu_dp_exact_results_20250905_071235.csv"
    
    try:
        df = pd.read_csv(csv_file)
        cpc_values = df['cpc'].values
        print(f"📊 Loaded {len(cpc_values)} CPC values from {csv_file}")
        return cpc_values
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    except KeyError:
        print(f"❌ 'cpc' column not found in {csv_file}")
        sys.exit(1)


def plot_paper_histogram(cpc_values, output_file="cpc_histogram_100k.png", bins=50):
    """
    Plot histogram of CPC values with log-normal distribution fit for paper publication.
    
    Args:
        cpc_values: Array of CPC values
        output_file: Output file path for saving the figure
        bins: Number of histogram bins
    """
    cpc_array = np.array(cpc_values)
    
    # Calculate geometric mean
    gm = np.exp(np.mean(np.log(cpc_array)))
    
    # Fit log-normal distribution
    shape, loc, scale = stats.lognorm.fit(cpc_array, floc=0)
    
    # Create figure with paper dimensions (80mm x 60mm)
    fig, ax = plt.subplots(1, 1, figsize=(3.15, 2.36))  # 80mm x 60mm in inches
    
    # Histogram (with proportionally thin edges)
    ax.hist(cpc_array, bins=bins, density=True, alpha=0.7, 
            edgecolor='black', linewidth=0.3)
    
    # Plot fitted log-normal distribution (red line, thin)
    x_range = np.linspace(cpc_array.min() * 0.9, cpc_array.max() * 1.1, 1000)
    pdf_fitted = stats.lognorm.pdf(x_range, shape, loc, scale)
    ax.plot(x_range, pdf_fitted, 'r-', linewidth=0.8, label='Log-normal fit')
    
    # Add vertical dashed line for geometric mean
    # ax.axvline(gm, color="green", linestyle="--", linewidth=0.8, 
               # label=f"GM={gm:.3f}"))
    
    # Labels and grid (no title as requested)
    ax.set_xlabel('CPC (Cost Per Customer)', fontsize=7)
    ax.set_ylabel('CPC Distribution', fontsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    
    # Adjust tick parameters for small figure
    ax.tick_params(labelsize=6, width=0.5)
    
    # Legend with smaller font
    ax.legend(fontsize=6, frameon=True, fancybox=True, shadow=False)
    
    plt.tight_layout()
    
    # Save PNG with high DPI for paper quality
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                format='png', facecolor='white')
    print(f"✅ Saved paper-ready PNG figure to {output_file}")
    
    # Also save as EPS for LaTeX
    eps_file = output_file.replace('.png', '.eps')
    plt.savefig(eps_file, dpi=300, bbox_inches='tight', 
                format='eps', facecolor='white')
    print(f"✅ Saved EPS version to {eps_file}")
    
    return fig, gm


def print_statistics(cpc_array, gm):
    """Print statistical information about the data."""
    print(f"\n📈 Statistics:")
    print(f"   Mean CPC: {np.mean(cpc_array):.6f}")
    print(f"   Std CPC:  {np.std(cpc_array):.6f}")
    print(f"   Min CPC:  {np.min(cpc_array):.6f}")
    print(f"   Max CPC:  {np.max(cpc_array):.6f}")
    print(f"   Geometric Mean: {gm:.6f}")
    
    # Test for log-normality
    shape, loc, scale = stats.lognorm.fit(cpc_array, floc=0)
    ks_stat, ks_pvalue = stats.kstest(cpc_array, 
                                      lambda x: stats.lognorm.cdf(x, shape, loc, scale))
    print(f"\n📊 Log-normal fit:")
    print(f"   Shape (σ): {shape:.4f}")
    print(f"   Scale (exp(μ)): {scale:.4f}")
    print(f"   KS test p-value: {ks_pvalue:.4f}")
    if ks_pvalue > 0.05:
        print(f"   ✅ Data is consistent with log-normal distribution (p > 0.05)")
    else:
        print(f"   ⚠️  Data may not follow log-normal distribution (p < 0.05)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper-ready CPC histogram with log-normal fit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--output', type=str, default='cpc_histogram_paper_100k.png',
                        help='Output figure file (PNG format)')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins')
    
    args = parser.parse_args()
    
    try:
        # Load data from hardcoded CSV file
        cpc_values = load_cpc_data()
        
        # Generate paper-ready plot
        fig, gm = plot_paper_histogram(cpc_values, args.output, args.bins)
        
        # Print statistics
        print_statistics(cpc_values, gm)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
