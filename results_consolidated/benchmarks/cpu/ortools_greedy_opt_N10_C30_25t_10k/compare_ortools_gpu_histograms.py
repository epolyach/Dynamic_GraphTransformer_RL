#!/usr/bin/env python3
"""
Compare histograms of CPC values from OR-Tools and GPU DP exact results
with log-normal distribution fits.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys


def load_cpc_data_ortools(json_file):
    """Load CPC values from OR-Tools JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['cpc'], data['n'], data.get('capacity', 'N/A')


def load_cpc_data_gpu(json_file):
    """Load CPC values from GPU DP exact JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert total costs to CPC
    n_customers = data['n_customers']
    cpc_values = [cost / n_customers for cost in data['all_costs']]
    
    return cpc_values, n_customers, data.get('capacity', 'N/A')


def plot_comparison_histograms(ortools_file, gpu_file, output_file=None, bins=50):
    """
    Plot comparison of OR-Tools and GPU DP exact results with log-normal fits.
    
    Args:
        ortools_file: Path to OR-Tools results JSON
        gpu_file: Path to GPU DP exact results JSON
        output_file: Optional output file path for saving the figure
        bins: Number of histogram bins
    """
    # Load data
    cpc_ortools, n_ortools, cap_ortools = load_cpc_data_ortools(ortools_file)
    cpc_gpu, n_gpu, cap_gpu = load_cpc_data_gpu(gpu_file)
    
    # Convert to numpy arrays
    cpc_ortools_array = np.array(cpc_ortools)
    cpc_gpu_array = np.array(cpc_gpu)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Common x-range for both plots
    x_min = min(cpc_ortools_array.min(), cpc_gpu_array.min()) * 0.9
    x_max = max(cpc_ortools_array.max(), cpc_gpu_array.max()) * 1.1
    x_range = np.linspace(x_min, x_max, 1000)
    
    # Plot 1: OR-Tools results
    ax1.hist(cpc_ortools_array, bins=bins, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black', label='OR-Tools data')
    
    # Fit and plot log-normal for OR-Tools
    shape_ortools, loc_ortools, scale_ortools = stats.lognorm.fit(cpc_ortools_array, floc=0)
    pdf_ortools = stats.lognorm.pdf(x_range, shape_ortools, loc_ortools, scale_ortools)
    ax1.plot(x_range, pdf_ortools, 'r-', linewidth=2, 
             label=f'Log-normal fit\n(σ={shape_ortools:.3f}, μ={np.log(scale_ortools):.3f})')
    
    ax1.set_xlabel('CPC (Cost Per Customer)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title(f'OR-Tools Greedy (optimal)\nN={n_ortools}, Capacity={cap_ortools}\n{len(cpc_ortools)} instances', 
                  fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min, x_max)
    
    # Plot 2: GPU DP exact results
    ax2.hist(cpc_gpu_array, bins=bins, density=True, alpha=0.7, 
             color='lightcoral', edgecolor='black', label='GPU DP exact data')
    
    # Fit and plot log-normal for GPU
    shape_gpu, loc_gpu, scale_gpu = stats.lognorm.fit(cpc_gpu_array, floc=0)
    pdf_gpu = stats.lognorm.pdf(x_range, shape_gpu, loc_gpu, scale_gpu)
    ax2.plot(x_range, pdf_gpu, 'r-', linewidth=2, 
             label=f'Log-normal fit\n(σ={shape_gpu:.3f}, μ={np.log(scale_gpu):.3f})')
    
    ax2.set_xlabel('CPC (Cost Per Customer)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title(f'GPU DP Exact (suboptimal)\nN={n_gpu}, Capacity={cap_gpu}\n{len(cpc_gpu)} instances', 
                  fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min, x_max)
    
    plt.suptitle('CPC Distribution Comparison: OR-Tools vs GPU DP Exact', fontsize=15)
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved comparison figure to {output_file}")
    else:
        plt.show()
    
    # Print statistics comparison
    print("\n📊 Statistics Comparison:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean CPC':<12} {'Std CPC':<12} {'Instances':<10}")
    print("-" * 60)
    print(f"{'OR-Tools (Greedy)':<20} {np.mean(cpc_ortools_array):<12.6f} {np.std(cpc_ortools_array):<12.6f} {len(cpc_ortools):<10}")
    print(f"{'GPU DP Exact':<20} {np.mean(cpc_gpu_array):<12.6f} {np.std(cpc_gpu_array):<12.6f} {len(cpc_gpu):<10}")
    print("-" * 60)
    
    # Print distribution parameters
    print("\n📊 Log-normal Distribution Parameters:")
    print("-" * 60)
    print(f"{'Method':<20} {'Shape (σ)':<12} {'Scale (exp(μ))':<15}")
    print("-" * 60)
    print(f"{'OR-Tools (Greedy)':<20} {shape_ortools:<12.4f} {scale_ortools:<15.4f}")
    print(f"{'GPU DP Exact':<20} {shape_gpu:<12.4f} {scale_gpu:<15.4f}")
    print("-" * 60)
    
    # KS test for both
    ks_stat_ortools, ks_p_ortools = stats.kstest(
        cpc_ortools_array, 
        lambda x: stats.lognorm.cdf(x, shape_ortools, loc_ortools, scale_ortools)
    )
    ks_stat_gpu, ks_p_gpu = stats.kstest(
        cpc_gpu_array, 
        lambda x: stats.lognorm.cdf(x, shape_gpu, loc_gpu, scale_gpu)
    )
    
    print("\n📊 Kolmogorov-Smirnov Test Results:")
    print("-" * 60)
    print(f"{'Method':<20} {'KS Statistic':<15} {'p-value':<15} {'Result'}")
    print("-" * 60)
    print(f"{'OR-Tools (Greedy)':<20} {ks_stat_ortools:<15.4f} {ks_p_ortools:<15.4f} {'✅ Log-normal' if ks_p_ortools > 0.05 else '⚠️  Not log-normal'}")
    print(f"{'GPU DP Exact':<20} {ks_stat_gpu:<15.4f} {ks_p_gpu:<15.4f} {'✅ Log-normal' if ks_p_gpu > 0.05 else '⚠️  Not log-normal'}")
    print("-" * 60)
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Compare OR-Tools and GPU DP exact CPC histograms with log-normal fits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ortools', type=str, default='ortools_n10.json',
                        help='OR-Tools results JSON file')
    parser.add_argument('--gpu', type=str, 
                        default='../../../benchmark_gpu/results/exact_dp_10:30_1000k/gpu_dp_exact_results_20250908_092225.json',
                        help='GPU DP exact results JSON file')
    parser.add_argument('--output', type=str, default='ortools_vs_gpu_comparison.png',
                        help='Output figure file')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins')
    
    args = parser.parse_args()
    
    try:
        # Check if files exist
        ortools_path = Path(args.ortools)
        if not ortools_path.exists():
            ortools_path = Path.cwd() / args.ortools
        
        gpu_path = Path(args.gpu)
        if not gpu_path.is_absolute():
            gpu_path = Path.cwd() / args.gpu
        
        if not ortools_path.exists():
            print(f"❌ OR-Tools file not found: {args.ortools}", file=sys.stderr)
            return 1
        
        if not gpu_path.exists():
            print(f"❌ GPU file not found: {args.gpu}", file=sys.stderr)
            print(f"   Searched at: {gpu_path.absolute()}")
            return 1
        
        print(f"📁 Loading data:")
        print(f"   OR-Tools: {ortools_path}")
        print(f"   GPU DP:   {gpu_path}")
        
        # Create comparison plot
        plot_comparison_histograms(ortools_path, gpu_path, args.output, args.bins)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
