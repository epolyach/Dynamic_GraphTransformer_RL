#!/usr/bin/env python3
"""
Create log-normal distribution figure for N=10 CVRP instances.
Final journal-quality version with specific formatting requirements.
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def load_cpc(data_file):
    ext = os.path.splitext(data_file)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(data_file)
        if 'cpc' not in df.columns:
            raise ValueError(f"'cpc' column not found in {data_file}; columns={list(df.columns)}")
        return df['cpc'].to_numpy()
    elif ext == '.json':
        with open(data_file, 'r') as f:
            data = json.load(f)
        if 'all_cpcs' not in data:
            raise ValueError(f"'all_cpcs' field not found in {data_file}; keys={list(data.keys())}")
        return np.array(data['all_cpcs'], dtype=float)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (use .csv or .json)")

def mm_to_inches(mm):
    return mm / 25.4

def setup_matplotlib_for_journal():
    """Configure matplotlib for journal-quality figures."""
    # Set font to serif for professional look
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
    
    # Set font sizes for journal articles (typically 7-8pt)
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['legend.fontsize'] = 7
    
    # Set line widths to be thinner for cleaner look
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    
    # Grid settings for fine grid
    plt.rcParams['grid.linewidth'] = 0.4
    plt.rcParams['grid.alpha'] = 0.4
    
    # Other settings
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fancybox'] = False

def main():
    p = argparse.ArgumentParser(description='Make log-normal distribution figure for N=10 (85mm x 65mm)')
    p.add_argument('data_file', help='CSV or JSON file with CPC data (cpc column or all_cpcs field)')
    p.add_argument('--bins', type=int, default=40, help='Number of histogram bins (default: 40)')
    p.add_argument('--out-prefix', type=str, default='log_norm_N10', help='Output file prefix')
    args = p.parse_args()

    # Configure matplotlib for journal quality
    setup_matplotlib_for_journal()
    
    cpc = load_cpc(args.data_file)
    log_cpc = np.log(cpc)

    mu, sigma = float(np.mean(log_cpc)), float(np.std(log_cpc, ddof=1))

    # Figure size 85mm x 65mm for journal
    fig, ax = plt.subplots(figsize=(mm_to_inches(85), mm_to_inches(65)))

    # Histogram with black edges
    n, bins, patches = ax.hist(log_cpc, bins=args.bins, density=True, alpha=0.7, 
                              color='lightcoral', edgecolor='black', linewidth=0.5)

    # Overlay normal PDF with thin line
    x = np.linspace(log_cpc.min(), log_cpc.max(), 400)
    normal_pdf = stats.norm.pdf(x, mu, sigma)
    line = ax.plot(x, normal_pdf, 'r-', linewidth=1.5, 
                   label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')

    # Axes labels
    ax.set_xlabel('log(CPC)')
    ax.set_ylabel('Density')
    
    # Set y-axis range from 0 to 2.5 as requested
    ax.set_ylim(0, 2.5)
    
    # Fine grid
    ax.grid(True, alpha=0.4, linewidth=0.4)
    
    # Position legend in upper left as requested
    ax.legend(loc='upper left', frameon=False, fontsize=7)
    
    # Adjust tick parameters - NO minor ticks as requested
    ax.tick_params(axis='both', which='major', labelsize=7, width=0.8)
    
    # Explicitly disable minor ticks
    ax.minorticks_off()
    
    # Tight layout with small padding
    plt.tight_layout(pad=0.2)

    # Save with requested filenames
    png_file = f"{args.out_prefix}.png"
    eps_file = f"{args.out_prefix}.eps"
    
    fig.savefig(png_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(eps_file, dpi=600, bbox_inches='tight', pad_inches=0.02, format='eps')
    
    plt.close(fig)
    
    print(f"Log-normal distribution figure saved:")
    print(f"  {png_file}")
    print(f"  {eps_file}")
    print(f"Settings applied:")
    print(f"  - Size: 85mm × 65mm")
    print(f"  - Font sizes: 7-8pt (journal standard)")
    print(f"  - Y-axis range: 0 to 2.5")
    print(f"  - Legend: upper left")
    print(f"  - Minor ticks: disabled")

if __name__ == '__main__':
    main()
