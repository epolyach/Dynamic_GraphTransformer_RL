#!/usr/bin/env python3
"""
Create a single-panel figure equivalent to panel 4 in cpc_lognormal_test.png:
Histogram of log(CPC) with overlaid Normal PDF fit.
Outputs 80mm x 80mm figures in PNG and EPS formats.
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


def main():
    p = argparse.ArgumentParser(description='Make single-panel log(CPC) histogram with normal fit (80mm x 80mm)')
    p.add_argument('data_file', help='CSV or JSON file with CPC data (cpc column or all_cpcs field)')
    p.add_argument('--bins', type=int, default=50, help='Number of histogram bins (default: 50)')
    p.add_argument('--out-prefix', type=str, default='cpc_panel4_80mm', help='Output file prefix')
    args = p.parse_args()

    cpc = load_cpc(args.data_file)
    log_cpc = np.log(cpc)

    mu, sigma = float(np.mean(log_cpc)), float(np.std(log_cpc, ddof=0))

    # Figure size 80mm x 80mm
    fig, ax = plt.subplots(figsize=(mm_to_inches(80), mm_to_inches(80)))

    # Histogram
    ax.hist(log_cpc, bins=args.bins, density=True, alpha=0.7, color='lightcoral', edgecolor='black')

    # Overlay normal PDF
    x = np.linspace(log_cpc.min(), log_cpc.max(), 400)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')

    ax.set_title('log(CPC) Distribution', fontsize=10, fontweight='bold')
    ax.set_xlabel('log(CPC)')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()

    # Save PNG and EPS
    png = f"{args.out_prefix}.png"
    eps = f"{args.out_prefix}.eps"
    fig.savefig(png, dpi=600, bbox_inches='tight')
    fig.savefig(eps, dpi=600, bbox_inches='tight')
    print(f"Saved: {png}\nSaved: {eps}")


if __name__ == '__main__':
    main()
