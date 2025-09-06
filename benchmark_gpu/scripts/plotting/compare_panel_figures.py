#!/usr/bin/env python3
"""
Compare the old and new panel figure styles side by side.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def mm_to_inches(mm):
    return mm / 25.4

def load_cpc_data():
    """Load the largest available dataset"""
    with open('gpu_dp_exact_results_20250905_071235.json', 'r') as f:
        data = json.load(f)
    return np.array(data['all_cpcs'], dtype=float)

def create_comparison():
    cpc = load_cpc_data()
    log_cpc = np.log(cpc)
    mu, sigma = float(np.mean(log_cpc)), float(np.std(log_cpc, ddof=1))
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(mm_to_inches(170), mm_to_inches(80)))
    
    # LEFT: Old style (recreated)
    plt.rcParams['font.size'] = 10  # Reset to default large size
    ax1.hist(log_cpc, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    x = np.linspace(log_cpc.min(), log_cpc.max(), 400)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
             label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
    ax1.set_title('OLD: 80mm×80mm, Large fonts', fontsize=10, fontweight='bold')
    ax1.set_xlabel('log(CPC)')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, fontsize=8)
    
    # RIGHT: New style
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 7
    
    ax2.hist(log_cpc, bins=40, density=True, alpha=0.7, 
             color='lightcoral', edgecolor='black', linewidth=0.5)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=1.5,
             label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
    ax2.set_title('NEW: 85mm×65mm, Journal fonts', fontsize=8, fontweight='bold')
    ax2.set_xlabel('log(CPC)', fontsize=8)
    ax2.set_ylabel('Density', fontsize=8)
    ax2.grid(True, alpha=0.4, linewidth=0.4)
    ax2.legend(loc='upper right', frameon=False, fontsize=7)
    ax2.tick_params(axis='both', which='major', labelsize=7, width=0.8)
    ax2.minorticks_on()
    
    plt.tight_layout()
    plt.savefig('panel_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison figure saved: panel_comparison.png")
    print("\nKey improvements in the NEW version:")
    print("- Font size reduced from ~10pt to 7-8pt (journal standard)")
    print("- Line width reduced from 2.0 to 1.5 (cleaner look)")
    print("- Histogram border width reduced to 0.5 (less intrusive)")
    print("- Grid made finer (0.4 width vs 0.3 alpha)")
    print("- Minor ticks added for professional appearance")
    print("- Size optimized for journal: 85mm×65mm vs 80mm×80mm")
    print("- Legend positioned to avoid overlap")
    print("- Ready for figure captions (no title in actual use)")

if __name__ == '__main__':
    create_comparison()
