#!/usr/bin/env python3
"""
Statistical tests for CPC distribution normality with command line interface.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path

def load_cpc_data_from_json(file_path):
    """Load CPC data from a single JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Processing JSON file: {file_path}")
        
        cpc_values = []
        
        if isinstance(data, dict):
            # Check for all_costs array and calculate CPC
            if 'all_costs' in data and 'n_customers' in data:
                costs = data['all_costs']
                n_customers = data['n_customers']
                cpc_values = [cost / n_customers for cost in costs]
                print(f"  Calculated {len(cpc_values)} CPC values from costs")
                
            # Check for all_cpcs array directly
            elif 'all_cpcs' in data:
                cpc_values = data['all_cpcs']
                print(f"  Found {len(cpc_values)} CPC values directly")
                
            # Check for direct CPC values or mean_cpc only
            elif 'mean_cpc' in data:
                print(f"  Found only mean_cpc: {data['mean_cpc']} (no individual values)")
                return np.array([])
                
        elif isinstance(data, list):
            # Handle list of results
            for entry in data:
                if isinstance(entry, dict):
                    if 'cpc' in entry:
                        cpc_values.append(entry['cpc'])
                    elif 'cost' in entry and 'n_customers' in entry:
                        cpc_values.append(entry['cost'] / entry['n_customers'])
            print(f"  Extracted {len(cpc_values)} CPC values from list")
            
        return np.array(cpc_values)
        
    except (json.JSONDecodeError, KeyError, TypeError, FileNotFoundError) as e:
        print(f"Error: Could not process {file_path}: {e}")
        return np.array([])

def load_cpc_data_from_csv(file_path):
    """Load CPC data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Processing CSV file: {file_path}")
        print(f"  CSV columns: {list(df.columns)}")
        
        # Try to find CPC column
        cpc_values = []
        
        if 'cpc' in df.columns:
            cpc_values = df['cpc'].dropna().values
            print(f"  Found {len(cpc_values)} CPC values in 'cpc' column")
            
        elif 'cost' in df.columns and 'n_customers' in df.columns:
            # Calculate CPC from cost and n_customers
            costs = df['cost'].dropna()
            n_customers = df['n_customers'].dropna()
            if len(costs) == len(n_customers):
                cpc_values = (costs / n_customers).values
                print(f"  Calculated {len(cpc_values)} CPC values from cost/n_customers")
            else:
                print("  Error: cost and n_customers columns have different lengths")
                
        elif 'cost' in df.columns:
            # Assume a fixed number of customers (ask user or use default)
            costs = df['cost'].dropna()
            print("  Found 'cost' column but no 'n_customers' column")
            n_customers = input("  Enter number of customers (default 10): ") or "10"
            try:
                n_customers = int(n_customers)
                cpc_values = (costs / n_customers).values
                print(f"  Calculated {len(cpc_values)} CPC values using n_customers={n_customers}")
            except ValueError:
                print("  Error: Invalid number of customers")
                return np.array([])
        else:
            print("  Error: No suitable columns found for CPC calculation")
            print("  Expected: 'cpc' column or 'cost' + 'n_customers' columns")
            return np.array([])
            
        return cpc_values
        
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
        print(f"Error: Could not process CSV file {file_path}: {e}")
        return np.array([])

def test_normality(data, alpha=0.05):
    """Perform multiple normality tests."""
    results = {}
    
    # Remove any infinite or NaN values
    data = data[np.isfinite(data)]
    
    if len(data) < 3:
        print("Error: Not enough data points for testing")
        return None
    
    print(f"\n{'='*60}")
    print("NORMALITY TEST RESULTS")
    print(f"{'='*60}")
    print(f"Sample size: {len(data):,}")
    print(f"Mean: {np.mean(data):.6f}")
    print(f"Standard deviation: {np.std(data):.6f}")
    print(f"Minimum: {np.min(data):.6f}")
    print(f"Maximum: {np.max(data):.6f}")
    print(f"Median: {np.median(data):.6f}")
    print(f"25th percentile: {np.percentile(data, 25):.6f}")
    print(f"75th percentile: {np.percentile(data, 75):.6f}")
    print(f"Skewness: {stats.skew(data):.4f}")
    print(f"Kurtosis: {stats.kurtosis(data):.4f}")
    print("-" * 60)
    
    # 1. Shapiro-Wilk test (most powerful for small-medium samples)
    if len(data) <= 5000:
        stat, p_value = stats.shapiro(data)
        results['Shapiro-Wilk'] = {'statistic': stat, 'p_value': p_value}
        print(f"1. Shapiro-Wilk test:")
        print(f"   Statistic: {stat:.6f}")
        print(f"   P-value: {p_value:.8f}")
        print(f"   Result: {'✓ Normal' if p_value > alpha else '✗ Not Normal'} (α = {alpha})")
        print()
    else:
        print(f"1. Shapiro-Wilk test: Skipped (n={len(data):,} > 5000)")
        print()
    
    # 2. Kolmogorov-Smirnov test
    stat, p_value = stats.kstest(data, lambda x: stats.norm.cdf(x, np.mean(data), np.std(data)))
    results['Kolmogorov-Smirnov'] = {'statistic': stat, 'p_value': p_value}
    print(f"2. Kolmogorov-Smirnov test:")
    print(f"   Statistic: {stat:.6f}")
    print(f"   P-value: {p_value:.8f}")
    print(f"   Result: {'✓ Normal' if p_value > alpha else '✗ Not Normal'} (α = {alpha})")
    print()
    
    # 3. Anderson-Darling test
    stat, critical_values, significance_levels = stats.anderson(data, dist='norm')
    results['Anderson-Darling'] = {'statistic': stat, 'critical_values': critical_values}
    print(f"3. Anderson-Darling test:")
    print(f"   Statistic: {stat:.6f}")
    for i, (cv, sl) in enumerate(zip(critical_values, significance_levels)):
        result = '✓ Normal' if stat < cv else '✗ Not Normal'
        print(f"   At {sl:4.1f}% significance level: {result} (critical value: {cv:.4f})")
    print()
    
    # 4. D'Agostino and Pearson's test
    if len(data) >= 20:
        stat, p_value = stats.normaltest(data)
        results['DAgostino-Pearson'] = {'statistic': stat, 'p_value': p_value}
        print(f"4. D'Agostino and Pearson's test:")
        print(f"   Statistic: {stat:.6f}")
        print(f"   P-value: {p_value:.8f}")
        print(f"   Result: {'✓ Normal' if p_value > alpha else '✗ Not Normal'} (α = {alpha})")
        print()
    
    # 5. Jarque-Bera test
    if len(data) >= 2000:
        stat, p_value = stats.jarque_bera(data)
        results['Jarque-Bera'] = {'statistic': stat, 'p_value': p_value}
        print(f"5. Jarque-Bera test:")
        print(f"   Statistic: {stat:.6f}")
        print(f"   P-value: {p_value:.8f}")
        print(f"   Result: {'✓ Normal' if p_value > alpha else '✗ Not Normal'} (α = {alpha})")
        print()
    
    return results

def plot_normality_diagnostics(data, output_file='cpc_normality_plots.png'):
    """Create diagnostic plots for normality assessment."""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram with normal overlay
    n_bins = min(50, max(10, int(np.sqrt(len(data)))))
    axes[0, 0].hist(data, bins=n_bins, density=True, alpha=0.7, color='skyblue', 
                    edgecolor='black', linewidth=0.5)
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(data.min(), data.max(), 200)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal PDF')
    axes[0, 0].set_title(f'Histogram with Normal Overlay\n(n={len(data):,})', fontsize=12)
    axes[0, 0].set_xlabel('CPC Values')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot
    bp = axes[1, 0].boxplot(data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[1, 0].set_title('Box Plot', fontsize=12)
    axes[1, 0].set_ylabel('CPC Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Empirical vs theoretical CDF
    x_sorted = np.sort(data)
    y_empirical = np.arange(1, len(data) + 1) / len(data)
    y_theoretical = stats.norm.cdf(x_sorted, mu, sigma)
    
    axes[1, 1].plot(x_sorted, y_empirical, 'b-', linewidth=2, label='Empirical CDF')
    axes[1, 1].plot(x_sorted, y_theoretical, 'r--', linewidth=2, label='Normal CDF')
    axes[1, 1].set_title('Empirical vs Theoretical CDF', fontsize=12)
    axes[1, 1].set_xlabel('CPC Values')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Normality diagnostic plots saved as: {output_file}")
    
def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Test CPC distribution for normality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_cpc_normality_cl.py data.json
  python test_cpc_normality_cl.py results.csv
  python test_cpc_normality_cl.py data.json --alpha 0.01 --plot output.png
        """
    )
    
    parser.add_argument('file', help='Input file (JSON or CSV)')
    parser.add_argument('--alpha', type=float, default=0.05, 
                       help='Significance level for tests (default: 0.05)')
    parser.add_argument('--plot', type=str, default='cpc_normality_plots.png',
                       help='Output file for diagnostic plots (default: cpc_normality_plots.png)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file).exists():
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    
    # Determine file type and load data
    file_path = Path(args.file)
    if file_path.suffix.lower() == '.json':
        cpc_data = load_cpc_data_from_json(args.file)
    elif file_path.suffix.lower() == '.csv':
        cpc_data = load_cpc_data_from_csv(args.file)
    else:
        print(f"Error: Unsupported file type '{file_path.suffix}'. Use .json or .csv")
        sys.exit(1)
    
    if len(cpc_data) == 0:
        print("Error: No CPC data found in the file")
        sys.exit(1)
    
    print(f"✅ Successfully loaded {len(cpc_data):,} CPC values")
    
    # Test normality
    results = test_normality(cpc_data, alpha=args.alpha)
    
    if results:
        # Create diagnostic plots
        plot_normality_diagnostics(cpc_data, args.plot)
        
        # Summary
        print("=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        
        normal_count = 0
        total_tests = 0
        
        for test_name, result in results.items():
            if 'p_value' in result:
                total_tests += 1
                if result['p_value'] > args.alpha:
                    normal_count += 1
                    print(f"✅ {test_name}: Normal (p = {result['p_value']:.6f})")
                else:
                    print(f"❌ {test_name}: Not Normal (p = {result['p_value']:.6f})")
            else:
                # Anderson-Darling doesn't have a single p-value
                if test_name == 'Anderson-Darling':
                    stat = result['statistic']
                    cv_5pct = result['critical_values'][2]  # 5% significance level
                    if stat < cv_5pct:
                        normal_count += 1
                        print(f"✅ {test_name}: Normal (stat = {stat:.4f} < {cv_5pct:.4f})")
                    else:
                        print(f"❌ {test_name}: Not Normal (stat = {stat:.4f} ≥ {cv_5pct:.4f})")
                    total_tests += 1
                    
        print(f"\n📊 Tests suggesting normality: {normal_count}/{total_tests}")
        
        if normal_count >= total_tests * 0.5:
            print("🎯 CONCLUSION: Data appears to be approximately NORMAL")
            print("   ✅ You can use parametric statistical methods:")
            print("      • t-tests for comparing means")
            print("      • ANOVA for multiple group comparisons")
            print("      • Linear regression")
            print("      • Pearson correlations")
        else:
            print("⚠️  CONCLUSION: Data does NOT appear to be normal")
            print("   ❌ Consider using non-parametric statistical methods:")
            print("      • Mann-Whitney U test")
            print("      • Kruskal-Wallis test")
            print("      • Spearman correlation")
            print("      • Wilcoxon tests")

if __name__ == "__main__":
    main()
