#!/usr/bin/env python3
"""
Statistical tests for CPC distribution normality.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

def load_cpc_data(file_pattern="*.json"):
    """Load CPC data from JSON result files."""
    cpc_values = []
    
    json_files = glob.glob(file_pattern)
    print(f"Found {len(json_files)} JSON files: {json_files}")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"Processing {file_path}...")
            
            # Extract costs and calculate CPC
            if isinstance(data, dict):
                # Check for all_costs array
                if 'all_costs' in data and 'n_customers' in data:
                    costs = data['all_costs']
                    n_customers = data['n_customers']
                    
                    # Calculate CPC for each cost
                    cpc_list = [cost / n_customers for cost in costs]
                    cpc_values.extend(cpc_list)
                    print(f"  Added {len(cpc_list)} CPC values from {file_path}")
                    
                # Check for direct CPC values or mean_cpc
                elif 'mean_cpc' in data:
                    # If we only have mean, we can't test distribution
                    print(f"  Found mean_cpc: {data['mean_cpc']} but no individual values")
                    
            elif isinstance(data, list):
                # Handle list of results
                for entry in data:
                    if isinstance(entry, dict):
                        if 'cpc' in entry:
                            cpc_values.append(entry['cpc'])
                        elif 'cost' in entry and 'n_customers' in entry:
                            cpc_values.append(entry['cost'] / entry['n_customers'])
                            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Could not process {file_path}: {e}")
            continue
    
    return np.array(cpc_values)

def test_normality(data, alpha=0.05):
    """Perform multiple normality tests."""
    results = {}
    
    # Remove any infinite or NaN values
    data = data[np.isfinite(data)]
    
    if len(data) < 3:
        print("Error: Not enough data points for testing")
        return None
    
    print(f"\nTesting normality for {len(data)} CPC values")
    print(f"Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Skewness: {stats.skew(data):.4f}")
    print(f"Kurtosis: {stats.kurtosis(data):.4f}")
    print("-" * 50)
    
    # 1. Shapiro-Wilk test (most powerful for small-medium samples)
    if len(data) <= 5000:
        stat, p_value = stats.shapiro(data)
        results['Shapiro-Wilk'] = {'statistic': stat, 'p_value': p_value}
        print(f"Shapiro-Wilk test:")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Result: {'Normal' if p_value > alpha else 'Not Normal'} (α = {alpha})")
        print()
    else:
        print(f"Shapiro-Wilk test: Skipped (n={len(data)} > 5000)")
        print()
    
    # 2. Kolmogorov-Smirnov test
    # Compare with normal distribution with same mean and std
    stat, p_value = stats.kstest(data, lambda x: stats.norm.cdf(x, np.mean(data), np.std(data)))
    results['Kolmogorov-Smirnov'] = {'statistic': stat, 'p_value': p_value}
    print(f"Kolmogorov-Smirnov test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Result: {'Normal' if p_value > alpha else 'Not Normal'} (α = {alpha})")
    print()
    
    # 3. Anderson-Darling test
    stat, critical_values, significance_levels = stats.anderson(data, dist='norm')
    results['Anderson-Darling'] = {'statistic': stat, 'critical_values': critical_values}
    print(f"Anderson-Darling test:")
    print(f"  Statistic: {stat:.4f}")
    for i, (cv, sl) in enumerate(zip(critical_values, significance_levels)):
        result = 'Normal' if stat < cv else 'Not Normal'
        print(f"  At {sl}% significance level: {result} (critical value: {cv:.4f})")
    print()
    
    # 4. D'Agostino and Pearson's test
    if len(data) >= 20:
        stat, p_value = stats.normaltest(data)
        results['DAgostino-Pearson'] = {'statistic': stat, 'p_value': p_value}
        print(f"D'Agostino and Pearson's test:")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Result: {'Normal' if p_value > alpha else 'Not Normal'} (α = {alpha})")
        print()
    
    # 5. Jarque-Bera test
    if len(data) >= 2000:
        stat, p_value = stats.jarque_bera(data)
        results['Jarque-Bera'] = {'statistic': stat, 'p_value': p_value}
        print(f"Jarque-Bera test:")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Result: {'Normal' if p_value > alpha else 'Not Normal'} (α = {alpha})")
        print()
    
    return results

def plot_normality_diagnostics(data, filename='cpc_normality_plots.png'):
    """Create diagnostic plots for normality assessment."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram with normal overlay
    axes[0, 0].hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(data.min(), data.max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal PDF')
    axes[0, 0].set_title('Histogram with Normal Overlay')
    axes[0, 0].set_xlabel('CPC Values')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normal)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot
    axes[1, 0].boxplot(data, vert=True)
    axes[1, 0].set_title('Box Plot')
    axes[1, 0].set_ylabel('CPC Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Empirical vs theoretical CDF
    x_sorted = np.sort(data)
    y_empirical = np.arange(1, len(data) + 1) / len(data)
    y_theoretical = stats.norm.cdf(x_sorted, mu, sigma)
    
    axes[1, 1].plot(x_sorted, y_empirical, 'b-', linewidth=2, label='Empirical CDF')
    axes[1, 1].plot(x_sorted, y_theoretical, 'r--', linewidth=2, label='Normal CDF')
    axes[1, 1].set_title('Empirical vs Theoretical CDF')
    axes[1, 1].set_xlabel('CPC Values')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Normality diagnostic plots saved as {filename}")
    
    # Don't show plot in headless environment, just save it
    # plt.show()

def main():
    """Main function to test CPC normality."""
    print("Loading CPC data from JSON files...")
    
    # Load CPC data
    cpc_data = load_cpc_data("*results*.json")
    
    if len(cpc_data) == 0:
        print("No CPC data found in JSON files.")
        return
    
    print(f"\nFound {len(cpc_data)} CPC values")
    
    # Test normality
    results = test_normality(cpc_data)
    
    if results:
        # Create diagnostic plots
        plot_normality_diagnostics(cpc_data)
        
        # Summary
        print("=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        
        normal_count = 0
        total_tests = 0
        
        for test_name, result in results.items():
            if 'p_value' in result:
                total_tests += 1
                if result['p_value'] > 0.05:
                    normal_count += 1
                    print(f"✓ {test_name}: Normal (p = {result['p_value']:.6f})")
                else:
                    print(f"✗ {test_name}: Not Normal (p = {result['p_value']:.6f})")
            else:
                # Anderson-Darling doesn't have a single p-value
                if test_name == 'Anderson-Darling':
                    stat = result['statistic']
                    cv_5pct = result['critical_values'][2]  # 5% significance level
                    if stat < cv_5pct:
                        normal_count += 1
                        print(f"✓ {test_name}: Normal (stat = {stat:.4f} < {cv_5pct:.4f})")
                    else:
                        print(f"✗ {test_name}: Not Normal (stat = {stat:.4f} ≥ {cv_5pct:.4f})")
                    total_tests += 1
                    
        print(f"\nTests suggesting normality: {normal_count}/{total_tests}")
        
        if normal_count >= total_tests * 0.5:
            print("🎯 CONCLUSION: Data appears to be approximately normal")
            print("   You can use parametric statistical methods (t-tests, ANOVA, etc.)")
        else:
            print("⚠️  CONCLUSION: Data does not appear to be normal")
            print("   Consider using non-parametric statistical methods")
            print("   (Mann-Whitney U, Kruskal-Wallis, Spearman correlation, etc.)")

if __name__ == "__main__":
    main()
