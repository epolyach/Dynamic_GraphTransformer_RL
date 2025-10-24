#!/usr/bin/env python3
"""
Analyze OR-Tools GLS statistics and check for calculation errors
"""
import json
import numpy as np
from scipy import stats

def analyze_file(filepath, label):
    print(f"\n{'='*60}")
    print(f"Analyzing: {label}")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract CPC values
        if 'results' in data and isinstance(data['results'], list):
            results = data['results']
            cpc_values = [r.get('cpc', r.get('cost_per_customer', None)) for r in results]
            cpc_values = [x for x in cpc_values if x is not None]
        else:
            # Try different structure
            cpc_values = []
            for key, value in data.items():
                if isinstance(value, dict) and 'cpc' in value:
                    cpc_values.append(value['cpc'])
                elif isinstance(value, (int, float)):
                    cpc_values.append(value)
        
        if not cpc_values:
            print("❌ No CPC values found in the data structure")
            print("Available keys:", list(data.keys()) if isinstance(data, dict) else "Not a dict")
            return
        
        cpc_array = np.array(cpc_values)
        n_instances = len(cpc_array)
        
        print(f"📊 Found {n_instances} instances")
        print(f"CPC range: {np.min(cpc_array):.6f} - {np.max(cpc_array):.6f}")
        
        # Calculate statistics
        mean_cpc = np.mean(cpc_array)
        median_cpc = np.median(cpc_array)
        
        # Geometric mean (handle potential zeros/negatives)
        if np.all(cpc_array > 0):
            geom_mean_cpc = stats.gmean(cpc_array)
        else:
            print("⚠️  Some CPC values are <= 0, using exp(mean(log)) for GM")
            geom_mean_cpc = np.exp(np.mean(np.log(cpc_array + 1e-10)))
        
        # Standard errors
        mean_se = stats.sem(cpc_array)
        median_se = 1.253 * mean_se  # approximate median SE
        geom_se = geom_mean_cpc * mean_se / mean_cpc  # approximate GM SE
        max_se = max(mean_se, median_se, geom_se)
        
        print(f"\n📈 CALCULATED STATISTICS:")
        print(f"Mean:     {mean_cpc:.6f} ± {mean_se:.6f}")
        print(f"Geometric Mean: {geom_mean_cpc:.6f} ± {geom_se:.6f}")  
        print(f"Median:   {median_cpc:.6f} ± {median_se:.6f}")
        print(f"Max SE:   {max_se:.6f}")
        
        # Check mathematical relationship
        print(f"\n🔍 MATHEMATICAL CHECKS:")
        print(f"GM ≤ Mean:     {geom_mean_cpc:.6f} ≤ {mean_cpc:.6f} → {'✓' if geom_mean_cpc <= mean_cpc else '❌'}")
        print(f"Median position: {geom_mean_cpc:.6f} ≤ {median_cpc:.6f} ≤ {mean_cpc:.6f}")
        
        # Check if median is between GM and mean
        median_in_range = geom_mean_cpc <= median_cpc <= mean_cpc
        print(f"Median in [GM, Mean]: {'✓' if median_in_range else '❌'}")
        
        if not median_in_range:
            print("⚠️  WARNING: Median should typically be between GM and Mean!")
            if median_cpc < geom_mean_cpc:
                print("   Median < GM suggests potential calculation error or unusual distribution")
            if median_cpc > mean_cpc:
                print("   Median > Mean suggests potential calculation error")
        
        # Distribution analysis
        skewness = stats.skew(cpc_array)
        kurtosis = stats.kurtosis(cpc_array)
        print(f"\n📊 DISTRIBUTION ANALYSIS:")
        print(f"Skewness: {skewness:.3f} ({'right' if skewness > 0 else 'left'} skewed)")
        print(f"Kurtosis: {kurtosis:.3f}")
        
        # Look for potential data issues
        print(f"\n🔧 DATA QUALITY CHECKS:")
        n_unique = len(np.unique(cpc_array))
        print(f"Unique values: {n_unique}/{n_instances} ({100*n_unique/n_instances:.1f}%)")
        
        if n_unique < n_instances * 0.5:
            print("⚠️  Low uniqueness - possible data duplication")
        
        # Check for outliers
        q1, q3 = np.percentile(cpc_array, [25, 75])
        iqr = q3 - q1
        outlier_bounds = [q1 - 1.5*iqr, q3 + 1.5*iqr]
        outliers = cpc_array[(cpc_array < outlier_bounds[0]) | (cpc_array > outlier_bounds[1])]
        print(f"Outliers: {len(outliers)} instances outside [{outlier_bounds[0]:.4f}, {outlier_bounds[1]:.4f}]")
        
        return {
            'mean': mean_cpc,
            'geom_mean': geom_mean_cpc, 
            'median': median_cpc,
            'mean_se': mean_se,
            'max_se': max_se,
            'n_instances': n_instances
        }
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

def main():
    print("OR-Tools GLS Statistics Analysis")
    
    # Analyze the 10s file that exists
    result_10s = analyze_file(
        "benchmark_cpu/results/ortools_gls_N20_C30_10s_10000i/ortools_n20.json",
        "OR-Tools GLS N20 C30 10s (existing)"
    )
    
    # Try to find the 2s file
    print(f"\n{'='*60}")
    print("Searching for 2s timeout data...")
    
    # Check if directory exists
    import os
    dir_2s = "benchmark_cpu/results/ortools_gls_N20_C30_2s_10000i"
    if os.path.exists(dir_2s):
        print(f"Directory exists: {dir_2s}")
        files_2s = os.listdir(dir_2s)
        print(f"Contents: {files_2s}")
        
        json_file_2s = os.path.join(dir_2s, "ortools_n20.json")
        if os.path.exists(json_file_2s):
            result_2s = analyze_file(json_file_2s, "OR-Tools GLS N20 C30 2s")
        else:
            print(f"❌ JSON file not found: {json_file_2s}")
    else:
        print(f"❌ Directory not found: {dir_2s}")
        print("Checking git history for this data...")

if __name__ == "__main__":
    main()
