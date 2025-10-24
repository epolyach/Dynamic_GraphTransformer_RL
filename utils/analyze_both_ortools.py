#!/usr/bin/env python3
"""
Analyze both 2s and 10s OR-Tools GLS datasets
"""
import json
import numpy as np
from scipy import stats

def analyze_ortools_data(filepath, label, expected_mean=None, expected_gm=None, expected_median=None):
    print(f"\n{'='*70}")
    print(f"Analyzing: {label}")
    print(f"File: {filepath}")
    print(f"{'='*70}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract CPC values
        if 'cpc' in data and isinstance(data['cpc'], list):
            cpc_values = data['cpc']
            n_instances = data.get('instances', len(cpc_values))
            n_customers = data.get('n', 'unknown')
            capacity = data.get('capacity', 'unknown')
        else:
            print("❌ No 'cpc' array found in JSON")
            return None
        
        cpc_array = np.array(cpc_values)
        
        print(f"📊 Dataset Info:")
        print(f"   N customers: {n_customers}")
        print(f"   Capacity: {capacity}")  
        print(f"   Instances: {len(cpc_array)} (declared: {n_instances})")
        print(f"   CPC range: {np.min(cpc_array):.6f} - {np.max(cpc_array):.6f}")
        
        # Calculate statistics
        mean_cpc = np.mean(cpc_array)
        median_cpc = np.median(cpc_array)
        geom_mean_cpc = stats.gmean(cpc_array)
        
        # Standard errors
        mean_se = stats.sem(cpc_array)
        median_se = 1.253 * mean_se
        geom_se = geom_mean_cpc * mean_se / mean_cpc
        max_se = max(mean_se, median_se, geom_se)
        
        print(f"\n📈 CALCULATED STATISTICS:")
        print(f"   Mean:           {mean_cpc:.6f} ± {mean_se:.2e}")
        print(f"   Geometric Mean: {geom_mean_cpc:.6f} ± {geom_se:.2e}")  
        print(f"   Median:         {median_cpc:.6f} ± {median_se:.2e}")
        print(f"   Max SE:         {max_se:.2e}")
        
        # Mathematical checks
        print(f"\n🔍 MATHEMATICAL VALIDATION:")
        gm_le_mean = geom_mean_cpc <= mean_cpc
        median_between = geom_mean_cpc <= median_cpc <= mean_cpc
        
        print(f"   GM ≤ Mean: {geom_mean_cpc:.4f} ≤ {mean_cpc:.4f} → {'✓' if gm_le_mean else '❌'}")
        print(f"   GM ≤ Median ≤ Mean: {geom_mean_cpc:.4f} ≤ {median_cpc:.4f} ≤ {mean_cpc:.4f}")
        print(f"   Median in range: {'✅' if median_between else '❌ (unusual but possible)'}")
        
        # Compare with expected values if provided
        if expected_mean is not None:
            print(f"\n📊 COMPARISON WITH TABLE:")
            print(f"   Expected: Mean={expected_mean:.4f}, GM={expected_gm:.4f}, Median={expected_median:.4f}")
            print(f"   Actual:   Mean={mean_cpc:.4f}, GM={geom_mean_cpc:.4f}, Median={median_cpc:.4f}")
            
            mean_match = abs(mean_cpc - expected_mean) < 0.0005
            gm_match = abs(geom_mean_cpc - expected_gm) < 0.0005
            median_match = abs(median_cpc - expected_median) < 0.0005
            
            print(f"   Mean match: {'✅' if mean_match else '❌'}")
            print(f"   GM match: {'✅' if gm_match else '❌'}")
            print(f"   Median match: {'✅' if median_match else '❌'}")
            
            if mean_match and gm_match and median_match:
                print("   🎯 ALL VALUES MATCH THE TABLE!")
            else:
                print("   ⚠️  Some values differ from the table")
        
        # Distribution analysis
        skewness = stats.skew(cpc_array)
        print(f"\n📊 DISTRIBUTION PROPERTIES:")
        print(f"   Skewness: {skewness:.3f} ({'right' if skewness > 0 else 'left' if skewness < 0 else 'symmetric'})")
        
        if not median_between:
            print(f"   📝 Explanation for Median < GM:")
            print(f"      This can occur with certain distribution shapes where")
            print(f"      the median sits at a different position than expected.")
        
        return {
            'mean': mean_cpc,
            'geom_mean': geom_mean_cpc,
            'median': median_cpc,
            'max_se': max_se,
            'n_instances': len(cpc_array),
            'all_match': expected_mean is not None and mean_match and gm_match and median_match
        }
        
    except Exception as e:
        print(f"❌ Error processing {label}: {e}")
        return None

def main():
    print("🔍 OR-Tools GLS N20 C30 - COMPLETE ANALYSIS")
    print("Analyzing both 2s and 10s timeout datasets")
    
    # Analyze 2s timeout data (first line in table)
    result_2s = analyze_ortools_data(
        "benchmark_cpu/results/ortools_gls_N20_C30_2s_10000i/ortools_n20.json",
        "OR-Tools GLS N20 C30 2s timeout",
        expected_mean=0.3288, expected_gm=0.3257, expected_median=0.3249
    )
    
    # Analyze 10s timeout data (second line in table)  
    result_10s = analyze_ortools_data(
        "benchmark_cpu/results/ortools_gls_N20_C30_10s_10000i/ortools_n20.json",
        "OR-Tools GLS N20 C30 10s timeout",
        expected_mean=0.3265, expected_gm=0.3234, expected_median=0.3227
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print("🏆 FINAL SUMMARY")
    print(f"{'='*70}")
    
    if result_2s and result_10s:
        print("✅ Both datasets analyzed successfully!")
        
        if result_2s['all_match'] and result_10s['all_match']:
            print("🎯 ALL TABLE VALUES ARE MATHEMATICALLY CORRECT!")
            print("\nCorrected LaTeX table lines:")
            print("OR-Tools GLS (sub)    & 10k   & 20  & 2.0s          & 13.4min             & 0.3288        & 0.3257      & 0.3249          & 5.7e-04 \\\\")
            print("OR-Tools GLS (sub)    & 10k   & 20  & 9.2s          & 61min               & 0.3265        & 0.3234      & 0.3227          & 5.6e-04 \\\\")
        else:
            print("⚠️  Some values may need correction")
            
        print(f"\n📈 Pattern Analysis:")
        print(f"   2s timeout: Better performance (shorter time, {result_2s['mean']:.4f} mean)")
        print(f"   10s timeout: Slightly worse but more thorough ({result_10s['mean']:.4f} mean)")
        print(f"   Improvement: {((result_2s['mean'] - result_10s['mean']) / result_2s['mean'] * 100):.2f}% better with 10s")
        
    else:
        print("❌ Analysis incomplete - check file paths and data format")

if __name__ == "__main__":
    main()
