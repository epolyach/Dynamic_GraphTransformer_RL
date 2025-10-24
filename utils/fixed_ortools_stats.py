#!/usr/bin/env python3
"""
Fixed analysis for OR-Tools GLS statistics
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
        
        # Extract CPC values from the correct structure
        if 'cpc' in data and isinstance(data['cpc'], list):
            cpc_values = data['cpc']
        else:
            print("❌ No 'cpc' array found in JSON")
            print("Available keys:", list(data.keys()))
            return None
        
        cpc_array = np.array(cpc_values)
        n_instances = len(cpc_array)
        
        print(f"📊 Found {n_instances} instances")
        print(f"CPC range: {np.min(cpc_array):.6f} - {np.max(cpc_array):.6f}")
        
        # Calculate statistics
        mean_cpc = np.mean(cpc_array)
        median_cpc = np.median(cpc_array)
        geom_mean_cpc = stats.gmean(cpc_array)
        
        # Standard errors
        mean_se = stats.sem(cpc_array)
        median_se = 1.253 * mean_se  # approximate median SE
        geom_se = geom_mean_cpc * mean_se / mean_cpc  # approximate GM SE
        max_se = max(mean_se, median_se, geom_se)
        
        print(f"\n📈 RECALCULATED STATISTICS:")
        print(f"Mean:           {mean_cpc:.6f} ± {mean_se:.6f}")
        print(f"Geometric Mean: {geom_mean_cpc:.6f} ± {geom_se:.6f}")  
        print(f"Median:         {median_cpc:.6f} ± {median_se:.6f}")
        print(f"Max SE:         {max_se:.6f}")
        
        # Check mathematical relationship
        print(f"\n🔍 MATHEMATICAL CHECKS:")
        print(f"GM ≤ Mean:     {geom_mean_cpc:.6f} ≤ {mean_cpc:.6f} → {'✓' if geom_mean_cpc <= mean_cpc else '❌'}")
        print(f"GM ≤ Median ≤ Mean: {geom_mean_cpc:.6f} ≤ {median_cpc:.6f} ≤ {mean_cpc:.6f}")
        
        # Check if median is between GM and mean (correct order)
        median_in_range = geom_mean_cpc <= median_cpc <= mean_cpc
        print(f"Median in [GM, Mean]: {'✓' if median_in_range else '❌'}")
        
        if not median_in_range:
            print("⚠️  WARNING: Median should be between GM and Mean!")
            if median_cpc < geom_mean_cpc:
                print("   → Median < GM: Unusual distribution or calculation error")
            if median_cpc > mean_cpc:
                print("   → Median > Mean: Very unusual distribution")
        else:
            print("✅ Statistics are mathematically consistent!")
        
        # Compare with table values
        print(f"\n📊 TABLE COMPARISON:")
        print(f"Your table shows: Mean=0.3265, GM=0.3234, Median=0.3227")
        print(f"Our calculation:  Mean={mean_cpc:.4f}, GM={geom_mean_cpc:.4f}, Median={median_cpc:.4f}")
        
        if abs(mean_cpc - 0.3265) < 0.001 and abs(geom_mean_cpc - 0.3234) < 0.001:
            print("✅ Values match the table!")
        else:
            print("❌ Values differ from the table - possible wrong file?")
        
        return {
            'mean': mean_cpc,
            'geom_mean': geom_mean_cpc, 
            'median': median_cpc,
            'max_se': max_se,
            'n_instances': n_instances
        }
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

def main():
    print("OR-Tools GLS Statistics - FIXED ANALYSIS")
    
    # Analyze the 10s file that exists
    result_10s = analyze_file(
        "benchmark_cpu/results/ortools_gls_N20_C30_10s_10000i/ortools_n20.json",
        "OR-Tools GLS N20 C30 10s"
    )
    
    # Try to restore 2s file from git
    print(f"\n{'='*60}")
    print("Attempting to restore 2s timeout data from git...")
    
    import subprocess
    import os
    
    # Check if we can find the 2s JSON file in git history
    try:
        result = subprocess.run([
            'git', 'log', '--all', '--full-history', '--', 
            'benchmark_cpu/results/ortools_gls_N20_C30_2s_10000i/ortools_n20.json'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0 and result.stdout.strip():
            print("📁 Found 2s JSON file in git history!")
            # Extract the commit hash
            lines = result.stdout.split('\n')
            commit_hash = None
            for line in lines:
                if line.startswith('commit '):
                    commit_hash = line.split()[1]
                    break
            
            if commit_hash:
                print(f"💾 Attempting to restore from commit: {commit_hash[:8]}")
                # Try to get the file content
                restore_result = subprocess.run([
                    'git', 'show', 
                    f'{commit_hash}:benchmark_cpu/results/ortools_gls_N20_C30_2s_10000i/ortools_n20.json'
                ], capture_output=True, text=True, cwd='.')
                
                if restore_result.returncode == 0:
                    # Temporarily save the file
                    temp_file = '/tmp/ortools_n20_2s.json'
                    with open(temp_file, 'w') as f:
                        f.write(restore_result.stdout)
                    
                    result_2s = analyze_file(temp_file, "OR-Tools GLS N20 C30 2s (restored)")
                    os.unlink(temp_file)
                else:
                    print("❌ Could not restore file content")
            else:
                print("❌ Could not find commit hash")
        else:
            print("❌ 2s JSON file not found in git history")
    except Exception as e:
        print(f"❌ Error accessing git: {e}")

if __name__ == "__main__":
    main()
