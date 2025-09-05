#!/usr/bin/env python3
"""
Monitor all OR-Tools benchmarks and generate comprehensive table
"""

import os
import time
import json
import glob
import numpy as np
from scipy import stats
from datetime import datetime
import subprocess

def find_latest_result(pattern):
    """Find the latest result file matching pattern"""
    files = glob.glob(pattern)
    if files:
        return sorted(files)[-1]
    return None

def compute_full_stats(cpcs):
    """Compute comprehensive statistics from CPC values"""
    cpcs = np.array(cpcs)
    log_cpcs = np.log(cpcs)
    n = len(cpcs)
    
    # Geometric statistics
    mu_log = log_cpcs.mean()
    sigma_log = log_cpcs.std(ddof=1)
    gm = np.exp(mu_log)
    gsd = np.exp(sigma_log)
    
    # 95% CI for GM
    se_log = sigma_log / np.sqrt(n)
    ci_lower = np.exp(mu_log - 1.96 * se_log)
    ci_upper = np.exp(mu_log + 1.96 * se_log)
    
    # Normality tests
    if n >= 20:
        ks_p = stats.kstest(log_cpcs, lambda x: stats.norm.cdf(x, loc=mu_log, scale=sigma_log)).pvalue
        dag_p = stats.normaltest(log_cpcs).pvalue if n >= 20 else np.nan
        jb_p = stats.jarque_bera(log_cpcs).pvalue if n >= 20 else np.nan
        ad_stat = stats.anderson(log_cpcs, dist='norm').statistic
    else:
        ks_p = dag_p = jb_p = ad_stat = np.nan
    
    return {
        'n': n,
        'gm': gm, 
        'gsd': gsd,
        'range_lower': gm * (gsd ** (-1.96)),
        'range_upper': gm * (gsd ** 1.96),
        'ci_lower': ci_lower, 
        'ci_upper': ci_upper,
        'ks_p': ks_p, 
        'dag_p': dag_p, 
        'jb_p': jb_p, 
        'ad_stat': ad_stat
    }

def generate_ortools_only_table(results):
    """Generate LaTeX table with OR-Tools GLS results only"""
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{OR-Tools GLS Performance on 10,000 random CVRP instances}")
    latex.append("\\begin{tabular}{rrrrrrrr}")
    latex.append("\\toprule")
    latex.append("N & Instances & GM & GSD & 95\\% Range & 95\\% CI(GM) & KS p-val & AD stat \\\\")
    latex.append("\\midrule")
    
    # Sort by N
    for n in sorted(results.keys()):
        data = results[n]
        stats = data['stats']
        
        row = f"{n} & "
        if data['instances'] < 10000:
            row += f"\\textbf{{{data['instances']}}}"
        else:
            row += f"{data['instances']}"
        
        row += f" & {stats['gm']:.4f} & {stats['gsd']:.4f} & "
        row += f"[{stats['range_lower']:.4f}, {stats['range_upper']:.4f}] & "
        row += f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] & "
        
        # P-value formatting
        if not np.isnan(stats['ks_p']):
            ks_str = f"{stats['ks_p']:.3f}" if stats['ks_p'] >= 0.001 else "<.001"
            if stats['ks_p'] < 0.05:
                ks_str = f"\\textbf{{{ks_str}}}"
        else:
            ks_str = "--"
        
        # AD statistic
        if not np.isnan(stats['ad_stat']):
            ad_str = f"{stats['ad_stat']:.2f}"
            if stats['ad_stat'] > 0.787:
                ad_str = f"\\textbf{{{ad_str}}}"
        else:
            ad_str = "--"
        
        row += f"{ks_str} & {ad_str} \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\begin{tablenotes}")
    latex.append("\\small")
    latex.append("\\item All results use OR-Tools with Guided Local Search (GLS) metaheuristic")
    latex.append("\\item Timeout: 2 seconds per instance for all runs")
    latex.append("\\item GM: Geometric Mean of CPC (Cost vs. Nearest Neighbor baseline)")
    latex.append("\\item GSD: Geometric Standard Deviation")
    latex.append("\\item KS p-val: Kolmogorov-Smirnov test for log-normality")
    latex.append("\\item AD stat: Anderson-Darling test statistic (critical value: 0.787)")
    latex.append("\\item Bold values indicate rejection of log-normality (p < 0.05 or AD > 0.787)")
    latex.append("\\item Bold instance counts indicate preliminary results")
    latex.append("\\end{tablenotes}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def check_process_status():
    """Check status of all benchmark processes"""
    status = {}
    
    # Check each N value
    for n in [10, 20, 50, 100]:
        # Check if process is running
        result = subprocess.run(['pgrep', '-f', f'--n {n}'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            status[n] = {'running': True, 'pids': result.stdout.strip().split('\n')}
        else:
            # Special case for N=100 (two different scripts)
            if n == 100:
                result = subprocess.run(['pgrep', '-f', 'benchmark_ortools_gls_fixed.py'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    status[n] = {'running': True, 'pids': result.stdout.strip().split('\n')}
                else:
                    status[n] = {'running': False}
            else:
                status[n] = {'running': False}
    
    return status

def main():
    print("="*70)
    print("OR-TOOLS GLS COMPREHENSIVE BENCHMARK MONITOR")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Collect all available results
    results = {}
    
    # Check N=10
    result_n10 = find_latest_result('ortools_gls_n10_10000inst_*.json')
    if result_n10:
        with open(result_n10, 'r') as f:
            data = json.load(f)
        results[10] = {
            'instances': data['instances'],
            'stats': compute_full_stats(data['all_cpcs'])
        }
    
    # Check N=20
    result_n20 = find_latest_result('ortools_gls_n20_10000inst_*.json')
    if result_n20:
        with open(result_n20, 'r') as f:
            data = json.load(f)
        results[20] = {
            'instances': data['instances'],
            'stats': compute_full_stats(data['all_cpcs'])
        }
    
    # Check N=50
    result_n50 = find_latest_result('ortools_gls_n50_10000inst_*.json')
    if result_n50:
        with open(result_n50, 'r') as f:
            data = json.load(f)
        results[50] = {
            'instances': data['instances'],
            'stats': compute_full_stats(data['all_cpcs'])
        }
    
    # Check N=100 (both 1000 and 10000 instance runs)
    result_n100_1k = find_latest_result('ortools_gls_n100_1000inst_*.json')
    result_n100_10k = find_latest_result('ortools_gls_n100_10000inst_*.json')
    
    if result_n100_10k:
        with open(result_n100_10k, 'r') as f:
            data = json.load(f)
        results[100] = {
            'instances': data['instances'],
            'stats': compute_full_stats(data['all_cpcs'])
        }
    elif result_n100_1k:
        with open(result_n100_1k, 'r') as f:
            data = json.load(f)
        results[100] = {
            'instances': data['instances'],
            'stats': compute_full_stats(data['all_cpcs'])
        }
    
    # Display results
    if results:
        print("COMPLETED RESULTS:")
        print("-"*40)
        for n in sorted(results.keys()):
            data = results[n]
            print(f"N={n:3d}: {data['instances']:5d} instances | "
                  f"GM={data['stats']['gm']:.4f} | "
                  f"GSD={data['stats']['gsd']:.4f}")
        print()
    
    # Check process status
    print("PROCESS STATUS:")
    print("-"*40)
    status = check_process_status()
    
    for n in [10, 20, 50, 100]:
        if n in status:
            if status[n]['running']:
                print(f"N={n:3d}: ⏳ RUNNING (PIDs: {', '.join(status[n]['pids'])})")
            else:
                if n in results:
                    print(f"N={n:3d}: ✓ COMPLETED")
                else:
                    print(f"N={n:3d}: ⏸ NOT RUNNING")
    
    # Generate table if we have results
    if results:
        print("\n" + "="*70)
        print("GENERATING LATEX TABLE...")
        print("="*70)
        
        latex_table = generate_ortools_only_table(results)
        
        # Save to file
        filename = 'ortools_gls_comprehensive_table.tex'
        with open(filename, 'w') as f:
            f.write(latex_table)
        
        print(f"Table saved to: {filename}")
        print("\nTable preview:")
        print("-"*40)
        # Show key lines
        for line in latex_table.split('\n'):
            if 'midrule' in line or 'bottomrule' in line:
                continue
            if line.strip() and not line.startswith('\\item'):
                print(line)

if __name__ == "__main__":
    main()
