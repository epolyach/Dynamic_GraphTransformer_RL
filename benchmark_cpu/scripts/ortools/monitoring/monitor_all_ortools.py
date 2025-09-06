#!/usr/bin/env python3
"""
Monitor all OR-Tools benchmarks and generate comprehensive tables
Enhanced to support multiple timeout configurations
"""

import os
import time
import json
import glob
import numpy as np
from scipy import stats
from datetime import datetime
import subprocess
import sys

def find_json_files(base_dir, pattern="*.json"):
    """Find all JSON result files in a directory"""
    if not os.path.exists(base_dir):
        return []
    return glob.glob(os.path.join(base_dir, pattern))

def parse_json_results(json_files):
    """Parse multiple JSON result files and aggregate CPC values"""
    all_cpcs = []
    metadata = None
    
    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if 'all_cpcs' in data:
                    all_cpcs.extend(data['all_cpcs'])
                    if metadata is None:
                        metadata = {
                            'n_customers': data.get('n_customers'),
                            'capacity': data.get('capacity'),
                            'time_limit': data.get('time_limit')
                        }
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return all_cpcs, metadata

def compute_full_stats(cpcs):
    """Compute comprehensive statistics from CPC values"""
    if len(cpcs) < 2:
        return None
        
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

def generate_latex_table(results, timeout_str, test_mode=False):
    """Generate LaTeX table for OR-Tools GLS results"""
    
    latex = []
    latex.append("\\begin{table*}[htbp]")
    latex.append("\\centering")
    
    if test_mode:
        latex.append(f"\\caption{{OR-Tools GLS Test Results ({timeout_str} timeout, 20 instances per configuration)}}")
    else:
        latex.append(f"\\caption{{OR-Tools GLS Performance with Log-normal Statistics ({timeout_str} timeout, 10,000 instances)}}")
    
    latex.append("\\label{tab:ortools-gls-" + timeout_str.replace(' ', '-') + "}")
    latex.append("\\begin{tabular}{@{}l c c S[table-format=1.4] S[table-format=1.4] c c c c c c@{}}")
    latex.append("\\toprule")
    latex.append("\\textbf{Method} & \\textbf{N} & \\textbf{Cap.} & {\\textbf{GM}} & {\\textbf{GSD}} & ")
    latex.append("\\textbf{95\\% Range} & \\textbf{95\\% CI} & \\textbf{KS} & \\textbf{D'Agost.} & \\textbf{JB} & \\textbf{AD} \\\\")
    latex.append("\\midrule")
    
    # Sort by N
    for n in sorted(results.keys()):
        data = results[n]
        stats = data['stats']
        
        if stats is None:
            continue
            
        row = f"OR-Tools GLS & {n} & {data.get('capacity', 'N/A')} & "
        row += f"{stats['gm']:.4f} & {stats['gsd']:.4f} & "
        row += f"[{stats['range_lower']:.4f}, {stats['range_upper']:.4f}] & "
        row += f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] & "
        
        # P-value formatting for normality tests
        if not np.isnan(stats['ks_p']):
            ks_str = f"{stats['ks_p']:.2f}" if stats['ks_p'] >= 0.01 else "$<$0.01"
        else:
            ks_str = "--"
            
        if not np.isnan(stats['dag_p']):
            dag_str = f"{stats['dag_p']:.2f}" if stats['dag_p'] >= 0.01 else "$<$0.01"
        else:
            dag_str = "--"
            
        if not np.isnan(stats['jb_p']):
            jb_str = f"{stats['jb_p']:.2f}" if stats['jb_p'] >= 0.01 else "$<$0.01"
        else:
            jb_str = "--"
        
        # AD statistic with significance marker
        if not np.isnan(stats['ad_stat']):
            ad_str = f"{stats['ad_stat']:.3f}"
            if stats['ad_stat'] > 0.787:
                ad_str += "*"
        else:
            ad_str = "--"
        
        row += f"${ks_str}$ & ${dag_str}$ & ${jb_str}$ & ${ad_str}$ \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\begin{tablenotes}")
    latex.append("\\small")
    latex.append("\\item GM: Geometric Mean, GSD: Geometric Standard Deviation")
    latex.append("\\item 95\\% Range: Individual value range GM $\\times$ [GSD$^{-1.96}$, GSD$^{+1.96}$]")
    latex.append("\\item 95\\% CI: Confidence interval for the geometric mean")
    latex.append("\\item KS: Kolmogorov-Smirnov, D'Agost.: D'Agostino, JB: Jarque-Bera (p-values for log(CPC) normality)")
    latex.append("\\item AD: Anderson-Darling test statistic (critical value at 5\\% = 0.787; * indicates rejection)")
    latex.append(f"\\item Time limit: {timeout_str} per instance")
    if test_mode:
        latex.append("\\item Test run with 20 instances per configuration")
    latex.append("\\end{tablenotes}")
    latex.append("\\end{table*}")
    
    return "\n".join(latex)

def monitor_directory(base_dir, timeout_label):
    """Monitor a specific result directory"""
    print(f"\nMonitoring {base_dir} ({timeout_label})...")
    
    if not os.path.exists(base_dir):
        print(f"  Directory not found: {base_dir}")
        return {}
    
    # Configuration mapping
    n_to_capacity = {
        10: 20,
        20: 30,
        50: 40,
        100: 50
    }
    
    results = {}
    
    for n in [10, 20, 50, 100]:
        # Find JSON files for this N value
        json_files = find_json_files(base_dir, f"*n{n}_*.json")
        
        if json_files:
            cpcs, metadata = parse_json_results(json_files)
            if cpcs:
                stats = compute_full_stats(cpcs)
                results[n] = {
                    'instances': len(cpcs),
                    'capacity': n_to_capacity[n],
                    'stats': stats,
                    'files': len(json_files)
                }
                print(f"  N={n:3d}: {len(cpcs):5d} instances from {len(json_files)} files")
            else:
                print(f"  N={n:3d}: No valid CPC data found")
        else:
            print(f"  N={n:3d}: No results found")
    
    return results

def main():
    print("=" * 70)
    print("OR-Tools GLS Benchmark Monitor")
    print(f"Time: {datetime.now()}")
    print("=" * 70)
    
    # Define directories to monitor
    directories = [
        ('results/ortools_gls_2s_test', '2s', True),
        ('results/ortools_gls_5s_test', '5s', True),
        ('results/ortools_gls_2s_production', '2s', False),
        ('results/ortools_gls_5s_production', '5s', False),
        # Legacy directories
        ('scripts', '2s', False),  # Check for legacy files in scripts/
    ]
    
    all_tables = []
    
    for dir_path, timeout_label, is_test in directories:
        results = monitor_directory(dir_path, timeout_label)
        
        if results:
            # Generate LaTeX table
            table = generate_latex_table(results, timeout_label, is_test)
            all_tables.append(table)
            
            # Save table to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            table_file = f'results/ortools_gls_table_{timeout_label}_{timestamp}.tex'
            os.makedirs('results', exist_ok=True)
            with open(table_file, 'w') as f:
                f.write(table)
            print(f"\nTable saved to: {table_file}")
    
    # Save all tables together
    if all_tables:
        combined_file = f'results/ortools_gls_all_tables_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tex'
        with open(combined_file, 'w') as f:
            f.write("% OR-Tools GLS Benchmark Results\n")
            f.write(f"% Generated: {datetime.now()}\n\n")
            f.write("\n\n".join(all_tables))
        print(f"\nAll tables saved to: {combined_file}")
    
    # Check for running processes
    print("\n" + "=" * 70)
    print("Process Status Check")
    print("=" * 70)
    
    # Look for benchmark processes
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    
    benchmark_processes = []
    for line in lines:
        if 'benchmark_ortools' in line and 'python' in line:
            benchmark_processes.append(line)
    
    if benchmark_processes:
        print(f"Found {len(benchmark_processes)} running benchmark process(es):")
        for proc in benchmark_processes:
            # Extract key info
            parts = proc.split()
            if len(parts) > 10:
                cmd = ' '.join(parts[10:])[:100]
                print(f"  PID {parts[1]}: {cmd}...")
    else:
        print("No benchmark processes currently running")
    
    print("\n" + "=" * 70)
    print("Monitor complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
