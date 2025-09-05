#!/usr/bin/env python3
"""
Fixed continuous monitor for all OR-Tools GLS benchmarks
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
        dag_p = stats.normaltest(log_cpcs).pvalue
        jb_p = stats.jarque_bera(log_cpcs).pvalue
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

def check_process_status():
    """Check which benchmarks are still running"""
    running = {}
    
    # Check N=10, 20, 50 individually
    for n in [10, 20, 50]:
        result = subprocess.run(['pgrep', '-f', f'benchmark_ortools_multi_n_fixed.py.*--n {n}'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = [p for p in result.stdout.strip().split('\n') if p]
            if pids:
                running[n] = pids
    
    # Check N=100 runs
    result_1k = subprocess.run(['pgrep', '-f', 'benchmark_ortools_gls_fixed.py.*--instances 1000'], 
                               capture_output=True, text=True)
    if result_1k.returncode == 0:
        pids = [p for p in result_1k.stdout.strip().split('\n') if p]
        if pids:
            running['100_1k'] = pids
    
    result_10k = subprocess.run(['pgrep', '-f', 'benchmark_ortools_gls_fixed.py.*--instances 10000'], 
                                capture_output=True, text=True)
    if result_10k.returncode == 0:
        pids = [p for p in result_10k.stdout.strip().split('\n') if p]
        if pids:
            running['100_10k'] = pids
    
    return running

def get_progress_info():
    """Get current progress from log files"""
    progress = {}
    
    for n in [10, 20, 50]:
        log_file = f'ortools_n{n}_10000inst.log'
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        if 'Progress:' in line:
                            # Parse: Progress: 600/10000 (6.0%) - Current GM: 0.813275 - ETA: 313.7 min
                            parts = line.strip().split(' - ')
                            if len(parts) >= 3:
                                prog_part = parts[0].split(': ')[1]  # "600/10000 (6.0%)"
                                gm_part = parts[1].split(': ')[1]    # "0.813275"
                                eta_part = parts[2].split(': ')[1]   # "313.7 min"
                                
                                progress[n] = {
                                    'progress': prog_part,
                                    'current_gm': float(gm_part),
                                    'eta': eta_part
                                }
                            break
            except:
                pass
    
    return progress

def collect_available_results():
    """Collect all available benchmark results"""
    results = {}
    
    # Check N=10, 20, 50
    for n in [10, 20, 50]:
        result_file = find_latest_result(f'ortools_gls_n{n}_10000inst_*.json')
        if result_file:
            with open(result_file, 'r') as f:
                data = json.load(f)
            results[n] = {
                'instances': data['instances'],
                'timeout': data.get('timeout_sec', 2),
                'stats': compute_full_stats(data['all_cpcs']),
                'file': result_file
            }
    
    # Check N=100 (prefer 10k instances, fallback to 1k)
    result_n100_10k = find_latest_result('ortools_gls_n100_10000inst_*.json')
    result_n100_1k = find_latest_result('ortools_gls_n100_1000inst_*.json')
    
    if result_n100_10k:
        with open(result_n100_10k, 'r') as f:
            data = json.load(f)
        results[100] = {
            'instances': data['instances'],
            'timeout': data.get('timeout_sec', 2),
            'stats': compute_full_stats(data['all_cpcs']),
            'file': result_n100_10k
        }
    elif result_n100_1k:
        with open(result_n100_1k, 'r') as f:
            data = json.load(f)
        results[100] = {
            'instances': data['instances'],
            'timeout': data.get('timeout_sec', 1),
            'stats': compute_full_stats(data['all_cpcs']),
            'file': result_n100_1k
        }
    
    return results

def generate_ortools_table(results):
    """Generate LaTeX table with OR-Tools GLS results"""
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{OR-Tools GLS Performance on CVRP instances}")
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
    latex.append("\\item GM: Geometric Mean of CPC (Cost vs. Nearest Neighbor baseline)")
    latex.append("\\item GSD: Geometric Standard Deviation")
    latex.append("\\item 95\\% Range: [GM×GSD$^{-1.96}$, GM×GSD$^{1.96}$]")
    latex.append("\\item 95\\% CI(GM): 95\\% confidence interval for geometric mean")
    latex.append("\\item KS p-val: Kolmogorov-Smirnov test for log-normality")
    latex.append("\\item AD stat: Anderson-Darling test statistic (critical value: 0.787)")
    latex.append("\\item Bold values indicate rejection of log-normality (p < 0.05 or AD > 0.787)")
    latex.append("\\item Bold instance counts indicate preliminary results (<10,000 instances)")
    latex.append("\\end{tablenotes}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def display_status(results, running, progress):
    """Display current status in terminal"""
    os.system('clear')
    
    print("="*80)
    print("OR-TOOLS GLS BENCHMARKS - CONTINUOUS MONITOR (FIXED)")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Show running processes with progress
    print("RUNNING PROCESSES:")
    print("-"*70)
    if not running:
        print("No processes currently running")
    else:
        for key, pids in running.items():
            if key == '100_1k':
                print(f"N=100 (1k inst):   PIDs: {', '.join(pids)}")
            elif key == '100_10k':
                print(f"N=100 (10k inst):  PIDs: {', '.join(pids)}")
            else:
                line = f"N={key:3d} (10k inst):  PIDs: {', '.join(pids)}"
                if key in progress:
                    p = progress[key]
                    line += f" | {p['progress']} | GM: {p['current_gm']:.4f} | ETA: {p['eta']}"
                print(line)
    print()
    
    # Show completed results
    print("COMPLETED RESULTS:")
    print("-"*70)
    if not results:
        print("No results available yet")
    else:
        for n in sorted(results.keys()):
            data = results[n]
            status = "PRELIMINARY" if data['instances'] < 10000 else "FINAL"
            print(f"N={n:3d}: {data['instances']:5d} instances | "
                  f"GM={data['stats']['gm']:.4f} | "
                  f"GSD={data['stats']['gsd']:.4f} | "
                  f"[{status}]")
    print()
    
    return len(results)

def monitor_loop():
    """Main monitoring loop"""
    last_result_count = 0
    table_generations = 0
    
    print("Starting OR-Tools GLS continuous monitor (FIXED)...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            # Collect current state
            results = collect_available_results()
            running = check_process_status()
            progress = get_progress_info()
            
            # Display status
            current_result_count = display_status(results, running, progress)
            
            # Generate table if new results are available
            if current_result_count > last_result_count and results:
                print("="*80)
                print("NEW RESULTS DETECTED - GENERATING LATEX TABLE")
                print("="*80)
                
                latex_table = generate_ortools_table(results)
                
                # Save table with timestamp
                table_generations += 1
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'ortools_gls_table_v{table_generations}_{timestamp}.tex'
                
                with open(filename, 'w') as f:
                    f.write(latex_table)
                
                print(f"✓ LaTeX table saved: {filename}")
                print(f"✓ Results included: N={sorted(results.keys())}")
                
                # Also save as latest
                with open('ortools_gls_latest.tex', 'w') as f:
                    f.write(latex_table)
                
                print("✓ Latest table saved: ortools_gls_latest.tex")
                print()
                
                last_result_count = current_result_count
            
            # Check if all benchmarks are complete
            if len(results) == 4 and not running:
                # All N values have final results
                all_final = all(data['instances'] >= 10000 for data in results.values())
                if all_final:
                    print("="*80)
                    print("🎉 ALL BENCHMARKS COMPLETED WITH FINAL RESULTS!")
                    print("="*80)
                    print("Final table generated with all N=10,20,50,100 on 10,000 instances")
                    break
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print(f"Latest table available in: ortools_gls_latest.tex")

if __name__ == "__main__":
    monitor_loop()
