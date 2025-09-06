#!/usr/bin/env python3
"""
Continuously monitor OR-Tools benchmarks and generate table when ready
"""

import time
import os
import json
import glob
import numpy as np
from scipy import stats
import subprocess
from datetime import datetime

def find_latest_result(pattern):
    """Find the latest result file matching pattern"""
    files = glob.glob(pattern)
    if files:
        return sorted(files)[-1]
    return None

def compute_stats(cpcs):
    """Compute all statistics from CPC values"""
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
        'gm': gm, 'gsd': gsd,
        'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'ks_p': ks_p, 'dag_p': dag_p, 'jb_p': jb_p, 'ad_stat': ad_stat,
        'n': n
    }

def generate_full_latex_table(new_stats_100=None, instances_100=1000):
    """Generate the complete LaTeX table with all results"""
    
    # Existing results
    existing_data = {
        'Exact (Gurobi)': [
            {'n': 10, 'instances': 10000, 'gm': 1.0003, 'gsd': 1.1052,
             'range': [0.8192, 1.2215], 'ci': [0.9981, 1.0025],
             'ks_p': 0.149, 'dag_p': 0.021, 'jb_p': 0.045, 'ad_stat': 0.717}
        ],
        '2-opt': [
            {'n': 10, 'instances': 10000, 'gm': 1.0017, 'gsd': 1.0928,
             'range': [0.8397, 1.1948], 'ci': [0.9996, 1.0038],
             'ks_p': 0.085, 'dag_p': 0.040, 'jb_p': 0.086, 'ad_stat': 0.676},
            {'n': 20, 'instances': 10000, 'gm': 1.0336, 'gsd': 1.0925,
             'range': [0.8670, 1.2318], 'ci': [1.0314, 1.0357],
             'ks_p': 0.014, 'dag_p': 0.010, 'jb_p': 0.019, 'ad_stat': 1.245},
            {'n': 50, 'instances': 10000, 'gm': 1.0654, 'gsd': 1.0765,
             'range': [0.9194, 1.2347], 'ci': [1.0635, 1.0673],
             'ks_p': 0.000, 'dag_p': 0.000, 'jb_p': 0.000, 'ad_stat': 8.071},
            {'n': 100, 'instances': 10000, 'gm': 1.0866, 'gsd': 1.0744,
             'range': [0.9408, 1.2548], 'ci': [1.0847, 1.0885],
             'ks_p': 0.000, 'dag_p': 0.000, 'jb_p': 0.000, 'ad_stat': 11.436}
        ],
        'Random Insertion': [
            {'n': 10, 'instances': 10000, 'gm': 1.0749, 'gsd': 1.1335,
             'range': [0.8366, 1.3816], 'ci': [1.0724, 1.0774],
             'ks_p': 0.000, 'dag_p': 0.000, 'jb_p': 0.000, 'ad_stat': 2.648},
            {'n': 20, 'instances': 10000, 'gm': 1.1099, 'gsd': 1.1153,
             'range': [0.8926, 1.3802], 'ci': [1.1076, 1.1122],
             'ks_p': 0.023, 'dag_p': 0.001, 'jb_p': 0.002, 'ad_stat': 1.077},
            {'n': 50, 'instances': 10000, 'gm': 1.1296, 'gsd': 1.0922,
             'range': [0.9479, 1.3460], 'ci': [1.1275, 1.1318],
             'ks_p': 0.007, 'dag_p': 0.023, 'jb_p': 0.046, 'ad_stat': 1.467},
            {'n': 100, 'instances': 10000, 'gm': 1.1414, 'gsd': 1.0820,
             'range': [0.9752, 1.3359], 'ci': [1.1394, 1.1435],
             'ks_p': 0.147, 'dag_p': 0.162, 'jb_p': 0.201, 'ad_stat': 0.716}
        ],
        'Nearest Neighbor': [
            {'n': 10, 'instances': 10000, 'gm': 1.1189, 'gsd': 1.1334,
             'range': [0.8706, 1.4375], 'ci': [1.1164, 1.1214],
             'ks_p': 0.012, 'dag_p': 0.029, 'jb_p': 0.058, 'ad_stat': 1.318},
            {'n': 20, 'instances': 10000, 'gm': 1.1634, 'gsd': 1.1048,
             'range': [0.9535, 1.4190], 'ci': [1.1611, 1.1656],
             'ks_p': 0.195, 'dag_p': 0.043, 'jb_p': 0.082, 'ad_stat': 0.640},
            {'n': 50, 'instances': 10000, 'gm': 1.2043, 'gsd': 1.0937,
             'range': [1.0082, 1.4383], 'ci': [1.2021, 1.2065],
             'ks_p': 0.178, 'dag_p': 0.072, 'jb_p': 0.129, 'ad_stat': 0.666},
            {'n': 100, 'instances': 10000, 'gm': 1.2234, 'gsd': 1.0808,
             'range': [1.0485, 1.4277], 'ci': [1.2214, 1.2255],
             'ks_p': 0.006, 'dag_p': 0.000, 'jb_p': 0.000, 'ad_stat': 1.587}
        ]
    }
    
    # Add OR-Tools GLS if available
    if new_stats_100:
        existing_data['OR-Tools GLS'] = [
            {'n': 100, 'instances': instances_100, 
             'gm': new_stats_100['gm'], 'gsd': new_stats_100['gsd'],
             'range': [new_stats_100['gm']*(new_stats_100['gsd']**(-1.96)), 
                      new_stats_100['gm']*(new_stats_100['gsd']**(1.96))],
             'ci': [new_stats_100['ci_lower'], new_stats_100['ci_upper']],
             'ks_p': new_stats_100['ks_p'], 'dag_p': new_stats_100['dag_p'], 
             'jb_p': new_stats_100['jb_p'], 'ad_stat': new_stats_100['ad_stat']}
        ]
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Performance comparison of heuristics on 10,000 random CVRP instances}")
    latex.append("\\begin{tabular}{llrrrrrrrr}")
    latex.append("\\toprule")
    latex.append("Method & N & Inst. & GM & GSD & 95\\% Range & 95\\% CI(GM) & KS & D'Ag. & AD \\\\")
    latex.append("\\midrule")
    
    for method in ['Exact (Gurobi)', '2-opt', 'Random Insertion', 'Nearest Neighbor', 'OR-Tools GLS']:
        if method in existing_data:
            for data in existing_data[method]:
                row = f"{method} & {data['n']} & "
                if data['instances'] < 10000:
                    row += f"\\textbf{{{data['instances']}}}"
                else:
                    row += f"{data['instances']}"
                row += f" & {data['gm']:.4f} & {data['gsd']:.4f} & "
                row += f"[{data['range'][0]:.4f}, {data['range'][1]:.4f}] & "
                row += f"[{data['ci'][0]:.4f}, {data['ci'][1]:.4f}] & "
                
                # P-values with significance indicators
                ks_str = f"{data['ks_p']:.3f}" if data['ks_p'] >= 0.001 else "<.001"
                if data['ks_p'] < 0.05:
                    ks_str = f"\\textbf{{{ks_str}}}"
                
                dag_str = f"{data['dag_p']:.3f}" if data['dag_p'] >= 0.001 else "<.001"
                if data['dag_p'] < 0.05:
                    dag_str = f"\\textbf{{{dag_str}}}"
                
                # AD statistic (not p-value)
                ad_str = f"{data['ad_stat']:.2f}"
                if data['ad_stat'] > 0.787:  # Critical value at 5% level
                    ad_str = f"\\textbf{{{ad_str}}}"
                
                row += f"{ks_str} & {dag_str} & {ad_str} \\\\"
                latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\begin{tablenotes}")
    latex.append("\\small")
    latex.append("\\item GM: Geometric Mean of CPC ratios")
    latex.append("\\item GSD: Geometric Standard Deviation")
    latex.append("\\item 95\\% Range: [GM×GSD$^{-1.96}$, GM×GSD$^{1.96}$]")
    latex.append("\\item 95\\% CI(GM): 95\\% confidence interval for the geometric mean")
    latex.append("\\item KS: Kolmogorov-Smirnov test p-value for log-normality")
    latex.append("\\item D'Ag.: D'Agostino test p-value for log-normality")
    latex.append("\\item AD: Anderson-Darling test statistic (critical value: 0.787 at 5\\% level)")
    latex.append("\\item Bold p-values indicate rejection of log-normality at 5\\% level (p < 0.05)")
    latex.append("\\item Bold AD values indicate rejection of normality (statistic > 0.787)")
    latex.append("\\item Bold instance counts indicate preliminary results (<10,000 instances)")
    latex.append("\\end{tablenotes}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def monitor_loop():
    """Main monitoring loop"""
    result_1000_reported = False
    result_10000_reported = False
    
    while True:
        os.system('clear')
        print(f"OR-Tools Benchmark Monitor - {datetime.now().strftime('%H:%M:%S')}")
        print("="*70)
        
        # Check 1000-instance results
        result_1000 = find_latest_result('ortools_gls_n100_1000inst_*.json')
        if result_1000 and not result_1000_reported:
            with open(result_1000, 'r') as f:
                data_1000 = json.load(f)
            stats_1000 = compute_stats(data_1000['all_cpcs'])
            
            print("\n✓ 1000-INSTANCE RESULTS AVAILABLE")
            print("-"*40)
            print(f"  GM:       {stats_1000['gm']:.6f}")
            print(f"  GSD:      {stats_1000['gsd']:.6f}")
            print(f"  95% CI:   [{stats_1000['ci_lower']:.6f}, {stats_1000['ci_upper']:.6f}]")
            print(f"  KS p-val: {stats_1000['ks_p']:.3f}")
            
            # Save LaTeX table
            latex_table = generate_full_latex_table(stats_1000, 1000)
            with open('table_with_ortools_1000.tex', 'w') as f:
                f.write(latex_table)
            print("\n✓ LaTeX table saved to: table_with_ortools_1000.tex")
            result_1000_reported = True
        
        # Check 10000-instance results
        result_10000 = find_latest_result('ortools_gls_n100_10000inst_*.json')
        if result_10000 and not result_10000_reported:
            with open(result_10000, 'r') as f:
                data_10000 = json.load(f)
            stats_10000 = compute_stats(data_10000['all_cpcs'])
            
            print("\n✓ 10,000-INSTANCE RESULTS AVAILABLE")
            print("-"*40)
            print(f"  GM:       {stats_10000['gm']:.6f}")
            print(f"  GSD:      {stats_10000['gsd']:.6f}")
            print(f"  95% CI:   [{stats_10000['ci_lower']:.6f}, {stats_10000['ci_upper']:.6f}]")
            print(f"  KS p-val: {stats_10000['ks_p']:.3f}")
            
            # Save final LaTeX table
            latex_table = generate_full_latex_table(stats_10000, 10000)
            with open('table_with_ortools_final.tex', 'w') as f:
                f.write(latex_table)
            print("\n✓ FINAL LaTeX table saved to: table_with_ortools_final.tex")
            result_10000_reported = True
        
        # Check running processes
        print("\n" + "="*70)
        print("PROCESS STATUS:")
        print("-"*40)
        
        proc_1000 = subprocess.run(['pgrep', '-f', 'instances 1000'], capture_output=True, text=True)
        if proc_1000.returncode == 0:
            print(f"⏳ 1000-instance: RUNNING (PID: {proc_1000.stdout.strip()})")
            # Check log
            try:
                result = subprocess.run(['tail', '-n', '5', 'ortools_1000inst.log'], 
                                      capture_output=True, text=True)
                for line in result.stdout.strip().split('\n'):
                    if 'Progress:' in line or 'Instance' in line:
                        print(f"     {line.strip()}")
            except:
                pass
        elif not result_1000_reported:
            print("⏸ 1000-instance: NOT RUNNING (checking for results...)")
        else:
            print("✓ 1000-instance: COMPLETED")
        
        proc_10000 = subprocess.run(['pgrep', '-f', 'instances 10000'], capture_output=True, text=True)
        if proc_10000.returncode == 0:
            print(f"⏳ 10000-instance: RUNNING (PID: {proc_10000.stdout.strip()})")
            # Check log
            try:
                result = subprocess.run(['tail', '-n', '5', 'ortools_10000inst.log'], 
                                      capture_output=True, text=True)
                for line in result.stdout.strip().split('\n'):
                    if 'Progress:' in line or 'Instance' in line:
                        print(f"     {line.strip()}")
            except:
                pass
        elif not result_10000_reported:
            print("⏸ 10000-instance: NOT RUNNING (checking for results...)")
        else:
            print("✓ 10000-instance: COMPLETED")
        
        if result_1000_reported and result_10000_reported:
            print("\n" + "="*70)
            print("✓✓ ALL BENCHMARKS COMPLETED!")
            print("="*70)
            break
        
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
