#!/usr/bin/env python3
"""
Monitor OR-Tools benchmarks and generate table when 1000-instance run completes
"""

import time
import os
import json
import glob
import numpy as np
from scipy import stats

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
        'ks_p': ks_p, 'dag_p': dag_p, 'jb_p': jb_p, 'ad_stat': ad_stat
    }

def check_and_report():
    """Check benchmark status and generate table if 1000-instance run is done"""
    
    # Check 1000-instance status
    result_1000 = find_latest_result('ortools_gls_n100_1000inst_*.json')
    
    if result_1000:
        print("\n" + "="*70)
        print("✓ 1000-INSTANCE BENCHMARK COMPLETED!")
        print("="*70)
        
        with open(result_1000, 'r') as f:
            data = json.load(f)
        
        stats = compute_stats(data['all_cpcs'])
        
        print(f"\nResults from {data['instances']} instances:")
        print(f"  GM:       {stats['gm']:.6f}")
        print(f"  GSD:      {stats['gsd']:.6f}")
        print(f"  95% CI:   [{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}]")
        print(f"  Range:    [{stats['gm']*(stats['gsd']**(-1.96)):.4f}, {stats['gm']*(stats['gsd']**(1.96)):.4f}]")
        
        print(f"\nNormality tests (p-values):")
        print(f"  KS:       {stats['ks_p']:.3f}")
        print(f"  D'Agost:  {stats['dag_p']:.3f}")
        print(f"  JB:       {stats['jb_p']:.3f}")
        print(f"  AD stat:  {stats['ad_stat']:.3f} {'(reject)' if stats['ad_stat'] > 0.787 else '(accept)'}")
        
        # LaTeX table row
        print("\n" + "="*70)
        print("LATEX TABLE ROW (1000 instances):")
        print("="*70)
        print(f"OR-Tools GLS & 100 & 50 & {stats['gm']:.4f} & {stats['gsd']:.4f} & "
              f"[{stats['gm']*(stats['gsd']**(-1.96)):.4f}, {stats['gm']*(stats['gsd']**(1.96)):.4f}] & "
              f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] \\\\")
        
        return True
    
    # Check if 1000-instance process is still running
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'instances 1000'], capture_output=True, text=True)
    if result.returncode == 0:
        print("⏳ 1000-instance benchmark still running...")
        return False
    else:
        print("⚠ 1000-instance benchmark not found (may have completed, checking for results...)")
        time.sleep(2)
        return False

# Check status of 10000-instance run
def check_10k_status():
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'instances 10000'], capture_output=True, text=True)
    if result.returncode == 0:
        print("⏳ 10,000-instance benchmark is running in parallel (PID: {})".format(result.stdout.strip()))
    
    # Try to get progress from log
    try:
        with open('ortools_10000inst.log', 'r') as f:
            lines = f.readlines()
            for line in reversed(lines[-20:]):
                if 'Progress:' in line:
                    print(f"   10k progress: {line.strip()}")
                    break
    except:
        pass

if __name__ == "__main__":
    print("Monitoring OR-Tools benchmarks...")
    print("="*70)
    
    if check_and_report():
        print("\n✓ Table generated with 1000-instance results!")
    
    print("")
    check_10k_status()
