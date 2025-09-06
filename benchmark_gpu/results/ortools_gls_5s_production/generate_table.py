#!/usr/bin/env python3
import json
import glob
import numpy as np
from scipy import stats
import math

def compute_stats(cpcs):
    cpcs = np.array(cpcs)
    log_cpcs = np.log(cpcs)
    n = len(cpcs)
    
    mu_log = log_cpcs.mean()
    sigma_log = log_cpcs.std(ddof=1)
    gm = float(np.exp(mu_log))
    gsd = float(np.exp(sigma_log))
    
    se_log = sigma_log / math.sqrt(n)
    ci_lower = float(np.exp(mu_log - 1.96 * se_log))
    ci_upper = float(np.exp(mu_log + 1.96 * se_log))
    
    # Normality tests
    ks_p = float(stats.kstest(log_cpcs, lambda x: stats.norm.cdf(x, loc=mu_log, scale=sigma_log)).pvalue)
    dag_p = float(stats.normaltest(log_cpcs).pvalue)
    jb_p = float(stats.jarque_bera(log_cpcs).pvalue)
    ad_stat = float(stats.anderson(log_cpcs, dist='norm').statistic)
    
    return {
        'n': n, 'gm': gm, 'gsd': gsd,
        'range_lower': gm * (gsd ** (-1.96)),
        'range_upper': gm * (gsd ** (1.96)),
        'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'ks_p': ks_p, 'dag_p': dag_p, 'jb_p': jb_p, 'ad_stat': ad_stat
    }

def format_p(p):
    return f"{p:.2f}" if p >= 0.01 else "$<$0.01"

# Load data and compute stats
results = {}
for file in glob.glob('OR_GLS_*.json'):
    with open(file, 'r') as f:
        data = json.load(f)
        n = data['n_customers']
        cap = data['capacity']
        cpcs = data['all_cpcs']
        results[n] = {'cap': cap, 'stats': compute_stats(cpcs)}

# Generate LaTeX table
lines = []
lines.append("\\begin{table*}[htbp]")
lines.append("\\centering")
lines.append("\\caption{OR-Tools GLS Performance with Log-normal Statistics (5s timeout, 10,000 instances)}")
lines.append("\\label{tab:ortools-gls-5s}")
lines.append("\\begin{tabular}{@{}l c c S[table-format=1.4] S[table-format=1.4] c c c c c c@{}}")
lines.append("\\toprule")
lines.append("\\textbf{Method} & \\textbf{N} & \\textbf{Cap.} & {\\textbf{GM}} & {\\textbf{GSD}} & ")
lines.append("\\textbf{95\\% Range} & \\textbf{95\\% CI} & \\textbf{KS} & \\textbf{D'Agost.} & \\textbf{JB} & \\textbf{AD} \\\\")
lines.append("\\midrule")

for n in sorted(results.keys()):
    data = results[n]
    s = data['stats']
    cap = data['cap']
    ks = format_p(s['ks_p'])
    dag = format_p(s['dag_p'])
    jb = format_p(s['jb_p'])
    ad = f"{s['ad_stat']:.3f}" + ("*" if s['ad_stat'] > 0.787 else "")
    
    lines.append(f"OR-Tools GLS & {n} & {cap} & {s['gm']:.4f} & {s['gsd']:.4f} & ")
    lines.append(f"[{s['range_lower']:.4f}, {s['range_upper']:.4f}] & ")
    lines.append(f"[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}] & ")
    lines.append(f"${ks}$ & ${dag}$ & ${jb}$ & ${ad}$ \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\begin{tablenotes}")
lines.append("\\small")
lines.append("\\item GM: Geometric Mean, GSD: Geometric Standard Deviation")
lines.append("\\item 95\\% Range: Individual value range GM $\\times$ [GSD$^{-1.96}$, GSD$^{+1.96}$]")
lines.append("\\item 95\\% CI: Confidence interval for the geometric mean")
lines.append("\\item KS: Kolmogorov-Smirnov, D'Agost.: D'Agostino, JB: Jarque-Bera (p-values for log(CPC) normality)")
lines.append("\\item AD: Anderson-Darling test statistic (critical value at 5\\% = 0.787; * indicates rejection)")
lines.append("\\item Time limit: 5s per instance")
lines.append("\\end{tablenotes}")
lines.append("\\end{table*}")

# Save table with proper newlines
with open('ortools_gls_5s_table.tex', 'w') as f:
    f.write('\n'.join(lines))

print("Generated ortools_gls_5s_table.tex")
