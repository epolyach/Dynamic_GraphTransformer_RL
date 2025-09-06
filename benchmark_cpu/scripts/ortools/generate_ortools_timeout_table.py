#!/usr/bin/env python3
"""
Generate a LaTeX table for OR-Tools GLS results including a Timeout column.
Scans results/ortools_gls_*s_test by default.

Usage:
  python3 scripts/generate_ortools_timeout_table.py \
    --mode test \
    --out results/ortools_gls_timeout_test_table.tex

Options:
  --mode: 'test' or 'production' (default: test)
  --dirs: explicit directories to scan (repeatable)
  --out: output .tex path (default under results/)

The table columns:
  Method, N, Cap., Timeout, GM, GSD, 95% Range, 95% CI, KS, D'Agost., JB, AD
"""

import argparse
import glob
import json
import math
import os
from datetime import datetime

import numpy as np
from scipy import stats


def compute_full_stats(cpcs):
    cpcs = np.array(cpcs, dtype=float)
    if len(cpcs) == 0:
        return None
    log_cpcs = np.log(cpcs)
    n = len(cpcs)

    mu_log = log_cpcs.mean()
    sigma_log = log_cpcs.std(ddof=1) if n > 1 else 0.0
    gm = float(np.exp(mu_log))
    gsd = float(np.exp(sigma_log))

    se_log = sigma_log / math.sqrt(n) if n > 0 else 0.0
    ci_lower = float(np.exp(mu_log - 1.96 * se_log))
    ci_upper = float(np.exp(mu_log + 1.96 * se_log))

    if n >= 20 and sigma_log > 0:
        ks_p = float(stats.kstest(log_cpcs, lambda x: stats.norm.cdf(x, loc=mu_log, scale=sigma_log)).pvalue)
        dag_p = float(stats.normaltest(log_cpcs).pvalue)
        jb_p = float(stats.jarque_bera(log_cpcs).pvalue)
        ad_stat = float(stats.anderson(log_cpcs, dist='norm').statistic)
    else:
        ks_p = dag_p = jb_p = ad_stat = float('nan')

    return {
        'n': n,
        'gm': gm,
        'gsd': gsd,
        'range_lower': gm * (gsd ** (-1.96)),
        'range_upper': gm * (gsd ** (1.96)),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ks_p': ks_p,
        'dag_p': dag_p,
        'jb_p': jb_p,
        'ad_stat': ad_stat,
    }


def gather_results(directories):
    """Return mapping {(n, timeout): {'cap': cap, 'stats': stats}}"""
    results = {}
    for d in directories:
        # detect timeout from dir name ..._gls_{timeout}s_...
        base = os.path.basename(d.rstrip('/'))
        timeout = None
        parts = base.split('_')
        for p in parts:
            if p.endswith('s') and p[:-1].isdigit():
                timeout = int(p[:-1])
        # fallback: search json
        json_files = glob.glob(os.path.join(d, '*.json'))
        if timeout is None and json_files:
            try:
                with open(json_files[0], 'r') as f:
                    timeout = int(json.load(f).get('time_limit', 0))
            except Exception:
                timeout = None
        if timeout is None:
            continue

        # collect cpcs by n
        by_n = {}
        caps = {}
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                n = int(data.get('n_customers') or data.get('n') or 0)
                cap = int(data.get('capacity') or 0)
                cpcs = data.get('all_cpcs') or []
                if n and cpcs:
                    by_n.setdefault(n, []).extend(cpcs)
                    if n not in caps:
                        caps[n] = cap
            except Exception:
                continue
        
        for n, cpcs in by_n.items():
            stats_obj = compute_full_stats(cpcs)
            if stats_obj:
                results[(n, timeout)] = {
                    'cap': caps.get(n, 'N/A'),
                    'stats': stats_obj,
                }
    return results


def format_p(p):
    if math.isnan(p):
        return "--"
    return f"{p:.2f}" if p >= 0.01 else "$<$0.01"


def generate_table(results_map, test_mode=True):
    # Order rows: by N ascending, then Timeout ascending
    keys = sorted(results_map.keys(), key=lambda k: (k[0], k[1]))

    lines = []
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    caption = "OR-Tools GLS Test Results with Timeout Column (20 instances per configuration)" if test_mode else "OR-Tools GLS Results with Timeout Column (10,000 instances per configuration)"
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:ortools-gls-timeout}")
    lines.append("\\begin{tabular}{@{}l c c c S[table-format=1.4] S[table-format=1.4] c c c c c c@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{N} & \\textbf{Cap.} & \\textbf{Timeout} & {\\textbf{GM}} & {\\textbf{GSD}} & \\textbf{95\\% Range} & \\textbf{95\\% CI} & \\textbf{KS} & \\textbf{D'Agost.} & \\textbf{JB} & \\textbf{AD} \\\\")
    lines.append("\\midrule")

    for (n, timeout) in keys:
        ent = results_map[(n, timeout)]
        s = ent['stats']
        cap = ent['cap']
        ks = format_p(s['ks_p'])
        dag = format_p(s['dag_p'])
        jb = format_p(s['jb_p'])
        if not math.isnan(s['ad_stat']):
            ad = f"{s['ad_stat']:.3f}"
            if s['ad_stat'] > 0.787:
                ad += "*"
        else:
            ad = "--"
        line = (
            f"OR-Tools GLS & {n} & {cap} & {timeout}s & "
            f"{s['gm']:.4f} & {s['gsd']:.4f} & "
            f"[{s['range_lower']:.4f}, {s['range_upper']:.4f}] & "
            f"[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}] & "
            f"${ks}$ & ${dag}$ & ${jb}$ & ${ad}$ \\\\" 
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\small")
    lines.append("\\item GM: Geometric Mean, GSD: Geometric Standard Deviation")
    lines.append("\\item 95\\% Range: GM $\\times$ [GSD$^{-1.96}$, GSD$^{+1.96}$]")
    lines.append("\\item KS: Kolmogorov-Smirnov, D'Agost.: D'Agostino, JB: Jarque-Bera (p-values for log(CPC) normality)")
    lines.append("\\item AD: Anderson-Darling test statistic (critical value at 5\\% = 0.787; * indicates rejection)")
    lines.append("\\item Timeout column indicates per-instance time limit used by OR-Tools GLS")
    if test_mode:
        lines.append("\\item Test run with 20 instances per configuration")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['test', 'production'], default='test')
    ap.add_argument('--dirs', action='append', default=[], help='Explicit result directories to scan (repeatable)')
    ap.add_argument('--out', default=None, help='Output .tex file path')
    args = ap.parse_args()

    if args.dirs:
        dirs = args.dirs
    else:
        pattern = f"results/ortools_gls_*s_{args.mode}"
        dirs = sorted(glob.glob(pattern))

    res = gather_results(dirs)
    table = generate_table(res, test_mode=(args.mode == 'test'))

    if not args.out:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out = f"results/ortools_gls_timeout_{args.mode}_table_{ts}.tex"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(table)

    print(f"Saved table to: {args.out}")

if __name__ == '__main__':
    main()
