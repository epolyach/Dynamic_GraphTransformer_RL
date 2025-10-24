#!/usr/bin/env python3
"""
Generate a LaTeX table line from OR-Tools GLS benchmark results.
Includes geometric mean, geometric std dev, and normality tests for log(CPC).
"""

import json
import argparse
import numpy as np
from scipy import stats
from pathlib import Path
import sys


def load_cpc_data(json_file):
    """Load CPC values from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['cpc'], data['n'], data.get('capacity', 'N/A'), len(data['cpc'])


def compute_statistics(cpc_values):
    """Compute all required statistics for the LaTeX table."""
    cpc_array = np.array(cpc_values)
    log_cpc = np.log(cpc_array)
    
    # Geometric mean and geometric standard deviation
    gm = np.exp(np.mean(log_cpc))
    gsd = np.exp(np.std(log_cpc))
    
    # 95% range using geometric std dev
    range_low = gm * (gsd ** -1.96)
    range_high = gm * (gsd ** 1.96)
    
    # 95% confidence interval for geometric mean
    n = len(cpc_values)
    se_log = np.std(log_cpc) / np.sqrt(n)
    ci_low = np.exp(np.mean(log_cpc) - 1.96 * se_log)
    ci_high = np.exp(np.mean(log_cpc) + 1.96 * se_log)
    
    # Normality tests on log(CPC)
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.kstest(log_cpc, 'norm', args=(np.mean(log_cpc), np.std(log_cpc)))
    
    # D'Agostino test
    try:
        dagostino_stat, dagostino_pvalue = stats.normaltest(log_cpc)
    except:
        dagostino_pvalue = 1.0  # Default if test fails
    
    # Jarque-Bera test
    try:
        jb_stat, jb_pvalue = stats.jarque_bera(log_cpc)
    except:
        jb_pvalue = 1.0  # Default if test fails
    
    # Anderson-Darling test
    ad_result = stats.anderson(log_cpc, dist='norm')
    ad_stat = ad_result.statistic
    # Critical value at 5% significance level
    ad_critical = ad_result.critical_values[2]  # Index 2 is for 5% significance
    
    return {
        'gm': gm,
        'gsd': gsd,
        'range_low': range_low,
        'range_high': range_high,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'ks_pvalue': ks_pvalue,
        'dagostino_pvalue': dagostino_pvalue,
        'jb_pvalue': jb_pvalue,
        'ad_stat': ad_stat,
        'ad_critical': ad_critical
    }


def format_pvalue(pvalue):
    """Format p-value for LaTeX table."""
    if pvalue < 0.01:
        return "$<$0.01"
    elif pvalue < 0.05:
        return f"$0.0{int(pvalue*100):d}$"
    else:
        return f"${pvalue:.2f}$"


def generate_latex_line(json_file, timeout, method_name="OR-Tools GLS"):
    """Generate a single LaTeX table line from JSON file."""
    
    # Load data
    cpc_values, n, capacity, num_instances = load_cpc_data(json_file)
    
    # Compute statistics
    stats_dict = compute_statistics(cpc_values)
    
    # Format Anderson-Darling statistic
    ad_str = f"{stats_dict['ad_stat']:.3f}"
    if stats_dict['ad_stat'] > stats_dict['ad_critical']:
        ad_str += "*"  # Mark if rejected at 5% level
    
    # Build LaTeX line
    line = f"{method_name} & {n} & {capacity} & {timeout} & "
    line += f"{stats_dict['gm']:.4f} & {stats_dict['gsd']:.4f} & "
    line += f"[{stats_dict['range_low']:.4f}, {stats_dict['range_high']:.4f}] & "
    line += f"[{stats_dict['ci_low']:.4f}, {stats_dict['ci_high']:.4f}] & "
    line += f"{format_pvalue(stats_dict['ks_pvalue'])} & "
    line += f"{format_pvalue(stats_dict['dagostino_pvalue'])} & "
    line += f"{format_pvalue(stats_dict['jb_pvalue'])} & "
    line += f"${ad_str}$ \\\\"
    
    return line, num_instances


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX table line from OR-Tools GLS results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file (e.g., ortools_n10.json)')
    parser.add_argument('--timeout', type=str, default='1s',
                        help='Timeout string for the table (e.g., 1s, 10s, 30s)')
    parser.add_argument('--method', type=str, default='OR-Tools GLS',
                        help='Method name for the first column')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to append the line (optional)')
    
    args = parser.parse_args()
    
    try:
        # Check if file exists
        json_path = Path(args.input)
        if not json_path.exists():
            json_path = Path.cwd() / args.input
        
        if not json_path.exists():
            print(f"❌ File not found: {args.input}", file=sys.stderr)
            return 1
        
        # Generate LaTeX line
        latex_line, num_instances = generate_latex_line(json_path, args.timeout, args.method)
        
        # Print or save the line
        if args.output:
            with open(args.output, 'a') as f:
                f.write(latex_line + '\n')
            print(f"✅ Appended LaTeX line to {args.output}")
        else:
            print("\n📋 LaTeX table line:")
            print(latex_line)
        
        # Print additional info
        print(f"\n📊 Generated from {num_instances} instances in {json_path.name}")
        
        # Also print a template if needed
        print("\n📝 Full table template (if needed):")
        print("""\\begin{table*}[htbp]
\\centering
\\caption{OR-Tools GLS Benchmark Results}
\\label{tab:ortools-gls-benchmark}
\\begin{tabular}{@{}l c c c S[table-format=1.4] S[table-format=1.4] c c c c c c@{}}
\\toprule
\\textbf{Method} & \\textbf{N} & \\textbf{Cap.} & \\textbf{Timeout} & {\\textbf{GM}} & {\\textbf{GSD}} & \\textbf{95\\% Range} & \\textbf{95\\% CI} & \\textbf{KS} & \\textbf{D'Agost.} & \\textbf{JB} & \\textbf{AD} \\\\
\\midrule""")
        print(latex_line)
        print("""\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item GM: Geometric Mean, GSD: Geometric Standard Deviation
\\item 95\\% Range: GM $\\times$ [GSD$^{-1.96}$, GSD$^{+1.96}$]
\\item KS: Kolmogorov-Smirnov, D'Agost.: D'Agostino, JB: Jarque-Bera (p-values for log(CPC) normality)
\\item AD: Anderson-Darling test statistic (critical value at 5\\% = 0.787; * indicates rejection)
\\end{tablenotes}
\\end{table*}""")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
