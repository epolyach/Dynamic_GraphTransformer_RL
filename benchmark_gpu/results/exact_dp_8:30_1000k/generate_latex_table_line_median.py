#!/usr/bin/env python3
"""
Generate a LaTeX table line from benchmark results using median-based statistics.
Includes median, percentiles, and normality tests for log(CPC).
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
    return data['all_cpcs'], data['n_customers'], data.get('capacity', 'N/A'), len(data['all_cpcs'])


def compute_statistics(cpc_values):
    """Compute all required statistics for the LaTeX table using median-based approach."""
    cpc_array = np.array(cpc_values)
    log_cpc = np.log(cpc_array)
    n = len(cpc_values)
    
    # Median instead of geometric mean
    median = np.median(cpc_array)
    
    # NaN instead of geometric standard deviation
    gsd = np.nan
    
    # 2.5 and 97.5 percentile range instead of 95% Range
    percentile_25 = np.percentile(cpc_array, 2.5)
    percentile_975 = np.percentile(cpc_array, 97.5)
    
    # 2.5 and 97.5 percentile range divided by sqrt(n) instead of 95% CI
    ci_low = percentile_25 / np.sqrt(n)
    ci_high = percentile_975 / np.sqrt(n)
    
    # Normality tests on log(CPC) - keeping the same tests
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
        'median': median,
        'gsd': gsd,
        'percentile_25': percentile_25,
        'percentile_975': percentile_975,
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
    """Generate a single LaTeX table line from JSON file using median-based statistics."""
    
    # Load data
    cpc_values, n, capacity, num_instances = load_cpc_data(json_file)
    
    # Compute statistics
    stats_dict = compute_statistics(cpc_values)
    
    # Format Anderson-Darling statistic
    ad_str = f"{stats_dict['ad_stat']:.3f}"
    if stats_dict['ad_stat'] > stats_dict['ad_critical']:
        ad_str += "*"  # Mark if rejected at 5% level
    
    # Build LaTeX line with median-based statistics
    line = f"{method_name} & {n} & {capacity} & {timeout} & "
    line += f"{stats_dict['median']:.4f} & NaN & "  # Median instead of GM, NaN instead of GSD
    line += f"[{stats_dict['percentile_25']:.4f}, {stats_dict['percentile_975']:.4f}] & "  # 2.5-97.5 percentile range
    line += f"[{stats_dict['ci_low']:.4f}, {stats_dict['ci_high']:.4f}] & "  # Percentile range / sqrt(n)
    line += f"{format_pvalue(stats_dict['ks_pvalue'])} & "
    line += f"{format_pvalue(stats_dict['dagostino_pvalue'])} & "
    line += f"{format_pvalue(stats_dict['jb_pvalue'])} & "
    line += f"${ad_str}$ \\\\\\"
    
    return line, num_instances


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX table line from benchmark results using median-based statistics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file (e.g., gpu_dp_exact_results_20250908_104018.json)')
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
            print("\n📋 LaTeX table line (median-based):")
            print(latex_line)
        
        # Print additional info
        print(f"\n📊 Generated from {num_instances} instances in {json_path.name}")
        
        # Also print a template if needed
        print("\n📝 Full table template (if needed):")
        print("""\\begin{table*}[htbp]
\\centering
\\caption{Benchmark Results (Median-based Statistics)}
\\label{tab:benchmark-median}
\\begin{tabular}{@{}l c c c S[table-format=1.4] c c c c c c c@{}}
\\toprule
\\textbf{Method} & \\textbf{N} & \\textbf{Cap.} & \\textbf{Timeout} & {\\textbf{Median}} & \\textbf{GSD} & \\textbf{2.5-97.5\\% Range} & \\textbf{Scaled Range} & \\textbf{KS} & \\textbf{D'Agost.} & \\textbf{JB} & \\textbf{AD} \\\\
\\midrule""")
        print(latex_line)
        print("""\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Median: Median CPC value, GSD: Not applicable (NaN)
\\item 2.5-97.5\\% Range: [2.5th percentile, 97.5th percentile] of CPC values
\\item Scaled Range: [2.5th percentile / √n, 97.5th percentile / √n]
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
