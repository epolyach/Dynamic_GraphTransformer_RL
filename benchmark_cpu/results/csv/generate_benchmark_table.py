#!/usr/bin/env python3
"""
Generate a LaTeX table from benchmark CSV data with statistics for multiple methods and problem sizes.
Computes Mean, GM, Median, and their SEs across instances for each (n, method) combination.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def format_pvalue(pvalue: float) -> str:
    """Format p-value for LaTeX output."""
    try:
        p = float(pvalue)
    except Exception:
        return "$\\mathrm{NA}$"
    if p < 0.01:
        return "$<0.01$"
    else:
        return f"${p:.2f}$"


def se_median_kde(x: np.ndarray) -> float:
    """Estimate SE of the median using KDE at the sample median."""
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return float('nan')

    med = np.median(x)

    try:
        kde = stats.gaussian_kde(x)
        fhat = float(kde(med)[0])
    except Exception:
        fhat = np.nan

    if not np.isfinite(fhat) or fhat <= 0.0:
        # Fallback: quantile-spacing estimator with h=0.01
        h = 0.01
        try:
            q_low = np.percentile(x, 100*(0.5 - h))
            q_high = np.percentile(x, 100*(0.5 + h))
            se_fallback = (q_high - q_low) / (4.0 * h * np.sqrt(n))
            return float(se_fallback)
        except Exception:
            return float('nan')

    return 1.0 / (2.0 * fhat * np.sqrt(n))


def compute_statistics(cpc_values: np.ndarray) -> dict:
    """Compute statistics for a set of CPC values."""
    x = np.asarray(cpc_values, dtype=float)
    
    # Remove NaN values
    x = x[~np.isnan(x)]
    
    if len(x) == 0:
        return {
            'mean': np.nan,
            'se_mean': np.nan,
            'gm': np.nan,
            'se_gm': np.nan,
            'median': np.nan,
            'se_median': np.nan,
            'p2_5': np.nan,
            'p97_5': np.nan,
            'ks_cpc_p': np.nan,
            'ks_log_p': np.nan,
        }
    
    if np.any(x <= 0):
        raise ValueError("CPC values must be strictly positive for log/GM computations")

    n_inst = x.size

    # Mean and SE(Mean)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n_inst > 1 else 0.0
    se_mean = sd / np.sqrt(n_inst) if n_inst > 0 else float('nan')

    # Log transform
    y = np.log(x)
    y_mean = float(np.mean(y))
    y_sd = float(np.std(y, ddof=1)) if n_inst > 1 else 0.0

    # GM and SE(GM) via GSD
    gm = float(np.exp(y_mean))
    gsd = float(np.exp(y_sd)) if n_inst > 1 else 1.0
    # SE(GM) = GM * log(GSD) / sqrt(n)
    se_gm = gm * np.log(gsd) / np.sqrt(n_inst) if n_inst > 0 and gsd > 1.0 else float('nan')

    # Median and SE(Median) via KDE
    median = float(np.median(x))
    se_median = float(se_median_kde(x))

    # 95% percentile range
    p2_5, p97_5 = np.percentile(x, [2.5, 97.5])

    # KS tests
    # CPC vs N(mean, sd)
    ks_cpc_stat, ks_cpc_p = stats.kstest(x, 'norm', args=(mean, sd if sd > 0 else 1.0))
    # log CPC vs N(y_mean, y_sd)
    ks_log_stat, ks_log_p = stats.kstest(y, 'norm', args=(y_mean, y_sd if y_sd > 0 else 1.0))

    return {
        'mean': mean,
        'se_mean': se_mean,
        'gm': gm,
        'se_gm': se_gm,
        'median': median,
        'se_median': se_median,
        'p2_5': float(p2_5),
        'p97_5': float(p97_5),
        'ks_cpc_p': float(ks_cpc_p),
        'ks_log_p': float(ks_log_p),
    }


def generate_latex_row(method: str, n: int, stats_dict: dict, time_str: str) -> str:
    """Generate a LaTeX table row for one method and problem size."""
    def f4(v):
        if np.isnan(v):
            return "---"
        return f"{v:.4f}"

    # Format errors with three significant digits in scientific notation
    def se3(v):
        if np.isnan(v):
            return "---"
        try:
            return f"{float(v):.2e}"
        except Exception:
            return "---"
    
    # Calculate max error
    errors = [stats_dict['se_mean'], stats_dict['se_gm'], stats_dict['se_median']]
    valid_errors = [e for e in errors if not np.isnan(e)]
    max_error = max(valid_errors) if valid_errors else np.nan

    percentile_range = f"[{f4(stats_dict['p2_5'])}, {f4(stats_dict['p97_5'])}]"
    if np.isnan(stats_dict['p2_5']) or np.isnan(stats_dict['p97_5']):
        percentile_range = "[---, ---]"

    ks_cpc = format_pvalue(stats_dict['ks_cpc_p']) if not np.isnan(stats_dict['ks_cpc_p']) else "---"
    ks_log = format_pvalue(stats_dict['ks_log_p']) if not np.isnan(stats_dict['ks_log_p']) else "---"

    line = (
        f"{method} & {n} & 30 & {time_str} & "
        f"{f4(stats_dict['mean'])} & {f4(stats_dict['gm'])} & {f4(stats_dict['median'])} & "
        f"{se3(max_error)} & "
        f"{percentile_range} & "
        f"{ks_cpc} & {ks_log} \\\\"
    )
    return line


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX benchmark table from CSV data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output LaTeX file')
    
    args = parser.parse_args()

    # Load CSV data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Define the methods and their display names
    # For n <= 8: exact_dp and ortools_greedy are optimal
    # For n > 8: ortools_greedy becomes sub-optimal
    
    table_rows = []
    
    # Process each n from 5 to 20
    for n in range(5, 21):
        print(f"Processing n={n}...")
        
        # Filter data for this n
        df_n = df[df['n'] == n]
        
        # Determine which methods to include
        if n <= 8:
            # Include exact_dp, ortools_greedy (optimal), ortools_gls
            methods = [
                ('exact_dp', 'Exact DP'),
                ('ortools_greedy', 'OR-Tools Greedy (opt)'),
                ('ortools_gls', 'OR-Tools GLS')
            ]
        else:
            # Only ortools_greedy (sub-optimal) and ortools_gls
            methods = [
                ('ortools_greedy', 'OR-Tools Greedy (sub)'),
                ('ortools_gls', 'OR-Tools GLS')
            ]
        
        for solver_name, display_name in methods:
            # Filter for this solver
            df_solver = df_n[df_n['solver'] == solver_name]
            
            if len(df_solver) == 0:
                continue
            
            # Get CPC values and compute statistics
            cpc_values = df_solver['cpc'].values
            stats = compute_statistics(cpc_values)
            
            # Calculate mean time
            mean_time = df_solver['time'].mean()
            if mean_time < 0.01:
                time_str = f"{mean_time*1000:.2f}ms"
            elif mean_time < 1:
                time_str = f"{mean_time:.3f}s"
            else:
                time_str = f"{mean_time:.2f}s"
            
            # Generate row
            row = generate_latex_row(display_name, n, stats, time_str)
            table_rows.append(row)
    
    # Generate complete table
    table = f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Benchmark Results: Comparison of Methods Across Problem Sizes}}
\\label{{tab:benchmark-comparison}}
\\begin{{tabular}}{{@{{}}l c c r r r r c c c c@{{}}}}
\\toprule
\\textbf{{Method}} & \\textbf{{N}} & \\textbf{{Cap.}} & \\textbf{{Time}} & \\textbf{{Mean}} & \\textbf{{GM}} & \\textbf{{Median}} & \\textbf{{Error}} & \\textbf{{95\\% Range}} & \\textbf{{KS CPC}} & \\textbf{{KS log(CPC)}} \\\\
\\midrule"""
    
    # Add rows with horizontal lines every few rows for readability
    for i, row in enumerate(table_rows):
        table += "\n" + row
        # Add a subtle separator between different n values
        if (i + 1) < len(table_rows):
            next_n = int(table_rows[i + 1].split(' & ')[1])
            current_n = int(row.split(' & ')[1])
            if next_n != current_n:
                table += "\n\\midrule"
    
    table += f"""
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item Statistics computed over 1000 instances per configuration
\\item Mean: Arithmetic mean; GM: Geometric mean
\\item Error: Maximum of SE(Mean), SE(GM), SE(Median) where:
\\item \\quad SE(Mean) = $s/\\sqrt{{n}}$; SE(GM) = $\\text{{GM}} \\times \\log(\\text{{GSD}})/\\sqrt{{n}}$; SE(Median) via KDE
\\item 95\\% Range: [2.5th percentile, 97.5th percentile] of CPC values
\\item KS: Kolmogorov-Smirnov test p-values for normality
\\item (opt): optimal solver; (sub): sub-optimal solver
\\end{{tablenotes}}
\\end{{table*}}"""
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(table + '\n')
        print(f"✅ Written complete LaTeX table to {args.output}")
    else:
        print("\n📋 Complete LaTeX table:")
        print(table)
    
    print(f"\n✅ Generated table with {len(table_rows)} rows")


if __name__ == '__main__':
    sys.exit(main())
