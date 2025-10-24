#!/usr/bin/env python3
"""
Generate a LaTeX table line comparing Mean, GM, Median and their SEs.
Adapted for OR-Tools Greedy (opt) JSON format with 'cpc', 'n', 'capacity' keys.

Usage:
  python3 generate_latex_comparison_adapted.py --input ortools_n6.json --method "OR-Tools Greedy (opt)" --timeout "—"
"""

import argparse
import json
import sys
from pathlib import Path

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


def load_json_data(json_path: Path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Adapt to OR-Tools JSON format
    if 'cpc' not in data:
        raise KeyError("Missing key 'cpc' in JSON")
    if 'n' not in data:
        raise KeyError("Missing key 'n' in JSON")
    if 'capacity' not in data:
        raise KeyError("Missing key 'capacity' in JSON")
    return data['cpc'], data['n'], data['capacity']


def se_median_kde(x: np.ndarray, kde_max_n: int = 0, random_seed: int = 0) -> float:
    """Estimate SE of the median using KDE at the sample median.

    SE(median) ≈ 1 / (2 f̂(m̂) sqrt(n)), where f̂ is Gaussian KDE evaluated at m̂.

    - kde_max_n: if >0 and len(x) > kde_max_n, subsample without replacement to speed up KDE.
    - Falls back to quantile-spacing estimator if KDE fails or returns non-positive density.
    """
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return float('nan')

    med = np.median(x)

    x_for_kde = x
    if kde_max_n and n > kde_max_n:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n, size=kde_max_n, replace=False)
        x_for_kde = x[idx]

    try:
        kde = stats.gaussian_kde(x_for_kde)
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


def compute_statistics(all_cpcs: list) -> dict:
    x = np.asarray(all_cpcs, dtype=float)
    if np.any(~np.isfinite(x)):
        raise ValueError("all_cpcs contains non-finite values")
    if np.any(x <= 0):
        raise ValueError("all_cpcs must be strictly positive for log/GM computations")

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
    se_gm = gm * np.log(gsd) / np.sqrt(n_inst) if n_inst > 0 and gsd > 1.0 else float('nan') if n_inst > 0 else float('nan')

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
        'n_inst': n_inst,
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


def generate_latex_row(method: str, n_customers, capacity, timeout: str, stats_dict: dict) -> str:
    def f4(v):
        return f"{v:.4f}"

    # Format errors with three significant digits in scientific notation
    def se3(v):
        try:
            return f"{float(v):.2e}"
        except Exception:
            return "NA"
    
    # Calculate max error
    max_error = max(
        stats_dict['se_mean'],
        stats_dict['se_gm'],
        stats_dict['se_median']
    )

    percentile_range = f"[{f4(stats_dict['p2_5'])}, {f4(stats_dict['p97_5'])}]"

    ks_cpc = format_pvalue(stats_dict['ks_cpc_p'])
    ks_log = format_pvalue(stats_dict['ks_log_p'])

    line = (
        f"{method} & {n_customers} & {capacity} & {timeout} & "
        f"{f4(stats_dict['mean'])} & {f4(stats_dict['gm'])} & {f4(stats_dict['median'])} & "
        f"{se3(max_error)} & "
        f"{percentile_range} & "
        f"{ks_cpc} & {ks_log} \\\\"
    )
    return line


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX line comparing Mean, GM, Median and their SEs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--method', type=str, required=True, help='Method name')
    parser.add_argument('--timeout', type=str, required=True, help='Timeout string')
    parser.add_argument('--output', type=str, default=None, help='Optional file to append the line to')
    parser.add_argument('--kde-max-n', type=int, default=0, help='If >0, subsample to this many points for KDE density only')
    parser.add_argument('--random-seed', type=int, default=0, help='Random seed for subsampling KDE')

    args = parser.parse_args()

    # Resolve file path
    json_path = Path(args.input)
    if not json_path.exists():
        json_path = Path.cwd() / args.input
    if not json_path.exists():
        print(f"❌ File not found: {args.input}", file=sys.stderr)
        return 1

    try:
        all_cpcs, n_customers, capacity = load_json_data(json_path)

        # Compute statistics; override se_median_kde with CLI options
        def se_median_kde_with_opts(x: np.ndarray) -> float:
            return se_median_kde(x, kde_max_n=args.kde_max_n, random_seed=args.random_seed)

        # Patch in our chosen KDE options
        stats_dict = compute_statistics(all_cpcs)
        # Recompute se_median with CLI options if non-default
        if args.kde_max_n != 0:
            x = np.asarray(all_cpcs, dtype=float)
            stats_dict['se_median'] = float(se_median_kde_with_opts(x))

        line = generate_latex_row(args.method, n_customers, capacity, args.timeout, stats_dict)

        if args.output:
            with open(args.output, 'a') as f:
                f.write(line + "\n")
            print(f"✅ Appended LaTeX line to {args.output}")
        else:
            print("\n📋 LaTeX table line (comparison):")
            print(line)
            
        # Print table header for reference
        print("\n📝 Table header (if needed):")
        print("\\textbf{Method} & \\textbf{N} & \\textbf{Cap.} & \\textbf{Timeout} & \\textbf{Mean} & \\textbf{GM} & \\textbf{Median} & \\textbf{Error} & \\textbf{95\\% Range} & \\textbf{KS CPC} & \\textbf{KS log(CPC)} \\\\")
        
        # Print additional stats
        print(f"\n📊 Statistics from {stats_dict['n_inst']} instances:")
        print(f"  SE(Mean): {stats_dict['se_mean']:.3e}")
        print(f"  SE(GM): {stats_dict['se_gm']:.3e}")
        print(f"  SE(Median): {stats_dict['se_median']:.3e}")
        max_err = max(stats_dict["se_mean"], stats_dict["se_gm"], stats_dict["se_median"])
        print(f"  Max Error: {max_err:.3e}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
