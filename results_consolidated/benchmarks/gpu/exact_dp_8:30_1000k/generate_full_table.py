#!/usr/bin/env python3
"""
Generate a complete LaTeX table with headers using the comparison script.
"""

import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate complete LaTeX table with comparison data')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--method', type=str, required=True, help='Method name')
    parser.add_argument('--timeout', type=str, required=True, help='Timeout string')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    
    args = parser.parse_args()
    
    # Get the data row from the comparison script
    try:
        result = subprocess.run([
            'python3', 'generate_latex_comparison.py',
            '--input', args.input,
            '--method', args.method,
            '--timeout', args.timeout
        ], capture_output=True, text=True, check=True)
        
        # Extract the data row (last line of output)
        lines = result.stdout.strip().split('\n')
        data_row = lines[-1]
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running comparison script: {e}", file=sys.stderr)
        return 1
    
    # Generate complete table with clean format and max error column
    table = f"""\\begin{{table*}}[htbp]
\\centering
\\caption{{Comparison of Mean, Geometric Mean, and Median with Standard Errors}}
\\label{{tab:central-tendency-comparison}}
\\begin{{tabular}}{{@{{}}l c c c r r r c c c c@{{}}}}
\\toprule
\\textbf{{Method}} & \\textbf{{N}} & \\textbf{{Cap.}} & \\textbf{{Timeout}} & \\textbf{{Mean}} & \\textbf{{GM}} & \\textbf{{Median}} & \\textbf{{Error}} & \\textbf{{95\\% Range}} & \\textbf{{KS CPC}} & \\textbf{{KS log(CPC)}} \\\\
\\midrule
{data_row}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item Mean: Arithmetic mean; GM: Geometric mean
\\item Error: Maximum of SE(Mean), SE(GM), SE(Median) where:
\\item \\quad SE(Mean) = $s/\\sqrt{{n}}$; SE(GM) = $\\text{{GM}} \\times \\log(\\text{{GSD}})/\\sqrt{{n}}$; SE(Median) via KDE at sample median
\\item 95\\% Range: [2.5th percentile, 97.5th percentile] of CPC values
\\item KS: Kolmogorov-Smirnov test p-values for normality (CPC and log(CPC))
\\end{{tablenotes}}
\\end{{table*}}"""
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(table + '\n')
        print(f"✅ Written complete LaTeX table to {args.output}")
    else:
        print("\n📋 Complete LaTeX table:")
        print(table)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
