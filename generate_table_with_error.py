#!/usr/bin/env python3
"""
Generate comparison table with relative error column
"""

import numpy as np

def main():
    # Data from previous runs
    data = [
        ("CPU", 1000, 0.464466, 0.090135, 0.002850),
        ("GPU", 1000, 0.460799, 0.091148, 0.002882),
        ("GPU", 10000, 0.466432, 0.089185, 0.000892),
        ("GPU", 100000, 0.466568, 0.089946, 0.000284),
    ]
    
    print("| Solver | Instances | Mean CPC | Std CPC | SEM     | 2×SEM/Mean(%) | 95% CI              |")
    print("|--------|-----------|----------|---------|---------|---------------|---------------------|")
    
    for solver, n, mean, std, sem in data:
        ci_lower = mean - 1.96 * sem
        ci_upper = mean + 1.96 * sem
        relative_error = (2 * sem / mean) * 100
        
        print(f"| {solver:6s} |    {n:6d} | {mean:8.6f} | {std:7.6f} | {sem:7.6f} |        {relative_error:5.2f}% | [{ci_lower:7.6f}, {ci_upper:7.6f}] |")

if __name__ == "__main__":
    main()
