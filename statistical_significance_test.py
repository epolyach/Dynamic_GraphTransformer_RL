#!/usr/bin/env python3
"""
Statistical significance test for CPC differences
Using Standard Error of the Mean (SEM) and confidence intervals
"""

import numpy as np
import scipy.stats as stats

def calculate_confidence_interval(mean, std, n, confidence=0.95):
    """Calculate confidence interval using SEM"""
    sem = std / np.sqrt(n)
    # For 95% confidence interval, use z-score of 1.96
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * sem
    return mean - margin_of_error, mean + margin_of_error, sem

def check_overlap(interval1, interval2):
    """Check if two intervals overlap"""
    return interval1[0] <= interval2[1] and interval2[0] <= interval1[1]

def main():
    print("=" * 70)
    print("Statistical Analysis of CPC Differences")
    print("=" * 70)
    
    # Data from your reported benchmarks
    cpu_data = {
        'mean_cpc': 0.465060,
        'std_cpc': 0.084619,
        'n_instances': 100,
        'label': 'CPU (benchmark_exact_cpu.py)'
    }
    
    gpu_data = {
        'mean_cpc': 0.478376,
        'std_cpc': 0.087646,
        'n_instances': 100,
        'label': 'GPU (benchmark_gpu_exact.py)'
    }
    
    # Also test with 1000 instances
    gpu_1000_data = {
        'mean_cpc': 0.486089,
        'std_cpc': 0.081292,
        'n_instances': 1000,
        'label': 'GPU (1000 instances)'
    }
    
    datasets = [cpu_data, gpu_data, gpu_1000_data]
    
    # Calculate confidence intervals for each dataset
    print("\n95% Confidence Intervals:")
    print("-" * 70)
    
    intervals = []
    for data in datasets:
        lower, upper, sem = calculate_confidence_interval(
            data['mean_cpc'], 
            data['std_cpc'], 
            data['n_instances']
        )
        intervals.append((lower, upper))
        
        print(f"\n{data['label']}:")
        print(f"  Mean CPC: {data['mean_cpc']:.6f}")
        print(f"  Std Dev:  {data['std_cpc']:.6f}")
        print(f"  N:        {data['n_instances']}")
        print(f"  SEM:      {sem:.6f}")
        print(f"  95% CI:   [{lower:.6f}, {upper:.6f}]")
        print(f"  Interval width: {upper - lower:.6f}")
    
    # Check overlaps
    print("\n" + "=" * 70)
    print("Overlap Analysis:")
    print("-" * 70)
    
    # CPU vs GPU (100 instances)
    overlap_100 = check_overlap(intervals[0], intervals[1])
    print(f"\nCPU (100) vs GPU (100):")
    print(f"  CPU interval: [{intervals[0][0]:.6f}, {intervals[0][1]:.6f}]")
    print(f"  GPU interval: [{intervals[1][0]:.6f}, {intervals[1][1]:.6f}]")
    print(f"  Overlapping: {overlap_100}")
    
    if not overlap_100:
        gap = max(intervals[0][0] - intervals[1][1], intervals[1][0] - intervals[0][1])
        print(f"  Gap between intervals: {gap:.6f}")
    
    # CPU vs GPU (1000 instances)
    overlap_1000 = check_overlap(intervals[0], intervals[2])
    print(f"\nCPU (100) vs GPU (1000):")
    print(f"  CPU interval: [{intervals[0][0]:.6f}, {intervals[0][1]:.6f}]")
    print(f"  GPU interval: [{intervals[2][0]:.6f}, {intervals[2][1]:.6f}]")
    print(f"  Overlapping: {overlap_1000}")
    
    if not overlap_1000:
        gap = max(intervals[0][0] - intervals[2][1], intervals[2][0] - intervals[0][1])
        print(f"  Gap between intervals: {gap:.6f}")
    
    # Statistical test (t-test)
    print("\n" + "=" * 70)
    print("Statistical Significance (Welch's t-test):")
    print("-" * 70)
    
    # Calculate t-statistic for CPU vs GPU (100)
    pooled_sem = np.sqrt(
        (cpu_data['std_cpc']**2 / cpu_data['n_instances']) + 
        (gpu_data['std_cpc']**2 / gpu_data['n_instances'])
    )
    t_stat = abs(cpu_data['mean_cpc'] - gpu_data['mean_cpc']) / pooled_sem
    
    # Degrees of freedom (Welch-Satterthwaite)
    df = (
        (cpu_data['std_cpc']**2 / cpu_data['n_instances'] + 
         gpu_data['std_cpc']**2 / gpu_data['n_instances'])**2
    ) / (
        (cpu_data['std_cpc']**2 / cpu_data['n_instances'])**2 / (cpu_data['n_instances'] - 1) +
        (gpu_data['std_cpc']**2 / gpu_data['n_instances'])**2 / (gpu_data['n_instances'] - 1)
    )
    
    p_value = 2 * (1 - stats.t.cdf(t_stat, df))
    
    print(f"\nCPU vs GPU (100 instances):")
    print(f"  Difference in means: {abs(cpu_data['mean_cpc'] - gpu_data['mean_cpc']):.6f}")
    print(f"  Combined SEM: {pooled_sem:.6f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  degrees of freedom: {df:.1f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant at α=0.05? {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Significant at α=0.01? {'Yes' if p_value < 0.01 else 'No'}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((cpu_data['n_instances'] - 1) * cpu_data['std_cpc']**2 + 
         (gpu_data['n_instances'] - 1) * gpu_data['std_cpc']**2) / 
        (cpu_data['n_instances'] + gpu_data['n_instances'] - 2)
    )
    cohens_d = abs(cpu_data['mean_cpc'] - gpu_data['mean_cpc']) / pooled_std
    
    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
    if cohens_d < 0.2:
        print("  Interpretation: Negligible effect")
    elif cohens_d < 0.5:
        print("  Interpretation: Small effect")
    elif cohens_d < 0.8:
        print("  Interpretation: Medium effect")
    else:
        print("  Interpretation: Large effect")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("-" * 70)
    
    if not overlap_100:
        print("The confidence intervals DO NOT overlap.")
        print("This indicates a statistically significant difference between CPU and GPU results.")
        print("\nThis strongly suggests the benchmarks are testing DIFFERENT instance distributions.")
    else:
        print("The confidence intervals overlap.")
        print("The differences might be due to random variation.")
    
    print("\nRECOMMENDATION:")
    print("Run both solvers on the EXACT SAME instances to verify they produce identical results.")

if __name__ == "__main__":
    main()
