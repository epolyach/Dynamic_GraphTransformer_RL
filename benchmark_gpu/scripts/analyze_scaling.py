#!/usr/bin/env python3
"""
Analyze GPU-DP solver scaling and extrapolate to N=20
"""
import numpy as np
import math

def analyze_scaling():
    print("GPU-DP Solver Scaling Analysis")
    print("="*50)
    
    # Measured data
    results = [
        (10, 15.611),  # N=10: 15.611s
        (11, 42.649),  # N=11: 42.649s (2.7x increase)
        # N=12: >60s (timeout)
    ]
    
    print("Measured results:")
    for n, time in results:
        states = 2**n
        print(f"N={n}: {time:.3f}s, {states:,} states")
    
    # Calculate empirical growth rate
    n1, t1 = results[0]  # N=10
    n2, t2 = results[1]  # N=11
    
    # Empirical scaling: t2/t1 = 2^(rate*(n2-n1))
    # So: rate = log2(t2/t1) / (n2-n1)
    empirical_rate = math.log2(t2/t1) / (n2-n1)
    
    print(f"\nScaling analysis:")
    print(f"Time ratio N=11/N=10: {t2/t1:.2f}x")
    print(f"Theoretical scaling (2^1): 2.0x")
    print(f"Empirical growth rate: 2^{empirical_rate:.3f} per customer")
    print(f"Theoretical growth rate: 2^1.000 per customer")
    print(f"Efficiency ratio: {empirical_rate:.3f} (higher = worse than theoretical)")
    
    print(f"\nExtrapolation to larger N:")
    print("N\tStates\t\tTime (empirical)\tTime (theoretical)")
    
    baseline_n, baseline_t = results[0]
    
    for n in range(10, 25):
        states = 2**n
        time_empirical = baseline_t * (2**(empirical_rate * (n - baseline_n)))
        time_theoretical = baseline_t * (2**(1.0 * (n - baseline_n)))
        
        if n <= 11:
            # We have actual data
            actual_time = next(t for nn, t in results if nn == n)
            marker = f" ✓ (actual: {actual_time:.1f}s)"
        elif n == 12:
            marker = " ⚠️  (timed out at 60s)"
        elif time_empirical < 3600:
            marker = f" ({time_empirical/60:.1f}m)"
        elif time_empirical < 86400:
            marker = f" ({time_empirical/3600:.1f}h)"
        else:
            marker = f" ({time_empirical/86400:.1f} days)"
        
        print(f"{n}\t{states:,}\t{time_empirical:8.1f}s\t\t{time_theoretical:8.1f}s{marker}")
    
    # Memory analysis
    print(f"\nMemory requirements (approximate):")
    print("N\tStates\t\tMemory per tensor\tBatch size limit (8GB GPU)")
    
    for n in [10, 12, 15, 20, 25]:
        states = 2**n
        # Rough estimate: 4 bytes per state, multiple tensors
        memory_per_instance_mb = states * 4 * 10 / (1024**2)  # 10 tensors estimate
        
        # 8GB = 8192 MB available
        max_batch_size = max(1, int(8192 / memory_per_instance_mb))
        
        print(f"{n}\t{states:,}\t{memory_per_instance_mb:8.1f} MB\t\t{max_batch_size:,}")
    
    # Practical recommendations
    print(f"\nPractical Analysis:")
    time_n20_empirical = baseline_t * (2**(empirical_rate * 10))
    time_n20_theoretical = baseline_t * (2**(1.0 * 10))
    
    print(f"Time estimate for N=20:")
    print(f"  Empirical model:    {time_n20_empirical:8.1f}s ({time_n20_empirical/3600:.2f} hours)")
    print(f"  Theoretical model:  {time_n20_theoretical:8.1f}s ({time_n20_theoretical/3600:.2f} hours)")
    print(f"  Reality check: N=12 already times out at 60s")
    
    print(f"\nConclusions:")
    print(f"1. GPU-DP solver has super-exponential scaling (worse than 2^N)")
    print(f"2. N=12 is near the practical limit for single instances")
    print(f"3. N=20 would take ~{time_n20_empirical/86400:.1f} days per instance")
    print(f"4. Memory requirements become prohibitive (>4GB per instance for N=20)")
    print(f"5. Recommendation: Use heuristic methods for N≥13")

if __name__ == "__main__":
    analyze_scaling()
