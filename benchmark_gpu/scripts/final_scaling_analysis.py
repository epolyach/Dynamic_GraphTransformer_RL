#!/usr/bin/env python3
"""
Final analysis of GPU-DP solver scaling with precise multiplication factors
"""
import numpy as np
import math

def main():
    print("GPU-DP Solver: PRECISE SCALING ANALYSIS")
    print("=" * 60)
    
    # Actual measured data (no timeouts)
    results = [
        (10, 15.749),   # N=10: 15.749s
        (11, 42.873),   # N=11: 42.873s  
        (12, 119.656),  # N=12: 119.656s
        (13, 338.090),  # N=13: 338.090s
        (14, 968.814),  # N=14: 968.814s
        # N=15: interrupted but estimated ~2716s based on 2.801x factor
    ]
    
    print("MEASURED RESULTS:")
    print("N\tTime (s)\tTime (min)\tStates\t\tMultiplier\tTheoretical\tEfficiency")
    print("-" * 85)
    
    multipliers = []
    
    for i, (n, time_s) in enumerate(results):
        states = 2**n
        time_min = time_s / 60
        
        if i == 0:
            print(f"{n}\t{time_s:8.1f}s\t{time_min:8.1f}m\t{states:,}\t\t-\t\t-\t\t-")
        else:
            prev_n, prev_time = results[i-1]
            multiplier = time_s / prev_time
            theoretical = 2.0  # Should double each step
            efficiency = multiplier / theoretical
            multipliers.append(multiplier)
            
            print(f"{n}\t{time_s:8.1f}s\t{time_min:8.1f}m\t{states:,}\t{multiplier:7.3f}x\t{theoretical:.3f}x\t\t{efficiency:.3f}")
    
    # Statistical analysis
    avg_multiplier = np.mean(multipliers)
    std_multiplier = np.std(multipliers)
    min_multiplier = np.min(multipliers)
    max_multiplier = np.max(multipliers)
    
    print(f"\nMULTIPLIER STATISTICS:")
    print(f"Average: {avg_multiplier:.3f}x per customer (theoretical: 2.000x)")
    print(f"Std Dev: {std_multiplier:.3f}x")
    print(f"Range:   {min_multiplier:.3f}x - {max_multiplier:.3f}x")
    print(f"Growth rate: 2^{math.log2(avg_multiplier):.3f} per customer")
    print(f"Efficiency loss: {((avg_multiplier/2.0 - 1) * 100):.1f}% worse than theoretical")
    
    # Trend analysis
    print(f"\nTREND ANALYSIS:")
    print("Multiplier is INCREASING with problem size:")
    for i, mult in enumerate(multipliers):
        n = results[i+1][0]
        print(f"  N={n-1}→{n}: {mult:.3f}x")
    
    # The multiplier itself is growing!
    if len(multipliers) >= 2:
        multiplier_growth = multipliers[-1] / multipliers[0]  # 2.87/2.72 = 1.055
        print(f"Multiplier growth: {multipliers[0]:.3f}x → {multipliers[-1]:.3f}x (+{multiplier_growth:.3f}x)")
    
    # Extrapolation with increasing multiplier
    print(f"\nEXTRAPOLATION TO LARGER N:")
    print("Using trend: multiplier increases by ~0.04x per step")
    print("N\tEstimated Time\t\tDescription")
    print("-" * 50)
    
    # More sophisticated extrapolation accounting for increasing multiplier
    current_n, current_time = results[-1]  # N=14, 968.814s
    current_mult = multipliers[-1]  # 2.87x
    mult_growth_rate = 0.04  # observed ~0.04 increase per step
    
    for target_n in range(15, 21):
        steps = target_n - current_n
        
        # Account for multiplier growth: each step the multiplier increases slightly
        total_mult = 1.0
        working_mult = current_mult
        
        for step in range(steps):
            total_mult *= working_mult
            working_mult += mult_growth_rate  # Multiplier gets worse each step
        
        extrapolated_time = current_time * total_mult
        
        if extrapolated_time < 3600:
            time_desc = f"{extrapolated_time/60:.1f} minutes"
        elif extrapolated_time < 86400:
            time_desc = f"{extrapolated_time/3600:.1f} hours"
        else:
            time_desc = f"{extrapolated_time/86400:.1f} days"
        
        print(f"{target_n}\t{extrapolated_time:10.0f}s\t\t{time_desc}")
    
    # Memory analysis
    print(f"\nMEMORY ANALYSIS:")
    print("N\tStates\t\tMemory/instance\tMax batch (8GB)")
    print("-" * 50)
    
    for n in [10, 12, 15, 18, 20]:
        states = 2**n
        # Conservative estimate: 8 bytes per state (float64), ~5 tensors
        memory_mb = states * 8 * 5 / (1024**2)
        max_batch = max(1, int(8192 / memory_mb)) if memory_mb > 0 else "∞"
        
        print(f"{n}\t{states:,}\t{memory_mb:10.1f} MB\t\t{max_batch}")
    
    print(f"\nKEY FINDINGS:")
    print(f"1. Scaling is WORSE than O(2^N) - multiplier ~2.8x instead of 2.0x")
    print(f"2. Multiplier is INCREASING with problem size (2.72x → 2.87x)")
    print(f"3. Super-exponential growth: O(2^{math.log2(avg_multiplier):.2f}^N)")
    print(f"4. N=15 estimated: ~45 minutes")
    print(f"5. N=20 estimated: ~{(current_time * (avg_multiplier**6))/86400:.1f} days")
    print(f"6. Practical limit for research: N≤15 (under 1 hour)")
    print(f"7. Absolute limit: N≤12 for routine use (under 2 minutes)")

if __name__ == "__main__":
    main()
