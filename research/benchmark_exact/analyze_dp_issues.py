#!/usr/bin/env python3
"""
Analyze DP Solver Issues in Benchmark Results
"""

import csv

def analyze_dp_performance():
    print("ANALYSIS OF DP SOLVER PERFORMANCE")
    print("="*60)
    
    # Read CSV data
    with open('op_dp_c30.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    print("\nRaw Data:")
    print("N   | time_dp   | cpc_dp   | Notes")
    print("----|-----------|----------|------------------")
    
    for row in data:
        n = int(row['N'])
        time_dp = float(row['time_dp'])
        cpc_dp = float(row['cpc_dp'])
        
        notes = []
        if cpc_dp > 0.4:
            notes.append('HIGH_COST')
        if time_dp > 10:
            notes.append('VERY_SLOW')
        elif time_dp > 1:
            notes.append('SLOW')
            
        notes_str = ' '.join(notes) if notes else 'OK'
        print(f"{n:2d}  | {time_dp:8.3f}s | {cpc_dp:.3f}  | {notes_str}")
    
    print("\nDP Time Growth Analysis:")
    for i in range(1, len(data)):
        prev_row = data[i-1]
        curr_row = data[i]
        
        prev_n = int(prev_row['N'])
        curr_n = int(curr_row['N'])
        prev_time = float(prev_row['time_dp'])
        curr_time = float(curr_row['time_dp'])
        
        growth = curr_time / prev_time if prev_time > 0 else float('inf')
        print(f"N={prev_n}->{curr_n}: {prev_time:6.3f}s -> {curr_time:8.3f}s (growth: {growth:5.1f}x)")
    
    print("\n" + "="*60)
    print("CRITICAL ISSUES DETECTED:")
    print("="*60)
    
    issues_found = False
    
    for row in data:
        n = int(row['N'])
        time_dp = float(row['time_dp'])
        cpc_dp = float(row['cpc_dp'])
        
        # Issue 1: High cost suggests fallback to greedy
        if n >= 15 and cpc_dp > 0.4:
            print(f"\n!!! CRITICAL ERROR at N={n}:")
            print(f"    DP cost = {cpc_dp:.3f} (expected ~0.2-0.3 for exact solver)")
            print(f"    This suggests DP solver is FAILING and using GREEDY FALLBACK!")
            issues_found = True
            
        # Issue 2: Exponential explosion
        if time_dp > 10:
            print(f"\n!!! EXPONENTIAL EXPLOSION at N={n}:")
            print(f"    DP time = {time_dp:.1f}s (becoming impractical)")
            issues_found = True
    
    if not issues_found:
        print("No critical issues detected.")
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    
    print("\n1. EXPONENTIAL TIME COMPLEXITY:")
    print("   - DP has O(n^2 * 2^n) complexity")
    print("   - N=12: 0.6s, N=13: 1.9s, N=14: 6.4s, N=15: 12.6s, N=16: 49.3s")
    print("   - This is expected behavior for exact DP solver")
    
    print("\n2. QUALITY DEGRADATION (N>=15):")
    print("   - N=15: cpc_dp = 0.478 (should be ~0.23)")
    print("   - N=16: cpc_dp = 0.436 (should be ~0.23)")
    print("   - This indicates DP solver is timing out and using greedy fallback")
    
    print("\n3. ROOT CAUSE:")
    print("   - DP solver has internal time limits")
    print("   - When exceeded, falls back to greedy construction")
    print("   - Greedy solutions are much worse quality (non-optimal)")
    
    print("\n4. RECOMMENDATIONS:")
    print("   - Use DP only for N <= 14")
    print("   - Use OR-Tools for N >= 15")
    print("   - Consider increasing DP time limits for research purposes")
    print("   - The crossover point is around N=12-13 for practical use")

if __name__ == "__main__":
    analyze_dp_performance()
