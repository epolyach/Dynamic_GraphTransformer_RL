#!/usr/bin/env python3
"""
Scaling test for GPU-DP solver - NO TIMEOUTS, measure actual multiplication factors
"""
import time
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from gpu_cvrp_solver_truly_optimal_fixed import GPUExactCVRPFixed

def time_solver(n_customers, capacity=30):
    """Time solver for given problem size - NO TIMEOUT"""
    print(f"\n{'='*60}")
    print(f"Testing N={n_customers}")
    print(f"State space size: 2^{n_customers} = {2**n_customers:,} states")
    print(f"Memory estimate: ~{2**n_customers * 4 / (1024**2):.1f} MB per tensor")
    
    # Generate single instance with fixed seed for consistency
    instance = _generate_instance(
        num_customers=n_customers,
        capacity=capacity,
        coord_range=100,
        demand_range=[1, 10],
        seed=42  # Same seed for all tests
    )
    
    # Initialize solver
    solver = GPUExactCVRPFixed()
    
    print(f"Starting solve at {time.strftime('%H:%M:%S')}")
    print("⏳ Running without timeout - please wait...")
    
    start_time = time.time()
    
    try:
        solutions = solver.solve_batch([instance], verbose=True)
        end_time = time.time()
        
        solve_time = end_time - start_time
        print(f"✅ COMPLETED in {solve_time:.3f} seconds")
        print(f"   Solution cost: {solutions[0].cost:.6f}")
        
        return solve_time, True
        
    except Exception as e:
        end_time = time.time()
        solve_time = end_time - start_time
        print(f"❌ FAILED after {solve_time:.3f} seconds")
        print(f"   Error: {e}")
        return solve_time, False

def main():
    print("GPU-DP Solver Scaling Analysis (NO TIMEOUTS)")
    print("Testing N=10, 11, 12, 13... until manual interrupt")
    print("Press Ctrl+C to stop at any point")
    
    results = []
    
    # Start from N=10 and go up
    n = 10
    while True:
        try:
            solve_time, success = time_solver(n)
            results.append((n, solve_time, success))
            
            # Calculate multiplication factor from previous
            if len(results) >= 2:
                prev_n, prev_time, prev_success = results[-2]
                if prev_success and success:
                    factor = solve_time / prev_time
                    theoretical_factor = 2**(n - prev_n)
                    print(f"📈 Multiplication factor vs N={prev_n}: {factor:.2f}x")
                    print(f"   Theoretical factor: {theoretical_factor:.2f}x")
                    print(f"   Efficiency ratio: {factor/theoretical_factor:.3f}")
            
            if not success:
                print(f"⚠️  Failed at N={n}, stopping")
                break
                
            n += 1
            
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted by user at N={n}")
            break
    
    # Final analysis
    print(f"\n{'='*70}")
    print("FINAL SCALING ANALYSIS")
    print(f"{'='*70}")
    
    successful_results = [(n, t) for n, t, success in results if success]
    
    if len(successful_results) >= 2:
        print("\nMeasured results:")
        print("N\tTime (s)\tFactor vs prev\tTheoretical\tEfficiency")
        
        for i, (n, solve_time) in enumerate(successful_results):
            if i == 0:
                print(f"{n}\t{solve_time:8.3f}s\t-\t\t-\t\t-")
            else:
                prev_n, prev_time = successful_results[i-1]
                factor = solve_time / prev_time
                theoretical = 2**(n - prev_n)
                efficiency = factor / theoretical
                print(f"{n}\t{solve_time:8.3f}s\t{factor:.2f}x\t\t{theoretical:.2f}x\t\t{efficiency:.3f}")
        
        # Calculate average multiplication factor
        factors = []
        for i in range(1, len(successful_results)):
            curr_n, curr_time = successful_results[i]
            prev_n, prev_time = successful_results[i-1]
            factor = curr_time / prev_time
            factors.append(factor)
        
        if factors:
            avg_factor = np.mean(factors)
            print(f"\nAverage multiplication factor: {avg_factor:.3f}x per customer")
            print(f"Theoretical multiplication factor: 2.000x per customer")
            print(f"Empirical growth rate: 2^{np.log2(avg_factor):.3f} per customer")
            
            # Extrapolate
            baseline_n, baseline_time = successful_results[0]
            print(f"\nExtrapolation from N={baseline_n}:")
            for target_n in [15, 16, 17, 18, 19, 20]:
                steps = target_n - baseline_n
                extrapolated_time = baseline_time * (avg_factor ** steps)
                
                if extrapolated_time < 3600:
                    time_str = f"{extrapolated_time:.1f}s ({extrapolated_time/60:.1f}m)"
                elif extrapolated_time < 86400:
                    time_str = f"{extrapolated_time:.0f}s ({extrapolated_time/3600:.1f}h)"
                else:
                    time_str = f"{extrapolated_time:.0f}s ({extrapolated_time/86400:.1f} days)"
                
                print(f"N={target_n}: {time_str}")

if __name__ == "__main__":
    main()
