#!/usr/bin/env python3
"""
Scaling test for GPU-DP solver from N=10 to N=16, then extrapolate
"""
import time
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from gpu_cvrp_solver_truly_optimal_fixed import GPUExactCVRPFixed

def time_solver(n_customers, capacity=30, timeout_seconds=300):
    """Time solver for given problem size with timeout"""
    print(f"\n{'='*50}")
    print(f"Testing N={n_customers}")
    print(f"State space size: 2^{n_customers} = {2**n_customers:,} states")
    print(f"Memory estimate: ~{2**n_customers * 4 / (1024**2):.1f} MB per tensor")
    print(f"Theoretical scaling vs N=10: {2**n_customers / 2**10:.1f}x")
    
    # Generate single instance
    instance = _generate_instance(
        num_customers=n_customers,
        capacity=capacity,
        coord_range=100,
        demand_range=[1, 10],
        seed=42  # Same seed for consistency
    )
    
    # Initialize solver
    solver = GPUExactCVRPFixed()
    
    print(f"Starting solve at {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    try:
        # Add timeout mechanism
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Solver timed out after {timeout_seconds} seconds")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        solutions = solver.solve_batch([instance], verbose=True)
        
        signal.alarm(0)  # Cancel alarm
        end_time = time.time()
        
        solve_time = end_time - start_time
        print(f"✓ SOLVED in {solve_time:.3f} seconds")
        print(f"  Solution cost: {solutions[0].cost:.6f}")
        return solve_time, True
        
    except TimeoutError as e:
        end_time = time.time()
        solve_time = end_time - start_time
        print(f"✗ TIMEOUT after {solve_time:.1f} seconds")
        return solve_time, False
        
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        end_time = time.time()
        solve_time = end_time - start_time
        print(f"✗ ERROR after {solve_time:.3f} seconds: {e}")
        return solve_time, False

def main():
    print("GPU-DP Solver Scaling Analysis")
    print("Testing problem sizes from N=10 to N=16")
    
    results = []
    
    # Test range: 10, 11, 12, 13, 14, 15, 16
    for n in range(10, 17):
        try:
            solve_time, success = time_solver(n, timeout_seconds=60)  # 1-minute timeout per size
            results.append((n, solve_time, success))
            
            # If we hit timeout, stop testing larger sizes
            if not success and solve_time >= 55:  # Nearly full timeout
                print(f"\n⚠️  Stopping at N={n} due to timeout")
                break
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted at N={n}")
            break
    
    # Analysis and extrapolation
    print(f"\n{'='*60}")
    print("SCALING ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    successful_results = [(n, t) for n, t, success in results if success]
    
    if len(successful_results) >= 2:
        print("\nSuccessful solves:")
        print("N\tTime (s)\tScaling vs N=10\tActual vs Theoretical")
        
        baseline_time = None
        for n, solve_time in successful_results:
            if n == 10:
                baseline_time = solve_time
                scaling_actual = 1.0
            else:
                scaling_actual = solve_time / baseline_time if baseline_time else 0
            
            scaling_theoretical = 2**(n-10)
            ratio = scaling_actual / scaling_theoretical if scaling_theoretical > 0 else 0
            
            print(f"{n}\t{solve_time:.3f}s\t\t{scaling_actual:.1f}x\t\t{ratio:.3f}")
        
        # Extrapolation to N=20
        if baseline_time and len(successful_results) >= 3:
            # Fit exponential growth: time = baseline * 2^(a*(n-10))
            times = np.array([t for _, t in successful_results])
            ns = np.array([n for n, _ in successful_results])
            
            # Linear regression on log scale
            log_times = np.log(times / baseline_time)
            coeffs = np.polyfit(ns - 10, log_times, 1)
            growth_rate = coeffs[0]
            
            print(f"\nGrowth analysis:")
            print(f"Empirical growth rate: 2^({growth_rate:.3f}*Δn)")
            print(f"Theoretical growth rate: 2^(1.0*Δn)")
            
            # Extrapolate to N=20
            extrapolated_time = baseline_time * (2 ** (growth_rate * 10))
            print(f"\nExtrapolation to N=20:")
            print(f"Estimated time: {extrapolated_time:.1f} seconds ({extrapolated_time/3600:.2f} hours)")
            print(f"Memory estimate: ~{2**20 * 4 / (1024**2):.1f} MB per tensor")
    
    # Failed attempts
    failed_results = [(n, t) for n, t, success in results if not success]
    if failed_results:
        print(f"\nFailed/timed out attempts:")
        for n, time_taken in failed_results:
            print(f"N={n}: stopped after {time_taken:.1f}s")

if __name__ == "__main__":
    main()
