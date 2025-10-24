#!/usr/bin/env python3
"""
Quick timing test for GPU-DP solver with N=20
"""
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generator.generator import _generate_instance
from gpu_cvrp_solver_truly_optimal_fixed import GPUExactCVRPFixed

def time_single_instance(n_customers=20, capacity=30):
    """Time a single instance solve"""
    print(f"Timing GPU-DP solver for N={n_customers}")
    print(f"State space size: 2^{n_customers} = {2**n_customers:,} states")
    print(f"Memory estimate: ~{2**n_customers * 4 / (1024**2):.1f} MB per tensor")
    
    # Generate single instance
    instance = _generate_instance(
        num_customers=n_customers,  # Fixed parameter name
        capacity=capacity,
        coord_range=100,
        demand_range=[1, 10],
        seed=42
    )
    
    # Initialize solver
    solver = GPUExactCVRPFixed()
    
    print(f"\nStarting solve at {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    try:
        solutions = solver.solve_batch([instance], verbose=True)
        end_time = time.time()
        
        solve_time = end_time - start_time
        print(f"\nSolved in {solve_time:.2f} seconds")
        print(f"Solution cost: {solutions[0].cost:.6f}")
        
        return solve_time
        
    except Exception as e:
        end_time = time.time()
        solve_time = end_time - start_time
        print(f"\nFailed after {solve_time:.2f} seconds")
        print(f"Error: {e}")
        return solve_time

if __name__ == "__main__":
    time_single_instance()
