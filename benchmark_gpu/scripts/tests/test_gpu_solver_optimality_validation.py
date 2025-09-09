#!/usr/bin/env python3
"""
Test to verify the fixed GPU solver actually produces optimal solutions.
"""

import sys
sys.path.append('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')

from src.generator.generator import _generate_instance
from src.benchmarking.solvers.cpu.exact_dp import solve as cpu_exact_solve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_cvrp_solver_truly_optimal as gpu_exact_fixed
import numpy as np

def test_gpu_fixed_optimality():
    """Test if the fixed GPU solver is truly optimal."""
    
    print("Testing Fixed GPU Solver Optimality")
    print("=" * 50)
    
    discrepancies = []
    
    # Test on small instances that CPU can handle
    for i in range(20):
        print(f"\nTest {i+1}/20:")
        
        # Generate instance  
        instance = _generate_instance(
            num_customers=6,  # Start with N=6
            capacity=20,
            coord_range=100,
            demand_range=[1, 10],
            seed=42000 + i
        )
        
        try:
            # CPU exact (guaranteed optimal)
            cpu_solution = cpu_exact_solve(instance, verbose=False)
            cpu_cost = cpu_solution.cost
            
            # GPU fixed (should be optimal)
            gpu_solution = gpu_exact_fixed.solve(instance, verbose=False)
            gpu_cost = gpu_solution.cost
            
            print(f"  CPU cost: {cpu_cost:.6f}")
            print(f"  GPU cost: {gpu_cost:.6f}")
            
            if abs(cpu_cost - gpu_cost) > 1e-4:
                discrepancy = ((gpu_cost - cpu_cost) / cpu_cost) * 100
                print(f"  DISCREPANCY: {discrepancy:.3f}%")
                discrepancies.append((i, cpu_cost, gpu_cost, discrepancy))
            else:
                print(f"  ✓ OPTIMAL")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            discrepancies.append((i, "ERROR", "ERROR", "ERROR"))
    
    print("\n" + "=" * 50)
    if discrepancies:
        print(f"FAILED: {len(discrepancies)} discrepancies found!")
        for i, cpu, gpu, diff in discrepancies[:5]:  # Show first 5
            print(f"  Test {i+1}: CPU={cpu}, GPU={gpu}, Diff={diff}")
        print("\nThe fixed GPU solver is NOT optimal!")
        return False
    else:
        print("SUCCESS: All tests passed!")
        print("The fixed GPU solver produces optimal solutions.")
        return True

if __name__ == "__main__":
    is_optimal = test_gpu_fixed_optimality()
    
    if not is_optimal:
        print("\n" + "="*50)
        print("RECOMMENDATION: Use the CPU exact solver for guaranteed")
        print("optimal results, or fix the GPU implementation issues.")
