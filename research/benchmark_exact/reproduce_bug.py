#!/usr/bin/env python3
"""
Reproduce the exact bug that's happening in the benchmark.
This will run the exact same logic as in benchmark_advanced_cli.py
"""

import numpy as np
import statistics
from advanced_solver import AdvancedCVRPSolver
from enhanced_generator import EnhancedCVRPGenerator, InstanceType

def reproduce_benchmark_bug():
    """Reproduce the exact benchmark scenario"""
    
    # Use exact parameters from benchmark
    n = 5
    capacity = 30
    demand_range = [1, 10]
    coord_range = 100
    num_instances = 5  # Just first 5 instances
    
    # Initialize generator and solvers exactly as in benchmark
    gen = EnhancedCVRPGenerator(config={})
    
    exact_solver = AdvancedCVRPSolver(
        time_limit=min(180 / 2, 60.0),  # Same as benchmark
        enable_heuristics=False,  # EXACT ONLY
        verbose=False  # Less verbose for debugging
    )
    heuristic_solver = AdvancedCVRPSolver(
        time_limit=min(180 / 2, 5.0),  # Same as benchmark
        enable_heuristics=True,  # HEURISTIC ONLY
        verbose=False
    )
    
    print("="*80)
    print("REPRODUCING BENCHMARK BUG")
    print("="*80)
    print("Parameters: N=5, capacity=30, demand_range=[1, 10]")
    print("Exact timeout: 60.0s, Heuristic timeout: 5.0s")
    print()
    
    exact_costs = []
    heuristic_costs = []
    
    for i in range(num_instances):
        print(f"--- Instance {i+1} ---")
        
        # Generate instance with exact same seed as benchmark
        seed = 4242 + n * 1000 + i
        print(f"Seed: {seed}")
        
        instance = gen.generate_instance(
            num_customers=n,
            capacity=capacity,
            coord_range=coord_range,
            demand_range=demand_range,
            seed=seed,
            instance_type=InstanceType.RANDOM,
            apply_augmentation=False,
        )
        
        print(f"Instance: coords shape {instance['coords'].shape}, demands {instance['demands']}")
        
        # Solve with exact
        try:
            sol_exact = exact_solver.solve(instance)
            if np.isfinite(sol_exact.cost) and sol_exact.cost > 0:
                exact_cost_per_customer = sol_exact.cost / n
                exact_costs.append(exact_cost_per_customer)
                print(f"  Exact: cost={sol_exact.cost:.6f}, per_customer={exact_cost_per_customer:.6f}, route={sol_exact.route}")
            else:
                print(f"  Exact: FAILED (cost={sol_exact.cost})")
                continue
        except Exception as e:
            print(f"  Exact: EXCEPTION {e}")
            continue
            
        # Solve with heuristic
        try:
            sol_heuristic = heuristic_solver.solve(instance)
            if np.isfinite(sol_heuristic.cost) and sol_heuristic.cost > 0:
                heuristic_cost_per_customer = sol_heuristic.cost / n
                heuristic_costs.append(heuristic_cost_per_customer)
                print(f"  Heuristic: cost={sol_heuristic.cost:.6f}, per_customer={heuristic_cost_per_customer:.6f}, route={sol_heuristic.route}")
            else:
                print(f"  Heuristic: FAILED (cost={sol_heuristic.cost})")
                continue
        except Exception as e:
            print(f"  Heuristic: EXCEPTION {e}")
            continue
        
        # Compare
        if len(exact_costs) > 0 and len(heuristic_costs) > 0:
            exact_cpc = exact_costs[-1]
            heuristic_cpc = heuristic_costs[-1]
            print(f"  Difference: exact-heuristic = {exact_cpc - heuristic_cpc:.6f}")
            
            if exact_cpc > heuristic_cpc:
                print(f"  ❌ BUG: Exact ({exact_cpc:.6f}) > Heuristic ({heuristic_cpc:.6f})")
            else:
                print(f"  ✅ OK: Exact ({exact_cpc:.6f}) <= Heuristic ({heuristic_cpc:.6f})")
        
        print()
    
    # Compute statistics exactly as in benchmark
    print("="*80)
    print("FINAL STATISTICS (same as benchmark computation)")
    print("="*80)
    
    if len(exact_costs) >= 1:
        exact_avg = float(statistics.mean(exact_costs))
        exact_std = float(statistics.stdev(exact_costs)) if len(exact_costs) >= 2 else 0.0
        print(f"Exact: avg_cpc={exact_avg:.6f}, std={exact_std:.6f}")
    else:
        exact_avg = exact_std = float('nan')
        print("Exact: No valid solutions")
    
    if len(heuristic_costs) >= 1:
        heuristic_avg = float(statistics.mean(heuristic_costs))
        heuristic_std = float(statistics.stdev(heuristic_costs)) if len(heuristic_costs) >= 2 else 0.0
        print(f"Heuristic: avg_cpc={heuristic_avg:.6f}, std={heuristic_std:.6f}")
    else:
        heuristic_avg = heuristic_std = float('nan')
        print("Heuristic: No valid solutions")
    
    print(f"Difference: exact_avg - heuristic_avg = {exact_avg - heuristic_avg:.6f}")
    
    if exact_avg > heuristic_avg:
        print("❌ REPRODUCE BUG: Exact average is worse than heuristic average!")
    else:
        print("✅ No bug reproduced.")
    
    # Let's also check the raw data
    print("\nRaw data:")
    print(f"Exact costs per customer: {exact_costs}")
    print(f"Heuristic costs per customer: {heuristic_costs}")

if __name__ == "__main__":
    reproduce_benchmark_bug()
