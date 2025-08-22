#!/usr/bin/env python3
"""
Fast CVRP Solver Benchmark CLI
Uses EXACTLY the same approach as the original exact baseline for speed.
"""

import argparse
import numpy as np
import time
import csv
import sys
import statistics
from pathlib import Path

# Local imports (self-contained)
from exact_solver import ExactCVRPSolver
from enhanced_generator import EnhancedCVRPGenerator, InstanceType

def run_dp_only_benchmark(n, num_instances, capacity, demand_range, timeout_seconds):
    """Run DP-only benchmark but output in comparison format."""
    print(f"N={n}: running up to {num_instances} instances (DP-only mode)")
    
    # Fixed problem settings (matching original)
    coord_range = 100  # integer grid scaled to [0,1] in generator
    
    # Initialize generator (matching original)
    gen = EnhancedCVRPGenerator(config={})
    
    # DP-only solver (matching original)
    solver = ExactCVRPSolver(time_limit=timeout_seconds, enable_or_tools=False, enable_gurobi=False, verbose=False)
    
    t0 = time.time()
    costs_per_customer = []
    solve_times = []
    solved = 0
    attempted = 0
    
    for i in range(num_instances):
        attempted = i + 1
        
        # Early termination if exceeding threshold (matching original)
        if time.time() - t0 > timeout_seconds:
            print(f"Time cap reached for N={n} at instance {i}. Elapsed {time.time() - t0:.2f}s")
            break
            
        # Seed per instance for reproducibility (matching original)
        seed = 4242 + n * 1000 + i
        instance = gen.generate_instance(
            num_customers=n,
            capacity=capacity,
            coord_range=coord_range,
            demand_range=demand_range,
            seed=seed,
            instance_type=InstanceType.RANDOM,
            apply_augmentation=False,
        )
        
        try:
            start_time = time.time()
            sol = solver.solve(instance)  # Use main solve method (like original)
            solve_time = time.time() - start_time
            
            if np.isfinite(sol.cost):
                costs_per_customer.append(sol.cost / max(1, n))
                solve_times.append(solve_time)
                solved += 1
        except (TimeoutError, Exception):
            continue  # Skip failed instances
    
    elapsed = time.time() - t0
    
    # Compute statistics only on successfully solved instances
    if len(costs_per_customer) >= 1:
        avg_cpc = float(statistics.mean(costs_per_customer))
        avg_time = float(statistics.mean(solve_times))
    else:
        avg_cpc = float('nan')
        avg_time = float('nan')
        
    if len(costs_per_customer) >= 2:
        std_cpc = float(statistics.stdev(costs_per_customer))
        # Standard error of the mean for time
        std_time = float(statistics.stdev(solve_times) / (len(solve_times) ** 0.5))
    else:
        std_cpc = float('nan')
        std_time = float('nan')
    
    # Return in comparison format (OR-Tools columns as NaN for DP-only mode)
    return {
        'N': n,
        'time_or': float('nan'),      # No OR-Tools data
        'time_or_std': float('nan'),  # No OR-Tools time std
        'cpc_or': float('nan'),       # No OR-Tools data
        'std_or': float('nan'),       # No OR-Tools cost std
        'time_dp': avg_time,          # DP time
        'time_dp_std': std_time,      # DP time std
        'cpc_dp': avg_cpc,            # DP cost per customer
        'std_dp': std_cpc,            # DP cost std
        'solved': solved,
        'attempted': attempted
    }

def run_comparison_benchmark(n, num_instances, capacity, demand_range, timeout_seconds):
    """Run DP vs OR-Tools comparison benchmark."""
    print(f"N={n}: running up to {num_instances} instances (DP vs OR-Tools)")
    
    coord_range = 100
    gen = EnhancedCVRPGenerator(config={})
    
    # Initialize both solvers
    solver_or = ExactCVRPSolver(time_limit=10.0, enable_or_tools=True, enable_gurobi=False, verbose=False)
    solver_dp = ExactCVRPSolver(time_limit=10.0, enable_or_tools=False, enable_gurobi=False, verbose=False)
    
    or_times = []
    or_costs = []
    dp_times = []  
    dp_costs = []
    
    t0 = time.time()
    attempted = 0
    
    for i in range(num_instances):
        attempted = i + 1
        
        if time.time() - t0 > timeout_seconds:
            print(f"Time cap reached for N={n} at instance {i}. Elapsed {time.time() - t0:.2f}s")
            break
            
        seed = 4242 + n * 1000 + i
        instance = gen.generate_instance(
            num_customers=n,
            capacity=capacity,
            coord_range=coord_range,
            demand_range=demand_range,
            seed=seed,
            instance_type=InstanceType.RANDOM,
            apply_augmentation=False,
        )
        
        # Solve with OR-Tools (pure OR-Tools, no fallback)
        try:
            start_time = time.time()
            solution_or = solver_or._solve_ortools_advanced(instance)
            solve_time_or = time.time() - start_time
            or_times.append(solve_time_or)
            or_costs.append(solution_or.cost / n)  # Cost per customer
        except (TimeoutError, Exception):
            # OR-Tools failed/timeout - skip this instance (no fallback)
            pass
        
        # Solve with DP (pure DP, no fallback)
        try:
            start_time = time.time()
            solution_dp = solver_dp._solve_dp_bitmasking(instance)
            solve_time_dp = time.time() - start_time  
            dp_times.append(solve_time_dp)
            dp_costs.append(solution_dp.cost / n)  # Cost per customer
        except (TimeoutError, Exception):
            # DP failed/timeout - skip this instance (no fallback)
            pass
    
    # Calculate statistics only on successfully solved instances
    valid_or_costs = [c for c in or_costs if c != float('inf')]
    valid_dp_costs = [c for c in dp_costs if c != float('inf')]
    
    return {
        'N': n,
        'time_or': round(statistics.mean(or_times) if or_times else float('nan'), 6),
        'time_or_std': round(statistics.stdev(or_times) / (len(or_times) ** 0.5) if len(or_times) > 1 else float('nan'), 6),
        'cpc_or': round(statistics.mean(valid_or_costs) if valid_or_costs else float('nan'), 6),
        'std_or': round(statistics.stdev(valid_or_costs) if len(valid_or_costs) > 1 else float('nan'), 6),
        'time_dp': round(statistics.mean(dp_times) if dp_times else float('nan'), 6),
        'time_dp_std': round(statistics.stdev(dp_times) / (len(dp_times) ** 0.5) if len(dp_times) > 1 else float('nan'), 6),
        'cpc_dp': round(statistics.mean(valid_dp_costs) if valid_dp_costs else float('nan'), 6),
        'std_dp': round(statistics.stdev(valid_dp_costs) if len(valid_dp_costs) > 1 else float('nan'), 6),
        'solved_or': len(valid_or_costs),
        'solved_dp': len(valid_dp_costs),
        'attempted': attempted
    }

def main():
    parser = argparse.ArgumentParser(description='Fast CVRP Solver Benchmark CLI')
    parser.add_argument('--instances', type=int, default=100, help='Number of instances per N (default: 100)')
    parser.add_argument('--timeout', type=float, default=60.0, help='Timeout per N in seconds (default: 60.0)')
    parser.add_argument('--n-start', type=int, default=5, help='Start N (default: 5)')
    parser.add_argument('--n-end', type=int, default=50, help='End N inclusive (default: 50)')
    parser.add_argument('--capacity', type=int, default=30, help='Vehicle capacity (default: 30)')
    parser.add_argument('--demand-min', type=int, default=1, help='Min demand (default: 1)')
    parser.add_argument('--demand-max', type=int, default=10, help='Max demand (default: 10)')
    parser.add_argument('--mode', choices=['dp-only', 'compare'], default='compare', 
                        help='Benchmark mode: dp-only (fast, like original) or compare (DP vs OR-Tools)')
    parser.add_argument('--output', type=str, default='benchmark_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FAST CVRP EXACT SOLVER BENCHMARK")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Problem size: N = {args.n_start} to {args.n_end}")
    print(f"Instances per N: {args.instances}")
    print(f"Vehicle capacity: {args.capacity}")
    print(f"Demand range: [{args.demand_min}, {args.demand_max}]")
    print(f"Timeout per N: {args.timeout}s")
    print(f"Output file: {args.output}")
    print()
    
    demand_range = [args.demand_min, args.demand_max]
    
    # Initialize CSV file with header (always use comparison format with time std)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['N', 'time_or', 'time_or_std', 'cpc_or', 'std_or', 'time_dp', 'time_dp_std', 'cpc_dp', 'std_dp'])
        writer.writeheader()
    
    rows_written = 0
    
    # Run benchmark for each N
    for n in range(args.n_start, args.n_end + 1):
        if args.mode == 'dp-only':
            result = run_dp_only_benchmark(n, args.instances, args.capacity, demand_range, args.timeout)
            
            # Write result immediately (using comparison format with time std)
            with open(args.output, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['N', 'time_or', 'time_or_std', 'cpc_or', 'std_or', 'time_dp', 'time_dp_std', 'cpc_dp', 'std_dp'])
                filtered_result = {k: v for k, v in result.items() if k in ['N', 'time_or', 'time_or_std', 'cpc_or', 'std_or', 'time_dp', 'time_dp_std', 'cpc_dp', 'std_dp']}
                writer.writerow(filtered_result)
            rows_written += 1
            
            print(f"Done N={n}: elapsed={result['time_dp']:.2f}s, solved={result['solved']} / {result['attempted']}, avg_cpc={result['cpc_dp'] if result['cpc_dp']==result['cpc_dp'] else 'nan':.4f}")
            
            # Stop once a full-N exceeds threshold (matching original)
            if result['time_dp'] > args.timeout:
                print(f"Stopping after N={n} (elapsed {result['time_dp']:.2f}s > {args.timeout:.0f}s).")
                break
                
        else:  # compare mode
            result = run_comparison_benchmark(n, args.instances, args.capacity, demand_range, args.timeout)
            
            # Write result immediately
            with open(args.output, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['N', 'time_or', 'time_or_std', 'cpc_or', 'std_or', 'time_dp', 'time_dp_std', 'cpc_dp', 'std_dp'])
                filtered_result = {k: v for k, v in result.items() if k in ['N', 'time_or', 'time_or_std', 'cpc_or', 'std_or', 'time_dp', 'time_dp_std', 'cpc_dp', 'std_dp']}
                writer.writerow(filtered_result)
            rows_written += 1
            
            print(f"Done N={n}: OR={result['time_or']:.4f}s ({result['solved_or']} solved), DP={result['time_dp']:.4f}s ({result['solved_dp']} solved) / {result['attempted']} attempted")
            
            # Stop if getting too slow
            if result['time_or'] > args.timeout / 5 or result['time_dp'] > args.timeout / 5:
                print(f"Stopping after N={n} (solvers getting slow)")
                break
    
    print(f"\n✅ Survey complete. Wrote {rows_written} rows to {args.output}")

if __name__ == '__main__':
    main()
