#!/usr/bin/env python3
"""
Advanced CVRP Solver Benchmark CLI for Large-Scale Studies
Supports exact and near-optimal algorithms for N up to 500+ customers.

EXACT ALGORITHMS (N ≤ 50):
- Dynamic Programming (N ≤ 12)
- Branch-and-Cut (N ≤ 20) 
- Enhanced OR-Tools (N ≤ 50)

NEAR-OPTIMAL ALGORITHMS (N ≤ 500+):
- HGS-CVRP (State-of-the-art heuristic)
- ALNS (Adaptive Large Neighborhood Search)
- Multi-start Local Search

Same interface as benchmark_cli.py but with:
- Support for much larger N (up to 500+)
- Multiple solver comparison modes
- Automatic algorithm selection
- Compatible CSV output for existing plotting code
"""

import argparse
import numpy as np
import time
import csv
import sys
import statistics
import logging
from pathlib import Path

# Local imports (self-contained)
from advanced_solver import AdvancedCVRPSolver, CVRPSolution
from enhanced_generator import EnhancedCVRPGenerator, InstanceType


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_exact_only_benchmark(n, num_instances, capacity, demand_range, timeout_seconds, verbose=False):
    """Run exact-only benchmark mode (exact algorithms only)."""
    print(f"N={n}: running up to {num_instances} instances (Exact-only mode)")
    
    # Fixed problem settings (matching original)
    coord_range = 100  # integer grid scaled to [0,1] in generator
    
    # Initialize generator (matching original)
    gen = EnhancedCVRPGenerator(config={})
    
    # Exact-only solver (no heuristics)
    solver = AdvancedCVRPSolver(
        time_limit=timeout_seconds, 
        enable_heuristics=False,  # Force exact algorithms only
        verbose=verbose
    )
    
    t0 = time.time()
    costs_per_customer = []
    solve_times = []
    algorithms_used = []
    solved = 0
    attempted = 0
    
    for i in range(num_instances):
        attempted = i + 1
        
        # Early termination if exceeding threshold
        if time.time() - t0 > timeout_seconds:
            print(f"Time cap reached for N={n} at instance {i}. Elapsed {time.time() - t0:.2f}s")
            break
            
        # Seed per instance for reproducibility
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
            sol = solver.solve(instance)
            
            if np.isfinite(sol.cost) and sol.cost > 0:
                costs_per_customer.append(sol.cost / max(1, n))
                solve_times.append(sol.solve_time)
                algorithms_used.append(sol.algorithm_used)
                solved += 1
                
                if verbose:
                    opt_str = "optimal" if sol.is_optimal else f"gap={sol.gap:.2%}"
                    print(f"  Instance {i+1}: {sol.algorithm_used}, cost={sol.cost:.4f}, time={sol.solve_time:.3f}s ({opt_str})")
                    
        except (TimeoutError, Exception) as e:
            if verbose:
                print(f"  Instance {i+1}: Failed - {e}")
            continue
    
    elapsed = time.time() - t0
    
    # Compute statistics
    if len(costs_per_customer) >= 1:
        avg_cpc = float(statistics.mean(costs_per_customer))
        avg_time = float(statistics.mean(solve_times))
        
        if len(costs_per_customer) >= 2:
            std_cpc = float(statistics.stdev(costs_per_customer))
            # Standard error of the mean for time
            std_time = float(statistics.stdev(solve_times) / (len(solve_times) ** 0.5))
        else:
            std_cpc = 0.0
            std_time = 0.0
    else:
        avg_cpc = float('nan')
        avg_time = float('nan')
        std_cpc = float('nan')
        std_time = float('nan')
    
    # Algorithm usage summary
    if algorithms_used:
        from collections import Counter
        algo_counts = Counter(algorithms_used)
        algo_summary = ", ".join([f"{algo}({count})" for algo, count in algo_counts.items()])
    else:
        algo_summary = "None"
    
    print(f"  → Solved: {solved}/{attempted}, Algorithms: {algo_summary}")
    
    # Return in exact-only format (compatible with plotting)
    return {
        'N': n,
        'time_exact': avg_time,
        'time_exact_std': std_time,
        'cpc_exact': avg_cpc,
        'std_exact': std_cpc,
        'solved': solved,
        'attempted': attempted,
        'algorithms': algo_summary
    }


def run_exact_vs_heuristic_benchmark(n, num_instances, capacity, demand_range, timeout_seconds, verbose=False):
    """Run exact vs heuristic comparison benchmark."""
    print(f"N={n}: running up to {num_instances} instances (Exact vs Heuristic)")
    
    coord_range = 100
    gen = EnhancedCVRPGenerator(config={})
    
    # Initialize solvers with clear separation
    exact_solver = AdvancedCVRPSolver(
        time_limit=min(timeout_seconds / 2, 60.0),  # Longer timeout for exact algorithms
        enable_heuristics=False,  # EXACT ONLY: DP, Gurobi, OR-Tools exact
        verbose=verbose
    )
    heuristic_solver = AdvancedCVRPSolver(
        time_limit=min(timeout_seconds / 2, 5.0),  # Shorter timeout for heuristics
        enable_heuristics=True,  # HEURISTIC ONLY: HGS-CVRP, ALNS
        verbose=verbose
    )
    
    exact_times = []
    exact_costs = []
    exact_algos = []
    heuristic_times = []
    heuristic_costs = []
    heuristic_algos = []
    
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
        
        # Solve with exact algorithm
        try:
            sol_exact = exact_solver.solve(instance)
            if np.isfinite(sol_exact.cost) and sol_exact.cost > 0:
                exact_times.append(sol_exact.solve_time)
                exact_costs.append(sol_exact.cost / n)
                exact_algos.append(sol_exact.algorithm_used)
                
                if verbose:
                    opt_str = "optimal" if sol_exact.is_optimal else f"gap={sol_exact.gap:.2%}"
                    print(f"  Instance {i+1} Exact: {sol_exact.algorithm_used}, cost={sol_exact.cost:.4f}, time={sol_exact.solve_time:.3f}s ({opt_str})")
                    
        except (TimeoutError, Exception) as e:
            if verbose:
                print(f"  Instance {i+1} Exact: Failed - {e}")
        
        # Solve with heuristic algorithm
        try:
            sol_heuristic = heuristic_solver.solve(instance)
            if np.isfinite(sol_heuristic.cost) and sol_heuristic.cost > 0:
                heuristic_times.append(sol_heuristic.solve_time)
                heuristic_costs.append(sol_heuristic.cost / n)
                heuristic_algos.append(sol_heuristic.algorithm_used)
                
                if verbose:
                    gap_str = f"gap~{sol_heuristic.gap:.1%}" if sol_heuristic.gap > 0 else "near-optimal"
                    print(f"  Instance {i+1} Heuristic: {sol_heuristic.algorithm_used}, cost={sol_heuristic.cost:.4f}, time={sol_heuristic.solve_time:.3f}s ({gap_str})")
                    
        except (TimeoutError, Exception) as e:
            if verbose:
                print(f"  Instance {i+1} Heuristic: Failed - {e}")
    
    # Calculate statistics
    def compute_stats(times, costs):
        if len(costs) >= 1:
            avg_time = float(statistics.mean(times)) if times else float('nan')
            avg_cost = float(statistics.mean(costs)) if costs else float('nan')
            
            if len(costs) >= 2:
                std_cost = float(statistics.stdev(costs))
                std_time = float(statistics.stdev(times) / (len(times) ** 0.5)) if times else float('nan')
            else:
                std_cost = 0.0
                std_time = 0.0
        else:
            avg_time = avg_cost = std_cost = std_time = float('nan')
        
        return avg_time, std_time, avg_cost, std_cost
    
    exact_time, exact_time_std, exact_cost, exact_std = compute_stats(exact_times, exact_costs)
    heur_time, heur_time_std, heur_cost, heur_std = compute_stats(heuristic_times, heuristic_costs)
    
    # Algorithm summaries
    from collections import Counter
    exact_algo_summary = ", ".join([f"{algo}({count})" for algo, count in Counter(exact_algos).items()]) if exact_algos else "None"
    heur_algo_summary = ", ".join([f"{algo}({count})" for algo, count in Counter(heuristic_algos).items()]) if heuristic_algos else "None"
    
    print(f"  → Exact solved: {len(exact_costs)}, Heuristic solved: {len(heuristic_costs)}")
    print(f"  → Exact algorithms: {exact_algo_summary}")
    print(f"  → Heuristic algorithms: {heur_algo_summary}")
    
    return {
        'N': n,
        'time_exact': exact_time,
        'time_exact_std': exact_time_std,
        'cpc_exact': exact_cost,
        'std_exact': exact_std,
        'time_heuristic': heur_time,
        'time_heuristic_std': heur_time_std,
        'cpc_heuristic': heur_cost,
        'std_heuristic': heur_std,
        'solved_exact': len(exact_costs),
        'solved_heuristic': len(heuristic_costs),
        'attempted': attempted,
        'exact_algorithms': exact_algo_summary,
        'heuristic_algorithms': heur_algo_summary
    }


def run_heuristic_only_benchmark(n, num_instances, capacity, demand_range, timeout_seconds, verbose=False):
    """Run heuristic-only benchmark mode for large N."""
    print(f"N={n}: running up to {num_instances} instances (Heuristic-only mode)")
    
    coord_range = 100
    gen = EnhancedCVRPGenerator(config={})
    
    # Heuristic-only solver
    solver = AdvancedCVRPSolver(
        time_limit=timeout_seconds,
        enable_heuristics=True,
        verbose=verbose
    )
    
    t0 = time.time()
    costs_per_customer = []
    solve_times = []
    algorithms_used = []
    gaps = []
    solved = 0
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
        
        try:
            sol = solver.solve(instance)
            
            if np.isfinite(sol.cost) and sol.cost > 0:
                costs_per_customer.append(sol.cost / max(1, n))
                solve_times.append(sol.solve_time)
                algorithms_used.append(sol.algorithm_used)
                gaps.append(sol.gap)
                solved += 1
                
                if verbose:
                    gap_str = f"gap~{sol.gap:.1%}" if sol.gap > 0 else "near-optimal"
                    print(f"  Instance {i+1}: {sol.algorithm_used}, cost={sol.cost:.4f}, time={sol.solve_time:.3f}s ({gap_str})")
                    
        except (TimeoutError, Exception) as e:
            if verbose:
                print(f"  Instance {i+1}: Failed - {e}")
            continue
    
    # Compute statistics
    if len(costs_per_customer) >= 1:
        avg_cpc = float(statistics.mean(costs_per_customer))
        avg_time = float(statistics.mean(solve_times))
        avg_gap = float(statistics.mean(gaps)) if gaps else 0.0
        
        if len(costs_per_customer) >= 2:
            std_cpc = float(statistics.stdev(costs_per_customer))
            std_time = float(statistics.stdev(solve_times) / (len(solve_times) ** 0.5))
        else:
            std_cpc = 0.0
            std_time = 0.0
    else:
        avg_cpc = avg_time = std_cpc = std_time = avg_gap = float('nan')
    
    # Algorithm usage summary
    if algorithms_used:
        from collections import Counter
        algo_counts = Counter(algorithms_used)
        algo_summary = ", ".join([f"{algo}({count})" for algo, count in algo_counts.items()])
    else:
        algo_summary = "None"
    
    print(f"  → Solved: {solved}/{attempted}, Average gap: {avg_gap:.2%}, Algorithms: {algo_summary}")
    
    return {
        'N': n,
        'time_heuristic': avg_time,
        'time_heuristic_std': std_time,
        'cpc_heuristic': avg_cpc,
        'std_heuristic': std_cpc,
        'avg_gap': avg_gap,
        'solved': solved,
        'attempted': attempted,
        'algorithms': algo_summary
    }


def main():
    parser = argparse.ArgumentParser(description='Advanced CVRP Solver Benchmark CLI for Large-Scale Studies')
    parser.add_argument('--instances', type=int, default=50, help='Number of instances per N (default: 50)')
    parser.add_argument('--timeout', type=float, default=120.0, help='Timeout per N in seconds (default: 120.0)')
    parser.add_argument('--n-start', type=int, default=5, help='Start N (default: 5)')
    parser.add_argument('--n-end', type=int, default=200, help='End N inclusive (default: 200)')
    parser.add_argument('--capacity', type=int, default=30, help='Vehicle capacity (default: 30)')
    parser.add_argument('--demand-min', type=int, default=1, help='Min demand (default: 1)')
    parser.add_argument('--demand-max', type=int, default=10, help='Max demand (default: 10)')
    parser.add_argument('--mode', choices=['exact-only', 'compare', 'heuristic-only'], default='compare',
                        help='Benchmark mode: exact-only, compare (exact vs heuristic), or heuristic-only')
    parser.add_argument('--output', type=str, default='benchmark_advanced_results.csv', help='Output CSV file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("="*70)
    print("ADVANCED CVRP SOLVER BENCHMARK - LARGE-SCALE STUDY")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Problem size: N = {args.n_start} to {args.n_end}")
    print(f"Instances per N: {args.instances}")
    print(f"Timeout per N: {args.timeout:.1f}s")
    print(f"Capacity: {args.capacity}, Demands: [{args.demand_min}, {args.demand_max}]")
    print(f"Output: {args.output}")
    print()
    
    # Check algorithm availability
    solver_test = AdvancedCVRPSolver(verbose=False)
    print("🔧 Available Algorithms:")
    print(f"   Dynamic Programming: ✅ (built-in)")
    print(f"   OR-Tools Enhanced: {'✅' if solver_test._has_ortools else '❌'}")
    print(f"   Gurobi Branch-Cut: {'✅' if solver_test._has_gurobi else '❌'}")  
    print(f"   HGS-CVRP (PyVRP): {'✅' if solver_test._has_pyvrp else '❌'}")
    print(f"   ALNS: ✅ (built-in)")
    print()
    
    # Generate N values (intelligent spacing)
    if args.n_end <= 20:
        n_values = list(range(args.n_start, args.n_end + 1))  # Dense for small N
    elif args.n_end <= 50:
        n_values = list(range(args.n_start, 21)) + list(range(25, args.n_end + 1, 5))  # Mixed spacing
    else:
        # Sparse spacing for large N
        n_values = (list(range(args.n_start, 21)) +  # Dense small
                   list(range(25, 51, 5)) +          # Medium spacing
                   list(range(60, args.n_end + 1, 10)))  # Sparse large
    
    # Remove duplicates and sort
    n_values = sorted(set(n_values))
    print(f"📊 Problem sizes to test: {n_values}")
    print()
    
    demand_range = [args.demand_min, args.demand_max]
    results = []
    
    total_start_time = time.time()
    
    for n in n_values:
        n_start_time = time.time()
        print(f"{'='*50}")
        print(f"Testing N = {n} customers")
        print(f"{'='*50}")
        
        try:
            if args.mode == 'exact-only':
                result = run_exact_only_benchmark(
                    n, args.instances, args.capacity, demand_range, args.timeout, args.verbose
                )
            elif args.mode == 'compare':
                result = run_exact_vs_heuristic_benchmark(
                    n, args.instances, args.capacity, demand_range, args.timeout, args.verbose
                )
            elif args.mode == 'heuristic-only':
                result = run_heuristic_only_benchmark(
                    n, args.instances, args.capacity, demand_range, args.timeout, args.verbose
                )
            
            results.append(result)
            
            n_elapsed = time.time() - n_start_time
            print(f"Completed N={n} in {n_elapsed:.1f}s")
            print()
            
            # Write incremental results
            write_results_csv(results, args.output, args.mode)
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error processing N={n}: {e}")
            continue
    
    total_elapsed = time.time() - total_start_time
    
    print("="*70)
    print("🏁 BENCHMARK COMPLETED")
    print("="*70)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Problem sizes completed: {len(results)}/{len(n_values)}")
    print(f"Results saved to: {args.output}")
    
    if results:
        print(f"\n📊 Summary:")
        total_instances = sum(r.get('attempted', 0) for r in results)
        if args.mode == 'exact-only':
            total_solved = sum(r.get('solved', 0) for r in results)
        elif args.mode == 'compare':
            total_solved_exact = sum(r.get('solved_exact', 0) for r in results)
            total_solved_heuristic = sum(r.get('solved_heuristic', 0) for r in results)
            print(f"   Exact solutions: {total_solved_exact}")
            print(f"   Heuristic solutions: {total_solved_heuristic}")
        else:  # heuristic-only
            total_solved = sum(r.get('solved', 0) for r in results)
            print(f"   Total solved: {total_solved}/{total_instances}")
        
        print(f"\n🎯 To visualize results:")
        print(f"   python plot_benchmark.py {args.output} --title \"Advanced CVRP Benchmark\"")
    
    print("\n✅ Done!")


def write_results_csv(results, filename, mode):
    """Write results to CSV file with mode-appropriate columns"""
    if not results:
        return
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['N']
        
        if mode == 'exact-only':
            fieldnames.extend(['time_exact', 'time_exact_std', 'cpc_exact', 'std_exact', 
                             'solved', 'attempted', 'algorithms'])
        elif mode == 'compare':
            fieldnames.extend(['time_exact', 'time_exact_std', 'cpc_exact', 'std_exact',
                             'time_heuristic', 'time_heuristic_std', 'cpc_heuristic', 'std_heuristic',
                             'solved_exact', 'solved_heuristic', 'attempted', 
                             'exact_algorithms', 'heuristic_algorithms'])
        else:  # heuristic-only
            fieldnames.extend(['time_heuristic', 'time_heuristic_std', 'cpc_heuristic', 'std_heuristic',
                             'avg_gap', 'solved', 'attempted', 'algorithms'])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Ensure all fields exist
            row = {field: result.get(field, '') for field in fieldnames}
            writer.writerow(row)


if __name__ == '__main__':
    main()
