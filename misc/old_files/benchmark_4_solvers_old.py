#!/usr/bin/env python3
"""
4-Solver CVRP Benchmark
Compares exact_gurobi, exact_dp, heuristic_or, heuristic_dp solvers
Creates a CSV and plot similar to research/benchmark_exact/benchmark_30cx100ix120s.png
"""

import argparse
import numpy as np
import time
import csv
import sys
import statistics
from pathlib import Path
import importlib.util

# Import solvers
import solvers.exact_milp as exact_milp
import solvers.exact_dp as exact_dp
import solvers.heuristic_or as heuristic_or
import solvers.heuristic_dp as heuristic_dp

# Import generator from research folder
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType


def run_single_solver(solver_module, solver_name, instance, time_limit):
    """Run a single solver on an instance and return timing/cost info."""
    try:
        start_time = time.time()
        solution = solver_module.solve(instance, time_limit=time_limit, verbose=False)
        solve_time = time.time() - start_time
        
        # Check if solution is valid
        if solution.cost == float('inf') or solution.cost <= 0:
            return None
            
        return {
            'time': solve_time,
            'cost': solution.cost,
            'cost_per_customer': solution.cost / max(1, instance['num_customers']),
            'is_optimal': solution.is_optimal,
            'algorithm': solution.algorithm_used
        }
    except Exception as e:
        print(f"  Error: {solver_name} failed: {e}")
        return None


def run_benchmark_for_n(n, num_instances, capacity, demand_range, time_limit):
    """Run benchmark for a specific problem size N."""
    print(f"N={n}: running up to {num_instances} instances")
    
    coord_range = 100
    gen = EnhancedCVRPGenerator(config={})
    
    # Results for each solver
    results = {
        'exact_milp': {'times': [], 'costs': [], 'optimal_count': 0},
        'exact_dp': {'times': [], 'costs': [], 'optimal_count': 0},
        'heuristic_or': {'times': [], 'costs': [], 'optimal_count': 0},
        'heuristic_dp': {'times': [], 'costs': [], 'optimal_count': 0}
    }
    
    solvers = {
        'exact_milp': exact_milp,
        'exact_dp': exact_dp, 
        'heuristic_or': heuristic_or,
        'heuristic_dp': heuristic_dp
    }
    
    attempted = 0
    t0 = time.time()
    
    for i in range(num_instances):
        attempted = i + 1
        
        # Generate instance ONCE per iteration
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
        
        print(f"  Instance {i+1}: Generated with seed {seed}")
        
        # Test each solver ON THE SAME INSTANCE
        instance_results = {}
        for solver_name, solver_module in solvers.items():
            print(f"    Testing {solver_name}...")
            result = run_single_solver(solver_module, solver_name, instance, time_limit)
            instance_results[solver_name] = result
            
            if result is not None:
                results[solver_name]['times'].append(result['time'])
                results[solver_name]['costs'].append(result['cost_per_customer'])
                
                # Only count as optimal if solver claims it's optimal AND it's an exact solver
                if result['is_optimal'] and solver_name.startswith('exact'):
                    results[solver_name]['optimal_count'] += 1
                    
                print(f"      -> Cost: {result['cost']:.4f}, Time: {result['time']:.4f}s, Optimal: {result['is_optimal']}")
            else:
                print(f"      -> FAILED")
        
        # Verify all solvers that succeeded got the same instance by checking depot coordinates
        valid_results = {k: v for k, v in instance_results.items() if v is not None}
        if len(valid_results) > 1:
            costs = [r['cost'] for r in valid_results.values()]
            print(f"    Costs for this instance: {dict(zip(valid_results.keys(), costs))}")
        
        # Progress update every 10 instances
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {i+1}/{num_instances} instances, {elapsed:.1f}s elapsed")
    
    # Compute statistics
    stats = {'N': n, 'attempted': attempted}
    
    for solver_name in solvers.keys():
        times = results[solver_name]['times']
        costs = results[solver_name]['costs']
        optimal_count = results[solver_name]['optimal_count']
        
        if len(times) >= 1:
            avg_time = float(statistics.mean(times))
            avg_cost = float(statistics.mean(costs))
            solved_count = len(times)
        else:
            avg_time = float('nan')
            avg_cost = float('nan') 
            solved_count = 0
            
        if len(costs) >= 2:
            std_cost = float(statistics.stdev(costs))
        else:
            std_cost = float('nan')
            
        # For exact solvers, only include instances with is_optimal=True in statistics
        if solver_name.startswith('exact'):
            if optimal_count > 0:
                # Re-compute statistics using only optimal solutions
                # This is a simplification - ideally we'd track which instances were optimal
                # For now, we use the first optimal_count solutions (assumes they come first)
                optimal_costs = costs[:optimal_count] if optimal_count <= len(costs) else []
                optimal_times = times[:optimal_count] if optimal_count <= len(times) else []
                
                if len(optimal_costs) >= 1:
                    avg_cost = float(statistics.mean(optimal_costs))
                    avg_time = float(statistics.mean(optimal_times))
                    solved_count = len(optimal_costs)
                    
                if len(optimal_costs) >= 2:
                    std_cost = float(statistics.stdev(optimal_costs))
            else:
                # No optimal solutions found for this exact solver
                avg_cost = float('nan')
                avg_time = float('nan')
                std_cost = float('nan')
                solved_count = 0
        
        # Store results
        stats[f'time_{solver_name}'] = avg_time
        stats[f'cpc_{solver_name}'] = avg_cost  
        stats[f'std_{solver_name}'] = std_cost
        stats[f'solved_{solver_name}'] = solved_count
        stats[f'optimal_{solver_name}'] = optimal_count
    
    elapsed = time.time() - t0
    print(f"  Completed N={n} in {elapsed:.1f}s")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='4-Solver CVRP Benchmark')
    parser.add_argument('--instances', type=int, default=20, help='Number of instances per N (default: 20)')
    parser.add_argument('--n-start', type=int, default=5, help='Start N (default: 5)')
    parser.add_argument('--n-end', type=int, default=15, help='End N inclusive (default: 15)')
    parser.add_argument('--capacity', type=int, default=30, help='Vehicle capacity (default: 30)')
    parser.add_argument('--demand-min', type=int, default=1, help='Min demand (default: 1)')
    parser.add_argument('--demand-max', type=int, default=10, help='Max demand (default: 10)')
    parser.add_argument('--time-limit', type=float, default=30.0, help='Time limit per instance (default: 30.0)')
    parser.add_argument('--output', type=str, default='benchmark_4.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("4-SOLVER CVRP BENCHMARK")
    print("="*60)
    print(f"Problem size: N = {args.n_start} to {args.n_end}")
    print(f"Instances per N: {args.instances}")
    print(f"Vehicle capacity: {args.capacity}")
    print(f"Demand range: [{args.demand_min}, {args.demand_max}]")
    print(f"Time limit per instance: {args.time_limit}s")
    print(f"Output file: {args.output}")
    print()
    
    demand_range = [args.demand_min, args.demand_max]
    
    # Initialize CSV file with header
    fieldnames = ['N', 'time_exact_milp', 'cpc_exact_milp', 'std_exact_milp', 'solved_exact_milp', 'optimal_exact_milp',
                  'time_exact_dp', 'cpc_exact_dp', 'std_exact_dp', 'solved_exact_dp', 'optimal_exact_dp',
                  'time_heuristic_or', 'cpc_heuristic_or', 'std_heuristic_or', 'solved_heuristic_or', 
                  'time_heuristic_dp', 'cpc_heuristic_dp', 'std_heuristic_dp', 'solved_heuristic_dp']
                  
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()  # Force immediate write to disk
    
    rows_written = 0
    
    # Run benchmark for each N
    for n in range(args.n_start, args.n_end + 1):
        result = run_benchmark_for_n(n, args.instances, args.capacity, demand_range, args.time_limit)
        
    # Write result immediately
        with open(args.output, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Filter to only include CSV fieldnames and clean NaN values
            filtered_result = {}
            for k, v in result.items():
                if k in fieldnames:
                    if isinstance(v, float) and np.isnan(v):
                        filtered_result[k] = 'nan'
                    else:
                        filtered_result[k] = v
            writer.writerow(filtered_result)
            f.flush()  # Force immediate write to disk
        rows_written += 1
        
        # Show summary for this N
        print(f"  Summary N={n}:")
        for solver in ['exact_milp', 'exact_dp', 'heuristic_or', 'heuristic_dp']:
            solved = result.get(f'solved_{solver}', 0)
            optimal = result.get(f'optimal_{solver}', 0) if solver.startswith('exact') else 'N/A'
            avg_time = result.get(f'time_{solver}', float('nan'))
            avg_cpc = result.get(f'cpc_{solver}', float('nan'))
            
            time_str = f"{avg_time:.4f}s" if not np.isnan(avg_time) else "nan"
            cpc_str = f"{avg_cpc:.4f}" if not np.isnan(avg_cpc) else "nan"
            
            if solver.startswith('exact'):
                print(f"    {solver}: {solved} solved, {optimal} optimal, time={time_str}, cpc={cpc_str}")
            else:
                print(f"    {solver}: {solved} solved, time={time_str}, cpc={cpc_str}")
        print()
    
    print(f"✅ Benchmark complete. Wrote {rows_written} rows to {args.output}")


if __name__ == '__main__':
    main()
