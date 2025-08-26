#!/usr/bin/env python3
"""
Modified CVRP CPU Benchmark (Mac repo)

Solver names:
- exact_dp: Dynamic programming exact solver (N ≤ 8)
- ortools_greedy: OR-Tools greedy/exact solver (formerly exact_ortools_vrp) — exact regime for N ≤ 8
- ortools_gls: OR-Tools Guided Local Search (formerly heuristic_or) — 2s timeout

Configuration:
- exact_dp: runs up to N=8 (inclusive)
- ortools_greedy: runs in exact mode up to N=8, then as heuristic with short timeout
- ortools_gls: runs with 2s timeout for all N ≤ 20
"""

import sys
import os
sys.path.append('research/benchmark_exact')

# Import solvers with new naming
import solvers.exact_dp as exact_dp
import solvers.exact.ortools_greedy as ortools_greedy
import solvers.ortools_gls as ortools_gls

from enhanced_generator import EnhancedCVRPGenerator, InstanceType
import numpy as np
import time
import argparse
import csv
from datetime import datetime


def get_solver_timeout(solver_name: str, n: int) -> float:
    """Get appropriate timeout for each solver based on N"""
    if solver_name == "exact_dp":
        # Only runs for N ≤ 8
        if n > 8:
            return None  # Don't run
        # Give generous timeout for exact solution, but cap at 60s
        return min(60.0, 2.0 * (2 ** (n - 5)))

    elif solver_name == "ortools_greedy":
        # In exact regime (N ≤ 8), give more time (but not too long)
        if n <= 8:
            return 15.0  # 15s for exact regime
        else:
            return 5.0   # 5s for heuristic/greedy regime

    elif solver_name == "ortools_gls":
        # Always 2 seconds as requested
        return 2.0

    return 10.0  # Default fallback


def should_run_solver(solver_name: str, n: int) -> bool:
    """Determine if a solver should run for given N"""
    if solver_name == "exact_dp":
        return n <= 8
    elif solver_name in ["ortools_greedy", "ortools_gls"]:
        return n <= 20
    return False


def run_single_instance(solver_module, solver_name: str, instance, n: int, timeout: float):
    """Run a single solver on an instance with timeout"""
    start_time = time.time()
    try:
        solution = solver_module.solve(instance, time_limit=timeout, verbose=False)
        solve_time = time.time() - start_time
        if solution is None:
            return None, solve_time, True
        return solution, solve_time, False
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"  {solver_name} error: {str(e)[:80]}")
        return None, solve_time, False


def benchmark_instance(instance, n: int, seed: int):
    """Run all applicable solvers on a single instance"""
    results = {}
    solvers = {
        'exact_dp': exact_dp,
        'ortools_greedy': ortools_greedy,
        'ortools_gls': ortools_gls,
    }

    for solver_name, solver_module in solvers.items():
        if not should_run_solver(solver_name, n):
            print(f"  {solver_name}: SKIPPED (N={n} out of range)")
            continue
        timeout = get_solver_timeout(solver_name, n)
        solution, solve_time, timed_out = run_single_instance(
            solver_module, solver_name, instance, n, timeout
        )
        if solution:
            cpc = solution.cost / n
            print(f"  {solver_name}: cost={solution.cost:.4f}, CPC={cpc:.4f}, time={solve_time:.3f}s")
            results[solver_name] = {
                'cost': solution.cost,
                'cpc': cpc,
                'time': solve_time,
                'routes': solution.vehicle_routes,
            }
        elif timed_out:
            print(f"  {solver_name}: TIMEOUT after {solve_time:.3f}s")
            results[solver_name] = {'timeout': True, 'time': solve_time}
        else:
            print(f"  {solver_name}: FAILED")
            results[solver_name] = {'failed': True, 'time': solve_time}
    return results


def run_benchmark(n_values, num_instances, output_prefix="benchmark"):
    gen = EnhancedCVRPGenerator(config={})
    all_results = {}

    csv_filename = f"results/{output_prefix}_modified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['n', 'instance_id', 'seed', 'solver', 'cost', 'cpc', 'time', 'timeout', 'failed'])

        for n in n_values:
            print(f"\n{'='*60}")
            print(f"Testing N={n} with {num_instances} instances")
            print(f"{'='*60}")
            all_results[n] = []
            for i in range(num_instances):
                seed = 1000 * n + i
                print(f"\nInstance {i+1}/{num_instances} (seed={seed}):")
                instance = gen.generate_instance(
                    num_customers=n,
                    capacity=30,
                    coord_range=100,
                    demand_range=[1, 10],
                    seed=seed,
                    instance_type=InstanceType.RANDOM,
                    apply_augmentation=False,
                )
                results = benchmark_instance(instance, n, seed)
                all_results[n].append(results)
                for solver_name, result in results.items():
                    row = [n, i+1, seed, solver_name]
                    if 'cost' in result:
                        row.extend([result['cost'], result['cpc'], result['time'], False, False])
                    elif 'timeout' in result:
                        row.extend([None, None, result['time'], True, False])
                    else:
                        row.extend([None, None, result['time'], False, True])
                    csv_writer.writerow(row)

    print(f"\n✅ Results saved to {csv_filename}")
    return all_results, csv_filename


def print_summary(all_results):
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    for n in sorted(all_results.keys()):
        print(f"\n--- N={n} ---")
        solver_stats = {}
        for instance_results in all_results[n]:
            for solver_name, result in instance_results.items():
                if solver_name not in solver_stats:
                    solver_stats[solver_name] = {'costs': [], 'times': [], 'timeouts': 0, 'failures': 0}
                if 'cost' in result:
                    solver_stats[solver_name]['costs'].append(result['cost'])
                    solver_stats[solver_name]['times'].append(result['time'])
                elif 'timeout' in result:
                    solver_stats[solver_name]['timeouts'] += 1
                    solver_stats[solver_name]['times'].append(result['time'])
                else:
                    solver_stats[solver_name]['failures'] += 1
        for solver_name in ['exact_dp', 'ortools_greedy', 'ortools_gls']:
            if solver_name not in solver_stats:
                continue
            stats = solver_stats[solver_name]
            if stats['costs']:
                avg_cost = np.mean(stats['costs'])
                avg_cpc = avg_cost / n
                avg_time = np.mean(stats['times'])
                success_rate = len(stats['costs']) / len(all_results[n]) * 100
                print(f"{solver_name:15s}: CPC={avg_cpc:.4f}, time={avg_time:.3f}s, success={success_rate:.0f}%")
                if stats['timeouts'] > 0:
                    print(f"{'':15s}  ({stats['timeouts']} timeouts)")
            elif stats['timeouts'] > 0:
                print(f"{solver_name:15s}: {stats['timeouts']} timeouts")
            else:
                print(f"{solver_name:15s}: skipped or failed")


def main():
    parser = argparse.ArgumentParser(description='Modified CVRP Benchmark (N≤8 exact, GLS=2s)')
    parser.add_argument('--n-values', type=int, nargs='+',
                        default=[5, 6, 7, 8, 9, 12, 15, 20],
                        help='N values to test')
    parser.add_argument('--instances', type=int, default=5,
                        help='Number of instances per N')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer instances')

    args = parser.parse_args()

    if args.quick:
        n_values = [5, 8, 12, 20]
        num_instances = 2
    else:
        n_values = args.n_values
        num_instances = args.instances

    print("="*70)
    print("MODIFIED CVRP BENCHMARK")
    print("="*70)
    print("\nSolver Configuration:")
    print("- exact_dp: Runs for N ≤ 8 (exact DP)")
    print("- ortools_greedy: Runs for N ≤ 20 (exact for N≤8, greedy after)")
    print("- ortools_gls: Runs for N ≤ 20 with 2s timeout (GLS)")
    print(f"\nTesting N values: {n_values}")
    print(f"Instances per N: {num_instances}")

    all_results, csv_file = run_benchmark(n_values, num_instances)
    print_summary(all_results)
    print(f"\n✅ Complete! Results saved to {csv_file}")


if __name__ == "__main__":
    main()

