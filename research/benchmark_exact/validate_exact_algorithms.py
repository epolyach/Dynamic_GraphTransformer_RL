#!/usr/bin/env python3
"""
Validate our exact algorithms against benchmark instances with known optimal solutions.
This will help us identify bugs in our exact solver implementations.
"""

import numpy as np
import os
import sys
from typing import Dict, List
from download_benchmarks import parse_vrp_instance, convert_to_our_format
from advanced_solver import AdvancedCVRPSolver

def load_benchmark_instances() -> List[Dict]:
    """Load all downloaded benchmark instances"""
    
    # Benchmark instances with known optimal solutions
    benchmark_data = [
        {'name': 'A-n32-k5', 'optimal_cost': 784, 'description': '32 customers, 5 vehicles'},
        {'name': 'A-n33-k5', 'optimal_cost': 661, 'description': '33 customers, 5 vehicles'},  
        {'name': 'A-n33-k6', 'optimal_cost': 742, 'description': '33 customers, 6 vehicles'},
        {'name': 'A-n37-k5', 'optimal_cost': 669, 'description': '37 customers, 5 vehicles'},
    ]
    
    instances = []
    
    if not os.path.exists('benchmarks'):
        print("❌ Benchmarks directory not found. Run download_benchmarks.py first.")
        return []
    
    for bench in benchmark_data:
        filename = f"benchmarks/{bench['name']}.vrp"
        if os.path.exists(filename):
            # Parse instance
            vrp_data = parse_vrp_instance(filename)
            if vrp_data:
                converted = convert_to_our_format(vrp_data)
                bench['data'] = converted
                instances.append(bench)
                print(f"✅ Loaded {bench['name']}: {converted['num_customers']} customers")
            else:
                print(f"❌ Failed to parse {bench['name']}")
        else:
            print(f"❌ File not found: {filename}")
    
    return instances

def validate_exact_algorithm(instance_data: Dict, algorithm_name: str, solver_config: Dict) -> Dict:
    """Validate a single exact algorithm on one instance"""
    
    instance = instance_data['data']
    known_optimal = instance_data['optimal_cost']
    
    print(f"  Testing {algorithm_name}...")
    
    try:
        # Create solver
        solver = AdvancedCVRPSolver(**solver_config)
        
        # Solve
        solution = solver.solve(instance)
        
        if solution and np.isfinite(solution.cost) and solution.cost > 0:
            # Calculate gap from known optimal
            gap = (solution.cost - known_optimal) / known_optimal * 100
            
            result = {
                'algorithm': algorithm_name,
                'found_cost': solution.cost,
                'known_optimal': known_optimal,
                'gap_percent': gap,
                'solve_time': solution.solve_time,
                'algorithm_used': solution.algorithm_used,
                'is_optimal_claimed': solution.is_optimal,
                'status': 'success'
            }
            
            # Determine if this is acceptable for an "exact" algorithm
            if abs(gap) < 0.01:  # Within 0.01%
                result['validation'] = '✅ OPTIMAL'
            elif abs(gap) < 1.0:   # Within 1%
                result['validation'] = '⚠️  NEAR-OPTIMAL' 
            else:
                result['validation'] = '❌ SUBOPTIMAL'
            
            print(f"    {result['validation']} Cost: {solution.cost:.2f} (optimal: {known_optimal:.2f}, gap: {gap:.2f}%)")
            print(f"    Time: {solution.solve_time:.3f}s, Algorithm: {solution.algorithm_used}")
            
            return result
        else:
            print(f"    ❌ FAILED - Invalid solution")
            return {
                'algorithm': algorithm_name,
                'status': 'failed',
                'validation': '❌ FAILED',
                'error': 'Invalid solution'
            }
            
    except Exception as e:
        print(f"    ❌ EXCEPTION: {e}")
        return {
            'algorithm': algorithm_name,
            'status': 'exception',
            'validation': '❌ EXCEPTION',
            'error': str(e)
        }

def run_validation():
    """Run complete validation of exact algorithms"""
    
    print("="*80)
    print("VALIDATING EXACT ALGORITHMS AGAINST BENCHMARK INSTANCES")
    print("="*80)
    print()
    
    # Load benchmark instances
    instances = load_benchmark_instances()
    if not instances:
        print("❌ No benchmark instances available.")
        return
    
    print(f"📊 Loaded {len(instances)} benchmark instances")
    print()
    
    # Define algorithm configurations to test
    algorithms = [
        {
            'name': 'Exact (No Heuristics)',
            'config': {
                'time_limit': 300.0,  # 5 minutes timeout
                'enable_heuristics': False,  # Pure exact algorithms
                'verbose': False
            }
        },
        {
            'name': 'Exact (Long Timeout)', 
            'config': {
                'time_limit': 600.0,  # 10 minutes timeout
                'enable_heuristics': False,
                'verbose': False
            }
        }
    ]
    
    # Results storage
    all_results = []
    
    # Test each instance
    for instance in instances:
        print(f"🎯 Testing Instance: {instance['name']} ({instance['description']})")
        print(f"   Known optimal cost: {instance['optimal_cost']}")
        print()
        
        instance_results = []
        
        # Test each algorithm
        for algo in algorithms:
            result = validate_exact_algorithm(instance, algo['name'], algo['config'])
            result['instance_name'] = instance['name']
            instance_results.append(result)
            all_results.append(result)
        
        print()
    
    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    # Group results by algorithm
    algo_summary = {}
    for result in all_results:
        algo = result.get('algorithm', 'Unknown')
        if algo not in algo_summary:
            algo_summary[algo] = {'optimal': 0, 'near_optimal': 0, 'suboptimal': 0, 'failed': 0, 'total': 0}
        
        algo_summary[algo]['total'] += 1
        
        validation = result.get('validation', '❌ UNKNOWN')
        if '✅ OPTIMAL' in validation:
            algo_summary[algo]['optimal'] += 1
        elif '⚠️  NEAR-OPTIMAL' in validation:
            algo_summary[algo]['near_optimal'] += 1
        elif '❌ SUBOPTIMAL' in validation:
            algo_summary[algo]['suboptimal'] += 1
        else:
            algo_summary[algo]['failed'] += 1
    
    for algo, stats in algo_summary.items():
        print(f"\n📊 {algo}:")
        print(f"   ✅ Optimal: {stats['optimal']}/{stats['total']} ({stats['optimal']/stats['total']*100:.1f}%)")
        print(f"   ⚠️  Near-optimal: {stats['near_optimal']}/{stats['total']} ({stats['near_optimal']/stats['total']*100:.1f}%)")
        print(f"   ❌ Suboptimal: {stats['suboptimal']}/{stats['total']} ({stats['suboptimal']/stats['total']*100:.1f}%)")
        print(f"   💀 Failed: {stats['failed']}/{stats['total']} ({stats['failed']/stats['total']*100:.1f}%)")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print(f"{'Instance':<12} {'Algorithm':<20} {'Found':<8} {'Optimal':<8} {'Gap%':<8} {'Time':<8} {'Status'}")
    print("-" * 80)
    
    for result in all_results:
        if result.get('status') == 'success':
            print(f"{result.get('instance_name', ''):<12} "
                  f"{result.get('algorithm', ''):<20} "
                  f"{result.get('found_cost', 0):<8.1f} "
                  f"{result.get('known_optimal', 0):<8.1f} "
                  f"{result.get('gap_percent', 0):<8.2f} "
                  f"{result.get('solve_time', 0):<8.2f} "
                  f"{result.get('validation', '')}")
        else:
            print(f"{result.get('instance_name', ''):<12} "
                  f"{result.get('algorithm', ''):<20} "
                  f"{'FAILED':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} "
                  f"{result.get('validation', '')}")
    
    print("\n" + "="*80)
    
    # Check if exact algorithms are actually exact
    exact_results = [r for r in all_results if r.get('status') == 'success']
    if exact_results:
        optimal_count = len([r for r in exact_results if '✅ OPTIMAL' in r.get('validation', '')])
        total_count = len(exact_results)
        
        if optimal_count == total_count:
            print("🎉 SUCCESS: All exact algorithms found optimal solutions!")
        else:
            print(f"❌ BUG DETECTED: {total_count - optimal_count}/{total_count} exact algorithm runs failed to find optimal solutions!")
            print("   The exact algorithms have bugs and need to be fixed.")
    
    return all_results

if __name__ == "__main__":
    results = run_validation()
