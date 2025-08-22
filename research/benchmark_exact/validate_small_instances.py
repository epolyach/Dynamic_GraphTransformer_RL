#!/usr/bin/env python3
"""
Validate exact algorithms on small benchmark instances (N=10-39).
Focus on instances where our exact algorithms should definitely find optimal solutions.
"""

import numpy as np
import os
import sys
from typing import Dict, List
from download_benchmarks import parse_vrp_instance, convert_to_our_format
from advanced_solver import AdvancedCVRPSolver

def load_small_benchmark_instances() -> List[Dict]:
    """Load small benchmark instances suitable for exact algorithms"""
    
    # Small benchmark instances (N=16-39) with known optimal solutions
    benchmark_data = [
        # P-series (smallest)
        {'name': 'P-n16-k8', 'optimal_cost': 450, 'description': '16 customers, 8 vehicles'},
        {'name': 'P-n19-k2', 'optimal_cost': 212, 'description': '19 customers, 2 vehicles'},
        {'name': 'P-n20-k2', 'optimal_cost': 216, 'description': '20 customers, 2 vehicles'},
        {'name': 'P-n21-k2', 'optimal_cost': 211, 'description': '21 customers, 2 vehicles'},
        {'name': 'P-n22-k2', 'optimal_cost': 216, 'description': '22 customers, 2 vehicles'},
        {'name': 'P-n22-k8', 'optimal_cost': 603, 'description': '22 customers, 8 vehicles'},
        {'name': 'P-n23-k8', 'optimal_cost': 529, 'description': '23 customers, 8 vehicles'},
        
        # E-series (medium small)
        {'name': 'E-n22-k4', 'optimal_cost': 375, 'description': '22 customers, 4 vehicles'},
        {'name': 'E-n23-k3', 'optimal_cost': 569, 'description': '23 customers, 3 vehicles'},
        {'name': 'E-n30-k3', 'optimal_cost': 534, 'description': '30 customers, 3 vehicles'},
        
        # B-series (medium)
        {'name': 'B-n31-k5', 'optimal_cost': 672, 'description': '31 customers, 5 vehicles'},
        {'name': 'B-n34-k5', 'optimal_cost': 788, 'description': '34 customers, 5 vehicles'},
        {'name': 'B-n35-k5', 'optimal_cost': 955, 'description': '35 customers, 5 vehicles'},
        {'name': 'B-n38-k6', 'optimal_cost': 805, 'description': '38 customers, 6 vehicles'},
        {'name': 'B-n39-k5', 'optimal_cost': 549, 'description': '39 customers, 5 vehicles'},
        
        # A-series (largest that we can handle)
        {'name': 'A-n32-k5', 'optimal_cost': 784, 'description': '32 customers, 5 vehicles'},
        {'name': 'A-n33-k5', 'optimal_cost': 661, 'description': '33 customers, 5 vehicles'},
        {'name': 'A-n33-k6', 'optimal_cost': 742, 'description': '33 customers, 6 vehicles'},
        {'name': 'A-n34-k5', 'optimal_cost': 778, 'description': '34 customers, 5 vehicles'},
        {'name': 'A-n36-k5', 'optimal_cost': 799, 'description': '36 customers, 5 vehicles'},
        {'name': 'A-n37-k5', 'optimal_cost': 669, 'description': '37 customers, 5 vehicles'},
        {'name': 'A-n37-k6', 'optimal_cost': 949, 'description': '37 customers, 6 vehicles'},
        {'name': 'A-n38-k5', 'optimal_cost': 730, 'description': '38 customers, 5 vehicles'},
        {'name': 'A-n39-k5', 'optimal_cost': 822, 'description': '39 customers, 5 vehicles'},
        {'name': 'A-n39-k6', 'optimal_cost': 831, 'description': '39 customers, 6 vehicles'},
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
                if converted:
                    bench['data'] = converted
                    instances.append(bench)
                    print(f"✅ Loaded {bench['name']}: {converted['num_customers']} customers")
                else:
                    print(f"⚠️  Skipped {bench['name']}: conversion failed")
            else:
                print(f"❌ Failed to parse {bench['name']}")
        else:
            print(f"❌ File not found: {filename}")
    
    return instances

def test_single_instance(instance_data: Dict, timeout: float = 300.0) -> Dict:
    """Test an exact algorithm on a single small instance"""
    
    instance = instance_data['data']
    known_optimal = instance_data['optimal_cost']
    n_customers = instance['num_customers']
    
    print(f"🎯 Testing {instance_data['name']} (N={n_customers}, optimal={known_optimal})")
    
    try:
        # Use exact solver with appropriate timeout
        solver = AdvancedCVRPSolver(
            time_limit=timeout,
            enable_heuristics=False,  # Pure exact algorithms
            verbose=False
        )
        
        # Solve
        solution = solver.solve(instance)
        
        if solution and np.isfinite(solution.cost) and solution.cost > 0:
            # Calculate gap from known optimal
            gap = (solution.cost - known_optimal) / known_optimal * 100
            
            result = {
                'instance': instance_data['name'],
                'n_customers': n_customers,
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
                print(f"   ✅ OPTIMAL: Found {solution.cost:.1f} (gap: {gap:.3f}%), time: {solution.solve_time:.3f}s, algo: {solution.algorithm_used}")
            elif abs(gap) < 1.0:   # Within 1%
                result['validation'] = '⚠️  NEAR-OPTIMAL'
                print(f"   ⚠️  NEAR-OPTIMAL: Found {solution.cost:.1f} (gap: {gap:.2f}%), time: {solution.solve_time:.3f}s, algo: {solution.algorithm_used}")
            else:
                result['validation'] = '❌ SUBOPTIMAL'
                print(f"   ❌ SUBOPTIMAL: Found {solution.cost:.1f} (gap: {gap:.1f}%), time: {solution.solve_time:.3f}s, algo: {solution.algorithm_used}")
            
            return result
        else:
            print(f"   ❌ FAILED - Invalid solution")
            return {
                'instance': instance_data['name'],
                'n_customers': n_customers,
                'status': 'failed',
                'validation': '❌ FAILED',
                'error': 'Invalid solution'
            }
            
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return {
            'instance': instance_data['name'],
            'n_customers': n_customers,
            'status': 'exception',
            'validation': '❌ EXCEPTION',
            'error': str(e)
        }

def run_small_validation():
    """Run validation on small benchmark instances"""
    
    print("="*80)
    print("VALIDATING EXACT ALGORITHMS ON SMALL BENCHMARK INSTANCES")
    print("="*80)
    print()
    
    # Load instances
    instances = load_small_benchmark_instances()
    if not instances:
        print("❌ No benchmark instances available.")
        return
    
    # Sort by number of customers for easier reading
    instances.sort(key=lambda x: x['data']['num_customers'])
    
    print(f"📊 Loaded {len(instances)} small benchmark instances")
    print()
    
    # Test each instance
    results = []
    optimal_count = 0
    near_optimal_count = 0
    suboptimal_count = 0
    failed_count = 0
    
    for instance in instances:
        result = test_single_instance(instance, timeout=300.0)  # 5 minutes per instance
        results.append(result)
        
        validation = result.get('validation', '❌ UNKNOWN')
        if '✅ OPTIMAL' in validation:
            optimal_count += 1
        elif '⚠️  NEAR-OPTIMAL' in validation:
            near_optimal_count += 1
        elif '❌ SUBOPTIMAL' in validation:
            suboptimal_count += 1
        else:
            failed_count += 1
        
        print()
    
    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    total = len(results)
    print(f"\n📊 Results ({total} instances tested):")
    print(f"   ✅ Optimal:      {optimal_count:2d}/{total} ({optimal_count/total*100:.1f}%)")
    print(f"   ⚠️  Near-optimal: {near_optimal_count:2d}/{total} ({near_optimal_count/total*100:.1f}%)")  
    print(f"   ❌ Suboptimal:   {suboptimal_count:2d}/{total} ({suboptimal_count/total*100:.1f}%)")
    print(f"   💀 Failed:       {failed_count:2d}/{total} ({failed_count/total*100:.1f}%)")
    
    # Check if exact algorithms are working
    success_count = optimal_count + near_optimal_count
    if success_count == total:
        print(f"\n🎉 SUCCESS: All exact algorithms found good solutions!")
        if optimal_count == total:
            print("   Perfect! All solutions are optimal.")
        else:
            print(f"   {optimal_count} optimal, {near_optimal_count} near-optimal (acceptable).")
    else:
        print(f"\n❌ BUG DETECTED: {total - success_count}/{total} exact algorithm runs failed to find good solutions!")
        print("   The exact algorithms still have bugs and need to be fixed.")
    
    # Detailed breakdown by size
    print(f"\n📋 DETAILED RESULTS:")
    print(f"{'Instance':<12} {'N':<3} {'Found':<8} {'Optimal':<8} {'Gap%':<8} {'Time':<8} {'Algorithm':<15} {'Status'}")
    print("-" * 90)
    
    for result in results:
        if result.get('status') == 'success':
            print(f"{result.get('instance', ''):<12} "
                  f"{result.get('n_customers', 0):<3} "
                  f"{result.get('found_cost', 0):<8.1f} "
                  f"{result.get('known_optimal', 0):<8.1f} "
                  f"{result.get('gap_percent', 0):<8.3f} "
                  f"{result.get('solve_time', 0):<8.2f} "
                  f"{result.get('algorithm_used', ''):<15} "
                  f"{result.get('validation', '')}")
        else:
            print(f"{result.get('instance', ''):<12} "
                  f"{result.get('n_customers', 0):<3} "
                  f"{'FAILED':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<15} "
                  f"{result.get('validation', '')}")
    
    print("\n" + "="*80)
    return results

if __name__ == "__main__":
    results = run_small_validation()
