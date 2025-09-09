#!/usr/bin/env python3
"""
Run SCIP benchmark for N=10, C=30, 1000 instances
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
import json

sys.path.append('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL')
from src.generator.generator import _generate_instance

# Import the SCIP solver
sys.path.append('/home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL/benchmark_gpu/scripts')
from gpu_cvrp_solver_scip_optimal_fixed import SCIPCVRPSolver

def run_benchmark():
    """Run SCIP benchmark for 1000 instances"""
    
    # Parameters
    n_customers = 10
    capacity = 30
    num_instances = 1000
    time_limit = 10  # 10 seconds per instance (can be adjusted)
    
    print("="*70)
    print("SCIP CVRP SOLVER - PRODUCTION RUN")
    print("="*70)
    print(f"Problem size: N={n_customers} customers")
    print(f"Vehicle capacity: C={capacity}")
    print(f"Number of instances: {num_instances}")
    print(f"Time limit per instance: {time_limit}s")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize solver
    solver = SCIPCVRPSolver(time_limit=time_limit, gap=0.0)
    
    # Results storage
    results = []
    
    # Progress tracking
    start_time = time.time()
    solved_count = 0
    optimal_count = 0
    
    print("\nStarting to solve instances...")
    print("Progress will be shown every 10 instances")
    print("-"*50)
    
    for i in range(num_instances):
        # Generate instance with deterministic seed
        instance = _generate_instance(
            num_customers=n_customers,
            capacity=capacity,
            demand_range=[1, 10],
            coord_range=1,
            seed=42 + i  # Deterministic seed
        )
        
        # Solve instance
        inst_start = time.time()
        try:
            solution = solver.solve(instance, verbose=False)
            inst_time = time.time() - inst_start
            
            # Store result
            result = {
                'instance_id': i,
                'seed': 42 + i,
                'cost': solution.cost if solution.cost < float('inf') else None,
                'num_routes': len(solution.routes) if solution.routes else None,
                'is_optimal': solution.is_optimal,
                'solve_time': inst_time,
                'success': solution.cost < float('inf')
            }
            
            if solution.cost < float('inf'):
                solved_count += 1
                if solution.is_optimal:
                    optimal_count += 1
            
        except Exception as e:
            print(f"Error on instance {i}: {str(e)}")
            result = {
                'instance_id': i,
                'seed': 42 + i,
                'cost': None,
                'num_routes': None,
                'is_optimal': False,
                'solve_time': time.time() - inst_start,
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
        
        # Progress report every 10 instances
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (num_instances - i - 1) / rate
            
            print(f"Instances: {i+1}/{num_instances} | "
                  f"Solved: {solved_count} | "
                  f"Optimal: {optimal_count} | "
                  f"Rate: {rate:.2f} inst/s | "
                  f"ETA: {eta/60:.1f} min")
            
            # Save intermediate results
            if (i + 1) % 100 == 0:
                df = pd.DataFrame(results)
                df.to_csv(f'scip_n{n_customers}_c{capacity}_intermediate_{i+1}.csv', index=False)
                print(f"  -> Saved intermediate results ({i+1} instances)")
    
    # Final statistics
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    
    # Calculate statistics
    df = pd.DataFrame(results)
    valid_results = df[df['success'] == True]
    
    if len(valid_results) > 0:
        avg_cost = valid_results['cost'].mean()
        std_cost = valid_results['cost'].std()
        avg_routes = valid_results['num_routes'].mean()
        avg_solve_time = valid_results['solve_time'].mean()
    else:
        avg_cost = std_cost = avg_routes = avg_solve_time = 0
    
    success_rate = (solved_count / num_instances) * 100
    optimal_rate = (optimal_count / num_instances) * 100
    
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Success rate: {success_rate:.1f}% ({solved_count}/{num_instances})")
    print(f"Optimal rate: {optimal_rate:.1f}% ({optimal_count}/{num_instances})")
    
    if len(valid_results) > 0:
        print(f"\nFor successfully solved instances:")
        print(f"  Average cost: {avg_cost:.4f} (std: {std_cost:.4f})")
        print(f"  Average routes: {avg_routes:.2f}")
        print(f"  Average solve time: {avg_solve_time:.2f}s")
        print(f"  Average CPC: {avg_cost/n_customers:.4f}")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"scip_n{n_customers}_c{capacity}_final_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nFinal results saved to: {output_file}")
    
    # Save summary
    summary = {
        'n_customers': n_customers,
        'capacity': capacity,
        'num_instances': num_instances,
        'time_limit': time_limit,
        'total_time_minutes': total_time/60,
        'success_rate': success_rate,
        'optimal_rate': optimal_rate,
        'solved_count': solved_count,
        'optimal_count': optimal_count,
        'avg_cost': avg_cost if len(valid_results) > 0 else None,
        'std_cost': std_cost if len(valid_results) > 0 else None,
        'avg_cpc': avg_cost/n_customers if len(valid_results) > 0 else None,
        'avg_routes': avg_routes if len(valid_results) > 0 else None,
        'avg_solve_time': avg_solve_time if len(valid_results) > 0 else None
    }
    
    summary_file = f"scip_n{n_customers}_c{capacity}_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    print("\n" + "="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    run_benchmark()
