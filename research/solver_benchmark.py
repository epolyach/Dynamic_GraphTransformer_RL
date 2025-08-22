#!/usr/bin/env python3
"""
CVRP Exact Solver Benchmark CLI
Fast version matching the original exact baseline approach.
Generates CSV with N,time_or,cpc_or,std_or,time_dp,cpc_dp,std_dp
"""

import argparse
import numpy as np
import time
import csv
import sys
from typing import Dict, List, Any
from pathlib import Path
import statistics

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
from research.exact_solver import ExactCVRPSolver
from src.data.enhanced_generator import EnhancedCVRPGenerator, InstanceType

def generate_cvrp_instance(num_customers: int, capacity: int = 30, demand_range: List[int] = None, seed: int = None) -> Dict[str, Any]:
    """
    Generate a random CVRP instance.
    
    Args:
        num_customers: Number of customers (excluding depot)
        capacity: Vehicle capacity
        demand_range: Range of possible demand values [min, max]
        seed: Random seed for reproducibility
        
    Returns:
        CVRP instance as a dictionary with 'coords', 'demands', 'distances', 'capacity'
    """
    if demand_range is None:
        demand_range = [1, 10]  # Default demand range
    
    if seed is not None:
        np.random.seed(seed)
        
    # Generate coordinates for depot (0) and customers (1...n)
    coords = np.random.rand(num_customers + 1, 2)  # Random coordinates in [0,1]×[0,1]
    
    # Generate demands (depot has no demand)
    demands = np.zeros(num_customers + 1, dtype=np.int32)
    for i in range(1, num_customers + 1):
        demands[i] = np.random.randint(demand_range[0], demand_range[1] + 1)
    
    # Compute distance matrix (Euclidean)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    return {
        'coords': coords,
        'demands': demands,
        'distances': distances,
        'capacity': capacity
    }

def run_benchmark(min_n: int = 5, max_n: int = 50, num_instances: int = 100, 
                  capacity: int = 30, demand_range: List[int] = None, 
                  time_limit_minutes: float = 1.0,
                  output_file: str = "cvrp_benchmark_results.csv"):
    """
    Run the solver benchmark for different problem sizes.
    
    Args:
        min_n: Minimum number of customers to test
        max_n: Maximum number of customers to test
        num_instances: Number of instances to generate for each size
        capacity: Vehicle capacity
        demand_range: Range of possible demand values [min, max]
        time_limit_minutes: Time limit in minutes before stopping
        output_file: CSV file to save results
    """
    if demand_range is None:
        demand_range = [1, 10]  # Default demand range from 1 to 10
    
    # Initialize results CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['N', 'time_or', 'cpc_or', 'std_or', 'time_dp', 'cpc_dp', 'std_dp'])
        csvfile.flush()  # Ensure header is written immediately
    
    # Run benchmark for each problem size
    n = min_n
    while n <= max_n:
        logger.info(f"Testing problem size N={n}...")
        
        # Track metrics
        or_times = []
        or_costs = []
        dp_times = []
        dp_costs = []
        
        # Create exact solver instances with very short time limits
        # This is to ensure we don't spend too much time on a single instance
        # The benchmark will still timeout based on total time
        solver_or = ExactCVRPSolver(time_limit=10.0, enable_or_tools=True, enable_gurobi=False, verbose=False)
        solver_dp = ExactCVRPSolver(time_limit=10.0, enable_or_tools=False, enable_gurobi=False, verbose=False)
        
        # Time the entire benchmark for this N
        start_time_n = time.time()
        
        for i in range(num_instances):
            # Generate a random instance with the same seed for fair comparison
            seed = n * 10000 + i
            instance = generate_cvrp_instance(n, capacity, demand_range, seed)
            
            # Solve with OR-Tools
            try:
                start_time = time.time()
                solution_or = solver_or._solve_ortools_advanced(instance)
                solve_time_or = time.time() - start_time
                or_times.append(solve_time_or)
                # Cost per customer
                or_costs.append(solution_or.cost / n)
                
                # Log progress occasionally
                if (i + 1) % 10 == 0:
                    logger.info(f"  Completed {i+1}/{num_instances} instances with OR-Tools, avg time: {statistics.mean(or_times):.4f}s")
            except Exception as e:
                logger.warning(f"  OR-Tools solver failed on instance {i+1}: {e}")
                # Use a large time value to indicate failure
                or_times.append(60.0)  # Use time limit as penalty
                or_costs.append(float('inf'))
            
            # Solve with DP
            try:
                start_time = time.time()
                solution_dp = solver_dp._solve_dp_bitmasking(instance)
                solve_time_dp = time.time() - start_time
                dp_times.append(solve_time_dp)
                # Cost per customer  
                dp_costs.append(solution_dp.cost / n)
                
                # Log progress occasionally
                if (i + 1) % 10 == 0:
                    logger.info(f"  Completed {i+1}/{num_instances} instances with DP, avg time: {statistics.mean(dp_times):.4f}s")
            except Exception as e:
                logger.warning(f"  DP solver failed on instance {i+1}: {e}")
                # Use a large time value to indicate failure
                dp_times.append(60.0)  # Use time limit as penalty
                dp_costs.append(float('inf'))
            
            # Check if we've exceeded the time limit for this N
            elapsed_time_n = time.time() - start_time_n
            if elapsed_time_n > time_limit_minutes * 60:
                logger.info(f"  Time limit reached after {i+1} instances. Moving to next N.")
                break
        
        # Calculate statistics (if we have any successful solves)
        if or_costs and any(cost != float('inf') for cost in or_costs):
            valid_or_costs = [cost for cost in or_costs if cost != float('inf')]
            mean_cost_or = statistics.mean(valid_or_costs) if valid_or_costs else float('inf')
            std_cost_or = statistics.stdev(valid_or_costs) if len(valid_or_costs) > 1 else 0.0
        else:
            mean_cost_or = float('inf')
            std_cost_or = 0.0
            
        if dp_costs and any(cost != float('inf') for cost in dp_costs):
            valid_dp_costs = [cost for cost in dp_costs if cost != float('inf')]
            mean_cost_dp = statistics.mean(valid_dp_costs) if valid_dp_costs else float('inf')
            std_cost_dp = statistics.stdev(valid_dp_costs) if len(valid_dp_costs) > 1 else 0.0
        else:
            mean_cost_dp = float('inf')
            std_cost_dp = 0.0
        
        mean_time_or = statistics.mean(or_times)
        mean_time_dp = statistics.mean(dp_times)
        
        # Write results to CSV
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                n, 
                f"{mean_time_or:.6f}", 
                f"{mean_cost_or:.6f}", 
                f"{std_cost_or:.6f}",
                f"{mean_time_dp:.6f}", 
                f"{mean_cost_dp:.6f}", 
                f"{std_cost_dp:.6f}"
            ])
            csvfile.flush()  # Ensure data is written immediately
        
        # Log summary
        logger.info(f"Results for N={n}:")
        logger.info(f"  OR-Tools: Avg time={mean_time_or:.6f}s, Avg cost={mean_cost_or:.6f}, Std={std_cost_or:.6f}")
        logger.info(f"  DP: Avg time={mean_time_dp:.6f}s, Avg cost={mean_cost_dp:.6f}, Std={std_cost_dp:.6f}")
        
        # Check if we need to stop (both solvers taking too long)
        elapsed_time_n = time.time() - start_time_n
        if elapsed_time_n > time_limit_minutes * 60:
            logger.info(f"Time limit exceeded for N={n}. Stopping benchmark.")
            break
        
        # Move to next problem size
        n += 1

if __name__ == "__main__":
    logger.info("Starting CVRP solver benchmark...")
    
    # Run the benchmark
    run_benchmark(
        min_n=5,
        max_n=50,
        num_instances=100,
        capacity=30,
        demand_range=[1, 10],
        time_limit_minutes=1.0,
        output_file="cvrp_benchmark_results.csv"
    )
    
    logger.info("Benchmark complete! Results saved to cvrp_benchmark_results.csv")
