#!/usr/bin/env python3
"""Analyze results from cpu_20.csv"""
import pandas as pd
import numpy as np

# Read the CSV
df = pd.read_csv('results/csv/cpu_20.csv')

print("=== Analysis of N=20 CVRP Results ===")
print(f"Total instances: {len(df)}")

# Group by solver
for solver in df['solver'].unique():
    solver_df = df[df['solver'] == solver]
    success_df = solver_df[solver_df['status'] == 'success']
    
    print(f"\n{solver}:")
    print(f"  Successful: {len(success_df)}/{len(solver_df)}")
    
    if len(success_df) > 0:
        costs = success_df['cpc'].values
        times = success_df['time'].values
        
        # Calculate statistics
        avg_cost = np.mean(costs)
        std_cost = np.std(costs, ddof=1)  # Sample standard deviation
        sem_cost = std_cost / np.sqrt(len(costs))  # Standard error of mean
        
        avg_time = np.mean(times)
        std_time = np.std(times, ddof=1)
        sem_time = std_time / np.sqrt(len(times))
        
        print(f"  Cost: {avg_cost:.4f} ± {sem_cost:.4f} (mean ± SEM)")
        print(f"        std dev = {std_cost:.4f}")
        print(f"  Time: {avg_time:.4f} ± {sem_time:.4f} seconds")
        print(f"        std dev = {std_time:.4f}s")
        print(f"  Min cost: {np.min(costs):.4f}")
        print(f"  Max cost: {np.max(costs):.4f}")
