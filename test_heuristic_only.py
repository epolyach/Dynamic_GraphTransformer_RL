#!/usr/bin/env python3
"""Test heuristic_or solver on N=20"""
import sys
import time
import numpy as np
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
import solvers.heuristic_or as heuristic_or

gen = EnhancedCVRPGenerator(config={})
costs = []
times = []

print("Testing heuristic_or on 10 instances of N=20...")
for i in range(10):
    instance = gen.generate_instance(
        num_customers=20, capacity=30, coord_range=100,
        demand_range=[1, 10], seed=7000+i,
        instance_type=InstanceType.RANDOM, apply_augmentation=False
    )
    
    start = time.time()
    solution = heuristic_or.solve(instance, time_limit=60.0, verbose=False)
    solve_time = time.time() - start
    
    costs.append(solution.cost)
    times.append(solve_time)
    print(f"Instance {i+1}: cost={solution.cost:.4f}, time={solve_time:.3f}s")

print(f"\nheuristic_or results:")
print(f"  Average cost: {np.mean(costs):.4f} ± {np.std(costs)/np.sqrt(len(costs)):.4f} (SEM)")
print(f"  Average time: {np.mean(times):.3f}s")
