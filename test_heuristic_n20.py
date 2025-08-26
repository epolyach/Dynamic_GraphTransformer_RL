#!/usr/bin/env python3
"""Test heuristic solver for N=20"""
import sys
import numpy as np
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
import solvers.heuristic_or as heuristic_or

# Generate a test instance for N=20
gen = EnhancedCVRPGenerator(config={})
instance = gen.generate_instance(
    num_customers=20,
    capacity=30,
    coord_range=100,
    demand_range=[1, 10],
    seed=42,
    instance_type=InstanceType.RANDOM,
    apply_augmentation=False,
)

print(f"Testing N=20 instance")
print(f"Total demand: {sum(instance['demands'][1:])}")
print(f"Capacity: {instance['capacity']}")

try:
    solution = heuristic_or.solve(instance, time_limit=60.0, verbose=True)
    print(f"Success! Cost: {solution.cost:.4f}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
