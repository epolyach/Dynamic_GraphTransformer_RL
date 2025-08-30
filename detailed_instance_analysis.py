#!/usr/bin/env python3
"""
Detailed analysis: Why are the CPCs different?
Let's generate instances using both methods and compare their characteristics
"""

import numpy as np
import sys
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType

def simple_generator(n_customers, seed):
    """GPU benchmark generator"""
    np.random.seed(seed)
    n = n_customers + 1
    coords = np.random.uniform(0, 1, size=(n, 2))
    coords[0] = [0.5, 0.5]
    demands = np.zeros(n, dtype=np.float32)
    demands[1:] = np.random.uniform(1, 10, size=n_customers)
    capacity = max(demands.sum() / 2, demands.max() * 2)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])
    return {'coords': coords, 'demands': demands, 'distances': distances,
            'capacity': capacity, 'n_customers': n_customers}

def enhanced_generator(n_customers, seed):
    """CPU benchmark generator"""
    gen = EnhancedCVRPGenerator(config={})
    instance = gen.generate_instance(
        num_customers=n_customers,
        capacity=30,  # Fixed capacity!
        coord_range=100,
        demand_range=[1, 10],
        seed=seed,
        instance_type=InstanceType.RANDOM,
        apply_augmentation=False,
    )
    return instance

def analyze_instances(n_customers=6, n_instances=20):
    """Compare instance characteristics from both generators"""
    
    print("=" * 70)
    print(f"Comparing Instance Generators (N={n_customers}, {n_instances} instances)")
    print("=" * 70)
    
    # Generate instances with both methods
    simple_instances = []
    enhanced_instances = []
    
    for i in range(n_instances):
        # Simple (GPU) generator
        simple_seed = 5000 + i
        simple_inst = simple_generator(n_customers, simple_seed)
        simple_instances.append(simple_inst)
        
        # Enhanced (CPU) generator  
        enhanced_seed = 4242 + n_customers * 1000 + i * 10
        enhanced_inst = enhanced_generator(n_customers, enhanced_seed)
        enhanced_instances.append(enhanced_inst)
    
    # Analyze characteristics
    print("\n1. CAPACITY ANALYSIS:")
    print("-" * 50)
    
    simple_caps = [inst['capacity'] for inst in simple_instances]
    enhanced_caps = [inst['capacity'] for inst in enhanced_instances]
    
    print(f"\nSimple Generator (GPU benchmark):")
    print(f"  Capacities: {simple_caps[:5]} ...")
    print(f"  Mean: {np.mean(simple_caps):.2f}")
    print(f"  Std:  {np.std(simple_caps):.2f}")
    print(f"  Min:  {np.min(simple_caps):.2f}")
    print(f"  Max:  {np.max(simple_caps):.2f}")
    
    print(f"\nEnhanced Generator (CPU benchmark):")
    print(f"  Capacities: {enhanced_caps[:5]} ...")
    print(f"  Mean: {np.mean(enhanced_caps):.2f}")
    print(f"  Std:  {np.std(enhanced_caps):.2f}")
    print(f"  Min:  {np.min(enhanced_caps):.2f}")
    print(f"  Max:  {np.max(enhanced_caps):.2f}")
    
    print("\n2. DEMAND ANALYSIS:")
    print("-" * 50)
    
    simple_demands_avg = [np.mean(inst['demands'][1:]) for inst in simple_instances]
    enhanced_demands_avg = [np.mean(inst['demands'][1:]) for inst in enhanced_instances]
    
    print(f"\nSimple Generator (GPU benchmark):")
    print(f"  Avg demands per instance: {simple_demands_avg[:5]}")
    print(f"  Overall mean demand: {np.mean(simple_demands_avg):.2f}")
    print(f"  Demand type: Continuous uniform [1, 10]")
    
    print(f"\nEnhanced Generator (CPU benchmark):")
    print(f"  Avg demands per instance: {enhanced_demands_avg[:5]}")
    print(f"  Overall mean demand: {np.mean(enhanced_demands_avg):.2f}")
    print(f"  Demand type: Integer uniform [1, 10]")
    
    print("\n3. COORDINATE ANALYSIS:")
    print("-" * 50)
    
    print(f"\nSimple Generator:")
    print(f"  Coordinate type: Continuous uniform [0, 1]")
    print(f"  Example coords: {simple_instances[0]['coords'][:3]}")
    
    print(f"\nEnhanced Generator:")
    print(f"  Coordinate type: Integer grid scaled to [0, 1]")
    print(f"  Example coords: {enhanced_instances[0]['coords'][:3]}")
    
    print("\n4. DISTANCE MATRIX STATISTICS:")
    print("-" * 50)
    
    simple_avg_dist = [np.mean(inst['distances'][np.triu_indices(n_customers+1, k=1)]) 
                       for inst in simple_instances]
    enhanced_avg_dist = [np.mean(inst['distances'][np.triu_indices(n_customers+1, k=1)]) 
                         for inst in enhanced_instances]
    
    print(f"\nSimple Generator:")
    print(f"  Mean pairwise distance: {np.mean(simple_avg_dist):.4f}")
    print(f"  Std of mean distances: {np.std(simple_avg_dist):.4f}")
    
    print(f"\nEnhanced Generator:")
    print(f"  Mean pairwise distance: {np.mean(enhanced_avg_dist):.4f}")
    print(f"  Std of mean distances: {np.std(enhanced_avg_dist):.4f}")
    
    print("\n5. KEY DIFFERENCES FOUND:")
    print("-" * 50)
    
    print("\n🔴 CRITICAL DIFFERENCE: Fixed vs Variable Capacity")
    print(f"  - Simple (GPU): Variable capacity based on demands (avg: {np.mean(simple_caps):.1f})")
    print(f"  - Enhanced (CPU): FIXED capacity = 30")
    print("\n  This affects the problem difficulty and optimal solutions!")
    
    print("\n🟡 Demand Distribution:")
    print(f"  - Simple: Continuous [1, 10]")
    print(f"  - Enhanced: Integer [1, 10]")
    
    print("\n🟡 Coordinate Distribution:")
    print(f"  - Simple: Continuous uniform")
    print(f"  - Enhanced: Integer grid scaled")
    
    # Calculate expected impact on CPC
    print("\n6. EXPECTED IMPACT ON CPC:")
    print("-" * 50)
    
    avg_demand_simple = np.mean([np.mean(inst['demands'][1:]) for inst in simple_instances])
    avg_demand_enhanced = np.mean([np.mean(inst['demands'][1:]) for inst in enhanced_instances])
    
    avg_capacity_simple = np.mean(simple_caps)
    avg_capacity_enhanced = 30
    
    vehicles_needed_simple = avg_demand_simple * n_customers / avg_capacity_simple
    vehicles_needed_enhanced = avg_demand_enhanced * n_customers / avg_capacity_enhanced
    
    print(f"\nEstimated vehicles needed:")
    print(f"  Simple (GPU):   {vehicles_needed_simple:.2f} vehicles")
    print(f"  Enhanced (CPU): {vehicles_needed_enhanced:.2f} vehicles")
    
    print(f"\nWith FIXED capacity=30, instances become EASIER (fewer vehicles needed)")
    print(f"This explains why CPU benchmark shows LOWER CPC values!")

if __name__ == "__main__":
    analyze_instances(6, 20)
