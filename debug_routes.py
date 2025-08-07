#!/usr/bin/env python3

import torch
import numpy as np
import sys
from run_comparative_study import (
    BaselinePointerNetwork, 
    generate_cvrp_instance,
    compute_route_cost,
    compute_naive_baseline_cost,
    validate_route
)

def debug_single_instance():
    """Debug a single CVRP instance"""
    print("🔍 DEBUG: Single CVRP Instance Analysis")
    print("="*50)
    
    # Create a simple instance
    instance = generate_cvrp_instance(
        num_customers=5, 
        capacity=3,
        coord_range=50, 
        demand_range=(1, 3), 
        seed=42
    )
    
    print(f"Instance created:")
    print(f"  Coordinates: {instance['coords']}")
    print(f"  Demands: {instance['demands']}")
    print(f"  Capacity: {instance['capacity']}")
    print()
    
    # Calculate naive baseline
    naive_cost = compute_naive_baseline_cost(instance)
    print(f"Naive baseline cost: {naive_cost:.2f}")
    print()
    
    # Let's manually check naive cost calculation
    distances = instance['distances']
    manual_naive = 0.0
    for customer_idx in range(1, len(instance['coords'])):
        depot_to_customer = distances[0, customer_idx]
        round_trip = depot_to_customer * 2
        print(f"  Customer {customer_idx}: depot distance = {depot_to_customer:.2f}, round trip = {round_trip:.2f}")
        manual_naive += round_trip
    
    print(f"Manual naive calculation: {manual_naive:.2f}")
    print()
    
    # Test route generation with baseline pointer
    print("Testing Baseline Pointer Network:")
    model = BaselinePointerNetwork(3, 64)
    
    with torch.no_grad():
        routes, log_probs = model([instance], greedy=True)
        route = routes[0]
        
        print(f"Generated route: {route}")
        
        # Validate
        n_customers = len(instance['coords']) - 1
        try:
            validate_route(route, n_customers, "DEBUG")
            print("✅ Route validation passed")
        except SystemExit:
            print("❌ Route validation failed")
            return
        
        # Calculate cost
        route_cost = compute_route_cost(route, distances)
        print(f"Route cost: {route_cost:.2f}")
        
        # Manual cost calculation
        manual_cost = 0.0
        for i in range(len(route) - 1):
            segment_cost = distances[route[i], route[i+1]]
            print(f"  {route[i]} -> {route[i+1]}: {segment_cost:.2f}")
            manual_cost += segment_cost
        
        print(f"Manual cost calculation: {manual_cost:.2f}")
        print()
        
        # Compare with naive
        print(f"COMPARISON:")
        print(f"  Naive baseline: {naive_cost:.2f}")
        print(f"  Route cost:     {route_cost:.2f}")
        print(f"  Difference:     {route_cost - naive_cost:.2f}")
        
        if route_cost > naive_cost:
            print("❌ PROBLEM: Route cost is HIGHER than naive baseline!")
        else:
            print("✅ Good: Route cost is lower than naive baseline")

if __name__ == "__main__":
    debug_single_instance()
