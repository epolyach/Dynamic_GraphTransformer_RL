#!/usr/bin/env python3
"""
Shared utility functions for CVRP solvers
"""
import numpy as np
from typing import List


def calculate_route_cost(vehicle_routes: List[List[int]], distances: np.ndarray) -> float:
    """
    Calculate the total cost of a route solution using double precision.
    This standardized function ensures all solvers report costs consistently.
    
    IMPORTANT: This function assumes vehicle_routes contain only customer nodes (no depot).
    It adds the depot-to-first-customer and last-customer-to-depot costs automatically.
    
    Args:
        vehicle_routes: List of routes, each route is a list of customer node indices (depot not included)
        distances: Distance matrix between all nodes (including depot at index 0)
        
    Returns:
        Total cost of the route solution including depot connections
    """
    total_cost = 0.0
    
    for route in vehicle_routes:
        if len(route) < 1:
            continue
            
        # Add depot-to-first and last-to-depot costs
        total_cost += float(distances[0][route[0]])  # depot (0) to first customer
        total_cost += float(distances[route[-1]][0])  # last customer to depot (0)
        
        # Add inter-customer costs
        for i in range(len(route) - 1):
            # Use double precision for accurate cost calculation
            total_cost += float(distances[route[i]][route[i + 1]])
    
    return float(total_cost)
