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
    
    Args:
        vehicle_routes: List of routes, each route is a list of node indices
        distances: Distance matrix between all nodes
        
    Returns:
        Total cost of the route solution
    """
    total_cost = 0.0
    for route in vehicle_routes:
        for i in range(len(route) - 1):
            # Use double precision for accurate cost calculation
            total_cost += float(distances[route[i]][route[i + 1]])
    return float(total_cost)
