"""
CVRP solver implementations for benchmarking.

This module provides exact and heuristic solvers for the Capacitated Vehicle
Routing Problem, organized by compute platform (CPU/GPU).
"""

from .types import CVRPSolution
from .utils import calculate_route_cost

__all__ = [
    'CVRPSolution',
    'calculate_route_cost',
]
