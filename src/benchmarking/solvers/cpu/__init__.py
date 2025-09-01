"""
CPU-based CVRP solvers.

This module contains CPU implementations of exact and heuristic solvers
for the Capacitated Vehicle Routing Problem.
"""

# Import solvers for easier access
from . import exact_dp
from . import exact_ortools_vrp_fixed
from . import ortools_gls

__all__ = [
    'exact_dp',
    'exact_ortools_vrp_fixed',
    'ortools_gls',
]
