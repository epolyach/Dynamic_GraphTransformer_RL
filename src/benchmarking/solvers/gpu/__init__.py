"""
GPU-based CVRP solvers.

This module contains GPU-accelerated implementations of exact and heuristic
solvers for the Capacitated Vehicle Routing Problem.
"""

# Import solvers for easier access
from . import exact_gpu
from . import exact_gpu_dp
from . import exact_gpu_improved

__all__ = [
    'exact_gpu',
    'exact_gpu_dp',
    'exact_gpu_improved',
]
