from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CVRPSolution:
    route: List[int]                     # Flattened route across vehicles (with depot nodes where applicable)
    cost: float                          # Total route cost (original distance scale)
    num_vehicles: int                    # Number of vehicles used
    vehicle_routes: List[List[int]]      # Per-vehicle customer routes (customers only or with depot; see solver docstring)
    solve_time: float                    # Wall-clock time in seconds
    algorithm_used: str                  # Solver identifier
    is_optimal: bool                     # True if solver guarantees optimality for this instance
    gap: Optional[float] = None          # Optional optimality gap if available
