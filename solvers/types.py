from dataclasses import dataclass
from typing import List

@dataclass
class CVRPSolution:
    route: List[int]
    cost: float
    num_vehicles: int
    vehicle_routes: List[List[int]]
    solve_time: float
    algorithm_used: str
    is_optimal: bool
    gap: float = 0.0

