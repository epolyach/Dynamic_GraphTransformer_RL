#!/usr/bin/env python3
import time
import numpy as np
from typing import Dict, Any, List, Tuple
from solvers.types import CVRPSolution


def _compute_route_cost(route: List[int], distances: np.ndarray) -> float:
    cost = 0.0
    for i in range(len(route) - 1):
        cost += float(distances[route[i]][route[i + 1]])
    return cost


def _split_by_capacity(route: List[int], demands: np.ndarray, capacity: int) -> List[List[int]]:
    vrs = []
    cur = [0]
    load = 0
    for node in route[1:]:
        if node == 0:
            if len(cur) > 1:
                if cur[-1] != 0:
                    cur.append(0)
                vrs.append(cur)
            cur = [0]
            load = 0
        else:
            d = int(demands[node])
            if load + d <= capacity:
                cur.append(node)
                load += d
            else:
                cur.append(0)
                vrs.append(cur)
                cur = [0, node]
                load = d
    if len(cur) > 1:
        if cur[-1] != 0:
            cur.append(0)
        vrs.append(cur)
    return vrs


def solve(instance: Dict[str, Any], time_limit: float = 300.0, verbose: bool = False) -> CVRPSolution:
    """Heuristic DP route constructor (nearest-feasible chaining with capacity resets).
    Not exact. is_optimal=False.
    """
    start = time.time()
    distances = instance['distances']
    demands = instance['demands']
    capacity = int(instance['capacity'])
    n = len(distances)

    route = [0]
    unvisited = set(range(1, n))
    cap_left = capacity
    cur = 0

    while unvisited:
        best = None
        best_d = float('inf')
        for c in unvisited:
            if int(demands[c]) <= cap_left:
                d = float(distances[cur][c])
                if d < best_d:
                    best_d = d
                    best = c
        if best is None:
            route.append(0)
            cap_left = capacity
            cur = 0
        else:
            route.append(best)
            unvisited.remove(best)
            cap_left -= int(demands[best])
            cur = best
    if route[-1] != 0:
        route.append(0)

    cost = _compute_route_cost(route, distances)
    vrs = _split_by_capacity(route, demands, capacity)
    t = time.time() - start
    return CVRPSolution(route=route, cost=cost, num_vehicles=len(vrs), vehicle_routes=vrs,
                        solve_time=t, algorithm_used='Heuristic-DP', is_optimal=False)

