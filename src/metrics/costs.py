import numpy as np
from typing import List

# Cost utilities extracted for reuse

def compute_route_cost(route: List[int], distances: np.ndarray) -> float:
    if len(route) <= 1:
        return 0.0
    cost = 0.0
    for i in range(len(route) - 1):
        cost += float(distances[route[i], route[i + 1]])
    return cost


def count_internal_depot_visits(route: List[int]) -> int:
    if not route or len(route) < 3:
        return 0
    return sum(1 for node in route[1:-1] if node == 0)


def compute_route_cost_with_penalty(route: List[int], distances: np.ndarray, penalty_per_visit: float = 0.0) -> float:
    base = compute_route_cost(route, distances)
    if penalty_per_visit and penalty_per_visit != 0.0:
        base += penalty_per_visit * count_internal_depot_visits(route)
    return base


def compute_normalized_cost(route: List[int], distances: np.ndarray, n_customers: int) -> float:
    total_cost = compute_route_cost(route, distances)
    return total_cost / n_customers if n_customers > 0 else 0.0


def compute_naive_baseline_cost(instance: dict, depot_penalty_per_visit: float = 0.0) -> float:
    """Compute cost of naive solution: depot->customer->depot for each customer.
    Includes optional depot penalty per internal return when configured.
    """
    distances = instance['distances']
    n_customers = len(instance['coords']) - 1
    naive_cost = 0.0
    for customer_idx in range(1, n_customers + 1):
        naive_cost += float(distances[0, customer_idx] * 2)
    if depot_penalty_per_visit and depot_penalty_per_visit != 0.0 and n_customers > 0:
        naive_cost += depot_penalty_per_visit * (n_customers - 1)
    return float(naive_cost)

