from typing import List
import numpy as np

# Validation utilities extracted for reuse

def quick_validate_route(route: List[int], n_customers: int) -> bool:
    """Lightweight validation: start/end depot, no consecutive depots, indices in range."""
    if not route:
        return False
    if route[0] != 0 or route[-1] != 0:
        return False
    for i in range(len(route) - 1):
        if route[i] == 0 and route[i + 1] == 0:
            return False
    max_idx = n_customers
    for node in route:
        if node < 0 or node > max_idx:
            return False
    return True


def analyze_route_trips(route: List[int], demands: np.ndarray, capacity: int):
    """Return trip breakdown and detect capacity violations per trip."""
    # Ensure capacity is an integer
    capacity = int(capacity) if not isinstance(capacity, int) else capacity
    
    trips = []
    current_trip = []
    for node in route:
        current_trip.append(node)
        if node == 0 and len(current_trip) > 1:
            trip_customers = [n for n in current_trip if n != 0]
            trip_demand = int(sum(int(demands[c]) for c in trip_customers))
            trips.append({
                'nodes': current_trip[:],
                'customers': trip_customers,
                'demand': trip_demand,
                'valid': trip_demand <= capacity,
            })
            current_trip = [0]
    return trips


def validate_route(route: List[int], n_customers: int, model_name: str = "Unknown", instance: dict | None = None) -> bool:
    """Rigorous CVRP validation: structure, coverage, capacity, index bounds.
    Raises ValueError with details on failure to preserve strict philosophy.
    """
    if len(route) == 0:
        raise ValueError(f"{model_name}: Empty route")
    if route[0] != 0 or route[-1] != 0:
        raise ValueError(f"{model_name}: Route must start and end at depot (0). Got start={route[0]}, end={route[-1]}")
    for i in range(len(route) - 1):
        if route[i] == 0 and route[i + 1] == 0:
            raise ValueError(f"{model_name}: Consecutive depot visits at positions {i}-{i+1}")
    max_idx = n_customers
    for i, node in enumerate(route):
        if node < 0 or node > max_idx:
            raise ValueError(f"{model_name}: Node index out of bounds at position {i}: {node} not in [0, {max_idx}]")
    # Customer coverage
    seen = [0] * (n_customers + 1)
    for node in route:
        if node != 0:
            seen[node] += 1
    missing = [i for i in range(1, n_customers + 1) if seen[i] == 0]
    dup = [i for i in range(1, n_customers + 1) if seen[i] > 1]
    if missing:
        raise ValueError(f"{model_name}: Missing customers: {missing}")
    if dup:
        raise ValueError(f"{model_name}: Customers visited more than once: {dup}")
    # Capacity
    if instance is not None:
        # Handle GPU tensors
        demands_raw = instance["demands"]
        if hasattr(demands_raw, "cpu"):  # GPU tensor
            demands = demands_raw.cpu().numpy().astype(int)
        else:  # CPU array
            demands = np.asarray(demands_raw).astype(int)
        capacity_raw = instance["capacity"]
        capacity = int(capacity_raw.item() if hasattr(capacity_raw, "item") else capacity_raw)
        current_load = 0
        for idx, node in enumerate(route[1:]):
            if node == 0:
                current_load = 0
            else:
                current_load += int(demands[node])
                if current_load > capacity:
                    raise ValueError(f"{model_name}: Capacity exceeded at step {idx+1}: load={current_load} > capacity={capacity}")
    return True

