from typing import Dict, Any, List, Optional, Callable
import numpy as np

# Canonical CVRP instance generator (single source of truth)
# Rules:
# - Coordinates sampled on integer grid [0, coord_range] then normalized to [0,1]
# - Demands are uniform integers in [demand_range[0], demand_range[1]]
# - Capacity is an integer taken from config
# - No augmentation by default
# - Seed discipline: caller provides seed; training uses non-overlapping seeds for train/val

def _generate_instance(num_customers: int,
                       capacity: int,
                       coord_range: int,
                       demand_range: List[int],
                       seed: Optional[int] = None) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(int(seed))

    # coords: (N+1, 2), depot at index 0
    coords = np.zeros((num_customers + 1, 2), dtype=np.float64)
    for i in range(num_customers + 1):
        coords[i] = np.random.randint(0, coord_range + 1, size=2) / coord_range

    # demands: (N+1,), depot has 0
    demands = np.zeros(num_customers + 1, dtype=np.int32)
    for i in range(1, num_customers + 1):
        demands[i] = np.random.randint(int(demand_range[0]), int(demand_range[1]) + 1)

    # distances: (N+1, N+1)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))

    return {
        'coords': coords,
        'demands': demands.astype(np.int32),
        'distances': distances,
        'capacity': int(capacity),
    }


def create_data_generator(config: Dict[str, Any]) -> Callable[[int, int, Optional[int]], List[Dict[str, Any]]]:
    """Create the canonical data generator used by both training and benchmarks.

    Returns a function(batch_size: int, epoch: int = 1, seed: Optional[int] = None) -> List[instances]
    """
    def gen(batch_size: int, epoch: int = 1, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        base_customers = int(config['problem']['num_customers'])
        capacity = int(config['problem']['vehicle_capacity'])
        coord_range = int(config['problem']['coord_range'])
        demand_range = list(config['problem']['demand_range'])

        instances: List[Dict[str, Any]] = []
        base_seed = (epoch * 1000) if seed is None else int(seed)
        for i in range(batch_size):
            instances.append(
                _generate_instance(
                    num_customers=base_customers,
                    capacity=capacity,
                    coord_range=coord_range,
                    demand_range=demand_range,
                    seed=base_seed + i,
                )
            )
        return instances

    return gen

