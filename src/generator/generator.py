from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
import multiprocessing as mp
from functools import partial

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


def _generate_instance_worker(args: Tuple[int, int, int, List[int], int]) -> Dict[str, Any]:
    """Worker function for parallel instance generation.
    
    Args:
        args: Tuple of (num_customers, capacity, coord_range, demand_range, seed)
    
    Returns:
        Generated instance dictionary
    """
    num_customers, capacity, coord_range, demand_range, seed = args
    return _generate_instance(num_customers, capacity, coord_range, demand_range, seed)


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


def create_parallel_data_generator(config: Dict[str, Any], 
                                   num_workers: int = 4) -> Callable[[int, int, Optional[int]], List[Dict[str, Any]]]:
    """Create a parallel data generator that uses multiprocessing to speed up instance generation.
    
    This can provide 4-5x speedup on multi-core CPUs by parallelizing the instance generation
    and distance matrix computation across multiple processes.
    
    Args:
        config: Configuration dictionary
        num_workers: Number of parallel worker processes (default: 4)
        
    Returns:
        A generator function with same signature as create_data_generator
    """
    base_customers = int(config['problem']['num_customers'])
    capacity = int(config['problem']['vehicle_capacity'])
    coord_range = int(config['problem']['coord_range'])
    demand_range = list(config['problem']['demand_range'])
    
    def gen(batch_size: int, epoch: int = 1, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        base_seed = (epoch * 1000) if seed is None else int(seed)
        
        # Prepare arguments for all instances
        args_list = [
            (base_customers, capacity, coord_range, demand_range, base_seed + i)
            for i in range(batch_size)
        ]
        
        # Use multiprocessing pool to generate instances in parallel
        # Note: Pool is created per call to avoid issues with CUDA and fork()
        # For better performance in production, consider using a persistent pool
        with mp.Pool(processes=num_workers) as pool:
            instances = pool.map(_generate_instance_worker, args_list)
        
        return instances
    
    return gen


class ParallelDataGeneratorPool:
    """Persistent multiprocessing pool for data generation.
    
    This class maintains a long-lived process pool to avoid the overhead of creating
    and destroying pools for each batch. It's more efficient for training loops that
    generate many batches.
    
    Usage:
        pool = ParallelDataGeneratorPool(config, num_workers=6)
        for epoch in range(num_epochs):
            for batch_idx in range(num_batches):
                instances = pool.generate_batch(batch_size, epoch, seed)
        pool.close()
    """
    
    def __init__(self, config: Dict[str, Any], num_workers: int = 4):
        """Initialize the persistent pool.
        
        Args:
            config: Configuration dictionary
            num_workers: Number of parallel worker processes
        """
        self.config = config
        self.num_workers = num_workers
        self.base_customers = int(config['problem']['num_customers'])
        self.capacity = int(config['problem']['vehicle_capacity'])
        self.coord_range = int(config['problem']['coord_range'])
        self.demand_range = list(config['problem']['demand_range'])
        
        # Create persistent pool
        # Use 'spawn' start method for better CUDA compatibility
        ctx = mp.get_context('spawn')
        self.pool = ctx.Pool(processes=num_workers)
    
    def generate_batch(self, batch_size: int, epoch: int = 1, 
                      seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate a batch of instances in parallel.
        
        Args:
            batch_size: Number of instances to generate
            epoch: Current epoch (used for seeding if seed is None)
            seed: Base seed for instance generation
            
        Returns:
            List of generated instances
        """
        base_seed = (epoch * 1000) if seed is None else int(seed)
        
        # Prepare arguments for all instances
        args_list = [
            (self.base_customers, self.capacity, self.coord_range, 
             self.demand_range, base_seed + i)
            for i in range(batch_size)
        ]
        
        # Generate instances in parallel using the persistent pool
        instances = self.pool.map(_generate_instance_worker, args_list)
        
        return instances
    
    def close(self):
        """Close the pool and free resources."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
    
    def __del__(self):
        """Ensure pool is closed on deletion."""
        self.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
        return False
