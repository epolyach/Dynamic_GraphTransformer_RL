import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Callable
import math
from enum import Enum


class InstanceType(Enum):
    """Different types of CVRP instances for diverse training."""
    RANDOM = "random"
    CLUSTERED = "clustered"
    UNIFORM_GRID = "uniform_grid"
    RADIAL = "radial"
    MIXED = "mixed"


class CVRPDataAugmentation:
    """Data augmentation techniques for CVRP instances."""
    
    @staticmethod
    def rotate_instance(coords: np.ndarray, angle: float) -> np.ndarray:
        """Rotate coordinates by given angle."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # Keep depot at origin for rotation
        depot = coords[0].copy()
        coords_centered = coords - depot
        coords_rotated = coords_centered @ rotation_matrix.T
        return coords_rotated + depot
    
    @staticmethod
    def scale_instance(coords: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale coordinates uniformly."""
        depot = coords[0].copy()
        coords_centered = coords - depot
        coords_scaled = coords_centered * scale_factor
        return coords_scaled + depot
    
    @staticmethod
    def add_noise(coords: np.ndarray, noise_std: float) -> np.ndarray:
        """Add Gaussian noise to coordinates."""
        noise = np.random.normal(0, noise_std, coords.shape)
        # Don't add noise to depot
        noise[0] = 0
        return coords + noise
    
    @staticmethod
    def flip_instance(coords: np.ndarray, axis: int) -> np.ndarray:
        """Flip coordinates along specified axis (0=x, 1=y)."""
        depot = coords[0].copy()
        coords_centered = coords - depot
        coords_flipped = coords_centered.copy()
        coords_flipped[:, axis] *= -1
        return coords_flipped + depot


class EnhancedCVRPGenerator:
    """Enhanced CVRP instance generator with diverse patterns and augmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.augmentation = CVRPDataAugmentation()
        
    def generate_instance(self, 
                         num_customers: int,
                         capacity: int, 
                         coord_range: int,
                         demand_range: List[int],
                         seed: Optional[int] = None,
                         instance_type: InstanceType = InstanceType.RANDOM,
                         apply_augmentation: bool = False) -> Dict[str, Any]:
        """Generate a single CVRP instance with specified characteristics."""
        
        if seed is not None:
            np.random.seed(seed)
            
        # Generate coordinates based on instance type
        coords = self._generate_coordinates(num_customers, coord_range, instance_type)
        
        # Apply augmentation if requested
        if apply_augmentation:
            coords = self._apply_random_augmentation(coords, coord_range)
            
        # Generate demands
        demands = self._generate_demands(num_customers, demand_range, capacity)
        
        # Compute distances
        distances = self._compute_distances(coords)
        
        return {
            'coords': coords,
            'demands': demands.astype(np.int32),
            'distances': distances,
            'capacity': int(capacity),
            'instance_type': instance_type.value,
            'num_customers': num_customers
        }
    
    def _generate_coordinates(self, num_customers: int, coord_range: int, 
                            instance_type: InstanceType) -> np.ndarray:
        """Generate coordinates based on instance type."""
        coords = np.zeros((num_customers + 1, 2), dtype=np.float64)
        
        if instance_type == InstanceType.RANDOM:
            # Standard random distribution
            for i in range(num_customers + 1):
                coords[i] = np.random.randint(0, coord_range + 1, size=2) / coord_range
                
        elif instance_type == InstanceType.CLUSTERED:
            # Generate clusters of customers
            num_clusters = max(2, num_customers // 5)
            cluster_centers = np.random.rand(num_clusters, 2)
            cluster_assignments = np.random.randint(0, num_clusters, num_customers)
            
            # Place depot at center
            coords[0] = np.array([0.5, 0.5])
            
            # Place customers around cluster centers
            for i in range(1, num_customers + 1):
                cluster_id = cluster_assignments[i - 1]
                center = cluster_centers[cluster_id]
                # Add noise around cluster center
                noise = np.random.normal(0, 0.1, 2)
                coords[i] = np.clip(center + noise, 0, 1)
                
        elif instance_type == InstanceType.UNIFORM_GRID:
            # Place customers on a grid
            grid_size = int(np.ceil(np.sqrt(num_customers + 1)))
            grid_coords = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(grid_coords) <= num_customers:
                        x = (i + 0.5) / grid_size
                        y = (j + 0.5) / grid_size
                        grid_coords.append([x, y])
            
            # Shuffle and assign
            np.random.shuffle(grid_coords)
            for i in range(num_customers + 1):
                coords[i] = grid_coords[i]
                
        elif instance_type == InstanceType.RADIAL:
            # Place customers in concentric circles
            coords[0] = np.array([0.5, 0.5])  # Depot at center
            
            num_circles = max(2, num_customers // 8)
            customers_per_circle = num_customers // num_circles
            
            customer_idx = 1
            for circle in range(num_circles):
                radius = (circle + 1) / (num_circles + 1) * 0.4  # Max radius 0.4
                circle_customers = customers_per_circle
                if circle == num_circles - 1:  # Last circle gets remaining customers
                    circle_customers = num_customers - customer_idx + 1
                    
                for i in range(circle_customers):
                    if customer_idx <= num_customers:
                        angle = 2 * np.pi * i / circle_customers
                        x = 0.5 + radius * np.cos(angle)
                        y = 0.5 + radius * np.sin(angle)
                        coords[customer_idx] = np.array([x, y])
                        customer_idx += 1
                        
        elif instance_type == InstanceType.MIXED:
            # Combine different patterns
            patterns = [InstanceType.RANDOM, InstanceType.CLUSTERED, InstanceType.RADIAL]
            chosen_pattern = np.random.choice(patterns)
            return self._generate_coordinates(num_customers, coord_range, chosen_pattern)
        
        return coords
    
    def _generate_demands(self, num_customers: int, demand_range: List[int], 
                         capacity: int) -> np.ndarray:
        """Generate demands with improved distribution."""
        demands = np.zeros(num_customers + 1, dtype=np.int32)
        
        # Depot has zero demand
        demands[0] = 0
        
        # Generate customer demands
        for i in range(1, num_customers + 1):
            # Use a mixture of uniform and capacity-aware demands
            if np.random.random() < 0.8:  # 80% uniform
                demands[i] = np.random.randint(demand_range[0], demand_range[1] + 1)
            else:  # 20% capacity-aware (larger demands)
                max_demand = min(demand_range[1], capacity // 2)
                demands[i] = np.random.randint(demand_range[0], max_demand + 1)
        
        # Ensure total demand is feasible (not exceeding reasonable number of vehicles)
        total_demand = demands.sum()
        max_vehicles = max(2, num_customers // 3)  # Reasonable upper bound
        
        if total_demand > capacity * max_vehicles:
            # Scale down demands proportionally
            scale_factor = (capacity * max_vehicles) / total_demand
            for i in range(1, num_customers + 1):
                demands[i] = max(demand_range[0], int(demands[i] * scale_factor))
        
        return demands
    
    def _compute_distances(self, coords: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix."""
        n = coords.shape[0]
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                
        return distances
    
    def _apply_random_augmentation(self, coords: np.ndarray, coord_range: int) -> np.ndarray:
        """Apply random augmentation to coordinates."""
        augmented = coords.copy()
        
        # Random rotation (0 to 2π)
        if np.random.random() < 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            augmented = self.augmentation.rotate_instance(augmented, angle)
        
        # Random scaling (0.8 to 1.2)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            augmented = self.augmentation.scale_instance(augmented, scale)
        
        # Random flip
        if np.random.random() < 0.3:
            axis = np.random.randint(0, 2)
            augmented = self.augmentation.flip_instance(augmented, axis)
        
        # Small noise (std = 1% of coordinate range)
        if np.random.random() < 0.4:
            noise_std = 0.01
            augmented = self.augmentation.add_noise(augmented, noise_std)
        
        # Ensure coordinates stay in valid range
        augmented = np.clip(augmented, 0, 1)
        
        return augmented
    
    def generate_batch(self, 
                      batch_size: int,
                      num_customers: int,
                      capacity: int,
                      coord_range: int,
                      demand_range: List[int],
                      base_seed: int = None,
                      instance_types: List[InstanceType] = None,
                      augmentation_prob: float = 0.3) -> List[Dict[str, Any]]:
        """Generate a batch of diverse CVRP instances."""
        
        if instance_types is None:
            instance_types = [InstanceType.RANDOM, InstanceType.CLUSTERED, 
                            InstanceType.RADIAL, InstanceType.MIXED]
        
        instances = []
        for i in range(batch_size):
            # Choose random instance type
            instance_type = np.random.choice(instance_types)
            
            # Decide on augmentation
            apply_aug = np.random.random() < augmentation_prob
            
            # Generate seed
            seed = (base_seed + i) if base_seed is not None else None
            
            # Generate instance
            instance = self.generate_instance(
                num_customers=num_customers,
                capacity=capacity,
                coord_range=coord_range,
                demand_range=demand_range,
                seed=seed,
                instance_type=instance_type,
                apply_augmentation=apply_aug
            )
            
            instances.append(instance)
            
        return instances


class CurriculumScheduler:
    """Curriculum learning scheduler for progressive difficulty."""
    
    def __init__(self, 
                 start_customers: int = 10,
                 max_customers: int = 50,
                 progression_epochs: int = 100):
        self.start_customers = start_customers
        self.max_customers = max_customers
        self.progression_epochs = progression_epochs
        
    def get_num_customers(self, epoch: int) -> int:
        """Get number of customers for current epoch."""
        progress = min(1.0, epoch / self.progression_epochs)
        # Smooth progression using cosine schedule
        smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
        
        num_customers = self.start_customers + int(
            (self.max_customers - self.start_customers) * smooth_progress
        )
        
        return num_customers
    
    def get_difficulty_params(self, epoch: int) -> Dict[str, Any]:
        """Get difficulty parameters for current epoch."""
        progress = min(1.0, epoch / self.progression_epochs)
        
        return {
            'num_customers': self.get_num_customers(epoch),
            'instance_types': self._get_instance_types(progress),
            'augmentation_prob': 0.2 + 0.3 * progress,  # Increase augmentation over time
            'capacity_scaling': 1.0 + 0.5 * progress,   # Slightly increase capacity
        }
    
    def _get_instance_types(self, progress: float) -> List[InstanceType]:
        """Get instance types based on training progress."""
        if progress < 0.3:
            return [InstanceType.RANDOM]
        elif progress < 0.6:
            return [InstanceType.RANDOM, InstanceType.CLUSTERED]
        elif progress < 0.8:
            return [InstanceType.RANDOM, InstanceType.CLUSTERED, InstanceType.RADIAL]
        else:
            return [InstanceType.RANDOM, InstanceType.CLUSTERED, 
                   InstanceType.RADIAL, InstanceType.MIXED]


def create_enhanced_data_generator(config: Dict[str, Any], 
                                 use_curriculum: bool = False) -> Callable:
    """Create an enhanced data generator function."""
    
    generator = EnhancedCVRPGenerator(config)
    
    # Setup curriculum if requested
    curriculum = None
    if use_curriculum:
        curriculum = CurriculumScheduler(
            start_customers=max(5, config['problem']['num_customers'] // 2),
            max_customers=config['problem']['num_customers'],
            progression_epochs=config['training']['num_epochs']
        )
    
    def data_generator_func(batch_size: int, epoch: int = 1, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Enhanced data generator function with curriculum and augmentation."""
        
        # Get base parameters
        base_customers = config['problem']['num_customers']
        capacity = config['problem']['vehicle_capacity']
        coord_range = config['problem']['coord_range']
        demand_range = config['problem']['demand_range']
        
        # Get configured instance types
        config_types = config['problem'].get('instance_types', ['random'])
        configured_instance_types = []
        for type_str in config_types:
            type_str = type_str.upper()
            if hasattr(InstanceType, type_str):
                configured_instance_types.append(getattr(InstanceType, type_str))
        
        # Apply curriculum if enabled
        if curriculum is not None:
            difficulty_params = curriculum.get_difficulty_params(epoch)
            num_customers = difficulty_params['num_customers']
            instance_types = difficulty_params['instance_types']
            augmentation_prob = difficulty_params['augmentation_prob']
            capacity = int(capacity * difficulty_params['capacity_scaling'])
        else:
            num_customers = base_customers
            # Use configured instance types or default to RANDOM only
            instance_types = configured_instance_types if configured_instance_types else [InstanceType.RANDOM]
            augmentation_prob = 0.3
        
        # Generate batch
        instances = generator.generate_batch(
            batch_size=batch_size,
            num_customers=num_customers,
            capacity=capacity,
            coord_range=coord_range,
            demand_range=demand_range,
            base_seed=seed if seed is not None else epoch * 1000,
            instance_types=instance_types,
            augmentation_prob=augmentation_prob
        )
        
        return instances
    
    return data_generator_func
