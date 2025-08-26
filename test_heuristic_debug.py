#!/usr/bin/env python3
"""Debug heuristic solver for N=20"""
import sys
import numpy as np
sys.path.append('research/benchmark_exact')
from enhanced_generator import EnhancedCVRPGenerator, InstanceType

# Generate a test instance for N=20
gen = EnhancedCVRPGenerator(config={})
instance = gen.generate_instance(
    num_customers=20,
    capacity=30,
    coord_range=100,
    demand_range=[1, 10],
    seed=42,
    instance_type=InstanceType.RANDOM,
    apply_augmentation=False,
)

coords = instance['coords']
demands = instance['demands']
capacity = instance['capacity']
n_customers = len(coords) - 1

total_demand = int(sum(int(d) for d in demands[1:]))
min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
max_vehicles_old = min(n_customers, min_vehicles + 2)
max_vehicles_new = min(n_customers, max(min_vehicles + 3, int(min_vehicles * 1.5)))

print(f"N = {n_customers}")
print(f"Total demand: {total_demand}")
print(f"Capacity: {capacity}")
print(f"Min vehicles needed: {min_vehicles}")
print(f"Max vehicles (old formula): {max_vehicles_old}")
print(f"Max vehicles (new formula): {max_vehicles_new}")
print(f"Will try vehicles from {min_vehicles} to {max_vehicles_new}")

# Now let's manually test OR-Tools with different vehicle counts
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

for n_vehicles in range(min_vehicles, min(n_customers, min_vehicles + 10)):
    print(f"\nTrying with {n_vehicles} vehicles...")
    
    try:
        manager = pywrapcp.RoutingIndexManager(len(coords), n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Scale distances for integer arithmetic
        scale = 10000
        scaled_distances = (instance['distances'] * scale).astype(int)
        
        def distance_callback(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return scaled_distances[i][j]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(demands[from_node])
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [int(capacity)] * n_vehicles,
            True,
            'Capacity'
        )
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 5
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            cost = solution.ObjectiveValue() / scale
            print(f"  ✓ Found solution with cost: {cost:.4f}")
            break
        else:
            print(f"  ✗ No solution found")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
