import torch
import numpy as np

def generate_cvrp_instance(num_customers=15, capacity=3, coord_range=50, demand_range=(1, 3), seed=None):
    if seed is not None:
        np.random.seed(seed)
    coords = np.random.uniform(0, coord_range, (num_customers + 1, 2))
    coords[0] = [coord_range/2, coord_range/2]
    demands = np.zeros(num_customers + 1)
    demands[1:] = np.random.randint(demand_range[0], demand_range[1] + 1, num_customers)
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    return {'coords': coords, 'demands': demands, 'distances': distances, 'capacity': capacity}

def compute_route_cost(route, distances):
    if len(route) <= 1:
        return 0.0
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distances[route[i], route[i + 1]]
    return cost

def compute_naive_baseline_cost(instance):
    distances = instance['distances']
    n_customers = len(instance['coords']) - 1
    naive_cost = 0.0
    for customer_idx in range(1, n_customers + 1):
        naive_cost += distances[0, customer_idx] * 2
    return naive_cost

# Test with a few instances
print("Testing cost calculations:")
for seed in [0, 1, 2]:
    instance = generate_cvrp_instance(seed=seed)
    naive = compute_naive_baseline_cost(instance)
    
    # Test some routes to see what could give low costs
    short_route = [1, 0, 2, 0, 3, 0]  # Visit first 3 customers individually  
    incomplete_route = [1, 2, 3, 0]   # Visit first 3 customers in one trip
    very_short = [1, 0]               # Just visit one customer
    
    short_cost = compute_route_cost(short_route, instance['distances'])
    incomplete_cost = compute_route_cost(incomplete_route, instance['distances'])
    very_short_cost = compute_route_cost(very_short, instance['distances'])
    
    print(f"\nSeed {seed}:")
    print(f"  Naive baseline: {naive:.1f}")
    print(f"  Short route {short_route} cost: {short_cost:.1f}")
    print(f"  Incomplete route {incomplete_route} cost: {incomplete_cost:.1f}")
    print(f"  Very short route {very_short} cost: {very_short_cost:.1f}")
    
    # Check if a route visiting only a few customers could get ~60
    if very_short_cost < 100:
        print(f"  *** Very short route gives cost {very_short_cost:.1f} - this could explain low validation costs!")
