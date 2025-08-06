import pickle
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def nearest_neighbor_cvrp(coordinates, demands, capacity):
    """
    Simple nearest neighbor heuristic for CVRP
    """
    n = len(coordinates)
    depot = 0
    unvisited = set(range(1, n))  # Exclude depot
    routes = []
    total_distance = 0
    
    while unvisited:
        route = [depot]
        current_load = 0
        current_node = depot
        route_distance = 0
        
        while True:
            # Find nearest unvisited customer that fits in vehicle capacity
            best_next = None
            best_distance = float('inf')
            
            for customer in unvisited:
                if current_load + demands[customer] <= capacity:
                    dist = euclidean_distance(coordinates[current_node], coordinates[customer])
                    if dist < best_distance:
                        best_distance = dist
                        best_next = customer
            
            if best_next is None:
                # No more customers can fit, return to depot
                break
                
            # Move to next customer
            route.append(best_next)
            current_load += demands[best_next]
            route_distance += best_distance
            unvisited.remove(best_next)
            current_node = best_next
        
        # Return to depot
        route.append(depot)
        route_distance += euclidean_distance(coordinates[current_node], coordinates[depot])
        
        routes.append({
            'route': route,
            'distance': route_distance,
            'load': current_load
        })
        total_distance += route_distance
    
    return routes, total_distance

def solve_nn_instance(data_instance, instance_id):
    """Solve a single CVRP instance using Nearest Neighbor"""
    print(f"\n{'='*50}")
    print(f"Solving Instance {instance_id + 1} with Nearest Neighbor")
    print(f"{'='*50}")
    
    try:
        coordinates = data_instance.x.numpy()
        demands = data_instance.demand.numpy().flatten()
        capacity = data_instance.capacity.item()
        
        print(f"Nodes: {len(coordinates)} (1 depot + {len(coordinates)-1} customers)")
        print(f"Vehicle capacity: {capacity:.1f}")
        print(f"Total demand: {np.sum(demands[1:]):.1f}")
        print(f"Min vehicles needed: {int(np.ceil(np.sum(demands[1:]) / capacity))}")
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None
    
    # Solve using nearest neighbor
    routes, total_distance = nearest_neighbor_cvrp(coordinates, demands, capacity)
    
    # Print solution
    print(f"\nNearest Neighbor Solution:")
    print(f"Number of vehicles used: {len(routes)}")
    
    for i, route_info in enumerate(routes):
        route = route_info['route']
        distance = route_info['distance']
        load = route_info['load']
        
        route_str = " -> ".join(map(str, route))
        print(f"Vehicle {i+1}: {route_str}")
        print(f"  Distance: {distance:.3f}, Load: {load:.1f}")
    
    print(f"\nTotal distance: {total_distance:.3f}")
    
    return {
        'instance_id': instance_id + 1,
        'status': 'solved',
        'total_distance': total_distance,
        'num_vehicles': len(routes),
        'num_nodes': len(coordinates),
        'total_demand': np.sum(demands[1:]),
        'capacity': capacity,
        'routes': routes
    }

def main():
    """Main function to test Nearest Neighbor on all instances"""
    print("Testing Nearest Neighbor Heuristic on CVRP instances")
    print("=" * 60)
    
    # Load dataset
    try:
        with open('data/cvrp20.pkl', 'rb') as f:
            data_list = pickle.load(f)
        print(f"Loaded {len(data_list)} instances from cvrp20.pkl")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Test all instances
    results = []
    
    for i, data_instance in enumerate(data_list):
        result = solve_nn_instance(data_instance, i)
        if result:
            results.append(result)
    
    # Summary statistics
    print("\n" + "="*60)
    print("NEAREST NEIGHBOR SUMMARY RESULTS")
    print("="*60)
    
    if results:
        distances = [r['total_distance'] for r in results]
        vehicles_used = [r['num_vehicles'] for r in results]
        
        print(f"Successfully solved: {len(results)}/{len(data_list)} instances")
        print(f"Average total distance: {np.mean(distances):.3f}")
        print(f"Best (shortest) distance: {np.min(distances):.3f}")
        print(f"Worst (longest) distance: {np.max(distances):.3f}")
        print(f"Standard deviation: {np.std(distances):.3f}")
        print(f"Average vehicles used: {np.mean(vehicles_used):.1f}")
        
        print("\nDetailed results:")
        for r in results:
            print(f"Instance {r['instance_id']:2d}: Distance: {r['total_distance']:6.3f} | "
                  f"Vehicles: {r['num_vehicles']:2d} | Demand: {r['total_demand']:4.1f} | "
                  f"Capacity: {r['capacity']:.1f}")
    else:
        print("No instances were successfully solved!")

if __name__ == '__main__':
    main()
