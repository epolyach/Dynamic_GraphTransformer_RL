import pickle
import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import sys
import os

# Add parent directory to path to import instance_creator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def create_distance_matrix(coordinates):
    """Create distance matrix from coordinates"""
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = euclidean_distance(coordinates[i], coordinates[j])
    
    # Scale distances for OR-Tools (convert to integers)
    distance_matrix = (distance_matrix * 1000).astype(int)
    return distance_matrix

def create_data_model(coordinates, demands, capacity, num_vehicles=4):
    """Create data model for OR-Tools"""
    data = {}
    data['distance_matrix'] = create_distance_matrix(coordinates)
    data['demands'] = [int(d * 10) for d in demands]  # Scale demands
    data['vehicle_capacities'] = [int(capacity * 10)] * num_vehicles  # Scale capacity
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, solution):
    """Print solution details"""
    print(f'Objective: {solution.ObjectiveValue() / 1000:.2f} distance units')
    print(f'Number of vehicles used: {solution.ObjectiveValue() // 1000000 if solution.ObjectiveValue() > 1000000 else data["num_vehicles"]}')
    
    total_distance = 0
    total_load = 0
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for vehicle {vehicle_id}:\n'
        route_distance = 0
        route_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += f' {node_index} Load({route_load}) -> '
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
        plan_output += f' {manager.IndexToNode(index)} Load({route_load})\n'
        plan_output += f'Distance of the route: {route_distance / 1000:.2f}\n'
        plan_output += f'Load of the route: {route_load / 10:.1f}\n'
        print(plan_output)
        
        total_distance += route_distance
        total_load += route_load
        
    print(f'Total distance of all routes: {total_distance / 1000:.2f}')
    print(f'Total load of all routes: {total_load / 10:.1f}')
    return total_distance / 1000

def solve_cvrp_instance(data_instance, instance_id):
    """Solve a single CVRP instance using Clarke-Wright"""
    print(f"\n{'='*50}")
    print(f"Solving Instance {instance_id + 1}")
    print(f"{'='*50}")
    
    # Extract data from PyTorch Geometric format
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
    
    # Create data model
    data = create_data_model(coordinates, demands, capacity)
    
    # Create routing manager and model
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], 
                                           data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    
    # Create distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    
    # Set search parameters - Clarke-Wright Savings Algorithm
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
    search_parameters.time_limit.seconds = 30  # 30 second time limit
    
    # Solve the problem
    print("Solving with Clarke-Wright Savings Algorithm...")
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        total_distance = print_solution(data, manager, routing, solution)
        return {
            'instance_id': instance_id + 1,
            'status': 'solved',
            'total_distance': total_distance,
            'objective_value': solution.ObjectiveValue(),
            'num_nodes': len(coordinates),
            'total_demand': np.sum(demands[1:]),
            'capacity': capacity
        }
    else:
        print('No solution found!')
        return {
            'instance_id': instance_id + 1,
            'status': 'no_solution',
            'total_distance': None,
            'objective_value': None,
            'num_nodes': len(coordinates),
            'total_demand': np.sum(demands[1:]),
            'capacity': capacity
        }

def main():
    """Main function to test Clarke-Wright on all instances"""
    print("Testing Clarke-Wright Savings Algorithm on CVRP instances")
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
        result = solve_cvrp_instance(data_instance, i)
        if result:
            results.append(result)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    solved_instances = [r for r in results if r['status'] == 'solved']
    
    if solved_instances:
        distances = [r['total_distance'] for r in solved_instances]
        print(f"Successfully solved: {len(solved_instances)}/{len(results)} instances")
        print(f"Average total distance: {np.mean(distances):.2f}")
        print(f"Best (shortest) distance: {np.min(distances):.2f}")
        print(f"Worst (longest) distance: {np.max(distances):.2f}")
        print(f"Standard deviation: {np.std(distances):.2f}")
        
        print("\nDetailed results:")
        for r in results:
            status_str = "✅ SOLVED" if r['status'] == 'solved' else "❌ FAILED"
            dist_str = f"{r['total_distance']:.2f}" if r['total_distance'] else "N/A"
            print(f"Instance {r['instance_id']:2d}: {status_str} | Distance: {dist_str:>8} | "
                  f"Nodes: {r['num_nodes']:2d} | Demand: {r['total_demand']:.1f} | Capacity: {r['capacity']:.1f}")
    else:
        print("No instances were successfully solved!")

if __name__ == '__main__':
    main()
