import numpy as np
from typing import List, Tuple, Dict

def validate_cvrp_route(route: List[int], demands: np.ndarray, capacity: float, n_customers: int) -> Dict:
    """
    Comprehensive CVRP route validation function.
    
    Args:
        route: List of node indices (including depot visits)
        demands: Array of demands [depot_demand, customer1_demand, ...]
        capacity: Vehicle capacity
        n_customers: Number of customers (excluding depot)
    
    Returns:
        Dict with validation results and details
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Basic checks
    if not route:
        result['valid'] = False
        result['errors'].append("Empty route")
        return result
    
    if route[0] != 0:
        result['valid'] = False
        result['errors'].append("Route must start at depot (node 0)")
    
    if route[-1] != 0:
        result['valid'] = False
        result['errors'].append("Route must end at depot (node 0)")
    
    # Check all customers served exactly once
    customers_in_route = [node for node in route if node != 0]
    unique_customers = set(customers_in_route)
    
    if len(customers_in_route) != len(unique_customers):
        duplicates = [x for x in customers_in_route if customers_in_route.count(x) > 1]
        result['valid'] = False
        result['errors'].append(f"Duplicate customers served: {set(duplicates)}")
    
    expected_customers = set(range(1, n_customers + 1))
    if unique_customers != expected_customers:
        missing = expected_customers - unique_customers
        extra = unique_customers - expected_customers
        if missing:
            result['valid'] = False
            result['errors'].append(f"Missing customers: {missing}")
        if extra:
            result['valid'] = False
            result['errors'].append(f"Extra/invalid customers: {extra}")
    
    # Check for consecutive depot visits (inefficient but not invalid)
    consecutive_depots = []
    for i in range(len(route) - 1):
        if route[i] == 0 and route[i + 1] == 0:
            consecutive_depots.append(i)
    
    if consecutive_depots:
        result['warnings'].append(f"Consecutive depot visits at positions: {consecutive_depots}")
    
    # Capacity validation - split route into tours
    tours = []
    current_tour = []
    
    for i, node in enumerate(route):
        if node == 0 and current_tour:  # End of tour
            tours.append(current_tour)
            current_tour = []
        elif node != 0:  # Customer
            current_tour.append(node)
    
    # Handle case where route doesn't end at depot properly
    if current_tour:
        tours.append(current_tour)
    
    capacity_violations = []
    tour_demands = []
    
    for tour_idx, tour in enumerate(tours):
        tour_demand = sum(demands[customer] for customer in tour)
        tour_demands.append(tour_demand)
        
        if tour_demand > capacity:
            capacity_violations.append({
                'tour': tour_idx + 1,
                'nodes': tour,
                'demand': tour_demand,
                'capacity': capacity,
                'violation': tour_demand - capacity
            })
    
    if capacity_violations:
        result['valid'] = False
        result['errors'].append(f"Capacity violations in {len(capacity_violations)} tours")
        result['capacity_violations'] = capacity_violations
    
    # Calculate statistics
    total_distance = 0
    if len(route) > 1:
        # This would need coordinate information to calculate actual distance
        # For now, just count segments
        result['stats']['total_segments'] = len(route) - 1
        result['stats']['depot_visits'] = route.count(0)
        result['stats']['num_tours'] = len(tours)
        result['stats']['avg_tour_demand'] = np.mean(tour_demands) if tour_demands else 0
        result['stats']['max_tour_demand'] = max(tour_demands) if tour_demands else 0
        result['stats']['tour_demands'] = tour_demands
    
    return result

def print_validation_report(validation_result: Dict, route: List[int] = None):
    """Print a human-readable validation report."""
    result = validation_result
    
    print("=" * 50)
    print("CVRP ROUTE VALIDATION REPORT")
    print("=" * 50)
    
    if route:
        print(f"Route: {route}")
        print(f"Route length: {len(route)}")
    
    print(f"Status: {'✓ VALID' if result['valid'] else '✗ INVALID'}")
    
    if result['errors']:
        print("\nERRORS:")
        for error in result['errors']:
            print(f"  ✗ {error}")
    
    if result['warnings']:
        print("\nWARNINGS:")
        for warning in result['warnings']:
            print(f"  ⚠ {warning}")
    
    if 'capacity_violations' in result:
        print(f"\nCAPACITY VIOLATIONS:")
        for violation in result['capacity_violations']:
            print(f"  Tour {violation['tour']}: nodes {violation['nodes']}")
            print(f"    Demand: {violation['demand']:.2f}, Capacity: {violation['capacity']:.2f}")
            print(f"    Violation: {violation['violation']:.2f}")
    
    if result['stats']:
        print(f"\nSTATISTICS:")
        stats = result['stats']
        for key, value in stats.items():
            if key != 'tour_demands':
                print(f"  {key}: {value}")
        if 'tour_demands' in stats:
            print(f"  Tour demands: {[f'{d:.1f}' for d in stats['tour_demands']]}")
    
    print("=" * 50)

# Test the validator with the current routes
if __name__ == "__main__":
    import torch
    import json
    
    # Load current test instance
    customers = 20
    n = customers + 1
    device = torch.device('cpu')
    test_seed = 42
    
    g = torch.Generator(device=device)
    g.manual_seed(test_seed)
    coords = (torch.randint(1, 101, (n, 2), generator=g, device=device, dtype=torch.int64).float() / 100.0)
    demands = (torch.randint(1, 11, (n, 1), generator=g, device=device, dtype=torch.int64).float() / 10.0)
    demands[0] = 0.0
    
    dvals = demands.detach().cpu().numpy().reshape(-1)
    
    # Test each model's route
    for model_name in ['dynamic_gt_rl', 'static_rl', 'greedy_baseline']:
        try:
            with open(f'utils/plots/test_instance_route_{model_name}.json', 'r') as f:
                route = json.load(f)
            
            print(f"\nValidating {model_name} route:")
            result = validate_cvrp_route(route, dvals, 3.0, customers)
            print_validation_report(result, route)
            
        except FileNotFoundError:
            print(f"Route file not found for {model_name}")
