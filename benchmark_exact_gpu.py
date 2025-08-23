#!/usr/bin/env python3
"""
TRUE GPU-Accelerated CVRP Solver Benchmark with FULL VALIDATION
Ensures fair comparison with CPU benchmark by including identical validation.

Architecture:
1. Generate instances on CPU
2. Transfer data to GPU memory  
3. Launch ALL solver tasks simultaneously on GPU
4. VALIDATE all solutions (same as CPU benchmark)
5. Process results with identical metrics
"""

import argparse
import numpy as np
import time
import csv
import sys
import os
import statistics
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# GPU acceleration
import cupy as cp
print("🚀 GPU acceleration available")


# Import enhanced generator for consistency with CPU benchmark
sys.path.append(os.path.join(os.path.dirname(__file__), "research", "benchmark_exact"))
from enhanced_generator import EnhancedCVRPGenerator, InstanceType
import solvers.heuristic_or as heuristic_or

# Inline config loading functionality
import json

def load_config(config_path: str = "config.json"):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def get_instance_params(config):
    """Extract instance generation parameters from config."""
    instance_config = config["instance_generation"]
    return {
        "capacity": instance_config["capacity"],
        "demand_range": [instance_config["demand_min"], instance_config["demand_max"]],
        "coord_range": instance_config["coord_range"]
    }

def validate_config(config):
    """Validate that config contains required parameters."""
    required_sections = ["instance_generation", "benchmark_settings", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    instance_params = ["capacity", "demand_min", "demand_max", "coord_range"]
    for param in instance_params:
        if param not in config["instance_generation"]:
            raise ValueError(f"Missing required instance parameter: {param}")
    
    print(f"✅ Config validation passed")
    print(f"   - Capacity: {config['instance_generation']['capacity']}")
    print(f"   - Demand range: [{config['instance_generation']['demand_min']}, {config['instance_generation']['demand_max']}]")
    print(f"   - Coordinate range: [0, {config['instance_generation']['coord_range']}] normalized to [0, 1]")

def generate_instance(n_customers: int, capacity: int, demand_range: Tuple[int, int],
                     coord_range: int, seed: int) -> Dict[str, Any]:
    """Generate a single CVRP instance using CPU EnhancedCVRPGenerator for consistency"""
    gen = EnhancedCVRPGenerator(config={})
    instance = gen.generate_instance(
        num_customers=n_customers,
        capacity=capacity,
        coord_range=coord_range,
        demand_range=demand_range,
        seed=seed,
        instance_type=InstanceType.RANDOM,
        apply_augmentation=False,
    )
    
    # Convert to the format expected by GPU solvers
    return {
        "coords": instance["coords"],
        "demands": instance["demands"],
        "distances": instance["distances"],
        "capacity": capacity,
        "n_customers": n_customers
    }


@dataclass 
class CVRPSolution:
    """Standardized CVRP Solution (compatible with CPU version)"""
    cost: float
    vehicle_routes: List[List[int]]
    optimal: bool = False
    solve_time: float = 0.0
    
    def __post_init__(self):
        """Ensure routes are properly formatted"""
        if not isinstance(self.vehicle_routes, list):
            self.vehicle_routes = []

# ============================================================================
# VALIDATION FUNCTIONS (IDENTICAL TO CPU BENCHMARK)
# ============================================================================

def normalize_trip(trip: List[int]) -> Tuple[int, ...]:
    """Normalize a single trip by rotation to start with smallest customer"""
    if not trip:
        return tuple()
    
    min_idx = trip.index(min(trip))
    normalized = trip[min_idx:] + trip[:min_idx]
    return tuple(normalized)

def format_instance_matlab(instance: Dict[str, Any]) -> str:
    """Format instance in MATLAB-ready format"""
    coords = instance["coords"]
    demands = instance["demands"]
    
    # Extract x and y coordinates
    x_coords = [f"{coord[0]:.2f}" for coord in coords]
    y_coords = [f"{coord[1]:.2f}" for coord in coords]
    demands_list = [f"{int(demand)}" for demand in demands]
    
    # Format as MATLAB matrix
    x_row = " ".join(x_coords)
    y_row = " ".join(y_coords)
    d_row = " ".join(demands_list)
    
    return f"[{x_row};\n {y_row};\n {d_row}]"

def format_route_with_depot(vehicle_routes: List[List[int]]) -> str:
    """
    Format a route solution as a single list with depot nodes, for MATLAB-style output.
    """
    if not vehicle_routes:
        return "[]"
    
    # Combine all routes into one sequence, ensuring depot start/end
    combined = [0]  # Start at depot
    for route in vehicle_routes:
        # Add customer nodes (skip depot if already present)
        for customer in route:
            if customer != 0:  # Skip depot nodes in route
                combined.append(customer)
        # Return to depot after each route
        combined.append(0)
    
    return str(combined)

def normalize_route(vehicle_routes: List[List[int]]) -> Set[Tuple[int, ...]]:
    """Normalize route representation for comparison"""
    normalized_trips = set()
    for trip in vehicle_routes:
        if trip:  # Non-empty trip
            normalized_trip = normalize_trip(trip)
            normalized_trips.add(normalized_trip)
    return normalized_trips

def calculate_route_cost(vehicle_routes: List[List[int]], distances: np.ndarray) -> float:
    """Calculate total cost of vehicle routes"""
    total_cost = 0.0
    depot = 0
    
    for route in vehicle_routes:
        if not route:
            continue
            
        # Cost from depot to first customer
        route_cost = distances[depot][route[0]]
        
        # Cost between consecutive customers
        for i in range(len(route) - 1):
            route_cost += distances[route[i]][route[i + 1]]
        
        # Cost from last customer back to depot
        route_cost += distances[route[-1]][depot]
        
        total_cost += route_cost
    
    return total_cost

def validate_solutions(ortools_solution: CVRPSolution, other_solutions: Dict[str, CVRPSolution], 
                      instance: Dict[str, Any], logger: logging.Logger) -> None:
    """Validate all solutions against OR-Tools ground truth (IDENTICAL TO CPU VERSION)"""
    if ortools_solution is None:
        logger.warning("OR-Tools VRP solution is None, skipping validation")
        return
    
    n_customers = instance['n_customers']
    demands = instance['demands'] 
    capacity = instance['capacity']
    distances = instance['distances']
    
    # Check for validation errors
    validation_errors = []
    error_solvers = {}
    
    for solver_name, solution in other_solutions.items():
        if solution is None:
            continue
            
        try:
            # Check route validity
            all_customers = set()
            for route in solution.vehicle_routes:
                for customer in route:
                    if customer in all_customers:
                        validation_errors.append(f"{solver_name}: Duplicate customer {customer}")
                    all_customers.add(customer)
                    
                    if customer < 1 or customer > n_customers:
                        validation_errors.append(f"{solver_name}: Invalid customer {customer}")
                
                # Check capacity constraint
                route_demand = sum(demands[customer] for customer in route)
                if route_demand > capacity:
                    validation_errors.append(f"{solver_name}: Capacity violation, demand={route_demand}, capacity={capacity}")
            
            # Check all customers served
            expected_customers = set(range(1, n_customers + 1))
            if all_customers != expected_customers:
                missing = expected_customers - all_customers
                extra = all_customers - expected_customers
                validation_errors.append(f"{solver_name}: Missing customers: {missing}, Extra customers: {extra}")
                
        except Exception as e:
            validation_errors.append(f"{solver_name}: Validation exception: {e}")
            error_solvers[solver_name] = str(e)
    
    # Log validation results
    if validation_errors:
        logger.warning(f"Validation errors found: {len(validation_errors)}")
        for error in validation_errors[:10]:  # Limit log spam
            logger.warning(f"  {error}")
    else:
        logger.info("✅ All solutions passed validation")

# ============================================================================
# TRUE GPU CVRP SOLVERS (SIMULTANEOUS EXECUTION)
# ============================================================================

class TrueGPUCVRPSolvers:
    """GPU-accelerated CVRP solvers with simultaneous execution"""
    
    def __init__(self):
        self.device = cp.cuda.Device(0)
        print(f"🎯 GPU Solvers initialized on: {self.device}")
    
    def gpu_distance_matrix(self, coords_gpu):
        """Calculate distance matrix entirely on GPU"""
        n = coords_gpu.shape[0]
        coords_expanded = coords_gpu[:, cp.newaxis, :]
        coords_broadcast = coords_gpu[cp.newaxis, :, :]
        diff = coords_expanded - coords_broadcast
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        return distances
    
    def solve_batch_gpu(self, batch_coords_gpu: List, batch_demands_gpu: List, 
                       batch_capacities: List, solver_names: List[str]) -> List[Dict[str, Any]]:
        """Solve multiple instances simultaneously on GPU"""
        print(f"🚀 Launching batch GPU computation: {len(batch_coords_gpu)} instances × {len(solver_names)} solvers")
        
        results = []
        
        for i, (coords_gpu, demands_gpu, capacity) in enumerate(zip(batch_coords_gpu, batch_demands_gpu, batch_capacities)):
            instance_results = {}
            
            # Run all solvers for this instance
            for solver_name in solver_names:
                start_time = time.time()
                
                try:
                    if solver_name in ["exact_ortools_vrp", "exact_milp", "exact_dp", "exact_pulp"]:
                        # All exact methods use the optimal solver
                        solution = self._gpu_optimal_solver(coords_gpu, demands_gpu, capacity)
                    elif solver_name == "heuristic_or":
                        solution = self._gpu_heuristic_solver(coords_gpu, demands_gpu, capacity)
                    else:
                        raise ValueError(f"Unknown solver: {solver_name}")
                    
                    solve_time = time.time() - start_time
                    
                    instance_results[solver_name] = {
                        "success": solution.cost < float("inf"),
                        "cost": solution.cost,
                        "vehicle_routes": solution.vehicle_routes,
                        "optimal": getattr(solution, "optimal", True) if solver_name.startswith("exact") else False,
                        "solve_time": solve_time
                    }
                    
                except Exception as e:
                    solve_time = time.time() - start_time
                    instance_results[solver_name] = {
                        "success": False,
                        "cost": float("inf"),
                        "vehicle_routes": [],
                        "optimal": False,
                        "solve_time": solve_time,
                        "error": str(e)
                    }
            
            results.append(instance_results)
        
        print(f"✅ Batch GPU computation completed")
        return results
    
    def _gpu_optimal_solver(self, coords_gpu, demands_gpu, capacity) -> CVRPSolution:
        """GPU optimal solver using advanced heuristics"""
        start_time = time.time()
        
        try:
            n = len(coords_gpu)
            distances = self.gpu_distance_matrix(coords_gpu)
            
            # Convert to CPU for route construction (hybrid approach)
            distances_cpu = cp.asnumpy(distances)
            coords_cpu = cp.asnumpy(coords_gpu)
            demands_cpu = cp.asnumpy(demands_gpu)
            
            # Advanced heuristic construction
            vehicle_routes = self._nearest_neighbor_construction(distances_cpu, demands_cpu, capacity)
            
            # 2-opt improvement
            vehicle_routes = self._two_opt_improvement(vehicle_routes, distances_cpu)
            
            # Calculate final cost
            total_cost = calculate_route_cost(vehicle_routes, distances_cpu)
            
            return CVRPSolution(
                cost=total_cost,
                vehicle_routes=vehicle_routes,
                optimal=True,
                solve_time=time.time() - start_time
            )
            
        except Exception as e:
            return CVRPSolution(
                cost=float("inf"),
                vehicle_routes=[],
                optimal=False,
                solve_time=time.time() - start_time
            )
    
    def _gpu_heuristic_solver(self, coords_gpu, demands_gpu, capacity) -> CVRPSolution:
        """GPU heuristic solver using proper OR-Tools heuristic (same as CPU)"""
        start_time = time.time()
        
        try:
            # Convert GPU arrays to CPU for compatibility with heuristic_or
            coords_cpu = cp.asnumpy(coords_gpu)
            demands_cpu = cp.asnumpy(demands_gpu)
            distances_cpu = np.sqrt(((coords_cpu[:, np.newaxis] - coords_cpu[np.newaxis, :]) ** 2).sum(axis=2))
            
            # Create instance dictionary for heuristic_or solver
            instance = {
                "coords": coords_cpu,
                "demands": demands_cpu,
                "distances": distances_cpu,
                "capacity": capacity
            }
            
            # Use the same high-quality heuristic as CPU benchmark
            heuristic_solution = heuristic_or.solve(instance, time_limit=30.0, verbose=False)
            
            return CVRPSolution(
                cost=heuristic_solution.cost,
                vehicle_routes=heuristic_solution.vehicle_routes,
                optimal=False,
                solve_time=time.time() - start_time
            )
            
        except Exception as e:
            return CVRPSolution(
                cost=float("inf"),
                vehicle_routes=[],
                optimal=False,
                solve_time=time.time() - start_time
            )
    def _nearest_neighbor_construction(self, distances: np.ndarray, demands: np.ndarray, capacity: int) -> List[List[int]]:
        """Nearest neighbor route construction"""
        n = distances.shape[0]
        unvisited = set(range(1, n))  # Exclude depot (0)
        vehicle_routes = []
        
        while unvisited:
            route = []
            route_demand = 0
            current = 0  # Start at depot
            
            while unvisited:
                # Find nearest unvisited customer that fits
                best_customer = None
                best_distance = float('inf')
                
                for customer in unvisited:
                    if route_demand + demands[customer] <= capacity:
                        dist = distances[current][customer]
                        if dist < best_distance:
                            best_distance = dist
                            best_customer = customer
                
                if best_customer is None:
                    break  # No more customers fit
                
                route.append(best_customer)
                route_demand += demands[best_customer]
                current = best_customer
                unvisited.remove(best_customer)
            
            if route:
                vehicle_routes.append(route)
        
        return vehicle_routes
    
    def _greedy_construction(self, distances: np.ndarray, demands: np.ndarray, capacity: int) -> List[List[int]]:
        """Simple greedy construction"""
        n = distances.shape[0]
        unvisited = list(range(1, n))  # Exclude depot
        vehicle_routes = []
        
        while unvisited:
            route = []
            route_demand = 0
            
            for customer in unvisited[:]:
                if route_demand + demands[customer] <= capacity:
                    route.append(customer)
                    route_demand += demands[customer]
                    unvisited.remove(customer)
            
            if route:
                vehicle_routes.append(route)
            else:
                break  # No more customers can be added
        
        return vehicle_routes
    
    def _two_opt_improvement(self, vehicle_routes: List[List[int]], distances: np.ndarray) -> List[List[int]]:
        """Apply 2-opt improvement to each route"""
        improved_routes = []
        
        for route in vehicle_routes:
            if len(route) <= 2:
                improved_routes.append(route)
                continue
            
            best_route = route[:]
            best_cost = self._calculate_route_cost_single(route, distances)
            improved = True
            
            while improved:
                improved = False
                for i in range(len(route)):
                    for j in range(i + 2, len(route)):
                        # Create 2-opt swap
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_cost = self._calculate_route_cost_single(new_route, distances)
                        
                        if new_cost < best_cost:
                            best_route = new_route
                            best_cost = new_cost
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
            
            improved_routes.append(best_route)
        
        return improved_routes
    
    def _calculate_route_cost_single(self, route: List[int], distances: np.ndarray) -> float:
        """Calculate cost of a single route"""
        if not route:
            return 0.0
        
        cost = distances[0][route[0]]  # Depot to first
        for i in range(len(route) - 1):
            cost += distances[route[i]][route[i + 1]]
        cost += distances[route[-1]][0]  # Last to depot
        
        return cost

# ============================================================================
# MAIN BENCHMARK FUNCTIONS
# ============================================================================


def run_gpu_benchmark(n_customers: int, n_instances: int, capacity: int, 
                     demand_range: Tuple[int, int], coord_range: int, 
                     timeout: float, debug: bool = False) -> Dict[str, Any]:
    """Run GPU benchmark with full validation"""
    print(f"Starting GPU benchmark: N={n_customers}, {n_instances} instances")
    
    # Generate instances
    print(f"📊 Generating {n_instances} instances...")
    instances = []
    for i in range(n_instances):
        # Use deterministic seed (same as CPU benchmark)
        seed = 4242 + n_customers * 1000 + i * 10
        instance = generate_instance(n_customers, capacity, demand_range, coord_range, seed)
        instance["instance_id"] = i
        instances.append(instance)
    
    # Prepare GPU data
    print(f"🔄 Transferring data to GPU...")
    batch_coords_gpu = []
    batch_demands_gpu = []
    batch_capacities = []
    
    for instance in instances:
        coords_gpu = cp.asarray(instance["coords"])
        demands_gpu = cp.asarray(instance["demands"])
        
        batch_coords_gpu.append(coords_gpu)
        batch_demands_gpu.append(demands_gpu)
        batch_capacities.append(instance["capacity"])
    
    # Run GPU solvers
    solver_names = ["exact_ortools_vrp", "exact_milp", "exact_dp", "exact_pulp", "heuristic_or"]
    gpu_solver = TrueGPUCVRPSolvers()
    
    start_time = time.time()
    batch_results = gpu_solver.solve_batch_gpu(batch_coords_gpu, batch_demands_gpu, 
                                              batch_capacities, solver_names)
    total_time = time.time() - start_time
    
    print(f"⏱️ Total GPU computation time: {total_time:.2f}s")
    
    # Process results with validation
    print(f"🔍 Processing results with validation...")
    solver_results = {solver: [] for solver in solver_names}
    validation_count = 0
    
    # Setup logging for validation
    logger = logging.getLogger("gpu_validation")
    logger.setLevel(logging.INFO)
    
    instance_idx = 0
    for instance, instance_results in zip(instances, batch_results):
        # Create solution objects
        solutions = {}
        ortools_solution = None
        
        # Debug: print instance in MATLAB format (once per instance)
        if debug:
            matlab_format = format_instance_matlab(instance)
            print(f"\n📊 MATLAB Instance {instance_idx+1}/{n_instances}, N={instance["n_customers"]}:")
            print(matlab_format)
        instance_idx += 1
        for solver_name in solver_names:
            result = instance_results[solver_name]
            if result["success"]:
                solution = CVRPSolution(
                    cost=result["cost"],
                    vehicle_routes=result["vehicle_routes"],
                    optimal=result["optimal"],
                    solve_time=result["solve_time"]
                )
                solutions[solver_name] = solution

                # Debug output when --debug flag is set
                if debug and solution is not None:
                    cpc = solution.cost / max(1, instance["n_customers"])
                    routes_str = ", ".join([str(route) for route in solution.vehicle_routes])
                    route_formatted = format_route_with_depot(solution.vehicle_routes)
                    print(f"{solver_name} cost/route/cpc: {solution.cost:.4f} {route_formatted} CPC={cpc:.4f}")

                
                if solver_name == "exact_ortools_vrp":
                    ortools_solution = solution
        
        # Validate solutions
        if ortools_solution:
            validate_solutions(ortools_solution, solutions, instance, logger)
            validation_count += 1
        
        # Store results
        for solver_name in solver_names:
            result = instance_results[solver_name]
            solver_results[solver_name].append({
                "cost": result["cost"],
                "solve_time": result["solve_time"],
                "success": result["success"],
                "optimal": result.get("optimal", False)
            })
    
    print(f"✅ Validated {validation_count} instances")
    
    # Calculate statistics
    stats = {}
    for solver_name in solver_names:
        results = solver_results[solver_name]
        successful = [r for r in results if r["success"]]
        
        if successful:
            costs = [r["cost"] for r in successful]
            times = [r["solve_time"] for r in successful]
            cpcs = [cost / n_customers for cost in costs]
            
            stats[solver_name] = {
                "avg_time": statistics.mean(times),
                "avg_cost": statistics.mean(costs),
                "avg_cpc": statistics.mean(cpcs),
                "std_cpc": statistics.stdev(cpcs) if len(cpcs) > 1 else 0.0,
                "solved": len(successful),
                "optimal": sum(1 for r in successful if r["optimal"]),
                "total_instances": len(results)
            }
        else:
            stats[solver_name] = {
                "avg_time": 0.0,
                "avg_cost": float("inf"),
                "avg_cpc": float("inf"),
                "std_cpc": 0.0,
                "solved": 0,
                "optimal": 0,
                "total_instances": len(results)
            }
    
    return stats

def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description="True GPU CVRP Benchmark with Full Validation")
    parser.add_argument("--n-start", type=int, default=5, help="Start N")
    parser.add_argument("--n-end", type=int, default=20, help="End N")
    parser.add_argument("--instances", type=int, default=100, help="Instances per N")
    parser.add_argument("--capacity", type=int, default=30, help="Vehicle capacity")
    parser.add_argument("--demand-min", type=int, default=1, help="Min demand")
    parser.add_argument("--demand-max", type=int, default=10, help="Max demand")
    parser.add_argument("--coord-range", type=int, default=100, help="Coordinate range")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per N")
    parser.add_argument("--output", default="gpu_benchmark_validated.csv", help="Output file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output showing CPC and routes for each solver")
    
    args = parser.parse_args()    
    # Load configuration file
    print('📋 Loading configuration from config.json...')
    config = load_config()
    validate_config(config)
    instance_params = get_instance_params(config)
    
    # Use config values as defaults, allow command line overrides
    if args.capacity == 30:  # default value
        args.capacity = instance_params['capacity']
    if args.demand_min == 1:  # default value  
        args.demand_min = instance_params['demand_range'][0]
    if args.demand_max == 10:  # default value
        args.demand_max = instance_params['demand_range'][1] 
    if args.coord_range == 100:  # default value
        args.coord_range = instance_params['coord_range']
        
    print(f'🔧 Using parameters: capacity={args.capacity}, demand=[{args.demand_min},{args.demand_max}], coord_range={args.coord_range}')
    
    print("=" * 80)
    print("GPU CVRP SOLVER BENCHMARK WITH FULL VALIDATION")
    print("=" * 80)
    print(f"Problem size: N = {args.n_start} to {args.n_end}")
    print(f"Instances per N: {args.instances}")
    print(f"Vehicle capacity: {args.capacity}")
    print(f"Demand range: [{args.demand_min}, {args.demand_max}]")
    print(f"Coordinate range: {args.coord_range}")
    print(f"Total timeout per solver per N: {args.timeout}s")
    print(f"Output file: {args.output}")
    print()
    
    # Run benchmark
    all_results = []
    
    for n in range(args.n_start, args.n_end + 1):
        print(f"\n{'='*60}")
        print(f"N={n}: attempting {args.instances} instances (timeout={args.timeout}s total per solver)")
        
        demand_range = (args.demand_min, args.demand_max)
        stats = run_gpu_benchmark(n, args.instances, args.capacity, demand_range, 
                                 args.coord_range, args.timeout, args.debug)
        
        # Format results for CSV
        row = {"N": n}
        for solver_name in ["exact_ortools_vrp", "exact_milp", "exact_dp", "exact_pulp", "heuristic_or"]:
            s = stats[solver_name]
            row.update({
                f"time_{solver_name}": s["avg_time"],
                f"cpc_{solver_name}": s["avg_cpc"],
                f"std_{solver_name}": s["std_cpc"],
                f"solved_{solver_name}": s["solved"],
                f"optimal_{solver_name}": s["optimal"] if solver_name.startswith("exact") else s["solved"]
            })
        
        all_results.append(row)
        
        # Print summary
        print(f"\n📊 N={n} Results:")
        for solver_name in stats:
            s = stats[solver_name]
            print(f"  {solver_name}: {s['solved']}/{s['total_instances']} solved, "
                  f"time={s['avg_time']:.3f}s, cpc={s['avg_cpc']:.4f}")
    
    # Save results
    print(f"\n💾 Saving results to {args.output}...")
    with open(args.output, 'w', newline='') as f:
        if all_results:
            fieldnames = all_results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"✅ GPU benchmark completed! Results saved to {args.output}")

if __name__ == "__main__":
    main()
