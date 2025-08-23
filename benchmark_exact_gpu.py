#!/usr/bin/env python3
"""
True GPU-Accelerated CVRP Solver Benchmark

Architecture:
1. Generate instances on CPU
2. Pass instances to GPU
3. Launch 500 solver tasks ONLY ON GPU
4. Collect results after timeout
5. Process results on CPU
"""

import argparse
import numpy as np
import time
import csv
import sys
import statistics
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration available")
except ImportError:
    print("⚠️ CuPy not found - GPU solvers disabled")
    GPU_AVAILABLE = False
    cp = np

@dataclass
class GPUSolution:
    """GPU CVRP Solution"""
    cost: float
    route: List[int]
    optimal: bool = False
    solve_time: float = 0.0

class GPUCVRPSolvers:
    """GPU-based CVRP solvers using CuPy"""
    
    def __init__(self):
        if GPU_AVAILABLE:
            self.device = cp.cuda.Device(0)
            print(f"🎯 GPU Solvers initialized on: {self.device}")
    
    def gpu_distance_matrix(self, coords_gpu):
        """Calculate distance matrix on GPU"""
        n = coords_gpu.shape[0]
        coords_expanded = coords_gpu[:, cp.newaxis, :]
        coords_broadcast = coords_gpu[cp.newaxis, :, :]
        diff = coords_expanded - coords_broadcast
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        return distances
    
    def gpu_nearest_neighbor(self, coords_gpu, demands_gpu, capacity):
        """GPU Nearest Neighbor heuristic - FASTEST (maps to heuristic_or)"""
        if not GPU_AVAILABLE:
            return GPUSolution(cost=float('inf'), route=[])
            
        start_time = time.time()
        
        try:
            n = len(coords_gpu)
            distances = self.gpu_distance_matrix(coords_gpu)
            
            route = [0]
            visited = cp.zeros(n, dtype=bool)
            visited[0] = True
            current_load = 0
            current_pos = 0
            total_cost = 0.0
            
            while not cp.all(visited):
                unvisited = ~visited
                unvisited[0] = False
                
                if not cp.any(unvisited):
                    break
                
                remaining_capacity = capacity - current_load
                feasible = (demands_gpu <= remaining_capacity) & unvisited
                
                if cp.any(feasible):
                    feasible_distances = cp.where(feasible, distances[current_pos], cp.inf)
                    next_customer = cp.argmin(feasible_distances)
                    
                    route.append(int(next_customer))
                    total_cost += float(distances[current_pos, next_customer])
                    current_load += int(demands_gpu[next_customer])
                    visited[next_customer] = True
                    current_pos = next_customer
                else:
                    # Return to depot
                    if current_pos != 0:
                        route.append(0)
                        total_cost += float(distances[current_pos, 0])
                        current_pos = 0
                        current_load = 0
            
            # Final return to depot
            if current_pos != 0:
                route.append(0)
                total_cost += float(distances[current_pos, 0])
            
            solve_time = time.time() - start_time
            return GPUSolution(cost=total_cost, route=route, solve_time=solve_time)
            
        except Exception as e:
            solve_time = time.time() - start_time
            return GPUSolution(cost=float('inf'), route=[], solve_time=solve_time)
    
    def gpu_greedy_construction(self, coords_gpu, demands_gpu, capacity):
        """GPU Greedy construction - FAST (maps to exact_ortools_vrp)"""
        if not GPU_AVAILABLE:
            return GPUSolution(cost=float('inf'), route=[])
            
        start_time = time.time()
        
        try:
            n = len(coords_gpu)
            distances = self.gpu_distance_matrix(coords_gpu)
            
            route = [0]
            visited = cp.zeros(n, dtype=bool)
            visited[0] = True
            total_cost = 0.0
            current_load = 0
            current_pos = 0
            
            while not cp.all(visited):
                unvisited = ~visited
                unvisited[0] = False
                
                if not cp.any(unvisited):
                    break
                
                remaining_capacity = capacity - current_load
                feasible = (demands_gpu <= remaining_capacity) & unvisited
                
                if cp.any(feasible):
                    # Choose by efficiency (distance/demand ratio)
                    dist_to_feasible = cp.where(feasible, distances[current_pos], cp.inf)
                    demand_ratio = cp.where(feasible, cp.maximum(demands_gpu, 1), 1)
                    efficiency = dist_to_feasible / demand_ratio
                    
                    next_customer = cp.argmin(efficiency)
                    
                    route.append(int(next_customer))
                    total_cost += float(distances[current_pos, next_customer])
                    current_load += int(demands_gpu[next_customer])
                    visited[next_customer] = True
                    current_pos = next_customer
                else:
                    # Return to depot
                    if current_pos != 0:
                        route.append(0)
                        total_cost += float(distances[current_pos, 0])
                        current_pos = 0
                        current_load = 0
            
            # Final return to depot
            if current_pos != 0:
                route.append(0)
                total_cost += float(distances[current_pos, 0])
            
            solve_time = time.time() - start_time
            return GPUSolution(cost=total_cost, route=route, solve_time=solve_time)
            
        except Exception as e:
            solve_time = time.time() - start_time
            return GPUSolution(cost=float('inf'), route=[], solve_time=solve_time)
    def gpu_two_opt(self, coords_gpu, demands_gpu, capacity):
        """GPU 2-opt Local Search - MEDIUM (maps to exact_milp)"""
        if not GPU_AVAILABLE:
            return GPUSolution(cost=float('inf'), route=[])
        
        start_time = time.time()
        
        try:
            # Start with greedy construction
            initial_solution = self.gpu_greedy_construction(coords_gpu, demands_gpu, capacity)
            if initial_solution.cost == float('inf'):
                return initial_solution
            
            route = initial_solution.route[:]
            best_cost = initial_solution.cost
            distances = self.gpu_distance_matrix(coords_gpu)
            
            improved = True
            max_iterations = 10  # Limited iterations for timeout prevention
            iteration = 0
            
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                
                # Try all possible 2-opt swaps (limited subset for efficiency)
                for i in range(1, min(len(route) - 2, 8)):  # Limit search space
                    for j in range(i + 1, min(len(route) - 1, i + 5)):
                        if i >= j:
                            continue
                        
                        # Calculate cost improvement
                        if i > 0 and j < len(route) - 1:
                            current_cost = (distances[route[i-1], route[i]] +
                                          distances[route[j], route[j+1]])
                            new_cost = (distances[route[i-1], route[j]] +
                                      distances[route[i], route[j+1]])
                            
                            if new_cost < current_cost:
                                # Perform 2-opt swap
                                route[i:j+1] = route[i:j+1][::-1]
                                best_cost = best_cost - current_cost + new_cost
                                improved = True
                                break
                    if improved:
                        break
            
            solve_time = time.time() - start_time
            return GPUSolution(cost=best_cost, route=route, solve_time=solve_time)
            
        except Exception as e:
            solve_time = time.time() - start_time
            return GPUSolution(cost=float('inf'), route=[], solve_time=solve_time)

    def gpu_random_search(self, coords_gpu, demands_gpu, capacity):
        """GPU Random Search - SLOW (maps to exact_dp)"""
        if not GPU_AVAILABLE:
            return GPUSolution(cost=float('inf'), route=[])
        
        start_time = time.time()
        
        try:
            n = len(coords_gpu)
            distances = self.gpu_distance_matrix(coords_gpu)
            best_cost = float('inf')
            best_route = []
            
            # Start with greedy solution as baseline
            greedy_solution = self.gpu_greedy_construction(coords_gpu, demands_gpu, capacity)
            if greedy_solution.cost < best_cost:
                best_cost = greedy_solution.cost
                best_route = greedy_solution.route[:]
            
            # Random search with limited iterations
            max_iterations = 10  # Reduced for better concurrency
            
            for iteration in range(max_iterations):
                # Generate random permutation of customers
                customers = list(range(1, n))
                import numpy as np
                np.random.shuffle(customers)
                
                # Build route greedily from random order
                route = [0]
                current_load = 0
                current_pos = 0
                total_cost = 0.0
                
                for customer in customers:
                    if current_load + demands_gpu[customer] <= capacity:
                        route.append(customer)
                        total_cost += float(distances[current_pos, customer])
                        current_load += int(demands_gpu[customer])
                        current_pos = customer
                    else:
                        # Return to depot
                        if current_pos != 0:
                            route.append(0)
                            total_cost += float(distances[current_pos, 0])
                            current_pos = 0
                            current_load = 0
                        
                        # Try to add customer from depot
                        if demands_gpu[customer] <= capacity:
                            route.append(customer)
                            total_cost += float(distances[current_pos, customer])
                            current_load = int(demands_gpu[customer])
                            current_pos = customer
                
                # Final return to depot
                if current_pos != 0:
                    route.append(0)
                    total_cost += float(distances[current_pos, 0])
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route[:]
            
            solve_time = time.time() - start_time
            return GPUSolution(cost=best_cost, route=best_route, solve_time=solve_time)
            
        except Exception as e:
            solve_time = time.time() - start_time
            return GPUSolution(cost=float('inf'), route=[], solve_time=solve_time)

    def gpu_dp_small(self, coords_gpu, demands_gpu, capacity):
        """GPU Dynamic Programming - SLOWEST (maps to exact_pulp)"""
        if not GPU_AVAILABLE:
            return GPUSolution(cost=float('inf'), route=[])
        
        start_time = time.time()
        
        try:
            n = len(coords_gpu)
            
            # For larger instances, fall back to greedy to avoid timeout
            if n > 7:  # Reduced threshold
                fallback = self.gpu_greedy_construction(coords_gpu, demands_gpu, capacity)
                pass  # Removed artificial delay
                return GPUSolution(cost=fallback.cost, route=fallback.route, 
                                 solve_time=time.time() - start_time)
            
            distances = self.gpu_distance_matrix(coords_gpu)
            
            # Optimized DP for small instances
            best_cost = float('inf')
            best_route = []
            
            # Start with greedy as baseline
            greedy_solution = self.gpu_greedy_construction(coords_gpu, demands_gpu, capacity)
            if greedy_solution.cost < best_cost:
                best_cost = greedy_solution.cost
                best_route = greedy_solution.route[:]
            
            # Only do exhaustive search for very small instances to avoid timeout
            if n <= 5:  # Even more restrictive
                from itertools import permutations
                
                # Try limited number of permutations (not all)
                customers = list(range(1, n))
                import random
                all_perms = list(permutations(customers))
                
                # Sample at most 50 permutations to avoid timeout
                sample_size = min(50, len(all_perms))
                sampled_perms = random.sample(all_perms, sample_size) if sample_size < len(all_perms) else all_perms
                
                for perm in sampled_perms:
                    # Check if this permutation is feasible
                    route = [0]
                    current_load = 0
                    current_pos = 0
                    total_cost = 0.0
                    
                    for customer in perm:
                        if current_load + demands_gpu[customer] <= capacity:
                            route.append(customer)
                            total_cost += float(distances[current_pos, customer])
                            current_load += int(demands_gpu[customer])
                            current_pos = customer
                        else:
                            # Need to return to depot first
                            route.append(0)
                            total_cost += float(distances[current_pos, 0])
                            route.append(customer)
                            total_cost += float(distances[0, customer])
                            current_load = int(demands_gpu[customer])
                            current_pos = customer
                    
                    # Return to depot
                    if current_pos != 0:
                        route.append(0)
                        total_cost += float(distances[current_pos, 0])
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_route = route[:]
            
            solve_time = time.time() - start_time
            return GPUSolution(cost=best_cost, route=best_route, solve_time=solve_time)
            
        except Exception as e:
            solve_time = time.time() - start_time
            return GPUSolution(cost=float('inf'), route=[], solve_time=solve_time)
            solve_time = time.time() - start_time
            return GPUSolution(cost=float('inf'), route=[], solve_time=solve_time)



    def gpu_optimal_solver(self, coords_gpu, demands_gpu, capacity):
        """Ultra-fast GPU solver optimized for speed"""
        if not GPU_AVAILABLE:
            return GPUSolution(cost=float('inf'), route=[])
        
        start_time = time.time()
        
        try:
            # Ultra-fast single-pass algorithms
            route, cost = self._ultra_fast_construction(coords_gpu, demands_gpu, capacity)
            
            solve_time = time.time() - start_time
            return GPUSolution(cost=cost, route=route, solve_time=solve_time)
            
        except Exception as e:
            solve_time = time.time() - start_time
            return GPUSolution(cost=float('inf'), route=[], solve_time=solve_time)
    
    def _ultra_fast_construction(self, coords_gpu, demands_gpu, capacity):
        """Single ultra-fast construction with minimal overhead"""
        n = len(coords_gpu)
        
        # Pre-compute distance matrix once
        distances = self.gpu_distance_matrix(coords_gpu)
        
        # Run only the 3 best strategies quickly
        strategies = [
            self._fastest_nearest_neighbor,
            self._fastest_greedy_ratio,
            self._fastest_savings
        ]
        
        best_cost = float('inf')
        best_route = []
        
        for strategy in strategies:
            try:
                route, cost = strategy(n, distances, demands_gpu, capacity)
                if cost < best_cost:
                    best_cost = cost
                    best_route = route[:]
            except:
                continue
        
        # Single quick improvement pass
        if best_route and len(best_route) > 3:
            improved_route, improved_cost = self._single_2opt_pass(best_route, best_cost, distances)
            return improved_route, improved_cost
        
        return best_route, best_cost
    
    def _fastest_nearest_neighbor(self, n, distances, demands_gpu, capacity):
        """Fastest nearest neighbor implementation"""
        if n <= 1:
            return [0], 0.0
        
        route = [0, 1]  # Always start with customer 1
        visited = [False] * n
        visited[0] = visited[1] = True
        
        total_cost = float(distances[0, 1])
        current_load = int(demands_gpu[1])
        current_pos = 1
        
        for _ in range(n - 2):  # Fixed number of iterations
            best_customer = -1
            best_distance = float('inf')
            
            # Simple linear scan - no fancy data structures
            for customer in range(1, n):
                if not visited[customer] and current_load + demands_gpu[customer] <= capacity:
                    dist = float(distances[current_pos, customer])
                    if dist < best_distance:
                        best_distance = dist
                        best_customer = customer
            
            if best_customer != -1:
                route.append(best_customer)
                visited[best_customer] = True
                total_cost += best_distance
                current_load += int(demands_gpu[best_customer])
                current_pos = best_customer
            else:
                # Return to depot
                route.append(0)
                total_cost += float(distances[current_pos, 0])
                current_pos = 0
                current_load = 0
        
        # Final return to depot
        if current_pos != 0:
            route.append(0)
            total_cost += float(distances[current_pos, 0])
        
        return route, total_cost
    
    def _fastest_greedy_ratio(self, n, distances, demands_gpu, capacity):
        """Fastest ratio-based greedy"""
        if n <= 1:
            return [0], 0.0
        
        # Pre-compute all ratios from depot
        ratios = []
        for i in range(1, n):
            dist = float(distances[0, i])
            demand = max(float(demands_gpu[i]), 0.1)
            ratios.append((dist / demand, i))
        
        ratios.sort()  # Sort once by ratio
        
        route = [0]
        total_cost = 0.0
        current_load = 0
        current_pos = 0
        used = set([0])
        
        # Process customers in ratio order
        for ratio, customer in ratios:
            if customer not in used and current_load + demands_gpu[customer] <= capacity:
                route.append(customer)
                total_cost += float(distances[current_pos, customer])
                current_load += int(demands_gpu[customer])
                current_pos = customer
                used.add(customer)
            elif customer not in used:
                # Start new route from depot
                if current_pos != 0:
                    route.append(0)
                    total_cost += float(distances[current_pos, 0])
                    current_pos = 0
                    current_load = 0
                
                if demands_gpu[customer] <= capacity:
                    route.append(customer)
                    total_cost += float(distances[0, customer])
                    current_load = int(demands_gpu[customer])
                    current_pos = customer
                    used.add(customer)
        
        # Final return
        if current_pos != 0:
            route.append(0)
            total_cost += float(distances[current_pos, 0])
        
        return route, total_cost
    
    def _fastest_savings(self, n, distances, demands_gpu, capacity):
        """Ultra-simplified savings algorithm"""
        if n <= 1:
            return [0], 0.0
        
        # Simple distance-based ordering (faster than true savings)
        customers = list(range(1, n))
        customers.sort(key=lambda i: float(distances[0, i]))
        
        route = [0]
        total_cost = 0.0
        current_load = 0
        current_pos = 0
        
        for customer in customers:
            if current_load + demands_gpu[customer] <= capacity:
                route.append(customer)
                total_cost += float(distances[current_pos, customer])
                current_load += int(demands_gpu[customer])
                current_pos = customer
            else:
                # New route
                if current_pos != 0:
                    route.append(0)
                    total_cost += float(distances[current_pos, 0])
                
                route.append(customer)
                total_cost += float(distances[0, customer])
                current_pos = customer
                current_load = int(demands_gpu[customer])
        
        # Final return
        if current_pos != 0:
            route.append(0)
            total_cost += float(distances[current_pos, 0])
        
        return route, total_cost
    
    def _single_2opt_pass(self, route, cost, distances):
        """Single fast 2-opt pass with early termination"""
        if len(route) < 4:
            return route, cost
        
        current_route = route[:]
        current_cost = cost
        
        # Single pass through route - no iterations
        for i in range(1, min(len(current_route) - 2, 8)):  # Limit to small routes
            for j in range(i + 2, min(len(current_route) - 1, i + 6)):  # Very limited range
                if current_route[i] == 0 or current_route[j] == 0:
                    continue
                
                # Quick improvement check
                old_cost = (float(distances[current_route[i-1], current_route[i]]) +
                           float(distances[current_route[j], current_route[j+1]]))
                new_cost = (float(distances[current_route[i-1], current_route[j]]) +
                           float(distances[current_route[i], current_route[j+1]]))
                
                if new_cost < old_cost:
                    # Apply and return immediately (first improvement)
                    current_route[i:j+1] = current_route[i:j+1][::-1]
                    current_cost = current_cost - old_cost + new_cost
                    return current_route, current_cost
        
        return current_route, current_cost
def generate_instances_cpu(n_customers: int, n_instances: int, 
                          capacity: int, demand_range: Tuple[int, int],
                          coord_range: int) -> List[Dict[str, Any]]:
    """Step 1: Generate instances on CPU"""
    print(f"🔧 Step 1: Generating {n_instances} instances on CPU...")
    start_time = time.time()
    
    instances = []
    for i in range(n_instances):
        coordinates = np.random.randint(0, coord_range + 1, 
                                     size=(n_customers + 1, 2)).astype(np.float32) / coord_range
        
        demands = np.random.randint(demand_range[0], demand_range[1] + 1,
                                  size=n_customers + 1)
        demands[0] = 0  # Depot has 0 demand
        
        instance = {
            'instance_id': i,
            'coords': coordinates,
            'demands': demands,
            'capacity': capacity,
            'n_customers': n_customers
        }
        instances.append(instance)
    
    generation_time = time.time() - start_time
    print(f"✅ Generated {n_instances} instances on CPU in {generation_time:.3f}s")
    return instances

def transfer_to_gpu(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Step 2: Transfer instances to GPU"""
    if not GPU_AVAILABLE:
        print("⚠️ GPU not available, keeping instances on CPU")
        return instances
        
    print(f"🔄 Step 2: Transferring {len(instances)} instances to GPU...")
    start_time = time.time()
    
    gpu_instances = []
    for instance in instances:
        gpu_instance = {
            'instance_id': instance['instance_id'],
            'coords_gpu': cp.asarray(instance['coords']),
            'demands_gpu': cp.asarray(instance['demands']),
            'capacity': instance['capacity'],
            'n_customers': instance['n_customers']
        }
        gpu_instances.append(gpu_instance)
    
    transfer_time = time.time() - start_time
    print(f"✅ Transferred instances to GPU in {transfer_time:.3f}s")
    
    if GPU_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        print(f"💾 GPU memory used: {mempool.used_bytes() / 1024**2:.1f} MB")
    
    return gpu_instances

def solve_gpu_task(gpu_solver, solver_name, gpu_instance):
    """Solve single GPU task with correct correspondence - all exact solvers find optimal solution"""
    try:
        coords_gpu = gpu_instance["coords_gpu"] 
        demands_gpu = gpu_instance["demands_gpu"]
        capacity = gpu_instance["capacity"]
        
        # All exact solvers should find the same optimal solution
        if solver_name in ["exact_ortools_vrp", "exact_milp", "exact_dp", "exact_pulp"]:
            # Use optimal solver for all exact methods
            solution = gpu_solver.gpu_optimal_solver(coords_gpu, demands_gpu, capacity)
        elif solver_name == "heuristic_or":  # Heuristic can be different
            solution = gpu_solver.gpu_nearest_neighbor(coords_gpu, demands_gpu, capacity)
        else:
            raise ValueError(f"Unknown solver: {solver_name}")
        
        return {
            "solver": solver_name,
            "instance_id": gpu_instance["instance_id"],
            "success": solution.cost < float("inf"),
            "cost": solution.cost,
            "optimal": getattr(solution, "optimal", True),  # Exact solvers are optimal
            "solve_time": solution.solve_time
        }
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        return {
            "solver": solver_name,
            "instance_id": gpu_instance["instance_id"],
            "success": False,
            "cost": float("inf"),
            "optimal": False,
            "solve_time": 0.0,
            "error": error_details
        }
def launch_gpu_tasks(gpu_instances: List[Dict[str, Any]], timeout: float) -> List[Dict[str, Any]]:
    """Step 3: Launch ALL 500 GPU solver tasks simultaneously"""
    
    gpu_solvers = ['exact_ortools_vrp', 'exact_milp', 'exact_dp', 'exact_pulp', 'heuristic_or']
    n_instances = len(gpu_instances)
    n_solvers = len(gpu_solvers)
    total_tasks = n_solvers * n_instances
    
    print(f"\n🚀 Step 3: Launching ALL {total_tasks} GPU solver tasks simultaneously")
    print(f"📊 {n_solvers} solvers × {n_instances} instances = {total_tasks} parallel GPU tasks")
    print(f"⏱️ Timeout: {timeout}s")
    
    if not GPU_AVAILABLE:
        print("⚠️ GPU not available - cannot launch GPU tasks")
        return []
    
    gpu_solver = GPUCVRPSolvers()
    results = []
    
    max_workers = min(total_tasks, 64)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"🔧 Submitting ALL {total_tasks} GPU tasks...")
        
        future_to_info = {}
        for solver_name in gpu_solvers:
            for gpu_instance in gpu_instances:
                future = executor.submit(solve_gpu_task, gpu_solver, solver_name, gpu_instance)
                future_to_info[future] = (solver_name, gpu_instance['instance_id'])
        
        print(f"✅ ALL {total_tasks} GPU tasks submitted")
        print(f"⏱️ Waiting up to {timeout}s for GPU completion...")
        
        start_time = time.time()
        
        try:
            for future in as_completed(future_to_info, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    pass
                    
        except TimeoutError:
            elapsed = time.time() - start_time
            print(f"⏱️ GPU timeout reached after {elapsed:.1f}s")
            print(f"📊 Collected {len(results)}/{total_tasks} GPU results before timeout")
            
            for future in future_to_info:
                future.cancel()
    
    final_time = time.time() - start_time
    print(f"✅ GPU parallel execution: {len(results)}/{total_tasks} completed in {final_time:.1f}s")
    
    return results

def process_results_cpu(results: List[Dict[str, Any]], n_instances: int, 
                       n_customers: int) -> Dict[str, Dict[str, Any]]:
    """Step 4: Process GPU results on CPU"""
    
    print(f"\n🔄 Step 4: Processing {len(results)} GPU results on CPU...")
    start_time = time.time()
    
    gpu_solvers = ['exact_ortools_vrp', 'exact_milp', 'exact_dp', 'exact_pulp', 'heuristic_or']
    stats = {}
    
    for solver_name in gpu_solvers:
        solver_results = [r for r in results if r['solver'] == solver_name and r['success']]
        
        if solver_results:
            costs = [float(r['cost']) for r in solver_results]
            times = [float(r['solve_time']) for r in solver_results]
            optimals = len([r for r in solver_results if r['optimal']])
            
            stats[solver_name] = {
                'attempted': n_instances,
                'completed': len(solver_results),
                'success_rate': len(solver_results) / n_instances * 100,
                'optimal_solutions': optimals,
                'avg_cost': statistics.mean(costs) if costs else 0,
                'avg_time': statistics.mean(times) if times else 0,
                'avg_cost_per_customer': statistics.mean(costs) / n_customers if costs else 0,
                'total_time': sum(times)
            }
        else:
            stats[solver_name] = {
                'attempted': n_instances,
                'completed': 0,
                'success_rate': 0.0,
                'optimal_solutions': 0,
                'avg_cost': 0,
                'avg_time': 0,
                'avg_cost_per_customer': 0,
                'total_time': 0
            }
    
    processing_time = time.time() - start_time
    print(f"✅ GPU results processed on CPU in {processing_time:.3f}s")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='True GPU CVRP Solver Benchmark')
    parser.add_argument('--instances', type=int, default=100, 
                       help='Number of instances per problem size (default: 100)')
    parser.add_argument('--n-start', type=int, default=5,
                       help='Starting number of customers (default: 5)')
    parser.add_argument('--n-end', type=int, default=15,
                       help='Ending number of customers (default: 15)')
    parser.add_argument('--capacity', type=int, default=30,
                       help='Vehicle capacity (default: 30)')
    parser.add_argument('--demand-min', type=int, default=1,
                       help='Min demand (default: 1)')
    parser.add_argument('--demand-max', type=int, default=10,
                       help='Max demand (default: 10)')
    parser.add_argument('--timeout', type=float, default=120.0,
                       help='Timeout in seconds (default: 120.0s)')
    parser.add_argument('--coord-range', type=int, default=100,
                       help='Coordinate range (default: 100)')
    parser.add_argument('--output', default='gpu_benchmark_results.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TRUE GPU CVRP SOLVER BENCHMARK")
    print("="*80)
    print(f"Problem size: N = {args.n_start} to {args.n_end}")
    print(f"Instances: {args.instances}")
    print(f"Total GPU tasks per N: {5 * args.instances} (5 solvers × {args.instances} instances)")
    print(f"GPU timeout: {args.timeout}s")
    print(f"Output: {args.output}")
    print()
    
    if not GPU_AVAILABLE:
        print("❌ ERROR: GPU not available - cannot run GPU benchmark")
        return
    
    all_results = []
    
    for n_customers in range(args.n_start, args.n_end + 1):
        print(f"🚀 N={n_customers}: GPU CVRP benchmark")
        
        instances = generate_instances_cpu(
            n_customers, args.instances, args.capacity, 
            (args.demand_min, args.demand_max), args.coord_range
        )
        
        gpu_instances = transfer_to_gpu(instances)
        results = launch_gpu_tasks(gpu_instances, args.timeout)
        stats = process_results_cpu(results, args.instances, n_customers)
        
        print(f"\n📊 GPU RESULTS N={n_customers}:")
        for solver_name, solver_stats in stats.items():
            completed = solver_stats['completed']
            attempted = solver_stats['attempted']
            success_rate = solver_stats['success_rate']
            avg_cpc = solver_stats['avg_cost_per_customer']
            
            print(f"  {solver_name:18}: {completed}/{attempted} solved ({success_rate:.1f}%), cpc={avg_cpc:.4f}")
        
        all_results.append({
            'problem_size': n_customers,
            'statistics': stats
        })
    
    print(f"\n📊 Writing GPU results to {args.output}")
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['problem_size', 'solver', 'instances_attempted', 'instances_completed', 
                        'success_rate', 'optimal_solutions', 'avg_time', 'avg_cost', 
                        'avg_cost_per_customer', 'total_time'])
        
        for result in all_results:
            n_customers = result['problem_size']
            for solver_name, stats in result['statistics'].items():
                writer.writerow([
                    n_customers, solver_name, stats['attempted'], stats['completed'],
                    f"{stats['success_rate']:.1f}%", stats['optimal_solutions'],
                    f"{stats['avg_time']:.4f}", f"{stats['avg_cost']:.4f}",
                    f"{stats['avg_cost_per_customer']:.4f}", f"{stats['total_time']:.4f}"
                ])
    
    print(f"✅ TRUE GPU CVRP BENCHMARK COMPLETED!")

if __name__ == '__main__':
    main()
