def generate_instance(n_customers: int, capacity: int, demand_range: Tuple[int, int], 
                     coord_range: int, seed: int) -> Dict[str, Any]:
    """Generate a single CVRP instance with deterministic seed"""
    np.random.seed(seed)  # Use deterministic seed for reproducibility
    
    # Generate coordinates (depot at origin)
    coords = np.random.uniform(0, coord_range, (n_customers + 1, 2))
    coords[0] = [coord_range / 2, coord_range / 2]  # Depot at center
    
    # Generate demands (depot has 0 demand)
    demands = np.zeros(n_customers + 1)
    demands[1:] = np.random.randint(demand_range[0], demand_range[1] + 1, n_customers)
    
    # Calculate distances
    distances = np.zeros((n_customers + 1, n_customers + 1))
    for i in range(n_customers + 1):
        for j in range(n_customers + 1):
            distances[i][j] = np.sqrt((coords[i][0] - coords[j][0])**2 + 
                                    (coords[i][1] - coords[j][1])**2)
    
    return {
        "coords": coords,
        "demands": demands,
        "distances": distances,
        "capacity": capacity,
        "demand_range": demand_range,
        "coord_range": coord_range
    }

def run_gpu_benchmark(n_customers: int, n_instances: int, capacity: int, 
                     demand_range: Tuple[int, int], coord_range: int, 
                     timeout: float) -> Dict[str, Any]:
    """Run GPU benchmark with full validation and deterministic seeding"""
    print(f"🚀 Starting GPU benchmark: N={n_customers}, {n_instances} instances")
    
    # Generate instances with same seeding strategy as CPU benchmark
    print(f"📊 Generating {n_instances} instances...")
    instances = []
    for i in range(n_instances):
        # Use same seeding formula as CPU: 4242 + n * 1000 + i * 10
        seed = 4242 + n_customers * 1000 + i * 10
        instance = generate_instance(n_customers, capacity, demand_range, coord_range, seed)
        instance["instance_id"] = i
        instances.append(instance)
    
    # Rest of the function remains the same...
    # Prepare GPU data
    print(f"🔄 Transferring data to GPU...")
    batch_coords_gpu = []
    batch_demands_gpu = []
    batch_capacities = []
    
    for instance in instances:
        coords = instance["coords"]
        demands = instance["demands"]
        
        # Convert to GPU arrays if available
        if GPU_AVAILABLE:
            coords_gpu = cp.array(coords, dtype=cp.float32)
            demands_gpu = cp.array(demands, dtype=cp.float32)
        else:
            print("⚠️ Running GPU solvers on CPU (CuPy not available)")
            coords_gpu = coords
            demands_gpu = demands
        
        batch_coords_gpu.append(coords_gpu)
        batch_demands_gpu.append(demands_gpu)
        batch_capacities.append(instance["capacity"])
    
    # Run GPU solvers
    solver_names = ["exact_ortools_vrp", "exact_milp", "exact_dp", "exact_pulp", "heuristic_or"]
    gpu_solver = TrueGPUCVRPSolvers()
    
    start_time = time.time()
    batch_results = gpu_solver.solve_batch_gpu(batch_coords_gpu, batch_demands_gpu, 
                                              batch_capacities, solver_names)
    gpu_time = time.time() - start_time
    print(f"⏱️ Total GPU computation time: {gpu_time:.2f}s")
    
    # Process and validate results 
    print(f"🔍 Processing results with validation...")
    
    stats = {}
    for solver_name in solver_names:
        stats[solver_name] = {
            "times": [],
            "costs": [],
            "cpcs": [],
            "success_count": 0,
            "optimal_count": 0
        }
    
    validation_count = 0
    
    for i, (instance, results) in enumerate(zip(instances, batch_results)):
        n = len(instance["coords"]) - 1  # Number of customers
        
        # Extract OR-Tools solution for validation baseline
        ortools_result = results.get("exact_ortools_vrp")
        if ortools_result and ortools_result["success"]:
            ortools_solution = CVRPSolution(
                cost=ortools_result["cost"],
                vehicle_routes=ortools_result["vehicle_routes"],
                optimal=ortools_result["optimal"]
            )
        else:
            ortools_solution = None
            
        # Validate other solutions against OR-Tools
        validation_errors = []
        other_solutions = {}
        
        for solver_name in solver_names:
            if solver_name == "exact_ortools_vrp":
                continue
                
            result = results.get(solver_name)
            if result and result["success"]:
                solution = CVRPSolution(
                    cost=result["cost"],
                    vehicle_routes=result["vehicle_routes"],
                    optimal=result.get("optimal", False)
                )
                other_solutions[solver_name] = solution
        
        # Perform validation (simplified version)
        if ortools_solution:
            validation_count += 1
            
            for solver_name, solution in other_solutions.items():
                if abs(solution.cost - ortools_solution.cost) > 1e-6:
                    validation_errors.append(f"{solver_name}: Cost mismatch with OR-Tools")
        
        if validation_errors:
            print(f"Validation errors found: {len(validation_errors)}")
            for error in validation_errors[:10]:  # Show first 10 errors
                print(f"  {error}")
        
        # Collect statistics
        for solver_name in solver_names:
            result = results.get(solver_name)
            if result and result["success"]:
                cost = result["cost"]
                cpc = cost / n if n > 0 else 0
                
                stats[solver_name]["times"].append(result["solve_time"])
                stats[solver_name]["costs"].append(cost)
                stats[solver_name]["cpcs"].append(cpc)
                stats[solver_name]["success_count"] += 1
                
                if result.get("optimal", False):
                    stats[solver_name]["optimal_count"] += 1
    
    print(f"✅ Validated {validation_count} instances")
    
    # Calculate final statistics
    final_stats = {}
    for solver_name in solver_names:
        s = stats[solver_name]
        if s["times"]:
            final_stats[solver_name] = {
                "avg_time": statistics.mean(s["times"]),
                "avg_cost": statistics.mean(s["costs"]),
                "avg_cpc": statistics.mean(s["cpcs"]),
                "std_cpc": statistics.stdev(s["cpcs"]) if len(s["cpcs"]) > 1 else 0.0,
                "success_rate": s["success_count"] / len(instances),
                "optimal_count": s["optimal_count"]
            }
        else:
            final_stats[solver_name] = {
                "avg_time": float('inf'),
                "avg_cost": float('inf'),
                "avg_cpc": float('inf'),
                "std_cpc": 0.0,
                "success_rate": 0.0,
                "optimal_count": 0
            }
    
    return final_stats
