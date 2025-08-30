    def solve_batch_gpu(self, batch_coords_gpu: List, batch_demands_gpu: List, 
                       batch_capacities: List, solver_names: List[str]) -> List[Dict[str, Any]]:
        """Solve multiple instances using proper exact solvers"""
        print(f"🚀 Launching batch GPU computation: {len(batch_coords_gpu)} instances × {len(solver_names)} solvers")
        
        results = []
        
        for i, (coords_gpu, demands_gpu, capacity) in enumerate(zip(batch_coords_gpu, batch_demands_gpu, batch_capacities)):
            instance_results = {}
            
            # Convert GPU data to CPU format for solver calls
            if GPU_AVAILABLE:
                coords_cpu = cp.asnumpy(coords_gpu)
                demands_cpu = cp.asnumpy(demands_gpu)
            else:
                coords_cpu = coords_gpu
                demands_cpu = demands_gpu
            
            # Calculate distance matrix
            n = len(coords_cpu)
            distances = np.zeros((n, n))
            for j in range(n):
                for k in range(n):
                    distances[j][k] = np.sqrt((coords_cpu[j][0] - coords_cpu[k][0])**2 + 
                                            (coords_cpu[j][1] - coords_cpu[k][1])**2)
            
            # Create instance dictionary for solver calls
            instance = {
                'coords': coords_cpu,
                'demands': demands_cpu,
                'distances': distances,
                'capacity': capacity
            }
            
            # Run each solver
            for solver_name in solver_names:
                start_time = time.time()
                
                try:
                    if solver_name == "exact_ortools_vrp":
                        solution = exact_ortools_vrp.solve(instance, time_limit=30.0, verbose=False)
                    elif solver_name == "exact_milp":
                        solution = exact_milp.solve(instance, time_limit=30.0, verbose=False)
                    elif solver_name == "exact_dp":
                        solution = exact_dp.solve(instance, time_limit=30.0, verbose=False)
                    elif solver_name == "exact_pulp":
                        solution = exact_pulp.solve(instance, time_limit=30.0, verbose=False)
                    elif solver_name == "heuristic_or":
                        solution = heuristic_or.solve(instance, time_limit=30.0, verbose=False)
                    else:
                        raise ValueError(f"Unknown solver: {solver_name}")
                    
                    solve_time = time.time() - start_time
                    
                    # Convert to GPU result format
                    instance_results[solver_name] = {
                        "success": solution.cost < float("inf"),
                        "cost": solution.cost,
                        "vehicle_routes": solution.vehicle_routes,
                        "optimal": getattr(solution, "is_optimal", True) if solver_name.startswith("exact") else False,
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
