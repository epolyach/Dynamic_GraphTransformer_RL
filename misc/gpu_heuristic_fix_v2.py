    def _gpu_heuristic_solver(self, coords_gpu, demands_gpu, capacity) -> CVRPSolution:
        """GPU heuristic solver - calls the real heuristic_or solver for consistency"""
        start_time = time.time()
        
        try:
            # Convert GPU data back to CPU format for heuristic solver
            if GPU_AVAILABLE:
                coords_cpu = cp.asnumpy(coords_gpu)
                demands_cpu = cp.asnumpy(demands_gpu)
            else:
                coords_cpu = coords_gpu
                demands_cpu = demands_gpu
            
            # Calculate distance matrix on CPU 
            n = len(coords_cpu)
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    distances[i][j] = np.sqrt((coords_cpu[i][0] - coords_cpu[j][0])**2 + 
                                            (coords_cpu[i][1] - coords_cpu[j][1])**2)
            
            # Create instance for heuristic solver (same format as CPU benchmark)
            instance = {
                'coords': coords_cpu,
                'demands': demands_cpu,
                'distances': distances,
                'capacity': capacity
            }
            
            # Call the actual heuristic OR-Tools solver (same as CPU benchmark)
            solver_solution = heuristic_or.solve(instance, time_limit=30.0, verbose=False)
            
            # Convert from solver CVRPSolution format to GPU CVRPSolution format
            return CVRPSolution(
                cost=solver_solution.cost,
                vehicle_routes=solver_solution.vehicle_routes,
                optimal=False,  # heuristic is not optimal
                solve_time=solver_solution.solve_time
            )
            
        except Exception as e:
            return CVRPSolution(
                cost=float("inf"),
                vehicle_routes=[],
                optimal=False,
                solve_time=time.time() - start_time
            )
