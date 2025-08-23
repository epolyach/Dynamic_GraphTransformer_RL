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
