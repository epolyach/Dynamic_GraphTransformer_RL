#!/usr/bin/env python3
import numpy as np
import time
from solvers.exact_dp import solve as exact_dp_solve
from solvers.exact.ortools_greedy import solve as ortools_exact_solve

def generate_instance(n_customers, seed):
    np.random.seed(seed)
    n = n_customers + 1
    coords = np.random.uniform(0, 1, size=(n, 2))
    coords[0] = [0.5, 0.5]
    demands = np.zeros(n, dtype=np.float32)
    demands[1:] = np.random.uniform(1, 10, size=n_customers)
    capacity = max(demands.sum() / 2, demands.max() * 2)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])
    return {'coords': coords, 'demands': demands, 'distances': distances, 
            'capacity': capacity, 'n_customers': n_customers}

def run_test(n_instances, solver_func, solver_name):
    n_customers = 6
    instances = [generate_instance(n_customers, 5000 + i) for i in range(n_instances)]
    
    start = time.time()
    results = []
    
    for i, instance in enumerate(instances):
        if i % 100 == 0 and i > 0:
            print(f"  {solver_name} progress: {i}/{n_instances}")
        result = solver_func(instance, time_limit=60.0, verbose=False)
        results.append(result)
    
    total_time = time.time() - start
    costs = np.array([r.cost for r in results])
    cpcs = costs / n_customers
    
    return cpcs.mean(), cpcs.std(), total_time / n_instances

# Test both solvers on same instances
print("\n| Solver      | Instances | Mean CPC | Std CPC | TPI (sec) |")
print("|-------------|-----------|----------|---------|-----------|")

# Exact DP - 100 instances
cpc_dp_100, std_dp_100, tpi_dp_100 = run_test(100, exact_dp_solve, "Exact-DP")
print(f"| Exact-DP    |       100 | {cpc_dp_100:8.6f} | {std_dp_100:7.6f} | {tpi_dp_100:9.6f} |")

# OR-Tools - 100 instances  
cpc_or_100, std_or_100, tpi_or_100 = run_test(100, ortools_exact_solve, "OR-Tools")
print(f"| OR-Tools    |       100 | {cpc_or_100:8.6f} | {std_or_100:7.6f} | {tpi_or_100:9.6f} |")

# Exact DP - 1000 instances
cpc_dp_1000, std_dp_1000, tpi_dp_1000 = run_test(1000, exact_dp_solve, "Exact-DP")
print(f"| Exact-DP    |      1000 | {cpc_dp_1000:8.6f} | {std_dp_1000:7.6f} | {tpi_dp_1000:9.6f} |")

# OR-Tools - 1000 instances
cpc_or_1000, std_or_1000, tpi_or_1000 = run_test(1000, ortools_exact_solve, "OR-Tools")
print(f"| OR-Tools    |      1000 | {cpc_or_1000:8.6f} | {std_or_1000:7.6f} | {tpi_or_1000:9.6f} |")
