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


def run_identical(n_customers=6, n_instances=100, seed=5000):
    inst = generate_instance(n_customers, seed)
    instances = [inst for _ in range(n_instances)]

    # Exact-DP
    start = time.time()
    dp_costs = []
    for i, ins in enumerate(instances):
        if i % 20 == 0 and i > 0:
            print(f"  Exact-DP progress: {i}/{n_instances}")
        res = exact_dp_solve(ins, time_limit=60.0, verbose=False)
        dp_costs.append(res.cost / n_customers)
    dp_tpi = (time.time() - start) / n_instances

    # OR-Tools exact
    start = time.time()
    or_costs = []
    for i, ins in enumerate(instances):
        if i % 20 == 0 and i > 0:
            print(f"  OR-Tools progress: {i}/{n_instances}")
        res = ortools_exact_solve(ins, time_limit=60.0, verbose=False)
        or_costs.append(res.cost / n_customers)
    or_tpi = (time.time() - start) / n_instances

    dp_cpc = np.array(dp_costs)
    or_cpc = np.array(or_costs)

    print("\n| Solver   | Instances | Mean CPC | Std CPC | TPI (sec) |")
    print("|----------|-----------|----------|---------|-----------|")
    print(f"| exact-dp | {n_instances:9d} | {dp_cpc.mean():8.6f} | {dp_cpc.std():7.6f} | {dp_tpi:9.6f} |")
    print(f"| ortools  | {n_instances:9d} | {or_cpc.mean():8.6f} | {or_cpc.std():7.6f} | {or_tpi:9.6f} |")

if __name__ == '__main__':
    run_identical(n_customers=6, n_instances=100, seed=6001)
