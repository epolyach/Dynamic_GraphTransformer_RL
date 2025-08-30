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

n_customers = 6
seeds = [5000 + i for i in range(100)]
instances = [generate_instance(n_customers, s) for s in seeds]

# Exact-DP on same instances
start = time.time()
dp_cpcs = []
for ins in instances:
    r = exact_dp_solve(ins, time_limit=60.0, verbose=False)
    dp_cpcs.append(r.cost / n_customers)

dp_tpi = (time.time() - start) / len(instances)

# OR-Tools exact on same instances
start = time.time()
or_cpcs = []
for ins in instances:
    r = ortools_exact_solve(ins, time_limit=60.0, verbose=False)
    or_cpcs.append(r.cost / n_customers)

or_tpi = (time.time() - start) / len(instances)

import math
import statistics as st

print("| Solver   | Instances | Mean CPC | Std CPC | TPI (sec) |")
print("|----------|-----------|----------|---------|-----------|")
print(f"| exact-dp | {len(instances):9d} | {st.mean(dp_cpcs):8.6f} | {st.pstdev(dp_cpcs):7.6f} | {dp_tpi:9.6f} |")
print(f"| ortools  | {len(instances):9d} | {st.mean(or_cpcs):8.6f} | {st.pstdev(or_cpcs):7.6f} | {or_tpi:9.6f} |")
