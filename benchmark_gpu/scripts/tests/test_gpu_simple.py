import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.benchmarking.solvers.gpu import heuristic_gpu_simple
from src.generator.generator import _generate_instance
import time

instances = []
for i in range(10):
    instance = _generate_instance(
        num_customers=100,
        capacity=50,
        coord_range=100,
        demand_range=[1, 10],
        seed=42000 + 100*1000 + i
    )
    instances.append(instance)

print('Testing heuristic_gpu_simple with 10 instances of N=100...')
start = time.time()
solutions = heuristic_gpu_simple.solve_batch(instances, verbose=False)
elapsed = time.time() - start
print(f'Time: {elapsed:.2f}s ({elapsed/10:.3f}s per instance)')
for i, sol in enumerate(solutions[:3]):
    print(f'Instance {i}: Cost={sol.cost:.2f}, CPC={sol.cost/100:.4f}')
