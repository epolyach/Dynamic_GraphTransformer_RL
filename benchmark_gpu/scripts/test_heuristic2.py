import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.benchmarking.solvers.gpu.heuristic_gpu_improved import solve_batch
from src.generator.generator import _generate_instance

# Test with a few small instances
instances = []
for i in range(5):
    instance = _generate_instance(
        num_customers=10,
        capacity=20,
        coord_range=100,
        demand_range=[1, 10],
        seed=42000 + i
    )
    instances.append(instance)

print("Testing heuristic_gpu_improved.solve_batch...")
solutions = solve_batch(instances, max_iterations=100, verbose=True)
for i, sol in enumerate(solutions):
    print(f'Instance {i}: Cost={sol.cost:.2f}, CPC={sol.cost/10:.4f}')
