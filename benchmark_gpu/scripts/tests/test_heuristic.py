import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.benchmarking.solvers.gpu import heuristic_gpu_improved
from src.generator.generator import _generate_instance

# Test with one small instance
instance = _generate_instance(
    num_customers=10,
    capacity=20,
    coord_range=100,
    demand_range=[1, 10],
    seed=42000
)

print("Testing heuristic_gpu_improved...")
solution = heuristic_gpu_improved.solve(instance, time_limit=1.0)
print('Cost:', solution.cost)
print('CPC:', solution.cost / 10)
print('Routes:', solution.routes[:2] if len(solution.routes) > 2 else solution.routes)
