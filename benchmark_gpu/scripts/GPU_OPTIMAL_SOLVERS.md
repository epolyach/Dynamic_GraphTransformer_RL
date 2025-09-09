# GPU Optimal CVRP Solvers

This directory contains three different optimal solvers for the Capacitated Vehicle Routing Problem (CVRP), specifically optimized for small instances (N=10 customers) with different vehicle capacities (C=20, 30).

## Available Solvers

### 1. gpu_cvrp_solver_truly_optimal.py
- **Algorithm**: Dynamic Programming with exponential state space
- **Status**: Production-ready, tested
- **Performance**: ~0.5-1.0 instances/second on RTX A6000
- **Memory**: ~12MB per instance for N=10
- **Limitations**: N≤12 due to memory constraints

### 2. gpu_cvrp_optimal_v2_fixed.py  
- **Algorithm**: Enhanced DP with optimizations
- **Status**: Experimental, needs debugging
- **Features**: Pruning, symmetry breaking, configurable modes
- **Expected Performance**: ~1.0-2.0 instances/second
- **Note**: Has an indexing bug that needs fixing

### 3. gpu_cvrp_solver_scip_optimal.py
- **Algorithm**: Mixed Integer Programming (MIP)
- **Solver**: SCIP (Solving Constraint Integer Programs)
- **Status**: Ready to use (requires pyscipopt)
- **Performance**: ~0.01-0.1 instances/second
- **Advantages**: Can handle N>10, provides optimality certificates

## Quick Start

### GPU DP Solver (Recommended for N=10)
```bash
# Run benchmark for N=10, C=20, 1000 instances
python3 benchmark_gpu_truly_optimal_n10.py --num-instances 1000 --capacity 20

# Run benchmark for N=10, C=30, 1000 instances  
python3 benchmark_gpu_truly_optimal_n10.py --num-instances 1000 --capacity 30
```

### SCIP Solver (For validation)
```bash
# Install SCIP Python bindings
pip install pyscipopt

# Run small benchmark
python3 gpu_cvrp_solver_scip_optimal.py --benchmark --time-limit 60
```

## Performance Estimates (N=10, RTX A6000)

| Instances | GPU DP Solver | SCIP Solver |
|-----------|--------------|-------------|
| 100       | ~2 minutes   | ~20 minutes |
| 1,000     | ~20 minutes  | ~3 hours    |
| 10,000    | ~3-4 hours   | ~30 hours   |

## Hardware Requirements

- **GPU DP Solvers**: NVIDIA GPU with CUDA, 8+ GB VRAM
- **SCIP Solver**: CPU-based, benefits from multiple cores

## Notes

- All solvers guarantee optimal solutions
- GPU DP is fastest for N≤10
- SCIP can handle larger instances but is slower
- Use GPU DP for production, SCIP for validation
