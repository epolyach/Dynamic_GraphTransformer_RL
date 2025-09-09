# GPU Scripts Cleanup Plan

## Files to KEEP (Working and Current):

### 1. Heuristic Solvers (Working, mentioned in README)
- `benchmark_gpu_multi_n.py` - Main heuristic benchmark (mentioned in README)
- `benchmark_gpu_heuristic_gls_advanced.py` - Most advanced heuristic version

### 2. Exact/Optimal Solvers (Current working versions)
- `gpu_cvrp_solver_truly_optimal_fixed.py` - FIXED DP solver (bug-free)
- `benchmark_gpu_truly_optimal_n10.py` - Benchmark script for N=10 (mentioned in README)
- `gpu_cvrp_solver_scip_optimal_fixed.py` - SCIP solver (working version)

### 3. Utilities
- `run_scip_n10_c30_1000.py` - Production script for SCIP runs
- `monitor_gpu_benchmark.sh` - Monitoring script

## Files to REMOVE (Obsolete/Buggy/Duplicates):

### Obsolete heuristic versions:
- `benchmark_gpu_heuristic_gls.py` (old version)
- `benchmark_gpu_heuristic_improved.py` (superseded)
- `benchmark_gpu_heuristic_multi.py` (superseded)

### Obsolete benchmark scripts:
- `benchmark_gpu_10k.py` (old)
- `benchmark_gpu_adaptive_n.py` (old)
- `benchmark_gpu_exact_matched.py` (old)
- `benchmark_gpu_dp_exact.py` (old/buggy)

### Buggy/experimental exact solvers:
- `gpu_cvrp_solver_truly_optimal.py` (BUGGY - num_vehicles always 1)
- `gpu_cvrp_solver_truly_optimal_backup.py` (backup of buggy version)
- `gpu_cvrp_optimal_v2.py` (experimental, unfinished)
- `gpu_cvrp_optimal_v2_fixed.py` (experimental, has indexing bug)
- `gpu_cvrp_solver_scip_optimal.py` (old version with CVRPSolution issues)
