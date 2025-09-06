# Scripts Inventory - benchmark_gpu/scripts/
Generated: 2025-09-06

## Summary
- Total scripts: 51 Python files
- Last major update: Sept 5, 2025 (most scripts at 21:57)
- Latest modifications: Sept 5 22:00+ (production deployment scripts)

## Script Categories

### 1. CPU-Based Scripts (Should move to training_cpu/)

#### OR-Tools Scripts (14 files) - ALL CPU-BASED
**Production Scripts:**
- `run_ortools_gls_parallel_test.py` (Sept 5 23:25) ⭐ **LATEST WITH THREAD SUPPORT**
  - Uses ProcessPoolExecutor for parallel processing
  - Handles N=10,20,50,100 with configurable threads
  - Most recent implementation for production benchmarks
  
- `run_ortools_gls_production.py` (Sept 5 22:17)
  - Production runner for OR-Tools GLS benchmarks
  - Manages batch processing for different N values
  
- `run_ortools_gls_extras.py` (Sept 5 22:56)
  - Extended timeout tests for OR-Tools
  
- `run_ortools_gls_test.py` (Sept 5 22:14)
  - Test runner for OR-Tools GLS

**Benchmark Scripts:**
- `benchmark_ortools_gls.py` - Basic OR-Tools GLS benchmark
- `benchmark_ortools_gls_fixed.py` - Fixed parameters version
- `benchmark_ortools_multi_n.py` - Multiple N sizes testing
- `benchmark_ortools_multi_n_fixed.py` - Fixed version for multi-N

**Monitoring Scripts:**
- `monitor_all_ortools.py` (Sept 5 22:13) - Active monitoring
- `monitor_all_ortools_backup.py` (Sept 5 22:11) - Backup version
- `monitor_ortools_fixed_n100.py` - N=100 specific monitoring
- `continuous_monitor_ortools.py` - Continuous monitoring
- `continuous_monitor_ortools_fixed.py` - Fixed params monitoring

**Data Processing:**
- `generate_ortools_timeout_table.py` (Sept 5 22:48) - Latest table generator

### 2. GPU-Based Scripts (21 files)

**Benchmark Scripts:**
- `benchmark_gpu_10k.py` - 10k instances benchmark
- `benchmark_gpu_adaptive_n.py` - Adaptive N sizing
- `benchmark_gpu_dp_exact.py` - Dynamic programming exact solver
- `benchmark_gpu_exact_matched.py` - Matched exact solutions
- `benchmark_gpu_heuristic_gls.py` - GPU GLS heuristic
- `benchmark_gpu_heuristic_gls_advanced.py` - Advanced GPU GLS
- `benchmark_gpu_heuristic_improved.py` - Improved heuristics
- `benchmark_gpu_heuristic_multi.py` - Multiple heuristics
- `benchmark_gpu_multi_n.py` - Multiple N sizes on GPU

**Example/Test Scripts:**
- `run_gpu_gls_example.py` - GLS example runner
- `run_gpu_heuristic_examples.py` - Heuristic examples
- `test_gpu_simple.py` - Simple GPU tests
- `test_gpu_results.py` - Results validation
- `test_gls.py` - GLS testing
- `test_heuristic.py` - Heuristic testing
- `test_heuristic2.py` - Additional heuristic tests
- `test_improved_solver.py` - Improved solver tests

**Other GPU Scripts:**
- `run_exact.py` - Exact solver runner
- `plot_gpu_benchmark.py` - GPU benchmark plotting
- `plot_cpu_gpu_comparison.py` - CPU/GPU comparison plots

### 3. Data Processing & Visualization (16 files)

**Table Generation:**
- `generate_final_latex_tables.py` - LaTeX table generation
- `generate_final_table.py` - Final results table
- `generate_final_table_with_n100.py` - N=100 specific tables
- `generate_gpu_latex_tables.py` - GPU results tables
- `generate_table_with_normality.py` - Statistical analysis tables
- `generate_table_with_normality_fixed.py` - Fixed version

**Figure Generation:**
- `make_log_norm_figure.py` - Log-normal distribution plots
- `make_log_norm_figure_cli.py` (Sept 6 16:31) - CLI version
- `make_panel4_fig.py` - 4-panel figures
- `make_panel4_fig_improved.py` - Improved panel figures
- `compare_panel_figures.py` - Figure comparison

**Statistical Analysis:**
- `test_cpc_lognormal.py` - Log-normal CPC testing
- `test_cpc_normality.py` - Normality testing
- `test_cpc_normality_cl.py` - CL normality testing

**Monitoring:**
- `continuous_monitor.py` - General continuous monitoring
- `monitor_and_generate_table.py` - Monitor and generate tables
- `monitor_progress.py` - Progress monitoring

## Scripts Processing N=10,20,50,100

**Primary Production Scripts:**
1. `run_ortools_gls_parallel_test.py` - Main parallel processor
2. `run_ortools_gls_production.py` - Production runner
3. `benchmark_ortools_multi_n.py` - Multi-N benchmarking
4. `benchmark_gpu_multi_n.py` - GPU multi-N benchmarking
5. `benchmark_gpu_adaptive_n.py` - Adaptive N sizing

## Recommended Actions

### Immediate Actions:
1. **Move CPU scripts to training_cpu/**
   - All 14 OR-Tools scripts should be moved
   - Update import paths in dependent scripts

2. **Keep Latest Implementations:**
   - `run_ortools_gls_parallel_test.py` - Latest with thread support ⭐
   - `src/benchmarking/solvers/cpu/ortools_gls.py` - Core solver

3. **Archive Obsolete Scripts:**
   - `monitor_all_ortools_backup.py` - Backup version
   - Scripts with `.backup` extension
   - Duplicate/fixed versions when originals exist

### Directory Structure Proposal:
```
benchmark_gpu/
├── scripts/
│   ├── gpu_benchmarks/    # GPU-specific benchmarks
│   ├── visualization/      # Plotting and figure generation
│   └── monitoring/         # Active monitoring scripts
training_cpu/
├── scripts/
│   └── ortools/           # All OR-Tools scripts
```

## Latest OR-Tools GLS Implementation

**File:** `run_ortools_gls_parallel_test.py`
**Key Features:**
- ProcessPoolExecutor for parallel instance processing
- Configurable thread count per N value
- Striped allocation of instances across threads
- Supports N=10,20,50,100 with automatic capacity calculation
- Each thread produces individual JSON output
- Timeout management per instance batch

**Usage Pattern:**
```python
# From run_ortools_gls_parallel_test.py
with ProcessPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for thread_id in range(num_threads):
        args = (n, capacity, timeout, thread_id, num_threads, 
                num_instances, output_dir)
        futures.append(executor.submit(run_thread_instances, args))
```

This is the most recent and sophisticated implementation for OR-Tools GLS benchmarking with full parallelization support.
