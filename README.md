# Dynamic GraphTransformer RL - GPU Branch

GPU-accelerated CVRP (Capacitated Vehicle Routing Problem) solvers with significant performance improvements over CPU implementations.

## Directory Structure

```
├── benchmark_exact_gpu.py    # GPU benchmark script
├── benchmark_exact_cpu.py    # CPU benchmark script  
├── plot_gpu_benchmark.py     # GPU results plotting
├── plot_cpu_gpu_comparison.py # CPU vs GPU comparison
├── activate_env.sh           # Environment activation
├── csv/                      # CSV benchmark results
├── plots/                    # PNG visualization outputs
├── logs/                     # Log files from benchmarks
└── misc/                     # Archived old scripts
```

## Setup

### Environment Activation
```bash
# Activate the virtual environment (CPU-optimized for benchmarking)
./activate_env.sh
```

The `activate_env.sh` script:
- Activates the Python virtual environment (`venv/`)
- Shows Python version and CPU thread information
- Provides quick reference for available configurations

### Requirements
```bash
pip install ortools matplotlib pandas seaborn numpy
```

## Core Scripts

### 1. GPU Benchmarking (`benchmark_exact_gpu.py`)

Run GPU-accelerated CVRP solver benchmarks with comprehensive validation.

#### Basic Usage
```bash
# Quick test (5 customers, 20 instances)
python3 benchmark_exact_gpu.py --instances 20 --n-start 5 --n-end 5 --timeout 10.0 --output csv/gpu_test.csv

# Small benchmark (5-8 customers)
python3 benchmark_exact_gpu.py --instances 10 --n-start 5 --n-end 8 --timeout 15.0 --output csv/gpu_small.csv

# Standard benchmark (5-12 customers)
python3 benchmark_exact_gpu.py --instances 20 --n-start 5 --n-end 12 --timeout 30.0 --output csv/gpu_standard.csv

# Extended benchmark (5-15 customers)
python3 benchmark_exact_gpu.py --instances 50 --n-start 5 --n-end 15 --timeout 60.0 --output csv/gpu_extended.csv
```

#### Advanced Parameters
```bash
# Custom vehicle capacity and demand range
python3 benchmark_exact_gpu.py \
  --instances 25 \
  --n-start 6 \
  --n-end 10 \
  --capacity 50 \
  --demand-min 2 \
  --demand-max 8 \
  --coord-range 200 \
  --timeout 45.0 \
  --output csv/gpu_custom.csv

# High-performance run (large instances)
python3 benchmark_exact_gpu.py \
  --instances 100 \
  --n-start 8 \
  --n-end 20 \
  --capacity 40 \
  --timeout 120.0 \
  --output csv/gpu_large.csv
```

### 2. CPU Benchmarking (`benchmark_exact_cpu.py`)

Run CPU-based CVRP solver benchmarks for comparison with GPU performance.

#### Basic Usage
```bash
# Quick test (5 customers)
python3 benchmark_exact_cpu.py --instances-min 5 --instances-max 5 --n-start 5 --n-end 5 --timeout 60.0 --output csv/cpu_test.csv

# Small benchmark (5-8 customers)
python3 benchmark_exact_cpu.py --instances-min 5 --instances-max 10 --n-start 5 --n-end 8 --timeout 90.0 --output csv/cpu_small.csv

# Standard benchmark (5-12 customers)
python3 benchmark_exact_cpu.py --instances-min 10 --instances-max 20 --n-start 5 --n-end 12 --timeout 120.0 --output csv/cpu_standard.csv
```

#### Advanced Parameters
```bash
# Variable instance count per problem size
python3 benchmark_exact_cpu.py \
  --instances-min 5 \
  --instances-max 25 \
  --n-start 6 \
  --n-end 15 \
  --capacity 45 \
  --demand-min 1 \
  --demand-max 12 \
  --coord-range 150 \
  --timeout 180.0 \
  --output csv/cpu_advanced.csv \
  --log logs/cpu_advanced.log

# Production benchmark with logging
python3 benchmark_exact_cpu.py \
  --instances-min 20 \
  --instances-max 50 \
  --n-start 5 \
  --n-end 20 \
  --timeout 300.0 \
  --output csv/cpu_production.csv \
  --log logs/cpu_production.log
```

### 3. GPU Results Plotting (`plot_gpu_benchmark.py`)

Visualize GPU benchmark results with performance metrics.

```bash
# Basic GPU results plot
python3 plot_gpu_benchmark.py csv/gpu_results.csv --output plots/gpu_analysis

# Custom title and output
python3 plot_gpu_benchmark.py csv/gpu_extended.csv \
  --output plots/gpu_detailed_analysis \
  --title "GPU CVRP Performance Analysis (Extended Dataset)"

# Multiple GPU result sets
python3 plot_gpu_benchmark.py csv/gpu_small.csv --output plots/gpu_small_analysis
python3 plot_gpu_benchmark.py csv/gpu_large.csv --output plots/gpu_large_analysis
```

### 4. CPU vs GPU Comparison (`plot_cpu_gpu_comparison.py`)

Generate side-by-side performance comparisons between CPU and GPU implementations.

```bash
# Basic comparison
python3 plot_cpu_gpu_comparison.py csv/cpu_results.csv csv/gpu_results.csv --output plots/cpu_gpu_comparison

# Custom comparison with title
python3 plot_cpu_gpu_comparison.py csv/cpu_standard.csv csv/gpu_standard.csv \
  --output plots/detailed_comparison \
  --title "CPU vs GPU CVRP Solver Performance Comparison"

# Multiple comparison scenarios
python3 plot_cpu_gpu_comparison.py csv/cpu_small.csv csv/gpu_small.csv --output plots/small_comparison
python3 plot_cpu_gpu_comparison.py csv/cpu_large.csv csv/gpu_large.csv --output plots/large_comparison
```

## Complete Workflow Examples

### Quick Performance Test
```bash
# 1. Run quick GPU benchmark
python3 benchmark_exact_gpu.py --instances 10 --n-start 5 --n-end 8 --timeout 20.0 --output csv/gpu_quick.csv

# 2. Run equivalent CPU benchmark  
python3 benchmark_exact_cpu.py --instances-min 10 --instances-max 10 --n-start 5 --n-end 8 --timeout 120.0 --output csv/cpu_quick.csv

# 3. Compare results
python3 plot_cpu_gpu_comparison.py csv/cpu_quick.csv csv/gpu_quick.csv --output plots/quick_comparison
```

### Comprehensive Analysis
```bash
# 1. Extended GPU benchmark
python3 benchmark_exact_gpu.py --instances 50 --n-start 5 --n-end 15 --timeout 60.0 --output csv/gpu_full.csv

# 2. Extended CPU benchmark
python3 benchmark_exact_cpu.py --instances-min 20 --instances-max 50 --n-start 5 --n-end 15 --timeout 300.0 --output csv/cpu_full.csv

# 3. Individual analysis
python3 plot_gpu_benchmark.py csv/gpu_full.csv --output plots/gpu_full_analysis --title "Comprehensive GPU Performance Analysis"

# 4. Comparative analysis
python3 plot_cpu_gpu_comparison.py csv/cpu_full.csv csv/gpu_full.csv \
  --output plots/comprehensive_comparison \
  --title "Comprehensive CPU vs GPU CVRP Performance Analysis"
```

### Research-Grade Benchmarking
```bash
# High-volume GPU benchmark
python3 benchmark_exact_gpu.py \
  --instances 100 \
  --n-start 5 \
  --n-end 20 \
  --capacity 40 \
  --demand-min 1 \
  --demand-max 10 \
  --coord-range 100 \
  --timeout 120.0 \
  --output csv/gpu_research.csv

# Equivalent CPU benchmark with detailed logging
python3 benchmark_exact_cpu.py \
  --instances-min 50 \
  --instances-max 100 \
  --n-start 5 \
  --n-end 20 \
  --capacity 40 \
  --demand-min 1 \
  --demand-max 10 \
  --coord-range 100 \
  --timeout 600.0 \
  --output csv/cpu_research.csv \
  --log logs/cpu_research.log

# Generate publication-ready plots
python3 plot_cpu_gpu_comparison.py csv/cpu_research.csv csv/gpu_research.csv \
  --output plots/research_grade_comparison \
  --title "Research-Grade CPU vs GPU CVRP Solver Performance"
```

## Output Organization

### `csv/` Directory
- **Purpose**: Stores all CSV benchmark results
- **Contents**: Timing data, cost-per-customer metrics, solver statistics
- **Examples**: `gpu_results.csv`, `cpu_results.csv`, `gpu_large_scale.csv`

### `plots/` Directory  
- **Purpose**: Stores all PNG visualization outputs
- **Contents**: Performance plots, comparison charts, analysis graphs
- **Examples**: `gpu_analysis.png`, `cpu_gpu_comparison.png`, `research_grade_comparison.png`

### `logs/` Directory
- **Purpose**: Stores detailed execution logs (mainly from CPU benchmarks)
- **Contents**: Debug information, solver progress, error messages
- **Examples**: `cpu_production.log`, `cpu_research.log`

### `misc/` Directory
- **Purpose**: Archive of old scripts and backup files
- **Contents**: Deprecated scripts, backup versions, experimental code

## Parameter Reference

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--instances` | Number of random CVRP instances per problem size | 10-100 |
| `--n-start` | Starting number of customers | 5-10 |
| `--n-end` | Ending number of customers | 8-20 |
| `--capacity` | Vehicle capacity constraint | 20-50 |
| `--demand-min/max` | Customer demand range | 1-10 |
| `--coord-range` | Coordinate space for customer locations | 50-200 |
| `--timeout` | Maximum solver time per problem size (seconds) | 10-600 |
| `--output` | Output CSV file path (use `csv/filename.csv`) | `csv/results.csv` |
| `--log` | Log file path (use `logs/filename.log`) | `logs/benchmark.log` |

## File Naming Conventions

### CSV Files
- `gpu_[descriptor].csv` - GPU benchmark results
- `cpu_[descriptor].csv` - CPU benchmark results
- Examples: `gpu_small.csv`, `cpu_production.csv`, `gpu_large_scale.csv`

### Plot Files
- `[descriptor]_analysis.png` - Individual performance analysis
- `[descriptor]_comparison.png` - CPU vs GPU comparisons  
- Examples: `gpu_analysis.png`, `comprehensive_comparison.png`

### Log Files
- `cpu_[descriptor].log` - CPU benchmark execution logs
- Examples: `cpu_production.log`, `cpu_research.log`
