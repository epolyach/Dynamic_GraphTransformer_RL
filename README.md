# CVRP Solver Benchmark Suite

A comprehensive benchmarking suite for Capacitated Vehicle Routing Problem (CVRP) solvers, featuring both CPU and GPU implementations with unified instance generation.

## 🚀 Quick Start

```bash
# Setup environment
./activate_env.sh              # CPU benchmarks
./activate_gpu_env.sh          # GPU benchmarks (requires NVIDIA GPU + CUDA)

# Quick test (original benchmark)
python3 benchmark_exact_cpu.py --n-start 5 --n-end 5 --instances-min 3 --instances-max 3

# Quick test (modified benchmark)
python3 benchmark_exact_cpu_modified.py --N 5 6 7 8 --instances 10 --quick
```

## 📊 Benchmarking

### CPU Benchmark (Original)

```bash
# Standard benchmark
python3 benchmark_exact_cpu.py --n-start 5 --n-end 15 --instances-min 20 --instances-max 20

# Key options
--timeout 120              # Timeout per solver per N (default: 60s)
--output results/csv/out.csv  # Output file
--debug                    # Enable debug output
```

### CPU Benchmark (Modified)

The modified benchmark provides optimized solver configurations:
- **exact_dp**: Runs only for N ≤ 8 (exact dynamic programming)
- **ortools_greedy**: Exact mode for N ≤ 8, heuristic mode for N > 8
- **ortools_gls**: Always runs with 2s timeout (Guided Local Search)

```bash
# Full benchmark with default N values
python3 benchmark_exact_cpu_modified.py

# Custom N values
python3 benchmark_exact_cpu_modified.py --N 5 6 7 8 9 10 12 15 18 20 --instances 10

# Quick test mode (3 instances per N)
python3 benchmark_exact_cpu_modified.py --N 5 10 15 --quick
```

### GPU Benchmark

```bash
# Standard benchmark  
python3 benchmark_exact_gpu.py --n-start 5 --n-end 15 --instances 50

# Key options
--timeout 300              # Solver timeout (default: 120s)
--output results/csv/out.csv  # Output file
```

**Requirements:** NVIDIA GPU with CUDA + CuPy (`pip install cupy-cuda12x`)

## 📈 Visualization

```bash
# Plot CPU results (handles both original and modified CSV formats)
python3 plot_cpu_benchmark.py --csv results/csv/benchmark_modified_*.csv --output plots/cpu_benchmark.png

# Plot GPU results  
python3 plot_gpu_benchmark.py results/csv/gpu_benchmark.csv --output plots

# Compare CPU vs GPU
python3 plot_cpu_gpu_comparison.py \
    results/csv/cpu_benchmark.csv \
    results/csv/gpu_benchmark.csv \
    --output plots/comparison
```

The `plot_cpu_benchmark.py` script now:
- Automatically handles both CSV formats (original with 'n_customers' and modified with 'n')
- Shows visual break between exact and heuristic modes for OR-Tools Greedy (at N=8/9)
- Displays separate statistics for exact vs heuristic performance

## 🔧 Configuration

Fixed parameters in `config.json`:
- Vehicle capacity: 30
- Customer demand: [1, 10] 
- Coordinates: [0, 100] → normalized to [0, 1]

## 📁 Project Structure

```
├── benchmark_exact_cpu.py           # Original CPU benchmark
├── benchmark_exact_cpu_modified.py  # Modified CPU benchmark with optimized configs
├── benchmark_exact_gpu.py           # GPU benchmark
├── plot_cpu_benchmark.py            # CPU results visualization (handles both formats)
├── plot_*.py                        # Other visualization scripts
├── config.json                      # Fixed parameters
├── solvers/                         # Solver implementations
│   ├── exact/
│   │   ├── exact_dp.py            # Dynamic programming (N≤8)
│   │   └── ortools_greedy.py      # OR-Tools with exact/heuristic modes
│   ├── exact_milp.py               # Exact MILP solver
│   ├── exact_ortools_vrp.py        # OR-Tools metaheuristic
│   ├── ortools_gls.py              # OR-Tools Guided Local Search
│   └── heuristic_or.py             # OR-Tools heuristic
└── results/
    ├── csv/                        # Benchmark results
    ├── plots/                      # Visualizations
    └── logs/                       # Execution logs & MATLAB format
```

## 🔬 Complete Workflow Example

```bash
# 1. Run modified CPU benchmark for comprehensive testing
python3 benchmark_exact_cpu_modified.py --N 5 6 7 8 9 10 12 15 18 20 --instances 10

# 2. Visualize results with mode transitions
python3 plot_cpu_benchmark.py --csv results/csv/benchmark_modified_*.csv --output plots/cpu_benchmark.png

# 3. Run GPU benchmark for comparison
python3 benchmark_exact_gpu.py --n-start 5 --n-end 20 --instances 10

# 4. Compare CPU vs GPU performance
python3 plot_cpu_gpu_comparison.py \
    results/csv/benchmark_modified_*.csv \
    results/csv/gpu_benchmark.csv
```

## 📊 Output Formats

### Modified Benchmark CSV Format
```csv
n_customers,instance_id,seed,solver,cost,cpc,time,timeout,failed
5,1,5000,exact_dp,2.7756,0.5551,0.0248,False,False
5,1,5000,ortools_greedy,2.7756,0.5551,0.0147,False,False
5,1,5000,ortools_gls,2.7756,0.5551,2.0031,False,False
```

### Original Benchmark CSV Format
```csv
n_customers,solver,instance_id,status,time,cpc
5,exact_milp,n5_s5242,success,0.0224,0.3741
5,exact_ortools_vrp,n5_s5242,success,0.0187,0.3741
```

### MATLAB Log (CPU only)
First 5 instances per N are logged in MATLAB format:
```matlab
N=5, Instance 1:
problem_matrix = [0.50 0.65 0.41 0.19 0.24;
                  0.50 0.66 0.15 0.80 0.81;
                  0 2 8 3 3];
Exact (MILP)              0.3741 [0, 1, 4, 3, 5, 2, 0]
Metaheuristic (OR-Tools)  0.3741 [0, 1, 4, 3, 5, 2, 0]
```

## 📈 Metrics

- **CPC (Cost Per Customer):** Primary quality metric = Total Cost / N
- **Solvers:**
  - Exact DP: Guaranteed optimal (N≤8 only)
  - OR-Tools Greedy: Exact for N≤8, fast heuristic for N>8
  - OR-Tools GLS: High-quality solutions with 2s time limit
  - Exact MILP: Guaranteed optimal (original benchmark)
  - Metaheuristic (OR-Tools): Near-optimal (original benchmark)

## 🐛 Debugging

```bash
# Debug mode for original benchmark
python3 benchmark_exact_cpu.py --n-start 5 --n-end 5 \
    --instances-min 1 --instances-max 1 --debug

# Quick test for modified benchmark
python3 benchmark_exact_cpu_modified.py --N 5 6 7 --quick

# GPU debug mode
python3 benchmark_exact_gpu.py --n-start 5 --n-end 5 \
    --instances 1 --debug
```

## ⚙️ Advanced Options

### Modified CPU Benchmark Options
| Option | Default | Description |
|--------|---------|-------------|
| `--N`, `--n-values` | [5,6,7,8,9,10,12,15,18,20] | Problem sizes to test |
| `--instances` | 10 | Instances per N |
| `--quick` | False | Quick mode (3 instances) |

### Original CPU Benchmark Options
| Option | Default | Description |
|--------|---------|-------------|
| `--n-start` | 5 | Starting problem size |
| `--n-end` | 15 | Ending problem size |
| `--instances-min` | 5 | Min instances per N |
| `--instances-max` | 20 | Max instances per N |
| `--timeout` | 60 | Timeout per solver per N (seconds) |
| `--output` | results/csv/cpu_benchmark.csv | Output file |
| `--log` | results/logs/benchmark_cpu.log | Log file |

### GPU Benchmark Options
| Option | Default | Description |
|--------|---------|-------------|
| `--n-start` | 5 | Starting problem size |
| `--n-end` | 20 | Ending problem size |
| `--instances` | 100 | Instances per size |
| `--timeout` | 120 | Solver timeout (seconds) |
| `--output` | results/csv/gpu_benchmark.csv | Output file |

## 📝 Notes

- **Instance Generation:** Identical seeds ensure fair CPU/GPU comparison
- **Route Format:** All routes include depot (0) at start/end
- **Solver Transitions:** OR-Tools Greedy switches from exact to heuristic at N>8
- **Visualization:** plot_cpu_benchmark.py shows visual break at N=8/9 transition
- **Auto-disable:** Original CPU benchmark disables consistently failing solvers
- **Incremental Save:** Results saved after each problem size
- **MATLAB Logs:** Auto-generated for first 5 instances per N (CPU only)

---

For development scripts and additional documentation, see the `misc/` directory.
