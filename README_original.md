# CVRP Solver Benchmark Suite

A comprehensive benchmarking suite for Capacitated Vehicle Routing Problem (CVRP) solvers, featuring both CPU and GPU implementations with unified instance generation.

## 🚀 Quick Start

```bash
# Setup environment
./activate_env.sh              # CPU benchmarks
./activate_gpu_env.sh          # GPU benchmarks (requires NVIDIA GPU + CUDA)

# Quick test
python3 benchmark_exact_cpu.py --n-start 5 --n-end 5 --instances-min 3 --instances-max 3
python3 benchmark_exact_gpu.py --n-start 5 --n-end 5 --instances 3
```

## 📊 Benchmarking

### CPU Benchmark

```bash
# Standard benchmark
python3 benchmark_exact_cpu.py --n-start 5 --n-end 15 --instances-min 20 --instances-max 20

# Key options
--timeout 120              # Timeout per solver per N (default: 60s)
--output results/csv/out.csv  # Output file
--debug                    # Enable debug output
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
# Plot CPU results
python3 plot_cpu_benchmark.py --csv results/csv/cpu_benchmark.csv --output plots

# Plot GPU results  
python3 plot_gpu_benchmark.py results/csv/gpu_benchmark.csv --output plots

# Compare CPU vs GPU
python3 plot_cpu_gpu_comparison.py \
    results/csv/cpu_benchmark.csv \
    results/csv/gpu_benchmark.csv \
    --output plots/comparison
```

## 🔧 Configuration

Fixed parameters in `config.json`:
- Vehicle capacity: 30
- Customer demand: [1, 10] 
- Coordinates: [0, 100] → normalized to [0, 1]

## 📁 Project Structure

```
├── benchmark_exact_cpu.py    # CPU benchmark
├── benchmark_exact_gpu.py    # GPU benchmark
├── plot_*.py                 # Visualization scripts
├── config.json              # Fixed parameters
├── solvers/                 # Solver implementations
│   ├── exact_milp.py       # Exact MILP solver
│   ├── exact_ortools_vrp.py # OR-Tools metaheuristic
│   └── heuristic_or.py     # OR-Tools heuristic
└── results/
    ├── csv/                # Benchmark results
    ├── plots/              # Visualizations
    └── logs/               # Execution logs & MATLAB format
```

## 🔬 Complete Workflow Example

```bash
# 1. Run CPU benchmark
python3 benchmark_exact_cpu.py --n-start 5 --n-end 15 \
    --instances-min 20 --instances-max 20

# 2. Run GPU benchmark
python3 benchmark_exact_gpu.py --n-start 5 --n-end 15 --instances 20

# 3. Visualize results
python3 plot_cpu_benchmark.py --csv results/csv/cpu_benchmark.csv
python3 plot_gpu_benchmark.py results/csv/gpu_benchmark.csv
python3 plot_cpu_gpu_comparison.py \
    results/csv/cpu_benchmark.csv \
    results/csv/gpu_benchmark.csv
```

## 📊 Output Formats

### CSV Format
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
  - Exact (MILP): Guaranteed optimal
  - Metaheuristic (OR-Tools): Near-optimal
  - Heuristic (OR-Tools): Fast approximate

## 🐛 Debugging

```bash
# Debug mode shows detailed output
python3 benchmark_exact_cpu.py --n-start 5 --n-end 5 \
    --instances-min 1 --instances-max 1 --debug

python3 benchmark_exact_gpu.py --n-start 5 --n-end 5 \
    --instances 1 --debug
```

## ⚙️ Advanced Options

### CPU Benchmark Options
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
- **Auto-disable:** CPU benchmark disables consistently failing solvers
- **Incremental Save:** Results saved after each problem size
- **MATLAB Logs:** Auto-generated for first 5 instances per N (CPU only)

---

For development scripts and additional documentation, see the `misc/` directory.
