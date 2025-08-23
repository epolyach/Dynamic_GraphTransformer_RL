# Dynamic GraphTransformer RL - CVRP Benchmark

GPU-accelerated CVRP (Capacitated Vehicle Routing Problem) solvers with unified, config-driven instance generation ensuring fair performance comparisons.

## Directory Structure

```
├── benchmark_exact_cpu.py       # CPU benchmark (config-driven, standalone)
├── benchmark_exact_gpu.py       # GPU benchmark (config-driven, unified generation, standalone)  
├── plot_gpu_benchmark.py        # GPU results plotting
├── plot_cpu_gpu_comparison.py   # CPU vs GPU comparison plotting
├── config.json                  # Fixed parameters for consistent benchmarking
├── csv/                         # CSV benchmark results
├── plots/                       # PNG visualization outputs
├── logs/                        # Log files from benchmarks
└── misc/                        # Development/utility scripts and backups
```

## Key Features

✅ **Config-Driven**: All parameters loaded from `config.json` for consistency  
✅ **Unified Instance Generation**: Both benchmarks generate identical instances  
✅ **Standalone**: Main files require no external dependencies beyond config.json  
✅ **Fair Comparison**: Performance differences reflect computational efficiency, not instance variations



## Setup & GPU Acceleration

### Environment Activation

**For CPU-only benchmarks:**
```bash
# Activate CPU-optimized environment
./activate_env.sh
```

**For GPU-accelerated benchmarks:**
```bash  
# Activate GPU environment with CuPy support
./activate_gpu_env.sh
```

### GPU Requirements

To enable true GPU acceleration, you need:
- **NVIDIA GPU** with CUDA support
- **CuPy** installed in the gpu_env environment

**Status Messages:**
- ✅ `🚀 GPU acceleration available` - True GPU acceleration enabled
- ❌ `⚠️ ImportError: No module named 'cupy' - CuPy not installed (benchmark will not run)` - Fallback to CPU

**Installation:**
```bash
# Activate GPU environment and install CuPy
source gpu_env/bin/activate
pip install cupy-cuda12x  # For CUDA 12.x
# or pip install cupy-cuda11x  # For CUDA 11.x

**Note:** The GPU benchmark now requires CuPy and will fail cleanly if not available - no CPU fallback ensures true GPU performance measurement.
```

## Configuration

All benchmark parameters are fixed in `config.json`:

```json
{
  "instance_generation": {
    "capacity": 30,
    "demand_min": 1,
    "demand_max": 10,
    "coord_range": 100,
    "coordinates_normalized": true
  }
}
```

**Fixed Parameters (DO NOT CHANGE):**
- **Capacity**: 30
- **Demand**: Integer range [1, 10]  
- **Coordinates**: Generated as integers [0, 100], normalized to [0, 1]

## Usage

### CPU Benchmark
```bash
# Quick test (N=5, 1 instance)
python3 benchmark_exact_cpu.py --n-start 5 --n-end 5 --instances-min 1 --instances-max 1 --timeout 30

# Small benchmark (N=5-8, 10 instances each)
python3 benchmark_exact_cpu.py --n-start 5 --n-end 8 --instances-min 10 --instances-max 10 --timeout 120

# Standard benchmark (N=5-12, 20 instances each)
python3 benchmark_exact_cpu.py --n-start 5 --n-end 12 --instances-min 20 --instances-max 20 --timeout 300
```

### GPU Benchmark
```bash
# Quick test (N=5, 1 instance)
python3 benchmark_exact_gpu.py --n-start 5 --n-end 5 --instances 1 --timeout 30

# Small benchmark (N=5-8, 10 instances each)
python3 benchmark_exact_gpu.py --n-start 5 --n-end 8 --instances 10 --timeout 120

# Standard benchmark (N=5-12, 20 instances each) 
python3 benchmark_exact_gpu.py --n-start 5 --n-end 12 --instances 20 --timeout 300

# Large benchmark (N=5-20, 100 instances each)
python3 benchmark_exact_gpu.py --n-start 5 --n-end 20 --instances 100 --timeout 600
```

### Plotting Results
```bash
# Plot GPU results only
python3 plot_gpu_benchmark.py csv/gpu_results.csv --output plots/gpu_analysis

# Compare CPU vs GPU results
python3 plot_cpu_gpu_comparison.py csv/cpu_results.csv csv/gpu_results.csv --output plots/comparison
```

## Benchmark Validation

Both benchmarks display configuration validation on startup:

```
📋 Loading configuration from config.json...
✅ Config validation passed
   - Capacity: 30
   - Demand range: [1, 10]
   - Coordinate range: [0, 100] normalized to [0, 1]
🔧 Using parameters: capacity=30, demand=[1,10], coord_range=100
```

## Complete Workflow Examples

### Performance Comparison
```bash
# 1. Run CPU benchmark
python3 benchmark_exact_cpu.py --n-start 5 --n-end 10 --instances-min 10 --instances-max 10 --output csv/cpu_test.csv

# 2. Run GPU benchmark (same problem sizes)
python3 benchmark_exact_gpu.py --n-start 5 --n-end 10 --instances 10 --output csv/gpu_test.csv

# 3. Generate comparison plot
python3 plot_cpu_gpu_comparison.py csv/cpu_test.csv csv/gpu_test.csv --output plots/performance_comparison
```

### Research-Grade Benchmarking
```bash
# Comprehensive GPU benchmark
python3 benchmark_exact_gpu.py --n-start 5 --n-end 15 --instances 50 --timeout 120 --output csv/gpu_research.csv

# Equivalent CPU benchmark
python3 benchmark_exact_cpu.py --n-start 5 --n-end 15 --instances-min 50 --instances-max 50 --timeout 600 --output csv/cpu_research.csv

# Publication-ready comparison
python3 plot_cpu_gpu_comparison.py csv/cpu_research.csv csv/gpu_research.csv \
  --output plots/research_comparison \
  --title "Research-Grade CPU vs GPU CVRP Performance"
```

## Output Organization

- **`csv/`** - Benchmark results with timing and cost-per-customer metrics
- **`plots/`** - Performance visualizations and comparison charts  
- **`logs/`** - Detailed execution logs (primarily from CPU benchmarks)
- **`misc/`** - Development utilities, backups, and archived scripts



### Debug Mode

Both benchmarks support detailed debug output when the `--debug` flag is set:

```bash
# CPU benchmark with debug output
python3 benchmark_exact_cpu.py --n-start 5 --n-end 5 --instances-min 1 --instances-max 1 --debug

# GPU benchmark with debug output  
python3 benchmark_exact_gpu.py --n-start 5 --n-end 5 --instances 1 --debug
```

**Debug Output Format:**
```
🐛 DEBUG [exact_ortools_vrp] Instance 1/1, N=5, Seed=9242
   └─ CPC: 0.523962, Cost: 2.619810, Routes: [[0, 1, 4, 3, 5, 2, 0]]
🐛 DEBUG [exact_milp] Instance 1/1, N=5, Seed=9242  
   └─ CPC: 0.523962, Cost: 2.619810, Routes: [[0, 1, 4, 3, 5, 2, 0]]
```

Shows for each solver:
- **Solver name** in brackets
- **Instance number** and **total instances**  
- **Problem size** (N) and **seed** used
- **Cost per customer** (CPC) and **total cost**
- **Vehicle routes** showing the solution

## Parameter Reference

| Parameter | CPU Benchmark | GPU Benchmark | Description |
|-----------|---------------|---------------|-------------|
| Problem sizes | `--n-start`, `--n-end` | `--n-start`, `--n-end` | Customer count range |
| Instances | `--instances-min`, `--instances-max` | `--instances` | Number of random instances per size |
| Timeout | `--timeout` | `--timeout` | Maximum solver time per size (seconds) |
| Output | `--output` | `--output` | CSV file path for results |
| Logging | `--log` | - | Log file path (CPU only) |

**Note**: Vehicle capacity, demand range, and coordinate range are fixed in `config.json` and cannot be overridden via command line.

## Identical Results Guarantee

Both benchmarks generate identical instances using the same seed formula and `EnhancedCVRPGenerator`. This ensures:
- Same customer locations and demands for identical seeds
- Same optimal solutions from exact solvers  
- Fair performance comparisons based purely on computational efficiency

Example verification:
- CPU N=5: `cpc_exact_ortools_vrp=0.523962001269`
- GPU N=5: `cpc_exact_ortools_vrp=0.523962001269` ✅ Identical!
