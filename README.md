# CVRP — Unified Training and Benchmarking

Single-source-of-truth generator, consistent configs, strict correctness (no hidden fallbacks).

- One strict generator used for everything (training, CPU benchmarks, GPU benchmarks)
- Four models kept for training: GAT+RL, GT+RL, DGT+RL, and GT-Greedy baseline
- YAML configs only; no ad-hoc defaults; reproducible via seeds

Strict CVRP specification (single source of truth):
- Graph: 1 depot (node 0) + N customers (nodes 1..N)
- Coordinates: each node sampled on integer grid [0..coord_range], then normalized to [0,1]
- Demands: d0 = 0 for depot; for i ≥ 1, di ~ UniformInteger[demand_min, demand_max]
- Vehicle capacity: fixed integer vehicle_capacity
- Distances: Euclidean distance matrix computed on normalized coordinates
- Augmentation: disabled for both training and benchmarking
- Seeds: training and validation seeds do not overlap; benchmarking uses explicit seeds for reproducibility
- Configuration: all parameters come from YAML (configs/default.yaml + overrides); no hidden defaults in code
- Validation: strict_validation flag in experiment section controls route validation (default: true)
- Implementation: src/generator/generator.py

Distance scaling knob (for solver internals):
- configs/default.yaml → benchmark.scaling.distance_scale (default 100000)
- Used by solver internals for stable integer math; costs are scaled back; CPC = cost/N

## 1) Training (CPU)
Location: training_cpu/

- Train a single model
  ```bash
  cd training_cpu
  python scripts/run_training.py --model GT+RL --config ../configs/small.yaml
  # Or use the convenience script:
  cd scripts
  ./run_training.sh ../../configs/small.yaml GT+RL
  ```

- Train all kept models (GAT+RL, GT+RL, DGT+RL, GT-Greedy)
  ```bash
  cd training_cpu
  python scripts/run_training.py --all --config ../configs/tiny.yaml
  ```

- Generate comparison plots from saved results
  ```bash
  cd training_cpu
  python scripts/make_comparative_plot.py --config ../configs/small.yaml
  ```

Notes:
- Training uses src/generator/generator.py; no augmentation/curriculum
- Models live in src/models; advanced trainer in training_cpu/lib/advanced_trainer.py
- Results saved locally in training_cpu/results/ (models in pytorch/, CSVs in csv/, plots in plots/)

## 2) CPU Benchmarks
Location: benchmark_cpu/

### OR-Tools GLS (Latest CPU Implementation) ⭐
**Main script:** `benchmark_cpu/scripts/ortools/production/run_ortools_gls.py`

Integrated OR-Tools GLS benchmark runner with ProcessPoolExecutor parallel processing, built-in instance generation, and deterministic seed management.

#### Usage
```bash
python3 benchmark_cpu/scripts/ortools/production/run_ortools_gls.py \
    --subfolder SUBFOLDER \
    --n N \
    --instances NI \
    --timeout TIMEOUT \
    --threads NT \
    [--capacity CAPACITY] \
    [--seed SEED] \
    [--verbose]
```

#### Arguments
- `--subfolder`: Output directory name in `benchmark_cpu/results/`
- `--n`: Problem size (number of customer nodes)
- `--instances`: Total number of instances to run
- `--timeout`: Base timeout in seconds per instance (will be increased on retries)
- `--threads`: Number of parallel threads
- `--capacity`: (Optional) Vehicle capacity. Auto-calculated if not provided
- `--seed`: (Optional) Base seed for reproducible instance generation (default: 42)
- `--verbose`: (Optional) Save detailed per-thread results in addition to CPC summary

#### Key Features
- **Integrated Generation & Solving**: No external script dependencies, all functionality in one place
- **Deterministic Instance Generation**: Each instance gets unique seed = base_seed + instance_id
- **Thread-Independent Reproducibility**: Same instances generated regardless of thread count
- **Striped Thread Allocation**: Optimal load balancing across threads
- **Automatic Retry Logic**: Failed instances retry with 2x, 4x, 8x timeout
- **Dual Output Modes**: 
  - Default: Only `ortools_n{N}.json` with CPC values
  - Verbose: Both detailed thread JSONs and CPC summary
- **Real-time Monitoring**: Live progress updates and logging to `benchmark_log.txt`
- **Complete Results**: Each thread produces JSON with instance data, solutions, and metadata
- **Proper CVRP Routes**: Routes include depot returns between vehicles

#### Examples

Quick test (N=10, 20 instances, 2 threads):
```bash
python3 benchmark_cpu/scripts/ortools/production/run_ortools_gls.py \
    --subfolder "test_n10" \
    --n 10 \
    --instances 20 \
    --timeout 1 \
    --threads 2 \
    --seed 42
```

Production run with verbose output (N=50, 1000 instances, 20 threads):
```bash
python3 benchmark_cpu/scripts/ortools/production/run_ortools_gls.py \
    --subfolder "production_n50" \
    --n 50 \
    --instances 1000 \
    --timeout 30 \
    --threads 20 \
    --seed 42 \
    --verbose
```

#### Output Structure
```
benchmark_cpu/results/
└── [subfolder]/
    ├── benchmark_log.txt                 # Real-time monitoring log
    ├── ortools_n{N}.json                 # CPC values (always created)
    ├── thread_00_n[N]_[timestamp].json   # Detailed results (--verbose only)
    ├── thread_01_n[N]_[timestamp].json   # Detailed results (--verbose only)
    └── ...                                # NT total thread files (--verbose only)
```

#### Analysis Tools

**Histogram Visualization:**
```bash
cd benchmark_cpu/results/[subfolder]
python3 make_log_norm_figure_cli.py --input ortools_n10.json --output histogram.png
```

**LaTeX Table Generation:**
```bash
cd benchmark_cpu/results/[subfolder]
python3 generate_latex_table_line.py --input ortools_n10.json --timeout 1s
```

#### Seed Management
- Seeds range from `base_seed` to `base_seed + instances - 1`
- Instance i always gets seed = `base_seed + i`
- Guarantees same instances regardless of thread count
- Example: With base_seed=100 and 6 instances:
  - Thread 0 (2 threads): processes instances [0,2,4] with seeds [100,102,104]
  - Thread 1 (2 threads): processes instances [1,3,5] with seeds [101,103,105]

### Legacy CPU Benchmarks
- Unified CPU benchmark
  ```bash
  cd benchmark_cpu
  python scripts/run_exact.py \
    --config ../configs/small.yaml \
    --n-start 5 --n-end 12 \
    --instances 20 \
    --time-limit 5 \
    --csv results/csv/cpu_benchmark.csv
  ```

- Plot CPU results
  ```bash
  cd benchmark_cpu
  python scripts/plot_cpu_benchmark.py \
    --csv results/csv/cpu_benchmark.csv \
    --output plots/cpu_benchmark.png
  ```

CPU Solvers (labels match plot):
- exact_dp (N ≤ 8)
- ortools_greedy (exact_ortools_vrp_fixed)
- ortools_gls (OR-Tools-GLS)

- exact_gpu_dp (exact for N ≤ 16)
- exact_gpu_improved (exact for tiny N; heuristic for larger N; raises on failure — no fallback)
- heuristic_gpu_gls (GPU-accelerated GLS)

## 4) Quick Start

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Quick training test
cd training_cpu
python scripts/run_training.py --model GT+RL --config ../configs/tiny.yaml

# 3. OR-Tools CPU benchmark test (CPC only)
python3 benchmark_cpu/scripts/ortools/production/run_ortools_gls.py \
    --subfolder "quicktest" --n 10 --instances 10 --timeout 1 --threads 2

# 4. OR-Tools CPU benchmark with detailed output
python3 benchmark_cpu/scripts/ortools/production/run_ortools_gls.py \
    --subfolder "quicktest_verbose" --n 10 --instances 10 --timeout 1 --threads 2 --verbose

# 5. GPU benchmark test (if CUDA available)
cd benchmark_gpu/scripts
# For heuristic solver:
python3 benchmark_gpu_multi_n.py
# For optimal solver (N=10, C=20):
python3 benchmark_gpu_truly_optimal_n10.py --num-instances 100 --capacity 20

# 6. Generate comparison plots
cd training_cpu
python scripts/make_comparative_plot.py --config ../configs/small.yaml
```

## 5) Directory Structure (Updated Sept 2025)

```
Dynamic_GraphTransformer_RL/
├── src/                          # Core implementation
│   ├── generator/                # CVRP instance generator
│   ├── models/                   # Neural network models
│   └── benchmarking/            # Solver implementations
│       ├── solvers/
│       │   ├── cpu/             # CPU solvers (OR-Tools, DP)
│       │   └── gpu/             # GPU solvers (CUDA)
├── configs/                      # YAML configuration files
├── training_cpu/                 # Neural network training
│   ├── scripts/                 # Training scripts
│   ├── lib/                     # Training utilities
│   └── results/                 # Training results
├── benchmark_cpu/                # CPU benchmarking ⭐
│   ├── scripts/
│   │   └── ortools/             # OR-Tools scripts
│   │       ├── production/      # Main OR-Tools runners
│   │       │   ├── run_ortools_gls.py           # Integrated benchmark runner
│   │       │   ├── make_log_norm_figure_cli.py  # Histogram visualization
│   │       │   └── generate_latex_table_line.py # LaTeX table generation
│   │       ├── benchmarks/      # Legacy benchmark scripts  
│   │       ├── monitoring/      # Progress monitoring
│   │       └── README.md        # OR-Tools documentation
│   └── results/                 # CPU benchmark results
│       └── [subfolder]/         # User-defined output directories
│           ├── benchmark_log.txt
│           ├── ortools_n{N}.json # CPC values
│           └── thread_*.json     # Detailed results (--verbose only)
├── benchmark_gpu/                # GPU benchmarking
│   ├── scripts/
│   │   ├── benchmark_gpu_*.py            # Core GPU benchmarks
│   │   ├── gpu_cvrp_solver_truly_optimal_fixed.py  # DP exact solver (bug-free)

## Configuration

The project uses YAML configuration files located in the `configs/` directory. Key configuration files include:

- `default.yaml`: Base configuration with all default parameters
- `tiny.yaml`: Quick experiments with reduced training (10 customers, fewer epochs)
- `small.yaml`: Small problem instances (20 customers)
- `medium.yaml`: Medium problem instances (50 customers)
- `large.yaml`: Large problem instances (100 customers)

### Relative Paths in Configurations

Configuration files support relative paths for the `working_dir_path` parameter. When using relative paths (starting with `../`), they are resolved relative to the script location, not the current working directory. This allows the same configuration file to work correctly for both CPU and GPU training:

```yaml
# In configs/tiny.yaml
working_dir_path: "../results/tiny"
```

This will resolve to:
- `training_cpu/results/tiny/` when using CPU training scripts
- `training_gpu/results/tiny/` when using GPU training scripts

This approach ensures clean separation of results while using a single configuration file.


│   │   ├── gpu_cvrp_solver_scip_optimal_fixed.py   # SCIP MIP solver
│   │   ├── plotting/            # Visualization
│   │   ├── table_generation/    # LaTeX tables
│   │   ├── monitoring/          # Progress tracking
│   │   ├── tests/               # Test scripts
│   │   └── examples/            # Example scripts
│   └── results/
│       ├── plots/               # Generated figures
│       ├── tables/              # LaTeX tables
│       ├── data/                # Results data
│       └── logs/                # Output logs
├── paper_dgt/                    # Research paper
├── test_ortools_parallel.py     # OR-Tools test runner
└── ORTOOLS_SETUP_SUMMARY.md     # Setup documentation
```

## 6) Key Features

### Latest OR-Tools Implementation
- **Location:** `benchmark_cpu/scripts/ortools/production/run_ortools_gls.py`
- **Features:** 
  - Integrated instance generation with deterministic seeds
  - ProcessPoolExecutor for parallel processing
  - Automatic retry logic with exponential backoff (1x, 2x, 4x, 8x timeout)
  - Dual output modes (CPC-only or detailed with --verbose)
  - Real-time logging and progress monitoring
- **Performance:** Scales across multiple CPU cores with optimal striped allocation
- **Reproducibility:** Same instances generated regardless of thread count
- **Analysis Tools:** 
  - `make_log_norm_figure_cli.py`: Generate histograms with log-normal fits
  - `generate_latex_table_line.py`: Create publication-ready LaTeX tables

### Organized Results
- All plots, tables, and data files properly organized by type
- Clear separation between CPU and GPU benchmarking results
- User-defined output directories via `--subfolder` argument
- Automatic CPC extraction for statistical analysis

### Strict Validation
- All generated routes are validated for CVRP constraints when `strict_validation: true` (default)
- Validation checks: depot start/end, capacity constraints, all customers visited exactly once
- Can be disabled for speed: set `strict_validation: false` in config

### Reproducibility
- Fixed seeds for training/validation/testing
- Deterministic instance generation with base_seed + instance_id
- All parameters in YAML configs
- No hidden defaults or fallbacks

## 7) Performance Benchmarks

### CPU vs GPU Performance Comparison

Benchmark results for N=6 customers showing statistical precision improvements with larger sample sizes:

| Solver | Instances | Mean CPC | Std CPC  | SEM      | 2×SEM/Mean(%) | 95% CI               |
|--------|-----------|----------|----------|----------|---------------|----------------------|
| CPU    |     1,000 | 0.464466 | 0.090135 | 0.002850 |        1.23% | [0.458880, 0.470052] |
| GPU    |     1,000 | 0.460799 | 0.091148 | 0.002882 |        1.25% | [0.455150, 0.466448] |
| GPU    |    10,000 | 0.466432 | 0.089185 | 0.000892 |        0.38% | [0.464684, 0.468180] |
| GPU    |   100,000 | 0.466568 | 0.089946 | 0.000284 |        0.12% | [0.466011, 0.467125] |

**Key Findings:**
- GPU and CPU solvers achieve equivalent solution quality (overlapping confidence intervals)
- Statistical precision improves dramatically with larger sample sizes (SEM reduces by ~10× with 10K instances)
- The latest OR-Tools GLS implementation provides robust parallel processing for production use

### OR-Tools GLS Statistical Properties (N=10, 1000 instances)

| Metric | Value |
|--------|-------|
| Geometric Mean | 0.4753 |
| Geometric Std Dev | 1.2001 |
| 95% Range | [0.3324, 0.6795] |
| KS Test p-value | 0.98 |
| Algorithm | OR-Tools-GLS |

**CPC values follow a log-normal distribution**, confirmed by multiple normality tests on log(CPC).

## 8) Citation

If you use this codebase in your research, please cite:

```bibtex
@inproceedings{polyachenko2025dynamic,
  title={Dynamic Graph Transformer with Reinforcement Learning for CVRP},
  author={Polyachenko, Evgeny},
  booktitle={Proceedings of ICORES 2025},
  year={2025}
}
```

## 9) GPU Optimal Solvers (N=10)

### Available Solvers

#### 1. GPU Dynamic Programming Solver
**Location:** `benchmark_gpu/scripts/gpu_cvrp_solver_truly_optimal_fixed.py`

Exact CVRP solver using Dynamic Programming with exponential state space exploration.

**Features:**
- Guarantees truly optimal solutions
- Batch processing of multiple instances on GPU
- Memory-efficient implementation for N≤12
- Full route recovery with optimal TSP tours

**Usage:**
```bash
cd benchmark_gpu/scripts
python3 benchmark_gpu_truly_optimal_n10.py --num-instances 10000 --capacity 20
```

#### 2. GPU Optimized DP Solver v2
**Location:** `benchmark_gpu/scripts/gpu_cvrp_solver_scip_optimal_fixed.py`

Enhanced version with advanced GPU optimizations.

**Features:**
- Multiple optimization strategies (pruning, symmetry breaking)
- Configurable solver modes (DP, Branch-and-Bound, Hybrid)
- Optimized memory layout for GPU access patterns
- Support for C=20, 30 and other capacity values

**Usage:**
```bash
python3 gpu_cvrp_solver_scip_optimal_fixed.py --benchmark --mode dp
```

#### 3. SCIP-based Optimal Solver
**Location:** `benchmark_gpu/scripts/gpu_cvrp_solver_scip_optimal_fixed.py`

Mixed Integer Programming solver using SCIP (Solving Constraint Integer Programs).

**Features:**
- Industry-standard MIP solver
- Miller-Tucker-Zemlin (MTZ) formulation for subtour elimination
- Provides optimality certificates
- Can handle larger instances (N>10) given enough time

**Installation:**
```bash
pip install pyscipopt
```

**Usage:**
```bash
python3 gpu_cvrp_solver_scip_optimal_fixed.py --benchmark --time-limit 60
```

### Performance Comparison (N=10)

| Solver | Algorithm | Throughput (inst/sec) | 1K instances | 10K instances | Optimality |
|--------|-----------|----------------------|--------------|---------------|------------|
| GPU DP v1 | Dynamic Programming | 0.5-1.0 | 0.5-1 hours | 5-10 hours | Guaranteed |
| GPU DP v2 | DP + Optimizations | 1.0-2.0 | 0.3-0.5 hours | 3-5 hours | Guaranteed |
| SCIP | Mixed Integer Programming | 0.01-0.1 | 3-30 hours | 1-10 days | Guaranteed* |

*SCIP guarantees optimality if solved to completion within time limit

### Recommended Use Cases

- **For production runs (1K-10K instances):** Use GPU DP solvers for best performance
- **For validation:** Use SCIP to verify a subset of solutions
- **For research:** Compare all three approaches to understand trade-offs

### Hardware Requirements

- **GPU DP Solvers:** NVIDIA GPU with CUDA support, 8+ GB VRAM recommended
- **SCIP Solver:** CPU-based, benefits from multiple cores

