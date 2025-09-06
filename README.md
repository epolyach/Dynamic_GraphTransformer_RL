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

The latest OR-Tools GLS implementation with ProcessPoolExecutor parallel processing:

```bash
# Direct execution
python3 benchmark_cpu/scripts/ortools/production/run_ortools_gls.py

# Or use the interactive test runner
./run_ortools_test.sh
```

**Features:**
- ProcessPoolExecutor for true parallel processing across CPU cores
- Configurable thread counts for different problem sizes
- Striped instance allocation for optimal load balancing
- Individual JSON output per thread
- Support for N=10, 20, 50, 100 with automatic capacity calculation

**Test configurations available:**
- Quick test: N=10,20 with 2s timeout (~10 seconds)
- Medium test: N=10,20,50,100 with 5s timeout (~40 seconds)  
- Full production: 18 threads, multiple configurations

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
- ortools_gls

## 3) GPU Benchmarks
Location: benchmark_gpu/

### Organized Structure (Updated Sept 2025)
```
benchmark_gpu/
├── scripts/
│   ├── benchmark_gpu_*.py        # Core GPU benchmarks (9 scripts)
│   ├── plotting/                 # Visualization scripts (7 files)
│   ├── table_generation/         # LaTeX table generation (6 files)
│   ├── monitoring/               # Progress monitoring (3 files)
│   ├── tests/                    # Test scripts (9 files)
│   ├── examples/                 # Example scripts (3 files)
│   └── archive/                  # Backup files
└── results/
    ├── plots/                    # Generated figures
    ├── tables/                   # LaTeX tables
    ├── data/                     # CSV/JSON results
    └── logs/                     # Output logs
```

### Core GPU Benchmarks
- High-precision GPU benchmark (10,000 instances)
  ```bash
  cd benchmark_gpu
  python scripts/benchmark_gpu_10k.py
  ```

- GPU vs CPU comparison benchmark (matched instances)
  ```bash
  cd benchmark_gpu
  python scripts/benchmark_gpu_exact_matched.py
  ```

- Adaptive N GPU benchmark (N=5 to N=20)
  ```bash
  cd benchmark_gpu
  python scripts/benchmark_gpu_adaptive_n.py
  ```

- Advanced GPU GLS heuristic
  ```bash
  cd benchmark_gpu
  python scripts/benchmark_gpu_heuristic_gls_advanced.py
  ```

### Visualization and Analysis
- Plot GPU benchmark results
  ```bash
  cd benchmark_gpu
  python scripts/plotting/plot_gpu_benchmark.py --csv results/data/gpu_benchmark_results.csv
  ```

- Plot CPU vs GPU comparison
  ```bash
  cd benchmark_gpu
  python scripts/plotting/plot_cpu_gpu_comparison.py \
    --cpu-csv ../benchmark_cpu/results/csv/cpu_benchmark.csv \
    --gpu-csv results/data/gpu_benchmark_results.csv
  ```

- Generate LaTeX tables
  ```bash
  cd benchmark_gpu
  python scripts/table_generation/generate_gpu_latex_tables.py
  ```

GPU solvers:
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

# 3. OR-Tools CPU benchmark test
./run_ortools_test.sh
# Choose option 1 for quick test

# 4. GPU benchmark test (if CUDA available)
cd benchmark_gpu
python scripts/benchmark_gpu_multi_n.py

# 5. Generate comparison plots
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
│   │       ├── benchmarks/      # Core benchmark scripts  
│   │       └── monitoring/      # Progress monitoring
│   └── results/                 # CPU benchmark results
├── benchmark_gpu/                # GPU benchmarking
│   ├── scripts/
│   │   ├── benchmark_gpu_*.py   # Core GPU benchmarks
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
├── run_ortools_test.sh          # Interactive test script
└── ORTOOLS_SETUP_SUMMARY.md     # Setup documentation
```

## 6) Key Features

### Latest OR-Tools Implementation
- **Location:** `benchmark_cpu/scripts/ortools/production/run_ortools_gls.py`
- **Features:** ProcessPoolExecutor, configurable threading, production-ready
- **Performance:** Scales across multiple CPU cores with optimal load balancing

### Organized Results
- All plots, tables, and data files properly organized by type
- Clear separation between CPU and GPU benchmarking results
- Automated test runners with multiple configuration options

### Strict Validation
- All generated routes are validated for CVRP constraints when `strict_validation: true` (default)
- Validation checks: depot start/end, capacity constraints, all customers visited exactly once
- Can be disabled for speed: set `strict_validation: false` in config

### Reproducibility
- Fixed seeds for training/validation/testing
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
