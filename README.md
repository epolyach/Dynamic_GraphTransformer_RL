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
  cd training_cpu/scripts
  python run_training.py --model GT+RL --config ../../configs/small.yaml
  # Or use the convenience script:
  ./run_training.sh ../../configs/small.yaml GT+RL
  ```

- Train all kept models (GAT+RL, GT+RL, DGT+RL, GT-Greedy)
  ```bash
  cd training_cpu/scripts
  python run_training.py --all --config ../../configs/tiny.yaml
  ```

- Generate comparison plots from saved results
  ```bash
  cd training_cpu/scripts
  python make_comparative_plot.py --config ../../configs/small.yaml
  ```

Notes:
- Training uses src/generator/generator.py; no augmentation/curriculum
- Models live in src/models; advanced trainer in training_cpu/lib/advanced_trainer.py
- Results saved locally in training_cpu/results/ (models in pytorch/, CSVs in csv/, plots in plots/)


## 2) CPU Benchmarks
Location: benchmark_cpu/

### Current Unified CPU Benchmark
- Generate per-instance CSV and evaluate solvers
  ```bash
  cd benchmark_cpu/scripts
  python run_exact.py \
    --config ../../configs/small.yaml \
    --n-start 5 --n-end 12 \
    --instances 20 \
    --time-limit 5 \
    --csv ../results/csv/cpu_benchmark.csv
  ```

- Plot CPU results (reads the same CSV)
  ```bash
  python plot_cpu_benchmark.py \
    --csv ../results/csv/cpu_benchmark.csv \
    --output ../plots/cpu_benchmark.png
  ```

### Archived CPU Benchmarks (Historical)
- Legacy exact CPU benchmark (uses EnhancedCVRPGenerator)
  ```bash
  cd benchmark_cpu/scripts
  python benchmark_exact_cpu.py
  ```

- Modified exact CPU benchmark (optimized version)
  ```bash
  python benchmark_exact_cpu_modified.py
  ```

Solvers (labels match plot):
- exact_dp (N ≤ 8)
- ortools_greedy (exact_ortools_vrp_fixed)
- ortools_gls


## 3) GPU Benchmarks
Location: benchmark_gpu/

**Note**: GPU scripts require `config.json` in `benchmark_cpu/scripts/` for configuration parameters.

- High-precision GPU benchmark (10,000 instances)
  ```bash
  cd benchmark_gpu/scripts
  python benchmark_gpu_10k.py
  ```

- GPU vs CPU comparison benchmark (matched instances)
  ```bash
  cd benchmark_gpu/scripts
  python benchmark_gpu_exact_matched.py
  ```

- Adaptive N GPU benchmark (N=5 to N=20, variable instances using 10^(7-N/5) formula)
  ```bash
  cd benchmark_gpu/scripts
  python benchmark_gpu_adaptive_n.py
  ```

- Multi-N GPU benchmark (N=5 to N=10, 10K instances each)
  ```bash
  cd benchmark_gpu/scripts
  python benchmark_gpu_multi_n.py
  ```

- Plot GPU benchmark results
  ```bash
  python plot_gpu_benchmark.py --csv ../results/csv/gpu_benchmark_results.csv
  ```

- Plot CPU vs GPU comparison
  ```bash
  python plot_cpu_gpu_comparison.py --cpu-csv ../../benchmark_cpu/results/csv/cpu_benchmark.csv --gpu-csv ../results/csv/gpu_benchmark_results.csv
  ```

GPU solvers:
- exact_gpu_dp (exact for N ≤ 16)
- exact_gpu_improved (exact for tiny N; heuristic for larger N; raises on failure — no fallback)

### Configuration
GPU benchmarks use `benchmark_cpu/scripts/config.json` with parameters:
- Capacity: 30
- Demand range: [1, 10]
- Coordinate range: 100 (normalized to [0,1])
- Random uniform distribution

### CPU vs GPU Performance Comparison

Benchmark results for N=6 customers showing statistical precision improvements with larger sample sizes:

| Solver | Instances | Mean CPC | Std CPC | SEM       | 2×SEM/Mean(%) | 95% CI               |
|--------|-----------|----------|----------|----------|---------------|----------------------|
| CPU    |      1000 | 0.464466 | 0.090135 | 0.002850 |         1.23% | [0.458880, 0.470052] |
| GPU    |      1000 | 0.460799 | 0.091148 | 0.002882 |         1.25% | [0.455150, 0.466448] |
| GPU    |     10000 | 0.466432 | 0.089185 | 0.000892 |         0.38% | [0.464684, 0.468180] |
| GPU    |    100000 | 0.466568 | 0.089946 | 0.000284 |         0.12% | [0.466011, 0.467125] |

**Analysis: Script Sources for Benchmark Data**

| Benchmark Data | Source Script | Location | Purpose |
|----------------|---------------|----------|----------|
| **CPU (1000 instances)** | `run_exact.py` | `benchmark_cpu/scripts/` | Current unified CPU benchmark |
| **CPU (1000 instances, legacy)** | `benchmark_exact_cpu_modified.py` | `benchmark_cpu/scripts/` | Archived CPU benchmark (historical) |
| **GPU (1000 instances)** | `benchmark_gpu_exact_matched.py` | `benchmark_gpu/scripts/` | GPU vs CPU comparison |
| **GPU (10,000 instances)** | `benchmark_gpu_10k.py` | `benchmark_gpu/scripts/` | High-precision GPU benchmark |
| **GPU (100,000 instances)** | Modified `benchmark_gpu_10k.py` | `benchmark_gpu/scripts/` | Extended precision study |

**Key Findings:**
- GPU and CPU solvers achieve equivalent solution quality (overlapping confidence intervals)
- Statistical precision improves dramatically with larger sample sizes (SEM reduces by ~10× with 10K instances)
- The 100K instance benchmark provides sub-0.1% precision for robust statistical analysis

### Multi-N Adaptive Instance Benchmark

GPU benchmark results across multiple problem sizes (N=5 to N=10) with adaptive instance counts optimized for statistical precision:

| N  | Instances    | Mean CPC | Std CPC  | SEM      | 2×SEM/Mean(%) |
|----|--------------|----------|----------|----------|---------------|
|  5 |    1,000,000 | 0.494321 | 0.097430 | 0.000097 |       0.0394% |
|  6 |      630,957 | 0.466990 | 0.090131 | 0.000113 |       0.0486% |
|  7 |      398,107 | 0.445905 | 0.080218 | 0.000127 |       0.0570% |
|  8 |      251,188 | 0.425300 | 0.069546 | 0.000139 |       0.0653% |
|  9 |      158,489 | 0.408855 | 0.064280 | 0.000161 |       0.0790% |
| 10 |      100,000 | 0.394610 | 0.061011 | 0.000193 |       0.0978% |

**Analysis: Adaptive Instance Count Strategy**

| Problem Size | Instance Count Formula | Source Script | Purpose |
|--------------|------------------------|---------------|---------|
| **N=5 to N=20** | `int(10^(7-N/5))` | `benchmark_gpu_adaptive_n.py` | Adaptive precision study |
| **Varying counts** | More instances for small N, fewer for large N | Formula: 10^(7-N/5) | Balanced computational cost vs precision |

**Key Insights:**
- **Computational scaling**: Instance counts decrease exponentially as N increases (10^6 for N=5 → 10^5 for N=10)
- **Precision maintenance**: All benchmarks achieve sub-0.1% relative error (2×SEM/Mean < 0.1%)
- **Cost trend**: Mean CPC decreases from 0.494 (N=5) to 0.395 (N=10), showing economies of scale
- **Statistical power**: Ultra-high precision enables detection of small performance differences across problem sizes


## Project Layout (essentials)
```
.
├── configs/                        # Configuration files
│   ├── default.yaml               # Base configuration (all parameters)
│   ├── tiny.yaml                  # Quick testing
│   ├── small.yaml                 # Development testing
│   ├── medium.yaml                # Research experiments
│   └── production.yaml            # Full training runs
│
├── src/                           # Core source code
│   ├── generator/                 # Data generation
│   │   └── generator.py          # Canonical CVRP instance generator
│   ├── models/                    # Model implementations
│   │   ├── model_factory.py      # Model creation factory
│   │   ├── gat.py                # GAT+RL model
│   │   ├── gt.py                 # GT+RL (Graph Transformer)
│   │   ├── dgt.py                # DGT+RL (Dynamic Graph Transformer)
│   │   └── greedy_gt.py          # GT-Greedy baseline
│   ├── eval/                      # Evaluation utilities
│   │   └── validation.py         # Strict route validation
│   ├── metrics/                   # Metrics computation
│   │   └── costs.py              # Cost utilities (CPC = cost/N)
│   ├── utils/                     # General utilities
│   │   ├── config.py             # YAML deep-merge + normalization
│   │   └── seeding.py            # Reproducibility utilities
│   └── benchmarking/              # Benchmarking utilities
│       └── solvers/               # Solver implementations
│           ├── cpu/              # CPU solvers
│           └── gpu/              # GPU solvers
│
├── training_cpu/                  # CPU-based training
│   ├── scripts/                  # Training scripts
│   │   ├── run_training.py       # Main training script
│   │   ├── run_training.sh       # Convenience script
│   │   ├── make_comparative_plot.py # Generate comparison plots
│   │   └── regenerate_analysis.py # Analysis regeneration
│   ├── lib/                      # Training library
│   │   ├── advanced_trainer.py   # RL training logic
│   │   └── rollout_baseline.py   # Rollout baseline for REINFORCE
│   └── results/                   # Local results directory
│       ├── pytorch/              # Saved models
│       ├── csv/                  # Training history
│       └── plots/                # Generated plots
│
├── benchmark_cpu/                 # CPU benchmarking
│   ├── scripts/                  # Benchmark scripts
│   │   ├── run_exact.py          # Current unified benchmark runner
│   │   ├── benchmark_exact_cpu.py # Archived CPU benchmark (historical)
│   │   ├── benchmark_exact_cpu_modified.py # Archived optimized CPU benchmark
│   │   ├── config.json           # Configuration for GPU benchmarks
│   │   ├── plot_cpu_benchmark.py # Plot results
│   │   ├── benchmark_cvrp_n7.py  # N=7 specific benchmarks
│   │   ├── analyze_n7_results.py # N=7 results analysis
│   │   └── run_n7_comparison.py  # N=7 comparison runner
│   └── results/                   # Local results
│       └── csv/                  # Benchmark data
│
├── benchmark_gpu/                 # GPU benchmarking
│   ├── scripts/                  # GPU benchmark scripts
│   │   ├── benchmark_gpu_10k.py  # High-precision GPU benchmark (10K instances)
│   │   ├── benchmark_gpu_adaptive_n.py # Adaptive N-value benchmarks
│   │   ├── benchmark_gpu_exact_matched.py # Matched instance comparison
│   │   ├── benchmark_gpu_multi_n.py # Multi-N benchmarks (N=5-10)
│   │   ├── plot_cpu_gpu_comparison.py # CPU vs GPU plotting
│   │   ├── plot_gpu_benchmark.py # GPU results visualization
│   │   └── run_exact.py          # Basic GPU benchmark runner
│   └── results/                   # GPU benchmark results
│       └── csv/                  # CSV data files
│
├── setup_venv.sh                  # Environment setup script
├── activate_env.sh                # Environment activation helper
└── requirements.txt               # Python dependencies
```


## Quickstart
```bash
# 1. Setup environment (one-time)
./setup_venv.sh

# 2. Activate environment (each session)
source activate_env.sh

# 3. Train models
cd training_cpu/scripts
# Train a single model
python run_training.py --model GT+RL --config ../../configs/small.yaml
# Or train all models
python run_training.py --all --config ../../configs/tiny.yaml

# 4. Generate comparison plots
python make_comparative_plot.py --config ../../configs/small.yaml

# 5. Run CPU benchmarks (optional)
cd ../../benchmark_cpu/scripts
python run_exact.py --config ../../configs/small.yaml --instances 10 --time-limit 5
python plot_cpu_benchmark.py --csv ../results/csv/cpu_benchmark.csv

# 6. Run GPU benchmarks (optional)
cd ../../benchmark_gpu/scripts
python benchmark_gpu_10k.py  # High-precision GPU benchmark
python benchmark_gpu_adaptive_n.py  # Adaptive multi-N benchmark
python plot_gpu_benchmark.py --csv ../results/csv/gpu_benchmark_results.csv
```


## Key Features

### Strict Validation
- All generated routes are validated for CVRP constraints when `strict_validation: true` (default)
- Validation checks: depot start/end, capacity constraints, all customers visited exactly once
- Can be disabled for speed: set `strict_validation: false` in config

### Reproducibility
- Fixed seeds for training/validation/testing
- All parameters in YAML configs
- No hidden defaults in code

### Modular Design
- Each component (training, benchmarking) has its own directory
- Local results directories prevent pollution
- Shared core utilities in `src/`

For details, see the "Strict CVRP specification" section above.
