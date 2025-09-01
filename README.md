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
  python run_training.py --model GT+RL --config ../configs/small.yaml
  # Or use the convenience script:
  ./run_training.sh ../configs/small.yaml GT+RL
  ```

- Train all kept models (GAT+RL, GT+RL, DGT+RL, GT-Greedy)
  ```bash
  cd training_cpu
  python run_training.py --all --config ../configs/tiny.yaml
  ```

- Generate comparison plots from saved results
  ```bash
  cd training_cpu
  python make_comparative_plot.py --config ../configs/small.yaml
  ```

Notes:
- Training uses src/generator/generator.py; no augmentation/curriculum
- Models live in src/models; advanced trainer in training_cpu/lib/advanced_trainer.py
- Results saved locally in training_cpu/results/ (models in pytorch/, CSVs in csv/, plots in plots/)


## 2) CPU Benchmarks
Location: benchmark_cpu/

- Generate per-instance CSV and evaluate solvers
  ```bash
  cd benchmark_cpu
  python run_exact.py \
    --config ../configs/small.yaml \
    --n-start 5 --n-end 12 \
    --instances 20 \
    --time-limit 5 \
    --csv results/csv/cpu_benchmark.csv
  ```

- Plot CPU results (reads the same CSV)
  ```bash
  python plot_cpu_benchmark.py \
    --csv results/csv/cpu_benchmark.csv \
    --output plots/cpu_benchmark.png
  ```

Solvers (labels match plot):
- exact_dp (N ≤ 8)
- ortools_greedy (exact_ortools_vrp_fixed)
- ortools_gls


## 3) GPU Benchmarks
Location: benchmark_gpu/

- Quick run on a fixed N and batch of instances
  ```bash
  cd benchmark_gpu
  python run_exact.py \
    --config ../configs/small.yaml \
    --seed 42 --instances 50 --time-limit 10
  ```

- GPU plotting scripts are present and will be harmonized with a unified CSV format later:
  - plot_gpu_benchmark.py
  - plot_cpu_gpu_comparison.py

Exact and heuristic GPU solvers:
- exact_gpu_dp (exact for N ≤ 16)
- exact_gpu_improved (exact for tiny N; heuristic for larger N; raises on failure — no fallback)


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
│   ├── run_training.py           # Main training script
│   ├── run_training.sh           # Convenience script
│   ├── make_comparative_plot.py  # Generate comparison plots
│   ├── lib/                      # Training library
│   │   ├── advanced_trainer.py   # RL training logic
│   │   └── rollout_baseline.py   # Rollout baseline for REINFORCE
│   └── results/                   # Local results directory
│       ├── pytorch/              # Saved models
│       ├── csv/                  # Training history
│       └── plots/                # Generated plots
│
├── benchmark_cpu/                 # CPU benchmarking
│   ├── run_exact.py              # Benchmark runner
│   ├── plot_cpu_benchmark.py     # Plot results
│   └── results/                   # Local results
│       └── csv/                  # Benchmark data
│
├── benchmark_gpu/                 # GPU benchmarking
│   ├── run_exact.py              # Benchmark runner
│   ├── plot_gpu_benchmark.py     # Plot results
│   └── plot_cpu_gpu_comparison.py # Compare CPU vs GPU
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
cd training_cpu
# Train a single model
python run_training.py --model GT+RL --config ../configs/small.yaml
# Or train all models
python run_training.py --all --config ../configs/tiny.yaml

# 4. Generate comparison plots
python make_comparative_plot.py --config ../configs/small.yaml

# 5. Run CPU benchmarks (optional)
cd ../benchmark_cpu
python run_exact.py --config ../configs/small.yaml --instances 10 --time-limit 5
python plot_cpu_benchmark.py --csv results/csv/cpu_benchmark.csv
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
