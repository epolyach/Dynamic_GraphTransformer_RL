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
  python scripts/regenerate_analysis.py --config ../configs/small.yaml
  ```

Notes:
- Training uses src/generator/generator.py; no augmentation/curriculum
- Models live in src/models; advanced trainer in training_cpu/lib/advanced_trainer.py
- Results saved locally in training_cpu/results/ (models in pytorch/, CSVs in csv/, plots in plots/)


## 1.5) Training (GPU)
Location: training_gpu/

GPU-optimized training with mixed precision support, efficient batch processing, and **hybrid baseline** capabilities.

### Quick Start
```bash
# Train a single model with GPU optimizations
cd /home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL
source venv/bin/activate
python training_gpu/scripts/run_training_gpu.py --config configs/medium.yaml --model GT+RL --device cuda:0

# Force retrain if model exists
python training_gpu/scripts/run_training_gpu.py --config configs/tiny.yaml --model GT+RL --force-retrain

# Train with optimized tiny config (large batch size)
python training_gpu/scripts/run_training_gpu.py --config configs/tiny_gpu_optimized.yaml --model GT+RL

# Train with hybrid baseline (automatic rollout→critic switching)
python training_gpu/scripts/run_training_gpu.py --config configs/experiment_5_curriculum.yaml --model GT+RL
```

### GPU-Specific Features
- **Mixed Precision Training**: Automatically enabled for N≥20 (FP16/FP32)
- **GPU Cost Computation**: Route costs computed on GPU via `src/metrics/gpu_costs.py`
- **Optimized Data Pipeline**: Numpy arrays converted to GPU tensors on arrival
- **Configurable Batch Sizes**: Use larger batches (2048-8192) for better GPU utilization
- **GPU-Specific Validation**: Handles GPU tensors without CPU transfers
- **Hybrid Baseline**: Combines rollout and critic baselines for superior performance

### Hybrid Baseline ⭐ NEW
The hybrid baseline implementation (`training_gpu/lib/critic_baseline.py`) provides an advanced training strategy that automatically switches between baseline types:

**How it works:**
1. **Early training (epochs 0-50)**: Uses rollout baseline for stable early learning
2. **Later training (epochs 50+)**: Switches to critic baseline for faster convergence
3. **Automatic switching**: Configured via `baseline_switch_epoch` parameter

**Configuration example:**
```yaml
# In configs/experiment_5_curriculum.yaml
training_advanced:
  use_hybrid_baseline: true
  baseline_switch_epoch: 50    # Switch at epoch 50

baseline:
  type: "hybrid"                # Enables hybrid baseline
  
  # Rollout baseline config (early epochs)
  rollout:
    update:
      frequency: 2
      warmup_epochs: 2
  
  # Critic baseline config (later epochs)
  critic:
    hidden_dim: 256
    num_layers: 2
    learning_rate: 5e-4
```

**Benefits:**
- More stable training in early epochs with rollout baseline
- Faster convergence in later epochs with learned critic
- Automatic mode detection and logging
- Seamless switching without training interruption

### Configuration Guidelines
```yaml
# Recommended GPU settings by problem size
gpu:
  mixed_precision: false  # For N≤10 (overhead > benefit)
  mixed_precision: true   # For N≥20 (significant speedup)
  batch_size: 4096       # For N=10 with 50GB GPU memory
  batch_size: 1024-2048  # For N=50-100
```

### Performance Optimizations Applied
Based on extensive profiling (see `MD/FINAL_FIX_SUMMARY.md`), several critical optimizations have been implemented:

1. **Eliminated GPU Transfer Overhead**: Distance matrices stay on CPU where they're used
2. **Fixed Code Duplication**: Removed 9x redundant cost computations
3. **Optimized Data Movement**: Only move necessary tensors (coords/demands) to GPU
4. **Result**: 2-3x speedup (from 48-70s to ~22s per epoch)

### Performance Notes
- **N≤20**: CPU training may be faster due to GPU overhead
- **N≥50**: GPU shows significant speedup, especially with large batches
- **Memory**: RTX A6000 (48GB) can handle batch_size=8192 for N=10
- **Hybrid Baseline**: Best for longer training runs (100+ epochs)

### Available Configurations
- `configs/tiny_gpu_optimized.yaml` - N=10, batch_size=4096, optimized for GPU
- `configs/medium_gpu.yaml` - N=50, balanced GPU settings
- `configs/experiment_5_curriculum.yaml` - Hybrid baseline with curriculum learning
- All standard configs now include GPU sections with appropriate settings

### Monitoring

### Sequential Training Scripts ⭐ NEW
For running multiple experiments sequentially without manual intervention:

**Basic sequential runner:**
```bash
./run_seq.sh config1.yaml config2.yaml config3.yaml
```

**SSH-persistent sequential runner (recommended):**
```bash
./run_seq_nohup.sh config1.yaml config2.yaml config3.yaml
```

**Example usage:**
```bash
# Run three tiny GPU experiments sequentially
./run_seq_nohup.sh configs/tiny_gpu_150.yaml configs/tiny_gpu_500.yaml configs/tiny_gpu_750.yaml

# Run experiment vs medium comparison
./run_seq_nohup.sh configs/experiment_rollout_only.yaml configs/medium_experiment_rollout_only.yaml
```

**Key Features:**
- **SSH-Resistant**: Uses `nohup` to survive SSH disconnections
- **Screen Sessions**: Each experiment runs in its own screen session
- **Automatic Sequencing**: Waits for each experiment to complete before starting next
- **Force Retrain**: Automatically overwrites existing results
- **Detailed Logging**: Timestamped logs with progress tracking
- **Early Stopping Disabled**: Ensures full epoch completion (100 epochs)

**Monitoring sequential training:**
```bash
# Monitor overall progress
tail -f nohup_sequential_TIMESTAMP.log

# Check if process is still running
ps -p PID
screen -ls

# Monitor individual experiment
screen -r seq_tiny_gpu_150_PID

# Kill if needed
kill PID
```
```bash
# Check GPU utilization during training
nvidia-smi -l 1

# View training progress in screen
screen -ls  # List sessions
screen -r gpu_tiny_training  # Attach to session
# Ctrl+A, D to detach
```

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

# 7. Sequential training (multiple configs automatically)
./run_seq_nohup.sh configs/tiny_gpu_150.yaml configs/tiny_gpu_500.yaml configs/tiny_gpu_750.yaml
cd training_cpu
python scripts/regenerate_analysis.py --config ../configs/small.yaml
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
│   │   ├── run_training.py      # Main training script
│   │   ├── regenerate_analysis.py # Re-analyze results
│   │   ├── run_training.sh      # Shell wrapper
│   │   ├── experiments/         # Experimental scripts
│   │   │   ├── run_training_gt_experiments.py
│   │   │   └── run_training_optimizer_experiment.py
│   │   ├── plotting/            # Plotting utilities
│   │   │   └── plot_training_80mm.py
│   │   └── table_generation/    # Table generation tools
│   ├── lib/                     # Training utilities
│   └── results/                 # Training results
├── training_gpu/                 # GPU-optimized training ⭐
│   ├── scripts/
│   │   └── run_training_gpu.py  # Main GPU training script
│   ├── lib/
│   │   ├── advanced_trainer_gpu.py    # GPU trainer with optimizations
│   │   ├── rollout_baseline_gpu_fixed.py  # Fixed rollout baseline
│   │   └── critic_baseline.py         # Hybrid baseline implementation
│   └── results/                 # GPU training results
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
├── MD/                           # Documentation of fixes and improvements
│   ├── FINAL_FIX_SUMMARY.md     # GPU transfer overhead fix
│   ├── PERFORMANCE_FIX_SUMMARY.md  # Code duplication fix
│   └── TRAINING_IMPLEMENTATION.md  # Training system documentation
├── run_seq.sh                    # Sequential training runner
├── run_seq_nohup.sh              # SSH-persistent sequential runner
├── paper_dgt/                    # Research paper
├── test_ortools_parallel.py     # OR-Tools test runner
└── ORTOOLS_SETUP_SUMMARY.md     # Setup documentation
```

## Configuration

The project uses YAML configuration files located in the `configs/` directory. Key configuration files include:

- `default.yaml`: Base configuration with all default parameters
- `tiny.yaml`: Quick experiments with reduced training (10 customers, fewer epochs)
- `small.yaml`: Small problem instances (20 customers)
- `medium.yaml`: Medium problem instances (50 customers)
- `large.yaml`: Large problem instances (100 customers)
- `experiment_5_curriculum.yaml`: Hybrid baseline with curriculum learning

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


## 6) Key Features

### Hybrid Baseline (NEW)
- **Location:** `training_gpu/lib/critic_baseline.py`
- **Features:**
  - Automatic switching between rollout and critic baselines
  - Configurable switch epoch (default: 50)
  - Combines stability of rollout baseline with efficiency of learned critic
  - Seamless mode transitions without training interruption
- **Performance:** Best for longer training runs where critic can learn value estimates

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
|--------|----------|
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
|--------|-----------|----------------------|--------------|---------------|--------------|
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

