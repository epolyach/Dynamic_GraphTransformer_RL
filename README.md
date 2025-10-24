# CVRP with Dynamic Graph Transformer and Reinforcement Learning

Deep Reinforcement Learning approach for solving the Capacitated Vehicle Routing Problem (CVRP) using Graph Attention Networks and Graph Transformers.

## Overview

This project implements and trains neural network models for CVRP using reinforcement learning:

**Supported Models:**
- **GAT+RL**: Graph Attention Network with RL
- **GT+RL**: Graph Transformer with RL
- **DGT+RL**: Dynamic Graph Transformer with RL (best performance)

**Key Features:**
- GPU-accelerated training with mixed precision (FP16/FP32)
- Curriculum learning for progressive problem difficulty
- Parallel CPU benchmarking with OR-Tools
- GPU-based exact solvers for validation
- Comprehensive configuration system

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/epolyach/Dynamic_GraphTransformer_RL.git
cd Dynamic_GraphTransformer_RL
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

4. (Optional) Install OR-Tools for benchmarking:
```bash
pip install ortools
```

---

## Quick Start

### Train a Model (GPU)

```bash
# Activate environment
source venv/bin/activate

# Train DGT model with small config
python training_gpu_prefetch/scripts/run_training_gpu.py --config configs/small.yaml --model DGT+RL --device cuda:0
```

### Run OR-Tools Benchmark

```bash
# Benchmark with OR-Tools GLS solver
python benchmark_cpu/scripts/ortools/production/run_ortools_gls.py --problem_size 20 --num_problems 1000 --time_limit 2.0 --parallel 8
```

---

## Training

### GPU Training (Recommended)

**Location:** `training_gpu_prefetch/`

Main training pipeline with GPU optimizations and batch prefetching.

#### Basic Usage

```bash
# Train with default config
python training_gpu_prefetch/scripts/run_training_gpu.py --config configs/default.yaml --model GT+RL

# Train with custom parameters
python training_gpu_prefetch/scripts/run_training_gpu.py --config configs/medium.yaml --model DGT+RL --device cuda:0 --epochs 200 --batch_size 1024
```

#### Training in Screen Session (SSH-Persistent)

For long training runs that survive SSH disconnections:

```bash
# Start training in screen
screen -dmS training bash -c "source venv/bin/activate && python training_gpu_prefetch/scripts/run_training_gpu.py --config configs/normal_gpu_1000.yaml --model DGT+RL; exec bash"

# Monitor training
screen -ls              # List sessions
screen -r training      # Attach to session
# Press Ctrl+A, then D to detach

# Check GPU usage
nvidia-smi -l 1
```

#### Key Arguments

- `--config PATH`: Configuration file
- `--model NAME`: Model (GAT+RL, GT+RL, DGT+RL)
- `--device DEVICE`: GPU device (cuda:0, cuda:1)
- `--epochs N`: Training epochs
- `--batch_size N`: Batch size
- `--mixed_precision`: Enable FP16
- `--resume PATH`: Resume from checkpoint
- `--force-retrain`: Overwrite existing model

#### Available Configs

- `tiny.yaml` - Quick experiments (N=10)
- `small.yaml` - Small problems (N=20)
- `medium.yaml` - Medium problems (N=50)
- `large.yaml` - Large problems (N=100)
- `normal_gpu_1000.yaml` - Production training

### Curriculum Learning

**Location:** `curriculum_learning/`

Progressive training from small to large problems.

```bash
cd curriculum_learning
source ~/CVRP/Dynamic_GraphTransformer_RL/venv/bin/activate
export PYTHONPATH="${HOME}/CVRP:${PYTHONPATH}"
screen -S curriculum bash -c ./start_training.sh
```

**Monitor:**
```bash
screen -ls                    # List sessions
screen -r curriculum          # Attach
tail -f logs/training_*.log   # View logs
```

---

## Benchmarking

### CPU Benchmarking (OR-Tools)

**Location:** `benchmark_cpu/scripts/ortools/production/run_ortools_gls.py`

```bash
# Run benchmark
python benchmark_cpu/scripts/ortools/production/run_ortools_gls.py --problem_size 20 --num_problems 1000 --time_limit 2.0 --parallel 8 --output results/benchmark.json

# With verbose output
python benchmark_cpu/scripts/ortools/production/run_ortools_gls.py --problem_size 50 --num_problems 100 --time_limit 5.0 --verbose
```

**Arguments:**
- `--problem_size N`: Number of customers
- `--num_problems N`: Number of instances
- `--time_limit T`: Time limit (seconds)
- `--parallel N`: Parallel workers
- `--output PATH`: Output JSON file
- `--verbose`: Detailed output

### GPU Benchmarking

**Location:** `benchmark_gpu/scripts/`

```bash
# GPU DP solver (exact solutions)
python benchmark_gpu/scripts/benchmark_gpu_truly_optimal_n10.py --num-instances 1000 --capacity 20
```

Features: Guarantees optimal solutions, batch GPU processing, supports N≤12

---

## Project Structure

```
Dynamic_GraphTransformer_RL/
├── src/                      # Core source code
│   ├── models/               # GAT, GT, DGT implementations
│   ├── generator/            # CVRP instance generator
│   ├── utils/                # Configuration and utilities
│   └── metrics/              # Cost computation
├── training_gpu_prefetch/    # Main GPU training pipeline
│   ├── scripts/              # Training scripts
│   ├── lib/                  # GPU training libraries
│   └── results/              # Training results
├── curriculum_learning/      # Curriculum training
│   ├── scripts/              # Training scripts
│   ├── schedulers/           # Curriculum schedulers
│   └── checkpoints/          # Model checkpoints
├── benchmark_cpu/            # CPU benchmarking (OR-Tools)
├── benchmark_gpu/            # GPU benchmarking
├── configs/                  # Configuration files
│   ├── default.yaml          # Base config
│   ├── tiny.yaml             # N=10
│   ├── small.yaml            # N=20
│   ├── medium.yaml           # N=50
│   └── large.yaml            # N=100
├── results_consolidated/     # Consolidated results
├── checkpoints/              # Model checkpoints (.pth)
├── utils/                    # Utility scripts
│   ├── activate_env.sh
│   ├── setup_venv.sh
│   └── generate_results_index.py
└── archive/                  # Archived docs and configs
```

---

## Configuration

All configs in `configs/` extend `default.yaml`.

**Key Parameters:**
```yaml
problem:
  num_customers: 20           # Problem size
  vehicle_capacity: 30        # Capacity
  demand_range: [1, 10]       # Demand range

training:
  batch_size: 512             # Batch size
  num_epochs: 100             # Epochs
  learning_rate: 1e-4         # Learning rate

model:
  hidden_dim: 256             # Hidden dimension
  num_heads: 4                # Attention heads
  num_layers: 4               # Transformer layers
```

---

## Results

Training results saved in:
- `training_gpu_prefetch/results/` - CSV history, configs, summaries
- `checkpoints/` - Model checkpoints (.pth files)
- `results_consolidated/` - Consolidated analysis

Benchmark results saved in:
- `benchmark_cpu/results/` - OR-Tools results
- `benchmark_gpu/results/` - GPU solver results

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{polyachenko2025dynamic,
  title={Dynamic Graph Transformer with Reinforcement Learning for CVRP},
  author={Polyachenko, Evgeny},
  booktitle={Proceedings of ICORES 2025},
  year={2025}
}
```
