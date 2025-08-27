# Dynamic Graph Transformer for CVRP - CPU Training Branch

This branch contains a minimal setup for training and evaluating CVRP models on CPU.

## Available Models

1. **GAT+RL** - Legacy Graph Attention Network with Reinforcement Learning
2. **GT+RL** - Advanced Graph Transformer with Reinforcement Learning  
3. **DGT+RL** - Dynamic Graph Transformer with Reinforcement Learning
4. **GT-Greedy** - Graph Transformer with Greedy decoding (baseline)

## Quick Start

### Setup Environment
```bash
# Create virtual environment
./setup_venv.sh

# Activate environment
source ./activate_env.sh
```

### Training

Train a model with a specific configuration:
```bash
python run_training.py --config configs/small.yaml --model GT+RL
```

Available configurations:
- `configs/small.yaml` - 20 customers, quick training
- `configs/medium.yaml` - 50 customers, moderate training
- `configs/production.yaml` - 100 customers, full training

### Generate Comparative Plots

After training, generate comparison plots:
```bash
python python python python python python python python python p
## Project Structure

```
.
├── configs/           # Training configurations
├── results/        ├── results/        ├── results/        ├── results/     # Model implementations
│   ├── training/     # Training logic
│   ├── utils/        # Utilities
│   └── ...
├── run_training.py   # Main training script
└── make_comparative_plot.py  # Plotting script
```

## Model Perfor## Model Perfor## Model Perel## Model Perfo| Training Time | Cost/Customer |
|-------|------------|---------------|---------------|
| GAT+RL | ~380K | ~60s | 0.35-0.40 |
| GT+RL | ~420K | ~70s | 0.33-0.38 |
| DGT+RL | ~520K | ~90s | 0.3| DGT+RL | ~520K | ~90 ~42| DGT+RL | ~520K | ~90s | 0.3| DGT+RL | ~520K | ~9irements.txt` for dependencies. Main requirements:
- PyTorch (CPU version)
- NumPy
- Matplotlib
- Seaborn
- PyYAML
- tqdm
