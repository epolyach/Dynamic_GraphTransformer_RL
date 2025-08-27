# Dynamic Graph Transformer for CVRP with Reinforcement Learning

A comprehensive implementation of state-of-the-art neural architectures for solving the Capacitated Vehicle Routing Problem (CVRP) using reinforcement learning.

## 🏗️ Model Architecture Hierarchy

This project implements three main neural architectures with progressive complexity:

### 1. **GAT+RL** (Legacy Baseline) - 1.3M parameters
- Edge-aware Graph Attention Network from legacy GAT_RL project
- Multi-head pointer attention decoder (8 heads)
- Serves as baseline for comparison

### 2. **GT+RL** (Advanced Graph Transformer) - 3.8M parameters  
- Spatial & positional encoding
- Distance-aware attention with learnable biases
- Dynamic state tracking and updates
- Multi-head pointer network
- Modern transformer improvements (Pre-LN, GLU)

### 3. **DGT+RL** (Dynamic Graph Transformer) - 8.1M parameters
- Everything from GT+RL PLUS:
- Temporal memory bank (32 slots)
- Dynamic edge processing
- Adaptive graph structure
- Multi-scale temporal attention
- Progressive refinement
- Learned update schedules
- Adaptive temperature control

### 4. **GT-Greedy** (Optional Baseline)
- Deterministic greedy version of GT for comparison
- No learning, pure attention-based routing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CPU-optimized (no GPU required)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Dynamic_GraphTransformer_RL

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

```bash
# Train a single model with small config
python3 run_training.py --config configs/small_quick.yaml --models GT+RL
```

## 📋 Training Models

### Train Individual Models

```bash
# Train only the advanced GT model
python3 run_training.py --models GT+RL

# Train only the state-of-the-art DGT model
python3 run_training.py --models DGT+RL

# Train the legacy baseline
python3 run_training.py --models GAT+RL
```

### Train Multiple Models

```bash
# Train all three main models (recommended for comparison)
python3 run_training.py --models GAT+RL GT+RL DGT+RL

# Train baseline and best model
python3 run_training.py --models GAT+RL DGT+RL

# Train all models including greedy baseline
python3 run_training.py --models all
```

### Training with Different Configurations

```bash
# Quick development testing (10 customers, ~5-10 min)
python3 run_training.py --config configs/small.yaml --models GT+RL

# Research experiments (20 customers, ~2-4 hours)
python3 run_training.py --config configs/medium.yaml --models GT+RL DGT+RL

# Publication results (100 customers, ~1-2 days)
python3 run_training.py --config configs/production.yaml --models all
```

### Advanced Training Options

```bash
# Force retrain existing models
python3 run_training.py --models DGT+RL --force-retrain

# Use curriculum learning (gradually increase difficulty)
python3 run_training.py --models GT+RL --use-curriculum

# Disable data augmentation
python3 run_training.py --models GAT+RL --no-augmentation

# Custom output directory
python3 run_training.py --models DGT+RL --output-dir results/experiment1

# Disable advanced features (for debugging)
python3 run_training.py --models GT+RL --disable-advanced
```

## 📊 Generating Comparative Plots

After training, generate performance comparison plots:

```bash
# Basic comparison plot
python3 make_comparative_plot.py --config configs/small.yaml

# With exact solver baseline (computes optimal solutions)
python3 make_comparative_plot.py --config configs/small.yaml --exact 50

# Custom filename suffix
python3 make_comparative_plot.py --config configs/medium.yaml --suffix final_results
```

## 🧪 Test Instance Analysis

Create and analyze test instances with all trained models:

```bash
# Basic test instance
python3 make_test_instance.py --config configs/small.yaml

# With exact optimal solution
python3 make_test_instance.py --config configs/small.yaml --exact

# Custom seed for reproducibility
python3 make_test_instance.py --config configs/medium.yaml --seed 42
```

## 🧹 Cleaning Results

```bash
# Clean results for a specific config
python3 erase_run.py --config configs/small.yaml

# Preview what would be removed (dry run)
python3 erase_run.py --config configs/medium.yaml --dry-run

# Force cleanup without confirmation
python3 erase_run.py --config configs/production.yaml --force
```

## 📂 Project Structure

```
.
├── run_training.py              # Main training script (NEW)
├── make_comparative_plot.py     # Performance comparison plots
├── make_test_instance.py        # Test instance generation
├── erase_run.py                 # Results cleanup utility
├── src/
│   ├── models/
│   │   ├── legacy_gat.py       # GAT+RL baseline
│   │   ├── gt.py                # GT+RL advanced model
│   │   ├── dgt.py               # DGT+RL state-of-the-art
│   │   ├── greedy_gt.py        # GT-Greedy baseline
│   │   └── model_factory.py    # Model creation factory
│   ├── training/
│   │   └── advanced_trainer.py # Training loop
│   ├── data/
│   │   └── enhanced_generator.py # Data generation
│   └── utils/
│       └── config.py            # Configuration loader
├── configs/
│   ├── default.yaml            # Base configuration
│   ├── small.yaml              # Quick testing (10 customers)
│   ├── small_quick.yaml        # Very quick test
│   ├── medium.yaml             # Research (20 customers)
│   └── production.yaml         # Publication (100 customers)
└── results/
    └── [config_name]/
        ├── pytorch/            # Model checkpoints
        ├── csv/                # Training histories
        ├── plots/              # Visualizations
        └── analysis/           # Comparative results
```

## ⚙️ Configuration System

### Problem Scales

- **Small** (`configs/small.yaml`): 10 customers, 32 epochs, ~5-10 min
- **Medium** (`configs/medium.yaml`): 20 customers, 64 epochs, ~2-4 hours  
- **Production** (`configs/production.yaml`): 100 customers, 150 epochs, ~1-2 days

### Key Configuration Options

Edit YAML files to customize:
- `problem.num_customers`: Number of customers (cities)
- `problem.capacity`: Vehicle capacity
- `training.num_epochs`: Training epochs
- `training.batch_size`: Batch size
- `model.hidden_dim`: Model hidden dimension
- `model.num_heads`: Attention heads
- `model.num_layers`: Transformer layers

## 📊 Expected Performance

### Relative Improvements (20 customers, capacity 30):
- **Naive Baseline**: ~1.33 cost/customer (individual deliveries)
- **GAT+RL**: ~0.60 cost/customer (55% improvement)
- **GT+RL**: ~0.56 cost/customer (58% improvement)
- **DGT+RL**: ~0.54 cost/customer (59% improvement)

### Training Times (CPU):
- **GAT+RL**: Fastest (~30 min for medium config)
- **GT+RL**: Moderate (~45 min for medium config)
- **DGT+RL**: Slowest but best quality (~90 min for medium config)

## 🔬 Model Comparison

| Model | Parameters | Key Features | Use Case |
|-------|------------|--------------|----------|
| GAT+RL | 1.3M | Edge-aware attention, proven baseline | Quick baseline comparison |
| GT+RL | 3.8M | Modern transformer, spatial encoding | Good balance of speed/quality |
| DGT+RL | 8.1M | Dynamic adaptation, memory bank | Best quality, research focus |
| GT-Greedy | 3.8M | Deterministic, no learning | Non-learning baseline |

## 🎯 Advanced Features

### Training Features
- **REINFORCE** with baseline for variance reduction
- **Curriculum Learning**: Gradually increase problem difficulty
- **Data Augmentation**: Rotation and reflection of instances
- **Early Stopping**: Prevent overfitting
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Gradient Clipping**: Prevent exploding gradients

### Model Features (DGT+RL)
- **Temporal Memory Bank**: Tracks decision patterns
- **Dynamic Edge Processing**: Updates edge representations
- **Adaptive Graph Structure**: Adjusts connectivity during solving
- **Multi-Scale Attention**: Different temporal horizons
- **Progressive Refinement**: Stage-specific transformations
- **Learned Schedules**: Network learns when to update

## 🚨 Common Issues & Solutions

### Out of Memory
```bash
# Reduce batch size in config
# Or use smaller model
python3 run_training.py --config configs/small.yaml --models GAT+RL
```

### Slow Training
```bash
# Use fewer epochs or smaller problem size
# Edit configs/small.yaml to reduce num_epochs or num_customers
```

### Model Not Converging
```bash
# Try different learning rate or more epochs
# Edit configs/default.yaml → training.learning_rate
```

## 📈 Monitoring Training

Training progress is saved to:
- **CSV files**: `results/[config]/csv/history_[model].csv`
- **Checkpoints**: `results/[config]/pytorch/model_[model].pt`
- **Logs**: Console output shows epoch-by-epoch progress

To monitor:
```bash
# Watch training progress
tail -f results/small/csv/history_gt_rl.csv

# Plot learning curves after training
python3 make_comparative_plot.py --config configs/small.yaml
```

## 🔄 Workflow Example

Complete workflow for research experiment:

```bash
# 1. Clean previous results (optional)
python3 erase_run.py --config configs/medium.yaml --force

# 2. Train models
python3 run_training.py --config configs/medium.yaml --models GAT+RL GT+RL DGT+RL

# 3. Generate comparison plots
python3 make_comparative_plot.py --config configs/medium.yaml

# 4. Analyze on test instances
python3 make_test_instance.py --config configs/medium.yaml --exact

# 5. Results will be in:
ls results/medium/
# pytorch/  csv/  plots/  analysis/
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{dynamic_graph_transformer_cvrp,
  title={Dynamic Graph Transformer for CVRP with Reinforcement Learning},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues.

## 📞 Support

For questions or issues, please open a GitHub issue.

---

**Note**: This implementation represents state-of-the-art neural approaches to CVRP, with careful attention to proper reinforcement learning, modern transformer architectures, and rigorous constraint validation.
