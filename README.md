# Dynamic Graph Transformer for CVRP - CPU Training Branch

This branch contains a minimal setup for training and evaluating CVRP models on CPU, with recent improvements to validation methodology and multi-model training support.

## Available Models

1. **GAT+RL** - Legacy Graph Attention Network with Reinforcement Learning
2. **GT+RL** - Advanced Graph Transformer with Reinforcement Learning  
3. **DGT+RL** - Dynamic Graph Transformer with Reinforcement Learning
4. **GT-Greedy** - Graph Transformer with Greedy decoding (baseline)

## Recent Improvements (August 2024)

### 1. Fixed Validation Strategy
- **Problem**: Validation was using different policy than training (fixed temp=0.1, greedy=True)
- **Solution**: Now uses same temperature and sampling strategy as training
- **Impact**: Reduced training-validation gap from 2-15% to <2%
- **Details**: See `VALIDATION_FIX_DOCUMENTATION.md`

### 2. Multi-Model Training Support
- **New Feature**: Added `--all` option to train all models sequentially
- **Benefit**: Train complete model suite with single command

## Quick Start

### Setup Environment
```bash
# Create virtual environment
./setup_venv.sh

# Activate environment
source ./activate_env.sh
```

### Training

#### Train a Single Model
```bash
python run_training.py --model DGT+RL --config configs/small.yaml
```

#### Train All Models (NEW)
```bash
python run_training.py --all --config configs/small.yaml
```

#### Force Retrain Existing Models
```bash
python run_training.py --all --force-retrain --config configs/small.yaml
```

### Available Configurations
- `configs/tiny.yaml` - 7 customers, very quick experiments
- `configs/small.yaml` - 10 customers, quick training
- `configs/medium.yaml` - 50 customers, moderate training  
- `configs/production.yaml` - 100 customers, full training

### Command Line Options

```bash
python run_training.py --help

Options:
  --config CONFIG       Path to configuration file
  --model {GAT+RL,GT+RL,DGT+RL,GT-Greedy}
                       Model to train (default: GT+RL)
  --all                Train all available models sequentially (NEW)
  --force-retrain      Force retraining even if saved model exists
  --output-dir OUTPUT_DIR
                       Override output directory from config
```

## Validation Methodology

The validation now follows best practices from RL literature:

1. **Temperature Matching**: Validation uses the same temperature as current training epoch
2. **Stochastic Sampling**: Validation samples from distribution (greedy=False) like training
3. **Seed Management**: Validation seeds have 1M offset to ensure no overlap with training data

This ensures we "validate what we train" - testing the same stochastic policy being learned.

## Project Structure

```
.
├── configs/                    # Training configurations
├── results/                    # Training outputs
│   └── {config_name}/
│       ├── csv/               # Training history CSVs
│       ├── plots/             # Visualization plots
│       └── pytorch/           # Saved model checkpoints
├── src/
│   ├── data/                  # Data generation
│   ├── models/                # Model implementations
│   ├── training/              # Training logic (includes validation fixes)
│   ├── eval/                  # Evaluation utilities
│   └── utils/                 # Helper utilities
├── run_training.py            # Main training script (supports --all)
├── analyze_validation_strategies.py  # Validation analysis tool
└── make_comparative_plot.py   # Plotting script
```

## Model Performance (Small Config - 10 customers)

| Model | Parameters | Training Time | Best Val Cost |
|-------|------------|---------------|---------------|
| GAT+RL | ~380K | ~60s | 0.47-0.48 |
| GT+RL | ~420K | ~70s | 0.46-0.47 |
| DGT+RL | ~520K | ~90s | 0.46-0.47 |
| GT-Greedy | ~420K | N/A | 0.49-0.50 |

*Note: Performance varies based on random seed and specific problem instances*

## Analyzing Results

### Generate Comparative Plots
After training, generate comparison plots:
```bash
python make_comparative_plot.py
```

### Analyze Validation Strategies
Compare different validation approaches:
```bash
python analyze_validation_strategies.py
```

## Key Files

- `run_training.py` - Main training script with multi-model support
- `src/training/advanced_trainer.py` - Contains improved validation logic
- `VALIDATION_FIX_DOCUMENTATION.md` - Detailed explanation of validation improvements
- `configs/*.yaml` - Configuration files for different problem sizes

## Requirements

See `requirements.txt` for full dependencies. Main requirements:
- PyTorch (CPU version)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- PyYAML
- tqdm

## Citation

Based on approaches from:
- Kool et al. (2019) "Attention, Learn to Solve Routing Problems!"
- Graph Transformer architectures for combinatorial optimization

## License

Research code for academic purposes.
