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
python3 run_training.py --model DGT+RL --config configs/small.yaml
```

#### Train All Models (NEW)
```bash
python3 run_training.py --all --config configs/small.yaml
```

#### Force Retrain Existing Models
```bash
python3 run_training.py --all --force-retrain --config configs/small.yaml
```

### Available Configurations
- `configs/tiny.yaml` - 7 customers, very quick experiments
- `configs/small.yaml` - 10 customers, quick training
- `configs/medium.yaml` - 50 customers, moderate training  
- `configs/production.yaml` - 100 customers, full training

### Command Line Options

```bash
python3 run_training.py --help

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
python3 make_comparative_plot.py --config configs/small.yaml
```

Note on required analysis artifact

make_comparative_plot.py loads a consolidated analysis file from your working directory:
- results/<scale>/analysis/enhanced_comparative_study.pt (preferred), or
- results/<scale>/analysis/comparative_study_complete.pt (legacy)

This file is a small Torch checkpoint containing:
- results: per-model histories (train_costs, val_costs, final_val_cost, etc.)
- training_times: per-model training time in seconds
- config: the configuration dict used for training (to infer problem size, etc.)

Which script produces it?

Historically this was created by a comparative study runner (run_comparative_study.py), referenced in src/models/__init__.py and src/training/__init__.py. That runner is not present in this CPU branch. Instead, you can generate the artifact from the saved model checkpoints that run_training.py writes under results/<scale>/pytorch/.

Generate the analysis artifact from existing checkpoints

1) Train models (single or all):
```bash
# Train all models with the selected config
python run_training.py --all --config configs/small.yaml
```

2) Build enhanced_comparative_study.pt from saved models (new script):
```bash
python3 build_analysis_artifact.py --config configs/small.yaml
```

Advanced (optional): one-liner alternative if you prefer inline:
```bash
python3 - <<'PY'
import os
from pathlib import Path
import torch

scale = 'small'  # change to medium/production as needed
base_dir = Path(f'results/{scale}')
pytorch_dir = base_dir/'pytorch'
analysis_dir = base_dir/'analysis'
analysis_dir.mkdir(parents=True, exist_ok=True)

results = {}
training_times = {}
config = None

for f in sorted(pytorch_dir.glob('model_*.pt')):
    m = torch.load(f, map_location='cpu', weights_only=False)
    name = m.get('model_name') or f.stem.replace('model_','').replace('_',' ').replace('plus','+')
    # Minimal structure expected by the plotter
    results[name] = {'history': m['history']}
    training_times[name] = m.get('training_time', 0.0)
    if config is None:
        config = m.get('config', {})

out_path = analysis_dir/'enhanced_comparative_study.pt'
torch.save({'results': results, 'training_times': training_times, 'config': config}, out_path)
print('Wrote', out_path)
PY
```

3) Generate plots:
```bash
python3 make_comparative_plot.py --config configs/small.yaml
```

The plotter will also read per-epoch series (loss/cost) directly from results/<scale>/csv/history_*.csv to enrich the figures.

### Analyze Validation Strategies
Compare different validation approaches:
```bash
python3 analyze_validation_strategies.py
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
