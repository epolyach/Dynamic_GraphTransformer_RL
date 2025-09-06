# GPU Training Usage Examples

## Configuration Hierarchy

The configuration system uses a hierarchical approach:
- `default.yaml` - Contains all base parameters (including GPU settings)
- Scale-specific configs (`tiny.yaml`, `small.yaml`, `medium.yaml`, `large.yaml`, `huge.yaml`) - Override specific parameters for different problem scales

## Basic Usage Examples

### 1. Train with Default Configuration

```bash
# Train DGT model with default configuration (includes GPU settings)
python training_gpu/scripts/run_training_gpu.py --model DGT+RL

# Train GAT model 
python training_gpu/scripts/run_training_gpu.py --model GAT+RL

# Train GT model
python training_gpu/scripts/run_training_gpu.py --model GT+RL

# Train Greedy baseline
python training_gpu/scripts/run_training_gpu.py --model GT-Greedy
```

### 2. Train with Scale-Specific Configurations

```bash
# Quick test with tiny configuration
python training_gpu/scripts/run_training_gpu.py --config configs/tiny.yaml --model DGT+RL

# Small problem instances
python training_gpu/scripts/run_training_gpu.py --config configs/small.yaml --model GAT+RL

# Medium scale
python training_gpu/scripts/run_training_gpu.py --config configs/medium.yaml --model GT+RL

# Large scale
python training_gpu/scripts/run_training_gpu.py --config configs/large.yaml --model DGT+RL

# Huge scale (requires significant GPU memory)
python training_gpu/scripts/run_training_gpu.py --config configs/huge.yaml --model DGT+RL --estimate_batch_size
```

### 3. Force Retrain Existing Models

```bash
# Force retrain even if model already exists
python training_gpu/scripts/run_training_gpu.py --model DGT+RL --force-retrain

# Retrain with different hyperparameters
python training_gpu/scripts/run_training_gpu.py --model GAT+RL --force-retrain --lr 0.0001
```

### 4. Train All Models Sequentially

```bash
# Train all models with default config
python training_gpu/scripts/run_training_gpu.py --all

# Train all with specific scale
python training_gpu/scripts/run_training_gpu.py --config configs/small.yaml --all

# Train all with force retrain
python training_gpu/scripts/run_training_gpu.py --all --force-retrain
```

### 5. Custom Output Directory

```bash
# Specify custom output directory
python training_gpu/scripts/run_training_gpu.py --model DGT+RL --output-dir results/my_experiment

# Output with timestamp
python training_gpu/scripts/run_training_gpu.py --model GAT+RL --output-dir results/exp_$(date +%Y%m%d_%H%M%S)
```

## GPU-Specific Options

### 6. Mixed Precision Training

```bash
# Enable mixed precision (enabled by default in default.yaml)
python training_gpu/scripts/run_training_gpu.py --model DGT+RL --mixed_precision

# Disable mixed precision if encountering numerical issues
python training_gpu/scripts/run_training_gpu.py --model GAT+RL --no_mixed_precision
```

### 7. Automatic Batch Size Estimation

```bash
# Automatically determine optimal batch size for your GPU
python training_gpu/scripts/run_training_gpu.py --model DGT+RL --estimate_batch_size

# Essential for large/huge configurations
python training_gpu/scripts/run_training_gpu.py --config configs/huge.yaml --model GT+RL --estimate_batch_size
```

### 8. Multi-GPU Selection

```bash
# Use specific GPU device
python training_gpu/scripts/run_training_gpu.py --model DGT+RL --device cuda:0

# Use second GPU
python training_gpu/scripts/run_training_gpu.py --model GAT+RL --device cuda:1

# Control memory fraction
python training_gpu/scripts/run_training_gpu.py --model GT+RL --memory_fraction 0.8
```

## Advanced Training Scenarios

### 9. Override Training Parameters

```bash
# Custom batch size and learning rate
python training_gpu/scripts/run_training_gpu.py --model DGT+RL --batch_size 1024 --lr 0.0001

# Custom epochs and problem size
python training_gpu/scripts/run_training_gpu.py --model GAT+RL --epochs 200 --problem_size 50

# Full custom configuration
python training_gpu/scripts/run_training_gpu.py \
    --config configs/medium.yaml \
    --model DGT+RL \
    --batch_size 768 \
    --epochs 150 \
    --lr 0.00005 \
    --device cuda:0 \
    --mixed_precision \
    --output-dir results/custom_exp
```

### 10. Checkpointing and Logging

```bash
# Specify checkpoint directory
python training_gpu/scripts/run_training_gpu.py --model DGT+RL --checkpoint_dir checkpoints/dgt_exp1

# Training with W&B logging
python training_gpu/scripts/run_training_gpu.py --model GAT+RL --wandb

# Benchmark mode
python training_gpu/scripts/run_training_gpu.py --model DGT+RL --benchmark
```

## Complete Example Workflows

### Experiment 1: Compare All Models on Default Scale

```bash
# Train all models with default configuration
python training_gpu/scripts/run_training_gpu.py \
    --all \
    --mixed_precision \
    --output-dir results/gpu_comparison

# Results will be in:
# - results/gpu_comparison/GAT_RL/
# - results/gpu_comparison/GT_RL/
# - results/gpu_comparison/DGT_RL/
# - results/gpu_comparison/GT-Greedy/
```

### Experiment 2: Scale Analysis

```bash
# Test different scales for the same model
for scale in tiny small medium large; do
    python training_gpu/scripts/run_training_gpu.py \
        --config configs/${scale}.yaml \
        --model DGT+RL \
        --estimate_batch_size \
        --output-dir results/scale_analysis/${scale}
done
```

### Experiment 3: Hyperparameter Search

```bash
# Test different learning rates with medium scale
for lr in 0.0001 0.0005 0.001; do
    python training_gpu/scripts/run_training_gpu.py \
        --config configs/medium.yaml \
        --model DGT+RL \
        --lr $lr \
        --output-dir results/lr_search/lr_$lr \
        --force-retrain
done
```

### Experiment 4: GPU Memory Optimization

```bash
# Find maximum batch size for large problems
python training_gpu/scripts/run_training_gpu.py \
    --config configs/large.yaml \
    --model DGT+RL \
    --estimate_batch_size \
    --mixed_precision \
    --memory_fraction 0.95 \
    --output-dir results/max_batch
```

## Configuration Files Overview

### Base Configuration (default.yaml)

Contains all parameters including GPU settings:

```yaml
# default.yaml structure
problem:
  num_customers: 20
  vehicle_capacity: 30
  ...

training:
  batch_size: 512
  learning_rate: 1e-4
  num_epochs: 100
  ...

gpu:
  enabled: true
  device: "cuda:0"
  mixed_precision: true
  memory_fraction: 0.95
  ...

experiment:
  device: "cuda"  # Override from "cpu" for GPU training
  ...
```

### Scale-Specific Configurations

Each scale config overrides specific parameters:

- `tiny.yaml` - N=10 customers, reduced training (quick tests)
- `small.yaml` - N=10 customers, full training
- `medium.yaml` - N=20 customers (default scale)
- `large.yaml` - N=50 customers
- `huge.yaml` - N=100 customers

## Model Name Formats

The script accepts multiple model name formats for compatibility:

- `GAT+RL` or `gat` - Graph Attention Network with RL
- `GT+RL` or `gt` - Graph Transformer with RL  
- `DGT+RL` or `dgt` - Dynamic Graph Transformer with RL
- `GT-Greedy` or `greedy` - Greedy baseline

## Tips for Optimal GPU Usage

1. **Use default.yaml** as the base - it includes all GPU optimizations
2. **Use scale-specific configs** to test different problem sizes
3. **Use `--estimate_batch_size`** for large/huge scales
4. **Enable `--force-retrain`** when testing hyperparameters
5. **Monitor GPU** with `nvidia-smi -l 1` in another terminal

## Quick Debug Commands

```bash
# Test with tiny config for quick debugging
python training_gpu/scripts/run_training_gpu.py --config configs/tiny.yaml --model DGT+RL

# Small batch size to debug OOM errors
python training_gpu/scripts/run_training_gpu.py --model DGT+RL --batch_size 32

# Disable GPU optimizations for debugging
python training_gpu/scripts/run_training_gpu.py --model GAT+RL --no_mixed_precision --memory_fraction 0.5
```
