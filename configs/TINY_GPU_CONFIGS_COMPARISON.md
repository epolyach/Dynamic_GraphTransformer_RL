# Tiny GPU Configs Comparison

This document compares the different tiny_gpu_* configurations for easy reference.

## Key Differences

| Config                | Batches/Epoch | Batch Size | Total Instances/Epoch | Use Case |
|-----------------------|---------------|------------|-----------------------|----------|
| tiny_gpu_150.yaml     | 150          | 512        | 76,800               | Quick testing |
| tiny_gpu_500.yaml     | 500          | 512        | 256,000              | Standard training |
| tiny_gpu_512.yaml     | 512          | 512        | 262,144              | Power-of-2 batches |
| tiny_gpu_750.yaml     | 750          | 512        | 384,000              | Extended training |
| **tiny_gpu_1000.yaml**| **1000**     | **512**    | **512,000**          | **Intensive training** |

## Total Training Instances (100 epochs)

| Config                | Total Instances |
|-----------------------|-----------------|
| tiny_gpu_150.yaml     | 7,680,000      |
| tiny_gpu_500.yaml     | 25,600,000     |
| tiny_gpu_512.yaml     | 26,214,400     |
| tiny_gpu_750.yaml     | 38,400,000     |
| **tiny_gpu_1000.yaml**| **51,200,000** |

## Common Settings

All configs share these settings:

```yaml
problem:
  num_customers: 10
  vehicle_capacity: 30

training:
  batch_size: 512
  num_epochs: 100
  learning_rate: 1e-4
  early_stopping:
    enabled: false

model:
  hidden_dim: 128
  num_heads: 4
  num_layers: 3

data_generation:
  num_workers: 6  # Parallel data generation
```

## When to Use Each Config

### tiny_gpu_150 - Quick Testing
- **Duration**: ~3.7 min per epoch (150/750 × 18.7 min)
- **Use**: Debugging, rapid prototyping, sanity checks
- **Pros**: Very fast iteration
- **Cons**: May not converge well due to limited data

### tiny_gpu_500 - Standard Training
- **Duration**: ~12.5 min per epoch (500/750 × 18.7 min)
- **Use**: Standard experiments, baseline comparisons
- **Pros**: Good balance of speed and convergence
- **Cons**: May need more epochs for harder problems

### tiny_gpu_512 - Power-of-2 Batches
- **Duration**: ~12.8 min per epoch (512/750 × 18.7 min)
- **Use**: When you want power-of-2 batch count (nice for some hardware)
- **Pros**: Similar to 500, slightly more data
- **Cons**: No significant advantage over 500

### tiny_gpu_750 - Extended Training
- **Duration**: ~18.7 min per epoch (measured: 1120 sec)
- **Use**: More thorough training, better convergence
- **Pros**: More training data per epoch
- **Cons**: Longer per-epoch time

### tiny_gpu_1000 - Intensive Training ⭐ NEW
- **Duration**: ~25 min per epoch (1000/750 × 18.7 min)
- **Use**: Maximum training data, best convergence
- **Pros**: 
  - Most training instances per epoch (512K)
  - Best for achieving optimal performance
  - Recommended for final/production models
- **Cons**: 
  - Longest training time
  - May be overkill for simple problems

## Performance Estimates (BEFORE Parallel Optimization)

Based on actual measurements (tiny_gpu_750 = 1120 sec/epoch):

| Config                | Time/Epoch      | Total (100 epochs)     | Speedup Potential |
|-----------------------|-----------------|------------------------|-------------------|
| tiny_gpu_150.yaml     | ~3.7 min        | ~6.2 hours            | → ~1.6 hours (3.8x) |
| tiny_gpu_500.yaml     | ~12.5 min       | ~20.8 hours           | → ~5.5 hours (3.8x) |
| tiny_gpu_512.yaml     | ~12.8 min       | ~21.3 hours           | → ~5.6 hours (3.8x) |
| tiny_gpu_750.yaml     | **~18.7 min**   | **~31.2 hours**       | **→ ~8.2 hours (3.8x)** |
| **tiny_gpu_1000.yaml**| **~25 min**     | **~41.7 hours**       | **→ ~11 hours (3.8x)** |

## Performance Estimates (AFTER Parallel Optimization)

With 6 workers providing 3.82x speedup:

| Config                | Time/Epoch      | Total (100 epochs)     |
|-----------------------|-----------------|------------------------|
| tiny_gpu_150.yaml     | ~1 min          | ~1.6 hours            |
| tiny_gpu_500.yaml     | ~3.3 min        | ~5.5 hours            |
| tiny_gpu_512.yaml     | ~3.4 min        | ~5.6 hours            |
| tiny_gpu_750.yaml     | **~4.9 min**    | **~8.2 hours**        |
| **tiny_gpu_1000.yaml**| **~6.5 min**    | **~11 hours**         |

**Note**: These are conservative estimates. Actual speedup depends on what percentage of total time is spent on data generation vs GPU computation.

## Recommended Workflow

1. **Development/Debugging**: Use `tiny_gpu_150.yaml`
   - Fast iterations (~1 min/epoch with parallel)
   - Test code changes
   - Validate hyperparameters

2. **Experimentation**: Use `tiny_gpu_500.yaml` or `tiny_gpu_512.yaml`
   - Compare different approaches (~3-4 min/epoch with parallel)
   - Tune hyperparameters
   - Ablation studies

3. **Production Training**: Use `tiny_gpu_750.yaml` or `tiny_gpu_1000.yaml`
   - Final model training (~5-7 min/epoch with parallel)
   - Best performance
   - Publication-ready results

## Running Training

```bash
# Quick test (150 steps/epoch)
python training_gpu/scripts/run_training_gpu.py \
    --config configs/tiny_gpu_150.yaml \
    --model GT+RL

# Standard training (500 steps/epoch)
python training_gpu/scripts/run_training_gpu.py \
    --config configs/tiny_gpu_500.yaml \
    --model GT+RL

# Extended training (750 steps/epoch)
python training_gpu/scripts/run_training_gpu.py \
    --config configs/tiny_gpu_750.yaml \
    --model GT+RL

# Intensive training (1000 steps/epoch) - NEW
python training_gpu/scripts/run_training_gpu.py \
    --config configs/tiny_gpu_1000.yaml \
    --model GT+RL
```

## Expected Impact of Parallel Optimization

**Current bottleneck**: 1 CPU core at 100%, GPU at 15%

**After optimization** (with 6 workers):
- CPU: 6 cores at 60-80% utilization
- GPU: 40-60% utilization (shifted bottleneck)
- Overall speedup: 3-4x (depends on GPU vs CPU time ratio)

The actual speedup will vary depending on:
- How much time is spent on data generation vs GPU computation
- GPU model and performance
- System I/O speed
- Memory bandwidth

## Notes

- All configs use **parallel data generation** (6 workers) for 3.8x data gen speedup
- All configs have **early_stopping disabled** - training runs for full 100 epochs
- Baseline updates every 2 epochs for efficiency
- Consistent model architecture across all configs for fair comparison
- Times are based on actual measurements from tiny_gpu_750 (1120 sec/epoch)
