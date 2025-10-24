# Medium GPU Configurations for n=20 Customers

## Created Configs

### 1. **medium_gpu_optimal.yaml** - Balanced Performance
- **Model**: hidden_dim=192, layers=3 (compromise between 128 and 256)
- **Batch**: 1024 × 50 steps = 51,200 instances/epoch
- **Temperature**: Fixed at 2.0 (lower than 2.5 for n=20)
- **Learning rate**: 1.5e-4 (slightly higher for faster convergence)
- **Expected EET**: ~150-180 seconds

### 2. **medium_gpu_annealing.yaml** - Temperature Annealing
- **Same model/batch as optimal**
- **Temperature**: Cosine annealing 2.5 → 0.3
- **Key difference**: Adaptive temperature for better exploration→exploitation
- **Expected benefit**: Better final performance, may converge slower initially

### 3. **medium_gpu_large_model.yaml** - Full Model Capacity
- **Model**: hidden_dim=256, layers=4 (paper defaults)
- **Batch**: 1024 × 40 steps (reduced due to memory)
- **Learning rate**: 1e-4 (more conservative for stability)
- **Expected EET**: ~200-250 seconds
- **Trade-off**: Higher quality vs longer training time

## Key Parameter Rationale for n=20

### Model Size
- **n=10**: hidden_dim=128 works well
- **n=20**: hidden_dim=192-256 recommended
- Larger problems benefit from more capacity but with diminishing returns

### Batch Size & Steps
- Kept batch_size=1024 (optimal for GPU utilization)
- Reduced steps to 40-50 (vs 75 for n=10) to maintain reasonable epoch time
- Total instances: ~40-50k per epoch (vs 76k for n=10)

### Temperature Strategy
- **n=20 needs lower temperature** (2.0 vs 2.5) for more focused exploration
- Annealing helps balance exploration/exploitation over training

### Learning Rate
- Slightly higher (1.5e-4) for n=20 to compensate for:
  - Larger solution space
  - More complex patterns to learn
  - Fewer instances per epoch

### Baseline Settings (Critical for Speed!)
- **eval_batches: 1** - Only 1024 instances (prevents initialization hang)
- **frequency: 3** - Update every 3 epochs (balance stability/adaptation)
- **warmup_epochs: 0** - Start learning immediately

### Entropy Regularization
- Higher entropy_coef (0.02 vs 0.01) for n=20
- Encourages more exploration in larger solution space
- Helps avoid premature convergence

## Expected Performance

| Config | EET (sec) | Final Cost | Convergence |
|--------|-----------|------------|-------------|
| medium_gpu_optimal | ~150-180 | Good | Fast |
| medium_gpu_annealing | ~150-180 | Better | Slower early, better late |
| medium_gpu_large_model | ~200-250 | Best | Slowest |

## Recommendations

1. **Start with medium_gpu_optimal.yaml** - Best speed/performance balance
2. **Try medium_gpu_annealing.yaml** if optimal plateaus early
3. **Use medium_gpu_large_model.yaml** for final/production models

All configs use:
- **eval_batches: 1** to avoid initialization delays
- **Optimized batch sizes** for A6000 GPU
- **Proven hyperparameters** from n=10 experiments
