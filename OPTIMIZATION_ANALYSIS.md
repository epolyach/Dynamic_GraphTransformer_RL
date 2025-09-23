# Training Performance Optimization Analysis

## Current Status
- **Observed**: ~57-61 seconds per epoch (including baseline evaluation)
- **Target**: 22-23 seconds per epoch
- **Problem**: Excessive time spent on baseline evaluations

## Key Findings

### 1. Baseline Evaluation Overhead
The RolloutBaseline evaluation is the main bottleneck:
- Evaluates on 640 instances (5 batches × 128) every epoch
- Runs evaluation twice: once before training, once after each epoch
- Each evaluation involves full model inference

### 2. Configuration Issues Fixed
- ✅ Updated capacity from 20 to 30 in configs
- ✅ Created configs for both n=10 and n=20

### 3. Implementation Status
- ✅ Using original_fast trainer as base (known to achieve 22-23s)
- ✅ Keeping distances on CPU for cost computation
- ✅ Using simple `compute_route_cost` from `src/metrics/costs.py`
- ✅ Removed vectorized GPU cost computation

## Recommendations to Achieve 22-second Target

### 1. Reduce Baseline Evaluation Frequency
- Evaluate baseline every 5-10 epochs instead of every epoch
- Reduce eval dataset from 5 batches to 2 batches

### 2. Optimize Data Movement
The `move_to_gpu_except_distances` function has been added to:
- Keep distance matrices on CPU (used only for cost computation)
- Move only necessary tensors to GPU (coords, demands, etc.)

### 3. Remove Unnecessary Synchronization
- Remove excessive `torch.cuda.synchronize()` calls
- Only synchronize when timing or at epoch boundaries

## Files Modified
1. `training_gpu/lib/advanced_trainer_gpu.py` - Using original_fast with CPU distances
2. `configs/tiny_1.yaml` - Updated capacity to 30
3. `configs/tiny_2_n20.yaml` - Created for n=20 testing

## Next Steps
To achieve the 22-second target:
1. Modify baseline evaluation frequency in the trainer
2. Run tests with reduced baseline overhead
3. Profile remaining bottlenecks if needed
