# Prefetch Training Pipeline Guide

## Scripts Created

Three scripts have been created for the `training_gpu_prefetch/` pipeline:

1. **`run_seq_prefetch.sh`** - Core script that runs configs sequentially in screen sessions
2. **`run_seq_nohup_prefetch.sh`** - Wrapper that adds nohup for SSH disconnection survival

## Current Running Training

**Config:** `configs/tiny_gpu_150.yaml`
**Process ID:** 1713821
**Screen Session:** `prefetch_tiny_gpu_150_1713821`
**Nohup Log:** `nohup_sequential_prefetch_20250930_204858.log`
**PID File:** `sequential_training_prefetch_20250930_204858.pid`

## Monitoring Commands

### Check if process is running:
```bash
ps -p 1713821
# or
ps -p $(cat sequential_training_prefetch_20250930_204858.pid)
```

### View screen sessions:
```bash
screen -ls
```

### Attach to the screen session (to see live output):
```bash
screen -r prefetch_tiny_gpu_150_1713821
# Press Ctrl+A then D to detach without stopping
```

### Monitor the log file in real-time:
```bash
tail -f nohup_sequential_prefetch_20250930_204858.log
```

### Check training progress:
```bash
# Check the sequential log
tail -f sequential_training_prefetch_*.log

# Or check the actual training logs in results
tail -f training_gpu_prefetch/results/tiny_gpu_150/training.log
```

## Usage Examples

### Single config:
```bash
./run_seq_nohup_prefetch.sh configs/tiny_gpu_150.yaml
```

### Multiple configs (run sequentially):
```bash
./run_seq_nohup_prefetch.sh configs/tiny_gpu_150.yaml configs/tiny_gpu_500.yaml configs/tiny_gpu_750.yaml
```

### Without nohup (if you want to see output directly):
```bash
./run_seq_prefetch.sh configs/tiny_gpu_150.yaml
```

## Stopping Training

### Graceful stop (kills the wrapper script):
```bash
kill 1713821
# or
kill $(cat sequential_training_prefetch_20250930_204858.pid)
```

### Force stop the actual training (in screen):
```bash
# List screen sessions
screen -ls

# Kill specific screen session
screen -X -S prefetch_tiny_gpu_150_1713821 quit
```

## Results Location

Training results will be saved in:
```
training_gpu_prefetch/results/<experiment_name>/
```

For the current run:
```
training_gpu_prefetch/results/tiny_gpu_150/
```

## Bugs Fixed

The following critical bugs were fixed in the prefetch pipeline:

1. **Training code indentation** - Training logic was not inside the prefetch loop
2. **Epoch statistics indentation** - Statistics were being computed inside the batch loop instead of once per epoch, causing:
   - Incorrect metric values
   - Multiple scheduler updates per epoch
   - Missing logging of baseline, temperature, and learning rate

Both issues have been resolved, and the pipeline should now correctly:
- Process all batches in each epoch
- Compute epoch statistics once per epoch
- Log baseline, temperature, and learning rate for every epoch
- Update the learning rate scheduler once per epoch

## Key Differences from Standard Pipeline

- **Script path:** `training_gpu_prefetch/scripts/run_training_gpu.py` (vs `training_gpu/scripts/`)
- **Results path:** `training_gpu_prefetch/results/` (vs `training_gpu/results/`)
- **Screen session prefix:** `prefetch_*` (vs `seq_*`)
- **Log file prefix:** `*_prefetch_*`
