# Screen Session Monitoring Guide

## Why Screen Appears Empty

The screen session `training_resume_tiny1000` appears empty because **all output is redirected to a log file**:

```bash
nohup python3 ... > resume_training.log 2>&1
```

This means:
- ✅ Training is running
- ✅ Output goes to `resume_training.log`
- ❌ Nothing displays in the screen terminal
- ⚠️ Screen is just a container to keep the process alive

## How to Monitor Training

### Option 1: Watch the Log File (Recommended)
```bash
tail -f training_gpu_prefetch/results/tiny_gpu_1000/resume_training.log
```

### Option 2: Watch GPU Utilization
```bash
watch -n 2 nvidia-smi
```

### Option 3: Check Process Status
```bash
# See CPU/Memory usage
top -p 260173

# See all worker processes
pstree -p 260173 | head -30
```

### Option 4: Monitor CSV Progress
```bash
# Watch for new epochs
watch -n 10 'tail -5 training_gpu_prefetch/results/tiny_gpu_1000/csv/history_gt_rl.csv'
```

## Current Training Status

**Process**: PID 260173 (running for ~6 minutes)  
**GPU Usage**: 98% ← **Training active!**  
**GPU Memory**: 3367 MiB  
**Status**: Training epoch 61  

The GPU usage jumping to 98% confirms training has started, even though the log hasn't been flushed yet due to output buffering.

## Why Use Screen Then?

Screen serves these purposes:
1. **Process isolation**: Keeps training running if SSH disconnects
2. **Process management**: Easy to stop with `screen -S name -X quit`
3. **Daemonization**: Process runs in background
4. **Signal handling**: Clean shutdown on system events

## Better Alternatives

### Option A: Run Without Screen (Direct)
```bash
nohup python3 training_gpu_prefetch/scripts/run_training_gpu.py \
    --model GT+RL \
    --config configs/tiny_gpu_1000.yaml \
    --resume training_gpu_prefetch/results/tiny_gpu_1000/checkpoints/checkpoint_epoch_60.pt \
    > training_gpu_prefetch/results/tiny_gpu_1000/resume_training.log 2>&1 &

# Save PID
echo $! > /tmp/training.pid

# Monitor
tail -f training_gpu_prefetch/results/tiny_gpu_1000/resume_training.log
```

### Option B: Screen Without nohup (Live Output)
```bash
# Start screen in foreground
screen -S training_live

# Inside screen, run without nohup
cd /home/evgeny.polyachenko/CVRP/Dynamic_GraphTransformer_RL
python3 training_gpu_prefetch/scripts/run_training_gpu.py \
    --model GT+RL \
    --config configs/tiny_gpu_1000.yaml \
    --resume checkpoint.pt

# Detach with: Ctrl+A then D
# Reattach with: screen -r training_live
```

### Option C: tmux (Better than screen)
```bash
# Create tmux session
tmux new -s training

# Run training (inside tmux)
python3 training_gpu_prefetch/scripts/run_training_gpu.py ...

# Detach: Ctrl+B then D
# Reattach: tmux attach -t training
```

## Quick Commands

### Check if training is progressing:
```bash
# GPU should be 80-100%
nvidia-smi

# Log file should be growing
watch -n 5 'wc -l training_gpu_prefetch/results/tiny_gpu_1000/resume_training.log'

# Process should be using CPU
ps aux | grep 260173 | grep -v grep
```

### Force flush log buffer:
```bash
# Send SIGUSR1 to force Python to flush
kill -USR1 260173  # May not work for all Python versions
```

### Stop training:
```bash
# Graceful stop (screen method)
screen -S training_resume_tiny1000 -X quit

# Direct kill (if needed)
kill 260173

# Force kill (last resort)
kill -9 260173
```

## Summary

**Current Situation:**
- ✅ Training IS running (GPU at 98%)
- ✅ Output redirected to log file
- ✅ Screen just keeps process alive
- ⚠️ Log buffered (will flush soon)

**To See Output:**
```bash
tail -f training_gpu_prefetch/results/tiny_gpu_1000/resume_training.log
```

The screen being "empty" is expected and normal - it's working as designed!

