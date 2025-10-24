# GPU Training Pipeline Guide

This guide explains how to run GPU-accelerated training with parallel data generation.

## Quick Start

### Test Parallel Data Generation (Optional but Recommended)

Before running full training, verify that parallel data generation is working:

```bash
./test_parallel_datagen.sh
```

While the test is running, open another terminal and monitor CPU usage:
```bash
htop
# or
watch -n 1 'ps aux | grep python | grep -v grep'
```

You should see 6 Python worker processes actively using CPU cores.

---

## Running Training

### Single Configuration with nohup (Recommended for SSH)

Run training that survives SSH disconnection:

```bash
./run_seq_nohup.sh configs/tiny_gpu_1000.yaml
```

This will:
- Start training in the background using nohup
- Create a timestamped log file: `nohup_sequential_TIMESTAMP.log`
- Save the process ID to a `.pid` file
- Continue running even if you close your SSH session

**Monitor Progress:**
```bash
tail -f nohup_sequential_*.log
```

**Check if Running:**
```bash
ps aux | grep run_training_gpu
# or
screen -ls
```

---

### Multiple Configurations Sequentially

Run multiple experiments one after another:

```bash
./run_seq_nohup.sh configs/tiny_gpu_150.yaml configs/tiny_gpu_500.yaml configs/tiny_gpu_1000.yaml
```

Each configuration will complete before the next one starts.

---

### Direct Training (No nohup)

If you want to run training interactively (will stop if SSH disconnects):

```bash
python training_gpu/scripts/run_training_gpu.py --config configs/tiny_gpu_1000.yaml --model GT+RL
```

**Available Models:**
- `GT+RL` - Graph Transformer with Reinforcement Learning (recommended)
- `GAT+RL` - Graph Attention Network with RL
- `DGT+RL` - Dynamic Graph Transformer with RL
- `GT-Greedy` - Greedy baseline

**Train All Models:**
```bash
python training_gpu/scripts/run_training_gpu.py --config configs/tiny_gpu_1000.yaml --all
```

---

## Configuration

### Parallel Data Generation Settings

Edit your config file (e.g., `configs/tiny_gpu_1000.yaml`):

```yaml
data_generation:
  num_workers: 6  # Number of CPU cores for parallel data generation
                  # 0 = sequential (single core)
                  # 4-8 = good for most systems
```

**Recommendations:**
- For systems with 8+ cores: use 6-8 workers
- For systems with 4-6 cores: use 4 workers
- Set to 0 if you experience issues

### GPU Settings

```yaml
gpu:
  device: cuda:0              # GPU device to use
  mixed_precision: true       # Use FP16 for faster training
  memory_fraction: 0.9        # Fraction of GPU memory to use
```

---

## Monitoring

### GPU Usage

```bash
nvtop
# or
watch -n 1 nvidia-smi
```

### CPU Usage (Data Generation)

```bash
htop
# Press F4 and type 'python' to filter
```

You should see:
- **1 main Python process** using GPU (shown in nvtop)
- **6 worker processes** generating data (shown in htop CPU usage)

### Training Progress

```bash
# If using nohup
tail -f nohup_sequential_*.log

# If using screen
screen -ls  # List sessions
screen -r <session_name>  # Attach to session
# Press Ctrl+A then D to detach
```

---

## Results

Training results are saved to:
```
training_gpu/results/<config_name>/
├── checkpoints/        # Model checkpoints
├── csv/               # Training history (CSV format)
│   └── history_gt_rl.csv
├── plots/             # Loss curves, route visualizations
└── final_model_gt_rl.pth  # Final trained model
```

---

## Troubleshooting

### Only 1 CPU Core Active

**Problem:** `nvtop` or `htop` shows only one Python process using CPU.

**Solution:** The parallel data generation fix has been applied. Make sure:
1. Your config has `data_generation.num_workers > 0`
2. Run the test: `./test_parallel_datagen.sh`
3. You should see the message: `[ParallelDataGeneratorPool] Initialized with N worker processes`

### Out of GPU Memory

**Solutions:**
1. Reduce batch size in config: `training.batch_size: 256`
2. Reduce GPU memory fraction: `gpu.memory_fraction: 0.7`
3. Disable mixed precision: `gpu.mixed_precision: false` (slower but uses less memory)

### Screen Session Won't Detach

Press: `Ctrl+A` then `D`

### Kill Running Training

```bash
# Find the process ID
ps aux | grep run_training_gpu

# Kill it
kill <PID>

# Or force kill
kill -9 <PID>
```

---

## Performance Tips

### Optimal Settings for Your System (RTX A6000, 47GB VRAM)

```yaml
training:
  batch_size: 512           # Large batch for your GPU
  num_batches_per_epoch: 1000

data_generation:
  num_workers: 6            # Parallel CPU generation

gpu:
  device: cuda:0
  mixed_precision: true     # 2x speed boost
  memory_fraction: 0.9
```

### Expected Speedup

With the parallel data generation fix:
- **Sequential (1 worker):** ~100% of 1 CPU core
- **Parallel (6 workers):** ~400-500% total CPU usage (across 6 cores)
- **Speedup:** 4-5x faster data generation

---

## Advanced Usage

### Override Config from Command Line

```bash
python training_gpu/scripts/run_training_gpu.py \
  --config configs/tiny_gpu_1000.yaml \
  --model GT+RL \
  --batch_size 1024 \
  --epochs 50 \
  --lr 1e-4 \
  --device cuda:0
```

### Force Retrain (Ignore Existing Checkpoints)

```bash
python training_gpu/scripts/run_training_gpu.py \
  --config configs/tiny_gpu_1000.yaml \
  --model GT+RL \
  --force-retrain
```

---

## Files Reference

- `run_seq_nohup.sh` - Run training with nohup (survives SSH disconnect)
- `run_seq.sh` - Run training sequentially (called by run_seq_nohup.sh)
- `test_parallel_datagen.sh` - Test parallel data generation
- `training_gpu/scripts/run_training_gpu.py` - Main training script
- `configs/` - Configuration files

