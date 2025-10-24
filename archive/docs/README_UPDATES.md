# README.md Updates Summary

## Changes Made

### 1. Updated GPU Training Section (Section 1.5)

#### Before:
- Only mentioned `training_gpu/` directory
- No information about screen sessions
- Direct command examples only

#### After:
- **Two Training Pipelines** clearly documented:
  - Standard Pipeline (`training_gpu/`)
  - Prefetch Pipeline (`training_gpu_prefetch/`)

- **Interactive Training Examples** for both pipelines

- **NEW: Screen Session Training** ⭐ RECOMMENDED
  - Complete screen command examples
  - Monitoring instructions
  - Key benefits explained

### 2. Correct Screen Command Format

The working screen command format is now documented:

```bash
# Prefetch pipeline
screen -dmS training bash -c "source venv/bin/activate && python training_gpu_prefetch/scripts/run_training_gpu.py --config configs/normal_gpu_1000.yaml --model GT+RL; exec bash"
```

**Key differences from previous command:**
- No `cd` to subdirectory
- Paths are relative to project root
- Uses correct `training_gpu_prefetch/` path

### 3. Updated Sequential Training Scripts

Now includes both standard and prefetch pipeline scripts:

#### Standard Pipeline:
- `./run_seq.sh`
- `./run_seq_nohup.sh`

#### Prefetch Pipeline:
- `./run_seq_prefetch.sh`
- `./run_seq_nohup_prefetch.sh`

### 4. Enhanced Examples

Added prefetch-specific examples:
```bash
# Prefetch pipeline - Run large experiments with prefetching
./run_seq_nohup_prefetch.sh configs/normal_gpu_1000.yaml configs/large.yaml
```

## Why These Changes Matter

1. **SSH Persistence**: Screen sessions survive SSH disconnections, critical for long training runs
2. **Clear Documentation**: Users know exactly which pipeline to use
3. **Correct Commands**: No more trial and error with paths and commands
4. **Better Organization**: Standard vs Prefetch pipelines clearly separated

## Files Modified

- `README.md` - Main documentation updated
- `README.md.backup` - Original backed up before changes

## Reference Documents

The following existing documents provide additional context:
- `PREFETCH_TRAINING_GUIDE.md` - Detailed prefetch pipeline guide
- `training_gpu_prefetch/README.md` - Prefetch-specific documentation
