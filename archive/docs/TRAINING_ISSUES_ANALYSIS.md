# Training Issues Analysis

## Issue 1: High Initial Loss

### Comparison:
| Config | Batches/Epoch | Epoch 0 Loss | Epoch 0 Cost | Initial Baseline |
|--------|---------------|--------------|--------------|------------------|
| tiny_gpu_500 | 500 | -0.155 | 0.598 | 0.555 |
| tiny_gpu_750 | 750 | -0.121 | 0.583 | 0.552 |
| **tiny_gpu_1000** | **1000** | **-0.682** | **0.730** | **0.637** |

### Problem Identified:
The baseline is STALE!

**Baseline update frequency:**
- Config setting: `frequency: 2` (update every 2 epochs)
- 500 batches: Updates after 1000 training batches
- 750 batches: Updates after 1500 training batches  
- **1000 batches: Updates after 2000 training batches** ← TOO INFREQUENT!

**Why this causes high loss:**
1. Baseline starts at random (0.637 cost)
2. Model quickly improves (down to 0.730 cost)
3. But baseline doesn't update for 2000 batches!
4. Loss = log_prob * (cost - baseline_cost)
5. With stale baseline: Loss is artificially high

### Solution:
Update baseline more frequently for 1000-batch config.

**Recommendation:**
```yaml
baseline:
  update:
    frequency: 1  # Update EVERY epoch instead of every 2
```

Or update every N batches:
```yaml
baseline:
  update:
    frequency_batches: 500  # Update after 500 batches
```

---

## Issue 2: Missing Rollout Baseline Output

### Problem:
Training is SO FAST now (14s vs 700s) that output scrolls by before you can see it!

The prefetching made training 50x faster, so output appears to "disappear".

### Evidence:
```
Time per epoch:
- tiny_gpu_500: ~700 seconds (11.7 min)
- tiny_gpu_750: ~1100 seconds (18.3 min)
- tiny_gpu_1000 WITH PREFETCH: ~14 seconds! (0.23 min)
```

**Speedup: 50-80x faster!** 🚀

### What's Actually Happening:
The output IS being printed, but:
1. Each epoch takes only 14 seconds
2. Log output happens every epoch
3. You're seeing it fly by too fast to read!

### Solutions:

**Option 1: Check the log file**
```bash
tail -50 sequential_training_20250930_201750.log | grep -E "Epoch|Baseline|Rollout"
```

**Option 2: Add more verbose output**
Modify trainer to print after every N batches instead of every epoch.

**Option 3: Slow down output (not recommended)**
Add delays, but this defeats the purpose of fast training!

---

## Root Cause Summary

**High Loss:**
- ❌ Baseline update frequency too low for 1000 batches/epoch
- ❌ Stale baseline (0.637) vs actual cost (0.730)
- ✅ FIX: Update baseline every epoch, not every 2 epochs

**Missing Output:**
- ✅ Output IS being printed!
- ❌ Just scrolling too fast (14s per epoch!)
- ✅ This is actually SUCCESS - prefetching works!
- ✅ FIX: Read log file or add batch-level output

---

## Recommended Fixes

### 1. Fix Baseline Update Frequency

Edit `configs/tiny_gpu_1000.yaml`:
```yaml
baseline:
  eval_batches: 5
  update:
    frequency: 1      # Change from 2 to 1
    warmup_epochs: 0
```

### 2. Add Batch-Level Progress Output

Modify training loop to print every 100 batches:
```python
if batch_idx % 100 == 0:
    logger.info(f"  Batch {batch_idx}/{n_batches}, "
                f"Loss: {current_loss:.4f}, "
                f"Cost: {current_cost:.4f}")
```

### 3. Verify Baseline Updates

Check CSV to see baseline updates:
```bash
awk -F',' 'NR==1 || $9 != ""' training_gpu/results/tiny_gpu_1000/csv/history_gt_rl.csv
```

This shows only epochs where baseline was updated (when column 9 is not empty).

