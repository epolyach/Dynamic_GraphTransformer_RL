# Configuration Cleanup Summary

## Changes Made

### 1. Removed from default.yaml:
- gat_training section (not using GAT+RL)
- model_gat section (not using GAT)
- Reduced from 169 to 152 lines

### 2. Updated src/utils/config.py:
- Removed num_instances from required_training validation
- num_instances is now auto-calculated as:
  num_instances = num_steps * (num_epochs + 1) * batch_size

### 3. Renamed parameter:
- num_batches_per_epoch -> num_steps (clearer, shorter)
- Updated in all .py and .yaml files

### 4. Created configs/example_n20.yaml:
- Minimal config showing only overrides needed
- Clean, simple structure

## Backups Created:
- configs/default.yaml.backup
- src/utils/config.py.backup

## Files Modified:
- src/utils/config.py
- training_gpu_prefetch/lib/advanced_trainer_gpu.py
- configs/default.yaml
- configs/example_n20.yaml
- curriculum_learning/configs/*.yaml
