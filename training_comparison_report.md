# Training Script Comparison Report

## Overview
Comparison between `../CVRP-GAT-RL/paper_replication_train.py` and 
`training_cpu/scripts/run_training.py` implementations.

## 1. Training Function Architecture

| Aspect                | paper_replication_train.py         | run_training.py                    |
|-----------------------|-------------------------------------|-------------------------------------|
| Training Function     | `train()` from src_batch            | `advanced_train_model()` from lib  |
| Complexity            | Simple, direct loop                 | Advanced with modern features       |
| Optimizer             | Basic Adam                          | AdamW with weight decay             |
| Gradient Clipping     | Fixed max_grad_norm=2.0             | Configurable via config             |
| Baseline              | Simple RolloutBaseline              | Advanced with significance testing  |

## 2. Data Generation

| Aspect                | paper_replication_train.py         | run_training.py                    |
|-----------------------|-------------------------------------|-------------------------------------|
| Method                | Pre-generates all data upfront      | Generates data on-the-fly          |
| Total Instances       | 768,000 fixed instances             | 768,000 per epoch (1500×512)       |
| Memory Usage          | High (stores all data)              | Low (generates per batch)           |
| Data Loader           | `instance_loader` fixed dataset    | `create_data_generator` function   |
| Validation Set        | Fixed 10,000 instances              | Generated dynamically               |
| Data Reuse            | Same instances every epoch          | New instances each epoch            |

## 3. Model Parameters

| Parameter         | paper_replication | run_training (default) | run_training (GAT) |
|-------------------|-------------------|------------------------|---------------------|
| hidden_dim        |               128 |                    256 |                 256 |
| edge_dim          |                16 |                     16 |                  16 |
| layers            |                 4 |                      4 |                   4 |
| dropout           |               0.6 |                    0.1 |                 0.6 |
| learning_rate     |             1e-4  |                  1e-4  |                2e-4 |
| temperature       |          Fixed 2.5|      Adaptive 2.5→0.15 |             3.0→0.3 |
| num_epochs        |               101 |                    100 |                 100 |
| batch_size        |               512 |                    512 |                 512 |
| vehicle_capacity  |                30 |                     30 |                  30 |

## 4. Advanced Training Features

| Feature                   | paper_replication | run_training |
|---------------------------|-------------------|--------------|
| LR Scheduling             | ❌                | ✅           |
| Early Stopping            | ❌                | ✅           |
| Temperature Scheduling    | ❌                | ✅           |
| CSV Logging               | ❌                | ✅           |
| Entropy Regularization    | ❌                | ✅           |
| Baseline Significance Test| ❌                | ✅           |
| Multiple Model Support    | ❌                | ✅           |
| Config Files              | ❌                | ✅           |
| CLI Arguments             | ❌                | ✅           |

## 5. Key Functional Differences

### Data Handling
- **paper_replication**: Uses same 768,000 instances every epoch (data reuse)
- **run_training**: Generates new 768,000 instances each epoch (more diverse)

### Model Architecture
- **paper_replication**: Pure GAT architecture only
- **run_training**: Supports GAT+RL, GT+RL, DGT+RL, GT-Greedy

### Training Algorithm
- **paper_replication**: Basic REINFORCE with simple baseline
- **run_training**: REINFORCE with entropy regularization, advanced variance reduction

### Configuration
- **paper_replication**: Hardcoded parameters in script
- **run_training**: YAML configuration with override capability

## 6. Module Dependencies

| Component         | paper_replication                  | run_training                       |
|-------------------|------------------------------------|------------------------------------|
| Model Module      | src_batch.model.Model              | src.models.model_factory           |
| Train Module      | src_batch.train.train_model        | training_cpu.lib.advanced_trainer  |
| Data Module       | src_batch.instance_creator         | src.generator.generator            |
| Baseline Module   | src_batch.RL.Rollout_Baseline      | training_cpu.lib.rollout_baseline  |

## 7. To Make Implementations Identical

### Required Changes for run_training.py:

1. **Create GAT-specific config**:
```yaml
model:
  hidden_dim: 128        # Reduce from 256
training:
  num_epochs: 101        # Increase from 100
training_advanced:
  use_adaptive_temperature: false
  temp_start: 2.5        # Fixed temperature
  use_lr_scheduling: false
  use_early_stopping: false
```

2. **Modify data generation** to pre-generate and reuse 768,000 instances

3. **Disable advanced features** by setting `use_advanced_features: false`

4. **Switch optimizer** from AdamW to basic Adam

## Summary

The implementations differ fundamentally in philosophy:
- **paper_replication_train.py**: Simple, fixed, paper-specific implementation
- **run_training.py**: Flexible, configurable framework with modern training techniques

The run_training.py implementation generates 768,000 new instances per epoch vs reusing the 
same 768,000 instances, which is a critical difference affecting training diversity.
