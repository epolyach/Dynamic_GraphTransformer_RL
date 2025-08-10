# Training Backup Directory

This directory contains legacy training modules that were moved during the `src/training/` cleanup.

## 📁 Contents

### Legacy Training Modules:
- **`train_model.py`** - Legacy REINFORCE training implementation with:
  - Rollout baseline mechanism
  - Tensorboard logging
  - CPU-only operation
  - Uses `RolloutBaseline` from `src/utils/RL/`
  
- **`main_train.py`** - Standalone training pipeline that:
  - Imports from `src_batch` (legacy compatibility layer)
  - Uses the legacy GAT model (`src_batch.model.Model`)
  - Large-scale training setup (768,000 instances)
  - CPU profiling capabilities

- **`utils.py`** - Training utility functions:
  - Cost scaling and normalization functions
  - Simple helper utilities

## 🚨 Current Status

- **❌ NOT USED**: These modules are not used by the current project
- **✅ GAT+RL Legacy**: The GAT+RL legacy comparison uses `src_batch.train.train_model` (which forwards to `../GAT_RL/`), not these modules
- **✅ Main Training**: The main comparative study uses inline training in `run_comparative_study.py`

## 🔄 Dependencies

If restored, these modules would require:
- `src/utils/RL/euclidean_cost.py`
- `src/utils/RL/Rollout_Baseline.py`
- Standard PyTorch training dependencies

## 📝 Restoration

To restore any of these modules (if needed for experimental work):

1. **Move files back to `src/training/`:**
   ```bash
   cp src/training_backup/<filename> src/training/
   ```

2. **Update imports if needed:**
   ```python
   from src.training.<module> import <function>
   ```

3. **Ensure dependencies are available:**
   - Check that `src/utils/RL/` utilities are accessible
   - Verify import paths match your usage

## 📝 Notes

- These represent an alternative training approach that was not adopted
- The main project uses inline training for better maintainability
- Keep this backup for potential future experimental directions
- The legacy GAT+RL comparison works independently through `src_batch/`

---

**Moved on**: August 10, 2024  
**Reason**: Clean slate approach - unused legacy training modules moved to backup
