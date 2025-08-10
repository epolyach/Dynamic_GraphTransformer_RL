# Models Backup Directory

This directory contains experimental and unused model implementations that were moved during the `src/models/` cleanup.

## 📁 Contents

### Core Experimental Models:
- **`ablation_models.py`** - Ablation study model variants (greedy, static RL, dynamic RL)
- **`DynamicGraphTransformerModel.py`** - Full dynamic graph transformer implementation
- **`graph_transformer.py`** - Graph transformer encoder implementation
- **`dynamic_updater.py`** - Dynamic graph update mechanisms
- **`model_factory.py`** - Factory pattern for creating different model variants

### Component Models:
- **`GAT_Encoder.py`** - Graph Attention Transformer encoder
- **`GAT_Decoder.py`** - Graph Attention Transformer decoder  
- **`EdgeGATConv.py`** - Custom edge-aware GAT convolution layer
- **`PointerAttention.py`** - Pointer network attention mechanism
- **`TransformerAttention.py`** - Transformer-based attention components
- **`mask_capacity.py`** - Capacity constraint masking utilities

### Legacy Models:
- **`Model.py`** - Basic wrapper/compatibility model

## 🔄 Restoration

To restore any of these models to active use:

1. **Move the required files back to `src/models/`:**
   ```bash
   cp src/models_backup/<filename> src/models/
   ```

2. **Update imports in your code:**
   ```python
   from src.models.<module> import <ClassName>
   ```

3. **Ensure dependencies are available:**
   - Some models may require `torch_geometric` for graph operations
   - Check import statements within each file

## 🚨 Current Status

- **Main comparative study** (`run_comparative_study.py`) uses **inline model definitions** and doesn't depend on these files
- **Ablation study** (`experiments/run_ablation_study.py`) is **temporarily disabled** due to missing dependencies
- **Legacy GAT compatibility** (`src_batch/`) may reference some of these components

## 🔧 To Re-enable Ablation Study:

1. Move required models back:
   ```bash
   cp src/models_backup/ablation_models.py src/models/
   cp src/models_backup/graph_transformer.py src/models/
   cp src/models_backup/dynamic_updater.py src/models/
   cp src/models_backup/GAT_Decoder.py src/models/
   ```

2. Remove the disable guard in `experiments/run_ablation_study.py`

3. Test imports:
   ```bash
   python -c "from src.models.ablation_models import create_ablation_model"
   ```

## 📝 Notes

- These models represent various experimental approaches and architectural explorations
- Some may be incomplete or require additional dependencies
- The main project functionality is preserved without these models
- Keep this backup for potential future research directions

---

**Moved on**: August 10, 2024  
**Reason**: Clean slate approach to remove unused experimental code from active codebase
