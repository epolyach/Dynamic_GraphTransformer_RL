# Training Pipeline Section - Integration Guide

## Files Created

1. **`training_pipeline_section.tex`** - The complete LaTeX section describing your GT+RL training pipeline
2. **`training_references.bib`** - Bibliography entries for the three key references cited
3. **`INTEGRATION_GUIDE.md`** - This file

## How to Integrate into Your Paper

### Step 1: Add to main.tex

Open `main.tex` and insert the training pipeline section where appropriate (likely after the "Model settings" section):

```latex
\input{training_pipeline_section}
```

Or simply copy-paste the content from `training_pipeline_section.tex` directly.

### Step 2: Add References to Your Bibliography

If you have an existing `.bib` file (e.g., `example.bib`), append the contents of `training_references.bib` to it:

```bash
cat training_references.bib >> example.bib
```

Or manually copy the three reference entries into your bibliography file.

### Step 3: Verify Citations

The section uses these three citations:
- `\cite{williams1992simple}` - REINFORCE algorithm
- `\cite{kool2018attention}` - Greedy rollout baseline (Kool et al., 2019)
- `\cite{loshchilov2017decoupled}` - AdamW optimizer

Make sure these compile correctly with your bibliography style.

## Section Structure

The generated section contains:

### Main Section: "Reinforcement Learning with Graph Transformers"
- **Preamble**: Introduces the GT+RL approach and its key innovations

### Subsection 1: "Training Pipeline and Batch Generation"
Contains 3 subsubsections:
1. **Instance Generation** - On-the-fly generation with parallel CPU workers
2. **REINFORCE Policy Gradient** - The core training algorithm with equations
3. **GPU Acceleration and Mixed Precision** - Training efficiency optimizations

### Subsection 2: "Graph Transformer Architecture"
Contains 4 subsubsections:
1. **Encoder: Distance-Aware Attention** - The key difference from GAT
2. **Spatial Positional Encoding** - Geometric information encoding
3. **Model Complexity** - Architecture parameters (256 dim, 4 layers, etc.)
4. **Decoder: Multi-Head Pointer Attention** - Sequential solution construction

### Subsection 3: "Greedy Rollout Baseline and Training Dynamics"
Contains 5 subsubsections:
1. **Rollout Baseline** - Following Kool et al. (2019)
2. **Baseline Update Strategy** - Statistical significance testing
3. **Temperature and Exploration** - Cosine annealing schedule
4. **Entropy Regularization** - Exploration bonus mechanism
5. **Optimization Details** - AdamW, learning rate, training duration

## Key Features of This Section

✅ **Concise but comprehensive** - Covers all essential aspects without excessive detail
✅ **Publication-ready** - Professional mathematical notation and structure
✅ **Well-referenced** - Three key citations to foundational work
✅ **Formula balance** - 7 equations covering the most important concepts
✅ **Implementation-grounded** - All parameters match your actual codebase

## Equations Included

1. **Policy gradient objective** (REINFORCE)
2. **Distance-aware attention** (key innovation)
3. **Spatial positional encoding**
4. **Decoder pointer mechanism**
5. **Advantage normalization**
6. **Temperature scheduling**
7. **Total loss with entropy**

## Parameters Documented

All key training parameters from your `configs/default.yaml`:
- Batch sizes: 256-512
- Hidden dimension: 256
- Attention heads: 4-8
- Transformer layers: 4
- Temperature: 2.5 → 0.15
- Entropy coefficient: 0.03
- Learning rate: 1e-4 → 1e-6
- Training epochs: 100
- Gradient clipping: 2.0

## Customization Notes

If you need to modify the section:

1. **Shorten it**: Remove some subsubsections or merge them
2. **Expand it**: Add more architectural details from `src/models/gt.py`
3. **Add figures**: Consider adding a diagram of the architecture
4. **Adjust parameters**: Update any values that differ in your final experiments

## Testing Compilation

To test that it compiles correctly:

```bash
cd !_paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX compilation method.

## Next Steps (Your Results Section)

After this training pipeline section, you'll want to add a "Results" section covering:
- Benchmark performance on different problem sizes (N=10, 20, 50, etc.)
- Comparison with OR-Tools and other baselines
- Training curves (loss, cost per customer over epochs)
- Computational efficiency (GPU utilization, training time)
- Solution quality analysis

The structure is ready for you to populate with your experimental data!

---

**Generated:** 2025-09-30
**Based on:** `training_gpu/` pipeline analysis
**Source files analyzed:** 
- `training_gpu/lib/advanced_trainer_gpu.py`
- `src/models/gt.py`
- `configs/default.yaml`
- `training_gpu/lib/rollout_baseline_gpu_fixed.py`
