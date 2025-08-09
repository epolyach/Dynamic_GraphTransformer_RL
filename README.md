# Dynamic Graph Transformer for Reinforcement Learning on CVRP

A comprehensive comparative study implementing and comparing 6 different neural network architectures for solving the Capacitated Vehicle Routing Problem (CVRP) using reinforcement learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA-capable GPU (optional, but recommended)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Dynamic_GraphTransformer_RL

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Comparative Study
```bash
# CPU version (auto-detects CUDA if available)
python src/experiments/run_comparative_study_cpu.py

# GPU version (explicit GPU optimization)
python src/experiments/run_comparative_study_gpu.py --customers 15 --epochs 10 --instances 800 --batch 8
```

## ⚖️ CPU vs GPU vs AMP Comparative Runs

Below are example commands to reproduce CPU vs GPU comparisons with identical parameters, and a GPU+AMP run.

Small sanity runs (~10–20 seconds on CPU):
- CPU:
  - python experiments/run_comparative_study_gpu.py --device cpu --problem_sizes 20 --instances 10 --out_dir results_cpu_small
- GPU (same params):
  - python experiments/run_comparative_study_gpu.py --device cuda --problem_sizes 20 --instances 10 --out_dir results_gpu_small
- GPU + AMP (same params):
  - python experiments/run_comparative_study_gpu.py --device cuda --problem_sizes 20 --instances 10 --amp --out_dir results_gpu_small_amp

Long runs (~10–20 minutes on CPU; tune instances/problem_sizes to hit target on your machine):
- CPU long:
  - python experiments/run_comparative_study_gpu.py --device cpu --problem_sizes 50 --instances 400 --out_dir results_cpu_long
- GPU long (same params):
  - python experiments/run_comparative_study_gpu.py --device cuda --problem_sizes 50 --instances 400 --out_dir results_gpu_long
- GPU long + AMP (same params):
  - python experiments/run_comparative_study_gpu.py --device cuda --problem_sizes 50 --instances 400 --amp --out_dir results_gpu_long_amp

Each run prints the total experiment time and writes CSV results and plots into the specified out_dir.

## 📋 Project Structure

```
.
├── run_comparative_study.py         # Main CPU version (latest)
├── run_comparative_study_gpu.py     # GPU-optimized version
├── pytorch/                         # Trained models & results
│   ├── model_pointer+rl.pt         # Pointer Network (21K params)
│   ├── model_gt-greedy.pt          # Graph Transformer Greedy (92K params)
│   ├── model_gt+rl.pt              # Graph Transformer RL (92K params)
│   ├── model_dgt+rl.pt             # Dynamic Graph Transformer (92K params)
│   ├── model_gat+rl.pt             # Graph Attention Transformer (59K params)
│   └── comparative_study_complete.pt # Complete study results
├── utils/
│   ├── plots/                      # All visualization files
│   └── comparative_results.csv     # Results data
├── src/                            # Source code modules
├── configs/                        # Configuration files
├── experiments/                    # Experiment setups
└── venv/                          # Python virtual environment
```

## 🏗️ Architecture Comparison

This study implements and compares 6 different neural network architectures:

### 1. **Pointer Network + RL**
- **Parameters**: ~21K
- **Architecture**: Simple attention-based pointer mechanism
- **Performance**: Good baseline, fast training
- **Use case**: Quick prototyping, baseline comparisons

### 2. **Graph Transformer (Greedy)**  
- **Parameters**: ~92K
- **Architecture**: Multi-head self-attention on graph nodes
- **Selection**: Always greedy (argmax)
- **Performance**: Deterministic, good for benchmarking

### 3. **Graph Transformer + RL**
- **Parameters**: ~92K  
- **Architecture**: Same as GT-Greedy but with RL training
- **Selection**: Probabilistic sampling during training
- **Performance**: Better exploration than greedy

### 4. **Dynamic Graph Transformer + RL**
- **Parameters**: ~92K
- **Architecture**: GT with dynamic state updates and gating
- **Features**: Adaptive node embeddings based on current state
- **Performance**: Handles complex routing constraints better

### 5. **Graph Attention Transformer + RL**
- **Parameters**: ~59K
- **Architecture**: GAT-style attention with edge features
- **Features**: Explicit distance and demand modeling
- **Performance**: Good balance of complexity and performance

### 6. **Hybrid Architecture + RL**
- **Parameters**: Variable
- **Architecture**: Combines pointer and graph attention mechanisms
- **Features**: Best of both approaches
- **Performance**: Most flexible, highest potential

## 📊 Performance Results

### Typical Performance (15 customers, 100 coordinate range):
- **Naive Baseline**: ~1.04 cost/customer (depot→customer→depot for each)
- **Pointer+RL**: ~0.64 cost/customer (38% improvement)
- **GT-Greedy**: ~0.62 cost/customer (40% improvement)  
- **GT+RL**: ~0.60 cost/customer (42% improvement)
- **DGT+RL**: ~0.58 cost/customer (44% improvement)
- **GAT+RL**: ~0.56 cost/customer (46% improvement)

### Training Performance:
- **Pointer+RL**: Fastest training (~20s), lowest memory
- **GT-Greedy**: Fast inference, deterministic results
- **GT+RL**: Good balance of speed and performance
- **DGT+RL**: Best route quality, moderate training time
- **GAT+RL**: Most parameter-efficient for performance achieved

## 🛠️ Technical Implementation

### Core Features
- **Sequential Route Generation**: All models generate complete routes through iterative decision-making
- **REINFORCE Learning**: Proper policy gradient implementation with baseline
- **Constraint Handling**: Vehicle capacity and customer visit constraints
- **GPU Optimization**: Efficient batching and tensor operations
- **Route Validation**: Comprehensive validation of generated solutions

### Data Generation
- **Coordinates**: Random integers [0, max_distance], normalized by /100
- **Demands**: Random integers [1, max_demand], normalized by /10
- **Capacity**: Fixed vehicle capacity constraint
- **Depot**: Randomly positioned (not centered)

### Training Process
1. **Instance Generation**: Create random CVRP instances
2. **Route Generation**: Models generate complete routes sequentially
3. **Cost Calculation**: Compute actual route costs using generated paths
4. **REINFORCE Update**: Update policy using cost-based advantages
5. **Validation**: Test on held-out instances with greedy selection

## 🔧 Configuration Options

### Command Line Arguments (GPU version):
```bash
--customers 15          # Number of customers (default: 15)
--epochs 10             # Training epochs (default: 10) 
--instances 800         # Training instances (default: 800)
--batch 8               # Batch size (default: 8)
--max_distance 100      # Coordinate range (default: 100)
--max_demand 10         # Demand range (default: 10)
--capacity 3            # Vehicle capacity (default: 3)
--device auto           # Device: cuda/cpu/auto (default: auto)
```

### CPU Version Configuration:
The CPU version uses fixed configuration optimized for stability:
- 15 customers, 10 epochs, 800 instances
- Batch size 8, capacity 3
- Auto-device detection (CUDA if available)

## 🧪 Experimental Features

### Recent Improvements
- **Fixed REINFORCE Implementation**: Correct advantage calculation
- **Proper Route Generation**: Sequential decision-making matching CVRP requirements  
- **GPU Optimization**: Efficient tensor operations and memory management
- **Architecture Matching**: GPU version now matches CPU sequential generation
- **Comprehensive Validation**: Route correctness and constraint satisfaction

### Architecture Evolution
The project evolved from single-action classification models to proper sequential route generation models, fixing fundamental architectural issues that prevented effective learning.

## 🚨 Common Issues & Solutions

### Installation Issues
```bash
# If PyTorch installation fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If CUDA issues on GPU
export CUDA_VISIBLE_DEVICES=0
python src/experiments/run_comparative_study_gpu.py --device cuda
```

### Memory Issues
```bash
# Reduce batch size for large problems
python src/experiments/run_comparative_study_gpu.py --batch 4 --customers 10

# Use CPU version for very large instances
python src/experiments/run_comparative_study_cpu.py
```

### Performance Issues
- **Slow training**: Reduce instances or customers
- **Poor convergence**: Increase epochs or adjust learning rate
- **Route validation errors**: Check constraint parameters

## 📈 Development History

### Key Milestones
1. **Initial Implementation**: Basic pointer network with single-step decisions
2. **Architecture Expansion**: Added 5 additional model architectures
3. **GPU Optimization**: Created GPU-optimized version with batching
4. **Critical Fixes**: Fixed REINFORCE advantages and route generation
5. **Architecture Alignment**: Made GPU version match CPU sequential approach
6. **Performance Validation**: Achieved 38-46% improvements over naive baseline

### Lessons Learned
- **Sequential vs Single-step**: CVRP requires sequential decision-making, not classification
- **Route Validation**: Critical for ensuring legitimate solutions
- **REINFORCE Implementation**: Advantage calculation direction matters significantly
- **Architecture Matters**: Different approaches excel in different scenarios
- **GPU Optimization**: Batching and tensor operations provide major speedups

## 🎯 Future Work

### Potential Improvements
- **Attention Mechanisms**: More sophisticated attention patterns
- **Multi-Vehicle Support**: Extend to multiple vehicle scenarios  
- **Dynamic Constraints**: Time windows, pickup/delivery constraints
- **Meta-Learning**: Adaptation to new problem instances
- **Hybrid Methods**: Combine with classical optimization

### Research Directions
- **Larger Scale**: Test on 50+ customer instances
- **Real-World Data**: Use actual delivery scenarios
- **Transfer Learning**: Pre-training on related routing problems
- **Architecture Search**: Automated neural architecture search
- **Interpretability**: Understanding learned routing strategies

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{dynamic_graph_transformer_cvrp,
  title={Dynamic Graph Transformer for Reinforcement Learning on CVRP},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or issues, please open a GitHub issue or contact [your-email].

---

**Note**: This project represents a comprehensive study of neural approaches to vehicle routing problems, with careful attention to proper implementation of reinforcement learning and sequential decision-making for combinatorial optimization.

## Unified CPU Orchestrator (training + evaluation + plots)

The CPU comparative study is now driven by a single orchestrator that trains requested models (only when needed), reuses cached runs, evaluates baselines on a deterministic validation set, and produces a consolidated table and plot (similar to Ver1).

- Entry point: src/experiments/run_comparative_study_cpu.py
- CSV output: results/comparative_study_cpu.csv
- Plot output: utils/plots/comparative_study_results.png

Supported models
- dynamic_gt_rl: trainable RL with dynamic graph updates during decoding (expected best)
- static_rl: trainable RL with static encoder and masking
- pointer_rl: trainable legacy RL (kept for compatibility)
- greedy_baseline: non-learning heuristic; greedy next-customer selection, no training
- naive_baseline: roundtrip heuristic baseline (worst), no training
- gat_rl: trainable internal GAT+RL (optional; add when enabled)
- gat_rl_legacy: legacy ../GAT_RL implementation via subprocess (optional)

Key behavior
- Trainable models (dynamic_gt_rl, static_rl, pointer_rl, gat_rl) are trained via the unified trainer and cached per configuration.
- Baselines (greedy_baseline, naive_baseline) are always evaluated on the same deterministic validation set derived from --seed and --instances.
- gat_rl_legacy can be integrated via subprocess to ../GAT_RL and cached similarly.

Caching
- Each trained run is stored under: results_train/cpu_{model}_C{customers}_I{instances}_E{epochs}_B{batch}
- A summary.json is written into the run directory with metrics and hyperparameters (including lr and seed).
- Changing customers, instances, epochs, batch, lr, or seed produces a new cached run.

Force retraining
- Use --recalculate_rl_weights to force retraining of all requested trainable models in this run (ignores existing summary.json caches).

Deterministic validation set
- The unified trainer uses a fixed validation subset per run: val_count = max(32, min(128, instances/5)), seeds = [seed .. seed+val_count-1].
- The orchestrator evaluates greedy_baseline and naive_baseline on this exact set for fair comparison.

Examples
- Default comparison (dynamic_gt_rl, static_rl, greedy_baseline, naive_baseline):
  python src/experiments/run_comparative_study_cpu.py

- Specific models:
  python src/experiments/run_comparative_study_cpu.py --models dynamic_gt_rl static_rl greedy_baseline naive_baseline

- Train if missing, else reuse cache:
  python src/experiments/run_comparative_study_cpu.py --models dynamic_gt_rl static_rl --customers 20 --instances 800 --epochs 20 --batch 32 --lr 1e-4 --seed 12345

- Force retraining (ignore cache) for requested trainable models:
  python src/experiments/run_comparative_study_cpu.py --models dynamic_gt_rl static_rl --recalculate_rl_weights

- Include legacy pointer and GAT when enabled:
  python src/experiments/run_comparative_study_cpu.py --models pointer_rl gat_rl gat_rl_legacy

Outputs
- results/comparative_study_cpu.csv: aggregated table with columns [Model, Val/Cust, CPU Time (s), OutDir]
- utils/plots/comparative_study_results.png: bar chart of Val/Cust for selected models
- results_train/...: per-model training directories with train_history.csv, best_route.png/json, summary.json

Notes

- greedy_baseline (heuristic) vs learned greedy:
  - greedy_baseline is a zero-training heuristic that greedily selects the next customer using fixed scoring; it does not update parameters and serves as a quick reference.
  - learned greedy means you train a model (e.g., static_rl, dynamic_gt_rl, or GAT+RL) with RL so its attention/policy parameters are learned; then at evaluation you decode greedily (argmax). This typically performs much better than a heuristic with random weights.
  - Recommended: report dynamic_gt_rl and static_rl using greedy decoding at evaluation (learned greedy). Keep greedy_baseline as the non-learning baseline for context.
- gat_rl_legacy is integrated via subprocess to ../GAT_RL and can be cached to avoid retraining every run.

### Example: CPU orchestrator with cached training
Run the top models and baselines with specific hyperparameters. Missing RL weights will be trained once and cached; subsequent runs reuse cache unless forced.

```bash
python src/experiments/run_comparative_study_cpu.py \
  --models dynamic_gt_rl static_rl greedy_baseline naive_baseline \
  --customers 20 --instances 200 --epochs 15 --batch 8 --lr 1e-4 --seed 12345
```

Force retraining of RL weights (ignore cache):
```bash
python src/experiments/run_comparative_study_cpu.py \
  --models dynamic_gt_rl static_rl \
  --customers 20 --instances 200 --epochs 15 --batch 8 --lr 1e-4 --seed 12345 \
  --recalculate_rl_weights
```

Outputs:
- results/comparative_study_cpu.csv
- utils/plots/comparative_study_results.png
- results_train/cpu_{model}_C{C}_I{I}_E{E}_B{B}/summary.json, train_history.csv, best_route.png/json
