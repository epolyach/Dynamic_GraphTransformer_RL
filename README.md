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
python run_comparative_study.py

# GPU version (explicit GPU optimization)
python run_comparative_study_gpu.py --customers 15 --epochs 10 --instances 800 --batch 8
```

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
python run_comparative_study_gpu.py --device cuda
```

### Memory Issues
```bash
# Reduce batch size for large problems
python run_comparative_study_gpu.py --batch 4 --customers 10

# Use CPU version for very large instances
python run_comparative_study.py
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
