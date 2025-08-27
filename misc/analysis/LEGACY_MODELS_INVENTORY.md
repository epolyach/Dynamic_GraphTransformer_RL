# Legacy GAT_RL Models Inventory

## Overview
The legacy GAT_RL project contains **one main deep learning model** with multiple architectural components, plus several classical heuristic algorithms for comparison.

## 1. Main Deep Learning Model Architecture

### **Model** (`model/Model.py`)
The main model that combines encoder and decoder:
```python
class Model(nn.Module):
    - encoder: ResidualEdgeGATEncoder
    - decoder: GAT_Decoder
```

### Components:

#### A. **Encoder Components**

##### **ResidualEdgeGATEncoder** (`encoder/GAT_Encoder.py`)
- **Purpose**: Encode graph structure with node and edge features
- **Architecture**:
  - Input embedding layers for nodes and edges
  - Batch normalization
  - Multiple EdgeGATConv layers with residual connections
  - Processes node features (coordinates + demand) and edge features (distances)
- **Key Features**:
  - Residual connections around GAT layers
  - Edge-aware attention mechanism
  - 4 layers by default

##### **EdgeGATConv** (`encoder/EdgeGATConv.py`)
- **Purpose**: Custom GAT layer incorporating edge features
- **Architecture**:
  - Message passing with edge-aware attention
  - Multi-head attention mechanism
  - Edge features integrated in attention coefficient computation
- **Key Innovation**: Unlike standard GAT, includes edge attributes in attention calculation

#### B. **Decoder Components**

##### **GAT_Decoder** (`decoder/GAT_Decoder.py`)
- **Purpose**: Sequential decision making for route construction
- **Architecture**:
  - Uses PointerAttention for node selection
  - Manages dynamic capacity and demand
  - Implements masking for feasible actions
- **Key Features**:
  - Dynamic state management
  - Capacity constraint enforcement
  - Sequential route generation

##### **PointerAttention** (`decoder/PointerAttention.py`) ⭐
- **Purpose**: Attention-based pointer network for node selection
- **Architecture**:
  - 8-head TransformerAttention layer
  - Compatibility score computation
  - Tanh activation with scaling (×10)
  - Temperature-controlled softmax
- **This is the counterpart to current Pointer+RL**

##### **TransformerAttention** (`decoder/TransformerAttention.py`)
- **Purpose**: Multi-head attention mechanism for state processing
- **Architecture**:
  - Configurable number of heads (default 8)
  - Q, K, V projections
  - Xavier initialization
  - Dropout for regularization
- **Used by**: PointerAttention

## 2. Reinforcement Learning Components

### **RolloutBaseline** (`RL/Rollout_Baseline.py`)
- **Purpose**: Baseline for policy gradient training
- **Features**:
  - Deep copy of model for stable baseline
  - Statistical significance testing (t-test)
  - Periodic baseline updates
  - Greedy rollout evaluation

### **euclidean_cost** (`RL/euclidean_cost.py`)
- **Purpose**: Reward/cost computation for routes
- **Features**:
  - Euclidean distance calculation
  - Tour cost evaluation
  - Batch processing support

## 3. Classical Heuristic Algorithms

### **Clarke-Wright Savings** (`RL/Clark_Wright.py`)
- **Type**: Classical constructive heuristic
- **Implementation**: Uses OR-Tools
- **Strategy**: SAVINGS first solution strategy
- **Status**: Partially implemented/commented code

### **Nearest Neighbor** (`RL/Nearest_Neighbor.py`)
- **Type**: Greedy constructive heuristic
- **Implementation**: Pure Python/NumPy
- **Strategy**: Always select nearest unvisited customer
- **Features**: Complete implementation with testing

### **Clarke-Wright Adapted** (`RL/Clark_Wright_adapted.py`)
- Appears to be a modified version of Clarke-Wright

### **Algorithm Comparison** (`RL/Algorithm_Comparison.py`)
- **Purpose**: Compare performance of classical algorithms
- **Contains**: Hard-coded results from Clarke-Wright and Nearest Neighbor
- **Metrics**: Distance, vehicles used, success rate

## 4. Training Infrastructure

### **main_train.py** (`train/main_train.py`)
- **Purpose**: Main training script
- **Configuration**:
  - 768,000 training instances
  - Batch size: 512
  - 100 epochs
  - Learning rate: 1e-4
  - Temperature: 2.5
- **Process**: Full batch iteration each epoch

### **train_model.py** (`train/train_model.py`)
- **Purpose**: Training loop implementation
- **Features**:
  - REINFORCE with baseline
  - Gradient clipping (max_norm=2.0)
  - TensorBoard logging
  - Model checkpointing

## 5. Data Generation and Utilities

### **InstanceGenerator** (`instance_creator/InstanceGenerator.py`)
- **Purpose**: Generate CVRP instances
- **Features**: Random instance generation with configurable parameters

### **CreateInstance_main** (`instance_creator/CreateInstance_main.py`)
- Main script for instance creation

### **instance_loader** (`instance_creator/instance_loader.py`)
- DataLoader utilities for batch processing

## 6. Main Inference Script

### **main.py**
- **Purpose**: Evaluation and testing
- **Features**:
  - Load trained models
  - Run inference on test instances
  - Compare with baseline
  - Save results to CSV

## Summary Statistics

| Category | Count | Details |
|----------|-------|---------|
| **Deep Learning Models** | 1 | GAT+PointerAttention |
| **Encoder Types** | 1 | ResidualEdgeGATEncoder with EdgeGATConv |
| **Decoder Types** | 1 | GAT_Decoder with PointerAttention |
| **Attention Mechanisms** | 2 | EdgeGATConv, TransformerAttention |
| **Classical Heuristics** | 2 | Clarke-Wright, Nearest Neighbor |
| **RL Components** | 1 | RolloutBaseline |

## Key Architectural Differences from Current Implementation

1. **Edge Features**: Legacy uses edge-aware GAT, current doesn't
2. **Multi-head Attention**: Legacy has 8 heads, current has none
3. **State Complexity**: Legacy tracks [GAT, first_node, current_node], current is simpler
4. **Parameter Count**: Legacy ~100K+, current ~21K
5. **Training Data**: Legacy processes all instances per epoch, current distributes across epochs

## Conclusion

The legacy GAT_RL project contains **ONE main deep learning model** that combines:
- **GAT Encoder** with edge-aware attention
- **Pointer Network Decoder** with 8-head transformer attention

The **PointerAttention** component is the direct counterpart to the current Pointer+RL implementation, but with significantly more complexity and architectural sophistication.
