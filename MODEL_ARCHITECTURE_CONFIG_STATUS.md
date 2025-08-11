# 📋 MODEL ARCHITECTURE CONSTANTS - YAML vs HARDCODED STATUS

## ✅ **ALREADY IN YAML FILES** (Well Configured)

### **Core Architecture Parameters**
```yaml
model:
  input_dim: 3              # ✅ Input dimensions (coords + demand)
  hidden_dim: 64/128/256    # ✅ Hidden dimensions (scales by config)
  num_heads: 4/8/16         # ✅ Attention heads (scales by config)
  num_layers: 2/4/6         # ✅ Transformer layers (scales by config)
```

### **Model-Specific Parameters**
```yaml
model:
  pointer_network:
    multiplier: 2           # ✅ Pointer network layer multiplier
    
  dynamic_graph_transformer:
    state_features: 4       # ✅ Dynamic state feature count
    residual_gate_init: -2.19722458  # ✅ Residual gate initialization
    
  legacy_gat:
    node_input_dim: 3       # ✅ Node input dimensions
    edge_input_dim: 1       # ✅ Edge input dimensions
    hidden_dim: 128         # ✅ GAT-specific hidden dim
    edge_dim: 16/32         # ✅ Edge embedding dimension
    num_layers: 4/6         # ✅ GAT layer count
    negative_slope: 0.2     # ✅ LeakyReLU slope
    dropout: 0.3-0.6        # ✅ GAT dropout rates
```

### **Inference Parameters**
```yaml
inference:
  max_steps_multiplier: 2   # ✅ Max routing steps multiplier
  log_prob_epsilon: 1e-12   # ✅ Numerical stability epsilon
  masked_score_value: -1e9  # ✅ Score masking value
  default_temperature: 1.0  # ✅ Default inference temperature
```

### **Ablation Study Configuration**
```yaml
model:
  encoder:
    dropout: 0.1            # ✅ Transformer dropout in ablation config
    activation: "relu"      # ✅ Activation functions
    
  positional_encoding:
    pe_type: "sinusoidal"   # ✅ Positional encoding type
    max_distance: 100.0     # ✅ PE maximum distance
    pe_dim: 64              # ✅ PE dimensions
```

## ❌ **STILL HARDCODED** (Not in YAML)

### **High Priority - Transformer Architecture**

1. **Transformer Dropout** (Lines 325, 505, 680)
```python
dropout=0.1  # TODO: Make configurable
```
- **Status:** ❌ Hardcoded in main script
- **YAML Status:** ✅ Present in `ablation_study.yaml` but NOT used by main script
- **Issue:** Main script doesn't read `model.encoder.dropout`

2. **Feedforward Multiplier** (Lines 324, 504, 679)
```python
dim_feedforward=hidden_dim * 2
```
- **Status:** ❌ Hardcoded multiplier of `2`
- **YAML Status:** ❌ Not present in any config file
- **Suggested:** `model.transformer.feedforward_multiplier: 2`

3. **Attention Scale Factor** (Line 175)
```python
/ (self.hidden_dim ** 0.5)
```
- **Status:** ❌ Hardcoded square root scaling
- **YAML Status:** ❌ Not present in any config file
- **Suggested:** `model.attention.scale_factor: 0.5`

### **Medium Priority - Edge Cases**

4. **Edge Embedding Divisor** (Line 902)
```python
hidden_dim // 4
```
- **Status:** ❌ Hardcoded divisor of `4`
- **YAML Status:** ❌ Not in main configs (legacy_gat.edge_dim is different)
- **Suggested:** `model.edge_embedding_divisor: 4`

5. **Pointer Network Input Multipliers** (Various lines)
```python
nn.Linear(hidden_dim * 2, hidden_dim)  # Most models
nn.Linear(hidden_dim * 3, hidden_dim)  # Dynamic GAT
```
- **Status:** ❌ Hardcoded multipliers
- **YAML Status:** ⚠️ Partially configured (`pointer_network.multiplier: 2`)
- **Issue:** Not actually used by the main script

## 🔧 **CONFIGURATION GAPS**

### **Gap 1: Transformer Dropout Not Connected**
- **YAML:** `ablation_study.yaml` has `model.encoder.dropout: 0.1`  
- **Code:** Hardcoded `dropout=0.1` in transformer layers
- **Fix:** Update script to read `config['model']['transformer_dropout']`

### **Gap 2: Pointer Network Multiplier Not Used**
- **YAML:** All configs have `model.pointer_network.multiplier: 2`
- **Code:** Hardcoded `hidden_dim * 2` and `hidden_dim * 3`
- **Fix:** Update script to use `config['model']['pointer_network']['multiplier']`

### **Gap 3: Missing Feedforward Configuration**
- **YAML:** No feedforward multiplier in any config
- **Code:** Hardcoded `hidden_dim * 2`  
- **Fix:** Add `model.transformer.feedforward_multiplier: 2` to configs

### **Gap 4: Edge Embedding Not Configurable**
- **YAML:** Legacy GAT has `edge_dim` but different concept
- **Code:** Hardcoded `hidden_dim // 4`
- **Fix:** Add `model.edge_embedding_divisor: 4` to configs

## 📊 **CURRENT STATUS SUMMARY**

| Parameter Category | In YAML | Used by Script | Status |
|-------------------|---------|----------------|--------|
| **Core Architecture** | ✅ | ✅ | ✅ **Working** |
| **Legacy GAT** | ✅ | ✅ | ✅ **Working** |  
| **Numerical Stability** | ✅ | ✅ | ✅ **Working** |
| **Dynamic GAT** | ✅ | ✅ | ✅ **Working** |
| **Transformer Dropout** | ⚠️ | ❌ | 🔧 **Needs Connection** |
| **Pointer Multipliers** | ⚠️ | ❌ | 🔧 **Needs Connection** |
| **Feedforward Multiplier** | ❌ | ❌ | 🔧 **Needs Implementation** |
| **Edge Embedding** | ❌ | ❌ | 🔧 **Needs Implementation** |
| **Attention Scaling** | ❌ | ❌ | 🔧 **Needs Implementation** |

## 🎯 **RECOMMENDATIONS**

### **Quick Wins** (Connect existing YAML to code)
1. **Fix transformer dropout connection** - Read from config instead of hardcode
2. **Fix pointer network multiplier** - Use existing YAML values

### **Easy Additions** (Add missing YAML keys) 
3. **Add feedforward_multiplier** to model config
4. **Add edge_embedding_divisor** to model config  
5. **Add attention_scale_factor** to model config

### **Final Result**
After these fixes: **~98% configurable** with only deep architectural constants remaining hardcoded.

## 💡 **PROPOSED YAML STRUCTURE ADDITIONS**

```yaml
model:
  transformer:
    dropout: 0.1                    # Fix: Connect to existing hardcoded value
    feedforward_multiplier: 2       # Add: Currently hardcoded * 2
    
  attention:
    scale_factor: 0.5              # Add: Currently hardcoded ** 0.5
    
  edge_embedding:
    divisor: 4                     # Add: Currently hardcoded // 4
    
  pointer_network:
    multiplier: 2                  # Fix: Connect existing YAML to code
    dynamic_multiplier: 3          # Add: For dynamic GAT's * 3 case
```
