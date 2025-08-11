# CURRENT REMAINING HARDCODED PARAMETERS

After adding configuration parameters to the YAML files, here are the hardcoded parameters that **still remain in the code** and need to be updated to use config values:

## 🔴 **CRITICAL HARDCODED VALUES** (Need immediate config integration)

### 1. **Pointer Network Input Multipliers** - Lines 144, 335, 515, 698, 919
```python
# BaselinePointerNetwork - Line 144
nn.Linear(hidden_dim * 2, hidden_dim)  # ❌ HARDCODED: multiplier = 2

# GraphTransformerNetwork - Line 335  
nn.Linear(hidden_dim * 2, hidden_dim)  # ❌ HARDCODED: multiplier = 2

# GraphTransformerGreedy - Line 515
nn.Linear(hidden_dim * 2, hidden_dim)  # ❌ HARDCODED: multiplier = 2

# DynamicGraphTransformer - Lines 691, 698
nn.Linear(hidden_dim * 2, hidden_dim)  # ❌ HARDCODED: update multiplier = 2
nn.Linear(hidden_dim * 3, hidden_dim)  # ❌ HARDCODED: pointer multiplier = 3

# GraphAttentionTransformer - Line 919
nn.Linear(hidden_dim * 2, hidden_dim)  # ❌ HARDCODED: multiplier = 2
```
**Config Available:** ✅ Added to YAML as `pointer_network.input_multiplier`, `dynamic_graph_transformer.update_input_multiplier`, etc.
**Status:** ❌ Code still uses hardcoded values

### 2. **Max Steps Multiplier** - Lines 152, 343, 524, 706, 927
```python
max_steps = len(instances[0]['coords']) * 2  # ❌ HARDCODED: multiplier = 2
```
**Config Available:** ✅ Already in YAML as `inference.max_steps_multiplier: 2`
**Status:** ❌ Code still uses hardcoded value instead of config

### 3. **Attention Temperature Scaling** - Line 175
```python
attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # ❌ HARDCODED: 0.5
```
**Config Available:** ✅ Added to YAML as `inference.attention_temperature_scaling: 0.5`
**Status:** ❌ Code still uses hardcoded `** 0.5`

### 4. **Residual Gate Initialization** - Line 689
```python
self.res_gate = nn.Parameter(torch.tensor(-2.19722458))  # ❌ HARDCODED
```
**Config Available:** ✅ Already in YAML as `dynamic_graph_transformer.residual_gate_init: -2.19722458`
**Status:** ❌ Code still uses hardcoded value

### 5. **Dynamic State Features Count** - Line 686
```python
self.state_encoder = nn.Linear(4, hidden_dim)  # ❌ HARDCODED: 4 features
```
**Config Available:** ✅ Already in YAML as `dynamic_graph_transformer.state_features: 4`
**Status:** ❌ Code still uses hardcoded `4`

## 🟡 **MODERATE HARDCODED VALUES** (Important but less critical)

### 6. **Feedforward Dimension Multiplier** - Lines 324, 504, 679
```python
dim_feedforward=hidden_dim * feedforward_multiplier  # ✅ GOOD: Uses config parameter
```
**Status:** ✅ Already properly uses config parameter

### 7. **Edge Embedding Divisor** - Line 902
```python
self.edge_embedding = nn.Linear(1, hidden_dim // edge_embedding_divisor)  # ✅ GOOD: Uses config
```
**Status:** ✅ Already properly uses config parameter

### 8. **Node Features Dimension** - Lines 156, 347, 528, 710, 931
```python
node_features = torch.zeros(batch_size, max_nodes, 3)  # ❌ HARDCODED: 3 features
```
**Config Available:** ✅ Available as `model.input_dim: 3` 
**Status:** ❌ Code uses hardcoded `3` instead of `config['model']['input_dim']`

## 🟢 **NUMERICAL STABILITY CONSTANTS** (Configurable for precision tuning)

### 9. **Large Negative Masking Values** - Lines 249, 429, 615, 825, 1015
```python
scores = scores.masked_fill(mask, -1e9)  # ❌ HARDCODED: -1e9
```
**Config Available:** ✅ Already in YAML as `inference.masked_score_value: -1e9`
**Status:** ❌ Code still uses hardcoded `-1e9`

### 10. **Log Probability Epsilon** - Lines 252, 432, 618, 828, 1018
```python
log_probs = torch.log(probs + 1e-12)  # ❌ HARDCODED: 1e-12
```
**Config Available:** ✅ Already in YAML as `inference.log_prob_epsilon: 1e-12`
**Status:** ❌ Code still uses hardcoded `1e-12`

## 🔵 **ACCEPTABLE HARDCODED VALUES** (Can remain as constants)

### 11. **Algorithm Safety Multiplier** - Line 1108
```python
naive_cost += distances[0, customer_idx] * 2  # ✅ ACCEPTABLE: Algorithmic constant
```
**Status:** ✅ Acceptable - this is algorithmic logic (depot→customer→depot)

### 12. **Output Dimensions** - Lines 146, 337, 517, 700, 921
```python
nn.Linear(hidden_dim, 1)  # ✅ ACCEPTABLE: Output dimension is always 1 for scores
```
**Status:** ✅ Acceptable - output dimension for pointer scoring is logically always 1

## 📊 **SUMMARY**

### ✅ **Already Fixed (Using Config):**
- Feedforward multipliers in transformer layers
- Edge embedding divisor 
- All training hyperparameters
- All problem parameters

### ❌ **Still Need Config Integration (10 parameters):**
1. Pointer network input multipliers (5 models)
2. Max steps multiplier 
3. Attention temperature scaling exponent
4. Residual gate initialization value
5. Dynamic state features count
6. Node features dimension 
7. Masked score large negative value
8. Log probability epsilon

### 🎯 **HIGH PRIORITY TO FIX:**
- Max steps multiplier (most impactful)
- Pointer network multipliers (affects all models)
- Numerical stability constants (precision tuning)
- Attention temperature scaling (fine-tuning)

The YAML configuration files already contain all these parameters, but the **code still needs to be updated** to read from config instead of using hardcoded values.
