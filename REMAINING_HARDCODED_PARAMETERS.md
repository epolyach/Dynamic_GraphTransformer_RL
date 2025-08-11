# Remaining Hardcoded Parameters in run_train_validation.py

After the recent updates to make feedforward multiplier and edge embedding divisor configurable, here are ALL the remaining hardcoded parameters that could potentially be made configurable:

## 🔥 CRITICAL HARDCODED VALUES (Most Important)

### 1. **Max Steps Multiplier** (Line 152, 343, 524, 706, 927)
```python
max_steps = len(instances[0]['coords']) * 2  # Hardcoded multiplier: 2
```
**Impact:** Controls maximum route length - affects convergence and computation time
**Suggested config:** `max_steps_multiplier: 2`

### 2. **Dynamic State Features Count** (Line 686)
```python
self.state_encoder = nn.Linear(4, hidden_dim)  # Hardcoded: 4 features
```
**Impact:** Number of dynamic state features (capacity_used, step_progress, visited_count, distance_from_depot)
**Suggested config:** `state_features_count: 4` (but changing this requires code changes)

### 3. **Attention Temperature Scaling** (Line 175)
```python
attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
```
**Impact:** Attention scaling factor - affects attention sharpness
**Suggested config:** `attention_temperature_scaling: 0.5`

### 4. **Gated Residual Initial Value** (Line 689)
```python
self.res_gate = nn.Parameter(torch.tensor(-2.19722458))  # sigmoid ≈ 0.1
```
**Impact:** Initial gate value for dynamic updates in DGT model
**Suggested config:** `residual_gate_init: -2.19722458`

## 🟡 MODERATE HARDCODED VALUES

### 5. **Pointer Network Architecture Multipliers**
```python
# BaselinePointerNetwork (Lines 144-146)
nn.Linear(hidden_dim * 2, hidden_dim)  # Input multiplier: 2
nn.Linear(hidden_dim, 1)               # Output: 1

# GraphTransformer models (Lines 335-337, 515-517)
nn.Linear(hidden_dim * 2, hidden_dim)  # Input multiplier: 2
nn.Linear(hidden_dim, 1)               # Output: 1

# DynamicGraphTransformer (Lines 698-700)
nn.Linear(hidden_dim * 3, hidden_dim)  # Input multiplier: 3
nn.Linear(hidden_dim, 1)               # Output: 1

# GAT model (Lines 919-921)
nn.Linear(hidden_dim * 2, hidden_dim)  # Input multiplier: 2
nn.Linear(hidden_dim, 1)               # Output: 1
```
**Impact:** Affects pointer network capacity and expressiveness
**Suggested config:** `pointer_input_multiplier`, `pointer_hidden_layers`

### 6. **Dynamic Update Architecture** (Lines 691-693)
```python
nn.Linear(hidden_dim * 2, hidden_dim)  # Input multiplier: 2
nn.ReLU()                              # Hardcoded activation
nn.Linear(hidden_dim, hidden_dim)      # Hidden-to-hidden
```
**Impact:** Architecture of dynamic update mechanism in DGT
**Suggested config:** `dynamic_update_multiplier: 2`, `dynamic_update_activation: "relu"`

### 7. **Node Feature Dimensions** (Lines 156, 347, 528, 710, 931)
```python
node_features = torch.zeros(batch_size, max_nodes, 3)  # Hardcoded: 3 features
```
**Impact:** Number of node features (x, y, demand)
**Note:** This is structural but could be made configurable for different input formats

## 🟢 NUMERICAL STABILITY & PRECISION CONSTANTS

### 8. **Masking Values**
```python
scores.masked_fill(mask, -1e9)     # Lines 249, 429, 615, 825, 1015
log_probs = torch.log(probs + 1e-12)  # Lines 252, 432, 618, 828, 1018
```
**Impact:** Numerical stability for masked softmax and log computations
**Suggested config:** `mask_value: -1e9`, `log_epsilon: 1e-12`

### 9. **Training Split Ratio** (Line 1291)
```python
train_val_split = config.get('training', {}).get('train_val_split', 0.8)
```
**Status:** ✅ Already configurable (with default 0.8)

### 10. **Default Forward Parameters** (Lines 149, 340, 520, 703, 924)
```python
def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
```
**Impact:** Default temperature for sampling
**Status:** Partially configurable (passed from training loop), default could be configurable

## 🔵 LEGACY MODEL HARDCODED VALUES

### 11. **Legacy Training Parameters** (Lines 1485-1487)
```python
n_steps = config['num_customers'] * legacy_training.get('max_steps_multiplier', 2)
T = legacy_training.get('temperature', 2.5)  # TODO: Make configurable
lr = legacy_training.get('learning_rate', 1e-4)  # TODO: Make configurable
```
**Status:** Partially configurable through `legacy_gat` config section

### 12. **Legacy Model Architecture** (Lines 1757-1763 - not shown in excerpts)
```python
LegacyGATModel(
    node_input_dim=3,          # Hardcoded
    edge_input_dim=1,          # Hardcoded  
    hidden_dim=...,            # ✅ Configurable
    edge_dim=legacy_config.get('edge_dim', 16),  # ✅ Configurable
    layers=...,                # ✅ Configurable
    negative_slope=legacy_config.get('negative_slope', 0.2),  # ✅ Configurable
    dropout=legacy_config.get('dropout', 0.6)  # ✅ Configurable
)
```

## 🟢 VALIDATION & UTILITY CONSTANTS

### 13. **Validation Frequency** (Line 1381 - not shown)
```python
validation_frequency = config.get('training', {}).get('validation_frequency', 3)
```
**Status:** ✅ Already configurable

### 14. **Naive Baseline Calculation** (Line 1108)
```python
naive_cost += distances[0, customer_idx] * 2  # Hardcoded multiplier: 2
```
**Impact:** Depot-to-customer-to-depot cost calculation
**Note:** This is algorithmic, not a tunable parameter

## 📋 PRIORITY RECOMMENDATIONS

### HIGH PRIORITY (Easy wins with significant impact):
1. **Max steps multiplier** - Configurable route length limits
2. **Attention temperature scaling** - Fine-tune attention sharpness  
3. **Pointer network multipliers** - Adjust model capacity
4. **Masking/epsilon values** - Numerical stability tuning

### MEDIUM PRIORITY:
5. **Dynamic update architecture** - DGT model customization
6. **Residual gate initialization** - DGT training dynamics
7. **Legacy model parameters** - Complete legacy model configuration

### LOW PRIORITY (Structural changes required):
8. **State feature count** - Requires architectural changes
9. **Node feature dimensions** - Input format dependent

## 🔧 IMPLEMENTATION NOTES

- Most parameters can be added to YAML config with sensible defaults
- Some parameters (like state_features_count) would require code restructuring
- Legacy model parameters are already partially configurable
- Numerical constants should be tunable for different precision requirements
- Route length multiplier is the most impactful remaining parameter
