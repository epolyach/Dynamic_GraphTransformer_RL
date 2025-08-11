# Remaining Hardcoded Parameters Analysis

After refactoring the `BaselinePointerNetwork` class to use config parameters, there are still several hardcoded parameters throughout the training script. Here's a comprehensive analysis:

## 1. Model Architecture Constants

### GraphTransformerNetwork and related classes (Lines 341, 435, 438)
```python
# Hardcoded pointer network dimensions
self.pointer = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),  # * 2 is hardcoded
    nn.ReLU(),
    nn.Linear(hidden_dim, 1)
)

# Hardcoded masking values in non-BaselinePointer models
scores = scores.masked_fill(mask, -1e9)  # Should use config['inference']['masked_score_value']
log_probs = torch.log(probs + 1e-12)    # Should use config['inference']['log_prob_epsilon']
```

### Max Steps Calculation (Lines 349, 530, 712, 933)
```python
# Still hardcoded * 2 multiplier in non-BaselinePointer models
max_steps = len(instances[0]['coords']) * 2  # Should use config['inference']['max_steps_multiplier']
```

### Default Temperature (Lines 346, 526, 709, 930)  
```python
# Default temperature hardcoded to 1.0
def forward(self, instances, max_steps=None, temperature=1.0, greedy=False):
```

### DynamicGraphTransformerNetwork Special Constants (Line 695)
```python
# Hardcoded gating parameter for residual connection
self.res_gate = nn.Parameter(torch.tensor(-2.19722458))  # sigmoid ~= 0.1
```

### State Encoder Input Dimension (Line 692)
```python
# Hardcoded to 4 features: capacity_used, step, visited_count, distance_from_depot
self.state_encoder = nn.Linear(4, hidden_dim)
```

### Pointer Network Input Dimensions (Lines 704, 925)
```python
# Hardcoded concatenation dimensions
nn.Linear(hidden_dim * 3, hidden_dim)  # DGT: node + context + state
nn.Linear(hidden_dim * 2, hidden_dim)  # Others: node + context
```

## 2. Training Logic Constants

### Train/Validation Split (Line 1297, 1490)
```python
# Hardcoded 80/20 split (though config fallback exists)
train_val_split = config.get('training', {}).get('train_val_split', 0.8)  # 0.8 is hardcoded fallback
split_idx = int(0.8 * len(data_list))  # Direct hardcoded 0.8 in legacy training
```

### Mathematical Constants (Lines 1289, 1317, 1358)
```python
# Hardcoded cosine schedule factor
cosine = 0.5 * (1 + math.cos(math.pi * t / total))  # 0.5 is hardcoded

# Hardcoded cosine temperature schedule
cosine_t = 0.5 * (1 + math.cos(math.pi * epoch / (config['num_epochs'] - 1)))  # 0.5 hardcoded

# Hardcoded cosine entropy decay
cosine_factor = 0.5 * (1 + math.cos(math.pi * epoch / (config['num_epochs'] - 1)))  # 0.5 hardcoded
```

### Naive Baseline Calculation (Line 1114)
```python
# Hardcoded multiplier for naive cost calculation  
naive_cost += distances[0, customer_idx] * 2  # depot->customer->depot = * 2
```

## 3. Legacy GAT Model Constants (Lines 1501-1502)
```python
# Legacy model still has some hardcoded parameters
T = legacy_training.get('temperature', 2.5)      # 2.5 fallback
lr = legacy_training.get('learning_rate', 1e-4)  # 1e-4 fallback
```

## 4. Validation and Evaluation Constants (Lines 1582)
```python
# Hardcoded validation epoch intervals
eval_epochs = list(range(0, num_epochs, 4))  # Every 4 epochs is hardcoded
```

## 5. Node Features and Input Dimensions

### Node Feature Tensor Dimensions (Multiple locations)
```python
# Hardcoded to 3 features: [x, y, demand]
node_features = torch.zeros(batch_size, max_nodes, 3)
```

## 6. Distance Calculation (Lines 1539, 1618)
```python
# Hardcoded Euclidean distance calculation in legacy evaluation
cost += float(((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2) ** 0.5)
```

## Priority for Refactoring

### High Priority (Should be configurable)
1. **Masking values (-1e9, 1e-12)** in non-BaselinePointer models - Critical for numerical stability
2. **Max steps multiplier (* 2)** in all models except BaselinePointer
3. **Default temperature (1.0)** in model forward signatures
4. **Mathematical constants (0.5)** in cosine schedules - affects learning dynamics

### Medium Priority  
1. **Pointer network dimensions** - affects model capacity
2. **Train/validation split ratios** - affects evaluation
3. **Validation frequency (4 epochs)** - affects training monitoring

### Low Priority (Architecture-specific)
1. **DGT gating parameter (-2.19722458)** - model-specific tuning
2. **State encoder dimensions (4)** - tied to feature engineering
3. **Node feature dimensions (3)** - tied to problem representation

## Recommended Configuration Extensions

Add to config YAML:
```yaml
model:
  pointer_network:
    context_multiplier: 2  # For hidden_dim * 2 concatenations
    enhanced_context_multiplier: 3  # For DGT hidden_dim * 3
  
  dynamic_graph_transformer:
    residual_gate_init: -2.19722458
    state_encoder_input_dim: 4
  
inference:
  default_temperature: 1.0
  
training:
  cosine_schedule_factor: 0.5
  validation_epoch_interval: 4

legacy_gat:
  temperature_fallback: 2.5
  learning_rate_fallback: 1e-4
```

## Implementation Strategy

1. **Phase 1**: Fix critical numerical stability parameters (masking values, max steps)
2. **Phase 2**: Make temperature and cosine schedule factors configurable  
3. **Phase 3**: Add architectural flexibility (pointer dimensions, validation intervals)
4. **Phase 4**: Fine-tuning parameters (gating, state dimensions)
