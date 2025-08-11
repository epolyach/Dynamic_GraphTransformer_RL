# Remaining Hardcoded Parameters in run_train_validation.py

After the recent updates to remove hardcoded fallbacks, here are the **remaining hardcoded parameters** that still need to be made configurable:

## 1. DynamicGraphTransformerNetwork Class (Lines 679-904)

### Constructor Hardcoded Values:
- **Line 702**: `self.state_encoder = nn.Linear(4, hidden_dim)` - hardcoded input dimension of 4
- **Line 705**: `self.res_gate = nn.Parameter(torch.tensor(-2.19722458))` - **MAJOR**: hardcoded gate initialization value
- **Line 707**: `nn.Linear(hidden_dim * 2, hidden_dim)` - hardcoded multiplier of 2
- **Line 714**: `nn.Linear(hidden_dim * 3, hidden_dim)` - hardcoded multiplier of 3

### Forward Method Hardcoded Values:
- **Line 719**: `temperature=1.0` - hardcoded default temperature  
- **Line 722**: `max_steps = len(instances[0]['coords']) * 2` - hardcoded multiplier of 2

### Route Generation Hardcoded Values:
- **Line 841**: `scores.masked_fill(mask, -1e9)` - hardcoded masked score value
- **Line 844**: `torch.log(probs + 1e-12)` - hardcoded epsilon for log probability

## 2. GraphAttentionTransformer Class (Lines 906-1086)

### Constructor Hardcoded Values:
- **Line 918**: `nn.Linear(1, hidden_dim // edge_embedding_divisor)` - hardcoded input dimension of 1
- **Line 935**: `nn.Linear(hidden_dim * 2, hidden_dim)` - hardcoded multiplier of 2

### Forward Method Hardcoded Values:
- **Line 940**: `temperature=1.0` - hardcoded default temperature
- **Line 943**: `max_steps = len(instances[0]['coords']) * 2` - hardcoded multiplier of 2

### Route Generation Hardcoded Values:
- **Line 1031**: `scores.masked_fill(mask, -1e9)` - hardcoded masked score value  
- **Line 1034**: `torch.log(probs + 1e-12)` - hardcoded epsilon for log probability

## 3. Legacy Training Function (Lines 1489-1655)

### Hardcoded Training Parameters:
- **Line 1504**: `split_idx = int(0.8 * len(data_list))` - hardcoded train/val split ratio
- **Line 1515**: `T = legacy_training.get('temperature', 2.5)` - hardcoded default temperature fallback
- **Line 1516**: `lr = legacy_training.get('learning_rate', 1e-4)` - hardcoded default learning rate fallback
- **Line 1553**: `cost += float(((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2) ** 0.5)` - hardcoded distance calculation

## 4. Model Initialization (Lines 1828-1839)

### Hardcoded Default Values:
- **Line 1830**: `dropout = config.get('model', {}).get('transformer_dropout', 0.1)` - hardcoded fallback
- **Line 1831**: `feedforward_multiplier = config.get('model', {}).get('feedforward_multiplier', 4)` - hardcoded fallback
- **Line 1832**: `edge_embedding_divisor = config.get('model', {}).get('edge_embedding_divisor', 4)` - hardcoded fallback

### Legacy Model Hardcoded Fallbacks:
- **Line 1850**: `hidden_dim=legacy_config.get('hidden_dim', config.get('hidden_dim', 128))` - hardcoded 128 fallback
- **Line 1851**: `edge_dim=legacy_config.get('edge_dim', 16)` - hardcoded 16 fallback  
- **Line 1852**: `layers=legacy_config.get('layers', config.get('num_layers', 4))` - hardcoded 4 fallback
- **Line 1853**: `negative_slope=legacy_config.get('negative_slope', 0.2)` - hardcoded 0.2 fallback
- **Line 1854**: `dropout=legacy_config.get('dropout', 0.6)` - hardcoded 0.6 fallback

## 5. Main Configuration (Line 1786)

- **Line 1786**: `config.setdefault('temperature', 1.0)` - hardcoded default temperature

## Priority for Configuration

### 🔴 HIGH PRIORITY (Critical for model behavior):
1. **Line 705**: `torch.tensor(-2.19722458)` - DGT gate initialization
2. **Lines 841, 1031**: `-1e9` - masked score values
3. **Lines 844, 1034**: `1e-12` - log probability epsilons  
4. **Lines 719, 940**: `temperature=1.0` - default temperatures
5. **Lines 722, 943**: `* 2` - max steps multipliers

### 🟡 MEDIUM PRIORITY (Architecture parameters):
1. **Lines 702, 707, 714**: Hidden dimension multipliers in DGT
2. **Lines 918, 935**: Hidden dimension multipliers in GAT
3. **Line 1515**: `2.5` - legacy temperature fallback

### 🟢 LOW PRIORITY (Fallback values):
1. Model initialization fallback values (lines 1830-1832, 1850-1854)
2. Default temperature in main config (line 1786)
3. Train/val split ratio (line 1504)

## Status Summary

- **✅ FIXED**: BaselinePointerNetwork, GraphTransformerNetwork, GraphTransformerGreedy now use config parameters
- **❌ REMAINING**: DynamicGraphTransformerNetwork and GraphAttentionTransformer still have hardcoded values
- **❌ REMAINING**: Legacy training function has several hardcoded parameters
- **❌ REMAINING**: Model initialization fallbacks are still hardcoded

## Next Steps

1. Update `DynamicGraphTransformerNetwork` to accept and use config parameters
2. Update `GraphAttentionTransformer` to accept and use config parameters  
3. Add corresponding config entries in YAML files
4. Update legacy training to use config parameters more extensively
5. Remove hardcoded fallbacks in model initialization
