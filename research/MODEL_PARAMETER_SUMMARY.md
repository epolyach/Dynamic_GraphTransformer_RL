# Dynamic Graph Transformer Model Parameter Summary

## Model Variants and Parameter Counts

### DGT-Super (Ultra-Lightweight)
- **Target**: ~30K parameters
- **Actual**: 28,021 parameters (0.93x target)
- **Reduction**: 12.8x smaller than GAT+RL baseline (360K)

#### Architecture:
- Hidden dimension: 48
- Attention heads: 3
- Attention layers: 2
- State encoding: 2 features (capacity + visit progress)
- Dynamic update: Simple residual connection
- Pointer network: 3-layer MLP

#### Key Design Decisions:
- Minimized hidden dimension while maintaining divisibility by heads
- Only 2 attention layers for basic representation learning
- Simplified state encoding with essential features only
- Lightweight pointer network with ReLU activations

### DGT-Lite (Balanced)
- **Target**: ~100K parameters
- **Actual**: 121,101 parameters (1.21x target)
- **Reduction**: 2.97x smaller than GAT+RL baseline (360K)

#### Architecture:
- Hidden dimension: 84 (divisible by 3 heads)
- Attention heads: 3
- Attention layers: 3
- State encoding: 4 features (capacity + visit + step + remaining demand)
- Dynamic update: 2-layer MLP with ReLU
- Pointer network: 3-layer MLP

#### Key Design Decisions:
- Balanced hidden dimension for good expressiveness
- 3 attention layers for deeper representation learning
- Enhanced state encoding with progress tracking
- More sophisticated dynamic update mechanism

## Baseline Comparison

| Model | Parameters | Reduction Factor | Key Features |
|-------|------------|------------------|--------------|
| GAT+RL (Baseline) | 360,000 | 1.0x | Graph attention + separate RL head |
| DGT-Lite | 121,101 | 2.97x smaller | Balanced performance/efficiency |
| DGT-Super | 28,021 | 12.8x smaller | Ultra-lightweight for deployment |

## Parameter Distribution Analysis

### DGT-Super Breakdown:
- Attention layers: ~18,816 params (67%)
- Pointer network: ~5,857 params (21%)
- Dynamic update: ~2,928 params (10%)
- Other components: ~420 params (2%)

### DGT-Lite Breakdown:
- Attention layers: ~84,672 params (70%)
- Pointer network: ~17,892 params (15%)
- Dynamic update: ~16,464 params (14%)
- Other components: ~2,073 params (1%)

## Design Principles

1. **Attention-First**: Both models prioritize attention mechanisms as the primary parameter allocation
2. **Efficient State Encoding**: Minimal but informative state features to guide decision-making
3. **Residual Connections**: Preserve gradient flow and enable deeper networks
4. **Divisible Dimensions**: Ensure clean division for multi-head attention
5. **Scalable Architecture**: Both models follow the same design pattern, differing only in scale

## Performance Expectations

- **DGT-Super**: Suitable for resource-constrained deployment, mobile devices, or real-time applications
- **DGT-Lite**: Better balance of performance and efficiency for most production scenarios

Both models maintain the core dynamic graph transformer innovations while achieving significant parameter reduction compared to traditional GAT+RL approaches.
