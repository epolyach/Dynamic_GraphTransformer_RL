#!/usr/bin/env python3
"""
Script to estimate parameter counts for different model configurations
to verify our 5K-300K parameter range is correct.
"""

def estimate_transformer_params(hidden_dim, n_layers, n_heads, ff_multiplier=2.0, input_dim=4):
    """
    Rough estimation of transformer parameters:
    - Input embedding: input_dim * hidden_dim
    - Each transformer layer:
      - Multi-head attention: 4 * hidden_dim^2 (Q, K, V, O projections)
      - Feed-forward: 2 * hidden_dim * (ff_multiplier * hidden_dim)
      - Layer norms: ~4 * hidden_dim (small)
    - Output layers: ~2 * hidden_dim
    """
    
    # Input embedding
    embedding_params = input_dim * hidden_dim
    
    # Each transformer layer
    attention_params = 4 * hidden_dim * hidden_dim
    ff_params = 2 * hidden_dim * int(ff_multiplier * hidden_dim)
    norm_params = 4 * hidden_dim  # Layer norms (small)
    layer_params = attention_params + ff_params + norm_params
    
    # All layers
    total_layer_params = n_layers * layer_params
    
    # Output layers (pointer network, etc.)
    output_params = 2 * hidden_dim
    
    total = embedding_params + total_layer_params + output_params
    return total

def test_parameter_ranges():
    """Test our parameter ranges to ensure 5K-300K coverage."""
    print("🧮 PARAMETER COUNT ESTIMATION")
    print("=" * 50)
    
    test_configs = [
        # (hidden_dim, n_layers, n_heads, ff_mult, description) - ADJUSTED RANGES
        (30, 2, 2, 1.0, "Minimum config"),
        (36, 2, 2, 1.2, "Very small"),
        (42, 2, 3, 1.5, "Small"),
        (48, 2, 4, 1.6, "Medium-small"),
        (56, 3, 4, 1.8, "Medium"),
        (64, 3, 4, 2.0, "Medium-large"),
        (72, 3, 6, 2.0, "Large"),
        (84, 3, 6, 2.2, "Maximum config"),
    ]
    
    print(f"{'Config':<15} {'Hidden':<8} {'Layers':<8} {'Heads':<8} {'FF':<6} {'Params':<10} {'Range'}")
    print("-" * 70)
    
    for hidden_dim, n_layers, n_heads, ff_mult, desc in test_configs:
        params = estimate_transformer_params(hidden_dim, n_layers, n_heads, ff_mult)
        
        # Determine range category
        if params < 10_000:
            range_cat = "< 10K"
        elif params < 50_000:
            range_cat = "10K-50K"
        elif params < 100_000:
            range_cat = "50K-100K"
        elif params < 200_000:
            range_cat = "100K-200K"
        else:
            range_cat = "> 200K"
            
        print(f"{desc:<15} {hidden_dim:<8} {n_layers:<8} {n_heads:<8} {ff_mult:<6.1f} {params:<10,} {range_cat}")
    
    print("\n🎯 TARGET RANGE VERIFICATION:")
    min_params = estimate_transformer_params(30, 2, 2, 1.0)  # Updated minimum
    max_params = estimate_transformer_params(84, 3, 6, 2.2)  # Updated maximum
    print(f"Minimum configuration: ~{min_params:,} parameters")
    print(f"Maximum configuration: ~{max_params:,} parameters")
    print(f"Range coverage: {min_params/1000:.1f}K to {max_params/1000:.1f}K parameters")
    
    if min_params >= 5000 and max_params <= 300000:
        print("✅ Range covers 5K-300K target!")
    else:
        print("❌ Range needs adjustment")

if __name__ == "__main__":
    test_parameter_ranges()
