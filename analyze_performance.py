#!/usr/bin/env python3
"""Analyze why large batch training is slow"""

print("="*70)
print("PERFORMANCE ANALYSIS: Why EET=220s with batch_size=3072")
print("="*70)

print("\n1. BASELINE INITIALIZATION BOTTLENECK:")
print("   - With batch_size=3072, eval_batches=5 means 15,360 instances")
print("   - Each instance requires a GREEDY ROLLOUT through the model")
print("   - That's 15,360 model inference calls just for initialization!")
print("   - At ~10ms per inference: 153 seconds just for baseline init")

print("\n2. MEMORY BANDWIDTH SATURATION:")
print("   - Batch 3072 requires moving ~3072 * 11 * 11 * 4 bytes = 1.4MB distances")
print("   - Attention scores: 3072 * 11 * 11 * 4 = 1.4MB per layer")
print("   - With 3 layers: ~4.2MB just for attention")
print("   - Memory bandwidth becomes the bottleneck, not compute")

print("\n3. ATTENTION COMPLEXITY:")
print("   - Attention is O(n²) in memory and compute")
print("   - batch=512: 512 * 11² = 61,952 attention ops")
print("   - batch=3072: 3072 * 11² = 371,712 attention ops (6x more)")
print("   - But GPU can't parallelize beyond a certain point")

print("\n4. OVERHEAD DOMINATES AT SMALL n=10:")
print("   - With only 10 customers, the model is tiny")
print("   - Most time is spent on memory transfers, not compute")
print("   - Large batches increase overhead without utilizing compute")

print("\n5. OPTIMAL BATCH SIZE:")
print("   - For n=10: batch_size=512-1024 is likely optimal")
print("   - For n=20: batch_size=1024-2048 might work better")
print("   - For n=50+: batch_size=2048-3072 would shine")

print("\nRECOMMENDATIONS:")
print("1. Reduce batch_size back to 1024 for n=10")
print("2. Set eval_batches=1 for large batches")
print("3. For n=10, the model is too small to benefit from huge batches")
print("="*70)
