#!/usr/bin/env python3
"""
Simple profiling script to check GPU usage during training
"""
import torch
import time
import psutil

def profile_gpu_training():
    print("🔍 Profiling GPU Usage...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Check initial GPU state
    
    # Simulate training workload
    batch_size = 8
    seq_len = 50
    hidden_dim = 128
    
    print(f"\n🧪 Testing batch size {batch_size}, seq_len {seq_len}, hidden_dim {hidden_dim}")
    
    # Create test tensors on GPU
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
    y = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Simple neural network operations
    linear = torch.nn.Linear(hidden_dim, hidden_dim).to(device)
    
    start_time = time.time()
    for i in range(100):
        out = linear(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        linear.zero_grad()
    
    end_time = time.time()
    
    print(f"⏱️  Time for 100 iterations: {end_time - start_time:.2f}s")
    print(f"   Time per iteration: {(end_time - start_time)*10:.1f}ms")
    
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(0) / 1e6
        reserved = torch.cuda.memory_reserved(0) / 1e6
        print(f"🔥 GPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")

if __name__ == "__main__":
    try:
        profile_gpu_training()
    except ImportError:
        print("Note: GPUtil not available, install with: pip install gputil")
        print("Continuing without GPU monitoring...")
        profile_gpu_training()
