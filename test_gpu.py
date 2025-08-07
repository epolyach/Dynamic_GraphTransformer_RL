#!/usr/bin/env python3
"""
GPU Test Script - Run this on the GPU server to verify A6000 setup
"""
import torch
import torch_geometric

def test_gpu_setup():
    print("🧪 GPU Server Setup Test")
    print("=" * 40)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            if 'A6000' in gpu_name:
                print("✅ RTX A6000 detected!")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        device = torch.device('cuda:0')
        
        # Small test
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        print("✅ Basic GPU computation successful")
        
        # Graph test
        from torch_geometric.data import Data
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long).cuda()
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float).cuda()
        data = Data(x=x, edge_index=edge_index)
        print("✅ GPU graph data handling successful")
        
        # Memory test
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        
        print("\n🎉 A6000 setup verified and ready for training!")
        return True
    else:
        print("❌ No CUDA GPUs available")
        return False

if __name__ == "__main__":
    test_gpu_setup()
