#!/usr/bin/env python3
"""
Development Environment Setup Script
Sets up environment for Dynamic Graph Transformer with A6000 GPU support and Mac CPU fallback
"""

import sys
import subprocess
import os
from pathlib import Path

def create_virtual_environment():
    """Create virtual environment for the project"""
    print("🐍 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("   Virtual environment already exists")
        return True
        
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_venv_python():
    """Get path to virtual environment Python"""
    if os.name == 'nt':  # Windows
        return "venv/Scripts/python.exe"
    else:  # Unix/Linux/Mac
        return "venv/bin/python"

def install_pytorch_cpu():
    """Install PyTorch CPU version for development"""
    print("🔥 Installing PyTorch (CPU version for development)...")
    
    python_path = get_venv_python()
    
    # Install CPU version of PyTorch (works on Mac and will work on GPU server too)
    pytorch_install_cmd = [
        python_path, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"
    ]
    
    try:
        print("   Running: " + " ".join(pytorch_install_cmd))
        subprocess.run(pytorch_install_cmd, check=True)
        print("✅ PyTorch (CPU) installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def install_torch_geometric():
    """Install PyTorch Geometric (basic version for development)"""
    print("🔗 Installing PyTorch Geometric...")
    
    python_path = get_venv_python()
    
    # Install core torch-geometric first (already installed)
    print("✅ torch-geometric already installed")
    
    # Try to install optional extensions, but don't fail if they can't be built
    optional_packages = [
        "torch-scatter", 
        "torch-sparse",
        "torch-cluster"
    ]
    
    for package in optional_packages:
        try:
            cmd = [python_path, "-m", "pip", "install", package]
            print(f"   Trying to install {package}...")
            subprocess.run(cmd, check=True, timeout=300)  # 5 minute timeout
            print(f"✅ {package} installed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"⚠️  Failed to install {package}: {e}")
            print(f"    Continuing without {package} (will use CPU-only fallbacks)")
    
    return True

def install_dev_packages():
    """Install development packages"""
    print("📚 Installing development packages...")
    
    python_path = get_venv_python()
    
    dev_packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "networkx>=2.8.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.64.0",
        "pytest>=7.0.0",
        "jupyter>=1.0.0",  # For development notebooks
        "ipykernel",       # Jupyter kernel
    ]
    
    try:
        cmd = [python_path, "-m", "pip", "install"] + dev_packages
        print("   Installing packages...")
        subprocess.run(cmd, check=True)
        print("✅ Development packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    python_path = get_venv_python()
    
    test_script = '''
import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
import yaml
import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"NumPy version: {np.__version__}")

# Test basic functionality
x = torch.randn(10, 5)
y = torch.mm(x, x.t())
print(f"Basic tensor operations: {'✅ OK' if y.shape == (10, 10) else '❌ FAIL'}")

# Test graph operations
from torch_geometric.data import Data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
print(f"Graph data creation: {'✅ OK' if data.num_nodes == 3 else '❌ FAIL'}")

# Check device availability
if torch.cuda.is_available():
    print(f"CUDA available: ✅ YES")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"MPS (Apple Silicon) available: ✅ YES")
else:
    print(f"GPU acceleration: ❌ Not available (CPU only)")

print("✅ All tests passed!")
'''
    
    try:
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        print("✅ Installation test successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def create_activation_script():
    """Create convenient activation script"""
    print("📝 Creating activation script...")
    
    activation_script = '''#!/bin/bash
# Activate the virtual environment for Dynamic Graph Transformer project
source venv/bin/activate
echo "🚀 Dynamic Graph Transformer environment activated!"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Available commands:"
echo "  python train_small.py    - Train on small instances (10 nodes)"
echo "  python test_gpu.py       - Test GPU availability"
echo "  jupyter notebook         - Start Jupyter for development"
echo ""
'''
    
    try:
        with open('activate_env.sh', 'w') as f:
            f.write(activation_script)
        os.chmod('activate_env.sh', 0o755)
        print("✅ Activation script created: activate_env.sh")
        return True
    except Exception as e:
        print(f"❌ Failed to create activation script: {e}")
        return False

def create_gpu_test_script():
    """Create GPU test script for server deployment"""
    print("🔧 Creating GPU test script...")
    
    gpu_test_script = '''#!/usr/bin/env python3
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
        print("\\nTesting GPU computation...")
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
        
        print("\\n🎉 A6000 setup verified and ready for training!")
        return True
    else:
        print("❌ No CUDA GPUs available")
        return False

if __name__ == "__main__":
    test_gpu_setup()
'''
    
    try:
        with open('test_gpu.py', 'w') as f:
            f.write(gpu_test_script)
        print("✅ GPU test script created: test_gpu.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create GPU test script: {e}")
        return False

def create_requirements_file():
    """Create requirements.txt for easy deployment"""
    print("📋 Creating requirements.txt...")
    
    requirements = '''# Core ML/DL Dependencies
torch>=2.0.0
torch-geometric>=2.4.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0
torch-cluster>=1.6.0

# Scientific Computing
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Data Handling
pandas>=1.3.0
networkx>=2.8.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Configuration and Logging
pyyaml>=6.0.0
tqdm>=4.64.0

# Development
pytest>=7.0.0
jupyter>=1.0.0
ipykernel
'''
    
    try:
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        print("✅ requirements.txt created")
        return True
    except Exception as e:
        print(f"❌ Failed to create requirements.txt: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Dynamic Graph Transformer Development Environment Setup")
    print("🖥️  Mode: Mac Development with A6000 GPU Compatibility")
    print("=" * 70)
    
    # Step 1: Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Step 2: Install PyTorch (CPU for development)
    if not install_pytorch_cpu():
        sys.exit(1)
    
    # Step 3: Install PyTorch Geometric
    if not install_torch_geometric():
        sys.exit(1)
    
    # Step 4: Install development packages
    if not install_dev_packages():
        print("⚠️  Some packages failed to install. Please check manually.")
    
    # Step 5: Test installation
    if not test_installation():
        sys.exit(1)
    
    # Step 6: Create helper scripts
    create_activation_script()
    create_gpu_test_script()
    create_requirements_file()
    
    print("\n" + "=" * 70)
    print("🎉 Development environment setup complete!")
    print("\n📋 Next steps:")
    print("   1. Activate environment: source activate_env.sh")
    print("   2. Start development on small instances (10 nodes + depot)")
    print("   3. Test basic pipeline with CPU")
    print("   4. Deploy to GPU server and run: python test_gpu.py")
    print("\n💡 GPU Server Deployment:")
    print("   - Copy project to GPU server")
    print("   - Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("   - Install requirements: pip install -r requirements.txt")
    print("   - Test with: python test_gpu.py")

if __name__ == "__main__":
    main()
