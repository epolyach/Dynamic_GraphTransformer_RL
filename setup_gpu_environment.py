#!/usr/bin/env python3
"""
GPU Environment Setup and Validation Script
Sets up the environment for Dynamic Graph Transformer with A6000 GPU support
"""

import sys
import subprocess
import importlib.metadata
from typing import List, Dict, Tuple
import os

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version compatible")
    return True

def check_cuda_availability():
    """Check CUDA installation and GPU availability"""
    print("\n🔧 Checking CUDA availability...")
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi found")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX A6000' in line or 'A6000' in line:
                    print(f"✅ Found A6000 GPU: {line.strip()}")
                    break
        else:
            print("❌ nvidia-smi not found - GPU drivers may not be installed")
            return False
            
    except FileNotFoundError:
        print("❌ nvidia-smi command not found")
        return False
    
    # Check CUDA version
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVCC (CUDA compiler) found")
        else:
            print("⚠️  NVCC not found - CUDA toolkit may not be installed")
    except FileNotFoundError:
        print("⚠️  NVCC not found - CUDA toolkit may not be installed")
    
    return True

def get_required_packages() -> List[Tuple[str, str]]:
    """Get list of required packages with versions"""
    return [
        # Core ML/DL Dependencies
        ("torch", ">=2.0.0"),
        ("torch-geometric", ">=2.4.0"),
        ("torch-scatter", ">=2.1.0"),
        ("torch-sparse", ">=0.6.0"),
        ("torch-cluster", ">=1.6.0"),
        
        # Scientific Computing
        ("numpy", ">=1.21.0"),
        ("scipy", ">=1.7.0"),
        ("scikit-learn", ">=1.0.0"),
        
        # Data Handling
        ("pandas", ">=1.3.0"),
        ("networkx", ">=2.8.0"),
        
        # Visualization
        ("matplotlib", ">=3.5.0"),
        ("seaborn", ">=0.11.0"),
        
        # Configuration and Logging
        ("pyyaml", ">=6.0.0"),
        ("tqdm", ">=4.64.0"),
        
        # Development
        ("pytest", ">=7.0.0"),
    ]

def check_package_installation() -> Dict[str, bool]:
    """Check which required packages are installed"""
    print("\n📦 Checking package installation...")
    
    required_packages = get_required_packages()
    installation_status = {}
    
    for package_name, version_req in required_packages:
        try:
            # Try to get the installed version using modern approach
            installed_version = importlib.metadata.version(package_name)
            print(f"✅ {package_name}: {installed_version}")
            installation_status[package_name] = True
        except importlib.metadata.PackageNotFoundError:
            print(f"❌ {package_name}: Not installed")
            installation_status[package_name] = False
        except Exception as e:
            print(f"⚠️  {package_name}: Error checking version ({e})")
            installation_status[package_name] = False
    
    return installation_status

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("\n🔥 Installing PyTorch with CUDA support...")
    
    # PyTorch with CUDA 12.1 support (compatible with A6000)
    pytorch_install_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    try:
        print("   Running: " + " ".join(pytorch_install_cmd))
        result = subprocess.run(pytorch_install_cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch with CUDA installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False

def install_torch_geometric():
    """Install PyTorch Geometric with compatible versions"""
    print("\n🔗 Installing PyTorch Geometric...")
    
    # Install PyTorch Geometric and extensions
    packages = [
        "torch-geometric",
        "torch-scatter", 
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv"
    ]
    
    for package in packages:
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            print(f"   Installing {package}...")
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def install_other_packages():
    """Install other required packages"""
    print("\n📚 Installing other required packages...")
    
    other_packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "networkx>=2.8.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.64.0",
        "pytest>=7.0.0"
    ]
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + other_packages
        print("   Installing packages...")
        subprocess.run(cmd, check=True, capture_output=True)
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def test_gpu_pytorch():
    """Test PyTorch GPU functionality"""
    print("\n🧪 Testing PyTorch GPU functionality...")
    
    try:
        import torch
        
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test GPU computation
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print("✅ GPU computation test successful")
            
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_torch_geometric():
    """Test PyTorch Geometric functionality"""
    print("\n🔗 Testing PyTorch Geometric functionality...")
    
    try:
        import torch
        import torch_geometric
        from torch_geometric.data import Data
        
        print(f"   PyTorch Geometric version: {torch_geometric.__version__}")
        
        # Create test graph data
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index)
        print(f"   Test graph: {data}")
        
        if torch.cuda.is_available():
            data = data.cuda()
            print("✅ PyTorch Geometric GPU test successful")
        else:
            print("✅ PyTorch Geometric CPU test successful")
            
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import PyTorch Geometric: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch Geometric test failed: {e}")
        return False

def create_environment_info():
    """Create environment info file"""
    print("\n💾 Creating environment info file...")
    
    try:
        import torch
        import torch_geometric
        
        env_info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'pytorch_geometric_version': torch_geometric.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            env_info['cuda_version'] = torch.version.cuda
            env_info['gpus'] = []
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_gb': torch.cuda.get_device_properties(i).total_memory / 1024**3
                }
                env_info['gpus'].append(gpu_info)
        
        # Save to file
        import yaml
        with open('environment_info.yaml', 'w') as f:
            yaml.dump(env_info, f, default_flow_style=False)
            
        print("✅ Environment info saved to environment_info.yaml")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create environment info: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Dynamic Graph Transformer GPU Environment Setup")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check CUDA availability  
    if not check_cuda_availability():
        print("\n⚠️  Warning: CUDA not detected. Proceeding with CPU-only setup.")
    
    # Step 3: Check existing packages
    installation_status = check_package_installation()
    
    # Step 4: Install packages if needed
    need_pytorch = not installation_status.get('torch', False)
    need_geometric = not installation_status.get('torch-geometric', False)
    
    if need_pytorch:
        if not install_pytorch_cuda():
            print("\n❌ Failed to install PyTorch. Please install manually.")
            sys.exit(1)
    
    if need_geometric:
        if not install_torch_geometric():
            print("\n❌ Failed to install PyTorch Geometric. Please install manually.")
            sys.exit(1)
    
    # Install other packages
    if not install_other_packages():
        print("\n⚠️  Some packages failed to install. Please check manually.")
    
    # Step 5: Test GPU functionality
    if not test_gpu_pytorch():
        print("\n⚠️  GPU tests failed. Continuing with CPU setup.")
    
    # Step 6: Test PyTorch Geometric
    if not test_torch_geometric():
        print("\n❌ PyTorch Geometric tests failed.")
        sys.exit(1)
    
    # Step 7: Create environment info
    create_environment_info()
    
    print("\n" + "=" * 60)
    print("🎉 Environment setup complete!")
    print("\n📋 Next steps:")
    print("   1. Run: python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\")") 
    print("   2. Check environment_info.yaml for details")
    print("   3. Proceed with model implementation")

if __name__ == "__main__":
    main()
