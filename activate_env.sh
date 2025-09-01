#!/bin/bash
# Setup and activate the virtual environment for Dynamic Graph Transformer project

# Check if venv exists, if not create and setup
if [ ! -d "venv" ]; then
    echo "================================================="
    echo "🔨 Virtual environment not found. Setting up..."
    echo "================================================="
    echo ""
    
    # Create virtual environment
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    # Activate it
    source venv/bin/activate
    
    # Upgrade pip
    echo "📦 Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install PyTorch (CPU version for MacOS)
    echo "🔥 Installing PyTorch (CPU version for MacOS)..."
    pip install torch torchvision torchaudio --quiet
    
    # Install other requirements
    echo "📦 Installing requirements from requirements.txt..."
    pip install -r requirements.txt --quiet
    
    echo ""
    echo "✅ Setup complete! Environment is now activated."
    echo ""
else
    # Just activate existing environment
    source venv/bin/activate
fi

# Show environment info
echo "🚀 Dynamic Graph Transformer environment activated! (CPU-optimized)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Not installed')"
echo "Pandas: $(python -c 'import pandas; print(pandas.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""
echo "Available configurations:"
echo "  🔬 configs/small.yaml       - Quick testing (10 epochs, 800 instances)"
echo "  🧪 configs/medium.yaml      - Research experiments (50 epochs, 10k instances)"
echo "  🏭 configs/production.yaml  - Publication results (200 epochs, 100k instances)"
echo "  🚀 configs/small_quick.yaml  - Ultra-fast testing (5 epochs, 100 instances)"
echo ""
echo "Usage examples:"
echo "  python run_comparative_study.py --config configs/small.yaml"
echo "  python run_enhanced_training.py --config configs/small_quick.yaml --models GT+RL"
echo ""
echo "Results will be saved to results/{config_name}/ respectively"
echo ""
