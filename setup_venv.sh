#!/bin/bash

# Setup script for Dynamic Graph Transformer RL project
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "=================================================="
echo "🚀 Setting up Dynamic Graph Transformer RL Project"
echo "=================================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "📦 Python version detected: $PYTHON_VERSION"

# Remove existing venv if it exists
if [ -d "venv" ]; then
    echo "⚠️  Removing existing virtual environment..."
    rm -rf venv
fi

# Create new virtual environment
echo "🔨 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for MacOS)
echo "🔥 Installing PyTorch (CPU version for MacOS)..."
pip install torch torchvision torchaudio

# Install other requirements
echo "📦 Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')"
python -c "import pandas; print(f'✓ Pandas {pandas.__version__}')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import yaml; print(f'✓ PyYAML installed')"
python -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')"

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source activate_env.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo ""
echo "To run the enhanced training script:"
echo "  ./run_training.sh"
echo ""
