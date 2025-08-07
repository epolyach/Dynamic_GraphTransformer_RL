#!/bin/bash
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
