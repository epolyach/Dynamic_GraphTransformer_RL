#!/bin/bash
# Activate the GPU environment for CVRP benchmarks with CuPy support
source gpu_env/bin/activate
echo "🚀 GPU-accelerated CVRP environment activated!"
echo "Python: $(which python)"
echo "CuPy: $(python -c 'import cupy; print(cupy.__version__)' 2>/dev/null || echo 'Not available')"

# Test GPU availability 
GPU_COUNT=$(python -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount())' 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "GPU Device: $GPU_COUNT GPU(s) available ✅"
else
    echo "GPU Device: No GPU detected ❌"
fi

echo ""
echo "Usage:"
echo "  🖥️  CPU Benchmark: python3 benchmark_exact_cpu.py --help"  
echo "  🚀 GPU Benchmark: python3 benchmark_exact_gpu.py --help"
echo "  📊 GPU Plotting:  python3 plot_gpu_benchmark.py --help"
echo "  📈 Comparison:    python3 plot_cpu_gpu_comparison.py --help"
echo ""
echo "Example with true GPU acceleration:"
echo "  python3 benchmark_exact_gpu.py --n-start 5 --n-end 8 --instances 10 --debug"
echo ""
