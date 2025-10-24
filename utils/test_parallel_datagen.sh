#!/bin/bash

# Quick test script for parallel data generation
# This will run a short test to verify multi-core CPU usage

echo "=============================================="
echo "Testing Parallel Data Generation"
echo "=============================================="
echo ""
echo "This test will generate 3 batches of 512 instances"
echo "using 16 worker processes."
echo ""
echo "MONITOR CPU USAGE:"
echo "  Open another terminal and run: htop"
echo "  or: watch -n 1 'ps aux | grep python'"
echo ""
echo "Press ENTER to start the test..."
read

# Activate environment
source venv/bin/activate

# Run the test
python3 << 'PYTEST'
import sys
from pathlib import Path
import time

# Setup path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.generator.generator import ParallelDataGeneratorPool
from src.utils.config import load_config

# Load config
config = load_config("configs/tiny_gpu_1000.yaml")

print("=" * 60)
print("Testing Parallel Data Generation")
print("=" * 60)
print()

# Create pool with 16 workers
print("Creating ParallelDataGeneratorPool with 16 workers...")
pool = ParallelDataGeneratorPool(config, num_workers=16)
print()

print("Generating 512 instances per batch...")
print("Monitor CPU usage now - you should see 6+ Python processes active")
print()

# Generate a few batches to warm up and observe
for i in range(3):
    print(f"Batch {i+1}/3: Generating...", end=" ", flush=True)
    start = time.time()
    instances = pool.generate_batch(batch_size=512, epoch=i)
    elapsed = time.time() - start
    print(f"Done! ({len(instances)} instances in {elapsed:.2f}s)")
    time.sleep(1)  # Pause between batches

pool.close()
print()
print("=" * 60)
print("Test complete! Pool closed.")
print("=" * 60)
PYTEST

echo ""
echo "Test completed!"
echo ""
echo "If you saw multiple Python processes in htop/ps, "
echo "parallel data generation is working correctly!"
