#!/usr/bin/env python3
import time
import os
import json

def check_progress():
    # Check for completed JSON files
    json_files = [f for f in os.listdir('.') if f.startswith('ortools_gls_n100_1000inst')]
    if json_files:
        latest = sorted(json_files)[-1]
        print(f"✓ Benchmark completed! Results in: {latest}")
        with open(latest, 'r') as f:
            data = json.load(f)
            print(f"  GM: {data['gm']:.6f}")
            print(f"  GSD: {data['gsd']:.6f}")
            print(f"  95% CI: [{data['ci_lower']:.6f}, {data['ci_upper']:.6f}]")
        return True
    
    # Check if process is still running
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'benchmark_ortools_gls_fixed.py'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"⏳ Benchmark still running (PID: {result.stdout.strip()})")
        print(f"   Check log with: tail -f ortools_1000inst.log")
        return False
    else:
        print("❌ Benchmark process not found")
        return True

print("Monitoring OR-Tools benchmark progress...")
print("="*50)
check_progress()
