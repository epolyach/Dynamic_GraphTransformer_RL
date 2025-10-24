#!/usr/bin/env python3
"""
Show sequential runner scripts summary
"""
import os
import subprocess

def show_script_info(script_path):
    if os.path.exists(script_path):
        print(f"\n📁 {script_path}")
        print(f"   ✅ Executable: {oct(os.stat(script_path).st_mode)[-3:]}")
        
        # Get file size
        size = os.path.getsize(script_path)
        print(f"   📏 Size: {size} bytes")
        
        # Show first few lines
        with open(script_path, 'r') as f:
            lines = f.readlines()[:5]
            print(f"   📝 First lines:")
            for i, line in enumerate(lines, 1):
                print(f"      {i:2d}: {line.rstrip()}")
        
        # Check git status
        try:
            result = subprocess.run(['git', 'log', '--oneline', '-1', script_path], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                commit_info = result.stdout.strip()
                print(f"   📋 Last commit: {commit_info}")
        except:
            pass
    else:
        print(f"\n❌ {script_path} - Not found")

def main():
    print("🔧 Sequential Training Scripts Status")
    print("="*60)
    
    scripts = ['run_seq.sh', 'run_seq_nohup.sh', 'remove/run_sequential_training.sh']
    
    for script in scripts:
        show_script_info(script)
    
    print(f"\n{'='*60}")
    print("📋 USAGE SUMMARY:")
    print("1. run_seq.sh - Basic sequential runner")
    print("   ./run_seq.sh config1.yaml config2.yaml config3.yaml")
    print("   - Runs configs one after another in screen sessions")
    print("   - Each waits for the previous to complete")
    
    print("\n2. run_seq_nohup.sh - Persistent sequential runner") 
    print("   ./run_seq_nohup.sh config1.yaml config2.yaml config3.yaml")
    print("   - Survives SSH disconnections using nohup")
    print("   - Wraps run_seq.sh with better logging")
    
    print("\n3. remove/run_sequential_training.sh - Legacy version")
    print("   - Older implementation in remove/ directory")
    
    print(f"\n✅ All scripts are pushed to git repository!")
    print("🚀 Ready to use for training experiments!")

if __name__ == "__main__":
    main()
