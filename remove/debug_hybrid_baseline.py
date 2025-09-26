# Add comprehensive debugging to the hybrid baseline initialization
with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    lines = f.readlines()

# Find the hybrid baseline initialization and add debug prints
for i, line in enumerate(lines):
    if 'if baseline_type == \'hybrid\':' in line:
        # Add debug print before the import
        lines.insert(i+1, '            print("[DEBUG] Hybrid baseline selected, importing HybridBaseline class...")\n')
        lines.insert(i+2, '            import sys; sys.stdout.flush()\n')
        
        # Find the HybridBaseline instantiation
        for j in range(i+3, min(i+20, len(lines))):
            if 'from .critic_baseline import HybridBaseline' in lines[j]:
                lines.insert(j+1, '            print("[DEBUG] Import successful, creating HybridBaseline instance...")\n')
                lines.insert(j+2, '            sys.stdout.flush()\n')
            elif 'baseline = HybridBaseline(' in lines[j]:
                lines.insert(j, '            print("[DEBUG] About to instantiate HybridBaseline with:")\n')
                lines.insert(j+1, f'            print(f"[DEBUG]   batch_size={{batch_size}}")\n')
                lines.insert(j+2, '            sys.stdout.flush()\n')
                break
        break

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.writelines(lines)

print("✅ Added debug logging to hybrid baseline initialization")

# Also add debug logging to the HybridBaseline __init__
with open('training_gpu/lib/critic_baseline.py', 'r') as f:
    content = f.read()

# Add debug print at the very start of HybridBaseline.__init__
init_start = 'def __init__(self, gpu_manager, model, config, data_generator, batch_size,'
init_debug = '''def __init__(self, gpu_manager, model, config, data_generator, batch_size,
                 move_to_gpu_except_distances, logger_print=print):
        print("[DEBUG HybridBaseline] Entering __init__")
        import sys; sys.stdout.flush()'''

content = content.replace(
    init_start + '\n                 move_to_gpu_except_distances, logger_print=print):',
    init_debug
)

# Add more debug prints throughout __init__
content = content.replace(
    'self.gpu_manager = gpu_manager',
    'print("[DEBUG HybridBaseline] Setting gpu_manager")\n        import sys; sys.stdout.flush()\n        self.gpu_manager = gpu_manager'
)

content = content.replace(
    'self.config = config',
    'print("[DEBUG HybridBaseline] Setting config")\n        sys.stdout.flush()\n        self.config = config'
)

content = content.replace(
    'baseline_config = config.get(\'baseline\', {})',
    'print("[DEBUG HybridBaseline] Getting baseline config")\n        sys.stdout.flush()\n        baseline_config = config.get(\'baseline\', {})'
)

with open('training_gpu/lib/critic_baseline.py', 'w') as f:
    f.write(content)

print("✅ Added debug logging to HybridBaseline class")
