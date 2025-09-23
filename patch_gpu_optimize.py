from pathlib import Path
p = Path('training_gpu/lib/advanced_trainer_gpu.py')
s = p.read_text()

# Add speed knobs
target = 'logger = logging.getLogger(__name__)'
addition = '''logger = logging.getLogger(__name__)

# Speed knobs for Ampere+ GPUs (e.g., A6000):
# - TF32 can accelerate large matmul-heavy models with minimal accuracy loss
# - High matmul precision hints PyTorch to use TF32 where possible
try:
    import torch.backends.cuda
    torch.backends.cuda.matmul.allow_tf32 = True  # enable TF32 matmul
    # cudnn TF32 mainly affects convs; harmless to enable
    import torch.backends.cudnn as cudnn
    cudnn.allow_tf32 = True
except Exception:
    pass

try:
    # PyTorch 2.0+: set float32 matmul precision policy
    torch.set_float32_matmul_precision('high')
except Exception:
    pass'''

s = s.replace(target, addition)

# Add torch.compile
needle = 'model = gpu_manager.to_device(model)'
addition = '''model = gpu_manager.to_device(model)

    # Optional: torch.compile for additional speedup (PyTorch 2.0+)
    # Controlled by config.training_advanced.compile
    try:
        compile_cfg = config.get('training_advanced', {}).get('compile', {})
        if compile_cfg and compile_cfg.get('enabled', False):
            compile_mode = compile_cfg.get('mode', 'default')
            # dynamic shapes in decoding can be tricky; default mode is safest
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
            print(f"[INIT] Model compiled with torch.compile(mode={compile_mode})")
    except Exception as e:
        print(f"[INIT] torch.compile skipped: {e}")'''

if needle in s:
    s = s.replace(needle, addition)
    
Path('training_gpu/lib/advanced_trainer_gpu.py').write_text(s)
print('Patched training_gpu/lib/advanced_trainer_gpu.py with TF32 and torch.compile support')
