# Add a trace print after EVERY line in HybridBaseline __init__
with open('training_gpu/lib/critic_baseline.py', 'r') as f:
    content = f.read()

# Find the HybridBaseline __init__ method and trace it
import re

# Find the __init__ method of HybridBaseline
pattern = r'(class HybridBaseline:.*?def __init__.*?\n)(.*?)((?=\n    def )|$)'

def add_traces(match):
    class_def = match.group(1)
    init_body = match.group(2)
    next_method = match.group(3)
    
    # Split init_body into lines
    lines = init_body.split('\n')
    traced_lines = []
    
    for i, line in enumerate(lines):
        traced_lines.append(line)
        # Add trace after each assignment or function call
        if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('print') and not line.strip().startswith('sys.'):
            indent = len(line) - len(line.lstrip())
            if '=' in line or line.strip().startswith('logger_print'):
                trace = ' ' * indent + f'print("[TRACE] Line executed: {line.strip()[:50]}"); sys.stdout.flush()\n'
                traced_lines.append(trace)
    
    return class_def + '\n'.join(traced_lines) + next_method

content = re.sub(pattern, add_traces, content, flags=re.DOTALL)

with open('training_gpu/lib/critic_baseline.py', 'w') as f:
    f.write(content)

print("✅ Added execution traces to HybridBaseline __init__")
