# Let's reduce eval_batches to 1 for faster initialization
import fileinput
import sys

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# Change default eval_batches from 2 to 1
content = content.replace(
    "eval_batches = baseline_config.get('eval_batches', 2)",
    "eval_batches = baseline_config.get('eval_batches', 1)"
)

# For initialization, use even fewer batches
content = content.replace(
    "for i in range(eval_batches):",
    "# Use only 1 batch for initialization to speed up\n        init_eval_batches = min(1, eval_batches)\n        for i in range(init_eval_batches):"
)

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("✅ Reduced eval_batches for faster initialization")
