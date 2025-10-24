import re

# Restore backup first
import shutil
shutil.copy('training_gpu/lib/advanced_trainer_gpu.py.backup_before_curriculum', 
            'training_gpu/lib/advanced_trainer_gpu.py')

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# 1. Add the curriculum function after the logger definition
curriculum_function = """

def get_current_batch_size(epoch, train_config):
    \"\"\"
    Get the current batch size based on epoch and curriculum schedule.
    
    Args:
        epoch: Current training epoch
        train_config: Training configuration dict
        
    Returns:
        int: Current batch size for this epoch
    \"\"\"
    # Check if curriculum is enabled
    curriculum = train_config.get('curriculum', {})
    if not curriculum.get('enabled', False):
        return train_config.get('batch_size', 32)
    
    # Get batch size schedule
    batch_size_schedule = curriculum.get('batch_size_schedule', [])
    if not batch_size_schedule:
        return train_config.get('batch_size', 32)
    
    # Sort schedule by epoch (in case it's not sorted)
    sorted_schedule = sorted(batch_size_schedule, key=lambda x: x['epoch'])
    
    # Find the appropriate batch size for current epoch
    current_batch_size = train_config.get('batch_size', 32)  # fallback
    
    for schedule_entry in sorted_schedule:
        if epoch >= schedule_entry['epoch']:
            current_batch_size = schedule_entry['batch_size']
        else:
            break
    
    return current_batch_size
"""

# Insert after logger definition
logger_pattern = r'(logger = logging\.getLogger\(__name__\))'
content = re.sub(logger_pattern, r'\1' + curriculum_function, content)

# 2. Modify the initial batch_size assignment
old_batch_size = r'    batch_size = train_config\.get\(\'batch_size\', 32\)'
new_batch_size = '''    # Initialize batch size (support curriculum learning)
    initial_batch_size = train_config.get('batch_size', 32)
    batch_size = get_current_batch_size(0, train_config)  # Start with epoch 0 batch size
    if batch_size != initial_batch_size:
        logger.info(f"Curriculum learning: Using batch_size={batch_size} instead of configured {initial_batch_size} for epoch 0")'''

content = re.sub(old_batch_size, new_batch_size, content)

# 3. Add batch size update at the beginning of each epoch
# Find the epoch loop and add curriculum batch size updating
epoch_start_pattern = r'(    for epoch in range\(.*?\):.*?epoch_start = time\.time\(\))'
batch_size_update = '''
        
        # Check if batch size should change for this epoch (curriculum learning)
        new_batch_size = get_current_batch_size(epoch, train_config)
        if new_batch_size != batch_size:
            old_batch_size = batch_size
            batch_size = new_batch_size
            effective_batch_size = batch_size * gradient_accumulation_steps
            logger.info(f"Curriculum learning: Batch size changed from {old_batch_size} to {batch_size} at epoch {epoch}")
            logger.info(f"New effective batch size: {effective_batch_size}")'''

content = re.sub(epoch_start_pattern, r'\1' + batch_size_update, content, flags=re.DOTALL)

# 4. Fix eval_batches configuration to respect config and handle large batch sizes
eval_batches_pattern = r'        eval_batches = baseline_config\.get\(\'eval_batches\', 2\)'
eval_batches_fix = '''        eval_batches = baseline_config.get('eval_batches', 2)
        # For large batch sizes, use fewer eval batches to prevent hangs
        if batch_size >= 2048:
            eval_batches = min(eval_batches, 1)
            logger.info(f"Large batch size ({batch_size}) detected: reducing eval_batches to {eval_batches}")
        elif batch_size >= 1024:
            eval_batches = min(eval_batches, 2)
            logger.info(f"Medium-large batch size ({batch_size}) detected: limiting eval_batches to {eval_batches}")'''

content = re.sub(eval_batches_pattern, eval_batches_fix, content)

# Write the modified content back
with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("Successfully applied curriculum learning modifications!")

# Verify the modifications
print("\nVerifying modifications:")
with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    lines = f.readlines()
    
for i, line in enumerate(lines):
    if 'get_current_batch_size' in line and 'def' in line:
        print(f"✓ Curriculum function added at line {i+1}")
    if 'Curriculum learning: Using batch_size' in line:
        print(f"✓ Initial batch size curriculum check at line {i+1}")
    if 'Curriculum learning: Batch size changed' in line:
        print(f"✓ Dynamic batch size update at line {i+1}")
    if 'Large batch size' in line and 'detected' in line:
        print(f"✓ Eval batches fix for large batch sizes at line {i+1}")
