import re

def modify_trainer_for_curriculum():
    with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
        content = f.read()
    
    # Add the curriculum batch size function at the top (after imports)
    curriculum_function = '''
def get_current_batch_size(epoch, train_config):
    """
    Get the current batch size based on epoch and curriculum schedule.
    
    Args:
        epoch: Current training epoch
        train_config: Training configuration dict
        
    Returns:
        int: Current batch size for this epoch
    """
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

'''
    
    # Find the location after imports to add the function
    # Look for the first function definition
    first_function_match = re.search(r'^def ', content, re.MULTILINE)
    if first_function_match:
        insert_pos = first_function_match.start()
        content = content[:insert_pos] + curriculum_function + '\n' + content[insert_pos:]
    
    # Modify the initial batch_size assignment to support curriculum
    old_batch_size_line = "    batch_size = train_config.get('batch_size', 32)"
    new_batch_size_line = '''    # Initialize batch size (may change with curriculum learning)
    initial_batch_size = train_config.get('batch_size', 32)
    batch_size = get_current_batch_size(0, train_config)  # Start with epoch 0 batch size
    if batch_size != initial_batch_size:
        logger.info(f"Curriculum learning: Using batch_size={batch_size} instead of configured {initial_batch_size} for epoch 0")'''
    
    content = content.replace(old_batch_size_line, new_batch_size_line)
    
    # Now we need to modify the training loop to handle dynamic batch size changes
    # Find the epoch loop and modify it
    epoch_loop_pattern = r'(for epoch in range\(n_epochs\):.*?)(\n        # Training loop.*?)(\n        for batch_idx in range\(n_batches\):)'
    
    def replace_epoch_loop(match):
        before = match.group(1)
        middle = match.group(2)
        after = match.group(3)
        
        # Add dynamic batch size handling at the start of each epoch
        batch_size_update = '''
        
        # Check if batch size should change for this epoch (curriculum learning)
        new_batch_size = get_current_batch_size(epoch, train_config)
        if new_batch_size != batch_size:
            old_batch_size = batch_size
            batch_size = new_batch_size
            effective_batch_size = batch_size * gradient_accumulation_steps
            logger.info(f"Curriculum learning: Batch size changed from {old_batch_size} to {batch_size} at epoch {epoch}")
            logger.info(f"New effective batch size: {effective_batch_size}")
            
            # Update n_batches calculation with new batch size
            n_batches = max(1, total_samples // batch_size)'''
        
        return before + batch_size_update + middle + after
    
    content = re.sub(epoch_loop_pattern, replace_epoch_loop, content, flags=re.DOTALL)
    
    # Also fix the eval_batches issue - ensure it uses the config value
    eval_batches_pattern = r'eval_batches = baseline_config\.get\(\'eval_batches\', 2\)'
    eval_batches_replacement = '''eval_batches = baseline_config.get('eval_batches', 2)
        # For large batch sizes, use fewer eval batches to prevent hangs
        if batch_size >= 2048:
            eval_batches = min(eval_batches, 1)
            logger.info(f"Large batch size ({batch_size}) detected: reducing eval_batches to {eval_batches}")
        elif batch_size >= 1024:
            eval_batches = min(eval_batches, 2)
            logger.info(f"Medium-large batch size ({batch_size}) detected: limiting eval_batches to {eval_batches}")'''
    
    content = re.sub(eval_batches_pattern, eval_batches_replacement, content)
    
    return content

# Apply the modifications
modified_content = modify_trainer_for_curriculum()

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(modified_content)

print("Trainer modified successfully for curriculum learning!")
