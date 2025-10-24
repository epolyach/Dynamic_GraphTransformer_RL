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

# Test the function
test_config = {
    'batch_size': 256,
    'curriculum': {
        'enabled': True,
        'batch_size_schedule': [
            {'epoch': 0, 'batch_size': 256},
            {'epoch': 25, 'batch_size': 512},
            {'epoch': 50, 'batch_size': 1024},
            {'epoch': 75, 'batch_size': 2048}
        ]
    }
}

# Test different epochs
for epoch in [0, 10, 25, 30, 50, 60, 75, 100]:
    batch_size = get_current_batch_size(epoch, test_config)
    print(f"Epoch {epoch}: batch_size = {batch_size}")
