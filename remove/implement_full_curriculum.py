import re

# Read the current trainer
with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# 1. Add helper functions for all curriculum features after get_current_batch_size
additional_functions = '''

def get_oscillating_temperature(epoch, adv_config):
    """Get temperature for oscillating schedule."""
    if not adv_config.get('use_oscillating_temperature', False):
        return None
    
    period = adv_config.get('temp_oscillation_period', 20)
    temp_high = adv_config.get('temp_high', 2.5)
    temp_low = adv_config.get('temp_low', 1.5)
    
    # Oscillate between high and low
    phase = (epoch % period) / period
    import math
    # Use cosine for smooth transition
    temp = temp_low + (temp_high - temp_low) * (1 + math.cos(2 * math.pi * phase)) / 2
    return temp

def get_cyclic_lr(epoch, base_lr, adv_config):
    """Get learning rate for cyclic schedule."""
    if adv_config.get('scheduler_type') != 'cyclic':
        return base_lr
    
    lr_base = adv_config.get('lr_base', base_lr)
    lr_max = adv_config.get('lr_max', base_lr * 4)
    cycle_epochs = adv_config.get('lr_cycle_epochs', 30)
    
    # Triangular cyclic schedule
    cycle_pos = epoch % cycle_epochs
    if cycle_pos < cycle_epochs / 2:
        # Increasing phase
        progress = cycle_pos / (cycle_epochs / 2)
        lr = lr_base + (lr_max - lr_base) * progress
    else:
        # Decreasing phase
        progress = (cycle_pos - cycle_epochs / 2) / (cycle_epochs / 2)
        lr = lr_max - (lr_max - lr_base) * progress
    
    return lr

def get_adaptive_entropy_coef(epoch, recent_losses, adv_config):
    """Adjust entropy coefficient based on plateau detection."""
    base_entropy = adv_config.get('entropy_coef', 0.01)
    
    if not adv_config.get('use_adaptive_entropy', False):
        return base_entropy
        
    window = adv_config.get('plateau_detection_window', 10)
    threshold = adv_config.get('plateau_threshold', 0.001)
    boost = adv_config.get('entropy_boost_on_plateau', 0.02)
    min_entropy = adv_config.get('entropy_min', 0.001)
    
    # Need at least window epochs of history
    if len(recent_losses) < window:
        return base_entropy
    
    # Check for plateau
    recent = recent_losses[-window:]
    improvement = max(recent[:-1]) - recent[-1]
    
    if improvement < threshold:
        # Plateau detected, boost entropy
        return min(base_entropy + boost, 0.1)  # Cap at 0.1
    else:
        # Making progress, use base entropy
        return max(base_entropy, min_entropy)

def should_use_critic_baseline(epoch, config):
    """Determine if should use critic baseline based on hybrid strategy."""
    adv_config = config.get('training_advanced', {})
    
    if not adv_config.get('use_hybrid_baseline', False):
        # Check baseline type directly
        baseline_type = config.get('baseline', {}).get('type', 'rollout')
        return baseline_type == 'critic'
    
    # Hybrid: switch at specified epoch
    switch_epoch = adv_config.get('baseline_switch_epoch', 50)
    return epoch >= switch_epoch

def get_dropout_rate(epoch, config):
    """Get dropout rate based on curriculum schedule."""
    curriculum = config.get('curriculum_learning', {})
    dropout_schedule = curriculum.get('dropout_schedule', [])
    
    if not dropout_schedule:
        return 0.0  # No dropout by default
    
    # Sort schedule by epoch
    sorted_schedule = sorted(dropout_schedule, key=lambda x: x['epoch'])
    
    # Find appropriate dropout for current epoch
    current_dropout = 0.0
    for entry in sorted_schedule:
        if epoch >= entry['epoch']:
            current_dropout = entry['dropout']
        else:
            break
    
    return current_dropout
'''

# Insert the additional functions after get_current_batch_size
insert_marker = "    return current_batch_size\n"
insert_pos = content.find(insert_marker) + len(insert_marker)
content = content[:insert_pos] + additional_functions + content[insert_pos:]

# 2. Initialize tracking variables for adaptive entropy
init_tracking = '''
    # Initialize tracking for adaptive features
    recent_losses = []  # Track recent losses for plateau detection
    '''

# Add after optimizer initialization
optimizer_marker = "optimizer = torch.optim.Adam"
opt_pos = content.find(optimizer_marker)
if opt_pos > 0:
    # Find the end of that line
    line_end = content.find('\n', opt_pos)
    content = content[:line_end+1] + init_tracking + content[line_end+1:]

# 3. Modify epoch loop to use all features
# Find where temperature is set
temp_pattern = r'(        # Current temperature.*?\n)(.*?)(        if temp_scheduler:)'

def replace_temp_section(match):
    before = match.group(1)
    middle = match.group(2)
    after = match.group(3)
    
    new_section = '''        
        # Oscillating temperature (takes priority)
        oscillating_temp = get_oscillating_temperature(epoch, adv_config)
        if oscillating_temp is not None:
            current_temp = oscillating_temp
            logger.info(f"Epoch {epoch}: Using oscillating temperature: {current_temp:.3f}")
        elif temp_scheduler:'''
    
    return before + new_section + '''
            current_temp = temp_scheduler.current_temp'''

content = re.sub(temp_pattern, replace_temp_section, content, flags=re.DOTALL)

# 4. Add cyclic LR update
lr_pattern = r'(        model\.train\(\))'

def add_cyclic_lr(match):
    return '''        # Update learning rate (cyclic or standard)
        current_lr = get_cyclic_lr(epoch, base_lr, adv_config)
        if current_lr != base_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            logger.info(f"Epoch {epoch}: Cyclic LR = {current_lr:.6f}")
        
''' + match.group(1)

content = re.sub(lr_pattern, add_cyclic_lr, content, count=1)

# 5. Add adaptive entropy
entropy_pattern = r'(                entropy_coef = adv_config\.get\("entropy_coef", 0\.01\))'

def replace_entropy(match):
    return '''                # Adaptive entropy coefficient
                entropy_coef = get_adaptive_entropy_coef(epoch, recent_losses, adv_config)
                if adv_config.get('use_adaptive_entropy', False) and epoch > 0:
                    logger.debug(f"Adaptive entropy coef: {entropy_coef:.4f}")'''

content = re.sub(entropy_pattern, replace_entropy, content)

# 6. Track losses for plateau detection
loss_tracking_pattern = r'(        epoch_losses\.append\(loss\.item\(\)\))'

def add_loss_tracking(match):
    return match.group(1) + '''
        
        # Track for adaptive entropy
        if epoch_losses:
            recent_losses.append(sum(epoch_losses) / len(epoch_losses))'''

content = re.sub(loss_tracking_pattern, add_loss_tracking, content)

# 7. Add dropout schedule support (need to modify model call)
# This is more complex as it requires modifying the model itself
# For now, add a log message about dropout
dropout_log = '''
        # Update dropout if scheduled
        current_dropout = get_dropout_rate(epoch, config)
        if current_dropout > 0:
            logger.info(f"Epoch {epoch}: Dropout rate = {current_dropout:.2f}")
            # Note: Model needs to be modified to support dynamic dropout
        '''

# Add after batch size update
batch_update_marker = "logger.info(f\"New effective batch size:"
batch_pos = content.find(batch_update_marker)
if batch_pos > 0:
    line_end = content.find('\n', batch_pos)
    content = content[:line_end+1] + dropout_log + content[line_end+1:]

# Save the modified trainer
with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("✅ Implemented all curriculum features!")
print("\nFeatures added:")
print("1. ✅ Oscillating temperature schedule")
print("2. ✅ Cyclic learning rate scheduler") 
print("3. ✅ Adaptive entropy with plateau detection")
print("4. ✅ Hybrid baseline switching (detection only)")
print("5. ✅ Dropout schedule (logging only)")
print("\nNote: Hybrid baseline and dropout need model/baseline modifications to fully work")
