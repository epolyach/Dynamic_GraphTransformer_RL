import re

with open('training_gpu/lib/advanced_trainer_gpu.py', 'r') as f:
    content = f.read()

# Find the baseline initialization section and replace it
pattern = r'(# Check if baseline is actually wanted.*?)\n(.*?use_baseline.*?\n.*?if use_baseline:)(.*?)(\n    else:\n        baseline = None)'

def replace_baseline_init(match):
    comment = match.group(1)
    check = match.group(2)
    old_init = match.group(3)
    else_clause = match.group(4)
    
    new_init = '''
        # Check baseline type from config
        if baseline_type == 'hybrid':
            # Use the new hybrid baseline with critic
            print("[INIT] Creating hybrid baseline (mean → rollout → critic)")
            from .critic_baseline import HybridBaseline
            baseline = HybridBaseline(
                gpu_manager=gpu_manager,
                model=model,
                config=config,
                data_generator=data_generator,
                batch_size=batch_size,
                move_to_gpu_except_distances=move_to_gpu_except_distances,
                logger_print=print
            )
            print("[INIT] Hybrid baseline ready")
        else:
            # Original rollout baseline initialization
            print(f"[INIT] Creating rollout baseline")
            eval_batches = baseline_config.get('eval_batches', 1)
            # Limit eval batches for large batch sizes
            if batch_size >= 2048:
                eval_batches = min(eval_batches, 1)
                logger.info(f"Large batch size ({batch_size}) detected: reducing eval_batches to {eval_batches}")
            elif batch_size >= 1024:
                eval_batches = min(eval_batches, 2)
                logger.info(f"Medium-large batch size ({batch_size}) detected: limiting eval_batches to {eval_batches}")
            
            print(f"[INIT] Building eval dataset: eval_batches={eval_batches}, batch_size={batch_size}")
            eval_dataset = []
            for i in range(eval_batches):
                seed_val = 123456 + i
                batch_data = data_generator(batch_size, seed=seed_val)
                gpu_batch = [move_to_gpu_except_distances(inst, gpu_manager) for inst in batch_data]
                eval_dataset.append(gpu_batch)
            print(f"[INIT] Eval dataset built: {len(eval_dataset)} batches")
            
            from .rollout_baseline_gpu_fixed import RolloutBaselineGPU
            baseline = RolloutBaselineGPU(
                gpu_manager=gpu_manager,
                model=model,
                eval_dataset=eval_dataset,
                config=config,
                logger_print=print
            )
            baseline._update_model(model, 0)'''
    
    return comment + '\n' + check + new_init + else_clause

content = re.sub(pattern, replace_baseline_init, content, flags=re.DOTALL)

# Also need to update baseline update calls to pass instances for critic training
# Find where baseline.update is called
update_pattern = r'(baseline\.update\(model, epoch)(.*?\))'

def add_instances_to_update(match):
    before = match.group(1)
    after = match.group(2)
    # Check if we're in a context where we have instances
    if 'costs' in after:
        # Add instances parameter
        return before + after.replace(')', ', instances=instances)')
    return before + after

content = re.sub(update_pattern, add_instances_to_update, content)

with open('training_gpu/lib/advanced_trainer_gpu.py', 'w') as f:
    f.write(content)

print("✅ Modified trainer to support hybrid baseline with critic")
