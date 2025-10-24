# CRITICAL BUG FOUND IN PREFETCH VERSION

## The Problem

When I wrapped the training loop with the prefetch context manager, I accidentally created TWO loops:

```python
with prefetcher_class(...) as prefetcher:
    
    for _internal_batch_idx in range(n_batches):
        batch_idx, instances = prefetcher.get_batch()
        if batch_idx is None:
            break
    # ← This is where the OLD code starts!
    
    # OLD CODE STILL HERE (unreachable!):
    batch_seed = epoch * n_batches * batch_size + batch_idx * batch_size 
    instances = data_generator(batch_size, epoch=epoch, seed=batch_seed)
    # ... rest of training code ...
```

## What Happens:
1. Prefetch loop runs and gets all batches
2. Loop exits (break when batch_idx is None)
3. **OLD training code is unreachable!**
4. Training never actually happens!
5. Baseline never updates!

## The Fix

Need to DELETE the old code and keep training logic INSIDE the prefetch loop:

```python
with prefetcher_class(...) as prefetcher:
    
    for _internal_batch_idx in range(n_batches):
        # Get batch from prefetcher
        batch_idx, instances = prefetcher.get_batch()
        if batch_idx is None:
            break
        
        # Move to GPU
        instances = [move_to_gpu_except_distances(...)]
        
        # Forward pass
        routes, log_probs, entropy = model(...)
        
        # Compute loss
        loss.backward()
        
        # ... ALL TRAINING CODE HERE ...
        
# OUTSIDE the prefetch loop:
# Baseline update (once per epoch)
if baseline is not None:
    baseline.epoch_callback(model, epoch)
```

## Lines to Fix

In `training_gpu_prefetch/lib/advanced_trainer_gpu.py`:

1. **Line 690-695**: DELETE duplicate data generation code
2. **Lines 696-850**: INDENT all training code to be INSIDE the for loop
3. **Line 882+**: Keep baseline update OUTSIDE (after with block)

