import sys
import time
sys.path.insert(0, '.')

import torch
from src.generator.generator import create_data_generator
from src.utils.config import load_config
from src.models.model_factory import ModelFactory

# Load config
config = load_config('configs/tiny.yaml')
config['experiment']['device'] = 'cpu'  # Force CPU

# Create model and data generator
model = ModelFactory.create_model('GT+RL', config)
data_generator = create_data_generator(config)

# Profile data generation
print("Profiling data generation...")
start = time.time()
for i in range(5):
    batch = data_generator(32, seed=i)
data_gen_time = time.time() - start
print(f"Data generation: {data_gen_time:.3f}s for 5 batches (avg: {data_gen_time/5:.3f}s per batch)")

# Profile forward pass
print("\nProfiling forward pass...")
batch = data_generator(32, seed=0)
start = time.time()
with torch.no_grad():
    routes, logp, ent = model(
        batch,
        max_steps=len(batch[0]['coords']) * 2,
        temperature=1.0,
        greedy=False,
        config=config
    )
forward_time = time.time() - start
print(f"Forward pass: {forward_time:.3f}s for batch of 32")

# Profile backward pass
print("\nProfiling backward pass...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
start = time.time()
routes, logp, ent = model(
    batch,
    max_steps=len(batch[0]['coords']) * 2,
    temperature=1.0,
    greedy=False,
    config=config
)
loss = -logp.mean()
loss.backward()
optimizer.step()
backward_time = time.time() - start
print(f"Backward pass: {backward_time:.3f}s for batch of 32")

print(f"\nTotal time per iteration: {data_gen_time/5 + forward_time + backward_time:.3f}s")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
