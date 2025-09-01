import numpy as np
import torch

def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
