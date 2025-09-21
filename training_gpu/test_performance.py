"""
Simple performance test for GPU trainer optimizations
"""

import time
import torch
import logging
from src.data_generators.cvrp_generator_cpu import CVRPDataGeneratorCPU
from training_gpu.lib.advanced_trainer_gpu import train_model_gpu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_performance():
    # Simple config for testing
    config = {
        'batch_size': 32,
        'batch': 5,  # Just 5 batches for quick test
        'epochs': 2,  # Just 2 epochs
        'val_size': 16,
        'validation_frequency': 1,
        'show_log_every': 1,
        'use_amp': True,
        'use_geometric_mean': False,
        'optimizer': {
            'type': 'Adam',
            'args': {'lr': 1e-4}
        }
    }
    
    # Create dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(100, 10)
        
        def forward(self, instances, **kwargs):
            batch_size = len(instances)
            # Dummy routes
            routes = [[0, 1, 2, 0] for _ in range(batch_size)]
            # Dummy log probs
            log_probs = [torch.randn(3, device='cuda') for _ in range(batch_size)]
            # Dummy entropy
            entropy = torch.randn(batch_size, device='cuda')
            return routes, log_probs, entropy
    
    model = DummyModel()
    data_generator = CVRPDataGeneratorCPU(n_customers=10)
    
    logger.info("Starting performance test...")
    start_time = time.time()
    
    try:
        history = train_model_gpu(
            model=model,
            data_generator=data_generator,
            baseline=None,
            config=config,
            logger=logger
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Test completed in {total_time:.2f} seconds")
        logger.info(f"Average time per epoch: {total_time/config['epochs']:.2f} seconds")
        
        if 'time' in history:
            logger.info(f"Epoch times: {history['time']}")
            avg_epoch_time = sum(history['time']) / len(history['time'])
            logger.info(f"Average epoch time from history: {avg_epoch_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_performance()
