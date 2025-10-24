"""
Batch Prefetcher - Continuous Data Generation Pipeline
Overlaps CPU data generation with GPU training for maximum utilization
"""

import threading
import queue
import time
from typing import Callable, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BatchPrefetcher:
    """
    Continuous batch prefetching system that generates data on CPU
    while GPU processes previous batches.
    
    Architecture:
    - Background thread continuously generates batches
    - Queue buffers 2-4 batches ahead
    - GPU never waits for data generation
    
    Expected improvement: 4-6x GPU utilization increase
    """
    
    def __init__(
        self,
        data_generator: Callable,
        batch_size: int,
        num_batches: int,
        epoch: int,
        queue_size: int = 3,
        seed_base: Optional[int] = None
    ):
        """
        Initialize the prefetcher.
        
        Args:
            data_generator: Function to generate data batches
            batch_size: Size of each batch
            num_batches: Total number of batches to generate
            epoch: Current training epoch
            queue_size: Number of batches to buffer (2-4 recommended)
            seed_base: Base seed for reproducibility
        """
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.epoch = epoch
        self.seed_base = seed_base if seed_base is not None else 0
        
        # Queue for buffering generated batches
        self.queue = queue.Queue(maxsize=queue_size)
        
        # Control flags
        self.stop_flag = threading.Event()
        self.error = None
        
        # Statistics
        self.batches_generated = 0
        self.generation_times = []
        self.queue_wait_times = []
        
        # Background thread
        self.worker_thread = None
        
        logger.info(f"BatchPrefetcher initialized: {num_batches} batches, "
                   f"queue_size={queue_size}, batch_size={batch_size}")
    
    def _worker(self):
        """Background worker thread that continuously generates batches."""
        try:
            for batch_idx in range(self.num_batches):
                if self.stop_flag.is_set():
                    logger.info("Prefetcher stopped by flag")
                    break
                
                # Generate batch with proper seeding
                gen_start = time.time()
                batch_seed = self.seed_base + batch_idx * self.batch_size
                
                instances = self.data_generator(
                    self.batch_size,
                    epoch=self.epoch,
                    seed=batch_seed
                )
                
                gen_time = time.time() - gen_start
                self.generation_times.append(gen_time)
                
                # Put batch in queue (blocks if queue is full)
                queue_start = time.time()
                self.queue.put((batch_idx, instances), timeout=60)
                queue_wait = time.time() - queue_start
                
                self.queue_wait_times.append(queue_wait)
                self.batches_generated += 1
                
                # Log progress periodically
                if (batch_idx + 1) % 100 == 0:
                    avg_gen_time = sum(self.generation_times[-100:]) / min(100, len(self.generation_times))
                    logger.debug(f"Prefetcher: {batch_idx + 1}/{self.num_batches} batches, "
                               f"avg_gen_time={avg_gen_time:.3f}s, queue_size={self.queue.qsize()}")
            
            # Signal completion
            self.queue.put((None, None))
            logger.info(f"Prefetcher completed: {self.batches_generated} batches generated")
            
        except Exception as e:
            logger.error(f"Prefetcher worker error: {e}")
            self.error = e
            self.queue.put((None, None))  # Unblock waiting consumer
    
    def start(self):
        """Start the background prefetching thread."""
        if self.worker_thread is not None:
            raise RuntimeError("Prefetcher already started")
        
        self.stop_flag.clear()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("Prefetcher started")
    
    def get_batch(self, timeout: float = 120.0) -> Optional[tuple]:
        """
        Get the next prefetched batch.
        
        Args:
            timeout: Maximum time to wait for batch (seconds)
            
        Returns:
            Tuple of (batch_idx, instances) or (None, None) if done
            
        Raises:
            RuntimeError: If prefetcher encountered an error
            queue.Empty: If timeout exceeded
        """
        if self.error:
            raise RuntimeError(f"Prefetcher failed: {self.error}")
        
        try:
            batch_idx, instances = self.queue.get(timeout=timeout)
            
            # Check for completion signal
            if batch_idx is None:
                return None, None
            
            return batch_idx, instances
            
        except queue.Empty:
            raise queue.Empty(f"Timeout waiting for batch (timeout={timeout}s)")
    
    def stop(self):
        """Stop the prefetcher and clean up."""
        logger.info("Stopping prefetcher...")
        self.stop_flag.set()
        
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                logger.warning("Prefetcher thread did not stop cleanly")
        
        # Clear remaining items in queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Prefetcher stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prefetcher statistics."""
        return {
            'batches_generated': self.batches_generated,
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0,
            'avg_queue_wait_time': sum(self.queue_wait_times) / len(self.queue_wait_times) if self.queue_wait_times else 0,
            'current_queue_size': self.queue.qsize(),
            'max_queue_size': self.queue.maxsize,
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class DummyPrefetcher:
    """
    Fallback prefetcher that generates data synchronously (no prefetching).
    Used when prefetching is disabled.
    """
    
    def __init__(self, data_generator, batch_size, num_batches, epoch, seed_base=None, **kwargs):
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.epoch = epoch
        self.seed_base = seed_base if seed_base is not None else 0
        self.current_batch = 0
    
    def start(self):
        """No-op for dummy prefetcher."""
        pass
    
    def get_batch(self, timeout=None):
        """Generate batch synchronously."""
        if self.current_batch >= self.num_batches:
            return None, None
        
        batch_seed = self.seed_base + self.current_batch * self.batch_size
        instances = self.data_generator(self.batch_size, epoch=self.epoch, seed=batch_seed)
        
        batch_idx = self.current_batch
        self.current_batch += 1
        
        return batch_idx, instances
    
    def stop(self):
        """No-op for dummy prefetcher."""
        pass
    
    def get_stats(self):
        """Return minimal stats."""
        return {'batches_generated': self.current_batch}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

