"""
Memory management utilities for 8GB RAM optimization
"""
import psutil
import gc
import threading
import time
from typing import Dict, Optional
from functools import wraps

class MemoryManager:
    def __init__(self, max_memory_percent: float = 85.0):
        self.max_memory_percent = max_memory_percent
        self.is_monitoring = False
        self.monitor_thread = None
        
    def get_memory_usage(self) -> Dict:
        """Get current memory usage"""
        mem = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent_used': mem.percent,
            'process_gb': process.memory_info().rss / (1024**3),
            'status': 'critical' if mem.percent > 90 else 'warning' if mem.percent > 80 else 'ok'
        }
    
    def is_critical(self) -> bool:
        """Check if memory is critically low"""
        usage = self.get_memory_usage()
        return usage['percent_used'] > self.max_memory_percent
    
    def force_cleanup(self):
        """Force garbage collection and clear caches"""
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        # Clear other caches
        import sys
        if 'transformers' in sys.modules:
            from transformers import logging
            logging.set_verbosity_error()
    
    def start_monitoring(self, interval_seconds: int = 10):
        """Start background memory monitoring"""
        self.is_monitoring = True
        
        def monitor():
            while self.is_monitoring:
                if self.is_critical():
                    print("⚠️ Memory critical, performing cleanup...")
                    self.force_cleanup()
                time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)


def memory_monitor(func):
    """Decorator to monitor memory usage of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        manager = MemoryManager()
        before = manager.get_memory_usage()
        
        result = func(*args, **kwargs)
        
        after = manager.get_memory_usage()
        used = after['process_gb'] - before['process_gb']
        
        print(f"📊 {func.__name__} used {used:.2f}GB RAM")
        
        if manager.is_critical():
            manager.force_cleanup()
        
        return result
    return wrapper


class LRUCache:
    """Least Recently Used cache with size limit"""
    
    def __init__(self, max_size_mb: int = 500):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_order = []
        
    def _get_size(self, value) -> int:
        """Get size of value in bytes"""
        import pickle
        return len(pickle.dumps(value))
    
    def _evict(self):
        """Evict oldest item"""
        if self.access_order:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
    
    def get(self, key):
        """Get item from cache"""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """Put item in cache"""
        size = self._get_size(value)
        
        # Evict until enough space
        total_size = sum(self._get_size(v) for v in self.cache.values())
        while total_size + size > self.max_size_bytes and self.cache:
            self._evict()
            total_size = sum(self._get_size(v) for v in self.cache.values())
        
        if key in self.cache:
            self.access_order.remove(key)
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()