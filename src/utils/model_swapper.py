"""
Smart model swapping to manage memory
"""
import gc
import time
from typing import Dict, Any, Callable
import psutil

class SmartModelLoader:
    def __init__(self, max_memory_mb: int = 7000):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.loaded_models: Dict[str, Dict] = {}
        self.model_loaders: Dict[str, Callable] = {}
        
    def register_model(self, name: str, loader: Callable, size_mb: int, priority: int = 1):
        """Register a model loader"""
        self.model_loaders[name] = {
            'loader': loader,
            'size_mb': size_mb,
            'priority': priority
        }
    
    def get_available_memory(self) -> int:
        """Get available memory in bytes"""
        return psutil.virtual_memory().available
    
    def load_model(self, name: str) -> Any:
        """Load a model, unloading others if needed"""
        if name in self.loaded_models:
            self.loaded_models[name]['last_used'] = time.time()
            return self.loaded_models[name]['model']
        
        if name not in self.model_loaders:
            raise ValueError(f"Model {name} not registered")
        
        model_info = self.model_loaders[name]
        
        # Unload low priority models if needed
        while self.get_available_memory() < model_info['size_mb'] * 1024 * 1024 * 1.2:
            if not self._unload_lowest_priority():
                break
        
        # Load the model
        print(f"Loading model: {name}")
        model = model_info['loader']()
        
        self.loaded_models[name] = {
            'model': model,
            'size_mb': model_info['size_mb'],
            'priority': model_info['priority'],
            'last_used': time.time()
        }
        
        return model
    
    def _unload_lowest_priority(self) -> bool:
        """Unload the lowest priority model"""
        if not self.loaded_models:
            return False
        
        lowest = min(self.loaded_models.items(), 
                    key=lambda x: (x[1]['priority'], x[1]['last_used']))
        
        print(f"Unloading model: {lowest[0]} to free memory")
        del self.loaded_models[lowest[0]]['model']
        del self.loaded_models[lowest[0]]
        gc.collect()
        
        return True
    
    def unload_model(self, name: str):
        """Unload specific model"""
        if name in self.loaded_models:
            del self.loaded_models[name]['model']
            del self.loaded_models[name]
            gc.collect()
    
    def unload_all(self):
        """Unload all models"""
        for name in list(self.loaded_models.keys()):
            self.unload_model(name)