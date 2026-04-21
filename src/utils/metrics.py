"""
Performance metrics collection
"""
import time
from typing import Dict, List
from collections import defaultdict
import threading

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.lock = threading.Lock()
        
    def record(self, name: str, value: float):
        """Record a metric value"""
        with self.lock:
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time()
            })
            
            # Keep only last 1000 entries
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    def record_time(self, name: str):
        """Decorator to record execution time"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start
                self.record(f"{name}_duration", duration)
                return result
            return wrapper
        return decorator
    
    def get_average(self, name: str, last_n: int = 100) -> float:
        """Get average of last N values"""
        values = [m['value'] for m in self.metrics.get(name, [])[-last_n:]]
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def get_stats(self, name: str) -> Dict:
        """Get statistics for a metric"""
        values = [m['value'] for m in self.metrics.get(name, [])]
        
        if not values:
            return {'count': 0}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else None
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all metrics"""
        return {name: self.get_stats(name) for name in self.metrics}
    
    def clear(self):
        """Clear all metrics"""
        with self.lock:
            self.metrics.clear()
    
    def report(self) -> str:
        """Generate metrics report"""
        stats = self.get_all_stats()
        lines = ["=== Metrics Report ==="]
        
        for name, stat in stats.items():
            if stat.get('count', 0) > 0:
                lines.append(f"{name}: avg={stat['avg']:.3f}, count={stat['count']}")
        
        return '\n'.join(lines)


# Global metrics collector
metrics = MetricsCollector()