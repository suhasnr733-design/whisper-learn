"""
Intelligent caching for transcriptions
"""
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict
import pickle
from datetime import datetime, timedelta

class TranscriptionCache:
    def __init__(self, cache_dir: str = "/tmp/whisper_cache", max_age_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(days=max_age_days)
        
    def _get_audio_hash(self, audio_path: str) -> str:
        """Generate hash from audio file"""
        hasher = hashlib.md5()
        with open(audio_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_cached(self, audio_path: str, model_size: str, **params) -> Optional[Dict]:
        """Retrieve cached transcription"""
        audio_hash = self._get_audio_hash(audio_path)
        param_str = json.dumps({**params, 'model': model_size}, sort_keys=True)
        cache_key = hashlib.md5(f"{audio_hash}_{param_str}".encode()).hexdigest()
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            # Check age
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time < self.max_age:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        return None
    
    def save_cache(self, audio_path: str, model_size: str, result: Dict, **params):
        """Save transcription to cache"""
        audio_hash = self._get_audio_hash(audio_path)
        param_str = json.dumps({**params, 'model': model_size}, sort_keys=True)
        cache_key = hashlib.md5(f"{audio_hash}_{param_str}".encode()).hexdigest()
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        # Clean old cache files
        self._clean_cache()
    
    def _clean_cache(self):
        """Remove old cache files"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time > self.max_age:
                cache_file.unlink()
    
    def clear_cache(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


class MemoryCache:
    """In-memory cache for frequently accessed data"""
    
    def __init__(self, max_size_mb: int = 500):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        
    def get(self, key: str):
        """Get item from cache"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, key: str, value):
        """Set item in cache with size tracking"""
        size = len(pickle.dumps(value))
        
        # Evict old items if needed
        while self._get_total_size() + size > self.max_size_bytes and self.cache:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def _get_total_size(self) -> int:
        """Calculate total cache size"""
        total = 0
        for value in self.cache.values():
            total += len(pickle.dumps(value))
        return total
    
    def _evict_oldest(self):
        """Remove least recently accessed item"""
        if self.access_times:
            oldest = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest]
            del self.access_times[oldest]
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()