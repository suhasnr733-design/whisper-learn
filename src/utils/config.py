"""
Configuration management
"""
import os
import json
from typing import Any, Dict, Optional
from pathlib import Path

class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Load from environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'WHISPER_MODEL': ('whisper_model', 'medium'),
            'USE_LLM': ('use_llm', 'false'),
            'USE_SUMMARIZATION': ('use_summarization', 'true'),
            'MAX_AUDIO_MINUTES': ('max_audio_minutes', 60),
            'MEMORY_LIMIT_MB': ('memory_limit_mb', 7000)
        }
        
        for env_var, (config_key, default) in env_mappings.items():
            value = os.environ.get(env_var, default)
            
            # Convert boolean strings
            if isinstance(default, bool):
                value = value.lower() == 'true'
            elif isinstance(default, int):
                value = int(value)
            
            self.config[config_key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def save(self, path: str):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @property
    def whisper_model(self) -> str:
        return self.get('whisper_model', 'medium')
    
    @property
    def use_llm(self) -> bool:
        return self.get('use_llm', False)
    
    @property
    def use_summarization(self) -> bool:
        return self.get('use_summarization', True)
    
    @property
    def max_audio_minutes(self) -> int:
        return self.get('max_audio_minutes', 60)
    
    @property
    def memory_limit_mb(self) -> int:
        return self.get('memory_limit_mb', 7000)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration"""
    return Config(config_path)