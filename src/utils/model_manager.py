# src/utils/model_manager.py
"""
Centralized model management with lazy loading
"""
import os
import gc
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import psutil

class ModelManager:
    """Manages all models with lazy loading and memory optimization"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.loaded_models = {}
        self.model_configs = self._load_configs()
        
    def _load_configs(self) -> Dict:
        """Load model configurations"""
        return {
            'whisper': {
                'sizes': {
                    'tiny': {'path': 'whisper/tiny.pt', 'ram_mb': 75, 'recommended': False},
                    'base': {'path': 'whisper/base.pt', 'ram_mb': 145, 'recommended': False},
                    'small': {'path': 'whisper/small.pt', 'ram_mb': 488, 'recommended': True},
                    'medium': {'path': 'whisper/medium.pt', 'ram_mb': 1500, 'recommended': True},
                    'large': {'path': 'whisper/large.pt', 'ram_mb': 2900, 'recommended': False}
                }
            },
            'embeddings': {
                'all-MiniLM-L6-v2': {'ram_mb': 80, 'recommended': True},
                'all-mpnet-base-v2': {'ram_mb': 420, 'recommended': False}
            },
            'summarization': {
                'bart-large-cnn': {'ram_mb': 1600, 'recommended': True},
                'pegasus-xsum': {'ram_mb': 1200, 'recommended': False}
            },
            'llm': {
                'phi-2': {'ram_mb': 1600, 'quantized_mb': 900, 'recommended': True},
                'tinyllama': {'ram_mb': 700, 'quantized_mb': 400, 'recommended': True}
            }
        }
    
    def get_available_ram_mb(self) -> int:
        """Get available RAM in MB"""
        return psutil.virtual_memory().available / (1024 * 1024)
    
    def can_load_model(self, model_type: str, model_name: str, quantized: bool = False) -> bool:
        """Check if model can be loaded with available RAM"""
        config = self.model_configs.get(model_type, {}).get(model_name, {})
        
        if quantized and 'quantized_mb' in config:
            required_mb = config['quantized_mb']
        else:
            required_mb = config.get('ram_mb', 1000)
        
        available = self.get_available_ram_mb()
        
        # Leave 500MB buffer
        return available > (required_mb + 500)
    
    def get_whisper(self, model_size: str = "medium"):
        """Get Whisper model (lazy load)"""
        model_key = f"whisper_{model_size}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        if not self.can_load_model('whisper', model_size):
            # Fallback to smaller model
            fallback = "small" if model_size == "medium" else "base"
            print(f"⚠️ Not enough RAM for {model_size}, using {fallback}")
            model_size = fallback
        
        import whisper
        print(f"Loading Whisper {model_size}...")
        model = whisper.load_model(model_size)
        
        self.loaded_models[model_key] = model
        return model
    
    def get_embeddings(self, model_name: str = "all-MiniLM-L6-v2"):
        """Get embeddings model"""
        model_key = f"embeddings_{model_name}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        from sentence_transformers import SentenceTransformer
        print(f"Loading embeddings model {model_name}...")
        model = SentenceTransformer(model_name)
        
        self.loaded_models[model_key] = model
        return model
    
    def get_summarizer(self, model_name: str = "facebook/bart-large-cnn"):
        """Get summarization pipeline"""
        model_key = f"summarizer_{model_name}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        if not self.can_load_model('summarization', 'bart-large-cnn'):
            print("⚠️ Not enough RAM for BART, using extractive summarization")
            return None
        
        from transformers import pipeline
        print(f"Loading summarization model...")
        
        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", model=model_name, device=device)
        
        self.loaded_models[model_key] = summarizer
        return summarizer
    
    def get_llm(self, model_name: str = "phi-2", quantized: bool = True):
        """Get LLM for Q&A"""
        model_key = f"llm_{model_name}_{'quantized' if quantized else 'full'}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        if not self.can_load_model('llm', model_name, quantized):
            print(f"⚠️ Not enough RAM for LLM, Q&A will use keyword matching")
            return None
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        # Map model names
        model_paths = {
            'phi-2': 'microsoft/phi-2',
            'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        }
        
        model_path = model_paths.get(model_name, model_name)
        print(f"Loading LLM {model_name}...")
        
        if quantized:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.loaded_models[model_key] = (model, tokenizer)
        return model, tokenizer
    
    def unload_model(self, model_key: str):
        """Unload a specific model to free memory"""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Unloaded {model_key}")
    
    def unload_all(self):
        """Unload all models"""
        for key in list(self.loaded_models.keys()):
            self.unload_model(key)
    
    def get_status(self) -> Dict:
        """Get model manager status"""
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'loaded_count': len(self.loaded_models),
            'available_ram_mb': self.get_available_ram_mb(),
            'models_dir': str(self.models_dir)
        }


# Global instance
model_manager = ModelManager()