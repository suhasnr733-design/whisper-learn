"""
Helper functions for quantized LLM loading
"""
import torch
from typing import Optional

def load_quantized_model(model_name: str, load_in_4bit: bool = True):
    """Load any model with quantization"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    return model, tokenizer


# Model configurations for 8GB RAM
AVAILABLE_MODELS = {
    'phi-2': {
        'name': 'microsoft/phi-2',
        'size_gb': 1.6,
        'quantized_size_gb': 0.9,
        'description': 'Best quality for 2.7B'
    },
    'tinyllama': {
        'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'size_gb': 0.7,
        'quantized_size_gb': 0.4,
        'description': 'Fastest, good enough'
    },
    'starcoder': {
        'name': 'bigcode/starcoderbase-1b',
        'size_gb': 0.8,
        'quantized_size_gb': 0.5,
        'description': 'Good for code lectures'
    }
}

def get_best_model_for_ram(available_ram_gb: float) -> str:
    """Recommend best model based on available RAM"""
    if available_ram_gb >= 6:
        return 'phi-2'
    else:
        return 'tinyllama'