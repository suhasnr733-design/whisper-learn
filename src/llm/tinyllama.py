"""
TinyLlama model (1.1B parameters) - Lighter alternative to Phi-2
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

class TinyLlamaQA:
    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load TinyLlama model"""
        if self.model is not None:
            return
        
        print("Loading TinyLlama model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("TinyLlama loaded successfully")
    
    def generate_answer(self, question: str, context: str, max_tokens: int = 200) -> str:
        """Generate answer using TinyLlama"""
        self.load_model()
        
        messages = [
            {"role": "system", "content": "You are a helpful teaching assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response