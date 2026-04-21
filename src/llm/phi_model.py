"""
Microsoft Phi-2 model for Q&A (2.7B parameters, 4-bit quantized)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional

class Phi2QA:
    def __init__(self, use_quantization: bool = True):
        self.model_name = "microsoft/phi-2"
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load Phi-2 with 4-bit quantization"""
        if self.model is not None:
            return
        
        print("Loading Phi-2 model...")
        
        if self.use_quantization:
            # 4-bit quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Phi-2 model loaded successfully")
    
    def generate_answer(self, question: str, context: str, max_tokens: int = 200) -> str:
        """Generate answer based on context"""
        self.load_model()
        
        prompt = f"""You are a helpful teaching assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        answer = response.split("Answer:")[-1].strip()
        if not answer:
            answer = response.split("answer:")[-1].strip()
        
        return answer
    
    def generate_qa_pair(self, context: str) -> Dict[str, str]:
        """Generate question-answer pair from context"""
        self.load_model()
        
        prompt = f"""Based on the following text, generate a question and its answer.

Text:
{context}

Generate a question that tests understanding of the key concept, then provide the answer.

Question:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse generated text
        parts = generated.split("Answer:")
        if len(parts) >= 2:
            return {
                'question': parts[0].split("Question:")[-1].strip(),
                'answer': parts[1].strip()
            }
        
        return {'question': '', 'answer': ''}
    
    def unload(self):
        """Unload model to free memory"""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()