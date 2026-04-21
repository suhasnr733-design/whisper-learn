"""
Abstractive summarization using transformer models
"""
from typing import Dict, Optional

class AbstractiveSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.pipeline = None
        
    def _load_model(self):
        """Lazy load the summarization model"""
        if self.pipeline is None:
            from transformers import pipeline
            import torch
            
            device = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline(
                "summarization",
                model=self.model_name,
                device=device
            )
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> Dict:
        """Generate abstractive summary"""
        self._load_model()
        
        # Handle long text by chunking
        if len(text) > 1024:
            chunks = self._chunk_text(text)
            summaries = []
            
            for chunk in chunks[:3]:  # Limit to first 3 chunks
                result = self.pipeline(
                    chunk,
                    max_length=max_length // 2,
                    min_length=min_length // 2,
                    do_sample=False
                )
                summaries.append(result[0]['summary_text'])
            
            summary = ' '.join(summaries)
        else:
            result = self.pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            summary = result[0]['summary_text']
        
        return {
            'summary': summary,
            'model': self.model_name,
            'original_length': len(text.split()),
            'summary_length': len(summary.split())
        }
    
    def _chunk_text(self, text: str, chunk_size: int = 512) -> list:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def unload(self):
        """Unload model to free memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            
        import gc
        gc.collect()