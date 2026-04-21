"""
Translation module using OPUS-MT (lightweight)
"""
from typing import List, Optional
import torch

class OPUSTranslator:
    def __init__(self, source_lang: str = "auto", target_lang: str = "en"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """Load OPUS-MT model"""
        from transformers import MarianMTModel, MarianTokenizer
        
        model_name = f"Helsinki-NLP/opus-mt-{self.source_lang}-{self.target_lang}"
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
        except:
            # Fallback to English-centric model
            model_name = "Helsinki-NLP/opus-mt-tc-big-en-multilingual"
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
    
    def translate(self, text: str, source_lang: Optional[str] = None) -> str:
        """Translate text to target language"""
        if self.model is None:
            self._load_model()
        
        if source_lang:
            self.source_lang = source_lang
        
        # Split long text
        sentences = text.split('. ')
        translated_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = self.model.generate(**inputs)
            
            decoded = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            translated_sentences.append(decoded)
        
        return '. '.join(translated_sentences)
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple texts"""
        results = []
        for text in texts:
            results.append(self.translate(text))
        return results


class LightweightTranslator:
    """Use OPUS-MT for translation with minimal memory"""
    
    def __init__(self):
        from transformers import pipeline
        self.translator = None
        
    def translate(self, text: str, target_lang: str = "en") -> str:
        """On-demand translation using pipeline"""
        if self.translator is None:
            self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ROMANCE")
        
        result = self.translator(text, max_length=512)
        return result[0]['translation_text']