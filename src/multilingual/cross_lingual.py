"""
Cross-lingual summarization and processing
"""
from typing import Dict, List
import numpy as np

class CrossLingualSummarizer:
    def __init__(self, source_lang: str = "auto", target_lang: str = "en"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator = None
        self.summarizer = None
        
    def _load_components(self):
        """Lazy load translation and summarization models"""
        if self.translator is None:
            from .translator import OPUSTranslator
            self.translator = OPUSTranslator(self.source_lang, self.target_lang)
        
        if self.summarizer is None:
            import sys
            sys.path.append('.')
            from src.summarization.hybrid_summarizer import HybridSummarizer
            self.summarizer = HybridSummarizer()
    
    def summarize_multilingual(self, text: str, original_lang: str) -> Dict:
        """Summarize text in original and target language"""
        self._load_components()
        
        # Translate to English if needed
        if original_lang != 'en':
            translated = self.translator.translate(text, original_lang)
        else:
            translated = text
        
        # Generate summary in English
        summary_en = self.summarizer.summarize(translated)
        
        # Translate summary back to original language
        if original_lang != 'en':
            summary_original = self.translator.translate(summary_en['summary'], 'en')
            summary_en['summary_original'] = summary_original
        
        return summary_en
    
    def process_lecture(self, transcript: str, language: str) -> Dict:
        """Complete multilingual lecture processing"""
        result = {
            'original_language': language,
            'transcript': transcript,
            'summary': None,
            'translation': None
        }
        
        # Translate if not English
        if language != 'en':
            self._load_components()
            result['translation'] = self.translator.translate(transcript, language)
            result['summary'] = self.summarize_multilingual(transcript, language)
        else:
            self._load_components()
            result['summary'] = self.summarizer.summarize(transcript)
        
        return result