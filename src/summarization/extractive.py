"""
Extractive summarization using TextRank and other methods
"""
from typing import List, Dict
import numpy as np

class ExtractiveSummarizer:
    def __init__(self, language: str = "english"):
        self.language = language
        
    def textrank_summary(self, text: str, num_sentences: int = 5) -> str:
        """TextRank based extractive summarization"""
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, num_sentences)
            
            return ' '.join(str(sentence) for sentence in summary)
            
        except ImportError:
            return self._simple_summary(text, num_sentences)
    
    def lsa_summary(self, text: str, num_sentences: int = 5) -> str:
        """LSA-based extractive summarization"""
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.lsa import LsaSummarizer
            
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, num_sentences)
            
            return ' '.join(str(sentence) for sentence in summary)
            
        except ImportError:
            return self._simple_summary(text, num_sentences)
    
    def _simple_summary(self, text: str, num_sentences: int) -> str:
        """Simple fallback summarization"""
        import re
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        
        if not sentences:
            return text[:500]
        
        # Score sentences by position (first sentences are more important)
        scores = [1.0 / (i + 1) for i in range(len(sentences))]
        
        # Get top sentences
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices.sort()
        
        summary_sentences = [sentences[i] for i in top_indices]
        return '. '.join(summary_sentences) + '.'
    
    def summarize(self, text: str, method: str = "textrank", num_sentences: int = 5) -> Dict:
        """Main summarization method"""
        if method == "textrank":
            summary = self.textrank_summary(text, num_sentences)
        elif method == "lsa":
            summary = self.lsa_summary(text, num_sentences)
        else:
            summary = self._simple_summary(text, num_sentences)
        
        return {
            'summary': summary,
            'method': method,
            'num_sentences': num_sentences,
            'original_length': len(text.split()),
            'summary_length': len(summary.split())
        }