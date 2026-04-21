"""
Hybrid summarization combining extractive and abstractive methods
"""
from typing import List, Dict
import numpy as np

class HybridSummarizer:
    def __init__(self, use_abstractive: bool = True):
        self.use_abstractive = use_abstractive
        self.abstractive_model = None
        
    def _load_abstractive(self):
        """Lazy load BART model"""
        if self.abstractive_model is None and self.use_abstractive:
            from transformers import pipeline
            self.abstractive_model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
    
    def extractive_summary(self, text: str, num_sentences: int = 5) -> str:
        """Extractive summarization using TextRank"""
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, num_sentences)
            
            return ' '.join(str(sentence) for sentence in summary)
        except ImportError:
            # Fallback: simple sentence scoring
            return self._simple_extractive_summary(text, num_sentences)
    
    def _simple_extractive_summary(self, text: str, num_sentences: int) -> str:
        """Simple extractive summary using sentence scoring"""
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text[:500]
        
        # Score sentences by length (simple heuristic)
        scores = [len(s.split()) for s in sentences]
        top_indices = np.argsort(scores)[-num_sentences:]
        
        summary_sentences = [sentences[i] for i in sorted(top_indices)]
        return '. '.join(summary_sentences) + '.'
    
    def abstractive_summary(self, text: str, max_length: int = 200, min_length: int = 50) -> str:
        """Abstractive summarization using BART"""
        self._load_abstractive()
        
        if self.abstractive_model is None:
            return self.extractive_summary(text)
        
        # Chunk if too long
        if len(text) > 2048:
            chunks = [text[i:i+1024] for i in range(0, len(text), 512)]
            summaries = []
            for chunk in chunks[:3]:  # Limit to first 3 chunks
                result = self.abstractive_model(
                    chunk,
                    max_length=max_length // 2,
                    min_length=min_length // 2,
                    do_sample=False
                )
                summaries.append(result[0]['summary_text'])
            return ' '.join(summaries)
        
        result = self.abstractive_model(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        return result[0]['summary_text']
    
    def summarize(self, text: str, style: str = "hybrid") -> Dict:
        """
        Generate summary in specified style
        styles: "extractive", "abstractive", "hybrid"
        """
        extractive = self.extractive_summary(text)
        
        if style == "extractive":
            summary = extractive
        elif style == "abstractive":
            summary = self.abstractive_summary(text)
        else:  # hybrid
            abstractive = self.abstractive_summary(text)
            # Combine both
            summary = f"{abstractive}\n\nKey points:\n{extractive}"
        
        return {
            'summary': summary,
            'style': style,
            'length': len(summary.split()),
            'original_length': len(text.split())
        }
    
    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """Extract key points using keyword extraction"""
        try:
            from keybert import KeyBERT
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=num_points
            )
            return [kw[0] for kw in keywords]
        except ImportError:
            # Fallback: simple TF-IDF
            return self._simple_keywords(text, num_points)
    
    def _simple_keywords(self, text: str, num_points: int) -> List[str]:
        """Simple keyword extraction using word frequency"""
        from collections import Counter
        import re
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stopwords = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'have'}
        words = [w for w in words if w not in stopwords]
        
        counter = Counter(words)
        return [word for word, _ in counter.most_common(num_points)]