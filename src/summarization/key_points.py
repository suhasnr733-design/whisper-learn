"""
Key point extraction from text
"""
from typing import List, Dict
import re

class KeyPointExtractor:
    def __init__(self, num_points: int = 5):
        self.num_points = num_points
        
    def extract_keywords(self, text: str) -> List[Dict]:
        """Extract keywords using RAKE or simple TF-IDF"""
        try:
            from rake_nltk import Rake
            rake = Rake()
            rake.extract_keywords_from_text(text)
            keywords = rake.get_ranked_phrases_with_scores()
            
            return [{'keyword': kw, 'score': score} 
                    for score, kw in keywords[:self.num_points]]
                    
        except ImportError:
            return self._simple_keywords(text)
    
    def _simple_keywords(self, text: str) -> List[Dict]:
        """Simple keyword extraction using frequency"""
        from collections import Counter
        
        # Clean text
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Remove stopwords
        stopwords = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 
                    'from', 'have', 'will', 'can', 'was', 'were', 'has'}
        words = [w for w in words if w not in stopwords]
        
        # Count frequencies
        counter = Counter(words)
        
        return [{'keyword': word, 'score': count} 
                for word, count in counter.most_common(self.num_points)]
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases using n-grams"""
        words = text.lower().split()
        
        # Extract bigrams and trigrams
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        all_phrases = bigrams + trigrams
        
        # Score by frequency
        from collections import Counter
        phrase_counts = Counter(all_phrases)
        
        # Filter by length and frequency
        filtered = [(phrase, count) for phrase, count in phrase_counts.items() 
                   if count > 1 and len(phrase) > 5]
        
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return [phrase for phrase, _ in filtered[:max_phrases]]
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points (important sentences)"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        
        # Score sentences
        scores = []
        for sent in sentences:
            score = 0
            
            # Longer sentences
            score += len(sent.split()) / 10
            
            # Sentences with key phrases
            key_phrases = self.extract_key_phrases(text, 5)
            for phrase in key_phrases:
                if phrase in sent.lower():
                    score += 5
            
            # Sentences at beginning of paragraphs (approximated)
            if sent[0].isupper() and len(sent) > 20:
                score += 3
            
            scores.append(score)
        
        # Get top sentences
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(top_indices[:self.num_points])
        
        key_points = [sentences[i] for i in top_indices]
        
        return key_points
    
    def summarize_key_points(self, text: str) -> Dict:
        """Full key point extraction"""
        return {
            'key_points': self.extract_key_points(text),
            'keywords': self.extract_keywords(text),
            'key_phrases': self.extract_key_phrases(text),
            'num_points': self.num_points
        }