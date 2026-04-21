"""
Semantic alignment of slides with transcript
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

class SlideAligner:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Lightweight embedding model (~80MB)"""
        self.model = SentenceTransformer(model_name)
        self.slide_embeddings = None
        self.slide_texts = None
        
    def embed_slides(self, slide_texts: List[str]):
        """Generate embeddings for all slides"""
        self.slide_texts = slide_texts
        self.slide_embeddings = self.model.encode(slide_texts, show_progress_bar=False)
        return self.slide_embeddings
    
    def align_transcript(self, transcript_segments: List[Dict]) -> List[Dict]:
        """Align transcript segments to slides"""
        if self.slide_embeddings is None:
            raise ValueError("Call embed_slides() first")
        
        # Extract transcript texts
        transcript_texts = [seg['text'] for seg in transcript_segments]
        transcript_embeddings = self.model.encode(transcript_texts, show_progress_bar=False)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(transcript_embeddings, self.slide_embeddings)
        
        # Assign best matching slide
        aligned_segments = []
        for i, seg in enumerate(transcript_segments):
            best_slide = np.argmax(similarity_matrix[i])
            confidence = similarity_matrix[i][best_slide]
            
            aligned_segments.append({
                'transcript_text': seg['text'],
                'start_time': seg.get('start', 0),
                'end_time': seg.get('end', 0),
                'slide_index': best_slide,
                'slide_text': self.slide_texts[best_slide],
                'confidence': float(confidence)
            })
        
        return aligned_segments
    
    def get_slide_timeline(self, aligned_segments: List[Dict]) -> List[Dict]:
        """Generate timeline of when each slide appears"""
        timeline = {}
        
        for seg in aligned_segments:
            slide_idx = seg['slide_index']
            if slide_idx not in timeline:
                timeline[slide_idx] = {
                    'slide_index': slide_idx,
                    'slide_text': seg['slide_text'],
                    'start_time': seg['start_time'],
                    'end_time': seg['end_time'],
                    'confidence': seg['confidence']
                }
            else:
                timeline[slide_idx]['end_time'] = seg['end_time']
        
        return list(timeline.values())


class TFIDFSlideAligner:
    """Lightweight alternative using TF-IDF"""
    
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.slide_vectors = None
        
    def fit_slides(self, slide_texts: List[str]):
        """Fit TF-IDF on slide texts"""
        self.slide_vectors = self.vectorizer.fit_transform(slide_texts)
    
    def align(self, transcript_text: str) -> int:
        """Find best matching slide for transcript text"""
        transcript_vector = self.vectorizer.transform([transcript_text])
        
        similarities = cosine_similarity(transcript_vector, self.slide_vectors)[0]
        best_slide = np.argmax(similarities)
        
        return best_slide, similarities[best_slide]