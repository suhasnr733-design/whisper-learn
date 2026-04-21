"""
Document retriever for RAG pipeline
"""
from typing import List, Dict, Optional
import numpy as np

class Retriever:
    def __init__(self, vector_store, embedding_generator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.5) -> List[Dict]:
        """Retrieve relevant documents"""
        # Generate query embedding
        query_embedding = self.embedding_generator.encode([query])
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)
        
        # Filter by threshold
        filtered = [r for r in results if r['similarity'] >= score_threshold]
        
        return filtered
    
    def retrieve_with_context(self, query: str, context_window: int = 1) -> List[Dict]:
        """Retrieve with surrounding context"""
        results = self.retrieve(query, top_k=3)
        
        expanded_results = []
        for result in results:
            # Get surrounding chunks if available
            idx = result.get('index', -1)
            if idx != -1 and hasattr(self.vector_store, 'texts'):
                start = max(0, idx - context_window)
                end = min(len(self.vector_store.texts), idx + context_window + 1)
                
                context_chunks = self.vector_store.texts[start:end]
                result['expanded_context'] = '\n'.join(context_chunks)
            
            expanded_results.append(result)
        
        return expanded_results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Dict]]:
        """Batch retrieval for multiple queries"""
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k))
        return results


class HybridRetriever:
    """Combines semantic search with keyword search"""
    
    def __init__(self, vector_store, embedding_generator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(max_features=500)
        
    def keyword_search(self, query: str, documents: List[str], top_k: int = 5) -> List[int]:
        """Traditional keyword-based search"""
        if not documents:
            return []
        
        # Fit TF-IDF on documents
        tfidf_matrix = self.tfidf.fit_transform(documents)
        
        # Transform query
        query_vec = self.tfidf.transform([query])
        
        # Compute similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        
        # Get top indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def hybrid_search(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """Combine semantic and keyword search"""
        # Semantic search
        semantic_results = self.vector_store.search(
            self.embedding_generator.encode([query]), 
            top_k
        )
        
        # Keyword search
        keyword_results = self.keyword_search(query, documents, top_k)
        
        # Combine and deduplicate
        combined = {}
        
        for res in semantic_results:
            combined[res.get('index', id(res))] = {
                'score': res['similarity'],
                'method': 'semantic',
                'text': res['text']
            }
        
        for idx, score in keyword_results:
            if idx not in combined:
                combined[idx] = {
                    'score': score,
                    'method': 'keyword',
                    'text': documents[idx]
                }
        
        # Sort by score
        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        
        return sorted_results[:top_k]