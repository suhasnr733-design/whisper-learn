"""
Embedding generation for text
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Model options:
        - all-MiniLM-L6-v2 (80MB, 384 dims) - Fastest
        - all-mpnet-base-v2 (420MB, 768 dims) - Better quality
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode search queries (optimized)"""
        return self.encode(queries)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents for indexing"""
        return self.encode(documents)


class CachedEmbeddingGenerator:
    """Embedding generator with caching to avoid recomputation"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_size: int = 10000):
        self.generator = EmbeddingGenerator(model_name)
        self.cache = {}
        self.cache_size = cache_size
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings with caching"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        uncached_indices = []
        uncached_texts = []
        embeddings = [None] * len(texts)
        
        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings[i] = self.cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Generate for uncached texts
        if uncached_texts:
            new_embeddings = self.generator.encode(uncached_texts)
            
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb
                
                # Add to cache
                if len(self.cache) >= self.cache_size:
                    # Remove oldest (simple FIFO)
                    oldest = next(iter(self.cache))
                    del self.cache[oldest]
                
                self.cache[texts[idx]] = emb
        
        return np.array(embeddings)