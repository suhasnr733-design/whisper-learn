"""
FAISS vector store for efficient similarity search
"""
import faiss
import numpy as np
from typing import List, Dict, Optional
import pickle
from pathlib import Path

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.texts = []
        self.metadata = []
        
    def create_index(self, index_type: str = "Flat"):
        """Create FAISS index"""
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        
        return self
    
    def add_embeddings(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict] = None):
        """Add embeddings to index"""
        if self.index is None:
            self.create_index()
        
        # Train IVF index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.texts.extend(texts)
        
        if metadata:
            self.metadata.extend(metadata)
        
        return len(self.texts)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar vectors"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.texts):
                results.append({
                    'text': self.texts[idx],
                    'distance': float(distances[0][i]),
                    'similarity': 1 / (1 + distances[0][i]),
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                })
        
        return results
    
    def save(self, path: str):
        """Save index to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / 'index.faiss'))
        
        # Save texts and metadata
        with open(path / 'data.pkl', 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
    
    def load(self, path: str):
        """Load index from disk"""
        path = Path(path)
        
        self.index = faiss.read_index(str(path / 'index.faiss'))
        
        with open(path / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']
            self.dimension = data['dimension']


class DocumentChunker:
    """Split documents into chunks for indexing"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def chunk_document(self, title: str, content: str) -> List[Dict]:
        """Chunk document with metadata"""
        chunks = self.chunk_text(content)
        
        return [{
            'text': chunk,
            'metadata': {
                'title': title,
                'chunk_id': i,
                'total_chunks': len(chunks)
            }
        } for i, chunk in enumerate(chunks)]