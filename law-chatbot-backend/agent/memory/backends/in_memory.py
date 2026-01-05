"""In-memory backend implementation for vector store."""
from typing import List, Dict, Optional, Tuple, Any
from .base import StoredChunk, VectorStoreBackend


class InMemoryVectorStore(VectorStoreBackend):
    """In-memory vector store implementation (for development/testing)."""
    
    def __init__(self, **kwargs):
        """Initialize in-memory store."""
        self.collections: Dict[str, Dict[str, StoredChunk]] = {}
    
    def add_chunks(
        self,
        chunks: List[StoredChunk],
        collection_name: str
    ) -> bool:
        """Add chunks to collection."""
        if collection_name not in self.collections:
            self.create_collection(collection_name)
        
        for chunk in chunks:
            self.collections[collection_name][chunk.id] = chunk
        return True
    
    def delete_chunks(
        self,
        chunk_ids: List[str],
        collection_name: str
    ) -> bool:
        """Delete chunks from collection."""
        if collection_name not in self.collections:
            return False
        
        for chunk_id in chunk_ids:
            self.collections[collection_name].pop(chunk_id, None)
        return True
    
    def search_by_similarity(
        self,
        query_embedding: List[float],
        collection_name: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[StoredChunk, float]]:
        """Search chunks by similarity."""
        if collection_name not in self.collections:
            return []
        
        import numpy as np
        
        query_vec = np.array(query_embedding)
        results = []
        
        for chunk in self.collections[collection_name].values():
            if chunk.embedding is None:
                continue
            
            emb_vec = np.array(chunk.embedding)
            similarity = np.dot(query_vec, emb_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(emb_vec) + 1e-10
            )
            
            if similarity >= threshold:
                results.append((chunk, float(similarity)))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_chunk(
        self,
        chunk_id: str,
        collection_name: str
    ) -> Optional[StoredChunk]:
        """Get a single chunk."""
        if collection_name not in self.collections:
            return None
        return self.collections[collection_name].get(chunk_id)
    
    def list_chunks(
        self,
        collection_name: str
    ) -> List[StoredChunk]:
        """List all chunks in collection."""
        if collection_name not in self.collections:
            return []
        return list(self.collections[collection_name].values())
    
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new collection."""
        if collection_name not in self.collections:
            self.collections[collection_name] = {}
        return True
    
    def delete_collection(
        self,
        collection_name: str
    ) -> bool:
        """Delete a collection."""
        if collection_name in self.collections:
            del self.collections[collection_name]
            return True
        return False
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        return list(self.collections.keys())
    
    def clear(self):
        """Clear all data."""
        self.collections.clear()
