"""Base definitions for vector store backends."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StoredChunk:
    """Represents a stored chunk in the vector store."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends."""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the backend with configuration."""
        pass
    
    @abstractmethod
    def add_chunks(
        self,
        chunks: List[StoredChunk],
        collection_name: str
    ) -> bool:
        """
        Add chunks to a collection.
        
        Args:
            chunks: List of chunks to add
            collection_name: Collection/table name
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def delete_chunks(
        self,
        chunk_ids: List[str],
        collection_name: str
    ) -> bool:
        """
        Delete chunks from a collection.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            collection_name: Collection/table name
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def search_by_similarity(
        self,
        query_embedding: List[float],
        collection_name: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[StoredChunk, float]]:
        """
        Search chunks by similarity.
        
        Args:
            query_embedding: Query embedding vector
            collection_name: Collection/table name
            top_k: Number of results
            threshold: Minimum similarity score
        
        Returns:
            List of (chunk, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    def get_chunk(
        self,
        chunk_id: str,
        collection_name: str
    ) -> Optional[StoredChunk]:
        """Get a single chunk by ID."""
        pass
    
    @abstractmethod
    def list_chunks(
        self,
        collection_name: str
    ) -> List[StoredChunk]:
        """List all chunks in a collection."""
        pass
    
    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new collection/table."""
        pass
    
    @abstractmethod
    def delete_collection(
        self,
        collection_name: str
    ) -> bool:
        """Delete a collection/table."""
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all data."""
        pass
