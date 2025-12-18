"""Factory to create vector store backends."""
from typing import List
from .in_memory import InMemoryVectorStore
from .chroma import ChromaVectorStore
from .qdrant import QdrantVectorStore
from .base import VectorStoreBackend


class VectorStoreFactory:
    """Factory for creating vector store backends."""
    
    _backends = {
        'in-memory': InMemoryVectorStore,
        'memory': InMemoryVectorStore,
        'chroma': ChromaVectorStore,
        'qdrant': QdrantVectorStore,
    }
    
    @classmethod
    def create(
        cls,
        backend_type: str,
        **kwargs
    ) -> VectorStoreBackend:
        """
        Create a vector store backend.
        
        Args:
            backend_type: Type of backend ('in-memory', 'chroma', 'qdrant')
            **kwargs: Backend-specific configuration
        
        Returns:
            VectorStoreBackend instance
        
        Raises:
            ValueError: If backend type not found
        """
        backend_type = backend_type.lower()
        
        if backend_type not in cls._backends:
            raise ValueError(
                f"Unknown backend: {backend_type}. "
                f"Available: {list(cls._backends.keys())}"
            )
        
        backend_class = cls._backends[backend_type]
        return backend_class(**kwargs)
    
    @classmethod
    def register_backend(
        cls,
        name: str,
        backend_class: type
    ) -> None:
        """Register a custom backend."""
        cls._backends[name.lower()] = backend_class
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List available backends."""
        return list(cls._backends.keys())
