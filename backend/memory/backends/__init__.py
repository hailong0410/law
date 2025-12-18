"""backends package - re-export backend implementations and factory.

Keep the original public names available for backward compatibility.
"""
from .base import StoredChunk, VectorStoreBackend
from .in_memory import InMemoryVectorStore
from .chroma import ChromaVectorStore
from .qdrant import QdrantVectorStore
from .factory import VectorStoreFactory

__all__ = [
    "StoredChunk",
    "VectorStoreBackend",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "QdrantVectorStore",
    "VectorStoreFactory",
]
