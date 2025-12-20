"""Qdrant backend implementation."""
from typing import List, Dict, Any, Optional, Tuple
from .base import StoredChunk, VectorStoreBackend


class QdrantVectorStore(VectorStoreBackend):
    """Qdrant vector store backend."""
    
    def __init__(self, **kwargs):
        """
        Initialize Qdrant backend.
        
        Args:
            url: Qdrant server URL (e.g., 'http://localhost:6333')
            api_key: Optional API key
            in_memory: Use in-memory Qdrant (default: False)
        """
        try:
            from qdrant_client import QdrantClient
            self.QdrantClient = QdrantClient
        except ImportError:
            raise ImportError("Qdrant not installed. Install with: pip install qdrant-client")
        
        url = kwargs.get('url', 'http://localhost:6333')
        api_key = kwargs.get('api_key', None)
        in_memory = kwargs.get('in_memory', False)
        
        if in_memory:
            self.client = QdrantClient(":memory:")
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(":memory:")
        
        self.collections = {}
        self.vector_size = kwargs.get('vector_size', 384)
    
    def add_chunks(
        self,
        chunks: List[StoredChunk],
        collection_name: str
    ) -> bool:
        """Add chunks to Qdrant collection."""
        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            # Create collection if not exists
            if collection_name not in self.collections:
                self.create_collection(collection_name)
            
            points = []
            for chunk in chunks:
                if chunk.embedding:
                    point = PointStruct(
                        id=hash(chunk.id) % (10 ** 8),  # Convert to numeric ID
                        vector=chunk.embedding,
                        payload={
                            "id": chunk.id,
                            "content": chunk.content,
                            **(chunk.metadata or {})
                        }
                    )
                    points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
            
            return True
        except Exception as e:
            print(f"Error adding chunks to Qdrant: {e}")
            return False
    
    def delete_chunks(
        self,
        chunk_ids: List[str],
        collection_name: str
    ) -> bool:
        """Delete chunks from Qdrant collection."""
        try:
            # Convert string IDs to numeric
            numeric_ids = [hash(chunk_id) % (10 ** 8) for chunk_id in chunk_ids]
            self.client.delete(
                collection_name=collection_name,
                points_selector=[{"points": numeric_ids}]
            )
            return True
        except Exception as e:
            print(f"Error deleting chunks from Qdrant: {e}")
            return False
    
    def search_by_similarity(
        self,
        query_embedding: List[float],
        collection_name: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[StoredChunk, float]]:
        """Search chunks in Qdrant."""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=threshold
            )
            
            output = []
            for result in results:
                chunk = StoredChunk(
                    id=result.payload.get("id", ""),
                    content=result.payload.get("content", ""),
                    metadata={k: v for k, v in result.payload.items() 
                             if k not in ["id", "content"]}
                )
                output.append((chunk, result.score))
            
            return output
        except Exception as e:
            print(f"Error searching in Qdrant: {e}")
            return []
    
    def get_chunk(
        self,
        chunk_id: str,
        collection_name: str
    ) -> Optional[StoredChunk]:
        """Get a chunk from Qdrant."""
        try:
            numeric_id = hash(chunk_id) % (10 ** 8)
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[numeric_id]
            )
            
            if result and len(result) > 0:
                payload = result[0].payload
                return StoredChunk(
                    id=payload.get("id", chunk_id),
                    content=payload.get("content", ""),
                    metadata={k: v for k, v in payload.items() 
                             if k not in ["id", "content"]}
                )
            return None
        except Exception as e:
            print(f"Error getting chunk from Qdrant: {e}")
            return None
    
    def list_chunks(
        self,
        collection_name: str
    ) -> List[StoredChunk]:
        """List all chunks in Qdrant collection."""
        try:
            # Qdrant doesn't have a simple list all method, use search with empty vector
            results = self.client.scroll(
                collection_name=collection_name,
                limit=10000
            )
            
            chunks = []
            for point in results[0]:
                chunk = StoredChunk(
                    id=point.payload.get("id", ""),
                    content=point.payload.get("content", ""),
                    metadata={k: v for k, v in point.payload.items() 
                             if k not in ["id", "content"]}
                )
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            print(f"Error listing chunks from Qdrant: {e}")
            return []
    
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a collection in Qdrant."""
        try:
            from qdrant_client.models import Distance, VectorParams
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            self.collections[collection_name] = True
            return True
        except Exception as e:
            # Collection might already exist
            return True
    
    def delete_collection(
        self,
        collection_name: str
    ) -> bool:
        """Delete a collection from Qdrant."""
        try:
            self.client.delete_collection(collection_name=collection_name)
            self.collections.pop(collection_name, None)
            return True
        except Exception as e:
            print(f"Error deleting collection from Qdrant: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections in Qdrant."""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            print(f"Error listing collections from Qdrant: {e}")
            return []
    
    def clear(self):
        """Clear all data."""
        try:
            for collection_name in self.list_collections():
                self.delete_collection(collection_name)
        except Exception as e:
            print(f"Error clearing Qdrant: {e}")
