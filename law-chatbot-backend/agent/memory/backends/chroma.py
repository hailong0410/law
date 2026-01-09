"""Chroma backend implementation."""
import os
import json
# Disable Chroma default embedding model download
os.environ['CHROMA_DISABLE_DEFAULT_EMBEDDING'] = '1'
from typing import List, Dict, Any, Optional, Tuple, Callable
from .base import StoredChunk, VectorStoreBackend
from agent.logging import logger


class ChromaVectorStore(VectorStoreBackend):
    """Chroma vector store backend."""
    
    def __init__(self, **kwargs):
        """
        Initialize Chroma backend.
        
        Args:
            persist_directory: Optional directory for persistence
            host: Chroma server host (for HttpClient)
            port: Chroma server port (for HttpClient)
            use_google_embedding: Whether to use Google Embedding API (default: True)
            embedding_api_key: Google API key (optional, uses GEMINI_API_KEY env var if not provided)
            embedding_model: Google embedding model name (default: "models/embedding-001")
            embedding_function: Custom embedding function (takes priority over use_google_embedding)
        """
        try:
            import chromadb
            self.chromadb = chromadb
        except ImportError:
            raise ImportError("Chroma not installed. Install with: pip install chromadb")
        
        self.persist_directory = kwargs.get('persist_directory', None)
        self.host = kwargs.get('host', None)
        self.port = kwargs.get('port', None)
        
        # Embedding configuration
        # Priority: embedding_function > use_google_embedding
        # IMPORTANT: Chroma default embedding is disabled - must provide embedding_function
        self.embedding_function = kwargs.get('embedding_function', None)
        
        if self.embedding_function is None:
            # Use Google Embedding if no custom function provided
            self.use_google_embedding = kwargs.get('use_google_embedding', False)  # Default to False
            self.embedding_api_key = kwargs.get('embedding_api_key', None)
            self.embedding_model = kwargs.get('embedding_model', "models/embedding-001")
            
            # Initialize embedding function if using Google Embedding
            if self.use_google_embedding:
                try:
                    from agent.llm.gemini import get_google_embedding_function
                    self.embedding_function = get_google_embedding_function(
                        api_key=self.embedding_api_key,
                        model=self.embedding_model
                    )
                except (ImportError, ValueError) as e:
                    # Cannot fallback to default - must provide valid embedding function
                    raise ValueError(
                        f"Could not initialize Google Embedding: {e}. "
                        "Please provide a valid embedding_function (e.g., VNPT embedding). "
                        "Chroma default embedding is disabled."
                    )
        else:
            # Custom embedding function provided (VNPT, Google, or other)
            self.use_google_embedding = False
        
        if self.host and self.port:
            self.client = chromadb.HttpClient(host=self.host, port=int(self.port))
        elif self.persist_directory:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.client = chromadb.EphemeralClient()
        
        self.collections = {}
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to ensure all values are ChromaDB-compatible types.
        ChromaDB only supports: str, int, float, bool, None
        """
        if not metadata:
            return {}
        
        sanitized = {}
        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue
            
            # Convert to supported types
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list to comma-separated string
                sanitized[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                # Convert dict to JSON string
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        
        return sanitized
    
    def add_chunks(
        self,
        chunks: List[StoredChunk],
        collection_name: str
    ) -> bool:
        """Add chunks to Chroma collection."""
        try:
            collection = self._get_or_create_collection(collection_name)
            
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
            # Sanitize metadata to ensure ChromaDB compatibility
            metadatas = [self._sanitize_metadata(chunk.metadata or {}) for chunk in chunks]
            
            if embeddings and len(embeddings) == len(chunks):
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            return True
        except Exception as e:
            # Only log error type, not full message (may contain large embeddings)
            error_type = type(e).__name__
            error_msg = str(e)
            # Truncate if too long (likely contains embedding data)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            logger.error(f"Error adding chunks to Chroma: {error_type}: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def delete_chunks(
        self,
        chunk_ids: List[str],
        collection_name: str
    ) -> bool:
        """Delete chunks from Chroma collection."""
        try:
            collection = self._get_or_create_collection(collection_name)
            collection.delete(ids=chunk_ids)
            return True
        except Exception as e:
            print(f"Error deleting chunks from Chroma: {e}")
            return False
    
    def search_by_similarity(
        self,
        query_embedding: Optional[List[float]] = None,
        collection_name: str = "default",
        top_k: int = 5,
        threshold: float = 0.0,
        query_text: Optional[str] = None
    ) -> List[Tuple[StoredChunk, float]]:
        """
        Search chunks in Chroma.
        
        Args:
            query_embedding: Query embedding vector (optional, use query_text instead)
            collection_name: Collection name
            top_k: Number of results
            threshold: Similarity threshold
            query_text: Query text (Chroma will generate embedding automatically)
        """
        try:
            collection = self._get_or_create_collection(collection_name)
            
            # Use query_text if provided (Chroma will generate embeddings via API)
            # Otherwise use query_embedding if provided
            if query_text:
                results = collection.query(
                    query_texts=[query_text],
                    n_results=top_k
                )
            elif query_embedding:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
            else:
                return []
            
            output = []
            if results['ids'] and len(results['ids']) > 0:
                for i, chunk_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= threshold:
                        chunk = StoredChunk(
                            id=chunk_id,
                            content=results['documents'][0][i] if results['documents'] else "",
                            metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                        )
                        output.append((chunk, similarity))
            
            return output
        except Exception as e:
            print(f"Error searching in Chroma: {e}")
            return []
    
    def get_chunk(
        self,
        chunk_id: str,
        collection_name: str
    ) -> Optional[StoredChunk]:
        """Get a chunk from Chroma."""
        try:
            collection = self._get_or_create_collection(collection_name)
            result = collection.get(ids=[chunk_id])
            
            if result['ids'] and len(result['ids']) > 0:
                return StoredChunk(
                    id=chunk_id,
                    content=result['documents'][0] if result['documents'] else "",
                    metadata=result['metadatas'][0] if result['metadatas'] else {}
                )
            return None
        except Exception as e:
            print(f"Error getting chunk from Chroma: {e}")
            return None
    
    def list_chunks(
        self,
        collection_name: str
    ) -> List[StoredChunk]:
        """List all chunks in Chroma collection."""
        try:
            logger.debug(f"Getting or creating collection: {collection_name}")
            collection = self._get_or_create_collection(collection_name)
            
            logger.debug(f"Calling collection.get() for collection: {collection_name}")
            result = collection.get()
            
            # Log result structure for debugging
            if result:
                ids_count = len(result.get('ids', [])) if result.get('ids') else 0
                docs_count = len(result.get('documents', [])) if result.get('documents') else 0
                metadata_count = len(result.get('metadatas', [])) if result.get('metadatas') else 0
                logger.debug(f"Collection.get() returned - IDs: {ids_count}, Documents: {docs_count}, Metadatas: {metadata_count}")
                
                if ids_count == 0:
                    logger.warning(f"Collection '{collection_name}' exists but contains 0 chunks")
                    # Try to get collection count
                    try:
                        count = collection.count()
                        logger.info(f"Collection '{collection_name}' count() method returns: {count}")
                    except Exception as count_error:
                        logger.debug(f"Could not get collection count: {count_error}")
            else:
                logger.warning(f"Collection.get() returned None or empty result for: {collection_name}")
            
            chunks = []
            ids = result.get('ids', []) if result else []
            
            if not ids:
                logger.info(f"No chunk IDs found in collection: {collection_name}")
                return []
            
            logger.debug(f"Processing {len(ids)} chunk(s) from collection: {collection_name}")
            
            for i, chunk_id in enumerate(ids):
                try:
                    chunk = StoredChunk(
                        id=chunk_id,
                        content=result['documents'][i] if result.get('documents') and i < len(result['documents']) else "",
                        metadata=result['metadatas'][i] if result.get('metadatas') and i < len(result['metadatas']) else {}
                    )
                    chunks.append(chunk)
                except Exception as chunk_error:
                    logger.warning(f"Error processing chunk {i} (ID: {chunk_id}): {chunk_error}")
                    continue
            
            logger.info(f"Successfully listed {len(chunks)} chunk(s) from collection: {collection_name}")
            return chunks
        except Exception as e:
            logger.error(f"Error listing chunks from Chroma collection '{collection_name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a collection in Chroma with embedding function if configured."""
        try:
            # Delete existing collection first to ensure it uses the correct embedding function
            try:
                self.client.delete_collection(name=collection_name)
            except:
                pass  # Collection doesn't exist, that's fine
            
            # Always use embedding_function if provided (VNPT, Google, or any custom function)
            # This prevents Chroma from using default embedding model
            if self.embedding_function:
                # ChromaDB requires metadata to be non-empty dict if provided
                # If metadata is None or empty, don't pass it
                create_kwargs = {
                    "name": collection_name,
                    "embedding_function": self.embedding_function
                }
                if metadata and len(metadata) > 0:
                    create_kwargs["metadata"] = metadata
                
                self.client.create_collection(**create_kwargs)
            else:
                # No embedding function provided - raise error to force explicit configuration
                raise ValueError(
                    "No embedding function provided. "
                    "Please provide embedding_function parameter (e.g., VNPT or Google embedding). "
                    "Chroma default embedding is disabled."
                )
            return True
        except Exception as e:
            # Re-raise ValueError to ensure proper error handling
            if isinstance(e, ValueError):
                raise
            # Collection might already exist or other error
            return True
    
    def delete_collection(
        self,
        collection_name: str
    ) -> bool:
        """Delete a collection from Chroma."""
        try:
            self.client.delete_collection(name=collection_name)
            self.collections.pop(collection_name, None)
            return True
        except Exception as e:
            print(f"Error deleting collection from Chroma: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections in Chroma."""
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            print(f"Error listing collections from Chroma: {e}")
            return []
    
    def clear(self):
        """Clear all data."""
        try:
            for collection_name in self.list_collections():
                self.delete_collection(collection_name)
        except Exception as e:
            print(f"Error clearing Chroma: {e}")
    
    def _get_or_create_collection(self, collection_name: str):
        """Get or create a collection with embedding function if configured."""
        if collection_name not in self.collections:
            # If embedding_function is provided (VNPT, Google, or any custom), always use it
            if self.embedding_function:
                try:
                    # Try to get existing collection first
                    logger.debug(f"Attempting to get existing collection: {collection_name}")
                    collection = self.client.get_collection(name=collection_name)
                    # Collection already exists, use it
                    logger.info(f"Successfully retrieved existing collection: {collection_name}")
                    try:
                        count = collection.count()
                        logger.info(f"Collection '{collection_name}' contains {count} item(s)")
                    except Exception as count_error:
                        logger.debug(f"Could not get collection count: {count_error}")
                except Exception as e:
                    # Collection doesn't exist, create new one
                    logger.info(f"Collection '{collection_name}' doesn't exist (error: {e}), creating new one")
                    try:
                        collection = self.client.create_collection(
                            name=collection_name,
                            embedding_function=self.embedding_function
                        )
                        logger.info(f"Successfully created new collection: {collection_name}")
                    except Exception as create_error:
                        # If creation fails (e.g., collection was created between get and create),
                        # try to get it again
                        if "already exists" in str(create_error).lower() or "409" in str(create_error):
                            logger.info(f"Collection was created by another process, getting it...")
                            collection = self.client.get_collection(name=collection_name)
                            try:
                                count = collection.count()
                                logger.info(f"Collection '{collection_name}' contains {count} item(s)")
                            except Exception:
                                pass
                        else:
                            logger.error(f"Error creating collection '{collection_name}': {create_error}")
                            raise create_error
            else:
                # No embedding function - raise error to force explicit configuration
                logger.error("No embedding function provided for ChromaDB collection")
                raise ValueError(
                    "No embedding function provided. "
                    "Please provide embedding_function parameter (e.g., VNPT or Google embedding). "
                    "Chroma default embedding is disabled."
                )
            
            self.collections[collection_name] = collection
        
        return self.collections[collection_name]
