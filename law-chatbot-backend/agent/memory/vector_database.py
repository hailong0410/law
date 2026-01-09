"""
Vector Database for RAG with document storage and retrieval.
Supports multiple collections and pluggable vector store backends.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
import numpy as np
from collections import defaultdict

from .chunking import ChunkedDocument, ChunkingFactory, ChunkingStrategy
from .backends import VectorStoreBackend, VectorStoreFactory, StoredChunk, InMemoryVectorStore
from agent.logging import logger


@dataclass
class DocumentRecord:
    """Record of a document in the database."""
    document_id: str
    content: str
    chunks: List[str]
    chunk_embeddings: List[List[float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunking_strategy: str = "sentence"
    collection_name: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "num_chunks": len(self.chunks),
            "chunks": self.chunks,
            "metadata": self.metadata,
            "chunking_strategy": self.chunking_strategy,
            "collection_name": self.collection_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class VectorDatabase:
    """
    Vector Database for RAG with advanced chunking and document management.
    Supports multiple collections and pluggable vector store backends.
    """
    
    def __init__(
        self,
        embedding_model=None,
        backend_type: str = "in-memory",
        **backend_kwargs
    ):
        """
        Initialize the Vector Database.
        
        Args:
            embedding_model: DEPRECATED - Embeddings are now handled by Chroma or API
            backend_type: Type of vector store backend ('in-memory', 'chroma', 'qdrant')
            **backend_kwargs: Backend-specific configuration
        """
        logger.info(f"Initializing VectorDatabase with backend type: {backend_type}")
        # embedding_model is deprecated - Chroma will handle embeddings automatically
        # or use API-based embeddings
        self.embedding_model = None
        
        # Initialize vector store backend
        self.backend: VectorStoreBackend = VectorStoreFactory.create(
            backend_type,
            **backend_kwargs
        )
        
        # Collections tracking
        self.collections: set = set()
        logger.debug("VectorDatabase initialization completed")
    
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new collection (table) in the database.
        
        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection
        
        Returns:
            True if successful
        """
        logger.info(f"Creating collection: {collection_name}")
        if collection_name in self.collections:
            logger.warning(f"Collection '{collection_name}' already exists")
            raise ValueError(f"Collection '{collection_name}' already exists")
        
        success = self.backend.create_collection(collection_name, metadata)
        if success:
            self.collections.add(collection_name)
            logger.info(f"Collection '{collection_name}' created successfully")
        return success
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
        
        Returns:
            True if successful
        """
        logger.info(f"Deleting collection: {collection_name}")
        success = self.backend.delete_collection(collection_name)
        if success:
            self.collections.discard(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
        return success
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        collections = self.backend.list_collections()
        self.collections = set(collections)
        return collections
    
    def add_document(
        self,
        content: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "sentence",
        max_chunk_length: int = 512,
        embedding: bool = True,
        collection_name: str = "default"
    ) -> str:
        """
        Add a document to the database.
        
        Args:
            content: Document content
            document_id: Optional document ID (auto-generated if not provided)
            metadata: Optional metadata dict
            chunking_strategy: Strategy for chunking
            max_chunk_length: Maximum length of each chunk
            embedding: Whether to generate embeddings for chunks
            collection_name: Name of the collection to add to
        
        Returns:
            Document ID
        
        Raises:
            ValueError: If document ID already exists or invalid strategy
        """
        # Create collection if not exists
        if collection_name not in self.collections:
            self.create_collection(collection_name)
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = self._generate_document_id()
        
        # Check if document already exists by querying backend
        # Search in all collections if collection_name not specified
        collections_to_check = [collection_name] if collection_name in self.collections else self.list_collections()
        for coll_name in collections_to_check:
            try:
                chunks = self.backend.list_chunks(coll_name)
                for chunk in chunks:
                    if chunk.metadata and chunk.metadata.get("document_id") == document_id:
                        raise ValueError(f"Document with ID '{document_id}' already exists in collection '{coll_name}'.")
            except Exception:
                pass  # Collection might not exist yet, that's fine
        
        # Validate chunking strategy
        try:
            chunker = ChunkingFactory.get_strategy(chunking_strategy)
        except ValueError as e:
            raise ValueError(f"Invalid chunking strategy: {e}")

        
        # Chunk the document
        chunks = chunker.chunk(content, max_chunk_length)
        
        if not chunks:
            raise ValueError("Document content is empty or resulted in no chunks")
        
        # Prepare chunks for storage backend
        # Embeddings will be generated by Chroma automatically (or via API)
        stored_chunks: List[StoredChunk] = []
        created_at = datetime.now().isoformat()
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}:{i}"
            
            # Don't generate embeddings here - let Chroma handle it or use API
            # If backend needs embeddings, they should be generated via API
            stored_chunk = StoredChunk(
                id=chunk_id,
                content=chunk_text,
                embedding=None,  # Let Chroma generate embeddings automatically
                metadata={
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunking_strategy": chunking_strategy,
                    "created_at": created_at,
                    "updated_at": created_at,
                    **(metadata or {})
                }
            )
            stored_chunks.append(stored_chunk)
        
        # Store chunks in backend
        self.backend.add_chunks(stored_chunks, collection_name)
        
        return document_id
    
    def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: Optional[str] = None,
        max_chunk_length: int = 512,
        embedding: bool = True,
        collection_name: Optional[str] = None
    ) -> None:
        """
        Update an existing document.
        
        Args:
            document_id: Document ID to update
            content: New content (if provided)
            metadata: New metadata (merged with existing)
            chunking_strategy: New chunking strategy
            max_chunk_length: Maximum chunk length
            embedding: Whether to regenerate embeddings
            collection_name: Collection name (optional, will search all if not provided)
        
        Raises:
            ValueError: If document not found
        """
        # Find document in backend
        doc_data = self.retrieve_by_document_id(document_id, collection_name)
        if not doc_data:
            raise ValueError(f"Document '{document_id}' not found")
        
        coll_name = doc_data.get("collection_name")
        existing_metadata = doc_data.get("metadata", {})
        existing_strategy = doc_data.get("chunking_strategy", "sentence")
        existing_created_at = existing_metadata.get("created_at", datetime.now().isoformat())
        
        # Merge metadata
        updated_metadata = {**existing_metadata, **(metadata or {})}
        updated_strategy = chunking_strategy or existing_strategy
        
        # Get old chunks to determine collection and remove them
        collections_to_search = [coll_name] if coll_name else self.list_collections()
        old_chunk_ids = []
        
        for coll in collections_to_search:
            try:
                chunks = self.backend.list_chunks(coll)
                for chunk in chunks:
                    if chunk.metadata and chunk.metadata.get("document_id") == document_id:
                        old_chunk_ids.append(chunk.id)
                if old_chunk_ids:
                    self.backend.delete_chunks(old_chunk_ids, coll)
                    break
            except Exception:
                continue
        
        # Re-chunk if content provided
        if content is not None:
            chunker = ChunkingFactory.get_strategy(updated_strategy)
            chunks = chunker.chunk(content, max_chunk_length)
        else:
            # Keep existing chunks but update metadata
            existing_chunks = self.backend.list_chunks(coll_name)
            matching_chunks = [
                chunk for chunk in existing_chunks
                if chunk.metadata and chunk.metadata.get("document_id") == document_id
            ]
            matching_chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0) if c.metadata else 0)
            chunks = [chunk.content for chunk in matching_chunks]
        
        # Re-generate chunks and store in backend
        stored_chunks: List[StoredChunk] = []
        updated_at = datetime.now().isoformat()
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}:{i}"
            
            stored_chunk = StoredChunk(
                id=chunk_id,
                content=chunk_text,
                embedding=None,  # Let Chroma generate embeddings automatically
                metadata={
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunking_strategy": updated_strategy,
                    "created_at": existing_created_at,
                    "updated_at": updated_at,
                    **updated_metadata
                }
            )
            stored_chunks.append(stored_chunk)
        
        self.backend.add_chunks(stored_chunks, coll_name)
    
    def delete_document(self, document_id: str, collection_name: Optional[str] = None) -> None:
        """
        Delete a document from the database.
        
        Args:
            document_id: Document ID to delete
            collection_name: Optional collection name (will search all if not provided)
        
        Raises:
            ValueError: If document not found
        """
        # Find document in backend
        doc_data = self.retrieve_by_document_id(document_id, collection_name)
        if not doc_data:
            raise ValueError(f"Document '{document_id}' not found")
        
        coll_name = doc_data.get("collection_name")
        
        # Find all chunks for this document
        collections_to_search = [coll_name] if coll_name else self.list_collections()
        chunk_ids = []
        
        for coll in collections_to_search:
            try:
                chunks = self.backend.list_chunks(coll)
                for chunk in chunks:
                    if chunk.metadata and chunk.metadata.get("document_id") == document_id:
                        chunk_ids.append(chunk.id)
                if chunk_ids:
                    self.backend.delete_chunks(chunk_ids, coll)
                    break
            except Exception:
                continue
        
        if not chunk_ids:
            raise ValueError(f"Document '{document_id}' not found")
    
    def get_document(self, document_id: str, collection_name: Optional[str] = None) -> Optional[DocumentRecord]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID
            collection_name: Optional collection name (will search all if not provided)
        
        Returns:
            DocumentRecord or None if not found
        """
        doc_data = self.retrieve_by_document_id(document_id, collection_name)
        if not doc_data:
            return None
        
        # Reconstruct DocumentRecord from retrieved data
        # Need to get chunks to reconstruct full content
        coll_name = doc_data.get("collection_name")
        chunks = self.backend.list_chunks(coll_name)
        matching_chunks = [
            chunk for chunk in chunks
            if chunk.metadata and chunk.metadata.get("document_id") == document_id
        ]
        matching_chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0) if c.metadata else 0)
        
        return DocumentRecord(
            document_id=document_id,
            content=doc_data.get("full_document", ""),
            chunks=[chunk.content for chunk in matching_chunks],
            metadata=doc_data.get("metadata", {}),
            chunking_strategy=doc_data.get("chunking_strategy", "sentence"),
            collection_name=coll_name,
            created_at=doc_data.get("created_at", datetime.now().isoformat()),
            updated_at=doc_data.get("updated_at", datetime.now().isoformat())
        )
    
    def search_by_metadata(
        self,
        key: str,
        value: Any,
        collection_name: Optional[str] = None
    ) -> List[DocumentRecord]:
        """
        Search documents by metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
            collection_name: Optional collection name (will search all if not provided)
        
        Returns:
            List of matching documents
        """
        collections_to_search = [collection_name] if collection_name else self.list_collections()
        matching_doc_ids = set()
        
        for coll_name in collections_to_search:
            try:
                chunks = self.backend.list_chunks(coll_name)
                for chunk in chunks:
                    if chunk.metadata:
                        chunk_value = chunk.metadata.get(key)
                        # Handle string comparison for metadata values that might be stored as strings
                        if chunk_value == value or str(chunk_value) == str(value):
                            doc_id = chunk.metadata.get("document_id")
                            if doc_id:
                                matching_doc_ids.add(doc_id)
            except Exception:
                continue
        
        # Convert to DocumentRecord objects
        results = []
        for doc_id in matching_doc_ids:
            doc_record = self.get_document(doc_id, collection_name)
            if doc_record:
                results.append(doc_record)
        
        return results
    
    def reconstruct_document(
        self,
        document_id: str,
        chunk_indices: Optional[List[int]] = None,
        collection_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Reconstruct document content from chunks.
        
        Args:
            document_id: Document ID
            chunk_indices: Optional list of specific chunk indices to reconstruct.
                          If None, all chunks are used.
            collection_name: Optional collection name (will search all if not provided)
        
        Returns:
            Reconstructed content or None if document not found
        """
        # Get document data from backend
        doc_data = self.retrieve_by_document_id(document_id, collection_name)
        if not doc_data:
            return None
        
        coll_name = doc_data.get("collection_name")
        chunking_strategy = doc_data.get("chunking_strategy", "sentence")
        
        # Get chunks from backend
        chunks = self.backend.list_chunks(coll_name)
        matching_chunks = [
            chunk for chunk in chunks
            if chunk.metadata and chunk.metadata.get("document_id") == document_id
        ]
        matching_chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0) if c.metadata else 0)
        
        if chunk_indices is None:
            # Use all chunks
            chunks_to_use = [chunk.content for chunk in matching_chunks]
        else:
            # Use specific chunks in order
            chunks_to_use = [
                matching_chunks[i].content
                for i in chunk_indices
                if i < len(matching_chunks)
            ]
        
        # Reconstruct based on chunking strategy
        if chunking_strategy == "markdown":
            return '\n'.join(chunks_to_use)
        elif chunking_strategy == "paragraph":
            return '\n\n'.join(chunks_to_use)
        elif chunking_strategy == "line":
            return '\n'.join(chunks_to_use)
        elif chunking_strategy == "sentence":
            return ' '.join(chunks_to_use)
        else:  # character
            return ''.join(chunks_to_use)
    
    def retrieve_by_similarity(
        self,
        query_embedding: Optional[List[float]] = None,
        collection_name: str = "default",
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        reconstruct: bool = True,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by similarity to query.
        
        Args:
            query_embedding: Query embedding vector (optional, use query_text instead)
            collection_name: Collection to search in
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            reconstruct: Whether to reconstruct full documents
            query_text: Query text (Chroma will generate embedding automatically via API)
        
        Returns:
            List of retrieved results with documents
        """
        logger.info(f"Searching collection '{collection_name}' for {top_k} similar chunks")
        # Search in backend - use query_text if provided (Chroma will use API for embeddings)
        if hasattr(self.backend, 'search_by_similarity'):
            # Try to pass query_text if backend supports it
            try:
                results_from_backend = self.backend.search_by_similarity(
                    query_embedding=query_embedding,
                    collection_name=collection_name,
                    top_k=top_k,
                    threshold=similarity_threshold,
                    query_text=query_text
                )
            except TypeError:
                # Backend doesn't support query_text, use query_embedding only
                if query_embedding:
                    results_from_backend = self.backend.search_by_similarity(
                        query_embedding,
                        collection_name,
                        top_k,
                        similarity_threshold
                    )
                else:
                    logger.warning("No query_embedding or query_text provided")
                    return []
        else:
            logger.warning("Backend does not support search_by_similarity")
            return []
        
        results = []
        processed_docs = set()
        
        for chunk, similarity in results_from_backend:
            doc_id = chunk.metadata.get("document_id", "") if chunk.metadata else ""
            chunk_idx = chunk.metadata.get("chunk_index", 0) if chunk.metadata else 0
            
            if not doc_id:
                continue
            
            # Get metadata from chunk (excluding internal fields)
            chunk_metadata = chunk.metadata or {}
            doc_metadata = {
                k: v for k, v in chunk_metadata.items()
                if k not in ["document_id", "chunk_index"]
            }
            
            # Build result
            result = {
                "document_id": doc_id,
                "chunk_index": chunk_idx,
                "chunk_content": chunk.content,
                "similarity_score": float(similarity),
                "metadata": doc_metadata,
            }
            
            # Add reconstructed document if requested and not already added
            if reconstruct and doc_id not in processed_docs:
                result["full_document"] = self.reconstruct_document(doc_id, collection_name=collection_name)
                processed_docs.add(doc_id)
            
            results.append(result)
        
        return results
    
    def retrieve_by_collection(
        self,
        collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all documents in a collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            List of documents in the collection
        """
        logger.info(f"Retrieving all documents from collection: {collection_name}")
        chunks = self.backend.list_chunks(collection_name)
        logger.info(f"Found {len(chunks)} chunk(s) in collection: {collection_name}")
        
        results = []
        processed_docs = set()
        skipped_chunks = 0
        doc_chunk_counts = {}
        doc_metadata_map = {}
        
        for chunk in chunks:
            doc_id = chunk.metadata.get("document_id", "") if chunk.metadata else ""
            
            if not doc_id:
                skipped_chunks += 1
                continue
            
            # Count chunks per document
            doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
            
            # Store metadata from first chunk (should be same for all chunks of same document)
            if doc_id not in doc_metadata_map:
                chunk_metadata = chunk.metadata or {}
                doc_metadata_map[doc_id] = {
                    k: v for k, v in chunk_metadata.items()
                    if k not in ["document_id", "chunk_index"]
                }
            
            # Add each unique document once
            if doc_id not in processed_docs:
                result = {
                    "document_id": doc_id,
                    "num_chunks": doc_chunk_counts[doc_id],
                    "metadata": doc_metadata_map[doc_id],
                    "full_document": self.reconstruct_document(doc_id, collection_name=collection_name)
                }
                results.append(result)
                processed_docs.add(doc_id)
        
        logger.info(f"Retrieved {len(results)} unique document(s) from collection: {collection_name} (skipped {skipped_chunks} chunk(s) without valid document_id)")
        if results:
            logger.info(f"Document IDs: {[r['document_id'][:12] + '...' for r in results[:5]]}")
        
        return results
    
    def retrieve_by_document_id(
        self,
        document_id: str,
        collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by its ID.
        This function retrieves from ChromaDB directly, not from memory.
        Works even after restart.
        
        Args:
            document_id: ID of the document to retrieve
            collection_name: Optional collection name. If not provided, searches all collections.
        
        Returns:
            Dictionary with document data or None if not found
        """
        logger.info(f"Retrieving document by ID: {document_id}")
        
        # Search in backend
        logger.debug(f"Searching for document {document_id} in backend")
        
        # If collection_name not provided, search in all collections
        collections_to_search = [collection_name] if collection_name else self.list_collections()
        
        if not collections_to_search:
            logger.warning("No collections found to search")
            return None
        
        # Search through collections
        for coll_name in collections_to_search:
            try:
                chunks = self.backend.list_chunks(coll_name)
                logger.debug(f"Searching in collection '{coll_name}': found {len(chunks)} chunks")
                
                # Filter chunks by document_id
                matching_chunks = []
                for chunk in chunks:
                    chunk_doc_id = chunk.metadata.get("document_id", "") if chunk.metadata else ""
                    if chunk_doc_id == document_id:
                        matching_chunks.append(chunk)
                
                if matching_chunks:
                    logger.info(f"Found {len(matching_chunks)} chunk(s) for document {document_id} in collection '{coll_name}'")
                    
                    # Sort chunks by chunk_index
                    matching_chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0) if c.metadata else 0)
                    
                    # Extract metadata from first chunk (should be same for all chunks)
                    first_chunk_metadata = matching_chunks[0].metadata or {}
                    
                    # Get document metadata (excluding internal fields)
                    doc_metadata = {
                        k: v for k, v in first_chunk_metadata.items()
                        if k not in ["document_id", "chunk_index"]
                    }
                    
                    # Reconstruct document content
                    chunk_contents = [chunk.content for chunk in matching_chunks]
                    
                    # Determine chunking strategy from metadata or default
                    chunking_strategy = first_chunk_metadata.get("chunking_strategy", "sentence")
                    
                    # Reconstruct based on chunking strategy
                    if chunking_strategy == "markdown":
                        reconstructed = '\n'.join(chunk_contents)
                    elif chunking_strategy == "paragraph":
                        reconstructed = '\n\n'.join(chunk_contents)
                    elif chunking_strategy == "line":
                        reconstructed = '\n'.join(chunk_contents)
                    elif chunking_strategy == "sentence":
                        reconstructed = ' '.join(chunk_contents)
                    else:  # character
                        reconstructed = ''.join(chunk_contents)
                    
                    return {
                        "document_id": document_id,
                        "num_chunks": len(matching_chunks),
                        "metadata": doc_metadata,
                        "full_document": reconstructed,
                        "chunking_strategy": chunking_strategy,
                        "collection_name": coll_name,
                        "created_at": first_chunk_metadata.get("created_at"),
                        "updated_at": first_chunk_metadata.get("updated_at")
                    }
                    
            except Exception as e:
                logger.warning(f"Error searching in collection '{coll_name}': {e}")
                continue
        
        logger.warning(f"Document {document_id} not found in any collection")
        return None
    
    def list_documents(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all documents in the database.
        
        Args:
            collection_name: Optional collection name (will list all collections if not provided)
        
        Returns:
            List of document dictionaries
        """
        collections_to_search = [collection_name] if collection_name else self.list_collections()
        processed_docs = set()
        results = []
        
        for coll_name in collections_to_search:
            try:
                chunks = self.backend.list_chunks(coll_name)
                doc_chunk_counts = {}
                doc_metadata_map = {}
                
                for chunk in chunks:
                    doc_id = chunk.metadata.get("document_id", "") if chunk.metadata else ""
                    if not doc_id or doc_id in processed_docs:
                        continue
                    
                    doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
                    
                    if doc_id not in doc_metadata_map:
                        chunk_metadata = chunk.metadata or {}
                        doc_metadata_map[doc_id] = {
                            k: v for k, v in chunk_metadata.items()
                            if k not in ["document_id", "chunk_index"]
                        }
                
                for doc_id in doc_chunk_counts:
                    if doc_id not in processed_docs:
                        doc_data = self.retrieve_by_document_id(doc_id, coll_name)
                        if doc_data:
                            results.append({
                                "document_id": doc_id,
                                "num_chunks": doc_chunk_counts[doc_id],
                                "metadata": doc_metadata_map[doc_id],
                                "chunking_strategy": doc_data.get("chunking_strategy", "sentence"),
                                "collection_name": coll_name,
                                "created_at": doc_data.get("created_at"),
                                "updated_at": doc_data.get("updated_at")
                            })
                            processed_docs.add(doc_id)
            except Exception as e:
                logger.warning(f"Error listing documents from collection '{coll_name}': {e}")
                continue
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        Reads directly from backend to ensure accuracy, especially for remote ChromaDB.
        """
        collections = self.list_collections()
        
        # Collect statistics from backend
        total_chunks = 0
        total_content_length = 0
        document_info = {}  # {doc_id: {chunks: [], collection: str, strategy: str}}
        chunking_strategies = set()
        
        # Read chunks from all collections
        for collection_name in collections:
            try:
                chunks = self.backend.list_chunks(collection_name)
                
                for chunk in chunks:
                    total_chunks += 1
                    total_content_length += len(chunk.content)
                    
                    # Extract document info from chunk metadata
                    if chunk.metadata:
                        doc_id = chunk.metadata.get("document_id")
                        if doc_id:
                            if doc_id not in document_info:
                                document_info[doc_id] = {
                                    "chunks": [],
                                    "collection": collection_name,
                                    "strategy": chunk.metadata.get("chunking_strategy", "unknown")
                                }
                            document_info[doc_id]["chunks"].append(chunk)
                            
                            strategy = chunk.metadata.get("chunking_strategy", "unknown")
                            if strategy:
                                chunking_strategies.add(strategy)
            except Exception as e:
                logger.warning(f"Error reading chunks from collection '{collection_name}': {e}")
                continue
        
        total_documents = len(document_info)
        
        # Calculate document details
        document_details = []
        for doc_id, info in document_info.items():
            doc_chunks = info["chunks"]
            doc_content_length = sum(len(chunk.content) for chunk in doc_chunks)
            
            document_details.append({
                "id": doc_id,
                "num_chunks": len(doc_chunks),
                "content_length": doc_content_length,
                "strategy": info["strategy"],
                "collection": info["collection"],
            })
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "average_chunks_per_doc": total_chunks / total_documents if total_documents > 0 else 0,
            "average_chunk_length": total_content_length / total_chunks if total_chunks > 0 else 0,
            "collections": collections,
            "chunking_strategies": sorted(list(chunking_strategies)),
            "documents": document_details
        }
    
    def clear_all(self):
        """Clear all data from database."""
        self.backend.clear()
        self.collections.clear()
    
    def _generate_document_id(self) -> str:
        """Generate a unique document ID."""
        return f"doc_{uuid.uuid4().hex[:8]}"
