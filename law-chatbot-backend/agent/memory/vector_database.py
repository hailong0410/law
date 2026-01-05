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
        
        # In-memory document metadata (always kept in memory)
        self.documents: Dict[str, DocumentRecord] = {}
        self.metadata_index: Dict[str, List[str]] = defaultdict(list)
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
            # Remove documents from this collection
            doc_ids_to_remove = [
                doc_id for doc_id, doc in self.documents.items()
                if doc.collection_name == collection_name
            ]
            for doc_id in doc_ids_to_remove:
                del self.documents[doc_id]
            logger.info(f"Collection '{collection_name}' deleted successfully. Removed {len(doc_ids_to_remove)} documents")
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
        
        # Check if document already exists
        if document_id in self.documents:
            raise ValueError(f"Document with ID '{document_id}' already exists.")
        
        # Validate chunking strategy
        try:
            chunker = ChunkingFactory.get_strategy(chunking_strategy)
        except ValueError as e:
            raise ValueError(f"Invalid chunking strategy: {e}")

        
        # Chunk the document
        chunks = chunker.chunk(content, max_chunk_length)
        
        if not chunks:
            raise ValueError("Document content is empty or resulted in no chunks")
        
        # Create document record
        doc_record = DocumentRecord(
            document_id=document_id,
            content=content,
            chunks=chunks,
            metadata=metadata or {},
            chunking_strategy=chunking_strategy,
            collection_name=collection_name
        )
        
        # Prepare chunks for storage backend
        # Embeddings will be generated by Chroma automatically (or via API)
        stored_chunks: List[StoredChunk] = []
        
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
                    **(metadata or {})
                }
            )
            stored_chunks.append(stored_chunk)
        
        # Store chunks in backend
        self.backend.add_chunks(stored_chunks, collection_name)
        
        # Store document metadata
        self.documents[document_id] = doc_record
        
        # Index metadata
        for key, value in (metadata or {}).items():
            self.metadata_index[f"{key}:{value}"].append(document_id)
        
        return document_id
    
    def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: Optional[str] = None,
        max_chunk_length: int = 512,
        embedding: bool = True
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
        
        Raises:
            ValueError: If document not found
        """
        if document_id not in self.documents:
            raise ValueError(f"Document '{document_id}' not found")
        
        doc_record = self.documents[document_id]
        
        # Remove old chunks from backend
        old_chunk_ids = [f"{document_id}:{i}" for i in range(len(doc_record.chunks))]
        self.backend.delete_chunks(old_chunk_ids, doc_record.collection_name)
        
        # Update content if provided
        if content is not None:
            doc_record.content = content
            doc_record.chunking_strategy = chunking_strategy or doc_record.chunking_strategy
            
            # Re-chunk document
            chunker = ChunkingFactory.get_strategy(doc_record.chunking_strategy)
            doc_record.chunks = chunker.chunk(content, max_chunk_length)
        
        # Update metadata
        if metadata:
            doc_record.metadata.update(metadata)
        
        # Update timestamp
        doc_record.updated_at = datetime.now().isoformat()
        
        # Re-generate chunks and store in backend
        # Embeddings will be generated by Chroma automatically (or via API)
        stored_chunks: List[StoredChunk] = []
        doc_record.chunk_embeddings = []
        
        for i, chunk_text in enumerate(doc_record.chunks):
            chunk_id = f"{document_id}:{i}"
            
            # Don't generate embeddings here - let Chroma handle it or use API
            stored_chunk = StoredChunk(
                id=chunk_id,
                content=chunk_text,
                embedding=None,  # Let Chroma generate embeddings automatically
                metadata={
                    "document_id": document_id,
                    "chunk_index": i,
                    **(doc_record.metadata or {})
                }
            )
            stored_chunks.append(stored_chunk)
        
        self.backend.add_chunks(stored_chunks, doc_record.collection_name)
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete a document from the database.
        
        Args:
            document_id: Document ID to delete
        
        Raises:
            ValueError: If document not found
        """
        if document_id not in self.documents:
            raise ValueError(f"Document '{document_id}' not found")
        
        doc_record = self.documents[document_id]
        
        # Remove chunks from backend
        chunk_ids = [f"{document_id}:{i}" for i in range(len(doc_record.chunks))]
        self.backend.delete_chunks(chunk_ids, doc_record.collection_name)
        
        # Remove metadata indices
        for key, value in doc_record.metadata.items():
            metadata_key = f"{key}:{value}"
            if metadata_key in self.metadata_index:
                if document_id in self.metadata_index[metadata_key]:
                    self.metadata_index[metadata_key].remove(document_id)
        
        # Remove document
        del self.documents[document_id]
    
    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID
        
        Returns:
            DocumentRecord or None if not found
        """
        return self.documents.get(document_id)
    
    def search_by_metadata(
        self,
        key: str,
        value: Any
    ) -> List[DocumentRecord]:
        """
        Search documents by metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        
        Returns:
            List of matching documents
        """
        metadata_key = f"{key}:{value}"
        doc_ids = self.metadata_index.get(metadata_key, [])
        return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]
    
    def reconstruct_document(
        self,
        document_id: str,
        chunk_indices: Optional[List[int]] = None
    ) -> Optional[str]:
        """
        Reconstruct document content from chunks.
        
        Args:
            document_id: Document ID
            chunk_indices: Optional list of specific chunk indices to reconstruct.
                          If None, all chunks are used.
        
        Returns:
            Reconstructed content or None if document not found
        """
        doc_record = self.documents.get(document_id)
        if not doc_record:
            return None
        
        if chunk_indices is None:
            # Use all chunks
            chunks_to_use = doc_record.chunks
        else:
            # Use specific chunks in order
            chunks_to_use = [
                doc_record.chunks[i]
                for i in chunk_indices
                if i < len(doc_record.chunks)
            ]
        
        # Reconstruct based on chunking strategy
        if doc_record.chunking_strategy == "markdown":
            return '\n'.join(chunks_to_use)
        elif doc_record.chunking_strategy == "paragraph":
            return '\n\n'.join(chunks_to_use)
        elif doc_record.chunking_strategy == "line":
            return '\n'.join(chunks_to_use)
        elif doc_record.chunking_strategy == "sentence":
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
            
            if not doc_id or doc_id not in self.documents:
                continue
            
            doc_record = self.documents[doc_id]
            
            # Build result
            result = {
                "document_id": doc_id,
                "chunk_index": chunk_idx,
                "chunk_content": chunk.content,
                "similarity_score": float(similarity),
                "metadata": doc_record.metadata,
            }
            
            # Add reconstructed document if requested and not already added
            if reconstruct and doc_id not in processed_docs:
                result["full_document"] = self.reconstruct_document(doc_id)
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
        
        results = []
        processed_docs = set()
        
        for chunk in chunks:
            doc_id = chunk.metadata.get("document_id", "") if chunk.metadata else ""
            
            if not doc_id or doc_id not in self.documents:
                continue
            
            doc_record = self.documents[doc_id]
            
            # Add each unique document once
            if doc_id not in processed_docs:
                result = {
                    "document_id": doc_id,
                    "num_chunks": len(doc_record.chunks),
                    "metadata": doc_record.metadata,
                    "full_document": self.reconstruct_document(doc_id)
                }
                results.append(result)
                processed_docs.add(doc_id)
        
        return results
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database."""
        return [doc.to_dict() for doc in self.documents.values()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        total_chunks = sum(len(doc.chunks) for doc in self.documents.values())
        total_content_length = sum(len(doc.content) for doc in self.documents.values())
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "average_chunks_per_doc": total_chunks / len(self.documents) if self.documents else 0,
            "average_chunk_length": total_content_length / total_chunks if total_chunks > 0 else 0,
            "collections": self.list_collections(),
            "chunking_strategies": list(set(
                doc.chunking_strategy for doc in self.documents.values()
            )),
            "documents": [
                {
                    "id": doc.document_id,
                    "num_chunks": len(doc.chunks),
                    "content_length": len(doc.content),
                    "strategy": doc.chunking_strategy,
                    "collection": doc.collection_name,
                }
                for doc in self.documents.values()
            ]
        }
    
    def clear_all(self):
        """Clear all data from database."""
        self.backend.clear()
        self.documents.clear()
        self.metadata_index.clear()
        self.collections.clear()
    
    def _generate_document_id(self) -> str:
        """Generate a unique document ID."""
        return f"doc_{uuid.uuid4().hex[:8]}"
