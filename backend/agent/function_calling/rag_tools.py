"""
RAG-specific tools for document retrieval and processing.
"""

from typing import List, Dict, Any, Optional
import json
from datetime import datetime


class VectorDatabaseRetriever:
    """Tool for retrieving documents from VectorDatabase."""
    
    def __init__(self, vector_db=None):
        """
        Initialize the VectorDatabase retriever.
        
        Args:
            vector_db: VectorDatabase instance
        """
        self.vector_db = vector_db
    
    def retrieve_by_similarity(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        reconstruct: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by similarity (requires embeddings).
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)
            reconstruct: Whether to reconstruct full documents
        
        Returns:
            List of retrieved chunks with full documents
        """
        """
        Real retrieval by similarity.

        Uses the configured embedding model from the VectorDatabase if available;
        otherwise falls back to a simple substring-based retrieval across documents
        (best-effort). Returns list of result dicts with similarity scores and
        optional reconstructed documents when requested.
        """
        if not self.vector_db:
            return []

        try:
            # Use Chroma's built-in embedding (via API or default model)
            # Chroma will automatically generate embeddings from query text
            if hasattr(self.vector_db.backend, 'search_by_similarity'):
                # Try to use query_text parameter if backend supports it
                try:
                    results = self.vector_db.backend.search_by_similarity(
                        query_embedding=None,
                        collection_name="default",
                        top_k=top_k,
                        threshold=similarity_threshold,
                        query_text=query
                    )
                    
                    # Convert backend results to expected format
                    formatted_results = []
                    for chunk, similarity in results:
                        doc_id = chunk.metadata.get("document_id", "") if chunk.metadata else ""
                        chunk_idx = chunk.metadata.get("chunk_index", 0) if chunk.metadata else 0
                        
                        if not doc_id or doc_id not in self.vector_db.documents:
                            continue
                        
                        doc_record = self.vector_db.documents[doc_id]
                        result = {
                            "document_id": doc_id,
                            "chunk_index": chunk_idx,
                            "chunk_content": chunk.content,
                            "similarity_score": float(similarity),
                            "metadata": doc_record.metadata,
                        }
                        
                        if reconstruct:
                            result["full_document"] = self.vector_db.reconstruct_document(doc_id)
                        
                        formatted_results.append(result)
                    
                    return formatted_results
                except TypeError:
                    # Backend doesn't support query_text, fall through to fallback
                    pass

            # Fallback: naive substring matching across documents
            results = []
            for doc in self.vector_db.list_documents():
                content = doc.get("content", "")
                if query.lower() in content.lower():
                    results.append({
                        "document_id": doc.get("document_id"),
                        "chunk_index": 0,
                        "chunk_content": content[:512],
                        "similarity_score": 1.0,
                        "metadata": doc.get("metadata", {}),
                        "full_document": content if reconstruct else None
                    })
                if len(results) >= top_k:
                    break

            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def retrieve_by_document_id(
        self,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: ID of the document
        
        Returns:
            Document data or None if not found
        """
        if not self.vector_db:
            # No vector DB configured — return None to indicate not found/unavailable
            return None
        
        try:
            doc = self.vector_db.get_document(document_id)
            if doc:
                reconstructed = self.vector_db.reconstruct_document(document_id)
                return {
                    "document_id": document_id,
                    "num_chunks": len(doc.chunks),
                    "content_preview": reconstructed[:500] if reconstructed else "",
                    "full_content": reconstructed,
                    "metadata": doc.metadata,
                    "chunking_strategy": doc.chunking_strategy,
                    "created_at": doc.created_at
                }
            return None
        except Exception as e:
            return {"error": str(e)}
    
    def search_by_metadata(
        self,
        key: str,
        value: Any
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        
        Returns:
            List of matching documents
        """
        if not self.vector_db:
            return []
        
        try:
            docs = self.vector_db.search_by_metadata(key, value)
            return [
                {
                    "document_id": doc.document_id,
                    "num_chunks": len(doc.chunks),
                    "metadata": doc.metadata,
                    "chunking_strategy": doc.chunking_strategy,
                }
                for doc in docs
            ]
        except Exception as e:
            return [{"error": str(e)}]
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database."""
        if not self.vector_db:
            return []
        
        try:
            return self.vector_db.list_documents()
        except Exception as e:
            return [{"error": str(e)}]
    
    def retrieve_from_collection(
        self,
        collection_name: str,
        top_k: int = 5,
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from a specific collection.
        
        Args:
            collection_name: Name of the collection to query
            top_k: Number of top documents to return
            query: Optional search query for similarity (if embeddings available)
        
        Returns:
            List of documents from the collection
        """
        if not self.vector_db:
            return []

        try:
            # Retrieve documents metadata from the VectorDatabase for that collection
            all_docs = self.vector_db.retrieve_by_collection(collection_name)
            if not all_docs:
                return []

            # If a query is provided, use Chroma's built-in embedding to search
            # Chroma will automatically generate embeddings from query text via API
            if query and hasattr(self.vector_db.backend, 'search_by_similarity'):
                try:
                    # Use backend's search_by_similarity with query_text
                    backend_results = self.vector_db.backend.search_by_similarity(
                        query_embedding=None,
                        collection_name=collection_name,
                        top_k=top_k,
                        threshold=0.0,
                        query_text=query
                    )
                    
                    # Convert to expected format
                    formatted_results = []
                    processed_docs = set()
                    for chunk, similarity in backend_results:
                        doc_id = chunk.metadata.get("document_id", "") if chunk.metadata else ""
                        if not doc_id or doc_id not in self.vector_db.documents:
                            continue
                        
                        if doc_id not in processed_docs:
                            doc_record = self.vector_db.documents[doc_id]
                            formatted_results.append({
                                "document_id": doc_id,
                                "num_chunks": len(doc_record.chunks),
                                "chunking_strategy": doc_record.chunking_strategy,
                                "metadata": doc_record.metadata,
                                "collection": collection_name,
                                "full_document": self.vector_db.reconstruct_document(doc_id),
                                "similarity_score": float(similarity)
                            })
                            processed_docs.add(doc_id)
                    
                    return formatted_results[:top_k]
                except (TypeError, AttributeError):
                    # Backend doesn't support query_text, fall through
                    pass

            results = []
            for doc in all_docs[:top_k]:
                results.append({
                    "document_id": doc.get("document_id", "unknown"),
                    "num_chunks": doc.get("num_chunks", 0),
                    "chunking_strategy": doc.get("chunking_strategy", "unknown"),
                    "metadata": doc.get("metadata", {}),
                    "collection": collection_name,
                    "full_document": doc.get("full_document")
                })

            return results
        except Exception as e:
            return [{"error": str(e), "collection": collection_name}]
    
    # Mock helper methods removed — production code should rely on a configured VectorDatabase.


class DocumentProcessor:
    """Tool for processing documents with various chunking strategies."""
    
    @staticmethod
    def add_document_to_db(
        vector_db,
        content: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "sentence",
        max_chunk_length: int = 512
    ) -> Dict[str, Any]:
        """
        Add a document to vector database.
        
        Args:
            vector_db: VectorDatabase instance
            content: Document content
            document_id: Optional document ID
            metadata: Document metadata
            chunking_strategy: Chunking strategy to use
            max_chunk_length: Maximum chunk length
        
        Returns:
            Information about added document
        """
        if not vector_db:
            return {"error": "Vector database not initialized"}
        
        try:
            doc_id = vector_db.add_document(
                content=content,
                document_id=document_id,
                metadata=metadata,
                chunking_strategy=chunking_strategy,
                max_chunk_length=max_chunk_length,
                embedding=False
            )
            
            doc = vector_db.get_document(doc_id)
            return {
                "success": True,
                "document_id": doc_id,
                "num_chunks": len(doc.chunks),
                "content_length": len(content),
                "chunking_strategy": chunking_strategy,
                "metadata": metadata or {}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def reconstruct_document(
        vector_db,
        document_id: str,
        chunk_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct a document from chunks.
        
        Args:
            vector_db: VectorDatabase instance
            document_id: Document ID to reconstruct
            chunk_indices: Optional specific chunks to reconstruct
        
        Returns:
            Reconstructed document with metadata
        """
        if not vector_db:
            return {"error": "Vector database not initialized"}
        
        try:
            reconstructed = vector_db.reconstruct_document(document_id, chunk_indices)
            
            if reconstructed is None:
                return {"error": f"Document '{document_id}' not found"}
            
            doc = vector_db.get_document(document_id)
            
            return {
                "success": True,
                "document_id": document_id,
                "content": reconstructed,
                "content_length": len(reconstructed),
                "num_chunks": len(doc.chunks),
                "reconstructed_chunks": chunk_indices if chunk_indices else "all",
                "metadata": doc.metadata
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def summarize(
        text: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Summarize text content.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
        
        Returns:
            Summarized text
        """
        # This is a placeholder - integrate with actual summarization model
        words = text.split()
        if max_length:
            words = words[:max(1, max_length // 6)]  # Rough estimate
        return " ".join(words[:50]) + "..."
    
    @staticmethod
    def extract_entities(text: str) -> List[Dict[str, str]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
        
        Returns:
            List of extracted entities
        """
        # This is a placeholder - integrate with NER model
        return [
            {
                "type": "placeholder",
                "value": "entity",
                "confidence": 0.95
            }
        ]
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 512,
        chunk_strategy: str = "character",
        overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Split text into chunks using specified strategy.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_strategy: Chunking strategy ('character', 'line', 'sentence', 'markdown', 'paragraph')
            overlap: Overlap between chunks (for character chunking)
        
        Returns:
            Chunks and metadata
        """
        try:
            from agent.memory.chunking import ChunkingFactory
            
            strategy = ChunkingFactory.get_strategy(chunk_strategy)
            chunks = strategy.chunk(text, chunk_size)
            
            return {
                "success": True,
                "strategy": chunk_strategy,
                "num_chunks": len(chunks),
                "chunks": chunks,
                "average_chunk_length": len(text) / len(chunks) if chunks else 0
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class DatabaseQuerier:
    """Tool for querying databases."""
    
    def __init__(self, db_connection=None):
        """
        Initialize the database querier.
        
        Args:
            db_connection: Database connection instance
        """
        self.db_connection = db_connection
    
    def query(
        self,
        sql: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Execute a database query.
        
        Args:
            sql: SQL query string
            limit: Maximum number of results
        
        Returns:
            Query results
        """
        if self.db_connection is None:
            return {"data": [], "count": 0, "message": "No database connected"}
        
        try:
            results = self.db_connection.execute(sql)
            return {
                "status": "success",
                "data": results[:limit],
                "count": len(results)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get database schema information."""
        if self.db_connection is None:
            return {"tables": []}
        
        try:
            schema = self.db_connection.get_schema()
            return {"tables": schema}
        except Exception as e:
            return {"error": str(e)}
