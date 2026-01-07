"""
Script to ingest law documents into ChromaDB vector database.

This script provides utilities to:
1. Ingest law documents from text files
2. Add documents to specific collections
3. Verify the ingestion was successful

Usage:
    python scripts/ingest_data.py --file path/to/law.txt --collection fire_protection_law
"""

import os
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent.memory import VectorDatabase
from agent.logging import logger


def initialize_vector_db():
    """Initialize VectorDatabase with ChromaDB backend."""
    print("=" * 60)
    print("Initializing ChromaDB Vector Database...")
    print("=" * 60)
    
    # Get configuration from environment
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./storage/chroma")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        print("✗ GEMINI_API_KEY not found in .env file!")
        print()
        print("Please add the following to your .env file:")
        print("GEMINI_API_KEY=your_gemini_api_key_here")
        print()
        print("Get a free API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Initialize Google Gemini embedding function
    print(f"✓ Using Google Gemini Embedding API")
    
    try:
        from agent.llm.gemini import get_google_embedding_function
        embedding_function = get_google_embedding_function(
            api_key=gemini_api_key,
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
        )
    except Exception as e:
        print(f"✗ Failed to initialize Google Gemini embedding: {e}")
        sys.exit(1)
    
    # Create VectorDatabase
    print(f"✓ Persist directory: {persist_dir}")
    
    db = VectorDatabase(
        backend_type="chroma",
        persist_directory=persist_dir,
        embedding_function=embedding_function
    )
    
    print("✓ VectorDatabase initialized successfully")
    print()
    return db


def ingest_file(db, file_path, collection_name, metadata=None, chunking_strategy="paragraph"):
    """
    Ingest a file into the vector database.
    
    Args:
        db: VectorDatabase instance
        file_path: Path to the file to ingest
        collection_name: Name of the collection to add to
        metadata: Optional metadata dict
        chunking_strategy: Chunking strategy (default: paragraph)
    """
    print("=" * 60)
    print(f"Ingesting file: {file_path}")
    print("=" * 60)
    
    # Check if file exists
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return None
    
    # Read file content
    print(f"✓ Reading file...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"✗ Failed to read file: {e}")
        return None
    
    if not content.strip():
        print(f"✗ File is empty")
        return None
    
    print(f"✓ File size: {len(content)} characters")
    
    # Add to database
    print(f"✓ Collection: {collection_name}")
    print(f"✓ Chunking strategy: {chunking_strategy}")
    print(f"✓ Adding document to vector database...")
    
    try:
        doc_id = db.add_document(
            content=content,
            metadata=metadata or {},
            collection_name=collection_name,
            chunking_strategy=chunking_strategy,
            max_chunk_length=512
        )
        
        print(f"✓ Document added successfully!")
        print(f"  Document ID: {doc_id}")
        
        # Get document info
        doc = db.get_document(doc_id)
        if doc:
            print(f"  Number of chunks: {len(doc.chunks)}")
            print(f"  Metadata: {doc.metadata}")
        
        return doc_id
        
    except Exception as e:
        print(f"✗ Failed to add document: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_ingestion(db, collection_name, test_query):
    """
    Verify the ingestion by running a test query.
    
    Args:
        db: VectorDatabase instance
        collection_name: Collection to query
        test_query: Test query string
    """
    print()
    print("=" * 60)
    print("Verifying ingestion with test query...")
    print("=" * 60)
    print(f"Query: '{test_query}'")
    print()
    
    try:
        results = db.retrieve_by_similarity(
            query_text=test_query,
            collection_name=collection_name,
            top_k=3
        )
        
        if results:
            print(f"✓ Found {len(results)} relevant chunks:")
            print()
            for i, res in enumerate(results, 1):
                print(f"--- Result {i} ---")
                print(f"Document ID: {res.get('document_id', 'N/A')}")
                print(f"Similarity Score: {res.get('similarity_score', 'N/A'):.4f}")
                preview = res['chunk_content'][:200].replace('\n', ' ')
                print(f"Content Preview: {preview}...")
                print()
        else:
            print("✗ No results found for test query")
            
    except Exception as e:
        print(f"✗ Query failed: {e}")
        import traceback
        traceback.print_exc()


def list_collections(db):
    """List all collections in the database."""
    print("=" * 60)
    print("Available Collections:")
    print("=" * 60)
    
    collections = db.list_collections()
    if collections:
        for i, col in enumerate(collections, 1):
            print(f"{i}. {col}")
    else:
        print("No collections found")
    print()


def main():
    parser = argparse.ArgumentParser(description="Ingest law documents into ChromaDB")
    parser.add_argument("--file", "-f", type=str, help="Path to the file to ingest")
    parser.add_argument("--collection", "-c", type=str, default="default", help="Collection name")
    parser.add_argument("--title", "-t", type=str, help="Document title (metadata)")
    parser.add_argument("--law-number", type=str, help="Law number (metadata)")
    parser.add_argument("--date", type=str, help="Law date (metadata)")
    parser.add_argument("--chunking", type=str, default="paragraph", 
                       choices=["sentence", "paragraph", "line", "markdown", "character"],
                       help="Chunking strategy")
    parser.add_argument("--query", "-q", type=str, help="Test query to verify ingestion")
    parser.add_argument("--list", "-l", action="store_true", help="List all collections")
    
    args = parser.parse_args()
    
    # Initialize database
    db = initialize_vector_db()
    
    # List collections if requested
    if args.list:
        list_collections(db)
        return
    
    # Ingest file if provided
    if args.file:
        # Prepare metadata
        metadata = {}
        if args.title:
            metadata["title"] = args.title
        if args.law_number:
            metadata["law_number"] = args.law_number
        if args.date:
            metadata["date"] = args.date
        metadata["type"] = "law"
        
        # Ingest file
        doc_id = ingest_file(
            db=db,
            file_path=args.file,
            collection_name=args.collection,
            metadata=metadata,
            chunking_strategy=args.chunking
        )
        
        # Verify if test query provided
        if doc_id and args.query:
            verify_ingestion(db, args.collection, args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
