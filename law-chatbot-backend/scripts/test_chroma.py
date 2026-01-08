"""
Script to test and inspect ChromaDB information.

Shows:
- All collections
- Number of records (chunks) in each collection
- Document statistics
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent.memory import VectorDatabase


def test_chroma():
    print("=" * 70)
    print("  ChromaDB Information Test")
    print("=" * 70)
    print()
    
    # Configuration
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./storage/chroma")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    
    print(f"💾 Storage: {persist_dir}")
    print(f"🔧 Embedding: {embedding_provider.upper()}")
    print()
    
    # Initialize embedding function
    print("🔧 Initializing embedding function...")
    embedding_function = None
    
    try:
        if embedding_provider == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("   ❌ OPENAI_API_KEY not found in .env file!")
                return
            from agent.llm.openai import get_openai_embedding_function
            embedding_function = get_openai_embedding_function(
                api_key=openai_api_key,
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
        else:  # Default to Gemini
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                print("   ❌ GEMINI_API_KEY not found in .env file!")
                return
            from agent.llm.gemini import get_google_embedding_function
            embedding_function = get_google_embedding_function(
                api_key=gemini_api_key,
                model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
            )
        print("   ✓ Embedding function initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize embedding: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Initialize VectorDatabase
    print("🗄️  Connecting to ChromaDB...")
    try:
        db = VectorDatabase(
            backend_type="chroma",
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
        print("   ✓ Connected to ChromaDB")
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # List all collections
    print("📦 Collections:")
    print("-" * 70)
    try:
        collections = db.list_collections()
        
        if not collections:
            print("   ⚠️  No collections found")
        else:
            print(f"   Found {len(collections)} collection(s):")
            print()
            
            total_chunks = 0
            total_documents = 0
            
            for collection_name in collections:
                print(f"   📁 Collection: {collection_name}")
                
                # Count chunks in this collection
                try:
                    chunks = db.backend.list_chunks(collection_name)
                    chunk_count = len(chunks)
                    total_chunks += chunk_count
                    
                    # Count unique documents
                    doc_ids = set()
                    for chunk in chunks:
                        if chunk.metadata and "document_id" in chunk.metadata:
                            doc_ids.add(chunk.metadata["document_id"])
                    
                    doc_count = len(doc_ids)
                    total_documents += doc_count
                    
                    print(f"      📊 Records (chunks): {chunk_count:,}")
                    print(f"      📄 Documents: {doc_count}")
                    
                    # Show document details
                    if doc_ids:
                        print(f"      📋 Documents ({len(doc_ids)}):")
                        for doc_id in sorted(doc_ids):
                            doc = db.documents.get(doc_id)
                            if doc:
                                print(f"         • Document ID: {doc_id}")
                                print(f"           - Chunks: {len(doc.chunks)}")
                                print(f"           - Strategy: {doc.chunking_strategy}")
                                print(f"           - Created: {doc.created_at}")
                                if doc.metadata:
                                    title = doc.metadata.get("title", "N/A")
                                    law_number = doc.metadata.get("law_number", "")
                                    print(f"           - Title: {title}")
                                    if law_number:
                                        print(f"           - Law Number: {law_number}")
                    
                except Exception as e:
                    print(f"      ❌ Error reading collection: {e}")
                
                print()
            
            print("-" * 70)
            print(f"   📊 Total Collections: {len(collections)}")
            print(f"   📊 Total Records (chunks): {total_chunks:,}")
            print(f"   📊 Total Documents: {total_documents}")
            
    except Exception as e:
        print(f"   ❌ Error listing collections: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Get statistics
    print("📈 Database Statistics:")
    print("-" * 70)
    try:
        stats = db.get_statistics()
        print(f"   Total Documents: {stats['total_documents']}")
        print(f"   Total Chunks: {stats['total_chunks']:,}")
        print(f"   Total Content Length: {stats['total_content_length']:,} characters")
        print(f"   Average Chunks per Document: {stats['average_chunks_per_doc']:.2f}")
        print(f"   Average Chunk Length: {stats['average_chunk_length']:.2f} characters")
        print(f"   Chunking Strategies: {', '.join(stats['chunking_strategies'])}")
        
        if stats['documents']:
            print()
            print("   Document Details:")
            for doc in stats['documents']:
                print(f"      • Document ID: {doc['id']}")
                print(f"        - Collection: {doc['collection']}")
                print(f"        - Chunks: {doc['num_chunks']}")
                print(f"        - Content Length: {doc['content_length']:,} chars")
                print(f"        - Strategy: {doc['strategy']}")
        
    except Exception as e:
        print(f"   ❌ Error getting statistics: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 70)
    print("  ✅ Test Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test_chroma()

