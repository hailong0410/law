"""
Quick script to ingest the Fire Protection Law into ChromaDB.

Hardcoded configuration - just run it!
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent.memory import VectorDatabase


def main():
    print("=" * 70)
    print("  Ingesting Fire Protection Law into ChromaDB")
    print("=" * 70)
    print()
    
    # Hardcoded configuration
    law_file = Path(__file__).parent.parent / "storage" / "data" / "law.txt"
    collection_name = "fire_protection_law"
    
    # Always use absolute path based on script location (not current directory)
    project_root = Path(__file__).parent.parent  # /app/ in container
    default_persist_dir = project_root / "storage" / "chroma"
    
    # Get from env or use default absolute path
    persist_dir_str = os.getenv("CHROMA_PERSIST_DIRECTORY", str(default_persist_dir))
    persist_dir = Path(persist_dir_str).resolve()  # Convert to absolute path
    
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    
    # Check if file exists
    if not law_file.exists():
        print(f"❌ File not found: {law_file}")
        print(f"   Please make sure the law.txt file exists in storage/data/")
        return
    
    print(f"📄 File: {law_file}")
    print(f"📦 Collection: {collection_name}")
    print(f"💾 Storage (absolute): {persist_dir}")
    print(f"🔧 Embedding: {embedding_provider.upper()}")
    print()
    
    # Initialize embedding function
    print(f"🔧 Initializing {embedding_provider.upper()} Embedding...")
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
    print("🗄️  Initializing ChromaDB...")
    try:
        db = VectorDatabase(
            backend_type="chroma",
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
        print("   ✓ ChromaDB initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize database: {e}")
        return
    
    print()
    
    # Read the law file
    print("📖 Reading law document...")
    try:
        with open(law_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            print("   ❌ File is empty!")
            return
        
        print(f"   ✓ Read {len(content):,} characters")
    except Exception as e:
        print(f"   ❌ Failed to read file: {e}")
        return
    
    print()
    
    # Add document to database
    print("💾 Adding document to ChromaDB...")
    print("   (This may take a few minutes depending on document size)")
    print()
    
    try:
        doc_id = db.add_document(
            content=content,
            metadata={
                "title": "Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ",
                "law_number": "55/2024/QH15",
                "date": "29/11/2024",
                "type": "law",
                "source": str(law_file)
            },
            collection_name=collection_name,
            chunking_strategy="paragraph",
            max_chunk_length=512
        )
        
        print(f"   ✅ Document added successfully!")
        print(f"   📝 Document ID: {doc_id}")
        
        # Get document info
        doc = db.get_document(doc_id)
        if doc:
            print(f"   📊 Number of chunks: {len(doc.chunks)}")
            print(f"   📋 Metadata: {doc.metadata}")
        
    except Exception as e:
        print(f"   ❌ Failed to add document: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Verify with a test query
    print("🔍 Verifying ingestion with test query...")
    test_query = "nguyên tắc phòng cháy chữa cháy"
    print(f"   Query: '{test_query}'")
    print()
    
    try:
        results = db.retrieve_by_similarity(
            query_text=test_query,
            collection_name=collection_name,
            top_k=3
        )
        
        if results:
            print(f"   ✅ Found {len(results)} relevant chunks:")
            print()
            for i, res in enumerate(results, 1):
                score = res.get('similarity_score', 0)
                content_preview = res['chunk_content'][:150].replace('\n', ' ')
                print(f"   {i}. Score: {score:.4f}")
                print(f"      {content_preview}...")
                print()
        else:
            print("   ⚠️  No results found (this might indicate an issue)")
    
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
    
    print()
    print("=" * 70)
    print("  ✅ Ingestion Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
