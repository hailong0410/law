"""
Quick script to ingest the Fire Protection Law into ChromaDB.

This is a simplified version that directly ingests the law.txt file
from the Document folder.
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
    
    # Configuration
    law_file = Path(r"d:\chatbot\law\Document\law.txt")
    collection_name = "fire_protection_law"
    persist_dir = "./storage/chroma"
    
    # Check if file exists
    if not law_file.exists():
        print(f"❌ File not found: {law_file}")
        print("   Please make sure the law.txt file exists in d:\\chatbot\\law\\Document\\")
        return
    
    print(f"📄 File: {law_file}")
    print(f"📦 Collection: {collection_name}")
    print(f"💾 Storage: {persist_dir}")
    print()
    
    # Initialize embedding function
    print("🔧 Initializing Google Gemini Embedding...")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        print("   ❌ GEMINI_API_KEY not found in .env file!")
        print()
        print("   Please add the following to your .env file:")
        print("   GEMINI_API_KEY=your_gemini_api_key_here")
        print()
        print("   You can get a free API key from: https://makersuite.google.com/app/apikey")
        return
    
    print("   ✓ Using Google Gemini Embedding API")
    
    try:
        from agent.llm.gemini import get_google_embedding_function
        embedding_function = get_google_embedding_function(
            api_key=gemini_api_key,
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        )
        print("   ✓ Embedding function initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize Google Gemini embedding: {e}")
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
            chunking_strategy="paragraph",  # Use paragraph for legal documents
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
    print("📌 Next steps:")
    print("   1. Start your backend server: python main.py")
    print("   2. Ask questions about fire protection law")
    print("   3. The agent will automatically retrieve relevant information")
    print()


if __name__ == "__main__":
    main()
