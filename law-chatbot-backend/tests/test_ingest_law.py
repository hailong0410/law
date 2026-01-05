
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.memory.vector_database import VectorDatabase
from agent.logging import logger

def test_ingest_fire_protection_law():
    """
    Test function to ingest the Fire Protection Law document into the vector database
    in the 'fire_protection_law' collection.
    """
    try:
        # 1. Initialize Vector Database
        # Using persistent storage if possible would be better, but we stick to default implementation for test
        # If the actual app uses a persistent backend (e.g. Chroma with persist_directory), 
        # this test should use the same configuration to be useful.
        # Assuming default for now as per codebase inspection.
        print("Initializing Vector Database...")
        db = VectorDatabase() 
        
        # 2. Read the document
        file_path = Path(r"d:\chatbot\law\Document\law.txt")
        if not file_path.exists():
            print(f"Error: File not found at {file_path}")
            return
            
        print(f"Reading document from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if not content:
            print("Error: Document is empty")
            return
            
        # 3. Add to Vector Database
        collection_name = "fire_protection_law"
        print(f"Adding document to collection '{collection_name}'...")
        
        doc_id = db.add_document(
            content=content,
            metadata={
                "title": "Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ",
                "law_number": "55/2024/QH15",
                "date": "29/11/2024",
                "type": "law"
            },
            collection_name=collection_name,
            chunking_strategy="paragraph" # Paragraph might be better for legal text structure
        )
        
        print(f"Successfully added document with ID: {doc_id}")
        
        # 4. Verify ingestion by simple query
        print("\nVerifying ingestion with a test query...")
        results = db.retrieve_by_similarity(
            query_text="nguyên tắc phòng cháy chữa cháy",
            collection_name=collection_name,
            top_k=3
        )
        
        if results:
            print(f"Found {len(results)} relevant chunks:")
            for i, res in enumerate(results):
                print(f"--- Result {i+1} (Score: {res.get('similarity_score', 'N/A')}) ---")
                preview = res['chunk_content'][:200].replace('\n', ' ')
                print(f"Content: {preview}...")
        else:
            print("Warning: No results found for test query.")
            
        print("\nTest completed successfully.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ingest_fire_protection_law()
