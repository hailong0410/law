"""
Script để ingest các file section law vào ChromaDB.
Mỗi section sẽ được chia thành các document, mỗi document chứa 3 Điều.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent.memory import VectorDatabase


def find_dieu_positions(content: str) -> List[Tuple[int, int, str]]:
    """
    Tìm tất cả các vị trí của các "Điều" trong nội dung.
    
    Returns:
        List of tuples: (start_line_index, end_line_index, dieu_number)
        end_line_index là vị trí bắt đầu của Điều tiếp theo hoặc cuối file
    """
    lines = content.split('\n')
    dieu_positions = []
    
    # Pattern để tìm dòng bắt đầu bằng "Điều" theo sau là số
    pattern = re.compile(r'^Điều\s+(\d+)\.')
    
    for i, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match:
            dieu_number = match.group(1)
            # Tìm vị trí bắt đầu của Điều tiếp theo
            start_pos = i
            end_pos = len(lines)  # Mặc định là cuối file
            
            # Tìm Điều tiếp theo
            for j in range(i + 1, len(lines)):
                next_match = pattern.match(lines[j].strip())
                if next_match:
                    end_pos = j
                    break
            
            dieu_positions.append((start_pos, end_pos, dieu_number))
    
    return dieu_positions


def extract_dieu_content(content: str, start_line: int, end_line: int) -> str:
    """Trích xuất nội dung của một Điều từ start_line đến end_line."""
    lines = content.split('\n')
    return '\n'.join(lines[start_line:end_line])


def split_section_by_dieu(content: str, max_dieu_per_doc: int = 3) -> List[Tuple[str, List[str]]]:
    """
    Chia section thành các document, mỗi document chứa tối đa max_dieu_per_doc Điều.
    Document đầu tiên sẽ bao gồm phần header (nếu có) trước Điều đầu tiên.
    
    Returns:
        List of tuples: (document_content, list_of_dieu_numbers)
    """
    dieu_positions = find_dieu_positions(content)
    
    if not dieu_positions:
        # Nếu không tìm thấy Điều nào, trả về toàn bộ nội dung
        return [(content, [])]
    
    documents = []
    lines = content.split('\n')
    
    # Lấy vị trí của Điều đầu tiên để xác định phần header
    first_dieu_start = dieu_positions[0][0]
    
    # Chia thành các nhóm, mỗi nhóm có tối đa max_dieu_per_doc Điều
    for i in range(0, len(dieu_positions), max_dieu_per_doc):
        group = dieu_positions[i:i + max_dieu_per_doc]
        
        # Lấy vị trí bắt đầu của Điều đầu tiên trong nhóm
        start_line = group[0][0]
        
        # Nếu đây là document đầu tiên, bao gồm phần header (từ dòng 0 đến Điều đầu tiên)
        if i == 0 and first_dieu_start > 0:
            # Bao gồm phần header
            start_line = 0
        
        # Lấy vị trí kết thúc (bắt đầu của Điều tiếp theo sau nhóm, hoặc cuối file)
        if i + max_dieu_per_doc < len(dieu_positions):
            # Có Điều tiếp theo sau nhóm này
            end_line = dieu_positions[i + max_dieu_per_doc][0]
        else:
            # Đây là nhóm cuối cùng
            end_line = len(lines)
        
        # Trích xuất nội dung
        doc_content = '\n'.join(lines[start_line:end_line])
        
        # Lấy danh sách số Điều trong nhóm này
        dieu_numbers = [dieu[2] for dieu in group]
        
        documents.append((doc_content, dieu_numbers))
    
    return documents


def main():
    print("=" * 70)
    print("  Ingesting Section Law Files into ChromaDB")
    print("=" * 70)
    print()
    
    # Configuration
    data_dir = Path(__file__).parent.parent / "storage" / "data"
    collection_name = "fire_protection_law"
    # Sử dụng đường dẫn tuyệt đối dựa trên project root
    default_persist_dir = Path(__file__).parent.parent / "storage" / "chroma"
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", str(default_persist_dir))
    # Hỗ trợ kết nối qua HTTP nếu có CHROMA_HOST và CHROMA_PORT
    chroma_host = os.getenv("CHROMA_HOST", None)
    chroma_port = os.getenv("CHROMA_PORT", None)
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    max_dieu_per_doc = 3
    
    print(f"📁 Data directory: {data_dir}")
    print(f"📦 Collection: {collection_name}")
    if chroma_host and chroma_port:
        print(f"🌐 ChromaDB Server: {chroma_host}:{chroma_port}")
    else:
        print(f"💾 Storage: {persist_dir}")
    print(f"🔧 Embedding: {embedding_provider.upper()}")
    print(f"📄 Max Điều per document: {max_dieu_per_doc}")
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
        # Nếu có CHROMA_HOST và CHROMA_PORT, kết nối qua HTTP (server mode)
        # Ngược lại, dùng PersistentClient (local file mode)
        if chroma_host and chroma_port:
            db = VectorDatabase(
                backend_type="chroma",
                host=chroma_host,
                port=int(chroma_port),
                embedding_function=embedding_function
            )
            print(f"   ✓ Connected to ChromaDB server at {chroma_host}:{chroma_port}")
        else:
            db = VectorDatabase(
                backend_type="chroma",
                persist_directory=persist_dir,
                embedding_function=embedding_function
            )
            print(f"   ✓ ChromaDB initialized (local mode: {persist_dir})")
    except Exception as e:
        print(f"   ❌ Failed to initialize database: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Tìm tất cả các file section
    section_files = sorted(data_dir.glob("section*_law.txt"))
    
    if not section_files:
        print(f"   ❌ No section law files found in {data_dir}")
        return
    
    print(f"📖 Found {len(section_files)} section file(s)")
    print()
    
    total_documents = 0
    total_dieu_count = 0
    
    # Xử lý từng file section
    for section_file in section_files:
        print(f"📄 Processing: {section_file.name}")
        
        try:
            # Đọc nội dung file
            with open(section_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                print(f"   ⚠️  File is empty, skipping...")
                continue
            
            # Lấy số section từ tên file (ví dụ: section1_law.txt -> 1)
            section_match = re.search(r'section(\d+)_law', section_file.name)
            section_number = section_match.group(1) if section_match else "unknown"
            
            # Chia section thành các document
            documents = split_section_by_dieu(content, max_dieu_per_doc)
            
            print(f"   ✓ Found {len(documents)} document(s) in this section")
            
            # Thêm từng document vào database
            for doc_idx, (doc_content, dieu_numbers) in enumerate(documents):
                if not doc_content.strip():
                    continue
                
                # Tạo metadata
                metadata = {
                    "title": f"Luật Phòng cháy, chữa cháy và cứu nạn, cứu hộ - Section {section_number}",
                    "law_number": "55/2024/QH15",
                    "date": "29/11/2024",
                    "type": "law",
                    "section": section_number,
                    "section_file": section_file.name,
                    "document_index": doc_idx + 1,
                    "dieu_count": len(dieu_numbers),
                    "source": str(section_file)
                }
                
                try:
                    doc_id = db.add_document(
                        content=doc_content,
                        metadata=metadata,
                        collection_name=collection_name,
                        chunking_strategy="paragraph",
                        max_chunk_length=512
                    )
                    
                    total_documents += 1
                    total_dieu_count += len(dieu_numbers)
                    
                    print(f"      ✅ Document {doc_idx + 1}: Điều {', '.join(dieu_numbers) if dieu_numbers else 'N/A'} (ID: {doc_id[:8]}...)")
                    
                except Exception as e:
                    print(f"      ❌ Failed to add document {doc_idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()
            
            print()
            
        except Exception as e:
            print(f"   ❌ Failed to process file: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    print()
    print("=" * 70)
    print(f"  ✅ Ingestion Complete!")
    print(f"  📊 Total documents added: {total_documents}")
    print(f"  📊 Total Điều processed: {total_dieu_count}")
    print("=" * 70)
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


if __name__ == "__main__":
    main()

