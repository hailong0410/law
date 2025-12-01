import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Article


BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "chroma_db"


# --------- HÀM CẮT VĂN BẢN THÀNH CHUNK NGẮN ---------
def split_text(text: str, max_chars: int = 500) -> list[str]:
    """
    Cắt nội dung điều luật thành các đoạn (~max_chars ký tự).
    Ưu tiên cắt theo xuống dòng; nếu đoạn quá dài thì cắt tiếp.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []

    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            # cắt tiếp nếu đoạn quá dài
            start = 0
            while start < len(p):
                end = start + max_chars
                chunks.append(p[start:end].strip())
                start = end

    return chunks or ["(trống)"]


# --------- HÀM BUILD VECTORSTORE ---------
def build_vectorstore():
    # 1. Lấy dữ liệu từ database
    db: Session = SessionLocal()
    articles = db.query(Article).all()
    print(f"Found {len(articles)} articles in database.")

    # 2. Khởi tạo thư mục Chroma
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Kết nối client
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 4. Xóa collection cũ (nếu có)
    try:
        client.delete_collection(name="law_articles")
        print("Old collection 'law_articles' deleted.")
    except Exception:
        print("No old collection found. Creating new one.")

    # 5. Tạo collection mới
    collection = client.create_collection(
        name="law_articles",
        metadata={"hnsw:space": "cosine"},
    )

    # 6. Load embedding model
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    all_ids = []
    all_texts = []
    all_metadatas = []

    # 7. Cắt chunk và gom dữ liệu
    for article in articles:
        chunks = split_text(article.noi_dung, max_chars=500)

        for idx, chunk in enumerate(chunks):
            doc_id = f"{article.id}_{idx}"

            all_ids.append(doc_id)
            all_texts.append(chunk)
            all_metadatas.append({
                "article_id": article.id,
                "so_dieu": article.so_dieu,
                "chuong": article.chuong,
                "tieu_de": article.tieu_de,
                "chunk_index": idx,
            })

    print(f"Total chunks: {len(all_texts)}")

    # 8. Tính embedding
    print("Encoding embeddings...")
    embeddings = embedder.encode(all_texts, show_progress_bar=True).tolist()

    # 9. Thêm dữ liệu vào Chroma
    print("Adding documents to Chroma...")
    collection.add(
        ids=all_ids,
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    print("DONE – Vectorstore built successfully!")
    db.close()


if __name__ == "__main__":
    build_vectorstore()
