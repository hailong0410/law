from pathlib import Path
from typing import List
from uuid import uuid4

import chromadb
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from gtts import gTTS
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# =========================
#   KHỞI TẠO CẤU HÌNH
# =========================

# __file__ = backend/app/routers/qa_router.py
# parents[0] = backend/app/routers
# parents[1] = backend/app
# => BASE_DIR = backend (nơi chứa chroma_db, tts_cache, law.db)
BASE_DIR = Path(__file__).resolve().parents[2]

CHROMA_DIR = BASE_DIR / "chroma_db"
TTS_DIR = BASE_DIR / "tts_cache"
TTS_DIR.mkdir(parents=True, exist_ok=True)

# Kết nối Chroma persistent client
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    collection = client.get_collection("law_articles")
except Exception as e:
    raise RuntimeError(
        "Không tìm thấy collection 'law_articles'. "
        "Hãy chạy: python -m scripts.build_vectorstore\n"
        f"Chi tiết: {e}"
    )

# Dùng cùng model với build_vectorstore
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedder = SentenceTransformer(MODEL_NAME)


# =========================
#   SCHEMA REQUEST/RESPONSE
# =========================

class AskRequest(BaseModel):
    question: str
    top_k: int = 5  # số đoạn muốn lấy


class ChunkAnswer(BaseModel):
    so_dieu: int
    chuong: str | None = None
    tieu_de: str | None = None
    chunk_text: str
    score: float  # điểm tương đồng (1 - distance)


class AskResponse(BaseModel):
    question: str
    results: List[ChunkAnswer]


# =========================
#   HÀM DÙNG CHUNG
# =========================

def retrieve_chunks(question: str, top_k: int = 5) -> AskResponse:
    """
    Nhận câu hỏi, truy vấn Chroma, trả về danh sách ChunkAnswer.
    """
    q = question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")

    # 1. Encode câu hỏi
    query_emb = embedder.encode([q]).tolist()

    # 2. Query Chroma
    try:
        res = collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi truy vấn Chroma: {e}")

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    results: list[ChunkAnswer] = []

    for doc, meta, dist in zip(docs, metas, dists):
        score = float(1.0 - dist)  # cosine distance -> similarity
        results.append(
            ChunkAnswer(
                so_dieu=meta.get("so_dieu"),
                chuong=meta.get("chuong"),
                tieu_de=meta.get("tieu_de"),
                chunk_text=doc,
                score=score,
            )
        )

    return AskResponse(question=q, results=results)


# =========================
#   ROUTES
# =========================

router = APIRouter(tags=["QA"])


@router.post("/ask", response_model=AskResponse)
def ask_law(req: AskRequest) -> AskResponse:
    """
    Nhận câu hỏi -> trả về các đoạn luật gần nhất (RAG retrieval).
    """
    return retrieve_chunks(req.question, req.top_k)


@router.post("/ask_tts")
def ask_law_tts(req: AskRequest):
    """
    Nhận câu hỏi -> tìm đoạn luật phù hợp nhất -> đọc ra file mp3.
    Trả về file audio/mp3 để client tải hoặc phát.
    """
    resp = retrieve_chunks(req.question, top_k=max(1, req.top_k))

    if not resp.results:
        text_to_read = (
            "Xin lỗi, tôi không tìm thấy điều luật phù hợp với câu hỏi của bạn."
        )
    else:
        best = resp.results[0]
        text_to_read = (
            f"Câu trả lời gợi ý dựa trên Luật phòng cháy chữa cháy. "
            f"Điều {best.so_dieu}, {best.tieu_de}. "
            f"Nội dung như sau: {best.chunk_text}"
        )

    # Tạo file mp3 tạm
    file_name = f"ans_{uuid4().hex}.mp3"
    file_path = TTS_DIR / file_name

    tts = gTTS(text=text_to_read, lang="vi")
    tts.save(str(file_path))

    # Trả về file audio
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=file_name,
    )
