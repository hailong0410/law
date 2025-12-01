from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from app.routers import qa_router, articles_router

# App khởi tạo
app = FastAPI(
    title="Law AI API",
    description="API hệ thống hỏi – đáp luật PCCC + TTS",
    version="1.0.0",
)

# ===============================
#  CORS – Cho phép frontend truy cập
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Cho phép mọi domain (dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
#  Đăng ký router
# ===============================
app.include_router(qa_router.router, prefix="/QA", tags=["QA"])
app.include_router(articles_router.router, prefix="/Articles", tags=["Articles"])

# ===============================
#  Endpoint gốc để test nhanh
# ===============================
@app.get("/")
def root():
    return {"message": "Law AI backend is running!"}
