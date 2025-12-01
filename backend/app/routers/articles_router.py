from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Article

router = APIRouter(prefix="/articles", tags=["Articles"])


@router.get("/")
def list_articles(keyword: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Article)

    if keyword:
        kw = f"%{keyword}%"
        query = query.filter(
            Article.tieu_de.ilike(kw) |
            Article.noi_dung.ilike(kw) |
            Article.chuong.ilike(kw)
        )

    return query.all()


@router.get("/{so_dieu}")
def get_article(so_dieu: int, db: Session = Depends(get_db)):
    article = db.query(Article).filter_by(so_dieu=so_dieu).first()
    if not article:
        raise HTTPException(status_code=404, detail="Không tìm thấy điều luật")
    return article
