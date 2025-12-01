import json
from pathlib import Path

from app.database import SessionLocal, engine
from app import models
from app.models import Article

# Đảm bảo bảng được tạo (phòng trường hợp chưa chạy server)
models.Base.metadata.create_all(bind=engine)

BASE_DIR = Path(__file__).resolve().parents[1]
JSON_PATH = BASE_DIR / "data" / "parsed_law.json"


def import_data():
    db = SessionLocal()

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    inserted = 0
    skipped = 0

    for item in data:
        # kiểm tra đã tồn tại Điều này chưa
        existing = db.query(Article).filter_by(so_dieu=item["so_dieu"]).first()
        if existing:
            skipped += 1
            continue  # bỏ qua, không insert lại

        article = Article(
            so_dieu=item["so_dieu"],
            chuong=item["chuong"],
            tieu_de=item["tieu_de"],
            noi_dung=item["noi_dung"],
            tom_tat=None,
            tags=None,
        )
        db.add(article)
        inserted += 1

    db.commit()
    db.close()

    print(f"DONE – Imported {inserted} articles, skipped {skipped} existed articles.")


if __name__ == "__main__":
    import_data()
