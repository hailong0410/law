import json
import re
from docx import Document


def read_docx_to_text(path: str) -> str:
    """Đọc toàn bộ nội dung file .docx thành 1 chuỗi text."""
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def split_into_articles(text: str):
    """
    Tách văn bản thành từng Điều.
    Mỗi điều có dạng 'Điều X.' đứng đầu dòng.
    Trả về list các dict: {so_dieu, tieu_de, noi_dung, chuong}.
    """
    # Regex nhận diện dòng Điều
    article_pattern = r"(Điều\s+\d+\.*)"

    # Tìm tất cả các vị trí "Điều X."
    parts = re.split(article_pattern, text)

    # parts sẽ theo dạng:  ["trước điều 1", "Điều 1", "nội dung 1", "Điều 2", "nội dung 2", ...]
    # nên phải ghép cặp 2 phần 1 lần
    articles = []

    current_chapter = ""

    # Regex nhận diện dòng CHƯƠNG (ví dụ: "Chương II", "Chương III. QUY ĐỊNH ...")
    chapter_pattern = r"Chương\s+[IVXLC]+\s*.*"

    # Nếu phần đầu có thông tin CHƯƠNG
    if parts and re.search(chapter_pattern, parts[0], re.IGNORECASE):
        current_chapter = re.search(chapter_pattern, parts[0], re.IGNORECASE).group().strip()

    # Bắt đầu từ index 1 để duyệt Điều
    for i in range(1, len(parts), 2):
        dieu_header = parts[i].strip()           # "Điều 20."
        dieu_text = parts[i+1].strip() if i+1 < len(parts) else ""

        # Lấy số điều
        so_dieu = int(re.findall(r"\d+", dieu_header)[0])

        # Kiểm tra trong nội dung có dòng CHƯƠNG không (cập nhật chương hiện tại)
        chapter_match = re.search(chapter_pattern, dieu_text, re.IGNORECASE)
        if chapter_match:
            current_chapter = chapter_match.group().strip()

        # Tách tiêu đề: dòng đầu tiên ngay sau "Điều 20."
        lines = [line for line in dieu_text.split("\n") if line.strip()]
        if not lines:
            tieu_de = ""
            noi_dung = ""
        else:
            tieu_de = lines[0].strip()
            noi_dung = "\n".join(lines[1:]).strip()

        articles.append({
            "so_dieu": so_dieu,
            "chuong": current_chapter,
            "tieu_de": tieu_de,
            "noi_dung": noi_dung
        })

    return articles


def save_json(data, output_path: str):
    """Lưu list dict vào file JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    INPUT_DOC = "../data/luat_phong_chay.docx"
    OUTPUT_JSON = "../data/parsed_law.json"

    print("Đang đọc file Word...")
    text = read_docx_to_text(INPUT_DOC)

    print("Đang tách các Điều...")
    articles = split_into_articles(text)

    print(f"Tách được {len(articles)} điều.")

    print("Đang lưu JSON...")
    save_json(articles, OUTPUT_JSON)

    print("Hoàn thành! File JSON đã tạo tại:", OUTPUT_JSON)
