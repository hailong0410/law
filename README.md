<<<<<<< HEAD
# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY CODE 

## 🧩 Yêu cầu trước khi cài đặt 

- Python 3.12+
- Node.js 18+ và npm
- Redis instance đang chạy
- Docker (tùy chọn, dùng cho local LLMs)

## ⚙️ Thiết lập Backend (/backend)

Đi tới thư mục backend:




=======
# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY CODE 

## 🧩 Yêu cầu trước khi cài đặt 

- Python 3.12+
- Node.js 18+ và npm
- Docker

## ⚙️ Thiết lập Backend (/backend)

1. **Đi tới thư mục backend**
```bash
cd backend
```

2. **Tạo & kích hoạt virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Cài dependencies**
```bash
pip install -r requirements.txt
```

4. **Tạo file môi trường .env**
```bash
cp .env.example .env
```

5. **Running**
```bash
uvicorn app.main:app --reload
```
**Backend chạy tại http://localhost:8000**

## 🎨 Thiết lập Frontend (/frontend)

1. **Đi tới thư mục frontend**
```bash
cd frontend
```

2. **Cài dependencies**
```bash
npm install
```

3. **Running**
```bash
npm run dev
```

**Frontend chạy tại http://localhost:5173**






>>>>>>> 1c67d78090dd092d7524f04e6b40fdfda97b6e11
