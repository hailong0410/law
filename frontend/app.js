// ===============================
// Cấu hình URL API backend
// ===============================
const API_QA_URL = "http://127.0.0.1:8000/QA/ask";
const API_TTS_URL = "http://127.0.0.1:8000/QA/tts";

// Gắn event cho nút
document.getElementById("ask-btn").addEventListener("click", askLaw);
document.getElementById("tts-btn").addEventListener("click", speakAnswer);

let lastAnswerText = "";

// ===============================
// Gửi câu hỏi lên backend
// ===============================
async function askLaw() {
    const questionBox = document.getElementById("question");
    const resultBox   = document.getElementById("result");
    const loading     = document.getElementById("loading-text");

    const question = questionBox.value.trim();
    if (!question) {
        alert("Vui lòng nhập câu hỏi!");
        return;
    }

    resultBox.innerHTML = "";
    loading.style.display = "block";

    const payload = {
        question: question,
        top_k: 5
    };

    try {
        const response = await fetch(API_QA_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error("Server trả về lỗi: " + response.status);
        }

        const data = await response.json();
        loading.style.display = "none";

        if (!data || !data.results || data.results.length === 0) {
            resultBox.innerHTML = "<p>Không tìm thấy điều luật phù hợp.</p>";
            lastAnswerText = "";
            return;
        }

        let html = `<h3>Kết quả:</h3>`;
        data.results.forEach(item => {
            html += `
                <div class="law-item">
                    <h4>Điều ${item.so_dieu} – ${item.tieu_de}</h4>
                    <p><b>Chương:</b> ${item.chuong}</p>
                    <p>${item.chunk_text}</p>
                    <p class="score">Độ liên quan: ${(item.score * 100).toFixed(2)}%</p>
                </div>
            `;
        });

        resultBox.innerHTML = html;

        // ghép nội dung để đọc TTS
        lastAnswerText = data.results.map(r => r.chunk_text).join(". ");

    } catch (error) {
        loading.style.display = "none";
        console.error("Lỗi khi gọi API /QA/ask:", error);
        alert("Có lỗi khi gọi API /QA/ask. Xem console để biết chi tiết.");
    }
}

// ===============================
// Gọi TTS backend để đọc nội dung
// ===============================
async function speakAnswer() {
    if (!lastAnswerText) {
        alert("Chưa có nội dung để đọc! Hãy hỏi luật trước.");
        return;
    }

    const payload = { text: lastAnswerText };

    try {
        const response = await fetch(API_TTS_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error("Lỗi TTS: " + response.status);
        }

        const data = await response.json();
        const audioBase64 = data.audio_base64;

        const audio = new Audio("data:audio/mp3;base64," + audioBase64);
        audio.play();

    } catch (err) {
        alert("Không thể phát âm thanh.");
        console.error("Lỗi TTS:", err);
    }
}
