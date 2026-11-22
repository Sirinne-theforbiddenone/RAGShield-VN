import os
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from dotenv import load_dotenv
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()


def test_xlmr_load():
    print("[1/2] Đang kiểm tra xlm-roberta-base từ cache...")
    t0 = time.perf_counter()
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    inputs = tokenizer("Kiểm tra độ trễ token hóa tiếng Việt.", return_tensors="pt")
    _ = model(**inputs)
    print(f"OK! Chạy thử xlm-roberta-base thành công ({time.perf_counter() - t0:.2f}s).")


def pick_active_chat_model(client: Groq) -> str:
    models_data = client.models.list().data
    all_ids = [m.id for m in models_data]

    excluded = ("whisper", "guard", "tts", "embed", "rerank", "audio")
    chat_models = [
        m_id for m_id in all_ids
        if not any(kw in m_id.lower() for kw in excluded)
    ]

    if not chat_models:
        raise RuntimeError(f"Không tìm thấy model chat khả dụng. Danh sách gốc: {all_ids}")

    preferred_prefixes = ("qwen", "llama-3.3", "llama-3", "llama", "gemma", "deepseek")
    for prefix in preferred_prefixes:
        for m_id in chat_models:
            if m_id.lower().startswith(prefix):
                return m_id
    return chat_models[0]


def test_llm_api():
    print("[2/2] Đang kiểm tra sinh văn bản tiếng Việt qua Groq API...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Thiếu GROQ_API_KEY trong file .env")

    client = Groq(api_key=api_key)
    model_id = "qwen/qwen3.8-27b"
    print(f"Model Groq được chọn: {model_id}")

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "Bạn là trợ lý nội bộ doanh nghiệp. Trả lời ngắn gọn bằng tiếng Việt."},
            {"role": "user", "content": "Quy trình xin nghỉ phép năm gồm những bước nào?"}
        ],
        temperature=0.0,
        seed=42,
        max_tokens=150
    )
    print("Phản hồi LLM:\n", resp.choices[0].message.content)


if __name__ == "__main__":
    test_xlmr_load()
    test_llm_api()