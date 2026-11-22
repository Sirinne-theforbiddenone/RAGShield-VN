# src/detector/inference.py
import os
import time
from pathlib import Path
from typing import Dict, List, Union

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

DEFAULT_HF_REPO = "Sirinne/ragshield-vn-xlmr-base"


def get_safe_device() -> str:
    """Kiểm tra GPU có thực thi được kernel PyTorch hiện tại không (tránh lỗi RTX 50-series sm_120)."""
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device="cuda") * 2.0
            return "cuda"
        except Exception as e:
            print(f"[WARN] GPU không tương thích với bản build PyTorch hiện tại ({e}). Chuyển sang CPU.")
    return "cpu"


def resolve_xlmr_model_path(model_path_or_repo: str | None = None) -> str:
    """
    Ưu tiên:
    1. Đường dẫn hoặc repo ID truyền trực tiếp vào hàm.
    2. Thư mục local nếu có ('models/ragshield-vn-xlmr-base' hoặc 'models/xlmr_injection_detector').
    3. Biến môi trường HF_MODEL_REPO_ID trong file .env.
    4. Mặc định tải trực tiếp từ Hugging Face Hub: 'Sirinne/ragshield-vn-xlmr-base'.
    """
    if model_path_or_repo:
        local_candidate = Path(model_path_or_repo)
        if local_candidate.exists() and (local_candidate / "config.json").exists():
            return str(local_candidate)
        if "/" in model_path_or_repo and not local_candidate.exists():
            return model_path_or_repo

    for folder_name in ["models/ragshield-vn-xlmr-base", "models/xlmr_injection_detector"]:
        local_dir = Path(folder_name)
        if local_dir.exists() and (local_dir / "config.json").exists():
            return str(local_dir)

    env_repo = (
        os.getenv("HF_MODEL_REPO_ID")
        or os.getenv("HUGGINGFACE_REPO_ID")
        or os.getenv("XLMR_MODEL_ID")
    )
    if env_repo:
        return env_repo.strip()

    return DEFAULT_HF_REPO


class XLMRDetector:
    """
    Bộ phát hiện Indirect Prompt Injection sử dụng mô hình XLM-RoBERTa đã fine-tune.
    Hỗ trợ tải từ thư mục local hoặc từ Hugging Face Hub (Sirinne/ragshield-vn-xlmr-base).
    """

    def __init__(
        self,
        model_path_or_repo: str | None = None,
        max_length: int = 384,
        device: str | None = None,
    ):
        resolved_source = resolve_xlmr_model_path(model_path_or_repo)
        self.model_source = resolved_source
        self.max_length = max_length
        self.device = device or get_safe_device()

        print(f"[XLMRDetector] Đang nạp model từ '{self.model_source}' trên thiết bị '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_source)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """Trả về xác suất p(injection) in [0.0, 1.0] cho 1 chuỗi hoặc danh sách chuỗi."""
        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)
        if not texts:
            return 0.0 if is_single else []

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].detach().cpu().numpy()
        scores = [float(p) for p in probs]
        return scores[0] if is_single else scores

    def predict_proba_batch(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """Chấm điểm theo batch cho danh sách các đoạn văn/câu."""
        if not texts:
            return []
        all_scores: List[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            scores = self.predict_proba(batch)
            if isinstance(scores, float):
                all_scores.append(scores)
            else:
                all_scores.extend(scores)
        return all_scores

    def evaluate_chunk(
        self,
        text: str,
        t_low: float = 0.45,
        t_high: float = 0.70,
    ) -> Dict[str, Union[float, str]]:
        """Đánh giá nhanh 1 chunk với 2 ngưỡng T_low và T_high."""
        t0 = time.perf_counter()
        score = float(self.predict_proba(text))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if score < t_low:
            decision = "ALLOW"
        elif score < t_high:
            decision = "SANITIZE_AND_FLAG"
        else:
            decision = "BLOCK_AND_QUARANTINE"

        return {
            "risk_score": round(score, 4),
            "decision": decision,
            "latency_ms": round(latency_ms, 2),
        }