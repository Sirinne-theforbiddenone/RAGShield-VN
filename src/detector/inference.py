import time
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class XLMRDetector:
    """
    Bộ phát hiện Indirect Prompt Injection dùng XLM-RoBERTa đã fine-tune.
    Hỗ trợ cả Single-chunk Inference và Batch Inference (tối ưu độ trễ cho RAG Guard).
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        max_length: int = 384,
        device: str = None,
        use_fp16: bool = True,
    ):
        self.model_name_or_path = str(model_name_or_path)
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(use_fp16 and self.device == "cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path
        )
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Dự đoán xác suất rủi ro (lớp INJECTION = 1) theo batch.
        Trả về mảng numpy 1D các xác suất p_i thuộc [0, 1].
        """
        if not texts:
            return np.array([], dtype=np.float32)

        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())

        return np.array(all_probs, dtype=np.float32)

    @torch.inference_mode()
    def score_single_with_latency(self, text: str) -> Tuple[float, float]:
        """Chấm điểm 1 chunk và đo độ trễ thực tế (ms)."""
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**encoded)
        prob = float(torch.softmax(outputs.logits.float(), dim=-1)[0, 1].item())

        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return prob, latency_ms