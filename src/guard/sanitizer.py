# src/guard/sanitizer.py
import re
from typing import Any, Dict, List

import numpy as np


VIETNAMESE_ABBREVIATIONS = [
    r"TP\.HCM",
    r"Tp\.HCM",
    r"v\.v\.",
    r"V\.v\.",
    r"PGS\.TS\.",
    r"GS\.TS\.",
    r"ThS\.",
    r"TS\.",
    r"BS\.",
    r"KS\.",
    r"CN\.",
    r"Tp\.",
    r"TP\.",
    r"Q\.",
    r"P\.",
    r"TT\.",
    r"Mr\.",
    r"Mrs\.",
    r"Ms\.",
    r"Dr\.",
    r"St\.",
]

# Các mẫu tấn công cấu trúc (Structural / Delimiter Spoofing / Markdown Exfiltration)
STRUCTURAL_HAZARD_RE = re.compile(
    r"(?:</?trusted_knowledge_base>|</?context>|\[(?:SYSTEM|ADMIN|SECURITY|INSTRUCTION)\b|!\[.*?\]\(https?://)",
    re.IGNORECASE,
)


def split_vietnamese_sentences_safe(text: str) -> List[str]:
    """
    Tách câu tiếng Việt an toàn, không cắt nhầm:
    - Từ viết tắt thông dụng (TP.HCM, TS., ThS., v.v., PGS.TS., Q., P.)
    - Số thập phân / phiên bản (v2.1, 3.5%, 100.000)
    - Đường dẫn URL / Markdown link ([text](https://example.com/a.b))
    Đồng thời tách riêng các thẻ giả mạo hệ thống ([SYSTEM...], </trusted_knowledge_base>).
    """
    if not text or not text.strip():
        return []

    protected = text.strip()
    placeholders: Dict[str, str] = {}
    counter = 0

    def _replace_with_token(match: re.Match) -> str:
        nonlocal counter
        token = f"__PROTECTED_{counter}__"
        placeholders[token] = match.group(0)
        counter += 1
        return token

    # 1. Bảo vệ Markdown links/images và URLs
    protected = re.sub(r"!?\[[^\]]*\]\([^)]+\)", _replace_with_token, protected)
    protected = re.sub(r"https?://\S+", _replace_with_token, protected)
    protected = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", _replace_with_token, protected)

    # 2. Bảo vệ từ viết tắt tiếng Việt
    abbr_pattern = r"(?<!\w)(?:" + "|".join(VIETNAMESE_ABBREVIATIONS) + r")"
    protected = re.sub(abbr_pattern, _replace_with_token, protected)

    # 3. Bảo vệ số thập phân / số tiền / phiên bản (VD: 3.14, v1.2, 1.000.000)
    protected = re.sub(r"\b\d+(?:\.\d+)+\b", _replace_with_token, protected)

    # 4. Chèn ngắt dòng trước các thẻ giả mạo cấu trúc nếu chúng dính liền sau câu trước
    protected = re.sub(
        r"\s*(?=(?:\[SYSTEM|\[ADMIN|\[SECURITY|\[INSTRUCTION|</trusted_knowledge_base>|</context>|###\s*(?:SYSTEM|INSTRUCTION|HỆ THỐNG)))",
        "\n",
        protected,
        flags=re.IGNORECASE,
    )

    # 5. Tách câu theo dấu kết thúc câu (. ! ? hoặc xuống dòng)
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", protected)

    sentences: List[str] = []
    for part in raw_parts:
        restored = part.strip()
        if not restored:
            continue
        for token, original in placeholders.items():
            restored = restored.replace(token, original)
        if restored.strip():
            sentences.append(restored.strip())

    return sentences


class ContextSanitizer:
    """
    Bộ lọc và làm sạch ngữ cảnh ở cấp độ câu (Sentence-level Sanitizer).
    Khớp 100% với cấu trúc dữ liệu của GuardPolicyEngine.inspect_chunks().
    """

    def __init__(
        self,
        detector: Any,
        sentence_threshold: float = 0.45,
        redaction_placeholder: str = "[NỘI DUNG BỊ LỌC DO RỦI RO BẢO MẬT]",
    ):
        self.detector = detector
        self.sentence_threshold = float(sentence_threshold)
        self.redaction_placeholder = str(redaction_placeholder)

    def sanitize_chunk(self, text: str) -> Dict[str, Any]:
        sentences = split_vietnamese_sentences_safe(text)
        if not sentences:
            return {
                "sanitized_text": text,
                "original_sentence_count": 0,
                "kept_sentence_count": 0,
                "removed_sentence_count": 0,
                "sentence_traces": [],
            }

        probs = np.asarray(self.detector.predict_proba(sentences), dtype=np.float32).reshape(-1)

        out_parts: List[str] = []
        sentence_traces: List[Dict[str, Any]] = []
        kept_count = 0
        removed_count = 0

        for sent, raw_p in zip(sentences, probs):
            score = float(raw_p)
            has_structural_hazard = bool(STRUCTURAL_HAZARD_RE.search(sent))
            effective_score = max(score, 0.99 if has_structural_hazard else score)

            if effective_score >= self.sentence_threshold or has_structural_hazard:
                action = "REMOVED"
                removed_count += 1
                if not out_parts or out_parts[-1] != self.redaction_placeholder:
                    out_parts.append(self.redaction_placeholder)
            else:
                action = "KEPT"
                kept_count += 1
                out_parts.append(sent)

            sentence_traces.append({
                "sentence": sent,
                "risk_score": round(effective_score, 4),
                "action": action,
            })

        return {
            "sanitized_text": " ".join(out_parts).strip(),
            "original_sentence_count": len(sentences),
            "kept_sentence_count": kept_count,
            "removed_sentence_count": removed_count,
            "sentence_traces": sentence_traces,
        }