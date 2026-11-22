import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.guard.logger import SecurityAuditLogger
from src.guard.sanitizer import ContextSanitizer

DEFAULT_SAFE_REFUSAL = (
    "Xin lỗi, tài liệu liên quan đến câu hỏi này hiện đang bị tạm khóa "
    "do phát hiện dấu hiệu vi phạm chính sách an toàn dữ liệu."
)


class GuardPolicyEngine:
    """
    Bộ điều phối chính sách bảo mật 3 mức (Policy-Driven Security Middleware):
    - ALLOW (p < T_low)
    - SANITIZE_AND_FLAG (T_low <= p < T_high hoặc Mixed Payload có chứa phần văn bản sạch)
    - BLOCK_AND_QUARANTINE (p >= T_high) + Retrieval Fallback (Top-K=5 chọn max_safe_chunks=3)
    """

    def __init__(
        self,
        detector: Any,
        policy_config_path: str = "configs/guard_policy.yaml",
        t_low: Optional[float] = None,
        t_high: Optional[float] = None,
        max_safe_chunks: Optional[int] = None,
        enable_sanitizer: bool = True,
        enable_audit_log: bool = True,
    ):
        self.detector = detector
        self.enable_sanitizer = enable_sanitizer
        self.enable_audit_log = enable_audit_log

        cfg = {}
        cfg_path = Path(policy_config_path)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        th_cfg = cfg.get("thresholds", {})
        ret_cfg = cfg.get("retrieval", {})
        san_cfg = cfg.get("sanitizer", {})

        self.t_low = float(t_low if t_low is not None else th_cfg.get("T_low", 0.45))
        self.t_high = float(t_high if t_high is not None else th_cfg.get("T_high", 0.70))
        self.top_k_fetch = int(ret_cfg.get("top_k_fetch", 5))
        self.max_safe_chunks = int(
            max_safe_chunks if max_safe_chunks is not None else ret_cfg.get("max_safe_chunks", 3)
        )
        self.mixed_payload_word_limit = int(san_cfg.get("mixed_payload_word_limit", 150))
        self.sentence_threshold = float(san_cfg.get("sentence_risk_threshold", self.t_low))
        self.redaction_placeholder = str(
            san_cfg.get("redaction_placeholder", "[NỘI DUNG BỊ LỌC DO RỦI RO BẢO MẬT]")
        )
        self.safe_refusal_message = str(cfg.get("safe_refusal_message", DEFAULT_SAFE_REFUSAL))

        self.sanitizer = ContextSanitizer(
            detector=self.detector,
            sentence_threshold=self.sentence_threshold,
            redaction_placeholder=self.redaction_placeholder,
        )
        log_path = str(cfg.get("audit_log_path", "logs/audit_log.jsonl"))
        self.logger = SecurityAuditLogger(log_path=log_path)

    def inspect_chunks(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: str = "",
    ) -> Dict[str, Any]:
        """
        Quét danh sách Top-K chunks từ Retriever, áp dụng Policy 3 mức + Retrieval Fallback.
        """
        t0 = time.perf_counter()

        if not retrieved_chunks:
            return {
                "safe_chunks": [],
                "safe_texts": [],
                "quarantined_chunks": [],
                "chunk_traces": [],
                "all_blocked": True,
                "safe_refusal_message": self.safe_refusal_message,
                "guard_latency_ms": 0.0,
                "summary": {"allow": 0, "sanitize": 0, "block": 0, "selected": 0},
            }

        # Bước 2a: Batch Inference chấm điểm toàn bộ các chunk truy xuất cùng lúc
        texts = [c["text"] for c in retrieved_chunks]
        probs = np.asarray(self.detector.predict_proba(texts), dtype=np.float32)

        safe_chunks: List[Dict[str, Any]] = []
        quarantined_chunks: List[Dict[str, Any]] = []
        chunk_traces: List[Dict[str, Any]] = []
        counts = {"allow": 0, "sanitize": 0, "block": 0, "fallback_used": 0}

        for rank_idx, (chunk, raw_prob) in enumerate(zip(retrieved_chunks, probs), start=1):
            p_i = float(raw_prob)
            word_count = len(chunk["text"].split())
            action = ""
            final_text = chunk["text"]
            sanitize_info = None
            reason = ""

            # Tình huống 1: Chunk hoàn toàn sạch (p_i < T_low)
            if p_i < self.t_low:
                action = "ALLOW"
                reason = f"Điểm rủi ro thấp ({p_i:.3f} < T_low={self.t_low:.2f})"
                counts["allow"] += 1

            # Tình huống 3 & 5: Chunk điểm lấp lửng (T_low <= p_i < T_high)
            # HOẶC Mixed Payload (đoạn văn dài > 150 từ hoặc nhiều câu có chứa phần sạch)
            elif self.enable_sanitizer and (
                p_i < self.t_high or word_count >= self.mixed_payload_word_limit
            ):
                sanitize_info = self.sanitizer.sanitize_chunk(chunk["text"])
                if sanitize_info["kept_sentence_count"] > 0 and sanitize_info["removed_sentence_count"] > 0:
                    # Kiểm tra lại phần văn bản sau khi đã cắt câu độc hại xem đã thực sự an toàn chưa
                    clean_sentences_only = " ".join(
                        s["sentence"]
                        for s in sanitize_info["sentence_traces"]
                        if s["action"] == "KEPT"
                    )
                    recheck_prob = float(self.detector.predict_proba([clean_sentences_only])[0])
                    if recheck_prob < self.t_high:
                        action = "SANITIZE_AND_FLAG"
                        final_text = sanitize_info["sanitized_text"]
                        reason = (
                            f"Đã lọc bỏ {sanitize_info['removed_sentence_count']}/"
                            f"{sanitize_info['original_sentence_count']} câu nghi ngờ "
                            f"(p_gốc={p_i:.3f} -> p_sạch={recheck_prob:.3f})"
                        )
                        counts["sanitize"] += 1
                    else:
                        action = "BLOCK_AND_QUARANTINE"
                        reason = f"Vẫn còn rủi ro cao sau khi tách câu ({recheck_prob:.3f} >= T_high)"
                        counts["block"] += 1
                elif sanitize_info["removed_sentence_count"] == 0 and p_i < self.t_high:
                    # Điểm đoạn văn ở mức trung bình nhưng không có câu đơn lẻ nào vượt ngưỡng độc hại
                    action = "SANITIZE_AND_FLAG"
                    final_text = chunk["text"]
                    reason = f"Cảnh báo mức trung bình ({p_i:.3f}), không phát hiện câu lệnh độc hại rời rạc"
                    counts["sanitize"] += 1
                else:
                    action = "BLOCK_AND_QUARANTINE"
                    reason = f"Toàn bộ các câu trong chunk đều vượt ngưỡng rủi ro (p={p_i:.3f})"
                    counts["block"] += 1

            # Tình huống 2 & 5 mở rộng: Điểm p_i >= T_high
            else:
                # Nếu bật Sanitizer, thử kiểm tra xem có phải đoạn văn hỗn hợp (Mixed Payload) nhiều câu không
                rescued_mixed = False
                if self.enable_sanitizer:
                    sanitize_info = self.sanitizer.sanitize_chunk(chunk["text"])
                    if (
                        sanitize_info["original_sentence_count"] >= 2
                        and sanitize_info["kept_sentence_count"] >= 1
                        and sanitize_info["removed_sentence_count"] >= 1
                    ):
                        clean_only_str = " ".join(
                            s["sentence"]
                            for s in sanitize_info["sentence_traces"]
                            if s["action"] == "KEPT"
                        )
                        recheck_p = float(self.detector.predict_proba([clean_only_str])[0])
                        if recheck_p < self.t_low:
                            action = "SANITIZE_AND_FLAG"
                            final_text = sanitize_info["sanitized_text"]
                            reason = (
                                f"Mixed Payload: Đã cắt bỏ {sanitize_info['removed_sentence_count']} câu độc hại, "
                                f"giữ lại {sanitize_info['kept_sentence_count']} câu sạch (p_sạch={recheck_p:.3f})"
                            )
                            counts["sanitize"] += 1
                            rescued_mixed = True

                if not rescued_mixed:
                    action = "BLOCK_AND_QUARANTINE"
                    reason = f"Điểm rủi ro vượt ngưỡng chặn ({p_i:.3f} >= T_high={self.t_high:.2f})"
                    counts["block"] += 1

            # Quyết định chọn vào Safe Context Pool (tối đa max_safe_chunks = 3)
            selected_for_llm = False
            is_fallback_chunk = False
            if action in ("ALLOW", "SANITIZE_AND_FLAG"):
                if len(safe_chunks) < self.max_safe_chunks:
                    selected_for_llm = True
                    # Nếu rank_idx > max_safe_chunks hoặc trước đó đã có chunk bị BLOCK thì đây là Retrieval Fallback
                    if counts["block"] > 0 and rank_idx > len(safe_chunks):
                        is_fallback_chunk = True
                        counts["fallback_used"] += 1
                    safe_chunk_copy = dict(chunk)
                    safe_chunk_copy["sanitized_text"] = final_text
                    safe_chunk_copy["risk_score"] = round(p_i, 4)
                    safe_chunk_copy["guard_action"] = action
                    safe_chunk_copy["is_fallback"] = is_fallback_chunk
                    safe_chunks.append(safe_chunk_copy)
            else:
                q_copy = dict(chunk)
                q_copy["risk_score"] = round(p_i, 4)
                q_copy["guard_action"] = action
                q_copy["reason"] = reason
                quarantined_chunks.append(q_copy)

            chunk_traces.append({
                "rank": rank_idx,
                "chunk_id": chunk.get("chunk_id", f"chunk_{rank_idx}"),
                "doc_id": chunk.get("doc_id", ""),
                "title": chunk.get("title", ""),
                "risk_score": round(p_i, 4),
                "word_count": word_count,
                "action": action,
                "selected_for_llm": selected_for_llm,
                "is_fallback": is_fallback_chunk,
                "reason": reason,
                "original_text": chunk["text"],
                "final_text": final_text if action != "BLOCK_AND_QUARANTINE" else "",
                "sentence_traces": sanitize_info["sentence_traces"] if sanitize_info else [],
            })

        # Tình huống 4: Tất cả các chunk đều bị chặn -> Kích hoạt Safe Refusal
        all_blocked = len(safe_chunks) == 0
        guard_latency_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "safe_chunks": safe_chunks,
            "safe_texts": [c["sanitized_text"] for c in safe_chunks],
            "quarantined_chunks": quarantined_chunks,
            "chunk_traces": chunk_traces,
            "all_blocked": all_blocked,
            "safe_refusal_message": self.safe_refusal_message if all_blocked else None,
            "guard_latency_ms": round(guard_latency_ms, 2),
            "summary": {
                "total_inspected": len(retrieved_chunks),
                "allow": counts["allow"],
                "sanitize": counts["sanitize"],
                "block": counts["block"],
                "selected_for_llm": len(safe_chunks),
                "fallback_used": counts["fallback_used"],
            },
        }

        if self.enable_audit_log:
            self.logger.log_event(query=query, guard_result=result)

        return result