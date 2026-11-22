import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class SecurityAuditLogger:
    """Ghi nhật ký sự kiện bảo mật (Audit Log) ra file JSONL (Trang 7 & 15)."""

    def __init__(self, log_path: str = "logs/audit_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, query: str, guard_result: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "all_blocked": guard_result.get("all_blocked", False),
            "guard_latency_ms": guard_result.get("guard_latency_ms", 0.0),
            "summary": guard_result.get("summary", {}),
            "chunk_traces": guard_result.get("chunk_traces", []),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record