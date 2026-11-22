import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Thêm thư mục gốc vào sys.path để import được src.*
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detector.baseline import BaselineDetector, load_jsonl
from src.detector.inference import XLMRDetector


def calibrate_dual_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_recall: float = 0.95,
    target_precision: float = 0.95,
) -> Dict[str, float]:
    """
    Hiệu chỉnh ngưỡng (Threshold Calibration - Trang 7):
    - T_low : Đạt Recall lớp Injection >= 95% (hầu như không lọt tấn công -> dưới T_low là ALLOW).
    - T_high: Đạt Precision lớp Injection >= 95% (chắc chắn là tấn công -> trên T_high là BLOCK).
    - T_f1  : Ngưỡng tối ưu F1-Score (dùng để so sánh phân loại nhị phân công bằng với Baseline).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    # Bỏ phần tử cuối (sklearn thêm precision=1, recall=0 ở cuối không có threshold tương ứng)
    p_arr = precisions[:-1]
    r_arr = recalls[:-1]
    t_arr = thresholds

    # 1. Tìm ngưỡng tối ưu F1
    f1_scores = (2 * p_arr * r_arr) / (p_arr + r_arr + 1e-8)
    best_f1_idx = int(np.argmax(f1_scores))
    t_best_f1 = float(t_arr[best_f1_idx])

    # 2. Tìm T_low: Ngưỡng cao nhất (nhưng không vượt quá 0.45) mà vẫn giữ Recall >= target_recall
    valid_low_indices = np.where(r_arr >= target_recall)[0]
    if len(valid_low_indices) > 0:
        raw_t_low = float(t_arr[valid_low_indices[-1]])
    else:
        raw_t_low = 0.35
    # Đảm bảo T_low nằm trong dải vận hành hợp lý [0.15, 0.45] theo kiến trúc Trang 7
    t_low = float(np.clip(raw_t_low, 0.15, 0.45))

    # 3. Tìm T_high: Ngưỡng thấp nhất (nhưng >= 0.70) mà đạt Precision >= target_precision
    valid_high_indices = np.where(p_arr >= target_precision)[0]
    if len(valid_high_indices) > 0:
        raw_t_high = float(t_arr[valid_high_indices[0]])
    else:
        raw_t_high = 0.80
    # Đảm bảo T_high nằm trong dải vận hành hợp lý [0.70, 0.90] và luôn > T_low
    t_high = float(np.clip(max(raw_t_high, t_low + 0.25), 0.70, 0.90))

    return {
        "T_low": round(t_low, 4),
        "T_high": round(t_high, 4),
        "T_best_f1": round(t_best_f1, 4),
        "raw_T_low_at_95_recall": round(raw_t_low, 4),
        "raw_T_high_at_95_precision": round(raw_t_high, 4),
    }


def evaluate_predictions(
    records: List[Dict],
    probs: np.ndarray,
    latencies_ms: List[float],
    threshold: float,
    t_low: float = 0.35,
    t_high: float = 0.80,
) -> Dict:
    """Tính toán toàn bộ chỉ số đánh giá Tầng 1 và phân bố hành động 3 mức."""
    y_true = np.array([int(r["label"]) for r in records])
    categories = [str(r.get("category", "")).lower() for r in records]
    y_pred = (probs >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    overall_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Tính False Positive Rate (FPR) trên tập Hard Negatives
    hard_neg_indices = [
        i
        for i, (label, cat) in enumerate(zip(y_true, categories))
        if label == 0 and ("hard" in cat or "neg" in cat)
    ]
    if len(hard_neg_indices) > 0:
        hn_preds = y_pred[hard_neg_indices]
        hn_probs = probs[hard_neg_indices]
        hn_fp = int(np.sum(hn_preds == 1))
        hn_total = len(hard_neg_indices)
        hard_negative_fpr = hn_fp / hn_total
        # Trong Policy 3 mức: chỉ khi p >= T_high thì Hard Negative mới bị BLOCK hoàn toàn
        hn_hard_blocked = int(np.sum(hn_probs >= t_high))
        hn_hard_block_rate = hn_hard_blocked / hn_total
    else:
        hn_fp, hn_total, hard_negative_fpr = 0, 0, 0.0
        hn_hard_blocked, hn_hard_block_rate = 0, 0.0

    # Thống kê phân bố vào 3 vùng Policy (ALLOW / SANITIZE / BLOCK)
    allow_cnt = int(np.sum(probs < t_low))
    sanitize_cnt = int(np.sum((probs >= t_low) & (probs < t_high)))
    block_cnt = int(np.sum(probs >= t_high))

    return {
        "num_samples": len(records),
        "threshold_used": round(threshold, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "overall_fpr": round(float(overall_fpr), 4),
        "hard_negative_stats": {
            "total_hard_negatives": hn_total,
            "false_positives_binary": hn_fp,
            "hard_negative_fpr": round(float(hard_negative_fpr), 4),
            "hard_blocked_at_T_high": hn_hard_blocked,
            "hard_block_rate_at_T_high": round(float(hn_hard_block_rate), 4),
        },
        "policy_tier_distribution": {
            "allow_count": allow_cnt,
            "sanitize_count": sanitize_cnt,
            "block_count": block_cnt,
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "latency_ms": {
            "mean_ms": round(float(np.mean(latencies_ms)), 3),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        },
    }


def plot_pr_curves(
    test_records: List[Dict],
    ood_records: List[Dict],
    xlmr_test_probs: np.ndarray,
    xlmr_ood_probs: np.ndarray,
    base_test_probs: np.ndarray,
    base_ood_probs: np.ndarray,
    out_path: Path,
):
    """Vẽ biểu đồ Precision-Recall Curve so sánh Baseline và XLM-RoBERTa trên Test & OOD."""
    y_test = np.array([int(r["label"]) for r in test_records])
    y_ood = np.array([int(r["label"]) for r in ood_records])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    for ax, y_true, xlmr_p, base_p, title in [
        (axes[0], y_test, xlmr_test_probs, base_test_probs, "In-Domain Test Set (n=76)"),
        (axes[1], y_ood, xlmr_ood_probs, base_ood_probs, "OOD Challenge Set (n=100)"),
    ]:
        p_xlmr, r_xlmr, _ = precision_recall_curve(y_true, xlmr_p)
        ap_xlmr = average_precision_score(y_true, xlmr_p)

        p_base, r_base, _ = precision_recall_curve(y_true, base_p)
        ap_base = average_precision_score(y_true, base_p)

        ax.plot(
            r_xlmr,
            p_xlmr,
            label=f"XLM-RoBERTa Fine-tuned (AP = {ap_xlmr:.3f})",
            color="#1f77b4",
            linewidth=2.5,
        )
        ax.plot(
            r_base,
            p_base,
            label=f"TF-IDF + LogReg Baseline (AP = {ap_base:.3f})",
            color="#ff7f0e",
            linestyle="--",
            linewidth=2.0,
        )
        ax.set_title(f"Precision-Recall Curve — {title}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall (Injection Detection Rate)", fontsize=10)
        ax.set_ylabel("Precision (1 - False Discovery Rate)", fontsize=10)
        ax.set_xlim([0.0, 1.02])
        ax.set_ylim([0.3, 1.03])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=9.5)

    plt.suptitle("RAGShield-VN: Detector PR-Curve Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def update_guard_policy_yaml(t_low: float, t_high: float, policy_path: Path):
    """Tạo hoặc cập nhật file configs/guard_policy.yaml với cặp ngưỡng đã hiệu chỉnh."""
    policy_cfg = {
        "thresholds": {
            "T_low": t_low,
            "T_high": t_high,
        },
        "retrieval": {
            "top_k_fetch": 5,
            "max_safe_chunks": 3,
        },
        "sanitizer": {
            "sentence_risk_threshold": 0.50,
            "mixed_payload_word_limit": 150,
            "redaction_placeholder": "[NỘI DUNG BỊ LỌC DO RỦI RO BẢO MẬT]",
        },
        "safe_refusal_message": (
            "Xin lỗi, tài liệu liên quan đến câu hỏi này hiện đang bị tạm khóa "
            "do phát hiện dấu hiệu vi phạm chính sách an toàn dữ liệu."
        ),
        "audit_log_path": "logs/audit_log.jsonl",
    }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with open(policy_path, "w", encoding="utf-8") as f:
        yaml.dump(policy_cfg, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá XLM-RoBERTa & Hiệu chỉnh ngưỡng (Ngày 7)")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Đường dẫn tới thư mục model hoặc tên repo trên Hugging Face Hub",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Xác định nguồn model: ưu tiên tham số truyền vào -> local_model_dir -> hub_model_id
    model_source = args.model_path
    if not model_source:
        local_dir = Path(cfg["model"]["local_model_dir"])
        if local_dir.exists():
            model_source = str(local_dir)
        else:
            model_source = cfg["model"]["hub_model_id"]

    print(f"1. Đang tải mô hình XLM-RoBERTa từ: {model_source}")
    detector = XLMRDetector(
        model_name_or_path=model_source,
        max_length=int(cfg["model"].get("max_length", 384)),
    )
    print(f"   Thiết bị suy luận: {detector.device.upper()} (FP16: {detector.use_fp16})")

    # Tải dữ liệu
    data_cfg = cfg["data"]
    train_data = load_jsonl(Path(data_cfg["train_path"]))
    val_data = load_jsonl(Path(data_cfg["val_path"]))
    test_data = load_jsonl(Path(data_cfg["test_path"]))
    ood_data = load_jsonl(Path(data_cfg["ood_path"]))

    # Hàm chạy inference và đo latency từng mẫu
    def run_split_inference(records: List[Dict]) -> Tuple[np.ndarray, List[float]]:
        # Warmup 2 lần đầu cho GPU
        if len(records) > 0:
            detector.score_single_with_latency(records[0]["text"])
        probs, lats = [], []
        for r in records:
            p, ms = detector.score_single_with_latency(r["text"])
            probs.append(p)
            lats.append(ms)
        return np.array(probs, dtype=np.float32), lats

    print("2. Đang chạy suy luận và đo độ trễ trên tập Val, Test và OOD...")
    val_probs, val_lats = run_split_inference(val_data)
    test_probs, test_lats = run_split_inference(test_data)
    ood_probs, ood_lats = run_split_inference(ood_data)

    # 3. Hiệu chỉnh ngưỡng trên tập Validation
    y_val = np.array([int(r["label"]) for r in val_data])
    calib = calibrate_dual_thresholds(y_val, val_probs)
    t_low = calib["T_low"]
    t_high = calib["T_high"]
    t_eval = 0.5  # Ngưỡng chuẩn 0.5 cho phân loại nhị phân

    print(
        f"3. Kết quả hiệu chỉnh ngưỡng (Threshold Calibration):\n"
        f"   + T_low  (Allow -> Sanitize) : {t_low:.4f} (Mục tiêu Recall >= 95%)\n"
        f"   + T_high (Sanitize -> Block) : {t_high:.4f} (Mục tiêu Precision >= 95%)\n"
        f"   + T_best_f1 trên Validation  : {calib['T_best_f1']:.4f}"
    )

    # 4. Đánh giá chi tiết trên 3 tập
    val_metrics = evaluate_predictions(val_data, val_probs, val_lats, t_eval, t_low, t_high)
    test_metrics = evaluate_predictions(test_data, test_probs, test_lats, t_eval, t_low, t_high)
    ood_metrics = evaluate_predictions(ood_data, ood_probs, ood_lats, t_eval, t_low, t_high)

    # 5. Chạy hoặc tải Baseline để vẽ PR-Curve và xuất bảng đối chứng
    base_model_path = Path("models/baseline_tfidf_lr.joblib")
    if base_model_path.exists():
        base_detector = BaselineDetector.load(base_model_path)
    else:
        base_detector = BaselineDetector()
        base_detector.fit(
            [r["text"] for r in train_data], [int(r["label"]) for r in train_data]
        )
        base_detector.calibrate_threshold(
            [r["text"] for r in val_data], [int(r["label"]) for r in val_data]
        )

    base_test_probs = base_detector.predict_proba([r["text"] for r in test_data])
    base_ood_probs = base_detector.predict_proba([r["text"] for r in ood_data])

    pr_fig_path = Path("results/figures/pr_curve_comparison.png")
    plot_pr_curves(
        test_data,
        ood_data,
        test_probs,
        ood_probs,
        base_test_probs,
        base_ood_probs,
        pr_fig_path,
    )
    print(f"4. Đã xuất biểu đồ PR-Curve tại: {pr_fig_path}")

    # Cập nhật configs/guard_policy.yaml
    policy_path = Path("configs/guard_policy.yaml")
    update_guard_policy_yaml(t_low, t_high, policy_path)
    print(f"5. Đã cập nhật ngưỡng vận hành (T_low, T_high) vào: {policy_path}")

    # Lưu kết quả JSON
    out_json_path = Path("results/xlmr_metrics.json")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_source": model_source,
        "device": detector.device,
        "fp16": detector.use_fp16,
        "threshold_calibration": calib,
        "metrics": {
            "val": val_metrics,
            "in_domain_test": test_metrics,
            "ood_challenge": ood_metrics,
        },
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"6. Đã lưu báo cáo đánh giá chi tiết tại: {out_json_path}")

    # In bảng so sánh trực quan
    base_json_path = Path("results/baseline_metrics.json")
    base_report = None
    if base_json_path.exists():
        with open(base_json_path, "r", encoding="utf-8") as f:
            base_report = json.load(f)

    print("\n" + "=" * 78)
    print("BẢNG SO SÁNH HIỆU NĂNG: BASELINE (TF-IDF + LR) vs. XLM-ROBERTA FINE-TUNED")
    print("=" * 78)
    print(
        f"{'Tập dữ liệu / Mô hình':28s} | {'F1-Score':8s} | {'Precision':9s} | "
        f"{'Recall':8s} | {'HardNeg FPR':11s} | {'Latency':9s}"
    )
    print("-" * 78)

    for split_key, split_label, xm in [
        ("val", "Validation", val_metrics),
        ("in_domain_test", "In-Domain Test", test_metrics),
        ("ood_challenge", "OOD Challenge", ood_metrics),
    ]:
        if base_report and split_key in base_report.get("metrics_calibrated", {}):
            bm = base_report["metrics_calibrated"][split_key]
            b_hn = bm["hard_negative_stats"]["hard_negative_fpr"] * 100
            print(
                f"{split_label + ' (Baseline)':28s} | {bm['f1_score']*100:7.2f}% | "
                f"{bm['precision']*100:8.2f}% | {bm['recall']*100:7.2f}% | "
                f"{b_hn:10.1f}% | {bm['latency_cpu_ms']['mean_ms']:6.2f} ms"
            )
        x_hn = xm["hard_negative_stats"]["hard_negative_fpr"] * 100
        print(
            f"{split_label + ' (XLM-R)':28s} | {xm['f1_score']*100:7.2f}% | "
            f"{xm['precision']*100:8.2f}% | {xm['recall']*100:7.2f}% | "
            f"{x_hn:10.1f}% | {xm['latency_ms']['mean_ms']:6.2f} ms"
        )
        print("-" * 78)


if __name__ == "__main__":
    main()