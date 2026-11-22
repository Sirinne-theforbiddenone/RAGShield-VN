import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

# Cố định seed tái lập
RANDOM_SEED = 42


def load_jsonl(file_path: Path) -> List[Dict]:
    """Đọc file JSONL và trả về danh sách các mẫu dữ liệu."""
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path}")
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class BaselineDetector:
    """
    Mô hình Baseline: TF-IDF (Word + Char n-grams) + Logistic Regression.
    Thiết kế hỗ trợ cả đánh giá độc lập (Ngày 5) và tích hợp vào Guard (Ngày 12 - Cấu hình C).
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    FeatureUnion(
                        [
                            (
                                "word_ngram",
                                TfidfVectorizer(
                                    analyzer="word",
                                    ngram_range=(1, 2),
                                    max_features=5000,
                                    lowercase=True,
                                    sublinear_tf=True,
                                ),
                            ),
                            (
                                "char_ngram",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    max_features=5000,
                                    lowercase=True,
                                    sublinear_tf=True,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        max_iter=1000,
                    ),
                ),
            ]
        )

    def fit(self, texts: List[str], labels: List[int]) -> float:
        """Huấn luyện mô hình và trả về thời gian train (giây)."""
        start_time = time.perf_counter()
        self.pipeline.fit(texts, labels)
        train_time_sec = time.perf_counter() - start_time
        return train_time_sec

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Trả về mảng xác suất lớp INJECTION (label=1) cho danh sách văn bản."""
        return self.pipeline.predict_proba(texts)[:, 1]

    def predict_single(self, text: str) -> Tuple[float, int, float]:
        """Dự đoán cho 1 chunk đơn lẻ (phục vụ đo latency thực tế trên CPU)."""
        t0 = time.perf_counter()
        prob = float(self.pipeline.predict_proba([text])[0, 1])
        pred = int(prob >= self.threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return prob, pred, latency_ms

    def calibrate_threshold(self, val_texts: List[str], val_labels: List[int]) -> float:
        """Tìm ngưỡng xác suất tối ưu hóa F1-Score trên tập Validation."""
        probs = self.predict_proba(val_texts)
        precisions, recalls, thresholds = precision_recall_curve(val_labels, probs)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # Bỏ phần tử cuối của sklearn
        self.threshold = float(thresholds[best_idx])
        return self.threshold

    def get_top_features(self, top_k: int = 15) -> Dict[str, List[Tuple[str, float]]]:
        """Trích xuất các đặc trưng (từ khóa/cụm ký tự) có trọng số cao nhất."""
        feature_union = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]

        feature_names = []
        for name, trans in feature_union.transformer_list:
            feats = [f"{name}__{f}" for f in trans.get_feature_names_out()]
            feature_names.extend(feats)

        coefs = clf.coef_[0]
        top_pos_idx = np.argsort(coefs)[-top_k:][::-1]
        top_neg_idx = np.argsort(coefs)[:top_k]

        return {
            "top_injection_features": [
                (feature_names[i], round(float(coefs[i]), 4)) for i in top_pos_idx
            ],
            "top_benign_features": [
                (feature_names[i], round(float(coefs[i]), 4)) for i in top_neg_idx
            ],
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: Path) -> "BaselineDetector":
        data = joblib.load(path)
        detector = cls(threshold=data["threshold"])
        detector.pipeline = data["pipeline"]
        return detector


def evaluate_split(
    detector: BaselineDetector, records: List[Dict], threshold: float
) -> Dict:
    """
    Đánh giá chi tiết trên một tập dữ liệu:
    - Precision, Recall, F1 (lớp Injection = 1)
    - FPR tổng thể và FPR riêng trên Hard Negatives
    - Độ trễ CPU (Mean & P95 ms/chunk)
    """
    texts = [r["text"] for r in records]
    y_true = np.array([int(r["label"]) for r in records])
    categories = [str(r.get("category", "")).lower() for r in records]

    # Đo xác suất & latency từng mẫu (mô phỏng quét từng chunk trên CPU)
    probs = []
    latencies_ms = []
    for text in texts:
        t0 = time.perf_counter()
        p = float(detector.pipeline.predict_proba([text])[0, 1])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        probs.append(p)

    probs = np.array(probs)
    y_pred = (probs >= threshold).astype(int)

    # Các chỉ số chính cho lớp INJECTION (label = 1)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    overall_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Tính False Positive Rate (FPR) riêng trên tập Hard Negatives
        # Sửa dòng lọc Hard Negatives trong hàm evaluate_split() của src/detector/baseline.py:
    hard_neg_indices = [
        i
        for i, (label, cat) in enumerate(zip(y_true, categories))
        if label == 0 and cat == "hard_negative"
    ]

    if len(hard_neg_indices) > 0:
        hn_preds = y_pred[hard_neg_indices]
        hn_fp = int(np.sum(hn_preds == 1))
        hn_total = len(hard_neg_indices)
        hard_negative_fpr = hn_fp / hn_total
    else:
        hn_fp = 0
        hn_total = 0
        hard_negative_fpr = 0.0

    return {
        "num_samples": len(records),
        "threshold_used": round(threshold, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "overall_fpr": round(float(overall_fpr), 4),
        "hard_negative_stats": {
            "total_hard_negatives": hn_total,
            "false_positives": hn_fp,
            "hard_negative_fpr": round(float(hard_negative_fpr), 4),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "latency_cpu_ms": {
            "mean_ms": round(float(np.mean(latencies_ms)), 3),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        },
    }


def main():
    data_dir = Path("data/processed")
    model_path = Path("models/baseline_tfidf_lr.joblib")
    results_path = Path("results/baseline_metrics.json")

    print("1. Đang tải dữ liệu từ data/processed/...")
    train_data = load_jsonl(data_dir / "train.jsonl")
    val_data = load_jsonl(data_dir / "val.jsonl")
    test_data = load_jsonl(data_dir / "test.jsonl")
    ood_data = load_jsonl(data_dir / "ood.jsonl")

    print(
        f"   Số lượng mẫu -> Train: {len(train_data)} | Val: {len(val_data)} "
        f"| Test: {len(test_data)} | OOD: {len(ood_data)}"
    )

    # 2. Huấn luyện mô hình Baseline
    print("\n2. Đang huấn luyện TF-IDF (Word + Char n-gram) + Logistic Regression...")
    detector = BaselineDetector(threshold=0.5)
    train_texts = [r["text"] for r in train_data]
    train_labels = [int(r["label"]) for r in train_data]
    train_time_sec = detector.fit(train_texts, train_labels)
    print(f"   Hoàn tất huấn luyện trong {train_time_sec * 1000:.2f} ms.")

    # 3. Hiệu chỉnh ngưỡng trên tập Validation
    val_texts = [r["text"] for r in val_data]
    val_labels = [int(r["label"]) for r in val_data]
    best_th = detector.calibrate_threshold(val_texts, val_labels)
    print(f"   Ngưỡng tối ưu F1 trên tập Validation: {best_th:.4f} (Mặc định: 0.5000)")

    # 4. Đánh giá trên Val, In-Domain Test và OOD Challenge
    print("\n3. Đang đánh giá mô hình trên các tập Val, Test và OOD...")
    val_metrics = evaluate_split(detector, val_data, threshold=best_th)
    test_metrics = evaluate_split(detector, test_data, threshold=best_th)
    ood_metrics = evaluate_split(detector, ood_data, threshold=best_th)

    # Cũng lưu kết quả ở ngưỡng mặc định 0.5 để đối chiếu nếu cần
    test_metrics_default_05 = evaluate_split(detector, test_data, threshold=0.5)
    ood_metrics_default_05 = evaluate_split(detector, ood_data, threshold=0.5)

    top_features = detector.get_top_features(top_k=15)

    # 5. Lưu trọng số và xuất báo cáo JSON
    detector.save(model_path)
    print(f"\n4. Đã lưu trọng số mô hình tại: {model_path}")

    report = {
        "model_name": "TF-IDF (word 1-2 + char_wb 3-5) + LogisticRegression",
        "training_time_sec": round(train_time_sec, 4),
        "calibrated_threshold": round(best_th, 4),
        "metrics_calibrated": {
            "val": val_metrics,
            "in_domain_test": test_metrics,
            "ood_challenge": ood_metrics,
        },
        "metrics_default_threshold_05": {
            "in_domain_test": test_metrics_default_05,
            "ood_challenge": ood_metrics_default_05,
        },
        "explainability_top_features": top_features,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"5. Đã xuất báo cáo đánh giá tại: {results_path}")
    print("\n" + "=" * 65)
    print("TÓM TẮT KẾT QUẢ BASELINE (NGƯỠNG CALIBRATED):")
    print("=" * 65)
    for split_name, m in [
        ("Validation", val_metrics),
        ("In-Domain Test", test_metrics),
        ("OOD Challenge", ood_metrics),
    ]:
        hn_fpr = m["hard_negative_stats"]["hard_negative_fpr"] * 100
        print(
            f"[{split_name:14s}] F1: {m['f1_score']*100:6.2f}% | "
            f"Precision: {m['precision']*100:6.2f}% | "
            f"Recall: {m['recall']*100:6.2f}% | "
            f"HardNeg FPR: {hn_fpr:5.1f}% | "
            f"Latency: {m['latency_cpu_ms']['mean_ms']:.2f} ms"
        )
    print("=" * 65)
    print("\nTop 8 đặc trưng báo hiệu INJECTION mạnh nhất:")
    for feat, weight in top_features["top_injection_features"][:8]:
        print(f"  + {feat:30s}: {weight:+.4f}")


if __name__ == "__main__":
    main()