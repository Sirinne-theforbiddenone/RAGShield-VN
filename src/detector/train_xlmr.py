import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def seed_everything(seed: int = 42):
    """Cố định toàn bộ seed để đảm bảo tính tái lập (Trang 7)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_jsonl(file_path: Path) -> List[Dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class RAGShieldDataset(Dataset):
    """PyTorch Dataset cho bài toán phát hiện Indirect Prompt Injection."""

    def __init__(self, records: List[Dict], tokenizer, max_length: int = 384):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        item = self.records[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Dùng DataCollatorWithPadding để dynamic padding nhanh hơn
        )
        encoding["labels"] = int(item["label"])
        return encoding


def compute_metrics(eval_pred):
    """Tính các chỉ số đánh giá cho lớp INJECTION (label=1) sau mỗi epoch."""
    logits, labels = eval_pred
    # Tính xác suất softmax cho lớp 1 (INJECTION)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

    return {
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(roc_auc), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa cho RAGShield-VN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Đường dẫn tới file model_config.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Thư mục lưu model (ghi đè local_model_dir trong config nếu truyền vào)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Đẩy model tốt nhất lên Hugging Face Hub sau khi train xong",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Tên repo trên Hugging Face Hub (vd: username/ragshield-vn-xlmr-base)",
    )
    args = parser.parse_args()

    # 1. Đọc cấu hình
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)

    output_dir = Path(args.output_dir or model_cfg["local_model_dir"])
    checkpoint_dir = output_dir.parent / "checkpoints_xlmr"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = bool(train_cfg.get("fp16", True) and torch.cuda.is_available())
    print(f"=== Thiết bị huấn luyện: {device.upper()} (FP16: {use_fp16}) ===")

    # 2. Tải dữ liệu
    train_records = load_jsonl(Path(data_cfg["train_path"]))
    val_records = load_jsonl(Path(data_cfg["val_path"]))
    print(f"=== Số mẫu Train: {len(train_records)} | Val: {len(val_records)} ===")

    # 3. Khởi tạo Tokenizer & Model
    pretrained_name = model_cfg["pretrained_name"]
    max_length = int(model_cfg.get("max_length", 384))

    print(f"=== Đang tải Tokenizer & Model: {pretrained_name} (max_length={max_length}) ===")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    id2label = {0: "BENIGN", 1: "INJECTION"}
    label2id = {"BENIGN": 0, "INJECTION": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=int(model_cfg.get("num_labels", 2)),
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = RAGShieldDataset(train_records, tokenizer, max_length=max_length)
    val_dataset = RAGShieldDataset(val_records, tokenizer, max_length=max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Thiết lập TrainingArguments (Tương thích cả transformers cũ và mới)
    ta_kwargs = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": float(train_cfg.get("epochs", 5)),
        "per_device_train_batch_size": int(train_cfg.get("batch_size", 16)),
        "per_device_eval_batch_size": int(train_cfg.get("batch_size", 16)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-5)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.10)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "fp16": use_fp16,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": train_cfg.get("metric_for_best_model", "f1"),
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": seed,
        "report_to": "none",
    }

    # Kiểm tra tên tham số evaluation_strategy vs eval_strategy
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(train_cfg.get("early_stopping_patience", 2))
            )
        ],
    )

    # 5. Bắt đầu Fine-tune
    print("\n=== BẮT ĐẦU HUẤN LUYỆN XLM-ROBERTA-BASE ===")
    train_result = trainer.train()

    # 6. Đánh giá nhanh trên tập Validation với checkpoint tốt nhất
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("\n" + "=" * 60)
    print("KẾT QUẢ CHECKPOINT TỐT NHẤT TRÊN TẬP VALIDATION:")
    print(f"  + Validation F1       : {val_metrics['eval_f1'] * 100:.2f}%")
    print(f"  + Validation Precision: {val_metrics['eval_precision'] * 100:.2f}%")
    print(f"  + Validation Recall   : {val_metrics['eval_recall'] * 100:.2f}%")
    print(f"  + Validation ROC-AUC  : {val_metrics['eval_roc_auc'] * 100:.2f}%")
    print("=" * 60)

    # 7. Lưu model và tokenizer tốt nhất ra thư mục đích
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n=== Đã lưu Model & Tokenizer tốt nhất tại: {output_dir} ===")

    # Lưu log lịch sử huấn luyện để vẽ biểu đồ nếu cần
    history_path = Path("results/xlmr_train_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_metrics": val_metrics,
                "train_runtime_sec": train_result.metrics.get("train_runtime", 0),
                "log_history": trainer.state.log_history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"=== Đã lưu lịch sử huấn luyện tại: {history_path} ===")

    # 8. Đẩy lên Hugging Face Hub (nếu bật cờ --push_to_hub)
    if args.push_to_hub:
        hub_id = args.hub_model_id or model_cfg.get("hub_model_id")
        print(f"\n=== Đang đẩy mô hình lên Hugging Face Hub: {hub_id} ===")
        trainer.model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)
        print(f"=== Hoàn tất đẩy lên Hub: https://huggingface.co/{hub_id} ===")


if __name__ == "__main__":
    main()