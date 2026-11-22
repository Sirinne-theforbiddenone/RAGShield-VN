import json
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_records(records: list, threshold: float = 0.85) -> list:
    """Loại bỏ các văn bản quá giống nhau (Cosine Similarity > threshold)."""
    if not records:
        return []
    texts = [r["text"] for r in records]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)

    keep_indices = []
    dropped_count = 0
    for i in range(len(records)):
        # Kiểm tra độ tương đồng với các mẫu đã giữ lại trước đó
        is_dup = False
        for kept_idx in keep_indices:
            if sim_matrix[i, kept_idx] > threshold:
                is_dup = True
                dropped_count += 1
                break
        if not is_dup:
            keep_indices.append(i)

    print(f"[Lọc trùng lặp] Giữ lại {len(keep_indices)}/{len(records)} mẫu (Đã loại {dropped_count} mẫu trùng).")
    return [records[i] for i in keep_indices]

def split_by_group(records: list, group_key: str, seed: int = 42):
    """Chia dữ liệu thành Train (70%), Val (15%), Test (15%) theo group_key."""
    groups = np.array([r[group_key] for r in records])
    indices = np.arange(len(records))

    # Bước 1: Tách 70% Train và 30% Temp (Val + Test)
    gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss_train.split(indices, groups=groups))

    train_records = [records[i] for i in train_idx]
    temp_records = [records[i] for i in temp_idx]
    temp_groups = np.array([r[group_key] for r in temp_records])

    # Bước 2: Chia đôi 30% Temp thành 15% Val và 15% Test
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_sub_idx, test_sub_idx = next(gss_val_test.split(np.arange(len(temp_records)), groups=temp_groups))

    val_records = [temp_records[i] for i in val_sub_idx]
    test_records = [temp_records[i] for i in test_sub_idx]

    return train_records, val_records, test_records

def verify_no_leakage(train_data: list, val_data: list, test_data: list):
    """Kiểm tra bắt buộc: Đảm bảo 0% rò rỉ template_id và base_doc_id giữa các tập."""
    def get_sets(data):
        tpls = {r["template_id"] for r in data if r["template_id"] != "none"}
        docs = {r["base_doc_id"] for r in data}
        return tpls, docs

    tr_tpl, tr_doc = get_sets(train_data)
    va_tpl, va_doc = get_sets(val_data)
    te_tpl, te_doc = get_sets(test_data)

    assert tr_tpl.isdisjoint(va_tpl) and tr_tpl.isdisjoint(te_tpl) and va_tpl.isdisjoint(te_tpl), "LỖI: Rò rỉ template_id!"
    assert tr_doc.isdisjoint(va_doc) and tr_doc.isdisjoint(te_doc) and va_doc.isdisjoint(te_doc), "LỖI: Rò rỉ base_doc_id!"
    print("[Kiểm tra bảo mật dữ liệu] THÀNH CÔNG: 0% rò rỉ template_id và base_doc_id giữa Train / Val / Test!")

def save_jsonl(records: list, filepath: str, split_name: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            r["split"] = split_name
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    with open("dataset_500.jsonl", "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f]

    # 1. Tách riêng Benign và Injection để lọc trùng lặp và chia nhóm chuẩn xác
    benign_all = deduplicate_records([r for r in all_records if r["label"] == 0], threshold=0.85)
    inj_all = deduplicate_records([r for r in all_records if r["label"] == 1], threshold=0.90)

    # 2. Chia nhóm độc lập cho từng nhãn
    ben_train, ben_val, ben_test = split_by_group(benign_all, group_key="base_doc_id", seed=42)
    inj_train, inj_val, inj_test = split_by_group(inj_all, group_key="template_id", seed=42)

    # 3. Gộp lại thành 3 tập hoàn chỉnh
    train_set = ben_train + inj_train
    val_set = ben_val + inj_val
    test_set = ben_test + inj_test

    # 4. Chạy kiểm tra chống rò rỉ
    verify_no_leakage(train_set, val_set, test_set)

    # 5. Lưu ra thư mục data/processed/
    save_jsonl(train_set, "data/processed/train.jsonl", "train")
    save_jsonl(val_set, "data/processed/val.jsonl", "val")
    save_jsonl(test_set, "data/processed/test.jsonl", "test")

    for name, ds in [("Train", train_set), ("Val", val_set), ("Test", test_set)]:
        n_ben = sum(1 for r in ds if r["label"] == 0)
        n_inj = sum(1 for r in ds if r["label"] == 1)
        print(f" -> {name:5s}: Tổng {len(ds):3d} mẫu (Benign: {n_ben}, Injection: {n_inj})")

if __name__ == "__main__":
    main()