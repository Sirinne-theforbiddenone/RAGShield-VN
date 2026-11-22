import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

import chromadb
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "ragshield_vn_kb"


def get_safe_device() -> str:
    """Kiểm tra xem GPU CUDA có thực thi được kernel thực tế không (hỗ trợ cả RTX 50-series)."""
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda") + 1
            return "cuda"
        except Exception as e:
            print(f"[Cảnh báo] CUDA chưa tương thích kernel ({e}). Tự động chuyển sang CPU.")
    return "cpu"


def split_vietnamese_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ\[<\"'])",
        text,
    )
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(text: str, max_words: int = 180, overlap_sentences: int = 1) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    sentences = split_vietnamese_sentences(text)
    chunks = []
    current_sents = []
    current_word_count = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_word_count + sent_words > max_words and current_sents:
            chunks.append(" ".join(current_sents))
            current_sents = current_sents[-overlap_sentences:] if overlap_sentences > 0 else []
            current_word_count = sum(len(s.split()) for s in current_sents)

        current_sents.append(sent)
        current_word_count += sent_words

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


def load_kb_documents(kb_path: Path, include_poisoned: bool = True) -> List[Dict]:
    if not kb_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {kb_path}. Hãy chạy `python scripts/build_knowledge_base.py` trước."
        )
    docs = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not include_poisoned and item.get("is_poisoned", False):
                continue
            docs.append(item)
    return docs


def build_vector_index(
    kb_path: Path = Path("data/knowledge_base/enterprise_kb.jsonl"),
    chroma_dir: Path = Path(DEFAULT_CHROMA_DIR),
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    include_poisoned: bool = True,
    max_words_per_chunk: int = 180,
    reset_db: bool = True,
):
    docs = load_kb_documents(kb_path, include_poisoned=include_poisoned)
    print(f"1. Đã đọc {len(docs)} tài liệu từ {kb_path} (include_poisoned={include_poisoned})")

    chunk_ids, chunk_texts, chunk_metadatas = [], [], []
    for doc in docs:
        chunks = chunk_document(doc["text"], max_words=max_words_per_chunk)
        for idx, ch_text in enumerate(chunks):
            cid = f"{doc['doc_id']}_chunk_{idx:02d}"
            chunk_ids.append(cid)
            chunk_texts.append(ch_text)
            chunk_metadatas.append({
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "category": doc.get("category", "standard_benign"),
                "is_poisoned": bool(doc.get("is_poisoned", False)),
                "word_count": len(ch_text.split()),
            })

    max_wc = max(m["word_count"] for m in chunk_metadatas) if chunk_metadatas else 0
    print(f"2. Đã chia thành {len(chunk_texts)} chunks (Độ dài lớn nhất: {max_wc} từ < 384 tokens).")

    device = get_safe_device()
    print(f"3. Đang tải mô hình Embedding: {embedding_model_name} (Thiết bị: {device.upper()})...")
    embedder = SentenceTransformer(embedding_model_name, device=device)
    embeddings = embedder.encode(
        chunk_texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    if reset_db and chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": embedding_model_name},
    )

    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=chunk_metadatas,
    )
    print(f"4. Đã lưu thành công {collection.count()} chunks vào ChromaDB tại: {chroma_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Ingest Knowledge Base vào ChromaDB (Ngày 8)")
    parser.add_argument("--kb_path", type=str, default="data/knowledge_base/enterprise_kb.jsonl")
    parser.add_argument("--chroma_dir", type=str, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--clean_only", action="store_true")
    args = parser.parse_args()

    build_vector_index(
        kb_path=Path(args.kb_path),
        chroma_dir=Path(args.chroma_dir),
        embedding_model_name=args.embedding_model,
        include_poisoned=not args.clean_only,
    )


if __name__ == "__main__":
    main()