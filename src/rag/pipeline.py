import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag.retriever_llm import LLMGenerator, VectorRetriever

DEFAULT_5_TEST_QUESTIONS = [
    "Nhân viên chính thức làm việc đủ 12 tháng được nghỉ bao nhiêu ngày phép năm và cần gửi đơn trước bao lâu?",
    "Hạn mức phụ cấp lưu trú công tác phí trong nước là bao nhiêu và thời hạn nộp hồ sơ thanh toán là mấy ngày?",
    "Khi gặp cảnh báo chứng chỉ bảo mật lúc đăng nhập FortiClient VPN nội bộ thì nhân viên cần xử lý như thế nào?",
    "Công ty quy định chu kỳ khấu hao nâng cấp Laptop cho khối Kỹ thuật là bao nhiêu tháng và cấu hình tối thiểu ra sao?",
    "Nếu phát hiện máy tính có dấu hiệu nhiễm mã độc Ransomware, nhân viên phải làm gì trong vòng 15 phút đầu tiên?",
]


def main():
    parser = argparse.ArgumentParser(description="CLI kiểm thử RAG Pipeline cơ bản (Ngày 8)")
    parser.add_argument("--query", type=str, default=None, help="Câu hỏi tiếng Việt bất kỳ")
    parser.add_argument("--top_k", type=int, default=3, help="Số lượng chunk đưa vào LLM")
    parser.add_argument(
        "--include_poisoned",
        action="store_true",
        help="Cho phép truy xuất cả các tài liệu bị tiêm mã độc (Mặc định Ngày 8: chỉ truy xuất 40 tài liệu sạch)",
    )
    args = parser.parse_args()

    retriever = VectorRetriever()
    llm = LLMGenerator()

    queries = [args.query] if args.query else DEFAULT_5_TEST_QUESTIONS
    clean_only = not args.include_poisoned

    print("=" * 80)
    print(f"KHỞI CHẠY RAG PIPELINE (LLM: {llm.provider}:{llm.model_name} | Clean-Only: {clean_only})")
    print("=" * 80)

    for idx, q in enumerate(queries, 1):
        print(f"\n[Câu hỏi {idx}/{len(queries)}]: {q}")
        chunks = retriever.retrieve(q, top_k=args.top_k, clean_only=clean_only)
        print("  -> Chunks truy xuất được:")
        for c in chunks:
            print(
                f"     * [{c['chunk_id']}] (Sim: {c['similarity']:.3f} | Cat: {c['category']}): "
                f"{c['text'][:95]}..."
            )

        res = llm.generate(q, [c["text"] for c in chunks], prompt_hardening=True)
        print(f"  -> [Câu trả lời ({res['llm_latency_ms']:.0f} ms)]:\n     {res['answer']}")
        print("-" * 80)


if __name__ == "__main__":
    main()