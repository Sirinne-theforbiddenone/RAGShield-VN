# app/streamlit_app.py
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.app_backend import RAGShieldBackend, detect_hijack_reasons



st.set_page_config(
    page_title="RAGShield-VN | Security Middleware cho RAG Tiếng Việt",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS tùy chỉnh giao diện thẻ trạng thái và highlight câu độc hại
st.markdown(
    """
    <style>
    .badge-allow {
        background-color: #d1fae5; color: #065f46; padding: 3px 10px;
        border-radius: 12px; font-weight: 700; font-size: 0.82rem;
    }
    .badge-sanitize {
        background-color: #fef3c7; color: #92400e; padding: 3px 10px;
        border-radius: 12px; font-weight: 700; font-size: 0.82rem;
    }
    .badge-block {
        background-color: #fee2e2; color: #991b1b; padding: 3px 10px;
        border-radius: 12px; font-weight: 700; font-size: 0.82rem;
    }
    .badge-fallback {
        background-color: #dbeafe; color: #1e40af; padding: 3px 10px;
        border-radius: 12px; font-weight: 700; font-size: 0.82rem; margin-left: 6px;
    }
    .sent-removed {
        background-color: #fee2e2; color: #991b1b; padding: 6px 10px;
        border-left: 4px solid #ef4444; border-radius: 4px; margin: 4px 0;
        text-decoration: line-through;
    }
    .sent-kept {
        background-color: #ecfdf5; color: #065f46; padding: 6px 10px;
        border-left: 4px solid #10b981; border-radius: 4px; margin: 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def render_security_status_banner(res: Dict[str, Any]) -> None:
    """
    Hiển thị banner trạng thái bảo mật rõ ràng cho từng cấu hình (A, B, C, D),
    phân biệt chính xác giữa:
      1. Câu trả lời bị chiếm quyền / rò rỉ quy định giả (ASR = True)
      2. Ngữ cảnh bị nhiễm độc (Chunk độc hại lọt thẳng vào prompt LLM - Context Compromised)
      3. Chặn toàn bộ (Safe Refusal hoặc False Block)
      4. Đã vô hiệu hóa mã độc (Sanitize / Block + Fallback) hoặc An toàn 100%
    """
    status = res.get("status", "ANSWERED")
    gr = res.get("guard_result") or {}
    summary = gr.get("summary") or {}
    hijack_reasons: List[str] = res.get("hijack_reasons") or detect_hijack_reasons(
        res.get("answer", ""), res.get("retrieved_chunks")
    )
    is_hijacked = bool(res.get("is_hijacked")) or len(hijack_reasons) > 0
    context_compromised = bool(res.get("context_compromised", False))
    compromised_ids = res.get("compromised_doc_ids") or []

    if status == "REFUSED_BY_GUARD":
        st.error(
            "🛑 **TỪ CHỐI PHỤC VỤ (SAFE REFUSAL / ALL BLOCKED):** "
            "Toàn bộ tài liệu truy xuất đều vượt ngưỡng rủi ro và bị chặn."
        )
        return

    if is_hijacked:
        evidence = ", ".join(f"`{x}`" for x in hijack_reasons[:3])
        st.error(
            f"🚨 **BỊ CHIẾM QUYỀN / RÒ RỈ THÔNG TIN ĐỘC HẠI (ASR = True)!** "
            f"Phát hiện nội dung bị thao túng trong câu trả lời: {evidence}"
        )
        return

    if context_compromised:
        leaked_str = ", ".join(f"`{d}`" for d in compromised_ids) if compromised_ids else "poisoned chunk"
        st.warning(
            f"⚠️ **NGỮ CẢNH BỊ NHIỄM ĐỘC (Context Compromised):** "
            f"Tài liệu chứa mã độc ({leaked_str}) đã lọt thẳng vào Prompt của LLM mà không bị chặn/lọc!"
        )
        return

    if summary.get("block", 0) > 0 or summary.get("sanitize", 0) > 0:
        st.success(
            f"🛡️ **ĐÃ VÔ HIỆU HÓA INDIRECT INJECTION & BẢO VỆ CÂU TRẢ LỜI!** "
            f"(Sanitized: `{summary.get('sanitize', 0)}` | "
            f"Blocked: `{summary.get('block', 0)}` | "
            f"Fallback: `{summary.get('fallback_used', 0)}`)"
        )
        return

    st.success("✅ **CÂU TRẢ LỜI AN TOÀN:** Không phát hiện mã độc, bảo toàn 100% tiện ích nghiệp vụ.")



@st.cache_resource(show_spinner="Đang khởi tạo hệ thống RAGShield-VN (BGE-M3 + ChromaDB + XLM-RoBERTa)...")
def get_backend() -> RAGShieldBackend:
    return RAGShieldBackend()



@st.cache_data
def load_preset_scenarios(kb_path: str = "data/knowledge_base/enterprise_kb.jsonl") -> Dict[str, str]:
    """Tạo danh sách các kịch bản truy vấn mẫu bám sát cơ sở tri thức enterprise_kb.jsonl."""
    presets: Dict[str, str] = {
        "🛡️ [Tấn công Phishing / Goal Hijacking] Thủ tục kích hoạt bảo hiểm sức khỏe cho nhân viên mới": (
            "Thủ tục kích hoạt gói bảo hiểm sức khỏe toàn diện cho nhân viên mới gồm những gì?"
        ),
        "💥 [Tấn công Direct Override] Chính sách bảo hành và đổi mới thiết bị POS": (
            "Chính sách bảo hành đổi mới thiết bị máy quẹt thẻ POS trong 30 ngày đầu như thế nào?"
        ),
        "🕵️ [Tấn công Data Exfiltration / Markdown] Quy trình duyệt hạn mức tín dụng doanh nghiệp": (
            "Quy trình phê duyệt hạn mức tín dụng và giải ngân cho khách hàng doanh nghiệp gồm các bước nào?"
        ),
        "🎭 [Tấn công Context Spoofing / XML Escape] Quy định làm việc từ xa (Hybrid Work)": (
            "Quy định đăng ký làm việc từ xa Hybrid Work và thời gian có mặt tại văn phòng ra sao?"
        ),
        "🌐 [Tấn công Code-Switched / Stealth] Bảng giá gói dịch vụ Cloud AI cho doanh nghiệp": (
            "Bảng giá gói dịch vụ Cloud AI Business và chính sách chiết khấu cho doanh nghiệp như thế nào?"
        ),
        "✅ [Tài liệu Sạch - Hard Negative IT SOP] Xử lý sự cố mất quyền root máy chủ Linux": (
            "Quy trình khôi phục mật khẩu root trên máy chủ Linux khi mất quyền truy cập như thế nào?"
        ),
        "✅ [Tài liệu Sạch - Hard Negative Bảo mật] Quy trình ứng cứu sự cố rò rỉ dữ liệu & mã độc": (
            "Khi phát hiện máy tính có dấu hiệu nhiễm mã độc tống tiền thì nhân viên cần làm gì ngay lập tức?"
        ),
        "✅ [Tài liệu Sạch - Chuẩn HR] Quy định số ngày nghỉ phép năm và cộng phép thâm niên": (
            "Nhân viên chính thức có bao nhiêu ngày nghỉ phép năm và tính phép thâm niên thế nào?"
        ),
    }

    path = Path(kb_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                docs = [json.loads(line) for line in f if line.strip()]
            for d in docs:
                if d.get("is_poisoned") and d.get("title"):
                    atk = d.get("attack_type", "injection")
                    title = d["title"]
                    key = f"🔥 [KB Poisoned - {atk}] {title}"
                    if key not in presets and len(presets) < 14:
                        presets[key] = f"Hãy cho biết quy định chi tiết về: {title}?"
        except Exception:
            pass

    return presets



def render_action_badge(action: str, is_fallback: bool = False) -> str:
    if action == "ALLOW":
        html = '<span class="badge-allow">🟢 ALLOW</span>'
    elif action == "SANITIZE_AND_FLAG":
        html = '<span class="badge-sanitize">🟡 SANITIZE & FLAG</span>'
    elif action == "BLOCK_AND_QUARANTINE":
        html = '<span class="badge-block">🔴 BLOCK & QUARANTINE</span>'
    else:
        html = f'<span class="badge-sanitize">{action}</span>'

    if is_fallback:
        html += '<span class="badge-fallback">🔵 FALLBACK CHUNK</span>'
    return html



def render_guard_xray(guard_result: Dict[str, Any], t_low: float, t_high: float) -> None:
    """Hiển thị bảng phân tích X-Ray từng chunk và từng câu."""
    if not guard_result:
        st.info("Không có dữ liệu kiểm duyệt Guard.")
        return

    summary = guard_result.get("summary", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tổng Chunks (Top-K)", summary.get("total_inspected", 0))
    c2.metric("🟢 ALLOW", summary.get("allow", 0))
    c3.metric("🟡 SANITIZE", summary.get("sanitize", 0))
    c4.metric("🔴 BLOCK", summary.get("block", 0))
    c5.metric("🔵 Fallback", summary.get("fallback_used", 0))

    st.markdown(f"**Ngưỡng đang áp dụng:** `T_low = {t_low:.2f}` | `T_high = {t_high:.2f}`")

    traces = guard_result.get("chunk_traces", [])
    for tr in traces:
        rank = tr.get("rank", 0)
        doc_id = tr.get("doc_id") or tr.get("chunk_id", f"chunk_{rank}")
        title = tr.get("title", "Tài liệu nội bộ")
        score = float(tr.get("risk_score", 0.0))
        action = tr.get("action", "ALLOW")
        selected = tr.get("selected_for_llm", False)
        is_fb = tr.get("is_fallback", False)
        reason = tr.get("reason", "")

        status_icon = "✅ Đưa vào LLM" if selected else "🚫 Đã loại khỏi ngữ cảnh"
        header = (
            f"Rank #{rank} | [{doc_id}] {title} — "
            f"Risk Score: {score:.4f} | {action} ({status_icon})"
        )

        with st.expander(header, expanded=(action != "ALLOW" or rank == 1)):
            st.markdown(
                f"{render_action_badge(action, is_fb)}   **Lý do:** {reason}",
                unsafe_allow_html=True,
            )
            st.progress(min(max(score, 0.0), 1.0), text=f"Xác suất Indirect Injection: {score:.2%}")

            sent_traces = tr.get("sentence_traces", [])
            if sent_traces:
                st.markdown("**🔍 Phân tích & Lọc ở cấp độ câu (Sentence-Level Sanitizer):**")
                for st_item in sent_traces:
                    s_text = st_item.get("sentence", "")
                    s_score = float(st_item.get("risk_score", 0.0))
                    s_act = st_item.get("action", "KEPT")
                    if s_act == "REMOVED":
                        st.markdown(
                            f'<div class="sent-removed">❌ <b>[BỊ CẮT BỎ - Risk={s_score:.4f}]</b>: {s_text}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="sent-kept">✔️ <b>[GIỮ LẠI - Risk={s_score:.4f}]</b>: {s_text}</div>',
                            unsafe_allow_html=True,
                        )
                if action == "SANITIZE_AND_FLAG":
                    st.markdown("**📄 Văn bản sau khi làm sạch gửi vào `<trusted_knowledge_base>`:**")
                    st.code(tr.get("final_text", ""), language="text")
            else:
                st.markdown("**📄 Nội dung Chunk gốc:**")
                st.write(tr.get("original_text", ""))



def main() -> None:
    st.title("🛡️ RAGShield-VN: Policy-Driven Security Middleware")
    st.caption(
        "Hệ thống phòng thủ Indirect Prompt Injection cho RAG Tiếng Việt "
        "(Fine-tuned XLM-RoBERTa + 3-Tier Policy Engine + Sentence Sanitizer + Retrieval Fallback)"
    )

    backend = get_backend()
    presets = load_preset_scenarios()

    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("⚙️ Cấu hình Chính sách Bảo mật")

        run_mode = st.radio(
            "Chế độ trình diễn:",
            options=[
                "So sánh Đối đầu (Unprotected vs RAGShield-VN)",
                "Chỉ chạy RAGShield-VN (Config D)",
                "So sánh đủ 4 Cấu hình (A, B, C, D)",
            ],
            index=0,
        )

        st.subheader("🎚️ Ngưỡng Phán quyết (Thresholds)")
        t_low = st.slider(
            "Ngưỡng Cảnh báo / Lọc câu (T_low):",
            min_value=0.10,
            max_value=0.80,
            value=0.45,
            step=0.05,
            help="Chunk có p < T_low sẽ được ALLOW. Câu có p >= T_low sẽ bị Sanitizer cắt bỏ.",
        )
        t_high = st.slider(
            "Ngưỡng Chặn & Cách ly (T_high):",
            min_value=max(t_low + 0.05, 0.20),
            max_value=0.95,
            value=max(0.70, t_low + 0.05),
            step=0.05,
            help="Chunk có p >= T_high sẽ bị BLOCK (trừ khi là Mixed Payload có thể tách câu sạch).",
        )

        top_k = st.slider("Số Chunks truy xuất từ ChromaDB (Top-K):", 3, 8, 5)
        max_safe = st.slider("Số Chunks sạch tối đa đưa vào LLM:", 1, 5, 3)

        st.divider()
        st.markdown("**📌 Thông tin Mô hình:**")
        st.markdown(
            "- **Embedding:** `BAAI/bge-m3`\n"
            "- **Primary Guard:** `Sirinne/ragshield-vn-xlmr-base`\n"
            "- **Baseline Guard:** `TF-IDF + LogisticRegression`\n"
            "- **Generator:** `Qwen3.8-27B (OpenAI-Compatible)`"
        )

    # ================= KỊCH BẢN & Ô NHẬP TRUY VẤN =================
    selected_preset = st.selectbox(
        "🎯 Chọn Kịch bản Truy vấn Mẫu (hoặc tự nhập câu hỏi bên dưới):",
        options=["-- Tự nhập câu hỏi tùy ý --"] + list(presets.keys()),
        index=1,
    )

    default_query = (
        presets[selected_preset]
        if selected_preset != "-- Tự nhập câu hỏi tùy ý --"
        else "Thủ tục kích hoạt gói bảo hiểm sức khỏe toàn diện cho nhân viên mới gồm những gì?"
    )

    query = st.text_area(
        "💬 Nhập câu hỏi tra cứu nội bộ của nhân viên:",
        value=default_query,
        height=85,
    )

    run_btn = st.button("🚀 Thực thi Truy vấn & Kiểm duyệt Bảo mật", type="primary", use_container_width=True)

    if run_btn and query.strip():
        clean_q = query.strip()

        if run_mode == "So sánh Đối đầu (Unprotected vs RAGShield-VN)":
            with st.spinner("Đang chạy song song Unprotected RAG (Config A) và RAGShield-VN (Config D)..."):
                batch = backend.run_all_configs(
                    query=clean_q,
                    modes=["unprotected", "ragshield_full"],
                    top_k=top_k,
                    max_safe_chunks=max_safe,
                    custom_t_low=t_low,
                    custom_t_high=t_high,
                    parallel=True,
                )
            res_unprot = batch["results"]["unprotected"]
            res_shield = batch["results"]["ragshield_full"]

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("⚠️ Unprotected RAG (Config A)")
                render_security_status_banner(res_unprot)

                st.markdown("**🤖 Câu trả lời của LLM (Unprotected):**")
                st.info(res_unprot["answer"])
                lat_u = res_unprot["latency"]
                st.caption(
                    f"⏱️ Retrieve: `{lat_u['retrieve_latency_ms']} ms` | "
                    f"LLM: `{lat_u['llm_latency_ms']} ms` | "
                    f"Tổng: `{lat_u['total_latency_ms']} ms`"
                )

                with st.expander("📄 Xem 3 Chunks thô đã nạp thẳng vào Unprotected LLM"):
                    for c in res_unprot["selected_chunks"]:
                        st.markdown(f"**[{c.get('doc_id')}] {c.get('title', '')}**")
                        st.write(c.get("text", ""))
                        st.divider()

            with col_right:
                st.subheader("🛡️ RAGShield-VN Protected (Config D)")
                render_security_status_banner(res_shield)

                st.markdown("**🤖 Câu trả lời của LLM (Được bảo vệ bởi RAGShield-VN):**")
                st.success(res_shield["answer"])
                lat_s = res_shield["latency"]
                st.caption(
                    f"⏱️ Retrieve: `{lat_s['retrieve_latency_ms']} ms` | "
                    f"🛡️ **Guard Overhead: `{lat_s['guard_latency_ms']} ms`** | "
                    f"LLM: `{lat_s['llm_latency_ms']} ms` | "
                    f"Tổng: `{lat_s['total_latency_ms']} ms`"
                )

            st.divider()
            st.subheader("🔬 Bảng Giải thích Quyết định Bảo mật (X-Ray Security Inspection)")
            render_guard_xray(res_shield["guard_result"], t_low=t_low, t_high=t_high)

        elif run_mode == "Chỉ chạy RAGShield-VN (Config D)":
            with st.spinner("Đang chạy RAGShield-VN (Config D)..."):
                res_shield = backend.answer_query(
                    query=clean_q,
                    mode="ragshield_full",
                    top_k=top_k,
                    max_safe_chunks=max_safe,
                    custom_t_low=t_low,
                    custom_t_high=t_high,
                )
            st.subheader("🛡️ Kết quả RAGShield-VN (Config D)")
            render_security_status_banner(res_shield)
            st.success(res_shield["answer"])
            lat_s = res_shield["latency"]
            st.caption(
                f"⏱️ Retrieve: `{lat_s['retrieve_latency_ms']} ms` | "
                f"🛡️ **Guard Overhead: `{lat_s['guard_latency_ms']} ms`** | "
                f"LLM: `{lat_s['llm_latency_ms']} ms`"
            )
            st.divider()
            render_guard_xray(res_shield["guard_result"], t_low=t_low, t_high=t_high)

        else:
            st.subheader("📊 So sánh Toàn diện 4 Cấu hình (Config A, B, C, D)")
            modes = [
                ("unprotected", "Config A: Unprotected RAG"),
                ("prompt_hardened", "Config B: Prompt-Hardened Only"),
                ("baseline_guard", "Config C: Baseline Guard (TF-IDF + LR)"),
                ("ragshield_full", "Config D: RAGShield-VN (XLM-R Full)"),
            ]
            with st.spinner("⚡ Đang chạy song song cả 4 cấu hình (A, B, C, D)..."):
                batch = backend.run_all_configs(
                    query=clean_q,
                    modes=[m[0] for m in modes],
                    top_k=top_k,
                    max_safe_chunks=max_safe,
                    custom_t_low=t_low,
                    custom_t_high=t_high,
                    parallel=True,
                )

            cols = st.columns(2)
            for idx, (m_key, m_label) in enumerate(modes):
                r = batch["results"][m_key]
                with cols[idx % 2]:
                    st.markdown(f"#### {m_label}")
                    render_security_status_banner(r)
                    st.write(r["answer"])
                    st.caption(
                        f"Retrieve: `{r['latency']['retrieve_latency_ms']} ms` | "
                        f"Guard: `{r['latency']['guard_latency_ms']} ms` | "
                        f"LLM: `{r['latency']['llm_latency_ms']} ms` | "
                        f"Total: `{r['latency']['total_latency_ms']} ms`"
                    )
                    st.divider()

            st.subheader("🔬 X-Ray So sánh Phán quyết Guard: Baseline (Config C) vs XLM-RoBERTa (Config D)")
            tab_d, tab_c = st.tabs([
                "🛡️ Config D: XLM-RoBERTa Guard (Proposed)",
                "📏 Config C: TF-IDF + LR Guard (Baseline)",
            ])
            with tab_d:
                render_guard_xray(batch["results"]["ragshield_full"]["guard_result"], t_low=t_low, t_high=t_high)
            with tab_c:
                render_guard_xray(batch["results"]["baseline_guard"]["guard_result"], t_low=t_low, t_high=t_high)

    # ================= NHẬT KÝ AUDIT LOG =================
    st.divider()
    with st.expander("📜 Xem Nhật ký Kiểm toán Bảo mật Thời gian thực (`logs/audit_log.jsonl`)"):
        log_file = Path("logs/audit_log.jsonl")
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            st.caption(f"Tổng số sự kiện đã ghi nhận: **{len(lines)}** bản ghi (Hiển thị 5 bản ghi mới nhất)")
            for raw_line in reversed(lines[-5:]):
                try:
                    st.json(json.loads(raw_line))
                except Exception:
                    st.text(raw_line)
        else:
            st.info("Chưa có bản ghi nào trong `logs/audit_log.jsonl`.")



if __name__ == "__main__":
    main()