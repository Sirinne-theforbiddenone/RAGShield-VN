# File: scripts/plot_results.py
"""
Day 13 & 14: High-Resolution Visualization Suite (300 DPI)
Chuẩn hóa nhãn học thuật:
  - Dự án / Khung hệ thống: RAGShield-VN Framework
  - Cấp độ Detector: Baseline (TF-IDF + LR) vs. Proposed (Fine-tuned XLM-RoBERTa-base)
  - Cấp độ E2E System: Config A (Unprotected), Config B (Prompt-Hardened),
    Config C (TF-IDF + LR Guard), Config D (XLM-RoBERTa Guard - Proposed Full)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
E2E_SUMMARY_JSON = PROJECT_ROOT / "results" / "e2e_benchmark_summary.json"
E2E_CSV = PROJECT_ROOT / "results" / "e2e_benchmark_results.csv"

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


def add_bar_labels(ax, bars, fmt="{:.1f}%", fontsize=9, offset=3):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color="#222222",
        )


def plot_fig1_detector_performance():
    splits = ["Validation\n(n=74)", "In-Domain Test\n(n=76)", "OOD Challenge\n(n=100)"]

    base_f1 = [87.18, 80.49, 66.67]
    xlmr_f1 = [100.00, 98.70, 84.75]
    base_hn_fpr = [25.00, 36.36, 96.15]
    xlmr_hn_fpr = [0.00, 0.00, 65.38]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(splits))
    width = 0.35

    b1 = ax1.bar(x - width / 2, base_f1, width, label="Baseline: TF-IDF + Logistic Regression", color="#6C8EBF", edgecolor="white")
    b2 = ax1.bar(x + width / 2, xlmr_f1, width, label="Proposed: Fine-tuned XLM-RoBERTa-base", color="#2E7D32", edgecolor="white")
    ax1.set_title("(a) So sánh F1-Score (%) ↑ (Cao hơn là tốt hơn)", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("F1-Score (%)", fontsize=10.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", frameon=True, fontsize=9.5)
    add_bar_labels(ax1, b1, fmt="{:.1f}%")
    add_bar_labels(ax1, b2, fmt="{:.1f}%")

    b3 = ax2.bar(x - width / 2, base_hn_fpr, width, label="Baseline: TF-IDF + Logistic Regression", color="#E07A5F", edgecolor="white")
    b4 = ax2.bar(x + width / 2, xlmr_hn_fpr, width, label="Proposed: Fine-tuned XLM-RoBERTa-base", color="#F4A261", edgecolor="white")
    ax2.set_title("(b) Tỷ lệ Báo động giả trên Hard Negatives (FPR %) ↓", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits, fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Hard Negative FPR (%)", fontsize=10.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    add_bar_labels(ax2, b3, fmt="{:.1f}%")
    add_bar_labels(ax2, b4, fmt="{:.1f}%")

    fig.suptitle(
        "RAGShield-VN — Đánh Giá Mô Hình Phát Hiện: TF-IDF + LR (Baseline) vs. Fine-tuned XLM-RoBERTa",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig1_detector_performance.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Đã cập nhật Hình 1: {out_path}")


def plot_fig2_e2e_benchmark(summary_data: dict, df_e2e: pd.DataFrame):
    configs = summary_data["configs"]
    method_labels = [
        "A. Unprotected\nRAG",
        "B. Prompt-Hardened\nOnly",
        "C. TF-IDF + LR\nGuard (Baseline)",
        "D. XLM-RoBERTa\nGuard (Proposed)",
    ]

    context_compromised_pct = []
    for c in configs:
        ckey = c["config_key"]
        if not df_e2e.empty:
            atk_rows = df_e2e[(df_e2e["config_key"] == ckey) & (df_e2e["query_type"] == "attack_target")]
            comp_rate = float(atk_rows["context_compromised"].mean() * 100.0) if len(atk_rows) else c["asr_pct"]
        else:
            comp_rate = c["asr_pct"]
        context_compromised_pct.append(comp_rate)

    benign_util = [c["benign_utility_pct"] for c in configs]
    hn_fbr = [c["hard_negative_fbr_pct"] for c in configs]

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(len(method_labels))
    width = 0.25

    bars_atk = ax.bar(
        x - width,
        context_compromised_pct,
        width,
        label="Tỷ lệ Lọt Mã Độc Vào Context / ASR (%) ↓",
        color="#D62828",
        edgecolor="white",
    )
    bars_util = ax.bar(
        x,
        benign_util,
        width,
        label="Benign Utility Retention (%) ↑",
        color="#2A9D8F",
        edgecolor="white",
    )
    bars_fbr = ax.bar(
        x + width,
        hn_fbr,
        width,
        label="Hard Negative False Block Rate (%) ↓",
        color="#F77F00",
        edgecolor="white",
    )

    ax.set_title(
        "RAGShield-VN — So Sánh 4 Cấu Hình Pipeline Trên 60 Truy Vấn (30 Attack + 30 Benign)",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_ylabel("Tỷ lệ (%)", fontsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    add_bar_labels(ax, bars_atk, fmt="{:.1f}%", fontsize=8.5)
    add_bar_labels(ax, bars_util, fmt="{:.1f}%", fontsize=8.5)
    add_bar_labels(ax, bars_fbr, fmt="{:.1f}%", fontsize=8.5)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig2_e2e_benchmark_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Đã cập nhật Hình 2: {out_path}")


def plot_fig3_guard_actions(summary_data: dict):
    configs_by_key = {c["config_key"]: c for c in summary_data["configs"]}
    c_base = configs_by_key["baseline_guard"]
    c_xlmr = configs_by_key["ragshield_full"]

    categories = [
        "ALLOW\n(Cho phép nguyên vẹn)",
        "SANITIZE_AND_FLAG\n(Cắt bỏ câu độc hại, giữ câu sạch)",
        "BLOCK_AND_QUARANTINE\n(Khóa & cách ly toàn bộ chunk)",
    ]
    base_counts = [
        c_base["total_allowed_chunks"],
        c_base["total_sanitized_chunks"],
        c_base["total_blocked_chunks"],
    ]
    xlmr_counts = [
        c_xlmr["total_allowed_chunks"],
        c_xlmr["total_sanitized_chunks"],
        c_xlmr["total_blocked_chunks"],
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(categories))
    width = 0.34

    b1 = ax.bar(x - width / 2, base_counts, width, label="Config C: TF-IDF + LR Guard (Baseline)", color="#577590", edgecolor="white")
    b2 = ax.bar(x + width / 2, xlmr_counts, width, label="Config D: XLM-RoBERTa Guard (Proposed)", color="#43AA8B", edgecolor="white")

    ax.set_title(
        "RAGShield-VN — Phân Bố Quyết Định Của Guard Policy Engine Trên 300 Chunks (60 Queries × Top-5)",
        fontsize=11.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, max(max(base_counts), max(xlmr_counts)) * 1.22)
    ax.set_ylabel("Số lượng Chunks", fontsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    add_bar_labels(ax, b1, fmt="{:.0f} chunks", fontsize=9)
    add_bar_labels(ax, b2, fmt="{:.0f} chunks", fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig3_guard_action_distribution.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Đã cập nhật Hình 3: {out_path}")


def plot_fig4_latency_breakdown(df_e2e: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    config_names = [
        "A. Unprotected\nRAG",
        "B. Prompt-Hardened\nOnly",
        "C. TF-IDF + LR\nGuard (CPU)",
        "D. XLM-RoBERTa\nGuard (GPU)",
    ]
    if not df_e2e.empty:
        guard_lat = [
            df_e2e[df_e2e["config_key"] == k]["guard_latency_ms"].mean()
            for k in ["unprotected", "prompt_hardened", "baseline_guard", "ragshield_full"]
        ]
        ret_lat = df_e2e[df_e2e["config_key"] == "ragshield_full"]["retrieval_latency_ms"].mean()
        xlmr_guard_lat = guard_lat[3]
    else:
        guard_lat = [0.0, 0.0, 15.76, 113.19]
        ret_lat = 28.5
        xlmr_guard_lat = 113.19

    colors = ["#ADB5BD", "#ADB5BD", "#4EA8DE", "#2B9348"]
    bars = ax1.bar(config_names, guard_lat, color=colors, width=0.5, edgecolor="white")
    ax1.axhline(150, color="#D62828", linestyle="--", linewidth=1.2, label="Ngưỡng SLA Tối đa (150 ms)")
    ax1.set_title("(a) Độ Trễ Trung Bình Tầng Guard Middleware (ms) ↓", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_ylabel("Độ trễ Guard (ms / query)", fontsize=10)
    ax1.set_ylim(0, 175)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=9)
    add_bar_labels(ax1, bars, fmt="{:.2f} ms", fontsize=9)

    stages = [
        "Vector Retrieval\n(ChromaDB + BGE-M3)",
        "XLM-RoBERTa Guard\n(Scoring + Sanitizer)",
        "Tổng Local Overhead\n(Trước khi gọi LLM)",
    ]
    stage_vals = [ret_lat, xlmr_guard_lat, ret_lat + xlmr_guard_lat]
    b_stages = ax2.bar(stages, stage_vals, color=["#457B9D", "#2A9D8F", "#1D3557"], width=0.48, edgecolor="white")
    ax2.set_title("(b) Phân Rã Thời Gian Xử Lý Local (Config D - ms)", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_ylabel("Thời gian xử lý (ms)", fontsize=10)
    ax2.set_ylim(0, max(stage_vals) * 1.28)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    add_bar_labels(ax2, b_stages, fmt="{:.1f} ms", fontsize=9)

    fig.suptitle(
        "RAGShield-VN — Phân Tích Độ Trễ Hệ Thống (Latency Overhead Analysis)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig4_latency_and_thresholds.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Đã cập nhật Hình 4: {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(E2E_SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    df_e2e = pd.read_csv(E2E_CSV) if E2E_CSV.exists() else pd.DataFrame()

    plot_fig1_detector_performance()
    plot_fig2_e2e_benchmark(summary_data, df_e2e)
    plot_fig3_guard_actions(summary_data)
    plot_fig4_latency_breakdown(df_e2e)


if __name__ == "__main__":
    main()