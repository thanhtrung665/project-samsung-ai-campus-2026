"""Vẽ hình kiến trúc tổng thể (images/fig_kien_truc.png) cho Phần 2 báo cáo."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "images" / "fig_kien_truc.png"
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12, 6.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")

def box(x, y, w, h, text, fc, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec="#334155", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=1.3, color="#334155", ls=ls))

# Lớp người dùng
box(0.2, 4.9, 2.2, 1.1, "Khách hàng\n(Web chat / Fanpage)", "#e0f2fe", bold=True)
box(0.2, 3.2, 2.2, 1.1, "Nhân viên / Quản lý ca\n(nhận chuyển tiếp)", "#e0f2fe")
# Agent
box(3.2, 3.9, 2.8, 2.1, "AGENT CSKH\nLLM đã tinh chỉnh M*\n(Qwen2.5 / LLaMA-3.2 3B\n+ adapter LoRA/QLoRA)\nchọn công cụ (JSON)", "#ede9fe", fs=9.5, bold=False)
arrow(2.4, 5.45, 3.2, 5.3); arrow(3.2, 4.6, 2.4, 4.75)
# Tools
box(6.8, 5.2, 2.3, 0.9, "sql_tool\n(hàm tham số hóa)", "#dcfce7")
box(6.8, 3.95, 2.3, 0.9, "rag_tool\n(Hybrid RAG)", "#dcfce7")
box(6.8, 2.7, 2.3, 0.9, "handoff_tool\n(chuyển nhân viên)", "#dcfce7")
for y in (5.65, 4.4, 3.15):
    arrow(6.0, 4.95, 6.8, y)
arrow(6.8, 3.15, 2.4, 3.75, ls="--")
# Data sources
box(9.8, 5.2, 2.0, 0.9, "SQL (SQLite)\nmenu, giá, chi nhánh,\nbàn, đặt bàn, KM", "#fef9c3", fs=8.5)
box(9.8, 3.75, 2.0, 1.25, "Hybrid RAG\nBM25 + dense bge-m3\n(Qdrant) → RRF\n→ reranker", "#fef9c3", fs=8.5)
box(9.8, 2.5, 2.0, 0.9, "NoSQL\ntài liệu gốc: món,\nchính sách, FAQ", "#fef9c3", fs=8.5)
arrow(9.1, 5.65, 9.8, 5.65); arrow(9.1, 4.4, 9.8, 4.4); arrow(10.8, 3.4, 10.8, 3.75)
# Offline training
box(0.2, 0.3, 11.6, 1.6, "", "#f1f5f9")
ax.text(0.4, 1.65, "HUẤN LUYỆN OFFLINE (chạy một lần)", fontsize=9.5, weight="bold", va="center")
steps = ["Dữ liệu hội thoại\n6.355 mẫu + RAFT,\ntool-call, từ chối", "Lọc & chia split\ntheo nhóm (chống rò rỉ)",
         "SFT LoRA / QLoRA\n{Qwen, LLaMA} × 2", "Đánh giá & chọn M*\n(test-gold đóng băng)", "Adapter M*\n(HF Hub)"]
for i, s in enumerate(steps):
    box(0.4 + i * 2.3, 0.45, 2.0, 1.0, s, "white", fs=8.5)
    if i:
        arrow(0.4 + i * 2.3 - 0.3, 0.95, 0.4 + i * 2.3, 0.95)
arrow(4.6, 1.9, 4.6, 3.9, ls="--")
ax.text(4.7, 2.6, "nạp adapter", fontsize=8.5, style="italic", color="#475569")
fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", OUT)
