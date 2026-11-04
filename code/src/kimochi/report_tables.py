"""Bảng / hình cho Phần 4–6 của báo cáo, đọc từ results/. File chưa có → ô 'TBD' (CLAUDE.md §3.2: không bịa số)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
TBD = "TBD"


def _csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def _json(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(v, nd=3, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return TBD
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", ".")
    if pct:
        return f"{v * 100:.1f}%".replace(".", ",")
    return f"{v:.{nd}f}".replace(".", ",")


def md_table(df: pd.DataFrame, align=None) -> str:
    cols = list(df.columns)
    align = align or ["l"] + ["c"] * (len(cols) - 1)
    sep = {"l": ":---", "c": ":---:", "r": "---:"}
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(sep[a] for a in align) + " |"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def status_note(files: list[str]) -> str:
    missing = [f for f in files if not (RES / f).exists()]
    return "" if not missing else f"*(Ô TBD: chưa có `{'`, `'.join(missing)}` — điền tự động sau khi chạy notebook Kaggle.)*"


# ---------------------------------------------------------------------------------------------- Phần 4
SYSTEMS = [("qwen", "B0", "zero"), ("qwen", "B1", "few"), ("qwen", "B2", "rag"),
           ("llama", "B0", "zero"), ("llama", "B1", "few"), ("llama", "B2", "rag")]


def baseline_table(eval_set="test_synth") -> str:
    b = _csv("baseline.csv")
    rows = []
    for mk, code, _ in SYSTEMS:
        r = None if b is None else b[(b.system == f"{mk}_{code}") & (b.eval_set == eval_set)]
        r = None if r is None or r.empty else r.iloc[0]
        lat = None if b is None else b[(b.system == f"{mk}_B2_latency")]
        rows.append({"Hệ thống": f"{'Qwen2.5-3B' if mk == 'qwen' else 'LLaMA-3.2-3B'} · {code}",
                     "ROUGE-L": fmt(r["rougeL"]) if r is not None else TBD,
                     "Fact-EM giá": fmt(r["fact_em"], pct=True) if r is not None else TBD,
                     "Bịa giá": fmt(r["hallucinated_price_rate"], pct=True) if r is not None else TBD,
                     "Từ chối": fmt(r["refusal_rate"], pct=True) if r is not None else TBD,
                     "p95 (s)": fmt(lat.iloc[0]["latency_p95"], 2) if code == "B2" and lat is not None and not lat.empty else "—"})
    return md_table(pd.DataFrame(rows))


def train_matrix_table() -> str:
    t = _csv("train_matrix.csv")
    rows = []
    for mk in ("qwen", "llama"):
        for q in ("fp16", "nf4"):
            r = None if t is None else t[(t.model == mk) & (t.quant == q)]
            ok = r is not None and not r.empty
            g = r.agg({"eval_loss": ["mean", "std"]}) if ok else None
            rows.append({"Mô hình": "Qwen2.5-3B" if mk == "qwen" else "LLaMA-3.2-3B",
                         "Kỹ thuật": "LoRA (fp16)" if q == "fp16" else "QLoRA (NF4)",
                         "Tham số huấn luyện": fmt(int(r.trainable_params.iloc[0])) if ok else TBD,
                         "Peak VRAM (GB)": fmt(r.peak_vram_gb.mean(), 2) if ok else TBD,
                         "Thời gian (phút)": fmt(r.train_minutes.mean(), 1) if ok else TBD,
                         "Val loss": (fmt(g.loc["mean", "eval_loss"]) + (f" ± {fmt(g.loc['std', 'eval_loss'])}" if len(r) > 1 else ""))
                         if ok else TBD,
                         "Seed": str(len(r)) if ok else TBD})
    return md_table(pd.DataFrame(rows))


def optuna_table() -> str:
    o = _json("optuna_best.json")
    tr = _csv("optuna_trials.csv")
    rows = [{"Siêu tham số": "Hạng LoRA r", "Không gian tìm kiếm": "{8, 16, 32}",
             "Giá trị tốt nhất": str(o["best_params"]["r"]) if o else TBD},
            {"Siêu tham số": "Learning rate", "Không gian tìm kiếm": "5e-5 – 3e-4 (log-uniform)",
             "Giá trị tốt nhất": f"{o['best_params']['lr']:.2e}" if o else TBD},
            {"Siêu tham số": "lora_alpha", "Không gian tìm kiếm": "cố định = 2r", "Giá trị tốt nhất": "2r"},
            {"Siêu tham số": "Số trial / bị cắt sớm", "Không gian tìm kiếm": "8 (MedianPruner)",
             "Giá trị tốt nhất": f"{len(tr)} / {int((tr.state == 'PRUNED').sum())}" if tr is not None else TBD},
            {"Siêu tham số": "Val loss tốt nhất", "Không gian tìm kiếm": "—", "Giá trị tốt nhất": fmt(o["best_val_loss"]) if o else TBD}]
    return md_table(pd.DataFrame(rows))


def data_ablation_table() -> str:
    v1, t = _csv("train_v1data.csv"), _csv("train_matrix.csv")
    v2 = None if t is None else t[(t.model == "qwen") & (t.quant == "nf4")]
    ev = _csv("finetuned_eval.csv")

    def ev_of(sys_prefix, col):
        if ev is None:
            return TBD
        r = ev[ev.system.str.startswith(sys_prefix) & (ev.eval_set == "test_synth")]
        return fmt(r[col].iloc[0], pct=col != "rougeL") if not r.empty and col in r else TBD
    rows = [{"Dữ liệu": "v1 (closed-book)", "Val loss": fmt(v1.eval_loss.iloc[0]) if v1 is not None else TBD,
             "Bịa giá (agent)": ev_of("qwen_nf4_s42_v1data_agent", "hallucinated_price_rate"),
             "Từ chối (agent)": ev_of("qwen_nf4_s42_v1data_agent", "refusal_rate")},
            {"Dữ liệu": "v2 (RAFT + công cụ)", "Val loss": fmt(v2.eval_loss.iloc[0]) if v2 is not None and not v2.empty else TBD,
             "Bịa giá (agent)": ev_of("qwen_nf4_s42_agent", "hallucinated_price_rate"),
             "Từ chối (agent)": ev_of("qwen_nf4_s42_agent", "refusal_rate")}]
    return md_table(pd.DataFrame(rows))


def retrieval_table() -> str:
    r = _csv("retrieval_test_synth.csv")
    names = {"bm25": "BM25", "bm25_filter": "BM25 + lọc tên món", "tfidf_char": "TF-IDF n-gram ký tự", "dense": "Dense bge-m3",
             "hybrid_sparse": "RRF(BM25, TF-IDF)", "hybrid_dense": "RRF(BM25, TF-IDF, dense)",
             "hybrid_dense_rerank": "RRF + reranker", "hybrid_filter": "RRF(BM25, TF-IDF) + lọc",
             "hybrid_dense_rerank_filter": "RRF + reranker + lọc"}
    rows = []
    for k, v in names.items():
        x = None if r is None else r[r.config == k]
        ok = x is not None and not x.empty and x.status.iloc[0] == "ok"
        rows.append({"Cấu hình": v, **{m: fmt(x[m].iloc[0]) if ok else TBD for m in ("Recall@1", "Recall@5", "MRR@10", "nDCG@5")},
                     "Recall@5 (câu nhiễu)": fmt(x["Recall@5_noisy"].iloc[0]) if ok else TBD})
    return md_table(pd.DataFrame(rows))


# ---------------------------------------------------------------------------------------------- Phần 5
def final_eval_table(eval_set=None) -> str:
    ev = _csv("finetuned_eval.csv")
    eval_set = eval_set or ("test_gold" if ev is not None and (ev.eval_set == "test_gold").any() else "test_synth")
    if ev is None:
        rows = [{"Hệ thống": s, "ROUGE-L": TBD, "Fact-EM giá": TBD, "Bịa giá": TBD, "Từ chối": TBD, "Judge (1–5)": TBD}
                for s in ("M1 Qwen LoRA", "M2 Qwen QLoRA", "M3 LLaMA LoRA", "M4 LLaMA QLoRA", "M* + RAG", "M* + RAG + Agent")]
        return md_table(pd.DataFrame(rows))
    j = _csv("judge_scores.csv")
    jm = j.groupby("system").judge.mean() if j is not None else pd.Series(dtype=float)
    e = ev[ev.eval_set == eval_set]
    rows = [{"Hệ thống": r.system, "ROUGE-L": fmt(r.rougeL), "Fact-EM giá": fmt(r.fact_em, pct=True),
             "Bịa giá": fmt(r.hallucinated_price_rate, pct=True), "Từ chối": fmt(r.refusal_rate, pct=True),
             "Judge (1–5)": fmt(jm.get(r.system), 2)} for r in e.itertuples()]
    return md_table(pd.DataFrame(rows))


def agent_table() -> str:
    a, t = _json("agent_summary.json"), _json("tool_routing_summary.json")
    rows = [{"Chỉ số": "Hoàn thành kịch bản đặt bàn", "Giá trị": fmt(a["task_success"], pct=True) if a else TBD, "Mục tiêu": "≥ 90%"},
            {"Chỉ số": "Trạng thái DB đúng", "Giá trị": fmt(a["db_state_ok"], pct=True) if a else TBD, "Mục tiêu": "—"},
            {"Chỉ số": "Tuân thủ xác nhận trước khi ghi (HITL)", "Giá trị": fmt(a["hitl_ok"], pct=True) if a else TBD, "Mục tiêu": "100%"},
            {"Chỉ số": "Độ chính xác định tuyến công cụ", "Giá trị": fmt(t["accuracy"], pct=True) if t else TBD, "Mục tiêu": "—"},
            {"Chỉ số": "Macro-F1 định tuyến", "Giá trị": fmt(t["macro_f1"]) if t else TBD, "Mục tiêu": "—"},
            {"Chỉ số": "Chi phí nhầm lẫn / 100 câu (đơn vị giả định)", "Giá trị": fmt(t["cost_per_100"], 1) if t else TBD, "Mục tiêu": "—"}]
    note = f" (kịch bản: {a['kind']}, n = {a['n']})" if a else ""
    return md_table(pd.DataFrame(rows)) + ("\n\n*Nguồn kịch bản" + note + ".*" if a else "")


def cost_table() -> str:
    from kimochi.agent_eval import COST
    names = {"get_menu_item": "tra giá", "check_availability": "kiểm tra bàn", "handoff_to_staff": "chuyển nhân viên", "none": "không gọi"}
    reasons = {("get_menu_item", "none"): "Trả lời giá không qua DB → nguy cơ bịa giá, sai cam kết với khách",
               ("check_availability", "none"): "Bỏ lỡ lượt đặt bàn → mất doanh thu trực tiếp",
               ("handoff_to_staff", "none"): "Khiếu nại không được chuyển → mất khách, đánh giá xấu",
               ("none", "handoff_to_staff"): "Chuyển thừa → tốn giờ công nhân viên",
               ("none", "get_menu_item"): "Gọi công cụ thừa → chỉ tăng độ trễ"}
    rows = [{"Kỳ vọng → Thực tế": f"{names[a]} → {names[b]}", "Hậu quả kinh doanh": reasons[(a, b)], "Trọng số chi phí": fmt(c, 1)}
            for (a, b), c in COST.items()]
    return md_table(pd.DataFrame(rows), ["l", "l", "c"])


def bm25_explain_table(query="com cuon kieu cali bao nhieu tien", top=6) -> str:
    from kimochi.retriever import HybridRetriever, load_docs, explain_bm25
    r = HybridRetriever(load_docs())
    ex = explain_bm25(r, query, top_terms=top)
    rows = [{"Token (đã bỏ dấu)": f"`{t}`", "Đóng góp điểm BM25": fmt(s, 2), "Tỷ trọng": fmt(s / ex["score"], pct=True)}
            for t, s in ex["terms"]]
    return md_table(pd.DataFrame(rows)), ex


def figure_or_tbd(name: str, caption: str, width: str = "90%") -> None:
    """In Markdown ảnh từ results/ (dùng với `#| output: asis`); chưa có file → ghi chú TBD thay cho hình."""
    p = RES / name
    if p.exists():
        print(f"![{caption}](results/{name}){{width={width}}}\n")
    else:
        print(f"> **[TBD – Hình]** {caption} *(sinh tự động khi có `results/{name}` từ notebook Kaggle).*\n")
