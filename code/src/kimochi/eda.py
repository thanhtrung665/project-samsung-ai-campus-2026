"""T08 — EDA cho Phần 3. Mỗi hàm vẽ 1 hình vào images/ và trả về các con số dùng để viết nhận xét."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
IMG = ROOT / "images"
SPLITS = ROOT / "data" / "splits"
KB = ROOT / "data" / "kb"
RESULTS = ROOT / "results"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110})
C = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]


def _save(fig, name):
    IMG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=200, bbox_inches="tight")
    return fig


def load_meta() -> pd.DataFrame:
    return pd.read_csv(SPLITS / "all_meta.csv")


def fig_intent_by_split(df):
    ct = pd.crosstab(df["intent"], df["split"]).reindex(columns=["train", "val", "test_synth", "dropped_leak"], fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    share = ct.div(ct.sum(axis=0), axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ct.sum(axis=1).plot.barh(ax=ax[0], color=C[0])
    ax[0].set_title("Số mẫu theo intent (toàn bộ)")
    for i, v in enumerate(ct.sum(axis=1)):
        ax[0].text(v, i, f" {v} ({v / len(df):.0%})", va="center", fontsize=8)
    share[["train", "val", "test_synth"]].plot.barh(ax=ax[1], color=C[:3])
    ax[1].set_title("Tỷ trọng intent trong từng tập")
    ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    _save(fig, "eda_intent_split.png")
    tot = ct.sum(axis=1)
    return {"share": (tot / tot.sum()).round(3).to_dict(),
            "max_abs_share_gap_train_test": float((share["train"] - share["test_synth"]).abs().max())}


def fig_lengths(df, answers: pd.Series):
    q, a = df["q_len_words"], answers.str.split().str.len()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for axi, s, t, col in [(ax[0], q, "Độ dài câu hỏi (từ)", C[0]), (ax[1], a, "Độ dài câu trả lời (từ)", C[1])]:
        axi.hist(s, bins=40, color=col, alpha=.85)
        axi.axvline(s.median(), color="k", ls="--", lw=1)
        axi.set_title(f"{t}\nTB={s.mean():.1f}, trung vị={s.median():.0f}, skew={s.skew():.2f}")
    _save(fig, "eda_lengths.png")
    return {"q_mean": q.mean(), "q_p95": q.quantile(.95), "q_skew": q.skew(), "a_mean": a.mean(), "a_p95": a.quantile(.95),
            "a_skew": a.skew(), "a_max": int(a.max())}


def fig_noise(df):
    g = df.groupby("intent")[["noisy_input", "unaccented"]].mean().sort_values("noisy_input")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    g.plot.barh(ax=ax, color=[C[2], C[3]])
    ax.legend(["Có teen code / viết tắt / không dấu", "Hoàn toàn không dấu"], loc="lower right", fontsize=8)
    ax.set_title("Mức độ 'bẩn' của câu hỏi theo intent")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    _save(fig, "eda_noise.png")
    return {"noisy_rate": df["noisy_input"].mean(), "unaccented_rate": df["unaccented"].mean()}


def fig_group_size(df):
    gs = df.groupby("group_id").size()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    gs.value_counts().sort_index().head(15).plot.bar(ax=ax, color=C[4])
    ax.set_xlabel("Số câu hỏi dùng chung 1 câu trả lời"); ax.set_ylabel("Số nhóm")
    ax.set_title("Phân phối kích thước nhóm diễn đạt lại (group_id)")
    _save(fig, "eda_group_size.png")
    return {"groups": len(gs), "group_mean": gs.mean(), "share_groups_ge3": float((gs >= 3).mean())}


def fig_leakage(df, thr=0.92):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for sp, col in [("val", C[0]), ("test_synth", C[1]), ("dropped_leak", C[3])]:
        s = df.loc[df["split"] == sp, "nn_cos_to_train"].dropna()
        ax.hist(s, bins=40, alpha=.6, color=col, label=f"{sp} (n={len(s)})")
    ax.axvline(thr, color="k", ls="--", lw=1); ax.text(thr, ax.get_ylim()[1] * .9, f" ngưỡng {thr}", fontsize=8)
    ax.set_xlabel("Cosine tới câu train gần nhất (TF-IDF ký tự)"); ax.legend(fontsize=8)
    ax.set_title("Kiểm tra rò rỉ dữ liệu giữa train và val/test")
    _save(fig, "eda_leakage.png")
    kept = df[df["split"].isin(["val", "test_synth"])]["nn_cos_to_train"]
    return {"kept_cos_mean": kept.mean(), "kept_cos_max": kept.max()}


def fig_funnel(kbq: dict, split_stats: dict):
    stages = [("Mẫu gốc", kbq["raw_rows"]),
              ("Gán được món cụ thể", kbq["rows_mapped_to_item"]),
              ("Câu trả lời khác nhau (nhóm)", split_stats["groups"]),
              ("Giữ sau kiểm rò rỉ", kbq["raw_rows"] - split_stats["leakage"]["dropped_leak"])]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh([s for s, _ in stages][::-1], [v for _, v in stages][::-1], color=C[5])
    for i, (_, v) in enumerate(stages[::-1]):
        ax.text(v, i, f" {v:,}".replace(",", "."), va="center", fontsize=9)
    ax.set_title("Phễu dữ liệu hội thoại")
    _save(fig, "eda_funnel.png")
    return dict(stages)


def fig_menu(menu: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    m = menu.dropna(subset=["price_vnd"])
    order = m.groupby("category")["price_vnd"].median().sort_values().index
    ax[0].boxplot([m.loc[m.category == c, "price_vnd"] / 1000 for c in order], orientation="horizontal", tick_labels=list(order))
    ax[0].set_xlabel("Giá (nghìn đồng)"); ax[0].set_title(f"Giá theo danh mục ({len(m)}/{len(menu)} món có giá)")
    al = menu["allergens"].fillna("").str.get_dummies(sep=";")
    by_cat = al.groupby(menu["category"]).mean()
    im = ax[1].imshow(by_cat.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax[1].set_xticks(range(by_cat.shape[1]), by_cat.columns, rotation=45, ha="right")
    ax[1].set_yticks(range(by_cat.shape[0]), by_cat.index)
    ax[1].set_title("Tỷ lệ món chứa nhóm dị ứng (suy luận)")
    fig.colorbar(im, ax=ax[1], fraction=.04)
    _save(fig, "eda_menu.png")
    missing = menu.loc[menu["price_vnd"].isna(), "category"].value_counts().to_dict()
    return {"priced": len(m), "items": len(menu), "price_median": m["price_vnd"].median(), "missing_price_by_cat": missing,
            "allergen_share": al.mean().round(2).to_dict()}


def run_all() -> dict:
    df = load_meta()
    answers = pd.concat([pd.read_json(SPLITS / f"{s}.jsonl", lines=True) for s in ("train", "val", "test_synth")])
    answers = answers["messages"].map(lambda m: m[2]["content"])
    kbq = json.loads((RESULTS / "kb_quality.json").read_text(encoding="utf-8"))
    sst = json.loads((RESULTS / "split_stats.json").read_text(encoding="utf-8"))
    menu = pd.read_csv(KB / "menu.csv")
    out = {"intent": fig_intent_by_split(df), "lengths": fig_lengths(df, answers), "noise": fig_noise(df),
           "groups": fig_group_size(df), "leakage": fig_leakage(df), "funnel": fig_funnel(kbq, sst), "menu": fig_menu(menu)}
    (RESULTS / "eda_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    plt.close("all")
    return out


if __name__ == "__main__":
    print(json.dumps(run_all(), ensure_ascii=False, indent=1, default=float))
