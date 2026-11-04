"""Biểu đồ kết quả cho NB2 / báo cáo. Mỗi hàm nhận DataFrame kết quả, lưu PNG vào results/ và trả về Figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]


def _save(fig, path):
    if path:
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def loss_curves(histories: dict[str, list[dict]], path=None):
    """histories: {tên ô: trainer.state.log_history}."""
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.2), sharey=True, squeeze=False)
    for ax, (name, h) in zip(axes[0], histories.items()):
        tr = [(x["step"], x["loss"]) for x in h if "loss" in x]
        ev = [(x["step"], x["eval_loss"]) for x in h if "eval_loss" in x]
        if tr:
            ax.plot(*zip(*tr), color=C[0], label="train")
        if ev:
            ax.plot(*zip(*ev), "o-", color=C[3], label="val")
            best = min(ev, key=lambda x: x[1])
            ax.axvline(best[0], color="k", ls=":", lw=1)
        ax.set_title(name, fontsize=9); ax.set_xlabel("bước")
    axes[0][0].set_ylabel("loss"); axes[0][0].legend(fontsize=8)
    return _save(fig, path)


def train_resources(tm: pd.DataFrame, path=None):
    """Peak VRAM, thời gian huấn luyện và val loss theo từng ô ma trận."""
    lab = tm["model"] + "\n" + tm["method"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for a, col, t, c in [(ax[0], "peak_vram_gb", "Peak VRAM (GB)", C[0]), (ax[1], "train_minutes", "Thời gian huấn luyện (phút)", C[2]),
                         (ax[2], "eval_loss", "Val loss", C[3])]:
        a.bar(lab, tm[col], color=c)
        for i, v in enumerate(tm[col]):
            a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        a.set_title(t); a.tick_params(axis="x", labelsize=8)
    return _save(fig, path)


def metric_bars(df: pd.DataFrame, metrics: list[str], ci: dict | None = None, title="", path=None):
    """df: index = hệ thống, cột = chỉ số; ci: {(hệ thống, chỉ số): (thấp, cao)} → thanh sai số."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        v = df[m].astype(float)
        err = None
        if ci:
            err = np.array([[v[s] - ci.get((s, m), (v[s], v[s]))[0] for s in v.index],
                            [ci.get((s, m), (v[s], v[s]))[1] - v[s] for s in v.index]])
        ax.bar(range(len(v)), v.values, yerr=err, capsize=3, color=[C[i % len(C)] for i in range(len(v))])
        ax.set_xticks(range(len(v)), v.index, rotation=35, ha="right", fontsize=8)
        ax.set_title(m)
        for i, x in enumerate(v.values):
            if not np.isnan(x):
                ax.text(i, x, f"{x:.2f}", ha="center", va="bottom", fontsize=7)
    if title:
        fig.suptitle(title)
    return _save(fig, path)


def quality_vs_vram(df: pd.DataFrame, quality="rougeL", path=None):
    """RQ2: chất lượng (trục y) theo peak VRAM (trục x), mỗi điểm là một ô ma trận."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, r in enumerate(df.itertuples()):
        ax.scatter(r.peak_vram_gb, getattr(r, quality), s=80, color=C[i % len(C)])
        ax.annotate(r.label, (r.peak_vram_gb, getattr(r, quality)), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Peak VRAM khi huấn luyện (GB)"); ax.set_ylabel(quality); ax.set_title("Đánh đổi chất lượng – bộ nhớ (RQ2)")
    return _save(fig, path)


def intent_heatmap(scored: dict[str, pd.DataFrame], metric="rougeL", path=None):
    """scored: {hệ thống: DataFrame đã chấm có cột intent}. Ô = trung bình chỉ số theo intent."""
    tab = pd.DataFrame({s: d.groupby("intent")[metric].mean() for s, d in scored.items()})
    fig, ax = plt.subplots(figsize=(1.6 * tab.shape[1] + 3, 0.45 * tab.shape[0] + 1.5))
    im = ax.imshow(tab.values, cmap="RdYlGn", vmin=np.nanmin(tab.values), vmax=np.nanmax(tab.values), aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(tab.shape[0]), tab.index, fontsize=8)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            if not np.isnan(tab.iat[i, j]):
                ax.text(j, i, f"{tab.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.04); ax.set_title(f"{metric} theo intent")
    _save(fig, path)
    return tab


def retrieval_bars(res: pd.DataFrame, path=None):
    r = res[res["status"] == "ok"].set_index("config")
    return metric_bars(r, ["Recall@1", "Recall@5", "MRR@10"], title="Ablation bộ truy xuất", path=path)
