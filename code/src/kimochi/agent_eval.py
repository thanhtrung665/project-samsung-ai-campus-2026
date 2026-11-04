"""T12 — Đánh giá Agent: (1) định tuyến công cụ (confusion matrix + chi phí kinh doanh), (2) kịch bản đặt bàn nhiều lượt.

Định dạng scenarios.jsonl (viết tay, khóa bằng code/freeze_tests.py):
{"id": "sc-01", "author": "...", "turns": ["đặt bàn 4 người 19h ngày 15/11 gv", "Minh 0901234567", "ok"],
 "expected": {"reservation": {"branch": "GV", "party_size": 4, "start_at": "2026-11-15 19:00", "phone": "0901234567"},
              "confirm_turn": 2, "must_not_call": []}}
- reservation = null ⇒ kịch bản KHÔNG được tạo đặt bàn (vd khách đổi ý, ngoài giờ).
- confirm_turn = chỉ số lượt user xác nhận; create_reservation gọi trước lượt này ⇒ vi phạm human-in-the-loop.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from kimochi import agent, tools

EXPECTED_TOOL = {"price_query": "get_menu_item", "store_info": "get_branch_info", "reservation_create": "check_availability",
                 "complaint": "handoff_to_staff"}
LABELS = ["get_menu_item", "get_branch_info", "check_availability", "search_menu", "handoff_to_staff", "none"]
# Chi phí kinh doanh tương đối của từng loại nhầm (giả định, dùng để xếp hạng mức nghiêm trọng — ghi rõ trong báo cáo)
COST = {("get_menu_item", "none"): 5,          # hỏi giá mà không tra DB → nguy cơ bịa giá
        ("check_availability", "none"): 4,     # mất lượt đặt bàn = mất doanh thu
        ("handoff_to_staff", "none"): 4,       # khiếu nại không được chuyển → mất khách
        ("none", "handoff_to_staff"): 1,       # chuyển thừa → tốn giờ công nhân viên
        ("none", "get_menu_item"): 0.5}        # gọi công cụ thừa → chỉ tăng độ trễ
DEFAULT_COST = 2


def expected_tool(intent: str) -> str:
    return EXPECTED_TOOL.get(intent, "none")


def routing_report(df: pd.DataFrame) -> dict:
    """df: cột intent, tools_called (list). Công cụ đầu tiên được gọi (hoặc 'none') so với công cụ kỳ vọng theo intent."""
    y_true = df["intent"].map(expected_tool)
    y_pred = df["tools_called"].map(lambda t: (t[0] if t else "none") if isinstance(t, (list, np.ndarray)) else "none")
    labels = [lab for lab in LABELS if lab in set(y_true) | set(y_pred)] + sorted((set(y_true) | set(y_pred)) - set(LABELS))
    cm = pd.crosstab(pd.Categorical(y_true, labels), pd.Categorical(y_pred, labels), rownames=["kỳ vọng"], colnames=["dự đoán"],
                     dropna=False)
    per = {}
    for lab in labels:
        tp = cm.loc[lab, lab]
        p = tp / cm[lab].sum() if cm[lab].sum() else np.nan
        r = tp / cm.loc[lab].sum() if cm.loc[lab].sum() else np.nan
        per[lab] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p and r else 0.0, "support": int(cm.loc[lab].sum())}
    cost = sum(cm.loc[t, p] * COST.get((t, p), DEFAULT_COST) for t in labels for p in labels if t != p)
    return {"confusion": cm, "per_label": pd.DataFrame(per).T, "accuracy": float((y_true == y_pred).mean()),
            "macro_f1": float(np.nanmean([v["f1"] for v in per.values() if v["support"]])),
            "business_cost": float(cost), "cost_per_100": float(cost / len(df) * 100)}


def plot_confusion(cm: pd.DataFrame, path: Path, title="Định tuyến công cụ của Agent"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    norm = cm.div(cm.sum(1).replace(0, 1), axis=0)
    ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.iat[i, j]), ha="center", va="center", color="white" if norm.iat[i, j] > .5 else "black", fontsize=9)
    ax.set_xlabel("Công cụ được gọi"); ax.set_ylabel("Công cụ kỳ vọng"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------------------------- kịch bản nhiều lượt
def run_scenario(sc: dict, generate_fn: Callable[[list[dict]], str], base_db: Path, workdir: Path) -> dict:
    db = workdir / f"{sc['id']}.db"
    shutil.copy(base_db, db)
    msgs = [agent.system_with_tools()]
    calls = []  # (turn, name, args)
    answers = []
    for t, user in enumerate(sc["turns"]):
        res = agent.run(msgs + [{"role": "user", "content": user}], generate_fn, db)
        msgs = res["messages"]
        answers.append(res["answer"])
        calls += [(t, c["call"]["name"], c["call"].get("arguments", {})) for c in res["trace"]]
    exp = sc.get("expected", {})
    con = sqlite3.connect(db)
    rows = con.execute("SELECT branch_id, party_size, start_at, phone FROM reservations WHERE status = 'CONFIRMED'").fetchall()
    con.close()
    want = exp.get("reservation")
    if want:
        ok = any(r[0] == want["branch"] and r[1] == int(want["party_size"]) and r[2] == want["start_at"] and
                 (not want.get("phone") or r[3] == want["phone"]) for r in rows)
    else:
        ok = len(rows) == 0
    creates = [t for t, n, _ in calls if n == "create_reservation"]
    ct = exp.get("confirm_turn")
    hitl_ok = all(t >= ct for t in creates) if ct is not None else True
    forbidden = [n for _, n, _ in calls if n in set(exp.get("must_not_call", []))]
    db.unlink(missing_ok=True)
    return {"id": sc["id"], "success": bool(ok and hitl_ok and not forbidden), "db_state_ok": bool(ok), "hitl_ok": hitl_ok,
            "forbidden_calls": forbidden, "n_tool_calls": len(calls), "n_reservations": len(rows),
            "tools": [n for _, n, _ in calls], "answers": answers}


def scenarios_from_v2_val(root: Path, limit: int = 30) -> list[dict]:
    """PROXY khi chưa có scenarios.jsonl viết tay: dựng kịch bản từ hội thoại đặt bàn trong val v2 (mẫu khuôn).
    Chỉ để kiểm tra pipeline — KHÔNG dùng làm số chính thức (cùng phân phối với dữ liệu huấn luyện)."""
    rows = pd.read_json(root / "data" / "splits_v2" / "val.jsonl", lines=True)
    out = []
    for r in rows.itertuples():
        if r.meta["format"] not in ("reservation_mt", "reservation_unavailable"):
            continue
        users = [m["content"] for m in r.messages if m["role"] == "user"]
        creates = [json.loads(m["content"][11:-12]) for m in r.messages
                   if m["role"] == "assistant" and m["content"].startswith("<tool_call>") and "create_reservation" in m["content"]]
        want = None
        if creates:
            a = creates[0]["arguments"]
            want = {"branch": a["branch"], "party_size": a["party_size"], "start_at": a["start_at"], "phone": a["phone"]}
        out.append({"id": f"proxy-{r.id}", "turns": users, "expected": {"reservation": want,
                    "confirm_turn": len(users) - 1 if creates else None, "must_not_call": []}})
    return out[:limit]
