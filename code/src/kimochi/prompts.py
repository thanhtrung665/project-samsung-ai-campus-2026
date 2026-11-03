"""Dựng lời nhắc cho các hệ thống B0 (zero-shot), B1 (5-shot), B2 / M*+RAG (có ngữ cảnh truy xuất)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from kimochi.prepare_sft import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[3]
SPLITS = ROOT / "data" / "splits"
SHOT_INTENTS = ["price_query", "allergy_diet", "recommendation", "menu_info", "store_info"]


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_json(SPLITS / f"{name}.jsonl", lines=True)
    df["query"] = df["messages"].map(lambda m: m[1]["content"])
    df["reference"] = df["messages"].map(lambda m: m[2]["content"])
    for k in ("intent", "item", "noisy_input", "group_id"):
        df[k] = df["meta"].map(lambda m, k=k: m.get(k))
    return df


def load_eval_set(name: str) -> pd.DataFrame:
    """test_synth / val (từ split) hoặc test_gold / test_adv (viết tay, định dạng code/freeze_tests.py)."""
    if name in ("test_synth", "val"):
        return load_split(name)[["id", "query", "reference", "intent", "item", "noisy_input"]]
    df = pd.read_json(SPLITS / f"{name}.jsonl", lines=True)
    for c in ("item", "noisy_input"):
        if c not in df:
            df[c] = None
    return df


def pick_shots(seed: int = 42) -> list[tuple[str, str]]:
    """5 ví dụ cố định từ train, mỗi intent 1 câu — dùng chung cho mọi model (công bằng)."""
    tr = load_split("train")
    return [tuple(tr[tr.intent == it].sample(1, random_state=seed)[["query", "reference"]].iloc[0]) for it in SHOT_INTENTS]


def format_context(chunks: list[dict]) -> str:
    """Định dạng ngữ cảnh DUY NHẤT cho cả huấn luyện (RAFT) và suy luận — chunk đã ≤ 600 ký tự nên không cắt."""
    return "\n".join(f"[{c.get('chunk_id', c['doc_id'])}] {c['title']}: {c['text']}" for c in chunks)


def build_messages(query: str, mode: str = "zero", shots=None, docs=None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if mode == "few":
        for q, a in shots:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    if mode == "rag" and docs:
        query = f"Ngữ cảnh:\n{format_context(docs)}\n\nCâu hỏi: {query}"
    msgs.append({"role": "user", "content": query})
    return msgs
