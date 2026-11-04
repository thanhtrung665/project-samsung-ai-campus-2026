"""Vòng lặp Agent tối giản (T12 dùng lại): LLM → nếu có <tool_call> thì chạy công cụ → đưa kết quả lại → tối đa N bước.

`generate_fn(messages) -> str` tách khỏi model cụ thể để kiểm thử được trên CPU (giả lập) và dùng llm.generate trên GPU.
Ghi DB chỉ trên bản sao (`db`) — không bao giờ ghi vào data/kimochi.db gốc khi đánh giá.
"""
from __future__ import annotations

import json
from typing import Callable

from kimochi import tools
from kimochi.prepare_sft import SYSTEM_PROMPT


def system_with_tools() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT + tools.TOOLS_PROMPT}


def run(messages: list[dict], generate_fn: Callable[[list[dict]], str], db, max_steps: int = 3) -> dict:
    trace, msgs = [], list(messages)
    for _ in range(max_steps):
        out = generate_fn(msgs)
        call = tools.parse_call(out)
        if call is None:
            return {"answer": out, "trace": trace, "messages": msgs + [{"role": "assistant", "content": out}]}
        result = tools.execute(call, db=db)
        trace.append({"call": call, "result": result})
        msgs += [{"role": "assistant", "content": out}, {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}]
    return {"answer": "Dạ em chuyển nhân viên hỗ trợ mình nha.", "trace": trace, "messages": msgs, "max_steps_hit": True}


def run_batch(model, tok, queries: list[str], db, contexts=None, max_steps: int = 3, bs: int = 8) -> list[dict]:
    """Chạy theo lô trên GPU: mỗi bước sinh cho mọi hội thoại chưa kết thúc."""
    from kimochi import llm
    from kimochi.prompts import format_context
    states = []
    for i, q in enumerate(queries):
        user = q if not contexts else f"Ngữ cảnh:\n{format_context(contexts[i])}\n\nCâu hỏi: {q}"
        states.append({"msgs": [system_with_tools(), {"role": "user", "content": user}], "trace": [], "answer": None, "lat": 0.0})
    for _ in range(max_steps):
        todo = [s for s in states if s["answer"] is None]
        if not todo:
            break
        outs, lats = llm.generate(model, tok, [s["msgs"] for s in todo], bs=bs)
        for s, out, lt in zip(todo, outs, lats):
            s["lat"] += lt
            call = tools.parse_call(out)
            if call is None:
                s["answer"] = out
                continue
            res = tools.execute(call, db=db)
            s["trace"].append({"call": call, "result": res})
            s["msgs"] += [{"role": "assistant", "content": out}, {"role": "tool", "content": json.dumps(res, ensure_ascii=False)}]
    for s in states:
        if s["answer"] is None:
            s["answer"] = "Dạ em chuyển nhân viên hỗ trợ mình nha."
    return [{"pred": s["answer"], "trace": s["trace"], "latency_batch_s": s["lat"],
             "tools_called": [t["call"]["name"] for t in s["trace"]]} for s in states]
