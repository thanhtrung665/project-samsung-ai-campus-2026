"""Chỉ số đánh giá (chạy CPU): ROUGE-L, fact-EM giá, bịa giá, từ chối, bootstrap CI, LLM-as-judge."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from kimochi.build_kb import fold, prices_in

ROOT = Path(__file__).resolve().parents[3]
REFUSAL = re.compile(r"chưa có thông tin|không có thông tin|chuyển (cho )?nhân viên|liên hệ hotline|không thể hỗ trợ|"
                     r"ngoài phạm vi|em chưa rõ|không chắc", re.I)


def rouge_l(pred: str, ref: str) -> float:
    a, b = fold(pred).split(), fold(ref).split()
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    lcs = dp[-1]
    p, r = lcs / len(a), lcs / len(b)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def menu_prices() -> dict[str, float | None]:
    m = pd.read_csv(ROOT / "data" / "kb" / "menu.csv")
    return {n: (None if pd.isna(p) else int(p)) for n, p in zip(m["name"], m["price_vnd"])}


def price_check(pred: str, item: str | None, intent: str | None, prices: dict) -> dict:
    """fact_ok: với câu hỏi giá của món có giá trong DB → câu trả lời phải nêu đúng giá và không nêu giá sai.
    hallucinated_price: món KHÔNG có giá trong DB nhưng câu trả lời vẫn đưa ra con số giá."""
    said = set(prices_in(pred))
    if not item or item not in prices:
        return {"fact_applicable": False, "fact_ok": np.nan, "hallucinated_price": np.nan}
    db = prices[item]
    if db is None:
        return {"fact_applicable": False, "fact_ok": np.nan, "hallucinated_price": float(bool(said))}
    if intent != "price_query" and not said:
        return {"fact_applicable": False, "fact_ok": np.nan, "hallucinated_price": np.nan}
    return {"fact_applicable": True, "fact_ok": float(db in said and said <= {db}), "hallucinated_price": np.nan}


def score_generations(df: pd.DataFrame) -> pd.DataFrame:
    """df cần cột: pred, reference, item, intent. Trả về df kèm các cột chỉ số từng mẫu."""
    prices = menu_prices()
    out = df.copy()
    out["rougeL"] = [rouge_l(p, r) for p, r in zip(out["pred"], out["reference"])]
    pc = pd.DataFrame([price_check(p, i, t, prices) for p, i, t in zip(out["pred"], out["item"], out["intent"])], index=out.index)
    out = pd.concat([out, pc], axis=1)
    out["refused"] = out["pred"].map(lambda s: bool(REFUSAL.search(s)))
    return out


def summarize(scored: pd.DataFrame) -> dict:
    s = {"n": len(scored), "rougeL": scored["rougeL"].mean(),
         "fact_em": scored["fact_ok"].mean(), "n_fact": int(scored["fact_applicable"].sum()),
         "hallucinated_price_rate": scored["hallucinated_price"].mean(),
         "refusal_rate": scored["refused"].mean()}
    if "latency_s" in scored:
        s["latency_p50"], s["latency_p95"] = scored["latency_s"].quantile([.5, .95]).tolist()
    if "judge" in scored:
        s["judge_mean"] = scored["judge"].mean()
    return s


def bootstrap_ci(x, n=1000, seed=0, alpha=.05):
    x = np.asarray(pd.Series(x, dtype=float).dropna(), dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a, b, n=1000, seed=0):
    """P(mean(a) <= mean(b)) qua bootstrap ghép cặp — nhỏ ⇒ a tốt hơn b có ý nghĩa."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_not_better": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return {"diff": float(a.mean() - b.mean()), "ci_low": float(np.quantile(d, .025)), "ci_high": float(np.quantile(d, .975)),
            "p_not_better": float((d <= 0).mean())}


# ---------------------------------------------------------------- LLM-as-judge (Gemini, model thầy khác Qwen/LLaMA)
JUDGE_PROMPT = """Bạn là giám khảo chấm chatbot CSKH nhà hàng Kimochi Sushi. Chấm CÂU TRẢ LỜI so với ĐÁP ÁN CHUẨN theo thang 1–5:
5 = đúng và đủ mọi dữ kiện (giá, giờ, thành phần), giọng lịch sự; 4 = đúng, thiếu chi tiết nhỏ; 3 = đúng một phần;
2 = phần lớn sai hoặc lạc đề; 1 = sai dữ kiện quan trọng (giá, dị ứng) hoặc bịa. Câu trả lời từ chối/chuyển nhân viên khi đáp án
chuẩn cũng không có thông tin thì được 5.
CÂU HỎI: {q}
ĐÁP ÁN CHUẨN: {ref}
CÂU TRẢ LỜI: {pred}
Chỉ trả về JSON: {{"score": <1-5>, "reason": "<ngắn>"}}"""


def judge(q, ref, pred, model="gemini-2.5-flash") -> tuple[float, str]:
    from google import genai  # pip install google-genai ; secret GEMINI_API_KEY
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    txt = client.models.generate_content(model=model, contents=JUDGE_PROMPT.format(q=q, ref=ref, pred=pred)).text
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        j = json.loads(m.group(0))
        return float(j["score"]), j.get("reason", "")
    except Exception:
        return np.nan, txt[:200]
