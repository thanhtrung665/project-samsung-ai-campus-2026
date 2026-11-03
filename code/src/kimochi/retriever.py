"""T09 — Hybrid RAG: BM25 + dense (bge-m3) + RRF (+ reranker, + lọc theo tên món), và đánh giá truy xuất.

Bộ đánh giá: câu hỏi val/test_synth có gắn món (meta.item) → tài liệu đúng = doc_menu_<item_id>.
Mọi văn bản được bỏ dấu trước khi tách từ cho BM25 (44% câu hỏi không dấu — xem Phần 3).

Dense / reranker cần tải mô hình từ Hugging Face → chạy trên Colab; nếu không tải được, cấu hình đó được đánh dấu
"skipped" trong kết quả (không điền số giả).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from kimochi.build_kb import fold

ROOT = Path(__file__).resolve().parents[3]
KB = ROOT / "data" / "kb"
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
RRF_K = 60


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", fold(text))
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]  # thêm bigram: "ca_hoi", "com_cuon"


def load_docs() -> list[dict]:
    return [json.loads(line) for line in open(KB / "docs.jsonl", encoding="utf-8")]


def doc_text(d: dict) -> str:
    return f"{d['title']}. {d['title']}. {d['text']}"  # nhân đôi tiêu đề = tăng trọng số tên món


@dataclass
class HybridRetriever:
    docs: list[dict]
    dense_model: object | None = None
    reranker: object | None = None

    def __post_init__(self):
        texts = [doc_text(d) for d in self.docs]
        self.ids = [d["doc_id"] for d in self.docs]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit([fold(t) for t in texts])
        self.tfidf_m = self.tfidf.transform([fold(t) for t in texts])
        self.dense_m = None
        if self.dense_model is not None:
            self.dense_m = self.dense_model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        self.title_f = [(fold(d["title"]), i) for i, d in enumerate(self.docs) if d["type"] == "menu"]
        self.title_f.sort(key=lambda x: -len(x[0]))

    # ---------------------------------------------------------------- từng bộ truy xuất → thứ hạng chỉ số tài liệu
    def rank_bm25(self, q):
        return np.argsort(-self.bm25.get_scores(tokenize(q)))

    def rank_tfidf(self, q):
        return np.argsort(-linear_kernel(self.tfidf.transform([fold(q)]), self.tfidf_m)[0])

    def rank_dense(self, q):
        v = self.dense_model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.argsort(-(self.dense_m @ v))

    @staticmethod
    def rrf(*rankings, k=RRF_K, depth=100):
        score: dict[int, float] = {}
        for r in rankings:
            for pos, idx in enumerate(r[:depth]):
                score[idx] = score.get(idx, 0.0) + 1.0 / (k + pos + 1)
        return np.array(sorted(score, key=lambda i: -score[i]))

    def title_filter(self, q, ranking):
        """Lọc metadata: nếu câu hỏi nhắc đúng tên một món (so khớp bỏ dấu, tên dài nhất) → đẩy tài liệu món đó lên đầu."""
        fq = fold(q)
        for t, i in self.title_f:
            if len(t) >= 6 and t in fq:
                return np.concatenate([[i], ranking[ranking != i]])
        return ranking

    def rerank(self, q, ranking, top=20):
        head = ranking[:top]
        scores = self.reranker.predict([(q, doc_text(self.docs[i])[:1500]) for i in head])
        return np.concatenate([head[np.argsort(-np.asarray(scores))], ranking[top:]])

    def search(self, q, config="hybrid_filter", k=5) -> list[str]:
        return [self.ids[i] for i in self._rank(q, config)[:k]]

    def _rank(self, q, config):
        if config == "bm25":
            return self.rank_bm25(q)
        if config == "bm25_filter":
            return self.title_filter(q, self.rank_bm25(q))
        if config == "tfidf_char":
            return self.rank_tfidf(q)
        if config == "dense":
            return self.rank_dense(q)
        sparse = [self.rank_bm25(q), self.rank_tfidf(q)]
        dense = [self.rank_dense(q)] if self.dense_m is not None and "dense" in config else []
        r = self.rrf(*sparse, *dense)
        if "rerank" in config:
            r = self.rerank(q, r)
        if "filter" in config:
            r = self.title_filter(q, r)
        return r


# ---------------------------------------------------------------- đánh giá
def eval_set(split="test_synth") -> pd.DataFrame:
    menu = pd.read_csv(KB / "menu.csv")
    id_of = dict(zip(menu["name"], menu["item_id"]))
    df = pd.read_json(SPLITS / f"{split}.jsonl", lines=True)
    df["query"] = df["messages"].map(lambda m: m[1]["content"])
    df["item"] = df["meta"].map(lambda m: m["item"])
    df["noisy"] = df["meta"].map(lambda m: m["noisy_input"])
    df = df[df["item"].isin(id_of)].copy()
    df["gold"] = df["item"].map(lambda n: f"doc_menu_{id_of[n]}")
    return df[["id", "query", "gold", "noisy"]]


def metrics(ranked: list[list[str]], gold: list[str]) -> dict:
    pos = np.array([r.index(g) + 1 if g in r else 0 for r, g in zip(ranked, gold)])
    hit = pos > 0
    safe = np.maximum(pos, 1)
    return {"Recall@1": float((pos == 1).mean()), "Recall@5": float((hit & (pos <= 5)).mean()),
            "MRR@10": float(np.where(hit & (pos <= 10), 1 / safe, 0).mean()),
            "nDCG@5": float(np.where(hit & (pos <= 5), 1 / np.log2(safe + 1), 0).mean())}


CONFIGS = ["bm25", "bm25_filter", "tfidf_char", "dense", "hybrid_sparse", "hybrid_dense", "hybrid_dense_rerank", "hybrid_filter",
           "hybrid_dense_rerank_filter"]


def load_models():
    dense = rer = None
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
        dense = SentenceTransformer("BAAI/bge-m3")
        rer = CrossEncoder("BAAI/bge-reranker-v2-m3")
    except Exception as e:  # không có mạng HF / GPU
        print("Dense/reranker không khả dụng:", type(e).__name__)
    return dense, rer


def run(split="test_synth", use_models=True) -> pd.DataFrame:
    dense, rer = load_models() if use_models else (None, None)
    r = HybridRetriever(load_docs(), dense, rer)
    ev = eval_set(split)
    rows = []
    for cfg in CONFIGS:
        need_dense, need_rer = "dense" in cfg, "rerank" in cfg
        if (need_dense and dense is None) or (need_rer and rer is None):
            rows.append({"config": cfg, "status": "skipped"})
            continue
        ranked = [r.search(q, cfg, k=10) for q in ev["query"]]
        m_all = metrics(ranked, list(ev["gold"]))
        m_noisy = metrics([x for x, n in zip(ranked, ev["noisy"]) if n], list(ev.loc[ev["noisy"], "gold"]))
        rows.append({"config": cfg, "status": "ok", "n": len(ev), **m_all, "Recall@5_noisy": m_noisy["Recall@5"]})
    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / f"retrieval_{split}.csv", index=False)
    return out


if __name__ == "__main__":
    print(run().round(3).to_string())
