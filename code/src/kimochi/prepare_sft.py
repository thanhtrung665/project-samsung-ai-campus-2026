"""T07 — Chuyển bộ hội thoại gốc sang định dạng SFT (messages + meta) và chia split chống rò rỉ.

Đầu ra: data/splits/{train,val,test_synth}.jsonl, data/splits/all_meta.csv, results/split_stats.json

- intent: gán bằng luật từ khóa trên Domain + câu hỏi (12 intent của SPEC §3.2; dữ liệu gốc không có
  reservation_manage / order_status / out_of_scope → sẽ được bổ sung bằng mẫu viết tay / sinh thêm).
- group_id: băm câu trả lời đã chuẩn hóa → mọi câu hỏi diễn đạt lại của cùng một đáp án nằm trọn trong 1 tập.
- split: StratifiedGroupKFold (10 fold) → 1 fold test_synth, 1 fold val, 8 fold train (~80/10/10), stratify theo intent.
- kiểm rò rỉ: cosine TF-IDF ký tự (proxy CPU cho embedding) giữa câu hỏi test và câu train gần nhất; loại cặp > 0,92.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from kimochi.build_kb import fold, load_raw

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "splits"
RESULTS = ROOT / "results"
SEED = 42
LEAK_THRESHOLD = 0.92

SYSTEM_PROMPT = (
    "Bạn là trợ lý chăm sóc khách hàng của nhà hàng Kimochi Sushi (2 chi nhánh Bình Thạnh và Gò Vấp, TP.HCM). "
    "Xưng 'em', gọi khách là 'khách iu' hoặc 'anh/chị', trả lời ngắn gọn, thân thiện. Chỉ dùng thông tin trong ngữ cảnh "
    "hoặc kết quả công cụ; nếu không có thông tin thì nói rõ và đề nghị chuyển nhân viên. Với câu hỏi dị ứng, luôn khuyên "
    "khách xác nhận lại với nhân viên."
)

INTENT_RULES = [  # thứ tự = độ ưu tiên
    ("complaint", r"phàn nàn|khiếu nại|góp ý|feedback|thái độ|chậm quá|không hài lòng|đồ bỏ quên|đổi trả"),
    ("reservation_create", r"đặt bàn|giữ bàn|bàn trống|còn bàn|đặt chỗ|phòng riêng|đặt trước"),
    ("store_info", r"chi nhánh|địa chỉ|giờ mở|mở cửa|đóng cửa|hotline|số điện thoại|gửi xe|đỗ xe|đường đi|google map|"
                   r"năm thành lập|kiến trúc|không gian|wifi|tuyển dụng|phỏng vấn"),
    ("policy_promo", r"ưu đãi|khuyến mãi|giảm giá|chính sách|ship|giao hàng|giao tận|mang đi|thanh toán|chuyển khoản|"
                     r"shopeefood|grab|voucher|thẻ thành viên|tích điểm"),
    ("allergy_diet", r"dị ứng|trẻ em|trẻ nhỏ|bé|bầu|ăn chay|món chay|thuần chay|không ăn|lưu ý khi dùng|người già|"
                     r"ăn kiêng|tanh|độ chín|sống"),
    ("price_query", r"\bgiá\b|bao nhiêu tiền|nhiêu tiền|bao nhiêu|mấy k\b|\d+\s*k\b|\bgia\b|bao nhieu"),
    ("recommendation", r"gợi ý|nên ăn|nên gọi|phù hợp|sinh nhật|nhóm|gia đình|cặp đôi|hẹn hò|dịp|tiệc|bán chạy|"
                       r"ngon nhất|best seller|combo cho|mấy người|liên hoan|kỷ niệm"),
    ("chitchat", r"^(xin chào|chào|hello|hi|cảm ơn|thanks|ok)\b"),
]
TEEN = re.compile(r"\b(k|ko|hok|dc|đc|bn|bnh|ns|mn|j|z|v|ak|ạk|nha|nhaa|shop|ad|iu|cx|vs)\b", re.I)


def intent_of(domain: str, query: str) -> str:
    d, q = domain.lower(), query.lower()
    for name, pat in INTENT_RULES:
        # domain kiểu "Giá & Hương vị: X" chứa cả câu hỏi không về giá → intent giá chỉ xét trên câu hỏi
        if re.search(pat, q) or (name != "price_query" and re.search(pat, d)):
            return name
    return "menu_info"


def is_unaccented(q: str) -> bool:
    letters = [c for c in q if c.isalpha()]
    return len(letters) >= 5 and fold(q) == q.lower()


def difficulty(q: str, intent: str) -> str:
    n = len(q.split())
    if intent in {"recommendation", "allergy_diet"} or n > 20:
        return "hard"
    return "easy" if n <= 7 else "medium"


def build_frame() -> pd.DataFrame:
    df = load_raw()
    df["intent"] = [intent_of(d, q) for d, q in zip(df["Domain"], df["user_query"])]
    df["group_id"] = df["responses"].map(lambda r: "g-" + hashlib.sha1(fold(r).encode()).hexdigest()[:10])
    df["noisy_input"] = df["user_query"].map(lambda q: bool(is_unaccented(q) or TEEN.search(q)))
    df["unaccented"] = df["user_query"].map(is_unaccented)
    df["needs_tool"] = df["intent"].map(lambda i: {"price_query": "sql", "store_info": "sql", "reservation_create": "sql",
                                                   "complaint": "handoff"}.get(i, "rag"))
    df["difficulty"] = [difficulty(q, i) for q, i in zip(df["user_query"], df["intent"])]
    df["q_len_words"] = df["user_query"].str.split().str.len()
    df["a_len_words"] = df["responses"].str.split().str.len()
    df["id"] = [f"kmc-{i:05d}" for i in range(len(df))]
    return df


def split(df: pd.DataFrame) -> pd.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_of = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["intent"], df["group_id"])):
        fold_of[idx] = k
    df["split"] = np.select([fold_of == 0, fold_of == 1], ["test_synth", "val"], "train")
    return df


def leakage_check(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit(df["user_query"].map(fold))
    tr = df[df["split"] == "train"]
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(vec.transform(tr["user_query"].map(fold)))
    stats = {}
    df["nn_cos_to_train"] = np.nan
    for sp in ("val", "test_synth"):
        m = df["split"] == sp
        dist, _ = nn.kneighbors(vec.transform(df.loc[m, "user_query"].map(fold)))
        df.loc[m, "nn_cos_to_train"] = 1 - dist[:, 0]
        stats[sp] = {"n": int(m.sum()), "cos_mean": float((1 - dist).mean()), "cos_p95": float(np.quantile(1 - dist, 0.95)),
                     "n_over_threshold": int(((1 - dist[:, 0]) > LEAK_THRESHOLD).sum())}
    drop = df["split"].isin(["val", "test_synth"]) & (df["nn_cos_to_train"] > LEAK_THRESHOLD)
    df.loc[drop, "split"] = "dropped_leak"
    stats["dropped_leak"] = int(drop.sum())
    return df, stats


def to_record(r) -> dict:
    return {
        "id": r.id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": r.user_query},
                     {"role": "assistant", "content": r.responses}],
        "meta": {"intent": r.intent, "format": "closed_book", "difficulty": r.difficulty, "noisy_input": bool(r.noisy_input),
                 "needs_tool": r.needs_tool, "group_id": r.group_id, "item": r.item if isinstance(r.item, str) else None,
                 "domain": r.Domain, "source": "kimochi_v1"},
    }


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    df = split(build_frame())
    df, leak = leakage_check(df)
    for sp in ("train", "val", "test_synth"):
        with open(OUT / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in df[df["split"] == sp].itertuples():
                f.write(json.dumps(to_record(r), ensure_ascii=False) + "\n")
    df.drop(columns=["responses"]).to_csv(OUT / "all_meta.csv", index=False, encoding="utf-8-sig")
    stats = {
        "split_sizes": df["split"].value_counts().to_dict(),
        "groups": int(df["group_id"].nunique()),
        "group_overlap_train_test": len(set(df.loc[df.split == "train", "group_id"]) & set(df.loc[df.split == "test_synth", "group_id"])),
        "intent_by_split": pd.crosstab(df["intent"], df["split"]).to_dict(),
        "noisy_input_rate": float(df["noisy_input"].mean()),
        "unaccented_rate": float(df["unaccented"].mean()),
        "leakage": leak,
    }
    (RESULTS / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: v for k, v in s.items() if k != "intent_by_split"}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(s["intent_by_split"]).fillna(0).astype(int))
