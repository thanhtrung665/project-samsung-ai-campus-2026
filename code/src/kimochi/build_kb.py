"""T05 — Dựng kho tri thức Kimochi từ bộ hội thoại gốc.

Đầu vào : data/KIMOCHI_DATA_TRAINING_CHATBOT.json  (Domain, user_query, responses)
Đầu ra  : data/kb/menu.csv            — 1 dòng / món (giá, danh mục, dị ứng, chay, cay)
          data/kb/docs.jsonl          — tài liệu cho RAG (1 món = 1 doc, 1 chủ đề chính sách = 1 doc)
          data/kimochi.db             — SQLite: categories, menu_items, branches, tables, reservations, promotions
          results/kb_quality.json     — thống kê chất lượng (xung đột giá, món thiếu giá…)

Nguyên tắc: chỉ trích thông tin CÓ trong dữ liệu; phần không có (bàn, đặt bàn) là MÔ PHỎNG và được đánh dấu `simulated=1`.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "KIMOCHI_DATA_TRAINING_CHATBOT.json"
KB_DIR = ROOT / "data" / "kb"
DB_PATH = ROOT / "data" / "kimochi.db"
RESULTS = ROOT / "results"

# ---------------------------------------------------------------- tiện ích văn bản
def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).strip()
    return re.sub(r"\s+", " ", s)


def fold(s: str) -> str:
    """Bỏ dấu + chữ thường — dùng để so khớp tên món."""
    s = unicodedata.normalize("NFD", s.lower()).replace("đ", "d")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# Tiền tố domain mô tả "khía cạnh" chứ không phải tên món
ASPECT_PREFIX = re.compile(
    r"^(tổng quan về món|cách thưởng thức|thành phần nguyên liệu của|các loại phần|chất lượng nguyên liệu của|"
    r"hương vị của|thời gian phục vụ món|khẩu phần ăn của|khẩu phần của|nước chấm kèm theo|đối tượng nên dùng món|"
    r"trẻ em có dùng được|phân loại và độ chín của món|điều chỉnh độ chín của|chính sách ship mang đi|"
    r"đối tượng cần lưu ý khi dùng|yêu cầu xem ảnh/hình ảnh của|giá món|ảnh món|món kèm theo|độ cay của|độ béo của|"
    r"độ ngọt của|độ giòn của vỏ|loại tôm dùng trong|điểm đặc biệt của|thời gian chế biến|mức độ bán chạy của|"
    r"hàm lượng dầu mỡ trong|việc bóc vỏ|sự phù hợp của|trẻ em và)\s+",
    re.I,
)


def item_from_domain(domain: str) -> str | None:
    """'Giá & Hương vị: Cơm Lươn' -> 'Cơm Lươn'; 'Tổng quan về món Sashimi Cá Ngừ' -> 'Sashimi Cá Ngừ'."""
    d = norm(domain)
    if ":" in d:
        name = d.split(":", 1)[1].strip()
    else:
        m = ASPECT_PREFIX.match(d)
        if not m:
            return None
        name = d[m.end():]
    name = re.sub(r"\s*\(([^)]*)\)\s*$", lambda m: f" ({m.group(1)})" if re.search(r"[A-Za-z]", m.group(1)) and not re.search(r"ăn|cay|trẻ", m.group(1)) else "", name)
    name = re.sub(r"\s+(không|với trẻ em|để .*|khi .*)$", "", name, flags=re.I).strip(" .")
    if name.lower() in {"kimochi sushi", "ẩm thực nhật bản"} or len(name) < 3:
        return None
    return name


PRICE_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})+)\s*(?:vnđ|vnd|đồng|đ)\b|(\d{2,4})\s*k\b", re.I)


def prices_in(text: str) -> list[int]:
    out = []
    for a, b in PRICE_RE.findall(text):
        v = int(re.sub(r"[.,]", "", a)) if a else int(b) * 1000
        if 5_000 <= v <= 3_000_000:
            out.append(v)
    return out


CATEGORY_RULES = [
    ("Combo", r"^combo|sushi box"),
    ("Sashimi", r"^sashimi"),
    ("Maki / Cơm cuộn", r"^cơm cuộn|maki|^cuộn"),
    ("Sushi / Nigiri", r"^sushi|nigiri"),
    ("Inari / Tàu hũ", r"inari|^tàu hũ|^đậu hũ"),
    ("Salad", r"^salad|^rong|^gỏi"),
    ("Súp", r"^súp|^canh"),
    ("Cơm / Mì", r"^cơm|^mì|^udon|^ramen|^soba"),
    ("Đồ uống", r"^trà|^nước|soda|^bia|sake|cà phê|sinh tố|coca|pepsi|sprite|^matcha|^sữa"),
    ("Tráng miệng", r"^kem|^chè|^bánh|mochi|pudding"),
    ("Món nóng / Khai vị", r"."),
]

ALLERGEN_RULES = {  # nhóm dị ứng theo 14 nhóm phổ biến (rút gọn)
    "giáp xác": r"tôm|cua|thanh cua|kani|ebi|ghẹ",
    "nhuyễn thể": r"\bsò\b|hàu|kaki|mực|bạch tuộc|bạch tuột|ika|hotate|nghêu|ốc",
    "cá": r"\bcá\b|salmon|tuna|lươn|unagi|saba|trứng cá|tobiko|ikura|mentai",
    "trứng": r"\btrứng\b(?! cá)|tamago|mayo|mayonnaise",
    "sữa": r"phô mai|cheese|sốt kem|\bkem\b|cream|bơ sữa|sữa",
    "đậu nành": r"đậu nành|miso|nước tương|shoyu|tàu hũ|đậu hũ|inari|edamame",
    "mè": r"\bmè\b|vừng|sesame",
    "gluten": r"tempura|chiên xù|bột chiên|panko|udon|ramen|\bmì\b|\bbánh\b",
}
VEG_NEG = re.compile(r"cá|tôm|cua|mực|sò|hàu|lươn|thịt|bò|gà|heo|bacon|trứng|bạch tu|salmon|tuna|kani|ebi|unagi|mentai", re.I)


def categorize(name: str) -> str:
    n = name.lower()
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, n):
            return cat
    return "Khác"


# ---------------------------------------------------------------- trích xuất
DISH_WORD = re.compile(
    r"sashimi|sushi|combo|cơm|cuộn|maki|salad|súp|tàu hũ|đậu hũ|inari|tôm|\bcá\b|mực|bạch tu|hàu|\bsò\b|lươn|\bgà\b|"
    r"\bbò\b|\bcua\b|trứng cút|khoai|rong|tempura|\btrà\b|soda|nước suối|\bkem\b|\bchè\b|mochi|sụn|edamame|"
    r"taco|coca|sprite|pepsi|\bbia\b|sake|matcha|cà phê|sữa|gan cá|bạch tuột|udon|ramen|mì",
    re.I,
)
NOT_DISH = re.compile(r"^(cá nhân|ăn cùng|món|phù hợp|nguyên liệu|sốt .* là gì|thuần chay|salad &|súp/|rolls|nhóm|sinh nhật)", re.I)


def clean_name(n: str | None) -> str | None:
    if not isinstance(n, str):
        return None
    n = re.sub(r"^(trẻ em|người lớn|cặp đôi|gia đình)\s*[-–]\s*", "", n, flags=re.I)
    n = re.split(r"\s+[-–]\s+", n)[0]
    n = re.sub(r"\s+(cho trẻ em|ăn kèm|ăn chơi|ngon nhất|là .*|có cần .*)$", "", n, flags=re.I)
    return n.strip() or None


def load_raw() -> pd.DataFrame:
    """Đọc dữ liệu gốc và gán mỗi dòng vào một món chính tắc (hoặc None = chủ đề chung)."""
    df = pd.DataFrame(json.loads(RAW.read_text(encoding="utf-8")))
    for c in df.columns:
        df[c] = df[c].map(norm)
    df["cand"] = df["Domain"].map(item_from_domain).map(clean_name)
    n_dom = df.groupby("cand")["Domain"].nunique()
    canon = sorted({c for c, k in n_dom.items() if k >= 2 and DISH_WORD.search(c) and not NOT_DISH.search(c)},
                   key=lambda c: -len(c))
    canon_f = [(fold(c), c) for c in canon]

    def assign(domain: str) -> str | None:
        f = fold(domain)
        for cf, c in canon_f:  # khớp tên dài nhất trước
            if cf in f:
                return c
        return None

    df["item"] = df["Domain"].map(assign)
    return df.drop(columns="cand")


def build_menu(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, conflicts = [], {}
    for item, g in df[df["item"].notna()].groupby("item"):
        resp = list(dict.fromkeys(g["responses"]))  # duy nhất, giữ thứ tự
        text = " ".join(resp)
        # giá: chỉ nhận giá xuất hiện trong câu có nhắc tên món hoặc domain có chữ "giá"
        price_votes: Counter[int] = Counter()
        for _, r in g.drop_duplicates("responses").iterrows():
            ps = prices_in(r["responses"])
            if ps and (fold(item)[:12] in fold(r["responses"]) or "giá" in r["Domain"].lower()):
                price_votes.update(ps[:1])  # giá đầu tiên của câu = giá món
        price = price_votes.most_common(1)[0][0] if price_votes else None
        if len(price_votes) > 1:
            conflicts[item] = dict(price_votes)
        ingr = g[g["Domain"].str.contains(r"thành phần|nguyên liệu|tổng quan|thông tin món|dị ứng|lưu ý", case=False, regex=True)]
        blob = (item + " " + " ".join(dict.fromkeys(ingr["responses"]))).lower()
        allergens = sorted(a for a, p in ALLERGEN_RULES.items() if re.search(p, blob))
        spicy = 2 if re.search(r"cay nồng|rất cay|cay nhiều", blob) else 1 if re.search(r"\bcay\b", item.lower()) or "cay nhẹ" in blob else 0
        rows.append({
            "name": item,
            "category": categorize(item),
            "price_vnd": price,
            "allergens": ";".join(allergens),
            "is_vegetarian": int(not VEG_NEG.search(item) and "chay" in blob),
            "spicy_level": spicy,
            "allergen_verified": 0,  # suy luận bằng từ khóa — PHẢI được quản lý nhà hàng xác nhận trước khi dùng thật
            "n_samples": len(g),
            "description": max(resp, key=len)[:600],
        })
    menu = pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)
    menu.insert(0, "item_id", [f"M{i+1:03d}" for i in range(len(menu))])
    return menu, conflicts


BRANCHES = [  # thông tin xuất hiện trong dữ liệu gốc (hotline, địa chỉ GV, giờ mở cửa)
    {"branch_id": "BT", "name": "Kimochi Sushi Bình Thạnh", "district": "Bình Thạnh", "address": None,
     "hotline": "0879096969", "open_time": "11:00", "close_time": "22:00", "seating": "bàn cao, bàn bệt"},
    {"branch_id": "GV", "name": "Kimochi Sushi Gò Vấp", "district": "Gò Vấp", "address": "Hẻm 590 Phan Văn Trị, Gò Vấp, TP.HCM",
     "hotline": "0888912344", "open_time": "11:00", "close_time": "22:00", "seating": "lầu 1: bàn cao, bàn bệt"},
]


def build_docs(df: pd.DataFrame, menu: pd.DataFrame) -> list[dict]:
    docs = []
    for r in menu.itertuples():
        body = " ".join(dict.fromkeys(df.loc[df["item"] == r.name, "responses"]))
        docs.append({"doc_id": f"doc_menu_{r.item_id}", "type": "menu", "title": r.name, "item_id": r.item_id,
                     "category": r.category, "allergens": r.allergens.split(";") if r.allergens else [],
                     "is_vegetarian": bool(r.is_vegetarian), "spicy_level": int(r.spicy_level), "text": body})
    pol = df[df["item"].isna()]
    for i, (topic, g) in enumerate(pol.groupby("Domain")):
        docs.append({"doc_id": f"doc_policy_{i+1:03d}", "type": "policy", "title": topic,
                     "text": " ".join(dict.fromkeys(g["responses"]))})
    return docs


SCHEMA = """
CREATE TABLE categories   (category_id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
CREATE TABLE menu_items   (item_id TEXT PRIMARY KEY, name TEXT NOT NULL, category_id INTEGER REFERENCES categories,
                           price_vnd INTEGER, allergens TEXT, allergen_verified INTEGER DEFAULT 0, is_vegetarian INTEGER, spicy_level INTEGER,
                           is_available INTEGER DEFAULT 1, description TEXT);
CREATE TABLE branches     (branch_id TEXT PRIMARY KEY, name TEXT, district TEXT, address TEXT, hotline TEXT,
                           open_time TEXT, close_time TEXT, seating TEXT);
CREATE TABLE tables       (table_id TEXT PRIMARY KEY, branch_id TEXT REFERENCES branches, seats INTEGER,
                           kind TEXT, simulated INTEGER DEFAULT 1);
CREATE TABLE reservations (reservation_id TEXT PRIMARY KEY, branch_id TEXT REFERENCES branches,
                           table_id TEXT REFERENCES tables, customer_name TEXT, phone TEXT, party_size INTEGER,
                           start_at TEXT, status TEXT CHECK (status IN ('CONFIRMED','CANCELLED','NO_SHOW','DONE')),
                           note TEXT, simulated INTEGER DEFAULT 1);
CREATE TABLE promotions   (promo_id TEXT PRIMARY KEY, title TEXT, condition TEXT, benefit TEXT,
                           valid_from TEXT, valid_to TEXT, simulated INTEGER DEFAULT 1);
CREATE INDEX ix_res_branch_time ON reservations(branch_id, start_at);
"""


def build_db(menu: pd.DataFrame) -> None:
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    cats = {c: i + 1 for i, c in enumerate(sorted(menu["category"].unique()))}
    con.executemany("INSERT INTO categories VALUES (?,?)", [(i, c) for c, i in cats.items()])
    con.executemany(
        "INSERT INTO menu_items VALUES (?,?,?,?,?,0,?,?,1,?)",
        [(r.item_id, r.name, cats[r.category], None if pd.isna(r.price_vnd) else int(r.price_vnd), r.allergens,
          int(r.is_vegetarian), int(r.spicy_level), r.description) for r in menu.itertuples()],
    )
    con.executemany("INSERT INTO branches VALUES (:branch_id,:name,:district,:address,:hotline,:open_time,:close_time,:seating)", BRANCHES)
    # MÔ PHỎNG: 12 bàn / chi nhánh (2,4,6,8 chỗ), không có số liệu thật
    tables = [(f"{b['branch_id']}-T{j+1:02d}", b["branch_id"], s, k)
              for b in BRANCHES for j, (s, k) in enumerate([(2, "bàn cao")] * 4 + [(4, "bàn cao")] * 4 + [(6, "bàn bệt")] * 3 + [(8, "bàn bệt")])]
    con.executemany("INSERT INTO tables (table_id, branch_id, seats, kind) VALUES (?,?,?,?)", tables)
    # Khuyến mãi xuất hiện trong dữ liệu: đặt bàn trước tặng chè đậu đỏ; giữ bàn 15 phút là chính sách (ở docs)
    con.execute("INSERT INTO promotions (promo_id,title,condition,benefit,valid_from,valid_to) VALUES "
                "('P01','Ưu đãi đặt bàn trước','Đặt bàn trước qua hotline/fanpage','Tặng chè đậu đỏ tráng miệng','2026-11-01','2026-11-30')")
    con.commit()
    con.close()


def cross_check_prices(df: pd.DataFrame, menu: pd.DataFrame) -> dict:
    """Đối chiếu giá trong DB với giá nhắc kèm tên món ở MỌI câu trả lời khác (vd câu gợi ý combo: "Tôm Rang Muối (174.900 VNĐ)")."""
    resp = list(dict.fromkeys(df["responses"]))
    out = {}
    for r in menu.dropna(subset=["price_vnd"]).itertuples():
        pat = re.compile(re.escape(r.name) + r"\s*\(?\s*(\d{1,3}(?:[.,]\d{3})+)\s*(?:vnđ|vnd|đ)", re.I)
        seen = Counter(int(re.sub(r"[.,]", "", m)) for t in resp for m in pat.findall(t))
        other = {p: c for p, c in seen.items() if p != int(r.price_vnd)}
        if other:
            out[r.name] = {"db": int(r.price_vnd), "other_mentions": other}
    return out


def main() -> dict:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    df = load_raw()
    menu, conflicts = build_menu(df)
    menu.to_csv(KB_DIR / "menu.csv", index=False, encoding="utf-8-sig")
    docs = build_docs(df, menu)
    with open(KB_DIR / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    pd.DataFrame(BRANCHES).to_csv(KB_DIR / "branches.csv", index=False, encoding="utf-8-sig")
    build_db(menu)
    q = {
        "raw_rows": len(df),
        "rows_mapped_to_item": int(df["item"].notna().sum()),
        "rows_policy_general": int(df["item"].isna().sum()),
        "menu_items": len(menu),
        "items_with_price": int(menu["price_vnd"].notna().sum()),
        "items_price_conflict": len(conflicts),
        "price_conflicts": conflicts,
        "items_price_mismatch_in_other_answers": len(xc := cross_check_prices(df, menu)),
        "price_mismatch_detail": xc,
        "docs_menu": sum(d["type"] == "menu" for d in docs),
        "docs_policy": sum(d["type"] == "policy" for d in docs),
        "category_counts": menu["category"].value_counts().to_dict(),
    }
    (RESULTS / "kb_quality.json").write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
    return q


if __name__ == "__main__":
    q = main()
    print(json.dumps({k: v for k, v in q.items() if k != "price_conflicts"}, ensure_ascii=False, indent=2))
