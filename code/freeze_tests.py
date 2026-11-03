"""T06 — Kiểm tra định dạng và đóng băng các bộ test viết tay bằng SHA-256.

Cách dùng (sau khi nhóm viết xong data/splits/test_gold.jsonl, test_adv.jsonl, scenarios.jsonl):
    python code/freeze_tests.py          # kiểm định dạng + ghi CHECKSUMS.txt
    python code/freeze_tests.py --check  # xác minh file chưa bị sửa sau khi khóa (chạy trước mỗi lần đánh giá)
Mỗi dòng test_gold / test_adv:
    {"id": "gold-001", "query": "...", "reference": "câu trả lời chuẩn", "intent": "price_query",
     "gold_doc_ids": ["doc_menu_M012"], "gold_sql": null, "author": "Trung"}
test_adv thêm "attack": "prompt_injection" | "out_of_scope" | "no_answer_in_kb".
scenarios.jsonl: {"id": "sc-01", "turns": ["...", "..."], "expected": {"reservation": {...}} , "author": "..."}
"""
import hashlib
import json
import sys
from pathlib import Path

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"
FILES = ["test_gold.jsonl", "test_adv.jsonl", "scenarios.jsonl"]
REQUIRED = {"test_gold.jsonl": {"id", "query", "reference", "intent", "author"},
            "test_adv.jsonl": {"id", "query", "reference", "intent", "attack", "author"},
            "scenarios.jsonl": {"id", "turns", "expected", "author"}}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(p: Path) -> int:
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"{p.name}: trùng id"
    for r in rows:
        miss = REQUIRED[p.name] - r.keys()
        assert not miss, f"{p.name} {r.get('id')}: thiếu {miss}"
    return len(rows)


def main():
    ck = SPLITS / "CHECKSUMS.txt"
    present = [SPLITS / f for f in FILES if (SPLITS / f).exists()]
    if "--check" in sys.argv:
        saved = dict(line.split()[::-1] for line in ck.read_text().splitlines())
        bad = [p.name for p in present if saved.get(p.name) != sha(p)]
        sys.exit(f"ĐÃ BỊ SỬA sau khi khóa: {bad}" if bad else print("OK — test chưa bị sửa"))
    for p in present:
        print(p.name, validate(p), "dòng")
    ck.write_text("".join(f"{sha(p)}  {p.name}\n" for p in present))
    print("Đã ghi", ck)


if __name__ == "__main__":
    main()
