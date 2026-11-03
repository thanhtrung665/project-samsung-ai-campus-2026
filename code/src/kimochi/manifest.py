"""Manifest dữ liệu: NB1 ghi md5 các file sinh ra, NB2 kiểm tra trước khi huấn luyện (bảo đảm đúng phiên bản dữ liệu)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ["data/kimochi.db", "data/kb/menu.csv", "data/kb/chunks.jsonl", "data/splits/train.jsonl", "data/splits/val.jsonl",
             "data/splits/test_synth.jsonl", "data/splits_v2/train.jsonl", "data/splits_v2/val.jsonl"]
OPTIONAL = ["data/splits/test_gold.jsonl", "data/splits/test_adv.jsonl", "data/splits/scenarios.jsonl"]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _rows(p: Path) -> int | None:
    return sum(1 for _ in p.open(encoding="utf-8")) if p.suffix == ".jsonl" else None


def write(root: Path = ROOT) -> dict:
    out = {}
    for a in ARTIFACTS + OPTIONAL:
        p = root / a
        if p.exists():
            out[a] = {"md5": md5(p), "rows": _rows(p)}
    (root / "data" / "MANIFEST.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def verify(root: Path = ROOT) -> list[str]:
    """Trả về danh sách file lệch / thiếu so với MANIFEST (rỗng = khớp)."""
    m = json.loads((root / "data" / "MANIFEST.json").read_text())
    bad = []
    for a, info in m.items():
        p = root / a
        if not p.exists():
            bad.append(f"thiếu {a}")
        elif md5(p) != info["md5"]:
            bad.append(f"khác md5: {a}")
    return bad
