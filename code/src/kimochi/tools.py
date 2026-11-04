"""Công cụ của Agent Kimochi trên SQLite + định dạng gọi công cụ dùng chung cho dữ liệu huấn luyện và lúc chạy.

Định dạng (độc lập model, Qwen và LLaMA đều sinh được dưới dạng văn bản):
    assistant: <tool_call>{"name": "get_menu_item", "arguments": {"item": "Cơm Lươn"}}</tool_call>
    tool     : {"item": "Cơm Lươn", "price_vnd": 209000, ...}
    assistant: câu trả lời cuối cho khách
Mọi truy vấn đều tham số hóa (không ghép chuỗi SQL). Ghi chỉ có create_reservation / cancel_reservation.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from kimochi.build_kb import fold

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data" / "kimochi.db"
SLOT_MINUTES = 120  # một lượt khách giữ bàn 2 giờ (giả định mô phỏng)


def new_reservation_id() -> str:
    """Có thể thay bằng bộ sinh có seed khi tạo dữ liệu huấn luyện (tái lập)."""
    return "KM" + uuid.uuid4().hex[:6].upper()

TOOL_SPECS = [
    {"name": "get_menu_item", "description": "Tra giá, danh mục, nhóm dị ứng (suy luận, chưa xác nhận) của một món theo tên.",
     "parameters": {"item": "tên món"}},
    {"name": "search_menu", "description": "Tìm món theo danh mục / giá tối đa / loại trừ nhóm dị ứng.",
     "parameters": {"category": "tùy chọn", "max_price": "tùy chọn, VNĐ", "exclude_allergen": "tùy chọn"}},
    {"name": "get_branch_info", "description": "Địa chỉ, hotline, giờ mở cửa của chi nhánh.", "parameters": {"branch": "BT | GV | all"}},
    {"name": "check_availability", "description": "Kiểm tra bàn trống.",
     "parameters": {"branch": "BT | GV", "party_size": "số khách", "start_at": "YYYY-MM-DD HH:MM"}},
    {"name": "create_reservation", "description": "Đặt bàn (chỉ gọi khi khách đã xác nhận đủ thông tin).",
     "parameters": {"branch": "BT | GV", "party_size": "số", "start_at": "YYYY-MM-DD HH:MM", "customer_name": "tên", "phone": "SĐT"}},
    {"name": "cancel_reservation", "description": "Hủy đặt bàn theo mã.", "parameters": {"reservation_id": "mã"}},
    {"name": "handoff_to_staff", "description": "Chuyển cuộc trò chuyện cho nhân viên (khiếu nại, ngoài khả năng).",
     "parameters": {"reason": "lý do"}},
]
TOOLS_PROMPT = ("\n\nBạn có thể gọi công cụ bằng cách trả lời DUY NHẤT một dòng "
                "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>. Giá, giờ, bàn trống PHẢI lấy từ công cụ.\n"
                "Công cụ: " + json.dumps(TOOL_SPECS, ensure_ascii=False))
CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def format_call(name: str, arguments: dict) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "</tool_call>"


def parse_call(text: str) -> dict | None:
    m = CALL_RE.search(text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
        return call if isinstance(call, dict) and "name" in call else None
    except json.JSONDecodeError:
        return None


def _con(db=DB, readonly=True):
    uri = f"file:{db}?mode=ro" if readonly else f"file:{db}"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con


def get_menu_item(item: str, db=DB) -> dict:
    with _con(db) as con:
        rows = con.execute("SELECT m.item_id, m.name, c.name AS category, m.price_vnd, m.allergens, m.allergen_verified, "
                           "m.is_available FROM menu_items m JOIN categories c USING(category_id)").fetchall()
    f = fold(item)
    best = max(rows, key=lambda r: (fold(r["name"]) == f, fold(r["name"]) in f or f in fold(r["name"]), -abs(len(r["name"]) - len(item))))
    if not (fold(best["name"]) in f or f in fold(best["name"])):
        return {"found": False, "query": item}
    return {"found": True, "item": best["name"], "category": best["category"], "price_vnd": best["price_vnd"],
            "allergens": [a for a in (best["allergens"] or "").split(";") if a], "allergen_verified": bool(best["allergen_verified"]),
            "available": bool(best["is_available"])}


def search_menu(category: str | None = None, max_price: int | None = None, exclude_allergen: str | None = None, db=DB) -> dict:
    sql = ("SELECT m.name, c.name AS category, m.price_vnd, m.allergens FROM menu_items m JOIN categories c USING(category_id) "
           "WHERE m.is_available = 1")
    args: list = []
    if category:
        sql += " AND c.name LIKE ?"
        args.append(f"%{category}%")
    if max_price:
        sql += " AND m.price_vnd IS NOT NULL AND m.price_vnd <= ?"
        args.append(int(max_price))
    if exclude_allergen:
        sql += " AND (m.allergens IS NULL OR m.allergens NOT LIKE ?)"
        args.append(f"%{exclude_allergen}%")
    with _con(db) as con:
        rows = [dict(r) for r in con.execute(sql + " ORDER BY m.price_vnd LIMIT 8", args)]
    return {"count": len(rows), "items": rows}


def get_branch_info(branch: str = "all", db=DB) -> dict:
    with _con(db) as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM branches WHERE ? = 'all' OR branch_id = ?", (branch, branch))]
    return {"branches": rows}


def _overlap_free_tables(con, branch, party_size, start: datetime):
    end = start + timedelta(minutes=SLOT_MINUTES)
    busy = {r[0] for r in con.execute(
        "SELECT table_id FROM reservations WHERE branch_id = ? AND status = 'CONFIRMED' AND start_at < ? AND "
        "datetime(start_at, ?) > ?", (branch, end.strftime("%Y-%m-%d %H:%M"), f"+{SLOT_MINUTES} minutes",
                                      start.strftime("%Y-%m-%d %H:%M")))}
    return [dict(r) for r in con.execute("SELECT table_id, seats, kind FROM tables WHERE branch_id = ? AND seats >= ? ORDER BY seats",
                                         (branch, int(party_size))) if r["table_id"] not in busy]


def _open(con, branch, start: datetime) -> bool:
    r = con.execute("SELECT open_time, close_time FROM branches WHERE branch_id = ?", (branch,)).fetchone()
    return r is not None and r["open_time"] <= start.strftime("%H:%M") <= r["close_time"]


def check_availability(branch: str, party_size: int, start_at: str, db=DB) -> dict:
    start = datetime.strptime(start_at, "%Y-%m-%d %H:%M")
    with _con(db) as con:
        if not _open(con, branch, start):
            return {"available": False, "reason": "ngoài giờ mở cửa (11:00–22:00)"}
        free = _overlap_free_tables(con, branch, party_size, start)
    return {"available": bool(free), "free_tables": len(free), "branch": branch, "start_at": start_at, "party_size": int(party_size)}


def create_reservation(branch: str, party_size: int, start_at: str, customer_name: str, phone: str, note: str = "", db=DB) -> dict:
    if not re.fullmatch(r"0\d{9}", str(phone)):
        return {"ok": False, "reason": "số điện thoại không hợp lệ"}
    start = datetime.strptime(start_at, "%Y-%m-%d %H:%M")
    with _con(db, readonly=False) as con:
        if not _open(con, branch, start):
            return {"ok": False, "reason": "ngoài giờ mở cửa"}
        free = _overlap_free_tables(con, branch, party_size, start)
        if not free:
            return {"ok": False, "reason": "hết bàn phù hợp"}
        rid = new_reservation_id()
        con.execute("INSERT INTO reservations VALUES (?,?,?,?,?,?,?,'CONFIRMED',?,1)",
                    (rid, branch, free[0]["table_id"], customer_name, phone, int(party_size), start_at, note))
    return {"ok": True, "reservation_id": rid, "table_id": free[0]["table_id"], "hold_minutes": 15}


def cancel_reservation(reservation_id: str, db=DB) -> dict:
    with _con(db, readonly=False) as con:
        n = con.execute("UPDATE reservations SET status = 'CANCELLED' WHERE reservation_id = ? AND status = 'CONFIRMED'",
                        (reservation_id,)).rowcount
    return {"ok": n == 1, "reservation_id": reservation_id}


def handoff_to_staff(reason: str) -> dict:
    return {"ok": True, "queued": True, "reason": reason}


REGISTRY = {f.__name__: f for f in (get_menu_item, search_menu, get_branch_info, check_availability, create_reservation,
                                    cancel_reservation, handoff_to_staff)}


def execute(call: dict, db=DB) -> dict:
    fn = REGISTRY.get(call.get("name"))
    if fn is None:
        return {"error": f"không có công cụ {call.get('name')}"}
    try:
        kw = dict(call.get("arguments") or {})
        if fn is not handoff_to_staff:
            kw["db"] = db
        return fn(**kw)
    except (TypeError, ValueError) as e:
        return {"error": str(e)}
