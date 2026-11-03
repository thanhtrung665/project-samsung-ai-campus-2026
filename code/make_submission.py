"""T17 — Kiểm tra hồ sơ theo rubric và đóng gói file nộp LMS.

    python code/make_submission.py --group 07 --topic-code CB01 --name "Chatbot CSKH Kimochi" [--run-notebooks] [--render]

Kiểm tra: đủ 5 hạng mục bắt buộc · PDF ≤ 45 trang · test viết tay chưa bị sửa (SHA-256) · không có file > 50 MB ·
không lộ secret · còn ô TBD / placeholder trong báo cáo (cảnh báo). Tạo [Nhom_XX]_[MaDeTai]_[TenDeTai].zip ở thư mục gốc repo.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["report_main.qmd", "report_main.pdf", "README.md", "requirements.txt", ".env.example", "code", "data", "demo", "images",
           "results"]
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git"}
EXCLUDE_SUFFIX = {".pyc", ".db-journal", ".zip", ".safetensors", ".bin"}
MAX_FILE_MB = 50
SECRET_RE = re.compile(r"(hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{30,})")
CPU_NOTEBOOKS = ["01_build_kb", "02_prepare_sft", "03_eda", "04_rag", "05_prepare_sft_v2"]


def slug(s: str) -> str:
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def files() -> list[Path]:
    out = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*") if f.is_file() and not (set(f.relative_to(ROOT).parts) & EXCLUDE_DIRS)
                    and f.suffix not in EXCLUDE_SUFFIX]
    return sorted(out)


def check(errors: list, warns: list):
    for need in ["report_main.qmd", "report_main.pdf", "code", "data", "demo"]:
        if not (ROOT / need).exists():
            errors.append(f"thiếu hạng mục bắt buộc: {need}")
    pdf = ROOT / "report_main.pdf"
    if pdf.exists():
        try:
            info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
            pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
            if pages > 45:
                errors.append(f"PDF {pages} trang > 45")
            else:
                print(f"✓ PDF {pages} trang (≤ 45)")
        except Exception:
            warns.append("không đọc được số trang PDF (cần pdfinfo)")
        if (ROOT / "report_main.qmd").stat().st_mtime > pdf.stat().st_mtime:
            warns.append("report_main.qmd mới hơn PDF → chạy lại với --render")
    qmd = (ROOT / "report_main.qmd").read_text(encoding="utf-8")
    for ph in ["[TÊN TRƯỜNG", "[TÊN KHOA", "[Học hàm", "Nhóm XX"]:
        if ph in qmd:
            warns.append(f"còn placeholder trong báo cáo: {ph}…")
    pdf_txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout if pdf.exists() else ""
    if pdf_txt.count("TBD"):
        warns.append(f"PDF còn {pdf_txt.count('TBD')} ô TBD (chưa có kết quả GPU)")
    ck = ROOT / "data" / "splits" / "CHECKSUMS.txt"
    if ck.exists():
        r = subprocess.run([sys.executable, str(ROOT / "code" / "freeze_tests.py"), "--check"], capture_output=True, text=True)
        if r.returncode:
            errors.append("test viết tay đã bị sửa sau khi khóa: " + (r.stderr or r.stdout).strip())
        else:
            print("✓ test viết tay khớp SHA-256")
    else:
        warns.append("chưa có test viết tay đã khóa (data/splits/CHECKSUMS.txt)")
    if not any((ROOT / "demo").glob("*.mp4")) and "http" not in (ROOT / "demo" / "README.md").read_text(encoding="utf-8").split("## Kịch bản")[0].replace(
            "https://huggingface.co", ""):
        warns.append("demo/: chưa có video (.mp4) hoặc link Space / video")
    for f in files():
        mb = f.stat().st_size / 2**20
        if mb > MAX_FILE_MB:
            errors.append(f"file > {MAX_FILE_MB} MB: {f.relative_to(ROOT)} ({mb:.0f} MB) → đưa lên Drive/Kaggle/HF, ghi link")
        if f.suffix in {".py", ".ipynb", ".md", ".qmd", ".json", ".jsonl", ".txt", ".csv", ".env", ".example"} and mb < 20:
            if SECRET_RE.search(f.read_text(encoding="utf-8", errors="ignore")):
                errors.append(f"nghi lộ secret trong {f.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="XX")
    ap.add_argument("--topic-code", default="MaDeTai")
    ap.add_argument("--name", default="Chatbot CSKH Kimochi Sushi")
    ap.add_argument("--run-notebooks", action="store_true", help="Restart & Run All các notebook CPU 01–05")
    ap.add_argument("--render", action="store_true", help="quarto render report_main.qmd --to pdf")
    a = ap.parse_args()
    if a.run_notebooks:
        for n in CPU_NOTEBOOKS:
            subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", f"{n}.ipynb"], cwd=ROOT / "code", check=True)
    if a.render:
        subprocess.run(["quarto", "render", "report_main.qmd", "--to", "pdf"], cwd=ROOT, check=True)
    errors, warns = [], []
    check(errors, warns)
    for w in warns:
        print("⚠", w)
    for e in errors:
        print("✗", e)
    if errors:
        sys.exit("Có lỗi — chưa đóng gói.")
    out = ROOT / f"Nhom_{slug(a.group)}_{slug(a.topic_code)}_{slug(a.name)}.zip"
    fl = files()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in fl:
            z.write(f, f.relative_to(ROOT))
    print(f"✓ {out.name}: {len(fl)} file, {out.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
