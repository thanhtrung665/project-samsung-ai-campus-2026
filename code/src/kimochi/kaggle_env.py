"""Chuẩn bị môi trường Kaggle: chép repo (dataset chỉ đọc) sang /kaggle/working/repo, nạp secrets.

Cách dùng ở ô đầu mỗi notebook Kaggle:
    import sys, glob; sys.path.insert(0, glob.glob('/kaggle/input/*/code/src')[0])
    from kimochi.kaggle_env import setup; ROOT = setup()
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

WORK = Path("/kaggle/working/repo")


def on_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def setup(secrets=("HF_TOKEN", "GEMINI_API_KEY")) -> Path:
    if not on_kaggle():
        root = Path(__file__).resolve().parents[3]
    else:
        src = next(Path(p).parents[2] for p in glob.glob("/kaggle/input/**/code/src/kimochi", recursive=True))
        if not WORK.exists():
            shutil.copytree(src, WORK, ignore=shutil.ignore_patterns("*.ipynb", "docs", "*.pdf"))
        root = WORK
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
            for name in secrets:
                try:
                    os.environ[name] = client.get_secret(name)
                except Exception:
                    print(f"(chưa khai báo secret {name})")
        except ImportError:
            pass
    sys.path[:] = [p for p in sys.path if not p.endswith("code/src")]
    sys.path.insert(0, str(root / "code" / "src"))
    for m in [m for m in sys.modules if m.startswith("kimochi")]:
        del sys.modules[m]  # nạp lại từ bản đã chép (ghi được)
    (root / "results").mkdir(exist_ok=True)
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    print("ROOT =", root)
    return root


def gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return ", ".join(f"{torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GB)"
                             for i in range(torch.cuda.device_count()))
    except ImportError:
        pass
    return "CPU"
