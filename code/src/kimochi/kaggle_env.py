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

# Notebook chạy trên 1 GPU: với "GPU T4 x2", Trainer tự bật DataParallel trong khi mô hình nạp cố định lên cuda:0
# → lỗi lệch kích thước lô ("size of tensor a (4) must match ... (8)"). Phải đặt TRƯỚC khi import torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Không để transformers nạp TensorFlow/JAX (bản TF trên Kaggle không khớp CUDA 13 → in "Could not find cuda drivers"; vô hại).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

WORK = Path("/kaggle/working/repo")


def on_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def setup(secrets=("HF_TOKEN", "GEMINI_API_KEY")) -> Path:
    if not on_kaggle():
        root = Path(__file__).resolve().parents[3]
    else:
        src = next(Path(p).parents[2] for p in glob.glob("/kaggle/input/**/code/src/kimochi", recursive=True))
        ign = shutil.ignore_patterns("*.ipynb", "docs", "*.pdf", "__pycache__")
        if not WORK.exists():
            shutil.copytree(src, WORK, ignore=ign)
        else:   # luôn làm mới code / dữ liệu / demo từ dataset (bản sửa mới có hiệu lực), giữ nguyên results/ đã sinh
            for sub in ("code", "data", "demo"):
                if (src / sub).exists():
                    shutil.rmtree(WORK / sub, ignore_errors=True)
                    shutil.copytree(src / sub, WORK / sub, ignore=ign)
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
