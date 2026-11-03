"""Nạp model, sinh câu trả lời theo lô và huấn luyện LoRA / QLoRA (chạy trên GPU Kaggle T4).

Phiên bản đã kiểm: transformers 4.56, trl 0.21, peft 0.17, bitsandbytes 0.47.
T4 không hỗ trợ bf16 → dùng fp16; tham số LoRA được ép về fp32 để GradScaler của AMP hoạt động.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import pandas as pd

MODELS = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",  # cần chấp nhận license trên HF + secret HF_TOKEN
}
LLAMA_MIRROR = "unsloth/Llama-3.2-3B-Instruct"  # bản sao không cần license, cùng trọng số
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _torch():
    import torch
    return torch


def bnb_config():
    from transformers import BitsAndBytesConfig
    torch = _torch()
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16)


def load(model_key: str, quant: str = "fp16", adapter: str | None = None, for_training: bool = False):
    """quant: 'fp16' (LoRA) hoặc 'nf4' (QLoRA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch = _torch()
    name = MODELS.get(model_key, model_key)
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except OSError:
        if model_key != "llama":
            raise
        print("Không truy cập được bản gốc LLaMA → dùng", LLAMA_MIRROR)
        name = LLAMA_MIRROR
        tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": torch.float16, "device_map": {"": 0}}
    if quant == "nf4":
        kw["quantization_config"] = bnb_config()
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    if not for_training:
        model.eval()
    model.config.use_cache = not for_training
    return model, tok, name


def free():
    """Gọi SAU khi đã `del model, tok` ở nơi gọi — hàm không thể xóa biến của notebook."""
    torch = _torch()
    gc.collect()
    torch.cuda.empty_cache()


def generate(model, tok, batch_messages: list[list[dict]], max_new_tokens=192, bs=8) -> tuple[list[str], list[float]]:
    """Sinh greedy theo lô (padding trái). Trả về câu trả lời và thời gian/mẫu (giây, chia đều trong lô)."""
    torch = _torch()
    tok.padding_side = "left"
    outs, lats = [], []
    for i in range(0, len(batch_messages), bs):
        chunk = batch_messages[i:i + bs]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        dt = time.perf_counter() - t0
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        lats += [dt / len(chunk)] * len(chunk)
    return [o.strip() for o in outs], lats


def latency_single(model, tok, messages: list[list[dict]], max_new_tokens=192) -> list[float]:
    """Độ trễ thật khi phục vụ từng khách (batch = 1) — dùng cho KPI p95 ≤ 5 s."""
    return [generate(model, tok, [m], max_new_tokens, bs=1)[1][0] for m in messages]


# ---------------------------------------------------------------- huấn luyện
def sft_dataset(root: Path, split: str, frac: float = 1.0, seed: int = 42, data_dir: str = "splits_v2"):
    """Mỗi lượt assistant → 1 mẫu prompt (toàn bộ hội thoại trước đó) / completion (đúng lượt đó).
    TRL chỉ tính loss trên completion → không học các lượt user / kết quả công cụ. Hỗ trợ hội thoại nhiều lượt và gọi công cụ.
    data_dir: 'splits_v2' (mặc định) hoặc 'splits' (v1, closed-book) để so sánh v1 ↔ v2."""
    from datasets import Dataset
    df = pd.read_json(root / "data" / data_dir / f"{split}.jsonl", lines=True)
    if frac < 1:
        # lấy mẫu theo nhóm để giữ nguyên tính chất chống rò rỉ
        groups = df["meta"].map(lambda m: m["group_id"])
        keep = pd.Series(groups.unique()).sample(frac=frac, random_state=seed)
        df = df[groups.isin(set(keep))]
    prompts, completions = [], []
    for msgs in df["messages"]:
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                prompts.append(msgs[:i])
                completions.append([m])
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def train(root: Path, model_key: str, quant: str, out_dir: str, seed: int = 42, lr: float = 2e-4, r: int = 16,
          epochs: float = 2, max_steps: int = -1, data_frac: float = 1.0, max_len: int = 1024, eval_steps: int = 50,
          per_device_bs: int = 4, grad_accum: int = 4, extra_callbacks=(), report_to="none", data_dir: str = "splits_v2") -> dict:
    """Huấn luyện 1 ô ma trận {model} × {LoRA fp16, QLoRA nf4}. Trả về chỉ số tài nguyên + lịch sử loss."""
    torch = _torch()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(seed)
    model, tok, name = load(model_key, quant, for_training=True)
    if quant == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=TARGET_MODULES,
                                             task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cfg = SFTConfig(
        output_dir=out_dir, seed=seed, learning_rate=lr, num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=per_device_bs, per_device_eval_batch_size=8, gradient_accumulation_steps=grad_accum,
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0, fp16=True, bf16=False,
        optim="paged_adamw_8bit" if quant == "nf4" else "adamw_torch",
        logging_steps=10, eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        max_length=max_len, completion_only_loss=True, packing=False, report_to=report_to, gradient_checkpointing=False,
    )
    trainer = SFTTrainer(model=model, args=cfg, processing_class=tok,
                         train_dataset=sft_dataset(root, "train", data_frac, seed, data_dir),
                         eval_dataset=sft_dataset(root, "val", data_dir=data_dir),
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=3), *extra_callbacks])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    ev = trainer.evaluate()
    trainer.model.save_pretrained(out_dir + "/adapter")
    tok.save_pretrained(out_dir + "/adapter")
    hist = trainer.state.log_history
    res = {"model": model_key, "base": name, "quant": quant, "seed": seed, "lr": lr, "r": r, "data_frac": data_frac,
           "data": data_dir, "trainable_params": n_train, "train_minutes": round(minutes, 1),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "eval_loss": ev["eval_loss"],
           "eval_ppl": math.exp(ev["eval_loss"]), "steps": trainer.state.global_step, "adapter": out_dir + "/adapter"}
    Path(out_dir, "history.json").write_text(json.dumps(hist))
    Path(out_dir, "result.json").write_text(json.dumps(res, indent=1))
    del trainer, model
    free()
    return res


def push_adapter(local_dir: str, repo_id: str, private: bool = True):
    """Đẩy adapter lên Hugging Face Hub (không commit trọng số vào git — CLAUDE.md §3.7)."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
