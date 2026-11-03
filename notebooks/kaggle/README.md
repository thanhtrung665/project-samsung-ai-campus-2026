# Chạy các notebook GPU trên Kaggle

> Cách gọn nhất: chạy **`../NB2_finetune_rag_mvp.ipynb`** (gộp K1–K7). Các notebook dưới đây là từng bước tách riêng.

| Notebook | Task | Thời gian ước tính (T4) | Đầu ra (`results/`) |
|---|---|---|---|
| `K1_rag_dense.ipynb` | T09 dense + reranker | 10–15 phút | `retrieval_val.csv`, `retrieval_test_synth.csv`, `retrieval_test_gold.csv` |
| `K2_baseline.ipynb` | T10 B0 / B1 / B2 × 2 model | 1–1,5 giờ | `baseline.csv`, `gen/*.jsonl` |
| `K3_optuna_qlora.ipynb` | T11 Optuna 8 trial | 2–3 giờ | `optuna_trials.csv`, `optuna_best.json`, `optuna_history.png` |
| `K4_train_matrix.ipynb` | T11 ma trận 2×2 + learning curve | 4–7 giờ (chia nhiều phiên được) | `train_matrix.csv`, `loss_curves.png`, `learning_curve.*`, adapter |
| `K5_eval.ipynb` | T13 đánh giá + judge | 1,5–2,5 giờ | `finetuned_eval.csv`, `ci_*.csv`, `judge_scores.csv`, `error_cases.csv` |

Thứ tự: K1 và K2 chạy song song được → K3 → K4 → K5. Thời gian là ước lượng, ghi số thật vào báo cáo từ `results/`.

## Chuẩn bị (một lần)
1. **Dataset:** nén thư mục repo (cần `code/src`, `data/`; không cần `docs/`) → Kaggle → *Datasets → New Dataset* → upload,
   đặt tên vd `kimochi-repo`. Khi repo thay đổi (vd đã viết `test_gold.jsonl`) → *New Version*.
2. **LLaMA:** chấp nhận license `meta-llama/Llama-3.2-3B-Instruct` trên Hugging Face. Nếu chưa được duyệt, code tự dùng bản
   sao `unsloth/Llama-3.2-3B-Instruct` (cùng trọng số) — ghi rõ trong báo cáo.
3. **Secrets** (*Add-ons → Secrets*): `HF_TOKEN` (token đọc; quyền ghi nếu đẩy adapter lên Hub), `GEMINI_API_KEY` (chỉ K5).

## Mỗi notebook
- *File → Import Notebook* → chọn file `.ipynb` trong thư mục này.
- *Settings*: Accelerator **GPU T4 x2**, Internet **On**. *Add Data*: dataset repo (K5 thêm output của K4).
- *Save Version → Save & Run All (Commit)* để chạy nền (không cần giữ trình duyệt) — đây cũng là "Restart & Run All" mà rubric yêu cầu.
- Xong: tab *Output* → tải `repo/results/` → chép vào `results/` của repo; adapter đẩy lên HF Hub hoặc tải `adapters.zip`
  (**không commit trọng số vào git**).

## Lưu ý
- Hạn mức GPU Kaggle ~30 giờ/tuần, phiên tối đa 12 giờ → nếu thiếu, cắt theo SPEC §7: 1 seed, Optuna 4 trial, bỏ LoRA fp16 của
  model kém hơn.
- `N_EVAL = 150` trong K2 để chạy thử nhanh trước khi chạy toàn bộ.
- Các notebook này chưa được chạy thử trên GPU (môi trường soạn không có GPU / Hugging Face); phần CPU (dữ liệu, chỉ số,
  lời nhắc) đã được kiểm thử. Lỗi phát sinh khi chạy lần đầu → báo lại kèm traceback.
