# Notebooks — Chatbot CSKH Kimochi Sushi

## Hai notebook tổng hợp (dùng để nộp / trình bày)
| Notebook | Chạy ở đâu | Nội dung | Đầu ra |
|---|---|---|---|
| **`NB1_data_pipeline.ipynb`** | CPU (~1 phút, đã chạy sẵn kết quả) | Nạp dữ liệu → kho tri thức SQLite + RAG → chất lượng dữ liệu (thiếu giá, giá trộn VAT) → nhãn intent & đặc trưng → chia train/val/test theo nhóm + kiểm rò rỉ → EDA có nhận xét kinh doanh → chuẩn hóa thành dữ liệu v2 (công cụ SQL, RAFT, hội thoại nhiều lượt, từ chối) → kiểm tra sẵn sàng → manifest md5 | `data/`, `images/eda_*`, `results/*_stats.json`, `data/MANIFEST.json` |
| **`NB2_finetune_rag_mvp.ipynb`** | Kaggle GPU T4 | Nhận dữ liệu (kiểm manifest) → thiết lập (Optuna tùy chọn, baseline) → fine-tune **LoRA 16-bit**, **QLoRA 4-bit** (+ 8-bit tùy chọn) × {Qwen2.5-3B, LLaMA-3.2-3B} → đánh giá trên test → trực quan & phân tích (CI 95%, VRAM ↔ chất lượng, intent, câu nhiễu, ca lỗi) → demo nhanh → **Hybrid-RAG** (BM25 + dense + RRF + reranker + lọc) & ablation → đánh giá hệ thống (gốc / +RAG / fine-tune / +RAG / +Agent), định tuyến công cụ, kịch bản đặt bàn, judge → **xuất mô hình gộp** + gói MVP → demo MVP | `results/*` (báo cáo tự đọc), `/kaggle/working/kimochi-mvp-model`, `kimochi-mvp-bundle.zip` |

Thứ tự: **NB1 → tải repo lên Kaggle (Dataset) → NB2**. NB2 dừng nếu dữ liệu lệch manifest của NB1.
NB2 có thể chạy nhiều phiên: thêm output phiên trước làm input, các bước đã xong tự bỏ qua.

## Notebook chi tiết theo từng bước (tham khảo / chạy riêng lẻ)
- `cpu/01_build_kb` · `02_prepare_sft` · `03_eda` · `04_rag` (BM25) · `05_prepare_sft_v2` — các bước của NB1 tách riêng.
- `kaggle/K1_rag_dense` · `K2_baseline` · `K3_optuna_qlora` · `K4_train_matrix` (+ đối chứng dữ liệu v1/v2) · `K5_eval` ·
  `K6_agent_eval` · `K7_demo` — các bước của NB2 tách riêng; hướng dẫn Kaggle: `kaggle/README.md`.

Mọi notebook tìm thư mục gốc repo tự động; logic nằm trong `code/src/kimochi/`. Sinh lại notebook từ mã:
`python code/make_main_notebooks.py` · `python code/make_notebooks.py` · `python code/make_kaggle_notebooks.py`.

> NB2 chưa được chạy trên GPU trong môi trường soạn (không có GPU / Hugging Face); các phần CPU của nó (nhận dữ liệu, chỉ số,
> truy xuất BM25, công cụ, agent, biểu đồ) đã được kiểm thử.
