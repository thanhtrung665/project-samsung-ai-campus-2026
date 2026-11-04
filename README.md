# Chatbot CSKH Kimochi Sushi — Hybrid RAG + Fine-tuning Qwen & LLaMA (LoRA / QLoRA)

Tiểu luận cuối khóa *Trí tuệ Nhân tạo Ứng dụng & Khoa học Dữ liệu* — Samsung Innovation Campus 2026. Tác giả: Hồ Thành Trung.

| Hạng mục nộp | Vị trí |
|---|---|
| Báo cáo Quarto / PDF | `report_main.qmd` · `report_main.pdf` (`quarto render report_main.qmd --to pdf`) |
| Notebook | **`notebooks/NB1_data_pipeline.ipynb`** (dữ liệu, CPU, đã chạy) · **`notebooks/NB2_finetune_rag_mvp.ipynb`** (fine-tune → RAG → MVP, GPU Kaggle) · bước chi tiết `notebooks/cpu/`, `notebooks/kaggle/` |
| Mã nguồn | thư viện `code/src/kimochi/` · script `code/*.py` |
| Dữ liệu | `data/` — hội thoại gốc, kho tri thức `kb/`, SQLite `kimochi.db`, split v1 `splits/` (tập test chính thức), dữ liệu huấn luyện v2 `splits_v2/` (xem dataset card) |
| Sản phẩm mẫu | `demo/` — ứng dụng Gradio, hướng dẫn chạy và kịch bản video |
| Kết quả | `results/` — mọi con số trong báo cáo được đọc từ đây |

## Chạy lại
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/NB1_data_pipeline.ipynb
```
Phần GPU (baseline, Optuna, huấn luyện ma trận LoRA/QLoRA, đánh giá, Agent, demo): chạy `notebooks/NB2_finetune_rag_mvp.ipynb` trên Kaggle (xem `notebooks/README.md`).
Demo không cần GPU (chế độ thử giao diện): `python demo/app.py`.
Kiểm tra hồ sơ và đóng gói: `python code/make_submission.py --group XX --topic-code XX --name "Chatbot CSKH Kimochi Sushi"`.

Dữ liệu bàn / đặt bàn là mô phỏng; thông tin dị ứng được suy luận tự động, chưa được nhà hàng xác nhận.
