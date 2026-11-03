# CLAUDE.md — Tiểu luận cuối khóa SIC 2026: Chatbot CSKH Kimochi Sushi

> Bộ 3 tài liệu: **`CLAUDE.md`** (luật + bối cảnh + cấu trúc) → [`SPEC.md`](SPEC.md) (yêu cầu, kiến trúc, kế hoạch, backlog
> task) → [`PROGRESS.md`](PROGRESS.md) (nhật ký, mới nhất lên đầu). Khi lệch nhau: **CLAUDE.md thắng**, rồi tới rubric gốc.

## 1. Dự án là gì

Tiểu luận cuối khóa học phần *Trí tuệ Nhân tạo Ứng dụng & Khoa học Dữ liệu* — Samsung Innovation Campus 2026.
Đề tài: **"Nghiên cứu xây dựng Chatbot chăm sóc khách hàng dựa trên công nghệ Hybrid-RAG kết hợp Fine-tuning mô hình Qwen &
LLaMA"** cho nhà hàng **Kimochi Sushi** (Nhật – Việt, 2 chi nhánh Bình Thạnh & Gò Vấp, TP.HCM, kênh chính: fanpage).
Tác giả: Hồ Thành Trung — 075205020424.

- **Hạn nộp: 23:59 ngày 11/11/2026** (LMS, file `[Nhom_XX]_[MaDeTai]_[TenDeTai].zip`).
- Đây là **phiên bản 2** của một dự án trước (`docs/KIMOCHI BÁO CÁO.pdf`: LLaMA 3.2-1B + LoRA, ~7.000 mẫu, RAG Qdrant,
  đo ROUGE/BERTScore, fallback 25,7%). Bản mới: so sánh **Qwen2.5-3B vs LLaMA-3.2-3B × LoRA vs QLoRA**, **Hybrid RAG**
  (BM25 + dense + RRF + reranker), **Agent** truy cập SQL + NoSQL (tra giá, bàn trống, đặt bàn).
- Nguyên tắc kỹ thuật cốt lõi: **fine-tuning dạy hành vi** (giọng điệu, bám context, gọi công cụ, từ chối);
  **SQL/RAG cung cấp sự thật** (giá, giờ, bàn trống, khuyến mãi) — không nhồi dữ kiện động vào trọng số.

## 2. Hồ sơ nộp bắt buộc (rubric `docs/huong_dan_tieu_luan_va_rubric_cuoi_khoa.pdf`)

1. `report_main.qmd` — nguồn Quarto hoàn chỉnh.
2. `report_main.pdf` — `quarto render report_main.qmd --to pdf` (XeLaTeX), **≤ 45 trang** (không tính code), `echo: false`.
3. `code/` — notebook `.ipynb` đã **Restart & Run All**, hiển thị đủ kết quả.
4. `data/` — dữ liệu mẫu (file > 50 MB → link Drive/Kaggle/HF).
5. `demo/` — link Web App (HF Spaces / Streamlit Cloud) **hoặc** video 2–3 phút.

Trọng số rubric: I Bài toán 15% · II EDA & tiền xử lý 20% · III Mô hình & chống quá khớp 25% · IV Đánh giá, XAI, đạo đức 15% ·
V Prototype 15% · VI Trình bày & mã nguồn 10% · **Bonus +0.5–1.0** (dữ liệu thật doanh nghiệp / người dùng thật thử).

## 3. Luật làm việc (bất biến)

1. **Mỗi việc có mã task trong SPEC §7** (vd `T05`). Việc chưa có mã → thêm dòng vào backlog trước. Kết thúc task: cập nhật trạng
   thái ở SPEC §7 + ghi `PROGRESS.md` (mới nhất lên đầu, ghi mã task).
2. **Không bịa số liệu kết quả.** Mọi con số trong báo cáo (metric, VRAM, thời gian, loss) phải sinh từ notebook trong `code/` và
   lưu ở `results/`. Ô chưa chạy để `TBD`, không điền ước lượng như kết quả thật. Giả định (ROI…) phải ghi rõ là giả định.
3. **Test-gold đóng băng:** viết tay, lưu SHA-256 trong `data/splits/CHECKSUMS.txt`; không dùng làm seed, không đưa cho LLM "thầy",
   không sửa sau khi khóa. Chia split theo `group_id` (GroupShuffleSplit), stratify theo intent, kiểm rò rỉ cosine > 0,92.
4. **Giá, giờ, địa chỉ, bàn trống chỉ lấy từ SQL**; câu trả lời có con số phải khớp DB (fact-EM). Công cụ SQL chỉ đọc, trừ
   `reservation_create/cancel` qua hàm tham số hóa (không ghép chuỗi SQL).
5. **Cùng ngân sách cho mọi ô thí nghiệm** (cùng dữ liệu, số bước, prompt đánh giá) — chỉ đổi model / kiểu lượng tử. Ghi seed.
6. **Liêm chính học thuật:** trích dẫn IEEE `\[n\]` thủ công trong danh mục cuối `.qmd`; Turnitin ≤ 20%; mục "Tuyên bố sử dụng AI"
   phải liệt kê đúng công cụ AI đã dùng (kể cả Claude Code).
7. **Không commit secrets** (HF token, OpenAI/Gemini key, Qdrant key) — dùng `.env` (gitignore) + `.env.example`.
   Không commit checkpoint/adapter lớn — đẩy lên Hugging Face Hub, ghi link.
8. Dữ liệu cá nhân khách hàng (tên, SĐT) trong mẫu thật phải ẩn danh trước khi đưa vào `data/`.
9. Mỗi biểu đồ/bảng trong báo cáo: có caption + label Quarto (`#fig-…` dưới hình, `#tbl-…` trên bảng) **và** 1–2 câu nhận xét
   kinh doanh. Hình đặt ở `images/`.

## 4. Cấu trúc repo (hiện tại → đích)

```
report_main.qmd                 # báo cáo (đã có Phần 1–2 + trang đầu + TLTK)
docs/                           # rubric, kế hoạch 12 ngày (.docx), báo cáo bản v1 (.pdf)
data/
  KIMOCHI_DATA_TRAINING_CHATBOT.json   # 6.355 cặp Q&A thật/bán thật (Domain, user_query, responses) — kế thừa v1
  kb/            (đích) menu.csv, branches.csv, policies/*.md, faq.jsonl  → nguồn SQL + NoSQL
  kimochi.db     (đích) SQLite: menu_items, categories, branches, tables, reservations, promotions
  kb/chunks.jsonl  đơn vị truy xuất RAG (≤ 600 ký tự) — dùng chung cho huấn luyện RAFT và lúc chạy
  splits/        v1: train/val/test_synth (+ test_gold/test_adv viết tay + CHECKSUMS.txt) — TẬP TEST CHÍNH THỨC
  splits_v2/     dữ liệu huấn luyện v2 (train/val + README dataset card) — chỉ train/val, không có test
notebooks/       NB1_data_pipeline (CPU) · NB2_finetune_rag_mvp (GPU) · cpu/01–05 · kaggle/K1–K7 (bước chi tiết)
code/            src/kimochi/ (build_kb, prepare_sft(_v2), eda, retriever, tools, agent, agent_eval, prompts, metrics, llm,
                 viz, manifest, report_tables) · script make_*_notebooks.py, make_submission.py, freeze_tests.py
results/         (đích) *.csv / *.json kết quả, đọc vào .qmd bằng chunk Python ẩn
images/          (đích) logo.png, fig_kien_truc.png, biểu đồ EDA/kết quả
demo/            (đích) app Gradio/Streamlit + README link Space / video
```

## 5. Stack & lệnh

- Huấn luyện: Colab/Kaggle **T4 16 GB**; HF `transformers` + `trl` SFTTrainer + `peft` (+ Unsloth), `bitsandbytes` (NF4,
  double quant), Optuna. Model: `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`.
- RAG: `rank_bm25` (tách từ tiếng Việt `underthesea`/`pyvi`), embedding `BAAI/bge-m3`, Qdrant (local hoặc Cloud), RRF,
  reranker `BAAI/bge-reranker-v2-m3`. NoSQL tài liệu gốc: MongoDB (hoặc JSON/TinyDB nếu thiếu thời gian — ghi rõ).
- Agent: LangGraph/ReAct đơn giản, tool JSON; SQL: SQLite. Đánh giá: RAGAS, LLM-as-judge (hiệu chỉnh với người chấm), bootstrap.
- Demo: Gradio trên Hugging Face Spaces.

```bash
quarto render report_main.qmd --to pdf      # cần quarto + xelatex + font Times New Roman
jupyter nbconvert --to notebook --execute --inplace notebooks/NB1_data_pipeline.ipynb   # = Restart & Run All
```

**Tính tái lập:** mọi bước sinh dữ liệu phải cho cùng md5 giữa các lần chạy (không dùng thứ tự `set` chuỗi, uuid, thời gian).

## 6. Lưu ý khi sửa `report_main.qmd`

- Giữ YAML hiện có (trang bìa tikz tự viết, `toc: false` + mục lục chèn thủ công). Rubric mẫu dùng `toc: true` — mục lục tự
  động phải có trong PDF cuối. Bổ sung `execute: echo: false, warning: false` khi thêm chunk code.
- Placeholder còn trống: `[TÊN TRƯỜNG ĐẠI HỌC]`, `[TÊN KHOA]`, giảng viên, `images/logo.png`, `images/fig_kien_truc.png`.
- Báo cáo tiếng Việt học thuật, ngôi "chúng tôi"; số thập phân dấu phẩy trong LaTeX `0{,}90`.
- Phần 1 nói Kimochi là **nhà hàng thật** (2 chi nhánh) còn kế hoạch nói "giả lập 3 chi nhánh" → **thống nhất theo nhà hàng
  thật** (lợi thế bonus) và ghi rõ phần dữ liệu nào là mô phỏng (bàn trống, đặt bàn, đơn hàng).
