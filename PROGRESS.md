# PROGRESS — Nhật ký dự án Chatbot Kimochi (mới nhất lên đầu)

## Phần I. Nhật ký

### 2026-11-03 (đêm) — T07b bộ dữ liệu SFT v2
- `prepare_sft_v2.py` + `05_prepare_sft_v2.ipynb` → `data/splits_v2/{train,val}.jsonl` (4.980 / 580 hội thoại, 433 nhiều lượt)
  + dataset card `data/splits_v2/README.md`. Làm sạch (−60 trùng, −239 nhãn mâu thuẫn); 235 câu hỏi giá → gọi `get_menu_item`
  (giá theo DB); xóa giá trong 839 câu trả lời khác, loại 40 dòng giá mâu thuẫn; 1.077 mẫu RAFT (20% chỉ nhiễu → từ chối);
  257 mẫu khuôn (đặt bàn nhiều lượt có xác nhận trước khi ghi, chi nhánh, tìm món, khiếu nại, ngoài phạm vi / injection, chào hỏi);
  3 biến thể system prompt. Test giữ nguyên v1 để so sánh v1 ↔ v2.
- `tools.py` (7 công cụ SQLite tham số hóa, định dạng `<tool_call>{json}</tool_call>`), `agent.py` (vòng lặp công cụ, chạy
  trên bản sao DB). KB chia chunk ≤ 600 ký tự (`data/kb/chunks.jsonl`, 1.019 chunk) để ngữ cảnh huấn luyện = ngữ cảnh chạy thật;
  retriever chuyển sang chunk: BM25 + lọc R@1 0,928 / R@5 0,965.
- `llm.sft_dataset`: mỗi lượt assistant → 1 mẫu prompt/completion (5.883 mẫu từ v2); mặc định `splits_v2`. K4 thêm ablation
  dữ liệu v1 ↔ v2; K5 thêm chế độ `agent` (đo tỷ lệ gọi công cụ cho câu hỏi giá).
- **Sửa tính tái lập:** sắp xếp tên món theo độ dài bị phụ thuộc thứ tự `set` (hash ngẫu nhiên), mã đặt bàn dùng uuid4 →
  nay cố định; kiểm 3 lần chạy cho cùng md5. Sửa cột dị ứng rỗng trong `eda_menu.png`.
- Báo cáo: thêm mục "Xây dựng bộ dữ liệu huấn luyện v2" cuối Phần 3 (PDF 25 trang).
- Việc còn lại cho dữ liệu: mẫu sinh bằng LLM thầy cho câu đa dạng hơn (tùy chọn), test-gold viết tay (T06, nhóm).

### 2026-11-03 (tối) — Notebook Kaggle cho T09–T13; sửa nhãn intent; phát hiện giá trộn VAT
- `code/kaggle/K1–K5` + `README.md` (hướng dẫn upload dataset, secrets, thứ tự chạy). Module mới `kaggle_env.py`, `prompts.py`,
  `metrics.py` (ROUGE-L, fact-EM giá, bịa giá, từ chối, bootstrap, judge Gemini), `llm.py` (nạp model, sinh theo lô, train
  LoRA fp16 / QLoRA nf4 với TRL 0.21 prompt/completion → loss chỉ trên câu trả lời). Phần CPU đã kiểm thử; **chưa chạy GPU**.
- **Sửa lỗi gán intent:** luật giá đọc chữ "Giá" trong Domain ("Giá & Hương vị: X") nên gán nhầm câu "ăn kèm gì" là
  price_query; `\bk\b` khớp teen code "k" (= không). Sau sửa: menu_info 60%, recommendation 14%, allergy 12,5%, price 8%;
  split 5.085 / 557 / 577, loại 136 câu rò rỉ. Đã chạy lại notebook 02–04 và cập nhật số trong Phần 3.
  Retrieval test_synth (339 câu): BM25 + lọc R@1 0,917 / R@5 0,962 / MRR 0,935.
- **Phát hiện chất lượng dữ liệu:** 26/74 món có giá mâu thuẫn giữa các câu trả lời; 16 món chênh đúng VAT 10%
  (109.000 ↔ 119.900), 10 món mâu thuẫn thật (Cơm Cuộn Lươn 95.000 ↔ 152.900). Đã đưa vào Phần 3 như lập luận cho thiết
  kế "giá chỉ đọc từ SQL". Cần nhà hàng xác nhận giá trước/sau VAT.

### 2026-11-03 (chiều) — T04, T05, T07 (một phần), T08, T09 (phần sparse)
- **T04** cấu trúc repo `code/ results/ images/ demo/`, `requirements.txt`, `.gitignore`, `.env.example`. Render PDF được
  (local dùng Liberation Serif thay Times New Roman vì môi trường không có font MS: `-M mainfont="Liberation Serif"`).
  Vẽ `images/fig_kien_truc.png` (`code/make_fig_architecture.py`) — trước đó thiếu hình làm render lỗi.
- **T05** `code/src/kimochi/build_kb.py` + `01_build_kb.ipynb`: 106 món chính tắc, 74 có giá; `data/kimochi.db` 6 bảng
  (bàn/đặt bàn mô phỏng); `data/kb/docs.jsonl` 106 tài liệu món + 545 tài liệu chủ đề. Phát hiện: 32 món thiếu giá (toàn bộ
  Sashimi), 1 xung đột giá Khoai tây chiên. Sửa 2 lỗi so khớp khi làm: `\\b` thoát sai, "mì" khớp trong "mình".
- **T07** `prepare_sft.py` + `02_prepare_sft.ipynb`: định dạng messages+meta, 8 intent theo luật, StratifiedGroupKFold theo
  group_id → train 5.084 / val 565 / test_synth 562; 0 nhóm trùng train↔test; loại 144 câu cosine > 0,92. Chưa làm: mẫu RAFT,
  tool-call, từ chối (cần LLM thầy + retriever).
- **T08** `eda.py` + `03_eda.ipynb`: 7 hình `images/eda_*.png`; viết **Phần 3** trong `.qmd` (PDF 24 trang). Phát hiện chính:
  intent lệch (52% menu_info, đặt bàn 1%), 44% câu hỏi không dấu, 99,3% đáp án dùng chung ≥ 3 câu hỏi.
- **T09 (sparse)** `retriever.py` + `04_rag.ipynb` trên 369 câu test có gắn món: BM25 R@1 0,827 / R@5 0,949; BM25 + lọc tên
  món R@1 0,883 / R@5 0,951 / MRR 0,912; RRF với TF-IDF ký tự *giảm* R@1 (0,718). Dense bge-m3 + reranker: `skipped`
  (Hugging Face bị chặn ở môi trường này) → chạy notebook trên Colab.
- **T06**: chỉ làm công cụ `code/freeze_tests.py` (kiểm định dạng + SHA-256). Không tự sinh test-gold: luật yêu cầu người viết,
  không qua LLM.
- Lưu ý: tài liệu RAG được dựng từ toàn bộ câu trả lời (kể cả của tập test) — hợp lệ vì KB là nguồn sự thật của nhà hàng, nhưng
  điểm truy xuất trên test_synth vì vậy lạc quan; số chính phải đo trên test-gold.

### 2026-11-03 — Phân tích repo, lập CLAUDE / SPEC / PROGRESS, tái lập kế hoạch
- Đọc rubric, kế hoạch 12 ngày (28/10), báo cáo v1, `report_main.qmd`, dữ liệu JSON.
- Tạo `CLAUDE.md` (luật + cấu trúc), `SPEC.md` (yêu cầu, kiến trúc, thí nghiệm, backlog T00–T18), `PROGRESS.md`.
- Phát hiện: dữ liệu 6.355 mẫu chỉ có 2.068 đáp án khác nhau → dùng làm `group_id` chia split; chưa có KB/SQL/test-gold/code.
- Mâu thuẫn cần chốt: báo cáo Phần 1 nói nhà hàng thật 2 chi nhánh, kế hoạch nói giả lập 3 chi nhánh → theo nhà hàng thật.
- Tiếp theo: **T04** dựng repo + render thử PDF, **T05** trích KB/SQLite.

### 2026-10-28 → 11-02 (trước khi có nhật ký)
- T00 kế hoạch 12 ngày + khung Quarto, trang bìa tikz.
- T01 Phần 1 hoàn chỉnh (pain points, 3 persona, empathy map, RQ1–RQ4, 7 KPI SMART, ROI ≈ 94%/hoàn vốn ~6,2 tháng + độ nhạy — giả định).
- T02 Phần 2 hoàn chỉnh (CRISP-DM có sơ đồ tikz, Transformer, Qwen2.5-3B vs LLaMA-3.2-3B, LoRA/QLoRA, Hybrid RAG, Agent SQL–NoSQL,
  so sánh 4 hướng tiếp cận); thiếu `images/fig_kien_truc.png`, `images/logo.png`.
- T03 push dữ liệu hội thoại `data/KIMOCHI_DATA_TRAINING_CHATBOT.json` (commit `e1722c0`).

## Phần II. Hiện trạng theo rubric (03/11)

| Trụ cột | TT | Ghi chú |
|---|---|---|
| Trang đầu (bìa, phân công, tuyên bố AI, viết tắt) | 🟡 | còn placeholder trường/khoa/GVHD; thiếu mục lục tự động + DM hình/bảng |
| I. Bài toán (15%) | ✅ | Phần 1 đủ, ROI có công thức + độ nhạy |
| Lý thuyết (Phần 2) | ✅ | đã có hình kiến trúc, còn thiếu logo |
| II. EDA & tiền xử lý (20%) | ✅ | Phần 3 + notebook 01–03; còn bổ sung mẫu RAFT/tool-call |
| III. Mô hình (25%) | ⬜ | chưa train |
| IV. Đánh giá / XAI / đạo đức (15%) | ⬜ | |
| V. Prototype (15%) | ⬜ | |
| VI. Trình bày & code (10%) | 🟡 | `code/` 4 notebook chạy sạch; PDF render được |
| Bonus | 🟡 | dữ liệu dựa trên nhà hàng thật; chưa có người dùng thử |

## Phần III. Quyết định

| Ngày | Quyết định | Lý do |
|---|---|---|
| 28/10 | Ma trận chính ở cỡ 3B (Qwen2.5-3B vs LLaMA-3.2-3B) | LoRA 16-bit & QLoRA đều vừa T4 16 GB → so sánh công bằng |
| 28/10 | Thay K-Fold bằng 3 seed + paired bootstrap | K-Fold với LLM tốn K lần chi phí |
| 03/11 | `group_id` = câu trả lời chuẩn hóa của dữ liệu cũ | 6.069 câu hỏi / 2.068 đáp án → tránh rò rỉ diễn đạt lại |
| 03/11 | Kimochi = nhà hàng thật 2 chi nhánh; phần giao dịch là mô phỏng | Khớp Phần 1, giữ cơ hội điểm thưởng |
| 03/11 | Huấn luyện trên dữ liệu v2; giá chỉ qua công cụ `get_menu_item` | 26/74 món có giá mâu thuẫn trong dữ liệu gốc |
| 03/11 | Gọi công cụ dạng văn bản `<tool_call>{json}</tool_call>` thay vì trường `tool_calls` | Không phụ thuộc chat template, Arrow lưu được, Qwen/LLaMA đều sinh được |
