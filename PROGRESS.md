# PROGRESS — Nhật ký dự án Chatbot Kimochi (mới nhất lên đầu)

## Phần I. Nhật ký

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
| Lý thuyết (Phần 2) | ✅ | thiếu 2 hình |
| II. EDA & tiền xử lý (20%) | ⬜ | có dữ liệu thô, chưa có notebook / Phần 3 |
| III. Mô hình (25%) | ⬜ | chưa train |
| IV. Đánh giá / XAI / đạo đức (15%) | ⬜ | |
| V. Prototype (15%) | ⬜ | |
| VI. Trình bày & code (10%) | 🟡 | chưa có `code/`; PDF chưa render thử |
| Bonus | 🟡 | dữ liệu dựa trên nhà hàng thật; chưa có người dùng thử |

## Phần III. Quyết định

| Ngày | Quyết định | Lý do |
|---|---|---|
| 28/10 | Ma trận chính ở cỡ 3B (Qwen2.5-3B vs LLaMA-3.2-3B) | LoRA 16-bit & QLoRA đều vừa T4 16 GB → so sánh công bằng |
| 28/10 | Thay K-Fold bằng 3 seed + paired bootstrap | K-Fold với LLM tốn K lần chi phí |
| 03/11 | `group_id` = câu trả lời chuẩn hóa của dữ liệu cũ | 6.069 câu hỏi / 2.068 đáp án → tránh rò rỉ diễn đạt lại |
| 03/11 | Kimochi = nhà hàng thật 2 chi nhánh; phần giao dịch là mô phỏng | Khớp Phần 1, giữ cơ hội điểm thưởng |
