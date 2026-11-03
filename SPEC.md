# SPEC — Chatbot CSKH Kimochi Sushi (Hybrid RAG + Fine-tuning Qwen & LLaMA)

> Nguồn: rubric `docs/huong_dan_tieu_luan_va_rubric_cuoi_khoa.pdf`, kế hoạch `docs/Ke-hoach-12-ngay-Kimochi.docx` (28/10),
> báo cáo v1 `docs/KIMOCHI BÁO CÁO.pdf`, nội dung đã viết trong `report_main.qmd`. Luật làm việc: `CLAUDE.md` §3.
> Cập nhật: 03/11/2026.

## 1. Mục tiêu & câu hỏi nghiên cứu

Trợ lý CSKH tiếng Việt cho fanpage Kimochi Sushi: trả lời menu/giá/dị ứng/khuyến mãi/chi nhánh, đặt & hủy bàn, chuyển nhân viên
khi khiếu nại, từ chối ngoài phạm vi / prompt injection.

- **RQ1** Qwen2.5-3B hay LLaMA-3.2-3B thích nghi tốt hơn với CSKH tiếng Việt?
- **RQ2** QLoRA đánh đổi bao nhiêu chất lượng lấy bao nhiêu VRAM so với LoRA?
- **RQ3** Hybrid RAG hơn BM25 / dense đơn lẻ bao nhiêu (Recall@5, MRR, nDCG@5)? (và FT + RAG có hơn RAG-only?)
- **RQ4** Agent SQL + NoSQL hoàn thành kịch bản đặt bàn nhiều lượt với tỷ lệ bao nhiêu?

KPI (khớp `@tbl-smart` trong báo cáo): deflection ≥ 70% · fallback ≤ 10% (v1: 25,7%) · fact-EM giá/giờ = 100% ·
Faithfulness ≥ 0,85 · Recall@5 ≥ 0,90 · đặt bàn thành công ≥ 90% · p95 latency ≤ 5 s.

## 2. Kiến trúc

```
Khách (Gradio) → Agent (LLM M* đã fine-tune, tool JSON)
                   ├─ sql_tool      → SQLite kimochi.db (giá, chi nhánh, giờ, bàn trống, đặt/hủy bàn — hàm tham số hóa)
                   ├─ rag_tool      → Hybrid RAG: BM25 + bge-m3 (Qdrant) → RRF → reranker → top-k chunk (tài liệu gốc ở NoSQL)
                   └─ handoff_tool  → chuyển nhân viên (khiếu nại, không đủ thông tin)
Offline: data/ → sinh + lọc dữ liệu → SFT LoRA/QLoRA → adapter (HF Hub) → agent nạp
```
Chunking theo cấu trúc: 1 món = 1 chunk (metadata `item_id, category, allergens[], is_vegetarian, spicy_level, price`);
chính sách theo tiêu đề 300–500 token, overlap 50; FAQ 1 cặp = 1 chunk.

## 3. Dữ liệu

### 3.1 Hiện có
`data/KIMOCHI_DATA_TRAINING_CHATBOT.json` — **6.355 mẫu** `{Domain, user_query, responses}`, 1.902 giá trị `Domain`
(dạng `"<Chủ đề>: <Tên món>"`, vd "Giá & Hương vị", "Đồ uống", "Gợi ý bữa ăn", "Món chay"…), 6.069 câu hỏi khác nhau nhưng chỉ
**2.068 câu trả lời khác nhau** (≈ 3 câu hỏi diễn đạt lại / 1 đáp án) → đây chính là `group_id` tự nhiên để chia split chống rò rỉ.
Độ dài TB: hỏi ~41 ký tự, đáp ~175 ký tự. Kế thừa từ v1 (sinh bằng ChatGPT/Gemini rồi duyệt tay, dựa trên thông tin thật nhà hàng).

### 3.2 Cần tạo
| Nguồn | Nội dung | Dùng cho |
|---|---|---|
| `data/kb/menu.csv` → `kimochi.db` | trích từ JSON hiện có + menu thật: món, danh mục, giá, dị ứng, chay, cay | SQL tool, fact-EM, EDA |
| `branches`, `tables`, `reservations`, `promotions` | 2 chi nhánh thật; bàn/đặt bàn **mô phỏng** (ghi rõ) | Agent giao dịch |
| `data/kb/policies/*.md`, `faq.jsonl` | đặt bàn/hủy/cọc, giao hàng, thẻ thành viên, khuyến mãi | RAG |
| Mẫu SFT (`messages` + `meta`) | chuyển đổi 6.355 mẫu + bổ sung RAFT (context + nhiễu, 20% chỉ nhiễu → từ chối), tool-call, out_of_scope | Fine-tune |
| `test_gold` (~150) / `test_adv` (~80) / 30–60 kịch bản nhiều lượt | **viết tay**, đóng băng SHA-256 | Đánh giá chính |

Kết quả trích KB (03/11): 106 món chính tắc, 74 có giá, 32 thiếu giá (Combo 10, Sashimi 9), 1 xung đột giá (Khoai tây chiên 53.900/49.000); dị ứng suy luận bằng từ khóa (`allergen_verified=0`). Intent thực tế trong dữ liệu cũ: menu_info 52%, price 15%, recommendation 15%, allergy 12%, policy 4%, store 2%, reservation 1%, complaint <1%; thiếu hẳn reservation_manage, order_status, chitchat, out_of_scope → phải bổ sung.

Taxonomy 12 intent: `menu_info, price_query, allergy_diet, recommendation, store_info, reservation_create, reservation_manage,
order_status, policy_promo, complaint, chitchat, out_of_scope`. Ánh xạ `Domain` → intent bằng luật từ khóa + duyệt tay.

Split: GroupShuffleSplit theo `group_id` (= câu trả lời chuẩn hóa), stratify intent, ~80/10/10 trên dữ liệu tổng hợp; kiểm cosine
test↔train gần nhất, loại > 0,92. Lọc: MinHash/cosine trùng, fact-check số với SQL, JSON tool-call parse được, độ dài bất thường
(ghi log lý do loại = "xử lý ngoại lai"). "Feature engineering" = `difficulty, noisy_input, needs_tool` + augmentation không dấu/teen code;
chứng minh bằng 1 ablation (có/không mẫu nhiễu RAFT → tỷ lệ bịa trên `test_adv`).

## 4. Thí nghiệm

| Mã | Hệ thống | Vai trò |
|---|---|---|
| B0 / B1 | Base Instruct zero-shot / 5-shot (cả 2 model) | Baseline |
| B2 | Base + Hybrid RAG | Tác động của RAG |
| M1–M4 | {Qwen, LLaMA} × {LoRA, QLoRA} | ≥ 3 mô hình nâng cao |
| M* + RAG (+ Agent) | Hệ thống đề xuất / sản phẩm | Kết luận |
| R1–R5 | Retrieval ablation: BM25, dense, BM25+dense RRF, +reranker, +metadata filter | RQ3 |

Cấu hình chung: all-linear target modules, r ∈ {8,16,32}, α = 2r, dropout 0,05, lr 5e-5–3e-4 cosine warmup 3%, 2–3 epoch early
stop, batch hiệu dụng 16, max_seq 1.024–2.048 (chốt theo EDA token), loss chỉ trên token assistant. QLoRA: NF4 + double quant,
fp16 compute, `paged_adamw_8bit`. Log peak VRAM, phút/epoch, tokens/s.

Thay K-Fold: **3 seed** (hoặc 1 nếu thiếu GPU) mean ± std + **paired bootstrap 1.000 lần** CI 95%. Optuna 8–12 trial (MedianPruner)
trên Qwen QLoRA, áp cho cả 4 ô. Learning curve: loss train/val theo bước + điểm test theo 25/50/100% dữ liệu. Kiểm catastrophic
forgetting: 50 câu kiến thức chung trước/sau.

## 5. Đánh giá, XAI, đạo đức

- Metric: Correctness (LLM-judge 1–5, hiệu chỉnh với 50 mẫu người chấm — báo Spearman/kappa), Faithfulness (RAGAS),
  fact-EM giá/giờ, tỷ lệ từ chối đúng, tool-call hợp lệ %, ROUGE-L/BERTScore (so sánh với v1), p95 latency, VRAM.
- Confusion matrix **bộ định tuyến công cụ** (sql/rag/handoff/none) + bảng chi phí kinh doanh từng loại sai
  (vd trả lời sai dị ứng ≫ chuyển nhân viên thừa).
- XAI cho LLM: trích dẫn nguồn `[doc_id]`, leave-one-out context (bỏ từng chunk → câu trả lời đổi?), attribution của reranker,
  ablation; 10 ca lỗi điển hình.
- Đạo đức: dị ứng (luôn khuyến nghị xác nhận nhân viên), dữ liệu cá nhân khi đặt bàn, prompt injection, thiên lệch câu không dấu/
  vùng miền, HITL (handoff).

## 6. Prototype & triển khai

Gradio trên HF Spaces (GPU nếu có; nếu không: QLoRA 4-bit/CPU hoặc chạy local + quay video 2–3 phút). Hiển thị: câu trả lời,
nguồn trích dẫn, câu SQL đã chạy, vết công cụ; công tắc what-if (đổi model, bật/tắt RAG, sửa giá món rồi hỏi lại).
Phần 6 báo cáo: tích hợp fanpage (Messenger webhook), giám sát drift (tỷ lệ fallback/handoff theo tuần, câu hỏi mới ngoài phân
phối), hạn chế & lộ trình. Bonus: 5–10 người dùng thật thử + phiếu chấm (SUS / 1–5).

## 7. Backlog task & kế hoạch (tái lập 03/11 — còn 9 ngày)

Trạng thái: ✅ xong · 🟡 đang làm / một phần · ⬜ chưa làm. **Ưu tiên P0 = bắt buộc để nộp được; P1 = ăn điểm; P2 = cắt được.**

| Mã | Ngày | Việc | Ưu tiên | Xong khi | TT |
|---|---|---|---|---|---|
| T00 | 28/10 | Kế hoạch 12 ngày, khung `.qmd` + trang bìa | P0 | `docs/Ke-hoach…docx`, YAML/titlepage | ✅ |
| T01 | – | Phần 1 báo cáo (pain, 3 persona, empathy map, RQ, SMART, ROI + độ nhạy) | P0 | §1 trong `.qmd` | ✅ |
| T02 | – | Phần 2 báo cáo (CRISP-DM, Transformer, Qwen/LLaMA, LoRA/QLoRA, Hybrid RAG, Agent, so sánh hướng tiếp cận) | P0 | §2 trong `.qmd` | ✅ (đã có hình kiến trúc; còn thiếu logo) |
| T03 | – | Thu thập dữ liệu hội thoại (6.355 mẫu) | P0 | `data/*.json` | ✅ |
| T04 | 03/11 | Dựng repo: `code/`, `results/`, `images/`, `requirements.txt`, `.gitignore`, `.env.example`; render thử PDF | P0 | PDF render được | ✅ 03/11 |
| T05 | 03/11 | Trích KB: menu/giá/dị ứng → `kb/menu.csv`, `kimochi.db` (SQLite schema + seed), policies/FAQ; data dictionary | P0 | `01_build_kb.ipynb` chạy; bảng data dictionary | ✅ 03/11 (106 món, 74 có giá) |
| T06 | 03–04/11 | Viết tay & đóng băng `test_gold` (150) + `test_adv` (80) + 30 kịch bản đặt bàn; SHA-256 | P0 | `data/splits/CHECKSUMS.txt` | 🟡 công cụ `code/freeze_tests.py` xong — **nhóm phải tự viết** |
| T07 | 04/11 | Chuyển đổi 6.355 mẫu → `messages`+`meta`, gán intent, bổ sung RAFT/tool-call/từ chối (LLM thầy), lọc, split theo group | P0 | `02_prepare_sft.ipynb`, train/val/test_synth.jsonl, phễu lọc | ✅ (thay bằng T07b cho dữ liệu huấn luyện; split v1 giữ làm tập test) |
| T07b | 03/11 | Bộ dữ liệu SFT v2: làm sạch, giá qua công cụ SQL, RAFT, mẫu khuôn (đặt bàn nhiều lượt, từ chối, khiếu nại), 3 system prompt, dataset card | P0 | `data/splits_v2/` + README, `05_prepare_sft_v2.ipynb` | ✅ 03/11 (4.980 / 580 hội thoại) |
| T08 | 04–05/11 | EDA 6–8 biểu đồ + nhận xét kinh doanh; viết **Phần 3** | P0 | `03_eda.ipynb`, §3 `.qmd` | ✅ 03/11 (7 hình; token-length chạy trên Colab) |
| T09 | 05/11 | Hybrid RAG + ablation R1–R5 (Recall@5, MRR, nDCG@5) | P0 | `04_rag.ipynb`, `results/retrieval_test_synth.csv` | 🟡 sparse xong (BM25+lọc R@5 0,951); dense/reranker chạy trên Colab |
| T10 | 05/11 | Baseline B0/B1/B2 cả 2 model trên test_gold | P0 | `results/baseline.csv` | 🟡 notebook `code/kaggle/K2` sẵn sàng — chờ chạy GPU |
| T11 | 05–06/11 | Optuna 8 trial Qwen QLoRA; train M1–M4 (1 seed bắt buộc, 3 seed nếu kịp); learning curve | P0 | 4 adapter trên HF Hub, `results/train_matrix.csv`, hình loss | 🟡 notebook K3, K4 sẵn sàng — chờ chạy GPU |
| T12 | 07/11 | Agent (sql/rag/handoff) gắn M* — `tools.py` + `agent.py` đã có (7 công cụ SQLite, vòng lặp tool_call); chạy kịch bản; confusion matrix công cụ; viết **Phần 4** | P0 | `06_agent.ipynb`, §4 `.qmd` | 🟡 `agent_eval.py` + `K6_agent_eval.ipynb` xong (kiểm thử CPU); chờ chạy GPU + `scenarios.jsonl` viết tay |
| T13 | 07–08/11 | Đánh giá toàn diện (judge hiệu chỉnh, RAGAS, fact-EM, bootstrap), XAI, 10 ca lỗi, đạo đức; viết **Phần 5** | P0 | `07_eval.ipynb`, §5 `.qmd` | 🟡 notebook K5 + khung Phần 5 trong `.qmd` (bảng tự điền từ `results/`); chờ GPU |
| T14 | 08/11 | Prototype Gradio + deploy HF Space / video 2–3 phút; viết **Phần 6** | P0 | link demo trong `demo/README.md` | 🟡 `demo/app.py` (Gradio, What-if giá) + K7 + kịch bản video; chờ adapter để quay video / Space |
| T15 | 08–09/11 | Người dùng thật thử (5–10 người) + phiếu chấm | P1 (bonus) | bảng kết quả trong §6 | ⬜ |
| T16 | 09/11 | Hoàn thiện: mục lục tự động, DM hình/bảng, placeholder bìa, bảng phân công, tuyên bố AI, TLTK đủ, ≤ 45 trang | P0 | PDF hoàn chỉnh | 🟡 Phần 4–6 viết xong khung (34 trang); còn điền trang bìa, phân công, số liệu GPU |
| T17 | 09/11 | Restart & Run All mọi notebook; đóng gói zip | P0 | zip đúng tên | 🟡 `code/make_submission.py` (kiểm tra rubric + zip) xong; chạy lần cuối 09/11 |
| T18 | 10–11/11 | Đệm: soát rubric, Turnitin, nộp LMS trước 23:59 11/11 | P0 | Đã nộp | ⬜ |

**Thứ tự cắt khi trễ** (không cắt test_gold và hiệu chỉnh judge): 3 seed → 1 seed · Optuna 8 → 4 trial · bỏ text-to-SQL tự do
(chỉ hàm tham số hóa) · MongoDB → JSON · M1–M4 → chỉ QLoRA cho cả 2 model + LoRA cho model tốt hơn · deploy Space → video.

## 8. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Trễ 4 ngày so với kế hoạch gốc (D2–D5 chưa có đầu ra) | Kế hoạch §7 nén; chạy GPU song song với viết báo cáo |
| Hết quota GPU Colab/Kaggle | Dùng Kaggle (30 h/tuần) + Colab; max_seq 1.024; Unsloth |
| LLaMA cần chấp nhận license HF | Xin quyền ngay T04 |
| Dữ liệu cũ trùng lặp cao (3 hỏi/1 đáp) → rò rỉ | Split theo group = câu trả lời |
| Rubric viết cho ML cổ điển | Bảng "dịch" yêu cầu → LLM (kế hoạch 28/10 §1) đưa vào báo cáo |
