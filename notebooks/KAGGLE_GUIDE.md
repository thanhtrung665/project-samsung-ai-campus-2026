# Hướng dẫn chạy NB2 trên Kaggle — từng bước

Repo: `thanhtrung665/project-samsung-ai-campus-2026`, nhánh `claude/practical-volta-8tfmy8`.
Thời gian GPU trong hướng dẫn là **ước lượng**; Kaggle cho ~30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ.

## Bước 0 — Chuẩn bị tài khoản (một lần, ~15 phút)
1. **Kaggle:** đăng nhập kaggle.com → *Settings* → **Phone verification** (bắt buộc để dùng GPU và Internet trong notebook).
2. **Hugging Face:** huggingface.co → *Settings → Access Tokens* → *Create new token* (loại **Read**; chọn **Write** nếu muốn đẩy
   adapter / mô hình lên Hub). Sao chép token `hf_...`.
3. **LLaMA:** mở https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct → điền form chấp nhận license (thường duyệt trong vài
   giờ). Chưa được duyệt thì notebook tự dùng bản sao `unsloth/Llama-3.2-3B-Instruct` — ghi rõ điều này trong báo cáo.
4. *(Tùy chọn)* **Gemini API key** cho bước LLM-as-judge: https://aistudio.google.com → *Get API key*.

## Bước 1 — Tải repo về máy
1. Mở https://github.com/thanhtrung665/project-samsung-ai-campus-2026/tree/claude/practical-volta-8tfmy8
2. Nút xanh **Code → Download ZIP** (đúng nhánh `claude/practical-volta-8tfmy8`).
3. Giải nén, kiểm tra có: `code/src/kimochi/`, `data/MANIFEST.json`, `data/splits_v2/`, `notebooks/NB2_finetune_rag_mvp.ipynb`.
4. *(Khuyến nghị)* Nếu nhóm đã viết bộ test tay (`data/splits/test_gold.jsonl`, `test_adv.jsonl`, `scenarios.jsonl`): chép vào
   `data/splits/`, chạy `python code/freeze_tests.py` rồi chạy lại NB1 để cập nhật `data/MANIFEST.json` **trước** khi tải lên.
5. Xóa thư mục `docs/` cho nhẹ (không cần trên Kaggle) → nén lại thành `kimochi-repo.zip`.

## Bước 2 — Tạo Kaggle Dataset chứa repo
1. kaggle.com → **Datasets → + New Dataset**.
2. Kéo thả `kimochi-repo.zip` (Kaggle tự giải nén) · Title: `kimochi-repo` · **Private** → **Create**.
3. Chờ trạng thái xử lý xong. Mỗi khi repo thay đổi (vd thêm test tay): mở dataset → **New Version** → tải zip mới.

## Bước 3 — Tạo notebook từ NB2
1. kaggle.com → **Code → + New Notebook**.
2. Menu **File → Import Notebook** → chọn `notebooks/NB2_finetune_rag_mvp.ipynb` từ thư mục đã giải nén → *Import*.
3. Panel bên phải (**Notebook / Session options**):
   - **Accelerator: GPU T4 x2**
   - **Internet: On**
   - **Persistence:** *Files only* (giữ `/kaggle/working` giữa các lần chạy tương tác).
4. **Add Input** (panel phải) → *Your Work / Datasets* → chọn **kimochi-repo** → *Add*.
5. Menu **Add-ons → Secrets → Add Secret**:
   - Label `HF_TOKEN`, Value = token Hugging Face → *Save*, **tích ô gắn vào notebook này**.
   - *(Tùy chọn)* `GEMINI_API_KEY`.

## Bước 4 — Chạy thử tương tác (~10 phút, kiểm tra môi trường)
Chạy **từng ô** (Shift + Enter) theo thứ tự:
1. Ô cài thư viện (`%%capture !pip install ...`) — 2–4 phút, không in gì là bình thường.
2. Ô môi trường → phải in `ROOT = /kaggle/working/repo` và `GPU: Tesla T4 (15 GB), Tesla T4 (15 GB)`.
3. Ô **§0 Cấu hình** → in `Ô đã có: []`.
4. Ô **§1 Nhận dữ liệu** → phải in `✓ Dữ liệu khớp manifest` và số mẫu huấn luyện (~5.843).

| Lỗi thường gặp | Cách xử lý |
|---|---|
| `GPU: CPU` | Chưa chọn Accelerator GPU / chưa xác minh số điện thoại |
| `IndexError` ở ô môi trường | Chưa Add Input dataset `kimochi-repo` |
| `Dữ liệu lệch manifest` | Repo tải lên khác bản NB1 đã chạy → chạy lại NB1 trên máy, tạo version dataset mới |
| Lỗi tải mô hình / `401` | Thiếu secret `HF_TOKEN` hoặc chưa tích gắn secret vào notebook |
| `CUDA out of memory` | *Run → Restart & clear outputs*, chạy lại; hoặc giảm `MAX_LEN` xuống 768 |

Tất cả OK → **dừng phiên tương tác** (nút ⏻ *Stop session*) để không tốn giờ GPU.

## Bước 5 — Chạy thật ở chế độ nền (khuyến nghị chia 2 phiên)
NB2 đầy đủ (2 mô hình × 2 kỹ thuật + đánh giá + RAG + xuất mô hình) có thể vượt 12 giờ. Chia:

**Phiên 1 — Qwen.** Sửa ô §0 Cấu hình:
```python
MODELS = ['qwen']
N_EVAL = None            # để 150 nếu muốn chạy nhanh trước
RUN_MVP_DEMO = False     # BẮT BUỘC False khi chạy nền
```
→ **Save Version** (góc trên phải) → **Save & Run All (Commit)** → *Save*. Kaggle chạy nền, có thể đóng trình duyệt.
Theo dõi: trang notebook → tab **Versions** / **Logs**.

**Phiên 2 — LLaMA + so sánh + xuất mô hình.** Khi phiên 1 xong:
1. Trong notebook → **Add Input → Your Work → Notebooks** → chọn chính notebook này (output của version 1) → *Add*.
   Notebook tự nạp lại adapter đã huấn luyện (`runs/`) và kết quả đã có (`results/`), bước nào xong thì bỏ qua.
2. Sửa `MODELS = ['qwen', 'llama']` → **Save & Run All (Commit)** lần nữa.

*(Thiếu giờ GPU?)* Cắt theo SPEC §7: giữ `METHODS` chỉ `QLoRA 4-bit` cho mô hình kém hơn, `RUN_BASELINE = False` ở phiên 2.

## Bước 6 — Lấy kết quả về repo
1. Mở version đã chạy xong → tab **Output**.
2. Tải thư mục `repo/results/` (hoặc *Download All*) → chép **đè** vào `results/` của repo trên máy.
3. Trên máy: `quarto render report_main.qmd --to pdf` → các ô TBD trong Phần 4–5 tự có số.
4. Tải `kimochi-mvp-bundle.zip` (gói MVP) nếu cần; **không commit** mô hình / adapter vào git (dùng HF Hub: đặt `HF_USER` ở §0).
5. Commit `results/` + PDF mới, push.

## Bước 7 — Demo MVP & quay video
1. Mở notebook ở chế độ **tương tác** (Edit), giữ Input là output của phiên 2.
2. Ở §0 đặt `RUN_MVP_DEMO = True`, chạy lần lượt các ô tới hết §9 (các bước đã xong tự bỏ qua, chỉ mất vài phút).
3. Ô §10 in link `https://xxxx.gradio.live` → mở trên trình duyệt → quay màn hình theo kịch bản 7 bước trong `demo/README.md`.
4. Xong → dừng ô §10 (■) và **Stop session**. Lưu video vào `demo/` (hoặc link YouTube/Drive trong `demo/README.md`).
5. *(Bonus T15)* Gửi link gradio.live (hiệu lực 72 giờ, trong lúc phiên còn chạy) cho 5–10 người dùng thử + phiếu chấm.
