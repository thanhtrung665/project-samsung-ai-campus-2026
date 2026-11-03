# Kimochi CSKH — bộ dữ liệu SFT v2

Sinh bởi `code/src/kimochi/prepare_sft_v2.py` (seed 42). Dùng cho tinh chỉnh LoRA / QLoRA Qwen2.5-3B và LLaMA-3.2-3B.

## Định dạng
JSONL, mỗi dòng: `id`, `messages` (system / user / assistant / tool — chuẩn chat của Hugging Face, áp `apply_chat_template` lúc
huấn luyện), `meta` (không đưa vào huấn luyện). Gọi công cụ là văn bản `<tool_call>{"name", "arguments"}</tool_call>`; kết quả
công cụ ở vai trò `tool`. Khi huấn luyện, mỗi lượt assistant thành một mẫu prompt/completion → loss chỉ tính trên lượt đó.

## Nguồn và xử lý
- `kimochi_v1`: 3991 mẫu hỏi–đáp gốc của nhà hàng (đã bỏ trùng, bỏ câu hỏi có đáp án mâu thuẫn, xóa phần giá
  trong câu trả lời không hỏi giá).
- `kimochi_v1+db`: 235 câu hỏi giá chuyển sang gọi `get_menu_item`; giá trong câu trả lời thay
  bằng giá trong `kimochi.db` (món chưa có giá → từ chối có căn cứ).
- `kimochi_v1+raft`: 1077 mẫu RAFT (1 chunk đúng + 2 chunk nhiễu; 20% chỉ có nhiễu → từ chối).
- `template_v2`: 257 mẫu theo khuôn: đặt bàn nhiều lượt (chạy công cụ thật trên bản sao DB, có bước
  xác nhận trước khi ghi), tra chi nhánh, tìm món theo điều kiện, khiếu nại → chuyển nhân viên, ngoài phạm vi / prompt injection,
  chào hỏi. Do người viết khuôn, không qua LLM.

Nhật ký loại bỏ: {"drop_exact_duplicate": 60, "drop_conflicting_question": 239, "price_stripped": 839, "drop_price_inconsistent": 7, "drop_price_unstrippable": 33}

## Phân bố
Theo định dạng:

|                         |   train |   val |
|:------------------------|--------:|------:|
| chitchat                |      16 |     1 |
| closed_book             |    3587 |   404 |
| handoff                 |      13 |     1 |
| oos_code                |       6 |     0 |
| oos_competitor          |       6 |     0 |
| oos_general             |       7 |     1 |
| oos_homework            |       5 |     1 |
| oos_injection           |       9 |     3 |
| oos_medical             |       4 |     0 |
| raft_neg                |     213 |    28 |
| raft_pos                |     748 |    88 |
| reservation_mt          |     127 |    19 |
| reservation_unavailable |      12 |     2 |
| tool_branch             |       6 |     4 |
| tool_no_price           |       3 |     2 |
| tool_price              |     207 |    23 |
| tool_search             |      11 |     3 |

Theo intent:

|                    |   train |   val |
|:-------------------|--------:|------:|
| allergy_diet       |     576 |    65 |
| chitchat           |      16 |     1 |
| complaint          |      31 |     4 |
| menu_info          |    2858 |   326 |
| out_of_scope       |      37 |     5 |
| policy_promo       |     177 |    16 |
| price_query        |     342 |    45 |
| recommendation     |     677 |    78 |
| reservation_create |     168 |    25 |
| store_info         |      98 |    15 |

## Chia tập
Kế thừa split của v1 theo `group_id` (StratifiedGroupKFold, kiểm rò rỉ cosine > 0,92). Mẫu khuôn chia 90/10 train/val.
**Tập kiểm tra không đổi:** `data/splits/test_synth.jsonl` (v1) và bộ viết tay `test_gold` / `test_adv` — để so sánh v1 ↔ v2.

## Hạn chế
- Câu hỏi và câu trả lời gốc do LLM sinh rồi người duyệt (phiên bản trước); văn phong lặp nhiều.
- Dị ứng suy luận bằng từ khóa, chưa được nhà hàng xác nhận. Bàn / đặt bàn là dữ liệu mô phỏng.
- Giá: 26/74 món có giá mâu thuẫn trong dữ liệu gốc (16 do VAT 10%); v2 chuẩn hóa theo giá trong DB — cần nhà hàng xác nhận.
- Mẫu khuôn có độ đa dạng ngôn ngữ thấp hơn hội thoại thật.
