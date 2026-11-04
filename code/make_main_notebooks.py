"""Sinh 2 notebook tổng hợp trong notebooks/:
  NB1_data_pipeline.ipynb     (CPU)  nạp dữ liệu → kho tri thức → chất lượng → nhãn & đặc trưng → split chống rò rỉ → EDA →
                                     chuẩn hóa dữ liệu v2 → kiểm tra sẵn sàng huấn luyện / đánh giá → manifest
  NB2_finetune_rag_mvp.ipynb  (GPU)  nhận dữ liệu → thiết lập → fine-tune LoRA 16-bit / QLoRA 4-bit (+8-bit tùy chọn) → đánh giá test →
                                     trực quan + phân tích → demo nhanh → Hybrid-RAG → đánh giá hệ thống, so sánh, trực quan →
                                     xuất mô hình hoàn chỉnh → demo MVP
Logic nằm trong code/src/kimochi/*; notebook chỉ điều phối + giải thích.
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


# ================================================================================================ NB1
LOCAL_SETUP = r'''
import sys, os, json, pathlib
import pandas as pd, numpy as np
from IPython.display import Image, display, Markdown
# tìm thư mục gốc repo (chứa code/src/kimochi) — chạy được cả local lẫn Kaggle / Colab
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / 'code/src/kimochi').exists())
os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'code/src'))
pd.set_option('display.max_colwidth', 90); pd.set_option('display.width', 160)
print('ROOT =', ROOT)
'''

NB1 = [
    md(r'''
# NB1 · Dữ liệu: nạp → tiền xử lý → EDA → chuẩn hóa → sẵn sàng huấn luyện & đánh giá
**Chatbot CSKH Kimochi Sushi** — Hybrid-RAG + Fine-tuning Qwen2.5-3B / LLaMA-3.2-3B (LoRA / QLoRA)

Chạy trên **CPU** (~2 phút), tái lập hoàn toàn (seed 42, cùng md5 mỗi lần chạy). Đầu ra dùng cho `NB2_finetune_rag_mvp.ipynb`.

| § | Nội dung | Đầu ra |
|---|---|---|
| 1 | Nạp & khảo sát dữ liệu gốc | — |
| 2 | Kho tri thức: SQLite + tài liệu RAG (chunk ≤ 600 ký tự) | `data/kimochi.db`, `data/kb/` |
| 3 | Chất lượng dữ liệu: thiếu giá, giá mâu thuẫn (VAT) | `results/kb_quality.json` |
| 4 | Gán nhãn intent & tạo đặc trưng | — |
| 5 | Chia train / val / test theo nhóm + kiểm rò rỉ | `data/splits/` (tập test chính thức) |
| 6 | EDA & phát hiện kinh doanh | `images/eda_*.png` |
| 7 | Chuẩn hóa → bộ dữ liệu huấn luyện v2 (công cụ SQL, RAFT, hội thoại nhiều lượt, từ chối) | `data/splits_v2/` |
| 8 | Kiểm tra sẵn sàng huấn luyện & đánh giá | — |
| 9 | Manifest md5 (NB2 kiểm tra lại) | `data/MANIFEST.json` |
'''),
    code(LOCAL_SETUP),

    md(r'''
## 1. Nạp & khảo sát dữ liệu gốc
`data/KIMOCHI_DATA_TRAINING_CHATBOT.json` — bộ hỏi–đáp kế thừa phiên bản trước, dựa trên thông tin thật của nhà hàng.
'''),
    code(r'''
raw = pd.read_json('data/KIMOCHI_DATA_TRAINING_CHATBOT.json')
print(raw.shape); display(raw.head(5))
print('Giá trị thiếu:', raw.isna().sum().to_dict())
print('Domain khác nhau:', raw.Domain.nunique(), '| câu hỏi khác nhau:', raw.user_query.nunique(),
      '| câu trả lời khác nhau:', raw.responses.nunique())
print('Cặp (hỏi, đáp) trùng nguyên văn:', raw.duplicated(['user_query', 'responses']).sum())
'''),
    md(r'''
**Nhận xét.** Không có giá trị thiếu, nhưng ~6.000 câu hỏi chỉ dùng ~2.000 câu trả lời → mỗi đáp án có ~3 cách hỏi (diễn đạt lại).
Đây là nguy cơ **rò rỉ dữ liệu** nếu chia train/test ngẫu nhiên theo dòng → §5 chia theo nhóm câu trả lời.
'''),

    md(r'''
## 2. Kho tri thức: cơ sở dữ liệu SQL + tài liệu RAG
Trích **tên món chính tắc**, giá, danh mục, nhóm dị ứng (suy luận từ khóa, gắn cờ *chưa xác nhận*) → SQLite 6 bảng;
mỗi món / chủ đề → một tài liệu, cắt thành **chunk ≤ 600 ký tự** (đơn vị truy xuất dùng chung cho huấn luyện RAFT và lúc chạy).
Bàn / đặt bàn là dữ liệu **mô phỏng** (`simulated = 1`).
'''),
    code(r'''
from kimochi import build_kb
kbq = build_kb.main()
menu = pd.read_csv('data/kb/menu.csv')
print({k: kbq[k] for k in ['raw_rows', 'rows_mapped_to_item', 'menu_items', 'items_with_price', 'chunks', 'docs_menu', 'docs_policy']})
display(menu.drop(columns='description').head(8))
'''),
    code(r'''
import sqlite3
con = sqlite3.connect('data/kimochi.db')
dd = pd.DataFrame([{'bảng': t, 'số dòng': con.execute(f'select count(*) from {t}').fetchone()[0],
                    'cột': ', '.join(r[1] for r in con.execute(f'pragma table_info({t})'))}
                   for t in ['categories', 'menu_items', 'branches', 'tables', 'reservations', 'promotions']])
display(dd)   # từ điển dữ liệu của cơ sở dữ liệu
chunks = pd.read_json('data/kb/chunks.jsonl', lines=True)
print('Chunk:', len(chunks), '| độ dài ký tự:', chunks.text.str.len().describe().round(0).to_dict())
'''),

    md(r'''
## 3. Chất lượng dữ liệu — thiếu giá và giá mâu thuẫn
Đối chiếu giá trong câu trả lời hỏi giá với giá nhắc kèm tên món ở **mọi** câu trả lời khác.
'''),
    code(r'''
miss = menu[menu.price_vnd.isna()].groupby('category').size().rename('món thiếu giá')
display(miss.to_frame())
xc = kbq['price_mismatch_detail']
vat = lambda db, p: abs(int(p) - round(db * 1.1, -2)) <= 200
tab = pd.DataFrame([{'món': k, 'giá DB': v['db'], 'giá khác': ', '.join(f'{int(p):,} (×{c})' for p, c in v['other_mentions'].items()),
                     'giải thích bằng VAT 10%': all(vat(v['db'], p) for p in v['other_mentions'])} for k, v in xc.items()])
print(f"{len(tab)}/{kbq['items_with_price']} món có giá mâu thuẫn; {tab['giải thích bằng VAT 10%'].sum()} món chênh đúng VAT 10%")
display(tab.head(12))
'''),
    md(r'''
**Nhận xét.** Dữ liệu trộn giá trước và sau VAT, cộng thêm một số mâu thuẫn không giải thích được. Nếu để mô hình *học thuộc*
giá từ hội thoại, nó sẽ học cả các con số mâu thuẫn → **nguyên tắc thiết kế: giá chỉ đọc qua công cụ SQL** (§7).
Giá thiếu được giữ NULL (không điền ước lượng) — điền sai giá gây thiệt hại trực tiếp.
'''),

    md(r'''
## 4. Gán nhãn intent & tạo đặc trưng
Intent theo luật từ khóa có thứ tự ưu tiên (intent giá chỉ xét câu hỏi, vì `Domain` kiểu "Giá & Hương vị: X" chứa cả câu hỏi
không về giá). Đặc trưng tạo mới: `group_id` (băm câu trả lời), `noisy_input` (không dấu / teen code), `needs_tool`, `difficulty`.
'''),
    code(r'''
from kimochi import prepare_sft
fr = prepare_sft.build_frame()
display(fr[['user_query', 'intent', 'group_id', 'noisy_input', 'unaccented', 'needs_tool', 'difficulty']].sample(8, random_state=1))
display(fr.intent.value_counts().rename('số mẫu').to_frame().assign(tỷ_lệ=lambda d: (d['số mẫu'] / len(fr)).round(3)))
'''),

    md(r'''
## 5. Chia train / val / test theo nhóm + kiểm tra rò rỉ
`StratifiedGroupKFold` (10 fold, seed 42): 1 fold test, 1 fold val, 8 fold train — chia **theo `group_id`**, phân tầng theo intent.
Sau đó so cosine (TF-IDF n-gram ký tự, văn bản bỏ dấu) giữa mỗi câu val/test và câu train gần nhất; loại cặp > 0,92.
**Tập test này (cùng bộ test viết tay `test_gold` / `test_adv`) là tập đánh giá chính thức cho mọi mô hình ở NB2.**
'''),
    code(r'''
st = prepare_sft.main()
print('Kích thước:', st['split_sizes'])
print('Số nhóm giao nhau train ↔ test:', st['group_overlap_train_test'])
print('Rò rỉ:', json.dumps(st['leakage'], indent=1))
display(pd.DataFrame(st['intent_by_split']).fillna(0).astype(int))
'''),

    md(r'''
## 6. EDA & phát hiện kinh doanh
Mỗi hình đi kèm nhận xét; số liệu đọc từ `results/eda_stats.json`.
'''),
    code(r'''
from kimochi import eda
E = eda.run_all()
def show(name, note):
    display(Image(f'images/{name}.png', width=820)); display(Markdown(note))
s = E['intent']['share']
show('eda_intent_split', f"**Intent lệch mạnh:** menu_info {s['menu_info']:.0%}, recommendation {s['recommendation']:.0%}, "
     f"allergy {s['allergy_diet']:.1%}, price {s['price_query']:.0%}; đặt bàn chỉ {s['reservation_create']:.1%}. Dữ liệu cũ thiên về "
     f"giới thiệu thực đơn, thiếu tác vụ giao dịch → bổ sung ở §7. Tỷ trọng intent giữa các tập lệch tối đa "
     f"{E['intent']['max_abs_share_gap_train_test']*100:.1f} điểm % (phân tầng tốt).")
L = E['lengths']
show('eda_lengths', f"Câu hỏi TB {L['q_mean']:.1f} từ (p95 {L['q_p95']:.0f}), câu trả lời TB {L['a_mean']:.1f} từ (max {L['a_max']}); "
     f"lệch phải (skew {L['q_skew']:.2f} / {L['a_skew']:.2f}). Mẫu đơn lượt ngắn → `max_length` 1.024 token là đủ kể cả khi thêm ngữ cảnh RAG.")
show('eda_noise', f"**{E['noise']['noisy_rate']:.0%} câu hỏi có nhiễu, {E['noise']['unaccented_rate']:.0%} hoàn toàn không dấu** → truy xuất "
     "phải chạy trên văn bản bỏ dấu; đánh giá riêng nhóm câu nhiễu (công bằng thuật toán).")
show('eda_group_size', f"{E['groups']['groups']:,} nhóm; {E['groups']['share_groups_ge3']:.1%} đáp án dùng chung ≥ 3 câu hỏi → bắt buộc chia theo nhóm.")
show('eda_leakage', f"Sau khi loại, cosine val/test → train TB {E['leakage']['kept_cos_mean']:.3f}, max {E['leakage']['kept_cos_max']:.3f} (< 0,92).")
show('eda_funnel', "Phễu dữ liệu: mẫu gốc → gán được món → nhóm đáp án → giữ sau kiểm rò rỉ.")
M = E['menu']
show('eda_menu', f"Giá trung vị {M['price_median']:,.0f} đ; {M['items'] - M['priced']}/{M['items']} món thiếu giá "
     f"({M['missing_price_by_cat']}). Dị ứng phổ biến: cá {M['allergen_share']['cá']:.0%}, trứng {M['allergen_share']['trứng']:.0%}, "
     f"nhuyễn thể {M['allergen_share']['nhuyễn thể']:.0%} — rủi ro tập trung ở Maki và Combo.")
'''),

    md(r'''
## 7. Chuẩn hóa → bộ dữ liệu huấn luyện v2
Nguyên tắc: **fine-tuning dạy hành vi, cơ sở dữ liệu cung cấp sự thật.**
1. Làm sạch: bỏ cặp trùng, bỏ câu hỏi có đáp án mâu thuẫn.
2. Câu hỏi giá → **gọi công cụ** `get_menu_item` (kết quả lấy từ SQLite thật); món chưa có giá → từ chối có căn cứ; xóa giá trong
   câu trả lời không hỏi giá; loại dòng giá mâu thuẫn ngoài quy luật VAT.
3. **RAFT**: câu hỏi + 1 chunk đúng + 2 chunk nhiễu khó (BM25); 20% chỉ có chunk nhiễu → đáp án từ chối.
4. Mẫu theo khuôn (chạy công cụ thật trên bản sao DB): đặt bàn nhiều lượt có **xác nhận trước khi ghi**, chi nhánh, tìm món,
   khiếu nại → chuyển nhân viên, ngoài phạm vi / prompt injection → từ chối, chào hỏi.
5. 3 biến thể system prompt. Định dạng chuẩn `messages` (system / user / assistant / tool) + `meta`.
'''),
    code(r'''
from kimochi import prepare_sft_v2
v2 = prepare_sft_v2.main()
print('Kích thước:', v2['split_sizes'], '| hội thoại nhiều lượt:', v2['multi_turn'])
print('Nhật ký làm sạch:', v2['cleaning_log']); print('Nguồn:', v2['source'])
fm = pd.DataFrame(v2['format_by_split']).fillna(0).astype(int); fm['tổng'] = fm.sum(axis=1)
display(fm.sort_values('tổng', ascending=False))
'''),
    code(r'''
import matplotlib.pyplot as plt
group = {'closed_book': 'Hỏi–đáp (đã làm sạch)', 'raft_pos': 'RAFT – có tài liệu đúng', 'raft_neg': 'RAFT – chỉ nhiễu → từ chối',
         'tool_price': 'Gọi công cụ giá', 'tool_no_price': 'Gọi công cụ giá', 'reservation_mt': 'Đặt bàn nhiều lượt',
         'reservation_unavailable': 'Đặt bàn nhiều lượt', 'tool_branch': 'Công cụ khác', 'tool_search': 'Công cụ khác',
         'handoff': 'Khiếu nại → chuyển NV', 'chitchat': 'Chào hỏi'}
agg = fm['tổng'].groupby(lambda k: group.get(k, 'Ngoài phạm vi / injection')).sum().sort_values()
fig, ax = plt.subplots(figsize=(7, 3.4)); agg.plot.barh(ax=ax, color='#2563eb')
for i, v in enumerate(agg): ax.text(v, i, f' {v}', va='center', fontsize=8)
ax.set_title('Thành phần bộ dữ liệu huấn luyện v2 (train + val)'); fig.tight_layout(); fig.savefig('images/v2_composition.png', dpi=200)
'''),
    code(r'''
v2_train = pd.read_json('data/splits_v2/train.jsonl', lines=True)
for fmt in ['tool_price', 'raft_neg', 'reservation_mt']:
    r = v2_train[v2_train.meta.map(lambda m: m['format']) == fmt].iloc[0]
    print('=' * 20, fmt)
    for m in r.messages[1:]:
        print(f"  [{m['role']:9}] {m['content'][:180]}")
'''),

    md(r'''
## 8. Kiểm tra sẵn sàng huấn luyện & đánh giá
- **Huấn luyện:** mỗi lượt assistant → 1 mẫu *prompt / completion* (loss chỉ trên completion). Kiểm tra số mẫu và cấu trúc.
- **Đánh giá:** tập test v1 (giữ nguyên để so sánh công bằng) + bộ viết tay nếu đã khóa; bộ truy xuất chạy được.
'''),
    code(r'''
from kimochi import llm, prompts, retriever
ds2, ds1 = llm.sft_dataset(ROOT, 'train'), llm.sft_dataset(ROOT, 'train', data_dir='splits')
dv2 = llm.sft_dataset(ROOT, 'val')
print(f'Mẫu huấn luyện v2: {len(ds2):,} (val {len(dv2):,}) | v1: {len(ds1):,}')
x = next(r for r in ds2 if len(r['prompt']) > 3)
print('Ví dụ prompt nhiều lượt — vai trò:', [m['role'] for m in x['prompt']], '→ completion:', x['completion'][0]['content'][:80])
assert all(r['completion'][0]['role'] == 'assistant' for r in ds2.select(range(500)))
tests = {n: len(prompts.load_eval_set(n)) for n in ['test_synth', 'test_gold', 'test_adv'] if (ROOT / f'data/splits/{n}.jsonl').exists()}
print('Tập đánh giá:', tests, '' if 'test_gold' in tests else '← CHƯA có test_gold viết tay (T06)')
R = retriever.HybridRetriever(retriever.load_docs())
for q in ['com cuon kieu cali bao nhieu tien', 'chi nhanh go vap mo cua may gio']:
    print(q, '→', [c['chunk_id'] for c in R.search_chunks(q, 'bm25_filter', k=3)])
try:   # độ dài token theo tokenizer thật (cần Internet tới Hugging Face)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    n = [len(tok.apply_chat_template(r['prompt'] + r['completion'], tokenize=True)) for r in ds2.select(range(2000))]
    print('Token / mẫu (Qwen):', pd.Series(n).describe(percentiles=[.5, .95, .99]).round(0).to_dict())
except Exception as e:
    print('Bỏ qua đếm token (không truy cập được Hugging Face):', type(e).__name__)
'''),

    md("## 9. Manifest dữ liệu\nGhi md5 các file dữ liệu → NB2 kiểm tra trước khi huấn luyện."),
    code(r'''
from kimochi import manifest
mf = manifest.write()
display(pd.DataFrame(mf).T)
assert manifest.verify() == []
print('✓ Dữ liệu sẵn sàng cho NB2')
'''),
]

# ================================================================================================ NB2
KAGGLE_SETUP = r'''
%%capture
!pip install -q "transformers==4.56.2" "trl==0.21.0" "peft==0.17.0" "bitsandbytes==0.48.2" "accelerate>=1.4" "datasets>=3.0" \
                rank-bm25 "sentence-transformers>=3.0" optuna gradio==5.49.1 google-genai
'''
NB2_ENV = r'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # chỉ dùng 1 GPU (xem kaggle_env.py) — phải đặt trước khi import torch
os.environ.update({'USE_TF': '0', 'USE_JAX': '0', 'TF_CPP_MIN_LOG_LEVEL': '3'})   # không nạp TensorFlow (chỉ dùng PyTorch)
import sys, glob, json, time, shutil, pathlib
cands = glob.glob('/kaggle/input/**/code/src', recursive=True)
if cands:                       # Kaggle: chép repo (dataset chỉ đọc) sang /kaggle/working/repo
    sys.path.insert(0, cands[0])
    from kimochi.kaggle_env import setup, gpu_info
    ROOT = setup()
else:                           # local / Colab: tìm thư mục gốc repo
    ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / 'code/src/kimochi').exists())
    sys.path.insert(0, str(ROOT / 'code/src'))
    from kimochi.kaggle_env import gpu_info
os.chdir(ROOT)
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
from kimochi import llm, prompts, metrics, retriever, agent, agent_eval, viz, manifest, tools
RES = ROOT / 'results'; GEN = RES / 'gen'; GEN.mkdir(parents=True, exist_ok=True)
WORK = pathlib.Path('/kaggle/working') if pathlib.Path('/kaggle/working').exists() else ROOT / 'runs'
RUNS = WORK / 'runs'; RUNS.mkdir(parents=True, exist_ok=True)
print('GPU:', gpu_info())
'''

NB2 = [
    md(r'''
# NB2 · Fine-tuning LoRA / QLoRA → đánh giá → Hybrid-RAG → xuất mô hình → demo MVP
**Chatbot CSKH Kimochi Sushi.** Chạy trên **Kaggle GPU T4** (Settings: Accelerator *GPU T4 x2*, Internet *On*; Add Data: dataset
chứa repo này, đã chạy NB1; Secrets: `HF_TOKEN`, tùy chọn `GEMINI_API_KEY`). Dùng *Save & Run All (Commit)* để chạy nền.

| § | Nội dung | Thời gian ước tính (T4) |
|---|---|---|
| 0 | Cấu hình thí nghiệm | — |
| 1 | Nhận dữ liệu & kiểm tra manifest | 1 phút |
| 2 | Thiết lập: siêu tham số (Optuna tùy chọn), mô hình cơ sở | 30–60 phút |
| 3 | Fine-tune lần lượt **LoRA 16-bit**, **QLoRA 4-bit** (+ QLoRA 8-bit tùy chọn) × {Qwen, LLaMA} | 3–6 giờ |
| 4 | Đánh giá các mô hình trên tập test | 40–80 phút |
| 5 | Trực quan & phân tích kết quả | 2 phút |
| 6 | Demo giao diện nhanh (mô hình tốt nhất) | tùy chọn |
| 7 | Xây dựng Hybrid-RAG (BM25 + dense + RRF + reranker + lọc) & ablation | 10–15 phút |
| 8 | Đánh giá hệ thống đầy đủ: closed-book ↔ RAG ↔ RAG + Agent; so sánh, trực quan | 30–60 phút |
| 9 | Xuất mô hình hoàn chỉnh (gộp adapter) + gói MVP | 10 phút |
| 10 | Demo MVP | — |

Thời gian là **ước lượng**. Phiên Kaggle tối đa 12 giờ → mọi bước **tự bỏ qua nếu đã có kết quả** (`RESUME_GLOB`): chạy lại
notebook với output phiên trước làm input là chạy tiếp. Kết quả ghi vào `results/` với cùng tên file mà báo cáo đọc.
'''),
    code(KAGGLE_SETUP),
    code(NB2_ENV),

    md("## 0. Cấu hình thí nghiệm\nChỉ đổi mô hình nền và kiểu lượng tử; mọi thứ khác giống nhau (CLAUDE.md §3.5)."),
    code(r'''
MODELS  = ['qwen', 'llama']                                   # Qwen2.5-3B-Instruct, LLaMA-3.2-3B-Instruct
METHODS = {'LoRA 16-bit': 'fp16', 'QLoRA 4-bit': 'nf4'}       # thêm 'QLoRA 8-bit': 'int8' nếu đủ thời gian GPU
SEED, EPOCHS, MAX_LEN = 42, 2, 1024
DEFAULT_HP = {'r': 16, 'lr': 2e-4}
RUN_OPTUNA, N_TRIALS = False, 6                               # True: dò r, lr trên Qwen QLoRA 4-bit (~1,5–2 giờ)
RUN_BASELINE = True                                           # B0 zero-shot của mô hình gốc để so sánh
N_EVAL = None                                                 # None = toàn bộ tập test; 150 để chạy thử nhanh
RUN_QUICK_DEMO = False                                        # §6: mở demo nhanh (không chặn notebook)
RUN_MVP_DEMO = False                                          # §10 — CHỈ bật khi chạy tương tác (ô demo chạy mãi, tốn GPU)
HF_USER = None                                                # vd 'thanhtrung665' → đẩy adapter + mô hình gộp lên HF Hub (private)
RESUME_GLOB = '/kaggle/input/**/runs/*/result.json'           # kết quả huấn luyện phiên trước (nếu có)
for p in glob.glob(RESUME_GLOB, recursive=True):              # nạp lại ô đã huấn luyện
    src = pathlib.Path(p).parent
    if not (RUNS / src.name).exists():
        shutil.copytree(src, RUNS / src.name, ignore=shutil.ignore_patterns('checkpoint-*'))
for p in glob.glob('/kaggle/input/**/repo/results', recursive=True):   # kết quả + câu trả lời đã sinh của phiên trước
    shutil.copytree(p, RES, dirs_exist_ok=True)
print('Ô đã có:', sorted(x.name for x in RUNS.iterdir()), '| file kết quả:', len(list(RES.rglob('*.*'))))
'''),

    md("## 1. Nhận dữ liệu\nKiểm tra md5 so với `data/MANIFEST.json` do NB1 ghi — bảo đảm huấn luyện đúng phiên bản dữ liệu."),
    code(r'''
bad = manifest.verify(ROOT)
assert not bad, f'Dữ liệu lệch manifest: {bad} → chạy lại NB1 rồi tạo version dataset mới'
print('✓ Dữ liệu khớp manifest')
train_ds, val_ds = llm.sft_dataset(ROOT, 'train'), llm.sft_dataset(ROOT, 'val')
EVAL_SETS = ['test_synth'] + [s for s in ['test_gold', 'test_adv'] if (ROOT / f'data/splits/{s}.jsonl').exists()]
evsets = {s: prompts.load_eval_set(s) for s in EVAL_SETS}
evsets = {s: (d.head(N_EVAL) if N_EVAL else d) for s, d in evsets.items()}
print(f'Huấn luyện: {len(train_ds):,} mẫu | kiểm định: {len(val_ds):,} | đánh giá:', {s: len(d) for s, d in evsets.items()})
if 'test_gold' not in EVAL_SETS:
    display(Markdown('> ⚠ Chưa có `test_gold` viết tay — kết quả trên `test_synth` là **trong phân phối**, chưa phải số chính thức.'))
'''),

    md(r'''
## 2. Thiết lập
### 2.1 Siêu tham số
Mặc định r = 16, lr = 2e-4 (α = 2r, dropout 0,05, cosine, warmup 3%, batch hiệu dụng 16, loss chỉ trên lượt trợ lý).
Bật `RUN_OPTUNA` để dò r ∈ {8, 16, 32}, lr ∈ [5e-5, 3e-4] bằng TPE + MedianPruner trên ô rẻ nhất (Qwen QLoRA 4-bit, 30% dữ liệu).
'''),
    code(r'''
best_path = RES / 'optuna_best.json'
if RUN_OPTUNA and not best_path.exists():
    import optuna
    from transformers import TrainerCallback
    class Prune(TrainerCallback):
        def __init__(self, t): self.t = t
        def on_evaluate(self, args, state, control, metrics=None, **kw):
            self.t.report(metrics['eval_loss'], state.global_step)
            if self.t.should_prune(): raise optuna.TrialPruned()
    def objective(t):
        r, lr = t.suggest_categorical('r', [8, 16, 32]), t.suggest_float('lr', 5e-5, 3e-4, log=True)
        res = llm.train(ROOT, 'qwen', 'nf4', str(WORK / f'optuna/t{t.number}'), seed=SEED, lr=lr, r=r, max_steps=120,
                        data_frac=0.3, eval_steps=30, max_len=MAX_LEN, extra_callbacks=[Prune(t)])
        return res['eval_loss']
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=30))
    study.optimize(objective, n_trials=N_TRIALS)
    study.trials_dataframe().to_csv(RES / 'optuna_trials.csv', index=False)
    best_path.write_text(json.dumps({'best_params': study.best_params, 'best_val_loss': study.best_value}, indent=1))
HP = json.loads(best_path.read_text())['best_params'] if best_path.exists() else DEFAULT_HP
print('Siêu tham số dùng cho mọi ô:', HP)
'''),
    md("### 2.2 Mô hình cơ sở (B0 zero-shot, mô hình gốc fp16)\nĐiểm mốc để đo phần cải thiện do fine-tuning."),
    code(r'''
def eval_model(model, tok, system_name, modes=('closed',), contexts_fn=None, db_src=ROOT / 'data/kimochi.db'):
    """Sinh trên mọi tập đánh giá cho các chế độ: closed (không ngữ cảnh) · rag (thêm chunk) · agent (công cụ + chunk)."""
    rows = []
    for es, ev in evsets.items():
        for mode in modes:
            f = GEN / f'{system_name}_{mode}_{es}.jsonl'
            if f.exists():
                out = pd.read_json(f, lines=True)
            elif mode == 'agent':
                db = WORK / 'eval.db'; shutil.copy(db_src, db)
                ag = agent.run_batch(model, tok, list(ev['query']), db, contexts=[contexts_fn(q) for q in ev['query']])
                out = ev.assign(pred=[a['pred'] for a in ag], latency_batch_s=[a['latency_batch_s'] for a in ag],
                                tools_called=[a['tools_called'] for a in ag])
            else:
                msgs = [prompts.build_messages(q, 'rag' if mode == 'rag' else 'zero', docs=contexts_fn(q) if mode == 'rag' else None)
                        for q in ev['query']]
                preds, lat = llm.generate(model, tok, msgs, bs=8)
                out = ev.assign(pred=preds, latency_batch_s=lat)
            out = out.assign(system=f'{system_name}_{mode}', eval_set=es)
            out.to_json(f, orient='records', lines=True, force_ascii=False)
            s = metrics.summarize(metrics.score_generations(out))
            if mode == 'agent':
                s['tool_call_rate_price'] = float(out[out.intent == 'price_query'].tools_called.map(lambda t: 'get_menu_item' in t).mean())
            rows.append({'system': f'{system_name}_{mode}', 'eval_set': es, **s})
    return rows

baseline_rows = []
if RUN_BASELINE:
    for mk in MODELS:
        model, tok, _ = llm.load(mk, 'fp16')
        baseline_rows += [dict(r, system=r['system'].replace('_closed', ''), model=mk) for r in eval_model(model, tok, f'{mk}_B0')]
        del model, tok; llm.free()
    pd.DataFrame(baseline_rows).to_csv(RES / 'baseline.csv', index=False)
    display(pd.DataFrame(baseline_rows).round(3))
'''),

    md(r'''
## 3. Fine-tuning: LoRA 16-bit → QLoRA 4-bit (→ QLoRA 8-bit tùy chọn)
- **LoRA 16-bit:** trọng số nền fp16 đóng băng, chỉ huấn luyện ma trận hạng thấp $B A$: $h = W_0x + \frac{\alpha}{r}BAx$.
- **QLoRA 4-bit:** trọng số nền lượng tử **NF4** + double quantization, tính toán fp16, bộ tối ưu `paged_adamw_8bit`.
- **QLoRA 8-bit (tùy chọn):** trọng số nền LLM.int8.

Mỗi ô ghi `result.json` (VRAM đỉnh, thời gian, val loss) + `history.json`; ô đã có thì bỏ qua.
'''),
    code(r'''
train_rows = []
for mk in MODELS:
    for method, q in METHODS.items():
        out = RUNS / f'{mk}_{q}_s{SEED}'
        if (out / 'result.json').exists():
            res = json.loads((out / 'result.json').read_text())
        else:
            print(f'▶ Huấn luyện {mk} · {method}')
            res = llm.train(ROOT, mk, q, str(out), seed=SEED, lr=HP['lr'], r=HP['r'], epochs=EPOCHS, max_len=MAX_LEN)
            if HF_USER:
                res['hub'] = llm.push_adapter(res['adapter'], f'{HF_USER}/kimochi-{mk}-{q}-s{SEED}')
                (out / 'result.json').write_text(json.dumps(res, indent=1))
        train_rows.append({**res, 'method': method, 'run': out.name})
        pd.DataFrame(train_rows).to_csv(RES / 'train_matrix.csv', index=False)
TM = pd.DataFrame(train_rows)
display(TM[['model', 'method', 'trainable_params', 'peak_vram_gb', 'train_minutes', 'steps', 'eval_loss', 'eval_ppl']].round(3))
'''),
    code(r'''
hist = {f"{r.model} · {r.method}": json.loads((RUNS / r.run / 'history.json').read_text()) for r in TM.itertuples()}
(RES / 'histories').mkdir(exist_ok=True)
for r in TM.itertuples():
    shutil.copy(RUNS / r.run / 'history.json', RES / 'histories' / f'{r.run}.json')
viz.loss_curves(hist, RES / 'loss_curves.png'); plt.show()
viz.train_resources(TM, RES / 'train_resources.png'); plt.show()
'''),
    md(r'''
**Đọc biểu đồ.** Đường chấm đứng = bước có val loss nhỏ nhất (checkpoint được giữ nhờ `load_best_model_at_end`). Val loss tăng lại
trong khi train loss vẫn giảm → **quá khớp**, dừng sớm đã chặn. So sánh cặp LoRA ↔ QLoRA cùng mô hình: chênh VRAM là lợi ích
của lượng tử hóa, chênh val loss là cái giá phải trả (RQ2).
'''),

    md("## 4. Đánh giá các mô hình fine-tune trên tập test (closed-book — đo riêng năng lực của mô hình)"),
    code(r'''
ft_rows = []
for r in TM.itertuples():
    model, tok, _ = llm.load(r.model, r.quant, adapter=str(RUNS / r.run / 'adapter'))
    ft_rows += [dict(x, model=r.model, method=r.method, run=r.run) for x in eval_model(model, tok, r.run)]
    del model, tok; llm.free()
FT = pd.DataFrame(ft_rows); FT.to_csv(RES / 'finetuned_eval.csv', index=False)
display(FT[['system', 'eval_set', 'rougeL', 'fact_em', 'hallucinated_price_rate', 'refusal_rate']].round(3))
'''),

    md("## 5. Trực quan & phân tích kết quả"),
    code(r'''
ES = 'test_gold' if 'test_gold' in EVAL_SETS else 'test_synth'
def scored(system_mode, es=ES):
    return metrics.score_generations(pd.read_json(GEN / f'{system_mode}_{es}.jsonl', lines=True))
systems = ([f'{mk}_B0_closed' for mk in MODELS] if RUN_BASELINE else []) + [f'{r}_closed' for r in TM.run]
S = {s: scored(s) for s in systems}
tab, ci = {}, {}
for s, d in S.items():
    m, lo, hi = metrics.bootstrap_ci(d['rougeL']); ci[(s, 'ROUGE-L')] = (lo, hi)
    tab[s] = {'ROUGE-L': m, 'Từ chối': d['refused'].mean(), 'Bịa giá': d['hallucinated_price'].mean()}
T = pd.DataFrame(tab).T
T.to_csv(RES / f'ci_{ES}.csv'); display(T.round(3))
viz.metric_bars(T, ['ROUGE-L', 'Từ chối', 'Bịa giá'], ci, f'Mô hình cơ sở ↔ fine-tune ({ES}, khoảng tin cậy 95%)',
                RES / 'eval_models.png'); plt.show()
'''),
    code(r'''
q = TM.merge(T['ROUGE-L'].rename('rougeL'), left_on=TM.run + '_closed', right_index=True)
q['label'] = q.model + ' · ' + q.method
viz.quality_vs_vram(q, 'rougeL', RES / 'quality_vs_vram.png'); plt.show()
ih = viz.intent_heatmap(S, 'rougeL', RES / 'intent_heatmap.png'); plt.show()
'''),
    code(r'''
# So sánh ghép cặp (thay K-Fold): mô hình tốt nhất ↔ từng hệ thống khác
best_sys = T['ROUGE-L'].idxmax()
cmp = pd.DataFrame({s: metrics.paired_bootstrap(S[best_sys]['rougeL'].values, S[s]['rougeL'].values)
                    for s in S if s != best_sys}).T
cmp.to_csv(RES / 'paired_bootstrap.csv'); display(cmp.round(3))
noisy = {s: d.groupby('noisy_input')['rougeL'].mean() for s, d in S.items()}
display(pd.DataFrame(noisy).T.rename(columns={False: 'câu sạch', True: 'câu nhiễu'}).round(3))   # công bằng: nhóm câu nhiễu
lines = [f"- Mô hình tốt nhất (ROUGE-L, {ES}): **{best_sys}**."]
for mk in MODELS:
    a, b = f'{mk}_fp16_s{SEED}_closed', f'{mk}_nf4_s{SEED}_closed'
    if a in T.index and b in T.index:
        va = TM.set_index('run').loc[[a[:-7], b[:-7]], 'peak_vram_gb']
        lines.append(f"- {mk}: QLoRA 4-bit so với LoRA 16-bit: ROUGE-L {T.loc[b, 'ROUGE-L'] - T.loc[a, 'ROUGE-L']:+.3f}, "
                     f"VRAM {va.iloc[1] - va.iloc[0]:+.2f} GB (RQ2).")
for meth in METHODS.values():
    a, b = f'qwen_{meth}_s{SEED}_closed', f'llama_{meth}_s{SEED}_closed'
    if a in T.index and b in T.index:
        lines.append(f"- {meth}: Qwen − LLaMA = {T.loc[a, 'ROUGE-L'] - T.loc[b, 'ROUGE-L']:+.3f} ROUGE-L (RQ1).")
display(Markdown('**Phân tích tự động**\n' + '\n'.join(lines)))
worst = S[best_sys].sort_values('rougeL').head(10)[['query', 'reference', 'pred', 'intent']]
worst.to_csv(RES / 'error_cases.csv', index=False); display(worst)
'''),
    md(r'''
**Gợi ý phân tích để đưa vào báo cáo (Phần 4–5):** (1) RQ1 – họ mô hình nào tốt hơn, chênh có nằm ngoài khoảng tin cậy không
(`paired_bootstrap.csv`, `p_not_better` < 0,05); (2) RQ2 – đánh đổi chất lượng ↔ VRAM (`quality_vs_vram.png`); (3) intent nào
yếu (`intent_heatmap.png`) và vì sao; (4) chênh lệch câu nhiễu ↔ câu sạch (công bằng); (5) phân nhóm 10 ca lỗi.
Lưu ý: closed-book **không có** công cụ / ngữ cảnh nên câu hỏi giá *phải* từ chối hoặc sai — §8 đo hệ thống đầy đủ.
'''),

    md("## 6. Demo giao diện nhanh (mô hình tốt nhất, chưa có Hybrid-RAG đầy đủ)\nBật `RUN_QUICK_DEMO`; link `*.gradio.live` hiện bên dưới, notebook không bị chặn."),
    code(r'''
BEST_RUN = best_sys.replace('_closed', '') if not best_sys.startswith(tuple(f'{m}_B0' for m in MODELS)) else TM.sort_values('eval_loss').run.iloc[0]
BEST = TM.set_index('run').loc[BEST_RUN]
print('Mô hình dùng cho các bước sau:', BEST_RUN)
if RUN_QUICK_DEMO:
    os.environ.update({'KIMOCHI_BACKEND': 'hf', 'KIMOCHI_MODEL': BEST.model, 'KIMOCHI_QUANT': BEST.quant,
                       'KIMOCHI_ADAPTER': str(RUNS / BEST_RUN / 'adapter')})
    sys.path.insert(0, str(ROOT / 'demo')); import app
    app.demo.queue().launch(share=True, prevent_thread_lock=True)
'''),
    code(r'''
if RUN_QUICK_DEMO:
    app.demo.close(); app._BACKEND = None; llm.free()   # đóng demo, giải phóng GPU trước §7
'''),

    md(r'''
## 7. Xây dựng hệ thống Hybrid-RAG
Đơn vị truy xuất: chunk ≤ 600 ký tự (giống dữ liệu RAFT). Thành phần: **BM25** trên văn bản bỏ dấu + bigram · **TF-IDF n-gram ký tự** ·
**dense `BAAI/bge-m3`** · hợp nhất **RRF** (k = 60) · **reranker `BAAI/bge-reranker-v2-m3`** (top-20) · **lọc theo tên món** (metadata).
Đánh giá: câu hỏi test có gắn món → tài liệu đúng = tài liệu của món đó.
'''),
    code(r'''
dense, rer = retriever.load_models()
RAG = retriever.HybridRetriever(retriever.load_docs(), dense, rer)
rows = []
ev = retriever.eval_set('test_synth')
for cfg in retriever.CONFIGS:
    if ('dense' in cfg and dense is None) or ('rerank' in cfg and rer is None):
        rows.append({'config': cfg, 'status': 'skipped'}); continue
    t0 = time.perf_counter(); ranked = [RAG.search(q, cfg, k=10) for q in ev['query']]; dt = (time.perf_counter() - t0) / len(ev)
    m = retriever.metrics(ranked, list(ev['gold']))
    mn = retriever.metrics([x for x, n in zip(ranked, ev['noisy']) if n], list(ev.loc[ev['noisy'], 'gold']))
    rows.append({'config': cfg, 'status': 'ok', 'n': len(ev), **m, 'Recall@5_noisy': mn['Recall@5'], 'ms_per_query': dt * 1000})
RR = pd.DataFrame(rows); RR.to_csv(RES / 'retrieval_test_synth.csv', index=False)
display(RR.round(3)); viz.retrieval_bars(RR, RES / 'retrieval_ablation.png'); plt.show()
ok = RR[RR.status == 'ok']
RAG_CFG = ok.sort_values(['MRR@10', 'ms_per_query'], ascending=[False, True]).config.iloc[0]
print('Cấu hình truy xuất được chọn (MRR@10 cao nhất):', RAG_CFG)
ctx = lambda q: RAG.search_chunks(q, RAG_CFG, k=3)
'''),
    code(r'''
ex = retriever.explain_bm25(RAG, 'com cuon kieu cali bao nhieu tien')       # XAI cho bộ truy xuất
print(ex['chunk_id'], ex['title'], round(ex['score'], 2)); display(pd.DataFrame(ex['terms'], columns=['token', 'đóng góp BM25']))
'''),

    md(r'''
## 8. Đánh giá hệ thống đầy đủ & so sánh
Cùng mô hình tốt nhất ở ba chế độ: **closed-book** → **+ Hybrid-RAG** → **+ RAG + Agent** (công cụ SQL chạy trên bản sao DB),
so với **mô hình gốc + RAG** (không fine-tune, trả lời RQ3). Thêm định tuyến công cụ, kịch bản đặt bàn nhiều lượt và LLM-as-judge.
'''),
    code(r'''
sys_rows = []
model, tok, _ = llm.load(BEST.model, BEST.quant, adapter=str(RUNS / BEST_RUN / 'adapter'))
sys_rows += eval_model(model, tok, BEST_RUN, modes=('closed', 'rag', 'agent'), contexts_fn=ctx)
lat1 = llm.latency_single(model, tok, [prompts.build_messages(q, 'rag', docs=ctx(q)) for q in evsets['test_synth'].head(30)['query']])
sys_rows.append({'system': f'{BEST_RUN}_rag', 'eval_set': 'latency30', 'latency_p50': float(np.median(lat1)),
                 'latency_p95': float(np.quantile(lat1, .95))})
gen_fn = lambda msgs: llm.generate(model, tok, [msgs], bs=1)[0][0]
sc_path = ROOT / 'data/splits/scenarios.jsonl'
scs = [json.loads(l) for l in sc_path.read_text(encoding='utf-8').splitlines() if l.strip()] if sc_path.exists() else agent_eval.scenarios_from_v2_val(ROOT)
kind = 'viết tay' if sc_path.exists() else 'PROXY (val v2)'
(WORK / 'sc').mkdir(exist_ok=True)
SC = pd.DataFrame([agent_eval.run_scenario(s, gen_fn, ROOT / 'data/kimochi.db', WORK / 'sc') for s in scs])
SC.to_json(RES / 'agent_scenarios.jsonl', orient='records', lines=True, force_ascii=False)
(RES / 'agent_summary.json').write_text(json.dumps({'kind': kind, 'n': len(SC), 'task_success': SC.success.mean(),
    'db_state_ok': SC.db_state_ok.mean(), 'hitl_ok': SC.hitl_ok.mean(), 'avg_tool_calls': SC.n_tool_calls.mean()}, indent=1, default=float))
del model, tok; llm.free()
base_mk = BEST.model
model, tok, _ = llm.load(base_mk, 'fp16')                                         # mô hình gốc + RAG (không fine-tune)
sys_rows += eval_model(model, tok, f'{base_mk}_B2', modes=('rag',), contexts_fn=ctx)
del model, tok; llm.free()
SYS = pd.DataFrame(sys_rows)
FT_all = pd.concat([pd.read_csv(RES / 'finetuned_eval.csv'), SYS[SYS.system.str.startswith(BEST_RUN)]]).drop_duplicates(['system', 'eval_set'], keep='last')
FT_all.to_csv(RES / 'finetuned_eval.csv', index=False)
pd.concat([pd.read_csv(RES / 'baseline.csv') if (RES / 'baseline.csv').exists() else pd.DataFrame(),
           SYS[SYS.system.str.startswith(f'{base_mk}_B2')].assign(system=f'{base_mk}_B2', model=base_mk)]
          ).drop_duplicates(['system', 'eval_set'], keep='last').to_csv(RES / 'baseline.csv', index=False)
display(SYS.round(3)); print('Kịch bản:', kind, json.loads((RES / 'agent_summary.json').read_text()))
'''),
    code(r'''
order = [f'{base_mk}_B0_closed', f'{base_mk}_B2_rag', f'{BEST_RUN}_closed', f'{BEST_RUN}_rag', f'{BEST_RUN}_agent']
names = {order[0]: 'Gốc', order[1]: 'Gốc + RAG', order[2]: 'Fine-tune', order[3]: 'Fine-tune + RAG', order[4]: 'Fine-tune + RAG + Agent'}
E2 = {names[s]: scored(s, 'test_synth') for s in order if (GEN / f'{s}_test_synth.jsonl').exists()}
cmpT, ci2 = {}, {}
for n, d in E2.items():
    for col, key in [('rougeL', 'ROUGE-L'), ('fact_ok', 'Fact-EM giá'), ('refused', 'Từ chối')]:
        m, lo, hi = metrics.bootstrap_ci(d[col]); ci2[(n, key)] = (lo, hi); cmpT.setdefault(n, {})[key] = m
    cmpT[n]['Bịa giá'] = d['hallucinated_price'].mean()
C2 = pd.DataFrame(cmpT).T; C2.to_csv(RES / 'system_comparison.csv'); display(C2.round(3))
viz.metric_bars(C2, ['ROUGE-L', 'Fact-EM giá', 'Bịa giá'], ci2, 'So sánh hệ thống trên test_synth (CI 95%)', RES / 'system_comparison.png'); plt.show()
ag = pd.read_json(GEN / f'{BEST_RUN}_agent_test_synth.jsonl', lines=True)
rep = agent_eval.routing_report(ag)
rep['confusion'].to_csv(RES / 'tool_routing_confusion.csv'); rep['per_label'].to_csv(RES / 'tool_routing_per_label.csv')
(RES / 'tool_routing_summary.json').write_text(json.dumps({k: rep[k] for k in ('accuracy', 'macro_f1', 'business_cost', 'cost_per_100')}, indent=1))
agent_eval.plot_confusion(rep['confusion'], RES / 'tool_routing_confusion.png'); display(Image(RES / 'tool_routing_confusion.png', width=520))
print({k: round(rep[k], 3) for k in ('accuracy', 'macro_f1', 'cost_per_100')})
'''),
    code(r'''
JUDGE_N = 100   # LLM-as-judge (Gemini) — cần secret GEMINI_API_KEY; hiệu chỉnh với data/splits/human_scores.csv nếu có
if os.environ.get('GEMINI_API_KEY'):
    from scipy.stats import spearmanr
    jr = []
    for n, s in [('Gốc + RAG', order[1]), ('Fine-tune + RAG', order[3]), ('Fine-tune + RAG + Agent', order[4])]:
        d = scored(s, ES).head(JUDGE_N)
        for r in d.itertuples():
            sc_, why = metrics.judge(r.query, r.reference, r.pred); jr.append({'id': r.id, 'system': s, 'judge': sc_, 'reason': why}); time.sleep(.4)
    J = pd.DataFrame(jr); J.to_csv(RES / 'judge_scores.csv', index=False); display(J.groupby('system').judge.agg(['mean', 'count']).round(2))
    hs = ROOT / 'data/splits/human_scores.csv'
    if hs.exists():
        h = pd.read_csv(hs).merge(J, on=['id', 'system']); print('Spearman judge ↔ người:', spearmanr(h.human_score, h.judge), 'n =', len(h))
else:
    print('Bỏ qua judge (chưa có GEMINI_API_KEY)')
'''),
    md(r'''
**Gợi ý phân tích:** RAG có cải thiện cả mô hình gốc lẫn mô hình fine-tune không (RQ3)? Agent có đưa **Fact-EM giá** lên gần 100% và
**bịa giá** về 0 không (giá đọc từ DB)? Nhầm lẫn định tuyến nào tốn chi phí nhất (ma trận nhầm lẫn)? Tỷ lệ hoàn thành kịch bản
đặt bàn và tuân thủ xác nhận trước khi ghi (`agent_summary.json`)? Độ trễ p95 so với KPI ≤ 5 s?
'''),

    md(r'''
## 9. Xuất mô hình hoàn chỉnh cho MVP
Gộp adapter vào trọng số fp16 (`merge_and_unload`) → thư mục safetensors độc lập (không cần PEFT khi chạy), kiểm tra nạp lại
ở NF4 và trả lời đúng, tùy chọn đẩy lên Hugging Face Hub, đóng gói **MVP bundle** (mô hình + app + kho tri thức + DB + hướng dẫn).
'''),
    code(r'''
EXPORT = WORK / 'kimochi-mvp-model'
if not (EXPORT / 'kimochi_export.json').exists():
    info = llm.export_merged(BEST.model, str(RUNS / BEST_RUN / 'adapter'), str(EXPORT))
else:
    info = json.loads((EXPORT / 'kimochi_export.json').read_text())
info.update({'run': BEST_RUN, 'method': BEST.method, 'retrieval_config': RAG_CFG}); (EXPORT / 'kimochi_export.json').write_text(json.dumps(info, indent=1))
print(info)
m2, t2, _ = llm.load(str(EXPORT), 'nf4')                                         # kiểm tra nạp lại + chạy agent 1 câu
db = WORK / 'smoke.db'; shutil.copy(ROOT / 'data/kimochi.db', db)
r = agent.run([agent.system_with_tools(), {'role': 'user', 'content': 'com luon bao nhieu tien vay shop'}],
              lambda m: llm.generate(m2, t2, [m], bs=1)[0][0], db)
print('Trả lời:', r['answer']); print('Công cụ:', [t['call'] for t in r['trace']])
del m2, t2; llm.free()
if HF_USER:
    print(llm.push_adapter(str(EXPORT), f'{HF_USER}/kimochi-mvp'))
'''),
    code(r'''
BUNDLE = WORK / 'kimochi-mvp'
if BUNDLE.exists(): shutil.rmtree(BUNDLE)
for sub in ['demo', 'code/src', 'data/kb']:
    shutil.copytree(ROOT / sub, BUNDLE / sub, ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy(ROOT / 'data/kimochi.db', BUNDLE / 'data/kimochi.db')
model_ref = f'{HF_USER}/kimochi-mvp' if HF_USER else '../kimochi-mvp-model'
(BUNDLE / 'run_mvp.sh').write_text(f"""#!/usr/bin/env bash
# Chạy MVP: cần GPU ≥ 8 GB (nạp NF4). Đổi KIMOCHI_MODEL nếu đặt mô hình ở nơi khác.
pip install -r demo/requirements.txt
KIMOCHI_BACKEND=hf KIMOCHI_MODEL={model_ref} KIMOCHI_QUANT=nf4 KIMOCHI_ADAPTER= python demo/app.py
""")
(BUNDLE / 'MVP_README.md').write_text(f"""# Kimochi Sushi — MVP
Mô hình: {info['base']} + adapter {BEST.method} (gộp, fp16, {info['size_gb']} GB) · truy xuất: `{RAG_CFG}`.
Chạy: `bash run_mvp.sh` → http://localhost:7860. Bàn / đặt bàn là dữ liệu mô phỏng; dị ứng chỉ tham khảo.
""")
shutil.make_archive(str(WORK / 'kimochi-mvp-bundle'), 'zip', BUNDLE)
print('Gói MVP:', WORK / 'kimochi-mvp-bundle.zip', '| mô hình:', EXPORT)
'''),

    md("## 10. Demo MVP\nChạy app với **mô hình đã xuất** (không cần adapter). Link `*.gradio.live` dùng để quay video 2–3 phút (kịch bản: `demo/README.md`) hoặc cho người dùng thật thử. Dừng ô này để tắt."),
    code(r'''
if RUN_MVP_DEMO:
    os.environ.update({'KIMOCHI_BACKEND': 'hf', 'KIMOCHI_MODEL': str(EXPORT), 'KIMOCHI_QUANT': 'nf4', 'KIMOCHI_ADAPTER': ''})
    sys.path.insert(0, str(ROOT / 'demo'))
    import importlib, app; importlib.reload(app)
    app.demo.queue().launch(share=True, debug=True)
'''),
]


def write(name, cells, kaggle=False):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    if kaggle:
        nb.metadata["kaggle"] = {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}
    nbf.write(nb, OUT / name)
    print("wrote", OUT / name)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["NB1", "NB2"]          # vd: python code/make_main_notebooks.py NB2  (không ghi đè NB1 đã chạy)
    if "NB1" in which:
        write("NB1_data_pipeline.ipynb", NB1)
    if "NB2" in which:
        write("NB2_finetune_rag_mvp.ipynb", NB2, kaggle=True)
