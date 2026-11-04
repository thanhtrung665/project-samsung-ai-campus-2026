"""Sinh các notebook mỏng gọi module trong code/src/kimochi (logic nằm ở .py để dễ review / test)."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE.parent / 'notebooks' / 'cpu'
SETUP = ("import sys, os, json, pathlib, pandas as pd\n"
         "# tìm thư mục gốc repo (chứa code/src/kimochi) rồi làm việc từ đó — notebook đặt ở đâu cũng chạy được\n"
         "ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / 'code/src/kimochi').exists())\n"
         "os.chdir(ROOT); sys.path.insert(0, str(ROOT / 'code/src'))\npd.set_option('display.max_colwidth', 80)")

NB = {
 "01_build_kb.ipynb": [
  ("md", "# 01 · Dựng kho tri thức Kimochi (T05)\nTrích món, giá, danh mục, dị ứng (suy luận) từ bộ hội thoại gốc → `data/kb/`, `data/kimochi.db`.\n"
         "Bàn / đặt bàn là **mô phỏng** (`simulated=1`). Dị ứng suy luận bằng từ khóa, `allergen_verified=0` → cần quản lý xác nhận."),
  ("code", SETUP),
  ("code", "from kimochi import build_kb\nq = build_kb.main()\n{k: v for k, v in q.items() if k != 'category_counts'}"),
  ("md", "## Từ điển dữ liệu — bảng `menu_items`"),
  ("code", "menu = pd.read_csv('data/kb/menu.csv')\nprint(menu.shape)\nmenu.drop(columns='description').head(10)"),
  ("code", "import sqlite3\ncon = sqlite3.connect('data/kimochi.db')\n"
           "for t in ['categories','menu_items','branches','tables','reservations','promotions']:\n"
           "    print(t, con.execute(f'select count(*) from {t}').fetchone()[0], [r[1] for r in con.execute(f'pragma table_info({t})')])"),
  ("md", "## Xung đột giá và món thiếu giá (phát hiện chất lượng dữ liệu)"),
  ("code", "print(q['price_conflicts'])\nmenu[menu.price_vnd.isna()][['item_id','name','category']]"),
 ],
 "02_prepare_sft.ipynb": [
  ("md", "# 02 · Chuẩn bị dữ liệu SFT & chia split chống rò rỉ (T07)\nIntent theo luật, `group_id` = câu trả lời chuẩn hóa, StratifiedGroupKFold, kiểm cosine > 0,92."),
  ("code", SETUP),
  ("code", "from kimochi import prepare_sft\ns = prepare_sft.main()\n{k: v for k, v in s.items() if k != 'intent_by_split'}"),
  ("code", "pd.DataFrame(s['intent_by_split']).fillna(0).astype(int)"),
  ("code", "print(json.dumps(json.loads(open('data/splits/train.jsonl', encoding='utf-8').readline()), ensure_ascii=False, indent=1))"),
 ],
 "03_eda.ipynb": [
  ("md", "# 03 · Phân tích khám phá dữ liệu (T08)\nMỗi hình lưu vào `images/eda_*.png` và được dùng trong Phần 3 báo cáo."),
  ("code", SETUP + "\nfrom IPython.display import Image, display"),
  ("code", "from kimochi import eda\nst = eda.run_all()"),
  ("code", "for f in ['eda_intent_split','eda_lengths','eda_noise','eda_group_size','eda_leakage','eda_funnel','eda_menu']:\n"
           "    display(Image(f'images/{f}.png', width=750))"),
  ("code", "st"),
  ("md", "## Độ dài token theo tokenizer Qwen / LLaMA (chạy trên Colab — cần truy cập Hugging Face)"),
  ("code", "try:\n    from transformers import AutoTokenizer\n    q = pd.read_json('data/splits/train.jsonl', lines=True)\n"
           "    texts = q.messages.map(lambda m: m[1]['content'] + ' ' + m[2]['content'])\n"
           "    for name in ['Qwen/Qwen2.5-3B-Instruct', 'unsloth/Llama-3.2-3B-Instruct']:\n"
           "        tok = AutoTokenizer.from_pretrained(name)\n        n = texts.map(lambda t: len(tok(t)['input_ids']))\n"
           "        print(name, n.describe().round(1).to_dict())\nexcept Exception as e:\n    print('Bỏ qua (không truy cập được HF):', type(e).__name__)"),
 ],
}

for name, cells in NB.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, OUT / name)
    print("wrote", name)

NB2 = {"05_prepare_sft_v2.ipynb": [
  ("md", "# 05 · Bộ dữ liệu SFT v2 (T07b)\nLàm sạch, giá qua công cụ SQL, RAFT, mẫu khuôn (đặt bàn nhiều lượt, từ chối, khiếu nại…). "
         "Dataset card: `data/splits_v2/README.md`."),
  ("code", SETUP),
  ("code", "from kimochi import prepare_sft_v2\ns = prepare_sft_v2.main()\n{k: v for k, v in s.items() if k not in ('format_by_split', 'intent_by_split')}"),
  ("code", "pd.DataFrame(s['format_by_split']).fillna(0).astype(int)"),
  ("code", "pd.DataFrame(s['intent_by_split']).fillna(0).astype(int)"),
  ("code", "rows = pd.read_json('data/splits_v2/train.jsonl', lines=True)\n"
           "for fmt in ['tool_price', 'raft_neg', 'reservation_mt']:\n"
           "    r = rows[rows.meta.map(lambda m: m['format']) == fmt].iloc[0]\n    print('=====', fmt)\n"
           "    for m in r.messages[1:]:\n        print(f\"  [{m['role']}] {m['content'][:200]}\")"),
  ("code", "from kimochi import llm\nfrom pathlib import Path\nprint('Số mẫu huấn luyện sau khi tách theo lượt assistant:', len(llm.sft_dataset(ROOT, 'train')))"),
],
"04_rag.ipynb": [
  ("md", "# 04 · Hybrid RAG & ablation truy xuất (T09)\nCấu hình: BM25, TF-IDF ký tự, dense bge-m3, RRF, reranker bge-reranker-v2-m3, lọc theo tên món.\n"
         "Dense/reranker cần Hugging Face (Colab): `pip install sentence-transformers`. Cấu hình không chạy được ghi `skipped`."),
  ("code", SETUP),
  ("code", "from kimochi import retriever\nres = retriever.run('test_synth')\nres.round(3)"),
  ("code", "r = retriever.HybridRetriever(retriever.load_docs())\nfor q in ['com cuon kieu cali gia bao nhieu', 'mon nao khong co tom cho be', 'chi nhanh go vap o dau']:\n"
           "    print(q, '->', r.search(q, 'bm25_filter', k=3))"),
]}
for name, cells in NB2.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s) for t, s in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, OUT / name)
