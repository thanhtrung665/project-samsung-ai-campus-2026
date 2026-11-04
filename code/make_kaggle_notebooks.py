"""Sinh các notebook Kaggle (GPU T4) cho T09 (dense), T10 (baseline), T11 (Optuna + ma trận 2×2 + learning curve), T13 (đánh giá).

Mọi notebook: Accelerator = GPU T4 x2 (dùng 1 GPU), Internet = On, Add Data = dataset chứa repo này,
Secrets: HF_TOKEN (bắt buộc với LLaMA gốc), GEMINI_API_KEY (chỉ K5 — LLM-as-judge).
Kết quả ghi vào /kaggle/working/repo/results → tải về và chép vào results/ của repo.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE.parent / 'notebooks' / 'kaggle'

INSTALL = """%%capture
!pip install -q "transformers==4.56.2" "trl==0.21.0" "peft==0.17.0" "bitsandbytes==0.47.0" "accelerate>=1.4" "datasets>=3.0" rank-bm25 optuna"""
SETUP = """import sys, glob, json, os, time
sys.path.insert(0, glob.glob('/kaggle/input/**/code/src', recursive=True)[0])
from kimochi.kaggle_env import setup, gpu_info
ROOT = setup()
import pandas as pd
from pathlib import Path
RES = ROOT / 'results'
print(gpu_info())"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


NB = {}

# ---------------------------------------------------------------------------------------------- K1 dense RAG
NB["K1_rag_dense.ipynb"] = [
    md("# K1 · Hybrid RAG đầy đủ: BM25 + dense bge-m3 + RRF + reranker (T09)\n"
       "Hoàn thiện ablation R1–R5 mà máy CPU đã đánh dấu `skipped`. ~10–15 phút trên T4."),
    code('%%capture\n!pip install -q "sentence-transformers>=3.0" rank-bm25'),
    code(SETUP),
    code("from kimochi import retriever\nfor split in ['val', 'test_synth']:\n    res = retriever.run(split)\n    display(res.round(3))"),
    md("Nếu đã có `test_gold.jsonl` có trường `gold_doc_ids`, đánh giá thêm trên bộ viết tay (số chính thức cho báo cáo):"),
    code("gold = ROOT / 'data/splits/test_gold.jsonl'\nif gold.exists():\n"
         "    import numpy as np\n    g = pd.read_json(gold, lines=True)\n    g = g[g.gold_doc_ids.map(lambda x: bool(x))]\n"
         "    dense, rer = retriever.load_models()\n    r = retriever.HybridRetriever(retriever.load_docs(), dense, rer)\n    rows = []\n"
         "    for cfg in retriever.CONFIGS:\n        try:\n            ranked = [r.search(q, cfg, k=10) for q in g['query']]\n"
         "        except Exception as e:\n            continue\n"
         "        # nhiều tài liệu đúng: lấy vị trí tài liệu đúng đầu tiên\n"
         "        first = [next((d for d in rk if d in set(gd)), gd[0]) for rk, gd in zip(ranked, g.gold_doc_ids)]\n"
         "        rows.append({'config': cfg, **retriever.metrics(ranked, first)})\n"
         "    out = pd.DataFrame(rows); out.to_csv(RES / 'retrieval_test_gold.csv', index=False); display(out.round(3))\n"
         "else:\n    print('Chưa có test_gold.jsonl — bỏ qua')"),
]

# ---------------------------------------------------------------------------------------------- K2 baseline
NB["K2_baseline.ipynb"] = [
    md("# K2 · Baseline B0 (zero-shot), B1 (5-shot), B2 (RAG) cho Qwen2.5-3B và LLaMA-3.2-3B (T10)\n"
       "Model gốc fp16, sinh greedy. ~25–40 phút / model với `N_EVAL = None` (toàn bộ test_synth)."),
    code(INSTALL),
    code(SETUP),
    code("N_EVAL = None          # None = toàn bộ; đặt 150 để chạy thử nhanh\nEVAL_SETS = ['test_synth'] + "
         "[s for s in ['test_gold', 'test_adv'] if (ROOT / f'data/splits/{s}.jsonl').exists()]\nMODEL_KEYS = ['qwen', 'llama']\nEVAL_SETS"),
    code("from kimochi import prompts, metrics, llm, retriever\nshots = prompts.pick_shots()\nR = retriever.HybridRetriever(retriever.load_docs())\n"
         "docs_by_id = {d['doc_id']: d for d in R.docs}\nGEN = RES / 'gen'; GEN.mkdir(exist_ok=True)\n\n"
         "def contexts(q, k=3):\n    return R.search_chunks(q, 'bm25_filter', k=k)"),
    code("rows = []\nfor mk in MODEL_KEYS:\n    model, tok, name = llm.load(mk, 'fp16')\n    for es in EVAL_SETS:\n"
         "        ev = prompts.load_eval_set(es)\n        ev = ev.head(N_EVAL) if N_EVAL else ev\n"
         "        for mode in ['zero', 'few', 'rag']:\n"
         "            msgs = [prompts.build_messages(q, mode, shots=shots, docs=contexts(q) if mode == 'rag' else None) for q in ev['query']]\n"
         "            preds, lat = llm.generate(model, tok, msgs, bs=8)\n"
         "            out = ev.assign(pred=preds, latency_batch_s=lat, system=f'{mk}_B{[\"zero\",\"few\",\"rag\"].index(mode)}', eval_set=es)\n"
         "            out.to_json(GEN / f'{mk}_{mode}_{es}.jsonl', orient='records', lines=True, force_ascii=False)\n"
         "            s = metrics.summarize(metrics.score_generations(out))\n"
         "            rows.append({'system': out.system.iloc[0], 'model': name, 'eval_set': es, **s}); print(rows[-1])\n"
         "    # độ trễ phục vụ thật (batch = 1) trên 30 câu, cấu hình RAG\n"
         "    q30 = prompts.load_eval_set('test_synth').head(30)['query']\n"
         "    lat1 = llm.latency_single(model, tok, [prompts.build_messages(q, 'rag', docs=contexts(q)) for q in q30])\n"
         "    rows.append({'system': f'{mk}_B2_latency', 'model': name, 'eval_set': 'latency30',\n"
         "                 'latency_p50': float(pd.Series(lat1).median()), 'latency_p95': float(pd.Series(lat1).quantile(.95))})\n"
         "    del model, tok; llm.free()\n"
         "base = pd.DataFrame(rows); base.to_csv(RES / 'baseline.csv', index=False); base.round(3)"),
    md("Kiểm tra định tính vài câu trả lời:"),
    code("pd.read_json(GEN / 'qwen_rag_test_synth.jsonl', lines=True)[['query', 'reference', 'pred']].head(8)"),
]

# ---------------------------------------------------------------------------------------------- K3 Optuna
NB["K3_optuna_qlora.ipynb"] = [
    md("# K3 · Tối ưu siêu tham số bằng Optuna trên Qwen QLoRA (T11)\n"
       "8 trial × (30% dữ liệu train, tối đa 120 bước), mục tiêu = val loss, MedianPruner cắt trial kém sớm. "
       "~2–3 giờ trên T4. Cấu hình tốt nhất lưu `results/optuna_best.json` và được K4 dùng cho cả 4 ô ma trận."),
    code(INSTALL),
    code(SETUP),
    code("N_TRIALS, FRAC, MAX_STEPS = 8, 0.3, 120\nimport optuna\nfrom transformers import TrainerCallback\nfrom kimochi import llm\n\n"
         "class OptunaPrune(TrainerCallback):\n    def __init__(self, trial): self.trial = trial\n"
         "    def on_evaluate(self, args, state, control, metrics=None, **kw):\n"
         "        self.trial.report(metrics['eval_loss'], state.global_step)\n"
         "        if self.trial.should_prune():\n            raise optuna.TrialPruned()\n\n"
         "def objective(trial):\n    r = trial.suggest_categorical('r', [8, 16, 32])\n"
         "    lr = trial.suggest_float('lr', 5e-5, 3e-4, log=True)\n    out = f'/kaggle/working/optuna/t{trial.number}'\n"
         "    try:\n        res = llm.train(ROOT, 'qwen', 'nf4', out, lr=lr, r=r, max_steps=MAX_STEPS, data_frac=FRAC, eval_steps=30,\n"
         "                        extra_callbacks=[OptunaPrune(trial)])\n    finally:\n        import gc, torch; gc.collect(); torch.cuda.empty_cache()\n"
         "    trial.set_user_attr('train_minutes', res['train_minutes']); trial.set_user_attr('peak_vram_gb', res['peak_vram_gb'])\n"
         "    return res['eval_loss']\n\n"
         "study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),\n"
         "                            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=30))\n"
         "study.optimize(objective, n_trials=N_TRIALS)"),
    code("df = study.trials_dataframe(); df.to_csv(RES / 'optuna_trials.csv', index=False)\n"
         "(RES / 'optuna_best.json').write_text(json.dumps({'best_params': study.best_params, 'best_val_loss': study.best_value}, indent=1))\n"
         "display(df[['number', 'value', 'params_r', 'params_lr', 'state']]); study.best_params"),
    code("import matplotlib.pyplot as plt\nfrom optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances\n"
         "ax = plot_optimization_history(study); plt.savefig(RES / 'optuna_history.png', dpi=200, bbox_inches='tight')"),
]

# ---------------------------------------------------------------------------------------------- K4 ma trận 2x2
NB["K4_train_matrix.ipynb"] = [
    md("# K4 · Huấn luyện ma trận {Qwen, LLaMA} × {LoRA fp16, QLoRA nf4} + learning curve (T11)\n"
       "Cùng dữ liệu, cùng siêu tham số (từ K3), cùng số epoch — chỉ đổi model / lượng tử hóa. Mỗi ô ~45–90 phút trên T4.\n\n"
       "**Phiên Kaggle tối đa 12 giờ** → nếu không đủ, chạy notebook nhiều lần, mỗi lần đặt `CELLS` khác nhau "
       "(kết quả từng ô được ghi riêng, ô đã xong tự bỏ qua nếu nạp lại thư mục output cũ làm dataset)."),
    code(INSTALL),
    code(SETUP),
    code("from kimochi import llm\nbest_path = RES / 'optuna_best.json'\n"
         "HP = json.loads(best_path.read_text())['best_params'] if best_path.exists() else {'r': 16, 'lr': 2e-4}\n"
         "CELLS = [('qwen', 'nf4'), ('qwen', 'fp16'), ('llama', 'nf4'), ('llama', 'fp16')]\nSEEDS = [42]        # thêm 7, 1234 nếu còn GPU (mean ± std)\n"
         "EPOCHS = 2\nDATA_DIR = 'splits_v2'   # bộ dữ liệu v2 (mặc định); 'splits' = v1 closed-book\nHF_USER = None      # vd 'thanhtrung665' → đẩy adapter lên HF Hub (private)\nprint(HP)"),
    code("results = []\nfor mk, q in CELLS:\n    for seed in SEEDS:\n        out = f'/kaggle/working/runs/{mk}_{q}_s{seed}'\n"
         "        if Path(out, 'result.json').exists():\n            results.append(json.loads(Path(out, 'result.json').read_text())); continue\n"
         "        res = llm.train(ROOT, mk, q, out, seed=seed, lr=HP['lr'], r=HP['r'], epochs=EPOCHS, data_dir=DATA_DIR)\n"
         "        if HF_USER:\n            res['hub'] = llm.push_adapter(res['adapter'], f'{HF_USER}/kimochi-{mk}-{q}-s{seed}')\n"
         "        results.append(res); print(res)\n        pd.DataFrame(results).to_csv(RES / 'train_matrix.csv', index=False)\n"
         "pd.DataFrame(results)"),
    md("## Ablation dữ liệu v1 ↔ v2 (cùng model, cùng siêu tham số) — chứng minh tác dụng của làm sạch, RAFT, mẫu công cụ"),
    code("out = '/kaggle/working/runs/qwen_nf4_s42_v1data'\nif not Path(out, 'result.json').exists():\n"
         "    r1 = llm.train(ROOT, 'qwen', 'nf4', out, lr=HP['lr'], r=HP['r'], epochs=EPOCHS, data_dir='splits')\n"
         "    pd.DataFrame([r1]).to_csv(RES / 'train_v1data.csv', index=False)\nprint(json.loads(Path(out, 'result.json').read_text()))"),
    md("## Đường cong loss (phát hiện quá khớp)"),
    code("import matplotlib.pyplot as plt\nfig, axes = plt.subplots(1, len(CELLS), figsize=(4 * len(CELLS), 3.2), sharey=True)\n"
         "for ax, (mk, q) in zip(axes, CELLS):\n    h = json.loads(Path(f'/kaggle/working/runs/{mk}_{q}_s{SEEDS[0]}/history.json').read_text())\n"
         "    tr = [(x['step'], x['loss']) for x in h if 'loss' in x]; ev = [(x['step'], x['eval_loss']) for x in h if 'eval_loss' in x]\n"
         "    ax.plot(*zip(*tr), label='train'); ax.plot(*zip(*ev), 'o-', label='val'); ax.set_title(f'{mk} {q}'); ax.set_xlabel('bước')\n"
         "axes[0].set_ylabel('loss'); axes[0].legend(); plt.tight_layout(); plt.savefig(RES / 'loss_curves.png', dpi=200)\n"
         "for (mk, q) in CELLS:\n    Path(RES / 'histories').mkdir(exist_ok=True)\n"
         "    Path(RES / 'histories' / f'{mk}_{q}.json').write_text(Path(f'/kaggle/working/runs/{mk}_{q}_s{SEEDS[0]}/history.json').read_text())"),
    md("## Learning curve theo cỡ dữ liệu (mô hình tốt nhất) — trả lời 'có đáng tạo thêm dữ liệu không'"),
    code("tm = pd.DataFrame(results)\nbest = tm.sort_values('eval_loss').iloc[0]; print('Ô tốt nhất:', best.model, best.quant)\nlc = []\n"
         "for frac in [0.25, 0.5]:\n    out = f'/kaggle/working/runs/lc_{best.model}_{best.quant}_{frac}'\n"
         "    r = llm.train(ROOT, best.model, best.quant, out, lr=HP['lr'], r=HP['r'], epochs=EPOCHS, data_frac=frac); lc.append(r)\n"
         "lc.append({**best.to_dict(), 'data_frac': 1.0})\nlc = pd.DataFrame(lc); lc.to_csv(RES / 'learning_curve.csv', index=False)\n"
         "plt.figure(figsize=(5, 3)); plt.plot(lc.data_frac * 100, lc.eval_loss, 'o-'); plt.xlabel('% dữ liệu train'); plt.ylabel('val loss')\n"
         "plt.savefig(RES / 'learning_curve.png', dpi=200, bbox_inches='tight'); lc[['data_frac', 'eval_loss', 'train_minutes']]"),
    code("!cd /kaggle/working && zip -qr adapters.zip runs -x '*/checkpoint-*'  # tải về nếu không dùng HF Hub\n!ls -la /kaggle/working"),
]

# ---------------------------------------------------------------------------------------------- K5 đánh giá
NB["K5_eval.ipynb"] = [
    md("# K5 · Đánh giá toàn diện M1–M4, M* + RAG (T13)\n"
       "Sinh trên test_synth / test_gold / test_adv cho từng adapter, closed-book và + RAG; fact-EM giá, tỷ lệ bịa giá, "
       "từ chối, ROUGE-L, độ trễ; paired bootstrap so với baseline; LLM-as-judge (Gemini) + hiệu chỉnh với người chấm.\n\n"
       "Add Data: dataset repo + output của K4 (thư mục `runs/`) hoặc dùng adapter trên HF Hub."),
    code(INSTALL + "\n!pip install -q google-genai"),
    code(SETUP),
    code("from kimochi import prompts, metrics, llm, retriever, agent\nimport shutil\n"
         "ADAPTERS = {Path(p).parent.name: str(Path(p).parent / 'adapter') for p in glob.glob('/kaggle/input/**/runs/*/result.json', recursive=True)\n"
         "            if not Path(p).parent.name.startswith('lc_')}\n"
         "# hoặc: ADAPTERS = {'qwen_nf4_s42': '<user>/kimochi-qwen-nf4-s42', ...}\nprint(ADAPTERS)\n"
         "EVAL_SETS = ['test_synth'] + [s for s in ['test_gold', 'test_adv'] if (ROOT / f'data/splits/{s}.jsonl').exists()]\n"
         "R = retriever.HybridRetriever(retriever.load_docs()); docs_by_id = {d['doc_id']: d for d in R.docs}\n"
         "ctx = lambda q: R.search_chunks(q, 'bm25_filter', k=3)\nGEN = RES / 'gen'; GEN.mkdir(exist_ok=True)"),
    code("rows = []\nfor run, ad in ADAPTERS.items():\n    mk, q = run.split('_')[:2]\n    model, tok, _ = llm.load(mk, q, adapter=ad)\n"
         "    for es in EVAL_SETS:\n        ev = prompts.load_eval_set(es)\n        for mode in ['closed', 'rag']:\n"
         "            msgs = [prompts.build_messages(x, 'rag' if mode == 'rag' else 'zero', docs=ctx(x) if mode == 'rag' else None) for x in ev['query']]\n"
         "            preds, lat = llm.generate(model, tok, msgs, bs=8)\n"
         "            out = ev.assign(pred=preds, latency_batch_s=lat, system=f'{run}_{mode}', eval_set=es)\n"
         "            out.to_json(GEN / f'{run}_{mode}_{es}.jsonl', orient='records', lines=True, force_ascii=False)\n"
         "            rows.append({'system': f'{run}_{mode}', 'eval_set': es, **metrics.summarize(metrics.score_generations(out))}); print(rows[-1])\n"
         "        # chế độ agent: system có mô tả công cụ + ngữ cảnh RAG; công cụ chạy trên BẢN SAO DB\n"
         "        db = Path('/kaggle/working/eval.db'); shutil.copy(ROOT / 'data/kimochi.db', db)\n"
         "        ag = agent.run_batch(model, tok, list(ev['query']), db, contexts=[ctx(x) for x in ev['query']])\n"
         "        out = ev.assign(pred=[a['pred'] for a in ag], latency_batch_s=[a['latency_batch_s'] for a in ag],\n"
         "                        tools_called=[a['tools_called'] for a in ag], system=f'{run}_agent', eval_set=es)\n"
         "        out.to_json(GEN / f'{run}_agent_{es}.jsonl', orient='records', lines=True, force_ascii=False)\n"
         "        sm = metrics.summarize(metrics.score_generations(out))\n"
         "        sm['tool_call_rate_price'] = float(out[out.intent == 'price_query'].tools_called.map(lambda t: 'get_menu_item' in t).mean())\n"
         "        rows.append({'system': f'{run}_agent', 'eval_set': es, **sm}); print(rows[-1])\n"
         "    q30 = prompts.load_eval_set('test_synth').head(30)['query']\n"
         "    lat1 = llm.latency_single(model, tok, [prompts.build_messages(x, 'rag', docs=ctx(x)) for x in q30])\n"
         "    rows.append({'system': f'{run}_rag', 'eval_set': 'latency30', 'latency_p50': float(pd.Series(lat1).median()),\n"
         "                 'latency_p95': float(pd.Series(lat1).quantile(.95))})\n    del model, tok; llm.free()\n"
         "ft = pd.DataFrame(rows); ft.to_csv(RES / 'finetuned_eval.csv', index=False); ft.round(3)"),
    md("## Khoảng tin cậy và so sánh ghép cặp (thay cho K-Fold)"),
    code("def load_gen(system, es):\n    f = {**{p.stem: p for p in GEN.glob('*.jsonl')}}\n"
         "    name = next(k for k in f if k.endswith(es) and k.startswith(system))\n"
         "    return metrics.score_generations(pd.read_json(f[name], lines=True))\n\n"
         "ES = 'test_gold' if 'test_gold' in EVAL_SETS else 'test_synth'\ncmp = []\n"
         "systems = [p.stem.rsplit('_' + ES, 1)[0] for p in GEN.glob(f'*_{ES}.jsonl')]\n"
         "for s in systems:\n    d = load_gen(s, ES)\n    m, lo, hi = metrics.bootstrap_ci(d['rougeL'])\n"
         "    cmp.append({'system': s, 'rougeL': m, 'ci_low': lo, 'ci_high': hi, 'fact_em': d['fact_ok'].mean()})\n"
         "cmp = pd.DataFrame(cmp).sort_values('rougeL', ascending=False); cmp.to_csv(RES / f'ci_{ES}.csv', index=False); display(cmp.round(3))\n"
         "best, ref = cmp.system.iloc[0], 'qwen_rag'  # so với baseline B2 cùng họ model\n"
         "if any(s.startswith(ref) for s in systems):\n"
         "    a, b = load_gen(best, ES), load_gen(ref, ES)\n    print(best, 'vs', ref, metrics.paired_bootstrap(a['rougeL'].values, b['rougeL'].values))"),
    md("## LLM-as-judge (Gemini) và hiệu chỉnh với người chấm\n"
       "Người chấm điền `data/splits/human_scores.csv` (cột `id`, `system`, `human_score` 1–5) cho ~50 mẫu; "
       "báo cáo tương quan Spearman giữa judge và người để chứng minh judge đáng tin."),
    code("JUDGE_N = 150\nfrom scipy.stats import spearmanr\njud = []\nfor s in cmp.system.head(3).tolist() + [x for x in systems if x in ('qwen_zero', 'qwen_rag', 'llama_rag')]:\n"
         "    d = load_gen(s, ES).head(JUDGE_N)\n    for r in d.itertuples():\n        sc, why = metrics.judge(r.query, r.reference, r.pred)\n"
         "        jud.append({'id': r.id, 'system': s, 'judge': sc, 'reason': why}); time.sleep(0.5)\n"
         "jud = pd.DataFrame(jud); jud.to_csv(RES / 'judge_scores.csv', index=False)\n"
         "display(jud.groupby('system').judge.agg(['mean', 'count']).round(2))\n"
         "hs = ROOT / 'data/splits/human_scores.csv'\nif hs.exists():\n    h = pd.read_csv(hs).merge(jud, on=['id', 'system'])\n"
         "    print('Spearman judge vs người:', spearmanr(h.human_score, h.judge), 'n =', len(h))"),
    md("## Mười ca lỗi điển hình (Phần 5 — phân tích lỗi)"),
    code("worst = load_gen(cmp.system.iloc[0], ES).sort_values('rougeL').head(10)[['query', 'reference', 'pred', 'intent']]\n"
         "worst.to_csv(RES / 'error_cases.csv', index=False); worst"),
]

# ---------------------------------------------------------------------------------------------- K6 agent
NB["K6_agent_eval.ipynb"] = [
    md("# K6 · Đánh giá Agent (T12): kịch bản đặt bàn nhiều lượt + định tuyến công cụ\n"
       "Kịch bản: `data/splits/scenarios.jsonl` (viết tay). Nếu chưa có → dùng kịch bản **proxy** dựng từ val v2 (chỉ kiểm tra "
       "pipeline, không phải số chính thức). Công cụ chạy trên bản sao DB. Định tuyến công cụ: đọc file `gen/*_agent_*.jsonl` "
       "của K5 (Add Data: output K5) hoặc tự chạy trên test_synth."),
    code(INSTALL),
    code(SETUP),
    code("from kimochi import llm, agent_eval as E, prompts, retriever, agent\nimport shutil\n"
         "ADAPTER = None   # None = tự tìm adapter tốt nhất trong output K4\nMODEL, QUANT = 'qwen', 'nf4'\n"
         "if ADAPTER is None:\n    found = sorted(glob.glob('/kaggle/input/**/runs/*_s42/adapter', recursive=True))\n"
         "    ADAPTER = found[0] if found else None\n    if ADAPTER: MODEL, QUANT = Path(ADAPTER).parent.name.split('_')[:2]\n"
         "print(MODEL, QUANT, ADAPTER)\nmodel, tok, _ = llm.load(MODEL, QUANT, adapter=ADAPTER)\n"
         "gen = lambda msgs: llm.generate(model, tok, [msgs], bs=1)[0][0]"),
    md("## Kịch bản nhiều lượt"),
    code("sc_path = ROOT / 'data/splits/scenarios.jsonl'\nif sc_path.exists():\n    scs = [json.loads(l) for l in sc_path.read_text(encoding='utf-8').splitlines() if l.strip()]; kind = 'viết tay'\n"
         "else:\n    scs = E.scenarios_from_v2_val(ROOT); kind = 'PROXY (val v2)'\nprint(len(scs), 'kịch bản', kind)\n"
         "W = Path('/kaggle/working/sc'); W.mkdir(exist_ok=True)\nres = pd.DataFrame([E.run_scenario(sc, gen, ROOT / 'data/kimochi.db', W) for sc in scs])\n"
         "res['kind'] = kind; res.to_json(RES / 'agent_scenarios.jsonl', orient='records', lines=True, force_ascii=False)\n"
         "summary = {'kind': kind, 'n': len(res), 'task_success': res.success.mean(), 'db_state_ok': res.db_state_ok.mean(),\n"
         "           'hitl_ok': res.hitl_ok.mean(), 'avg_tool_calls': res.n_tool_calls.mean()}\n"
         "(RES / 'agent_summary.json').write_text(json.dumps(summary, indent=1, default=float)); summary"),
    code("res[~res.success][['id', 'db_state_ok', 'hitl_ok', 'tools', 'answers']].head(10)  # phân tích lỗi"),
    md("## Định tuyến công cụ (confusion matrix + chi phí kinh doanh)"),
    code("files = glob.glob('/kaggle/input/**/gen/*_agent_test_*.jsonl', recursive=True)\n"
         "if files:\n    ag = pd.read_json(sorted(files)[0], lines=True); src = Path(sorted(files)[0]).name\n"
         "else:\n    ev = prompts.load_eval_set('test_synth')\n    R = retriever.HybridRetriever(retriever.load_docs())\n"
         "    db = Path('/kaggle/working/route.db'); shutil.copy(ROOT / 'data/kimochi.db', db)\n"
         "    out = agent.run_batch(model, tok, list(ev['query']), db, contexts=[R.search_chunks(q, 'bm25_filter', k=3) for q in ev['query']])\n"
         "    ag = ev.assign(tools_called=[o['tools_called'] for o in out]); src = 'test_synth (chạy trực tiếp)'\n"
         "rep = E.routing_report(ag)\nprint(src, '| accuracy', round(rep['accuracy'], 3), '| macro-F1', round(rep['macro_f1'], 3), '| chi phí / 100 câu', round(rep['cost_per_100'], 1))\n"
         "rep['confusion'].to_csv(RES / 'tool_routing_confusion.csv'); rep['per_label'].to_csv(RES / 'tool_routing_per_label.csv')\n"
         "(RES / 'tool_routing_summary.json').write_text(json.dumps({k: rep[k] for k in ('accuracy', 'macro_f1', 'business_cost', 'cost_per_100')} | {'source': src}, indent=1))\n"
         "E.plot_confusion(rep['confusion'], RES / 'tool_routing_confusion.png'); display(rep['confusion']); rep['per_label'].round(3)"),
]

# ---------------------------------------------------------------------------------------------- K7 demo
NB["K7_demo.ipynb"] = [
    md("# K7 · Chạy demo Gradio với mô hình tinh chỉnh tốt nhất (T14)\n"
       "Tạo link công khai `*.gradio.live` (hiệu lực 72 giờ) để quay video 2–3 phút hoặc cho người dùng thật thử (T15). "
       "Add Data: dataset repo + output K4 (hoặc đặt `ADAPTER` = HF repo). Kịch bản quay: `demo/README.md`."),
    code(INSTALL + "\n!pip install -q gradio==5.49.1"),
    code(SETUP),
    code("ADAPTER = None   # vd '/kaggle/input/kimochi-k4/runs/qwen_nf4_s42/adapter' hoặc 'user/kimochi-qwen-nf4-s42'\n"
         "MODEL, QUANT = 'qwen', 'nf4'\n"
         "if ADAPTER is None:\n    found = sorted(glob.glob('/kaggle/input/**/runs/*_s42/adapter', recursive=True))\n"
         "    ADAPTER = found[0] if found else None\nprint('Adapter:', ADAPTER)\n"
         "os.environ.update({'KIMOCHI_BACKEND': 'hf', 'KIMOCHI_MODEL': MODEL, 'KIMOCHI_QUANT': QUANT, 'KIMOCHI_ADAPTER': ADAPTER or ''})"),
    code("sys.path.insert(0, str(ROOT / 'demo'))\nimport app\napp.demo.queue().launch(share=True, debug=True)  # dừng ô này để tắt demo"),
]


if __name__ == "__main__":
    for name, cells in NB.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata.update({"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}})
        nbf.write(nb, OUT / name)
        print("wrote", name)
