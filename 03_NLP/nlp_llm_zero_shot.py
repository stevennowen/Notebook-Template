# -*- coding: utf-8 -*-
# =====================================================================================================
# nlp_llm_zero_shot.py — versi SCRIPT dari nlp_llm_zero_shot.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python nlp_llm_zero_shot.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb nlp_llm_zero_shot.py
# =====================================================================================================
import os as _os
import sys as _sys

_os.environ.setdefault("MPLBACKEND", "Agg")      # mode script: figure disimpan, tidak membuka jendela
for _st in (_sys.stdout, _sys.stderr):           # console Windows (cp1252) tidak bisa mencetak emoji → UTF-8
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
try:                                             # display() hanya ada di Jupyter → di script = print
    from IPython.display import display  # noqa: F401
except Exception:
    display = print
# ---- akhir header ----

# %% [markdown]
# # LLM Lokal — Zero/Few-shot Classification & Ekstraksi
#
# Kapan dipakai: data berlabel **sangat sedikit/tidak ada**, label baru yang dijelaskan dengan kata-kata, atau soal
# ekstraksi informasi (ambil nama produk/harga/lokasi dari teks) — juga untuk soal olimpiade yang memang meminta LLM
# (IOAI *Concepts*, AICC *Scientific Facts*). Semua berjalan **lokal** (tanpa API), sesuai aturan lomba offline.
#
# ## Pipeline
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json) → data test (+ train berlabel opsional)
# → Prompt: instruksi + daftar label (+ deskripsi label) + contoh few-shot (seimbang / paling mirip lewat TF-IDF)
# → Klasifikasi dengan SKOR LABEL: log-probabilitas tiap label sebagai jawaban (bukan parsing teks bebas → selalu valid)
# → Kalibrasi bias label (contextual calibration: input kosong "N/A")
# → Evaluasi di sampel train berlabel vs baseline TF-IDF + Logistic Regression (CV) → pilih / ensemble
# → (Opsional) probe: embedding LLM + Logistic Regression (kalau label cukup)
# → Mode ekstraksi: generate JSON terstruktur → parse aman → tabel
# → Submission + ringkasan laporan; resep LoRA fine-tune
# ```
#
# **Dua format:** `nlp_llm_zero_shot.ipynb` / `.py`. Headless: `python tools/run_task.py --task llm --format py`.
# **Kalau label train > ±500 per kelas**, model biasa (`nlp_classification`) hampir selalu lebih baik & lebih cepat.

# %%
import os


class CFG:
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "nlp_llm_zero_shot"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"
    TASK = "classify"              # "classify" | "extract"
    TRAIN_PATH = "./data/train.csv"            # opsional (contoh few-shot + evaluasi)
    TEST_PATH = "./data/test.csv"
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    ID_COL = "id"
    TEXT_COL = "text"
    TARGET_COL = "label"
    LABELS = None                  # None = label unik di train; atau isi manual, mis. ["positif", "negatif", "netral"]
    LABEL_DESCRIPTIONS = {}        # opsional: {"netral": "tidak memuji maupun mengeluh"}
    TASK_INSTRUCTION = "Classify the text into exactly one of the categories."
    EXTRACT_FIELDS = {"product": "nama produk", "price": "harga dalam rupiah (angka)", "city": "kota"}   # TASK="extract"
    OUTPUT_DIR = "./outputs/nlp_llm_zero_shot/"
    LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # kecil & multibahasa; alternatif: "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-3B-Instruct"
    N_SHOTS = 2                    # contoh per label di prompt (0 = zero-shot)
    SHOT_SELECTION = "similar"     # "balanced" | "similar" (TF-IDF nearest neighbours per teks)
    CALIBRATE = True
    MAX_TEXT_CHARS = 1200
    BATCH_SIZE = 8
    MAX_NEW_TOKENS = 128           # mode extract
    USE_GPU = True
    DTYPE = "auto"                 # "auto" | "float16" | "bfloat16" | "float32"
    EVAL_SAMPLE = 300              # jumlah contoh train berlabel untuk evaluasi LLM
    RUN_BASELINE = True
    RUN_EMBED_PROBE = True         # embedding rata-rata LLM + Logistic Regression (CV)
    ENSEMBLE = True                # rata-rata probabilitas LLM & baseline kalau lebih baik di evaluasi
    FAST_TEST_ROWS = 40
    FAST_EVAL_SAMPLE = 40
    FIG_DPI = 150

# %%
# >>> USER OVERRIDE
# CFG.LABELS = ["positif", "negatif", "netral"]
# CFG.LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# CFG.RUN_MODE = "fast"
pass

# %%
# >>> CONFIG FILE / HEADLESS RUNNER — jangan diedit
# tools/run_task.py mengisi env TEMPLATE_CONFIG (JSON). Kalau notebook/script dijalankan dari folder kerja yang berisi
# task_config.json (dibuat tools/new_task.py), file itu dipakai otomatis. Nilai di JSON MENIMPA CFG & USER OVERRIDE.
import json as _json
import os as _os


def apply_template_config(cfg, filename="task_config.json"):
    """Terapkan config JSON ke CFG. Baris yang diawali // = komentar. Key berawalan _ diabaikan."""
    path = _os.environ.get("TEMPLATE_CONFIG") or (filename if _os.path.exists(filename) else None)
    if _os.environ.get("TEMPLATE_AUTO_INSTALL", "1") == "0" and hasattr(cfg, "AUTO_INSTALL"):
        cfg.AUTO_INSTALL = False
    if not path:
        print("Config file: tidak dipakai (USER OVERRIDE di atas yang berlaku).")
        return {}
    if not _os.path.exists(path):
        raise FileNotFoundError(f"Config file tidak ditemukan: {path}")
    with open(path, encoding="utf-8") as f:
        txt = "\n".join(l for l in f.read().splitlines() if not l.strip().startswith("//"))
    conf = _json.loads(txt) if txt.strip() else {}
    applied = {k: v for k, v in conf.items() if not str(k).startswith("_")}
    unknown = [k for k in applied if not hasattr(cfg, k)]
    for k, v in applied.items():
        setattr(cfg, k, v)
    print(f"Config file: {path} → {len(applied)} key diterapkan ke CFG")
    for k, v in applied.items():
        print(f"  CFG.{k} = {v!r}"[:160])
    if unknown:
        print("  ⚠️ key tidak dikenal (tetap di-set, cek ejaan):", unknown)
    return applied


TEMPLATE_CONFIG_APPLIED = apply_template_config(CFG)

# %% [markdown]
# # 1. Install & import
#
# Butuh `torch` (manual, sesuai GPU) + `transformers` + `accelerate`. Model di-download dari HuggingFace sekali
# (1.5B parameter ≈ 3 GB) — **siapkan sebelum hari-H** kalau lokasi lomba tanpa internet.
#
# | Hardware | Model yang realistis |
# |---|---|
# | CPU saja | 0.5B–1.5B (lambat: ± 1–3 detik per teks) |
# | GPU 6–8 GB | 1.5B–3B (float16) |
# | GPU ≥ 12 GB | 7B–8B (float16) |

# %%
import importlib.util
import subprocess
import sys

for imp, pip_name in {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "matplotlib": "matplotlib",
                      "transformers": "transformers", "accelerate": "accelerate"}.items():
    if importlib.util.find_spec(imp) is None and CFG.AUTO_INSTALL:
        try:
            print("Installing", pip_name)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as e:  # noqa
            print("install gagal:", pip_name, e)

import json
import random
import re
import time
import warnings

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except Exception:  # noqa
    display = print
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if CFG.USE_GPU and torch.cuda.is_available() else "cpu"
if CFG.RUN_MODE == "fast":
    print(f"[FAST MODE] test ≤ {CFG.FAST_TEST_ROWS}, evaluasi ≤ {CFG.FAST_EVAL_SAMPLE}")
OUT = CFG.OUTPUT_DIR
FIG_DIR, TAB_DIR, SUB_DIR = (os.path.join(OUT, d) for d in ("figures", "tables", "submissions"))
for d in (OUT, FIG_DIR, TAB_DIR, SUB_DIR):
    os.makedirs(d, exist_ok=True)
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
_FIG = [0]


def save_fig(name, fig=None):
    fig = fig or plt.gcf()
    _FIG[0] += 1
    p = os.path.join(FIG_DIR, f"fig{_FIG[0]:02d}_{name}.png")
    fig.savefig(p, dpi=CFG.FIG_DPI, bbox_inches="tight")
    print("  [fig] saved:", p)
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)


def save_table(df, name):
    p = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(p, index=False)
    print("  [table] saved:", p)


def smart_read(path, **kw):
    """Baca tabel (csv/tsv/txt/xlsx/parquet/json). Separator CSV dipilih dari baris header (aman untuk CSV 1 kolom)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, **kw)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".json", ".jsonl"):
        return pd.read_json(path, lines=ext == ".jsonl")
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                head = f.readline()
            seps = [",", ";", "\t", "|"]
            sep = max(seps, key=head.count) if any(s in head for s in seps) else ","
            return pd.read_csv(path, sep=sep, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kw)

# %% [markdown]
# # 2. Data

# %%
train = smart_read(CFG.TRAIN_PATH) if CFG.TRAIN_PATH and os.path.exists(CFG.TRAIN_PATH) else None
test = smart_read(CFG.TEST_PATH)
HAS_LABELS = train is not None and CFG.TARGET_COL in train.columns
if HAS_LABELS:
    train = train.dropna(subset=[CFG.TEXT_COL, CFG.TARGET_COL]).reset_index(drop=True)
    train[CFG.TARGET_COL] = train[CFG.TARGET_COL].astype(str)
LABELS = [str(l) for l in (CFG.LABELS or (sorted(train[CFG.TARGET_COL].unique()) if HAS_LABELS else []))]
if CFG.TASK == "classify" and not LABELS:
    raise ValueError("Isi CFG.LABELS (daftar kategori) — tidak ada train berlabel untuk menebaknya.")
if CFG.RUN_MODE == "fast":
    test = test.head(CFG.FAST_TEST_ROWS)
    CFG.EVAL_SAMPLE = min(CFG.EVAL_SAMPLE, CFG.FAST_EVAL_SAMPLE)
clean = lambda t: re.sub(r"\s+", " ", str(t)).strip()[:CFG.MAX_TEXT_CHARS]  # noqa: E731
print(f"test {len(test)} | train berlabel {len(train) if HAS_LABELS else 0} | label: {LABELS}")
if HAS_LABELS:
    display(train[CFG.TARGET_COL].value_counts().to_frame("n"))

# %% [markdown]
# # 3. Muat LLM
#
# `DTYPE="auto"` → float16 di GPU (bfloat16 kalau didukung), float32 di CPU.

# %%
t0 = time.time()
tok = AutoTokenizer.from_pretrained(CFG.LLM_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(CFG.DTYPE)
if dtype is None:
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if DEVICE == "cuda" else torch.float32
import transformers as _tf
_new_api = tuple(int(x) for x in re.findall(r"\d+", _tf.__version__)[:2]) >= (4, 56)   # argumen `dtype` (lama: `torch_dtype`)
llm = AutoModelForCausalLM.from_pretrained(CFG.LLM_MODEL, **({"dtype": dtype} if _new_api else {"torch_dtype": dtype})).to(DEVICE).eval()
print(f"LLM: {CFG.LLM_MODEL} | {sum(p.numel() for p in llm.parameters()) / 1e9:.2f} B parameter | {dtype} | {DEVICE} | {time.time() - t0:.0f}s")
HAS_CHAT = bool(getattr(tok, "chat_template", None))

# %% [markdown]
# # 4. Prompt & skor label
#
# Untuk setiap teks, prompt meminta satu kategori. Lalu **setiap label** ditempelkan sebagai jawaban dan dihitung
# rata-rata log-probabilitas token-tokennya → softmax antar label = probabilitas. Jawaban selalu salah satu label
# (tidak perlu parsing). **Kalibrasi**: skor label untuk input kosong ("N/A") dikurangkan → mengurangi bias model
# yang suka label tertentu.

# %%
def label_block():
    return "\n".join(f"- {l}" + (f": {CFG.LABEL_DESCRIPTIONS[l]}" if l in CFG.LABEL_DESCRIPTIONS else "") for l in LABELS)


def build_prompt(text, shots):
    sys_msg = f"{CFG.TASK_INSTRUCTION}\nCategories:\n{label_block()}\nAnswer with the category name only."
    msgs = [{"role": "system", "content": sys_msg}]
    for t, l in shots:
        msgs += [{"role": "user", "content": f"Text: {clean(t)}"}, {"role": "assistant", "content": l}]
    msgs.append({"role": "user", "content": f"Text: {clean(text)}"})
    if HAS_CHAT:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa
            pass
    s = sys_msg + "\n\n" + "".join(f"Text: {clean(t)}\nCategory: {l}\n\n" for t, l in shots)
    return s + f"Text: {clean(text)}\nCategory:"


if HAS_LABELS and CFG.N_SHOTS > 0:
    pool = train.sample(frac=1, random_state=CFG.SEED)
    tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(pool[CFG.TEXT_COL].astype(str))
    pool_vec = tfv.transform(pool[CFG.TEXT_COL].astype(str))


def pick_shots(text, exclude_idx=None):
    if not HAS_LABELS or CFG.N_SHOTS <= 0:
        return []
    shots = []
    if CFG.SHOT_SELECTION == "similar":
        sims = (pool_vec @ tfv.transform([str(text)]).T).toarray().ravel()
        order = np.argsort(-sims)
    else:
        order = np.arange(len(pool))
    for lab in LABELS:
        cand = [i for i in order if pool[CFG.TARGET_COL].iloc[i] == lab and pool.index[i] != exclude_idx]
        shots += [(pool[CFG.TEXT_COL].iloc[i], lab) for i in cand[:CFG.N_SHOTS]]
    random.Random(hash(str(text)) % 1000).shuffle(shots)
    return shots


LABEL_IDS = [tok(" " + l if not HAS_CHAT else l, add_special_tokens=False)["input_ids"] for l in LABELS]


@torch.no_grad()
def label_scores(prompts):
    """Return (n, K) rata-rata log-prob token label (dengan semua label untuk setiap prompt)."""
    out = np.zeros((len(prompts), len(LABELS)), np.float32)
    seqs, owners = [], []
    for i, p in enumerate(prompts):
        pid = tok(p, add_special_tokens=False)["input_ids"]
        for k, lid in enumerate(LABEL_IDS):
            seqs.append((pid, lid))
            owners.append((i, k))
    for b in range(0, len(seqs), CFG.BATCH_SIZE * max(1, len(LABELS))):
        chunk = seqs[b:b + CFG.BATCH_SIZE * max(1, len(LABELS))]
        ids = [p + l for p, l in chunk]
        L = max(map(len, ids))
        inp = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for j, s in enumerate(ids):
            inp[j, L - len(s):] = torch.tensor(s)
            att[j, L - len(s):] = 1
        logits = llm(input_ids=inp.to(DEVICE), attention_mask=att.to(DEVICE)).logits.float()
        logp = F.log_softmax(logits[:, :-1], -1)
        tgt = inp[:, 1:].to(DEVICE)
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for j, (p, l) in enumerate(chunk):
            n_l = len(l)
            i, k = owners[b + j]
            out[i, k] = tok_lp[j, -n_l:].mean().item()
    return out


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


CAL = np.zeros(len(LABELS), np.float32)
if CFG.TASK == "classify" and CFG.CALIBRATE:
    CAL = label_scores([build_prompt(c, pick_shots(c) if CFG.SHOT_SELECTION != "similar" else []) for c in ["N/A", "", "[MASK]"]]).mean(0)
    print("Bias label (log-prob input kosong):", dict(zip(LABELS, np.round(CAL, 3))))


def llm_predict(texts, exclude=None):
    t0_ = time.time()
    prompts = [build_prompt(t, pick_shots(t, exclude[i] if exclude is not None else None)) for i, t in enumerate(texts)]
    s = label_scores(prompts)
    raw = softmax(s)
    cal = softmax(s - CAL)
    print(f"  {len(texts)} teks diproses dalam {time.time() - t0_:.0f}s ({(time.time() - t0_) / max(1, len(texts)):.2f}s/teks)")
    return raw, cal


if CFG.TASK == "classify":
    print("Contoh prompt:\n" + build_prompt(test[CFG.TEXT_COL].iloc[0], pick_shots(test[CFG.TEXT_COL].iloc[0]))[:1200])

# %% [markdown]
# # 5. Evaluasi di data berlabel (kalau ada)
#
# Contoh few-shot untuk sebuah teks evaluasi **tidak boleh** teks itu sendiri (dikecualikan) → evaluasi jujur.
# Pembanding: TF-IDF + Logistic Regression dengan cross-validation pada seluruh train.

# %%
RESULTS = {}
if CFG.TASK == "classify" and HAS_LABELS:
    ev = train.sample(min(CFG.EVAL_SAMPLE, len(train)), random_state=CFG.SEED)
    y_ev = ev[CFG.TARGET_COL].map({l: i for i, l in enumerate(LABELS)}).values
    p_raw, p_cal = llm_predict(ev[CFG.TEXT_COL].tolist(), exclude=ev.index.tolist())
    RESULTS["llm_raw"], RESULTS["llm_calibrated"] = p_raw, p_cal
    if CFG.RUN_BASELINE and train[CFG.TARGET_COL].value_counts().min() >= 2:
        n_sp = int(min(5, train[CFG.TARGET_COL].value_counts().min()))
        base = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1), LogisticRegression(max_iter=2000, C=4))
        oof = cross_val_predict(base, train[CFG.TEXT_COL].astype(str), train[CFG.TARGET_COL], cv=StratifiedKFold(n_sp, shuffle=True, random_state=CFG.SEED),
                                method="predict_proba")
        cls_order = sorted(train[CFG.TARGET_COL].unique())
        oof = pd.DataFrame(oof, columns=cls_order).reindex(columns=LABELS, fill_value=0).values
        RESULTS["tfidf_logreg"] = oof[train.index.get_indexer(ev.index)]
    if CFG.RUN_EMBED_PROBE and len(train) >= 20:
        @torch.no_grad()
        def llm_embed(texts):
            embs = []
            for i in range(0, len(texts), CFG.BATCH_SIZE):
                enc = tok([clean(t) for t in texts[i:i + CFG.BATCH_SIZE]], return_tensors="pt", padding=True, truncation=True, max_length=256)
                h = llm(**{k: v.to(DEVICE) for k, v in enc.items()}, output_hidden_states=True).hidden_states[-1].float()
                m = enc["attention_mask"].to(DEVICE).unsqueeze(-1)
                embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
            return np.concatenate(embs)
        E = llm_embed(train[CFG.TEXT_COL].tolist())
        n_sp = int(min(5, train[CFG.TARGET_COL].value_counts().min()))
        if n_sp >= 2:
            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
            oofp = cross_val_predict(probe, E, train[CFG.TARGET_COL], cv=StratifiedKFold(n_sp, shuffle=True, random_state=CFG.SEED), method="predict_proba")
            oofp = pd.DataFrame(oofp, columns=sorted(train[CFG.TARGET_COL].unique())).reindex(columns=LABELS, fill_value=0).values
            RESULTS["llm_embed_probe"] = oofp[train.index.get_indexer(ev.index)]
    if CFG.ENSEMBLE and "tfidf_logreg" in RESULTS:
        RESULTS["ensemble_llm_tfidf"] = (RESULTS["llm_calibrated"] + RESULTS["tfidf_logreg"]) / 2
    board = pd.DataFrame([{"model": k, "accuracy": accuracy_score(y_ev, v.argmax(1)), "f1_macro": f1_score(y_ev, v.argmax(1), average="macro")}
                          for k, v in RESULTS.items()]).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    display(board.round(4))
    save_table(board, "leaderboard")
    FINAL = board["model"].iloc[0]
    cm = confusion_matrix(y_ev, RESULTS[FINAL].argmax(1), labels=range(len(LABELS)))
    fig, ax = plt.subplots(figsize=(1.5 + .6 * len(LABELS), 1.2 + .6 * len(LABELS)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, fontsize=8)
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_title(f"Confusion matrix ({FINAL})")
    save_fig("confusion_matrix", fig)
else:
    FINAL = "llm_calibrated" if CFG.CALIBRATE else "llm_raw"
print("Metode final:", FINAL if CFG.TASK == "classify" else "extract")

# %% [markdown]
# # 6. Prediksi test (classify) / ekstraksi (extract)

# %%
@torch.no_grad()
def generate(prompts):
    outs = []
    for i in range(0, len(prompts), CFG.BATCH_SIZE):
        enc = tok(prompts[i:i + CFG.BATCH_SIZE], return_tensors="pt", padding=True, add_special_tokens=not HAS_CHAT)
        g = llm.generate(**{k: v.to(DEVICE) for k, v in enc.items()}, max_new_tokens=CFG.MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def parse_json(s):
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group())
        except Exception:  # noqa
            pass
    out = {}
    for k in CFG.EXTRACT_FIELDS:
        mm = re.search(rf'"?{k}"?\s*[:=]\s*"?([^",\n}}]+)', s)
        out[k] = mm.group(1).strip() if mm else None
    return out


SUB_PATH = None
texts = test[CFG.TEXT_COL].astype(str).tolist()
ids = test[CFG.ID_COL].values if CFG.ID_COL in test.columns else np.arange(len(test))
if CFG.TASK == "classify":
    p_raw, p_cal = llm_predict(texts)
    P = {"llm_raw": p_raw, "llm_calibrated": p_cal}.get(FINAL)
    if P is None:
        base_te = None
        if FINAL in ("tfidf_logreg", "ensemble_llm_tfidf"):
            base.fit(train[CFG.TEXT_COL].astype(str), train[CFG.TARGET_COL])
            base_te = pd.DataFrame(base.predict_proba(texts), columns=base.classes_).reindex(columns=LABELS, fill_value=0).values
        if FINAL == "tfidf_logreg":
            P = base_te
        elif FINAL == "ensemble_llm_tfidf":
            P = (p_cal + base_te) / 2
        elif FINAL == "llm_embed_probe":
            probe.fit(E, train[CFG.TARGET_COL])
            P = pd.DataFrame(probe.predict_proba(llm_embed(texts)), columns=probe.classes_).reindex(columns=LABELS, fill_value=0).values
    pred = np.array(LABELS, dtype=object)[P.argmax(1)]
    sample = smart_read(CFG.SAMPLE_SUB_PATH) if os.path.exists(CFG.SAMPLE_SUB_PATH) else None
    id_name = sample.columns[0] if sample is not None else CFG.ID_COL
    val_name = sample.columns[1] if sample is not None and sample.shape[1] > 1 else CFG.TARGET_COL
    sub = pd.DataFrame({id_name: ids, val_name: pred})
    if sample is not None and CFG.RUN_MODE != "fast":
        sub = sample[[id_name]].merge(sub, on=id_name, how="left")
    SUB_PATH = os.path.join(SUB_DIR, "submission.csv")
    sub.to_csv(SUB_PATH, index=False)
    pd.DataFrame(P, columns=LABELS).assign(**{id_name: ids}).to_csv(os.path.join(TAB_DIR, "test_probabilities.csv"), index=False)
    print(f"✅ Submission: {SUB_PATH} | {len(sub)} baris | distribusi: {pd.Series(pred).value_counts().to_dict()}")
else:
    fields = "\n".join(f'- "{k}": {v}' for k, v in CFG.EXTRACT_FIELDS.items())
    sysmsg = f"Extract the following fields from the text and answer ONLY with a JSON object (use null if absent):\n{fields}"
    prompts = []
    for t in texts:
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": clean(t)}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) if HAS_CHAT else
                       f"{sysmsg}\n\nText: {clean(t)}\nJSON:")
    raw_out = generate(prompts)
    rows = [{CFG.ID_COL: i, **{k: parse_json(o).get(k) for k in CFG.EXTRACT_FIELDS}, "_raw": o} for i, o in zip(ids, raw_out)]
    ext = pd.DataFrame(rows)
    SUB_PATH = os.path.join(SUB_DIR, "extraction.csv")
    ext.drop(columns="_raw").to_csv(SUB_PATH, index=False)
    ext.to_csv(os.path.join(TAB_DIR, "extraction_with_raw.csv"), index=False)
    print(f"✅ Ekstraksi: {SUB_PATH} | kolom terisi: {ext.drop(columns=[CFG.ID_COL, '_raw']).notna().mean().round(2).to_dict()}")
    display(ext.head())

# %% [markdown]
# # 7. Ringkasan untuk laporan

# %%
summ = {"task": CFG.TASK, "llm": CFG.LLM_MODEL, "n_shots": CFG.N_SHOTS, "shot_selection": CFG.SHOT_SELECTION,
        "calibrated": CFG.CALIBRATE, "final": FINAL if CFG.TASK == "classify" else "extract", "n_test": len(test), "submission": SUB_PATH}
if CFG.TASK == "classify" and HAS_LABELS:
    summ.update({"eval_n": len(ev), "eval_f1_macro": round(float(board.f1_macro.iloc[0]), 4),
                 "leaderboard": board.round(4).to_dict("records")})
text = f"""# Local LLM summary

- Model **{CFG.LLM_MODEL}**, {CFG.N_SHOTS}-shot per label ({CFG.SHOT_SELECTION} examples), label-likelihood scoring{' with contextual calibration' if CFG.CALIBRATE else ''}.
{('- Evaluation on ' + str(summ.get('eval_n')) + ' labelled texts: best = **' + str(FINAL) + '**, macro F1 ' + str(summ.get('eval_f1_macro')) + '.') if 'eval_n' in summ else '- No labelled data: predictions are zero-shot.'}

**Draft sentences (paraphrase):**
- "Instead of parsing free-form generations, we scored every candidate label by its average token log-probability under the prompt, guaranteeing a valid label for every text."
- "Label bias was reduced with contextual calibration, subtracting the scores obtained for a content-free input."
"""
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(text)
json.dump(summ, open(os.path.join(OUT, "run_summary.json"), "w"), indent=2, default=str)
print(text)
print("✅ Notebook selesai.")

# %% [markdown]
# # 8. Troubleshooting, recipes (LoRA), checklist
#
# | Gejala | Solusi |
# |---|---|
# | Model tidak bisa di-download | siapkan cache HuggingFace sebelum lomba (`huggingface-cli download <model>`) dan salin folder `~/.cache/huggingface` |
# | `CUDA out of memory` | model lebih kecil, `BATCH_SIZE=2`, `MAX_TEXT_CHARS=600` |
# | Semua prediksi satu label | `CALIBRATE=True`, tambah `LABEL_DESCRIPTIONS`, `N_SHOTS=3` |
# | LLM kalah dari TF-IDF | wajar kalau label train banyak → pakai ensemble atau `nlp_classification` |
# | Lambat di CPU | model 0.5B (`Qwen/Qwen2.5-0.5B-Instruct`), `N_SHOTS=1`, `SHOT_SELECTION="balanced"` |
#
# **Recipe — LoRA fine-tune** (butuh `pip install peft`, GPU; tidak dijalankan otomatis oleh template ini):
#
# ```python
# from peft import LoraConfig, get_peft_model
# lora = get_peft_model(llm.train(), LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
#                                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
# opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=2e-4)
# for epoch in range(2):
#     for text, lab in zip(train[CFG.TEXT_COL], train[CFG.TARGET_COL]):
#         p_ids = tok(build_prompt(text, []), add_special_tokens=False)["input_ids"]
#         l_ids = tok(lab, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
#         ids = torch.tensor([p_ids + l_ids], device=DEVICE)
#         labels = torch.tensor([[-100] * len(p_ids) + l_ids], device=DEVICE)   # loss hanya di jawaban
#         loss = lora(input_ids=ids, labels=labels).loss
#         loss.backward(); opt.step(); opt.zero_grad()
# llm = lora.eval()   # lalu jalankan ulang section 5–6 (skor label)
# ```
#
# - [ ] Model & versi dicatat di laporan (dan diizinkan aturan lomba)
# - [ ] Evaluasi memakai contoh few-shot yang tidak sama dengan teks yang dinilai
# - [ ] Label submission persis sama penulisannya dengan aturan
