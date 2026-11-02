# -*- coding: utf-8 -*-
# =====================================================================================================
# nlp_token_classification.py — versi SCRIPT dari nlp_token_classification.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python nlp_token_classification.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb nlp_token_classification.py
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
# # NLP Token Classification — NER / Slot Filling / POS (label per kata)
#
# Untuk soal yang memberi label **setiap kata**: Named Entity Recognition (orang, lokasi, organisasi, produk),
# *slot filling* (NEOAI *Intent Detection and Slot Filling*), ekstraksi atribut dari ulasan/alamat, POS tagging.
# Skema label BIO: `B-LOC` = awal entitas lokasi, `I-LOC` = lanjutan, `O` = bukan entitas.
#
# ## Pipeline
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json)
# → Baca format apa pun: CoNLL (token<TAB>tag), JSONL tokens/tags, JSONL text+span, CSV panjang (kalimat, token, tag)
# → Normalisasi label → BIO (IOB1 / tag polos diubah otomatis) + cek label tidak valid
# → EDA: panjang kalimat, jumlah entitas per tipe, panjang entitas
# → Baseline cepat: fitur kata (huruf besar, prefix/suffix, bentuk, tetangga) + Logistic Regression
# → Transformer fine-tune (AutoModelForTokenClassification): label di sub-kata pertama, AMP, early stopping
# → Post-processing BIO (I- tanpa B- diperbaiki) → metrik **entity-level F1** (seqeval-style, strict) per tipe
# → Prediksi test → file submission (format sama dengan input / CSV) + ringkasan laporan
# ```
#
# **Dua format:** `nlp_token_classification.ipynb` / `.py`. Headless: `python tools/run_task.py --task ner --format py`.
# Untuk klasifikasi **satu label per kalimat** (intent, sentimen) pakai `nlp_classification`.

# %%
import os


class CFG:
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "nlp_token_classification"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"
    TRAIN_PATH = "./data/train.conll"
    TEST_PATH = "./data/test.conll"
    VALID_PATH = None              # kalau panitia memberi data validasi terpisah
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    FORMAT = "auto"                # "auto" | "conll" | "jsonl_tokens" | "jsonl_spans" | "csv_long"
    TOKEN_COL, TAG_COL, SENT_COL = None, None, None   # csv_long: None = auto
    OUTPUT_DIR = "./outputs/nlp_token_classification/"
    RUN_BASELINE = True
    RUN_TRANSFORMER = "auto"       # "auto" (kalau transformers + model tersedia) | True | False
    TRANSFORMER_MODELS = ["indobenchmark/indobert-base-p1"]   # Inggris: "bert-base-cased"; multibahasa: "xlm-roberta-base"
    MAX_LEN = 128
    EPOCHS = 5
    BATCH_SIZE = 16
    LR = 3e-5
    USE_AMP = True
    USE_GPU = True
    VAL_FRAC = 0.15
    PATIENCE = 2
    SUBMISSION_FORMAT = "auto"     # "auto" | "conll" | "csv_long" | "jsonl"
    FAST_MAX_SENT = 600
    FAST_EPOCHS = 2
    FIG_DPI = 150

# %%
# >>> USER OVERRIDE
# CFG.TRANSFORMER_MODELS = ["xlm-roberta-base"]
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
# Baseline hanya butuh scikit-learn. Transformer butuh `torch` (manual) + `transformers` (auto-install).
# Model dari HuggingFace di-download sekali (butuh internet) lalu tersimpan di cache.

# %%
import importlib.util
import subprocess
import sys

for imp, pip_name in {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "matplotlib": "matplotlib",
                      "transformers": "transformers"}.items():
    if importlib.util.find_spec(imp) is None and CFG.AUTO_INSTALL and (imp != "transformers" or importlib.util.find_spec("torch")):
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
from collections import Counter

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except Exception:  # noqa
    display = print
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TF = HAS_TORCH and importlib.util.find_spec("transformers") is not None
if HAS_TORCH:
    import torch
DEVICE = "cuda" if HAS_TORCH and CFG.USE_GPU and torch.cuda.is_available() else "cpu"
print(f"torch {HAS_TORCH} | transformers {HAS_TF} | device {DEVICE}")
if CFG.RUN_MODE == "fast":
    CFG.EPOCHS = min(CFG.EPOCHS, CFG.FAST_EPOCHS)
OUT = CFG.OUTPUT_DIR
FIG_DIR, TAB_DIR, SUB_DIR = (os.path.join(OUT, d) for d in ("figures", "tables", "submissions"))
for d in (OUT, FIG_DIR, TAB_DIR, SUB_DIR):
    os.makedirs(d, exist_ok=True)
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
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

# %% [markdown]
# # 2. Baca data (format otomatis)
#
# Hasil: daftar kalimat, masing-masing `{"id", "tokens": [...], "tags": [...] atau None}`.
#
# | Format | Contoh |
# |---|---|
# | CoNLL | `Budi\tB-PER` per baris, kalimat dipisah baris kosong (kolom terakhir = tag) |
# | JSONL tokens | `{"tokens": ["Budi", "ke", "Bandung"], "tags": ["B-PER", "O", "B-LOC"]}` |
# | JSONL spans | `{"text": "Budi ke Bandung", "entities": [{"start": 0, "end": 4, "label": "PER"}]}` |
# | CSV panjang | kolom `sentence_id, token, tag` (satu baris per token) |

# %%
def detect_format(path):
    if CFG.FORMAT != "auto":
        return CFG.FORMAT
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".json"):
        first = open(path, encoding="utf-8").readline()
        return "jsonl_spans" if '"text"' in first and ("entities" in first or "spans" in first or "labels" in first) else "jsonl_tokens"
    if ext in (".csv", ".tsv", ".xlsx"):
        return "csv_long"
    return "conll"


def read_conll(path):
    sents, toks, tags = [], [], []
    has_tags = True
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("-DOCSTART-"):
            if toks:
                sents.append((toks, tags if has_tags else None))
            toks, tags = [], []
            continue
        parts = re.split(r"\t| ", line.strip())
        toks.append(parts[0])
        if len(parts) > 1:
            tags.append(parts[-1])
        else:
            has_tags = False
    if toks:
        sents.append((toks, tags if has_tags else None))
    return [{"id": i, "tokens": t, "tags": g} for i, (t, g) in enumerate(sents)]


def read_jsonl(path, spans):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if path.endswith(".jsonl") else json.load(open(path, encoding="utf-8"))
    out = []
    for i, r in enumerate(rows):
        if not spans:
            tk = r.get("tokens") or r.get("words")
            tg = r.get("tags") or r.get("ner_tags") or r.get("labels")
            out.append({"id": r.get("id", i), "tokens": tk, "tags": tg})
            continue
        text = r["text"]
        ents = r.get("entities") or r.get("spans") or r.get("labels") or []
        toks, tags = [], []
        for m in re.finditer(r"\w+|[^\w\s]", text):
            lab = "O"
            for e in ents:
                s, e_, l = (e["start"], e["end"], e["label"]) if isinstance(e, dict) else e[:3]
                if m.start() >= s and m.end() <= e_:
                    lab = ("B-" if m.start() == s or not tags or not tags[-1].endswith(l) else "I-") + l
            toks.append(m.group())
            tags.append(lab)
        out.append({"id": r.get("id", i), "tokens": toks, "tags": tags if ents or "entities" in r else None, "text": text,
                    "offsets": [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]})
    return out


def read_csv_long(path):
    df = pd.read_csv(path, sep=None, engine="python") if not path.endswith(".xlsx") else pd.read_excel(path)
    low = {c.lower(): c for c in df.columns}
    sc = CFG.SENT_COL or next((low[c] for c in ("sentence_id", "sentence #", "sent_id", "id", "sentence") if c in low), df.columns[0])
    tc = CFG.TOKEN_COL or next((low[c] for c in ("token", "word", "tokens", "kata") if c in low), df.columns[1])
    gc = CFG.TAG_COL or next((low[c] for c in ("tag", "label", "ner", "ner_tag", "slot") if c in low), None)
    df[sc] = df[sc].ffill()
    return [{"id": sid, "tokens": g[tc].astype(str).tolist(), "tags": g[gc].astype(str).tolist() if gc else None}
            for sid, g in df.groupby(sc, sort=False)]


def load_any(path):
    fmt = detect_format(path)
    data = read_conll(path) if fmt == "conll" else read_csv_long(path) if fmt == "csv_long" else read_jsonl(path, fmt == "jsonl_spans")
    return data, fmt


def to_bio(tags):
    """IOB1 / tag polos ('PER') → BIO yang valid."""
    out, prev = [], "O"
    for t in tags:
        t = str(t)
        if t in ("O", "0", "", "nan"):
            out.append("O")
            prev = "O"
            continue
        if not re.match(r"^[BIES]-|^[BI]_", t):
            t = ("I-" if prev.endswith(t) and prev != "O" else "B-") + t
        t = t.replace("_", "-", 1) if re.match(r"^[BI]_", t) else t
        t = "B-" + t[2:] if t.startswith(("S-",)) else ("I-" + t[2:] if t.startswith("E-") else t)
        if t.startswith("I-") and (prev == "O" or prev[2:] != t[2:]):
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


train_sents, FMT = load_any(CFG.TRAIN_PATH)
test_sents = load_any(CFG.TEST_PATH)[0] if CFG.TEST_PATH and os.path.exists(CFG.TEST_PATH) else []
valid_sents = load_any(CFG.VALID_PATH)[0] if CFG.VALID_PATH and os.path.exists(CFG.VALID_PATH) else None
train_sents = [s for s in train_sents if s["tokens"] and s["tags"] and len(s["tokens"]) == len(s["tags"])]
for s in train_sents + (valid_sents or []):
    s["tags"] = to_bio(s["tags"])
if CFG.RUN_MODE == "fast" and len(train_sents) > CFG.FAST_MAX_SENT:
    train_sents = random.Random(CFG.SEED).sample(train_sents, CFG.FAST_MAX_SENT)
print(f"Format: {FMT} | kalimat train {len(train_sents)} | test {len(test_sents)} | valid terpisah: {valid_sents is not None}")
LABELS = sorted({t for s in train_sents for t in s["tags"]}, key=lambda t: (t != "O", t[2:], t[:2]))
L2I = {l: i for i, l in enumerate(LABELS)}
TYPES = sorted({t[2:] for t in LABELS if t != "O"})
print(f"Label ({len(LABELS)}): {LABELS}")

# %% [markdown]
# # 3. EDA

# %%
def spans(tags):
    """BIO → set (tipe, awal, akhir) — dipakai metrik entity-level."""
    out, cur = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        if cur and not (t.startswith("I-") and t[2:] == cur[0]):
            out.append((cur[0], cur[1], i))
            cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None):
            cur = (t[2:], i)
    return out


lens = [len(s["tokens"]) for s in train_sents]
ent = Counter(e[0] for s in train_sents for e in spans(s["tags"]))
ent_len = [e[2] - e[1] for s in train_sents for e in spans(s["tags"])]
print(f"Panjang kalimat: median {np.median(lens):.0f}, max {max(lens)} | entitas: {sum(ent.values())} | per tipe: {dict(ent)}")
save_table(pd.DataFrame(ent.most_common(), columns=["type", "count"]), "entity_type_counts")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.3))
ax[0].hist(lens, bins=30, color="#4C72B0")
ax[0].set_title("Panjang kalimat (token)")
ax[1].bar(list(ent), list(ent.values()), color="#55A868")
ax[1].set_title("Entitas per tipe")
ax[2].hist(ent_len, bins=range(1, max(ent_len + [1]) + 2), align="left", color="#C44E52")
ax[2].set_title("Panjang entitas (token)")
save_fig("eda", fig)
ex = train_sents[0]
print("Contoh:", " ".join(f"{w}/{t}" if t != "O" else w for w, t in zip(ex["tokens"], ex["tags"])))

# %% [markdown]
# # 4. Split & metrik entity-level
#
# Metrik standar NER = **F1 entitas strict**: prediksi benar hanya kalau tipe DAN batas awal–akhir entitas tepat
# (sama seperti `seqeval`). Akurasi per token menyesatkan karena mayoritas token = `O`.

# %%
if valid_sents:
    tr_s, va_s = train_sents, valid_sents
else:
    tr_s, va_s = train_test_split(train_sents, test_size=CFG.VAL_FRAC, random_state=CFG.SEED)
print(f"train {len(tr_s)} | valid {len(va_s)}")


def entity_scores(true_seqs, pred_seqs):
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(true_seqs, pred_seqs):
        ts, ps = set(spans(t)), set(spans(p))
        for e in ps & ts:
            tp[e[0]] += 1
        for e in ps - ts:
            fp[e[0]] += 1
        for e in ts - ps:
            fn[e[0]] += 1
    rows = []
    for ty in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[ty] / max(1, tp[ty] + fp[ty])
        r = tp[ty] / max(1, tp[ty] + fn[ty])
        rows.append({"type": ty, "precision": p, "recall": r, "f1": 2 * p * r / max(1e-9, p + r), "support": tp[ty] + fn[ty]})
    P = sum(tp.values()) / max(1, sum(tp.values()) + sum(fp.values()))
    R = sum(tp.values()) / max(1, sum(tp.values()) + sum(fn.values()))
    return {"f1": 2 * P * R / max(1e-9, P + R), "precision": P, "recall": R, "per_type": pd.DataFrame(rows)}


def fix_bio(tags):
    return to_bio(tags)

# %% [markdown]
# # 5. Baseline: fitur kata + Logistic Regression
#
# Fitur per token: kata (lowercase), awalan/akhiran 2–3 huruf, huruf kapital, angka, bentuk (`Xxxx`, `dd`), dan kata
# tetangga ±2. Cepat (detik) dan sering sudah kuat untuk entitas yang polanya jelas.

# %%
def shape(w):
    return re.sub(r"(.)\1+", r"\1\1", re.sub("[A-Z]", "X", re.sub("[a-z]", "x", re.sub(r"\d", "d", w))))


def tok_feats(toks, i):
    w = toks[i]
    f = {"w": w.lower(), "p2": w[:2].lower(), "p3": w[:3].lower(), "s2": w[-2:].lower(), "s3": w[-3:].lower(),
         "title": w.istitle(), "upper": w.isupper(), "digit": w.isdigit(), "shape": shape(w), "len": min(len(w), 10),
         "bos": i == 0, "eos": i == len(toks) - 1}
    for o in (-2, -1, 1, 2):
        j = i + o
        if 0 <= j < len(toks):
            f[f"w{o}"] = toks[j].lower()
            f[f"t{o}"] = toks[j].istitle()
    return f


def sent_feats(sents):
    return [tok_feats(s["tokens"], i) for s in sents for i in range(len(s["tokens"]))]


RESULTS = {}
if CFG.RUN_BASELINE:
    t0 = time.time()
    dv = DictVectorizer()
    Xb = dv.fit_transform(sent_feats(tr_s))
    yb = [t for s in tr_s for t in s["tags"]]
    base = LogisticRegression(max_iter=1000, C=2.0)
    base.fit(Xb, yb)

    def base_predict(sents):
        if not sents:
            return []
        flat = base.predict(dv.transform(sent_feats(sents)))
        out, k = [], 0
        for s in sents:
            out.append(fix_bio(list(flat[k:k + len(s["tokens"])])))
            k += len(s["tokens"])
        return out

    va_pred = base_predict(va_s)
    sc = entity_scores([s["tags"] for s in va_s], va_pred)
    RESULTS["baseline_logreg"] = {"f1": sc["f1"], "scores": sc, "predict": base_predict}
    print(f"==> baseline_logreg: entity F1 {sc['f1']:.4f} (P {sc['precision']:.3f} R {sc['recall']:.3f}) | {time.time() - t0:.0f}s")

# %% [markdown]
# # 6. Transformer fine-tuning
#
# Tokenizer memecah kata menjadi sub-kata; label diberikan ke **sub-kata pertama** setiap kata, sisanya `-100`
# (diabaikan loss). Prediksi kata = prediksi sub-kata pertama. Kalimat lebih panjang dari `MAX_LEN` dipotong
# (token sisanya diprediksi `O` — naikkan `MAX_LEN` kalau ini sering terjadi).

# %%
def model_available(name):
    if not HAS_TF:
        return False
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(name)
        return True
    except Exception as e:  # noqa
        print(f"  model {name} tidak tersedia (offline / nama salah): {str(e)[:100]}")
        return False


def run_transformer(name):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("butuh fast tokenizer (word_ids)")
    model = AutoModelForTokenClassification.from_pretrained(name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                                                            label2id=L2I, ignore_mismatched_sizes=True).to(DEVICE)

    def encode(sents, with_labels=True):
        enc = tok([s["tokens"] for s in sents], is_split_into_words=True, truncation=True, max_length=CFG.MAX_LEN)
        labels = []
        for i, s in enumerate(sents):
            wids, prev, lab = enc.word_ids(i), None, []
            for w in wids:
                lab.append(-100 if w is None or w == prev or not with_labels else L2I.get(s["tags"][w], 0))
                prev = w
            labels.append(lab)
        return enc, labels

    def batches(sents, shuffle):
        idx = list(range(len(sents)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), CFG.BATCH_SIZE):
            b = [sents[j] for j in idx[i:i + CFG.BATCH_SIZE]]
            enc, lab = encode(b, b[0]["tags"] is not None)
            pad = tok.pad({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, return_tensors="pt")
            L = max(len(x) for x in lab)
            yield b, enc, pad, torch.tensor([x + [-100] * (L - len(x)) for x in lab])

    def predict(sents):
        model.eval()
        out = []
        with torch.no_grad():
            for b, enc, pad, _ in batches(sents, False):
                with torch.autocast(device_type="cuda", enabled=CFG.USE_AMP and DEVICE == "cuda"):
                    lo = model(**{k: v.to(DEVICE) for k, v in pad.items()}).logits.float()
                pr = lo.argmax(-1).cpu().numpy()
                for i, s in enumerate(b):
                    tags, seen = ["O"] * len(s["tokens"]), set()
                    for pos, w in enumerate(enc.word_ids(i)):
                        if w is not None and w not in seen:
                            tags[w] = LABELS[pr[i, pos]]
                            seen.add(w)
                    out.append(fix_bio(tags))
        return out

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=0.01)
    steps = CFG.EPOCHS * ((len(tr_s) + CFG.BATCH_SIZE - 1) // CFG.BATCH_SIZE)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG.LR, total_steps=max(1, steps), pct_start=0.1, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP and DEVICE == "cuda")
    best, best_state, wait = -1, None, 0
    for ep in range(CFG.EPOCHS):
        model.train()
        losses = []
        for _, _, pad, lab in batches(tr_s, True):
            with torch.autocast(device_type="cuda", enabled=CFG.USE_AMP and DEVICE == "cuda"):
                loss = model(**{k: v.to(DEVICE) for k, v in pad.items()}, labels=lab.to(DEVICE)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            losses.append(loss.item())
        sc = entity_scores([s["tags"] for s in va_s], predict(va_s))
        print(f"  [{name}] ep {ep + 1} loss {np.mean(losses):.4f} | val entity F1 {sc['f1']:.4f}")
        if sc["f1"] > best:
            best, wait = sc["f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG.PATIENCE:
                break
    model.load_state_dict(best_state)
    sc = entity_scores([s["tags"] for s in va_s], predict(va_s))
    return {"f1": sc["f1"], "scores": sc, "predict": predict}


use_tf = CFG.RUN_TRANSFORMER is True or (CFG.RUN_TRANSFORMER == "auto" and HAS_TF)
if use_tf:
    for name in CFG.TRANSFORMER_MODELS:
        if not model_available(name):
            continue
        t0 = time.time()
        try:
            RESULTS[f"tf_{name.split('/')[-1]}"] = run_transformer(name)
            print(f"==> tf_{name.split('/')[-1]}: entity F1 {RESULTS[f'tf_{name.split(chr(47))[-1]}']['f1']:.4f} | {time.time() - t0:.0f}s")
        except Exception as e:  # noqa
            print(f"❌ transformer {name} gagal: {type(e).__name__}: {str(e)[:200]}")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
else:
    print("ℹ️ Transformer dilewati (transformers/torch tidak ada atau RUN_TRANSFORMER=False).")

# %% [markdown]
# # 7. Evaluasi & pemilihan model
#
# Tabel per tipe entitas: tipe dengan recall rendah → entitas sering terlewat (tambah data/aturan gazetteer);
# precision rendah → banyak deteksi palsu.

# %%
if not RESULTS:
    raise RuntimeError("Tidak ada model yang berhasil.")
board = pd.DataFrame([{"model": k, "entity_f1": v["f1"], "precision": v["scores"]["precision"], "recall": v["scores"]["recall"]}
                      for k, v in RESULTS.items()]).sort_values("entity_f1", ascending=False).reset_index(drop=True)
display(board.round(4))
save_table(board, "leaderboard")
FINAL = board["model"].iloc[0]
per_type = RESULTS[FINAL]["scores"]["per_type"]
display(per_type.round(4))
save_table(per_type, "per_type_scores")
fig, ax = plt.subplots(figsize=(max(5, len(per_type) * .9), 3.3))
x = np.arange(len(per_type))
for j, m in enumerate(["precision", "recall", "f1"]):
    ax.bar(x + (j - 1) * .27, per_type[m], .27, label=m)
ax.set_xticks(x, per_type["type"])
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.set_title(f"Skor per tipe entitas (valid) — {FINAL}")
save_fig("per_type_scores", fig)
va_pred = RESULTS[FINAL]["predict"](va_s)
errs = [(" ".join(s["tokens"]), spans(s["tags"]), spans(p)) for s, p in zip(va_s, va_pred) if spans(s["tags"]) != spans(p)][:5]
for t, g, p in errs:
    print(f"- {t[:120]}\n    GT  : {g}\n    pred: {p}")

# %% [markdown]
# # 8. Prediksi test + submission
#
# Format otomatis: sama dengan format input (CoNLL → file .conll berisi `token<TAB>tag`), atau mengikuti
# `sample_submission.csv` kalau ada (CSV panjang: id kalimat / token / tag).

# %%
SUB_PATH = None
if test_sents:
    preds = RESULTS[FINAL]["predict"](test_sents)
    fmt = CFG.SUBMISSION_FORMAT
    sample = pd.read_csv(CFG.SAMPLE_SUB_PATH) if os.path.exists(CFG.SAMPLE_SUB_PATH) else None
    if fmt == "auto":
        fmt = "csv_long" if sample is not None else ("jsonl" if FMT.startswith("jsonl") else "conll")
    if fmt == "conll":
        SUB_PATH = os.path.join(SUB_DIR, "submission.conll")
        with open(SUB_PATH, "w", encoding="utf-8") as f:
            for s, p in zip(test_sents, preds):
                f.write("\n".join(f"{w}\t{t}" for w, t in zip(s["tokens"], p)) + "\n\n")
    elif fmt == "jsonl":
        SUB_PATH = os.path.join(SUB_DIR, "submission.jsonl")
        with open(SUB_PATH, "w", encoding="utf-8") as f:
            for s, p in zip(test_sents, preds):
                row = {"id": s["id"], "tokens": s["tokens"], "tags": p}
                if s.get("offsets"):
                    row["entities"] = [{"start": s["offsets"][a][0], "end": s["offsets"][b - 1][1], "label": ty}
                                       for ty, a, b in spans(p)]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        rows = [{"sentence_id": s["id"], "token_index": i, "token": w, "tag": t}
                for s, p in zip(test_sents, preds) for i, (w, t) in enumerate(zip(s["tokens"], p))]
        sub = pd.DataFrame(rows)
        if sample is not None and len(sample) == len(sub):
            sub = pd.DataFrame({sample.columns[0]: sample.iloc[:, 0], sample.columns[-1]: sub["tag"].values})
        SUB_PATH = os.path.join(SUB_DIR, "submission.csv")
        sub.to_csv(SUB_PATH, index=False)
    n_ent = sum(len(spans(p)) for p in preds)
    print(f"✅ Submission ({fmt}): {SUB_PATH} | {len(test_sents)} kalimat | {n_ent} entitas diprediksi")
    if sample is not None and fmt == "csv_long" and len(sample) != sum(len(s["tokens"]) for s in test_sents):
        print("⚠️ jumlah token ≠ jumlah baris sample_submission → cek tokenisasi / format yang diminta")

# %% [markdown]
# # 9. Ringkasan untuk laporan

# %%
best = board.iloc[0]
summ = {"final_model": FINAL, "labels": LABELS, "n_train_sent": len(tr_s), "n_valid_sent": len(va_s),
        "valid_entity_f1": round(float(best.entity_f1), 4), "precision": round(float(best.precision), 4),
        "recall": round(float(best.recall), 4), "submission": SUB_PATH}
text = f"""# Token classification summary

- Entity types: {', '.join(TYPES)}; {len(tr_s)} training / {len(va_s)} validation sentences.
- Final model **{FINAL}**: strict entity-level F1 **{summ['valid_entity_f1']}** (P {summ['precision']}, R {summ['recall']}).
- Leaderboard: {', '.join(f"{r.model} {r.entity_f1:.3f}" for r in board.itertuples())}.

**Draft sentences (paraphrase):**
- "We cast the task as BIO sequence labelling and evaluated with strict entity-level F1, where an entity counts only if both its type and boundaries match."
- "A feature-based logistic regression baseline was compared with a fine-tuned transformer whose labels are assigned to the first sub-word of each word."
"""
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(text)
json.dump(summ, open(os.path.join(OUT, "run_summary.json"), "w"), indent=2, default=str)
print(text)
print("✅ Notebook selesai.")

# %% [markdown]
# # 10. Troubleshooting, recipes, checklist
#
# | Gejala | Solusi |
# |---|---|
# | `tidak tersedia (offline)` | download model sekali saat online, atau pakai model yang sudah ada di cache (`TRANSFORMER_MODELS`) |
# | F1 transformer < baseline | data sangat kecil/epoch kurang → `EPOCHS=10`, `LR=5e-5`; atau model bahasa yang cocok (IndoBERT untuk teks Indonesia) |
# | Entitas panjang terpotong | `MAX_LEN=256` |
# | Label aneh (`B_PER`, `S-LOC`) | dinormalisasi otomatis ke BIO; cek daftar label yang dicetak |
#
# ```python
# CFG.TRANSFORMER_MODELS = ["indobenchmark/indobert-base-p1", "xlm-roberta-base"]   # Indonesia
# CFG.TRANSFORMER_MODELS = ["bert-base-cased"]                                         # Inggris (cased penting untuk NER!)
# ```
#
# - [ ] Label BIO valid dan tipe entitas sesuai soal
# - [ ] Metrik = entity-level F1 (bukan akurasi token) kecuali aturan berkata lain
# - [ ] Jumlah token/kalimat submission = data test
