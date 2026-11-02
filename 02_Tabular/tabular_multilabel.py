# -*- coding: utf-8 -*-
# =====================================================================================================
# tabular_multilabel.py — versi SCRIPT dari tabular_multilabel.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python tabular_multilabel.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb tabular_multilabel.py
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
# # Tabular Multilabel Template — beberapa target 0/1 sekaligus
#
# Untuk soal tabular yang meminta **beberapa label per baris** (satu nasabah bisa "tertarik promo" DAN "akan
# upgrade"; satu transaksi bisa punya beberapa tag; prediksi beberapa penyakit sekaligus). Bentuk data yang dikenali:
#
# | Bentuk target | Contoh | Setting |
# |---|---|---|
# | beberapa kolom 0/1 | `promo, upgrade, churn` | `TARGET_COL = ["promo", "upgrade", "churn"]` (atau auto dari sample_submission) |
# | satu kolom berisi daftar | `tags = "promo|upgrade"` | `TARGET_COL = "tags"`, `LABEL_SEP = "|"` |
#
# Untuk satu target saja (biner/multikelas/regresi) pakai `tabular_master`.
#
# ## Pipeline
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json) → load (csv/xlsx/parquet) → target multilabel (kolom 0/1 atau daftar)
# → EDA label: prevalensi, jumlah label per baris, korelasi antar label (co-occurrence)
# → preprocessing fold-safe: numerik (median), kategorikal (ordinal / one-hot), tanggal → bagian tanggal
# → model per label (Binary Relevance): Logistic Regression, HistGradientBoosting, LightGBM (kalau ada)
#   + Classifier Chain (memakai prediksi label lain sebagai fitur) — dibandingkan dengan CV
# → OOF: AUC per label, micro/macro F1, Hamming loss, subset accuracy; threshold per label di-tune di OOF
# → submission (probabilitas atau 0/1, kolom mengikuti sample_submission) + validator + ringkasan laporan
# ```
#
# **Dua format:** `tabular_multilabel.ipynb` / `.py`. Headless: `python tools/run_task.py --task tabular-multilabel --format py`.

# %%
import os


class CFG:
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "tabular_multilabel"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"
    TRAIN_PATH = "./data/train.csv"
    TEST_PATH = "./data/test.csv"
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    OUTPUT_DIR = "./outputs/tabular_multilabel/"
    ID_COL = "id"
    TARGET_COL = None            # list kolom 0/1, atau satu kolom berisi daftar label; None = auto dari sample_submission
    LABEL_SEP = None             # satu kolom daftar: pemisah ("|", ",", " ", ";"); None = auto
    DROP_COLS = []
    METRIC = "auc_macro"         # "auc_macro" | "f1_micro" | "f1_macro" | "logloss" | "hamming" — sesuai aturan lomba
    SUBMISSION_MODE = "auto"     # "auto" | "proba" | "label"  (auto: dari isi sample_submission)
    MODELS = ["logreg", "hgb", "lgb"]
    USE_CHAIN = True             # Classifier Chain (urutan label: prevalensi tertinggi dulu)
    N_FOLDS = 5
    MAX_ONEHOT_CARD = 30         # kategorikal ≤ ini → one-hot (logreg); lebih → ordinal
    TUNE_THRESHOLDS = True
    FAST_MAX_ROWS = 3000
    FAST_N_FOLDS = 3
    FIG_DPI = 150

# %%
# >>> USER OVERRIDE
# CFG.TARGET_COL = ["promo", "upgrade", "churn"]
# CFG.METRIC = "f1_micro"
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
# Inti: numpy, pandas, scikit-learn, matplotlib. LightGBM opsional (di-skip kalau tidak ada).

# %%
import importlib.util
import subprocess
import sys

for imp, pip_name in {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "matplotlib": "matplotlib",
                      "lightgbm": "lightgbm", "openpyxl": "openpyxl"}.items():
    if importlib.util.find_spec(imp) is None:
        if CFG.AUTO_INSTALL:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            except Exception as e:  # noqa
                print("install gagal:", pip_name, e)
        else:
            print(f"  {imp}: tidak ada (AUTO_INSTALL off)")

import json
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, log_loss, hamming_loss, accuracy_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except Exception:  # noqa
    display = print
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:  # noqa
    HAS_LGB = False
print("LightGBM:", HAS_LGB)
if CFG.RUN_MODE == "fast":
    CFG.N_FOLDS = min(CFG.N_FOLDS, CFG.FAST_N_FOLDS)
OUT = CFG.OUTPUT_DIR
FIG_DIR, TAB_DIR, SUB_DIR = (os.path.join(OUT, d) for d in ("figures", "tables", "submissions"))
for d in (OUT, FIG_DIR, TAB_DIR, SUB_DIR):
    os.makedirs(d, exist_ok=True)
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
# # 2. Load data & bentuk target multilabel
#
# **Jika** `TARGET_COL=None` **maka** kolom target = kolom `sample_submission` (selain ID) yang juga ada di train.
# **Jika** satu kolom berisi daftar label **maka** dipecah menjadi kolom 0/1 per label (`LABEL_SEP` auto).

# %%
train = smart_read(CFG.TRAIN_PATH)
test = smart_read(CFG.TEST_PATH) if os.path.exists(CFG.TEST_PATH) else None
sample = smart_read(CFG.SAMPLE_SUB_PATH) if os.path.exists(CFG.SAMPLE_SUB_PATH) else None
ID = CFG.ID_COL if CFG.ID_COL in train.columns else None
print(f"train {train.shape} | test {None if test is None else test.shape} | sample {None if sample is None else sample.shape}")

tc = CFG.TARGET_COL
if tc is None and sample is not None:
    tc = [c for c in sample.columns if c != (CFG.ID_COL if CFG.ID_COL in sample.columns else sample.columns[0]) and c in train.columns]
if tc is None or (isinstance(tc, list) and not tc):
    raise ValueError("Target tidak bisa ditebak → isi CFG.TARGET_COL (list kolom 0/1 atau satu kolom daftar label).")
if isinstance(tc, str):
    s = train[tc].fillna("").astype(str)
    sep = CFG.LABEL_SEP or max(["|", ",", ";", " "], key=lambda x: s.str.contains(x, regex=False).mean())
    lists = s.map(lambda v: [t.strip() for t in v.split(sep) if t.strip()])
    LABELS = sorted({t for l in lists for t in l})
    Y = pd.DataFrame({lab: lists.map(lambda l, lab=lab: int(lab in l)) for lab in LABELS})
    TARGET_SRC = [tc]
    print(f"Target daftar '{tc}' (pemisah {sep!r}) → {len(LABELS)} label")
else:
    LABELS = list(tc)
    Y = train[LABELS].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).clip(0, 1)
    TARGET_SRC = LABELS
X = train.drop(columns=[c for c in TARGET_SRC + list(CFG.DROP_COLS) + ([ID] if ID else []) if c in train.columns])
X_test = test[[c for c in X.columns if c in test.columns]].copy() if test is not None else None
const = [l for l in LABELS if Y[l].nunique() < 2]
if const:
    print("⚠️ label konstan (selalu 0/1) → diprediksi konstan:", const)
if CFG.RUN_MODE == "fast" and len(X) > CFG.FAST_MAX_ROWS:
    idx = np.random.RandomState(CFG.SEED).choice(len(X), CFG.FAST_MAX_ROWS, replace=False)
    X, Y = X.iloc[idx].reset_index(drop=True), Y.iloc[idx].reset_index(drop=True)
    print(f"[FAST MODE] {len(X)} baris")
print("Label:", LABELS)

# %% [markdown]
# # 3. EDA label
#
# - *Prevalensi* sangat kecil (< 2%) → AUC/F1 label itu tidak stabil; threshold per label penting.
# - *Co-occurrence* tinggi antar label → Classifier Chain biasanya membantu.

# %%
prev = Y.mean().sort_values(ascending=False)
card = Y.sum(1)
print(f"Rata-rata label per baris: {card.mean():.2f} | baris tanpa label: {(card == 0).mean():.1%}")
display(prev.to_frame("prevalensi").round(4))
save_table(prev.rename_axis("label").reset_index(name="prevalence"), "label_prevalence")
corr = Y.corr().fillna(0)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].bar(prev.index.astype(str), prev.values, color="#4C72B0")
ax[0].set_title("Prevalensi label")
ax[0].tick_params(axis="x", rotation=45)
ax[1].hist(card, bins=range(card.max() + 2), align="left", color="#55A868")
ax[1].set_title("Jumlah label per baris")
im = ax[2].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax[2].set_xticks(range(len(LABELS)), LABELS, rotation=45, fontsize=7)
ax[2].set_yticks(range(len(LABELS)), LABELS, fontsize=7)
ax[2].set_title("Korelasi antar label")
plt.colorbar(im, ax=ax[2])
save_fig("label_eda", fig)

# %% [markdown]
# # 4. Preprocessing (fold-safe) & fitur
#
# - Teks tanggal → tahun/bulan/hari/hari-dalam-minggu (kolom asli dibuang).
# - Teks angka (`"1.234"`, `"Rp 5.000"`) → angka kalau ≥ 90% baris bisa dikonversi.
# - Encoder & imputer di-*fit* di dalam setiap fold (lewat `Pipeline`) → tidak ada kebocoran ke validasi.

# %%
def prepare(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            num = pd.to_numeric(df[c].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
            if num.notna().mean() > 0.9 and df[c].astype(str).str.contains(r"\d").mean() > 0.9 and \
                    not df[c].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.5:
                df[c] = num
                continue
            dt = pd.to_datetime(df[c], errors="coerce", format="mixed") if df[c].astype(str).str.contains(r"\d{4}|/").mean() > .8 else None
            if dt is not None and dt.notna().mean() > 0.9:
                df[c + "_year"], df[c + "_month"], df[c + "_dow"] = dt.dt.year, dt.dt.month, dt.dt.dayofweek
                df[c + "_ord"] = (dt - pd.Timestamp("2000-01-01")).dt.days
                df = df.drop(columns=c)
    return df


X = prepare(X)
X_test = prepare(X_test).reindex(columns=X.columns) if X_test is not None else None
NUM = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
CAT = [c for c in X.columns if c not in NUM]
for c in CAT:
    X[c] = X[c].astype(str)
    if X_test is not None:
        X_test[c] = X_test[c].astype(str)
LOW = [c for c in CAT if X[c].nunique() <= CFG.MAX_ONEHOT_CARD]
HIGH = [c for c in CAT if c not in LOW]
print(f"numerik {len(NUM)} | kategorikal one-hot {len(LOW)} | kategorikal ordinal {len(HIGH)}")


def make_pre(kind):
    if kind == "logreg":
        tr = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM)]
        if LOW:
            tr.append(("low", OneHotEncoder(handle_unknown="ignore", min_frequency=5), LOW))
        if HIGH:
            tr.append(("high", make_pipeline(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                                             StandardScaler()), HIGH))
    else:
        tr = [("num", "passthrough", NUM)]
        if CAT:
            tr.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                             encoded_missing_value=-1), CAT))
    return ColumnTransformer(tr, remainder="drop")


def make_model(kind, extra=0):
    if kind == "logreg":
        return make_pipeline(make_pre(kind), LogisticRegression(max_iter=2000, C=1.0))
    if kind == "hgb":
        cat_idx = list(range(len(NUM), len(NUM) + len(CAT))) if CAT else None
        return make_pipeline(make_pre(kind), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, early_stopping=True,
                                                                            categorical_features=cat_idx, random_state=CFG.SEED))
    if kind == "lgb" and HAS_LGB:
        return make_pipeline(make_pre(kind), lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31,
                                                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                                random_state=CFG.SEED, verbose=-1))
    return None

# %% [markdown]
# # 5. Training: Binary Relevance (+ Classifier Chain)
#
# **Binary Relevance** = satu model biner per label. **Classifier Chain** = model label ke-k juga menerima prediksi
# OOF label 1..k-1 sebagai fitur (memanfaatkan korelasi label). Keduanya dinilai dengan OOF yang sama.

# %%
folds = list(KFold(CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED).split(X))
RESULTS = {}


def fit_predict_label(kind, Xtr, ytr, Xva, Xte):
    if ytr.nunique() < 2:
        v = float(ytr.mean())
        return np.full(len(Xva), v), (np.full(len(Xte), v) if Xte is not None else None)
    m = make_model(kind)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], (m.predict_proba(Xte)[:, 1] if Xte is not None else None)


def run_binary_relevance(kind):
    oof = np.zeros((len(X), len(LABELS)))
    te = np.zeros((len(X_test), len(LABELS))) if X_test is not None else None
    for tr, va in folds:
        for j, lab in enumerate(LABELS):
            p_va, p_te = fit_predict_label(kind, X.iloc[tr], Y[lab].iloc[tr], X.iloc[va], X_test)
            oof[va, j] = p_va
            if te is not None:
                te[:, j] += p_te / len(folds)
    return oof, te


def run_chain(kind):
    order = list(prev.index)
    oof = np.zeros((len(X), len(LABELS)))
    te = np.zeros((len(X_test), len(LABELS))) if X_test is not None else None
    for tr, va in folds:
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        Xte = X_test.copy() if X_test is not None else None
        for lab in order:
            j = LABELS.index(lab)
            p_va, p_te = fit_predict_label(kind, Xtr, Y[lab].iloc[tr], Xva, Xte)
            oof[va, j] = p_va
            if te is not None:
                te[:, j] += p_te / len(folds)
            # fitur chain: label asli di train (teacher forcing), prediksi di valid/test
            Xtr[f"chain_{lab}"] = Y[lab].iloc[tr].values
            Xva[f"chain_{lab}"] = p_va
            if Xte is not None:
                Xte[f"chain_{lab}"] = p_te
            if f"chain_{lab}" not in NUM:
                NUM.append(f"chain_{lab}")
        for lab in order:
            NUM.remove(f"chain_{lab}")
    return oof, te


def scores(oof, th=None):
    th = np.full(len(LABELS), 0.5) if th is None else np.asarray(th)
    yb = (oof >= th).astype(int)
    aucs = [roc_auc_score(Y[l], oof[:, j]) if Y[l].nunique() > 1 else np.nan for j, l in enumerate(LABELS)]
    return {"auc_macro": float(np.nanmean(aucs)), "f1_micro": f1_score(Y.values, yb, average="micro", zero_division=0),
            "f1_macro": f1_score(Y.values, yb, average="macro", zero_division=0),
            "logloss": float(np.mean([log_loss(Y[l], np.clip(oof[:, j], 1e-6, 1 - 1e-6), labels=[0, 1]) for j, l in enumerate(LABELS)])),
            "hamming": hamming_loss(Y.values, yb), "subset_acc": accuracy_score(Y.values, yb),
            "map": float(np.nanmean([average_precision_score(Y[l], oof[:, j]) if Y[l].nunique() > 1 else np.nan
                                     for j, l in enumerate(LABELS)])), "auc_per_label": aucs}


GREATER = CFG.METRIC not in ("logloss", "hamming")
for kind in CFG.MODELS:
    if make_model(kind) is None:
        print(f"skip {kind} (library tidak ada)")
        continue
    t0 = time.time()
    try:
        RESULTS[f"br_{kind}"] = run_binary_relevance(kind)
        sc = scores(RESULTS[f"br_{kind}"][0])
        print(f"==> br_{kind}: {CFG.METRIC} = {sc[CFG.METRIC]:.5f} | auc_macro {sc['auc_macro']:.4f} | f1_micro {sc['f1_micro']:.4f} ({time.time() - t0:.0f}s)")
        if CFG.USE_CHAIN and len(LABELS) > 1:
            t0 = time.time()
            RESULTS[f"chain_{kind}"] = run_chain(kind)
            sc = scores(RESULTS[f"chain_{kind}"][0])
            print(f"==> chain_{kind}: {CFG.METRIC} = {sc[CFG.METRIC]:.5f} | auc_macro {sc['auc_macro']:.4f} ({time.time() - t0:.0f}s)")
    except Exception as e:  # noqa
        print(f"❌ {kind} gagal: {type(e).__name__}: {str(e)[:200]}")
if not RESULTS:
    raise RuntimeError("Semua model gagal — cek data/kolom.")
board = pd.DataFrame([{"model": k, **{m: v for m, v in scores(o).items() if m != "auc_per_label"}} for k, (o, _) in RESULTS.items()])
board = board.sort_values(CFG.METRIC, ascending=not GREATER).reset_index(drop=True)
display(board.round(4))
save_table(board, "leaderboard")

# %% [markdown]
# # 6. Ensemble + threshold per label
#
# Ensemble = rata-rata probabilitas 2–3 model teratas (dipakai kalau lebih baik di OOF). Threshold per label dipilih
# untuk F1 label itu di OOF (relevan kalau submission berupa 0/1 atau metrik F1).

# %%
top = board["model"].head(3).tolist()
ens_oof = np.mean([RESULTS[k][0] for k in top], 0)
ens_te = np.mean([RESULTS[k][1] for k in top], 0) if X_test is not None else None
best_key = board["model"].iloc[0]
cand = {best_key: RESULTS[best_key], "ensemble_top3": (ens_oof, ens_te)}
sel = max(cand, key=lambda k: scores(cand[k][0])[CFG.METRIC] * (1 if GREATER else -1))
FINAL_OOF, FINAL_TE = cand[sel]
print(f"Model final: {sel} | OOF {CFG.METRIC} = {scores(FINAL_OOF)[CFG.METRIC]:.5f}")
TH = np.full(len(LABELS), 0.5)
if CFG.TUNE_THRESHOLDS:
    for j, lab in enumerate(LABELS):
        if Y[lab].nunique() < 2:
            continue
        grid = np.arange(0.05, 0.96, 0.01)
        f1s = [f1_score(Y[lab], FINAL_OOF[:, j] >= t, zero_division=0) for t in grid]
        TH[j] = grid[int(np.argmax(f1s))]
s05, sth = scores(FINAL_OOF), scores(FINAL_OOF, TH)
print(f"F1 micro: th=0.5 → {s05['f1_micro']:.4f} | th per label → {sth['f1_micro']:.4f}")
per_label = pd.DataFrame({"label": LABELS, "prevalence": Y.mean().values, "auc": sth["auc_per_label"], "threshold": TH,
                          "f1": [f1_score(Y[l], FINAL_OOF[:, j] >= TH[j], zero_division=0) for j, l in enumerate(LABELS)]})
display(per_label.round(4))
save_table(per_label, "per_label_scores")
fig, ax = plt.subplots(figsize=(max(5, .5 * len(LABELS)), 3.5))
ax.bar(per_label["label"], per_label["auc"], color="#4C72B0", label="AUC")
ax.bar(per_label["label"], per_label["f1"], color="#C44E52", alpha=.6, label="F1 (th tuned)")
ax.set_ylim(0, 1)
ax.legend()
ax.tick_params(axis="x", rotation=45)
ax.set_title("Skor OOF per label")
save_fig("per_label_scores", fig)

# %% [markdown]
# # 7. Submission + validator

# %%
SUB_PATH = None
if FINAL_TE is not None:
    mode = CFG.SUBMISSION_MODE
    if mode == "auto":
        mode = "proba"
        if sample is not None:
            vals = sample[[c for c in sample.columns if c in LABELS]].values.ravel()
            mode = "label" if len(vals) and set(pd.unique(vals)) <= {0, 1} and CFG.METRIC.startswith(("f1", "hamming")) else "proba"
    out = FINAL_TE if mode == "proba" else (FINAL_TE >= TH).astype(int)
    id_name = (sample.columns[0] if sample is not None else (CFG.ID_COL or "id"))
    ids = test[CFG.ID_COL] if CFG.ID_COL in test.columns else np.arange(len(test))
    if isinstance(CFG.TARGET_COL, str):
        sub = pd.DataFrame({id_name: ids, CFG.TARGET_COL: [
            (CFG.LABEL_SEP or " ").join(l for l, v in zip(LABELS, row) if v >= (0.5 if mode == "proba" else 1))
            for row in ((FINAL_TE >= TH).astype(int))]})
    else:
        sub = pd.DataFrame(out, columns=LABELS)
        sub.insert(0, id_name, ids.values if hasattr(ids, "values") else ids)
        if sample is not None:
            sub = sub[[c for c in sample.columns if c in sub.columns]]
    SUB_PATH = os.path.join(SUB_DIR, "submission.csv")
    sub.to_csv(SUB_PATH, index=False)
    print(f"✅ Submission ({mode}): {SUB_PATH} | shape {sub.shape}")
    if sample is not None:
        print("✅ Validator: kolom & baris sama dengan sample_submission" if list(sub.columns) == list(sample.columns)
              and len(sub) == len(sample) else "⚠️ Validator: kolom/baris berbeda dari sample_submission — cek!")
else:
    print("ℹ️ Tidak ada test → submission dilewati.")

# %% [markdown]
# # 8. Ringkasan untuk laporan

# %%
fs = scores(FINAL_OOF, TH)
summ = {"final_model": sel, "n_labels": len(LABELS), "n_train": len(X), "oof_auc_macro": round(fs["auc_macro"], 4),
        "oof_f1_micro": round(fs["f1_micro"], 4), "oof_f1_macro": round(fs["f1_macro"], 4), "hamming": round(fs["hamming"], 4),
        "subset_accuracy": round(fs["subset_acc"], 4), "submission": SUB_PATH}
text = f"""# Multilabel summary

- {len(LABELS)} labels, {len(X)} training rows, {CFG.N_FOLDS}-fold CV. Final model: **{sel}**.
- OOF macro AUC **{summ['oof_auc_macro']}**, micro F1 **{summ['oof_f1_micro']}** (per-label thresholds), Hamming loss {summ['hamming']}.
- Hardest label: **{per_label.sort_values('auc').iloc[0]['label']}** (AUC {per_label['auc'].min():.3f}).

**Draft sentences (paraphrase):**
- "We framed the task as multilabel classification and compared binary relevance with classifier chains under the same cross-validation."
- "Decision thresholds were tuned per label on out-of-fold predictions to maximise F1."
"""
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(text)
json.dump(summ, open(os.path.join(OUT, "run_summary.json"), "w"), indent=2, default=str)
print(text)
print("✅ Notebook selesai.")

# %% [markdown]
# # 9. Troubleshooting, recipes, checklist
#
# | Gejala | Solusi |
# |---|---|
# | `Target tidak bisa ditebak` | isi `CFG.TARGET_COL` (list kolom atau satu kolom daftar) |
# | AUC satu label NaN | label konstan di train/fold → dibuang dari rata-rata; cek prevalensi |
# | F1 micro rendah tapi AUC tinggi | threshold — pastikan `TUNE_THRESHOLDS=True` & submission mode `label` |
# | Lambat | `MODELS=["lgb"]`, `USE_CHAIN=False`, `RUN_MODE="fast"` |
#
# ```python
# CFG.MODELS, CFG.USE_CHAIN = ["lgb", "logreg"], True     # default kuat
# CFG.METRIC, CFG.SUBMISSION_MODE = "f1_micro", "label"   # lomba berbasis F1 & 0/1
# ```
#
# - [ ] Kolom label & urutan kolom submission = sample_submission
# - [ ] Probabilitas vs 0/1 sesuai aturan; threshold dari OOF
# - [ ] Metrik lomba (micro/macro/sample-based) sama dengan `CFG.METRIC`
