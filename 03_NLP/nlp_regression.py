# -*- coding: utf-8 -*-
# =====================================================================================================
# nlp_regression.py — versi SCRIPT dari nlp_regression.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python nlp_regression.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb nlp_regression.py
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
# # NLP Text Regression — Competition Template
#
# Template untuk lomba di mana **input utama berupa teks** dan **target berupa angka**. Kode lengkap + panduan
# membaca output + aturan "Jika X maka Y" + catatan untuk laporan, supaya bisa dikerjakan mandiri dari nol
# sampai submit.
#
# **Contoh task**
#
# | Contoh | Target | Catatan |
# |---|---|---|
# | ulasan produk → rating | 1–5 (diskrit, ordinal) | metric sering RMSE/MAE/QWK → rounding & optimized thresholds |
# | esai → skor | 0–100 / 1–6 | QWK umum dipakai |
# | dokumen hukum → lama hukuman (bulan) | ≥ 0, sangat skewed | `TARGET_TRANSFORM="log1p"`, clip, rounding |
# | iklan properti → harga | ≥ 0, long-tail | `log1p` hampir wajib, metric bisa RMSLE |
# | artikel → readability / sentiment score | kontinu | Pearson/Spearman |
#
# ## Pipeline besar
#
# ```text
# CFG + USER OVERRIDE
# → install/cek library → load data (smart_read) → parsing target numerik → sanity check
# → analisis target (distribusi, skew, diskrit?, transform) → text cleaning (2 versi)
# → Text EDA (panjang & MAX_LEN, bahasa → model, n-gram per bin target, korelasi fitur-target, duplikat, drift)
# → K-Fold binned-stratified (atau group)
# → [A] Classical: TF-IDF word+char → Ridge / SGD(Huber) / LinearSVR / LightGBM(SVD)
# → [B] Sentence-embedding (frozen) → Ridge / SVR / LightGBM / XGB / CatBoost
# → [C] Transformer fine-tuning K-Fold (regression head, target distandarisasi, MSE/Huber, long text, OOM retry)
# → (opsional) pseudo-labeling → perbandingan model → ensemble (mean / weighted / rank / stacking)
# → post-processing (clip, rounding ke nilai valid, optimized rounding thresholds untuk QWK)
# → error analysis (residual per rentang target & panjang teks) → interpretability → submission + validator
# ```
#
# ## Cara pakai cepat (5 langkah)
#
# 1. Taruh data di `./data/train.csv`, `./data/test.csv`, `./data/sample_submission.csv` (atau ubah path).
# 2. Di cell **USER OVERRIDE**: isi `ID_COL`, `TEXT_COL`, `TARGET_COL`, `METRIC` sesuai aturan lomba.
# 3. Jalankan dulu `CFG.RUN_MODE = "fast"` → pastikan semua cell jalan.
# 4. Baca hasil analisis target & EDA (transform, MAX_LEN, model), sesuaikan CFG.
# 5. `CFG.RUN_MODE = "full"` → Run All → ambil `outputs/.../submissions/submission.csv`.
#
# ## Pilih model transformer (ringkas)
#
# | Kondisi data | Model key | Catatan |
# |---|---|---|
# | Bahasa Indonesia formal (dokumen, berita) | `indobert_base` | |
# | Bahasa Indonesia sosmed / slang | `indobertweet` | |
# | Campur ID-EN / multibahasa | `xlm_roberta_base` | |
# | English | `deberta_v3_base` / `modernbert_base` | DeBERTa-v3 = juara banyak lomba regresi teks (esai) |
# | Tidak ada GPU | **jangan fine-tune**; pakai Classical + Sentence-embedding | |
#
# > **Prinsip:** semua keputusan (transform, model, rounding, ensemble) diambil berdasarkan **skor OOF**.

# %% [markdown]
# ## 0. Catatan platform & hardware
#
# - Notebook **self-contained**, jalan di Jupyter / VSCode lokal.
# - **Tanpa GPU**: Classical (TF-IDF + Ridge) dan Sentence-embedding tetap cepat. Fine-tuning di CPU sangat lambat
#   → default `RUN_TRANSFORMER = "auto"` = hanya jalan kalau ada CUDA.
# - **GPU kecil (4–6 GB)**: model `base`, `BATCH_SIZE=8`, `MAX_LEN<=256`, `USE_AMP=True`,
#   `GRADIENT_CHECKPOINTING=True` kalau masih OOM (OOM juga di-retry otomatis dengan batch lebih kecil).
# - **Internet diblokir?** model HuggingFace harus sudah ada di cache atau folder lokal (isi `MODEL_POOL` dengan path).
# - Output di `CFG.OUTPUT_DIR` (`figures/`, `tables/`, `models/`, `submissions/`, `preds/`, `cache/`).

# %%
# ============================================================
# Import dasar (library standar + numpy/pandas). Library opsional di-import di cell setelah CFG.
# ============================================================
import os
import re
import gc
import sys
import json
import html
import math
import time
import random
import hashlib
import warnings
import inspect
import traceback
import importlib
import importlib.util
import subprocess
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 200)

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print

NOTEBOOK_NAME = "nlp_regression"
print("Python:", sys.version.split()[0], "| numpy:", np.__version__, "| pandas:", pd.__version__)

# %% [markdown]
# ## 1. CFG — pusat kontrol notebook
#
# Semua "knob" ada di sini. Ubah lewat cell `USER OVERRIDE` di bawahnya (jangan edit class ini langsung saat lomba).
#
# Yang hampir pasti diubah: path, `ID_COL`, `TEXT_COL`, `TARGET_COL`, `METRIC`, `RUN_MODE`, `TRANSFORMER_MODELS`,
# dan (kalau target punya range tertentu) `MIN_TARGET`/`MAX_TARGET`/`VALID_VALUES`.

# %%
class CFG:
    # ======================================================
    # RUN MODE
    # ======================================================
    RUN_MODE = "full"              # "fast" (debug: subsample, fold & epoch sedikit) | "full" (final)
    FAST_SAMPLE_ROWS = 3000
    FAST_N_FOLDS = 3
    FAST_EPOCHS = 1
    SEED = 42
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"            # install package yang BELUM ada (tidak pernah upgrade, tidak pernah torch) (env TEMPLATE_AUTO_INSTALL=0 → off)
    INSTALL_OPTIONAL = True

    # ======================================================
    # PATHS
    # ======================================================
    TRAIN_PATH = "./data/train.csv"
    TEST_PATH = "./data/test.csv"              # tidak ada → CV saja
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    OUTPUT_DIR = f"./outputs/{NOTEBOOK_NAME}/"

    # ======================================================
    # KOLOM
    # ======================================================
    ID_COL = "id"
    TEXT_COL = "text"
    EXTRA_TEXT_COLS = []           # kolom teks tambahan digabung: "title | text"
    TARGET_COL = "target"          # angka; string angka ("1.234,5", "Rp 10.000") di-parse otomatis
    META_COLS = []                 # metadata kategorikal → disisipkan ke teks "[kategori: fashion] ..."
    USE_METADATA_AS_TEXT = True
    GROUP_COL = None               # mis. "user_id" → GroupKFold
    SUBMISSION_TARGET_COL = None   # None = otomatis dari sample_submission / TARGET_COL

    # ======================================================
    # METRIC & TARGET
    # ======================================================
    # "rmse", "mae", "r2", "pearson", "spearman", "qwk" (target diskrit ordinal), "rmsle", "mape", "medae"
    METRIC = "rmse"
    # Transform target saat training (evaluasi tetap di skala asli):
    # "none" | "log1p" (target ≥ 0 & skewed) | "sqrt" | "auto" (log1p kalau skew > 1 dan min ≥ 0)
    TARGET_TRANSFORM = "auto"
    CLIP_PREDICTIONS = True        # clip prediksi ke [MIN_TARGET, MAX_TARGET]
    MIN_TARGET = None              # None = min target train
    MAX_TARGET = None              # None = max target train
    VALID_VALUES = None            # nilai valid target, mis. [1, 2, 3, 4, 5]; None = deteksi otomatis kalau diskrit
    ROUND_TO_VALID = "auto"        # "auto" = coba rounding & pakai kalau OOF membaik | True | False
    OPTIMIZE_ROUNDING = True       # cari threshold rounding optimal (berguna untuk QWK / rating)
    MIN_TUNING_GAIN = 0.002        # post-processing hanya dipakai kalau gain cross-validated ≥ ini (relatif ke skala metric)

    # ======================================================
    # CROSS-VALIDATION
    # ======================================================
    N_FOLDS = 5
    CV_STRATEGY = "auto"           # "auto" (binned stratified / group) | "binned" | "kfold" | "group" | "stratified_group"
    TARGET_BINS = 10               # jumlah bin target untuk stratifikasi

    # ======================================================
    # TEXT CLEANING (lihat penjelasan section 8)
    # ======================================================
    NORMALIZE_URL = True
    NORMALIZE_USER = True
    KEEP_HASHTAG_TEXT = True
    REMOVE_EMOJI = False
    TRANSFORMER_LOWERCASE = False
    TRANSFORMER_USE_SLANG = False
    CLASSICAL_LOWERCASE = True
    USE_SLANG_NORMALIZATION = True
    CUSTOM_SLANG_DICT = {}
    REDUCE_ELONGATION = True
    REMOVE_STOPWORDS_CLASSICAL = False
    USE_STEMMING = False
    REMOVE_NUMBERS_CLASSICAL = False   # HATI-HATI: angka di teks sering sinyal kuat untuk regresi (luas, umur, dll.)

    # ======================================================
    # EDA
    # ======================================================
    RUN_EDA = True
    EDA_TOKENIZER_MODEL = None
    MAX_LEN_PERCENTILE = 95
    EDA_TARGET_GROUPS = 3          # target dibagi jadi N kelompok (low/mid/high) untuk perbandingan teks
    TOP_NGRAMS = 15
    NEAR_DUP_THRESHOLD = 0.90
    NEAR_DUP_MAX_ROWS = 20000
    RUN_ADVERSARIAL_VALIDATION = True
    MAKE_WORDCLOUD = True

    # ======================================================
    # [A] CLASSICAL (TF-IDF)
    # ======================================================
    RUN_CLASSICAL = True
    CLASSICAL_MODELS = ["ridge", "sgd", "linearsvr", "lgbm_svd"]   # pilihan: ridge, sgd, linearsvr, lgbm_svd
    TFIDF_WORD_NGRAM = (1, 2)
    TFIDF_CHAR_NGRAM = (2, 5)
    USE_CHAR_TFIDF = True
    TFIDF_WORD_MAX_FEATURES = 200_000
    TFIDF_CHAR_MAX_FEATURES = 300_000
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95
    RIDGE_ALPHA = 1.0              # makin besar = regularisasi makin kuat (coba 0.3, 1, 3, 10)
    SVD_COMPONENTS = 256
    LGBM_N_ESTIMATORS = 2000

    # ======================================================
    # [B] SENTENCE EMBEDDING
    # ======================================================
    RUN_EMBEDDINGS = True
    EMBEDDING_MODELS = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
    EMBEDDING_PREFIXES = {}
    EMBEDDING_HEADS = ["ridge", "svr", "lgbm"]   # + "xgb", "catboost" kalau terpasang
    EMB_BATCH_SIZE = 64
    EMB_MAX_LEN = 256
    CACHE_EMBEDDINGS = True

    # ======================================================
    # [C] TRANSFORMER FINE-TUNING
    # ======================================================
    RUN_TRANSFORMER = "auto"       # "auto" = hanya kalau CUDA | True | False
    TRAINER_BACKEND = "custom_loop"  # "custom_loop" | "hf_trainer"
    MODEL_POOL = {
        "deberta_v3_base": "microsoft/deberta-v3-base",
        "deberta_v3_large": "microsoft/deberta-v3-large",
        "roberta_base": "FacebookAI/roberta-base",
        "modernbert_base": "answerdotai/ModernBERT-base",
        "modernbert_large": "answerdotai/ModernBERT-large",
        "indobert_base": "indobenchmark/indobert-base-p2",
        "indobert_large": "indobenchmark/indobert-large-p2",
        "indolem_indobert": "indolem/indobert-base-uncased",
        "indobertweet": "indolem/indobertweet-base-uncased",
        "xlm_roberta_base": "FacebookAI/xlm-roberta-base",
        "xlm_roberta_large": "FacebookAI/xlm-roberta-large",
        "mdeberta_v3_base": "microsoft/mdeberta-v3-base",
    }
    TRANSFORMER_MODELS = ["auto"]  # ["auto"] = rekomendasi deteksi bahasa
    MAX_LEN = "auto"
    LONG_TEXT_STRATEGY = "head_tail"   # "truncate" | "head_tail" | "chunk"
    HEAD_RATIO = 0.5
    CHUNK_OVERLAP = 64
    CHUNK_POOLING = "mean"
    MAX_CHUNKS_PER_DOC = 8
    BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 64
    GRAD_ACCUM_STEPS = 1
    EPOCHS = 3
    LR = 2e-5
    HEAD_LR = 1e-4
    LLRD = 1.0
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    SCHEDULER = "linear"
    MAX_GRAD_NORM = 1.0
    USE_AMP = True
    AMP_DTYPE = "fp16"
    GRADIENT_CHECKPOINTING = False
    POOLING = "mean"               # mean pooling biasanya lebih stabil untuk regresi
    MULTI_SAMPLE_DROPOUT = 0
    HEAD_DROPOUT = 0.0             # regresi: dropout di head sering MERUGIKAN (memperbesar variance output)
    BACKBONE_DROPOUT = 0.0         # trik populer lomba regresi teks: matikan dropout backbone. None = default model
    REG_LOSS = "mse"               # "mse" | "huber" (tahan outlier) | "mae"
    EARLY_STOPPING_PATIENCE = 2
    EVALS_PER_EPOCH = 1
    OOM_MAX_RETRIES = 3
    SAVE_FOLD_MODELS = False
    FOLDS_TO_RUN = None
    NUM_WORKERS = 0
    DEBUG_SIMULATE_OOM = False

    # ======================================================
    # PSEUDO-LABELING
    # ======================================================
    RUN_PSEUDO_LABEL = False
    PSEUDO_MAX_DISAGREEMENT = 0.5  # ambil test rows dengan std prediksi antar model ≤ kuantil ini (0.5 = 50% paling sepakat)
    PSEUDO_MAX_FRACTION = 0.5

    # ======================================================
    # ENSEMBLE, OUTPUT
    # ======================================================
    RUN_ENSEMBLE = True
    ENSEMBLE_METHODS = ["mean", "weighted", "rank", "stacking"]
    ENSEMBLE_TOP_K = 6
    MANUAL_WEIGHTS = {}
    FINAL_PREDICTION = "auto"
    RUN_INTERPRETABILITY = True
    N_EXPLAIN_EXAMPLES = 3
    USE_SHAP = True
    FIG_DPI = 200
    SHOW_FIGS = True

# %% [markdown]
# ## 2. USER OVERRIDE — ubah CFG di sini
#
# ```python
# CFG.TEXT_COL = "ulasan"
# CFG.TARGET_COL = "rating"
# CFG.METRIC = "rmse"            # sesuai aturan lomba
# CFG.VALID_VALUES = [1, 2, 3, 4, 5]
# CFG.RUN_MODE = "fast"
# ```
#
# **Jika** target harga/hukuman (≥ 0, ekor panjang) → biarkan `TARGET_TRANSFORM="auto"` (akan memilih log1p).
# **Jika** metric QWK → `CFG.METRIC = "qwk"`, rounding & threshold optimal otomatis dicoba.
# **Jika** laptop tanpa GPU → biarkan `RUN_TRANSFORMER = "auto"`.

# %%
# >>> USER OVERRIDE
# Contoh (hapus tanda # untuk mengaktifkan):
# CFG.TRAIN_PATH = "./data/train.csv"
# CFG.TEST_PATH = "./data/test.csv"
# CFG.SAMPLE_SUB_PATH = "./data/sample_submission.csv"
# CFG.ID_COL = "id"
# CFG.TEXT_COL = "text"
# CFG.TARGET_COL = "target"
# CFG.METRIC = "rmse"
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
        if isinstance(v, list) and isinstance(getattr(cfg, k, None), tuple):
            v = applied[k] = tuple(v)                # JSON tidak punya tuple: [2, 5] → (2, 5)
        setattr(cfg, k, v)
    print(f"Config file: {path} → {len(applied)} key diterapkan ke CFG")
    for k, v in applied.items():
        print(f"  CFG.{k} = {v!r}"[:160])
    if unknown:
        print("  ⚠️ key tidak dikenal (tetap di-set, cek ejaan):", unknown)
    return applied


TEMPLATE_CONFIG_APPLIED = apply_template_config(CFG)

# %% [markdown]
# ## 3. Cek & install library + finalisasi CFG
#
# **Tujuan:** memastikan semua library yang dibutuhkan ada, tanpa membuat notebook crash kalau library opsional
# tidak ada.
#
# - `ensure_packages()` mengecek tiap package dengan `importlib.util.find_spec`. Kalau belum ada **dan**
#   `CFG.AUTO_INSTALL = True`, dia menjalankan `pip install`. Gagal install = hanya warning, notebook lanjut.
# - **Torch tidak pernah di-install otomatis** (ukurannya besar dan versi CUDA harus cocok dengan driver).
#   Kalau torch tidak ada: classical + TF-IDF tetap jalan; transformer & sentence-embedding di-skip.
# - Setelah itu tiap library opsional di-import dalam `try/except` dan diberi flag `HAS_xxx`.
#
# **Cara membaca output:** tabel `Environment` menampilkan versi tiap library. `False` di kolom available berarti
# blok yang membutuhkan library itu akan di-skip dengan pesan jelas.
#
# **Jika** muncul `HAS_TORCH=False` padahal kamu butuh transformer → install manual sesuai CUDA:
# `pip install torch --index-url https://download.pytorch.org/whl/cu121` (CPU: `.../whl/cpu`), lalu restart kernel.
# **Jika** `pip install` gagal karena proxy/offline → set `CFG.AUTO_INSTALL = False` dan pakai yang ada.
# **Jika** baru saja install package → kadang perlu **Restart Kernel** supaya ter-load.

# %%
def finalize_cfg():
    """Terapkan RUN_MODE, buat folder output, dan rapikan beberapa nilai CFG."""
    CFG.RUN_MODE = str(CFG.RUN_MODE).lower()
    if CFG.RUN_MODE == "fast":
        CFG.N_FOLDS = min(CFG.N_FOLDS, CFG.FAST_N_FOLDS)
        CFG.EPOCHS = min(CFG.EPOCHS, CFG.FAST_EPOCHS)
        CFG.TFIDF_WORD_MAX_FEATURES = min(CFG.TFIDF_WORD_MAX_FEATURES, 50_000)
        CFG.TFIDF_CHAR_MAX_FEATURES = min(CFG.TFIDF_CHAR_MAX_FEATURES, 80_000)
        CFG.LGBM_N_ESTIMATORS = min(CFG.LGBM_N_ESTIMATORS, 300)
    CFG.OUTPUT_DIR = os.path.join(CFG.OUTPUT_DIR, "")
    for sub in ["", "figures", "tables", "models", "submissions", "preds", "cache"]:
        os.makedirs(os.path.join(CFG.OUTPUT_DIR, sub), exist_ok=True)
    print(f"RUN_MODE = {CFG.RUN_MODE} | N_FOLDS = {CFG.N_FOLDS} | EPOCHS = {CFG.EPOCHS}")
    print("Output dir:", os.path.abspath(CFG.OUTPUT_DIR))


# import-name → pip-name
CORE_PACKAGES = {
    "sklearn": "scikit-learn", "scipy": "scipy", "matplotlib": "matplotlib", "tqdm": "tqdm",
}
OPTIONAL_PACKAGES = {
    "seaborn": "seaborn", "lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost",
    "transformers": "transformers", "sentence_transformers": "sentence-transformers",
    "accelerate": "accelerate", "sentencepiece": "sentencepiece", "google.protobuf": "protobuf",
    "wordcloud": "wordcloud", "shap": "shap", "openpyxl": "openpyxl", "pyarrow": "pyarrow",
}
# Package yang menarik torch sebagai dependency → hanya di-install kalau torch SUDAH ada
TORCH_DEPENDENT = {"sentence_transformers", "accelerate"}


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def ensure_packages(packages, auto_install=True):
    """Cek package; install yang belum ada kalau auto_install=True. Tidak pernah raise error."""
    torch_present = _has_module("torch")
    status = {}
    for mod, pip_name in packages.items():
        if _has_module(mod):
            status[mod] = "ok"
            continue
        if mod in TORCH_DEPENDENT and not torch_present:
            status[mod] = "skip (butuh torch)"
            continue
        if not auto_install:
            status[mod] = "missing"
            continue
        try:
            print(f"Installing {pip_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *([] if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else ["--user"]), pip_name])
            importlib.invalidate_caches()
            status[mod] = "installed" if _has_module(mod) else "install gagal"
        except Exception as e:  # jangan pernah gagal di sini
            status[mod] = f"install gagal ({type(e).__name__})"
    return status


finalize_cfg()
_pkgs = dict(CORE_PACKAGES)
if CFG.INSTALL_OPTIONAL:
    _pkgs.update(OPTIONAL_PACKAGES)
PKG_STATUS = ensure_packages(_pkgs, auto_install=CFG.AUTO_INSTALL)
if not _has_module("torch"):
    print("⚠️ torch tidak terpasang. Transformer & sentence-embedding akan di-skip.\n"
          "   Install manual (pilih sesuai GPU): pip install torch --index-url https://download.pytorch.org/whl/cu121")

# %%
# ============================================================
# Import library inti + opsional (dengan flag HAS_xxx)
# ============================================================
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import rankdata, pearsonr, spearmanr, ks_2samp

import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier, SGDClassifier, SGDRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.naive_bayes import ComplementNB
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics as skm

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x

try:
    import seaborn as sns
    HAS_SNS = True
    sns.set_theme(style="whitegrid")
except Exception:
    sns, HAS_SNS = None, False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    lgb, HAS_LGB = None, False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb, HAS_XGB = None, False

try:
    import catboost as cb
    HAS_CB = True
except Exception:
    cb, HAS_CB = None, False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    torch, nn, F, HAS_TORCH = None, None, None, False

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    HAS_TRANSFORMERS = HAS_TORCH
except Exception:
    transformers, HAS_TRANSFORMERS = None, False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = HAS_TORCH
except Exception:
    SentenceTransformer, HAS_ST = None, False

HAS_ACCELERATE = _has_module("accelerate")
HAS_SENTENCEPIECE = _has_module("sentencepiece")

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    WordCloud, HAS_WORDCLOUD = None, False

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap, HAS_SHAP = None, False

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    HAS_SASTRAWI = True
except Exception:
    StemmerFactory, HAS_SASTRAWI = None, False

def _detect_cuda():
    """True hanya kalau CUDA benar-benar bisa dipakai (ada device & alokasi tensor berhasil)."""
    if not HAS_TORCH:
        return False
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return False
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


HAS_CUDA = _detect_cuda()
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu") if HAS_TORCH else "cpu"
FAILED_MODELS = {}   # nama model -> pesan error (model yang gagal tidak menghentikan notebook)


def _ver(mod):
    try:
        return getattr(mod, "__version__", "?") if mod is not None else "-"
    except Exception:
        return "?"


env_rows = [
    ("python", sys.version.split()[0], True), ("numpy", np.__version__, True), ("pandas", pd.__version__, True),
    ("scikit-learn", sklearn.__version__, True), ("torch", _ver(torch), HAS_TORCH),
    ("cuda", (torch.version.cuda if HAS_TORCH else "-"), HAS_CUDA),
    ("transformers", _ver(transformers), HAS_TRANSFORMERS),
    ("sentence-transformers", _ver(sys.modules.get("sentence_transformers")), HAS_ST),
    ("accelerate (HF Trainer)", "", HAS_ACCELERATE), ("sentencepiece", "", HAS_SENTENCEPIECE),
    ("lightgbm", _ver(lgb), HAS_LGB), ("xgboost", _ver(xgb), HAS_XGB), ("catboost", _ver(cb), HAS_CB),
    ("seaborn", _ver(sns), HAS_SNS), ("wordcloud", "", HAS_WORDCLOUD), ("shap", _ver(shap), HAS_SHAP),
    ("Sastrawi", "", HAS_SASTRAWI),
]
ENV_DF = pd.DataFrame(env_rows, columns=["library", "version", "available"])
display(ENV_DF)
if HAS_CUDA:
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"✅ GPU: {props.name} | VRAM {props.total_memory / 1024 ** 3:.1f} GB")
        if props.total_memory / 1024 ** 3 < 7:
            print("   VRAM kecil → BATCH_SIZE 8, MAX_LEN ≤ 256, USE_AMP=True, model base (bukan large).")
    except Exception as e:
        print("✅ GPU terdeteksi (properti tidak bisa dibaca:", short_err(e, 80) if "short_err" in globals() else e, ")")
else:
    print("⚠️ Tidak ada CUDA GPU → fine-tuning transformer di-skip kalau RUN_TRANSFORMER='auto'.")

# %% [markdown]
# ## 4. Utilities: seed, figure/table saver, smart_read, timer, experiment log
#
# - `seed_everything` → hasil bisa direproduksi (random, numpy, torch).
# - `save_fig(name)` → simpan figure ke `figures/figNN_name.png` dengan DPI `CFG.FIG_DPI`
#   (nomor otomatis, jadi gampang dirujuk di laporan: "Gambar 3 menunjukkan ...").
# - `save_table(df, name)` → simpan tabel ke `tables/name.csv` (copy-paste ke laporan).
# - `smart_read(path)` → baca csv/tsv/xlsx/parquet/json/feather/pickle, deteksi separator & encoding otomatis.

# %%
def seed_everything(seed=42):
    """Set seed untuk random, numpy, PYTHONHASHSEED, dan torch (kalau ada)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


seed_everything(CFG.SEED)
plt.rcParams.update({"figure.dpi": 100, "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
_FIG_COUNTER = [0]


def save_fig(name, fig=None):
    """Simpan figure aktif ke figures/figNN_<name>.png lalu tampilkan/tutup."""
    fig = fig or plt.gcf()
    _FIG_COUNTER[0] += 1
    fname = f"fig{_FIG_COUNTER[0]:02d}_{re.sub(r'[^A-Za-z0-9_]+', '_', name)}.png"
    path = os.path.join(CFG.OUTPUT_DIR, "figures", fname)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight")
    if CFG.SHOW_FIGS:
        plt.show()
    plt.close(fig)
    print(f"[figure saved] {path}")
    return path


def save_table(df, name, index=False):
    """Simpan DataFrame ke tables/<name>.csv."""
    path = os.path.join(CFG.OUTPUT_DIR, "tables", f"{name}.csv")
    df.to_csv(path, index=index)
    print(f"[table saved] {path}")
    return path


class Timer:
    """Context manager kecil untuk mengukur waktu: with Timer('nama'): ..."""

    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.seconds = time.time() - self.t0
        print(f"[time] {self.name}: {self.seconds:.1f}s")


def smart_read(path, **kwargs):
    """Baca file tabular apapun berdasarkan ekstensi. CSV: coba beberapa encoding & sniff separator."""
    if path is None or not os.path.exists(str(path)):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    ext = os.path.splitext(str(path))[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)
    if ext == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if ext in [".json", ".jsonl"]:
        try:
            return pd.read_json(path, **kwargs)
        except ValueError:
            return pd.read_json(path, lines=True, **kwargs)
    if ext == ".feather":
        return pd.read_feather(path, **kwargs)
    if ext in [".pkl", ".pickle"]:
        return pd.read_pickle(path, **kwargs)
    sep = "\t" if ext == ".tsv" else None
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            if sep is not None:
                return pd.read_csv(path, sep=sep, encoding=enc, **kwargs)
            df = pd.read_csv(path, encoding=enc, **kwargs)
            if df.shape[1] == 1 and any(s in str(df.columns[0]) for s in [";", "\t", "|"]):
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc, **kwargs)
            return df
        except UnicodeDecodeError as e:
            last_err = e
        except pd.errors.ParserError as e:
            last_err = e
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc, **kwargs)
            except Exception as e2:
                last_err = e2
    raise last_err


def log_experiment(name, score, extra=None):
    """Tambahkan satu baris ke experiment_log.csv (riwayat semua eksperimen lintas run)."""
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "run_mode": CFG.RUN_MODE, "model": name,
           "metric": CFG.METRIC, "score": score, "n_folds": CFG.N_FOLDS}
    if extra:
        row.update(extra)
    path = os.path.join(CFG.OUTPUT_DIR, "experiment_log.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def free_memory():
    """Bersihkan RAM & VRAM."""
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def short_err(e, n=300):
    """Ringkas pesan error untuk FAILED_MODELS."""
    return f"{type(e).__name__}: {str(e)[:n]}"


print("Utilities siap.")

# %% [markdown]
# ## 5. Load data
#
# **Tujuan:** membaca train/test/sample_submission dan memastikan kolom di CFG ada.
#
# **Cara membaca output:** `Train shape` / `Test shape`. Test **tidak ada** → notebook tetap jalan (hanya CV).
# Kalau `TEXT_COL` tidak ditemukan → dipilih otomatis kolom teks terpanjang (⚠️ cek!). Kalau `TARGET_COL` tidak
# ditemukan → ditebak dari kolom yang ada di train tapi tidak di test.
#
# **Error umum:** `FileNotFoundError` → path salah (cek `os.getcwd()`); separator `;` dan encoding sudah otomatis.

# %%
print("Working dir:", os.getcwd())
train_raw = smart_read(CFG.TRAIN_PATH)
HAS_TEST = bool(CFG.TEST_PATH) and os.path.exists(str(CFG.TEST_PATH))
test_raw = smart_read(CFG.TEST_PATH) if HAS_TEST else None
sample_sub = smart_read(CFG.SAMPLE_SUB_PATH) if (CFG.SAMPLE_SUB_PATH and os.path.exists(str(CFG.SAMPLE_SUB_PATH))) else None

print("Train shape:", train_raw.shape)
print("Test shape :", test_raw.shape if HAS_TEST else "TIDAK ADA → hanya CV, submission di-skip")
print("Sample sub :", sample_sub.shape if sample_sub is not None else "tidak ada")


def guess_text_col(df):
    """Pilih kolom object dengan rata-rata panjang string terbesar."""
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return None
    lens = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
    return max(lens, key=lens.get)


if CFG.TEXT_COL not in train_raw.columns:
    guess = guess_text_col(train_raw.drop(columns=[CFG.TARGET_COL], errors="ignore"))
    print(f"⚠️ TEXT_COL '{CFG.TEXT_COL}' tidak ada. Kolom: {list(train_raw.columns)} → memakai '{guess}'")
    CFG.TEXT_COL = guess
if CFG.TARGET_COL not in train_raw.columns:
    cand = [c for c in train_raw.columns if HAS_TEST and c not in test_raw.columns and c != CFG.ID_COL]
    if len(cand) == 1:
        print(f"⚠️ TARGET_COL '{CFG.TARGET_COL}' tidak ada → memakai '{cand[0]}'")
        CFG.TARGET_COL = cand[0]
    else:
        raise KeyError(f"TARGET_COL '{CFG.TARGET_COL}' tidak ditemukan. Kolom train: {list(train_raw.columns)}")
missing_extra = [c for c in CFG.EXTRA_TEXT_COLS + CFG.META_COLS if c not in train_raw.columns]
if missing_extra:
    print(f"⚠️ Kolom EXTRA/META tidak ada dan diabaikan: {missing_extra}")
    CFG.EXTRA_TEXT_COLS = [c for c in CFG.EXTRA_TEXT_COLS if c in train_raw.columns]
    CFG.META_COLS = [c for c in CFG.META_COLS if c in train_raw.columns]
display(train_raw.head())
if HAS_TEST:
    display(test_raw.head())

# %% [markdown]
# ## 6. Parsing target & analisis target
#
# **Parsing:** target yang tersimpan sebagai teks (mis. `"5,510.1"`, `"1.234,5"`, `"Rp 10.000"`, `"12 bulan"`)
# diubah ke angka. Baris dengan target kosong/tidak bisa di-parse **dibuang** (dicetak jumlahnya).
#
# **Analisis yang dicetak:**
# - statistik deskriptif, **skewness**, jumlah outlier (aturan IQR),
# - apakah target **diskrit** (≤ 20 nilai unik bulat, mis. rating 1–5) → `VALID_VALUES` otomatis,
# - rekomendasi transform.
#
# **Jika X maka Y:**
# - skew > 1 dan target ≥ 0 → `log1p` (dipilih otomatis kalau `TARGET_TRANSFORM="auto"`). Model belajar error
#   relatif, bukan absolut → prediksi nilai kecil tidak "ditarik" oleh nilai ekstrem.
# - target diskrit (rating) → regresi tetap bagus untuk RMSE; untuk QWK/accuracy coba rounding & threshold
#   optimal (section post-processing). Alternatif: perlakukan sebagai klasifikasi ordinal (notebook klasifikasi).
# - banyak outlier ekstrem → `REG_LOSS="huber"`, cek apakah outlier = salah input.
#
# **Untuk laporan:** histogram target (asli & setelah transform) + kalimat tentang skewness & transform.

# %%
def parse_numeric(series):
    """Parse angka dari string: buang simbol, tangani pemisah ribuan/desimal (format ID & EN)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = re.sub(r"[^0-9,.\-eE]", "", str(v).strip())
        if not s:
            return np.nan
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".") if re.search(r",\d{1,2}$", s) else s.replace(",", "")
        elif s.count(".") > 1:
            s = s.replace(".", "")
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.apply(one)


train = train_raw.copy()
test = test_raw.copy() if HAS_TEST else None
train[CFG.TARGET_COL] = parse_numeric(train[CFG.TARGET_COL])
n_bad = int(train[CFG.TARGET_COL].isna().sum() + np.isinf(train[CFG.TARGET_COL]).sum())
if n_bad:
    print(f"⚠️ {n_bad} baris target kosong / tidak bisa di-parse → dibuang.")
train = train[np.isfinite(train[CFG.TARGET_COL])].reset_index(drop=True)
TASK = "regression"

if CFG.RUN_MODE == "fast" and len(train) > CFG.FAST_SAMPLE_ROWS:
    train = train.sample(CFG.FAST_SAMPLE_ROWS, random_state=CFG.SEED).reset_index(drop=True)
    print(f"FAST mode: train di-subsample → {len(train)} baris")

Y = train[CFG.TARGET_COL].values.astype(float)
y_desc = pd.Series(Y).describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
SKEW = float(pd.Series(Y).skew())
q1, q3 = np.percentile(Y, [25, 75])
n_out = int(((Y < q1 - 3 * (q3 - q1)) | (Y > q3 + 3 * (q3 - q1))).sum())
uniq = np.unique(Y)
IS_DISCRETE = len(uniq) <= 20 and np.allclose(uniq, np.round(uniq))
if CFG.VALID_VALUES is None and IS_DISCRETE:
    CFG.VALID_VALUES = [float(v) for v in uniq]
VALID_VALUES = np.array(sorted(CFG.VALID_VALUES), dtype=float) if CFG.VALID_VALUES is not None else None
MIN_T = float(np.min(Y)) if CFG.MIN_TARGET is None else float(CFG.MIN_TARGET)
MAX_T = float(np.max(Y)) if CFG.MAX_TARGET is None else float(CFG.MAX_TARGET)

if CFG.TARGET_TRANSFORM == "auto":
    CFG.TARGET_TRANSFORM = "log1p" if (SKEW > 1 and Y.min() >= 0 and not IS_DISCRETE) else "none"
if CFG.TARGET_TRANSFORM in ("log1p", "sqrt") and Y.min() < 0:
    print(f"⚠️ {CFG.TARGET_TRANSFORM} butuh target ≥ 0 → transform dimatikan.")
    CFG.TARGET_TRANSFORM = "none"

display(y_desc.to_frame("target").T.round(4))
print(f"Skewness = {SKEW:.3f}", "⚠️ skewed" if abs(SKEW) > 1 else "✅")
print(f"Outliers (3×IQR) = {n_out} ({n_out / len(Y):.1%})")
print(f"Discrete target = {IS_DISCRETE} | VALID_VALUES = {None if VALID_VALUES is None else VALID_VALUES.tolist()}")
print(f"Clip range = [{MIN_T}, {MAX_T}] | TARGET_TRANSFORM = {CFG.TARGET_TRANSFORM}")
if IS_DISCRETE and CFG.METRIC not in ("qwk",):
    print("💡 Target diskrit: kalau metric resmi QWK/accuracy, set CFG.METRIC='qwk'. Rounding dicoba di post-processing.")


def transform_target(y):
    """Skala asli → skala training."""
    y = np.asarray(y, dtype=float)
    if CFG.TARGET_TRANSFORM == "log1p":
        return np.log1p(np.clip(y, 0, None))
    if CFG.TARGET_TRANSFORM == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    return y


def inverse_transform_target(z):
    """Skala training → skala asli (aman dari overflow)."""
    z = np.nan_to_num(np.asarray(z, dtype=float), nan=0.0, posinf=50.0, neginf=-50.0)
    if CFG.TARGET_TRANSFORM == "log1p":
        return np.expm1(np.clip(z, -20, np.log1p(max(MAX_T, 1.0)) + 2))
    if CFG.TARGET_TRANSFORM == "sqrt":
        return np.clip(z, 0, None) ** 2
    return z


Y_T = transform_target(Y)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(Y, bins=min(50, max(10, len(uniq))), color="#4C72B0")
axes[0].set_title("Target distribution (original)")
axes[0].set_xlabel(CFG.TARGET_COL)
axes[0].set_ylabel("Count")
axes[1].hist(Y_T, bins=50, color="#DD8452")
axes[1].set_title(f"Target distribution (transform: {CFG.TARGET_TRANSFORM})")
axes[1].set_xlabel(f"{CFG.TARGET_TRANSFORM}({CFG.TARGET_COL})")
axes[1].set_ylabel("Count")
save_fig("target_distribution", fig)
save_table(y_desc.to_frame("target").reset_index(), "target_summary")

# %% [markdown]
# ## 7. Inspeksi dasar & sanity check
#
# Cek: tipe data & missing, teks kosong, ID unik, ID test ada di train, kolom train vs test, dan **leakage
# heuristik**: korelasi ID dengan target (data terurut), angka target tertulis di teks, metadata yang menentukan target.
#
# **Jika** korelasi ID–target tinggi → urutan data mengandung informasi (mis. waktu). Jangan pakai ID sebagai
# fitur; pertimbangkan validasi berbasis waktu. **Jika** target tertulis di teks (mis. "rating 5/5") → model akan
# "menyontek"; pastikan pola sama juga ada di test, atau bersihkan.

# %%
def build_raw_text(df):
    """Gabungkan TEXT_COL + EXTRA_TEXT_COLS."""
    out = df[CFG.TEXT_COL].fillna("").astype(str)
    for c in CFG.EXTRA_TEXT_COLS:
        if c in df.columns:
            out = out.str.cat(df[c].fillna("").astype(str), sep=" | ")
    return out


train["raw_text"] = build_raw_text(train)
if HAS_TEST:
    test["raw_text"] = build_raw_text(test)

info = pd.DataFrame({"dtype": train.dtypes.astype(str), "n_missing": train.isna().sum(),
                     "pct_missing": (train.isna().mean() * 100).round(2), "n_unique": train.nunique()})
display(info)
checks = []
n_empty = int((train["raw_text"].str.strip() == "").sum())
checks.append(("empty texts (train)", n_empty, "⚠️" if n_empty else "✅"))
if HAS_TEST:
    n_empty_te = int((test["raw_text"].str.strip() == "").sum())
    checks.append(("empty texts (test)", n_empty_te, "⚠️" if n_empty_te else "✅"))
if CFG.ID_COL in train.columns:
    dup_id = int(train[CFG.ID_COL].duplicated().sum())
    checks.append(("duplicate IDs (train)", dup_id, "⚠️" if dup_id else "✅"))
    if HAS_TEST and CFG.ID_COL in test.columns:
        overlap = len(set(train[CFG.ID_COL]) & set(test[CFG.ID_COL]))
        checks.append(("test IDs also in train", overlap, "⚠️" if overlap else "✅"))
if HAS_TEST:
    only_tr = sorted(set(train.columns) - set(test.columns) - {CFG.TARGET_COL})
    only_te = sorted(set(test.columns) - set(train.columns))
    checks.append(("cols only in train", len(only_tr), str(only_tr[:10]) if only_tr else "✅"))
    checks.append(("cols only in test", len(only_te), str(only_te[:10]) if only_te else "✅"))
display(pd.DataFrame(checks, columns=["check", "value", "status"]))

leak_notes = []
if CFG.ID_COL in train.columns:
    id_num = pd.to_numeric(train[CFG.ID_COL].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    if id_num.notna().mean() > 0.9:
        rho = spearmanr(id_num.fillna(id_num.median()), Y).correlation
        leak_notes.append(f"Spearman(ID, target) = {rho:.3f}" + ("  ⚠️ ID berkorelasi dengan target!" if abs(rho) > 0.3 else "  ✅"))
if IS_DISCRETE:
    pat = train["raw_text"].str.contains(r"\b[1-5]\s*/\s*5\b|\b(bintang|stars?)\s*[1-5]\b", case=False, regex=True)
    if pat.mean() > 0.05:
        leak_notes.append(f"{pat.mean():.0%} teks menyebut rating eksplisit ('4/5', 'bintang 5') → kemungkinan leakage")
for mc in CFG.META_COLS:
    r2_meta = train.groupby(mc)[CFG.TARGET_COL].transform("mean")
    share = 1 - np.var(Y - r2_meta) / max(np.var(Y), 1e-12)
    leak_notes.append(f"META '{mc}': menjelaskan {share:.1%} variance target (rata-rata per kategori)")
print("Leakage / signal heuristics:")
for n in leak_notes or ["✅ tidak ada tanda leakage yang jelas"]:
    print("  -", n)

# %% [markdown]
# ## 8. Text cleaning & normalisasi Bahasa Indonesia
#
# **Prinsip utama (hafalkan):**
#
# | Jalur model | Cleaning yang disarankan | Alasan |
# |---|---|---|
# | **Transformer** (IndoBERT, XLM-R, DeBERTa) | **minimal**: decode HTML, URL→`<URL>`, mention→`<USER>`, rapikan spasi | tokenizer subword sudah menangani variasi kata; slang, tanda baca, emoji, huruf kapital bisa jadi sinyal. Cleaning agresif sering **menurunkan** skor. |
# | **TF-IDF / classical** | **lebih banyak**: lowercase, slang→baku (`gak`→`tidak`), elongation (`mantappp`→`mantap`), opsional stopword & stemming | TF-IDF menghitung kata persis; menyatukan variasi ejaan = fitur lebih padat & kuat. |
#
# Karena itu notebook membuat **dua kolom teks**:
# - `text_tf` → untuk transformer & sentence-embedding (cleaning minimal),
# - `text_cl` → untuk TF-IDF (normalisasi lebih kuat).
#
# **Kamus slang** di bawah hanya contoh kecil (~100 kata umum). Tambahkan kata khas data lomba lewat
# `CFG.CUSTOM_SLANG_DICT` (cek daftar top kata di EDA untuk menemukan slang yang sering muncul).
#
# **Stopword:** daftar kecil bawaan (ID + EN), **tanpa download**. Default **tidak** dihapus karena kata seperti
# "tidak", "bukan", "kurang" membalik makna (sentimen!). Daftar stopword di sini sudah mengecualikan kata negasi.
#
# **Stemming** (Sastrawi) hanya kalau package `Sastrawi` terpasang; lambat dan jarang menaikkan skor klasifikasi.
#
# **Jika** skor TF-IDF turun setelah mengaktifkan stopword/stemming → matikan lagi (selalu bandingkan OOF).
# **Untuk laporan:** jelaskan dua strategi preprocessing ini dan alasannya (1 paragraf + tabel contoh before/after).

# %%
SLANG_DICT_ID = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak",
    "engga": "tidak", "tdk": "tidak", "kagak": "tidak", "yg": "yang", "yng": "yang",
    "bgt": "banget", "bngt": "banget", "bet": "banget", "dgn": "dengan", "dg": "dengan", "sdh": "sudah",
    "udh": "sudah", "udah": "sudah", "dah": "sudah", "blm": "belum", "blom": "belum", "krn": "karena",
    "karna": "karena", "tp": "tapi", "tpi": "tapi", "jg": "juga", "jga": "juga", "aja": "saja", "aj": "saja",
    "sy": "saya", "gw": "saya", "gue": "saya", "gua": "saya", "lu": "kamu", "lo": "kamu", "elo": "kamu",
    "km": "kamu", "kmu": "kamu", "bgs": "bagus", "bkn": "bukan", "org": "orang", "utk": "untuk", "untk": "untuk",
    "sm": "sama", "sma": "sama", "dr": "dari", "dri": "dari", "klo": "kalau", "kalo": "kalau", "kl": "kalau",
    "gmn": "bagaimana", "gimana": "bagaimana", "knp": "kenapa", "napa": "kenapa", "emg": "memang",
    "emang": "memang", "bnyk": "banyak", "byk": "banyak", "msh": "masih", "lg": "lagi", "lgi": "lagi",
    "skrg": "sekarang", "skrng": "sekarang", "trs": "terus", "trus": "terus", "sampe": "sampai",
    "smpe": "sampai", "bs": "bisa", "bsa": "bisa", "pake": "pakai", "pk": "pakai", "mantul": "mantap",
    "mantab": "mantap", "thx": "terima kasih", "makasih": "terima kasih", "mksh": "terima kasih",
    "trims": "terima kasih", "tq": "terima kasih", "ok": "oke", "okey": "oke", "okay": "oke", "bener": "benar",
    "bnr": "benar", "dpt": "dapat", "dapet": "dapat", "hrs": "harus", "jd": "jadi", "jdi": "jadi",
    "kyk": "seperti", "kayak": "seperti", "kek": "seperti", "cm": "cuma", "doang": "saja", "lemot": "lambat",
    "ancur": "hancur", "nyesel": "menyesal", "bikin": "membuat", "gede": "besar", "dikit": "sedikit",
    "cepet": "cepat", "sll": "selalu", "slalu": "selalu", "brg": "barang", "hrg": "harga", "ongkir": "ongkos kirim",
    "rekomen": "rekomendasi", "pengen": "ingin", "pgn": "ingin", "tau": "tahu", "tw": "tahu",
    "gpp": "tidak apa apa", "gapapa": "tidak apa apa", "wkwk": "haha", "wkwkwk": "haha", "btw": "ngomong ngomong",
}

STOPWORDS_ID = set("""
ada adalah adanya agar akan akhirnya aku amat anda antara apa apakah apalagi atas atau ataupun bagai bagaimana
bagi bahkan bahwa banyak beberapa begitu belakang berapa berbagai beri berikut bersama biasa bila bisa boleh
cukup dalam dan dapat dari daripada demikian dengan di dia diri dirinya hal hampir hanya harus hingga ia ialah
ini itu jadi jika juga kami kamu kan karena kata ke kemudian kepada ketika kita lagi lain lalu lebih maka
masih mau melalui memang mereka meski mungkin namun nanti nya oleh pada para pernah pula pun saat saja saya
sebagai sebelum sedang sehingga sejak sekali selain seluruh semua sendiri seperti serta setelah sini situ
suatu sudah supaya tadi tanpa tapi telah tentang terhadap tersebut tetapi untuk walau yaitu yakni yang
""".split())
STOPWORDS_EN = set("""
a about above after again all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
him his how i if in into is it its itself just me more most my myself of off on once only or other our ours
out over own same she should so some such than that the their theirs them then there these they this those
through to too under until up very was we were what when where which while who whom why will with would you
your yours
""".split())
# kata negasi/intensitas JANGAN dibuang (membalik makna)
KEEP_WORDS = {"tidak", "bukan", "belum", "jangan", "kurang", "no", "not", "nor", "never", "very", "banget", "sangat"}
STOPWORDS_ALL = (STOPWORDS_ID | STOPWORDS_EN) - KEEP_WORDS

SLANG_DICT = {**SLANG_DICT_ID, **{str(k).lower(): str(v) for k, v in CFG.CUSTOM_SLANG_DICT.items()}}
_STEMMER = StemmerFactory().create_stemmer() if (CFG.USE_STEMMING and HAS_SASTRAWI) else None
if CFG.USE_STEMMING and not HAS_SASTRAWI:
    print("⚠️ USE_STEMMING=True tapi Sastrawi tidak terpasang → stemming di-skip (pip install Sastrawi).")

_RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
_RE_USER = re.compile(r"@\w+")
_RE_WS = re.compile(r"\s+")
_RE_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF]+")
_RE_TOKEN = re.compile(r"[a-zA-ZÀ-ɏ]+|\d+|<url>|<user>")


def basic_clean(text, lowercase=False):
    """Cleaning minimal (untuk transformer): HTML, URL, mention, hashtag, emoji opsional, whitespace."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = html.unescape(str(text))
    if CFG.NORMALIZE_URL:
        t = _RE_URL.sub(" <URL> ", t)
    if CFG.NORMALIZE_USER:
        t = _RE_USER.sub(" <USER> ", t)
    if CFG.KEEP_HASHTAG_TEXT:
        t = t.replace("#", " ")
    if CFG.REMOVE_EMOJI:
        t = _RE_EMOJI.sub(" ", t)
    t = _RE_WS.sub(" ", t).strip()
    return t.lower() if lowercase else t


def normalize_slang_tokens(tokens):
    """Ganti token slang dengan bentuk baku (bisa jadi >1 kata)."""
    out = []
    for tok in tokens:
        rep = SLANG_DICT.get(tok)
        out.extend(rep.split() if rep else [tok])
    return out


def classical_normalize(text):
    """Normalisasi lebih kuat untuk TF-IDF: lowercase, elongation, reduplikasi, slang, stopword, stemming."""
    t = basic_clean(text, lowercase=CFG.CLASSICAL_LOWERCASE)
    if CFG.REDUCE_ELONGATION:
        t = re.sub(r"([a-zA-Z])\1{2,}", r"\1", t)            # mantappp -> mantap
        t = re.sub(r"\b([a-zA-Z]+)2\b", r"\1 \1", t)          # bagus2 -> bagus bagus
    if CFG.REMOVE_NUMBERS_CLASSICAL:
        t = re.sub(r"\d+", " ", t)
    tokens = t.split()
    if CFG.USE_SLANG_NORMALIZATION:
        tokens = normalize_slang_tokens([tk.strip(".,!?;:\"'()[]") or tk for tk in tokens])
    if CFG.REMOVE_STOPWORDS_CLASSICAL:
        tokens = [tk for tk in tokens if tk.lower() not in STOPWORDS_ALL]
    t = " ".join(tokens)
    if _STEMMER is not None:
        t = _STEMMER.stem(t)
    return t


def add_meta_prefix(df, texts):
    """Sisipkan metadata sebagai teks: "[platform: shopee] ..."."""
    if not (CFG.USE_METADATA_AS_TEXT and CFG.META_COLS):
        return texts
    prefix = pd.Series([""] * len(df), index=df.index)
    for c in CFG.META_COLS:
        if c in df.columns:
            prefix = prefix + "[" + str(c) + ": " + df[c].fillna("NA").astype(str) + "] "
    return prefix + texts


def build_texts(df):
    """Kembalikan (text_tf, text_cl) untuk satu DataFrame."""
    tf = df["raw_text"].apply(lambda s: basic_clean(s, lowercase=CFG.TRANSFORMER_LOWERCASE))
    if CFG.TRANSFORMER_USE_SLANG:
        tf = tf.apply(lambda s: " ".join(normalize_slang_tokens(s.split())))
    cl = df["raw_text"].apply(classical_normalize)
    return add_meta_prefix(df, tf), add_meta_prefix(df, cl)


with Timer("text cleaning"):
    train["text_tf"], train["text_cl"] = build_texts(train)
    if HAS_TEST:
        test["text_tf"], test["text_cl"] = build_texts(test)

_examples = train[["raw_text", "text_tf", "text_cl"]].sample(min(6, len(train)), random_state=CFG.SEED)
print("Contoh before/after cleaning:")
display(_examples)
save_table(_examples, "cleaning_examples")

# %% [markdown]
# ### 8.1 Deteksi bahasa (heuristik stopword) → rekomendasi model
#
# **Cara kerja:** untuk tiap teks dihitung proporsi token yang termasuk stopword Indonesia vs stopword English.
# Teks diberi label `id`, `en`, `mixed`, atau `unknown`. Juga dihitung "skor sosmed" (proporsi teks berisi
# mention/hashtag/URL/emoji/slang).
#
# **Aturan rekomendasi:**
# - mayoritas `id` (> 60%) + skor sosmed tinggi → `indobertweet`; formal → `indobert_base`
# - mayoritas `en` (> 60%) → `deberta_v3_base`
# - campur → `xlm_roberta_base`
#
# Kalau `CFG.TRANSFORMER_MODELS = ["auto"]`, rekomendasi ini dipakai otomatis. Ini heuristik — kalau kamu tahu
# domainnya, pilih manual. **Untuk laporan:** sebutkan komposisi bahasa dan alasan pemilihan model.

# %%
def detect_language(text):
    """Heuristik bahasa per teks berbasis rasio stopword ID vs EN."""
    toks = [t.lower() for t in _RE_TOKEN.findall(str(text))]
    if not toks:
        return "unknown"
    n_id = sum(t in STOPWORDS_ID or t in SLANG_DICT_ID for t in toks)
    n_en = sum(t in STOPWORDS_EN for t in toks)
    if n_id + n_en == 0:
        return "unknown"
    r = n_id / (n_id + n_en)
    return "id" if r >= 0.7 else ("en" if r <= 0.3 else "mixed")


_lang_src = train["raw_text"].sample(min(len(train), 20000), random_state=CFG.SEED)
LANG_SERIES = train["raw_text"].apply(detect_language)
lang_share = LANG_SERIES.value_counts(normalize=True)
_social = _lang_src.str.contains(r"@\w+|#\w+|https?://|www\.", regex=True) | _lang_src.str.contains(_RE_EMOJI)
_slang_rate = _lang_src.str.lower().str.split().apply(lambda ts: any(t in SLANG_DICT_ID for t in ts))
SOCIAL_SCORE = float((_social | _slang_rate).mean())

share_id = lang_share.get("id", 0.0)
share_en = lang_share.get("en", 0.0)
if share_id > 0.6:
    RECOMMENDED_MODEL = "indobertweet" if SOCIAL_SCORE > 0.3 else "indobert_base"
elif share_en > 0.6:
    RECOMMENDED_MODEL = "deberta_v3_base"
else:
    RECOMMENDED_MODEL = "xlm_roberta_base"

print("Language share:", {k: f"{v:.1%}" for k, v in lang_share.items()})
print(f"Social-media/slang score: {SOCIAL_SCORE:.1%}")
print(f"➡️ Rekomendasi model transformer: {RECOMMENDED_MODEL} ({CFG.MODEL_POOL.get(RECOMMENDED_MODEL)})")
if CFG.TRANSFORMER_MODELS == ["auto"] or CFG.TRANSFORMER_MODELS == "auto":
    CFG.TRANSFORMER_MODELS = [RECOMMENDED_MODEL]
    print(f"   TRANSFORMER_MODELS auto → {CFG.TRANSFORMER_MODELS}")
if RECOMMENDED_MODEL in ("deberta_v3_base", "xlm_roberta_base") and not HAS_SENTENCEPIECE:
    print("   ⚠️ model ini mungkin butuh sentencepiece (tergantung versi transformers) → kalau load tokenizer gagal: "
          "pip install sentencepiece protobuf, lalu restart kernel")

fig, ax = plt.subplots(figsize=(6, 3.5))
lang_share.sort_index().plot(kind="bar", ax=ax, color="#55A868")
ax.set_title("Language composition (stopword heuristic)")
ax.set_xlabel("Detected language")
ax.set_ylabel("Share of texts")
save_fig("language_composition", fig)

# %% [markdown]
# ## 9. Text EDA — helper functions
#
# Cell ini hanya mendefinisikan fungsi EDA (panjang teks, token, n-gram, duplikat, near-duplicate, drift,
# meta-feature). Pemanggilannya di section berikutnya. Semua fungsi menerima `groups` (kelas untuk klasifikasi,
# bin target untuk regresi) supaya bisa membandingkan antar kelompok.

# %%
def resolve_model_name(key_or_name):
    """Key MODEL_POOL → nama HF (atau kembalikan apa adanya jika sudah nama/path)."""
    return CFG.MODEL_POOL.get(key_or_name, key_or_name)


_TOKENIZER_CACHE = {}


def load_tokenizer(model_name):
    """Load tokenizer dengan cache + pesan error yang jelas. Return None kalau gagal."""
    if not HAS_TRANSFORMERS:
        return None
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        msg = str(e)
        hint = ""
        if "sentencepiece" in msg.lower() or "tiktoken" in msg.lower():
            hint = " → pip install sentencepiece protobuf (lalu restart kernel)"
        elif "connect" in msg.lower() or "offline" in msg.lower() or "resolve" in msg.lower():
            hint = " → internet/HF Hub tidak bisa diakses; pakai model di cache/folder lokal"
        print(f"⚠️ Gagal load tokenizer '{model_name}': {type(e).__name__}{hint}")
        tok = None
    _TOKENIZER_CACHE[model_name] = tok
    return tok


def token_lengths(texts, tokenizer, max_rows=20000, seed=42):
    """Panjang token (tanpa truncation) untuk sampel teks. Fallback: kata × 1.3."""
    s = pd.Series(list(texts))
    if len(s) > max_rows:
        s = s.sample(max_rows, random_state=seed)
    if tokenizer is None:
        return (s.str.split().str.len().fillna(0) * 1.3).astype(int).values, False
    lens = []
    lst = s.tolist()
    for i in range(0, len(lst), 1000):
        enc = tokenizer(lst[i:i + 1000], add_special_tokens=True, truncation=False)["input_ids"]
        lens.extend(len(x) for x in enc)
    return np.array(lens), True


def recommend_max_len(tok_lens, percentile=95, cap=512):
    """MAX_LEN = persentil panjang token, dibulatkan ke atas kelipatan 32, dibatasi [32, cap]."""
    p = np.percentile(tok_lens, percentile)
    rec = int(min(cap, max(32, math.ceil(p / 32) * 32)))
    return rec


def plot_length_distributions(len_df, group=None, title_suffix=""):
    """Histogram char/word/token length, di-split per group (maks 8 group)."""
    cols = [c for c in ["n_chars", "n_words", "n_tokens"] if c in len_df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 3.8))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        data = len_df[c].clip(upper=len_df[c].quantile(0.99))
        if group is not None and pd.Series(group).nunique() <= 8:
            for g in pd.Series(group).dropna().unique():
                ax.hist(data[np.asarray(group) == g], bins=40, alpha=0.5, label=str(g), density=True)
            ax.legend(fontsize=8)
            ax.set_ylabel("Density")
        else:
            ax.hist(data, bins=40, color="#4C72B0")
            ax.set_ylabel("Count")
        ax.set_title(f"{c} distribution{title_suffix}")
        ax.set_xlabel(c.replace("n_", "Number of "))
    return fig


def top_ngrams_by_group(texts, groups, ngram_range=(1, 1), top_k=15, stop_words=None, min_df=2):
    """Top n-gram per group berdasarkan frekuensi & log-odds (kata khas group vs sisanya)."""
    vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=50000,
                          stop_words=list(stop_words) if stop_words else None, token_pattern=r"(?u)\b\w\w+\b")
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    groups = np.asarray(groups)
    rows = []
    total = np.asarray(X.sum(0)).ravel()
    for g in pd.Series(groups).dropna().unique():
        m = groups == g
        cnt_g = np.asarray(X[m].sum(0)).ravel()
        cnt_r = total - cnt_g
        lo = np.log((cnt_g + 1) / (cnt_g.sum() + len(vocab))) - np.log((cnt_r + 1) / (cnt_r.sum() + len(vocab)))
        lo[cnt_g < 3] = -np.inf
        for rank, j in enumerate(np.argsort(-cnt_g)[:top_k]):
            rows.append({"group": g, "type": "frequent", "rank": rank + 1, "ngram": vocab[j], "count": int(cnt_g[j]),
                         "log_odds": float(lo[j])})
        for rank, j in enumerate(np.argsort(-lo)[:top_k]):
            if np.isfinite(lo[j]):
                rows.append({"group": g, "type": "distinctive", "rank": rank + 1, "ngram": vocab[j],
                             "count": int(cnt_g[j]), "log_odds": float(lo[j])})
    return pd.DataFrame(rows)


def plot_distinctive_ngrams(ng_df, max_groups=6, title="Most distinctive n-grams per group (log-odds)"):
    """Bar chart n-gram paling khas per group."""
    d = ng_df[ng_df["type"] == "distinctive"]
    gs = list(d["group"].unique())[:max_groups]
    if not gs:
        return None
    fig, axes = plt.subplots(1, len(gs), figsize=(4.2 * len(gs), 4.5))
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, gs):
        sub = d[d["group"] == g].head(12).iloc[::-1]
        ax.barh(sub["ngram"], sub["log_odds"], color="#C44E52")
        ax.set_title(f"{g}", fontsize=11)
        ax.set_xlabel("Log-odds vs rest")
    fig.suptitle(title)
    return fig


def normalize_for_dup(s):
    """Normalisasi agresif untuk deteksi duplikat: lowercase, buang non-alfanumerik, rapikan spasi."""
    s = str(s).lower()
    s = re.sub(r"[^0-9a-zÀ-ɏ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def near_duplicate_pairs(texts, threshold=0.9, max_rows=20000, seed=42):
    """Pasangan near-duplicate via char TF-IDF + cosine nearest neighbor."""
    s = pd.Series(list(texts))
    idx = np.arange(len(s))
    if len(s) > max_rows:
        idx = np.sort(np.random.RandomState(seed).choice(len(s), max_rows, replace=False))
    sub = s.iloc[idx].tolist()
    if len(sub) < 3:
        return pd.DataFrame(columns=["i", "j", "similarity"])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200000)
    try:
        X = vec.fit_transform(sub)
    except ValueError:
        return pd.DataFrame(columns=["i", "j", "similarity"])
    nn_ = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute").fit(X)
    dist, nb = nn_.kneighbors(X)
    rows = []
    for a in range(len(sub)):
        b = nb[a, 1]
        sim = 1 - dist[a, 1]
        if sim >= threshold and a < b:
            rows.append({"i": int(idx[a]), "j": int(idx[b]), "similarity": float(sim)})
    return pd.DataFrame(rows, columns=["i", "j", "similarity"])


def vocab_overlap(train_texts, test_texts, min_count=1):
    """Persentase token test yang ada di vocab train + daftar kata OOV terbanyak."""
    tok = lambda s: re.findall(r"(?u)\b\w\w+\b", str(s).lower())
    tr_cnt = Counter(t for s in train_texts for t in tok(s))
    te_cnt = Counter(t for s in test_texts for t in tok(s))
    tr_vocab = {t for t, c in tr_cnt.items() if c >= min_count}
    total_te = sum(te_cnt.values()) or 1
    covered = sum(c for t, c in te_cnt.items() if t in tr_vocab)
    oov = Counter({t: c for t, c in te_cnt.items() if t not in tr_vocab})
    type_cov = len(set(te_cnt) & tr_vocab) / max(len(te_cnt), 1)
    return covered / total_te, type_cov, oov.most_common(20)


def adversarial_validation(train_texts, test_texts, max_rows=20000, seed=42):
    """AUC model yang membedakan train vs test (TF-IDF + LogReg, 3-fold). ~0.5 = mirip; >0.7 = drift."""
    tr = pd.Series(list(train_texts)).sample(min(len(train_texts), max_rows), random_state=seed)
    te = pd.Series(list(test_texts)).sample(min(len(test_texts), max_rows), random_state=seed)
    texts = pd.concat([tr, te], ignore_index=True)
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    oof = np.zeros(len(y))
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    coef_sum = None
    vec_last = None
    for tr_i, va_i in skf.split(texts, y):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        Xtr = vec.fit_transform(texts.iloc[tr_i])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, y[tr_i])
        oof[va_i] = clf.predict_proba(vec.transform(texts.iloc[va_i]))[:, 1]
        vec_last, coef_sum = vec, clf.coef_.ravel()
    auc = skm.roc_auc_score(y, oof)
    names = np.array(vec_last.get_feature_names_out())
    order = np.argsort(coef_sum)
    return auc, list(names[order[-15:]][::-1]), list(names[order[:15]])


def meta_features(texts):
    """Fitur sederhana dari teks (untuk EDA & laporan)."""
    s = pd.Series(list(texts)).fillna("").astype(str)
    letters = s.str.count(r"[A-Za-z]").replace(0, 1)
    return pd.DataFrame({
        "n_chars": s.str.len(), "n_words": s.str.split().str.len().fillna(0),
        "n_exclaim": s.str.count("!"), "n_question": s.str.count(r"\?"),
        "upper_ratio": s.str.count(r"[A-Z]") / letters, "n_url": s.str.count(r"https?://|www\.|<URL>"),
        "n_mention": s.str.count(r"@\w+|<USER>"), "n_hashtag": s.str.count(r"#\w+"),
        "n_emoji": s.str.count(_RE_EMOJI.pattern), "n_digits": s.str.count(r"\d"),
    })


print("EDA helpers siap.")

# %% [markdown]
# ## 10. Text EDA
#
# Untuk regresi, target dibagi menjadi `EDA_TARGET_GROUPS` kelompok (mis. low / mid / high berdasarkan kuantil)
# supaya teks bisa dibandingkan antar rentang target.
#
# ### 10.1 Panjang teks + rekomendasi `MAX_LEN` + korelasi panjang dengan target
#
# **Cara membaca:** tabel `% truncated at L` = persentase teks yang terpotong kalau `MAX_LEN = L`.
# `MAX_LEN` rekomendasi = persentil `MAX_LEN_PERCENTILE` panjang token (kelipatan 32).
#
# **Jika X maka Y:**
# - Spearman(panjang, target) besar (|ρ| > 0.3) → panjang teks adalah sinyal kuat (umum di skor esai!) →
#   jangan potong teks terlalu pendek; `head_tail`/`chunk` membantu; sebut di laporan.
# - Banyak teks > 512 token → `LONG_TEXT_STRATEGY="chunk"` atau `"head_tail"`, atau `modernbert_base`.
#
# **Untuk laporan:** histogram panjang + scatter panjang vs target + nilai korelasinya.

# %%
try:
    _grp = pd.qcut(Y, q=CFG.EDA_TARGET_GROUPS, duplicates="drop")
    _label_names = np.array([f"{iv.left:.3g}–{iv.right:.3g}" for iv in _grp])
except Exception:
    _label_names = np.array(["all"] * len(Y))


def _group_key(s):
    try:
        return float(str(s).split("–")[0])
    except ValueError:
        return 0.0


GROUP_ORDER = sorted(pd.unique(_label_names), key=_group_key)
EDA_TOK_NAME = resolve_model_name(CFG.EDA_TOKENIZER_MODEL or CFG.TRANSFORMER_MODELS[0])
eda_tok = load_tokenizer(EDA_TOK_NAME) if HAS_TRANSFORMERS else None
tok_lens_tr, exact_tok = token_lengths(train["text_tf"], eda_tok, seed=CFG.SEED)
print(f"Tokenizer EDA: {EDA_TOK_NAME if exact_tok else 'tidak tersedia → estimasi kata×1.3'}")

len_df = pd.DataFrame({"n_chars": train["text_tf"].str.len(), "n_words": train["text_tf"].str.split().str.len()})
if len(tok_lens_tr) == len(train):
    len_df["n_tokens"] = tok_lens_tr
summary_len = len_df.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).T.round(1)
display(summary_len)
save_table(summary_len.reset_index(), "text_length_summary")
trunc = pd.DataFrame({"MAX_LEN": [64, 128, 256, 384, 512],
                      "% truncated": [float((tok_lens_tr > L).mean() * 100) for L in [64, 128, 256, 384, 512]]}).round(2)
display(trunc)
RECOMMENDED_MAX_LEN = recommend_max_len(tok_lens_tr, CFG.MAX_LEN_PERCENTILE)
print(f"➡️ MAX_LEN rekomendasi (p{CFG.MAX_LEN_PERCENTILE}) = {RECOMMENDED_MAX_LEN}")
if str(CFG.MAX_LEN).lower() == "auto":
    CFG.MAX_LEN = RECOMMENDED_MAX_LEN
    print(f"   CFG.MAX_LEN auto → {CFG.MAX_LEN}")
rho_len = spearmanr(len_df["n_words"], Y).correlation
print(f"Spearman(word count, target) = {rho_len:.3f}", "⚠️ panjang teks = sinyal kuat" if abs(rho_len) > 0.3 else "")

if CFG.RUN_EDA:
    fig = plot_length_distributions(len_df, _label_names if len(tok_lens_tr) == len(train) else None, " by target group")
    save_fig("text_length_distribution", fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(len_df["n_words"], Y, s=8, alpha=0.4)
    ax.set_xscale("symlog")
    ax.set_title(f"Word count vs target (Spearman ρ = {rho_len:.2f})")
    ax.set_xlabel("Number of words")
    ax.set_ylabel(CFG.TARGET_COL)
    save_fig("length_vs_target", fig)
    if HAS_TEST:
        ks = ks_2samp(len_df["n_words"], test["text_tf"].str.split().str.len())
        print(f"KS test word-length train vs test: stat={ks.statistic:.3f}", "⚠️ berbeda" if ks.statistic > 0.1 else "✅ mirip")

# %% [markdown]
# ### 10.2 N-gram khas per kelompok target
#
# `distinctive` = n-gram dengan log-odds tertinggi di kelompok target tsb dibanding kelompok lain → kata yang
# berasosiasi dengan target rendah vs tinggi (mis. "rusak", "kecewa" di rating rendah). Bagus untuk laporan.
# Hubungan linear kata–target yang lebih presisi ada di section interpretability (koefisien Ridge).

# %%
if CFG.RUN_EDA:
    try:
        NGRAM_DF = top_ngrams_by_group(train["text_cl"], _label_names, (1, 2), CFG.TOP_NGRAMS, STOPWORDS_ALL)
        save_table(NGRAM_DF, "top_ngrams_per_target_group")
        display(NGRAM_DF[NGRAM_DF["type"] == "distinctive"].groupby("group").head(8))
        fig = plot_distinctive_ngrams(NGRAM_DF, title="Most distinctive n-grams per target group (log-odds)")
        if fig is not None:
            save_fig("distinctive_ngrams_per_target_group", fig)
    except Exception as e:
        print("n-gram EDA dilewati:", short_err(e))

# %% [markdown]
# ### 10.3 Duplikat, near-duplicate, dan target konflik
#
# Teks identik (setelah normalisasi) dengan target berbeda = **noise label** → batas bawah error yang tidak bisa
# dikalahkan model. Std target di dalam grup duplikat dicetak.
#
# **Jika** duplikat banyak (> 5%) → `CFG.GROUP_COL = "dup_group"` supaya duplikat tidak terpecah antar fold (CV jujur).
# **Jika** banyak teks test sudah ada di train → cek aturan lomba (nilai train bisa jadi patokan).

# %%
train["text_norm"] = train["raw_text"].apply(normalize_for_dup)
train["dup_group"] = train.groupby("text_norm").ngroup()
_nonempty = train["text_norm"] != ""
n_dup = int(train.loc[_nonempty, "text_norm"].duplicated().sum())
print(f"Exact duplicates (normalized) in train: {n_dup} ({n_dup / len(train):.1%})")
if n_dup:
    g = pd.DataFrame({"t": train["text_norm"], "y": Y})[_nonempty.values].groupby("t")["y"].agg(["size", "std"])
    conf = g[(g["size"] > 1) & (g["std"] > 0)]
    print(f"Duplicate texts with DIFFERENT targets: {len(conf)}", "⚠️ noise" if len(conf) else "✅")
    if len(conf):
        print(f"   mean within-duplicate target std = {conf['std'].mean():.4f}")
if HAS_TEST:
    test["text_norm"] = test["raw_text"].apply(normalize_for_dup)
    overlap = (test["text_norm"].isin(set(train.loc[_nonempty, "text_norm"])) & (test["text_norm"] != "")).mean()
    print(f"Test texts that also appear in train: {overlap:.1%}", "⚠️" if overlap > 0.02 else "✅")
if CFG.RUN_EDA:
    with Timer("near-duplicate search"):
        NEAR_DUP = near_duplicate_pairs(train["text_norm"], CFG.NEAR_DUP_THRESHOLD, CFG.NEAR_DUP_MAX_ROWS, CFG.SEED)
    if len(NEAR_DUP):
        NEAR_DUP = NEAR_DUP[train["text_norm"].values[NEAR_DUP["i"]] != train["text_norm"].values[NEAR_DUP["j"]]]
    print(f"Near-duplicate pairs (sim ≥ {CFG.NEAR_DUP_THRESHOLD}, excluding exact): {len(NEAR_DUP)}")
    if len(NEAR_DUP):
        show = NEAR_DUP.head(8).copy()
        show["y_i"], show["y_j"] = Y[show["i"]], Y[show["j"]]
        show["text_i"] = train["raw_text"].values[show["i"]]
        display(show)
if n_dup / max(len(train), 1) > 0.05 and not CFG.GROUP_COL:
    print("💡 Duplikat > 5% → pertimbangkan CFG.GROUP_COL = 'dup_group'.")

# %% [markdown]
# ### 10.4 Drift train vs test (vocabulary & adversarial validation)
#
# Token coverage < 90% → banyak kata baru di test. Adversarial AUC > 0.7 → distribusi test berbeda → CV mungkin
# tidak mencerminkan leaderboard; andalkan model yang lebih general dan hindari post-processing yang agresif.

# %%
if CFG.RUN_EDA and HAS_TEST and len(test) > 0:
    cov_tok, cov_type, oov_top = vocab_overlap(train["text_cl"], test["text_cl"])
    print(f"Token coverage: {cov_tok:.1%} | type coverage: {cov_type:.1%}", "⚠️" if cov_tok < 0.9 else "✅")
    print("Top OOV words in test:", oov_top[:15])
    if CFG.RUN_ADVERSARIAL_VALIDATION:
        try:
            ADV_AUC, adv_te, adv_tr = adversarial_validation(train["text_cl"], test["text_cl"], seed=CFG.SEED)
            print(f"Adversarial validation AUC = {ADV_AUC:.3f}", "⚠️ DRIFT" if ADV_AUC > 0.7 else "✅ mirip")
            print("  kata khas TEST :", adv_te[:10])
            print("  kata khas TRAIN:", adv_tr[:10])
        except Exception as e:
            print("Adversarial validation dilewati:", short_err(e))
elif not HAS_TEST:
    print("Test tidak ada → analisis drift dilewati.")

# %% [markdown]
# ### 10.5 Meta-feature vs target (+ word cloud opsional)
#
# Korelasi Spearman antara fitur sederhana (jumlah kata, tanda seru, huruf kapital, angka, URL, ...) dan target.
# **Jika** |ρ| besar → fitur ini sinyal kuat; cocok dibahas di laporan. Fitur angka (`n_digits`) penting untuk
# data seperti iklan properti (luas, jumlah kamar).

# %%
if CFG.RUN_EDA:
    MF = meta_features(train["raw_text"])
    corr_mf = pd.DataFrame({"feature": MF.columns,
                            "spearman_with_target": [spearmanr(MF[c], Y).correlation if MF[c].nunique() > 1 else np.nan
                                                     for c in MF.columns]}).sort_values("spearman_with_target", key=np.abs,
                                                                                         ascending=False)
    display(corr_mf.round(3))
    save_table(corr_mf, "meta_feature_target_correlation")
    if CFG.MAKE_WORDCLOUD and HAS_WORDCLOUD:
        try:
            gs = GROUP_ORDER[:: max(1, len(GROUP_ORDER) - 1)][:2]
            fig, axes = plt.subplots(1, len(gs), figsize=(6 * len(gs), 4))
            for ax, gname in zip(np.atleast_1d(axes), gs):
                txt = " ".join(train.loc[_label_names == gname, "text_cl"].tolist())
                ax.imshow(WordCloud(width=600, height=400, background_color="white", stopwords=STOPWORDS_ALL,
                                    max_words=100).generate(txt or "empty"))
                ax.axis("off")
                ax.set_title(f"Word cloud: target {gname}")
            save_fig("wordclouds", fig)
        except Exception as e:
            print("Word cloud dilewati:", short_err(e))
    elif CFG.MAKE_WORDCLOUD:
        print("ℹ️ wordcloud tidak terpasang → dilewati.")

# %% [markdown]
# ## 11. Metric registry & post-processing dasar
#
# | metric | lebih besar lebih baik? | catatan |
# |---|---|---|
# | `rmse` | tidak | sensitif outlier; standar umum |
# | `mae` | tidak | robust outlier; prediksi optimal ≈ median |
# | `r2` | ya | 1 = sempurna, 0 = setara tebak rata-rata |
# | `pearson` / `spearman` | ya | korelasi; skala prediksi tidak penting (spearman hanya urutan) |
# | `qwk` | ya | Quadratic Weighted Kappa untuk target diskrit ordinal; prediksi dibulatkan ke `VALID_VALUES` |
# | `rmsle` | tidak | RMSE di skala log → cocok untuk harga; latih dengan `log1p` |
# | `mape` / `medae` | tidak | persentase error / median absolute error |
#
# Semua skor dihitung di **skala asli** (setelah inverse transform) dan setelah **clip** ke `[MIN_TARGET, MAX_TARGET]`.
# **Jika** metric lomba tidak ada → tambahkan ke dict `METRICS`.

# %%
EPS = 1e-9


def clip_pred(p):
    """Clip prediksi ke rentang valid (kalau CLIP_PREDICTIONS)."""
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=float(np.median(Y)))
    return np.clip(p, MIN_T, MAX_T) if CFG.CLIP_PREDICTIONS else p


def round_to_valid(p, thresholds=None):
    """Bulatkan prediksi ke VALID_VALUES (nearest) atau pakai threshold rounding optimal."""
    vals = VALID_VALUES if VALID_VALUES is not None else np.unique(np.round(Y))
    p = np.asarray(p, dtype=float)
    if thresholds is not None:
        return vals[np.searchsorted(np.asarray(thresholds), p, side="right")]
    idx = np.abs(p[:, None] - vals[None, :]).argmin(1)
    return vals[idx]


def _corr(fn, y, p):
    if np.std(p) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(fn(y, p)[0] if fn is pearsonr else fn(y, p).correlation)


METRICS = {
    # name: (greater_is_better, fn(y_true, y_pred))
    "rmse": (False, lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))),
    "mae": (False, lambda y, p: float(np.mean(np.abs(y - p)))),
    "medae": (False, lambda y, p: float(np.median(np.abs(y - p)))),
    "r2": (True, lambda y, p: float(skm.r2_score(y, p))),
    "pearson": (True, lambda y, p: _corr(pearsonr, y, p)),
    "spearman": (True, lambda y, p: _corr(spearmanr, y, p)),
    "qwk": (True, lambda y, p: float(skm.cohen_kappa_score(np.round(y).astype(int),
                                                            np.round(round_to_valid(p)).astype(int),
                                                            weights="quadratic"))),
    "rmsle": (False, lambda y, p: float(np.sqrt(np.mean((np.log1p(np.clip(y, 0, None)) -
                                                          np.log1p(np.clip(p, 0, None))) ** 2)))),
    "mape": (False, lambda y, p: float(np.mean(np.abs(y - p) / np.clip(np.abs(y), EPS, None)))),
}
assert CFG.METRIC in METRICS, f"METRIC '{CFG.METRIC}' tidak dikenal. Pilihan: {list(METRICS)}"
GREATER_IS_BETTER = METRICS[CFG.METRIC][0]
REPORT_METRICS = ["rmse", "mae", "r2", "pearson", "spearman"] + (["qwk"] if (IS_DISCRETE or CFG.METRIC == "qwk") else []) \
    + (["rmsle"] if Y.min() >= 0 else [])
if CFG.METRIC not in REPORT_METRICS:
    REPORT_METRICS = [CFG.METRIC] + REPORT_METRICS
if CFG.METRIC == "qwk" and VALID_VALUES is None:
    print("⚠️ METRIC qwk tapi target tidak terdeteksi diskrit → prediksi dibulatkan ke integer terdekat.")


def compute_score(y_true, pred, metric=None):
    """Skor satu metric (prediksi di-clip dulu)."""
    metric = metric or CFG.METRIC
    try:
        return METRICS[metric][1](np.asarray(y_true, dtype=float), clip_pred(pred))
    except Exception:
        return np.nan


def evaluate_all(y_true, pred):
    return {m: compute_score(y_true, pred, m) for m in REPORT_METRICS}


def is_better(a, b):
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return True
    return a > b if GREATER_IS_BETTER else a < b


print(f"Main metric: {CFG.METRIC} ({'higher' if GREATER_IS_BETTER else 'lower'} is better) | report: {REPORT_METRICS}")
print(f"Baseline (predict train mean): {CFG.METRIC} = {compute_score(Y, np.full(len(Y), Y.mean())):.5f}  "
      "← model harus jauh lebih baik dari ini")

# %% [markdown]
# ## 12. Cross-validation folds (binned stratified)
#
# Untuk regresi, `StratifiedKFold` dilakukan pada **bin kuantil target** (`TARGET_BINS`) supaya distribusi target
# di tiap fold mirip (penting kalau target skewed — tanpa ini satu fold bisa kebagian semua nilai ekstrem).
# `GROUP_COL` diisi → `StratifiedGroupKFold` (group tidak terpecah).
#
# **Cara membaca:** tabel `describe` target per fold harus mirip (mean, std, max).
# **Untuk laporan:** "Evaluasi memakai 5-fold CV terstratifikasi berdasarkan bin target; skor = skor OOF."

# %%
def make_folds(n_folds, seed):
    """Fold id per baris sesuai CV_STRATEGY."""
    folds = np.full(len(train), -1)
    strat = CFG.CV_STRATEGY
    groups = train[CFG.GROUP_COL].astype(str).values if (CFG.GROUP_COL and CFG.GROUP_COL in train.columns) else None
    if strat == "auto":
        strat = "stratified_group" if groups is not None else "binned"
    if strat in ("group", "stratified_group") and groups is None:
        print("⚠️ GROUP_COL tidak ada → fallback binned.")
        strat = "binned"
    try:
        bins = pd.qcut(Y, q=CFG.TARGET_BINS, labels=False, duplicates="drop")
    except Exception:
        bins = np.zeros(len(Y), int)
    if strat == "stratified_group":
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            it = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(train, bins, groups)
        except Exception:
            it = GroupKFold(n_splits=n_folds).split(train, bins, groups)
    elif strat == "group":
        it = GroupKFold(n_splits=n_folds).split(train, bins, groups)
    elif strat == "kfold":
        it = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(train)
    else:
        try:
            it = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(train, bins))
        except ValueError:
            print("Binning gagal → KFold biasa")
            it = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(train)
    for f, (_, va) in enumerate(it):
        folds[va] = f
    globals()["CV_STRATEGY_USED"] = strat
    print(f"CV strategy: {strat}, n_folds={n_folds}")
    return folds


train["fold"] = make_folds(CFG.N_FOLDS, CFG.SEED)
FOLDS = train["fold"].values
display(pd.DataFrame({"fold": FOLDS, "y": Y}).groupby("fold")["y"].describe().round(3))

# %% [markdown]
# ## 13. Result registry
#
# `register_result(name, oof, test_pred)` menyimpan prediksi OOF/test (skala asli), menulis `.npy`, menghitung skor per
# fold, dan menambah baris ke `experiment_log.csv`.
#
# **Jika** fold std besar (mis. RMSE std > 5% dari RMSE) → model tidak stabil (data kecil / outlier di fold tertentu).

# %%
RESULTS = {}


def register_result(name, oof, test_pred, family, seconds=None, mask=None):
    """Simpan hasil satu model + print ringkasan."""
    mask = np.ones(len(Y), bool) if mask is None else mask
    oof = np.asarray(oof, dtype=float)
    score = compute_score(Y[mask], oof[mask])
    fold_scores = [compute_score(Y[(FOLDS == f) & mask], oof[(FOLDS == f) & mask]) for f in np.unique(FOLDS[mask])]
    RESULTS[name] = {"oof": oof, "test": None if test_pred is None else np.asarray(test_pred, dtype=float),
                     "score": score, "family": family, "fold_mean": float(np.nanmean(fold_scores)),
                     "fold_std": float(np.nanstd(fold_scores)), "seconds": seconds, "partial": bool(~mask.all())}
    np.save(os.path.join(CFG.OUTPUT_DIR, "preds", f"oof_{name}.npy"), oof)
    if test_pred is not None:
        np.save(os.path.join(CFG.OUTPUT_DIR, "preds", f"test_{name}.npy"), np.asarray(test_pred, dtype=float))
    log_experiment(name, score, {"family": family, "fold_std": RESULTS[name]["fold_std"]})
    rel = RESULTS[name]["fold_std"] / max(abs(np.nanmean(fold_scores)), EPS)
    flag = "⚠️ fold variance tinggi" if rel > 0.05 else "✅"
    part = " (PARTIAL folds)" if RESULTS[name]["partial"] else ""
    print(f"  ==> {name}: OOF {CFG.METRIC} = {score:.5f} | folds {np.nanmean(fold_scores):.4f} ± "
          f"{np.nanstd(fold_scores):.4f} {flag}{part}")
    return score


def fold_standardizer(y_t):
    """Mean/std target (skala transform) untuk standardisasi di dalam fold."""
    mu, sd = float(np.mean(y_t)), float(np.std(y_t))
    return mu, (sd if sd > 1e-12 else 1.0)


print("Registry siap.")

# %% [markdown]
# ## 14. Feature builder TF-IDF (word + char) & helper LightGBM
#
# **TF-IDF word (1–2 gram)** menangkap kata & frasa ("tidak bagus"). **TF-IDF char_wb (2–5 gram)** menangkap
# potongan kata → tahan typo, slang, imbuhan Indonesia (me-, di-, -kan, -nya). Keduanya digabung (hstack).
#
# **Anti-leakage:** vectorizer **di-fit hanya pada fold train**, lalu dipakai untuk transform fold valid & test.
#
# **Jika** RAM habis (MemoryError) → kecilkan `TFIDF_CHAR_MAX_FEATURES` (mis. 100k) atau `USE_CHAR_TFIDF=False`.
# **Jika** error `empty vocabulary` → teks terlalu pendek/sedikit; helper otomatis mencoba ulang dengan `min_df=1`.

# %%
def build_tfidf(train_texts, other_texts_list):
    """Fit TF-IDF word(+char) di train_texts, transform semua teks lain. Return (X_train, [X_other...], vecs)."""
    def _make(min_df):
        vecs = [TfidfVectorizer(ngram_range=CFG.TFIDF_WORD_NGRAM, min_df=min_df, max_df=CFG.TFIDF_MAX_DF,
                                max_features=CFG.TFIDF_WORD_MAX_FEATURES, sublinear_tf=True,
                                strip_accents="unicode", token_pattern=r"(?u)\b\w+\b", dtype=np.float32)]
        if CFG.USE_CHAR_TFIDF:
            vecs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=CFG.TFIDF_CHAR_NGRAM, min_df=min_df,
                                        max_df=1.0, max_features=CFG.TFIDF_CHAR_MAX_FEATURES, sublinear_tf=True,
                                        dtype=np.float32))
        return vecs

    for min_df in (CFG.TFIDF_MIN_DF, 1):
        try:
            vecs = _make(min_df)
            Xtr = sparse.hstack([v.fit_transform(train_texts) for v in vecs]).tocsr()
            break
        except ValueError as e:
            if min_df == 1:
                raise
            print(f"   TF-IDF retry min_df=1 ({short_err(e, 80)})")
    others = [sparse.hstack([v.transform(t) for v in vecs]).tocsr() if len(t) else None for t in other_texts_list]
    return Xtr, others, vecs


def fit_lgbm_inner_es(X, y, kind="classification", seed=42):
    """LightGBM dengan early stopping pada inner holdout 10% dari data train fold (tidak menyentuh fold valid)."""
    rs = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rs.permutation(n)
    n_hold = max(1, int(n * 0.1))
    hold, fit_i = idx[:n_hold], idx[n_hold:]
    params = dict(n_estimators=CFG.LGBM_N_ESTIMATORS, learning_rate=0.05, num_leaves=31, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, random_state=seed,
                  verbose=-1, n_jobs=-1)
    model = lgb.LGBMClassifier(**params) if kind == "classification" else lgb.LGBMRegressor(**params)
    try:
        model.fit(X[fit_i], y[fit_i], eval_set=[(X[hold], y[hold])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    except TypeError:  # lightgbm versi lama
        model.fit(X[fit_i], y[fit_i], eval_set=[(X[hold], y[hold])], early_stopping_rounds=100, verbose=False)
    except Exception:
        model = lgb.LGBMClassifier(**params) if kind == "classification" else lgb.LGBMRegressor(**params)
        model.set_params(n_estimators=min(300, CFG.LGBM_N_ESTIMATORS))
        model.fit(X, y)
    return model


def make_calibrated(estimator, cv=3):
    """CalibratedClassifierCV dengan shim nama argumen (estimator vs base_estimator)."""
    try:
        return CalibratedClassifierCV(estimator=estimator, cv=cv, method="sigmoid")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, cv=cv, method="sigmoid")


def softmax_np(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -50, 50)))


print("Feature helpers siap.")

# %% [markdown]
# ## 15. [A] Classical baselines: TF-IDF + model linear
#
# **Wajib dijalankan**: cepat di CPU, sering sangat kuat, dan menjadi patokan untuk transformer.
#
# | model | ide | kapan bagus |
# |---|---|---|
# | `ridge` | regresi linear + L2 | baseline paling stabil untuk teks; tuning `RIDGE_ALPHA` |
# | `sgd` | SGDRegressor loss Huber | tahan outlier, cepat untuk data besar |
# | `linearsvr` | Linear SVR (squared epsilon-insensitive) | variasi linear lain (bagus untuk ensemble) |
# | `lgbm_svd` | TruncatedSVD → LightGBM | non-linear; menangkap interaksi |
#
# Semua model dilatih pada target **transform + distandarisasi di dalam fold** (mean/std dihitung dari fold train),
# lalu prediksi dikembalikan ke skala asli.
#
# **Cara membaca:** OOF score vs baseline "predict mean" (dicetak di section 11).
# **Jika** Ridge hampir sama dengan baseline → teks hampir tidak mengandung sinyal ATAU kolom teks/target salah.
# **Untuk laporan:** "TF-IDF + Ridge sebagai baseline mencapai RMSE X".

# %%
def fit_predict_reg(name, Xtr, ytr_orig, X_list, cache=None):
    """Fit model regresi pada fitur sparse/dense. Return list prediksi skala asli."""
    cache = {} if cache is None else cache
    yt = transform_target(ytr_orig)
    mu, sd = fold_standardizer(yt)
    z = (yt - mu) / sd
    if name == "ridge":
        model = Ridge(alpha=CFG.RIDGE_ALPHA, random_state=CFG.SEED)
    elif name == "sgd":
        model = SGDRegressor(loss="huber", epsilon=1.0, penalty="l2", alpha=1e-5, max_iter=50, tol=1e-4,
                             random_state=CFG.SEED)
    elif name == "linearsvr":
        model = LinearSVR(C=0.1, epsilon=0.0, loss="squared_epsilon_insensitive", max_iter=5000,
                          random_state=CFG.SEED)
    elif name == "lgbm_svd":
        if "svd" not in cache:
            n_comp = int(min(CFG.SVD_COMPONENTS, Xtr.shape[1] - 1, Xtr.shape[0] - 1))
            svd = TruncatedSVD(n_components=max(2, n_comp), random_state=CFG.SEED)
            cache["svd"] = (svd.fit_transform(Xtr), [svd.transform(X) if X is not None else None for X in X_list])
        Ztr, Zs = cache["svd"]
        model = fit_lgbm_inner_es(Ztr, z, "regression", CFG.SEED)
        return [None if Zx is None else inverse_transform_target(model.predict(Zx) * sd + mu) for Zx in Zs]
    else:
        raise ValueError(f"Model classical tidak dikenal: {name}")
    model.fit(Xtr, z)
    return [None if X is None else inverse_transform_target(model.predict(X) * sd + mu) for X in X_list]


CLASSICAL_MODELS = [m for m in CFG.CLASSICAL_MODELS if not (m == "lgbm_svd" and not HAS_LGB)]
if "lgbm_svd" in CFG.CLASSICAL_MODELS and not HAS_LGB:
    print("ℹ️ lightgbm tidak ada → lgbm_svd dilewati.")

if CFG.RUN_CLASSICAL:
    n_te = len(test) if HAS_TEST else 0
    oofs = {m: np.zeros(len(train)) for m in CLASSICAL_MODELS}
    tests = {m: np.zeros(n_te) for m in CLASSICAL_MODELS}
    times = defaultdict(float)
    te_texts = test["text_cl"].values if HAS_TEST else []
    for fold in range(CFG.N_FOLDS):
        tr_idx, va_idx = np.where(FOLDS != fold)[0], np.where(FOLDS == fold)[0]
        t0 = time.time()
        Xtr, (Xva, Xte), _ = build_tfidf(train["text_cl"].values[tr_idx], [train["text_cl"].values[va_idx], te_texts])
        print(f"Fold {fold}: TF-IDF {Xtr.shape} built in {time.time() - t0:.1f}s")
        cache = {}
        for m in CLASSICAL_MODELS:
            key = f"tfidf_{m}"
            if key in FAILED_MODELS:
                continue
            t0 = time.time()
            try:
                pv, pt = fit_predict_reg(m, Xtr, Y[tr_idx], [Xva, Xte], cache)
                oofs[m][va_idx] = pv
                if HAS_TEST:
                    tests[m] += pt / CFG.N_FOLDS
                times[m] += time.time() - t0
            except Exception as e:
                FAILED_MODELS[key] = short_err(e)
                print(f"   ❌ {key} gagal: {FAILED_MODELS[key]}")
        del Xtr, Xva, Xte
        gc.collect()
    for m in CLASSICAL_MODELS:
        if f"tfidf_{m}" not in FAILED_MODELS:
            register_result(f"tfidf_{m}", oofs[m], tests[m] if HAS_TEST else None, "classical", times[m])
else:
    print("RUN_CLASSICAL=False → dilewati.")

# %% [markdown]
# ## 16. [B] Sentence embeddings (frozen) — helper
#
# **Ide:** model sentence-embedding (sudah dilatih untuk memahami makna kalimat) mengubah tiap teks menjadi vektor
# (mis. 384/768 dimensi). Model **tidak di-fine-tune**, cukup encode sekali → lalu model ringan
# (LogReg / Ridge / LightGBM) dilatih di atas vektor itu. **Cepat, jalan di CPU**, dan karena sifatnya berbeda dari
# TF-IDF (semantik vs kata persis), biasanya **menambah skor ensemble**.
#
# **Model yang direkomendasikan**
#
# | model | ukuran | catatan |
# |---|---|---|
# | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | cepat, multibahasa (termasuk Indonesia) |
# | `intfloat/multilingual-e5-small` | 118M | kuat; **wajib prefix `"query: "`** (otomatis ditambahkan) |
# | `intfloat/multilingual-e5-base` | 278M | lebih kuat, lebih lambat |
# | `sentence-transformers/all-MiniLM-L6-v2` | 23M | English only, sangat cepat |
#
# **Catatan prefix E5:** model keluarga E5 dilatih dengan awalan `"query: "` / `"passage: "`. Untuk klasifikasi/
# regresi pakai `"query: "` di semua teks. Tanpa prefix, kualitas embedding turun.
#
# **Cache:** embedding disimpan ke `cache/emb_<model>_<hash>.npy`. Run ulang dengan teks & model sama = instan.
# Kalau `sentence-transformers` tidak terpasang, notebook memakai `transformers` + mean pooling (hasil setara).
#
# **Jika** download model gagal (offline) → isi `EMBEDDING_MODELS` dengan path folder lokal model.

# %%
def embedding_prefix(model_name):
    """Prefix teks untuk model embedding tertentu (E5 → 'query: ')."""
    if model_name in CFG.EMBEDDING_PREFIXES:
        return CFG.EMBEDDING_PREFIXES[model_name]
    if "e5" in model_name.lower().split("/")[-1]:
        return "query: "
    return ""


def _hash_texts(texts, extra=""):
    h = hashlib.md5(extra.encode("utf-8"))
    for t in texts:
        h.update(str(t).encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def autocast_ctx(enabled=True, dtype=None):
    """torch.autocast hanya kalau CUDA tersedia & diminta; selain itu context kosong."""
    import contextlib
    if not (enabled and HAS_CUDA):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype or torch.float16)


def _mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def _encode_with_transformers(texts, model_name, batch_size, max_len):
    """Fallback tanpa sentence-transformers: AutoModel + mean pooling + L2 normalize."""
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    out = []
    order = np.argsort([len(t) for t in texts])
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with autocast_ctx(True):
                h = mdl(**enc)[0]
            out.append((idx, _mean_pool(h.float(), enc["attention_mask"]).cpu().numpy()))
    emb = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
    for idx, e in out:
        emb[idx] = e
    del mdl
    free_memory()
    return normalize(emb)


def encode_sentences(texts, model_name, batch_size=64, max_len=256):
    """Encode list teks → np.ndarray (n, dim), dengan cache disk."""
    texts = [embedding_prefix(model_name) + str(t) for t in texts]
    key = _hash_texts(texts, f"{model_name}|{max_len}")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model_name)[-60:]
    path = os.path.join(CFG.OUTPUT_DIR, "cache", f"emb_{safe}_{key}.npy")
    if CFG.CACHE_EMBEDDINGS and os.path.exists(path):
        print(f"   loaded cached embeddings: {path}")
        return np.load(path)
    t0 = time.time()
    emb = None
    if HAS_ST:
        try:
            st_model = SentenceTransformer(model_name, device=str(DEVICE))
            st_model.max_seq_length = max_len
            emb = st_model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True,
                                  normalize_embeddings=True)
            del st_model
            free_memory()
        except Exception as e:
            print(f"   sentence-transformers gagal ({short_err(e, 120)}) → fallback transformers mean pooling")
    if emb is None:
        emb = _encode_with_transformers(texts, model_name, batch_size, max_len)
    emb = np.asarray(emb, dtype=np.float32)
    if CFG.CACHE_EMBEDDINGS:
        np.save(path, emb)
    print(f"   encoded {len(texts)} texts with {model_name} → {emb.shape} in {time.time() - t0:.1f}s")
    return emb


print("Embedding helpers siap.")

# %% [markdown]
# ### 16.1 Sentence embedding + Ridge / SVR / LightGBM (K-Fold)
#
# **Head** (`CFG.EMBEDDING_HEADS`): `ridge` (cepat, stabil), `svr` (RBF SVR — sering terbaik di embedding untuk
# data < 20k baris; lambat untuk data besar), `lgbm`, `xgb`, `catboost` (GBDT; kalau terpasang).
#
# **Jika** SVR jauh lebih baik dari Ridge → hubungan embedding–target non-linear. **Jika** data > 30k baris →
# hindari `svr` (kompleksitas O(n²)).
# **Untuk laporan:** bandingkan embedding vs TF-IDF (semantik vs kata persis) dan kontribusinya di ensemble.

# %%
def fit_predict_dense_reg(head, Ztr, ytr_orig, Z_list):
    """Fit head regresi pada fitur dense. Target transform + standardisasi di dalam fold."""
    yt = transform_target(ytr_orig)
    mu, sd = fold_standardizer(yt)
    z = (yt - mu) / sd
    if head == "ridge":
        sc = StandardScaler().fit(Ztr)
        m = Ridge(alpha=10.0).fit(sc.transform(Ztr), z)
        preds = [None if Zx is None else m.predict(sc.transform(Zx)) for Zx in Z_list]
    elif head == "svr":
        from sklearn.svm import SVR
        sc = StandardScaler().fit(Ztr)
        m = SVR(C=3.0, epsilon=0.05, kernel="rbf").fit(sc.transform(Ztr), z)
        preds = [None if Zx is None else m.predict(sc.transform(Zx)) for Zx in Z_list]
    elif head == "lgbm":
        if not HAS_LGB:
            raise ImportError("lightgbm tidak terpasang")
        m = fit_lgbm_inner_es(Ztr, z, "regression", CFG.SEED)
        preds = [None if Zx is None else m.predict(Zx) for Zx in Z_list]
    elif head == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost tidak terpasang")
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.5,
                             tree_method="hist", random_state=CFG.SEED, n_jobs=-1).fit(Ztr, z)
        preds = [None if Zx is None else m.predict(Zx) for Zx in Z_list]
    elif head == "catboost":
        if not HAS_CB:
            raise ImportError("catboost tidak terpasang")
        m = cb.CatBoostRegressor(iterations=600, learning_rate=0.08, depth=6, random_seed=CFG.SEED,
                                 verbose=False).fit(Ztr, z)
        preds = [None if Zx is None else m.predict(Zx) for Zx in Z_list]
    else:
        raise ValueError(f"Embedding head tidak dikenal: {head}")
    return [None if p is None else inverse_transform_target(np.asarray(p) * sd + mu) for p in preds]


EMB_CACHE = {}
if CFG.RUN_EMBEDDINGS and HAS_TORCH and HAS_TRANSFORMERS:
    all_texts = train["text_tf"].tolist() + (test["text_tf"].tolist() if HAS_TEST else [])
    for emb_name in CFG.EMBEDDING_MODELS:
        short = re.sub(r"[^A-Za-z0-9]+", "_", emb_name.split("/")[-1])[:40]
        try:
            with Timer(f"encode {emb_name}"):
                E = encode_sentences(all_texts, emb_name, CFG.EMB_BATCH_SIZE, CFG.EMB_MAX_LEN)
        except Exception as e:
            FAILED_MODELS[f"emb_{short}"] = short_err(e)
            print(f"❌ embedding {emb_name} gagal: {FAILED_MODELS[f'emb_{short}']}")
            continue
        E_tr, E_te = E[:len(train)], (E[len(train):] if HAS_TEST else None)
        EMB_CACHE[emb_name] = (E_tr, E_te)
        for head in CFG.EMBEDDING_HEADS:
            key = f"emb_{short}_{head}"
            oof, tp = np.zeros(len(train)), (np.zeros(len(test)) if HAS_TEST else None)
            t0 = time.time()
            try:
                for fold in range(CFG.N_FOLDS):
                    tr_idx, va_idx = np.where(FOLDS != fold)[0], np.where(FOLDS == fold)[0]
                    pv, pt = fit_predict_dense_reg(head, E_tr[tr_idx], Y[tr_idx], [E_tr[va_idx], E_te])
                    oof[va_idx] = pv
                    if HAS_TEST:
                        tp += pt / CFG.N_FOLDS
                register_result(key, oof, tp, "embedding", time.time() - t0)
            except Exception as e:
                FAILED_MODELS[key] = short_err(e)
                print(f"   ❌ {key} gagal: {FAILED_MODELS[key]}")
elif CFG.RUN_EMBEDDINGS:
    print("⚠️ torch/transformers tidak tersedia → sentence embedding dilewati.")
else:
    print("RUN_EMBEDDINGS=False → dilewati.")

# %% [markdown]
# ## 17. [C] Transformer fine-tuning — engine
#
# Engine ini dipakai oleh dua backend:
#
# | backend | kelebihan | kekurangan |
# |---|---|---|
# | `custom_loop` (default) | PyTorch murni → **paling tahan perbedaan versi transformers**; AMP, grad accumulation, warmup, LLRD, early stopping, best-checkpoint di memori, gradient clipping | — |
# | `hf_trainer` | API HuggingFace `Trainer` | butuh `accelerate`; argumen sering berubah antar versi (sudah di-shim: `eval_strategy`/`evaluation_strategy`, `processing_class`/`tokenizer`, `compute_loss(num_items_in_batch)`) |
#
# **Fitur engine:**
# - **Long text**: `truncate` (potong akhir), `head_tail` (ambil awal + akhir — bagian penting sering di awal
#   dan kesimpulan di akhir), `chunk` (teks dipotong jadi beberapa window ber-overlap; saat training tiap chunk
#   mewarisi label dokumen, saat prediksi logits chunk digabung dengan mean/max per dokumen).
# - **Head**: pooling `cls` atau `mean`, multi-sample dropout opsional.
# - **Loss**: cross-entropy (+ class weights, label smoothing), focal loss, BCE untuk multilabel, MSE/Huber untuk regresi.
# - **Memory**: AMP fp16/bf16 (hanya CUDA), gradient checkpointing, **OOM auto-retry** (batch size dibagi 2,
#   grad accumulation dikali 2 → batch efektif sama).
# - **Best checkpoint** dipilih berdasarkan metric lomba pada fold valid (bukan loss), disimpan di RAM
#   (menghindari bug reload checkpoint dari disk).
#
# **Error umum & solusi** (detail di tabel Troubleshooting):
# - `CUDA out of memory` → otomatis retry; kalau tetap gagal: kecilkan `MAX_LEN`, `GRADIENT_CHECKPOINTING=True`.
# - `loss NaN` → `USE_AMP=False`, turunkan `LR` (1e-5).
# - `sentencepiece`/`tiktoken` error saat load tokenizer → `pip install sentencepiece protobuf`, restart kernel.
# - Warning "Some weights ... newly initialized" → **normal** (head klasifikasi memang baru).

# %%
if HAS_TORCH and HAS_TRANSFORMERS:
    from torch.utils.data import Dataset, DataLoader

    def is_oom_error(e):
        """Deteksi CUDA OOM lintas versi torch."""
        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        return (oom_cls is not None and isinstance(e, oom_cls)) or "out of memory" in str(e).lower()

    def make_grad_scaler(enabled):
        """GradScaler shim (torch.amp vs torch.cuda.amp)."""
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def amp_settings():
        """(use_amp, dtype, use_scaler) sesuai CFG & hardware."""
        use_amp = bool(CFG.USE_AMP and HAS_CUDA)
        if use_amp and CFG.AMP_DTYPE == "bf16" and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        return use_amp, torch.float16, use_amp

    # ---------------------------------------------------------------- tokenisasi & long-text strategy
    def special_token_layout(tokenizer):
        """Cari token spesial di depan/belakang untuk satu kalimat (CLS..SEP, <s>..</s>, dll.)."""
        with_sp = tokenizer("a", add_special_tokens=True)["input_ids"]
        without = tokenizer("a", add_special_tokens=False)["input_ids"]
        n = len(without)
        for s in range(len(with_sp) - n + 1):
            if with_sp[s:s + n] == without:
                return list(with_sp[:s]), list(with_sp[s + n:])
        return list(with_sp[:1]), list(with_sp[-1:])

    def tokenize_docs(tokenizer, texts, batch=2000):
        """Token id per dokumen TANPA token spesial & tanpa truncation (dipotong nanti sesuai strategi)."""
        out = []
        texts = [str(t) for t in texts]
        for i in range(0, len(texts), batch):
            out.extend(tokenizer(texts[i:i + batch], add_special_tokens=False, truncation=False)["input_ids"])
        return out

    def build_items(doc_ids, doc_indices, tokenizer, max_len, strategy):
        """Ubah dokumen → item model sesuai strategi. Return (list input_ids, array doc_position)."""
        prefix, suffix = special_token_layout(tokenizer)
        budget = max(8, int(max_len) - len(prefix) - len(suffix))
        items, owner = [], []
        for pos, d in enumerate(doc_indices):
            ids = doc_ids[d]
            if strategy == "chunk" and len(ids) > budget:
                step = max(1, budget - CFG.CHUNK_OVERLAP)
                starts, s = [], 0
                while True:
                    starts.append(s)
                    if s + budget >= len(ids):
                        break
                    s += step
                if len(starts) > CFG.MAX_CHUNKS_PER_DOC:
                    pick = np.unique(np.linspace(0, len(starts) - 1, CFG.MAX_CHUNKS_PER_DOC).round().astype(int))
                    starts = [starts[k] for k in pick]
                chunks = [ids[s:s + budget] for s in starts]
            elif strategy == "head_tail" and len(ids) > budget:
                h = int(budget * CFG.HEAD_RATIO)
                t = budget - h
                chunks = [ids[:h] + (ids[-t:] if t > 0 else [])]
            else:
                chunks = [ids[:budget]]
            for c in chunks:
                items.append(prefix + list(c) + suffix)
                owner.append(pos)
        return items, np.asarray(owner)

    class TokenDataset(Dataset):
        """Dataset list token id (+ label opsional)."""

        def __init__(self, ids_list, labels=None):
            self.ids = ids_list
            self.labels = labels

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            item = {"input_ids": self.ids[i]}
            if self.labels is not None:
                item["labels"] = self.labels[i]
            return item

    def make_collate(pad_id, use_tti, label_dtype):
        """Dynamic padding collate (dipakai custom loop & HF Trainer)."""
        def collate(batch):
            L = max(len(b["input_ids"]) for b in batch)
            ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for i, b in enumerate(batch):
                n = len(b["input_ids"])
                ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
                mask[i, :n] = 1
            out = {"input_ids": ids, "attention_mask": mask}
            if use_tti:
                out["token_type_ids"] = torch.zeros_like(ids)
            if "labels" in batch[0]:
                out["labels"] = torch.tensor(np.array([b["labels"] for b in batch]), dtype=label_dtype)
            return out
        return collate

    # ---------------------------------------------------------------- model
    class TextModel(nn.Module):
        """Backbone AutoModel + pooling (cls/mean) + (multi-sample) dropout + linear head."""

        def __init__(self, model_name, num_outputs):
            super().__init__()
            config = AutoConfig.from_pretrained(model_name)
            bd = getattr(CFG, "BACKBONE_DROPOUT", None)
            if bd is not None:
                for attr in ["hidden_dropout_prob", "attention_probs_dropout_prob", "dropout", "attention_dropout"]:
                    if hasattr(config, attr):
                        setattr(config, attr, bd)
            self.backbone = AutoModel.from_pretrained(model_name, config=config)
            if CFG.GRADIENT_CHECKPOINTING:
                try:
                    self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    self.backbone.gradient_checkpointing_enable()
                except Exception as e:
                    print("   gradient checkpointing tidak didukung:", short_err(e, 80))
                if hasattr(self.backbone.config, "use_cache"):
                    self.backbone.config.use_cache = False
            hidden = getattr(config, "hidden_size", None) or getattr(config, "dim", None) or getattr(config, "d_model")
            n_drop = max(1, int(CFG.MULTI_SAMPLE_DROPOUT))
            self.dropouts = nn.ModuleList([nn.Dropout(CFG.HEAD_DROPOUT) for _ in range(n_drop)])
            self.head = nn.Linear(hidden, num_outputs)
            fwd = inspect.signature(self.backbone.forward).parameters
            self.accepts_tti = "token_type_ids" in fwd

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
            kw = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None and self.accepts_tti:
                kw["token_type_ids"] = token_type_ids
            h = self.backbone(**kw)[0]
            if CFG.POOLING == "mean":
                m = attention_mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            else:
                pooled = h[:, 0]
            if self.training and len(self.dropouts) > 1:
                logits = torch.stack([self.head(d(pooled)) for d in self.dropouts]).mean(0)
            else:
                logits = self.head(self.dropouts[0](pooled))
            return {"logits": logits}

    # ---------------------------------------------------------------- loss
    def build_loss_fn(spec):
        """Loss sesuai task: CE/focal (single-label), BCE (multilabel), MSE/Huber (regresi)."""
        task = spec["task"]
        cw = spec.get("class_weights")

        def loss_fn(logits, labels):
            logits = logits.float()
            if task in ("binary", "multiclass"):
                w = cw.to(logits.device) if cw is not None else None
                if CFG.LOSS == "focal":
                    logp = F.log_softmax(logits, dim=-1)
                    logp_t = logp.gather(1, labels.long().view(-1, 1)).squeeze(1)
                    p_t = logp_t.exp()
                    loss = -((1 - p_t) ** CFG.FOCAL_GAMMA) * logp_t
                    if w is not None:
                        loss = loss * w[labels.long()]
                    return loss.mean()
                return F.cross_entropy(logits, labels.long(), weight=w, label_smoothing=float(CFG.LABEL_SMOOTHING))
            if task == "multilabel":
                y = labels.float()
                if CFG.LABEL_SMOOTHING:
                    y = y * (1 - CFG.LABEL_SMOOTHING) + 0.5 * CFG.LABEL_SMOOTHING
                pw = cw.to(logits.device) if cw is not None else None
                return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            pred = logits.view(-1)
            y = labels.float().view(-1)
            if getattr(CFG, "REG_LOSS", "mse") == "huber":
                return F.smooth_l1_loss(pred, y, beta=1.0)
            if getattr(CFG, "REG_LOSS", "mse") == "mae":
                return F.l1_loss(pred, y)
            return F.mse_loss(pred, y)
        return loss_fn

    # ---------------------------------------------------------------- optimizer & scheduler
    def build_optimizer(model):
        """AdamW dengan layer-wise LR decay (LLRD) dan tanpa weight decay untuk bias/LayerNorm."""
        n_layers = getattr(model.backbone.config, "num_hidden_layers", 12) or 12
        groups = defaultdict(list)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            no_wd = p.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()
            wd = 0.0 if no_wd else CFG.WEIGHT_DECAY
            if name.startswith("backbone."):
                mt = re.search(r"\.(?:layer|layers|block|blocks)\.(\d+)\.", name)
                if mt:
                    depth = int(mt.group(1)) + 1
                elif "embed" in name:
                    depth = 0
                else:
                    depth = n_layers
                lr = CFG.LR * (CFG.LLRD ** (n_layers - depth))
            else:
                lr = CFG.HEAD_LR or CFG.LR
            groups[(round(lr, 12), wd)].append(p)
        param_groups = [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]
        return torch.optim.AdamW(param_groups, lr=CFG.LR, eps=1e-6)

    def build_scheduler(optimizer, total_steps):
        """Linear warmup lalu linear/cosine decay (LambdaLR, tanpa dependensi transformers)."""
        warm = int(total_steps * CFG.WARMUP_RATIO)

        def lr_lambda(step):
            if step < warm:
                return float(step + 1) / float(max(1, warm))
            prog = float(step - warm) / float(max(1, total_steps - warm))
            if CFG.SCHEDULER == "cosine":
                return max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))
            return max(0.0, 1.0 - prog)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_logits(model, ids_list, collate, n_out, batch_size=None):
        """Prediksi logits untuk list item (diurutkan berdasarkan panjang → lebih cepat)."""
        batch_size = batch_size or CFG.EVAL_BATCH_SIZE
        model.eval()
        use_amp, dtype, _ = amp_settings()
        out = np.zeros((len(ids_list), n_out), dtype=np.float32)
        order = np.argsort([len(x) for x in ids_list])
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch = collate([{"input_ids": ids_list[i]} for i in idx])
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with autocast_ctx(use_amp, dtype):
                logits = model(**batch)["logits"]
            out[idx] = logits.float().cpu().numpy()
        return out

    def aggregate_logits(item_logits, owner, n_docs):
        """Gabungkan logits beberapa chunk menjadi satu per dokumen (mean / max)."""
        if len(owner) == n_docs and np.all(owner == np.arange(n_docs)):
            return item_logits
        if CFG.CHUNK_POOLING == "max":
            return pd.DataFrame(item_logits).groupby(owner).max().values
        sums = np.zeros((n_docs, item_logits.shape[1]), dtype=np.float64)
        np.add.at(sums, owner, item_logits)
        cnt = np.bincount(owner, minlength=n_docs).reshape(-1, 1)
        return (sums / np.clip(cnt, 1, None)).astype(np.float32)

    # ---------------------------------------------------------------- backend: custom loop
    def train_custom_loop(model, train_ds, collate, eval_fn, loss_fn, batch_size, accum, tag):
        """Training loop PyTorch: AMP, grad accumulation, warmup, clipping, early stopping, best state."""
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                            num_workers=CFG.NUM_WORKERS, drop_last=len(train_ds) > batch_size)
        steps_per_epoch = math.ceil(len(loader) / accum)
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer, steps_per_epoch * CFG.EPOCHS)
        use_amp, dtype, use_scaler = amp_settings()
        scaler = make_grad_scaler(use_scaler)
        n_eval = max(1, int(CFG.EVALS_PER_EPOCH))
        eval_at = {max(0, int(round(len(loader) * k / n_eval)) - 1) for k in range(1, n_eval + 1)}
        best, best_state, bad, history, nan_steps = None, None, 0, [], 0
        stop = False
        for epoch in range(CFG.EPOCHS):
            model.train()
            run_loss, n_seen = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                labels = batch.pop("labels")
                with autocast_ctx(use_amp, dtype):
                    logits = model(**batch)["logits"]
                loss = loss_fn(logits, labels)
                if not torch.isfinite(loss):
                    nan_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    if nan_steps > 10:
                        raise FloatingPointError("Loss NaN/Inf berulang → set USE_AMP=False dan/atau turunkan LR")
                    continue
                scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                run_loss += float(loss.item())
                n_seen += 1
                if step in eval_at:
                    score = eval_fn(model)
                    ep = epoch + (step + 1) / len(loader)
                    history.append({"epoch": round(ep, 3), "train_loss": run_loss / max(n_seen, 1),
                                    "valid_score": score})
                    improved = is_better(score, best)
                    print(f"   [{tag}] epoch {ep:.2f} | train_loss {run_loss / max(n_seen, 1):.4f} | "
                          f"valid {CFG.METRIC} {score:.5f} {'*' if improved else ''}")
                    if improved:
                        best, bad = score, 0
                        best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= CFG.EARLY_STOPPING_PATIENCE:
                            print(f"   [{tag}] early stopping")
                            stop = True
                            break
                    model.train()
            if stop:
                break
        return best_state, best, history

    # ---------------------------------------------------------------- backend: HF Trainer
    def filter_kwargs(callable_obj, kwargs, verbose=True):
        """Buang argumen yang tidak dikenal oleh versi library terpasang (berdasarkan signature)."""
        params = inspect.signature(callable_obj).parameters
        kept = {k: v for k, v in kwargs.items() if k in params}
        dropped = sorted(set(kwargs) - set(kept))
        if dropped and verbose:
            print(f"   (argumen tidak didukung versi ini, di-skip: {dropped})")
        return kept

    def train_hf_trainer(model, train_ds, valid_ds, collate, eval_from_logits, loss_fn, batch_size, accum, tag,
                         tokenizer):
        """Training dengan transformers.Trainer + shim kompatibilitas versi."""
        from transformers import Trainer, TrainingArguments, TrainerCallback
        steps_per_epoch = math.ceil(math.ceil(len(train_ds) / batch_size) / accum)
        total = steps_per_epoch * CFG.EPOCHS
        use_amp, dtype, _ = amp_settings()
        n_eval = max(1, int(CFG.EVALS_PER_EPOCH))
        ta = dict(output_dir=os.path.join(CFG.OUTPUT_DIR, "models", "hf_tmp"),
                  per_device_train_batch_size=batch_size, per_device_eval_batch_size=CFG.EVAL_BATCH_SIZE,
                  gradient_accumulation_steps=accum, num_train_epochs=CFG.EPOCHS, learning_rate=CFG.LR,
                  weight_decay=CFG.WEIGHT_DECAY, warmup_steps=int(total * CFG.WARMUP_RATIO),
                  max_grad_norm=CFG.MAX_GRAD_NORM, logging_strategy="no", save_strategy="no", report_to="none",
                  seed=CFG.SEED, fp16=bool(use_amp and dtype == torch.float16),
                  bf16=bool(use_amp and dtype == torch.bfloat16), dataloader_num_workers=CFG.NUM_WORKERS,
                  disable_tqdm=True, remove_unused_columns=False, lr_scheduler_type=CFG.SCHEDULER)
        strategy = "epoch" if n_eval == 1 else "steps"
        sig = inspect.signature(TrainingArguments.__init__).parameters
        if "eval_strategy" in sig:
            ta["eval_strategy"] = strategy
        else:
            ta["evaluation_strategy"] = strategy
        if strategy == "steps":
            ta["eval_steps"] = max(1, steps_per_epoch // n_eval)
        args = TrainingArguments(**filter_kwargs(TrainingArguments.__init__, ta))

        state = {"best": None, "best_state": None, "bad": 0, "history": []}

        class BestStateCallback(TrainerCallback):
            """Simpan state terbaik di RAM + early stopping berdasarkan metric lomba."""

            def on_evaluate(self, args_, st, control, metrics=None, **kw):
                score = (metrics or {}).get("eval_score")
                if score is None:
                    return control
                improved = is_better(score, state["best"])
                state["history"].append({"epoch": round(float(st.epoch or 0), 3), "train_loss": np.nan,
                                         "valid_score": score})
                print(f"   [{tag}] epoch {float(st.epoch or 0):.2f} | valid {CFG.METRIC} {score:.5f} "
                      f"{'*' if improved else ''}")
                if improved:
                    state["best"], state["bad"] = score, 0
                    state["best_state"] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                else:
                    state["bad"] += 1
                    if state["bad"] >= CFG.EARLY_STOPPING_PATIENCE:
                        print(f"   [{tag}] early stopping")
                        control.should_training_stop = True
                return control

        class LossTrainer(Trainer):
            def compute_loss(self, model_, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
                labels = inputs.pop("labels")
                outputs = model_(**inputs)
                loss = loss_fn(outputs["logits"], labels)
                return (loss, outputs) if return_outputs else loss

        def compute_metrics(p):
            preds = p.predictions[0] if isinstance(p.predictions, (tuple, list)) else p.predictions
            return {"score": float(eval_from_logits(np.asarray(preds, dtype=np.float32)))}

        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer, total)
        tkw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds, data_collator=collate,
                   compute_metrics=compute_metrics, callbacks=[BestStateCallback()], optimizers=(optimizer, scheduler))
        tsig = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in tsig:
            tkw["processing_class"] = tokenizer
        elif "tokenizer" in tsig:
            tkw["tokenizer"] = tokenizer
        trainer = LossTrainer(**tkw)
        trainer.train()
        if state["best_state"] is None:  # fallback: evaluasi manual sekali
            trainer.evaluate()
        del trainer
        return state["best_state"], state["best"], state["history"]

    # ---------------------------------------------------------------- satu fold
    def train_transformer_fold(model_name, tokenizer, doc_ids_tr, tr_idx, va_idx, doc_ids_te, labels, spec,
                               fold, batch_size, accum, logits_to_pred, score_fn, save_tag):
        """Latih satu fold. Return (valid_logits_doc, test_logits_doc, history)."""
        seed_everything(CFG.SEED + fold)
        if CFG.DEBUG_SIMULATE_OOM and batch_size == CFG.BATCH_SIZE:
            raise RuntimeError("CUDA out of memory. (simulated for testing)")
        strat, max_len = CFG.LONG_TEXT_STRATEGY, int(CFG.MAX_LEN)
        tr_items, tr_owner = build_items(doc_ids_tr, tr_idx, tokenizer, max_len, strat)
        va_items, va_owner = build_items(doc_ids_tr, va_idx, tokenizer, max_len, strat)
        label_dtype = torch.long if spec["task"] in ("binary", "multiclass") else torch.float32
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        use_tti = "token_type_ids" in getattr(tokenizer, "model_input_names", [])
        collate = make_collate(pad_id, use_tti, label_dtype)
        lab_tr = labels[tr_idx][tr_owner]
        train_ds = TokenDataset(tr_items, list(lab_tr))
        n_out = spec["num_outputs"]
        model = TextModel(model_name, n_out).to(DEVICE)
        loss_fn = build_loss_fn(spec)

        def eval_from_logits(item_logits):
            doc_logits = aggregate_logits(item_logits, va_owner, len(va_idx))
            return score_fn(logits_to_pred(doc_logits))

        tag = f"{save_tag} f{fold}"
        if CFG.TRAINER_BACKEND == "hf_trainer":
            valid_ds = TokenDataset(va_items, list(labels[va_idx][va_owner]))
            best_state, best, hist = train_hf_trainer(model, train_ds, valid_ds, collate, eval_from_logits,
                                                      loss_fn, batch_size, accum, tag, tokenizer)
        else:
            def eval_fn(m):
                return eval_from_logits(predict_logits(m, va_items, collate, n_out))
            best_state, best, hist = train_custom_loop(model, train_ds, collate, eval_fn, loss_fn, batch_size,
                                                       accum, tag)
        if best_state is not None:
            model.load_state_dict(best_state)
        if CFG.SAVE_FOLD_MODELS:
            torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "models", f"{save_tag}_fold{fold}.pt"))
        va_logits = aggregate_logits(predict_logits(model, va_items, collate, n_out), va_owner, len(va_idx))
        te_logits = None
        if doc_ids_te is not None:
            te_items, te_owner = build_items(doc_ids_te, np.arange(len(doc_ids_te)), tokenizer, max_len, strat)
            te_logits = aggregate_logits(predict_logits(model, te_items, collate, n_out), te_owner, len(doc_ids_te))
        for h in hist:
            h.update({"model": save_tag, "fold": fold})
        del model, best_state, train_ds
        free_memory()
        return va_logits, te_logits, hist

    def run_with_oom_retry(fn, batch_size, accum):
        """Jalankan fn(bs, accum); kalau CUDA OOM → bs/2, accum*2, ulangi (maks CFG.OOM_MAX_RETRIES)."""
        for attempt in range(CFG.OOM_MAX_RETRIES + 1):
            try:
                return fn(batch_size, accum)
            except Exception as e:
                if is_oom_error(e) and batch_size > 1 and attempt < CFG.OOM_MAX_RETRIES:
                    free_memory()
                    batch_size, accum = max(1, batch_size // 2), accum * 2
                    print(f"   ⚠️ CUDA OOM → retry dengan BATCH_SIZE={batch_size}, GRAD_ACCUM_STEPS={accum}")
                    continue
                raise

    def run_transformer_cv(model_key, spec, train_texts, labels, test_texts, logits_to_pred, score_fn_rows):
        """K-Fold fine-tuning satu model. Return (oof, test_pred, mask, history)."""
        model_name = resolve_model_name(model_key)
        tokenizer = load_tokenizer(model_name)
        if tokenizer is None:
            raise RuntimeError(f"Tokenizer '{model_name}' tidak bisa di-load (lihat pesan di atas).")
        doc_ids_tr = tokenize_docs(tokenizer, train_texts)
        doc_ids_te = tokenize_docs(tokenizer, test_texts) if test_texts is not None else None
        folds_to_run = list(CFG.FOLDS_TO_RUN) if CFG.FOLDS_TO_RUN is not None else list(range(CFG.N_FOLDS))
        oof, test_pred, mask, history = None, None, np.zeros(len(train_texts), bool), []
        safe_key = re.sub(r"[^A-Za-z0-9]+", "_", str(model_key))[-40:]
        for fold in folds_to_run:
            tr_idx, va_idx = np.where(FOLDS != fold)[0], np.where(FOLDS == fold)[0]
            t0 = time.time()

            def attempt(bs, acc):
                return train_transformer_fold(model_name, tokenizer, doc_ids_tr, tr_idx, va_idx, doc_ids_te, labels,
                                              spec, fold, bs, acc, logits_to_pred,
                                              lambda pred: score_fn_rows(va_idx, pred), safe_key)
            va_logits, te_logits, hist = run_with_oom_retry(attempt, CFG.BATCH_SIZE, CFG.GRAD_ACCUM_STEPS)
            va_pred = logits_to_pred(va_logits)
            if oof is None:
                oof = np.zeros((len(train_texts),) + va_pred.shape[1:], dtype=np.float64)
            oof[va_idx] = va_pred
            mask[va_idx] = True
            if te_logits is not None:
                tp = logits_to_pred(te_logits) / len(folds_to_run)
                test_pred = tp if test_pred is None else test_pred + tp
            history.extend(hist)
            print(f"   fold {fold}: {CFG.METRIC} = {score_fn_rows(va_idx, va_pred):.5f} ({time.time() - t0:.0f}s)")
        return oof, test_pred, mask, history

    print(f"Transformer engine siap | device={DEVICE} | backend={CFG.TRAINER_BACKEND}")
else:
    print("⚠️ torch/transformers tidak tersedia → engine transformer tidak didefinisikan (section [C] di-skip).")

# %% [markdown]
# ### 17.1 Jalankan fine-tuning K-Fold (regression head)
#
# **Detail khusus regresi:**
# - Output = 1 neuron (linear). Target = `transform(y)` lalu **distandarisasi** (mean 0, std 1) → training jauh
#   lebih stabil daripada memprediksi angka mentah (mis. harga jutaan).
# - Loss: `REG_LOSS="mse"` (default; cocok untuk RMSE), `"huber"` (tahan outlier), `"mae"` (untuk metric MAE).
# - `BACKBONE_DROPOUT=0.0` & `HEAD_DROPOUT=0.0`: trik populer lomba regresi teks — dropout membuat skala output
#   berbeda antara train & inference, yang merugikan regresi.
# - Best checkpoint dipilih berdasarkan metric lomba di skala asli.
#
# **Sebelum full run:** `CFG.FOLDS_TO_RUN = [0]` + `EPOCHS=1` untuk cek waktu & VRAM.
#
# **Jika X maka Y:**
# - Skor valid sama dengan baseline mean (prediksi hampir konstan) → LR terlalu kecil/besar atau epoch kurang;
#   coba `LR=3e-5`, `EPOCHS=4`, `HEAD_LR=1e-3`, cek transform target.
# - Loss NaN → `USE_AMP=False`, LR lebih kecil.
# - Transformer kalah dari TF-IDF+Ridge → cek `MAX_LEN` (teks terpotong?), bahasa model, jumlah data (< 1k baris
#   sering lebih cocok embedding + Ridge/SVR).
#
# **Untuk laporan:** konfigurasi (model, MAX_LEN, LR, epoch, loss), skor per fold, learning curve.

# %%
RUN_TF = CFG.RUN_TRANSFORMER
if str(RUN_TF).lower() == "auto":
    RUN_TF = HAS_CUDA
    if not HAS_CUDA:
        print("ℹ️ RUN_TRANSFORMER='auto' dan tidak ada GPU → fine-tuning dilewati.")
if RUN_TF and not (HAS_TORCH and HAS_TRANSFORMERS):
    print("⚠️ torch/transformers tidak tersedia → fine-tuning dilewati.")
    RUN_TF = False
if RUN_TF and CFG.TRAINER_BACKEND == "hf_trainer" and not HAS_ACCELERATE:
    print("⚠️ HF Trainer butuh package 'accelerate' → fallback ke custom_loop.")
    CFG.TRAINER_BACKEND = "custom_loop"

TF_HISTORY = []
if RUN_TF:
    spec = {"task": "regression", "num_outputs": 1, "class_weights": None}
    TF_MU, TF_SD = fold_standardizer(Y_T)
    train_labels = ((Y_T - TF_MU) / TF_SD).astype(np.float32)

    def logits_to_pred_reg(logits):
        """Output standar → skala asli."""
        return inverse_transform_target(np.asarray(logits).reshape(len(logits), -1)[:, 0] * TF_SD + TF_MU)

    def score_rows(idx, pred):
        return compute_score(Y[idx], pred)

    print(f"Backend={CFG.TRAINER_BACKEND} | MAX_LEN={CFG.MAX_LEN} | strategy={CFG.LONG_TEXT_STRATEGY} | "
          f"batch={CFG.BATCH_SIZE}x{CFG.GRAD_ACCUM_STEPS} | epochs={CFG.EPOCHS} | LR={CFG.LR} | loss={CFG.REG_LOSS}")
    for mk in CFG.TRANSFORMER_MODELS:
        key = "tf_" + re.sub(r"[^A-Za-z0-9]+", "_", str(mk).split("/")[-1])
        print("=" * 80, f"\nModel {mk} → {resolve_model_name(mk)}")
        t0 = time.time()
        try:
            oof, tp, mask, hist = run_transformer_cv(mk, spec, train["text_tf"].values, train_labels,
                                                     test["text_tf"].values if HAS_TEST else None,
                                                     logits_to_pred_reg, score_rows)
            register_result(key, oof, tp, "transformer", time.time() - t0, mask=mask)
            TF_HISTORY.extend(hist)
        except Exception as e:
            FAILED_MODELS[key] = short_err(e)
            print(f"❌ {key} gagal: {FAILED_MODELS[key]}")
            print("".join(traceback.format_exception(type(e), e, e.__traceback__)[-3:]))
            msg = str(e).lower()
            if "sentencepiece" in msg or "tiktoken" in msg:
                print("   → pip install sentencepiece protobuf, lalu restart kernel")
            elif "out of memory" in msg:
                print("   → kecilkan MAX_LEN/BATCH_SIZE, GRADIENT_CHECKPOINTING=True, atau model lebih kecil")
            elif "connect" in msg or "resolve" in msg or "offline" in msg:
                print("   → model tidak bisa di-download; pakai path lokal / cache HF")
        free_memory()

if TF_HISTORY:
    H = pd.DataFrame(TF_HISTORY)
    save_table(H, "transformer_learning_curves")
    for mname, g in H.groupby("model"):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for f, gf in g.groupby("fold"):
            ax.plot(gf["epoch"], gf["valid_score"], marker="o", label=f"fold {f}")
        ax.set_title(f"Validation {CFG.METRIC} per epoch — {mname}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Validation {CFG.METRIC}")
        ax.legend(fontsize=8)
        save_fig(f"learning_curve_{mname}", fig)

# %% [markdown]
# ## 18. Pseudo-labeling (opsional, default OFF)
#
# Untuk regresi tidak ada "confidence" probabilitas, jadi dipakai **kesepakatan antar model**: baris test dengan
# standar deviasi prediksi antar model paling kecil (kuantil `PSEUDO_MAX_DISAGREEMENT`) diberi target = prediksi
# rata-rata model, lalu ditambahkan ke **fold train saja** untuk melatih ulang TF-IDF + Ridge.
#
# ⚠️ Peringatan: pseudo-label yang bias memperkuat bias (prediksi cenderung "ke tengah"); prediksi test berasal dari
# model yang pernah melihat semua train → CV sedikit optimis; anggap gain kecil sebagai noise; cek aturan lomba.

# %%
if CFG.RUN_PSEUDO_LABEL and HAS_TEST and RESULTS:
    full_res = {k: v for k, v in RESULTS.items() if not v["partial"] and v["test"] is not None}
    top = sorted(full_res, key=lambda k: -full_res[k]["score"] if GREATER_IS_BETTER else full_res[k]["score"])[:4]
    P = np.vstack([full_res[k]["test"] for k in top])
    disagreement = P.std(0) if len(top) > 1 else np.zeros(P.shape[1])
    cut = np.quantile(disagreement, CFG.PSEUDO_MAX_DISAGREEMENT)
    cand = np.where(disagreement <= cut)[0][:int(CFG.PSEUDO_MAX_FRACTION * len(train))]
    print(f"Pseudo-label source: mean of {top} | selected {len(cand)} / {P.shape[1]} test rows")
    if len(cand):
        pseudo_y = P.mean(0)[cand]
        pseudo_txt = test["text_cl"].values[cand]
        oof_pl, tp_pl = np.zeros(len(train)), np.zeros(len(test))
        for fold in range(CFG.N_FOLDS):
            tr_idx, va_idx = np.where(FOLDS != fold)[0], np.where(FOLDS == fold)[0]
            txt_tr = np.concatenate([train["text_cl"].values[tr_idx], pseudo_txt])
            y_tr = np.concatenate([Y[tr_idx], pseudo_y])
            Xtr, (Xva, Xte), _ = build_tfidf(txt_tr, [train["text_cl"].values[va_idx], test["text_cl"].values])
            pv, pt = fit_predict_reg("ridge", Xtr, y_tr, [Xva, Xte])
            oof_pl[va_idx] = pv
            tp_pl += pt / CFG.N_FOLDS
        register_result("tfidf_ridge_pseudo", oof_pl, tp_pl, "pseudo")
        if "tfidf_ridge" in RESULTS:
            gain = RESULTS["tfidf_ridge_pseudo"]["score"] - RESULTS["tfidf_ridge"]["score"]
            gain = gain if GREATER_IS_BETTER else -gain
            print(f"Gain vs tfidf_ridge: {gain:+.5f}", "✅ membantu" if gain > 0 else "⚠️ tidak membantu")
elif CFG.RUN_PSEUDO_LABEL:
    print("Pseudo-labeling butuh test set dan minimal satu model → dilewati.")
else:
    print("RUN_PSEUDO_LABEL=False → dilewati.")

# %% [markdown]
# ## 19. Perbandingan model
#
# Tabel semua model: skor OOF metric utama + metric lain, mean ± std per fold, waktu.
#
# **Cara membaca:** bandingkan dengan baseline "predict mean". Selisih sangat kecil antar model (mis. < 0.5% RMSE)
# biasanya tidak signifikan. Model `partial=True` tidak ikut ensemble.
# **Untuk laporan:** tabel + bar chart ini = inti bagian "Hasil Eksperimen".

# %%
if not RESULTS:
    raise RuntimeError("Tidak ada model yang berhasil. Cek FAILED_MODELS: " + json.dumps(FAILED_MODELS, indent=1))

rows = []
for name, r in RESULTS.items():
    row = {"model": name, "family": r["family"], CFG.METRIC: r["score"], "fold_mean": r["fold_mean"],
           "fold_std": r["fold_std"], "seconds": r["seconds"], "partial": r["partial"]}
    if not r["partial"]:
        row.update({m: v for m, v in evaluate_all(Y, r["oof"]).items() if m != CFG.METRIC})
    rows.append(row)
SCORE_DF = pd.DataFrame(rows).sort_values(CFG.METRIC, ascending=not GREATER_IS_BETTER).reset_index(drop=True)
display(SCORE_DF.round(5))
save_table(SCORE_DF, "model_comparison")
if FAILED_MODELS:
    print("Model gagal:", FAILED_MODELS)

fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(SCORE_DF) + 1)))
d = SCORE_DF.iloc[::-1]
ax.barh(d["model"], d[CFG.METRIC], xerr=d["fold_std"],
        color=["#DD8452" if f == "transformer" else "#55A868" if f == "embedding" else "#4C72B0" for f in d["family"]])
ax.axvline(compute_score(Y, np.full(len(Y), Y.mean())), color="gray", ls="--", label="predict-mean baseline")
ax.legend()
ax.set_title(f"Model comparison (OOF {CFG.METRIC}, error bar = fold std)")
ax.set_xlabel(f"OOF {CFG.METRIC}")
ax.set_ylabel("Model")
save_fig("model_comparison", fig)

# %% [markdown]
# ## 20. Ensemble
#
# | metode | cara | catatan |
# |---|---|---|
# | `mean` | rata-rata prediksi | aman |
# | `weighted` | bobot ≥ 0 (jumlah 1) dioptimasi scipy pada OOF | metric halus (rmse/mae/rmsle/r2/pearson) dioptimasi langsung; lainnya pakai RMSE |
# | `rank` | rata-rata rank, dipetakan ke distribusi prediksi rata-rata | bagus untuk metric korelasi (spearman) |
# | `stacking` | Ridge(positive) di atas OOF semua model (fold sama) | rawan overfit di data kecil |
#
# **Jika** ensemble tidak lebih baik dari model terbaik → pakai model tunggal. **Untuk laporan:** heatmap korelasi
# OOF + skor ensemble vs model tunggal.

# %%
cands = [k for k in SCORE_DF["model"] if not RESULTS[k]["partial"]][:CFG.ENSEMBLE_TOP_K]
cands = [k for k in cands if (RESULTS[k]["test"] is not None) or not HAS_TEST]
print("Ensemble candidates:", cands)
if len(cands) >= 2:
    corr = pd.DataFrame({k: RESULTS[k]["oof"] for k in cands}).corr(method="pearson")
    fig, ax = plt.subplots(figsize=(1.0 * len(cands) + 3, 0.8 * len(cands) + 2))
    im = ax.imshow(corr.values, cmap="viridis", vmin=min(0.5, corr.values.min()), vmax=1)
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels(cands, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cands)))
    ax.set_yticklabels(cands, fontsize=8)
    for i in range(len(cands)):
        for j in range(len(cands)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Pearson correlation of OOF predictions")
    save_fig("oof_correlation", fig)


def blend(preds, w):
    w = np.asarray(w, dtype=float)
    w = w / max(w.sum(), EPS)
    return sum(wi * np.asarray(p) for wi, p in zip(w, preds))


OPT_METRIC = CFG.METRIC if CFG.METRIC in ("rmse", "mae", "rmsle", "r2", "pearson") else "rmse"


def optimize_weights(oofs):
    """Bobot non-negatif (jumlah 1) yang mengoptimalkan OPT_METRIC pada OOF."""
    n = len(oofs)
    sign = -1 if METRICS[OPT_METRIC][0] else 1

    def obj(w):
        return sign * compute_score(Y, blend(oofs, np.clip(w, 0, None) + 1e-12), OPT_METRIC)
    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1},))
    w = res.x if res.success and np.all(np.isfinite(res.x)) else np.ones(n) / n
    w = np.clip(w, 0, None)
    return w / max(w.sum(), EPS)


def rank_to_scale(rank_pred, ref_pred):
    """Petakan skor rank ke distribusi nilai ref_pred (urutan tetap, skala asli kembali)."""
    out = np.empty(len(rank_pred))
    out[np.argsort(rank_pred, kind="mergesort")] = np.sort(np.asarray(ref_pred, dtype=float))
    return out


def run_stacking(names):
    """Ridge(positive=True) di atas OOF (skala transform); OOF stacking memakai fold yang sama."""
    Fo = np.column_stack([transform_target(RESULTS[k]["oof"]) for k in names])
    Ft = np.column_stack([transform_target(RESULTS[k]["test"]) for k in names]) if HAS_TEST else None

    def make():
        try:
            return Ridge(alpha=1.0, positive=True)
        except TypeError:
            return Ridge(alpha=1.0)
    oof_s = np.zeros(len(Y))
    for f in range(CFG.N_FOLDS):
        tr, va = FOLDS != f, FOLDS == f
        oof_s[va] = make().fit(Fo[tr], Y_T[tr]).predict(Fo[va])
    te_s = make().fit(Fo, Y_T).predict(Ft) if HAS_TEST else None
    return inverse_transform_target(oof_s), (inverse_transform_target(te_s) if HAS_TEST else None)


ENSEMBLE_WEIGHTS = {}
if CFG.RUN_ENSEMBLE and len(cands) >= 2:
    oofs = [RESULTS[k]["oof"] for k in cands]
    tests = [RESULTS[k]["test"] for k in cands] if HAS_TEST else None
    for method in CFG.ENSEMBLE_METHODS:
        try:
            if method == "mean":
                w = np.ones(len(cands))
                if CFG.MANUAL_WEIGHTS:
                    w = np.array([CFG.MANUAL_WEIGHTS.get(k, 0.0) for k in cands]) + EPS
                o, t = blend(oofs, w), (blend(tests, w) if HAS_TEST else None)
            elif method == "weighted":
                w = optimize_weights(oofs)
                ENSEMBLE_WEIGHTS = {k: round(float(v), 4) for k, v in zip(cands, w)}
                print(f"Optimized weights ({OPT_METRIC}):", ENSEMBLE_WEIGHTS)
                o, t = blend(oofs, w), (blend(tests, w) if HAS_TEST else None)
            elif method == "rank":
                o = rank_to_scale(np.mean([rankdata(p) for p in oofs], axis=0), blend(oofs, np.ones(len(oofs))))
                t = rank_to_scale(np.mean([rankdata(p) for p in tests], axis=0),
                                  blend(tests, np.ones(len(tests)))) if HAS_TEST else None
            elif method == "stacking":
                o, t = run_stacking(cands)
            else:
                print("Metode ensemble tidak dikenal:", method)
                continue
            register_result(f"ens_{method}", o, t, "ensemble")
        except Exception as e:
            FAILED_MODELS[f"ens_{method}"] = short_err(e)
            print(f"❌ ens_{method} gagal: {FAILED_MODELS[f'ens_{method}']}")
else:
    print("Ensemble dilewati (RUN_ENSEMBLE=False atau < 2 kandidat).")

valid_final = {k: v for k, v in RESULTS.items() if not v["partial"] and (v["test"] is not None or not HAS_TEST)}
if CFG.FINAL_PREDICTION != "auto" and CFG.FINAL_PREDICTION in valid_final:
    FINAL_NAME = CFG.FINAL_PREDICTION
else:
    FINAL_NAME = max(valid_final, key=lambda k: valid_final[k]["score"] if GREATER_IS_BETTER else -valid_final[k]["score"])
best_single = max([k for k in valid_final if valid_final[k]["family"] != "ensemble"] or [FINAL_NAME],
                  key=lambda k: valid_final[k]["score"] if GREATER_IS_BETTER else -valid_final[k]["score"])
print(f"➡️ FINAL prediction: {FINAL_NAME} (OOF {CFG.METRIC} = {RESULTS[FINAL_NAME]['score']:.5f}) | "
      f"best single: {best_single} ({RESULTS[best_single]['score']:.5f})")
FINAL_OOF_RAW = RESULTS[FINAL_NAME]["oof"]
FINAL_TEST_RAW = RESULTS[FINAL_NAME]["test"]

# %% [markdown]
# ## 21. Post-processing: clip, rounding ke nilai valid, optimized rounding thresholds
#
# Kandidat yang dibandingkan (semua di OOF):
# 1. `clip` — prediksi dibatasi ke `[MIN_TARGET, MAX_TARGET]` (selalu aman).
# 2. `round_nearest` — dibulatkan ke `VALID_VALUES` terdekat (mis. rating 1–5).
# 3. `round_optimized` — threshold pembulatan dicari supaya metric maksimal (OptimizedRounder; klasik untuk QWK).
#    Contoh: batas 3→4 bukan 3.5 tetapi 3.62.
#
# **Anti-overfit:** setiap kandidat dievaluasi dengan **cross-validated check** (threshold dicari di K−1 fold,
# dievaluasi di fold sisanya). Rounding dipakai hanya kalau gain CV ≥ `MIN_TUNING_GAIN`.
#
# **Jika** metric RMSE/MAE dan target rating → rounding biasanya **memperburuk** RMSE (prediksi kontinu lebih baik)
# tapi memperbaiki accuracy/QWK. Ikuti metric resmi!
# **Untuk laporan:** tabel kandidat post-processing + alasan memilih.

# %%
def find_rounding_thresholds(y, p, vals):
    """Coordinate search threshold antar nilai valid untuk memaksimalkan/meminimalkan metric."""
    th = (vals[:-1] + vals[1:]) / 2.0
    sign = 1 if GREATER_IS_BETTER else -1

    def score_th(t):
        return sign * compute_score(y, round_to_valid(p, np.sort(t)))
    best = score_th(th)
    for _ in range(3):
        for i in range(len(th)):
            lo, hi = vals[i], vals[i + 1]
            for cand in np.linspace(lo, hi, 21)[1:-1]:
                trial = th.copy()
                trial[i] = cand
                s = score_th(trial)
                if s > best + 1e-12:
                    best, th = s, trial
    return np.sort(th)


POST = {"name": "clip", "thresholds": None}
post_rows = [("raw (no clip)", METRICS[CFG.METRIC][1](Y, FINAL_OOF_RAW), np.nan),
             ("clip", compute_score(Y, FINAL_OOF_RAW), 0.0)]
if VALID_VALUES is not None and len(VALID_VALUES) >= 2 and CFG.ROUND_TO_VALID in ("auto", True):
    base_cv = [compute_score(Y[FOLDS == f], FINAL_OOF_RAW[FOLDS == f]) for f in range(CFG.N_FOLDS)]
    near = round_to_valid(clip_pred(FINAL_OOF_RAW))
    near_cv = [compute_score(Y[FOLDS == f], near[FOLDS == f]) for f in range(CFG.N_FOLDS)]
    sign = 1 if GREATER_IS_BETTER else -1
    gain_near = sign * (np.mean(near_cv) - np.mean(base_cv))
    post_rows.append(("round_nearest", compute_score(Y, near), gain_near))
    gain_opt, th_full = -np.inf, None
    if CFG.OPTIMIZE_ROUNDING:
        opt_cv = []
        for f in range(CFG.N_FOLDS):
            tr, va = FOLDS != f, FOLDS == f
            th_f = find_rounding_thresholds(Y[tr], clip_pred(FINAL_OOF_RAW[tr]), VALID_VALUES)
            opt_cv.append(compute_score(Y[va], round_to_valid(clip_pred(FINAL_OOF_RAW[va]), th_f)))
        gain_opt = sign * (np.mean(opt_cv) - np.mean(base_cv))
        th_full = find_rounding_thresholds(Y, clip_pred(FINAL_OOF_RAW), VALID_VALUES)
        post_rows.append(("round_optimized", compute_score(Y, round_to_valid(clip_pred(FINAL_OOF_RAW), th_full)), gain_opt))
        print("Optimized thresholds:", np.round(th_full, 3).tolist())
    force = CFG.ROUND_TO_VALID is True
    if max(gain_near, gain_opt) >= CFG.MIN_TUNING_GAIN or force:
        if gain_opt > gain_near and th_full is not None:
            POST = {"name": "round_optimized", "thresholds": th_full}
        else:
            POST = {"name": "round_nearest", "thresholds": None}
post_df = pd.DataFrame(post_rows, columns=["postprocess", f"OOF {CFG.METRIC}", "cross-validated gain"])
display(post_df.round(5))
save_table(post_df, "postprocessing_candidates")
print(f"➡️ Post-processing dipakai: {POST['name']}")


def apply_post(p):
    """Terapkan post-processing terpilih."""
    p = clip_pred(p)
    if POST["name"] == "round_nearest":
        return round_to_valid(p)
    if POST["name"] == "round_optimized":
        return round_to_valid(p, POST["thresholds"])
    return p


FINAL_OOF = apply_post(FINAL_OOF_RAW)
FINAL_TEST = apply_post(FINAL_TEST_RAW) if FINAL_TEST_RAW is not None else None
FINAL_METRICS = evaluate_all(Y, FINAL_OOF)
display(pd.Series(FINAL_METRICS).round(5).to_frame("value"))
save_table(pd.Series(FINAL_METRICS).to_frame("value").reset_index(), "final_oof_metrics")
log_experiment(f"FINAL::{FINAL_NAME}", FINAL_METRICS[CFG.METRIC], {"post": POST["name"]})

# %% [markdown]
# ## 22. Error analysis
#
# **Yang dilihat:**
# 1. **Scatter prediksi vs aktual** — titik harus mengikuti garis diagonal. Pola "mendatar" = prediksi ditarik ke
#    rata-rata (regression to the mean): nilai rendah di-overestimate, nilai tinggi di-underestimate.
# 2. **Residual per rentang target** (bias & MAE per bin) — memperlihatkan di rentang mana model paling lemah.
# 3. **Residual per panjang teks** — teks pendek kurang konteks? teks panjang terpotong?
# 4. **Worst errors** — baca teksnya: label salah? informasi target ada di bagian teks yang terpotong?
# 5. Target diskrit: confusion matrix prediksi yang dibulatkan.
#
# **Jika X maka Y:**
# - bias besar di ekstrem (rentang rendah positif, rentang tinggi negatif) → wajar; coba `TARGET_TRANSFORM`, loss
#   MSE (bukan MAE), ensemble dengan model non-linear, atau stacking (bisa "melebarkan" prediksi).
# - error besar di teks panjang → `MAX_LEN` lebih besar / `LONG_TEXT_STRATEGY="chunk"`.
# - worst errors berisi target tidak masuk akal → outlier/label error; `REG_LOSS="huber"`.
#
# **Untuk laporan:** scatter pred vs actual + tabel residual per rentang target.

# %%
RES = pd.DataFrame({"text": train["raw_text"].values, "y_true": Y, "y_pred": FINAL_OOF, "fold": FOLDS})
if CFG.ID_COL in train.columns:
    RES.insert(0, CFG.ID_COL, train[CFG.ID_COL].values)
RES["residual"] = RES["y_pred"] - RES["y_true"]
RES["abs_error"] = RES["residual"].abs()
RES["n_words"] = train["text_tf"].str.split().str.len().values

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
lims = [min(Y.min(), FINAL_OOF.min()), max(Y.max(), FINAL_OOF.max())]
axes[0].scatter(Y, FINAL_OOF, s=8, alpha=0.35)
axes[0].plot(lims, lims, "r--", label="y = x")
axes[0].set_title(f"Predicted vs actual (OOF, {FINAL_NAME})")
axes[0].set_xlabel(f"Actual {CFG.TARGET_COL}")
axes[0].set_ylabel(f"Predicted {CFG.TARGET_COL}")
axes[0].legend()
axes[1].hist(RES["residual"], bins=50, color="#8172B2")
axes[1].set_title("Residual distribution (pred − actual)")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
save_fig("pred_vs_actual_and_residuals", fig)

try:
    RES["target_bin"] = pd.qcut(RES["y_true"], q=min(5, len(np.unique(Y))), duplicates="drop").astype(str)
except Exception:
    RES["target_bin"] = "all"
by_bin = RES.groupby("target_bin", sort=False).agg(n=("residual", "size"), mean_true=("y_true", "mean"),
                                                   bias=("residual", "mean"), mae=("abs_error", "mean")
                                                   ).sort_values("mean_true")
print("Residual per rentang target (bias > 0 = overestimate):")
display(by_bin.round(4))
save_table(by_bin.reset_index(), "residual_by_target_range")
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(by_bin.index.astype(str), by_bin["bias"], color=["#C44E52" if b > 0 else "#4C72B0" for b in by_bin["bias"]])
ax.axhline(0, color="k", lw=0.8)
ax.set_title("Mean residual (bias) by actual target range")
ax.set_xlabel("Actual target range")
ax.set_ylabel("Mean residual (pred − actual)")
plt.xticks(rotation=20)
save_fig("bias_by_target_range", fig)

try:
    RES["len_bucket"] = pd.qcut(RES["n_words"], q=min(5, RES["n_words"].nunique()), duplicates="drop").astype(str)
except Exception:
    RES["len_bucket"] = "all"
by_len = RES.groupby("len_bucket", sort=False).agg(n=("abs_error", "size"), mae=("abs_error", "mean"),
                                                   min_words=("n_words", "min")).sort_values("min_words")
print("Error per panjang teks:")
display(by_len.round(4))
save_table(by_len.reset_index(), "error_by_length_bucket")
print("Worst predictions (abs error terbesar):")
display(RES.sort_values("abs_error", ascending=False).head(15))
save_table(RES.sort_values("abs_error", ascending=False).head(200), "worst_errors")

if VALID_VALUES is not None and len(VALID_VALUES) <= 15:
    yr = round_to_valid(Y)
    pr = round_to_valid(FINAL_OOF) if POST["name"] == "clip" else FINAL_OOF
    cm = skm.confusion_matrix(yr, pr, labels=VALID_VALUES)
    fig, ax = plt.subplots(figsize=(1 + 0.7 * len(VALID_VALUES), 0.6 * len(VALID_VALUES) + 1.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(VALID_VALUES)))
    ax.set_yticks(range(len(VALID_VALUES)))
    ax.set_xticklabels([f"{v:g}" for v in VALID_VALUES])
    ax.set_yticklabels([f"{v:g}" for v in VALID_VALUES])
    for i in range(len(VALID_VALUES)):
        for j in range(len(VALID_VALUES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix of rounded OOF predictions")
    ax.set_xlabel("Predicted (rounded)")
    ax.set_ylabel("Actual")
    save_fig("rounded_confusion_matrix", fig)

# %% [markdown]
# ## 23. Interpretability
#
# **23.1 Koefisien Ridge (global):** Ridge pada TF-IDF word (1–2 gram) di-fit di seluruh train (target transform).
# Koefisien **positif** = n-gram yang menaikkan prediksi target, **negatif** = menurunkan. Sangat bagus untuk laporan
# ("kata 'strategis', 'jakarta' menaikkan harga; 'renovasi' menurunkan").
#
# **23.2 Token occlusion (lokal):** untuk beberapa contoh, tiap kata dihapus lalu dilihat perubahan prediksi.
# SHAP LinearExplainer dijalankan juga kalau package `shap` ada.
#
# **Jika** kata berpengaruh tidak masuk akal (nama, URL, template) → model belajar shortcut.

# %%
INTERP_MODEL = None
if CFG.RUN_INTERPRETABILITY:
    try:
        vec_i = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=100000, sublinear_tf=True)
        Xi = vec_i.fit_transform(train["text_cl"])
        mu_i, sd_i = fold_standardizer(Y_T)
        ridge_i = Ridge(alpha=CFG.RIDGE_ALPHA).fit(Xi, (Y_T - mu_i) / sd_i)
        INTERP_MODEL = (vec_i, ridge_i, mu_i, sd_i)
        names = np.array(vec_i.get_feature_names_out())
        coef = ridge_i.coef_.ravel()
        top_pos, top_neg = np.argsort(-coef)[:20], np.argsort(coef)[:20]
        COEF_DF = pd.concat([pd.DataFrame({"direction": "increases target", "ngram": names[top_pos], "coef": coef[top_pos]}),
                             pd.DataFrame({"direction": "decreases target", "ngram": names[top_neg], "coef": coef[top_neg]})])
        save_table(COEF_DF, "ridge_top_ngrams")
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, idx, col, title in [(axes[0], top_pos[:15][::-1], "#C44E52", "Increase target"),
                                    (axes[1], top_neg[:15][::-1], "#4C72B0", "Decrease target")]:
            ax.barh(names[idx], coef[idx], color=col)
            ax.set_title(title)
            ax.set_xlabel("Ridge coefficient (standardized target)")
        fig.suptitle("Top weighted n-grams (TF-IDF + Ridge)")
        save_fig("ridge_top_ngrams", fig)
    except Exception as e:
        print("Ridge interpretability dilewati:", short_err(e))


def occlusion_explain_reg(text, predict_fn, top_k=8):
    """Hapus tiap kata, ukur perubahan prediksi (skala asli). Return list (kata, delta)."""
    words = str(text).split()
    if not words:
        return []
    base = predict_fn([" ".join(words)])[0]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words[:200]))]
    deltas = base - predict_fn(variants)
    order = np.argsort(-np.abs(deltas))[:top_k]
    return [(words[i], float(deltas[i])) for i in order]


if CFG.RUN_INTERPRETABILITY and INTERP_MODEL is not None:
    vec_i, ridge_i, mu_i, sd_i = INTERP_MODEL

    def predict_interp(texts):
        X = vec_i.transform([classical_normalize(t) for t in texts])
        return inverse_transform_target(ridge_i.predict(X) * sd_i + mu_i)

    ex_idx = list(RES.sort_values("abs_error", ascending=False).index[:max(1, CFG.N_EXPLAIN_EXAMPLES // 2)])
    ex_idx += list(RES.sort_values("abs_error").index[:CFG.N_EXPLAIN_EXAMPLES - len(ex_idx)])
    for k, i in enumerate(ex_idx):
        txt = train["raw_text"].iloc[i]
        expl = occlusion_explain_reg(txt, predict_interp)
        print(f"\n[{k + 1}] true={Y[i]:.4g} | interp-model pred={predict_interp([txt])[0]:.4g}")
        print("    text:", str(txt)[:300])
        print("    kata berpengaruh (Δpred jika kata dihapus):", [(w, round(d, 4)) for w, d in expl])
        if k == 0 and expl:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            w_, d_ = zip(*expl[::-1])
            ax.barh(w_, d_, color=["#C44E52" if v > 0 else "#4C72B0" for v in d_])
            ax.set_title("Token occlusion — contribution to prediction")
            ax.set_xlabel("Drop in prediction when word removed")
            save_fig("occlusion_example", fig)
    if CFG.USE_SHAP and HAS_SHAP:
        try:
            explainer = shap.LinearExplainer(ridge_i, vec_i.transform(train["text_cl"].iloc[:min(500, len(train))]))
            sv = explainer.shap_values(vec_i.transform(train["text_cl"].iloc[ex_idx]))
            sv = np.asarray(sv.todense() if sparse.issparse(sv) else sv)
            fn = np.array(vec_i.get_feature_names_out())
            for r in range(sv.shape[0]):
                top = np.argsort(-np.abs(sv[r]))[:8]
                print(f"SHAP example {r + 1}:", [(fn[j], round(float(sv[r, j]), 3)) for j in top])
        except Exception as e:
            print("SHAP dilewati:", short_err(e))
    elif CFG.USE_SHAP:
        print("ℹ️ shap tidak terpasang → hanya occlusion.")

# %% [markdown]
# ## 24. Submission + validator
#
# Prediksi test final (setelah post-processing) ditulis ke kolom target. Kalau ada sample_submission, urutan ID &
# nama kolom mengikuti sample (merge berdasarkan ID).
#
# **Validator** mengecek: jumlah baris, kolom & urutan, urutan ID, tidak ada NaN/inf, prediksi numerik dan di dalam
# rentang `[MIN_TARGET, MAX_TARGET]`. Semua harus ✅.
# **Jika** distribusi prediksi test sangat berbeda dari target train → cek drift / bug (mis. lupa inverse transform).

# %%
def validate_submission(sub, sample=None):
    """Cek format submission regresi. Return True kalau semua lolos."""
    pred_cols = [c for c in sub.columns if c != CFG.ID_COL]
    vals = sub[pred_cols].apply(pd.to_numeric, errors="coerce")
    checks = [("no NaN / inf", bool(np.isfinite(vals.values).all())), ("rows > 0", len(sub) > 0),
              ("within [MIN, MAX]", bool(((vals >= MIN_T - 1e-9) & (vals <= MAX_T + 1e-9)).all().all())
               if CFG.CLIP_PREDICTIONS else True)]
    if sample is not None:
        checks += [("same n_rows as sample", len(sub) == len(sample)),
                   ("same columns & order", list(sub.columns) == list(sample.columns))]
        if CFG.ID_COL in sample.columns and CFG.ID_COL in sub.columns:
            checks.append(("same ID order", (sub[CFG.ID_COL].astype(str).values == sample[CFG.ID_COL].astype(str).values).all()))
    elif HAS_TEST:
        checks.append(("same n_rows as test", len(sub) == len(test)))
    ok = all(v for _, v in checks)
    display(pd.DataFrame([(n, "✅" if v else "❌") for n, v in checks], columns=["check", "status"]))
    print("SUBMISSION VALID ✅" if ok else "❌ SUBMISSION TIDAK VALID — perbaiki sebelum upload!")
    return ok


SUBMISSION = None
if HAS_TEST and FINAL_TEST is not None:
    pred_col = CFG.SUBMISSION_TARGET_COL or CFG.TARGET_COL
    if sample_sub is not None:
        others = [c for c in sample_sub.columns if c != CFG.ID_COL]
        if pred_col not in sample_sub.columns and len(others) == 1:
            pred_col = others[0]
            print(f"ℹ️ kolom prediksi mengikuti sample_submission: '{pred_col}'")
    pred_df = pd.DataFrame({pred_col: FINAL_TEST})
    if IS_DISCRETE and POST["name"] != "clip" and np.allclose(FINAL_TEST, np.round(FINAL_TEST)):
        pred_df[pred_col] = np.round(FINAL_TEST).astype(int)
    if CFG.ID_COL in test.columns:
        pred_df.insert(0, CFG.ID_COL, test[CFG.ID_COL].values)
    if sample_sub is not None and CFG.ID_COL in sample_sub.columns and CFG.ID_COL in pred_df.columns:
        SUBMISSION = sample_sub[[CFG.ID_COL]].merge(pred_df, on=CFG.ID_COL, how="left")
        SUBMISSION = SUBMISSION[[c for c in sample_sub.columns if c in SUBMISSION.columns]]
    else:
        SUBMISSION = pred_df
    display(SUBMISSION.head())
    SUB_OK = validate_submission(SUBMISSION, sample_sub)
    sub_path = os.path.join(CFG.OUTPUT_DIR, "submissions", "submission.csv")
    SUBMISSION.to_csv(sub_path, index=False)
    tag = f"{FINAL_NAME}_{CFG.METRIC}{RESULTS[FINAL_NAME]['score']:.4f}_{time.strftime('%m%d_%H%M')}"
    SUBMISSION.to_csv(os.path.join(CFG.OUTPUT_DIR, "submissions", f"submission_{tag}.csv"), index=False)
    print("Saved:", sub_path)
    cmp = pd.DataFrame({"train_target": pd.Series(Y).describe(), "test_pred": pd.Series(FINAL_TEST).describe()})
    display(cmp.round(4))
else:
    print("Test tidak ada → submission dilewati (hasil CV tetap tersimpan).")

# %% [markdown]
# ## 25. Simpan OOF, prediksi test, konfigurasi & ringkasan untuk laporan
#
# File: `preds/oof_<model>.npy` & `preds/test_<model>.npy` (semua model), `oof_final.csv`, `test_pred_final.csv`,
# `cfg_used.json`, `experiment_log.csv`. Cetakan "REPORT SUMMARY" berisi angka kunci untuk laporan.

# %%
oof_out = pd.DataFrame({"fold": FOLDS, "y_true": Y, "y_pred_raw": FINAL_OOF_RAW, "y_pred_final": FINAL_OOF})
if CFG.ID_COL in train.columns:
    oof_out.insert(0, CFG.ID_COL, train[CFG.ID_COL].values)
oof_out.to_csv(os.path.join(CFG.OUTPUT_DIR, "oof_final.csv"), index=False)
if HAS_TEST and FINAL_TEST is not None:
    te_out = pd.DataFrame({"y_pred_raw": FINAL_TEST_RAW, "y_pred_final": FINAL_TEST})
    if CFG.ID_COL in test.columns:
        te_out.insert(0, CFG.ID_COL, test[CFG.ID_COL].values)
    te_out.to_csv(os.path.join(CFG.OUTPUT_DIR, "test_pred_final.csv"), index=False)

cfg_dump = {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict, tuple)) else str(v))
            for k, v in vars(CFG).items() if not k.startswith("_")}
cfg_dump.update({"_FINAL_MODEL": FINAL_NAME, "_POSTPROCESS": POST["name"],
                 "_THRESHOLDS": None if POST["thresholds"] is None else np.asarray(POST["thresholds"]).tolist()})
with open(os.path.join(CFG.OUTPUT_DIR, "cfg_used.json"), "w", encoding="utf-8") as f:
    json.dump(cfg_dump, f, indent=2, default=str)

print("=" * 80)
print("REPORT SUMMARY")
print("=" * 80)
print(f"Data         : {len(train)} train rows" + (f", {len(test)} test rows" if HAS_TEST else ", no test file"))
print(f"Target       : mean={Y.mean():.4g}, std={Y.std():.4g}, skew={SKEW:.2f}, discrete={IS_DISCRETE}, "
      f"transform={CFG.TARGET_TRANSFORM}")
print(f"Language     : {dict((k, round(v, 3)) for k, v in lang_share.items())} → model {CFG.TRANSFORMER_MODELS}")
print(f"MAX_LEN      : {CFG.MAX_LEN} | Spearman(length, target) = {rho_len:.3f}")
print(f"Validation   : {CFG.N_FOLDS}-fold {CV_STRATEGY_USED} CV, metric = {CFG.METRIC}")
print(f"Baseline     : predict-mean {CFG.METRIC} = {compute_score(Y, np.full(len(Y), Y.mean())):.5f}")
print("Top models   :")
for _, r in SCORE_DF.head(5).iterrows():
    print(f"   {r['model']:<40s} {CFG.METRIC}={r[CFG.METRIC]:.5f} ± {r['fold_std']:.4f}")
print(f"Final        : {FINAL_NAME} + {POST['name']} | OOF {CFG.METRIC} = {FINAL_METRICS[CFG.METRIC]:.5f}")
print(f"Failed       : {list(FAILED_MODELS) or 'none'}")
print(f"Figures      : {len(os.listdir(os.path.join(CFG.OUTPUT_DIR, 'figures')))} files in {os.path.join(CFG.OUTPUT_DIR, 'figures')}")

# %% [markdown]
# ## 26. Experiment recipes
#
# **R1 — Laptop tanpa GPU**
# ```python
# CFG.RUN_TRANSFORMER = False
# CFG.CLASSICAL_MODELS = ["ridge", "linearsvr", "lgbm_svd"]
# CFG.EMBEDDING_HEADS = ["ridge", "svr"]
# ```
#
# **R2 — Rating 1–5 dengan metric QWK**
# ```python
# CFG.METRIC = "qwk"; CFG.VALID_VALUES = [1, 2, 3, 4, 5]
# CFG.TARGET_TRANSFORM = "none"; CFG.OPTIMIZE_ROUNDING = True
# ```
#
# **R3 — Rating dengan metric RMSE**
# ```python
# CFG.METRIC = "rmse"; CFG.ROUND_TO_VALID = False      # prediksi kontinu biasanya lebih baik untuk RMSE
# ```
#
# **R4 — Harga / hukuman (≥ 0, sangat skewed)**
# ```python
# CFG.TARGET_TRANSFORM = "log1p"; CFG.REG_LOSS = "huber"; CFG.MIN_TARGET = 0
# CFG.METRIC = "rmsle"      # kalau metric resmi RMSLE
# CFG.REMOVE_NUMBERS_CLASSICAL = False   # angka (luas, kamar) adalah sinyal
# ```
#
# **R5 — Dokumen panjang (putusan hukum, artikel)**
# ```python
# CFG.LONG_TEXT_STRATEGY = "chunk"; CFG.MAX_LEN = 512; CFG.MAX_CHUNKS_PER_DOC = 6
# CFG.TRANSFORMER_MODELS = ["indobert_base"]
# # atau ekstrak bagian penting dulu (mis. paragraf "menjatuhkan pidana") lewat kolom baru → CFG.TEXT_COL
# ```
#
# **R6 — Esai English (skor)**
# ```python
# CFG.TRANSFORMER_MODELS = ["deberta_v3_base"]; CFG.MAX_LEN = 512; CFG.LLRD = 0.9
# CFG.POOLING = "mean"; CFG.BACKBONE_DROPOUT = 0.0; CFG.EPOCHS = 4
# ```
#
# **R7 — Bahasa Indonesia sosmed**
# ```python
# CFG.TRANSFORMER_MODELS = ["indobertweet"]; CFG.MAX_LEN = 128; CFG.TRANSFORMER_LOWERCASE = True
# ```
#
# **R8 — GPU kecil (4–6 GB)**
# ```python
# CFG.BATCH_SIZE = 8; CFG.GRAD_ACCUM_STEPS = 2; CFG.USE_AMP = True; CFG.GRADIENT_CHECKPOINTING = True
# ```
#
# **R9 — Data kecil (< 2k baris)**
# ```python
# CFG.N_FOLDS = 10; CFG.EMBEDDING_MODELS = ["intfloat/multilingual-e5-base"]; CFG.EMBEDDING_HEADS = ["ridge", "svr"]
# ```
#
# **R10 — Cek cepat satu fold transformer**
# ```python
# CFG.FOLDS_TO_RUN = [0]; CFG.EPOCHS = 1
# ```

# %% [markdown]
# ## 27. Troubleshooting
#
# | Error / gejala | Penyebab | Solusi |
# |---|---|---|
# | `FileNotFoundError` | path salah | cek `os.getcwd()`, path absolut |
# | `KeyError` kolom | nama kolom beda | set `TEXT_COL` / `TARGET_COL` di USER OVERRIDE |
# | Banyak baris dibuang "target tidak bisa di-parse" | format angka aneh | cek contoh nilai target; ubah `parse_numeric` |
# | `UnicodeDecodeError` / kolom tergabung | encoding / separator | `smart_read` menangani; kalau gagal, simpan ulang CSV UTF-8 |
# | `CUDA out of memory` | batch/MAX_LEN besar | auto-retry; manual `BATCH_SIZE=8`, `MAX_LEN=256`, `GRADIENT_CHECKPOINTING=True` |
# | Loss `nan` | fp16 overflow / LR besar / target tidak distandarisasi | `USE_AMP=False`, `LR=1e-5` (target sudah otomatis distandarisasi) |
# | Prediksi transformer hampir konstan | LR terlalu kecil/besar, epoch kurang, dropout | `LR=3e-5`, `HEAD_LR=1e-3`, `EPOCHS=4`, `BACKBONE_DROPOUT=0.0` |
# | Tokenizer error `sentencepiece` / `tiktoken` | DeBERTa-v3 / XLM-R / mDeBERTa | `pip install sentencepiece protobuf`, restart kernel |
# | `OSError` / `ConnectionError` saat load model | HF Hub tidak bisa diakses | path folder lokal; `HF_HUB_OFFLINE=1` kalau sudah di cache |
# | Warning `Some weights ... newly initialized` | head regresi baru | **normal** |
# | `ImportError: ... accelerate` | HF Trainer tanpa accelerate | otomatis pakai `custom_loop`; atau `pip install accelerate` |
# | `evaluation_strategy` / `tokenizer` TypeError | versi transformers | sudah di-shim; gunakan `custom_loop` |
# | Overflow / prediksi sangat besar (`inf`) | log1p + prediksi ekstrem | inverse transform sudah di-clip; cek `MAX_TARGET` |
# | RMSE bagus tapi QWK jelek | prediksi kontinu tidak dibulatkan | `METRIC="qwk"` + `OPTIMIZE_ROUNDING=True` |
# | Rounding memperburuk skor | metric kontinu (RMSE/MAE) | `ROUND_TO_VALID=False` |
# | Skor fold sangat bervariasi | outlier terkumpul di satu fold / data kecil | `TARGET_BINS` (sudah binned), `REG_LOSS="huber"`, lebih banyak fold |
# | CV bagus, leaderboard jelek | leakage / drift / duplikat | adversarial validation, `GROUP_COL="dup_group"`, kurangi tuning |
# | `MemoryError` TF-IDF | fitur char terlalu banyak | turunkan `TFIDF_CHAR_MAX_FEATURES` / `USE_CHAR_TFIDF=False` |
# | SVR sangat lambat | data besar (O(n²)) | hapus `"svr"` dari `EMBEDDING_HEADS` |
# | `ValueError: Input contains NaN` | prediksi NaN dari model gagal | cek `FAILED_MODELS`; NaN otomatis diisi median saat skor |
# | Submission ditolak (kolom/baris) | format beda dengan sample | lihat validator; set `SUBMISSION_TARGET_COL` |
# | Submission harus integer | target diskrit | rounding dipilih otomatis bila membantu; atau `ROUND_TO_VALID=True` |
# | `pip install` gagal | proxy/offline | `CFG.AUTO_INSTALL=False` |
# | `DataLoader worker exited` (Windows) | multiprocessing | `NUM_WORKERS=0` |
# | Kernel mati | RAM habis | turunkan `NEAR_DUP_MAX_ROWS`, fitur TF-IDF, `EMB_BATCH_SIZE` |

# %% [markdown]
# ## 28. Checklist sebelum submit
#
# - [ ] `RUN_MODE = "full"` dan semua cell jalan tanpa error dari atas.
# - [ ] `METRIC` = metric resmi lomba; `TARGET_TRANSFORM` sesuai (cek histogram target).
# - [ ] Skor OOF jauh lebih baik dari baseline "predict mean"; fold std kecil.
# - [ ] Tidak ada model penting di `FAILED_MODELS`; tidak ada tanda leakage.
# - [ ] Post-processing (clip/rounding) dipilih berdasarkan gain cross-validated.
# - [ ] Validator submission semua ✅ (baris, kolom, urutan ID, tanpa NaN, dalam rentang).
# - [ ] Distribusi prediksi test masuk akal dibanding target train (mean/std/min/max).
# - [ ] File `submission.csv` terbaru yang di-upload.
# - [ ] Figure laporan ada di `figures/` (`CFG.FIG_DPI = 300` untuk final); `cfg_used.json` dicatat di laporan.
