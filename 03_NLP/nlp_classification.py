# -*- coding: utf-8 -*-
# =====================================================================================================
# nlp_classification.py — versi SCRIPT dari nlp_classification.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python nlp_classification.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb nlp_classification.py
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
# # NLP Text Classification — Competition Template (Binary / Multiclass / Multilabel)
#
# Template ini dibuat supaya kamu bisa **jalan sendirian dari nol sampai submit** tanpa bantuan siapa pun:
# kode lengkap + panduan cara membaca output + aturan "Jika X maka Y" + catatan untuk laporan.
#
# **Task yang didukung**
#
# | Task | Contoh | Output model |
# |---|---|---|
# | `binary` | sentimen positif/negatif, hoax/bukan, spam/bukan | 1 probabilitas kelas positif |
# | `multiclass` | topik berita, emosi (marah/sedih/senang/...), kategori keluhan | probabilitas per kelas (jumlah = 1) |
# | `multilabel` | aspek ulasan (harga & pengiriman sekaligus), tag artikel | probabilitas independen per label |
#
# ## Pipeline besar
#
# ```text
# CFG + USER OVERRIDE
# → install/cek library  → load data (smart_read)  → encode target  → sanity check
# → text cleaning (2 versi: untuk transformer & untuk TF-IDF)  → Text EDA lengkap
#   (panjang teks + rekomendasi MAX_LEN, bahasa, n-gram per kelas, duplikat, drift train-test)
# → K-Fold (stratified / group)
# → [A] Classical: TF-IDF word+char  → LogReg / LinearSVC / ComplementNB / SGD / Ridge / NB-SVM / LightGBM(SVD)
# → [B] Sentence-embedding (frozen) → LogReg / LightGBM
# → [C] Transformer fine-tuning K-Fold (custom loop atau HF Trainer, long-text strategy, OOM auto-retry)
# → (opsional) pseudo-labeling
# → perbandingan model → ensemble (mean / weighted-optimized / rank / stacking)
# → threshold / bias tuning → error analysis → interpretability (top n-gram, occlusion)
# → submission + validator → simpan OOF/test/log → ringkasan untuk laporan
# ```
#
# ## Cara pakai cepat (5 langkah)
#
# 1. Taruh data di `./data/train.csv`, `./data/test.csv`, `./data/sample_submission.csv` (atau ubah path).
# 2. Di cell **USER OVERRIDE**: isi `ID_COL`, `TEXT_COL`, `TARGET_COL`, dan `METRIC` sesuai aturan lomba.
# 3. Jalankan dulu dengan `CFG.RUN_MODE = "fast"` (subsample, fold sedikit) → pastikan semua cell jalan.
# 4. Baca hasil EDA (MAX_LEN rekomendasi, bahasa → rekomendasi model), sesuaikan CFG.
# 5. Ganti ke `CFG.RUN_MODE = "full"` → Run All → ambil `outputs/.../submissions/submission.csv`.
#
# ## Pilih model transformer (ringkas)
#
# | Kondisi data | Model key (di `MODEL_POOL`) | Catatan |
# |---|---|---|
# | Bahasa Indonesia formal (berita, dokumen) | `indobert_base` | default kuat untuk ID |
# | Bahasa Indonesia sosmed / slang / alay | `indobertweet` | dilatih di tweet ID |
# | Campur ID-EN (code-mix) / multibahasa | `xlm_roberta_base` | butuh `sentencepiece` |
# | English | `deberta_v3_base` (terkuat) / `modernbert_base` (cepat, konteks panjang) | DeBERTa butuh `sentencepiece` |
# | Tidak ada GPU | **jangan fine-tune**; pakai Classical + Sentence-embedding | tetap kompetitif |
#
# > **Prinsip penting:** selalu percaya **CV (OOF score)**, bukan feeling. Semua keputusan
# > (cleaning, model, threshold, ensemble) diambil berdasarkan OOF.

# %% [markdown]
# ## 0. Catatan platform & hardware
#
# - Notebook ini **self-contained**: tidak butuh file helper lain. Jalan di Jupyter / VSCode lokal.
# - **Tanpa GPU**: Classical (TF-IDF) dan Sentence-embedding tetap jalan cepat di CPU. Fine-tuning transformer
#   di CPU sangat lambat (bisa berjam-jam) → default `RUN_TRANSFORMER = "auto"` = hanya jalan kalau ada CUDA.
# - **Dengan GPU kecil (4–6 GB VRAM)**: pakai model `base`, `BATCH_SIZE=8`, `MAX_LEN<=256`, `USE_AMP=True`,
#   kalau masih OOM → `GRADIENT_CHECKPOINTING=True`. Notebook juga otomatis mengulang dengan batch size setengah
#   kalau kena CUDA OOM.
# - **Internet diblokir?** Model HuggingFace harus sudah ter-download di cache (`~/.cache/huggingface/hub`)
#   atau folder lokal; isi `MODEL_POOL` dengan path folder lokal. Lihat tabel Troubleshooting di akhir.
# - Semua output tersimpan di `CFG.OUTPUT_DIR` (`figures/`, `tables/`, `models/`, `submissions/`, `preds/`).

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

NOTEBOOK_NAME = "nlp_classification"
print("Python:", sys.version.split()[0], "| numpy:", np.__version__, "| pandas:", pd.__version__)

# %% [markdown]
# ## 1. CFG — pusat kontrol notebook
#
# Semua "knob" ada di sini. **Jangan edit langsung di class ini saat lomba** — ubah lewat cell
# `USER OVERRIDE` di bawahnya supaya default tetap utuh dan perubahanmu terlihat jelas di satu tempat.
#
# Yang hampir pasti kamu ubah: `TRAIN_PATH/TEST_PATH/SAMPLE_SUB_PATH`, `ID_COL`, `TEXT_COL`, `TARGET_COL`,
# `METRIC`, `SUBMISSION_MODE`, `RUN_MODE`, `TRANSFORMER_MODELS`.
#
# **Cara membaca komentar:** tiap knob punya penjelasan *apa fungsinya*, *kapan diubah*, dan *nilai umum*.

# %%
class CFG:
    # ======================================================
    # RUN MODE
    # ======================================================
    # "fast": debugging cepat → subsample data, fold & epoch sedikit, model classical lebih sedikit.
    # "full": run final untuk submission. Selalu cek "fast" dulu sampai semua cell lolos.
    RUN_MODE = "full"
    FAST_SAMPLE_ROWS = 3000        # jumlah baris train saat fast (stratified subsample)
    FAST_N_FOLDS = 3
    FAST_EPOCHS = 1
    SEED = 42

    # Install otomatis package yang belum ada (pip). Torch TIDAK pernah di-install otomatis.
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"  # (env TEMPLATE_AUTO_INSTALL=0 → off)
    INSTALL_OPTIONAL = True        # ikut install yang opsional (lightgbm, sentence-transformers, dll.)

    # ======================================================
    # PATHS
    # ======================================================
    TRAIN_PATH = "./data/train.csv"
    TEST_PATH = "./data/test.csv"              # kalau tidak ada → notebook tetap jalan (CV saja)
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    OUTPUT_DIR = f"./outputs/{NOTEBOOK_NAME}/"

    # ======================================================
    # KOLOM
    # ======================================================
    ID_COL = "id"                  # kalau tidak ada kolom id, biarkan; submission pakai urutan baris
    TEXT_COL = "text"              # kolom teks utama
    EXTRA_TEXT_COLS = []           # kolom teks tambahan yang digabung, contoh ["title"] → "title | text"
    TARGET_COL = "label"           # kolom target (string atau angka, akan di-encode otomatis)
    META_COLS = []                 # metadata kategorikal (mis. ["platform", "keyword"]) → disisipkan ke teks
    USE_METADATA_AS_TEXT = True    # True: "[platform: shopee] teks..." (berguna untuk transformer & TF-IDF)
    GROUP_COL = None               # mis. "user_id"/"article_id" → GroupKFold supaya tidak bocor antar fold
    SUBMISSION_TARGET_COL = None   # None = otomatis (pakai kolom di sample_submission / TARGET_COL)

    # ======================================================
    # TASK & METRIC
    # ======================================================
    # "auto" → binary kalau 2 kelas, multiclass kalau >2, multilabel kalau TARGET_COLS / MULTILABEL_SEP diisi.
    TASK_TYPE = "auto"             # "auto" | "binary" | "multiclass" | "multilabel"
    TARGET_COLS = []               # multilabel format kolom 0/1 terpisah, mis. ["harga", "kualitas"]
    MULTILABEL_SEP = None          # multilabel format 1 kolom, mis. "harga|kualitas" → MULTILABEL_SEP = "|"
    POSITIVE_LABEL = None          # binary: label mana yang "positif" (mis. "positif", 1). None = kelas ke-2 urut abjad
    CLASS_ORDER = None             # urutan kelas manual (penting untuk QWK / label ordinal), mis. ["low","mid","high"]

    # METRIC sesuai aturan lomba:
    #   binary     : "f1", "accuracy", "roc_auc", "log_loss", "mcc", "balanced_accuracy", "pr_auc", "precision", "recall"
    #   multiclass : "f1_macro", "f1_micro", "f1_weighted", "accuracy", "log_loss", "roc_auc_ovr", "qwk", "mcc"
    #   multilabel : "f1_micro", "f1_macro", "f1_samples", "roc_auc_macro", "hamming_loss"
    #   "auto"     : binary→f1, multiclass→f1_macro, multilabel→f1_micro
    METRIC = "auto"
    # "label" = submit kelas akhir; "probability" = submit probabilitas (biasanya untuk AUC/logloss)
    SUBMISSION_MODE = "label"

    # ======================================================
    # CROSS-VALIDATION
    # ======================================================
    N_FOLDS = 5                    # 5 standar; 3 kalau data besar/waktu sempit; 10 kalau data kecil (<2k)
    CV_STRATEGY = "auto"           # "auto" | "stratified" | "group" | "stratified_group" | "kfold"

    # ======================================================
    # TEXT CLEANING
    # ======================================================
    # Transformer: cleaning MINIMAL (tokenizer sudah pintar, kata slang/emoji bisa jadi sinyal).
    # TF-IDF     : normalisasi lebih banyak membantu (lowercase, slang → baku, elongation, dll.).
    NORMALIZE_URL = True           # "http://..." → "<URL>"
    NORMALIZE_USER = True          # "@akun" → "<USER>"
    KEEP_HASHTAG_TEXT = True       # "#promo" → "promo"
    REMOVE_EMOJI = False           # emoji sering sinyal sentimen → default dibiarkan
    TRANSFORMER_LOWERCASE = False  # model uncased (indobertweet, indolem) sudah lowercase sendiri
    TRANSFORMER_USE_SLANG = False  # normalisasi slang juga untuk transformer (biasanya tidak perlu)
    CLASSICAL_LOWERCASE = True
    USE_SLANG_NORMALIZATION = True # "gak"→"tidak", "yg"→"yang", "bgt"→"banget" (untuk TF-IDF)
    CUSTOM_SLANG_DICT = {}         # tambahan kamus slang khusus lomba, mis. {"bgsd": "bagus"}
    REDUCE_ELONGATION = True       # "mantappp" → "mantap", "bagus2" → "bagus bagus"
    REMOVE_STOPWORDS_CLASSICAL = False  # stopword removal: kadang menurunkan skor (kata "tidak" penting!)
    USE_STEMMING = False           # butuh Sastrawi (opsional); lambat; jarang membantu untuk klasifikasi
    REMOVE_NUMBERS_CLASSICAL = False

    # ======================================================
    # EDA
    # ======================================================
    RUN_EDA = True
    EDA_TOKENIZER_MODEL = None     # None = tokenizer dari model transformer pertama; bisa nama HF lain
    MAX_LEN_PERCENTILE = 95        # MAX_LEN rekomendasi = persentil panjang token ini
    TOP_NGRAMS = 15
    NEAR_DUP_THRESHOLD = 0.90      # cosine similarity char-TF-IDF ≥ ini dianggap near-duplicate
    NEAR_DUP_MAX_ROWS = 20000      # batasi baris untuk cek near-duplicate (biar tidak lama)
    RUN_ADVERSARIAL_VALIDATION = True
    MAKE_WORDCLOUD = True          # hanya jalan kalau package wordcloud terpasang

    # ======================================================
    # [A] CLASSICAL (TF-IDF)
    # ======================================================
    RUN_CLASSICAL = True
    # pilihan: "logreg", "linearsvc", "complement_nb", "sgd", "ridge", "nbsvm", "lgbm_svd"
    CLASSICAL_MODELS = ["logreg", "linearsvc", "complement_nb", "sgd", "ridge", "nbsvm", "lgbm_svd"]
    TFIDF_WORD_NGRAM = (1, 2)
    TFIDF_CHAR_NGRAM = (2, 5)
    USE_CHAR_TFIDF = True          # char n-gram tahan typo/slang → sangat berguna untuk teks Indonesia sosmed
    TFIDF_WORD_MAX_FEATURES = 200_000
    TFIDF_CHAR_MAX_FEATURES = 300_000
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.95
    LOGREG_C = 4.0                 # makin besar = regularisasi makin lemah (coba 1, 4, 10)
    SVD_COMPONENTS = 256           # untuk lgbm_svd
    LGBM_N_ESTIMATORS = 2000       # dengan early stopping internal (inner holdout di fold train)

    # ======================================================
    # [B] SENTENCE EMBEDDING (frozen) + model ringan
    # ======================================================
    RUN_EMBEDDINGS = True
    # Rekomendasi: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (cepat, multibahasa),
    # "intfloat/multilingual-e5-small" / "-base" (kuat; WAJIB prefix "query: " → otomatis ditangani).
    EMBEDDING_MODELS = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
    EMBEDDING_PREFIXES = {}        # override prefix manual, mis. {"intfloat/multilingual-e5-base": "query: "}
    EMBEDDING_HEADS = ["logreg", "lgbm"]
    EMB_BATCH_SIZE = 64
    EMB_MAX_LEN = 256
    CACHE_EMBEDDINGS = True        # simpan ke disk → run berikutnya tidak perlu encode ulang

    # ======================================================
    # [C] TRANSFORMER FINE-TUNING
    # ======================================================
    RUN_TRANSFORMER = "auto"       # "auto" = hanya kalau CUDA tersedia; True = paksa (CPU lambat!); False = skip
    TRAINER_BACKEND = "custom_loop"  # "custom_loop" (PyTorch murni, paling tahan beda versi) | "hf_trainer"
    MODEL_POOL = {
        # English
        "deberta_v3_base": "microsoft/deberta-v3-base",
        "deberta_v3_large": "microsoft/deberta-v3-large",
        "roberta_base": "FacebookAI/roberta-base",
        "modernbert_base": "answerdotai/ModernBERT-base",
        "modernbert_large": "answerdotai/ModernBERT-large",
        # Indonesia
        "indobert_base": "indobenchmark/indobert-base-p2",
        "indobert_large": "indobenchmark/indobert-large-p2",
        "indolem_indobert": "indolem/indobert-base-uncased",
        "indobertweet": "indolem/indobertweet-base-uncased",
        # Multilingual / code-mix
        "xlm_roberta_base": "FacebookAI/xlm-roberta-base",
        "xlm_roberta_large": "FacebookAI/xlm-roberta-large",
        "mdeberta_v3_base": "microsoft/mdeberta-v3-base",
    }
    # ["auto"] = dipilih otomatis dari deteksi bahasa di EDA. Bisa key MODEL_POOL atau nama HF/path lokal.
    TRANSFORMER_MODELS = ["auto"]
    MAX_LEN = "auto"               # "auto" = rekomendasi EDA (persentil token), atau angka: 128 / 256 / 512
    LONG_TEXT_STRATEGY = "head_tail"  # "truncate" | "head_tail" (awal+akhir) | "chunk" (potong & rata-rata)
    HEAD_RATIO = 0.5               # head_tail: porsi token dari awal teks (sisanya dari akhir)
    CHUNK_OVERLAP = 64             # chunk: token overlap antar potongan
    CHUNK_POOLING = "mean"         # chunk: gabung logits potongan dengan "mean" atau "max"
    MAX_CHUNKS_PER_DOC = 8
    BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 64
    GRAD_ACCUM_STEPS = 1           # batch efektif = BATCH_SIZE * GRAD_ACCUM_STEPS
    EPOCHS = 3
    LR = 2e-5                      # backbone LR (1e-5 large, 2e-5~3e-5 base)
    HEAD_LR = 1e-4                 # LR untuk classification head (lebih besar dari backbone)
    LLRD = 1.0                     # layer-wise LR decay: 1.0 = off; 0.9 = layer bawah LR lebih kecil
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    SCHEDULER = "linear"           # "linear" | "cosine"
    MAX_GRAD_NORM = 1.0            # gradient clipping
    USE_AMP = True                 # mixed precision (hanya aktif jika CUDA). NaN loss? → set False
    AMP_DTYPE = "fp16"             # "fp16" | "bf16" (bf16 hanya GPU Ampere+)
    GRADIENT_CHECKPOINTING = False # hemat VRAM ~30-50%, training ~20-30% lebih lambat
    POOLING = "cls"                # "cls" | "mean" (mean pooling sering lebih stabil)
    MULTI_SAMPLE_DROPOUT = 0       # 0 = off; 5 = rata-rata 5 dropout head (stabilkan training)
    HEAD_DROPOUT = 0.1
    LOSS = "ce"                    # "ce" | "focal" (focal untuk imbalance berat)
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.0          # 0.05–0.1 bisa bantu kalau label noisy
    USE_CLASS_WEIGHTS = False      # True untuk imbalance (bobot balanced)
    EARLY_STOPPING_PATIENCE = 2    # stop kalau skor valid tidak membaik N evaluasi berturut-turut
    EVALS_PER_EPOCH = 1            # 2 = evaluasi tiap setengah epoch (untuk dataset besar)
    OOM_MAX_RETRIES = 3            # CUDA OOM → ulang dengan batch size setengah
    SAVE_FOLD_MODELS = False       # simpan bobot tiap fold (.pt) — besar! aktifkan kalau perlu inference ulang
    FOLDS_TO_RUN = None            # None = semua fold; [0] = hanya fold 0 (eksperimen cepat)
    NUM_WORKERS = 0                # DataLoader workers (Windows: 0 paling aman)
    DEBUG_SIMULATE_OOM = False     # hanya untuk testing alur OOM-retry

    # ======================================================
    # PSEUDO-LABELING (opsional, hati-hati)
    # ======================================================
    RUN_PSEUDO_LABEL = False
    PSEUDO_CONF_THRESHOLD = 0.95   # ambil prediksi test dengan confidence ≥ ini
    PSEUDO_MAX_FRACTION = 0.5      # maksimal pseudo rows = fraksi ini × jumlah train

    # ======================================================
    # ENSEMBLE, THRESHOLD, OUTPUT
    # ======================================================
    RUN_ENSEMBLE = True
    ENSEMBLE_METHODS = ["mean", "weighted", "rank", "stacking"]
    ENSEMBLE_TOP_K = 6             # hanya top-K model terbaik (OOF) yang ikut ensemble
    MANUAL_WEIGHTS = {}            # mis. {"tfidf_logreg": 0.3, "tf_indobert_base": 0.7}
    FINAL_PREDICTION = "auto"      # "auto" = skor OOF terbaik; atau nama model/ensemble tertentu
    TUNE_THRESHOLD = True          # binary: threshold; multiclass: bias per kelas; multilabel: threshold per label
    MIN_TUNING_GAIN = 0.002        # tuning hanya dipakai kalau gain OOF ≥ ini (hindari overfit)
    RUN_INTERPRETABILITY = True
    N_EXPLAIN_EXAMPLES = 3
    USE_SHAP = True                # hanya jika package shap ada
    FIG_DPI = 200                  # 300 untuk figure final laporan
    SHOW_FIGS = True

# %% [markdown]
# ## 2. USER OVERRIDE — ubah CFG di sini
#
# Tulis semua perubahan untuk lomba di cell ini. Contoh:
#
# ```python
# CFG.TRAIN_PATH = "./data/train.csv"
# CFG.TEXT_COL = "tweet"
# CFG.TARGET_COL = "sentiment"
# CFG.METRIC = "f1_macro"
# CFG.RUN_MODE = "fast"          # ganti "full" untuk run final
# CFG.TRANSFORMER_MODELS = ["indobertweet"]
# ```
#
# **Jika** lomba minta probabilitas → `CFG.SUBMISSION_MODE = "probability"`.
# **Jika** ada kolom user/sumber yang berulang → `CFG.GROUP_COL = "user_id"`.
# **Jika** laptop tanpa GPU → biarkan `RUN_TRANSFORMER = "auto"` (otomatis skip fine-tuning).

# %%
# >>> USER OVERRIDE
# Contoh (hapus tanda # untuk mengaktifkan):
# CFG.TRAIN_PATH = "./data/train.csv"
# CFG.TEST_PATH = "./data/test.csv"
# CFG.SAMPLE_SUB_PATH = "./data/sample_submission.csv"
# CFG.ID_COL = "id"
# CFG.TEXT_COL = "text"
# CFG.TARGET_COL = "label"
# CFG.METRIC = "f1_macro"
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
# **Tujuan:** membaca train/test/sample_submission dan memastikan kolom yang disebut di CFG benar-benar ada.
#
# **Cara membaca output:**
# - `Train shape` / `Test shape`: jumlah baris & kolom. Test **tidak ada** → notebook tetap jalan (hanya CV),
#   submission di-skip dengan pesan.
# - Kalau `TEXT_COL` tidak ditemukan, notebook memilih otomatis kolom teks dengan rata-rata panjang terbesar
#   dan memberi ⚠️ — **cek apakah pilihannya benar**, lalu set `CFG.TEXT_COL` di USER OVERRIDE.
# - Kalau `TARGET_COL` tidak ditemukan, notebook menebak kolom yang ada di train tapi tidak ada di test.
#
# **Error umum:** `FileNotFoundError` → path salah (cek `os.getcwd()` dan struktur folder);
# `UnicodeDecodeError` sudah ditangani otomatis (utf-8 → utf-8-sig → latin-1);
# kolom tergabung jadi satu (mis. `"id;text;label"`) → separator `;` otomatis terdeteksi.

# %%
print("Working dir:", os.getcwd())
train_raw = smart_read(CFG.TRAIN_PATH)
HAS_TEST = bool(CFG.TEST_PATH) and os.path.exists(str(CFG.TEST_PATH))
test_raw = smart_read(CFG.TEST_PATH) if HAS_TEST else None
sample_sub = smart_read(CFG.SAMPLE_SUB_PATH) if (CFG.SAMPLE_SUB_PATH and os.path.exists(str(CFG.SAMPLE_SUB_PATH))) else None

print("Train shape:", train_raw.shape)
print("Test shape :", test_raw.shape if HAS_TEST else "TIDAK ADA → hanya CV, submission di-skip")
print("Sample sub :", sample_sub.shape if sample_sub is not None else "tidak ada (submission dibuat dari test + ID_COL)")


def guess_text_col(df):
    """Pilih kolom object dengan rata-rata panjang string terbesar."""
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return None
    lens = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
    return max(lens, key=lens.get)


if CFG.TEXT_COL not in train_raw.columns:
    guess = guess_text_col(train_raw)
    print(f"⚠️ TEXT_COL '{CFG.TEXT_COL}' tidak ada. Kolom tersedia: {list(train_raw.columns)}")
    print(f"   → memakai tebakan otomatis TEXT_COL = '{guess}'. Kalau salah, set CFG.TEXT_COL di USER OVERRIDE.")
    CFG.TEXT_COL = guess

if not CFG.TARGET_COLS and CFG.TARGET_COL not in train_raw.columns:
    cand = [c for c in train_raw.columns if HAS_TEST and c not in test_raw.columns and c != CFG.ID_COL]
    if len(cand) == 1:
        print(f"⚠️ TARGET_COL '{CFG.TARGET_COL}' tidak ada → memakai '{cand[0]}' (ada di train, tidak ada di test).")
        CFG.TARGET_COL = cand[0]
    else:
        raise KeyError(f"TARGET_COL '{CFG.TARGET_COL}' tidak ditemukan. Kolom train: {list(train_raw.columns)}")

missing_extra = [c for c in CFG.EXTRA_TEXT_COLS + CFG.META_COLS if c not in train_raw.columns]
if missing_extra:
    print(f"⚠️ Kolom EXTRA/META tidak ada di train dan diabaikan: {missing_extra}")
    CFG.EXTRA_TEXT_COLS = [c for c in CFG.EXTRA_TEXT_COLS if c in train_raw.columns]
    CFG.META_COLS = [c for c in CFG.META_COLS if c in train_raw.columns]
if CFG.ID_COL not in train_raw.columns:
    print(f"ℹ️ ID_COL '{CFG.ID_COL}' tidak ada di train (tidak masalah; OOF disimpan dengan index baris).")

display(train_raw.head())
if HAS_TEST:
    display(test_raw.head())

# %% [markdown]
# ## 6. Encode target & deteksi task
#
# **Tujuan:** mengubah label (string/angka) menjadi indeks `0..K-1`, menentukan task (binary/multiclass/multilabel)
# dan metric default.
#
# - Binary: kelas positif (indeks 1) = `CFG.POSITIVE_LABEL` kalau diisi, kalau tidak = kelas ke-2 setelah diurutkan.
#   **Penting untuk F1 binary**: F1 dihitung untuk kelas positif. Pastikan kelas positif = kelas yang diminta lomba
#   (mis. "hoax", "spam", "1").
# - Multiclass ordinal (mis. rating "buruk/sedang/baik") + metric QWK → isi `CFG.CLASS_ORDER` supaya urutannya benar.
# - Multilabel: isi `CFG.TARGET_COLS` (kolom 0/1) **atau** `CFG.MULTILABEL_SEP` (satu kolom "a|b|c").
#
# **Cara membaca output:** distribusi kelas & `imbalance ratio` (kelas terbesar / terkecil).
# - **Jika** ratio > 5 → data imbalance: pakai metric F1/macro-F1, `USE_CLASS_WEIGHTS=True` atau `LOSS="focal"`,
#   dan threshold tuning wajib.
# - **Jika** ada kelas dengan < `N_FOLDS` sampel → stratified split akan warning; pertimbangkan gabung kelas langka.
#
# **Untuk laporan:** tampilkan figure distribusi kelas + sebutkan imbalance ratio.

# %%
train = train_raw.copy()
test = test_raw.copy() if HAS_TEST else None

if str(CFG.TASK_TYPE).lower() == "auto":
    if CFG.TARGET_COLS or CFG.MULTILABEL_SEP:
        TASK = "multilabel"
    else:
        _n = train[CFG.TARGET_COL].dropna().nunique()
        TASK = "binary" if _n == 2 else "multiclass"
else:
    TASK = str(CFG.TASK_TYPE).lower()
assert TASK in ("binary", "multiclass", "multilabel"), f"TASK_TYPE tidak dikenal: {TASK}"

if TASK == "multilabel":
    if CFG.TARGET_COLS:
        CLASSES = list(CFG.TARGET_COLS)
        Y = train[CLASSES].fillna(0).astype(int).clip(0, 1).values
    else:
        sep = CFG.MULTILABEL_SEP or "|"
        _lists = train[CFG.TARGET_COL].fillna("").astype(str).apply(
            lambda s: [t.strip() for t in s.split(sep) if t.strip()])
        CLASSES = sorted({t for lst in _lists for t in lst})
        Y = np.array([[int(c in set(lst)) for c in CLASSES] for lst in _lists], dtype=int)
    NUM_CLASSES = len(CLASSES)
else:
    n_before = len(train)
    train = train[train[CFG.TARGET_COL].notna()].reset_index(drop=True)
    if len(train) < n_before:
        print(f"⚠️ {n_before - len(train)} baris dengan target kosong dibuang.")
    raw_target = train[CFG.TARGET_COL]
    if CFG.CLASS_ORDER:
        CLASSES = list(CFG.CLASS_ORDER)
        unknown = set(raw_target.unique()) - set(CLASSES)
        assert not unknown, f"Label tidak ada di CLASS_ORDER: {unknown}"
    else:
        CLASSES = sorted(raw_target.unique().tolist(), key=lambda v: (str(type(v)), v))
    if TASK == "binary" and CFG.POSITIVE_LABEL is not None:
        assert CFG.POSITIVE_LABEL in CLASSES, f"POSITIVE_LABEL {CFG.POSITIVE_LABEL} tidak ada di {CLASSES}"
        CLASSES = [c for c in CLASSES if c != CFG.POSITIVE_LABEL] + [CFG.POSITIVE_LABEL]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    Y = raw_target.map(CLASS_TO_IDX).astype(int).values
    NUM_CLASSES = len(CLASSES)
    if TASK == "binary" and NUM_CLASSES != 2:
        raise ValueError(f"TASK_TYPE binary tapi jumlah kelas = {NUM_CLASSES}. Set CFG.TASK_TYPE='multiclass'.")

if CFG.METRIC == "auto":
    CFG.METRIC = {"binary": "f1", "multiclass": "f1_macro", "multilabel": "f1_micro"}[TASK]
    print(f"ℹ️ METRIC auto → '{CFG.METRIC}'. GANTI sesuai metric resmi lomba di USER OVERRIDE!")


def decode_labels(idx):
    """Indeks → label asli."""
    return np.array(CLASSES, dtype=object)[np.asarray(idx, dtype=int)]


print(f"TASK = {TASK} | NUM_CLASSES = {NUM_CLASSES} | METRIC = {CFG.METRIC}")
print("Classes (index → label):", {i: c for i, c in enumerate(CLASSES)})
if TASK == "binary":
    print(f"Kelas POSITIF (index 1) = {CLASSES[1]!r}  ← pastikan ini kelas yang dimaksud metric F1/AUC lomba")

# %% [markdown]
# ### 6.1 Subsample untuk RUN_MODE="fast"
#
# Di mode fast, train diambil subsample stratified sebanyak `FAST_SAMPLE_ROWS` supaya seluruh notebook selesai
# cepat. **Skor di mode fast tidak mewakili skor final** — gunakan hanya untuk memastikan kode jalan.

# %%
if CFG.RUN_MODE == "fast" and len(train) > CFG.FAST_SAMPLE_ROWS:
    rs = np.random.RandomState(CFG.SEED)
    if TASK == "multilabel":
        keep = rs.choice(len(train), CFG.FAST_SAMPLE_ROWS, replace=False)
    else:
        keep = []
        frac = CFG.FAST_SAMPLE_ROWS / len(train)
        for c in np.unique(Y):
            idx = np.where(Y == c)[0]
            k = max(min(len(idx), CFG.N_FOLDS), int(round(len(idx) * frac)))
            keep.extend(rs.choice(idx, k, replace=False))
        keep = np.array(sorted(keep))
    train = train.iloc[keep].reset_index(drop=True)
    Y = Y[keep]
    print(f"FAST mode: train di-subsample → {len(train)} baris")
train = train.reset_index(drop=True)

# %% [markdown]
# ## 7. Inspeksi dasar & sanity check
#
# **Apa yang dicek:**
# 1. Tipe data & missing value per kolom (train dan test).
# 2. Teks kosong (setelah strip) — model akan melihat string kosong; terlalu banyak → cek parsing file.
# 3. ID unik? ID duplikat di train / ID test muncul di train → kemungkinan data bocor atau salah join.
# 4. Kolom train vs test berbeda (selain target).
# 5. **Leakage heuristik**: (a) ID berkorelasi dengan target (data terurut per kelas), (b) nama kelas tertulis
#    langsung di teks, (c) nilai metadata yang hampir selalu menentukan kelas.
#
# **Jika** leakage terdeteksi → jangan langsung senang. Pastikan fitur itu juga tersedia & valid di test. ID
# berkorelasi dengan target biasanya artefak urutan data → **jangan** dipakai sebagai fitur.

# %%
def build_raw_text(df):
    """Gabungkan TEXT_COL + EXTRA_TEXT_COLS menjadi satu string."""
    parts = [df[CFG.TEXT_COL].fillna("").astype(str)]
    for c in CFG.EXTRA_TEXT_COLS:
        if c in df.columns:
            parts.append(df[c].fillna("").astype(str))
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p, sep=" | ")
    return out


train["raw_text"] = build_raw_text(train)
if HAS_TEST:
    test["raw_text"] = build_raw_text(test)

info = pd.DataFrame({"dtype": train.dtypes.astype(str), "n_missing": train.isna().sum(),
                     "pct_missing": (train.isna().mean() * 100).round(2), "n_unique": train.nunique()})
print("Train column summary:")
display(info)
if HAS_TEST:
    print("Test missing values:")
    display(test.isna().sum().to_frame("n_missing").T)

checks = []
n_empty = int((train["raw_text"].str.strip() == "").sum())
checks.append(("empty texts (train)", n_empty, "⚠️ cek parsing / isi dengan ''" if n_empty else "✅"))
if HAS_TEST:
    n_empty_te = int((test["raw_text"].str.strip() == "").sum())
    checks.append(("empty texts (test)", n_empty_te, "⚠️" if n_empty_te else "✅"))
if CFG.ID_COL in train.columns:
    dup_id = int(train[CFG.ID_COL].duplicated().sum())
    checks.append(("duplicate IDs (train)", dup_id, "⚠️ ID tidak unik" if dup_id else "✅"))
    if HAS_TEST and CFG.ID_COL in test.columns:
        overlap = len(set(train[CFG.ID_COL]) & set(test[CFG.ID_COL]))
        checks.append(("test IDs also in train", overlap, "⚠️ cek join/leak" if overlap else "✅"))
if HAS_TEST:
    tcols = set(test.columns) - {"raw_text"}
    trcols = set(train.columns) - {"raw_text", CFG.TARGET_COL} - set(CFG.TARGET_COLS)
    only_tr, only_te = sorted(trcols - tcols), sorted(tcols - trcols)
    checks.append(("cols only in train", len(only_tr), str(only_tr[:10]) if only_tr else "✅"))
    checks.append(("cols only in test", len(only_te), str(only_te[:10]) if only_te else "✅"))
display(pd.DataFrame(checks, columns=["check", "value", "status"]))

# ---- leakage heuristics
leak_notes = []
if TASK != "multilabel":
    y_first = Y if TASK == "binary" else (Y == np.bincount(Y).argmax()).astype(int)
    if CFG.ID_COL in train.columns:
        id_num = pd.to_numeric(train[CFG.ID_COL].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        if id_num.notna().mean() > 0.9 and len(np.unique(y_first)) == 2:
            auc_id = skm.roc_auc_score(y_first, id_num.fillna(id_num.median()))
            auc_id = max(auc_id, 1 - auc_id)
            leak_notes.append(f"AUC(ID → target) = {auc_id:.3f}" + ("  ⚠️ ID berkorelasi dengan target (urutan data)!"
                                                                      if auc_id > 0.65 else "  ✅"))
    low = train["raw_text"].str.lower()
    for c in CLASSES[:20]:
        cs = str(c).lower()
        if len(cs) >= 3:
            rate = low.str.contains(re.escape(cs), regex=True).mean()
            if rate > 0.3:
                leak_notes.append(f"Label '{c}' tertulis di {rate:.0%} teks → cek apakah ini leakage")
    for mc in CFG.META_COLS:
        tab = pd.crosstab(train[mc].fillna("NA"), Y)
        purity = (tab.max(axis=1) / tab.sum(axis=1))
        big = tab.sum(axis=1) >= max(5, len(train) * 0.01)
        if big.any() and (purity[big] > 0.98).mean() > 0.5:
            leak_notes.append(f"META '{mc}': mayoritas nilai hampir murni satu kelas → sinyal sangat kuat/leak?")
print("Leakage heuristics:")
for n in leak_notes or ["✅ tidak ada tanda leakage yang jelas"]:
    print("  -", n)

# ---- class distribution
if TASK == "multilabel":
    dist = pd.DataFrame({"label": CLASSES, "count": Y.sum(0), "pct": (Y.mean(0) * 100).round(2)})
    print("Labels per row:", pd.Series(Y.sum(1)).value_counts().sort_index().to_dict())
else:
    cnt = np.bincount(Y, minlength=NUM_CLASSES)
    dist = pd.DataFrame({"label": CLASSES, "count": cnt, "pct": (cnt / cnt.sum() * 100).round(2)})
display(dist)
save_table(dist, "class_distribution")
IMBALANCE_RATIO = float(dist["count"].max() / max(dist["count"].min(), 1))
print(f"Imbalance ratio (max/min) = {IMBALANCE_RATIO:.2f}",
      "⚠️ imbalance → F1/macro-F1, class weights / focal, threshold tuning" if IMBALANCE_RATIO > 5 else "✅")
if (dist["count"] < CFG.N_FOLDS).any():
    print(f"⚠️ Ada kelas dengan < {CFG.N_FOLDS} sampel: {dist.loc[dist['count'] < CFG.N_FOLDS, 'label'].tolist()}")

fig, ax = plt.subplots(figsize=(max(6, 0.6 * NUM_CLASSES + 3), 4))
ax.bar(dist["label"].astype(str), dist["count"], color="#4C72B0")
for i, v in enumerate(dist["count"]):
    ax.text(i, v, f"{v}\n({dist['pct'].iloc[i]:.1f}%)", ha="center", va="bottom", fontsize=9)
ax.set_title("Class distribution (train)")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
plt.xticks(rotation=30 if NUM_CLASSES > 5 else 0, ha="right" if NUM_CLASSES > 5 else "center")
save_fig("class_distribution", fig)

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
# ### 10.1 Panjang teks (karakter / kata / token) + rekomendasi `MAX_LEN`
#
# **Tujuan:** menentukan `MAX_LEN` transformer. Token dihitung dengan tokenizer model yang akan dipakai
# (subword — biasanya 1 kata ≈ 1.2–1.6 token untuk Bahasa Indonesia; makin banyak slang makin banyak token).
#
# **Cara membaca output:**
# - Tabel `% truncated at L` = persentase teks yang **terpotong** kalau `MAX_LEN = L`.
# - `MAX_LEN` rekomendasi = persentil ke-`MAX_LEN_PERCENTILE` (default 95) dibulatkan ke kelipatan 32.
#
# **Jika X maka Y:**
# - Rekomendasi ≤ 128 (tweet/komentar) → `MAX_LEN=128`, batch bisa besar (32).
# - 128–384 (review/paragraf) → `MAX_LEN=256`, batch 16.
# - Banyak teks > 512 token (artikel/dokumen) → `LONG_TEXT_STRATEGY="head_tail"` (cepat, sering sama bagusnya)
#   atau `"chunk"` (lebih lengkap, lebih lambat), atau model konteks panjang (`modernbert_base`, max 8192).
# - Distribusi panjang **berbeda jauh antar kelas** → panjang teks sendiri adalah sinyal; model classical akan
#   menangkapnya; sebut di laporan.
#
# **Untuk laporan:** figure distribusi panjang + kalimat "95% teks ≤ N token sehingga MAX_LEN = N".

# %%
_label_names = decode_labels(Y) if TASK != "multilabel" else np.array([f"{k} labels" for k in Y.sum(1)])
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
if (tok_lens_tr > 512).mean() > 0.1:
    print("   ⚠️ >10% teks lebih dari 512 token → pertimbangkan LONG_TEXT_STRATEGY='chunk' atau 'head_tail'.")

if CFG.RUN_EDA:
    groups_for_len = _label_names if len(tok_lens_tr) == len(train) else None
    fig = plot_length_distributions(len_df, groups_for_len, " by class")
    save_fig("text_length_distribution", fig)
    by_class = len_df.assign(cls=_label_names).groupby("cls").median().round(1)
    print("Median length per class:")
    display(by_class)
    if HAS_TEST:
        te_words = test["text_tf"].str.split().str.len()
        ks = ks_2samp(len_df["n_words"], te_words)
        print(f"KS test word-length train vs test: stat={ks.statistic:.3f}, p={ks.pvalue:.3g}",
              "⚠️ panjang teks test berbeda" if ks.statistic > 0.1 else "✅ mirip")

# %% [markdown]
# ### 10.2 Kata & n-gram khas per kelas
#
# - **frequent** = n-gram paling sering di kelas tsb (sering didominasi kata umum).
# - **distinctive** = n-gram dengan **log-odds** tertinggi (kelas ini vs kelas lain) → kata yang benar-benar
#   membedakan. Ini yang paling menarik untuk laporan ("kata 'kecewa', 'rusak' khas ulasan negatif").
#
# Dihitung dari `text_cl` dengan stopword dibuang (hanya untuk EDA, tidak memengaruhi model).
#
# **Jika** distinctive n-gram terlihat seperti template/artefak (mis. nama sumber, "baca selengkapnya") →
# kemungkinan shortcut/leakage; pertimbangkan dibersihkan di `classical_normalize`.
# **Jika** muncul banyak slang yang belum dinormalisasi → tambahkan ke `CFG.CUSTOM_SLANG_DICT`.

# %%
if CFG.RUN_EDA:
    try:
        if TASK == "multilabel":
            parts = []
            for j, lab in enumerate(CLASSES[:8]):
                g = np.where(Y[:, j] == 1, lab, "__rest__")
                d = top_ngrams_by_group(train["text_cl"], g, (1, 2), CFG.TOP_NGRAMS, STOPWORDS_ALL)
                parts.append(d[d["group"] == lab])
            NGRAM_DF = pd.concat(parts, ignore_index=True)
        else:
            NGRAM_DF = top_ngrams_by_group(train["text_cl"], _label_names, (1, 2), CFG.TOP_NGRAMS, STOPWORDS_ALL)
        save_table(NGRAM_DF, "top_ngrams_per_class")
        display(NGRAM_DF[NGRAM_DF["type"] == "distinctive"].groupby("group").head(8))
        fig = plot_distinctive_ngrams(NGRAM_DF)
        if fig is not None:
            save_fig("distinctive_ngrams_per_class", fig)
    except Exception as e:
        print("n-gram EDA dilewati:", short_err(e))

# %% [markdown]
# ### 10.3 Duplikat, near-duplicate, dan label konflik
#
# - **Exact duplicate** dihitung setelah normalisasi agresif (lowercase, tanpa tanda baca).
# - **Label konflik** = teks sama tapi label berbeda → label noise. Batas atas akurasi turun.
# - **Near-duplicate** = cosine similarity char-TF-IDF ≥ `NEAR_DUP_THRESHOLD` (mis. retweet dengan sedikit beda).
# - **Train–test overlap**: teks test yang persis ada di train.
#
# **Jika X maka Y:**
# - Banyak duplikat di train → kalau duplikat jatuh di fold berbeda, CV jadi **terlalu optimis**. Solusi: set
#   `CFG.GROUP_COL = "dup_group"` (kolom ini dibuat otomatis di bawah) supaya duplikat selalu di fold yang sama.
# - Banyak label konflik → pertimbangkan `LABEL_SMOOTHING=0.1`, atau dedup dengan label mayoritas.
# - Banyak test ada di train → untuk baris itu, label train bisa langsung dipakai (cek aturan lomba!).
#
# **Untuk laporan:** sebutkan jumlah duplikat & konflik dan cara penanganannya.

# %%
train["text_norm"] = train["raw_text"].apply(normalize_for_dup)
train["dup_group"] = train.groupby("text_norm").ngroup()
_nonempty = train["text_norm"] != ""
n_dup = int(train.loc[_nonempty, "text_norm"].duplicated().sum())
print(f"Exact duplicates (normalized) in train: {n_dup} ({n_dup / len(train):.1%})")
CONFLICT_DF = pd.DataFrame()
if TASK != "multilabel" and n_dup:
    g = pd.DataFrame({"t": train["text_norm"], "y": Y})[_nonempty.values].groupby("t")["y"].nunique()
    conflict_texts = g[g > 1].index
    CONFLICT_DF = train[train["text_norm"].isin(conflict_texts)][["raw_text", CFG.TARGET_COL]]
    print(f"Texts with CONFLICTING labels: {len(conflict_texts)}",
          "⚠️ label noise" if len(conflict_texts) else "✅")
    if len(conflict_texts):
        display(CONFLICT_DF.sort_values("raw_text").head(10))
        save_table(CONFLICT_DF, "conflicting_labels")
if HAS_TEST:
    test["text_norm"] = test["raw_text"].apply(normalize_for_dup)
    overlap = (test["text_norm"].isin(set(train.loc[_nonempty, "text_norm"])) & (test["text_norm"] != "")).mean()
    print(f"Test texts that also appear in train: {overlap:.1%}", "⚠️ cek aturan lomba" if overlap > 0.02 else "✅")

if CFG.RUN_EDA:
    with Timer("near-duplicate search"):
        NEAR_DUP = near_duplicate_pairs(train["text_norm"], CFG.NEAR_DUP_THRESHOLD, CFG.NEAR_DUP_MAX_ROWS, CFG.SEED)
    NEAR_DUP = NEAR_DUP[train["text_norm"].values[NEAR_DUP["i"]] != train["text_norm"].values[NEAR_DUP["j"]]] \
        if len(NEAR_DUP) else NEAR_DUP
    print(f"Near-duplicate pairs (sim ≥ {CFG.NEAR_DUP_THRESHOLD}, excluding exact): {len(NEAR_DUP)}")
    if len(NEAR_DUP):
        show = NEAR_DUP.head(8).copy()
        show["text_i"] = train["raw_text"].values[show["i"]]
        show["text_j"] = train["raw_text"].values[show["j"]]
        display(show)
if n_dup / max(len(train), 1) > 0.05 and not CFG.GROUP_COL:
    print("💡 Duplikat > 5% → pertimbangkan CFG.GROUP_COL = 'dup_group' supaya CV tidak bocor.")

# %% [markdown]
# ### 10.4 Drift train vs test: overlap vocabulary & adversarial validation
#
# - **Token coverage** = % kemunculan token test yang ada di vocab train. < 90% → banyak kata baru di test
#   (topik/periode berbeda) → model berbasis subword (transformer) & char n-gram lebih tahan.
# - **Adversarial validation**: model dilatih membedakan baris train vs test.
#   AUC ≈ 0.5 → distribusi mirip ✅; AUC > 0.7 → **drift**: CV mungkin tidak mencerminkan leaderboard.
#   Kata-kata yang paling membedakan dicetak → cari tahu penyebabnya (periode waktu, sumber, template).
#
# **Jika** drift tinggi → (1) pilih fold/validasi yang mirip test (mis. data terbaru), (2) hindari fitur yang
# drift, (3) andalkan model yang lebih general (transformer/embedding), (4) jangan overfit threshold.

# %%
if CFG.RUN_EDA and HAS_TEST and len(test) > 0:
    cov_tok, cov_type, oov_top = vocab_overlap(train["text_cl"], test["text_cl"])
    print(f"Token coverage (test tokens in train vocab): {cov_tok:.1%} | type coverage: {cov_type:.1%}",
          "⚠️ banyak kata baru di test" if cov_tok < 0.9 else "✅")
    print("Top OOV words in test:", oov_top[:15])
    if CFG.RUN_ADVERSARIAL_VALIDATION:
        try:
            ADV_AUC, adv_test_words, adv_train_words = adversarial_validation(train["text_cl"], test["text_cl"],
                                                                              seed=CFG.SEED)
            print(f"Adversarial validation AUC = {ADV_AUC:.3f}",
                  "⚠️ DRIFT train vs test" if ADV_AUC > 0.7 else "✅ train & test mirip")
            print("  kata khas TEST :", adv_test_words[:10])
            print("  kata khas TRAIN:", adv_train_words[:10])
        except Exception as e:
            print("Adversarial validation dilewati:", short_err(e))
elif not HAS_TEST:
    print("Test tidak ada → analisis drift dilewati.")

# %% [markdown]
# ### 10.5 Meta-feature sederhana per kelas (+ word cloud opsional)
#
# Rata-rata jumlah kata, tanda seru, tanda tanya, rasio huruf kapital, URL, mention, emoji per kelas.
# **Jika** satu fitur sangat berbeda antar kelas (mis. ulasan negatif lebih banyak "!") → bagus untuk narasi
# laporan, dan menjelaskan kenapa cleaning yang membuang tanda baca bisa menurunkan skor transformer.

# %%
if CFG.RUN_EDA:
    MF = meta_features(train["raw_text"])
    if TASK != "multilabel":
        mf_tab = MF.assign(cls=_label_names).groupby("cls").mean().round(3)
        display(mf_tab)
        save_table(mf_tab.reset_index(), "meta_features_by_class")
        if NUM_CLASSES <= 12:
            fig, ax = plt.subplots(figsize=(max(6, NUM_CLASSES * 0.9 + 2), 4))
            data = [MF["n_words"].values[_label_names == c] for c in CLASSES]
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(CLASSES) + 1))
            ax.set_xticklabels([str(c) for c in CLASSES], rotation=30 if NUM_CLASSES > 5 else 0)
            ax.set_title("Word count by class")
            ax.set_xlabel("Class")
            ax.set_ylabel("Number of words")
            save_fig("word_count_by_class", fig)
    if CFG.MAKE_WORDCLOUD and HAS_WORDCLOUD and TASK != "multilabel":
        try:
            show_cls = CLASSES[:4]
            fig, axes = plt.subplots(1, len(show_cls), figsize=(5 * len(show_cls), 4))
            for ax, c in zip(np.atleast_1d(axes), show_cls):
                txt = " ".join(train.loc[_label_names == c, "text_cl"].tolist())
                wc = WordCloud(width=600, height=400, background_color="white",
                               stopwords=STOPWORDS_ALL, max_words=100).generate(txt or "empty")
                ax.imshow(wc)
                ax.axis("off")
                ax.set_title(f"Word cloud: {c}")
            save_fig("wordclouds", fig)
        except Exception as e:
            print("Word cloud dilewati:", short_err(e))
    elif CFG.MAKE_WORDCLOUD and not HAS_WORDCLOUD:
        print("ℹ️ wordcloud tidak terpasang → dilewati (opsional: pip install wordcloud).")

# %% [markdown]
# ## 11. Metric registry & aturan keputusan (threshold / bias)
#
# Semua model menghasilkan **probabilitas**. Untuk metric berbasis label (F1, accuracy, QWK, MCC) probabilitas
# diubah ke label lewat **decision rule**:
# - binary → `threshold` (default 0.5, nanti di-tuning),
# - multiclass → `argmax(log p + bias)` dengan `bias` per kelas (default 0, nanti di-tuning untuk macro-F1),
# - multilabel → threshold per label.
#
# | metric | tipe | lebih besar lebih baik? | catatan |
# |---|---|---|---|
# | f1 / precision / recall | label | ya | binary, untuk kelas positif (index 1) |
# | f1_macro / f1_micro / f1_weighted | label | ya | macro = semua kelas setara (bagus untuk imbalance) |
# | accuracy / balanced_accuracy / mcc | label | ya | MCC robust untuk imbalance |
# | qwk | label | ya | ordinal (kelas berurutan) → isi `CLASS_ORDER` |
# | roc_auc / roc_auc_ovr / pr_auc | proba | ya | tidak butuh threshold |
# | log_loss | proba | **tidak** | sensitif terhadap kalibrasi & prediksi terlalu yakin |
# | f1_samples / roc_auc_macro / hamming_loss | multilabel | ya / ya / **tidak** | |
#
# **Jika** metric lomba tidak ada di daftar → tambahkan ke dict `METRICS` (format: kind, greater, fn).

# %%
EPS = 1e-7


def default_decision():
    """Decision rule default per task."""
    if TASK == "binary":
        return 0.5
    if TASK == "multiclass":
        return np.zeros(NUM_CLASSES)
    return np.full(NUM_CLASSES, 0.5)


def proba_to_labels(proba, decision=None):
    """Probabilitas → label sesuai decision rule (threshold / bias / threshold per label)."""
    decision = default_decision() if decision is None else decision
    proba = np.asarray(proba, dtype=float)
    if TASK == "binary":
        return (proba >= float(decision)).astype(int)
    if TASK == "multiclass":
        return np.argmax(np.log(np.clip(proba, EPS, 1)) + np.asarray(decision)[None, :], axis=1)
    return (proba >= np.asarray(decision)[None, :]).astype(int)


def _safe(fn):
    def wrap(*a):
        try:
            return float(fn(*a))
        except Exception:
            return np.nan
    return wrap


def _logloss(y, p):
    if TASK == "binary":
        p = np.clip(p, EPS, 1 - EPS)
        return skm.log_loss(y, np.c_[1 - p, p], labels=[0, 1])
    if TASK == "multiclass":
        p = np.clip(p, EPS, 1)
        return skm.log_loss(y, p / p.sum(1, keepdims=True), labels=list(range(NUM_CLASSES)))
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


METRICS = {
    # name: (kind, greater_is_better, fn(y_true, y_label, proba))
    "f1": ("label", True, _safe(lambda y, l, p: skm.f1_score(y, l, zero_division=0))),
    "precision": ("label", True, _safe(lambda y, l, p: skm.precision_score(y, l, zero_division=0))),
    "recall": ("label", True, _safe(lambda y, l, p: skm.recall_score(y, l, zero_division=0))),
    "f1_macro": ("label", True, _safe(lambda y, l, p: skm.f1_score(y, l, average="macro", zero_division=0))),
    "f1_micro": ("label", True, _safe(lambda y, l, p: skm.f1_score(y, l, average="micro", zero_division=0))),
    "f1_weighted": ("label", True, _safe(lambda y, l, p: skm.f1_score(y, l, average="weighted", zero_division=0))),
    "f1_samples": ("label", True, _safe(lambda y, l, p: skm.f1_score(y, l, average="samples", zero_division=0))),
    "accuracy": ("label", True, _safe(lambda y, l, p: skm.accuracy_score(y, l))),
    "balanced_accuracy": ("label", True, _safe(lambda y, l, p: skm.balanced_accuracy_score(y, l))),
    "mcc": ("label", True, _safe(lambda y, l, p: skm.matthews_corrcoef(y, l))),
    "qwk": ("label", True, _safe(lambda y, l, p: skm.cohen_kappa_score(y, l, weights="quadratic"))),
    "hamming_loss": ("label", False, _safe(lambda y, l, p: skm.hamming_loss(y, l))),
    "roc_auc": ("proba", True, _safe(lambda y, l, p: skm.roc_auc_score(y, p))),
    "pr_auc": ("proba", True, _safe(lambda y, l, p: skm.average_precision_score(y, p))),
    "roc_auc_ovr": ("proba", True, _safe(lambda y, l, p: skm.roc_auc_score(
        y, np.clip(p, EPS, 1) / np.clip(p, EPS, 1).sum(1, keepdims=True), multi_class="ovr", average="macro",
        labels=list(range(NUM_CLASSES))))),
    "roc_auc_macro": ("proba", True, _safe(lambda y, l, p: skm.roc_auc_score(y, p, average="macro"))),
    "log_loss": ("proba", False, _safe(lambda y, l, p: _logloss(y, p))),
}
TASK_METRICS = {
    "binary": ["f1", "precision", "recall", "accuracy", "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "log_loss"],
    "multiclass": ["f1_macro", "f1_micro", "f1_weighted", "accuracy", "balanced_accuracy", "mcc", "qwk",
                   "roc_auc_ovr", "log_loss"],
    "multilabel": ["f1_micro", "f1_macro", "f1_samples", "accuracy", "hamming_loss", "roc_auc_macro", "log_loss"],
}
assert CFG.METRIC in METRICS, f"METRIC '{CFG.METRIC}' tidak dikenal. Pilihan: {list(METRICS)}"
if CFG.METRIC not in TASK_METRICS[TASK]:
    print(f"⚠️ METRIC '{CFG.METRIC}' tidak lazim untuk task {TASK}. Pilihan umum: {TASK_METRICS[TASK]}")
METRIC_KIND, GREATER_IS_BETTER, _ = METRICS[CFG.METRIC]


def compute_score(y_true, proba, metric=None, decision=None):
    """Skor satu metric dari probabilitas."""
    metric = metric or CFG.METRIC
    kind, _, fn = METRICS[metric]
    labels = proba_to_labels(proba, decision) if kind == "label" else None
    return fn(y_true, labels, proba)


def evaluate_all(y_true, proba, decision=None):
    """Semua metric relevan untuk task (dict)."""
    return {m: compute_score(y_true, proba, m, decision) for m in TASK_METRICS[TASK]}


def is_better(a, b):
    """True kalau skor a lebih baik dari b sesuai arah metric."""
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return True
    return a > b if GREATER_IS_BETTER else a < b


print(f"Main metric: {CFG.METRIC} ({METRIC_KIND}-based, {'higher' if GREATER_IS_BETTER else 'lower'} is better)")

# %% [markdown]
# ## 12. Cross-validation folds
#
# **Kenapa K-Fold?** Semua model dievaluasi dengan prediksi **out-of-fold (OOF)**: tiap baris diprediksi oleh model
# yang tidak pernah melihat baris itu. OOF score = estimasi jujur skor leaderboard, dan OOF dipakai untuk
# ensemble & threshold tuning tanpa bocor.
#
# Strategi (`CV_STRATEGY="auto"`):
# - default → `StratifiedKFold` (proporsi kelas sama di tiap fold),
# - `GROUP_COL` diisi → `StratifiedGroupKFold` (satu group tidak pernah terpecah di train & valid),
# - multilabel → stratifikasi berdasarkan label paling langka per baris (pendekatan sederhana).
#
# **Cara membaca output:** ukuran tiap fold harus mirip, dan proporsi kelas per fold harus mirip.
# **Jika** ada kelas yang hilang di sebagian fold → kelas terlalu langka; turunkan `N_FOLDS` atau gabung kelas.
# **Untuk laporan:** "Evaluasi memakai Stratified 5-Fold CV; skor yang dilaporkan adalah skor OOF".

# %%
def make_folds(n_folds, seed):
    """Buat array fold id sesuai CV_STRATEGY."""
    folds = np.full(len(train), -1)
    strat = CFG.CV_STRATEGY
    groups = train[CFG.GROUP_COL].astype(str).values if (CFG.GROUP_COL and CFG.GROUP_COL in train.columns) else None
    if strat == "auto":
        strat = "stratified_group" if groups is not None else "stratified"
    if TASK == "multilabel":
        freq = Y.sum(0)
        y_strat = np.array([np.argmin(np.where(r == 1, freq, np.inf)) if r.sum() else NUM_CLASSES for r in Y])
    else:
        y_strat = Y
    if strat in ("group", "stratified_group") and groups is None:
        print("⚠️ GROUP_COL tidak diisi/ tidak ada → fallback stratified.")
        strat = "stratified"
    if strat == "stratified_group":
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            it = splitter.split(train, y_strat, groups)
        except Exception:
            print("StratifiedGroupKFold tidak tersedia → GroupKFold")
            it = GroupKFold(n_splits=n_folds).split(train, y_strat, groups)
    elif strat == "group":
        it = GroupKFold(n_splits=n_folds).split(train, y_strat, groups)
    elif strat == "kfold":
        it = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(train)
    else:
        it = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(train, y_strat)
    for f, (_, va) in enumerate(it):
        folds[va] = f
    print(f"CV strategy: {strat}, n_folds={n_folds}")
    globals()["CV_STRATEGY_USED"] = strat
    return folds


train["fold"] = make_folds(CFG.N_FOLDS, CFG.SEED)
FOLDS = train["fold"].values
display(pd.Series(FOLDS).value_counts().sort_index().rename("n_rows").to_frame().T)
if TASK != "multilabel":
    display(pd.crosstab(FOLDS, decode_labels(Y), normalize="index").round(3))

# %% [markdown]
# ## 13. Result registry (semua model disimpan di satu tempat)
#
# `register_result(name, oof, test_pred)` menyimpan OOF & test prediction ke `RESULTS`, menulis file `.npy` ke
# `preds/`, menghitung skor per fold, dan menambah baris ke `experiment_log.csv`.
#
# **Cara membaca output tiap model:** `OOF <metric> = x | fold mean ± std`.
# - **Jika** std antar fold > 0.02 (untuk metric 0–1) → model tidak stabil: data kecil/noisy, coba regularisasi
#   lebih kuat, lebih banyak fold, atau ensemble.
# - **Jika** skor fold sangat berbeda satu sama lain → cek apakah ada group/duplikat yang bocor antar fold.

# %%
RESULTS = {}


def empty_pred(n):
    """Array prediksi kosong sesuai bentuk task."""
    return np.zeros(n) if TASK == "binary" else np.zeros((n, NUM_CLASSES))


def register_result(name, oof, test_pred, family, seconds=None, mask=None):
    """Simpan hasil satu model + print ringkasan skor."""
    mask = np.ones(len(Y), bool) if mask is None else mask
    score = compute_score(Y[mask], oof[mask])
    fold_scores = [compute_score(Y[(FOLDS == f) & mask], oof[(FOLDS == f) & mask])
                   for f in np.unique(FOLDS[mask])]
    RESULTS[name] = {"oof": oof, "test": test_pred, "score": score, "family": family,
                     "fold_mean": float(np.nanmean(fold_scores)), "fold_std": float(np.nanstd(fold_scores)),
                     "seconds": seconds, "partial": bool(~mask.all())}
    np.save(os.path.join(CFG.OUTPUT_DIR, "preds", f"oof_{name}.npy"), oof)
    if test_pred is not None:
        np.save(os.path.join(CFG.OUTPUT_DIR, "preds", f"test_{name}.npy"), test_pred)
    log_experiment(name, score, {"family": family, "fold_std": RESULTS[name]["fold_std"]})
    flag = "⚠️ fold variance tinggi" if RESULTS[name]["fold_std"] > 0.02 else "✅"
    part = " (PARTIAL folds)" if RESULTS[name]["partial"] else ""
    print(f"  ==> {name}: OOF {CFG.METRIC} = {score:.5f} | folds {np.nanmean(fold_scores):.4f} ± "
          f"{np.nanstd(fold_scores):.4f} {flag}{part}")
    return score


def align_proba(proba, classes_seen, n_classes=None):
    """Model yang dilatih tanpa sebagian kelas → kembalikan matriks (n, n_classes) penuh."""
    full = np.zeros((proba.shape[0], n_classes or NUM_CLASSES))
    full[:, np.asarray(classes_seen, dtype=int)] = proba
    return full


def to_task_output(proba_full):
    """(n, K) → bentuk output task (binary: kolom positif)."""
    return proba_full[:, 1] if TASK == "binary" else proba_full


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
# **Wajib dijalankan** — cepat (detik–menit di CPU), sering sangat kompetitif untuk teks pendek, dan menjadi
# sanity check: kalau transformer kalah dari TF-IDF, ada yang salah (LR, MAX_LEN, label, cleaning).
#
# | model | ide | kapan bagus |
# |---|---|---|
# | `logreg` | Logistic Regression | baseline terkuat & paling stabil; probabilitas cukup terkalibrasi |
# | `linearsvc` | Linear SVM + kalibrasi sigmoid | sering sedikit di atas logreg untuk teks |
# | `complement_nb` | Naive Bayes untuk teks imbalance | sangat cepat; bagus untuk data kecil |
# | `sgd` | SGD modified_huber | cepat untuk data besar (>100k) |
# | `ridge` | RidgeClassifier + kalibrasi | variasi linear lain (bagus untuk ensemble) |
# | `nbsvm` | fitur log-count ratio (NB) + LogReg | klasik kuat untuk sentimen |
# | `lgbm_svd` | TruncatedSVD (dense) → LightGBM | non-linear; berbeda dari model linear → bagus untuk ensemble |
#
# **Cara membaca output:** tiap model mencetak OOF score + mean ± std per fold. Bandingkan di tabel perbandingan
# (section 19). **Jika** semua model sangat jelek (≈ tebakan acak) → cek encoding label & kolom teks.
# **Jika** satu model gagal (error) → tercatat di `FAILED_MODELS`, model lain tetap jalan.
#
# **Untuk laporan:** tabel skor baseline (TF-IDF + LogReg) sebagai pembanding model utama.

# %%
def fit_predict_nbsvm(Xtr, ytr_bin, X_list, C=4.0, alpha=1.0):
    """NB-SVM (Wang & Manning 2012): skala fitur dengan log-count ratio r, lalu LogReg. Return prob kelas 1."""
    p = alpha + np.asarray(Xtr[ytr_bin == 1].sum(0)).ravel()
    q = alpha + np.asarray(Xtr[ytr_bin == 0].sum(0)).ravel()
    r = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
    R = sparse.diags(r)
    clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
    clf.fit(Xtr @ R, ytr_bin)
    return [clf.predict_proba(X @ R)[:, 1] if X is not None else None for X in X_list]


def _class_weight():
    return "balanced" if CFG.USE_CLASS_WEIGHTS else None


def _fit_single_label(name, Xtr, ytr, X_list, n_classes, cache):
    """Fit satu model untuk target single-label (binary atau multiclass). Return list proba (n, n_classes)."""
    seen = np.unique(ytr)
    if len(seen) == 1:  # fold hanya punya satu kelas
        return [None if X is None else align_proba(np.ones((X.shape[0], 1)), seen, n_classes) for X in X_list]
    if name == "nbsvm":
        outs = []
        for c in range(n_classes):
            outs.append(fit_predict_nbsvm(Xtr, (ytr == c).astype(int), X_list, C=CFG.LOGREG_C)
                        if c in seen else [None if X is None else np.zeros(X.shape[0]) for X in X_list])
        res = []
        for k, X in enumerate(X_list):
            if X is None:
                res.append(None)
                continue
            P = np.column_stack([outs[c][k] for c in range(n_classes)])
            if n_classes == 2:
                P = np.column_stack([1 - outs[1][k], outs[1][k]])
            res.append(P / np.clip(P.sum(1, keepdims=True), EPS, None))
        return res
    if name == "lgbm_svd":
        if "svd" not in cache:
            n_comp = int(min(CFG.SVD_COMPONENTS, Xtr.shape[1] - 1, Xtr.shape[0] - 1))
            svd = TruncatedSVD(n_components=max(2, n_comp), random_state=CFG.SEED)
            cache["svd"] = (svd.fit_transform(Xtr), [svd.transform(X) if X is not None else None for X in X_list])
        Ztr, Zs = cache["svd"]
        model = fit_lgbm_inner_es(Ztr, ytr, "classification", CFG.SEED)
        return [None if Z is None else align_proba(model.predict_proba(Z), model.classes_, n_classes) for Z in Zs]

    if name == "logreg":
        model = LogisticRegression(C=CFG.LOGREG_C, max_iter=2000, class_weight=_class_weight(),
                                   solver="liblinear" if len(seen) == 2 else "lbfgs")
    elif name == "linearsvc":
        min_count = np.bincount(ytr).min() if len(seen) == n_classes else 0
        base = LinearSVC(C=0.5, class_weight=_class_weight())
        model = make_calibrated(base, cv=3) if min_count >= 3 else base
    elif name == "complement_nb":
        model = ComplementNB(alpha=0.3)
    elif name == "sgd":
        model = SGDClassifier(loss="modified_huber", alpha=2e-5, max_iter=50, tol=1e-4,
                              class_weight=_class_weight(), random_state=CFG.SEED)
    elif name == "ridge":
        min_count = np.bincount(ytr).min() if len(seen) == n_classes else 0
        base = RidgeClassifier(alpha=1.0, class_weight=_class_weight())
        model = make_calibrated(base, cv=3) if min_count >= 3 else base
    else:
        raise ValueError(f"Model classical tidak dikenal: {name}")
    model.fit(Xtr, ytr)
    classes = getattr(model, "classes_", seen)
    out = []
    for X in X_list:
        if X is None:
            out.append(None)
            continue
        if hasattr(model, "predict_proba"):
            P = model.predict_proba(X)
        else:  # decision_function → probabilitas pseudo
            d = model.decision_function(X)
            P = np.column_stack([-d, d]) if d.ndim == 1 else d
            P = softmax_np(P * 2.0)
        out.append(align_proba(P, classes, n_classes))
    return out


def fit_predict_classical(name, Xtr, ytr, X_list):
    """Wrapper task-aware. Return list prediksi dengan bentuk output task."""
    if TASK != "multilabel":
        res = _fit_single_label(name, Xtr, ytr, X_list, NUM_CLASSES, cache={})
        return [None if P is None else to_task_output(P) for P in res]
    cols = []
    cache = {}
    for j in range(NUM_CLASSES):
        res = _fit_single_label(name, Xtr, ytr[:, j], X_list, 2, cache)
        cols.append([None if P is None else P[:, 1] for P in res])
    return [None if cols[0][k] is None else np.column_stack([c[k] for c in cols]) for k in range(len(X_list))]


CLASSICAL_MODELS = [m for m in CFG.CLASSICAL_MODELS if not (m == "lgbm_svd" and not HAS_LGB)]
if "lgbm_svd" in CFG.CLASSICAL_MODELS and not HAS_LGB:
    print("ℹ️ lightgbm tidak ada → lgbm_svd dilewati.")

if CFG.RUN_CLASSICAL:
    n_te = len(test) if HAS_TEST else 0
    oofs = {m: empty_pred(len(train)) for m in CLASSICAL_MODELS}
    tests = {m: empty_pred(n_te) for m in CLASSICAL_MODELS}
    times = defaultdict(float)
    te_texts = test["text_cl"].values if HAS_TEST else []
    for fold in range(CFG.N_FOLDS):
        tr_idx, va_idx = np.where(FOLDS != fold)[0], np.where(FOLDS == fold)[0]
        t0 = time.time()
        Xtr, (Xva, Xte), _ = build_tfidf(train["text_cl"].values[tr_idx], [train["text_cl"].values[va_idx], te_texts])
        print(f"Fold {fold}: TF-IDF {Xtr.shape} built in {time.time() - t0:.1f}s")
        for m in CLASSICAL_MODELS:
            key = f"tfidf_{m}"
            if key in FAILED_MODELS:
                continue
            t0 = time.time()
            try:
                pv, pt = fit_predict_classical(m, Xtr, Y[tr_idx], [Xva, Xte])
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
        key = f"tfidf_{m}"
        if key not in FAILED_MODELS:
            register_result(key, oofs[m], tests[m] if HAS_TEST else None, "classical", times[m])
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
# ### 16.1 Sentence embedding + LogReg / LightGBM (K-Fold)
#
# **Head yang tersedia** (`CFG.EMBEDDING_HEADS`): `logreg` (StandardScaler + LogReg; biasanya terbaik),
# `ridge` (RidgeClassifier terkalibrasi), `lgbm` (LightGBM, non-linear), `xgb`, `catboost` (kalau terpasang).
#
# **Cara membaca output:** skor OOF tiap kombinasi `emb_<model>_<head>`.
# - **Jika** embedding + LogReg ≈ TF-IDF → bagus, keduanya saling melengkapi di ensemble.
# - **Jika** jauh di bawah TF-IDF → model embedding kurang cocok dengan bahasa/domain; coba `multilingual-e5-base`.
# - **Jika** LightGBM jauh di bawah LogReg → normal untuk embedding (fitur dense & linear-separable).
#
# **Untuk laporan:** "Representasi semantik dari model sentence-embedding multibahasa memberi skor X dan
# meningkatkan ensemble sebesar Y".

# %%
def fit_predict_dense(head, Ztr, ytr, Z_list, n_classes=None):
    """Fit head pada fitur dense (embedding). Return list prob (n, n_classes)."""
    n_classes = n_classes or NUM_CLASSES
    seen = np.unique(ytr)
    if len(seen) == 1:
        return [None if Z is None else align_proba(np.ones((Z.shape[0], 1)), seen, n_classes) for Z in Z_list]
    if head == "logreg":
        sc = StandardScaler().fit(Ztr)
        m = LogisticRegression(C=0.5, max_iter=3000, class_weight=_class_weight()).fit(sc.transform(Ztr), ytr)
        return [None if Z is None else align_proba(m.predict_proba(sc.transform(Z)), m.classes_, n_classes)
                for Z in Z_list]
    if head == "ridge":
        base = RidgeClassifier(alpha=10.0, class_weight=_class_weight())
        m = make_calibrated(base, 3) if np.bincount(ytr).min() >= 3 else base
        m.fit(Ztr, ytr)
        out = []
        for Z in Z_list:
            if Z is None:
                out.append(None)
            elif hasattr(m, "predict_proba"):
                out.append(align_proba(m.predict_proba(Z), m.classes_, n_classes))
            else:
                d = m.decision_function(Z)
                d = np.column_stack([-d, d]) if d.ndim == 1 else d
                out.append(align_proba(softmax_np(d * 2.0), m.classes_, n_classes))
        return out
    if head == "lgbm":
        if not HAS_LGB:
            raise ImportError("lightgbm tidak terpasang")
        m = fit_lgbm_inner_es(Ztr, ytr, "classification", CFG.SEED)
        return [None if Z is None else align_proba(m.predict_proba(Z), m.classes_, n_classes) for Z in Z_list]
    if head == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost tidak terpasang")
        enc = {c: i for i, c in enumerate(seen)}
        m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                              colsample_bytree=0.5, tree_method="hist", random_state=CFG.SEED, n_jobs=-1)
        m.fit(Ztr, np.array([enc[v] for v in ytr]))
        return [None if Z is None else align_proba(m.predict_proba(Z), seen, n_classes) for Z in Z_list]
    if head == "catboost":
        if not HAS_CB:
            raise ImportError("catboost tidak terpasang")
        m = cb.CatBoostClassifier(iterations=500, learning_rate=0.08, depth=6, random_seed=CFG.SEED, verbose=False)
        m.fit(Ztr, ytr)
        return [None if Z is None else align_proba(m.predict_proba(Z), m.classes_.astype(int), n_classes)
                for Z in Z_list]
    raise ValueError(f"Embedding head tidak dikenal: {head}")


def fit_predict_dense_task(head, Ztr, ytr, Z_list):
    """Task-aware wrapper. Multilabel → satu model biner per label."""
    if TASK != "multilabel":
        return [None if P is None else to_task_output(P) for P in fit_predict_dense(head, Ztr, ytr, Z_list)]
    per_label = []
    for j in range(NUM_CLASSES):
        res = fit_predict_dense(head, Ztr, ytr[:, j], Z_list, n_classes=2)
        per_label.append([None if P is None else P[:, 1] for P in res])
    out = []
    for k in range(len(Z_list)):
        out.append(None if per_label[0][k] is None else np.column_stack([pl[k] for pl in per_label]))
    return out


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
            oof, tp = empty_pred(len(train)), (empty_pred(len(test)) if HAS_TEST else None)
            t0 = time.time()
            try:
                for fold in range(CFG.N_FOLDS):
                    tr_idx, va_idx = np.where(FOLDS != fold)[0], np.where(FOLDS == fold)[0]
                    pv, pt = fit_predict_dense_task(head, E_tr[tr_idx], Y[tr_idx], [E_tr[va_idx], E_te])
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
# ### 17.1 Jalankan fine-tuning K-Fold
#
# **Sebelum menjalankan full:** coba `CFG.FOLDS_TO_RUN = [0]` + `EPOCHS=1` untuk mengecek waktu per fold & VRAM.
# Estimasi waktu total ≈ waktu 1 fold × N_FOLDS × jumlah model.
#
# **Cara membaca log:** tiap evaluasi mencetak `train_loss` dan skor valid; tanda `*` = skor terbaik baru
# (state disimpan). Di akhir tiap fold dicetak skor fold, lalu ringkasan OOF.
#
# **Jika X maka Y:**
# - train_loss turun tapi valid tidak membaik sejak epoch 1 → overfit cepat: turunkan LR, `EPOCHS=2`, naikkan
#   `WEIGHT_DECAY`, atau `LABEL_SMOOTHING=0.1`.
# - train_loss tidak turun sama sekali (≈ ln(K)) → LR terlalu besar/kecil, atau label salah. Coba `LR=2e-5`,
#   `USE_AMP=False`.
# - Skor fold sangat bervariasi → data kecil; pakai `MULTI_SAMPLE_DROPOUT=5`, `LLRD=0.9`, atau seed averaging.
# - Transformer kalah dari TF-IDF → cek `MAX_LEN` (terlalu pendek?), model bahasa yang salah, epoch kurang.
#
# **Untuk laporan:** sebutkan model, MAX_LEN, LR, epoch, batch, strategi teks panjang, dan skor OOF per fold;
# sertakan figure learning curve.

# %%
RUN_TF = CFG.RUN_TRANSFORMER
if str(RUN_TF).lower() == "auto":
    RUN_TF = HAS_CUDA
    if not HAS_CUDA:
        print("ℹ️ RUN_TRANSFORMER='auto' dan tidak ada GPU → fine-tuning dilewati (set True untuk paksa di CPU).")
if RUN_TF and not (HAS_TORCH and HAS_TRANSFORMERS):
    print("⚠️ torch/transformers tidak tersedia → fine-tuning dilewati.")
    RUN_TF = False
if RUN_TF and CFG.TRAINER_BACKEND == "hf_trainer" and not HAS_ACCELERATE:
    print("⚠️ HF Trainer butuh package 'accelerate' (pip install accelerate) → fallback ke custom_loop.")
    CFG.TRAINER_BACKEND = "custom_loop"

TF_HISTORY = []
if RUN_TF:
    spec = {"task": TASK, "num_outputs": NUM_CLASSES, "class_weights": None}
    if CFG.USE_CLASS_WEIGHTS:
        if TASK == "multilabel":
            pos = Y.sum(0).clip(1, None)
            spec["class_weights"] = torch.tensor(np.clip((len(Y) - pos) / pos, 1, 10), dtype=torch.float32)
        else:
            cnt = np.bincount(Y, minlength=NUM_CLASSES).clip(1, None)
            spec["class_weights"] = torch.tensor(len(Y) / (NUM_CLASSES * cnt), dtype=torch.float32)
        print("Class weights:", np.round(spec["class_weights"].numpy(), 3))
    train_labels = Y.astype(np.float32) if TASK == "multilabel" else Y.astype(np.int64)

    def logits_to_pred_cls(logits):
        """Logits → probabilitas bentuk output task."""
        if TASK == "multilabel":
            return sigmoid_np(logits)
        P = softmax_np(logits)
        return P[:, 1] if TASK == "binary" else P

    def score_rows(idx, pred):
        return compute_score(Y[idx], pred)

    print(f"Backend={CFG.TRAINER_BACKEND} | MAX_LEN={CFG.MAX_LEN} | strategy={CFG.LONG_TEXT_STRATEGY} | "
          f"batch={CFG.BATCH_SIZE}x{CFG.GRAD_ACCUM_STEPS} | epochs={CFG.EPOCHS} | LR={CFG.LR} | AMP={CFG.USE_AMP and HAS_CUDA}")
    for mk in CFG.TRANSFORMER_MODELS:
        key = "tf_" + re.sub(r"[^A-Za-z0-9]+", "_", str(mk).split("/")[-1])
        print("=" * 80, f"\nModel {mk} → {resolve_model_name(mk)}")
        t0 = time.time()
        try:
            oof, tp, mask, hist = run_transformer_cv(mk, spec, train["text_tf"].values, train_labels,
                                                     test["text_tf"].values if HAS_TEST else None,
                                                     logits_to_pred_cls, score_rows)
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
# **Ide:** ambil prediksi test yang **sangat yakin** (confidence ≥ `PSEUDO_CONF_THRESHOLD`) dari model terbaik,
# jadikan label "semu", tambahkan ke data train, latih ulang. Membantu kalau test besar & train kecil.
#
# ⚠️ **Peringatan:**
# 1. Pseudo-label yang salah memperkuat kesalahan model (confirmation bias). Pakai threshold tinggi.
# 2. Pseudo rows **hanya ditambahkan ke fold train**, tidak pernah ke fold valid — OOF tetap dihitung di data asli.
# 3. Tetap ada sedikit bias optimis: prediksi test berasal dari model yang pernah melihat semua baris train.
#    Anggap gain < 0.003 sebagai noise.
# 4. Cek aturan lomba: sebagian lomba melarang pemakaian data test untuk training.
#
# Implementasi di sini memakai TF-IDF + LogReg (cepat) sebagai demonstrasi. Kalau gain nyata, pola yang sama bisa
# diterapkan ke transformer (tambahkan pseudo rows ke `train` lalu jalankan ulang section 17).

# %%
if CFG.RUN_PSEUDO_LABEL and HAS_TEST and RESULTS:
    full_res = {k: v for k, v in RESULTS.items() if not v["partial"] and v["test"] is not None}
    src = max(full_res, key=lambda k: full_res[k]["score"] if GREATER_IS_BETTER else -full_res[k]["score"])
    P_te = full_res[src]["test"]
    if TASK == "binary":
        conf = np.maximum(P_te, 1 - P_te)
    elif TASK == "multiclass":
        conf = P_te.max(1)
    else:
        conf = np.minimum(P_te, 1 - P_te).max(1) * -1 + 1
    cand = np.where(conf >= CFG.PSEUDO_CONF_THRESHOLD)[0]
    cand = cand[np.argsort(-conf[cand])][:int(CFG.PSEUDO_MAX_FRACTION * len(train))]
    print(f"Pseudo-label source: {src} | selected {len(cand)} / {len(P_te)} test rows")
    if len(cand) > 0:
        pseudo_y = proba_to_labels(P_te[cand])
        pseudo_txt = test["text_cl"].values[cand]
        oof_pl, tp_pl = empty_pred(len(train)), empty_pred(len(test))
        for fold in range(CFG.N_FOLDS):
            tr_idx, va_idx = np.where(FOLDS != fold)[0], np.where(FOLDS == fold)[0]
            txt_tr = np.concatenate([train["text_cl"].values[tr_idx], pseudo_txt])
            y_tr = np.concatenate([Y[tr_idx], pseudo_y])
            Xtr, (Xva, Xte), _ = build_tfidf(txt_tr, [train["text_cl"].values[va_idx], test["text_cl"].values])
            pv, pt = fit_predict_classical("logreg", Xtr, y_tr, [Xva, Xte])
            oof_pl[va_idx] = pv
            tp_pl += pt / CFG.N_FOLDS
        register_result("tfidf_logreg_pseudo", oof_pl, tp_pl, "pseudo")
        if "tfidf_logreg" in RESULTS:
            gain = RESULTS["tfidf_logreg_pseudo"]["score"] - RESULTS["tfidf_logreg"]["score"]
            gain = gain if GREATER_IS_BETTER else -gain
            print(f"Gain vs tfidf_logreg: {gain:+.5f}", "✅ membantu" if gain > 0.003 else "⚠️ tidak signifikan")
elif CFG.RUN_PSEUDO_LABEL:
    print("Pseudo-labeling butuh test set dan minimal satu model → dilewati.")
else:
    print("RUN_PSEUDO_LABEL=False → dilewati.")

# %% [markdown]
# ## 19. Perbandingan model
#
# Tabel semua model dengan skor OOF metric utama + semua metric lain, mean ± std per fold, dan waktu training.
#
# **Cara membaca:**
# - Urutkan berdasarkan metric utama. Selisih < 0.002 antar model biasanya **tidak signifikan**.
# - `fold_std` besar = tidak stabil → kurang bisa dipercaya di leaderboard.
# - Model `partial=True` (hanya sebagian fold) tidak ikut ensemble.
#
# **Untuk laporan:** tabel ini (atau versi ringkas 4–6 model) + bar chart = inti bagian "Hasil Eksperimen".

# %%
if not RESULTS:
    raise RuntimeError("Tidak ada model yang berhasil. Cek FAILED_MODELS: " + json.dumps(FAILED_MODELS, indent=1))

rows = []
for name, r in RESULTS.items():
    row = {"model": name, "family": r["family"], CFG.METRIC: r["score"], "fold_mean": r["fold_mean"],
           "fold_std": r["fold_std"], "seconds": r["seconds"], "partial": r["partial"]}
    if not r["partial"]:
        for m, v in evaluate_all(Y, r["oof"]).items():
            if m != CFG.METRIC:
                row[m] = v
    rows.append(row)
SCORE_DF = pd.DataFrame(rows).sort_values(CFG.METRIC, ascending=not GREATER_IS_BETTER).reset_index(drop=True)
display(SCORE_DF.round(5))
save_table(SCORE_DF, "model_comparison")
if FAILED_MODELS:
    print("Model gagal:", FAILED_MODELS)

fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(SCORE_DF) + 1)))
d = SCORE_DF.iloc[::-1]
ax.barh(d["model"], d[CFG.METRIC], xerr=d["fold_std"], color=["#DD8452" if f == "transformer" else
                                                                 "#55A868" if f == "embedding" else "#4C72B0"
                                                                 for f in d["family"]])
lo = np.nanmin(d[CFG.METRIC])
hi = np.nanmax(d[CFG.METRIC])
ax.set_xlim(max(0, lo - (hi - lo) * 0.5 - 0.01), hi + (hi - lo) * 0.3 + 0.01)
ax.set_title(f"Model comparison (OOF {CFG.METRIC}, error bar = fold std)")
ax.set_xlabel(f"OOF {CFG.METRIC}")
ax.set_ylabel("Model")
save_fig("model_comparison", fig)

# %% [markdown]
# ## 20. Ensemble
#
# Metode (`CFG.ENSEMBLE_METHODS`), semua dihitung dari OOF top-`ENSEMBLE_TOP_K` model:
#
# | metode | cara | catatan |
# |---|---|---|
# | `mean` | rata-rata probabilitas | paling aman, tidak bisa overfit |
# | `weighted` | bobot dioptimasi (scipy SLSQP) meminimalkan log-loss OOF | bisa sedikit overfit OOF |
# | `rank` | rata-rata **rank** prediksi, lalu dipetakan kembali ke skala probabilitas rata-rata | bagus untuk AUC & model dengan skala probabilitas berbeda; tidak dipakai untuk log_loss |
# | `stacking` | Logistic Regression di atas logit OOF semua model (CV fold yang sama) | paling fleksibel, paling rawan overfit pada data kecil |
#
# **Korelasi OOF antar model:** model dengan korelasi rendah (< 0.9) tapi skor bagus = pasangan ensemble terbaik.
# Korelasi > 0.98 = hampir identik, menambah sedikit.
#
# **Jika** ensemble tidak lebih baik dari model tunggal terbaik → pakai model tunggal (lebih sederhana untuk laporan).
# **Untuk laporan:** heatmap korelasi + tabel skor ensemble vs model terbaik.

# %%
cands = [k for k in SCORE_DF["model"] if not RESULTS[k]["partial"]][:CFG.ENSEMBLE_TOP_K]
need_test = HAS_TEST
cands = [k for k in cands if (RESULTS[k]["test"] is not None) or not need_test]
print("Ensemble candidates:", cands)


def flat(p):
    return np.asarray(p).reshape(len(p), -1)


if len(cands) >= 2:
    C = pd.DataFrame({k: flat(RESULTS[k]["oof"]).mean(1) if TASK != "binary" else RESULTS[k]["oof"] for k in cands})
    corr = C.corr(method="spearman")
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
    ax.set_title("Spearman correlation of OOF predictions")
    save_fig("oof_correlation", fig)


def blend(preds, w):
    w = np.asarray(w, dtype=float)
    w = w / max(w.sum(), EPS)
    return sum(wi * p for wi, p in zip(w, preds))


def optimize_weights(oofs):
    """Bobot non-negatif (jumlah 1) yang meminimalkan log-loss OOF."""
    n = len(oofs)

    def obj(w):
        return compute_score(Y, blend(oofs, w), "log_loss")
    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1},))
    w = res.x if res.success and np.all(np.isfinite(res.x)) else np.ones(n) / n
    return np.clip(w, 0, None) / max(np.clip(w, 0, None).sum(), EPS)


def rank_transform(p):
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        return rankdata(p) / len(p)
    return np.column_stack([rankdata(p[:, j]) / len(p) for j in range(p.shape[1])])


def rank_to_scale(rank_pred, ref_pred):
    """Petakan skor rank ke distribusi nilai ref_pred (urutan tetap, skala probabilitas kembali)."""
    r, ref = np.asarray(rank_pred, dtype=float), np.asarray(ref_pred, dtype=float)
    if r.ndim == 1:
        out = np.empty_like(r)
        out[np.argsort(r, kind="mergesort")] = np.sort(ref)
        return out
    return np.column_stack([rank_to_scale(r[:, j], ref[:, j]) for j in range(r.shape[1])])


def stack_features(preds):
    cols = []
    for p in preds:
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        cols.append((np.log(p) - np.log(1 - p)).reshape(len(p), -1))
    return np.hstack(cols)


def run_stacking(names):
    """Meta-model LogReg di atas logit OOF; OOF stacking memakai fold yang sama."""
    Xo = [RESULTS[k]["oof"] for k in names]
    Xt = [RESULTS[k]["test"] for k in names] if HAS_TEST else None
    if TASK == "multilabel":
        oof_s, te_s = np.zeros_like(Y, dtype=float), (np.zeros((len(test), NUM_CLASSES)) if HAS_TEST else None)
        for j in range(NUM_CLASSES):
            Fo = stack_features([p[:, j] for p in Xo])
            Ft = stack_features([p[:, j] for p in Xt]) if HAS_TEST else None
            for f in range(CFG.N_FOLDS):
                tr, va = FOLDS != f, FOLDS == f
                if len(np.unique(Y[tr, j])) < 2:
                    oof_s[va, j] = Y[tr, j].mean()
                    continue
                oof_s[va, j] = LogisticRegression(C=1.0, max_iter=2000).fit(Fo[tr], Y[tr, j]).predict_proba(Fo[va])[:, 1]
            if HAS_TEST:
                te_s[:, j] = LogisticRegression(C=1.0, max_iter=2000).fit(Fo, Y[:, j]).predict_proba(Ft)[:, 1] \
                    if len(np.unique(Y[:, j])) > 1 else Y[:, j].mean()
        return oof_s, te_s
    Fo = stack_features(Xo)
    Ft = stack_features(Xt) if HAS_TEST else None
    oof_s = np.zeros((len(Y), NUM_CLASSES))
    for f in range(CFG.N_FOLDS):
        tr, va = FOLDS != f, FOLDS == f
        m = LogisticRegression(C=1.0, max_iter=3000).fit(Fo[tr], Y[tr])
        oof_s[va] = align_proba(m.predict_proba(Fo[va]), m.classes_)
    te_s = None
    if HAS_TEST:
        m = LogisticRegression(C=1.0, max_iter=3000).fit(Fo, Y)
        te_s = to_task_output(align_proba(m.predict_proba(Ft), m.classes_))
    return to_task_output(oof_s), te_s


ENSEMBLE_WEIGHTS = {}
if CFG.RUN_ENSEMBLE and len(cands) >= 2:
    oofs = [RESULTS[k]["oof"] for k in cands]
    tests = [RESULTS[k]["test"] for k in cands] if HAS_TEST else None
    for method in CFG.ENSEMBLE_METHODS:
        try:
            if method == "mean":
                w = np.ones(len(cands)) / len(cands)
                if CFG.MANUAL_WEIGHTS:
                    w = np.array([CFG.MANUAL_WEIGHTS.get(k, 0.0) for k in cands]) + EPS
                o, t = blend(oofs, w), (blend(tests, w) if HAS_TEST else None)
            elif method == "weighted":
                w = optimize_weights(oofs)
                ENSEMBLE_WEIGHTS = {k: round(float(v), 4) for k, v in zip(cands, w)}
                print("Optimized weights:", ENSEMBLE_WEIGHTS)
                o, t = blend(oofs, w), (blend(tests, w) if HAS_TEST else None)
            elif method == "rank":
                if CFG.METRIC == "log_loss":
                    print("rank ensemble dilewati (tidak valid untuk log_loss)")
                    continue
                # rank-average, lalu dipetakan kembali ke skala probabilitas rata-rata (supaya threshold 0.5 tetap bermakna)
                o = rank_to_scale(np.mean([rank_transform(p) for p in oofs], axis=0), blend(oofs, np.ones(len(oofs))))
                t = rank_to_scale(np.mean([rank_transform(p) for p in tests], axis=0),
                                  blend(tests, np.ones(len(tests)))) if HAS_TEST else None
                if TASK == "multiclass":
                    o, t = o / o.sum(1, keepdims=True), (t / t.sum(1, keepdims=True) if HAS_TEST else None)
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
    print("Ensemble dilewati (RUN_ENSEMBLE=False atau < 2 model kandidat).")

# ---- pilih prediksi final
valid_final = {k: v for k, v in RESULTS.items() if not v["partial"] and (v["test"] is not None or not HAS_TEST)}
if CFG.FINAL_PREDICTION != "auto" and CFG.FINAL_PREDICTION in valid_final:
    FINAL_NAME = CFG.FINAL_PREDICTION
else:
    FINAL_NAME = max(valid_final, key=lambda k: valid_final[k]["score"] if GREATER_IS_BETTER else -valid_final[k]["score"])
best_single = max([k for k in valid_final if valid_final[k]["family"] != "ensemble"] or [FINAL_NAME],
                  key=lambda k: valid_final[k]["score"] if GREATER_IS_BETTER else -valid_final[k]["score"])
print(f"➡️ FINAL prediction: {FINAL_NAME} (OOF {CFG.METRIC} = {RESULTS[FINAL_NAME]['score']:.5f}) | "
      f"best single: {best_single} ({RESULTS[best_single]['score']:.5f})")
if RESULTS[FINAL_NAME]["family"] == "ensemble" and abs(RESULTS[FINAL_NAME]["score"] - RESULTS[best_single]["score"]) < CFG.MIN_TUNING_GAIN:
    print("   ℹ️ gain ensemble sangat kecil — model tunggal juga pilihan yang wajar (lebih mudah dijelaskan).")
FINAL_OOF = RESULTS[FINAL_NAME]["oof"]
FINAL_TEST = RESULTS[FINAL_NAME]["test"]

# %% [markdown]
# ## 21. Threshold / bias tuning
#
# - **Binary**: cari threshold 0.02–0.98 yang memaksimalkan metric (kalau metric berbasis label). Untuk data
#   imbalance, threshold optimal F1 sering jauh dari 0.5 (mis. 0.3).
# - **Multiclass**: tambahkan **bias per kelas** pada log-probabilitas (coordinate ascent) untuk macro-F1 —
#   efeknya mirip menurunkan threshold kelas minoritas.
# - **Multilabel**: threshold terpisah per label.
#
# **Anti-overfit:** dilakukan **cross-validated check** — tuning di K−1 fold, evaluasi di fold sisanya. Kalau gain
# CV-check < `MIN_TUNING_GAIN`, tuning **tidak dipakai**.
#
# **Jika** metric berbasis probabilitas (AUC, log_loss) → threshold tidak memengaruhi skor; tuning hanya untuk
# membuat label (dipakai kalau `SUBMISSION_MODE="label"`).
# **Untuk laporan:** plot metric vs threshold + threshold terpilih.

# %%
TUNE_METRIC = CFG.METRIC if METRIC_KIND == "label" else ("f1" if TASK == "binary" else
                                                          "f1_macro" if TASK == "multiclass" else "f1_micro")


def tune_decision(y, p):
    """Cari decision rule terbaik untuk TUNE_METRIC pada (y, p)."""
    sign = 1 if METRICS[TUNE_METRIC][1] else -1
    if TASK == "binary":
        grid = np.arange(0.02, 0.981, 0.01)
        scores = [sign * compute_score(y, p, TUNE_METRIC, t) for t in grid]
        return float(grid[int(np.nanargmax(scores))])
    if TASK == "multiclass":
        bias = np.zeros(NUM_CLASSES)
        best = sign * compute_score(y, p, TUNE_METRIC, bias)
        for _ in range(2):
            for c in range(NUM_CLASSES):
                for b in np.linspace(-2, 2, 21):
                    trial = bias.copy()
                    trial[c] = b
                    s = sign * compute_score(y, p, TUNE_METRIC, trial)
                    if s > best + 1e-9:
                        best, bias = s, trial
        return bias
    th = np.full(NUM_CLASSES, 0.5)
    for j in range(NUM_CLASSES):
        grid = np.arange(0.05, 0.951, 0.01)
        f1s = [skm.f1_score(y[:, j], (p[:, j] >= t).astype(int), zero_division=0) for t in grid]
        th[j] = grid[int(np.argmax(f1s))]
    return th


FINAL_DECISION = default_decision()
if CFG.TUNE_THRESHOLD:
    base = compute_score(Y, FINAL_OOF, TUNE_METRIC)
    tuned = tune_decision(Y, FINAL_OOF)
    full_gain = compute_score(Y, FINAL_OOF, TUNE_METRIC, tuned) - base
    cv_scores_def, cv_scores_tuned = [], []
    for f in range(CFG.N_FOLDS):
        tr, va = FOLDS != f, FOLDS == f
        d_f = tune_decision(Y[tr], FINAL_OOF[tr])
        cv_scores_def.append(compute_score(Y[va], FINAL_OOF[va], TUNE_METRIC))
        cv_scores_tuned.append(compute_score(Y[va], FINAL_OOF[va], TUNE_METRIC, d_f))
    cv_gain = np.nanmean(cv_scores_tuned) - np.nanmean(cv_scores_def)
    if not METRICS[TUNE_METRIC][1]:
        full_gain, cv_gain = -full_gain, -cv_gain
    print(f"Tuning for {TUNE_METRIC}: default={base:.5f} | in-sample gain={full_gain:+.5f} | "
          f"cross-validated gain={cv_gain:+.5f}")
    if cv_gain >= CFG.MIN_TUNING_GAIN:
        FINAL_DECISION = tuned
        print("✅ Tuning dipakai. Decision:", np.round(tuned, 3))
    else:
        print("ℹ️ Gain tidak cukup / tidak robust → decision default dipakai.")
    if TASK == "binary":
        grid = np.arange(0.02, 0.981, 0.01)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(grid, [compute_score(Y, FINAL_OOF, TUNE_METRIC, t) for t in grid])
        ax.axvline(float(FINAL_DECISION), color="r", ls="--", label=f"chosen = {float(FINAL_DECISION):.2f}")
        ax.set_title(f"OOF {TUNE_METRIC} vs decision threshold ({FINAL_NAME})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(TUNE_METRIC)
        ax.legend()
        save_fig("threshold_curve", fig)

FINAL_METRICS = evaluate_all(Y, FINAL_OOF, FINAL_DECISION)
print("Final OOF metrics (with chosen decision):")
display(pd.Series(FINAL_METRICS).round(5).to_frame("value"))
save_table(pd.Series(FINAL_METRICS).to_frame("value").reset_index(), "final_oof_metrics")
log_experiment(f"FINAL::{FINAL_NAME}", FINAL_METRICS[CFG.METRIC], {"decision": str(np.round(FINAL_DECISION, 3))})

# %% [markdown]
# ## 22. Error analysis
#
# **Apa yang dilihat:**
# 1. **Classification report** per kelas (precision, recall, F1, support) → kelas mana yang paling sulit.
# 2. **Confusion matrix** (jumlah & dinormalisasi per baris) → kelas mana yang saling tertukar.
# 3. **Most confident errors** → model yakin tapi salah: sering = **label salah** di data, sarkasme, atau negasi.
# 4. **Error per bucket panjang teks** → apakah model lemah di teks pendek (konteks kurang) atau panjang (terpotong)?
# 5. **Error per bahasa** (heuristik) dan **kalibrasi** (binary).
#
# **Jika X maka Y:**
# - Dua kelas sering tertukar → cek apakah definisinya tumpang tindih; tambahkan contoh/fitur khusus; bias tuning.
# - Recall kelas minoritas rendah → class weights / focal / threshold lebih rendah.
# - Error tinggi di teks panjang → naikkan `MAX_LEN` atau `LONG_TEXT_STRATEGY="chunk"`.
# - Error tinggi di teks pendek → TF-IDF char n-gram & ensemble membantu.
# - Banyak "confident errors" yang ternyata labelnya salah → label noise: `LABEL_SMOOTHING`, jangan overfit.
#
# **Untuk laporan:** confusion matrix (dinormalisasi) + 2–3 contoh error yang menarik beserta analisisnya.

# %%
pred_lab = proba_to_labels(FINAL_OOF, FINAL_DECISION)
if TASK != "multilabel":
    rep = pd.DataFrame(skm.classification_report(Y, pred_lab, labels=list(range(NUM_CLASSES)),
                                                 target_names=[str(c) for c in CLASSES], output_dict=True,
                                                 zero_division=0)).T
    display(rep.round(4))
    save_table(rep.reset_index(), "classification_report")
    cm = skm.confusion_matrix(Y, pred_lab, labels=list(range(NUM_CLASSES)))
    cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, axes = plt.subplots(1, 2, figsize=(min(16, 4 + 1.6 * NUM_CLASSES), min(7, 2.5 + 0.6 * NUM_CLASSES)))
    for ax, mat, title, fmt in [(axes[0], cm, "Confusion matrix (counts)", "d"),
                                (axes[1], cm_norm, "Confusion matrix (row-normalized)", ".2f")]:
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels([str(c) for c in CLASSES], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([str(c) for c in CLASSES], fontsize=8)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    save_fig("confusion_matrix", fig)
else:
    per_lab = pd.DataFrame({"label": CLASSES,
                            "f1": [skm.f1_score(Y[:, j], pred_lab[:, j], zero_division=0) for j in range(NUM_CLASSES)],
                            "precision": [skm.precision_score(Y[:, j], pred_lab[:, j], zero_division=0) for j in range(NUM_CLASSES)],
                            "recall": [skm.recall_score(Y[:, j], pred_lab[:, j], zero_division=0) for j in range(NUM_CLASSES)],
                            "support": Y.sum(0)})
    display(per_lab.round(4))
    save_table(per_lab, "multilabel_per_label_report")
    fig, ax = plt.subplots(figsize=(max(6, NUM_CLASSES * 0.7), 4))
    ax.bar(per_lab["label"].astype(str), per_lab["f1"], color="#4C72B0")
    ax.set_title("Per-label F1 (OOF)")
    ax.set_xlabel("Label")
    ax.set_ylabel("F1")
    save_fig("multilabel_per_label_f1", fig)

# ---- confident errors
ERR = pd.DataFrame({"text": train["raw_text"].values, "fold": FOLDS})
if CFG.ID_COL in train.columns:
    ERR.insert(0, CFG.ID_COL, train[CFG.ID_COL].values)
if TASK == "binary":
    ERR["true"] = decode_labels(Y)
    ERR["pred"] = decode_labels(pred_lab)
    ERR["p_positive"] = FINAL_OOF
    ERR["confidence"] = np.where(pred_lab == 1, FINAL_OOF, 1 - FINAL_OOF)
    ERR["correct"] = pred_lab == Y
elif TASK == "multiclass":
    ERR["true"] = decode_labels(Y)
    ERR["pred"] = decode_labels(pred_lab)
    ERR["confidence"] = FINAL_OOF[np.arange(len(Y)), pred_lab]
    ERR["p_true"] = FINAL_OOF[np.arange(len(Y)), Y]
    ERR["correct"] = pred_lab == Y
else:
    ERR["true"] = ["|".join(np.array(CLASSES)[r == 1]) for r in Y]
    ERR["pred"] = ["|".join(np.array(CLASSES)[r == 1]) for r in pred_lab]
    ERR["confidence"] = np.abs(FINAL_OOF - 0.5).mean(1) + 0.5
    ERR["correct"] = (pred_lab == Y).all(1)
ERR["n_words"] = train["text_tf"].str.split().str.len().values
ERR["language"] = LANG_SERIES.values
print(f"OOF row accuracy: {ERR['correct'].mean():.4f}")
print("Most confident ERRORS (cek: label salah? sarkasme? negasi?):")
display(ERR[~ERR["correct"]].sort_values("confidence", ascending=False).head(15))
save_table(ERR[~ERR["correct"]].sort_values("confidence", ascending=False).head(200), "top_confident_errors")

# ---- error by length bucket & language
try:
    ERR["len_bucket"] = pd.qcut(ERR["n_words"], q=min(5, ERR["n_words"].nunique()), duplicates="drop").astype(str)
except Exception:
    ERR["len_bucket"] = "all"
bucket = ERR.groupby("len_bucket", sort=False).agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                                  min_words=("n_words", "min")).sort_values("min_words")
display(bucket.round(4))
save_table(bucket.reset_index(), "error_by_length_bucket")
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(bucket.index.astype(str), bucket["accuracy"], color="#8172B2")
ax.set_ylim(max(0, bucket["accuracy"].min() - 0.1), 1)
ax.set_title("OOF accuracy by text length (word-count quintile)")
ax.set_xlabel("Word-count bucket")
ax.set_ylabel("Accuracy")
plt.xticks(rotation=20)
save_fig("accuracy_by_length", fig)
display(ERR.groupby("language").agg(n=("correct", "size"), accuracy=("correct", "mean")).round(4))

if TASK == "binary":
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(Y, np.clip(FINAL_OOF, 0, 1), n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(mean_pred, frac_pos, marker="o", label=FINAL_NAME)
    ax.set_title("Reliability diagram (OOF)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.legend()
    save_fig("calibration_curve", fig)

# %% [markdown]
# ## 23. Interpretability
#
# ### 23.1 Top n-gram berbobot dari Logistic Regression (global)
#
# LogReg pada TF-IDF word (1–2 gram) di-fit pada **seluruh** train. Koefisien positif besar = n-gram yang
# mendorong prediksi ke kelas tersebut. Ini penjelasan paling mudah dipahami juri → **sangat bagus untuk laporan**
# ("model belajar bahwa 'tidak sesuai', 'kecewa', 'rusak' adalah indikator ulasan negatif").
#
# ### 23.2 Penjelasan lokal (per contoh): token occlusion (+ SHAP opsional)
#
# Untuk beberapa contoh, tiap kata dihapus satu per satu lalu dilihat seberapa besar probabilitas kelas prediksi
# turun. Kata dengan penurunan terbesar = kata paling berpengaruh. Metode ini model-agnostic dan tidak butuh
# library tambahan. Kalau package `shap` terpasang, SHAP LinearExplainer juga dijalankan untuk model LogReg.
#
# **Jika** kata berpengaruh terlihat tidak masuk akal (nama orang, URL, template) → model belajar shortcut;
# bersihkan teks atau pertimbangkan bahwa test mungkin tidak punya pola yang sama.

# %%
INTERP_MODEL = None
if CFG.RUN_INTERPRETABILITY:
    try:
        vec_i = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=100000, sublinear_tf=True,
                                stop_words=list(STOPWORDS_ALL) if CFG.REMOVE_STOPWORDS_CLASSICAL else None)
        Xi = vec_i.fit_transform(train["text_cl"])
        names = np.array(vec_i.get_feature_names_out())
        coef_rows = []
        if TASK == "multilabel":
            targets = [(CLASSES[j], LogisticRegression(C=4, max_iter=2000).fit(Xi, Y[:, j]).coef_.ravel())
                       for j in range(NUM_CLASSES) if len(np.unique(Y[:, j])) > 1]
        else:
            lr_i = LogisticRegression(C=4, max_iter=3000).fit(Xi, Y)
            INTERP_MODEL = (vec_i, lr_i)
            if TASK == "binary":
                targets = [(CLASSES[1], lr_i.coef_.ravel()), (CLASSES[0], -lr_i.coef_.ravel())]
            else:
                targets = [(CLASSES[c], lr_i.coef_[k]) for k, c in enumerate(lr_i.classes_)]
        for lab, coef in targets:
            for r, j in enumerate(np.argsort(-coef)[:20]):
                coef_rows.append({"class": lab, "rank": r + 1, "ngram": names[j], "weight": float(coef[j])})
        COEF_DF = pd.DataFrame(coef_rows)
        save_table(COEF_DF, "logreg_top_ngrams_per_class")
        show = targets[:6]
        fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 5))
        for ax, (lab, coef) in zip(np.atleast_1d(axes), show):
            top = np.argsort(-coef)[:15][::-1]
            ax.barh(names[top], coef[top], color="#4C72B0")
            ax.set_title(f"Class: {lab}", fontsize=11)
            ax.set_xlabel("LogReg coefficient")
        fig.suptitle("Top weighted n-grams (TF-IDF + Logistic Regression)")
        save_fig("logreg_top_ngrams", fig)
    except Exception as e:
        print("Top n-gram interpretability dilewati:", short_err(e))


def occlusion_explain(text, predict_fn, target_idx, top_k=8):
    """Hapus tiap kata, ukur penurunan prob kelas target. Return list (kata, delta)."""
    words = str(text).split()
    if not words:
        return []
    base = predict_fn([" ".join(words)])[0][target_idx]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words[:200]))]
    probs = predict_fn(variants)[:, target_idx]
    deltas = base - probs
    order = np.argsort(-np.abs(deltas))[:top_k]
    return [(words[i], float(deltas[i])) for i in order]


if CFG.RUN_INTERPRETABILITY and INTERP_MODEL is not None:
    vec_i, lr_i = INTERP_MODEL

    def predict_interp(texts):
        cleaned = [classical_normalize(t) for t in texts]
        return align_proba(lr_i.predict_proba(vec_i.transform(cleaned)), lr_i.classes_)

    ex_idx = list(ERR[~ERR["correct"]].sort_values("confidence", ascending=False).index[:max(1, CFG.N_EXPLAIN_EXAMPLES // 2)])
    ex_idx += list(ERR[ERR["correct"]].sort_values("confidence", ascending=False).index[:CFG.N_EXPLAIN_EXAMPLES - len(ex_idx)])
    for k, i in enumerate(ex_idx):
        txt = train["raw_text"].iloc[i]
        tgt = int(np.argmax(predict_interp([txt])[0]))
        expl = occlusion_explain(txt, predict_interp, tgt)
        print(f"\n[{k + 1}] true={decode_labels([Y[i]])[0]!r} | interp-model pred={CLASSES[tgt]!r}")
        print("    text:", str(txt)[:300])
        print("    kata berpengaruh (Δprob jika kata dihapus):", [(w, round(d, 3)) for w, d in expl])
        if k == 0 and expl:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            w_, d_ = zip(*expl[::-1])
            ax.barh(w_, d_, color=["#C44E52" if v > 0 else "#4C72B0" for v in d_])
            ax.set_title(f"Token occlusion — predicted class '{CLASSES[tgt]}'")
            ax.set_xlabel("Drop in probability when word removed")
            save_fig("occlusion_example", fig)
    if CFG.USE_SHAP and HAS_SHAP:
        try:
            Xs = vec_i.transform(train["text_cl"].iloc[ex_idx])
            explainer = shap.LinearExplainer(lr_i, vec_i.transform(train["text_cl"].iloc[:min(500, len(train))]))
            sv = explainer.shap_values(Xs)
            sv0 = sv[0] if isinstance(sv, list) else (sv[..., -1] if np.ndim(sv) == 3 else sv)
            sv0 = np.asarray(sv0.todense() if sparse.issparse(sv0) else sv0)
            fn = np.array(vec_i.get_feature_names_out())
            for r in range(min(len(ex_idx), sv0.shape[0])):
                top = np.argsort(-np.abs(sv0[r]))[:8]
                print(f"SHAP example {r + 1}:", [(fn[j], round(float(sv0[r, j]), 3)) for j in top])
        except Exception as e:
            print("SHAP dilewati:", short_err(e))
    elif CFG.USE_SHAP:
        print("ℹ️ shap tidak terpasang → hanya occlusion (opsional: pip install shap).")

# %% [markdown]
# ## 24. Submission + validator
#
# **Format:**
# - `SUBMISSION_MODE="label"` → label asli (string/angka sesuai train) memakai decision rule hasil tuning.
# - `SUBMISSION_MODE="probability"` → binary: probabilitas kelas positif; multiclass/multilabel: satu kolom per
#   kelas (kalau sample_submission punya kolom bernama kelas, kolom itu yang diisi).
# - Multilabel label mode: format kolom 0/1 (kalau `TARGET_COLS`) atau string gabungan dengan `MULTILABEL_SEP`.
#
# **Validator** mengecek terhadap sample_submission: jumlah baris, nama & urutan kolom, urutan ID, tidak ada NaN,
# nilai label valid / probabilitas di [0, 1]. Semua harus ✅ sebelum upload.
#
# **Error umum:** urutan ID beda → sudah ditangani dengan merge berdasarkan ID; tipe label beda (mis. `1` vs `"1"`)
# → cek output validator kolom `label values`.

# %%
def build_prediction_frame(test_pred, decision):
    """DataFrame prediksi test (tanpa ID) sesuai SUBMISSION_MODE."""
    target_col = CFG.SUBMISSION_TARGET_COL or CFG.TARGET_COL
    if CFG.SUBMISSION_MODE == "probability":
        if TASK == "binary":
            return pd.DataFrame({target_col: test_pred})
        return pd.DataFrame(test_pred, columns=[str(c) for c in CLASSES])
    labels = proba_to_labels(test_pred, decision)
    if TASK == "multilabel":
        if CFG.TARGET_COLS:
            return pd.DataFrame(labels, columns=CLASSES)
        sep = CFG.MULTILABEL_SEP or "|"
        return pd.DataFrame({target_col: [sep.join(np.array(CLASSES)[r == 1]) for r in labels]})
    return pd.DataFrame({target_col: decode_labels(labels)})


def validate_submission(sub, sample=None):
    """Cek format submission. Return True jika semua lolos."""
    checks = [("no NaN", not sub.isna().any().any()), ("rows > 0", len(sub) > 0)]
    if sample is not None:
        checks += [("same n_rows as sample", len(sub) == len(sample)),
                   ("same columns & order", list(sub.columns) == list(sample.columns))]
        if CFG.ID_COL in sample.columns and CFG.ID_COL in sub.columns:
            checks.append(("same ID order", (sub[CFG.ID_COL].astype(str).values == sample[CFG.ID_COL].astype(str).values).all()))
    elif HAS_TEST:
        checks.append(("same n_rows as test", len(sub) == len(test)))
    pred_cols = [c for c in sub.columns if c != CFG.ID_COL]
    if CFG.SUBMISSION_MODE == "probability":
        vals = sub[pred_cols].apply(pd.to_numeric, errors="coerce")
        checks.append(("probabilities in [0,1]", bool(((vals >= 0) & (vals <= 1)).all().all())))
    elif TASK != "multilabel":
        allowed = set(map(str, CLASSES))
        checks.append(("label values valid", set(sub[pred_cols[0]].astype(str)) <= allowed))
    ok = all(v for _, v in checks)
    display(pd.DataFrame([(n, "✅" if v else "❌") for n, v in checks], columns=["check", "status"]))
    print("SUBMISSION VALID ✅" if ok else "❌ SUBMISSION TIDAK VALID — perbaiki sebelum upload!")
    return ok


SUBMISSION = None
if HAS_TEST and FINAL_TEST is not None:
    pred_df = build_prediction_frame(FINAL_TEST, FINAL_DECISION)
    if sample_sub is not None and CFG.ID_COL in sample_sub.columns and CFG.ID_COL in test.columns:
        pred_df.insert(0, CFG.ID_COL, test[CFG.ID_COL].values)
        pred_cols_s = [c for c in sample_sub.columns if c != CFG.ID_COL]
        own_cols = [c for c in pred_df.columns if c != CFG.ID_COL]
        if len(pred_cols_s) == len(own_cols) and set(pred_cols_s) != set(own_cols):
            pred_df = pred_df.rename(columns=dict(zip(own_cols, pred_cols_s)))
            print(f"ℹ️ Kolom prediksi di-rename agar sama dengan sample_submission: {pred_cols_s}")
        SUBMISSION = sample_sub[[CFG.ID_COL]].merge(pred_df, on=CFG.ID_COL, how="left")
        SUBMISSION = SUBMISSION[[c for c in sample_sub.columns if c in SUBMISSION.columns]]
    else:
        if CFG.ID_COL in test.columns:
            pred_df.insert(0, CFG.ID_COL, test[CFG.ID_COL].values)
        SUBMISSION = pred_df
    display(SUBMISSION.head())
    SUB_OK = validate_submission(SUBMISSION, sample_sub)
    sub_path = os.path.join(CFG.OUTPUT_DIR, "submissions", "submission.csv")
    SUBMISSION.to_csv(sub_path, index=False)
    tag = f"{FINAL_NAME}_{CFG.METRIC}{RESULTS[FINAL_NAME]['score']:.4f}_{time.strftime('%m%d_%H%M')}"
    SUBMISSION.to_csv(os.path.join(CFG.OUTPUT_DIR, "submissions", f"submission_{tag}.csv"), index=False)
    print("Saved:", sub_path)
    if TASK != "multilabel" and CFG.SUBMISSION_MODE == "label":
        print("Distribusi label prediksi test vs train:")
        display(pd.concat([SUBMISSION.iloc[:, -1].value_counts(normalize=True).rename("test_pred"),
                           pd.Series(decode_labels(Y)).value_counts(normalize=True).rename("train")], axis=1).round(3))
else:
    print("Test tidak ada → submission dilewati (hasil CV tetap tersimpan).")

# %% [markdown]
# ## 25. Simpan OOF, prediksi test, konfigurasi & ringkasan untuk laporan
#
# File yang disimpan:
# - `preds/oof_<model>.npy`, `preds/test_<model>.npy` → semua model (untuk ensemble ulang tanpa training ulang),
# - `oof_final.csv`, `test_pred_final.csv` → prediksi final dengan ID,
# - `cfg_used.json` → konfigurasi yang dipakai (reproducibility),
# - `experiment_log.csv` → riwayat semua run.
#
# **Untuk laporan:** cetakan "REPORT SUMMARY" di bawah berisi angka-angka kunci yang bisa langsung dipakai.

# %%
oof_out = pd.DataFrame({"fold": FOLDS})
if CFG.ID_COL in train.columns:
    oof_out.insert(0, CFG.ID_COL, train[CFG.ID_COL].values)
if TASK == "multilabel":
    for j, c in enumerate(CLASSES):
        oof_out[f"true_{c}"] = Y[:, j]
        oof_out[f"prob_{c}"] = FINAL_OOF[:, j]
else:
    oof_out["true"] = decode_labels(Y)
    oof_out["pred"] = decode_labels(pred_lab)
    if TASK == "binary":
        oof_out["prob_positive"] = FINAL_OOF
    else:
        for j, c in enumerate(CLASSES):
            oof_out[f"prob_{c}"] = FINAL_OOF[:, j]
oof_out.to_csv(os.path.join(CFG.OUTPUT_DIR, "oof_final.csv"), index=False)
if HAS_TEST and FINAL_TEST is not None:
    te_out = pd.DataFrame(np.asarray(FINAL_TEST).reshape(len(test), -1),
                          columns=["prob_positive"] if TASK == "binary" else [f"prob_{c}" for c in CLASSES])
    if CFG.ID_COL in test.columns:
        te_out.insert(0, CFG.ID_COL, test[CFG.ID_COL].values)
    te_out.to_csv(os.path.join(CFG.OUTPUT_DIR, "test_pred_final.csv"), index=False)

cfg_dump = {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict, tuple)) else str(v))
            for k, v in vars(CFG).items() if not k.startswith("_")}
cfg_dump.update({"_TASK": TASK, "_CLASSES": [str(c) for c in CLASSES], "_FINAL_MODEL": FINAL_NAME,
                 "_FINAL_DECISION": np.asarray(FINAL_DECISION).tolist()})
with open(os.path.join(CFG.OUTPUT_DIR, "cfg_used.json"), "w", encoding="utf-8") as f:
    json.dump(cfg_dump, f, indent=2, default=str)

print("=" * 80)
print("REPORT SUMMARY")
print("=" * 80)
print(f"Data         : {len(train)} train rows" + (f", {len(test)} test rows" if HAS_TEST else ", no test file"))
print(f"Task         : {TASK} with {NUM_CLASSES} classes {CLASSES[:10]}")
print(f"Imbalance    : max/min = {IMBALANCE_RATIO:.2f}")
print(f"Language     : {dict((k, round(v, 3)) for k, v in lang_share.items())} → model {CFG.TRANSFORMER_MODELS}")
print(f"MAX_LEN      : {CFG.MAX_LEN} (p{CFG.MAX_LEN_PERCENTILE} token length)")
print(f"Validation   : {CFG.N_FOLDS}-fold {CV_STRATEGY_USED} CV, metric = {CFG.METRIC}")
print("Top models   :")
for _, r in SCORE_DF.head(5).iterrows():
    print(f"   {r['model']:<40s} {CFG.METRIC}={r[CFG.METRIC]:.5f} ± {r['fold_std']:.4f}")
print(f"Final        : {FINAL_NAME} | OOF {CFG.METRIC} = {FINAL_METRICS[CFG.METRIC]:.5f} | decision = {np.round(FINAL_DECISION, 3)}")
print(f"Failed       : {list(FAILED_MODELS) or 'none'}")
print(f"Figures      : {len(os.listdir(os.path.join(CFG.OUTPUT_DIR, 'figures')))} files in {os.path.join(CFG.OUTPUT_DIR, 'figures')}")

# %% [markdown]
# ## 26. Catatan multilabel (sudah didukung)
#
# Multilabel = satu teks bisa punya **beberapa** label sekaligus (mis. ulasan menyebut *harga* dan *pengiriman*).
#
# **Cara mengaktifkan:**
# ```python
# CFG.TASK_TYPE = "multilabel"
# # format A: satu kolom berisi "harga|pengiriman"
# CFG.TARGET_COL = "aspects"; CFG.MULTILABEL_SEP = "|"
# # format B: beberapa kolom 0/1
# CFG.TARGET_COLS = ["harga", "pengiriman", "kualitas"]
# CFG.METRIC = "f1_micro"          # atau "f1_macro", "f1_samples", "roc_auc_macro"
# ```
#
# **Apa yang berubah otomatis:**
# - Target jadi matriks 0/1 `(n, L)`; output model = **sigmoid per label** (bukan softmax).
# - Transformer memakai `BCEWithLogitsLoss` (+ `pos_weight` kalau `USE_CLASS_WEIGHTS=True`).
# - Classical/embedding: satu model biner per label (one-vs-rest).
# - Threshold di-tuning **per label** (section 21).
# - Fold: stratifikasi berdasarkan label paling langka di tiap baris (pendekatan sederhana; package
#   `iterative-stratification` lebih presisi kalau tersedia).
# - Submission: format kolom 0/1 (format B) atau string gabungan (format A).
#
# **Tips:** label sangat langka (< 20 contoh) → threshold per label tidak stabil; pertimbangkan threshold global.

# %% [markdown]
# ## 27. Experiment recipes
#
# **R1 — Laptop tanpa GPU (cepat & aman)**
# ```python
# CFG.RUN_TRANSFORMER = False
# CFG.CLASSICAL_MODELS = ["logreg", "linearsvc", "nbsvm", "complement_nb"]
# CFG.EMBEDDING_MODELS = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
# ```
#
# **R2 — Bahasa Indonesia sosmed (tweet, komentar)**
# ```python
# CFG.TRANSFORMER_MODELS = ["indobertweet"]; CFG.MAX_LEN = 128; CFG.BATCH_SIZE = 32
# CFG.USE_SLANG_NORMALIZATION = True     # untuk TF-IDF
# CFG.TRANSFORMER_LOWERCASE = True       # indobertweet uncased
# ```
#
# **R3 — Bahasa Indonesia formal (berita, dokumen)**
# ```python
# CFG.TRANSFORMER_MODELS = ["indobert_base"]; CFG.MAX_LEN = 256; CFG.LONG_TEXT_STRATEGY = "head_tail"
# ```
#
# **R4 — Campuran ID-EN / multibahasa**
# ```python
# CFG.TRANSFORMER_MODELS = ["xlm_roberta_base"]; CFG.LR = 2e-5; CFG.LLRD = 0.9
# CFG.EMBEDDING_MODELS = ["intfloat/multilingual-e5-small"]
# ```
#
# **R5 — English**
# ```python
# CFG.TRANSFORMER_MODELS = ["deberta_v3_base", "modernbert_base"]   # deberta butuh sentencepiece
# ```
#
# **R6 — Imbalance berat (rasio > 10)**
# ```python
# CFG.METRIC = "f1_macro"; CFG.USE_CLASS_WEIGHTS = True; CFG.LOSS = "focal"; CFG.TUNE_THRESHOLD = True
# ```
#
# **R7 — Dokumen panjang (> 512 token)**
# ```python
# CFG.LONG_TEXT_STRATEGY = "chunk"; CFG.MAX_LEN = 512; CFG.CHUNK_OVERLAP = 128; CFG.MAX_CHUNKS_PER_DOC = 6
# # atau model konteks panjang:
# CFG.TRANSFORMER_MODELS = ["modernbert_base"]; CFG.MAX_LEN = 1024; CFG.LONG_TEXT_STRATEGY = "truncate"
# ```
#
# **R8 — GPU kecil (4–6 GB)**
# ```python
# CFG.BATCH_SIZE = 8; CFG.GRAD_ACCUM_STEPS = 2; CFG.USE_AMP = True; CFG.GRADIENT_CHECKPOINTING = True
# CFG.MAX_LEN = 256
# ```
#
# **R9 — Data kecil (< 2k baris), skor fold tidak stabil**
# ```python
# CFG.N_FOLDS = 10; CFG.MULTI_SAMPLE_DROPOUT = 5; CFG.LLRD = 0.9; CFG.POOLING = "mean"; CFG.EPOCHS = 4
# ```
#
# **R10 — Final push (waktu cukup)**
# ```python
# CFG.RUN_MODE = "full"; CFG.TRANSFORMER_MODELS = ["indobert_base", "xlm_roberta_base"]
# CFG.ENSEMBLE_METHODS = ["mean", "weighted", "rank", "stacking"]
# ```
#
# **R11 — Cek cepat satu fold transformer**
# ```python
# CFG.FOLDS_TO_RUN = [0]; CFG.EPOCHS = 1     # lihat waktu & VRAM, lalu kembalikan ke None
# ```

# %% [markdown]
# ## 28. Troubleshooting
#
# | Error / gejala | Penyebab | Solusi |
# |---|---|---|
# | `FileNotFoundError` | path data salah | cek `os.getcwd()`, pakai path absolut, cek nama file |
# | `KeyError: 'text'` / kolom tidak ada | nama kolom beda | set `CFG.TEXT_COL` / `TARGET_COL` di USER OVERRIDE |
# | `UnicodeDecodeError` | encoding file | `smart_read` sudah mencoba utf-8/utf-8-sig/latin-1; untuk Excel simpan ulang sebagai CSV UTF-8 |
# | Semua kolom terbaca jadi satu | separator `;` | otomatis dideteksi; kalau gagal `pd.read_csv(path, sep=";")` |
# | `CUDA out of memory` | batch/MAX_LEN terlalu besar | auto-retry (batch/2); manual: `BATCH_SIZE=8`, `MAX_LEN=256`, `GRADIENT_CHECKPOINTING=True`, model base |
# | Loss `nan` / skor tiba-tiba 0 | fp16 overflow / LR terlalu besar | `USE_AMP=False`, `LR=1e-5`, `MAX_GRAD_NORM=1.0` |
# | `ValueError: Couldn't instantiate the backend tokenizer` / butuh `sentencepiece`/`tiktoken` | DeBERTa-v3, XLM-R, mDeBERTa butuh sentencepiece | `pip install sentencepiece protobuf`, **restart kernel** |
# | `OSError: ... is not a valid model identifier` / `ConnectionError` | internet/HF diblokir atau nama salah | cek nama di huggingface.co; offline: download dulu (`save_pretrained`) lalu pakai path folder; `HF_HUB_OFFLINE=1` |
# | Warning `Some weights of ... were not initialized` / `newly initialized` | head klasifikasi baru | **normal**, abaikan |
# | `ImportError: ... requires accelerate` | HF Trainer tanpa accelerate | otomatis fallback ke `custom_loop`; atau `pip install accelerate` |
# | `TypeError: __init__() got an unexpected keyword argument 'evaluation_strategy'` | versi transformers baru | sudah di-shim (`eval_strategy`), pakai `custom_loop` |
# | `TypeError: compute_loss() got an unexpected keyword argument 'num_items_in_batch'` | versi transformers ≥ 4.46 | sudah ditangani (signature menerima `num_items_in_batch`) |
# | Transformer sangat lambat | jalan di CPU | `RUN_TRANSFORMER="auto"`/False; pakai classical + embedding |
# | Skor transformer < TF-IDF | MAX_LEN terlalu kecil, model bahasa salah, LR/epoch | cek EDA token length, ganti model sesuai bahasa, `LR=2e-5`, `EPOCHS=3–4` |
# | Skor fold sangat bervariasi | data kecil / duplikat bocor | `GROUP_COL="dup_group"`, lebih banyak fold, ensemble |
# | CV bagus, leaderboard jelek | leakage / drift / threshold overfit | cek adversarial validation, duplikat, jangan pakai tuning dengan gain kecil |
# | `MemoryError` saat TF-IDF | fitur char terlalu banyak | turunkan `TFIDF_CHAR_MAX_FEATURES`, `USE_CHAR_TFIDF=False` |
# | `ValueError: empty vocabulary` | teks kosong/terlalu pendek | cek `TEXT_COL`; helper otomatis `min_df=1` |
# | Warning `least populated class has only 1 member` | kelas sangat langka | gabung kelas langka atau turunkan `N_FOLDS` |
# | `ValueError: Input contains NaN` di model | prediksi NaN | cek FAILED_MODELS; sumber biasanya loss NaN transformer |
# | Submission ditolak: jumlah baris/kolom beda | ID/kolom tidak sama dengan sample | lihat output validator; set `SUBMISSION_TARGET_COL` |
# | Label submission salah tipe (`1.0` vs `1`) | target numerik dibaca float | pastikan target train bertipe benar sebelum encode |
# | `pip install` gagal | proxy/offline/izin | `CFG.AUTO_INSTALL=False`, install manual atau pakai yang ada |
# | `DataLoader worker exited unexpectedly` (Windows) | multiprocessing | `CFG.NUM_WORKERS = 0` |
# | Embedding E5 jelek | lupa prefix | otomatis `"query: "`; cek `EMBEDDING_PREFIXES` |
# | `RuntimeError: Expected all tensors to be on the same device` | bug kustom kode | restart kernel, jalankan dari atas |
# | Kernel mati (dead kernel) | RAM habis | turunkan `NEAR_DUP_MAX_ROWS`, TF-IDF features, `EMB_BATCH_SIZE` |

# %% [markdown]
# ## 29. Checklist sebelum submit
#
# - [ ] `RUN_MODE = "full"` (bukan "fast") dan semua cell jalan dari atas tanpa error.
# - [ ] `METRIC` = metric resmi lomba; kelas positif (binary) sudah benar.
# - [ ] Tidak ada model penting di `FAILED_MODELS`.
# - [ ] Skor OOF masuk akal & fold std kecil; tidak ada tanda leakage (ID, duplikat, adversarial AUC).
# - [ ] Threshold/bias hanya dipakai kalau cross-validated gain cukup.
# - [ ] Validator submission semua ✅ (jumlah baris, kolom, urutan ID, tidak ada NaN, nilai valid).
# - [ ] Distribusi label prediksi test masuk akal dibanding train.
# - [ ] File `submission.csv` terbaru yang di-upload (cek timestamp di `submissions/`).
# - [ ] Figure untuk laporan sudah ada di `figures/` (DPI 300 untuk final: `CFG.FIG_DPI = 300`).
# - [ ] Catat konfigurasi final (`cfg_used.json`) & skor di laporan.
