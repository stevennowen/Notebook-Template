# -*- coding: utf-8 -*-
# =====================================================================================================
# cv_image_classification.py — versi SCRIPT dari cv_image_classification.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python cv_image_classification.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb cv_image_classification.py
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
# # Computer Vision Competition Template — Image Classification
#
# Template lengkap untuk lomba **klasifikasi gambar** (binary / multiclass). Notebook ini dirancang supaya bisa
# dijalankan **sendirian dari atas ke bawah**, di laptop siapa pun (CPU-only atau GPU), tanpa bantuan siapa-siapa.
#
# ---
#
# ## Pipeline besar
#
# ```text
# CFG + USER OVERRIDE
# → Install / cek library (timm, torchvision, lightgbm opsional)
# → Deteksi layout data otomatis (CSV+folder / ImageFolder / test folder / pixel CSV ala MNIST)
# → Sanity check data (file hilang, label kosong)
# → Image EDA: distribusi kelas, contoh gambar, ukuran & aspect ratio, mode channel,
#   file korup, duplikat (aHash), mean/std dataset, brightness, train-vs-test shift
# → Cross-validation folds (stratified, duplicate-aware)
# → Transforms + augmentasi (torchvision ATAU fallback PIL buatan sendiri)
# → Overfit-one-batch test (cek pipeline training benar)
# → Baseline cepat: frozen features / pixel features + LogisticRegression / LightGBM
# → Deep learning: timm → torchvision → SimpleCNN (pure PyTorch) — K-fold, AMP, AdamW, cosine+warmup,
#   label smoothing, mixup/cutmix, class weights, early stopping, freeze→unfreeze, OOM auto-retry
# → Inference dengan TTA (hflip + multi-scale) + rata-rata antar fold
# → Ensemble (hill climbing di OOF) + threshold / class-weight optimization
# → Evaluasi: confusion matrix, per-class report, gambar salah prediksi, confidence histogram, Grad-CAM
# → Submission (cocok dengan sample_submission) + validator
# → Simpan model, OOF, test prediction, experiment log
# → Troubleshooting, Recipes, Checklist
# ```
#
# ---
#
# ## Cara pakai cepat (5 langkah)
#
# 1. Taruh data di folder `./data/` (lihat 4 layout yang didukung di bawah).
# 2. Buka cell **`# >>> USER OVERRIDE`** (tepat setelah `CFG`) dan isi path / nama kolom / metric lomba.
#    Kalau layout standar, cukup biarkan `DATA_LAYOUT = "auto"`.
# 3. Set `CFG.RUN_MODE = "fast"` dulu → jalankan semua cell (Run All). Pastikan tidak ada error dan hasil masuk akal.
# 4. Ganti ke `CFG.RUN_MODE = "full"` → Run All lagi untuk hasil final.
# 5. Ambil file submission di `outputs/cv_image_classification/submissions/` dan figure untuk laporan di `figures/`.
#
# ## 4 layout data yang didukung (`CFG.DATA_LAYOUT = "auto"` mendeteksi sendiri)
#
# | Layout | Struktur | Contoh lomba |
# |---|---|---|
# | `csv` | `train.csv` berisi kolom nama file/path gambar + kolom label, gambar di folder (`./data/train/`, `./data/train_images/`, `./data/images/`) | Kaggle "Plant Pathology", "Cassava" |
# | `folder` | `./data/train/<nama_kelas>/*.jpg` (ImageFolder) | dataset Keras / ImageNet-style |
# | test tanpa label | `./data/test/*.jpg` saja (tanpa CSV) — id = nama file | banyak lomba lokal |
# | `pixels` | `train.csv` berisi kolom `label` + kolom pixel (`pixel0..pixel783`) → di-reshape jadi gambar | Kaggle "Digit Recognizer" (MNIST), Fashion-MNIST |
#
# ## Kalau tidak punya GPU
#
# Notebook tetap jalan. Pakai recipe **"CPU-only quick"** di bagian akhir: `IMG_SIZE=128`, model kecil
# (`mobilenetv3_large_100` / `resnet18`) dan andalkan **baseline frozen features + LogisticRegression**,
# yang sering sudah sangat kompetitif dan hanya butuh beberapa menit.
#
# ## Kalau `timm` / `torchvision` tidak ada
#
# Notebook otomatis turun ke fallback: model **SimpleCNN** (pure PyTorch) + augmentasi buatan sendiri (PIL/numpy).
# Kalau bahkan `torch` tidak ada, bagian deep learning di-skip dan submission tetap dibuat dari baseline pixel features.

# %% [markdown]
# # 1. Config (`CFG`)
#
# **Tujuan:** semua "kenop" notebook ada di satu tempat. Jangan edit angka di tengah-tengah notebook — ubah di sini
# atau (lebih baik) di cell `USER OVERRIDE` setelahnya.
#
# **Yang paling sering diubah:**
#
# - Path: `TRAIN_CSV`, `TEST_CSV`, `SAMPLE_SUB_PATH`, `TRAIN_IMAGE_DIR`, `TEST_IMAGE_DIR`
# - Kolom: `IMAGE_COL`, `TARGET_COL`, `ID_COL` (biarkan `None` = auto-detect)
# - `METRIC` sesuai rules lomba, `SUBMISSION_MODE` (label atau probabilitas)
# - `MODEL_POOL`, `IMG_SIZE`, `EPOCHS`, `BATCH_SIZE`
#
# **Jika** laptop panitia CPU-only **maka** turunkan `IMG_SIZE` ke 128, `EPOCHS` ke 5–8, pakai `mobilenetv3_large_100`.
# **Jika** GPU kecil (4–6 GB) **maka** `BATCH_SIZE=16–32` + `USE_AMP=True`; kalau OOM, notebook otomatis mengecilkan batch.

# %%
import os

class CFG:
    # =====================================================================
    # MODE JALAN
    # =====================================================================
    # "fast" = debug cepat: subsample data, fold & epoch sedikit, gambar kecil, 1 model.
    # "full" = run final. Selalu jalankan "fast" dulu sampai tidak ada error.
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "cv_image_classification"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"          # True: coba pip install library opsional yang hilang (TIDAK pernah install torch) (env TEMPLATE_AUTO_INSTALL=0 → off)
    INSTALL_TIMM = True          # timm di-install dengan --no-deps supaya TIDAK mengganti versi torch yang ada

    # =====================================================================
    # PATH DATA
    # =====================================================================
    # "auto" | "csv" | "folder" | "pixels"  (lihat tabel layout di atas)
    DATA_LAYOUT = "auto"
    TRAIN_CSV = "./data/train.csv"
    TEST_CSV = "./data/test.csv"                     # boleh tidak ada
    SAMPLE_SUB_PATH = "./data/sample_submission.csv" # boleh tidak ada (tapi sangat disarankan)
    TRAIN_IMAGE_DIR = "./data/train"                 # csv: folder gambar train | folder: root berisi subfolder kelas
    TEST_IMAGE_DIR = "./data/test"                   # folder gambar test
    # Folder lain yang dicoba otomatis kalau gambar tidak ketemu di TRAIN_IMAGE_DIR / TEST_IMAGE_DIR
    IMAGE_DIR_CANDIDATES = ["./data/train_images", "./data/images", "./data/train", "./data/imgs"]
    TEST_IMAGE_DIR_CANDIDATES = ["./data/test_images", "./data/test", "./data/images"]
    EXTRA_TRAIN_DIRS = []        # layout folder: folder tambahan berlabel (mis. "./data/val") yang digabung ke train
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".pgm")

    # =====================================================================
    # KOLOM (None = auto-detect)
    # =====================================================================
    IMAGE_COL = None     # kolom nama file / path gambar di train.csv (mis. "image_id", "filename")
    TEST_IMAGE_COL = None  # kalau nama kolom gambar di test.csv beda dari train
    TARGET_COL = None    # kolom label (mis. "label", "class")
    ID_COL = None        # kolom id di sample_submission/test (mis. "id", "image_id"); None = kolom pertama sample_sub
    SUB_TARGET_COL = None  # nama kolom target di submission; None = kolom ke-2 sample_sub / TARGET_COL
    GROUP_COL = None     # kolom grup (mis. "patient_id") → GroupKFold supaya gambar 1 pasien tidak bocor antar fold

    # Layout "pixels" (MNIST-like)
    PIXEL_SHAPE = None   # None = auto (sqrt jumlah pixel). Contoh: (28, 28) atau (32, 32, 3)
    PIXEL_PREFIX = None  # mis. "pixel" → hanya kolom yang diawali "pixel" dianggap pixel
    PIXEL_CHANNEL_FIRST = False  # True kalau urutan pixel (C,H,W) seperti CIFAR mentah

    # =====================================================================
    # OUTPUT
    # =====================================================================
    OUTPUT_DIR = "./outputs/cv_image_classification/"
    FIG_DPI = 200        # 300 untuk figure laporan final

    # =====================================================================
    # TASK & METRIC
    # =====================================================================
    # "accuracy" | "f1_macro" | "f1" (binary, kelas positif = kelas ke-2 setelah sort) | "logloss" | "roc_auc" | "qwk"
    METRIC = "f1_macro"
    # "label" = submit nama kelas | "proba" = submit probabilitas (binary: 1 kolom P(kelas positif); multiclass: 1 kolom/kelas)
    SUBMISSION_MODE = "label"
    OPTIMIZE_DECISION = True     # optimasi threshold (binary) / class weight (multiclass) di OOF untuk metric label-based
    MIN_DECISION_GAIN = 0.002    # hanya pakai hasil optimasi kalau naik >= ini (anti-overfit OOF)

    # =====================================================================
    # EDA
    # =====================================================================
    RUN_EDA = True
    SCAN_MAX_IMAGES = None       # None = scan semua gambar (cek file korup). Isi angka kalau dataset sangat besar.
    STAT_SIZE = 64               # ukuran thumbnail untuk statistik (mean/std, brightness, hash)
    LOAD_TRUNCATED_IMAGES = True # True: gambar JPEG terpotong tetap bisa dibaca (sisa diisi abu-abu)
    DROP_TRUNCATED = False       # True: gambar terpotong dibuang dari train
    NEAR_DUP_HAMMING = 0         # 0 = hanya duplikat persis (hash sama). 2–4 = near-duplicate (lebih lambat)
    NEAR_DUP_MAX_N = 6000        # batas jumlah gambar untuk near-duplicate check (O(N^2))
    DROP_DUPLICATES = False      # True: buang duplikat persis di train (sisakan 1); label konflik dibuang
    DUP_AWARE_CV = True          # True: gambar duplikat dipaksa di fold yang sama (CV lebih jujur)
    GRID_PER_CLASS = 6           # jumlah contoh gambar per kelas di sample grid
    MAX_CLASSES_IN_GRID = 12

    # =====================================================================
    # BACKBONE / MODEL
    # =====================================================================
    USE_TIMM = True              # False = paksa tidak pakai timm walau terinstall
    USE_TORCHVISION = True       # False = paksa tidak pakai torchvision (transform & model)
    USE_GPU = True
    # Nama model gaya timm. Kalau timm tidak ada → dicari padanan torchvision → kalau tidak ada → "simple_cnn".
    # Pilihan umum (dari ringan ke berat):
    #   "mobilenetv3_large_100" (CPU-friendly), "resnet18", "efficientnet_b0", "resnet50",
    #   "convnext_tiny", "vit_small_patch16_224", "eva02_small_patch14_224.mim_in22k" (timm only), "simple_cnn"
    MODEL_POOL = ["efficientnet_b0", "convnext_tiny"]
    PRETRAINED = True            # pakai ImageNet weights (download sekali, butuh internet)
    IMG_SIZE = 224               # 224 standar; 128/160 untuk CPU; 288–384 kalau GPU kuat & detail kecil penting
    NORM = "auto"                # "auto" (imagenet kalau pretrained, dataset kalau scratch) | "imagenet" | "dataset" | "none"
    SIMPLE_CNN_WIDTHS = (32, 64, 128, 256)
    DROPOUT = 0.2                # dropout sebelum classifier (SimpleCNN / timm drop_rate)

    # =====================================================================
    # AUGMENTASI
    # =====================================================================
    AUG_LEVEL = "medium"         # "none" | "light" | "medium" | "strong"
    HFLIP = True                 # matikan kalau orientasi kiri-kanan bermakna (mis. teks, angka)
    VFLIP = False                # nyalakan untuk citra satelit / mikroskop / tampak atas
    ROTATION_DEG = 15
    COLOR_JITTER = 0.2           # 0 = mati
    RANDOM_ERASING = 0.0         # prob random erasing (0.1–0.25 untuk regularisasi)
    MIXUP_ALPHA = 0.0            # 0.2–0.4 untuk mengaktifkan mixup
    CUTMIX_ALPHA = 0.0           # 1.0 untuk mengaktifkan cutmix
    MIX_PROB = 0.5               # peluang sebuah batch di-mix

    # =====================================================================
    # VALIDASI
    # =====================================================================
    CV_STRATEGY = "kfold"        # "kfold" | "holdout"
    N_FOLDS = 5
    HOLDOUT_SIZE = 0.2
    FOLDS_TO_RUN = None          # None = semua fold; contoh [0] untuk cepat (OOF hanya fold 0)

    # =====================================================================
    # TRAINING DEEP LEARNING
    # =====================================================================
    RUN_DEEP = True
    EPOCHS = 10
    BATCH_SIZE = 32
    ACCUM_STEPS = 1              # gradient accumulation: effective batch = BATCH_SIZE * ACCUM_STEPS
    LR = 3e-4                    # LR backbone pretrained (fine-tuning)
    LR_SCRATCH = 1e-3            # LR kalau model dilatih dari nol (SimpleCNN / pretrained gagal)
    HEAD_LR_MULT = 10.0          # LR head = LR * ini (head baru butuh LR lebih besar)
    WEIGHT_DECAY = 1e-2
    WARMUP_EPOCHS = 1
    MIN_LR_RATIO = 0.01          # LR akhir cosine = LR * ini
    LABEL_SMOOTHING = 0.1
    CLASS_WEIGHT = None          # None | "balanced"  (untuk data imbalanced)
    USE_WEIGHTED_SAMPLER = False # alternatif class weight: oversampling kelas kecil (jangan dipakai bersamaan)
    FREEZE_EPOCHS = 0            # >0: epoch awal hanya melatih head (linear probe), lalu unfreeze semua
    GRAD_CLIP = 1.0              # 0 = mati
    PATIENCE = 3                 # early stopping (epoch tanpa perbaikan)
    USE_AMP = True               # mixed precision (hanya aktif kalau CUDA)
    # GTX 16xx (1650/1660) punya bug fp16 di cuDNN → output NaN pada ukuran batch tertentu. True = AMP dimatikan
    # otomatis di GPU tersebut. Selain itu notebook juga punya pengaman: kalau muncul NaN saat AMP, batch dihitung
    # ulang dalam fp32 dan AMP dimatikan untuk sisa run.
    AMP_DISABLE_ON_GTX16 = True
    NUM_WORKERS = 0 if os.name == "nt" else 2   # Windows: 0 (lihat penjelasan di section DataLoader)
    CACHE_IMAGES = "auto"        # "auto" | True | False — simpan gambar ter-resize di RAM (sangat mempercepat di Windows)
    CACHE_MAX_GB = 2.0
    OOM_RETRIES = 3              # CUDA OOM → batch size dibagi 2 & accum dikali 2, lalu ulang
    MIN_BATCH_SIZE = 4
    DETERMINISTIC = False        # True = hasil identik tiap run (lebih lambat)
    MAX_MINUTES_PER_MODEL = None # batas waktu per model; fold sisanya di-skip kalau lewat
    SKIP_TRAIN_IF_CKPT = False   # True: kalau checkpoint fold sudah ada, load saja (tidak training ulang)
    RUN_OVERFIT_TEST = True      # sanity check: model harus bisa menghafal 1 batch
    OVERFIT_STEPS = 25

    # =====================================================================
    # INFERENCE / TTA
    # =====================================================================
    TTA_HFLIP = True
    TTA_SCALES = [1.0]           # contoh [1.0, 1.15] = multi-scale (resize lebih besar lalu center crop)
    TTA_ON_OOF = True            # OOF dihitung dengan TTA yang sama dengan test (lebih konsisten)

    # =====================================================================
    # BASELINE CEPAT (frozen features)
    # =====================================================================
    RUN_FEATURE_BASELINE = True
    FEATURE_SOURCE = "auto"      # "auto" (backbone pretrained kalau ada, else pixel) | "backbone" | "pixels"
    FEATURE_BACKBONE = None      # None = model pertama di MODEL_POOL
    FEAT_PIXEL_SIZE = 32         # ukuran resize untuk pixel features
    PCA_COMPONENTS = 128
    LOGREG_C = 1.0
    RUN_LGB_ON_FEATURES = True

    # =====================================================================
    # ENSEMBLE & EVALUASI
    # =====================================================================
    RUN_ENSEMBLE = True
    ENSEMBLE_ITERS = 25
    RUN_GRADCAM = True
    N_GRADCAM = 8
    N_MISCLASSIFIED = 16

    # =====================================================================
    # FAST MODE OVERRIDES (dipakai kalau RUN_MODE == "fast")
    # =====================================================================
    FAST_SAMPLE = 800            # maksimal jumlah gambar train
    FAST_EPOCHS = 2
    FAST_FOLDS = 2
    FAST_IMG_SIZE = 96
    FAST_N_MODELS = 1

# %% [markdown]
# # 2. USER OVERRIDE (isi untuk lomba spesifik)
#
# Tulis perubahan CFG di cell bawah ini, contoh:
#
# ```python
# CFG.TRAIN_CSV = "./data/train.csv"
# CFG.TRAIN_IMAGE_DIR = "./data/train_images"
# CFG.IMAGE_COL = "image_id"
# CFG.TARGET_COL = "label"
# CFG.METRIC = "accuracy"
# CFG.RUN_MODE = "fast"
# ```
#
# Cell ini sengaja dipisah supaya `CFG` default tetap utuh dan kamu tahu persis apa saja yang diubah untuk lomba ini.

# %%
# >>> USER OVERRIDE
# Contoh (hapus tanda # untuk mengaktifkan):
# CFG.RUN_MODE = "fast"
# CFG.DATA_LAYOUT = "auto"
# CFG.TRAIN_CSV = "./data/train.csv"
# CFG.TRAIN_IMAGE_DIR = "./data/train_images"
# CFG.METRIC = "accuracy"
# CFG.MODEL_POOL = ["efficientnet_b0"]
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
# # 3. Install / cek library
#
# **Tujuan:** memastikan library opsional tersedia. Fungsi `ensure_packages` hanya meng-install kalau
# `CFG.AUTO_INSTALL = True` dan package benar-benar belum ada. Kegagalan install **tidak** menghentikan notebook.
#
# **Penting soal PyTorch:**
#
# - `torch` **tidak pernah** di-install otomatis (ukurannya besar dan harus cocok dengan versi CUDA).
# - `torchvision` juga tidak di-install otomatis karena versinya harus pas dengan torch
#   (aturan: torch `2.X` ↔ torchvision `0.(X+15)`, mis. torch 2.5 ↔ torchvision 0.20, torch 2.8 ↔ torchvision 0.23).
# - `timm` di-install dengan `--no-deps` supaya pip tidak "meng-upgrade" torch kamu ke versi CPU.
#
# **Cara membaca output:** tiap package ditandai `ok` / `installed` / `missing` / `failed`.
# **Jika** `timm` failed **maka** notebook tetap jalan dengan torchvision atau SimpleCNN.
# **Jika** torch missing **maka** bagian deep learning di-skip; baseline pixel features tetap membuat submission.

# %%
import sys
import subprocess
import importlib
import importlib.util

PIP_NAMES = {
    "sklearn": "scikit-learn", "PIL": "pillow", "yaml": "pyyaml", "lightgbm": "lightgbm",
    "seaborn": "seaborn", "timm": "timm", "huggingface_hub": "huggingface_hub", "safetensors": "safetensors",
}


def ensure_packages(packages, auto_install=True, no_deps=()):
    """Cek package (import name -> pip name). Install yang hilang kalau auto_install=True.

    Tidak pernah raise: kegagalan install hanya dicetak.
    """
    status = {}
    for imp_name, pip_name in packages.items():
        if importlib.util.find_spec(imp_name) is not None:
            status[imp_name] = "ok"
            continue
        if not auto_install:
            status[imp_name] = "missing"
            continue
        cmd = [sys.executable, "-m", "pip", "install", "-q", pip_name]
        if imp_name in no_deps:
            cmd.insert(5, "--no-deps")
        try:
            print("Installing", pip_name, "...")
            subprocess.check_call(cmd)
            importlib.invalidate_caches()
            status[imp_name] = "installed" if importlib.util.find_spec(imp_name) is not None else "failed"
        except Exception as e:  # noqa
            status[imp_name] = f"failed ({type(e).__name__})"
    for k, v in status.items():
        print(f"  {k:16s}: {v}")
    return status


_torch_present = importlib.util.find_spec("torch") is not None
base_pkgs = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "PIL": "pillow",
             "matplotlib": "matplotlib", "seaborn": "seaborn", "lightgbm": "lightgbm"}
print("Core / optional packages:")
PKG_STATUS = ensure_packages(base_pkgs, auto_install=CFG.AUTO_INSTALL)

if _torch_present and CFG.USE_TIMM:
    print("timm (+ dependensi ringan):")
    timm_pkgs = {"yaml": "pyyaml", "huggingface_hub": "huggingface_hub", "safetensors": "safetensors", "timm": "timm"}
    PKG_STATUS.update(ensure_packages(timm_pkgs, auto_install=CFG.AUTO_INSTALL and CFG.INSTALL_TIMM, no_deps=("timm",)))

if not _torch_present:
    print("\n⚠️ PyTorch tidak terinstall. Deep learning akan di-skip (baseline tetap jalan).")
    print("   Install manual (pilih sesuai GPU) dari https://pytorch.org/get-started/locally/ , contoh CPU:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
elif importlib.util.find_spec("torchvision") is None:
    try:
        import torch as _t
        _minor = int(_t.__version__.split(".")[1])
        _cuda = (_t.version.cuda or "").replace(".", "")
        _idx = f"cu{_cuda}" if _cuda else "cpu"
        print(f"\nInfo: torchvision tidak ada. Kalau mau, install versi yang cocok dengan torch {_t.__version__}:")
        print(f"   pip install torchvision==0.{_minor + 15}.* --no-deps --index-url https://download.pytorch.org/whl/{_idx}")
    except Exception:
        pass

# %% [markdown]
# # 4. Imports, flag ketersediaan, device, helper umum
#
# **Tujuan:** import semua library dengan aman. Library opsional di-import di dalam `try/except` dan menghasilkan
# flag `HAS_xxx`. Semua bagian notebook yang butuh library itu akan di-skip dengan pesan jelas kalau flag-nya `False`.
#
# **Cara membaca output:** baris `Backend` memberi tahu jalur yang dipakai:
#
# - `timm` → jalur terbaik (ratusan model pretrained).
# - `torchvision` → fallback kedua (resnet, efficientnet, convnext, mobilenet).
# - `simple_cnn` → fallback terakhir (dilatih dari nol, cocok untuk gambar kecil / dataset sederhana).
#
# `Device: cuda` berarti GPU terpakai. **Jika** kamu punya GPU NVIDIA tapi tertulis `cpu` **maka** torch yang terinstall
# adalah versi CPU — lihat Troubleshooting.

# %%
import gc
import json
import math
import time
import random
import types
import warnings
import traceback
import contextlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image, ImageFile, ImageOps, ImageEnhance

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, f1_score, log_loss, roc_auc_score, cohen_kappa_score,
                             confusion_matrix, classification_report, precision_recall_fscore_support)

warnings.filterwarnings("ignore")

try:
    from IPython.display import display
except Exception:  # noqa
    display = print

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except ImportError:
    HAS_SGKF = False

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# ---- PyTorch (opsional tapi sangat disarankan) ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAS_TORCH = True
except Exception as _e:  # noqa
    HAS_TORCH = False
    torch = None
    F = None
    # stub supaya definisi class di bawah tidak error walau torch tidak ada
    nn = types.SimpleNamespace(Module=object, Linear=type("Linear", (), {}), Conv2d=type("Conv2d", (), {}))
    Dataset = object

HAS_TV = False
if HAS_TORCH and CFG.USE_TORCHVISION:
    try:
        import torchvision
        import torchvision.transforms as T
        import torchvision.models as tvm
        HAS_TV = True
    except Exception as _e:  # noqa
        print("torchvision tidak bisa di-import:", type(_e).__name__, str(_e)[:120])

HAS_TIMM = False
if HAS_TORCH and CFG.USE_TIMM:
    try:
        import timm
        HAS_TIMM = True
    except Exception as _e:  # noqa
        print("timm tidak bisa di-import:", type(_e).__name__, str(_e)[:120])

ImageFile.LOAD_TRUNCATED_IMAGES = CFG.LOAD_TRUNCATED_IMAGES
Image.MAX_IMAGE_PIXELS = 300_000_000   # cegah DecompressionBombError untuk gambar sangat besar yang sah
RESAMPLE = getattr(Image, "Resampling", Image)
BILINEAR = RESAMPLE.BILINEAR

if HAS_TORCH:
    DEVICE = torch.device("cuda" if (CFG.USE_GPU and torch.cuda.is_available()) else "cpu")
else:
    DEVICE = "cpu"
USE_AMP = bool(HAS_TORCH and CFG.USE_AMP and str(DEVICE) == "cuda")
if USE_AMP and CFG.AMP_DISABLE_ON_GTX16 and "GTX 16" in torch.cuda.get_device_name(0).upper():
    USE_AMP = False
    print("⚠️ GPU GTX 16xx terdeteksi → AMP (fp16) dimatikan (bug fp16 menghasilkan NaN). Training tetap di GPU (fp32).")
BACKEND = "timm" if HAS_TIMM else ("torchvision" if HAS_TV else ("simple_cnn" if HAS_TORCH else "none (no torch)"))
FAILED_MODELS = {}

print("Python      :", sys.version.split()[0])
print("numpy/pandas:", np.__version__, "/", pd.__version__)
print("torch       :", torch.__version__ if HAS_TORCH else "NOT INSTALLED")
print("torchvision :", torchvision.__version__ if HAS_TV else "not used")
print("timm        :", timm.__version__ if HAS_TIMM else "not used")
print("lightgbm    :", lgb.__version__ if HAS_LGB else "not installed")
print("Backend     :", BACKEND)
print("Device      :", DEVICE, "| AMP:", USE_AMP)
if HAS_TORCH and str(DEVICE) == "cuda":
    _p = torch.cuda.get_device_properties(0)
    print(f"GPU         : {_p.name} ({_p.total_memory / 1024**3:.1f} GB)")

# %% [markdown]
# ## 4b. Terapkan RUN_MODE, buat folder output, helper (seed, save_fig, smart_read)
#
# **Cara membaca output:** ringkasan config final yang benar-benar dipakai. Cek sekali lagi path & metric di sini
# sebelum lanjut — kesalahan path adalah penyebab error nomor 1.

# %%
def apply_run_mode():
    """Override CFG untuk mode fast (debug cepat)."""
    if CFG.RUN_MODE == "fast":
        CFG.EPOCHS = min(CFG.EPOCHS, CFG.FAST_EPOCHS)
        CFG.N_FOLDS = min(CFG.N_FOLDS, CFG.FAST_FOLDS)
        CFG.IMG_SIZE = min(CFG.IMG_SIZE, CFG.FAST_IMG_SIZE)
        CFG.MODEL_POOL = list(CFG.MODEL_POOL)[:CFG.FAST_N_MODELS]
        CFG.WARMUP_EPOCHS = min(CFG.WARMUP_EPOCHS, 1)
        CFG.OVERFIT_STEPS = min(CFG.OVERFIT_STEPS, 15)
        print(f"[FAST MODE]: epochs={CFG.EPOCHS}, folds={CFG.N_FOLDS}, img={CFG.IMG_SIZE}, models={CFG.MODEL_POOL}")
    else:
        print("[FULL MODE]")


apply_run_mode()

OUT_DIR = CFG.OUTPUT_DIR
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
MODEL_DIR = os.path.join(OUT_DIR, "models")
SUB_DIR = os.path.join(OUT_DIR, "submissions")
for _d in (OUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR):
    os.makedirs(_d, exist_ok=True)


def seed_everything(seed=42, deterministic=False):
    """Set seed random, numpy, PYTHONHASHSEED, dan torch (kalau ada)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not deterministic


seed_everything(CFG.SEED, CFG.DETERMINISTIC)

plt.rcParams.update({"figure.dpi": 100, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
_FIG_COUNTER = [0]


def save_fig(name, fig=None, dpi=None, show=True):
    """Simpan figure ke figures/figNN_<name>.png lalu tutup (hemat memori)."""
    fig = fig if fig is not None else plt.gcf()
    _FIG_COUNTER[0] += 1
    path = os.path.join(FIG_DIR, f"fig{_FIG_COUNTER[0]:02d}_{name}.png")
    try:
        fig.savefig(path, dpi=dpi or CFG.FIG_DPI, bbox_inches="tight")
        print("  [fig] saved:", path)
    except Exception as e:  # noqa
        print("  gagal simpan figure:", e)
    if show:
        plt.show()
    plt.close(fig)
    return path


def save_table(df, name, index=False):
    """Simpan DataFrame ke tables/<name>.csv."""
    path = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(path, index=index)
    print("  [table] saved:", path)
    return path


def smart_read(path, **kwargs):
    """Baca tabel dengan format apa pun (csv/tsv/xlsx/parquet/json/feather/pickle) + fallback encoding/separator."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, **kwargs)
    if ext == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if ext == ".json":
        try:
            return pd.read_json(path, **kwargs)
        except ValueError:
            return pd.read_json(path, lines=True, **kwargs)
    if ext == ".feather":
        return pd.read_feather(path, **kwargs)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path, **kwargs)
    sep = "\t" if ext == ".tsv" else ","
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, **kwargs)
            if df.shape[1] == 1 and ext not in (".tsv",):
                df2 = pd.read_csv(path, sep=None, engine="python", encoding=enc, **kwargs)
                if df2.shape[1] > 1:
                    df = df2
            return df
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:  # noqa
            last_err = e
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc, **kwargs)
            except Exception as e2:  # noqa
                last_err = e2
    raise last_err


def file_exists(path):
    return bool(path) and os.path.isfile(str(path))


def progress(it, every=500, total=None, desc=""):
    """Iterator sederhana dengan print progress (tanpa tqdm)."""
    total = total if total is not None else (len(it) if hasattr(it, "__len__") else None)
    t0 = time.time()
    for i, x in enumerate(it):
        if i and i % every == 0:
            print(f"  {desc} {i}/{total if total else '?'} ({time.time() - t0:.0f}s)")
        yield x


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

print("\nConfig summary")
print("-" * 60)
for k in ["RUN_MODE", "DATA_LAYOUT", "TRAIN_CSV", "TEST_CSV", "SAMPLE_SUB_PATH", "TRAIN_IMAGE_DIR", "TEST_IMAGE_DIR",
          "IMAGE_COL", "TARGET_COL", "ID_COL", "METRIC", "SUBMISSION_MODE", "MODEL_POOL", "IMG_SIZE", "EPOCHS",
          "BATCH_SIZE", "N_FOLDS", "CV_STRATEGY", "OUTPUT_DIR"]:
    print(f"{k:18s}: {getattr(CFG, k)}")

# %% [markdown]
# # 5. Load data — deteksi layout otomatis
#
# **Tujuan:** apa pun bentuk datanya, ubah menjadi format seragam:
#
# - `train_df` dengan kolom `ref` (path gambar, atau penunjuk ke array pixel), `label` (label asli), `src_id`
# - `test_df` dengan kolom `id` (id untuk submission) dan `ref`
#
# **Logika auto-detect (`DATA_LAYOUT = "auto"`):**
#
# 1. Kalau `TRAIN_CSV` ada dan punya kolom yang isinya berakhiran `.jpg/.png/...` → layout **csv**.
# 2. Kalau `TRAIN_CSV` ada dan punya ≥ 64 kolom numerik → layout **pixels** (MNIST-like).
# 3. Kalau `TRAIN_CSV` ada dan punya kolom bernama `image_id`, `filename`, `id`, ... yang bisa di-resolve ke file → **csv**.
# 4. Kalau ada folder (`TRAIN_IMAGE_DIR`) dengan ≥ 2 subfolder berisi gambar → layout **folder**.
#
# Nama file di CSV boleh tanpa ekstensi (`img_001`), boleh path relatif (`train/img_001.jpg`), boleh path absolut —
# resolver mencoba semua kombinasi folder kandidat + ekstensi.
#
# **Cara membaca output:** cek `Layout terdeteksi`, jumlah baris train/test, jumlah file yang **tidak ketemu**.
#
# **Jika X maka Y:**
#
# - Jika `file tidak ditemukan` banyak → cek `TRAIN_IMAGE_DIR`; buka satu nama file di CSV dan cari manual di folder.
# - Jika layout salah terdeteksi → set manual `CFG.DATA_LAYOUT = "csv"` / `"folder"` / `"pixels"`.
# - Jika kolom label salah → set `CFG.TARGET_COL`. Jika kolom gambar salah → set `CFG.IMAGE_COL`.
# - Jika test tidak ditemukan → notebook tetap jalan (CV saja), submission di-skip.

# %%
IMAGE_EXTS = tuple(e.lower() for e in CFG.IMAGE_EXTS)
PX_PREFIX = "__px__"
PIXEL_STORE = {}            # {"train": array (N,H,W[,C]) uint8, "test": ...} untuk layout pixels
COMMON_IMG_COLS = ["image", "image_path", "image_name", "image_id", "img", "img_path", "img_name", "filename",
                   "file_name", "file", "path", "filepath", "id_code", "name", "id"]
COMMON_TARGET_COLS = ["label", "labels", "target", "class", "class_name", "classname", "category", "y",
                      "diagnosis", "species", "class_id"]


def is_image_file(name):
    return str(name).lower().endswith(IMAGE_EXTS)


def list_images(folder, recursive=True):
    """Semua file gambar di folder (urut, deterministik)."""
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for f in sorted(files):
            if is_image_file(f):
                out.append(os.path.join(root, f))
        if not recursive:
            break
    return out


def has_any_image(folder):
    for root, dirs, files in os.walk(folder):
        if any(is_image_file(f) for f in files):
            return True
    return False


def existing_dirs(cands):
    """Folder kandidat yang benar-benar ada (tanpa duplikat)."""
    out, seen = [], set()
    for d in cands:
        if d and os.path.isdir(d) and os.path.abspath(d) not in seen:
            out.append(d)
            seen.add(os.path.abspath(d))
    return out


class PathResolver:
    """Mencocokkan nilai di CSV (nama file / id / path) ke path file gambar yang nyata."""

    def __init__(self, dirs, flat_dirs=()):
        self.dirs = list(dirs)
        self.by_name, self.by_stem, self.by_lower = {}, {}, {}
        for d in self.dirs:
            for p in list_images(d, recursive=True):
                self._add(p)
        for d in flat_dirs:
            if d and os.path.isdir(d):
                for p in list_images(d, recursive=False):
                    self._add(p)
        print(f"  PathResolver: {len(self.by_name)} file gambar terindeks dari {self.dirs + list(flat_dirs)}")

    def _add(self, p):
        name = os.path.basename(p)
        self.by_name.setdefault(name, p)
        self.by_stem.setdefault(os.path.splitext(name)[0], p)
        self.by_lower.setdefault(name.lower(), p)

    def resolve(self, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        v = str(v).strip().replace("\\", "/")
        if os.path.isfile(v):
            return v
        for d in self.dirs:
            p = os.path.join(d, v)
            if os.path.isfile(p):
                return p
        name = os.path.basename(v)
        for key, table in ((name, self.by_name), (name, self.by_stem), (name.lower(), self.by_lower),
                           (os.path.splitext(name)[0], self.by_stem)):
            if key in table:
                return table[key]
        for d in self.dirs:
            for ext in IMAGE_EXTS:
                p = os.path.join(d, v + ext)
                if os.path.isfile(p):
                    return p
        return None


def guess_image_col(df, prefer=None):
    """Tebak kolom berisi nama file gambar."""
    for c in (prefer, CFG.IMAGE_COL):
        if c and c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().astype(str).head(200)
            if len(s) and s.map(is_image_file).mean() > 0.8:
                return c
    return None


def guess_named_image_col(df):
    lower = {str(c).lower(): c for c in df.columns}
    for name in COMMON_IMG_COLS:
        if name in lower:
            return lower[name]
    return None


def guess_target_col(df, exclude=()):
    if CFG.TARGET_COL and CFG.TARGET_COL in df.columns:
        return CFG.TARGET_COL
    lower = {str(c).lower(): c for c in df.columns if c not in exclude}
    for name in COMMON_TARGET_COLS:
        if name in lower:
            return lower[name]
    return None


def numeric_cols(df, exclude=()):
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def find_class_root(cands):
    """Cari folder ImageFolder: punya >= 2 subfolder yang berisi gambar."""
    skip = {"test", "tests", "test_images", "testing"}
    for d in existing_dirs(cands):
        subs = [s for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s)) and s.lower() not in skip]
        with_imgs = [s for s in subs if has_any_image(os.path.join(d, s))]
        if len(with_imgs) >= 2:
            return d, with_imgs
    return None, []


def detect_layout():
    if CFG.DATA_LAYOUT != "auto":
        return CFG.DATA_LAYOUT
    if file_exists(CFG.TRAIN_CSV):
        head = smart_read(CFG.TRAIN_CSV, nrows=300)
        if guess_image_col(head):
            return "csv"
        tgt = guess_target_col(head)
        if len(numeric_cols(head, exclude=[tgt, CFG.ID_COL])) >= 64:
            return "pixels"
        if guess_named_image_col(head):
            return "csv"
        print("⚠️ TRAIN_CSV ada tapi tidak terlihat seperti layout csv/pixels → coba layout folder")
    root, classes = find_class_root([CFG.TRAIN_IMAGE_DIR, "./data/train", "./data"])
    if root:
        return "folder"
    raise FileNotFoundError(
        "Data train tidak ditemukan. Cek CFG.TRAIN_CSV / CFG.TRAIN_IMAGE_DIR. "
        f"TRAIN_CSV={CFG.TRAIN_CSV} (ada={file_exists(CFG.TRAIN_CSV)}), TRAIN_IMAGE_DIR={CFG.TRAIN_IMAGE_DIR} "
        f"(ada={os.path.isdir(CFG.TRAIN_IMAGE_DIR)}). Working dir sekarang: {os.getcwd()}")


def parse_pixels(df, cols):
    """Ubah kolom pixel menjadi array uint8 (N,H,W) atau (N,H,W,C)."""
    arr = df[cols].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    if arr.max() <= 1.0 + 1e-6:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    n_pix = arr.shape[1]
    shape = CFG.PIXEL_SHAPE
    if shape is None:
        s = int(round(math.sqrt(n_pix)))
        s3 = int(round(math.sqrt(n_pix / 3)))
        if s * s == n_pix:
            shape = (s, s)
        elif s3 * s3 * 3 == n_pix:
            shape = (s3, s3, 3)
        else:
            raise ValueError(f"Tidak bisa menebak bentuk gambar dari {n_pix} pixel. Set CFG.PIXEL_SHAPE manual.")
    if CFG.PIXEL_CHANNEL_FIRST and len(shape) == 3:
        c, h, w = shape if shape[0] in (1, 3) else (shape[2], shape[0], shape[1])
        arr = arr.reshape(len(arr), c, h, w).transpose(0, 2, 3, 1)
    else:
        arr = arr.reshape((len(arr),) + tuple(shape))
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def load_sample_submission():
    if file_exists(CFG.SAMPLE_SUB_PATH):
        ss = smart_read(CFG.SAMPLE_SUB_PATH)
        print(f"sample_submission: {ss.shape} | kolom: {list(ss.columns)}")
        return ss
    print("Info: sample_submission tidak ditemukan → submission dibuat dengan format [id, target].")
    return None


def load_data():
    """Return train_df, test_df (atau None), layout, info dict."""
    layout = detect_layout()
    print("Layout terdeteksi:", layout)
    info = {"layout": layout}
    sample_sub = load_sample_submission()
    test_df = None

    if layout == "csv":
        tr = smart_read(CFG.TRAIN_CSV)
        img_col = guess_image_col(tr) or guess_named_image_col(tr)
        tgt = guess_target_col(tr, exclude=[img_col])
        if img_col is None or tgt is None:
            raise ValueError(f"Kolom gambar/label tidak terdeteksi (img={img_col}, target={tgt}). "
                             f"Kolom tersedia: {list(tr.columns)}. Set CFG.IMAGE_COL & CFG.TARGET_COL.")
        print(f"Kolom gambar: '{img_col}' | kolom label: '{tgt}'")
        res = PathResolver(existing_dirs([CFG.TRAIN_IMAGE_DIR] + CFG.IMAGE_DIR_CANDIDATES),
                           flat_dirs=[os.path.dirname(CFG.TRAIN_CSV) or "."])
        refs = [res.resolve(v) for v in tr[img_col].values]
        train_df = pd.DataFrame({"ref": refs, "label": tr[tgt].values, "src_id": tr[img_col].astype(str).values})
        if CFG.GROUP_COL and CFG.GROUP_COL in tr.columns:
            train_df["group"] = tr[CFG.GROUP_COL].values
        info.update(img_col=img_col, target_col=tgt)

        # ---- test ----
        test_dirs = existing_dirs([CFG.TEST_IMAGE_DIR] + CFG.TEST_IMAGE_DIR_CANDIDATES)
        te, te_img_col, te_id_col = None, None, None
        if file_exists(CFG.TEST_CSV):
            te = smart_read(CFG.TEST_CSV)
            te_img_col = guess_image_col(te, prefer=CFG.TEST_IMAGE_COL or img_col) or guess_named_image_col(te)
            if CFG.ID_COL and CFG.ID_COL in te.columns:
                te_id_col = CFG.ID_COL
            elif sample_sub is not None and sample_sub.columns[0] in te.columns:
                te_id_col = sample_sub.columns[0]
            else:
                te_id_col = te_img_col
        elif sample_sub is not None and test_dirs:
            te = sample_sub
            te_id_col = CFG.ID_COL if (CFG.ID_COL and CFG.ID_COL in te.columns) else te.columns[0]
            te_img_col = te_id_col
            print("Info: test.csv tidak ada → daftar gambar test diambil dari kolom id sample_submission")
        if te is not None and te_img_col is not None:
            res_te = PathResolver(test_dirs, flat_dirs=[os.path.dirname(CFG.TEST_CSV) or "."])
            test_df = pd.DataFrame({"id": te[te_id_col].values,
                                    "ref": [res_te.resolve(v) for v in te[te_img_col].values]})
        elif test_dirs:
            files = list_images(test_dirs[0])
            if files:
                test_df = pd.DataFrame({"id": [os.path.basename(p) for p in files], "ref": files})
                print(f"Info: test diambil dari folder {test_dirs[0]} (id = nama file)")

    elif layout == "folder":
        root, classes = find_class_root([CFG.TRAIN_IMAGE_DIR, "./data/train", "./data"])
        if root is None:
            raise FileNotFoundError(f"Folder ImageFolder tidak ditemukan di {CFG.TRAIN_IMAGE_DIR}")
        rows = []
        for d in [root] + list(CFG.EXTRA_TRAIN_DIRS):
            if not os.path.isdir(d):
                print("⚠️ EXTRA_TRAIN_DIR tidak ada:", d)
                continue
            for c in sorted(os.listdir(d)):
                cdir = os.path.join(d, c)
                if os.path.isdir(cdir) and c.lower() not in ("test", "tests", "test_images"):
                    for p in list_images(cdir):
                        rows.append((p, c, os.path.relpath(p, d)))
        train_df = pd.DataFrame(rows, columns=["ref", "label", "src_id"])
        print(f"Root kelas: {root} | {len(classes)} kelas: {classes[:20]}")
        test_dirs = existing_dirs([CFG.TEST_IMAGE_DIR] + CFG.TEST_IMAGE_DIR_CANDIDATES)
        test_dirs = [d for d in test_dirs if os.path.abspath(d) != os.path.abspath(root)]
        if test_dirs:
            files = list_images(test_dirs[0])
            if files:
                test_df = pd.DataFrame({"id": [os.path.basename(p) for p in files], "ref": files})
                print(f"Test folder: {test_dirs[0]} ({len(files)} gambar, id = nama file)")

    elif layout == "pixels":
        tr = smart_read(CFG.TRAIN_CSV)
        tgt = guess_target_col(tr)
        if tgt is None:
            raise ValueError(f"Kolom label tidak terdeteksi. Set CFG.TARGET_COL. Kolom awal: {list(tr.columns[:10])}")
        id_col = CFG.ID_COL if (CFG.ID_COL and CFG.ID_COL in tr.columns) else None
        pix_cols = numeric_cols(tr, exclude=[tgt, id_col])
        if CFG.PIXEL_PREFIX:
            pix_cols = [c for c in pix_cols if str(c).startswith(CFG.PIXEL_PREFIX)]
        PIXEL_STORE["train"] = parse_pixels(tr, pix_cols)
        print(f"Pixel layout: {len(pix_cols)} kolom pixel → shape gambar {PIXEL_STORE['train'].shape[1:]}")
        train_df = pd.DataFrame({"ref": [f"{PX_PREFIX}/train/{i}" for i in range(len(tr))],
                                 "label": tr[tgt].values,
                                 "src_id": tr[id_col].astype(str).values if id_col else np.arange(len(tr)).astype(str)})
        info.update(target_col=tgt, pix_cols=pix_cols)
        if file_exists(CFG.TEST_CSV):
            te = smart_read(CFG.TEST_CSV)
            missing_cols = [c for c in pix_cols if c not in te.columns]
            if missing_cols:
                print(f"⚠️ {len(missing_cols)} kolom pixel tidak ada di test → diisi 0")
                for c in missing_cols:
                    te[c] = 0
            PIXEL_STORE["test"] = parse_pixels(te, pix_cols)
            te_id_col = CFG.ID_COL if (CFG.ID_COL and CFG.ID_COL in te.columns) else None
            if te_id_col is None and sample_sub is not None and len(sample_sub) == len(te):
                ids = sample_sub.iloc[:, 0].values
                print("Info: test tanpa kolom id → id diambil dari sample_submission (urutan sama)")
            else:
                ids = te[te_id_col].values if te_id_col else np.arange(1, len(te) + 1)
            test_df = pd.DataFrame({"id": ids, "ref": [f"{PX_PREFIX}/test/{i}" for i in range(len(te))]})
    else:
        raise ValueError(f"DATA_LAYOUT tidak dikenal: {layout}")

    return train_df, test_df, sample_sub, info


train_df, test_df, sample_sub, DATA_INFO = load_data()

# ---- sanity: label kosong & file tidak ditemukan ----
n0 = len(train_df)
na_label = train_df["label"].isna()
if na_label.any():
    print(f"⚠️ {na_label.sum()} baris label kosong → dibuang")
    train_df = train_df[~na_label]
missing = train_df["ref"].isna()
if missing.any():
    print(f"⚠️ {missing.sum()} gambar train TIDAK ditemukan → dibuang. Contoh src_id:",
          train_df.loc[missing, "src_id"].head(5).tolist())
    train_df = train_df[~missing]
train_df = train_df.reset_index(drop=True)
print(f"Train: {n0} baris → {len(train_df)} baris valid")

HAS_TEST = test_df is not None and len(test_df) > 0
if HAS_TEST:
    test_df = test_df.reset_index(drop=True)
    n_miss_te = int(test_df["ref"].isna().sum())
    print(f"Test : {len(test_df)} baris | gambar tidak ditemukan: {n_miss_te}")
    if n_miss_te:
        print("⚠️ Gambar test yang hilang TIDAK dibuang (submission harus lengkap) → diprediksi dari gambar hitam. "
              "Perbaiki path kalau jumlahnya banyak!")
else:
    print("Info: Test tidak ditemukan → notebook menjalankan CV saja, submission di-skip.")

display(train_df.head())
if HAS_TEST:
    display(test_df.head())

# %% [markdown]
# ## 5b. Encode label + FAST subsample
#
# Label asli (string atau angka) di-encode ke `0..K-1` dengan urutan **sorted**. Mapping disimpan ke
# `tables/label_mapping.csv` — dipakai saat membuat submission (prediksi dikembalikan ke label asli).
#
# **Jika** jumlah kelas = 2 → task **binary** (kelas positif = kelas ke-2 setelah sort, penting untuk metric `f1`/`roc_auc`).
# **Jika** jumlah kelas sangat banyak (>100) dan per kelas sedikit → pertimbangkan pretrained backbone + LogisticRegression.

# %%
def _sort_key(v):
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


CLASSES = sorted(pd.unique(train_df["label"]).tolist(), key=_sort_key)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_NAMES = [str(c) for c in CLASSES]
N_CLASSES = len(CLASSES)
train_df["y"] = train_df["label"].map(CLASS_TO_IDX).astype(int)
TASK = "binary" if N_CLASSES == 2 else "multiclass"
if N_CLASSES < 2:
    raise ValueError("Hanya ada 1 kelas di train — cek kolom label.")
if CFG.METRIC == "f1" and TASK != "binary":
    print("Info: METRIC 'f1' untuk multiclass → diganti 'f1_macro'")
    CFG.METRIC = "f1_macro"
GREATER_IS_BETTER = CFG.METRIC != "logloss"

if CFG.RUN_MODE == "fast" and len(train_df) > CFG.FAST_SAMPLE:
    keep, _ = train_test_split(np.arange(len(train_df)), train_size=CFG.FAST_SAMPLE, random_state=CFG.SEED,
                               stratify=train_df["y"] if train_df["y"].value_counts().min() >= 2 else None)
    train_df = train_df.iloc[np.sort(keep)].reset_index(drop=True)
    print(f"[FAST MODE] train di-subsample ke {len(train_df)} gambar")

save_table(pd.DataFrame({"idx": range(N_CLASSES), "label": CLASSES}), "label_mapping")
print(f"Task: {TASK} | {N_CLASSES} kelas | metric: {CFG.METRIC} ({'maximize' if GREATER_IS_BETTER else 'minimize'})")
print("Kelas:", CLASS_NAMES[:30], "..." if N_CLASSES > 30 else "")

# %% [markdown]
# # 6. Image loading helper
#
# Semua gambar dibaca lewat `load_image()` yang:
#
# - memperbaiki rotasi dari EXIF (foto HP sering tersimpan miring),
# - mengubah semua mode ke **RGB**: `L` (grayscale) → 3 channel, `RGBA`/`LA`/`P` transparan → ditempel di background
#   putih (bukan hitam, supaya objek tidak hilang), `CMYK` → RGB, `I;16`/`I`/`F` (16-bit, medis) → dinormalisasi ke 8-bit,
# - `safe_load_image()` tidak pernah crash: gambar rusak / hilang diganti gambar hitam + dicatat di `LOAD_ERRORS`.

# %%
LOAD_ERRORS = Counter()


def to_rgb(img):
    """Konversi PIL image mode apa pun ke RGB."""
    mode = img.mode
    if mode == "RGB":
        return img
    if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(a).convert("RGB")
    return img.convert("RGB")


def load_image(ref):
    """Baca 1 gambar sebagai PIL RGB. ref = path file atau '__px__/<split>/<idx>'."""
    if isinstance(ref, str) and ref.startswith(PX_PREFIX):
        _, split, idx = ref.split("/")
        return to_rgb(Image.fromarray(PIXEL_STORE[split][int(idx)]))
    img = Image.open(ref)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa
        pass
    img.load()
    return to_rgb(img)


def safe_load_image(ref, fallback_size=None):
    """Seperti load_image tapi tidak pernah error (gambar hitam kalau gagal)."""
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        LOAD_ERRORS["missing"] += 1
        s = fallback_size or CFG.IMG_SIZE
        return Image.new("RGB", (s, s))
    try:
        return load_image(ref)
    except Exception as e:  # noqa
        LOAD_ERRORS[type(e).__name__] += 1
        s = fallback_size or CFG.IMG_SIZE
        return Image.new("RGB", (s, s))


_img = safe_load_image(train_df["ref"].iloc[0])
print("Contoh gambar pertama:", train_df["ref"].iloc[0], "| size:", _img.size, "| mode:", _img.mode)

# %% [markdown]
# # 7. Image EDA
#
# ## 7a. Scan semua gambar: file korup, ukuran, mode channel, hash, brightness, mean/std
#
# **Tujuan:** satu kali baca semua gambar untuk mengumpulkan metadata. Per gambar dicek:
#
# - `PIL.Image.verify()` lalu `load()` penuh → mendeteksi file **korup** (tidak bisa dibaca) dan **terpotong (truncated)**.
# - `width`, `height`, `mode` (RGB / L / RGBA / CMYK / P / I;16), ukuran file.
# - **average hash (aHash)** 8×8 → untuk mendeteksi gambar duplikat (diimplementasi sendiri, tanpa library tambahan).
# - brightness (rata-rata intensitas) dan statistik per channel untuk **mean/std normalisasi**.
#
# **Cara membaca output:**
#
# - `corrupted` = tidak bisa dibuka sama sekali → **dibuang** dari train (di test tetap dipertahankan, diganti gambar hitam).
# - `truncated` = file terpotong tapi masih bisa dibaca dengan `ImageFile.LOAD_TRUNCATED_IMAGES = True` → dipertahankan
#   (kecuali `CFG.DROP_TRUNCATED = True`).
#
# **Jika** banyak file korup (> 1%) → cek apakah download/extract zip sempurna (ukuran zip, extract ulang).
# **Jika** scan terlalu lama (dataset > 100k gambar) → set `CFG.SCAN_MAX_IMAGES = 20000` (file korup di luar sampel
# tetap aman karena loader memakai `safe_load_image`).

# %%
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
THUMBS = {}              # ref -> thumbnail grayscale 32x32 (uint8) untuk verifikasi duplikat
DUP_PIXEL_TOL = 0.03     # rata-rata selisih pixel (0-1) maksimum supaya 2 gambar dianggap duplikat


def ahash_bits(img, hash_size=8):
    """Average hash: grayscale → resize hash_size x hash_size → bit = pixel > mean."""
    g = img.convert("L").resize((hash_size, hash_size), BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    return (a > a.mean()).flatten()


def bits_to_hex(bits):
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()


def scan_image(ref):
    """Kumpulkan metadata 1 gambar. Return (record dict, channel_sum, channel_sqsum, n_pixels)."""
    rec = {"ref": ref, "ok": True, "truncated": False, "error": "", "width": np.nan, "height": np.nan,
           "mode": "", "file_kb": np.nan, "ahash": "", "ahash8": "", "brightness": np.nan, "pix_std": np.nan}
    zeros = np.zeros(3)
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        rec.update(ok=False, error="missing")
        return rec, zeros, zeros, 0
    try:
        if isinstance(ref, str) and ref.startswith(PX_PREFIX):
            _, split, idx = ref.split("/")
            raw = Image.fromarray(PIXEL_STORE[split][int(idx)])
        else:
            rec["file_kb"] = os.path.getsize(ref) / 1024
            prev = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            try:
                with Image.open(ref) as im:
                    im.verify()
                raw = Image.open(ref)
                raw.load()
            except Exception as e_strict:  # noqa
                # coba lagi dengan mode toleran → kalau berhasil, berarti truncated (masih bisa dipakai)
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                raw = Image.open(ref)
                raw.load()
                rec["truncated"] = True
                rec["error"] = f"truncated: {type(e_strict).__name__}"
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = prev
        rec["width"], rec["height"], rec["mode"] = raw.size[0], raw.size[1], raw.mode
        rgb = to_rgb(raw)
        thumb = rgb.resize((CFG.STAT_SIZE, CFG.STAT_SIZE), BILINEAR)
        a = np.asarray(thumb, dtype=np.float64) / 255.0
        rec["brightness"] = float(a.mean())
        rec["pix_std"] = float(a.std())
        THUMBS[ref] = np.asarray(thumb.convert("L").resize((32, 32), BILINEAR), dtype=np.uint8)
        rec["ahash"] = bits_to_hex(ahash_bits(thumb, 16))   # 256-bit: kunci duplikat persis
        rec["ahash8"] = bits_to_hex(ahash_bits(thumb, 8))   # 64-bit: untuk near-duplicate (hamming)
        flat = a.reshape(-1, 3)
        return rec, flat.sum(0), (flat ** 2).sum(0), flat.shape[0]
    except Exception as e:  # noqa
        rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:100]}")
        return rec, zeros, zeros, 0


def scan_dataset(refs, name):
    """Scan list gambar → DataFrame metadata + mean/std per channel."""
    refs = list(refs)
    if CFG.SCAN_MAX_IMAGES and len(refs) > CFG.SCAN_MAX_IMAGES:
        rng = np.random.RandomState(CFG.SEED)
        refs = [refs[i] for i in sorted(rng.choice(len(refs), CFG.SCAN_MAX_IMAGES, replace=False))]
        print(f"  ({name}: scan dibatasi ke {len(refs)} gambar)")
    recs, s, sq, n = [], np.zeros(3), np.zeros(3), 0
    t0 = time.time()
    for ref in progress(refs, every=2000, desc=f"scan {name}"):
        rec, cs, csq, npx = scan_image(ref)
        recs.append(rec)
        s += cs
        sq += csq
        n += npx
    meta = pd.DataFrame(recs)
    mean = s / max(n, 1)
    std = np.sqrt(np.maximum(sq / max(n, 1) - mean ** 2, 1e-12))
    print(f"  scan {name}: {len(meta)} gambar dalam {time.time() - t0:.1f}s")
    return meta, mean, std


DATASET_MEAN, DATASET_STD = IMAGENET_MEAN, IMAGENET_STD
meta_train, meta_test = None, None
if CFG.RUN_EDA:
    meta_train, DATASET_MEAN, DATASET_STD = scan_dataset(train_df["ref"], "train")
    if HAS_TEST:
        meta_test, _, _ = scan_dataset(test_df["ref"], "test")
    DATASET_MEAN = tuple(float(v) for v in np.round(DATASET_MEAN, 4))
    DATASET_STD = tuple(float(v) for v in np.round(DATASET_STD, 4))
    meta_train = meta_train.merge(train_df[["ref", "label", "y"]].drop_duplicates("ref"), on="ref", how="left")

    bad = meta_train[~meta_train["ok"]]
    trunc = meta_train[meta_train["truncated"]]
    print(f"\nTrain corrupted : {len(bad)}")
    if len(bad):
        display(bad[["ref", "error"]].head(20))
    print(f"Train truncated : {len(trunc)}" + ("  (dipertahankan)" if not CFG.DROP_TRUNCATED else "  (dibuang)"))
    if len(trunc):
        display(trunc[["ref", "error"]].head(10))
    drop_refs = set(bad["ref"])
    if CFG.DROP_TRUNCATED:
        drop_refs |= set(trunc["ref"])
    if drop_refs:
        before = len(train_df)
        train_df = train_df[~train_df["ref"].isin(drop_refs)].reset_index(drop=True)
        print(f"⚠️ {before - len(train_df)} gambar train bermasalah dibuang → {len(train_df)} tersisa")
        save_table(bad[["ref", "error"]], "corrupted_train_files")
    else:
        print("✅ Tidak ada file train korup")
    if meta_test is not None:
        bad_te = meta_test[~meta_test["ok"]]
        print(f"Test corrupted  : {len(bad_te)}" + (" → akan diprediksi dari gambar hitam (cek manual!)" if len(bad_te) else ""))
        if len(bad_te):
            display(bad_te[["ref", "error"]].head(10))
    print("\nMean dataset (RGB):", DATASET_MEAN, "| Std:", DATASET_STD)
    print("ImageNet mean     :", IMAGENET_MEAN, "| Std:", IMAGENET_STD)
else:
    print("RUN_EDA=False → scan di-skip (mean/std dataset = ImageNet)")

# %% [markdown]
# ## 7b. Distribusi kelas
#
# **Cara membaca:** bar chart jumlah gambar per kelas + persentase. `Imbalance ratio` = kelas terbesar / terkecil.
#
# **Jika X maka Y:**
#
# - Ratio < 3 → aman, tidak perlu treatment khusus.
# - Ratio 3–10 → coba `CFG.CLASS_WEIGHT = "balanced"` **atau** `USE_WEIGHTED_SAMPLER = True`, dan pakai metric macro-F1
#   untuk monitoring (accuracy bisa menipu).
# - Ratio > 10 → wajib treatment imbalance + cek per-class recall di confusion matrix; pertimbangkan threshold tuning.
# - Ada kelas dengan < `N_FOLDS` gambar → stratified K-fold otomatis mengurangi jumlah fold (lihat section CV).
#
# **Untuk laporan:** cantumkan figure ini + tabel jumlah per kelas di bagian "Deskripsi Data".

# %%
class_counts = train_df["label"].value_counts().reindex(CLASSES).fillna(0).astype(int)
IMB_RATIO = class_counts.max() / max(class_counts.min(), 1)
cc_table = pd.DataFrame({"label": CLASS_NAMES, "count": class_counts.values,
                         "pct": (class_counts.values / class_counts.sum() * 100).round(2)})
display(cc_table)
save_table(cc_table, "class_distribution")

fig, ax = plt.subplots(figsize=(max(6, min(0.5 * N_CLASSES + 3, 18)), 4))
ax.bar(range(N_CLASSES), class_counts.values, color="#4C72B0")
ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=45 if N_CLASSES > 6 else 0, ha="right" if N_CLASSES > 6 else "center")
for i, v in enumerate(class_counts.values):
    ax.text(i, v, f"{v}\n({v / class_counts.sum() * 100:.1f}%)", ha="center", va="bottom", fontsize=8)
ax.set_title(f"Class Distribution (train, n={len(train_df)}, imbalance ratio={IMB_RATIO:.1f})")
ax.set_xlabel("Class")
ax.set_ylabel("Number of images")
ax.set_ylim(0, class_counts.max() * 1.25)
save_fig("class_distribution", fig)

if IMB_RATIO > 10:
    print(f"⚠️ Imbalance berat (ratio {IMB_RATIO:.1f}). Set CLASS_WEIGHT='balanced' / USE_WEIGHTED_SAMPLER=True.")
elif IMB_RATIO > 3:
    print(f"⚠️ Imbalance sedang (ratio {IMB_RATIO:.1f}). Pertimbangkan CLASS_WEIGHT='balanced'.")
else:
    print(f"✅ Distribusi kelas cukup seimbang (ratio {IMB_RATIO:.1f}).")

# %% [markdown]
# ## 7c. Contoh gambar per kelas
#
# **Cara membaca:** setiap baris = 1 kelas. Lihat dengan mata:
#
# - Apakah label masuk akal? (label noise sering terlihat langsung)
# - Apa yang membedakan antar kelas — warna, bentuk, tekstur, detail kecil? → menentukan augmentasi:
#   kalau **warna** adalah pembeda utama, kecilkan `COLOR_JITTER`; kalau **orientasi** bermakna, matikan `HFLIP`/rotasi.
# - Apakah objek kecil di gambar besar? → butuh `IMG_SIZE` lebih besar.
#
# **Untuk laporan:** figure ini bagus sebagai "Gambar 1. Contoh data per kelas".

# %%
def show_grid(refs, titles, ncols, fname, suptitle, cell=2.0, title_colors=None):
    n = len(refs)
    if n == 0:
        print("(tidak ada gambar untuk ditampilkan)")
        return
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows + 0.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (r, t) in enumerate(zip(refs, titles)):
        if r is None:
            continue
        ax = axes.ravel()[i]
        img = safe_load_image(r)
        img.thumbnail((256, 256))
        ax.imshow(img)
        ax.set_title(str(t), fontsize=7, color=(title_colors[i] if title_colors else "black"))
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    save_fig(fname, fig)


rng = np.random.RandomState(CFG.SEED)
grid_classes = CLASSES if N_CLASSES <= CFG.MAX_CLASSES_IN_GRID else \
    [CLASSES[i] for i in sorted(rng.choice(N_CLASSES, CFG.MAX_CLASSES_IN_GRID, replace=False))]
g_refs, g_titles = [], []
for c in grid_classes:
    sub = train_df[train_df["label"] == c]
    pick = sub.sample(min(CFG.GRID_PER_CLASS, len(sub)), random_state=CFG.SEED)
    for _, r in pick.iterrows():
        g_refs.append(r["ref"])
        g_titles.append(f"{c}")
    for _ in range(CFG.GRID_PER_CLASS - len(pick)):
        g_refs.append(None)
        g_titles.append("")
show_grid([r for r in g_refs], g_titles, CFG.GRID_PER_CLASS, "sample_grid_per_class",
          "Sample Images per Class (one row per class)")

# %% [markdown]
# ## 7d. Ukuran gambar, aspect ratio, mode channel
#
# **Cara membaca:**
#
# - Scatter width vs height: titik menumpuk di satu tempat = ukuran seragam. Tersebar = perlu resize (otomatis dilakukan).
# - Aspect ratio (w/h): jauh dari 1 (mis. 2.0 panorama, 0.5 potret) → resize paksa ke persegi akan "menggepengkan"
#   objek. Biasanya tetap OK untuk CNN, tapi kalau bentuk penting pertimbangkan padding (letterbox) di transform.
# - Mode channel: `L` = grayscale, `RGBA` = ada transparansi, `CMYK` = hasil scan/cetak, `P` = palette (gif/png),
#   `I;16` = 16-bit (medis). **Semua otomatis dikonversi ke RGB** oleh `to_rgb()`.
#
# **Jika X maka Y:**
#
# - Median ukuran jauh lebih kecil dari `IMG_SIZE` (mis. 32 px vs 224) → upsampling besar tidak menambah info;
#   coba `IMG_SIZE` 64–128 (lebih cepat) atau tetap 224 untuk pretrained (sering tetap membantu).
# - Median ukuran sangat besar (> 1000 px) dan objek kecil → naikkan `IMG_SIZE` (288–384) kalau GPU kuat.
# - Distribusi ukuran train vs test **berbeda jauh** → hati-hati domain shift; ukuran gambar bisa jadi "shortcut" (leak).
#
# **Untuk laporan:** tulis rentang ukuran gambar (min / median / max) dan bahwa semua gambar di-resize ke `IMG_SIZE`.

# %%
if meta_train is not None:
    mt = meta_train[meta_train["ok"]].copy()
    mt["aspect"] = mt["width"] / mt["height"]
    size_summary = mt[["width", "height", "aspect", "file_kb"]].describe().T.round(2)
    display(size_summary)
    save_table(size_summary.reset_index().rename(columns={"index": "stat"}), "image_size_summary")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(mt["width"], mt["height"], s=8, alpha=0.4, label="train")
    if meta_test is not None and meta_test["ok"].any():
        mte = meta_test[meta_test["ok"]]
        axes[0].scatter(mte["width"], mte["height"], s=8, alpha=0.4, label="test", color="#DD8452")
    axes[0].set_title("Image Size (width vs height)")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].legend()
    axes[1].hist(mt["aspect"].clip(0, 4), bins=40, color="#55A868")
    axes[1].set_title("Aspect Ratio Distribution (w/h)")
    axes[1].set_xlabel("Aspect ratio")
    axes[1].set_ylabel("Count")
    mode_tr = meta_train["mode"].replace("", "unreadable").value_counts()
    mode_df = pd.DataFrame({"train": mode_tr})
    if meta_test is not None:
        mode_df["test"] = meta_test["mode"].replace("", "unreadable").value_counts()
    mode_df = mode_df.fillna(0).astype(int)
    mode_df.plot(kind="bar", ax=axes[2], color=["#4C72B0", "#DD8452"][:mode_df.shape[1]])
    axes[2].set_title("Image Mode (channels)")
    axes[2].set_xlabel("PIL mode")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=0)
    axes[2].set_yscale("log")   # skala log supaya mode yang jarang tetap terlihat
    plt.tight_layout()
    save_fig("image_size_aspect_mode", fig)
    display(mode_df)

    med_side = float(np.median(np.minimum(mt["width"], mt["height"])))
    print(f"Median sisi terpendek: {med_side:.0f}px | IMG_SIZE: {CFG.IMG_SIZE}")
    if med_side < CFG.IMG_SIZE / 2:
        print("Info: Gambar asli jauh lebih kecil dari IMG_SIZE (upsampling). Coba IMG_SIZE lebih kecil untuk kecepatan.")
    if mt["aspect"].std() > 0.3:
        print("⚠️ Aspect ratio sangat bervariasi → resize ke persegi akan menggepengkan sebagian gambar.")
    non_rgb = {str(k): int(v) for k, v in mode_tr.items() if k not in ("RGB", "unreadable")}
    if non_rgb:
        print(f"Info: {sum(non_rgb.values())} gambar train bukan RGB (mode: {non_rgb}) → dikonversi otomatis ke RGB.")
    if (mt["pix_std"] < 0.02).any():
        print(f"⚠️ {(mt['pix_std'] < 0.02).sum()} gambar hampir polos/kosong (std pixel < 0.02) → cek manual.")
else:
    print("EDA scan di-skip.")

# %% [markdown]
# ## 7e. Duplikat gambar (average hash)
#
# **Cara kerja aHash:** gambar → grayscale kecil → setiap pixel jadi bit 1 kalau > rata-rata → fingerprint.
# Duplikat persis memakai hash 16×16 (256 bit, sangat jarang salah); dua gambar dengan hash sama hampir pasti duplikat
# (walau beda ukuran / kompresi). `NEAR_DUP_HAMMING > 0` juga menangkap near-duplicate memakai hash 8×8 (beda ≤ k bit),
# tapi gambar yang memang mirip (mis. X-ray, objek sederhana di background polos) bisa ikut tertangkap.
#
# **Cara membaca output:**
#
# - `duplicate groups (train)`: grup gambar identik di train. Kalau duplikat tersebar di fold berbeda → CV terlalu
#   optimis. Dengan `DUP_AWARE_CV = True`, duplikat dipaksa di fold yang sama.
# - `label conflict`: gambar identik dengan label **berbeda** → label noise. Buang (`DROP_DUPLICATES=True`) atau periksa.
# - `train-test overlap`: gambar test yang identik dengan gambar train. Cek rules lomba sebelum memanfaatkannya.
#
# **Untuk laporan:** sebutkan jumlah duplikat yang ditemukan dan bagaimana kamu menanganinya (data cleaning).

# %%
DUP_GROUP = None
if meta_train is not None:
    mtr = meta_train[meta_train["ok"] & (meta_train["ahash"] != "") & meta_train["ref"].isin(set(train_df["ref"]))]

    def thumb_diff(a, b):
        return float(np.abs(THUMBS[a].astype(np.float32) - THUMBS[b].astype(np.float32)).mean() / 255.0)

    def verified_keys(meta):
        """Hash sama belum tentu duplikat (gambar sederhana bisa punya hash sama). Verifikasi dengan selisih
        pixel thumbnail: anggota grup hash dipecah menjadi cluster yang benar-benar mirip."""
        keys = {}
        for h, grp in meta.groupby("ahash"):
            reps = []
            for r in grp["ref"]:
                for k, rep in enumerate(reps):
                    if thumb_diff(r, rep) < DUP_PIXEL_TOL:
                        keys[r] = f"{h}_{k}"
                        break
                else:
                    reps.append(r)
                    keys[r] = f"{h}_{len(reps) - 1}"
        return keys

    mtr = mtr.copy()
    mtr["dup_key"] = mtr["ref"].map(verified_keys(mtr))
    dup = mtr[mtr.duplicated("dup_key", keep=False)].sort_values("dup_key")
    n_groups = dup["dup_key"].nunique()
    print(f"Duplicate groups (train, hash persis): {n_groups} grup, {len(dup)} gambar")
    conflict_hashes = dup.groupby("dup_key")["y"].nunique()
    conflict_hashes = conflict_hashes[conflict_hashes > 1].index
    print(f"Label conflict (gambar sama, label beda): {len(conflict_hashes)} grup")
    if len(dup):
        display(dup[["ref", "label", "dup_key"]].head(12))
        save_table(dup[["ref", "label", "dup_key"]], "duplicate_groups_train")
        k = min(4, n_groups)
        d_refs, d_titles = [], []
        for h in dup["dup_key"].unique()[:k]:
            grp = dup[dup["dup_key"] == h].head(4)
            for _, r in grp.iterrows():
                d_refs.append(r["ref"])
                d_titles.append(f"{r['label']} | {h[:6]}")
            for _ in range(4 - len(grp)):
                d_refs.append(None)
                d_titles.append("")
        show_grid(d_refs, d_titles, 4, "duplicate_examples", "Duplicate Image Groups (same average hash)")

    # near-duplicate (opsional)
    if CFG.NEAR_DUP_HAMMING > 0 and 1 < len(mtr) <= CFG.NEAR_DUP_MAX_N:
        H = np.array([int(h, 16) for h in mtr["ahash8"]], dtype=np.uint64)
        pairs = []
        for i0 in range(0, len(H), 500):
            x = (H[i0:i0 + 500, None] ^ H[None, :]).view(np.uint8).reshape(-1, len(H), 8)
            dist = POPCOUNT8[x].sum(-1)
            ii, jj = np.where(dist <= CFG.NEAR_DUP_HAMMING)
            for a, b in zip(ii + i0, jj):
                if a < b and H[a] != H[b]:
                    pairs.append((mtr["ref"].iloc[a], mtr["ref"].iloc[b], int(dist[a - i0, b])))
        print(f"Near-duplicate pairs (hamming <= {CFG.NEAR_DUP_HAMMING}, hash beda): {len(pairs)}")
        if pairs:
            save_table(pd.DataFrame(pairs, columns=["ref_a", "ref_b", "hamming"]), "near_duplicates_train")

    if CFG.DROP_DUPLICATES and len(dup):
        bad_refs = set(mtr[mtr["dup_key"].isin(conflict_hashes)]["ref"])
        keep_first = mtr[~mtr["dup_key"].isin(conflict_hashes)].drop_duplicates("dup_key", keep="first")["ref"]
        dup_refs = set(dup["ref"]) - set(keep_first) - bad_refs
        before = len(train_df)
        train_df = train_df[~train_df["ref"].isin(dup_refs | bad_refs)].reset_index(drop=True)
        print(f"DROP_DUPLICATES: {before - len(train_df)} gambar dibuang → {len(train_df)} tersisa")

    # group id untuk duplicate-aware CV
    h_map = dict(zip(mtr["ref"], mtr["dup_key"]))
    if len(dup) and CFG.DUP_AWARE_CV:
        DUP_GROUP = train_df["ref"].map(h_map).fillna(train_df["ref"]).values

    if meta_test is not None:
        by_hash = mtr.groupby("ahash")["ref"].apply(list).to_dict()
        overlap_pairs = []
        for _, r in meta_test[meta_test["ok"] & (meta_test["ahash"] != "")].iterrows():
            for tr_ref in by_hash.get(r["ahash"], []):
                if thumb_diff(r["ref"], tr_ref) < DUP_PIXEL_TOL:
                    overlap_pairs.append((r["ref"], tr_ref))
                    break
        print(f"Train-test overlap (terverifikasi): {len(overlap_pairs)} gambar test identik dengan gambar train")
        if overlap_pairs:
            save_table(pd.DataFrame(overlap_pairs, columns=["test_ref", "train_ref"]), "train_test_overlap")
        if len(overlap_pairs) > 0.05 * max(len(meta_test), 1):
            print("⚠️ Overlap besar → kemungkinan data leak; laporkan/cek rules.")
    if n_groups == 0:
        print("✅ Tidak ada duplikat persis di train")

# %% [markdown]
# ## 7f. Brightness & train-vs-test shift
#
# **Cara membaca:** histogram brightness (rata-rata intensitas 0–1) per kelas dan train vs test.
#
# **Jika X maka Y:**
#
# - Satu kelas jauh lebih terang/gelap dari yang lain → model bisa belajar "shortcut" brightness, bukan objeknya.
#   Tambah `COLOR_JITTER` (0.3–0.4) supaya model tidak bergantung pada kecerahan.
# - Distribusi test bergeser dari train → domain shift; augmentasi warna/brightness lebih kuat, dan percayai CV sedikit
#   lebih rendah.
# - Brightness sangat rendah (< 0.15) → dataset gelap (malam / X-ray); normalisasi `NORM="dataset"` bisa membantu
#   model dari nol.
#
# **Untuk laporan:** cukup 1 kalimat kalau tidak ada shift; kalau ada, tampilkan figure ini.

# %%
if meta_train is not None and meta_train["ok"].any():
    mt = meta_train[meta_train["ok"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for c in grid_classes:
        vals = mt.loc[mt["label"] == c, "brightness"].dropna()
        if len(vals):
            axes[0].hist(vals, bins=30, alpha=0.45, label=str(c), density=True)
    axes[0].set_title("Brightness Distribution per Class")
    axes[0].set_xlabel("Mean brightness (0-1)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].hist(mt["brightness"].dropna(), bins=30, alpha=0.6, density=True, label="train")
    if meta_test is not None and meta_test["ok"].any():
        axes[1].hist(meta_test.loc[meta_test["ok"], "brightness"].dropna(), bins=30, alpha=0.6, density=True,
                     label="test")
    axes[1].set_title("Brightness: Train vs Test")
    axes[1].set_xlabel("Mean brightness (0-1)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    plt.tight_layout()
    save_fig("brightness_distribution", fig)

    bstats = mt.groupby("label")["brightness"].agg(["mean", "std", "count"]).round(3)
    display(bstats)
    spread = bstats["mean"].max() - bstats["mean"].min()
    if spread > 0.15:
        print(f"⚠️ Rata-rata brightness antar kelas berbeda {spread:.2f} → risiko shortcut, naikkan COLOR_JITTER.")
    if meta_test is not None and meta_test["ok"].any():
        diff = abs(meta_test.loc[meta_test["ok"], "brightness"].mean() - mt["brightness"].mean())
        print(f"Selisih mean brightness train vs test: {diff:.3f}",
              "⚠️ (shift)" if diff > 0.05 else "✅ (mirip)")

    eda_summary = pd.DataFrame([
        ("n_train", len(train_df)), ("n_test", len(test_df) if HAS_TEST else 0), ("n_classes", N_CLASSES),
        ("imbalance_ratio", round(IMB_RATIO, 2)),
        ("corrupted_train", int((~meta_train["ok"]).sum())), ("truncated_train", int(meta_train["truncated"].sum())),
        ("median_width", float(mt["width"].median())), ("median_height", float(mt["height"].median())),
        ("dataset_mean_rgb", str(DATASET_MEAN)), ("dataset_std_rgb", str(DATASET_STD)),
    ], columns=["item", "value"])
    display(eda_summary)
    save_table(eda_summary, "eda_summary")

# %% [markdown]
# # 8. Cross-validation folds
#
# **Tujuan:** membagi train menjadi K fold. Semua skor yang kita percayai adalah skor **out-of-fold (OOF)**: setiap
# gambar diprediksi oleh model yang tidak pernah melihatnya saat training.
#
# **Aturan pembagian (otomatis):**
#
# 1. `GROUP_COL` diisi (mis. `patient_id`) → `StratifiedGroupKFold` (gambar dari grup yang sama tidak terpisah).
# 2. Ada duplikat & `DUP_AWARE_CV=True` → duplikat diperlakukan sebagai 1 grup.
# 3. Selain itu → `StratifiedKFold` (proporsi kelas sama di setiap fold).
# 4. Ada kelas dengan anggota < `N_FOLDS` → jumlah fold dikurangi otomatis; kalau ada kelas dengan 1 gambar → `KFold` biasa.
# 5. `CV_STRATEGY="holdout"` → 1 split train/valid (lebih cepat, tapi skor lebih "berisik").
#
# **Cara membaca output:** tabel jumlah gambar per fold per kelas — harus kurang lebih rata.
#
# **Jika** dataset kecil (< 2000 gambar) → pakai 5 fold, bukan holdout. **Jika** waktu sangat terbatas → `FOLDS_TO_RUN=[0]`
# (cepat, tapi OOF hanya 1/K data dan ensemble kurang kuat).

# %%
def make_folds(df):
    y = df["y"].values
    folds = np.full(len(df), -1, dtype=int)
    min_count = int(pd.Series(y).value_counts().min())
    groups = None
    if CFG.GROUP_COL and "group" in df.columns:
        groups = df["group"].astype(str).values
        print("CV: grouping dengan GROUP_COL =", CFG.GROUP_COL)
    elif DUP_GROUP is not None and len(DUP_GROUP) == len(df):
        groups = np.asarray(DUP_GROUP).astype(str)
        print("CV: duplicate-aware (duplikat di fold yang sama)")

    if CFG.CV_STRATEGY == "holdout":
        idx = np.arange(len(df))
        if groups is not None:
            uniq = pd.unique(groups)
            _, va_g = train_test_split(uniq, test_size=CFG.HOLDOUT_SIZE, random_state=CFG.SEED)
            va = idx[np.isin(groups, va_g)]
        else:
            _, va = train_test_split(idx, test_size=CFG.HOLDOUT_SIZE, random_state=CFG.SEED,
                                     stratify=y if min_count >= 2 else None)
        folds[va] = 0
        return folds, 1

    n_splits = CFG.N_FOLDS
    if min_count < n_splits:
        n_splits = max(2, min_count)
        print(f"⚠️ Kelas terkecil hanya {min_count} gambar → N_FOLDS diturunkan ke {n_splits}")
    if groups is not None:
        if HAS_SGKF and min_count >= 2:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=CFG.SEED)
        else:
            splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(df, y, groups)
    elif min_count >= 2:
        split_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.SEED).split(df, y)
    else:
        print("⚠️ Ada kelas dengan 1 gambar → KFold biasa (tidak stratified)")
        split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=CFG.SEED).split(df)
    for k, (_, va) in enumerate(split_iter):
        folds[va] = k
    return folds, n_splits


train_df["fold"], N_SPLITS = make_folds(train_df)
FOLDS = sorted(train_df.loc[train_df["fold"] >= 0, "fold"].unique().tolist())
if CFG.FOLDS_TO_RUN is not None:
    FOLDS = [f for f in FOLDS if f in CFG.FOLDS_TO_RUN]
fold_table = pd.crosstab(train_df["fold"], train_df["label"])
display(fold_table)
print("Fold yang dijalankan:", FOLDS)
Y = train_df["y"].values

# %% [markdown]
# # 9. Metrics
#
# Semua metric dihitung dari matriks probabilitas `P` (N × K). Untuk binary, `P[:, 1]` = probabilitas kelas positif.
#
# | Metric | Kapan dipakai | Catatan |
# |---|---|---|
# | `accuracy` | kelas seimbang | menipu kalau imbalanced |
# | `f1_macro` | imbalanced / semua kelas sama penting | rata-rata F1 tiap kelas |
# | `f1` | binary, fokus kelas positif | threshold tuning membantu |
# | `logloss` | lomba minta probabilitas | sensitif ke prediksi yakin-tapi-salah → label smoothing & temperature membantu |
# | `roc_auc` | binary / ovr | hanya peduli urutan probabilitas |
# | `qwk` | label ordinal (tingkat keparahan 0–4) | salah jauh dihukum lebih berat |

# %%
def decide(P, decision=None):
    """Ubah probabilitas → label index, memakai threshold (binary) / class weights (multiclass) kalau ada."""
    P = np.asarray(P)
    if decision is not None and decision.get("type") == "threshold":
        return (P[:, 1] >= decision["value"]).astype(int)
    if decision is not None and decision.get("type") == "class_weights":
        return np.argmax(P * np.asarray(decision["value"])[None, :], axis=1)
    return np.argmax(P, axis=1)


def compute_metrics(y, P, decision=None):
    """Semua metric penting dalam 1 dict."""
    y = np.asarray(y)
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-7, 1 - 1e-7)
    P = P / P.sum(1, keepdims=True)
    pred = decide(P, decision)
    out = {"accuracy": accuracy_score(y, pred),
           "f1_macro": f1_score(y, pred, average="macro", labels=list(range(N_CLASSES)), zero_division=0)}
    if N_CLASSES == 2:
        out["f1"] = f1_score(y, pred, pos_label=1, zero_division=0)
    try:
        out["logloss"] = log_loss(y, P, labels=list(range(N_CLASSES)))
    except Exception:  # noqa
        out["logloss"] = np.nan
    try:
        if N_CLASSES == 2:
            out["roc_auc"] = roc_auc_score(y, P[:, 1])
        else:
            out["roc_auc"] = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(N_CLASSES)))
    except Exception:  # noqa
        out["roc_auc"] = np.nan
    try:
        out["qwk"] = cohen_kappa_score(y, pred, weights="quadratic")
    except Exception:  # noqa
        out["qwk"] = np.nan
    return out


def primary_score(y, P, decision=None):
    return compute_metrics(y, P, decision)[CFG.METRIC]


def is_better(new, best):
    if best is None or (isinstance(best, float) and np.isnan(best)):
        return True
    return new > best if GREATER_IS_BETTER else new < best


print("Contoh metric untuk prediksi uniform (baseline 'tebak rata'):")
_uniform = np.full((len(Y), N_CLASSES), 1.0 / N_CLASSES)
print({k: round(v, 4) for k, v in compute_metrics(Y, _uniform).items()})
_prior = np.tile(np.bincount(Y, minlength=N_CLASSES) / len(Y), (len(Y), 1))
print("Prediksi kelas mayoritas → accuracy:", round(accuracy_score(Y, _prior.argmax(1)), 4),
      "(model harus jauh di atas ini)")

# %% [markdown]
# # 10. Transforms & augmentasi
#
# **Dua jalur:**
#
# - `torchvision.transforms` kalau tersedia.
# - **Fallback buatan sendiri** (PIL + numpy) kalau torchvision tidak ada: resize, random resized crop, flip, rotasi,
#   color jitter (brightness/contrast/saturation), random erasing, normalize → tensor.
#
# **Level augmentasi (`AUG_LEVEL`):**
#
# | Level | Isi | Kapan |
# |---|---|---|
# | `none` | resize saja | debug / data sangat bersih |
# | `light` | flip + crop ringan | dataset besar (> 20k) |
# | `medium` | + rotasi, color jitter | default |
# | `strong` | crop agresif, rotasi besar, jitter kuat, random erasing | dataset kecil (< 2k) / overfit |
#
# **Jika** train accuracy ≈ 100% tapi valid jauh lebih rendah (overfit) → naikkan `AUG_LEVEL`, aktifkan mixup/cutmix.
# **Jika** train dan valid sama-sama rendah (underfit) → turunkan augmentasi, tambah epoch, model lebih besar.
#
# **Hati-hati:** augmentasi harus "label-preserving". Contoh: flip horizontal pada angka/huruf bisa mengubah makna
# (matikan `HFLIP`), rotasi 90° pada citra satelit aman (nyalakan `VFLIP`).
#
# **Progressive resizing (tips GPU):** latih dulu di `IMG_SIZE=128` beberapa epoch, lalu lanjut di 224 dengan
# `SKIP_TRAIN_IF_CKPT=False` — model belajar cepat di resolusi kecil lalu menyempurnakan detail. Di template ini
# cara paling sederhana: jalankan sekali dengan `IMG_SIZE=128` untuk eksplorasi cepat, run final di 224+.

# %%
AUG_PRESETS = {
    "none":   dict(scale=(1.0, 1.0), rot=0, jitter=0.0, erase=0.0),
    "light":  dict(scale=(0.85, 1.0), rot=0, jitter=0.0, erase=0.0),
    "medium": dict(scale=(0.7, 1.0), rot=CFG.ROTATION_DEG, jitter=CFG.COLOR_JITTER, erase=CFG.RANDOM_ERASING),
    "strong": dict(scale=(0.5, 1.0), rot=max(CFG.ROTATION_DEG, 30), jitter=max(CFG.COLOR_JITTER, 0.4),
                   erase=max(CFG.RANDOM_ERASING, 0.25)),
}


# ---------- fallback transforms (PIL + numpy), dipakai kalau torchvision tidak ada ----------
class PCompose:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, img):
        for op in self.ops:
            img = op(img)
        return img

    def __repr__(self):
        return "PCompose(" + ", ".join(type(o).__name__ for o in self.ops) + ")"


class PResize:
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, img):
        return img.resize((self.size, self.size), BILINEAR)


class PCenterCrop:
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, img):
        w, h = img.size
        left, top = max((w - self.size) // 2, 0), max((h - self.size) // 2, 0)
        return img.crop((left, top, left + self.size, top + self.size))


class PRandomResizedCrop:
    def __init__(self, size, scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3)):
        self.size, self.scale, self.ratio = int(size), scale, ratio

    def __call__(self, img):
        w, h = img.size
        area = w * h
        for _ in range(10):
            target = area * random.uniform(*self.scale)
            ar = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            cw, ch = int(round(math.sqrt(target * ar))), int(round(math.sqrt(target / ar)))
            if 0 < cw <= w and 0 < ch <= h:
                x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
                return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((self.size, self.size), BILINEAR)
        return img.resize((self.size, self.size), BILINEAR)


class PRandomFlip:
    def __init__(self, p=0.5, vertical=False):
        self.p, self.vertical = p, vertical

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.flip(img) if self.vertical else ImageOps.mirror(img)
        return img


class PRandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        if self.degrees <= 0:
            return img
        return img.rotate(random.uniform(-self.degrees, self.degrees), resample=BILINEAR)


class PColorJitter:
    def __init__(self, strength):
        self.s = strength

    def __call__(self, img):
        if self.s <= 0:
            return img
        for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            img = enh(img).enhance(random.uniform(max(0.0, 1 - self.s), 1 + self.s))
        return img


class PToTensorNormalize:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img):
        a = (np.asarray(img, dtype=np.float32) / 255.0 - self.mean) / self.std
        a = np.ascontiguousarray(a.transpose(2, 0, 1))
        return torch.from_numpy(a) if HAS_TORCH else a


class PRandomErasing:
    """Random erasing pada tensor/array (C,H,W)."""

    def __init__(self, p=0.25, scale=(0.02, 0.2)):
        self.p, self.scale = p, scale

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        _, h, w = x.shape
        area = h * w * random.uniform(*self.scale)
        eh = min(h, max(1, int(math.sqrt(area))))
        ew = min(w, max(1, int(area / eh)))
        y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
        x = x.clone() if HAS_TORCH else x.copy()
        x[:, y0:y0 + eh, x0:x0 + ew] = 0
        return x


def get_norm(meta=None):
    """Pilih mean/std normalisasi sesuai CFG.NORM dan status pretrained model."""
    if CFG.NORM == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if CFG.NORM == "dataset":
        return DATASET_MEAN, DATASET_STD
    if CFG.NORM == "none":
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    if meta is not None and meta.get("pretrained_ok"):
        return meta.get("mean", IMAGENET_MEAN), meta.get("std", IMAGENET_STD)
    return DATASET_MEAN, DATASET_STD


def build_train_transform(size, mean, std):
    a = AUG_PRESETS.get(CFG.AUG_LEVEL, AUG_PRESETS["medium"])
    use_flip = CFG.HFLIP and CFG.AUG_LEVEL != "none"
    use_vflip = CFG.VFLIP and CFG.AUG_LEVEL != "none"
    if HAS_TV:
        ops = [T.RandomResizedCrop(size, scale=a["scale"])] if a["scale"][0] < 1.0 else [T.Resize((size, size))]
        if use_flip:
            ops.append(T.RandomHorizontalFlip())
        if use_vflip:
            ops.append(T.RandomVerticalFlip())
        if a["rot"] > 0:
            ops.append(T.RandomRotation(a["rot"]))
        if a["jitter"] > 0:
            ops.append(T.ColorJitter(a["jitter"], a["jitter"], a["jitter"]))
        ops += [T.ToTensor(), T.Normalize(mean, std)]
        if a["erase"] > 0:
            ops.append(T.RandomErasing(p=a["erase"]))
        return T.Compose(ops)
    ops = [PRandomResizedCrop(size, scale=a["scale"])] if a["scale"][0] < 1.0 else [PResize(size)]
    if use_flip:
        ops.append(PRandomFlip(0.5))
    if use_vflip:
        ops.append(PRandomFlip(0.5, vertical=True))
    if a["rot"] > 0:
        ops.append(PRandomRotation(a["rot"]))
    if a["jitter"] > 0:
        ops.append(PColorJitter(a["jitter"]))
    ops.append(PToTensorNormalize(mean, std))
    if a["erase"] > 0:
        ops.append(PRandomErasing(a["erase"]))
    return PCompose(ops)


def build_eval_transform(size, mean, std, scale=1.0):
    """Resize (scale>1: resize lebih besar lalu center crop = zoom ringan untuk TTA multi-scale)."""
    big = int(round(size * scale))
    if HAS_TV:
        ops = [T.Resize((big, big))]
        if big != size:
            ops.append(T.CenterCrop(size))
        return T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])
    ops = [PResize(big)] + ([PCenterCrop(size)] if big != size else [])
    return PCompose(ops + [PToTensorNormalize(mean, std)])


print("Transform backend:", "torchvision" if HAS_TV else "fallback PIL/numpy")
print("Train transform  :", build_train_transform(CFG.IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD))

# %% [markdown]
# ## 10b. Preview augmentasi
#
# **Cara membaca:** kolom pertama = gambar asli, kolom berikutnya = hasil augmentasi acak. Pastikan objek masih
# terlihat dan label masih benar. **Jika** objek sering terpotong habis → naikkan batas bawah `scale` (preset lebih ringan).

# %%
def denorm_to_img(x, mean, std):
    a = x.numpy() if HAS_TORCH and hasattr(x, "numpy") else np.asarray(x)
    a = a.transpose(1, 2, 0) * np.asarray(std).reshape(1, 1, 3) + np.asarray(mean).reshape(1, 1, 3)
    return np.clip(a, 0, 1)


_tf_prev = build_train_transform(CFG.IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD)
_n_prev = min(4, len(train_df))
_ref_prev = train_df.groupby("y").head(1)["ref"].tolist()[:_n_prev]
fig, axes = plt.subplots(len(_ref_prev), 6, figsize=(12, 2.1 * len(_ref_prev)), squeeze=False)
for i, r in enumerate(_ref_prev):
    im = safe_load_image(r)
    axes[i, 0].imshow(im.resize((CFG.IMG_SIZE, CFG.IMG_SIZE)))
    axes[i, 0].set_title("original", fontsize=8)
    for j in range(1, 6):
        axes[i, j].imshow(denorm_to_img(_tf_prev(im), IMAGENET_MEAN, IMAGENET_STD))
        axes[i, j].set_title(f"aug {j}", fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
fig.suptitle(f"Augmentation Preview (AUG_LEVEL={CFG.AUG_LEVEL})")
plt.tight_layout()
_ = save_fig("augmentation_preview", fig)

# %% [markdown]
# # 11. Dataset, DataLoader, cache gambar
#
# **Kenapa `NUM_WORKERS = 0` di Windows?** DataLoader dengan `num_workers > 0` membuat proses baru memakai metode
# *spawn*. Proses baru harus meng-import ulang class `Dataset` — tapi class yang didefinisikan di dalam notebook tidak
# bisa di-import oleh proses lain, sehingga muncul error seperti
# `AttributeError: Can't get attribute 'ImageDataset' on <module '__main__'>` atau notebook **hang** tanpa pesan.
# Solusi aman: `NUM_WORKERS=0` + **cache gambar di RAM** (`CACHE_IMAGES`). Setelah epoch pertama, semua gambar sudah
# ter-decode & ter-resize di memori sehingga epoch berikutnya jauh lebih cepat. Di Linux/Mac, `NUM_WORKERS=2–8` aman.
#
# **Cache:** gambar disimpan sudah di-resize ke `IMG_SIZE × 1.15` (persegi). Estimasi memori dicetak di bawah;
# kalau melebihi `CACHE_MAX_GB`, cache dimatikan otomatis (mode `"auto"`).

# %%
IMAGE_CACHE = {}
CACHE_STATE = {"enabled": False, "size": None}


def setup_cache(n_images):
    tta_max = max([1.0] + list(CFG.TTA_SCALES or [1.0]))
    size = int(round(CFG.IMG_SIZE * max(1.15, tta_max)))
    est_gb = n_images * size * size * 3 / 1024 ** 3
    if CFG.CACHE_IMAGES == "auto":
        enabled = est_gb <= CFG.CACHE_MAX_GB
    else:
        enabled = bool(CFG.CACHE_IMAGES)
    if CACHE_STATE["size"] != size:
        IMAGE_CACHE.clear()
    CACHE_STATE.update(enabled=enabled, size=size)
    print(f"Image cache: {'ON' if enabled else 'OFF'} | size {size}px | estimasi {est_gb:.2f} GB untuk {n_images} gambar")


def get_image(ref):
    """Ambil gambar (dari cache kalau aktif)."""
    if CACHE_STATE["enabled"]:
        key = ref if ref is not None else "__none__"
        im = IMAGE_CACHE.get(key)
        if im is None:
            s = CACHE_STATE["size"]
            im = safe_load_image(ref).resize((s, s), BILINEAR)
            IMAGE_CACHE[key] = im
        return im
    return safe_load_image(ref)


class ImageDataset(Dataset):
    """Dataset generik: refs (list path/penunjuk pixel) + labels (opsional) + transform."""

    def __init__(self, refs, labels=None, transform=None):
        self.refs = list(refs)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        x = self.transform(get_image(self.refs[i]))
        y = -1 if self.labels is None else int(self.labels[i])
        return x, y


def make_loader(refs, labels, transform, batch_size, shuffle=False, sampler=None, drop_last=False):
    ds = ImageDataset(refs, labels, transform)
    g = torch.Generator()
    g.manual_seed(CFG.SEED)
    kw = dict(batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler,
              num_workers=CFG.NUM_WORKERS, pin_memory=(str(DEVICE) == "cuda"), drop_last=drop_last, generator=g)
    if CFG.NUM_WORKERS > 0:
        kw["persistent_workers"] = True
    return DataLoader(ds, **kw)


if HAS_TORCH:
    setup_cache(len(train_df) + (len(test_df) if HAS_TEST else 0))
    _dl = make_loader(train_df["ref"].values[:8], Y[:8], build_eval_transform(CFG.IMG_SIZE, IMAGENET_MEAN,
                      IMAGENET_STD), batch_size=8)
    _xb, _yb = next(iter(_dl))
    print("Batch check → x:", tuple(_xb.shape), _xb.dtype, f"range [{_xb.min():.2f}, {_xb.max():.2f}]",
          "| y:", _yb.tolist())
    assert _xb.shape[1] == 3 and _xb.shape[2] == CFG.IMG_SIZE, "Shape batch tidak sesuai"
    assert torch.isfinite(_xb).all(), "Ada NaN/inf di batch input"
    print("✅ DataLoader OK")
else:
    print("torch tidak ada → DataLoader di-skip")

# %% [markdown]
# # 12. Model factory: timm → torchvision → SimpleCNN
#
# **Urutan pencarian untuk setiap nama di `MODEL_POOL`:**
#
# 1. **timm** (kalau terinstall dan nama ada di `timm.list_models()`).
# 2. **torchvision** — nama gaya timm dipetakan ke padanannya (mis. `mobilenetv3_large_100` → `mobilenet_v3_large`).
# 3. **SimpleCNN** — CNN kecil pure-PyTorch (4 blok conv-BN-ReLU), dilatih dari nol. Selalu tersedia selama torch ada.
#
# Pretrained weights di-download sekali (butuh internet). **Jika** download gagal (offline / firewall) → otomatis
# lanjut dengan bobot acak + peringatan (hasil jauh lebih jelek: perbaiki koneksi atau pakai SimpleCNN + epoch lebih banyak).
#
# **Pilihan model (rule of thumb):**
#
# | Model | Params | Kapan |
# |---|---|---|
# | `mobilenetv3_large_100` | 5M | CPU-only, cepat |
# | `resnet18` / `resnet50` | 11M / 25M | baseline klasik, stabil |
# | `efficientnet_b0` | 5M | default bagus akurasi/kecepatan |
# | `convnext_tiny` | 28M | kuat, butuh GPU |
# | `vit_small_patch16_224` | 22M | dataset besar / pretrained kuat; butuh `IMG_SIZE` kelipatan 16 |
# | `eva02_small_patch14_224.mim_in22k` | 22M | sangat kuat (timm only), GPU |
# | `simple_cnn` | ~1M | tanpa timm/torchvision, gambar kecil (≤ 64 px), MNIST-like |
#
# **Error umum:** `RuntimeError: Unknown model` (timm) → cek ejaan dengan `timm.list_models("*effic*")`.
# `size mismatch for classifier` → jangan load state_dict dari model dengan jumlah kelas berbeda (lihat Troubleshooting).

# %%
TV_EQUIV = {
    "resnet18": "resnet18", "resnet34": "resnet34", "resnet50": "resnet50", "resnet101": "resnet101",
    "efficientnet_b0": "efficientnet_b0", "efficientnet_b1": "efficientnet_b1", "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3", "convnext_tiny": "convnext_tiny", "convnext_small": "convnext_small",
    "mobilenetv3_large_100": "mobilenet_v3_large", "mobilenetv3_small_100": "mobilenet_v3_small",
    "mobilenetv2_100": "mobilenet_v2", "densenet121": "densenet121", "regnety_008": "regnet_y_800mf",
    "regnety_016": "regnet_y_1_6gf", "vit_base_patch16_224": "vit_b_16", "efficientnetv2_s": "efficientnet_v2_s",
    "tf_efficientnetv2_s": "efficientnet_v2_s", "shufflenet_v2_x1_0": "shufflenet_v2_x1_0",
}
TRANSFORMER_KEYS = ("vit", "deit", "eva", "swin", "beit", "cait", "xcit")


class SimpleCNN(nn.Module):
    """CNN kecil pure-PyTorch. Menerima ukuran gambar berapa pun (global average pooling)."""

    def __init__(self, num_classes, widths=(32, 64, 128, 256), dropout=0.2):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))


def resolve_model(name):
    """Return (backend, nama_model) untuk satu entri MODEL_POOL."""
    if name == "simple_cnn" or not HAS_TORCH:
        return ("simple", "simple_cnn")
    if HAS_TIMM:
        base = name.split(".")[0]
        try:
            ok = timm.is_model(base) if hasattr(timm, "is_model") else (base in timm.list_models())
        except Exception:  # noqa
            ok = base in timm.list_models()
        if ok:
            return ("timm", name)
        similar = timm.list_models(f"*{base.split('_')[0]}*")[:8]
        print(f"⚠️ '{name}' tidak ada di timm. Mirip: {similar}")
    if HAS_TV:
        tv_name = TV_EQUIV.get(name, name)
        if hasattr(tvm, tv_name) and callable(getattr(tvm, tv_name)):
            return ("torchvision", tv_name)
        print(f"⚠️ '{name}' tidak punya padanan torchvision.")
    print(f"⚠️ '{name}' → diganti 'simple_cnn' (fallback pure PyTorch)")
    return ("simple", "simple_cnn")


def last_linear_name(model):
    name = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            name = n
    return name


def set_submodule(model, name, new):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def head_param_names(model):
    ln = last_linear_name(model)
    return {n for n, _ in model.named_parameters() if ln is not None and (n == ln or n.startswith(ln + "."))}


def build_model(spec, num_classes, pretrained=True, verbose=True):
    """Bangun model + metadata (backend, pretrained_ok, mean/std)."""
    backend, name = spec
    meta = {"backend": backend, "name": name, "pretrained_ok": False, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    model = None
    if backend == "timm":
        kw = {"drop_rate": CFG.DROPOUT}
        if any(k in name for k in TRANSFORMER_KEYS):
            kw["img_size"] = CFG.IMG_SIZE
        if pretrained:
            for kwargs in (kw, {}):
                try:
                    model = timm.create_model(name, pretrained=True, num_classes=num_classes, **kwargs)
                    meta["pretrained_ok"] = True
                    break
                except Exception as e:  # noqa
                    err = e
            if model is None and verbose:
                print(f"⚠️ pretrained timm '{name}' gagal ({type(err).__name__}: {str(err)[:150]}) → bobot acak")
        if model is None:
            try:
                model = timm.create_model(name, pretrained=False, num_classes=num_classes, **kw)
            except TypeError:
                model = timm.create_model(name, pretrained=False, num_classes=num_classes)
        cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None) or {}
        if "mean" in cfg and "std" in cfg:
            meta["mean"], meta["std"] = tuple(cfg["mean"]), tuple(cfg["std"])
    elif backend == "torchvision":
        fn = getattr(tvm, name)
        if pretrained:
            try:
                try:
                    model = fn(weights="DEFAULT", progress=False)
                except TypeError:
                    model = fn(pretrained=True, progress=False)        # torchvision < 0.13
                meta["pretrained_ok"] = True
            except Exception as e:  # noqa
                model = None
                if verbose:
                    print(f"⚠️ pretrained torchvision '{name}' gagal ({type(e).__name__}: {str(e)[:150]}) → bobot acak")
        if model is None:
            try:
                model = fn(weights=None)
            except TypeError:
                model = fn(pretrained=False)
        ln = last_linear_name(model)
        old = dict(model.named_modules())[ln]
        set_submodule(model, ln, nn.Linear(old.in_features, num_classes))
    else:
        model = SimpleCNN(num_classes, widths=CFG.SIMPLE_CNN_WIDTHS, dropout=CFG.DROPOUT)
    meta["mean"], meta["std"] = get_norm(meta)
    meta["n_params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
    model.cv_meta = meta
    return model


RESOLVED_MODELS = []
if HAS_TORCH:
    for _m in CFG.MODEL_POOL:
        spec = resolve_model(_m)
        if spec not in RESOLVED_MODELS:
            RESOLVED_MODELS.append(spec)
    print("Model yang akan dilatih:")
    for spec in RESOLVED_MODELS:
        print("  ", spec)
else:
    print("torch tidak ada → deep learning di-skip")

# %% [markdown]
# # 13. Sanity check: overfit satu batch
#
# **Tujuan:** sebelum training lama, pastikan pipeline benar. Model yang sehat **harus** bisa menghafal 1 batch kecil
# (loss turun mendekati 0, akurasi batch → 100%) dalam beberapa puluh langkah.
#
# **Cara membaca:** `loss awal → loss akhir`.
#
# - Loss turun drastis → ✅ pipeline (label, transform, model, optimizer) benar.
# - Loss tidak turun → ada bug: label teracak/tidak sinkron dengan gambar, LR terlalu kecil/besar, input semua nol
#   (cek batch check & preview augmentasi), atau model salah jumlah kelas.

# %%
def train_mode_backbone_frozen(model, freeze):
    heads = head_param_names(model)
    for n, p in model.named_parameters():
        p.requires_grad = (n in heads) or (not freeze)


if HAS_TORCH and CFG.RUN_DEEP and CFG.RUN_OVERFIT_TEST and RESOLVED_MODELS:
    try:
        seed_everything(CFG.SEED)
        spec = RESOLVED_MODELS[0]
        _m = build_model(spec, N_CLASSES, pretrained=CFG.PRETRAINED, verbose=False).to(DEVICE)
        mean, std = _m.cv_meta["mean"], _m.cv_meta["std"]
        bs = min(16, len(train_df))
        idx = train_df.groupby("y").head(max(1, bs // N_CLASSES)).index[:bs]
        _dl = make_loader(train_df.loc[idx, "ref"].values, Y[idx], build_eval_transform(CFG.IMG_SIZE, mean, std),
                          batch_size=bs)
        xb, yb = next(iter(_dl))
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt = torch.optim.Adam(_m.parameters(), lr=1e-3)
        _m.train()
        losses = []
        for step in range(CFG.OVERFIT_STEPS):
            opt.zero_grad()
            loss = F.cross_entropy(_m(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        _m.eval()
        with torch.no_grad():
            acc = (_m(xb).argmax(1) == yb).float().mean().item()
        print(f"Overfit test [{spec[1]}]: loss {losses[0]:.3f} → {losses[-1]:.3f} | batch acc (eval) {acc:.2f}")
        if losses[-1] < 0.5 * losses[0] or acc >= 0.9:
            print("✅ Model bisa belajar — pipeline OK")
        else:
            print("⚠️ Loss tidak turun signifikan. Cek label/transform/LR (lihat penjelasan di atas). "
                  "Untuk model besar + BatchNorm, langkah sedikit kadang belum cukup — naikkan OVERFIT_STEPS.")
        del _m, opt, xb, yb
    except Exception as e:  # noqa
        print("Overfit test gagal dijalankan:", type(e).__name__, str(e)[:200])
    finally:
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
else:
    print("Overfit test di-skip")

# %% [markdown]
# # 14. Baseline cepat: frozen features + LogisticRegression / LightGBM
#
# **Tujuan:** dapat skor referensi dalam hitungan menit, bahkan di CPU.
#
# - **Backbone features** (kalau timm/torchvision + pretrained tersedia): gambar → backbone pretrained (tanpa head) →
#   vektor embedding (mis. 1280 dimensi untuk efficientnet_b0) → LogisticRegression. Ini **linear probe** dan sering
#   mengejutkan kuatnya untuk dataset kecil (< 2000 gambar).
# - **Pixel features** (fallback, tidak butuh torch): gambar di-resize ke 32×32 → pixel mentah + histogram warna +
#   statistik gradien → StandardScaler → PCA → LogisticRegression / LightGBM.
#
# PCA & scaler di-fit **di dalam fold** (pipeline) sehingga tidak ada leakage.
#
# **Cara membaca output:** skor CV per model baseline. Bandingkan nanti dengan deep model.
#
# **Jika X maka Y:**
#
# - Baseline backbone ≥ deep fine-tuning → dataset terlalu kecil untuk fine-tuning / overfit; pakai `FREEZE_EPOCHS`,
#   augmentasi lebih kuat, atau cukup submit baseline/ensemble.
# - Baseline pixel sudah tinggi (> 0.9) → task mudah (warna/bentuk sederhana); SimpleCNN pun cukup.
# - Waktu tinggal sedikit / CPU-only → submit hasil ini (submission tetap dibuat di bagian akhir).
#
# **Untuk laporan:** sebutkan sebagai "baseline" di tabel perbandingan model.

# %%
def pixel_features(refs, size=None):
    """Fitur sederhana tanpa deep learning: pixel kecil + histogram warna + gradien."""
    size = size or CFG.FEAT_PIXEL_SIZE
    feats = []
    for r in progress(list(refs), every=5000, desc="pixel feats"):
        img = safe_load_image(r).resize((size, size), BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        hist = np.concatenate([np.histogram(a[..., c], bins=8, range=(0, 1))[0] / (size * size) for c in range(3)])
        gray = a.mean(2)
        gx, gy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
        extra = np.array([gx.mean(), gy.mean(), gx.std(), gy.std(), a.mean(), a.std()], dtype=np.float32)
        feats.append(np.concatenate([a.ravel(), hist.astype(np.float32), extra]))
    return np.stack(feats).astype(np.float32)


def backbone_features(refs, spec, batch_size=64):
    """Embedding dari backbone pretrained (head diganti Identity). Return None kalau pretrained tidak tersedia."""
    model = build_model(spec, N_CLASSES, pretrained=True)
    if not model.cv_meta["pretrained_ok"]:
        print("  backbone tidak pretrained → embedding tidak berguna, pakai pixel features")
        return None, None
    meta = model.cv_meta
    set_submodule(model, last_linear_name(model), nn.Identity())
    model = model.to(DEVICE).eval()
    tf = build_eval_transform(CFG.IMG_SIZE, meta["mean"], meta["std"])
    out = []
    with torch.no_grad():
        for xb, _ in make_loader(refs, None, tf, batch_size=batch_size):
            xb = xb.to(DEVICE)
            with autocast_ctx(USE_AMP):
                f = model(xb)
            if not torch.isfinite(f).all():
                f = fp32_fallback(model, xb)
            if f.ndim > 2:
                f = f.flatten(2).mean(-1)
            out.append(f.float().cpu().numpy())
    del model
    gc.collect()
    return np.concatenate(out), meta


def autocast_ctx(enabled):
    """Autocast yang kompatibel dengan torch lama & baru."""
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


class AmpNaNError(RuntimeError):
    """Loss NaN saat AMP aktif → fold diulang dalam fp32."""


def disable_amp(reason):
    global USE_AMP
    if USE_AMP:
        USE_AMP = False
        print(f"  ⚠️ {reason} → AMP (fp16) dimatikan untuk sisa run (fp32 lebih lambat tapi stabil).")


def fp32_fallback(model, x):
    """Hitung ulang output dalam fp32 (dipakai kalau fp16 menghasilkan NaN/inf)."""
    disable_amp("Output NaN/inf dengan fp16")
    return model(x.float())


def class_weight_dict(y):
    counts = np.bincount(y, minlength=N_CLASSES)
    w = len(y) / (N_CLASSES * np.maximum(counts, 1))
    return {i: float(w[i]) for i in range(N_CLASSES)}


CANDIDATES = {}   # nama -> {"oof": (N,K) dengan NaN di baris tanpa prediksi, "test": (M,K) atau None, "cv": skor}


def register_candidate(name, oof, test_pred, extra=None):
    in_folds = np.isin(train_df["fold"].values, FOLDS)
    bad = in_folds & ~np.isfinite(oof).all(1)
    if bad.any():
        print(f"  ⚠️ {name}: {bad.sum()} baris OOF NaN/inf → diisi prior kelas (cek AMP / data)")
        oof[bad] = np.bincount(Y, minlength=N_CLASSES) / len(Y)
    if test_pred is not None and not np.isfinite(test_pred).all():
        badt = ~np.isfinite(test_pred).all(1)
        print(f"  ⚠️ {name}: {badt.sum()} baris prediksi test NaN/inf → diisi prior kelas")
        test_pred[badt] = np.bincount(Y, minlength=N_CLASSES) / len(Y)
    mask = ~np.isnan(oof[:, 0])
    sc = primary_score(Y[mask], oof[mask]) if mask.any() else np.nan
    CANDIDATES[name] = {"oof": oof, "test": test_pred, "cv": sc, **(extra or {})}
    return sc


FEAT_INFO = {}
if CFG.RUN_FEATURE_BASELINE:
    t0 = time.time()
    X_feat = X_feat_test = None
    src = CFG.FEATURE_SOURCE
    if src in ("auto", "backbone") and HAS_TORCH and (HAS_TIMM or HAS_TV) and CFG.PRETRAINED:
        fb = resolve_model(CFG.FEATURE_BACKBONE) if CFG.FEATURE_BACKBONE else \
            (RESOLVED_MODELS[0] if RESOLVED_MODELS else ("simple", "simple_cnn"))
        if fb[0] != "simple":
            try:
                print("Ekstraksi embedding dengan backbone:", fb)
                X_feat, _ = backbone_features(train_df["ref"].values, fb)
                if X_feat is not None and HAS_TEST:
                    X_feat_test, _ = backbone_features(test_df["ref"].values, fb)
                FEAT_INFO["source"] = f"backbone:{fb[1]}"
            except Exception as e:  # noqa
                print("⚠️ Ekstraksi backbone gagal:", type(e).__name__, str(e)[:200])
                X_feat = None
    if X_feat is None:
        print("Pixel features (resize", CFG.FEAT_PIXEL_SIZE, ")")
        X_feat = pixel_features(train_df["ref"].values)
        X_feat_test = pixel_features(test_df["ref"].values) if HAS_TEST else None
        FEAT_INFO["source"] = "pixels"
    print(f"Feature matrix: {X_feat.shape} | sumber: {FEAT_INFO['source']} | {time.time() - t0:.1f}s")

    def make_logreg(n_train):
        n_comp = int(min(CFG.PCA_COMPONENTS, n_train - 1, X_feat.shape[1]))
        steps = [StandardScaler()]
        if X_feat.shape[1] > n_comp:
            steps.append(PCA(n_components=n_comp, random_state=CFG.SEED))
        cw = "balanced" if CFG.CLASS_WEIGHT == "balanced" else None
        steps.append(LogisticRegression(C=CFG.LOGREG_C, max_iter=3000, class_weight=cw))
        return make_pipeline(*steps)

    def make_lgbm(n_train):
        n_comp = int(min(64, n_train - 1, X_feat.shape[1]))
        steps = [StandardScaler()]
        if X_feat.shape[1] > n_comp:
            steps.append(PCA(n_components=n_comp, random_state=CFG.SEED))
        steps.append(lgb.LGBMClassifier(n_estimators=300 if CFG.RUN_MODE == "full" else 100, learning_rate=0.05,
                                        num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                        class_weight="balanced" if CFG.CLASS_WEIGHT == "balanced" else None,
                                        random_state=CFG.SEED, verbose=-1))
        return make_pipeline(*steps)

    base_models = {"feat_logreg": make_logreg}
    if HAS_LGB and CFG.RUN_LGB_ON_FEATURES:
        base_models["feat_lgbm"] = make_lgbm
    elif not HAS_LGB:
        print("Info: lightgbm tidak ada → feat_lgbm di-skip")

    base_rows = []
    for bname, factory in base_models.items():
        try:
            t1 = time.time()
            oof = np.full((len(train_df), N_CLASSES), np.nan)
            tpred = np.zeros((len(test_df), N_CLASSES)) if HAS_TEST else None
            fold_scores = []
            for k in FOLDS:
                tr = np.where(train_df["fold"].values != k)[0]
                va = np.where(train_df["fold"].values == k)[0]
                mdl = factory(len(tr))
                mdl.fit(X_feat[tr], Y[tr])
                p = np.zeros((len(va), N_CLASSES))
                p[:, mdl.classes_] = mdl.predict_proba(X_feat[va])
                oof[va] = p
                fold_scores.append(primary_score(Y[va], p))
                if HAS_TEST:
                    pt = np.zeros((len(test_df), N_CLASSES))
                    pt[:, mdl.classes_] = mdl.predict_proba(X_feat_test)
                    tpred += pt / len(FOLDS)
            sc = register_candidate(bname, oof, tpred, {"backend": FEAT_INFO["source"], "fold_scores": fold_scores})
            base_rows.append({"model": bname, "features": FEAT_INFO["source"], f"cv_{CFG.METRIC}": round(sc, 4),
                              "fold_std": round(float(np.std(fold_scores)), 4), "sec": round(time.time() - t1, 1)})
            print(f"  {bname:12s} CV {CFG.METRIC} = {sc:.4f} (fold std {np.std(fold_scores):.4f})")
        except Exception as e:  # noqa
            FAILED_MODELS[bname] = f"{type(e).__name__}: {e}"
            print(f"  ⚠️ {bname} gagal: {type(e).__name__}: {str(e)[:200]}")
    if base_rows:
        display(pd.DataFrame(base_rows))
        save_table(pd.DataFrame(base_rows), "baseline_results")
else:
    print("RUN_FEATURE_BASELINE=False → di-skip")

# %% [markdown]
# # 15. Training deep learning (K-fold)
#
# **Yang terjadi di setiap fold:**
#
# 1. Bangun model (pretrained), optimizer **AdamW** dengan 2 grup LR (backbone `LR`, head `LR × HEAD_LR_MULT`).
# 2. Scheduler **linear warmup → cosine decay** per step.
# 3. Loss **CrossEntropy + label smoothing** (+ class weights kalau `CLASS_WEIGHT="balanced"`).
# 4. Opsional: **mixup / cutmix**, **weighted sampler**, **gradient accumulation**, **freeze→unfreeze**
#    (`FREEZE_EPOCHS` epoch pertama hanya head = linear probe, lalu semua layer dilatih).
# 5. **AMP** (mixed precision) otomatis kalau CUDA → ~2× lebih cepat & hemat memori.
# 6. Setiap epoch: evaluasi di fold validasi → simpan checkpoint **terbaik menurut `METRIC`**; **early stopping**
#    kalau `PATIENCE` epoch tanpa perbaikan.
# 7. Load checkpoint terbaik → prediksi validasi (OOF) & test **dengan TTA**.
# 8. **CUDA OOM** → batch size otomatis dibagi 2 (accum dikali 2 supaya effective batch tetap), lalu fold diulang.
#
# **Cara membaca log per epoch:** `train_loss` harus turun. `val_<metric>` naik lalu mendatar. Tanda `*` = checkpoint
# baru tersimpan.
#
# **Jika X maka Y:**
#
# - `train_loss` turun tapi `val` memburuk sejak epoch awal → overfit: augmentasi lebih kuat, mixup, `FREEZE_EPOCHS=1–2`,
#   LR lebih kecil, model lebih kecil.
# - `train_loss` tidak turun → LR terlalu kecil/besar (coba 1e-4 dan 1e-3), cek overfit test.
# - `loss = nan` → matikan AMP (`USE_AMP=False`), turunkan LR, cek gambar korup.
# - Skor antar fold beda jauh (std > 0.02–0.03) → model tidak stabil / data kecil: tambah epoch, ensemble, seed lain.
# - Best epoch selalu epoch terakhir → model masih belajar, tambah `EPOCHS`.
# - Best epoch selalu epoch 1–2 → LR terlalu besar atau overfit cepat.
#
# **Untuk laporan:** figure learning curve + tabel skor CV per fold (mean ± std). Tuliskan hyperparameter utama
# (model, IMG_SIZE, epoch, LR, batch, augmentasi) di bagian Metodologi.

# %%
def make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_criterion(class_weights=None):
    w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    try:
        return nn.CrossEntropyLoss(weight=w, label_smoothing=CFG.LABEL_SMOOTHING)
    except TypeError:  # torch < 1.10
        return nn.CrossEntropyLoss(weight=w)


def warmup_cosine(total_steps, warmup_steps, min_ratio):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return fn


def mix_batch(x, y):
    """Mixup atau CutMix. Return x_mix, y_a, y_b, lam."""
    idx = torch.randperm(x.size(0), device=x.device)
    use_cutmix = CFG.CUTMIX_ALPHA > 0 and (CFG.MIXUP_ALPHA <= 0 or random.random() < 0.5)
    if use_cutmix:
        lam = np.random.beta(CFG.CUTMIX_ALPHA, CFG.CUTMIX_ALPHA)
        H, W = x.shape[2], x.shape[3]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / float(H * W)
    else:
        lam = float(np.random.beta(CFG.MIXUP_ALPHA, CFG.MIXUP_ALPHA))
        x = lam * x + (1 - lam) * x[idx]
    return x, y, y[idx], lam


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, accum):
    model.train()
    total, n, nan_batches = 0.0, 0, 0
    nb = len(loader)
    optimizer.zero_grad(set_to_none=True)
    use_mix = CFG.MIXUP_ALPHA > 0 or CFG.CUTMIX_ALPHA > 0
    params = [p for p in model.parameters() if p.requires_grad]
    for step, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        mixed = use_mix and random.random() < CFG.MIX_PROB
        if mixed:
            x, ya, yb, lam = mix_batch(x, y)
        with autocast_ctx(USE_AMP):
            out = model(x)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb) if mixed else criterion(out, y)
        if not torch.isfinite(loss):
            if USE_AMP:
                raise AmpNaNError("Loss NaN/inf dengan AMP")
            nan_batches += 1
            optimizer.zero_grad(set_to_none=True)
            if nan_batches > max(3, 0.2 * nb):
                raise FloatingPointError("Loss NaN/inf berulang kali → matikan AMP / turunkan LR / cek data.")
            continue
        scaler.scale(loss / accum).backward()
        if (step + 1) % accum == 0 or (step + 1) == nb:
            if CFG.GRAD_CLIP and CFG.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, CFG.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def predict_proba(model, refs, mean, std, tta=False, batch_size=64):
    """Probabilitas (N,K). TTA: rata-rata hflip & multi-scale."""
    refs = list(refs)
    if len(refs) == 0:
        return np.zeros((0, N_CLASSES))
    scales = list(CFG.TTA_SCALES) if (tta and CFG.TTA_SCALES) else [1.0]
    model.eval()
    acc = np.zeros((len(refs), N_CLASSES))
    for sc in scales:
        tf = build_eval_transform(CFG.IMG_SIZE, mean, std, scale=sc)
        outs = []
        with torch.no_grad():
            for x, _ in make_loader(refs, None, tf, batch_size=batch_size):
                x = x.to(DEVICE, non_blocking=True)
                views = [x, torch.flip(x, dims=[3])] if (tta and CFG.TTA_HFLIP) else [x]
                p = 0
                for v in views:
                    with autocast_ctx(USE_AMP):
                        logits = model(v)
                    if not torch.isfinite(logits).all():
                        logits = fp32_fallback(model, v)
                    p = p + torch.softmax(logits.float(), 1) / len(views)
                outs.append(p.float().cpu().numpy())
        acc += np.concatenate(outs)
    return acc / len(scales)


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


def ckpt_path(spec, fold):
    return os.path.join(MODEL_DIR, f"{safe_name(spec[1])}_s{CFG.IMG_SIZE}_fold{fold}.pt")


def is_oom(e):
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def train_fold(spec, fold, batch_size, accum):
    """Latih 1 fold. Return dict (val_probs, test_probs, history, best_score, best_epoch)."""
    seed_everything(CFG.SEED + fold, CFG.DETERMINISTIC)
    tr_idx = np.where(train_df["fold"].values != fold)[0]
    va_idx = np.where(train_df["fold"].values == fold)[0]
    model = build_model(spec, N_CLASSES, pretrained=CFG.PRETRAINED, verbose=(fold == FOLDS[0])).to(DEVICE)
    meta = model.cv_meta
    mean, std = meta["mean"], meta["std"]
    path = ckpt_path(spec, fold)
    history = []
    best_score, best_epoch, best_loss = None, -1, np.inf

    if CFG.SKIP_TRAIN_IF_CKPT and os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"  fold {fold}: checkpoint ditemukan → skip training ({path})")
    else:
        y_tr = Y[tr_idx]
        sampler = None
        if CFG.USE_WEIGHTED_SAMPLER:
            cnt = np.bincount(y_tr, minlength=N_CLASSES)
            sampler = WeightedRandomSampler(torch.as_tensor(1.0 / cnt[y_tr], dtype=torch.double), len(y_tr),
                                            replacement=True)
        tr_loader = make_loader(train_df["ref"].values[tr_idx], y_tr,
                                build_train_transform(CFG.IMG_SIZE, mean, std), batch_size=batch_size,
                                shuffle=True, sampler=sampler, drop_last=len(tr_idx) > batch_size)
        cw = None
        if CFG.CLASS_WEIGHT == "balanced":
            d = class_weight_dict(y_tr)
            cw = [d[i] for i in range(N_CLASSES)]
        criterion = make_criterion(cw)
        heads = head_param_names(model)
        base_lr = CFG.LR if meta["pretrained_ok"] else CFG.LR_SCRATCH
        head_mult = CFG.HEAD_LR_MULT if meta["pretrained_ok"] else 1.0
        body_p = [p for n, p in model.named_parameters() if n not in heads]
        head_p = [p for n, p in model.named_parameters() if n in heads]
        groups = [{"params": body_p, "lr": base_lr}, {"params": head_p, "lr": base_lr * head_mult}]
        groups = [g for g in groups if len(g["params"])]
        optimizer = torch.optim.AdamW(groups, lr=base_lr, weight_decay=CFG.WEIGHT_DECAY)
        steps_per_epoch = max(1, math.ceil(len(tr_loader) / accum))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_cosine(steps_per_epoch * CFG.EPOCHS, steps_per_epoch * CFG.WARMUP_EPOCHS,
                                     CFG.MIN_LR_RATIO))
        scaler = make_grad_scaler(USE_AMP)
        freeze_epochs = CFG.FREEZE_EPOCHS if meta["pretrained_ok"] else 0
        bad_epochs = 0
        for epoch in range(CFG.EPOCHS):
            t0 = time.time()
            if freeze_epochs > 0:
                train_mode_backbone_frozen(model, freeze=epoch < freeze_epochs)
            tr_loss = train_one_epoch(model, tr_loader, optimizer, scheduler, scaler, criterion, accum)
            vp = predict_proba(model, train_df["ref"].values[va_idx], mean, std, tta=False, batch_size=batch_size * 2)
            m = compute_metrics(Y[va_idx], vp)
            score = m[CFG.METRIC]
            # skor sama (mis. metric sudah 1.0) → pakai val_loss lebih kecil sebagai tie-breaker
            improved = is_better(score, best_score) or (best_score is not None and score == best_score
                                                        and m["logloss"] < best_loss)
            if improved:
                best_score, best_epoch, bad_epochs, best_loss = score, epoch, 0, m["logloss"]
                torch.save(model.state_dict(), path)
            else:
                bad_epochs += 1
            lr_now = optimizer.param_groups[0]["lr"]
            history.append({"fold": fold, "epoch": epoch + 1, "lr": lr_now, "train_loss": tr_loss,
                            "val_loss": m["logloss"], f"val_{CFG.METRIC}": score, "sec": time.time() - t0})
            frz = " [frozen]" if epoch < freeze_epochs else ""
            print(f"  fold {fold} ep {epoch + 1:02d}/{CFG.EPOCHS} | lr {lr_now:.2e} | train_loss {tr_loss:.4f} | "
                  f"val_loss {m['logloss']:.4f} | val_{CFG.METRIC} {score:.4f} | {time.time() - t0:.1f}s"
                  f"{' *' if improved else ''}{frz}")
            if bad_epochs >= CFG.PATIENCE:
                print(f"  early stopping (patience {CFG.PATIENCE})")
                break
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        if freeze_epochs > 0:
            train_mode_backbone_frozen(model, freeze=False)

    bs_inf = batch_size * 2
    val_probs = predict_proba(model, train_df["ref"].values[va_idx], mean, std, tta=CFG.TTA_ON_OOF, batch_size=bs_inf)
    test_probs = predict_proba(model, test_df["ref"].values, mean, std, tta=True, batch_size=bs_inf) if HAS_TEST else None
    val_score = primary_score(Y[va_idx], val_probs)
    del model
    return {"va_idx": va_idx, "val_probs": val_probs, "test_probs": test_probs, "history": history,
            "best_epoch": best_epoch + 1, "score": val_score, "meta": meta}


def train_fold_with_retry(spec, fold):
    bs, accum = CFG.BATCH_SIZE, CFG.ACCUM_STEPS
    for attempt in range(CFG.OOM_RETRIES + 1):
        oom = False
        try:
            return train_fold(spec, fold, bs, accum), bs
        except AmpNaNError:
            disable_amp("Loss NaN/inf saat training dengan fp16")
            print(f"  ulang fold {fold} dalam fp32")
            gc.collect()
            continue
        except RuntimeError as e:
            if not is_oom(e) or bs // 2 < CFG.MIN_BATCH_SIZE:
                raise
            oom = True
        if oom:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bs, accum = bs // 2, accum * 2
            print(f"  ⚠️ CUDA OOM → ulang fold {fold} dengan batch_size={bs}, accum={accum}")
    raise RuntimeError("OOM terus walau batch sudah kecil → turunkan IMG_SIZE / pakai model lebih kecil")


def run_cv(spec):
    name = spec[1]
    oof = np.full((len(train_df), N_CLASSES), np.nan)
    tpred = np.zeros((len(test_df), N_CLASSES)) if HAS_TEST else None
    hist, fold_scores, best_epochs, n_test_folds = [], [], [], 0
    t0 = time.time()
    meta = None
    for fold in FOLDS:
        if CFG.MAX_MINUTES_PER_MODEL and (time.time() - t0) / 60 > CFG.MAX_MINUTES_PER_MODEL and fold_scores:
            print(f"  batas waktu {CFG.MAX_MINUTES_PER_MODEL} menit tercapai → fold sisanya di-skip")
            break
        res, used_bs = train_fold_with_retry(spec, fold)
        meta = res["meta"]
        oof[res["va_idx"]] = res["val_probs"]
        if HAS_TEST:
            tpred += res["test_probs"]
            n_test_folds += 1
        hist += res["history"]
        fold_scores.append(res["score"])
        best_epochs.append(res["best_epoch"])
        print(f"  >> fold {fold}: {CFG.METRIC} = {res['score']:.4f} (best epoch {res['best_epoch']}, batch {used_bs})")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if HAS_TEST and n_test_folds:
        tpred /= n_test_folds
    return {"oof": oof, "test": tpred, "history": pd.DataFrame(hist), "fold_scores": fold_scores,
            "best_epochs": best_epochs, "minutes": (time.time() - t0) / 60, "meta": meta}


DEEP_RESULTS = {}
if HAS_TORCH and CFG.RUN_DEEP and RESOLVED_MODELS:
    for spec in RESOLVED_MODELS:
        tag = f"{spec[0]}:{spec[1]}"
        print("=" * 90)
        print(f"MODEL {tag} | img {CFG.IMG_SIZE} | epochs {CFG.EPOCHS} | folds {FOLDS} | device {DEVICE}")
        print("=" * 90)
        try:
            res = run_cv(spec)
            DEEP_RESULTS[spec[1]] = res
            sc = register_candidate(spec[1], res["oof"], res["test"],
                                    {"backend": spec[0], "fold_scores": res["fold_scores"]})
            fs = res["fold_scores"]
            print(f"✅ {tag}: OOF {CFG.METRIC} = {sc:.4f} | fold mean {np.mean(fs):.4f} ± {np.std(fs):.4f} | "
                  f"{res['minutes']:.1f} min | pretrained={res['meta']['pretrained_ok']}")
            if len(fs) > 1 and np.std(fs) > 0.03:
                print("⚠️ Variansi antar fold tinggi → model kurang stabil (lihat 'Jika X maka Y' di atas)")
            res["history"].to_csv(os.path.join(TAB_DIR, f"history_{safe_name(spec[1])}.csv"), index=False)
        except Exception as e:  # noqa
            FAILED_MODELS[spec[1]] = f"{type(e).__name__}: {e}"
            print(f"❌ {tag} gagal: {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc(limit=3)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
else:
    print("Deep learning di-skip (RUN_DEEP=False / torch tidak ada / tidak ada model).")

print("\nFAILED_MODELS:", FAILED_MODELS if FAILED_MODELS else "tidak ada")
if LOAD_ERRORS:
    print("⚠️ Error saat load gambar (diganti gambar hitam):", dict(LOAD_ERRORS))

# %% [markdown]
# ## 15b. Learning curves & ringkasan CV
#
# **Cara membaca learning curve:**
#
# - Kiri (train loss): harus turun mulus.
# - Kanan (val metric per epoch, tiap garis = 1 fold): naik lalu mendatar. Garis antar fold berdekatan = stabil.
# - Gap besar antara train loss sangat rendah & val memburuk = overfit.
#
# **Untuk laporan:** "Gambar X. Learning curve model terbaik" + tabel `cv_results.csv` (mean ± std per model).

# %%
for name, res in DEEP_RESULTS.items():
    h = res["history"]
    if h.empty:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for f, g in h.groupby("fold"):
        axes[0].plot(g["epoch"], g["train_loss"], marker="o", label=f"fold {f} train")
        axes[0].plot(g["epoch"], g["val_loss"], marker="x", linestyle="--", label=f"fold {f} val")
        axes[1].plot(g["epoch"], g[f"val_{CFG.METRIC}"], marker="o", label=f"fold {f}")
    axes[0].set_title(f"Loss per Epoch — {name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(fontsize=7)
    axes[1].set_title(f"Validation {CFG.METRIC} per Epoch — {name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(CFG.METRIC)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    save_fig(f"learning_curve_{safe_name(name)}", fig)

cv_rows = []
for name, c in CANDIDATES.items():
    mask = ~np.isnan(c["oof"][:, 0])
    m = compute_metrics(Y[mask], c["oof"][mask])
    fs = c.get("fold_scores", [])
    cv_rows.append({"model": name, "backend": c.get("backend", ""), f"oof_{CFG.METRIC}": round(c["cv"], 4),
                    "fold_mean": round(float(np.mean(fs)), 4) if fs else np.nan,
                    "fold_std": round(float(np.std(fs)), 4) if fs else np.nan,
                    **{k: round(v, 4) for k, v in m.items()}, "n_oof": int(mask.sum())})
CV_TABLE = pd.DataFrame(cv_rows)
if len(CV_TABLE):
    CV_TABLE = CV_TABLE.sort_values(f"oof_{CFG.METRIC}", ascending=not GREATER_IS_BETTER).reset_index(drop=True)
    display(CV_TABLE)
    save_table(CV_TABLE, "cv_results")
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(CV_TABLE) + 1)))
    ax.barh(CV_TABLE["model"], CV_TABLE[f"oof_{CFG.METRIC}"], xerr=CV_TABLE["fold_std"].fillna(0), color="#4C72B0")
    ax.invert_yaxis()
    ax.set_title(f"Model Comparison (OOF {CFG.METRIC})")
    ax.set_xlabel(CFG.METRIC)
    ax.set_ylabel("Model")
    save_fig("model_comparison", fig)
else:
    print("Tidak ada model yang berhasil — cek FAILED_MODELS dan Troubleshooting.")

# %% [markdown]
# # 16. Ensemble (hill climbing di OOF)
#
# **Tujuan:** menggabungkan beberapa model (deep + baseline) dengan bobot yang dicari di OOF.
# Algoritma *greedy hill climbing* (Caruana): mulai dari model terbaik, lalu berulang kali menambahkan model
# (boleh sama) yang paling menaikkan skor OOF. Bobot = berapa kali model dipilih.
#
# Hanya model yang punya prediksi OOF di **baris yang sama** dan prediksi test yang ikut.
#
# **Cara membaca:** bobot tiap model + skor ensemble vs model tunggal terbaik. Final = yang skornya lebih baik.
#
# **Jika X maka Y:**
#
# - Ensemble hanya naik sangat sedikit (< 0.002) → model-model terlalu mirip; tambah model yang beda arsitektur
#   (CNN + ViT) atau beda resolusi.
# - Bobot hampir semua ke 1 model → model lain jauh lebih lemah; tidak apa-apa.
# - Dataset sangat kecil → ensemble OOF bisa overfit; lebih aman rata-rata sederhana top-2/top-3.
#
# **Untuk laporan:** tuliskan komposisi ensemble final dan peningkatannya dibanding model tunggal.

# %%
def hill_climb(names, oofs, y, iters=25):
    scores = {n: primary_score(y, oofs[n]) for n in names}
    order = sorted(names, key=lambda n: scores[n], reverse=GREATER_IS_BETTER)
    chosen = [order[0]]
    cur = oofs[order[0]].copy()
    best = scores[order[0]]
    for _ in range(iters):
        cand_best, cand_name = None, None
        for n in names:
            trial = (cur * len(chosen) + oofs[n]) / (len(chosen) + 1)
            s = primary_score(y, trial)
            if cand_best is None or is_better(s, cand_best):
                cand_best, cand_name = s, n
        if cand_name is None or not is_better(cand_best, best) or abs(cand_best - best) < 1e-6:
            break
        chosen.append(cand_name)
        cur = (cur * (len(chosen) - 1) + oofs[cand_name]) / len(chosen)
        best = cand_best
    w = Counter(chosen)
    return {n: w[n] / len(chosen) for n in w}, best


FINAL = None
if CANDIDATES:
    common = None
    for n, c in CANDIDATES.items():
        m = ~np.isnan(c["oof"][:, 0])
        common = m if common is None else (common & m)
    usable = [n for n, c in CANDIDATES.items() if (c["test"] is not None or not HAS_TEST)]
    best_single = sorted(usable, key=lambda n: CANDIDATES[n]["cv"], reverse=GREATER_IS_BETTER)[0]
    FINAL = {"name": best_single, "oof": CANDIDATES[best_single]["oof"], "test": CANDIDATES[best_single]["test"],
             "weights": {best_single: 1.0}, "cv": CANDIDATES[best_single]["cv"]}
    print(f"Model tunggal terbaik: {best_single} (OOF {CFG.METRIC} {FINAL['cv']:.4f})")
    if CFG.RUN_ENSEMBLE and len(usable) >= 2 and common.sum() > 0:
        oofs = {n: CANDIDATES[n]["oof"][common] for n in usable}
        weights, ens_score = hill_climb(usable, oofs, Y[common], CFG.ENSEMBLE_ITERS)
        single_on_common = primary_score(Y[common], oofs[best_single])
        print("Bobot ensemble:", {k: round(v, 3) for k, v in weights.items()})
        print(f"Ensemble OOF {CFG.METRIC}: {ens_score:.4f} vs best single {single_on_common:.4f} "
              f"(baris OOF bersama: {int(common.sum())})")
        if len(weights) > 1 and is_better(ens_score, single_on_common):
            ens_oof = np.full_like(CANDIDATES[best_single]["oof"], np.nan)
            ens_oof[common] = sum(w * CANDIDATES[n]["oof"][common] for n, w in weights.items())
            ens_test = sum(w * CANDIDATES[n]["test"] for n, w in weights.items()) if HAS_TEST else None
            FINAL = {"name": "ensemble", "oof": ens_oof, "test": ens_test, "weights": weights, "cv": ens_score}
            print("✅ Ensemble dipakai sebagai prediksi final")
        else:
            print("Info: Ensemble tidak lebih baik → pakai model tunggal terbaik")
    save_table(pd.DataFrame([{"model": k, "weight": v} for k, v in FINAL["weights"].items()]), "final_blend_weights")
else:
    print("⚠️ Tidak ada kandidat prediksi. Cek error di atas.")

# %% [markdown]
# # 17. Optimasi keputusan: threshold / class weights / temperature
#
# - **Binary + metric label-based (f1, accuracy, f1_macro, qwk):** cari threshold terbaik di OOF (default argmax = 0.5).
# - **Multiclass + metric label-based:** cari pengali probabilitas per kelas (coordinate ascent) — membantu kelas
#   minoritas yang jarang diprediksi.
# - **logloss:** temperature scaling (`P^(1/T)` dinormalisasi) — memperbaiki model yang terlalu yakin / kurang yakin.
# - **roc_auc:** tidak perlu (hanya urutan yang penting).
#
# Hasil optimasi hanya dipakai kalau naik ≥ `MIN_DECISION_GAIN` (mencegah overfit ke OOF).
#
# **Cara membaca:** skor sebelum → sesudah. **Jika** threshold optimal jauh dari 0.5 (mis. 0.2) → model under-predict
# kelas positif (wajar untuk data imbalanced).
#
# **Untuk laporan:** sebutkan threshold yang dipakai dan alasannya (dioptimasi di OOF, bukan di test).

# %%
DECISION = None
TEMPERATURE = 1.0


def apply_temperature(P, T_):
    Q = np.power(np.clip(P, 1e-9, 1), 1.0 / T_)
    return Q / Q.sum(1, keepdims=True)


if FINAL is not None and CFG.OPTIMIZE_DECISION:
    mask = ~np.isnan(FINAL["oof"][:, 0])
    P_oof, y_oof = FINAL["oof"][mask], Y[mask]
    base = primary_score(y_oof, P_oof)
    label_metrics = ("accuracy", "f1", "f1_macro", "qwk")
    if CFG.METRIC in label_metrics and N_CLASSES == 2:
        grid = np.linspace(0.05, 0.95, 91)
        scores = [primary_score(y_oof, P_oof, {"type": "threshold", "value": t}) for t in grid]
        bi = int(np.argmax(scores))
        print(f"Threshold terbaik: {grid[bi]:.2f} → {CFG.METRIC} {scores[bi]:.4f} (argmax/0.5: {base:.4f})")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(grid, scores)
        ax.axvline(grid[bi], color="red", linestyle="--", label=f"best={grid[bi]:.2f}")
        ax.set_title(f"Threshold vs OOF {CFG.METRIC}")
        ax.set_xlabel("Threshold on P(positive)")
        ax.set_ylabel(CFG.METRIC)
        ax.legend()
        save_fig("threshold_curve", fig)
        if scores[bi] - base >= CFG.MIN_DECISION_GAIN:
            DECISION = {"type": "threshold", "value": float(grid[bi])}
    elif CFG.METRIC in label_metrics:
        w = np.ones(N_CLASSES)
        best = base
        for _ in range(2):
            for k in range(N_CLASSES):
                for v in (0.5, 0.67, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0):
                    trial = w.copy()
                    trial[k] = w[k] * v
                    s = primary_score(y_oof, P_oof, {"type": "class_weights", "value": trial})
                    if s > best + 1e-6:
                        best, w = s, trial
        print(f"Class weights terbaik: {np.round(w, 3).tolist()} → {CFG.METRIC} {best:.4f} (argmax: {base:.4f})")
        if best - base >= CFG.MIN_DECISION_GAIN:
            DECISION = {"type": "class_weights", "value": w.tolist()}
    elif CFG.METRIC == "logloss":
        grid = np.linspace(0.5, 3.0, 26)
        lls = [log_loss(y_oof, apply_temperature(P_oof, t_), labels=list(range(N_CLASSES))) for t_ in grid]
        bi = int(np.argmin(lls))
        print(f"Temperature terbaik: {grid[bi]:.2f} → logloss {lls[bi]:.4f} (T=1: {base:.4f})")
        if base - lls[bi] >= CFG.MIN_DECISION_GAIN:
            TEMPERATURE = float(grid[bi])
            FINAL["oof"] = np.where(np.isnan(FINAL["oof"]), np.nan, apply_temperature(np.nan_to_num(FINAL["oof"], nan=1.0 / N_CLASSES), TEMPERATURE))
            if FINAL["test"] is not None:
                FINAL["test"] = apply_temperature(FINAL["test"], TEMPERATURE)
    else:
        print("Metric berbasis ranking/probabilitas → tidak ada optimasi keputusan.")
    print("Keputusan final:", DECISION if DECISION else "argmax biasa", "| temperature:", TEMPERATURE)

# %% [markdown]
# # 18. Evaluasi model final
#
# ## 18a. Metric lengkap, confusion matrix, per-class report
#
# **Cara membaca confusion matrix:** baris = label asli, kolom = prediksi. Diagonal = benar. Versi ternormalisasi
# (per baris) = recall per kelas. Sel off-diagonal besar = pasangan kelas yang sering tertukar.
#
# **Jika X maka Y:**
#
# - Dua kelas sering tertukar → lihat contoh gambarnya (section 18b); mungkin butuh resolusi lebih tinggi atau
#   label memang ambigu.
# - Recall kelas minoritas rendah → `CLASS_WEIGHT="balanced"`, weighted sampler, atau optimasi class weights (section 17).
# - Precision rendah tapi recall tinggi untuk satu kelas → model terlalu sering menebak kelas itu.
#
# **Untuk laporan:** confusion matrix ternormalisasi + tabel per-class precision/recall/F1 adalah figure wajib.

# %%
EVAL = {}
if FINAL is not None:
    mask = ~np.isnan(FINAL["oof"][:, 0])
    P_oof, y_oof = FINAL["oof"][mask], Y[mask]
    pred_oof = decide(P_oof, DECISION)
    metrics_final = compute_metrics(y_oof, P_oof, DECISION)
    mt_df = pd.DataFrame([{"model": FINAL["name"], **{k: round(v, 4) for k, v in metrics_final.items()}}])
    display(mt_df)
    save_table(mt_df, "final_oof_metrics")

    rep = classification_report(y_oof, pred_oof, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T.round(4)
    display(rep_df)
    save_table(rep_df.reset_index().rename(columns={"index": "class"}), "classification_report")

    cm = confusion_matrix(y_oof, pred_oof, labels=list(range(N_CLASSES)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(5, min(0.6 * N_CLASSES + 3, 16))
    fig, axes = plt.subplots(1, 2, figsize=(2 * size, size * 0.85))
    for ax, mat, title, fmt in ((axes[0], cm, "Confusion Matrix (counts)", "d"),
                                (axes[1], cmn, "Confusion Matrix (row-normalized = recall)", ".2f")):
        if HAS_SNS:
            sns.heatmap(mat, annot=N_CLASSES <= 20, fmt=fmt, cmap="Blues", ax=ax,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
        else:
            ax.imshow(mat, cmap="Blues")
            if N_CLASSES <= 20:
                for i in range(N_CLASSES):
                    for j in range(N_CLASSES):
                        ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=8)
            ax.set_xticks(range(N_CLASSES))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
            ax.set_yticks(range(N_CLASSES))
            ax.set_yticklabels(CLASS_NAMES)
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    plt.tight_layout()
    save_fig("confusion_matrix", fig)

    pr, rc, f1s, sup = precision_recall_fscore_support(y_oof, pred_oof, labels=list(range(N_CLASSES)), zero_division=0)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * N_CLASSES + 3), 4))
    xs = np.arange(N_CLASSES)
    ax.bar(xs - 0.25, pr, 0.25, label="precision")
    ax.bar(xs, rc, 0.25, label="recall")
    ax.bar(xs + 0.25, f1s, 0.25, label="F1")
    ax.set_xticks(xs)
    ax.set_xticklabels(CLASS_NAMES, rotation=45 if N_CLASSES > 6 else 0, ha="right" if N_CLASSES > 6 else "center")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class Precision / Recall / F1 (OOF)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.legend()
    save_fig("per_class_metrics", fig)
    worst = int(np.argmin(f1s))
    print(f"Kelas terlemah: '{CLASS_NAMES[worst]}' (F1 {f1s[worst]:.3f}, support {sup[worst]})")
    EVAL.update(mask=mask, P=P_oof, y=y_oof, pred=pred_oof)
else:
    print("Tidak ada prediksi final untuk dievaluasi.")

# %% [markdown]
# ## 18b. Gambar yang salah diprediksi & confidence histogram
#
# **Cara membaca grid:** judul `T: label asli | P: prediksi (confidence)`. Diurutkan dari salah yang **paling yakin**.
#
# - Model sangat yakin tapi salah + gambar terlihat seperti label prediksi → kemungkinan **label noise** di data.
# - Gambar blur / gelap / objek kecil → butuh augmentasi atau resolusi lebih tinggi.
#
# **Confidence histogram:** prediksi benar seharusnya menumpuk di confidence tinggi, prediksi salah di confidence rendah.
# Kalau prediksi salah juga banyak di confidence > 0.9 → model overconfident (label smoothing / temperature membantu).
#
# **Untuk laporan:** 1 figure error analysis (gambar salah) sangat bernilai di bagian Diskusi.

# %%
if EVAL:
    idx_valid = np.where(EVAL["mask"])[0]
    conf = EVAL["P"].max(1)
    wrong = np.where(EVAL["pred"] != EVAL["y"])[0]
    print(f"Salah prediksi: {len(wrong)} dari {len(EVAL['y'])} ({len(wrong) / max(len(EVAL['y']), 1) * 100:.1f}%)")
    if len(wrong):
        order = wrong[np.argsort(-conf[wrong])][:CFG.N_MISCLASSIFIED]
        refs = train_df["ref"].values[idx_valid[order]]
        titles = [f"T: {CLASS_NAMES[EVAL['y'][i]]} | P: {CLASS_NAMES[EVAL['pred'][i]]} ({conf[i]:.2f})" for i in order]
        show_grid(list(refs), titles, 4, "misclassified_examples",
                  "Most Confident Misclassifications (OOF)", cell=2.6, title_colors=["red"] * len(refs))
        mis_df = pd.DataFrame({"ref": train_df["ref"].values[idx_valid[wrong]],
                               "true": [CLASS_NAMES[i] for i in EVAL["y"][wrong]],
                               "pred": [CLASS_NAMES[i] for i in EVAL["pred"][wrong]],
                               "confidence": conf[wrong].round(4)}).sort_values("confidence", ascending=False)
        save_table(mis_df, "misclassified_oof")

    fig, ax = plt.subplots(figsize=(8, 4))
    ok = EVAL["pred"] == EVAL["y"]
    ax.hist(conf[ok], bins=20, range=(0, 1), alpha=0.6, label=f"correct (n={ok.sum()})", color="#55A868")
    ax.hist(conf[~ok], bins=20, range=(0, 1), alpha=0.6, label=f"wrong (n={(~ok).sum()})", color="#C44E52")
    ax.set_title("Prediction Confidence (max probability, OOF)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.legend()
    save_fig("confidence_histogram", fig)
    if (~ok).sum() and np.mean(conf[~ok] > 0.9) > 0.3:
        print("⚠️ Banyak prediksi salah dengan confidence > 0.9 → overconfident / label noise.")

# %% [markdown]
# ## 18c. Grad-CAM (interpretability)
#
# **Apa itu:** peta panas yang menunjukkan bagian gambar mana yang paling memengaruhi prediksi. Diimplementasi manual
# dengan *forward hook* (simpan aktivasi) dan *tensor hook* (simpan gradien) di **layer Conv2d terakhir**:
# `bobot channel = rata-rata gradien`, `CAM = ReLU(Σ bobot × aktivasi)`, lalu di-upsample ke ukuran gambar.
#
# Hanya untuk backbone **CNN** (ResNet, EfficientNet, ConvNeXt, MobileNet, SimpleCNN). Untuk ViT/EVA/Swin di-skip
# (butuh metode berbeda, mis. attention rollout).
#
# **Cara membaca:** area merah/kuning = area yang "dilihat" model. **Bagus** kalau menyorot objek. **Curiga** kalau
# menyorot background, watermark, tepi gambar, atau teks → model belajar shortcut (data leak / bias).
#
# **Untuk laporan:** figure Grad-CAM membuat bagian interpretasi jauh lebih kuat ("model fokus pada area lesi").

# %%
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self.h = layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(self._bwd)

    def _bwd(self, g):
        self.grads = g

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = self.model(x)
            idx = out.argmax(1) if class_idx is None else torch.as_tensor(class_idx, device=x.device)
            out.gather(1, idx.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.flatten(1).min(1)[0][:, None, None]
        cam = cam / (cam.flatten(1).max(1)[0][:, None, None] + 1e-8)
        return cam.detach().cpu().numpy(), idx.detach().cpu().numpy()

    def remove(self):
        self.h.remove()


def last_conv_layer(model):
    layer = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            layer = m
    return layer


if HAS_TORCH and CFG.RUN_GRADCAM and DEEP_RESULTS:
    try:
        cnn_names = [n for n in DEEP_RESULTS if not any(k in n for k in TRANSFORMER_KEYS)]
        if not cnn_names:
            print("Semua deep model berbasis transformer → Grad-CAM di-skip.")
        else:
            gname = sorted(cnn_names, key=lambda n: CANDIDATES[n]["cv"], reverse=GREATER_IS_BETTER)[0]
            spec = [s for s in RESOLVED_MODELS if s[1] == gname][0]
            f0 = FOLDS[0]
            model = build_model(spec, N_CLASSES, pretrained=False, verbose=False)
            model.load_state_dict(torch.load(ckpt_path(spec, f0), map_location="cpu"))
            model = model.to(DEVICE).eval()
            meta = DEEP_RESULTS[gname]["meta"]
            layer = last_conv_layer(model)
            if layer is None:
                print("Tidak ada Conv2d di model → Grad-CAM di-skip")
            else:
                va_idx = np.where(train_df["fold"].values == f0)[0]
                pv = DEEP_RESULTS[gname]["oof"][va_idx]
                correct = va_idx[pv.argmax(1) == Y[va_idx]]
                wrong_ = va_idx[pv.argmax(1) != Y[va_idx]]
                n_w = min(len(wrong_), CFG.N_GRADCAM // 2)
                pick = list(wrong_[:n_w]) + list(correct[:CFG.N_GRADCAM - n_w])
                tf = build_eval_transform(CFG.IMG_SIZE, meta["mean"], meta["std"])
                xb = torch.stack([tf(get_image(train_df["ref"].values[i])) for i in pick]).to(DEVICE)
                cam_obj = GradCAM(model, layer)
                cams, preds = cam_obj(xb)
                cam_obj.remove()
                n = len(pick)
                fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8), squeeze=False)
                for j, i in enumerate(pick):
                    img = np.asarray(safe_load_image(train_df["ref"].values[i]).resize((CFG.IMG_SIZE, CFG.IMG_SIZE)))
                    axes[0, j].imshow(img)
                    axes[0, j].set_title(f"T: {CLASS_NAMES[Y[i]]}", fontsize=7)
                    axes[1, j].imshow(img)
                    axes[1, j].imshow(cams[j], cmap="jet", alpha=0.45)
                    col = "green" if preds[j] == Y[i] else "red"
                    axes[1, j].set_title(f"P: {CLASS_NAMES[preds[j]]}", fontsize=7, color=col)
                for ax in axes.ravel():
                    ax.axis("off")
                fig.suptitle(f"Grad-CAM — {gname} (fold {f0}); top: image, bottom: heatmap (red title = wrong)")
                plt.tight_layout()
                save_fig("gradcam", fig)
            del model
    except Exception as e:  # noqa
        print("⚠️ Grad-CAM gagal:", type(e).__name__, str(e)[:200])
else:
    print("Grad-CAM di-skip (butuh torch + deep model yang berhasil).")

# %% [markdown]
# # 19. Submission + validator
#
# **Logika:**
#
# 1. Kalau `sample_submission` ada → kolom & urutan id **persis mengikuti sample**. Id dicocokkan dengan test (persis,
#    lalu tanpa ekstensi / tanpa folder: `img_1.jpg` ≈ `img_1`).
# 2. `SUBMISSION_MODE="label"` → label asli (string/angka sesuai train). `"proba"` → binary: 1 kolom P(kelas positif);
#    multiclass: 1 kolom per kelas (nama kolom diambil dari sample kalau cocok dengan nama kelas).
# 3. Validator mengecek: jumlah baris, nama kolom, urutan id, tidak ada NaN, label valid.
#
# **Jika** validator gagal di "id tidak cocok" → lihat contoh id di sample vs test yang dicetak; biasanya beda ekstensi
# atau prefix folder. **Jika** test tidak ada → section ini di-skip.

# %%
def norm_key(x):
    return os.path.splitext(os.path.basename(str(x).replace("\\", "/")))[0]


def cast_like(values, ref_series):
    """Samakan dtype label dengan contoh di sample_submission (mis. int)."""
    try:
        if ref_series is not None and pd.api.types.is_integer_dtype(ref_series):
            return pd.Series(values).astype(float).astype(int).values
    except Exception:  # noqa
        pass
    return values


def build_submission(P_test, decision):
    labels_idx = decide(P_test, decision)
    labels = np.array([CLASSES[i] for i in labels_idx], dtype=object)
    test_ids = test_df["id"].values
    if sample_sub is not None:
        sub = sample_sub.copy()
        id_col = CFG.ID_COL if (CFG.ID_COL and CFG.ID_COL in sub.columns) else sub.columns[0]
        pos = {str(v): i for i, v in enumerate(test_ids)}
        pos_norm = {norm_key(v): i for i, v in enumerate(test_ids)}
        order = []
        for v in sub[id_col].values:
            i = pos.get(str(v), pos_norm.get(norm_key(v)))
            order.append(-1 if i is None else i)
        order = np.array(order)
        n_miss = int((order < 0).sum())
        if n_miss:
            if n_miss == len(order) and len(sub) == len(test_df):
                print("⚠️ Id sample_submission tidak cocok sama sekali dengan id test → diasumsikan URUTAN SAMA.")
                order = np.arange(len(sub))
            else:
                print(f"⚠️ {n_miss} id di sample tidak ditemukan di test → diisi kelas mayoritas / prior. "
                      f"Contoh sample id: {sub[id_col].head(3).tolist()} | contoh test id: {list(test_ids[:3])}")
        prior = np.bincount(Y, minlength=N_CLASSES) / len(Y)
        P_sub = np.array([P_test[i] if i >= 0 else prior for i in order])
        lab_sub = np.array([labels[i] if i >= 0 else CLASSES[int(prior.argmax())] for i in order], dtype=object)
        other_cols = [c for c in sub.columns if c != id_col]
        if CFG.SUBMISSION_MODE == "proba":
            name_map = {str(c): c for c in other_cols}
            if all(cn in name_map for cn in CLASS_NAMES):
                for k, cn in enumerate(CLASS_NAMES):
                    sub[name_map[cn]] = P_sub[:, k]
            elif N_CLASSES == 2 and len(other_cols) == 1:
                sub[other_cols[0]] = P_sub[:, 1]
            elif len(other_cols) == N_CLASSES:
                for k, c in enumerate(other_cols):
                    sub[c] = P_sub[:, k]
            else:
                print("⚠️ Kolom sample_submission tidak cocok untuk mode proba "
                      f"({other_cols} vs {N_CLASSES} kelas) → dibuat 1 kolom per kelas. Cek format lomba / "
                      "pakai SUBMISSION_MODE='label'.")
                sub = sub[[id_col]].copy()
                for k, cn in enumerate(CLASS_NAMES):
                    sub[cn] = P_sub[:, k]
        else:
            tcol = CFG.SUB_TARGET_COL or (other_cols[0] if other_cols else (DATA_INFO.get("target_col") or "label"))
            sub[tcol] = cast_like(lab_sub, sample_sub[tcol] if tcol in sample_sub.columns else None)
        return sub
    tcol = CFG.SUB_TARGET_COL or DATA_INFO.get("target_col") or "label"
    id_col = CFG.ID_COL or "id"
    if CFG.SUBMISSION_MODE == "proba":
        sub = pd.DataFrame({id_col: test_ids})
        if N_CLASSES == 2:
            sub[tcol] = P_test[:, 1]
        else:
            for k, cn in enumerate(CLASS_NAMES):
                sub[cn] = P_test[:, k]
        return sub
    return pd.DataFrame({id_col: test_ids, tcol: labels})


def validate_submission(sub, sample):
    ok = True
    print("Validasi submission:")
    if sub.isna().any().any():
        print("  ❌ Ada NaN di submission")
        ok = False
    if sample is not None:
        checks = [("jumlah baris sama", len(sub) == len(sample)),
                  ("kolom sama & urutan sama", list(sub.columns) == list(sample.columns)),
                  ("urutan id sama", len(sub) == len(sample) and
                   (sub.iloc[:, 0].astype(str).values == sample.iloc[:, 0].astype(str).values).all())]
        for name, c in checks:
            print(f"  {'✅' if c else '❌'} {name}")
            ok &= bool(c)
    else:
        print(f"  ✅ {len(sub)} baris (tanpa sample_submission untuk dibandingkan)")
    if CFG.SUBMISSION_MODE == "label":
        tcol = sub.columns[-1]
        valid = set(map(str, CLASSES))
        bad = ~sub[tcol].astype(str).isin(valid)
        print(f"  {'✅' if not bad.any() else '❌'} semua label termasuk kelas train ({bad.sum()} tidak valid)")
        ok &= not bad.any()
        print("  Distribusi prediksi:", sub[tcol].value_counts().head(10).to_dict())
    print("  HASIL:", "✅ VALID" if ok else "❌ PERBAIKI DULU")
    return ok


SUBMISSION = None
if HAS_TEST and FINAL is not None and FINAL["test"] is not None:
    try:
        SUBMISSION = build_submission(FINAL["test"], DECISION)
    except Exception as e:  # noqa
        print("❌ Gagal membuat submission:", type(e).__name__, e)
        print("   Fallback: format [id, label] sederhana. Cek ID_COL / SUB_TARGET_COL / SUBMISSION_MODE.")
        SUBMISSION = pd.DataFrame({"id": test_df["id"].values,
                                   "label": [CLASSES[i] for i in decide(FINAL["test"], DECISION)]})
if SUBMISSION is not None:
    display(SUBMISSION.head())
    validate_submission(SUBMISSION, sample_sub)
    stamp = time.strftime("%Y%m%d_%H%M")
    sub_path = os.path.join(SUB_DIR, f"submission_{safe_name(FINAL['name'])}_{CFG.METRIC}{FINAL['cv']:.4f}_{stamp}.csv")
    SUBMISSION.to_csv(sub_path, index=False)
    SUBMISSION.to_csv(os.path.join(SUB_DIR, "submission.csv"), index=False)
    print("Submission disimpan:", sub_path, "dan submission.csv")
    # cek distribusi prediksi test vs train
    pt = np.bincount(decide(FINAL["test"], DECISION), minlength=N_CLASSES) / len(test_df)
    ptr = np.bincount(Y, minlength=N_CLASSES) / len(Y)
    dist = pd.DataFrame({"class": CLASS_NAMES, "train_pct": (ptr * 100).round(1), "test_pred_pct": (pt * 100).round(1)})
    display(dist)
    if np.abs(pt - ptr).max() > 0.2:
        print("⚠️ Distribusi prediksi test sangat berbeda dari train → cek domain shift / bug urutan id.")
else:
    print("Info: Test tidak tersedia atau tidak ada prediksi → submission di-skip.")

# %% [markdown]
# # 20. Simpan artefak: OOF, test prediction, config, experiment log
#
# Semua yang dibutuhkan untuk reproduksi & laporan:
#
# - `tables/oof_<model>.csv` dan `tables/test_pred_<model>.csv` (probabilitas per kelas) → untuk ensemble ulang tanpa training.
# - `models/*.pt` → bobot terbaik per fold (load dengan `build_model(spec, N_CLASSES, pretrained=False)` + `load_state_dict`).
# - `tables/experiment_log.csv` → setiap run menambah 1 baris (model, setting, skor). Bandingkan eksperimen di sini.
# - `config_used.json` → snapshot CFG.

# %%
prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
for name, c in CANDIDATES.items():
    o = pd.DataFrame(c["oof"], columns=prob_cols)
    o.insert(0, "label", train_df["label"].values)
    o.insert(0, "fold", train_df["fold"].values)
    o.insert(0, "ref", train_df["ref"].values)
    o.to_csv(os.path.join(TAB_DIR, f"oof_{safe_name(name)}.csv"), index=False)
    if HAS_TEST and c["test"] is not None:
        t_ = pd.DataFrame(c["test"], columns=prob_cols)
        t_.insert(0, "id", test_df["id"].values)
        t_.to_csv(os.path.join(TAB_DIR, f"test_pred_{safe_name(name)}.csv"), index=False)
if FINAL is not None:
    np.save(os.path.join(TAB_DIR, "final_oof.npy"), FINAL["oof"])
    if FINAL["test"] is not None:
        np.save(os.path.join(TAB_DIR, "final_test.npy"), FINAL["test"])

cfg_dump = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in vars(CFG).items() if not k.startswith("_")}
cfg_dump.update(backend=BACKEND, device=str(DEVICE), n_train=len(train_df), n_test=len(test_df) if HAS_TEST else 0,
                classes=CLASS_NAMES, decision=str(DECISION), temperature=TEMPERATURE,
                final=FINAL["name"] if FINAL else None)
with open(os.path.join(OUT_DIR, "config_used.json"), "w", encoding="utf-8") as f:
    json.dump(cfg_dump, f, indent=2, default=str)

log_path = os.path.join(TAB_DIR, "experiment_log.csv")
log_rows = []
for name, c in CANDIDATES.items():
    log_rows.append({"time": time.strftime("%Y-%m-%d %H:%M"), "run_mode": CFG.RUN_MODE, "model": name,
                     "backend": c.get("backend", ""), "img_size": CFG.IMG_SIZE, "epochs": CFG.EPOCHS,
                     "folds": str(FOLDS), "aug": CFG.AUG_LEVEL, "lr": CFG.LR, "batch": CFG.BATCH_SIZE,
                     "metric": CFG.METRIC, "oof_score": round(c["cv"], 5),
                     "fold_std": round(float(np.std(c.get("fold_scores", [np.nan]))), 5),
                     "final": FINAL is not None and (FINAL["name"] == name or name in FINAL["weights"])})
if log_rows:
    log_df = pd.DataFrame(log_rows)
    if os.path.isfile(log_path):
        log_df = pd.concat([pd.read_csv(log_path), log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    display(log_df.tail(10))
print("Artefak tersimpan di:", os.path.abspath(OUT_DIR))
print("Figures:", sorted(os.listdir(FIG_DIR))[:40])

# %% [markdown]
# # 21. Ringkasan untuk laporan
#
# Cell di bawah mencetak ringkasan yang bisa langsung disalin ke laporan (lalu diparafrasekan).
#
# **Struktur laporan yang disarankan (maks 10 halaman):**
#
# 1. **Pendahuluan** — masalah & tujuan (klasifikasi K kelas, metric lomba).
# 2. **Deskripsi data & EDA** — jumlah gambar, distribusi kelas (`class_distribution`), contoh gambar (`sample_grid`),
#    ukuran & mode gambar, data cleaning (file korup, duplikat).
# 3. **Metodologi** — preprocessing (resize, normalisasi), augmentasi (`augmentation_preview`), arsitektur
#    (backbone pretrained / SimpleCNN), skema validasi (Stratified K-Fold, duplicate-aware), hyperparameter, TTA, ensemble.
# 4. **Hasil** — tabel `cv_results`, `model_comparison`, learning curve, confusion matrix, per-class metrics.
# 5. **Analisis & interpretasi** — Grad-CAM, contoh salah prediksi, kelas yang sulit & alasannya.
# 6. **Kesimpulan & saran** — apa yang paling membantu, keterbatasan (data kecil, imbalance, domain shift).

# %%
print("=" * 70)
print("RINGKASAN")
print("=" * 70)
print(f"Data        : {len(train_df)} gambar train, {len(test_df) if HAS_TEST else 0} test, {N_CLASSES} kelas "
      f"(layout {DATA_INFO['layout']}, imbalance ratio {IMB_RATIO:.1f})")
print(f"Validasi    : {CFG.CV_STRATEGY}, {len(FOLDS)} fold dijalankan, metric {CFG.METRIC}")
print(f"Backend     : {BACKEND} | device {DEVICE} | IMG_SIZE {CFG.IMG_SIZE} | epochs {CFG.EPOCHS}")
if len(CV_TABLE):
    for _, r in CV_TABLE.iterrows():
        print(f"  - {r['model']:28s} OOF {CFG.METRIC} = {r[f'oof_{CFG.METRIC}']:.4f} (fold std {r['fold_std']})")
if FINAL is not None:
    print(f"Final       : {FINAL['name']} | bobot {FINAL['weights']} | OOF {CFG.METRIC} {FINAL['cv']:.4f}")
    if EVAL:
        print("Metric lengkap (OOF):", {k: round(v, 4) for k, v in compute_metrics(EVAL['y'], EVAL['P'], DECISION).items()})
print("Gagal       :", FAILED_MODELS if FAILED_MODELS else "-")
print("Submission  :", "dibuat" if SUBMISSION is not None else "tidak dibuat")

# %% [markdown]
# # 22. Debugging: sanity checks ringkas
#
# Cell ini mengulang pengecekan penting dalam 1 tempat. Jalankan kapan saja setelah ada yang aneh.

# %%
print("Sanity checks")
print("-" * 60)
print("train_df shape          :", train_df.shape, "| kolom:", list(train_df.columns))
print("label NaN               :", int(train_df["label"].isna().sum()))
print("ref None                :", int(train_df["ref"].isna().sum()))
print("kelas train             :", N_CLASSES, "| min/max per kelas:", int(class_counts.min()), "/", int(class_counts.max()))
print("fold assignment         :", train_df["fold"].value_counts().sort_index().to_dict())
if HAS_TEST:
    print("test_df shape           :", test_df.shape, "| id unik:", test_df["id"].nunique() == len(test_df))
    overlap_ids = set(map(str, test_df["id"])) & set(map(str, train_df["src_id"]))
    print("id test yang juga ada di train (src_id):", len(overlap_ids), "⚠️ cek leakage" if overlap_ids else "✅")
if FINAL is not None:
    o = FINAL["oof"][~np.isnan(FINAL["oof"][:, 0])]
    print("OOF probs valid (0..1, sum=1):", bool(np.all(o >= -1e-6) and np.allclose(o.sum(1), 1, atol=1e-3)))
    if FINAL["test"] is not None:
        print("Test probs finite        :", bool(np.isfinite(FINAL["test"]).all()), "| shape:", FINAL["test"].shape)
print("Load errors              :", dict(LOAD_ERRORS) if LOAD_ERRORS else "tidak ada")
print("FAILED_MODELS            :", list(FAILED_MODELS) if FAILED_MODELS else "tidak ada")
if CANDIDATES:
    best = max(c["cv"] for c in CANDIDATES.values()) if GREATER_IS_BETTER else min(c["cv"] for c in CANDIDATES.values())
    if GREATER_IS_BETTER and CFG.METRIC in ("accuracy", "f1_macro", "f1", "roc_auc") and best > 0.995:
        print("⚠️ Skor CV hampir sempurna → cek leakage (duplikat train/valid, nama file mengandung label, dsb).")

# %% [markdown]
# # 23. Troubleshooting
#
# | Error / gejala | Penyebab | Solusi |
# |---|---|---|
# | `FileNotFoundError: Data train tidak ditemukan` | path salah / working directory beda | `print(os.getcwd())`; set `CFG.TRAIN_CSV`, `CFG.TRAIN_IMAGE_DIR` dengan path absolut |
# | Banyak `gambar train TIDAK ditemukan` | nama file di CSV tanpa ekstensi / beda folder / beda huruf besar-kecil | cek 1 nama file manual; set `CFG.TRAIN_IMAGE_DIR` ke folder yang benar; tambahkan folder ke `IMAGE_DIR_CANDIDATES` |
# | `CUDA out of memory` | batch / IMG_SIZE / model terlalu besar | otomatis retry dengan batch kecil; kalau tetap gagal: `IMG_SIZE` 224→160, model lebih kecil, `USE_AMP=True`, restart kernel |
# | `AttributeError: Can't get attribute 'ImageDataset' on <module '__main__'>` / notebook hang saat training | `NUM_WORKERS > 0` di Windows (spawn tidak bisa import class dari notebook) | `CFG.NUM_WORKERS = 0` (default Windows) + `CACHE_IMAGES=True` |
# | `BrokenPipeError` / `DataLoader worker exited unexpectedly` | worker crash (RAM habis / Windows) | `NUM_WORKERS=0`, kecilkan batch, matikan cache kalau RAM kecil |
# | `OSError: image file is truncated` | JPEG terpotong | `ImageFile.LOAD_TRUNCATED_IMAGES = True` (sudah default via `CFG.LOAD_TRUNCATED_IMAGES`) |
# | `PIL.UnidentifiedImageError: cannot identify image file` | file korup / bukan gambar / format tidak didukung (HEIC, SVG) | sudah dibuang otomatis di EDA; konversi HEIC ke JPG kalau banyak |
# | `DecompressionBombError` | gambar sangat besar (> 89M pixel) | `Image.MAX_IMAGE_PIXELS = None` atau angka besar (sudah 300M) |
# | `RuntimeError: Unknown model (xxx)` (timm) | salah ejaan nama model | `timm.list_models("*convnext*")`, `timm.list_models(pretrained=True)`; notebook mencetak nama mirip |
# | `pretrained gagal (... ConnectionError / HTTPError / SSLError)` | offline / firewall / proxy kampus | sambungkan internet lalu jalankan ulang (weights di-cache di `~/.cache/huggingface` atau `~/.cache/torch`); atau `PRETRAINED=False` + epoch lebih banyak |
# | `size mismatch for fc.weight / classifier.weight / head.weight` | load checkpoint dengan jumlah kelas berbeda / model berbeda | bangun model dengan `num_classes` sama seperti saat training; hapus checkpoint lama di `models/` atau `SKIP_TRAIN_IF_CKPT=False` |
# | `Expected more than 1 value per channel when training` | batch berisi 1 sampel + BatchNorm | sudah ditangani `drop_last=True`; kalau tetap, naikkan batch / kurangi `MIN_BATCH_SIZE` jangan < 2 |
# | `loss = nan` / `FloatingPointError: Loss NaN/inf` | LR terlalu besar, AMP overflow, gambar rusak | `USE_AMP=False`, `LR` /10, `GRAD_CLIP=1.0`, cek file korup |
# | `RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.FloatTensor)` | model tidak dipindah ke GPU / campur device | pastikan `model.to(DEVICE)`; jangan ubah DEVICE di tengah notebook; restart kernel |
# | `Device: cpu` padahal ada GPU NVIDIA | torch versi CPU terinstall | install ulang torch versi CUDA dari pytorch.org (sesuai versi driver), jangan lewat `pip install torch` biasa |
# | `ImportError` / `RuntimeError: operator torchvision::nms does not exist` | versi torchvision tidak cocok dengan torch | install torchvision sesuai torch (2.X ↔ 0.(X+15)) atau `CFG.USE_TORCHVISION=False` (fallback tetap jalan) |
# | `ValueError: n_splits=5 cannot be greater than the number of members in each class` | kelas sangat kecil | sudah otomatis menurunkan fold; atau gabungkan kelas langka / `CV_STRATEGY="holdout"` |
# | `ValueError: Tidak bisa menebak bentuk gambar dari N pixel` | layout pixels dengan bentuk tidak persegi | set `CFG.PIXEL_SHAPE = (H, W)` atau `(H, W, 3)`; cek `PIXEL_PREFIX` kalau ada kolom numerik non-pixel |
# | ViT error `Input image size (96*96) doesn't match model (224*224)` | ViT butuh ukuran tetap | timm: notebook mengirim `img_size=IMG_SIZE`; pakai kelipatan patch (16/14); torchvision ViT: pakai `IMG_SIZE=224` |
# | Submission: `id di sample tidak ditemukan di test` | id beda format (ekstensi, folder, angka vs string) | lihat contoh id yang dicetak; set `CFG.ID_COL`; pastikan `TEST_CSV`/`TEST_IMAGE_DIR` benar |
# | Skor CV sangat tinggi tapi leaderboard jelek | leakage (duplikat lintas fold, grup pasien bocor) atau domain shift | `DUP_AWARE_CV=True`, `GROUP_COL`, cek brightness/size shift train-vs-test, cek Grad-CAM menyorot background |
# | Training sangat lambat di CPU | model/IMG_SIZE besar | recipe CPU-only: `IMG_SIZE=128`, `mobilenetv3_large_100`, `EPOCHS=5`, atau pakai baseline frozen features saja |
# | `KeyError` kolom label / gambar | auto-detect salah | set `CFG.TARGET_COL`, `CFG.IMAGE_COL`, `CFG.ID_COL` secara eksplisit |
# | Memori RAM penuh | cache gambar terlalu besar | `CACHE_IMAGES=False` atau `CACHE_MAX_GB` lebih kecil |

# %% [markdown]
# # 24. Experiment Recipes
#
# Salin ke cell `USER OVERRIDE`.
#
# **A. CPU-only quick (laptop tanpa GPU, ~10–30 menit)**
#
# ```python
# CFG.MODEL_POOL = ["mobilenetv3_large_100"]
# CFG.IMG_SIZE = 128
# CFG.EPOCHS = 5
# CFG.BATCH_SIZE = 32
# CFG.N_FOLDS = 3                 # atau CFG.CV_STRATEGY = "holdout"
# CFG.TTA_SCALES = [1.0]
# CFG.RUN_FEATURE_BASELINE = True # frozen features + LogReg sering sudah kompetitif
# ```
#
# **B. GPU strong (GPU ≥ 8 GB, ~1–3 jam)**
#
# ```python
# CFG.MODEL_POOL = ["convnext_tiny", "efficientnet_b2", "eva02_small_patch14_224.mim_in22k"]
# CFG.IMG_SIZE = 288              # eva02 butuh kelipatan 14: pakai 224/280 kalau model itu dipakai
# CFG.EPOCHS = 15
# CFG.BATCH_SIZE = 32
# CFG.LR = 2e-4
# CFG.AUG_LEVEL = "medium"
# CFG.MIXUP_ALPHA = 0.2; CFG.CUTMIX_ALPHA = 1.0
# CFG.TTA_SCALES = [1.0, 1.15]
# CFG.N_FOLDS = 5
# ```
#
# **C. Dataset kecil (< 1500 gambar)**
#
# ```python
# CFG.MODEL_POOL = ["efficientnet_b0", "resnet50"]
# CFG.AUG_LEVEL = "strong"
# CFG.FREEZE_EPOCHS = 2           # linear probe dulu supaya head stabil
# CFG.EPOCHS = 20; CFG.PATIENCE = 5
# CFG.LR = 1e-4
# CFG.N_FOLDS = 5
# CFG.MIXUP_ALPHA = 0.2
# ```
#
# **D. Imbalanced (ratio > 5)**
#
# ```python
# CFG.METRIC = "f1_macro"         # kalau lomba memakai metric ini
# CFG.CLASS_WEIGHT = "balanced"   # ATAU CFG.USE_WEIGHTED_SAMPLER = True (jangan dua-duanya)
# CFG.OPTIMIZE_DECISION = True    # threshold / class weights di OOF
# CFG.LABEL_SMOOTHING = 0.05
# ```
#
# **E. Gambar kecil grayscale (MNIST-like, 28×28)**
#
# ```python
# CFG.DATA_LAYOUT = "pixels"
# CFG.MODEL_POOL = ["simple_cnn"] # atau resnet18 pretrained dengan IMG_SIZE=64
# CFG.IMG_SIZE = 32
# CFG.HFLIP = False               # angka/huruf tidak boleh di-flip
# CFG.ROTATION_DEG = 10
# CFG.EPOCHS = 15; CFG.BATCH_SIZE = 128
# CFG.NORM = "dataset"
# ```
#
# **F. Citra satelit / mikroskop / tampak atas**
#
# ```python
# CFG.VFLIP = True; CFG.ROTATION_DEG = 90
# ```
#
# **G. Waktu hampir habis**
#
# ```python
# CFG.FOLDS_TO_RUN = [0]          # 1 fold saja
# CFG.EPOCHS = 3
# CFG.SKIP_TRAIN_IF_CKPT = True   # pakai checkpoint yang sudah ada
# ```

# %% [markdown]
# # 25. Checklist sebelum submit
#
# - [ ] `RUN_MODE = "full"` (bukan fast) untuk submission final.
# - [ ] Path data benar; jumlah gambar train/test sesuai deskripsi lomba; tidak ada (atau sedikit & dijelaskan) file hilang.
# - [ ] `METRIC` sama dengan metric resmi lomba; `SUBMISSION_MODE` (label vs proba) sesuai format yang diminta.
# - [ ] Skor CV masuk akal (jauh di atas baseline kelas mayoritas, tidak "terlalu sempurna" → cek leakage).
# - [ ] Variansi antar fold wajar (std < 0.02–0.03).
# - [ ] Validator submission semua ✅ (jumlah baris, kolom, urutan id, tidak ada NaN, label valid).
# - [ ] Distribusi prediksi test mirip distribusi train (kecuali ada alasan).
# - [ ] Buka file `submission.csv` manual: header benar, tidak ada kolom index tambahan.
# - [ ] Figure laporan tersimpan di `figures/` (DPI 300 untuk final: `CFG.FIG_DPI = 300`).
# - [ ] `experiment_log.csv` dicatat → tulis di laporan model/setting mana yang dipakai dan kenapa.
# - [ ] Simpan notebook + `outputs/` (model, OOF, config) sebagai bukti reproducibility.
