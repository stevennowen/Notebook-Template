# -*- coding: utf-8 -*-
# =====================================================================================================
# cv_segmentation.py — versi SCRIPT dari cv_segmentation.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python cv_segmentation.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb cv_segmentation.py
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
# # Computer Vision Competition Template — Image Segmentation (Semantic)
#
# Template lengkap untuk lomba **segmentasi gambar**: setiap pixel diberi kelas (biner: objek vs background, atau
# multikelas: jalan/sawah/air/...). Contoh lomba: Gemastik XIX 3b (underwater scene), Hology 9 Final (sawah dari citra
# satelit), ARA 7 (pothole). Notebook ini bisa dijalankan **dari atas ke bawah** di laptop CPU maupun GPU.
#
# ---
#
# ## Pipeline besar
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json kalau dijalankan lewat tools/run_task.py)
# → Install / cek library (segmentation_models_pytorch opsional; fallback torchvision / U-Net buatan sendiri)
# → Deteksi layout data: folder images + masks  ATAU  train.csv berisi RLE (Kaggle style)
# → Baca mask: index kelas (0,1,2,...), biner 0/255, atau RGB berwarna (palette otomatis)
# → EDA: ukuran gambar, frekuensi pixel per kelas, % mask kosong, contoh overlay
# → K-fold (stratified: mask kosong vs ada objek / kelas dominan)
# → Augmentasi (flip, rotate90, crop-resize, brightness/contrast) — tanpa library tambahan
# → Model: smp U-Net/FPN/DeepLabV3+ (encoder pretrained) → torchvision DeepLabV3/FCN → U-Net kecil
# → Loss Dice + BCE/CE, AMP, AdamW + cosine, early stopping, OOM auto-retry
# → OOF: Dice / IoU per kelas → tuning threshold + min area (post-processing)
# → Inference: TTA flip, rata-rata fold, resize balik ke ukuran asli
# → Submission: RLE CSV (cocok sample_submission) atau folder mask PNG (+ ZIP) + validator
# → Ringkasan untuk laporan, troubleshooting, recipes, checklist
# ```
#
# ## Cara pakai cepat (5 langkah)
#
# 1. Taruh data di `./data/` (lihat layout di bawah).
# 2. Isi cell **`# >>> USER OVERRIDE`**: path folder gambar/mask, metric lomba, format submission.
# 3. `CFG.RUN_MODE = "fast"` → Run All. Cek: kelas terdeteksi benar? overlay mask pas dengan gambar?
# 4. `CFG.RUN_MODE = "full"` → Run All lagi untuk hasil final.
# 5. Ambil submission di `outputs/cv_segmentation/submissions/` dan figure di `figures/`.
#
# **Dua format:** file ini ada sebagai notebook (`cv_segmentation.ipynb`) dan script (`cv_segmentation.py`) dengan isi
# sama. Script: `python cv_segmentation.py` atau buka di VS Code dan jalankan per `# %%`. Headless:
# `python tools/run_task.py --task segmentation --format py --config task_config.json`.
#
# ## Layout data yang didukung (`CFG.DATA_LAYOUT = "auto"`)
#
# | Layout | Struktur | Catatan |
# |---|---|---|
# | `folder` | `data/train/images/*.png` + `data/train/masks/*.png` (nama file sama; mask boleh berakhiran `_mask`, `_label`, `_gt`) | mask = index kelas, 0/255, atau RGB berwarna |
# | `csv_rle` | `data/train.csv` berisi kolom id gambar + kolom RLE (+ kolom kelas kalau multikelas, 1 baris per kelas) | RLE kosong/NaN = tidak ada objek |
#
# Gambar test: `data/test/images/` (atau `TEST_IMAGE_DIR`). Format submission dideteksi dari `sample_submission.csv`:
# ada kolom berisi RLE → submission RLE; tidak ada sample_submission → folder mask PNG.

# %% [markdown]
# # 1. Config (`CFG`)
#
# Semua "kenop" ada di sini. Ubah lewat cell `USER OVERRIDE` di bawahnya (atau `task_config.json`).
#
# **Jika** CPU-only **maka** `IMG_SIZE=128–256`, `EPOCHS=8–15`, `ENCODER="resnet18"` / `ARCH="unet_small"`.
# **Jika** GPU 4–8 GB **maka** `IMG_SIZE=384–512`, `BATCH_SIZE=4–8`, `USE_AMP=True` (OOM → batch otomatis dikecilkan).
# **Jika** objek kecil (retak, pothole kecil) **maka** naikkan `IMG_SIZE` atau pakai `TILE_SIZE` (crop acak resolusi asli).

# %%
import os


class CFG:
    # ---------------- mode jalan
    RUN_MODE = "full"            # "fast" = debug cepat (subsample, 2 fold, epoch & gambar kecil) | "full" = final
    SEED = 42
    NOTEBOOK_NAME = "cv_segmentation"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"  # pip install library opsional (TIDAK pernah torch)
    INSTALL_SMP = True           # segmentation_models_pytorch (+timm) di-install dengan --no-deps (torch tidak diganti)

    # ---------------- path data
    DATA_LAYOUT = "auto"         # "auto" | "folder" | "csv_rle"
    TRAIN_IMAGE_DIR = "./data/train/images"
    TRAIN_MASK_DIR = "./data/train/masks"
    TEST_IMAGE_DIR = "./data/test/images"
    TRAIN_CSV = "./data/train.csv"            # layout csv_rle
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    IMAGE_DIR_CANDIDATES = ["./data/train/images", "./data/train_images", "./data/images", "./data/train", "./data/img"]
    MASK_DIR_CANDIDATES = ["./data/train/masks", "./data/train_masks", "./data/masks", "./data/labels", "./data/mask",
                           "./data/annotations", "./data/gt"]
    TEST_DIR_CANDIDATES = ["./data/test/images", "./data/test_images", "./data/test"]
    MASK_SUFFIXES = ["", "_mask", "_masks", "_label", "_labels", "_gt", "_seg", "-mask"]
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    OUTPUT_DIR = "./outputs/cv_segmentation/"

    # ---------------- mask & kelas
    MASK_MODE = "auto"           # "auto" | "index" (0..K-1) | "binary255" (0/255) | "rgb" (warna per kelas)
    N_CLASSES = None             # None = auto dari mask (termasuk background)
    CLASS_NAMES = None           # mis. ["background", "road", "water"]; None = class_0, class_1, ...
    IGNORE_INDEX = None          # nilai mask yang diabaikan saat training (mis. 255 di beberapa dataset)
    RGB_PALETTE = None           # layout rgb: {(r,g,b): idx}; None = dibangun otomatis dari warna yang muncul
    # csv_rle
    RLE_ID_COL = None            # None = auto (kolom pertama yang cocok nama gambar)
    RLE_COL = None               # None = auto (kolom berisi "1 3 10 5 ...")
    RLE_CLASS_COL = None         # multikelas: kolom kelas (1 baris per gambar per kelas)
    RLE_ORDER = "F"              # "F" = column-major (Kaggle umum) | "C" = row-major. Cek dengan figure overlay!
    RLE_ONE_INDEXED = True

    # ---------------- model
    ARCH = "auto"                # "auto" | "Unet" | "UnetPlusPlus" | "FPN" | "DeepLabV3Plus" (smp) | "deeplabv3" | "fcn" (torchvision) | "unet_small"
    ENCODER = "resnet34"         # encoder smp (resnet18/34/50, efficientnet-b0..b4, mit_b0..)
    PRETRAINED = True            # encoder ImageNet (butuh internet sekali; cache di ~/.cache)
    USE_GPU = True
    IMG_SIZE = 320               # sisi gambar saat training (kelipatan 32)
    TILE_SIZE = None             # mis. 512 → crop acak ukuran asli (objek kecil) — inference tetap full image (resize)
    BATCH_SIZE = 8
    EPOCHS = 25
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    LOSS = "dice_ce"             # "dice_ce" | "ce" | "dice" | "focal_dice"
    DICE_WEIGHT = 0.5
    USE_AMP = True
    PATIENCE = 6                 # early stopping (epoch tanpa perbaikan)
    N_FOLDS = 5
    FOLDS_TO_RUN = None          # mis. [0] untuk 1 fold saja (cepat)
    NUM_WORKERS = 0 if os.name == "nt" else 2
    MAX_MINUTES_PER_FOLD = None

    # ---------------- augmentasi
    AUG_HFLIP = True
    AUG_VFLIP = True             # matikan untuk foto yang orientasinya penting (langit selalu di atas)
    AUG_ROT90 = True             # citra satelit / mikroskop: True; foto jalan: False
    AUG_SCALE = (0.75, 1.0)      # random resized crop (luas relatif)
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST = 0.2

    # ---------------- post-processing & submission
    TTA = ["none", "hflip"]      # + "vflip"
    TUNE_THRESHOLD = True        # biner: cari threshold terbaik di OOF
    THRESHOLD = 0.5
    MIN_AREA = 0                 # buang komponen < MIN_AREA pixel (di-tune kalau TUNE_MIN_AREA)
    TUNE_MIN_AREA = True
    METRIC = "dice"              # "dice" | "iou" (mIoU) — sesuai aturan lomba
    METRIC_INCLUDE_BG = False    # multikelas: rata-rata tanpa background (umum di lomba)
    SUBMISSION_FORMAT = "auto"   # "auto" | "rle" | "mask_png"
    SUB_ID_COL = None            # None = kolom pertama sample_submission
    SUB_RLE_COL = None           # None = kolom ke-2 sample_submission
    EMPTY_RLE = ""               # isi untuk mask kosong ("" atau "-" atau "1 0" — cek aturan)

    # ---------------- fast mode
    FAST_SAMPLE = 120
    FAST_EPOCHS = 3
    FAST_FOLDS = 2
    FAST_IMG_SIZE = 128
    FIG_DPI = 150

# %% [markdown]
# # 2. USER OVERRIDE (isi untuk lomba spesifik)
#
# ```python
# CFG.TRAIN_IMAGE_DIR = "./data/train/images"
# CFG.TRAIN_MASK_DIR = "./data/train/masks"
# CFG.TEST_IMAGE_DIR = "./data/test/images"
# CFG.METRIC = "iou"
# CFG.RUN_MODE = "fast"
# ```

# %%
# >>> USER OVERRIDE
# CFG.RUN_MODE = "fast"
# CFG.CLASS_NAMES = ["background", "sawah"]
# CFG.IMG_SIZE = 384
# CFG.ARCH = "Unet"; CFG.ENCODER = "efficientnet-b3"
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
# - `torch` & `torchvision` **tidak pernah** di-install otomatis (harus cocok dengan CUDA) → https://pytorch.org/get-started/locally/
# - `segmentation_models_pytorch` (smp) + `timm` di-install dengan `--no-deps` supaya torch kamu tidak diganti.
# - Gagal install **tidak** menghentikan notebook: model jatuh ke torchvision DeepLabV3/FCN atau U-Net kecil.

# %%
import importlib
import importlib.util
import subprocess
import sys


def ensure_packages(packages, auto_install=True, no_deps=()):
    """Cek package (import name → pip name); install yang hilang kalau auto_install. Tidak pernah raise."""
    status = {}
    for imp, pip_name in packages.items():
        if importlib.util.find_spec(imp) is not None:
            status[imp] = "ok"
            continue
        if not auto_install:
            status[imp] = "missing"
            continue
        cmd = [sys.executable, "-m", "pip", "install", "-q", pip_name] + (["--no-deps"] if imp in no_deps else [])
        try:
            print("Installing", pip_name, "...")
            subprocess.check_call(cmd)
            importlib.invalidate_caches()
            status[imp] = "installed" if importlib.util.find_spec(imp) else "failed"
        except Exception as e:  # noqa
            status[imp] = f"failed ({type(e).__name__})"
    for k, v in status.items():
        print(f"  {k:30s}: {v}")
    return status


print("Core:")
PKG_STATUS = ensure_packages({"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "PIL": "pillow",
                              "matplotlib": "matplotlib"}, CFG.AUTO_INSTALL)
if importlib.util.find_spec("torch") is None:
    print("\n❌ PyTorch belum terinstall — segmentasi butuh torch. Install dari https://pytorch.org/get-started/locally/")
    print("   CPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
else:
    print("Segmentation models (opsional):")
    PKG_STATUS.update(ensure_packages(
        {"yaml": "pyyaml", "huggingface_hub": "huggingface_hub", "safetensors": "safetensors", "timm": "timm",
         "segmentation_models_pytorch": "segmentation-models-pytorch"},
        CFG.AUTO_INSTALL and CFG.INSTALL_SMP, no_deps=("timm", "segmentation_models_pytorch")))

# %% [markdown]
# # 4. Import, flag ketersediaan, device, helper
#
# **Cara membaca output:** `Model backend` = jalur model yang akan dipakai. `Device: cuda` = GPU terpakai.

# %%
import gc
import glob
import json
import math
import random
import time
import traceback
import warnings
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 300_000_000
try:
    from IPython.display import display
except Exception:  # noqa
    display = print

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except Exception:  # noqa
    HAS_TORCH = False
    torch, nn, F = None, None, None
    Dataset = object

HAS_TV = HAS_SMP = False
if HAS_TORCH:
    try:
        import torchvision
        HAS_TV = True
    except Exception as e:  # noqa
        print("torchvision tidak bisa di-import:", str(e)[:120])
    try:
        import segmentation_models_pytorch as smp
        HAS_SMP = True
    except Exception as e:  # noqa
        if importlib.util.find_spec("segmentation_models_pytorch"):
            print("smp ada tapi gagal di-import:", str(e)[:150])

DEVICE = "cuda" if (HAS_TORCH and CFG.USE_GPU and torch.cuda.is_available()) else "cpu"
SMP_ARCHS = {"Unet", "UnetPlusPlus", "FPN", "DeepLabV3Plus", "DeepLabV3", "PSPNet", "Linknet", "MAnet", "PAN"}
if CFG.ARCH == "auto":
    MODEL_BACKEND = "smp" if HAS_SMP else ("torchvision" if HAS_TV else "unet_small")
elif CFG.ARCH in SMP_ARCHS:
    MODEL_BACKEND = "smp" if HAS_SMP else ("torchvision" if HAS_TV else "unet_small")
    if not HAS_SMP:
        print(f"⚠️ ARCH={CFG.ARCH} butuh smp (tidak ada) → fallback {MODEL_BACKEND}")
elif CFG.ARCH in ("deeplabv3", "fcn"):
    MODEL_BACKEND = "torchvision" if HAS_TV else "unet_small"
else:
    MODEL_BACKEND = "unet_small"
print(f"torch: {torch.__version__ if HAS_TORCH else '-'} | torchvision: {HAS_TV} | smp: {HAS_SMP}")
print(f"Device: {DEVICE} | Model backend: {MODEL_BACKEND}")
if HAS_TORCH and DEVICE == "cpu" and CFG.USE_GPU:
    print("ℹ️ Tidak ada GPU CUDA yang terdeteksi → CPU. Kecilkan IMG_SIZE/EPOCHS kalau terlalu lama.")

# %%
def apply_run_mode():
    if CFG.RUN_MODE == "fast":
        CFG.EPOCHS = min(CFG.EPOCHS, CFG.FAST_EPOCHS)
        CFG.N_FOLDS = min(CFG.N_FOLDS, CFG.FAST_FOLDS)
        CFG.IMG_SIZE = min(CFG.IMG_SIZE, CFG.FAST_IMG_SIZE)
        CFG.PATIENCE = min(CFG.PATIENCE, 2)
        print(f"[FAST MODE] epochs={CFG.EPOCHS} folds={CFG.N_FOLDS} img={CFG.IMG_SIZE} sample≤{CFG.FAST_SAMPLE}")
    else:
        print("[FULL MODE]")
    CFG.IMG_SIZE = max(32, int(round(CFG.IMG_SIZE / 32)) * 32)


apply_run_mode()
OUT_DIR = CFG.OUTPUT_DIR
FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR = (os.path.join(OUT_DIR, d) for d in ("figures", "tables", "models", "submissions"))
for _d in (OUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR):
    os.makedirs(_d, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


seed_everything(CFG.SEED)
_FIG_COUNTER = [0]


def save_fig(name, fig=None, show=True):
    """Simpan figure ke figures/figNN_<name>.png (nomor urut otomatis) lalu tutup."""
    fig = fig if fig is not None else plt.gcf()
    _FIG_COUNTER[0] += 1
    path = os.path.join(FIG_DIR, f"fig{_FIG_COUNTER[0]:02d}_{name}.png")
    try:
        fig.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight")
        print("  [fig] saved:", path)
    except Exception as e:  # noqa
        print("  gagal simpan figure:", e)
    if show and "agg" not in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)
    return path


def save_table(df, name, index=False):
    path = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(path, index=index)
    print("  [table] saved:", path)
    return path


EXPERIMENT_LOG = []


def log_experiment(**kw):
    kw.update({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run_mode": CFG.RUN_MODE, "arch": CFG.ARCH,
               "encoder": CFG.ENCODER, "img_size": CFG.IMG_SIZE})
    EXPERIMENT_LOG.append(kw)
    pd.DataFrame(EXPERIMENT_LOG).to_csv(os.path.join(OUT_DIR, "experiment_log.csv"), index=False)


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
            seps = [",", ";", "	", "|"]
            sep = max(seps, key=head.count) if any(s in head for s in seps) else ","
            return pd.read_csv(path, sep=sep, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kw)

# %% [markdown]
# # 5. Load data — deteksi layout otomatis
#
# Hasil: `train_df` (kolom `img_path`, `mask_path` atau `rle_rows`, `stem`) dan `test_df` (`img_path`, `stem`).
#
# **Jika X maka Y:**
# - `0 pasangan gambar–mask` → nama file mask beda dari gambar. Cek contoh nama di output; tambahkan akhiran ke `MASK_SUFFIXES`.
# - Layout salah → set `CFG.DATA_LAYOUT = "folder"` atau `"csv_rle"`.
# - Test tidak ditemukan → training & evaluasi OOF tetap jalan, submission di-skip.

# %%
IMAGE_EXTS = tuple(e.lower() for e in CFG.IMAGE_EXTS)


def list_images(folder):
    out = []
    if folder and os.path.isdir(folder):
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(IMAGE_EXTS)]
    return out


def first_existing(cands):
    for d in cands:
        if d and os.path.isdir(d) and list_images(d):
            return d
    return None


def strip_suffix(stem):
    for suf in sorted(CFG.MASK_SUFFIXES, key=len, reverse=True):
        if suf and stem.lower().endswith(suf.lower()):
            return stem[: -len(suf)]
    return stem


def pair_images_masks(img_dir, mask_dir):
    imgs = list_images(img_dir)
    masks = list_images(mask_dir)
    if os.path.abspath(img_dir) == os.path.abspath(mask_dir or ""):     # gambar & mask satu folder (img.png + img_mask.png)
        masks = [m for m in imgs if strip_suffix(os.path.splitext(os.path.basename(m))[0]) !=
                 os.path.splitext(os.path.basename(m))[0]]
        imgs = [p for p in imgs if p not in set(masks)]
    by_stem = {}
    for m in masks:
        by_stem.setdefault(strip_suffix(os.path.splitext(os.path.basename(m))[0]), m)
    rows, unmatched = [], []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = by_stem.get(stem) or by_stem.get(strip_suffix(stem))
        (rows.append({"stem": stem, "img_path": p, "mask_path": m}) if m else unmatched.append(stem))
    return pd.DataFrame(rows), unmatched, len(masks)


def detect_rle_columns(df):
    def looks_rle(v):
        if not isinstance(v, str):
            return False
        toks = v.split()
        return len(toks) >= 2 and len(toks) % 2 == 0 and all(t.isdigit() for t in toks[:6])
    rle_col = CFG.RLE_COL or next((c for c in df.columns if df[c].map(looks_rle).mean() > 0.05), None)
    id_col = CFG.RLE_ID_COL or next((c for c in df.columns if c != rle_col and df[c].dtype == object), df.columns[0])
    cls_col = CFG.RLE_CLASS_COL
    if cls_col is None:
        for c in df.columns:
            if c not in (rle_col, id_col) and 1 < df[c].nunique() <= 50 and df.groupby(id_col)[c].nunique().max() > 1:
                cls_col = c
                break
    return id_col, rle_col, cls_col


TRAIN_IMG_DIR = CFG.TRAIN_IMAGE_DIR if os.path.isdir(CFG.TRAIN_IMAGE_DIR) else first_existing(CFG.IMAGE_DIR_CANDIDATES)
TRAIN_MSK_DIR = CFG.TRAIN_MASK_DIR if os.path.isdir(CFG.TRAIN_MASK_DIR) else first_existing(CFG.MASK_DIR_CANDIDATES)
TEST_IMG_DIR = CFG.TEST_IMAGE_DIR if os.path.isdir(CFG.TEST_IMAGE_DIR) else first_existing(CFG.TEST_DIR_CANDIDATES)
LAYOUT = CFG.DATA_LAYOUT
if LAYOUT == "auto":
    if TRAIN_MSK_DIR and TRAIN_IMG_DIR:
        LAYOUT = "folder"
    elif os.path.exists(CFG.TRAIN_CSV):
        LAYOUT = "csv_rle"
    else:
        raise FileNotFoundError("Tidak menemukan folder mask maupun train.csv. Set TRAIN_IMAGE_DIR / TRAIN_MASK_DIR / TRAIN_CSV.")
print(f"Layout terdeteksi: {LAYOUT} | train images: {TRAIN_IMG_DIR} | masks: {TRAIN_MSK_DIR} | test: {TEST_IMG_DIR}")

RLE_META = {}
if LAYOUT == "folder":
    train_df, unmatched, n_masks = pair_images_masks(TRAIN_IMG_DIR, TRAIN_MSK_DIR)
    print(f"Pasangan gambar–mask: {len(train_df)} | gambar tanpa mask: {len(unmatched)} | file mask: {n_masks}")
    if unmatched:
        print("  contoh tanpa mask:", unmatched[:5])
    if train_df.empty:
        raise ValueError("0 pasangan gambar–mask. Cek nama file dan CFG.MASK_SUFFIXES.")
else:
    raw = smart_read(CFG.TRAIN_CSV)
    id_col, rle_col, cls_col = detect_rle_columns(raw)
    if rle_col is None:
        raise ValueError("Kolom RLE tidak ditemukan di train.csv → set CFG.RLE_COL.")
    RLE_META = {"id_col": id_col, "rle_col": rle_col, "cls_col": cls_col}
    print("Kolom RLE:", RLE_META)
    imgs = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(TRAIN_IMG_DIR)}
    rows = []
    for stem, g in raw.groupby(raw[id_col].astype(str).map(lambda s: os.path.splitext(os.path.basename(s))[0])):
        if stem in imgs:
            rows.append({"stem": stem, "img_path": imgs[stem], "mask_path": None,
                         "rle_rows": [(r[cls_col] if cls_col else 1, r[rle_col]) for _, r in g.iterrows()]})
    train_df = pd.DataFrame(rows)
    print(f"Gambar dengan baris RLE: {len(train_df)} dari {raw[id_col].nunique()} id unik")

test_df = pd.DataFrame({"img_path": list_images(TEST_IMG_DIR)}) if TEST_IMG_DIR else pd.DataFrame(columns=["img_path"])
test_df["stem"] = test_df["img_path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
SAMPLE_SUB = smart_read(CFG.SAMPLE_SUB_PATH) if os.path.exists(CFG.SAMPLE_SUB_PATH) else None
print(f"Train: {len(train_df)} | Test: {len(test_df)} | sample_submission: {None if SAMPLE_SUB is None else SAMPLE_SUB.shape}")

# %% [markdown]
# # 6. Mask → index kelas
#
# Semua bentuk mask diubah menjadi array `H×W` berisi index kelas `0..K-1` (0 = background):
#
# - **index**: nilai pixel sudah 0,1,2,... → dipakai langsung.
# - **binary255**: nilai {0,255} → 0/1.
# - **rgb**: setiap warna unik = satu kelas (hitam = background). Palette disimpan ke `tables/class_palette.csv`.
# - **csv_rle**: RLE di-decode per kelas (urutan `RLE_ORDER`). Kalau overlay di EDA terlihat "miring/terputus",
#   ganti `RLE_ORDER` ("F" ↔ "C").

# %%
def rle_decode(rle, shape, order=None, one_indexed=None):
    order = order or CFG.RLE_ORDER
    one_indexed = CFG.RLE_ONE_INDEXED if one_indexed is None else one_indexed
    h, w = shape
    m = np.zeros(h * w, dtype=np.uint8)
    if not isinstance(rle, str) or not rle.strip() or rle.strip() in ("-", "nan"):
        return m.reshape((h, w), order=order)
    s = np.asarray(rle.split(), dtype=np.int64)
    starts, lengths = s[0::2] - (1 if one_indexed else 0), s[1::2]
    for a, l in zip(starts, lengths):
        m[a: a + l] = 1
    return m.reshape((h, w), order=order)


def rle_encode(mask, order=None, one_indexed=None):
    order = order or CFG.RLE_ORDER
    one_indexed = CFG.RLE_ONE_INDEXED if one_indexed is None else one_indexed
    px = np.asarray(mask, dtype=np.uint8).flatten(order=order)
    px = np.concatenate([[0], px, [0]])
    runs = np.where(px[1:] != px[:-1])[0] + (1 if one_indexed else 0)
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def read_image(path):
    img = Image.open(path)
    if img.mode in ("I;16", "I", "F"):
        a = np.asarray(img).astype(np.float32)
        a = (255 * (a - a.min()) / max(1e-6, a.max() - a.min())).astype(np.uint8)
        img = Image.fromarray(a)
    return np.asarray(img.convert("RGB"))


def scan_masks(paths, max_n=400):
    """Tentukan MASK_MODE & daftar nilai/warna dari sampel mask."""
    vals, colors, is_rgb = Counter(), Counter(), False
    for p in paths[:max_n]:
        a = np.asarray(Image.open(p))
        if a.ndim == 3 and a.shape[2] >= 3 and not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])):
            is_rgb = True
            c = a[..., :3].reshape(-1, 3)
            u, n = np.unique(c, axis=0, return_counts=True)
            for uu, nn_ in zip(map(tuple, u), n):
                colors[uu] += int(nn_)
        else:
            a = a[..., 0] if a.ndim == 3 else a
            u, n = np.unique(a, return_counts=True)
            for uu, nn_ in zip(u.tolist(), n):
                vals[uu] += int(nn_)
    if is_rgb:
        return "rgb", colors
    keys = sorted(vals)
    if set(keys) <= {0, 255} or (len(keys) == 2 and keys[-1] > 1 and keys[0] == 0):
        return "binary255", vals
    return "index", vals


if LAYOUT == "folder":
    detected_mode, mask_stats = scan_masks(train_df["mask_path"].tolist())
    MASK_MODE = detected_mode if CFG.MASK_MODE == "auto" else CFG.MASK_MODE
    if MASK_MODE == "rgb":
        palette = CFG.RGB_PALETTE or {}
        if not palette:
            cols = [c for c, _ in mask_stats.most_common() if mask_stats[c] > 50]
            cols = sorted(cols, key=lambda c: (c != (0, 0, 0), c))
            palette = {tuple(c): i for i, c in enumerate(cols)}
        RGB_PALETTE = {tuple(k): int(v) for k, v in palette.items()}
        N_CLASSES = CFG.N_CLASSES or len(RGB_PALETTE)
        save_table(pd.DataFrame([{"class_idx": v, "rgb": k} for k, v in RGB_PALETTE.items()]), "class_palette")
        VALUE_MAP = None
    elif MASK_MODE == "binary255":
        N_CLASSES, VALUE_MAP, RGB_PALETTE = 2, None, None
    else:
        values = sorted(v for v in mask_stats if v != CFG.IGNORE_INDEX)
        VALUE_MAP = {v: i for i, v in enumerate(values)} if values != list(range(len(values))) else None
        N_CLASSES = CFG.N_CLASSES or len(values)
        RGB_PALETTE = None
        if VALUE_MAP:
            print("ℹ️ Nilai mask tidak berurutan → dipetakan:", VALUE_MAP)
else:
    MASK_MODE, VALUE_MAP, RGB_PALETTE = "rle", None, None
    cls_vals = sorted({c for rr in train_df["rle_rows"] for c, _ in rr}, key=str)
    RLE_CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(cls_vals)}
    N_CLASSES = CFG.N_CLASSES or (len(cls_vals) + 1)

BINARY = N_CLASSES == 2
CLASS_NAMES = CFG.CLASS_NAMES or (["background", "object"] if BINARY else [f"class_{i}" for i in range(N_CLASSES)])
if MASK_MODE == "rle" and not CFG.CLASS_NAMES and not BINARY:
    CLASS_NAMES = ["background"] + [str(c) for c in cls_vals]
print(f"MASK_MODE: {MASK_MODE} | N_CLASSES: {N_CLASSES} ({'biner' if BINARY else 'multikelas'}) | kelas: {CLASS_NAMES}")


def read_mask(row, shape=None):
    """Return mask index kelas (H, W) uint8 — sumber apa pun."""
    if MASK_MODE == "rle":
        h, w = shape if shape is not None else read_image(row["img_path"]).shape[:2]
        m = np.zeros((h, w), np.uint8)
        for c, rle in row["rle_rows"]:
            m[rle_decode(rle, (h, w)) > 0] = RLE_CLASS_TO_IDX.get(c, 1)
        return m
    a = np.asarray(Image.open(row["mask_path"]))
    if MASK_MODE == "rgb":
        a = a[..., :3]
        out = np.zeros(a.shape[:2], np.uint8)
        for col, idx in RGB_PALETTE.items():
            out[np.all(a == np.array(col, dtype=a.dtype), axis=-1)] = idx
        return out
    a = a[..., 0] if a.ndim == 3 else a
    if MASK_MODE == "binary255":
        return (a > 127).astype(np.uint8) if a.max() > 1 else (a > 0).astype(np.uint8)
    if VALUE_MAP:
        out = np.zeros(a.shape, np.uint8)
        for v, i in VALUE_MAP.items():
            out[a == v] = i
        if CFG.IGNORE_INDEX is not None:
            out[a == CFG.IGNORE_INDEX] = 255
        return out
    return a.astype(np.uint8)

# %% [markdown]
# # 7. EDA
#
# **Cara membaca:**
# - *Frekuensi pixel per kelas*: kelas yang < 1% pixel → sulit; pakai loss Dice (sudah default) dan cek Dice per kelas.
# - *% mask kosong*: kalau tinggi (> 30%), model cenderung memprediksi kosong → tuning `MIN_AREA` & threshold penting.
# - *Overlay*: **wajib dilihat**. Kalau mask tidak pas dengan objek → orientasi RLE / pasangan file salah.

# %%
def image_size(p):
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:  # noqa
        return (np.nan, np.nan)


eda_n = min(len(train_df), 400)
eda_rows = train_df.sample(eda_n, random_state=CFG.SEED) if len(train_df) > eda_n else train_df
sizes = pd.DataFrame([image_size(p) for p in eda_rows["img_path"]], columns=["w", "h"])
pix = np.zeros(N_CLASSES, np.int64)
empty, dominant = [], {}
for _, r in eda_rows.iterrows():
    m = read_mask(r)
    cnt = np.bincount(m[m < N_CLASSES].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    pix += cnt
    empty.append(cnt[1:].sum() == 0)
    dominant[r["stem"]] = int(np.argmax(cnt[1:]) + 1) if cnt[1:].sum() else 0
class_freq = pd.DataFrame({"class": CLASS_NAMES[:N_CLASSES], "pixel_pct": 100 * pix / max(1, pix.sum())})
print("Ukuran gambar (w×h) paling umum:", sizes.value_counts().head(3).to_dict())
print(f"Mask kosong (tanpa objek): {100 * np.mean(empty):.1f}%")
display(class_freq.round(3))
save_table(class_freq, "class_pixel_frequency")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].bar(class_freq["class"], class_freq["pixel_pct"], color="#4C72B0")
axes[0].set_title("Frekuensi pixel per kelas (%)")
axes[0].tick_params(axis="x", rotation=30)
axes[1].scatter(sizes["w"], sizes["h"], s=8, alpha=.5)
axes[1].set_title("Ukuran gambar (lebar vs tinggi)")
save_fig("class_frequency_and_sizes", fig)

CMAP = plt.get_cmap("tab10")


def overlay(img, mask, alpha=0.45):
    out = img.astype(np.float32).copy()
    for c in range(1, N_CLASSES):
        col = np.array(CMAP(c % 10)[:3]) * 255
        out[mask == c] = (1 - alpha) * out[mask == c] + alpha * col
    return out.clip(0, 255).astype(np.uint8)


k = min(6, len(train_df))
fig, axes = plt.subplots(2, k, figsize=(2.6 * k, 5.4))
axes = np.atleast_2d(axes)
for j, (_, r) in enumerate(train_df.sample(k, random_state=1).iterrows()):
    img = read_image(r["img_path"])
    m = read_mask(r, img.shape[:2])
    axes[0, j].imshow(img)
    axes[0, j].set_title(r["stem"][:18], fontsize=8)
    axes[1, j].imshow(overlay(img, m))
    for a in axes[:, j]:
        a.axis("off")
axes[0, 0].set_ylabel("image")
fig.suptitle("Contoh gambar (atas) dan overlay mask (bawah)")
save_fig("sample_overlays", fig)

# %% [markdown]
# # 8. Cross-validation folds
#
# Stratifikasi berdasarkan kelas dominan (0 = mask kosong) supaya setiap fold punya proporsi gambar kosong/berisi
# yang mirip. **Jika** gambar berasal dari satu scene besar yang dipotong (tile citra satelit) **maka** sebaiknya
# fold per scene (GroupKFold) — set kolom grup manual: `train_df["group"] = ...` lalu ganti splitter.

# %%
if CFG.RUN_MODE == "fast" and len(train_df) > CFG.FAST_SAMPLE:
    train_df = train_df.sample(CFG.FAST_SAMPLE, random_state=CFG.SEED).reset_index(drop=True)
    print(f"[FAST MODE] train di-subsample ke {len(train_df)} gambar")
train_df = train_df.reset_index(drop=True)
strat = train_df["stem"].map(dominant).fillna(-1).astype(int) if dominant else pd.Series(0, index=train_df.index)
if len(train_df) < 4:
    raise ValueError("Terlalu sedikit gambar train (< 4).")
n_folds = max(2, min(CFG.N_FOLDS, len(train_df) // 2))
min_count = strat.value_counts().min()
splitter = StratifiedKFold(n_folds, shuffle=True, random_state=CFG.SEED) if min_count >= n_folds else KFold(n_folds, shuffle=True, random_state=CFG.SEED)
train_df["fold"] = -1
for f, (_, va) in enumerate(splitter.split(train_df, strat if min_count >= n_folds else None)):
    train_df.loc[va, "fold"] = f
print(train_df["fold"].value_counts().sort_index().to_dict())

# %% [markdown]
# # 9. Dataset, augmentasi, DataLoader
#
# Augmentasi diterapkan **bersamaan** ke gambar dan mask (flip/rotate/crop harus identik). Mask di-resize dengan
# *nearest neighbour* supaya nilai kelas tidak tercampur.

# %%
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
RESAMPLE = getattr(Image, "Resampling", Image)


def resize_pair(img, mask, size):
    im = np.asarray(Image.fromarray(img).resize((size, size), RESAMPLE.BILINEAR))
    mk = None if mask is None else np.asarray(Image.fromarray(mask).resize((size, size), RESAMPLE.NEAREST))
    return im, mk


def augment(img, mask, rng):
    h, w = img.shape[:2]
    if CFG.TILE_SIZE and min(h, w) > CFG.TILE_SIZE:
        t = CFG.TILE_SIZE
        y0, x0 = rng.integers(0, h - t + 1), rng.integers(0, w - t + 1)
        img, mask = img[y0:y0 + t, x0:x0 + t], mask[y0:y0 + t, x0:x0 + t]
    elif CFG.AUG_SCALE and CFG.AUG_SCALE[0] < 1:
        s = math.sqrt(rng.uniform(*CFG.AUG_SCALE))
        ch, cw = max(8, int(h * s)), max(8, int(w * s))
        y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
        img, mask = img[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]
    if CFG.AUG_HFLIP and rng.random() < .5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if CFG.AUG_VFLIP and rng.random() < .5:
        img, mask = img[::-1], mask[::-1]
    if CFG.AUG_ROT90 and rng.random() < .5:
        k = int(rng.integers(1, 4))
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    img = img.astype(np.float32)
    if CFG.AUG_BRIGHTNESS:
        img = img * (1 + rng.uniform(-CFG.AUG_BRIGHTNESS, CFG.AUG_BRIGHTNESS))
    if CFG.AUG_CONTRAST:
        mu = img.mean()
        img = (img - mu) * (1 + rng.uniform(-CFG.AUG_CONTRAST, CFG.AUG_CONTRAST)) + mu
    return np.ascontiguousarray(img.clip(0, 255).astype(np.uint8)), np.ascontiguousarray(mask)


def to_tensor_img(img):
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class SegDataset(Dataset):
    def __init__(self, df, train=True, with_mask=True, size=None):
        self.df, self.train, self.with_mask = df.reset_index(drop=True), train, with_mask
        self.size = size or CFG.IMG_SIZE
        self.rng = np.random.default_rng(CFG.SEED)
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def load(self, i):
        if i in self.cache:
            return self.cache[i]
        r = self.df.iloc[i]
        img = read_image(r["img_path"])
        mask = read_mask(r, img.shape[:2]) if self.with_mask else None
        if len(self.cache) < 3000 and img.size < 2_000_000:
            self.cache[i] = (img, mask)
        return img, mask

    def __getitem__(self, i):
        img, mask = self.load(i)
        if self.train:
            img, mask = augment(img, mask, self.rng)
        img, mask = resize_pair(img, mask, self.size)
        x = to_tensor_img(img)
        if mask is None:
            return x, torch.zeros(1)
        return x, torch.from_numpy(mask.astype(np.int64))

# %% [markdown]
# # 10. Model factory: smp → torchvision → U-Net kecil
#
# | Backend | Kapan | Catatan |
# |---|---|---|
# | `smp` (U-Net, FPN, DeepLabV3+, UNet++) | default kalau terinstall | encoder pretrained ImageNet → biasanya terbaik |
# | `torchvision` (DeepLabV3 / FCN) | smp tidak ada | backbone pretrained ResNet / MobileNet |
# | `unet_small` | tanpa smp & torchvision / dataset sangat kecil / gambar abu-abu sederhana | dilatih dari nol |

# %%
OUT_CH = 1 if BINARY else N_CLASSES


class ConvBlock(nn.Module if HAS_TORCH else object):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module if HAS_TORCH else object):
    """U-Net 4 level (±7 juta parameter dengan base=32). Tanpa pretrained."""

    def __init__(self, out_ch, base=32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8]
        self.downs = nn.ModuleList([ConvBlock(3 if i == 0 else ch[i - 1], ch[i]) for i in range(4)])
        self.bottom = ConvBlock(ch[3], ch[3] * 2)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch[i] * 2, ch[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([ConvBlock(ch[i] * 2, ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for up, dec, s in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, s], 1))
        return self.head(x)


class TVWrapper(nn.Module if HAS_TORCH else object):
    """torchvision segmentation model → output tensor (bukan dict), ukuran sama dengan input."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)["out"]
        return out if out.shape[-2:] == x.shape[-2:] else F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model():
    if MODEL_BACKEND == "smp":
        arch = CFG.ARCH if CFG.ARCH in SMP_ARCHS else "Unet"
        try:
            return getattr(smp, arch)(encoder_name=CFG.ENCODER, encoder_weights="imagenet" if CFG.PRETRAINED else None,
                                      in_channels=3, classes=OUT_CH), f"smp.{arch}({CFG.ENCODER})"
        except Exception as e:  # noqa
            print(f"⚠️ smp {arch}/{CFG.ENCODER} gagal ({str(e)[:120]}) → coba tanpa pretrained")
            try:
                return getattr(smp, arch)(encoder_name=CFG.ENCODER, encoder_weights=None, in_channels=3,
                                          classes=OUT_CH), f"smp.{arch}({CFG.ENCODER}, scratch)"
            except Exception as e2:  # noqa
                print("   gagal lagi:", str(e2)[:120], "→ fallback torchvision/unet_small")
    if MODEL_BACKEND in ("smp", "torchvision") and HAS_TV:
        seg = torchvision.models.segmentation
        name = "fcn" if CFG.ARCH == "fcn" else "deeplabv3"
        for builder, wb in [(f"{name}_resnet50", "ResNet50_Weights"), ("deeplabv3_mobilenet_v3_large", "MobileNet_V3_Large_Weights")]:
            try:
                wbk = getattr(torchvision.models, wb).IMAGENET1K_V1 if CFG.PRETRAINED else None
                m = getattr(seg, builder)(weights=None, weights_backbone=wbk, num_classes=OUT_CH, aux_loss=False)
                return TVWrapper(m), f"torchvision.{builder}{'' if CFG.PRETRAINED else ' (scratch)'}"
            except Exception as e:  # noqa
                print(f"⚠️ torchvision {builder} gagal: {str(e)[:120]}")
    return UNetSmall(OUT_CH), "unet_small (scratch)"


if HAS_TORCH:
    _m, MODEL_NAME = build_model()
    n_params = sum(p.numel() for p in _m.parameters()) / 1e6
    print(f"Model: {MODEL_NAME} | {n_params:.1f} M parameter | output channel: {OUT_CH}")
    del _m

# %% [markdown]
# # 11. Loss & metrik
#
# - **Biner**: BCE + Dice pada channel tunggal (sigmoid).
# - **Multikelas**: Cross-Entropy + Dice multikelas (softmax). `IGNORE_INDEX` (255) tidak dihitung.
# - Metrik: **Dice** = 2|A∩B| / (|A|+|B|), **IoU** = |A∩B| / |A∪B|, dihitung per kelas lalu dirata-rata
#   (tanpa background kalau `METRIC_INCLUDE_BG=False`). Gambar yang kelasnya tidak ada di GT **dan** prediksi → skor 1.

# %%
def dice_loss(logits, target):
    valid = (target != 255)
    if BINARY:
        p = torch.sigmoid(logits[:, 0])
        t = (target == 1).float()
        p, t = p * valid, t * valid
        inter = (p * t).sum((1, 2))
        return (1 - (2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()
    p = torch.softmax(logits, 1)
    t = F.one_hot(target.clamp(0, N_CLASSES - 1), N_CLASSES).permute(0, 3, 1, 2).float()
    v = valid.unsqueeze(1).float()
    p, t = p * v, t * v
    inter = (p * t).sum((0, 2, 3))
    d = (2 * inter + 1) / (p.sum((0, 2, 3)) + t.sum((0, 2, 3)) + 1)
    return 1 - (d[1:].mean() if N_CLASSES > 1 else d.mean())


def seg_loss(logits, target):
    valid = (target != 255)
    if BINARY:
        bce = F.binary_cross_entropy_with_logits(logits[:, 0], (target == 1).float(), reduction="none")
        base = (bce * valid).sum() / valid.sum().clamp(min=1)
        if CFG.LOSS == "focal_dice":
            pt = torch.exp(-bce)
            base = (((1 - pt) ** 2 * bce) * valid).sum() / valid.sum().clamp(min=1)
    else:
        base = F.cross_entropy(logits, target, ignore_index=255)
    if CFG.LOSS == "ce":
        return base
    if CFG.LOSS == "dice":
        return dice_loss(logits, target)
    return (1 - CFG.DICE_WEIGHT) * base + CFG.DICE_WEIGHT * dice_loss(logits, target)


def per_class_scores(pred, gt, n_classes):
    """pred, gt: (H, W) int. Return arrays dice[c], iou[c] (NaN = kelas tidak ada di GT & pred)."""
    dice, iou = np.full(n_classes, np.nan), np.full(n_classes, np.nan)
    valid = gt != 255
    for c in range(n_classes):
        p, g = (pred == c) & valid, (gt == c) & valid
        sp, sg = p.sum(), g.sum()
        if sp + sg == 0:
            dice[c], iou[c] = 1.0, 1.0
            continue
        inter = (p & g).sum()
        dice[c] = 2 * inter / (sp + sg)
        iou[c] = inter / (sp + sg - inter)
    return dice, iou


def summarize_scores(rows):
    d = np.nanmean(np.stack([r[0] for r in rows]), 0)
    i = np.nanmean(np.stack([r[1] for r in rows]), 0)
    sl = slice(0, None) if CFG.METRIC_INCLUDE_BG or N_CLASSES == 1 else slice(1, None)
    return {"dice": float(np.nanmean(d[sl])), "iou": float(np.nanmean(i[sl])), "dice_per_class": d, "iou_per_class": i}

# %% [markdown]
# # 12. Training K-fold
#
# **Cara membaca output per epoch:** `train_loss` turun & `val_<metric>` naik = sehat. `val` stagnan sejak awal →
# cek overlay (mask salah?) atau LR terlalu besar/kecil. `train_loss` turun tapi `val` turun → overfit (perbanyak
# augmentasi, kurangi epoch, encoder lebih kecil).
#
# OOF (prediksi validasi) disimpan sebagai probabilitas resolusi `IMG_SIZE` untuk tuning threshold di section 13.

# %%
def make_loader(df, train, bs, with_mask=True):
    return DataLoader(SegDataset(df, train=train, with_mask=with_mask), batch_size=bs, shuffle=train,
                      num_workers=CFG.NUM_WORKERS, drop_last=train and len(df) > bs, pin_memory=DEVICE == "cuda")


def predict_probs(model, loader, tta=("none",)):
    """Return probs (N, C_out, S, S) float16."""
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            acc = 0
            for t in tta:
                xi = torch.flip(x, [3]) if t == "hflip" else (torch.flip(x, [2]) if t == "vflip" else x)
                with torch.autocast(device_type="cuda", enabled=CFG.USE_AMP and DEVICE == "cuda"):
                    lo = model(xi).float()
                p = torch.sigmoid(lo) if BINARY else torch.softmax(lo, 1)
                p = torch.flip(p, [3]) if t == "hflip" else (torch.flip(p, [2]) if t == "vflip" else p)
                acc = acc + p
            outs.append((acc / len(tta)).cpu().numpy().astype(np.float16))
    return np.concatenate(outs) if outs else np.zeros((0, OUT_CH, CFG.IMG_SIZE, CFG.IMG_SIZE), np.float16)


def probs_to_mask(p, threshold=None):
    """p: (C_out, H, W) → mask index (H, W)."""
    if BINARY:
        return (p[0] > (CFG.THRESHOLD if threshold is None else threshold)).astype(np.uint8)
    return p.argmax(0).astype(np.uint8)


def train_fold(fold):
    tr, va = train_df[train_df.fold != fold], train_df[train_df.fold == fold]
    bs = CFG.BATCH_SIZE
    while True:
        try:
            seed_everything(CFG.SEED + fold)
            model, name = build_model()
            model.to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
            steps = max(1, CFG.EPOCHS * math.ceil(len(tr) / bs))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG.LR, total_steps=steps, pct_start=0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP and DEVICE == "cuda")
            tl, vl = make_loader(tr, True, bs), make_loader(va, False, bs)
            best, best_ep, hist, t0 = -1, -1, [], time.time()
            ckpt = os.path.join(MODEL_DIR, f"fold{fold}.pt")
            va_masks = [resize_pair(np.zeros((2, 2, 3), np.uint8), read_mask(r), CFG.IMG_SIZE)[1] for _, r in va.iterrows()]
            for ep in range(CFG.EPOCHS):
                model.train()
                losses = []
                for x, y in tl:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", enabled=CFG.USE_AMP and DEVICE == "cuda"):
                        lo = model(x)
                    loss = seg_loss(lo.float(), y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    if sched.last_epoch < steps - 1:
                        sched.step()
                    losses.append(loss.item())
                probs = predict_probs(model, vl)
                sc = summarize_scores([per_class_scores(probs_to_mask(p.astype(np.float32)), g, N_CLASSES)
                                       for p, g in zip(probs, va_masks)])
                score = sc[CFG.METRIC]
                hist.append({"fold": fold, "epoch": ep + 1, "train_loss": float(np.mean(losses)), f"val_{CFG.METRIC}": score})
                flag = ""
                if score > best:
                    best, best_ep, flag = score, ep, " *"
                    torch.save(model.state_dict(), ckpt)
                print(f"  fold {fold} ep {ep + 1:02d}/{CFG.EPOCHS} loss {np.mean(losses):.4f} | val {CFG.METRIC} {score:.4f}{flag}")
                if ep - best_ep >= CFG.PATIENCE:
                    print("  early stopping")
                    break
                if CFG.MAX_MINUTES_PER_FOLD and (time.time() - t0) / 60 > CFG.MAX_MINUTES_PER_FOLD:
                    print("  batas waktu fold tercapai")
                    break
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            probs = predict_probs(model, vl, CFG.TTA)
            del model, opt
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            return {"fold": fold, "best": best, "best_epoch": best_ep + 1, "probs": probs, "masks": va_masks,
                    "index": va.index.values, "history": hist, "name": name}
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                print(f"⚠️ CUDA OOM → BATCH_SIZE diturunkan ke {bs}, ulangi fold {fold}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise


FOLD_RESULTS = []
if HAS_TORCH:
    folds = CFG.FOLDS_TO_RUN if CFG.FOLDS_TO_RUN is not None else list(range(n_folds))
    for f in folds:
        print(f"===== FOLD {f} =====")
        FOLD_RESULTS.append(train_fold(f))
    hist = pd.DataFrame([h for r in FOLD_RESULTS for h in r["history"]])
    save_table(hist, "training_history")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for f, g in hist.groupby("fold"):
        ax[0].plot(g["epoch"], g["train_loss"], label=f"fold {f}")
        ax[1].plot(g["epoch"], g[f"val_{CFG.METRIC}"], label=f"fold {f}")
    ax[0].set_title("Train loss")
    ax[1].set_title(f"Validation {CFG.METRIC}")
    ax[1].legend(fontsize=8)
    save_fig("learning_curves", fig)
    print("CV per fold:", {r["fold"]: round(r["best"], 4) for r in FOLD_RESULTS})
else:
    print("❌ torch tidak ada → training dilewati.")

# %% [markdown]
# # 13. OOF: tuning threshold & min area (post-processing)
#
# - **Biner**: cari threshold probabilitas terbaik (0.2–0.8) di OOF, lalu `MIN_AREA` (buang komponen kecil = noise).
# - **Multikelas**: argmax; `MIN_AREA` diterapkan per kelas (komponen kecil → background).
#
# Hasil tuning hanya dipakai kalau **lebih baik** dari default di OOF.

# %%
def remove_small(mask, min_area):
    if min_area <= 0:
        return mask
    try:
        import cv2
        out = mask.copy()
        for c in range(1, N_CLASSES):
            n, lab, stats, _ = cv2.connectedComponentsWithStats((mask == c).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    out[lab == i] = 0
        return out
    except ImportError:
        from scipy import ndimage
        out = mask.copy()
        for c in range(1, N_CLASSES):
            lab, n = ndimage.label(mask == c)
            if n:
                sizes_ = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
                for i, s in enumerate(sizes_, 1):
                    if s < min_area:
                        out[lab == i] = 0
        return out


def oof_score(th=None, min_area=0):
    rows = []
    for r in FOLD_RESULTS:
        for p, g in zip(r["probs"], r["masks"]):
            rows.append(per_class_scores(remove_small(probs_to_mask(p.astype(np.float32), th), min_area), g, N_CLASSES))
    return summarize_scores(rows)


BEST_TH, BEST_MIN_AREA = CFG.THRESHOLD, CFG.MIN_AREA
if FOLD_RESULTS:
    base = oof_score(CFG.THRESHOLD, CFG.MIN_AREA)
    print(f"OOF default: dice {base['dice']:.4f} | iou {base['iou']:.4f}")
    best_val = base[CFG.METRIC]
    if BINARY and CFG.TUNE_THRESHOLD:
        for th in np.arange(0.2, 0.81, 0.05):
            s = oof_score(th, CFG.MIN_AREA)[CFG.METRIC]
            if s > best_val + 1e-4:
                best_val, BEST_TH = s, float(round(th, 2))
    if CFG.TUNE_MIN_AREA:
        scale = CFG.IMG_SIZE ** 2
        for ma in [int(scale * f) for f in (0.0005, 0.001, 0.002, 0.005, 0.01)]:
            s = oof_score(BEST_TH, ma)[CFG.METRIC]
            if s > best_val + 1e-4:
                best_val, BEST_MIN_AREA = s, ma
    FINAL_OOF = oof_score(BEST_TH, BEST_MIN_AREA)
    print(f"Post-processing terpilih: threshold={BEST_TH} | min_area={BEST_MIN_AREA} px (pada {CFG.IMG_SIZE}²)")
    print(f"OOF final: dice {FINAL_OOF['dice']:.4f} | iou {FINAL_OOF['iou']:.4f}")
    per_class = pd.DataFrame({"class": CLASS_NAMES[:N_CLASSES], "dice": FINAL_OOF["dice_per_class"], "iou": FINAL_OOF["iou_per_class"]})
    display(per_class.round(4))
    save_table(per_class, "oof_per_class_scores")
    log_experiment(model=FOLD_RESULTS[0]["name"], oof_dice=FINAL_OOF["dice"], oof_iou=FINAL_OOF["iou"],
                   threshold=BEST_TH, min_area=BEST_MIN_AREA)

# %% [markdown]
# # 14. Evaluasi visual & debugging
#
# Figure *worst predictions* = gambar dengan skor terendah di OOF. Pola umum:
# - objek kecil hilang → naikkan `IMG_SIZE` / `TILE_SIZE`, turunkan `MIN_AREA`;
# - batas objek kasar → resolusi lebih tinggi, encoder lebih kuat, atau FPN/UNet++;
# - prediksi di area yang jelas background → tambah augmentasi warna, cek label noise.

# %%
if FOLD_RESULTS:
    recs = []
    for r in FOLD_RESULTS:
        for idx, p, g in zip(r["index"], r["probs"], r["masks"]):
            pm = remove_small(probs_to_mask(p.astype(np.float32), BEST_TH), BEST_MIN_AREA)
            d, i = per_class_scores(pm, g, N_CLASSES)
            sl = slice(1, None) if not CFG.METRIC_INCLUDE_BG and N_CLASSES > 1 else slice(0, None)
            recs.append({"idx": idx, "stem": train_df.loc[idx, "stem"], "dice": np.nanmean(d[sl]), "iou": np.nanmean(i[sl]),
                         "pred": pm, "gt": g})
    per_img = pd.DataFrame(recs).sort_values(CFG.METRIC)
    save_table(per_img[["stem", "dice", "iou"]], "oof_per_image_scores")
    k = min(5, len(per_img))
    fig, axes = plt.subplots(3, k, figsize=(2.7 * k, 8))
    axes = np.atleast_2d(axes).reshape(3, k)
    for j, (_, r) in enumerate(per_img.head(k).iterrows()):
        img = np.asarray(Image.fromarray(read_image(train_df.loc[r["idx"], "img_path"])).resize((CFG.IMG_SIZE, CFG.IMG_SIZE)))
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{r['stem'][:14]}\n{CFG.METRIC}={r[CFG.METRIC]:.2f}", fontsize=8)
        axes[1, j].imshow(overlay(img, r["gt"]))
        axes[2, j].imshow(overlay(img, r["pred"]))
        for a in axes[:, j]:
            a.axis("off")
    axes[1, 0].set_title("ground truth", fontsize=8, loc="left")
    axes[2, 0].set_title("prediksi", fontsize=8, loc="left")
    fig.suptitle("OOF — prediksi terburuk (atas: gambar, tengah: GT, bawah: prediksi)")
    save_fig("oof_worst_predictions", fig)
    if not BINARY:
        cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
        for r in recs:
            v = r["gt"] != 255
            cm += np.bincount(N_CLASSES * r["gt"][v].astype(int) + r["pred"][v], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(1 + .6 * N_CLASSES, 1 + .6 * N_CLASSES))
        ax.imshow(cmn, cmap="Blues")
        ax.set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=45, fontsize=7)
        ax.set_yticks(range(N_CLASSES), CLASS_NAMES, fontsize=7)
        for a_ in range(N_CLASSES):
            for b_ in range(N_CLASSES):
                ax.text(b_, a_, f"{cmn[a_, b_]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_xlabel("prediksi")
        ax.set_ylabel("ground truth")
        ax.set_title("Confusion matrix pixel (dinormalisasi per baris)")
        save_fig("pixel_confusion_matrix", fig)

# %% [markdown]
# # 15. Inference test + submission
#
# Prediksi = rata-rata probabilitas semua fold (+ TTA), di-resize balik ke **ukuran asli** gambar test, lalu
# post-processing (threshold/min area hasil tuning, diskalakan ke ukuran asli).
#
# Format submission:
# - **rle**: satu baris per gambar (biner) atau per gambar-per-kelas (multikelas: kolom id berisi `<id>_<kelas>` atau
#   kolom kelas terpisah — dicocokkan dengan `sample_submission`).
# - **mask_png**: mask index kelas disimpan sebagai PNG di `submissions/masks/` + ZIP.

# %%
def detect_sub_format():
    if CFG.SUBMISSION_FORMAT != "auto":
        return CFG.SUBMISSION_FORMAT
    if SAMPLE_SUB is not None and SAMPLE_SUB.shape[1] >= 2:
        return "rle"
    return "mask_png"


def predict_test():
    ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "fold*.pt")))
    if not ckpts or test_df.empty:
        return None
    loader = DataLoader(SegDataset(test_df, train=False, with_mask=False), batch_size=CFG.BATCH_SIZE, shuffle=False,
                        num_workers=CFG.NUM_WORKERS)
    acc = None
    for c in ckpts:
        model, _ = build_model()
        model.load_state_dict(torch.load(c, map_location=DEVICE))
        model.to(DEVICE)
        p = predict_probs(model, loader, CFG.TTA).astype(np.float32)
        acc = p if acc is None else acc + p
        del model
    return acc / len(ckpts)


SUB_PATH = None
if HAS_TORCH and FOLD_RESULTS and not test_df.empty:
    TEST_PROBS = predict_test()
    fmt = detect_sub_format()
    masks_out = {}
    for (_, r), p in zip(test_df.iterrows(), TEST_PROBS):
        w, h = image_size(r["img_path"])
        pt = torch.from_numpy(p[None])
        pt = F.interpolate(pt, size=(int(h), int(w)), mode="bilinear", align_corners=False)[0].numpy()
        m = probs_to_mask(pt, BEST_TH)
        ma = int(BEST_MIN_AREA * (h * w) / (CFG.IMG_SIZE ** 2))
        masks_out[r["stem"]] = remove_small(m, ma)
    if fmt == "rle":
        sub_id = CFG.SUB_ID_COL or (SAMPLE_SUB.columns[0] if SAMPLE_SUB is not None else "id")
        sub_rle = CFG.SUB_RLE_COL or (SAMPLE_SUB.columns[1] if SAMPLE_SUB is not None else "rle")
        rows = []
        if SAMPLE_SUB is not None:
            for v in SAMPLE_SUB[sub_id].astype(str):
                stem = os.path.splitext(os.path.basename(v))[0]
                cls = None
                if stem not in masks_out and "_" in stem:     # format <id>_<kelas>
                    stem, cls = stem.rsplit("_", 1)
                m = masks_out.get(stem)
                if m is None:
                    rows.append({sub_id: v, sub_rle: CFG.EMPTY_RLE})
                    continue
                if cls is not None:
                    ci = (CLASS_NAMES.index(cls) if cls in CLASS_NAMES else int(cls) if cls.isdigit() else 1)
                    mm = (m == ci)
                else:
                    mm = (m > 0)
                rle = rle_encode(mm)
                rows.append({sub_id: v, sub_rle: rle if rle else CFG.EMPTY_RLE})
        else:
            rows = [{sub_id: s, sub_rle: rle_encode(m > 0) or CFG.EMPTY_RLE} for s, m in masks_out.items()]
        sub = pd.DataFrame(rows)
        missing = (SAMPLE_SUB is not None) and sub[sub_id].astype(str).isin(set(masks_out)).mean() < 0.5 and \
            not any("_" in s for s in SAMPLE_SUB[sub_id].astype(str))
        SUB_PATH = os.path.join(SUB_DIR, "submission.csv")
        sub.to_csv(SUB_PATH, index=False)
        print(f"✅ Submission RLE: {SUB_PATH} | {len(sub)} baris | kosong: {(sub[sub_rle] == CFG.EMPTY_RLE).mean():.1%}")
        if missing:
            print("⚠️ Banyak id di sample_submission tidak cocok dengan nama file test — cek SUB_ID_COL / nama file.")
        # validator: decode ulang 1 baris → harus sama dengan mask
        st0 = next(iter(masks_out))
        m0 = masks_out[st0] > 0
        assert np.array_equal(rle_decode(rle_encode(m0), m0.shape) > 0, m0), "RLE encode/decode tidak konsisten!"
        print("✅ Validator: RLE encode→decode konsisten | kolom:", list(sub.columns))
    else:
        mdir = os.path.join(SUB_DIR, "masks")
        os.makedirs(mdir, exist_ok=True)
        for s, m in masks_out.items():
            Image.fromarray((m * (255 if BINARY else 1)).astype(np.uint8)).save(os.path.join(mdir, f"{s}.png"))
        SUB_PATH = os.path.join(SUB_DIR, "submission_masks.zip")
        with zipfile.ZipFile(SUB_PATH, "w", zipfile.ZIP_DEFLATED) as z:
            for p in glob.glob(os.path.join(mdir, "*.png")):
                z.write(p, os.path.basename(p))
        print(f"✅ Submission mask PNG: {mdir} ({len(masks_out)} file) + {SUB_PATH}")
        print("   Biner: 0/255. Multikelas: nilai pixel = index kelas. Sesuaikan dengan aturan lomba!")
    k = min(4, len(masks_out))
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3))
    for a, (s, m) in zip(np.atleast_1d(axes), list(masks_out.items())[:k]):
        a.imshow(overlay(read_image(test_df.set_index("stem").loc[s, "img_path"]), m))
        a.set_title(s[:18], fontsize=8)
        a.axis("off")
    fig.suptitle("Prediksi test (overlay)")
    save_fig("test_predictions", fig)
elif test_df.empty:
    print("ℹ️ Tidak ada gambar test → submission dilewati.")

# %% [markdown]
# # 16. Ringkasan untuk laporan
#
# Kalimat siap parafrase (English) + angka utama tersimpan di `outputs/cv_segmentation/summary.md`.

# %%
if FOLD_RESULTS:
    summ = {"model": FOLD_RESULTS[0]["name"], "img_size": CFG.IMG_SIZE, "n_train": len(train_df), "n_test": len(test_df),
            "n_classes": N_CLASSES, "folds": len(FOLD_RESULTS), "oof_dice": round(FINAL_OOF["dice"], 4),
            "oof_iou": round(FINAL_OOF["iou"], 4), "threshold": BEST_TH, "min_area": BEST_MIN_AREA, "submission": SUB_PATH}
    worst_cls = per_class.iloc[1:].sort_values("dice").head(1) if N_CLASSES > 2 else per_class.tail(1)
    text = f"""# Segmentation summary

- Model: **{summ['model']}**, input {CFG.IMG_SIZE}×{CFG.IMG_SIZE}, {summ['folds']}-fold CV, loss {CFG.LOSS}, TTA {CFG.TTA}.
- Out-of-fold Dice **{summ['oof_dice']:.4f}**, mIoU **{summ['oof_iou']:.4f}** ({'without' if not CFG.METRIC_INCLUDE_BG else 'with'} background).
- Post-processing tuned on OOF: threshold {BEST_TH}, minimum component area {BEST_MIN_AREA} px.
- Hardest class: **{worst_cls['class'].iloc[0]}** (Dice {worst_cls['dice'].iloc[0]:.3f}).

**Draft sentences (paraphrase):**
- "We trained a {summ['model']} segmentation network with a combined {'BCE' if BINARY else 'cross-entropy'} and Dice loss
  using {summ['folds']}-fold cross-validation, reaching an out-of-fold Dice of {summ['oof_dice']:.3f}."
- "Threshold and minimum-area post-processing were selected on out-of-fold predictions, never on the test set."
"""
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, default=str)
    print(text)
print("✅ Notebook selesai.")

# %% [markdown]
# # 17. Troubleshooting
#
# | Gejala | Penyebab umum | Solusi |
# |---|---|---|
# | `0 pasangan gambar–mask` | nama mask ≠ nama gambar | lihat contoh nama; tambah akhiran ke `MASK_SUFFIXES` |
# | Overlay mask bergeser/terputus-putus (RLE) | orientasi RLE salah | `RLE_ORDER = "C"` (atau sebaliknya), cek `RLE_ONE_INDEXED` |
# | N_CLASSES = 256 / aneh | mask JPEG (kompresi mengubah nilai) | `MASK_MODE="binary255"` atau set `N_CLASSES` + `VALUE_MAP` manual |
# | Dice ~0 dan prediksi kosong | kelas objek sangat kecil | `LOSS="dice"`, turunkan threshold, `TILE_SIZE`, `IMG_SIZE` lebih besar |
# | `CUDA out of memory` | gambar/batch terlalu besar | otomatis batch dikecilkan; turunkan `IMG_SIZE`, `USE_AMP=True` |
# | Sangat lambat di CPU | model besar | `ARCH="unet_small"` atau `ENCODER="resnet18"`, `IMG_SIZE=128`, `FOLDS_TO_RUN=[0]` |
# | smp gagal download weights | offline | `PRETRAINED=False` atau salin cache `~/.cache/torch` dari komputer lain |
# | Submission ditolak | format RLE/ID beda | bandingkan 3 baris `sample_submission` vs file kita; `EMPTY_RLE`, `SUB_ID_COL` |
#
# # 18. Experiment recipes (salin ke USER OVERRIDE)
#
# ```python
# # baseline kuat (GPU)
# CFG.ARCH, CFG.ENCODER, CFG.IMG_SIZE, CFG.EPOCHS = "Unet", "efficientnet-b3", 512, 30
# # objek kecil di gambar besar
# CFG.TILE_SIZE, CFG.IMG_SIZE = 512, 512
# # citra satelit / mikroskop (orientasi bebas)
# CFG.AUG_ROT90, CFG.AUG_VFLIP, CFG.TTA = True, True, ["none", "hflip", "vflip"]
# # cepat untuk cek ide
# CFG.FOLDS_TO_RUN, CFG.EPOCHS = [0], 8
# ```
#
# # 19. Checklist sebelum submit
#
# - [ ] Overlay EDA: mask pas dengan objek (orientasi RLE benar)
# - [ ] Kelas & jumlah kelas sesuai soal; background = 0
# - [ ] Metric = metric lomba (`dice` vs `iou`, dengan/tanpa background)
# - [ ] Threshold/min area dipilih dari OOF, bukan dari test
# - [ ] Submission: jumlah baris = sample_submission, ID cocok, mask kosong ditulis sesuai aturan
# - [ ] Ukuran mask submission = ukuran asli gambar test
