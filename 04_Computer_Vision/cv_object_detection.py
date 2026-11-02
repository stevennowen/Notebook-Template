# -*- coding: utf-8 -*-
# =====================================================================================================
# cv_object_detection.py — versi SCRIPT dari cv_object_detection.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python cv_object_detection.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb cv_object_detection.py
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
# # Computer Vision Competition Template — Object Detection & Counting
#
# Template untuk lomba yang meminta **kotak (bounding box) + kelas** per objek, atau **jumlah objek** per gambar
# (counting: ayam, koin, kerumunan, kendaraan, sel). Contoh: Hology 8 (overcrowding), Chicken/Coin/Dice Counting,
# deteksi pothole/kapal. Bisa dijalankan dari atas ke bawah di CPU maupun GPU.
#
# ---
#
# ## Pipeline besar
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json)
# → Install / cek library (ultralytics opsional; fallback torchvision Faster R-CNN)
# → Baca anotasi format apa pun: YOLO txt / COCO json / Pascal VOC xml / CSV (xyxy, xywh, bbox string)
# → Tabel anotasi seragam (pixel xyxy) + sanity check (box keluar gambar, ukuran nol, kelas aneh)
# → EDA: objek per gambar, ukuran & rasio box, kelas, contoh gambar + box
# → Split train/val (stratified jumlah objek) → dataset YOLO otomatis (data.yaml)
# → Training: YOLO (ultralytics) ATAU Faster R-CNN (torchvision)
# → Evaluasi sendiri di val: mAP@0.5, mAP@0.5:0.95, precision/recall, MAE jumlah objek
# → Tuning confidence threshold (F1 untuk deteksi, MAE untuk counting) + kalibrasi count
# → Prediksi test (+TTA) → submission: box CSV / PredictionString / count / COCO json + validator
# → Ringkasan laporan, troubleshooting, recipes, checklist
# ```
#
# ## Cara pakai cepat
#
# 1. Taruh data di `./data/` (format anotasi di bawah dideteksi otomatis).
# 2. Isi `# >>> USER OVERRIDE`: path, `TASK_MODE` (`"detect"` atau `"count"`), format submission.
# 3. `RUN_MODE="fast"` → Run All → **lihat figure contoh box** (box harus pas dengan objek!).
# 4. `RUN_MODE="full"` → Run All.
# 5. Submission di `outputs/cv_object_detection/submissions/`.
#
# **Dua format:** `cv_object_detection.ipynb` dan `cv_object_detection.py` (isi sama; script bisa dijalankan
# `python cv_object_detection.py` atau per `# %%` di VS Code). Headless: `python tools/run_task.py --task detection --format py`.
#
# ## Format anotasi yang dikenali (`DATA_FORMAT="auto"`)
#
# | Format | Bentuk | Koordinat |
# |---|---|---|
# | `yolo` | `labels/<nama>.txt` per gambar: `cls cx cy w h` | ternormalisasi 0–1 |
# | `coco` | satu `.json` berisi `images`, `annotations`, `categories` | `bbox = [x, y, w, h]` pixel |
# | `voc` | `.xml` per gambar (`<object><name>`, `<bndbox>`) | `xmin ymin xmax ymax` pixel |
# | `csv` | satu baris per box: kolom gambar + `xmin,ymin,xmax,ymax` / `x,y,w,h` / `bbox="[x,y,w,h]"` (+ label) | pixel (atau 0–1, dideteksi) |
#
# Untuk lomba **counting tanpa box** (hanya `train.csv` berisi jumlah): pakai `cv_image_classification` dengan target
# angka (regresi) — lihat Recipes.

# %% [markdown]
# # 1. Config (`CFG`)
#
# **Jika** CPU-only **maka** `YOLO_MODEL="yolo11n.pt"`, `IMG_SIZE=416–512`, `EPOCHS=30`.
# **Jika** objek sangat kecil (kerumunan, sel) **maka** `IMG_SIZE=960–1280` dan model `s`/`m` di GPU.
# **Jika** lomba counting **maka** `TASK_MODE="count"` → threshold dipilih untuk MAE jumlah, bukan mAP.

# %%
import os


class CFG:
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "cv_object_detection"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"  # pip install ultralytics kalau belum ada (TIDAK pernah torch)
    TASK_MODE = "detect"         # "detect" (box + kelas) | "count" (jumlah objek per gambar)

    # ---------------- data
    DATA_FORMAT = "auto"         # "auto" | "yolo" | "coco" | "voc" | "csv"
    TRAIN_IMAGE_DIR = "./data/train/images"
    TRAIN_LABEL_DIR = "./data/train/labels"      # yolo txt / voc xml
    ANNOTATION_PATH = None       # coco .json atau csv (None = cari otomatis di ./data)
    TEST_IMAGE_DIR = "./data/test/images"
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    IMAGE_DIR_CANDIDATES = ["./data/train/images", "./data/train", "./data/images", "./data/train_images"]
    LABEL_DIR_CANDIDATES = ["./data/train/labels", "./data/labels", "./data/annotations", "./data/train/annotations", "./data/Annotations"]
    TEST_DIR_CANDIDATES = ["./data/test/images", "./data/test", "./data/test_images"]
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    CLASS_NAMES = None           # None = dari classes.txt / categories COCO / label unik CSV
    CSV_IMAGE_COL = None         # csv: kolom nama gambar (None = auto)
    CSV_LABEL_COL = None         # csv: kolom kelas (None = auto; tidak ada = 1 kelas)
    CSV_BOX_FORMAT = "auto"      # "auto" | "xyxy" | "xywh"
    OUTPUT_DIR = "./outputs/cv_object_detection/"

    # ---------------- model
    BACKEND = "auto"             # "auto" | "ultralytics" | "torchvision"
    YOLO_MODEL = "yolo11n.pt"    # yolo11n/s/m/l/x.pt (pretrained COCO) — n paling ringan
    PRETRAINED = True            # False = dari nol (tanpa download weights; jauh lebih lemah)
    IMG_SIZE = 640
    EPOCHS = 60
    BATCH_SIZE = 16
    PATIENCE = 15
    LR0 = None                   # None = default ultralytics
    AMP = True
    USE_GPU = True
    NUM_WORKERS = 0 if os.name == "nt" else 4    # Windows: 0 (mode script aman)
    VAL_FRAC = 0.2
    AUG_MOSAIC = 1.0             # 0 kalau gambar sudah padat / objek di tepi penting
    AUG_FLIPUD = 0.0             # 0.5 untuk citra udara/satelit/mikroskop
    TV_MODEL = "fasterrcnn_mobilenet_v3_large_fpn"   # fallback torchvision (atau fasterrcnn_resnet50_fpn_v2)
    TV_LR = 1e-4

    # ---------------- inference & submission
    CONF_THRESHOLD = None        # None = di-tune di val (F1 untuk detect, MAE untuk count)
    NMS_IOU = 0.6
    MAX_DET = 300
    TTA = False                  # ultralytics augment=True saat predict
    COUNT_CALIBRATION = True     # count: kalibrasi linear dari fitur deteksi (dicek dengan CV di val)
    SUBMISSION_FORMAT = "auto"   # "auto" | "boxes_csv" | "prediction_string" | "count" | "coco_json"
    SUB_BOX_FORMAT = "xyxy"      # boxes_csv: "xyxy" | "xywh" (pixel)
    SUB_ID_COL = None

    # ---------------- fast mode
    FAST_SAMPLE = 150
    FAST_EPOCHS = 3
    FAST_IMG_SIZE = 320
    FIG_DPI = 150

# %% [markdown]
# # 2. USER OVERRIDE

# %%
# >>> USER OVERRIDE
# CFG.RUN_MODE = "fast"
# CFG.TASK_MODE = "count"
# CFG.YOLO_MODEL = "yolo11s.pt"; CFG.IMG_SIZE = 960
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
# # 3. Install / cek library
#
# `ultralytics` (YOLO) di-install kalau belum ada. Torch tidak pernah di-install otomatis. Tanpa ultralytics →
# fallback Faster R-CNN dari torchvision (lebih lambat, tetap valid).

# %%
import importlib
import importlib.util
import subprocess
import sys


def ensure_packages(packages, auto_install=True):
    status = {}
    for imp, pip_name in packages.items():
        if importlib.util.find_spec(imp) is not None:
            status[imp] = "ok"
            continue
        if not auto_install:
            status[imp] = "missing"
            continue
        try:
            print("Installing", pip_name, "...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            importlib.invalidate_caches()
            status[imp] = "installed" if importlib.util.find_spec(imp) else "failed"
        except Exception as e:  # noqa
            status[imp] = f"failed ({type(e).__name__})"
    for k, v in status.items():
        print(f"  {k:14s}: {v}")
    return status


PKG_STATUS = ensure_packages({"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "PIL": "pillow",
                              "matplotlib": "matplotlib", "yaml": "pyyaml"}, CFG.AUTO_INSTALL)
if importlib.util.find_spec("torch") is None:
    print("❌ PyTorch belum ada → https://pytorch.org/get-started/locally/ (deteksi butuh torch)")
elif CFG.BACKEND in ("auto", "ultralytics"):
    PKG_STATUS.update(ensure_packages({"ultralytics": "ultralytics"}, CFG.AUTO_INSTALL))

# %% [markdown]
# # 4. Import, flag, device, helper

# %%
import gc
import glob
import json
import math
import random
import shutil
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    from IPython.display import display
except Exception:  # noqa
    display = print

try:
    import torch
    HAS_TORCH = True
except Exception:  # noqa
    torch, HAS_TORCH = None, False
HAS_ULTRA = HAS_TV = False
if HAS_TORCH:
    try:
        from ultralytics import YOLO
        HAS_ULTRA = True
    except Exception as e:  # noqa
        if importlib.util.find_spec("ultralytics"):
            print("ultralytics gagal di-import:", str(e)[:150])
    try:
        import torchvision
        from torchvision.ops import box_iou as tv_box_iou
        HAS_TV = True
    except Exception:  # noqa
        pass

if CFG.BACKEND == "auto":
    BACKEND = "ultralytics" if HAS_ULTRA else ("torchvision" if HAS_TV else None)
else:
    BACKEND = CFG.BACKEND if {"ultralytics": HAS_ULTRA, "torchvision": HAS_TV}.get(CFG.BACKEND) else (
        "torchvision" if HAS_TV else None)
DEVICE = "cuda" if (HAS_TORCH and CFG.USE_GPU and torch.cuda.is_available()) else "cpu"
print(f"torch: {torch.__version__ if HAS_TORCH else '-'} | ultralytics: {HAS_ULTRA} | torchvision: {HAS_TV}")
print(f"Backend: {BACKEND} | Device: {DEVICE}")
if BACKEND is None:
    print("❌ Tidak ada backend deteksi (butuh torch + ultralytics/torchvision). EDA tetap jalan.")

if CFG.RUN_MODE == "fast":
    CFG.EPOCHS = min(CFG.EPOCHS, CFG.FAST_EPOCHS)
    CFG.IMG_SIZE = min(CFG.IMG_SIZE, CFG.FAST_IMG_SIZE)
    print(f"[FAST MODE] epochs={CFG.EPOCHS} img={CFG.IMG_SIZE} sample≤{CFG.FAST_SAMPLE}")
OUT_DIR = os.path.abspath(CFG.OUTPUT_DIR)
FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR = (os.path.join(OUT_DIR, d) for d in ("figures", "tables", "models", "submissions"))
for _d in (OUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR):
    os.makedirs(_d, exist_ok=True)
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
if HAS_TORCH:
    torch.manual_seed(CFG.SEED)
_FIG_COUNTER = [0]


def save_fig(name, fig=None, show=True):
    fig = fig if fig is not None else plt.gcf()
    _FIG_COUNTER[0] += 1
    path = os.path.join(FIG_DIR, f"fig{_FIG_COUNTER[0]:02d}_{name}.png")
    fig.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight")
    print("  [fig] saved:", path)
    if show and "agg" not in matplotlib.get_backend().lower():
        plt.show()
    plt.close(fig)
    return path


def save_table(df, name, index=False):
    path = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(path, index=index)
    print("  [table] saved:", path)
    return path


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


def list_images(folder):
    out = []
    if folder and os.path.isdir(folder):
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(tuple(CFG.IMAGE_EXTS))]
    return out


def first_dir(primary, cands, need_images=True):
    for d in [primary] + list(cands):
        if d and os.path.isdir(d) and (list_images(d) if need_images else os.listdir(d)):
            return d
    return None


def img_size(p):
    with Image.open(p) as im:
        return im.size

# %% [markdown]
# # 5. Baca anotasi → tabel seragam
#
# Hasil: `IMAGES` (satu baris per gambar train: path, lebar, tinggi) dan `ANN` (satu baris per box:
# `stem, cls, x1, y1, x2, y2` dalam **pixel**). Gambar tanpa box tetap ada di `IMAGES` (contoh negatif).
#
# **Jika X maka Y:**
# - Format salah terdeteksi → set `CFG.DATA_FORMAT`.
# - Box di figure contoh tidak pas (bergeser/terlalu besar) → koordinat CSV mungkin `xywh` bukan `xyxy`
#   (`CSV_BOX_FORMAT`), atau ternormalisasi.
# - `box tidak valid` banyak → anotasi rusak; baris itu dibuang otomatis (dilaporkan).

# %%
TRAIN_IMG_DIR = first_dir(CFG.TRAIN_IMAGE_DIR, CFG.IMAGE_DIR_CANDIDATES)
TEST_IMG_DIR = first_dir(CFG.TEST_IMAGE_DIR, CFG.TEST_DIR_CANDIDATES)
LABEL_DIR = first_dir(CFG.TRAIN_LABEL_DIR, CFG.LABEL_DIR_CANDIDATES, need_images=False)
if TRAIN_IMG_DIR is None:
    raise FileNotFoundError("Folder gambar train tidak ditemukan → set CFG.TRAIN_IMAGE_DIR")
_img_paths = list_images(TRAIN_IMG_DIR)
IMG_BY_STEM = {os.path.splitext(os.path.basename(p))[0]: p for p in _img_paths}
IMG_BY_NAME = {os.path.basename(p): p for p in _img_paths}


def find_annotation_file():
    if CFG.ANNOTATION_PATH and os.path.exists(CFG.ANNOTATION_PATH):
        return CFG.ANNOTATION_PATH
    cands = sorted(glob.glob("./data/**/*.json", recursive=True)) + sorted(glob.glob("./data/*.csv"))
    cands = [c for c in cands if "sample_sub" not in c.lower() and "test" not in os.path.basename(c).lower()
             and not os.path.basename(c).startswith("_")]
    return cands[0] if cands else None


def detect_format():
    if CFG.DATA_FORMAT != "auto":
        return CFG.DATA_FORMAT
    if LABEL_DIR:
        files = os.listdir(LABEL_DIR)
        if any(f.endswith(".txt") and f != "classes.txt" for f in files):
            return "yolo"
        if any(f.endswith(".xml") for f in files):
            return "voc"
    f = find_annotation_file()
    if f and f.endswith(".json"):
        return "coco"
    if f and f.endswith((".csv", ".xlsx")):
        return "csv"
    raise FileNotFoundError("Anotasi tidak ditemukan (labels/*.txt, *.xml, *.json, atau train.csv).")


def read_class_names_file():
    for p in ["./data/classes.txt", os.path.join(LABEL_DIR or "", "classes.txt"), "./data/obj.names"]:
        if p and os.path.exists(p):
            return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    for p in glob.glob("./data/**/*.yaml", recursive=True):
        try:
            import yaml
            y = yaml.safe_load(open(p, encoding="utf-8"))
            n = y.get("names") if isinstance(y, dict) else None
            if n:
                return list(n.values()) if isinstance(n, dict) else list(n)
        except Exception:  # noqa
            pass
    return None


def load_yolo():
    rows = []
    for stem, p in IMG_BY_STEM.items():
        lp = os.path.join(LABEL_DIR, stem + ".txt")
        if not os.path.exists(lp):
            continue
        w, h = img_size(p)
        for line in open(lp, encoding="utf-8"):
            t = line.split()
            if len(t) >= 5:
                c, cx, cy, bw, bh = int(float(t[0])), *map(float, t[1:5])
                rows.append({"stem": stem, "cls": c, "x1": (cx - bw / 2) * w, "y1": (cy - bh / 2) * h,
                             "x2": (cx + bw / 2) * w, "y2": (cy + bh / 2) * h})
    return pd.DataFrame(rows), read_class_names_file()


def load_coco(path):
    d = json.load(open(path, encoding="utf-8"))
    cats = sorted(d.get("categories", []), key=lambda c: c["id"])
    cat_idx = {c["id"]: i for i, c in enumerate(cats)}
    imgs = {im["id"]: os.path.splitext(os.path.basename(im["file_name"]))[0] for im in d["images"]}
    rows = []
    for a in d.get("annotations", []):
        x, y, w, h = a["bbox"]
        rows.append({"stem": imgs.get(a["image_id"]), "cls": cat_idx.get(a["category_id"], 0),
                     "x1": x, "y1": y, "x2": x + w, "y2": y + h})
    return pd.DataFrame(rows), [c["name"] for c in cats] or None


def load_voc():
    rows, names = [], []
    for xp in sorted(glob.glob(os.path.join(LABEL_DIR, "**", "*.xml"), recursive=True)):
        root = ET.parse(xp).getroot()
        fn = root.findtext("filename") or os.path.basename(xp)
        stem = os.path.splitext(os.path.basename(fn))[0]
        if stem not in IMG_BY_STEM:
            stem = os.path.splitext(os.path.basename(xp))[0]
        for o in root.iter("object"):
            n = o.findtext("name", "object").strip()
            if n not in names:
                names.append(n)
            b = o.find("bndbox")
            rows.append({"stem": stem, "cls": names.index(n), "x1": float(b.findtext("xmin")), "y1": float(b.findtext("ymin")),
                         "x2": float(b.findtext("xmax")), "y2": float(b.findtext("ymax"))})
    return pd.DataFrame(rows), names


def load_csv(path):
    df = smart_read(path)
    low = {c.lower(): c for c in df.columns}
    img_col = CFG.CSV_IMAGE_COL or next((low[c] for c in ["image", "image_id", "image_name", "filename", "file_name",
                                                         "img", "name", "path", "id"] if c in low), df.columns[0])
    lab_col = CFG.CSV_LABEL_COL or next((low[c] for c in ["label", "class", "class_name", "category", "cls", "class_id"]
                                         if c in low), None)
    get = lambda *names: next((low[n] for n in names if n in low), None)  # noqa: E731
    fmt, cols = CFG.CSV_BOX_FORMAT, None
    if "bbox" in low:
        vals = df[low["bbox"]].astype(str).str.findall(r"[-\d.]+").map(lambda v: [float(x) for x in v[:4]])
        df[["_a", "_b", "_c", "_d"]] = pd.DataFrame(vals.tolist(), index=df.index)
        cols, fmt = ["_a", "_b", "_c", "_d"], ("xywh" if fmt == "auto" else fmt)
    elif get("xmin", "x_min", "x1", "left") and get("xmax", "x_max", "x2", "right"):
        cols = [get("xmin", "x_min", "x1", "left"), get("ymin", "y_min", "y1", "top"),
                get("xmax", "x_max", "x2", "right"), get("ymax", "y_max", "y2", "bottom")]
        fmt = "xyxy" if fmt == "auto" else fmt
    elif get("x", "x_center", "cx") and get("w", "width", "bw"):
        cols = [get("x", "x_center", "cx"), get("y", "y_center", "cy"), get("w", "width", "bw"), get("h", "height", "bh")]
        fmt = "xywh" if fmt == "auto" else fmt
    if cols is None:
        raise ValueError(f"Kolom box tidak dikenali di {path}: {list(df.columns)}")
    names = sorted(df[lab_col].dropna().astype(str).unique().tolist()) if lab_col else ["object"]
    rows = []
    for _, r in df.iterrows():
        a, b, c, d = (float(r[k]) for k in cols)
        if any(np.isnan([a, b, c, d])):
            continue
        stem = os.path.splitext(os.path.basename(str(r[img_col])))[0]
        x1, y1, x2, y2 = (a, b, a + c, b + d) if fmt == "xywh" else (a, b, c, d)
        if max(x1, y1, x2, y2) <= 1.0 and stem in IMG_BY_STEM:      # ternormalisasi
            w, h = img_size(IMG_BY_STEM[stem])
            x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
        rows.append({"stem": stem, "cls": names.index(str(r[lab_col])) if lab_col else 0,
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return pd.DataFrame(rows), names


DATA_FORMAT = detect_format()
if DATA_FORMAT == "yolo":
    ANN, names_found = load_yolo()
elif DATA_FORMAT == "coco":
    ANN, names_found = load_coco(find_annotation_file())
elif DATA_FORMAT == "voc":
    ANN, names_found = load_voc()
else:
    ANN, names_found = load_csv(find_annotation_file())
print(f"Format anotasi: {DATA_FORMAT} | box: {len(ANN)} | label dir: {LABEL_DIR}")

IMAGES = pd.DataFrame({"stem": list(IMG_BY_STEM), "path": list(IMG_BY_STEM.values())})
if DATA_FORMAT in ("coco", "csv"):
    labelled = set(ANN["stem"])
    IMAGES = IMAGES[IMAGES["stem"].isin(labelled) | (len(labelled) == 0)]
wh = IMAGES["path"].map(img_size)
IMAGES["W"], IMAGES["H"] = wh.map(lambda s: s[0]), wh.map(lambda s: s[1])
IMAGES = IMAGES.reset_index(drop=True)

# sanity check box
ANN = ANN[ANN["stem"].isin(set(IMAGES["stem"]))].merge(IMAGES[["stem", "W", "H"]], on="stem")
before = len(ANN)
ANN["x1"], ANN["x2"] = ANN["x1"].clip(0, ANN["W"]), ANN["x2"].clip(0, ANN["W"])
ANN["y1"], ANN["y2"] = ANN["y1"].clip(0, ANN["H"]), ANN["y2"].clip(0, ANN["H"])
ANN = ANN[(ANN["x2"] - ANN["x1"] > 1) & (ANN["y2"] - ANN["y1"] > 1)].reset_index(drop=True)
print(f"Box tidak valid (ukuran ≤1px setelah di-clip) dibuang: {before - len(ANN)}")
N_CLS = int(max(ANN["cls"].max() + 1 if len(ANN) else 1, len(CFG.CLASS_NAMES or names_found or [])))
CLASS_NAMES = list(CFG.CLASS_NAMES or names_found or [f"class_{i}" for i in range(N_CLS)])[:N_CLS]
CLASS_NAMES += [f"class_{i}" for i in range(len(CLASS_NAMES), N_CLS)]
counts = ANN.groupby("stem").size()
IMAGES["n_obj"] = IMAGES["stem"].map(counts).fillna(0).astype(int)
print(f"Gambar train: {len(IMAGES)} | kelas ({N_CLS}): {CLASS_NAMES} | gambar tanpa objek: {(IMAGES.n_obj == 0).sum()}")
TEST = pd.DataFrame({"path": list_images(TEST_IMG_DIR)}) if TEST_IMG_DIR else pd.DataFrame(columns=["path"])
TEST["stem"] = TEST["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
SAMPLE_SUB = smart_read(CFG.SAMPLE_SUB_PATH) if os.path.exists(CFG.SAMPLE_SUB_PATH) else None
print(f"Gambar test: {len(TEST)} | sample_submission: {None if SAMPLE_SUB is None else list(SAMPLE_SUB.columns)}")

# %% [markdown]
# # 6. EDA
#
# **Cara membaca:**
# - *Objek per gambar* lebar (1–200) → counting padat: `MAX_DET` ≥ maksimum, `IMG_SIZE` besar.
# - *Ukuran box relatif* < 1% sisi gambar → objek kecil: naikkan `IMG_SIZE`, matikan mosaic kalau objek terpotong.
# - *Contoh gambar*: box harus pas di objek. Kalau tidak → konversi koordinat salah (lihat section 5).

# %%
ANN["bw"], ANN["bh"] = ANN["x2"] - ANN["x1"], ANN["y2"] - ANN["y1"]
ANN["rel_size"] = np.sqrt(ANN["bw"] * ANN["bh"] / (ANN["W"] * ANN["H"]))
summary_eda = {"images": len(IMAGES), "boxes": len(ANN), "obj_per_img_mean": IMAGES.n_obj.mean(),
               "obj_per_img_max": int(IMAGES.n_obj.max()), "rel_size_median": ANN["rel_size"].median()}
print({k: round(v, 3) if isinstance(v, float) else v for k, v in summary_eda.items()})
cls_counts = ANN["cls"].map(dict(enumerate(CLASS_NAMES))).value_counts()
display(cls_counts.to_frame("boxes"))
save_table(cls_counts.rename_axis("class").reset_index(name="boxes"), "class_box_counts")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].hist(IMAGES["n_obj"], bins=min(40, IMAGES.n_obj.max() + 1), color="#4C72B0")
ax[0].set_title("Objek per gambar")
ax[1].hist(ANN["rel_size"], bins=40, color="#55A868")
ax[1].set_title("Ukuran box relatif (√area box / area gambar)")
ax[2].bar(cls_counts.index.astype(str), cls_counts.values, color="#C44E52")
ax[2].set_title("Box per kelas")
ax[2].tick_params(axis="x", rotation=30)
save_fig("eda_boxes", fig)


def draw_boxes(ax, path, boxes, title="", color_by_cls=True, scores=None):
    ax.imshow(Image.open(path).convert("RGB"))
    for j, b in enumerate(boxes.itertuples() if isinstance(boxes, pd.DataFrame) else boxes):
        x1, y1, x2, y2, c = (b.x1, b.y1, b.x2, b.y2, b.cls) if hasattr(b, "x1") else b[:5]
        col = plt.get_cmap("tab10")(int(c) % 10)
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.3, color=col))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


k = min(6, len(IMAGES))
fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.2))
for a, (_, r) in zip(np.atleast_1d(axes), IMAGES.sample(k, random_state=1).iterrows()):
    draw_boxes(a, r["path"], ANN[ANN.stem == r["stem"]], f"{r['stem'][:16]} ({r['n_obj']} obj)")
fig.suptitle("Contoh anotasi (cek: box pas dengan objek)")
save_fig("sample_annotations", fig)

# %% [markdown]
# # 7. Split train/val + dataset YOLO
#
# Deteksi memakai **holdout** (K-fold YOLO mahal). Split distratifikasi menurut jumlah objek (kuantil) supaya val
# mewakili gambar sepi & padat. Dataset YOLO ditulis ke `outputs/.../yolo_dataset/` (gambar di-copy, label dari `ANN`).

# %%
if CFG.RUN_MODE == "fast" and len(IMAGES) > CFG.FAST_SAMPLE:
    IMAGES = IMAGES.sample(CFG.FAST_SAMPLE, random_state=CFG.SEED).reset_index(drop=True)
    ANN = ANN[ANN.stem.isin(set(IMAGES.stem))].reset_index(drop=True)
    print(f"[FAST MODE] {len(IMAGES)} gambar")
bins = pd.qcut(IMAGES["n_obj"].rank(method="first"), q=min(5, max(2, len(IMAGES) // 10)), labels=False)
tr_idx, va_idx = train_test_split(np.arange(len(IMAGES)), test_size=CFG.VAL_FRAC, random_state=CFG.SEED,
                                  stratify=bins if bins.value_counts().min() >= 2 else None)
IMAGES["split"] = "train"
IMAGES.loc[va_idx, "split"] = "val"
print(IMAGES["split"].value_counts().to_dict())

YOLO_DIR = os.path.join(OUT_DIR, "yolo_dataset")


def write_yolo_dataset():
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            d = os.path.join(YOLO_DIR, sub, sp)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
    g = ANN.groupby("stem")
    for _, r in IMAGES.iterrows():
        ext = os.path.splitext(r["path"])[1]
        shutil.copyfile(r["path"], os.path.join(YOLO_DIR, "images", r["split"], r["stem"] + ext))
        lines = []
        if r["stem"] in g.groups:
            for b in g.get_group(r["stem"]).itertuples():
                cx, cy = (b.x1 + b.x2) / 2 / r["W"], (b.y1 + b.y2) / 2 / r["H"]
                lines.append(f"{int(b.cls)} {cx:.6f} {cy:.6f} {(b.x2 - b.x1) / r['W']:.6f} {(b.y2 - b.y1) / r['H']:.6f}")
        with open(os.path.join(YOLO_DIR, "labels", r["split"], r["stem"] + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    yml = os.path.join(YOLO_DIR, "data.yaml")
    with open(yml, "w", encoding="utf-8") as f:
        f.write(f"path: {YOLO_DIR.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {N_CLS}\nnames: {json.dumps(CLASS_NAMES)}\n")
    return yml


DATA_YAML = write_yolo_dataset()
print("data.yaml:", DATA_YAML)

# %% [markdown]
# # 8. Training
#
# **ultralytics**: `model.train(...)` — log per epoch (box_loss, cls_loss, mAP val) tercetak; hasil di
# `outputs/.../yolo_runs/train/` (results.csv, confusion matrix, PR curve). Model terbaik: `weights/best.pt`.
#
# **torchvision**: Faster R-CNN, loop training sederhana (AdamW), model terbaik menurut mAP@0.5 val.
#
# **Jika** `mAP` val tetap 0 setelah beberapa epoch **maka** cek figure anotasi (box salah) dan jumlah epoch
# (training dari nol butuh jauh lebih banyak epoch daripada pretrained).

# %%
def yolo_weights_name():
    m = CFG.YOLO_MODEL
    if not CFG.PRETRAINED and m.endswith(".pt"):
        return m[:-3] + ".yaml"
    return m


TRAINED = {}
if BACKEND == "ultralytics":
    t0 = time.time()
    model = YOLO(yolo_weights_name())
    kw = dict(data=DATA_YAML, epochs=CFG.EPOCHS, imgsz=CFG.IMG_SIZE, batch=CFG.BATCH_SIZE, patience=CFG.PATIENCE,
              device=0 if DEVICE == "cuda" else "cpu", workers=CFG.NUM_WORKERS, project=os.path.join(OUT_DIR, "yolo_runs"),
              name="train", exist_ok=True, seed=CFG.SEED, amp=CFG.AMP and DEVICE == "cuda", verbose=False,
              mosaic=CFG.AUG_MOSAIC, flipud=CFG.AUG_FLIPUD, pretrained=CFG.PRETRAINED, plots=True)
    if CFG.LR0:
        kw["lr0"] = CFG.LR0
    try:
        model.train(**kw)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️ OOM → batch dikecilkan ke", max(1, CFG.BATCH_SIZE // 2))
            torch.cuda.empty_cache()
            kw["batch"] = max(1, CFG.BATCH_SIZE // 2)
            model = YOLO(yolo_weights_name())
            model.train(**kw)
        else:
            raise
    run_dir = os.path.join(OUT_DIR, "yolo_runs", "train")
    best = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best):
        best = os.path.join(run_dir, "weights", "last.pt")
    shutil.copyfile(best, os.path.join(MODEL_DIR, "best.pt"))
    TRAINED = {"backend": "ultralytics", "weights": os.path.join(MODEL_DIR, "best.pt"), "seconds": time.time() - t0,
               "name": f"YOLO {CFG.YOLO_MODEL}{'' if CFG.PRETRAINED else ' (scratch)'}"}
    res_csv = os.path.join(run_dir, "results.csv")
    if os.path.exists(res_csv):
        res = pd.read_csv(res_csv)
        res.columns = [c.strip() for c in res.columns]
        save_table(res, "yolo_training_history")
        mcols = [c for c in res.columns if "mAP50" in c]
        if mcols:
            fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
            for c in [c for c in res.columns if c.startswith("train/")]:
                ax[0].plot(res["epoch"], res[c], label=c.split("/")[1])
            ax[0].legend(fontsize=7)
            ax[0].set_title("Train loss")
            for c in mcols:
                ax[1].plot(res["epoch"], res[c], label=c.split("/")[-1])
            ax[1].legend(fontsize=7)
            ax[1].set_title("Validation mAP")
            save_fig("learning_curves", fig)

elif BACKEND == "torchvision":
    from torchvision.models import detection as tvd
    import torchvision.transforms.functional as TF

    def build_tv():
        fn = getattr(tvd, CFG.TV_MODEL)
        if CFG.PRETRAINED:
            try:
                m = fn(weights="DEFAULT")
                in_f = m.roi_heads.box_predictor.cls_score.in_features
                m.roi_heads.box_predictor = tvd.faster_rcnn.FastRCNNPredictor(in_f, N_CLS + 1)
                return m
            except Exception as e:  # noqa
                print("⚠️ weights pretrained gagal:", str(e)[:120], "→ dari nol")
        return fn(weights=None, weights_backbone=None, num_classes=N_CLS + 1)

    def tv_batches(df, bs, shuffle):
        idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
        g = ANN.groupby("stem")
        for s in range(0, len(idx), bs):
            imgs, tg = [], []
            for i in idx[s:s + bs]:
                r = df.iloc[i]
                im = Image.open(r["path"]).convert("RGB")
                b = g.get_group(r["stem"])[["x1", "y1", "x2", "y2"]].values if r["stem"] in g.groups else np.zeros((0, 4))
                c = g.get_group(r["stem"])["cls"].values + 1 if r["stem"] in g.groups else np.zeros(0)
                t = TF.to_tensor(im)
                if shuffle and random.random() < .5:
                    t = t.flip(-1)
                    b = b.copy()
                    b[:, [0, 2]] = r["W"] - b[:, [2, 0]]
                imgs.append(t.to(DEVICE))
                tg.append({"boxes": torch.tensor(b, dtype=torch.float32, device=DEVICE).reshape(-1, 4),
                           "labels": torch.tensor(c, dtype=torch.int64, device=DEVICE)})
            yield imgs, tg

    t0 = time.time()
    tvm = build_tv().to(DEVICE)
    opt = torch.optim.AdamW([p for p in tvm.parameters() if p.requires_grad], lr=CFG.TV_LR, weight_decay=1e-4)
    trd = IMAGES[IMAGES.split == "train"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, CFG.EPOCHS))
    for ep in range(CFG.EPOCHS):
        tvm.train()
        losses = []
        for imgs, tg in tv_batches(trd, max(1, CFG.BATCH_SIZE // 4), True):
            loss = sum(tvm(imgs, tg).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tvm.parameters(), 10.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        print(f"  ep {ep + 1:02d}/{CFG.EPOCHS} loss {np.mean(losses):.4f}")
    torch.save(tvm.state_dict(), os.path.join(MODEL_DIR, "fasterrcnn_last.pt"))
    TRAINED = {"backend": "torchvision", "model": tvm, "seconds": time.time() - t0,
               "name": f"torchvision {CFG.TV_MODEL}{'' if CFG.PRETRAINED else ' (scratch)'}"}
print("Model terlatih:", TRAINED.get("name"), f"| {TRAINED.get('seconds', 0):.0f}s" if TRAINED else "")

# %% [markdown]
# # 9. Evaluasi di val (implementasi sendiri — sama untuk semua backend)
#
# - **mAP@0.5** dan **mAP@0.5:0.95** (rata-rata 10 threshold IoU, gaya COCO), per kelas.
# - **Precision/Recall/F1** pada threshold confidence tertentu → kurva F1 vs confidence.
# - **Counting**: MAE & RMSE jumlah objek per gambar vs threshold.
#
# Prediksi val dibuat dengan confidence sangat rendah (0.001) supaya bisa di-tune setelahnya.

# %%
def predict_images(paths, conf=0.001):
    """Return dict stem → array (n, 6): x1, y1, x2, y2, score, cls."""
    out = {}
    if not TRAINED or not paths:
        return out
    if TRAINED["backend"] == "ultralytics":
        m = YOLO(TRAINED["weights"])
        for p, r in zip(paths, m.predict(source=paths, imgsz=CFG.IMG_SIZE, conf=conf, iou=CFG.NMS_IOU, max_det=CFG.MAX_DET,
                                         augment=CFG.TTA, device=0 if DEVICE == "cuda" else "cpu", verbose=False, stream=True)):
            b = r.boxes
            arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1) \
                if len(b) else np.zeros((0, 6))
            out[os.path.splitext(os.path.basename(p))[0]] = arr
    else:
        import torchvision.transforms.functional as TF
        m = TRAINED["model"].eval()
        m.roi_heads.score_thresh, m.roi_heads.detections_per_img = conf, CFG.MAX_DET
        with torch.no_grad():
            for p in paths:
                o = m([TF.to_tensor(Image.open(p).convert("RGB")).to(DEVICE)])[0]
                arr = np.concatenate([o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()[:, None],
                                      (o["labels"].cpu().numpy() - 1)[:, None]], 1) if len(o["boxes"]) else np.zeros((0, 6))
                out[os.path.splitext(os.path.basename(p))[0]] = arr
    return out


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def average_precision(preds, gts, cls, iou_th):
    """preds/gts: dict stem → array. AP (all-point interpolation) satu kelas."""
    scores, tps, n_gt = [], [], 0
    for stem, g in gts.items():
        gc_ = g[g[:, 4] == cls][:, :4] if len(g) else np.zeros((0, 4))
        n_gt += len(gc_)
        p = preds.get(stem, np.zeros((0, 6)))
        p = p[p[:, 5] == cls]
        p = p[np.argsort(-p[:, 4])]
        used = np.zeros(len(gc_), bool)
        ious = iou_matrix(p[:, :4], gc_)
        for i in range(len(p)):
            j = int(np.argmax(ious[i])) if len(gc_) else -1
            ok = j >= 0 and ious[i, j] >= iou_th and not used[j]
            if ok:
                used[j] = True
            scores.append(p[i, 4])
            tps.append(1 if ok else 0)
    if n_gt == 0:
        return np.nan
    if not scores:
        return 0.0
    o = np.argsort(-np.array(scores))
    tp = np.cumsum(np.array(tps)[o])
    fp = np.cumsum(1 - np.array(tps)[o])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate(preds, gts):
    ap50 = [average_precision(preds, gts, c, 0.5) for c in range(N_CLS)]
    ap_all = [np.nanmean([average_precision(preds, gts, c, t) for t in np.arange(0.5, 0.96, 0.05)]) for c in range(N_CLS)]
    return {"mAP50": float(np.nanmean(ap50)), "mAP50_95": float(np.nanmean(ap_all)), "ap50_per_class": ap50, "ap_per_class": ap_all}


def prf_at(preds, gts, conf, iou_th=0.5):
    tp = fp = fn = 0
    for stem, g in gts.items():
        p = preds.get(stem, np.zeros((0, 6)))
        p = p[p[:, 4] >= conf]
        for c in range(N_CLS):
            pc, gc_ = p[p[:, 5] == c][:, :4], (g[g[:, 4] == c][:, :4] if len(g) else np.zeros((0, 4)))
            ious = iou_matrix(pc, gc_)
            matched = set()
            for i in np.argsort(-p[p[:, 5] == c][:, 4]) if len(pc) else []:
                j = int(np.argmax(ious[i])) if len(gc_) else -1
                if j >= 0 and ious[i, j] >= iou_th and j not in matched:
                    matched.add(j)
                    tp += 1
                else:
                    fp += 1
            fn += len(gc_) - len(matched)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


VAL = IMAGES[IMAGES.split == "val"]
GTS = {s: (ANN[ANN.stem == s][["x1", "y1", "x2", "y2", "cls"]].values if s in set(ANN.stem) else np.zeros((0, 5)))
       for s in VAL["stem"]}
VAL_PREDS = predict_images(VAL["path"].tolist()) if TRAINED else {}
CONF_TH, METRICS = CFG.CONF_THRESHOLD, {}
if VAL_PREDS:
    METRICS = evaluate(VAL_PREDS, GTS)
    print(f"Val mAP@0.5 = {METRICS['mAP50']:.4f} | mAP@0.5:0.95 = {METRICS['mAP50_95']:.4f}")
    ap_tab = pd.DataFrame({"class": CLASS_NAMES, "AP50": METRICS["ap50_per_class"], "AP50_95": METRICS["ap_per_class"]})
    display(ap_tab.round(4))
    save_table(ap_tab, "val_ap_per_class")
    grid = np.round(np.arange(0.05, 0.91, 0.05), 2)
    true_n = np.array([len(GTS[s]) for s in VAL["stem"]])
    rows = []
    for c in grid:
        p, r, f1 = prf_at(VAL_PREDS, GTS, c)
        pred_n = np.array([(VAL_PREDS.get(s, np.zeros((0, 6)))[:, 4] >= c).sum() for s in VAL["stem"]])
        rows.append({"conf": c, "precision": p, "recall": r, "f1": f1, "count_mae": np.abs(pred_n - true_n).mean(),
                     "count_rmse": np.sqrt(((pred_n - true_n) ** 2).mean())})
    thr = pd.DataFrame(rows)
    save_table(thr, "val_threshold_sweep")
    if CONF_TH is None:
        CONF_TH = float(thr.loc[thr["count_mae"].idxmin() if CFG.TASK_MODE == "count" else thr["f1"].idxmax(), "conf"])
    best_row = thr[thr.conf == CONF_TH].iloc[0] if (thr.conf == CONF_TH).any() else thr.iloc[0]
    print(f"Confidence threshold = {CONF_TH} ({'min MAE count' if CFG.TASK_MODE == 'count' else 'max F1'}) | "
          f"P {best_row.precision:.3f} R {best_row.recall:.3f} F1 {best_row.f1:.3f} | count MAE {best_row.count_mae:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(thr.conf, thr.precision, label="precision")
    ax[0].plot(thr.conf, thr.recall, label="recall")
    ax[0].plot(thr.conf, thr.f1, label="F1")
    ax[0].axvline(CONF_TH, ls="--", c="gray")
    ax[0].legend()
    ax[0].set_title("Precision / recall / F1 vs confidence (val)")
    ax[1].plot(thr.conf, thr.count_mae, label="MAE")
    ax[1].plot(thr.conf, thr.count_rmse, label="RMSE")
    ax[1].axvline(CONF_TH, ls="--", c="gray")
    ax[1].legend()
    ax[1].set_title("Error jumlah objek vs confidence (val)")
    save_fig("threshold_sweep", fig)

# %% [markdown]
# ## 9b. Kalibrasi jumlah (counting)
#
# Detektor sering konsisten *kurang* menghitung objek yang berhimpitan. Kalibrasi: regresi Ridge dari fitur deteksi
# (jumlah box di beberapa threshold + jumlah skor) → jumlah sebenarnya. Dinilai dengan **CV di val**; hanya dipakai
# kalau MAE-nya lebih kecil dari hitungan langsung.

# %%
COUNT_MODEL = None


def count_features(pred):
    s = pred[:, 4] if len(pred) else np.zeros(0)
    return [(s >= t).sum() for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)] + [s.sum(), (s ** 2).sum()]


if VAL_PREDS and CFG.TASK_MODE == "count" and CFG.COUNT_CALIBRATION and len(VAL) >= 10:
    Xc = np.array([count_features(VAL_PREDS.get(s, np.zeros((0, 6)))) for s in VAL["stem"]])
    yc = true_n
    direct = np.array([(VAL_PREDS.get(s, np.zeros((0, 6)))[:, 4] >= CONF_TH).sum() for s in VAL["stem"]])
    cvp = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), Xc, yc, cv=KFold(min(5, len(VAL) // 2), shuffle=True,
                                                                                     random_state=CFG.SEED))
    mae_direct, mae_cal = np.abs(direct - yc).mean(), np.abs(np.clip(np.round(cvp), 0, None) - yc).mean()
    print(f"Count MAE langsung = {mae_direct:.3f} | terkalibrasi (CV) = {mae_cal:.3f}")
    if mae_cal < mae_direct:
        COUNT_MODEL = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xc, yc)
        print("✅ Kalibrasi dipakai untuk submission count.")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(yc, direct, s=12, alpha=.6, label="langsung")
    ax.scatter(yc, np.round(cvp), s=12, alpha=.6, label="kalibrasi (CV)")
    lim = [0, max(yc.max(), direct.max()) + 1]
    ax.plot(lim, lim, "k--", lw=.8)
    ax.set_xlabel("jumlah sebenarnya")
    ax.set_ylabel("jumlah prediksi")
    ax.legend()
    ax.set_title("Counting di val")
    save_fig("count_calibration", fig)

# %% [markdown]
# # 10. Visual debugging di val
#
# Hijau = ground truth, merah = prediksi (≥ threshold). Lihat gambar dengan error jumlah terbesar.

# %%
if VAL_PREDS:
    err = pd.Series({s: abs((VAL_PREDS[s][:, 4] >= CONF_TH).sum() - len(GTS[s])) for s in VAL["stem"]}).sort_values(ascending=False)
    k = min(4, len(err))
    fig, axes = plt.subplots(1, k, figsize=(3.6 * k, 3.6))
    for a, s in zip(np.atleast_1d(axes), err.index[:k]):
        p = VAL_PREDS[s]
        p = p[p[:, 4] >= CONF_TH]
        a.imshow(Image.open(IMAGES.set_index("stem").loc[s, "path"]).convert("RGB"))
        for b in GTS[s]:
            a.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="lime", lw=1.2))
        for b in p:
            a.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="red", lw=1, ls="--"))
        a.set_title(f"{s[:14]} GT={len(GTS[s])} pred={len(p)}", fontsize=8)
        a.axis("off")
    fig.suptitle("Val: error terbesar (hijau = GT, merah = prediksi)")
    save_fig("val_worst_images", fig)

# %% [markdown]
# # 11. Prediksi test + submission
#
# Format otomatis dari `sample_submission`:
#
# | sample_submission | Format |
# |---|---|
# | 2 kolom, kolom ke-2 angka (`count`, `n`, `jumlah`) / `TASK_MODE="count"` | `count` |
# | ada kolom `PredictionString` | `prediction_string`: `"cls conf x1 y1 x2 y2 ..."` per gambar |
# | ada kolom box (`xmin`/`x`/`bbox`) | `boxes_csv`: satu baris per box |
# | tidak ada | `boxes_csv` + `coco_json` |
#
# Selalu cek urutan koordinat yang diminta lomba (`SUB_BOX_FORMAT`)!

# %%
def detect_sub_format():
    if CFG.SUBMISSION_FORMAT != "auto":
        return CFG.SUBMISSION_FORMAT
    if CFG.TASK_MODE == "count":
        return "count"
    if SAMPLE_SUB is not None:
        cols = [c.lower() for c in SAMPLE_SUB.columns]
        if any("predictionstring" in c.replace("_", "") for c in cols):
            return "prediction_string"
        if len(cols) == 2 and pd.api.types.is_numeric_dtype(SAMPLE_SUB.iloc[:, 1]):
            return "count"
    return "boxes_csv"


SUB_PATH = None
if TRAINED and len(TEST):
    TEST_PREDS = predict_images(TEST["path"].tolist())
    fmt = detect_sub_format()
    id_col = CFG.SUB_ID_COL or (SAMPLE_SUB.columns[0] if SAMPLE_SUB is not None else "image_id")
    keep = {s: p[p[:, 4] >= CONF_TH] for s, p in TEST_PREDS.items()}
    order = SAMPLE_SUB[id_col].astype(str).tolist() if SAMPLE_SUB is not None else list(TEST["stem"])
    stem_of = lambda v: os.path.splitext(os.path.basename(str(v)))[0]  # noqa: E731
    if fmt == "count":
        val_col = SAMPLE_SUB.columns[1] if SAMPLE_SUB is not None and SAMPLE_SUB.shape[1] > 1 else "count"
        rows = []
        for v in order:
            p = TEST_PREDS.get(stem_of(v), np.zeros((0, 6)))
            n = (p[:, 4] >= CONF_TH).sum() if COUNT_MODEL is None else max(0, int(round(COUNT_MODEL.predict([count_features(p)])[0])))
            rows.append({id_col: v, val_col: int(n)})
        sub = pd.DataFrame(rows)
    elif fmt == "prediction_string":
        ps_col = next(c for c in SAMPLE_SUB.columns if "predictionstring" in c.lower().replace("_", ""))
        sub = pd.DataFrame({id_col: order, ps_col: [" ".join(f"{int(b[5])} {b[4]:.4f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
                                                            for b in keep.get(stem_of(v), [])) for v in order]})
    else:
        rows = []
        for v in order:
            for b in keep.get(stem_of(v), []):
                x1, y1, x2, y2 = b[:4]
                box = {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2} if CFG.SUB_BOX_FORMAT == "xyxy" else \
                    {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                rows.append({id_col: v, "label": CLASS_NAMES[int(b[5])], "confidence": round(float(b[4]), 4),
                             **{k: round(float(z), 1) for k, z in box.items()}})
        sub = pd.DataFrame(rows)
        coco = {"images": [{"id": i, "file_name": os.path.basename(p)} for i, p in enumerate(TEST["path"])],
                "annotations": [{"image_id": i, "category_id": int(b[5]), "score": float(b[4]),
                                 "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]}
                                for i, s in enumerate(TEST["stem"]) for b in keep.get(s, [])],
                "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]}
        json.dump(coco, open(os.path.join(SUB_DIR, "predictions_coco.json"), "w"))
    SUB_PATH = os.path.join(SUB_DIR, "submission.csv")
    sub.to_csv(SUB_PATH, index=False)
    print(f"✅ Submission ({fmt}): {SUB_PATH} | {len(sub)} baris | kolom {list(sub.columns)}")
    if SAMPLE_SUB is not None and fmt in ("count", "prediction_string"):
        ok = len(sub) == len(SAMPLE_SUB) and list(sub.columns) == list(SAMPLE_SUB.columns)
        print("✅ Validator: jumlah baris & kolom sama dengan sample_submission" if ok else
              "⚠️ Validator: baris/kolom berbeda dari sample_submission — cek!")
    missing = sum(stem_of(v) not in TEST_PREDS for v in order)
    if missing:
        print(f"⚠️ {missing} id di submission tidak punya gambar test yang cocok (nama file beda?)")
    k = min(4, len(TEST))
    fig, axes = plt.subplots(1, k, figsize=(3.4 * k, 3.4))
    for a, (_, r) in zip(np.atleast_1d(axes), TEST.head(k).iterrows()):
        draw_boxes(a, r["path"], keep.get(r["stem"], np.zeros((0, 6)))[:, [0, 1, 2, 3, 5]],
                   f"{r['stem'][:14]} ({len(keep.get(r['stem'], []))} obj)")
    fig.suptitle(f"Prediksi test (conf ≥ {CONF_TH})")
    save_fig("test_predictions", fig)
elif not len(TEST):
    print("ℹ️ Tidak ada gambar test → submission dilewati.")

# %% [markdown]
# # 12. Ringkasan untuk laporan

# %%
if TRAINED:
    summ = {"model": TRAINED.get("name"), "task_mode": CFG.TASK_MODE, "img_size": CFG.IMG_SIZE, "n_train_images": int((IMAGES.split == "train").sum()),
            "n_val_images": len(VAL), "n_boxes": len(ANN), "classes": CLASS_NAMES, "val_mAP50": round(METRICS.get("mAP50", float("nan")), 4),
            "val_mAP50_95": round(METRICS.get("mAP50_95", float("nan")), 4), "conf_threshold": CONF_TH,
            "count_calibrated": COUNT_MODEL is not None, "submission": SUB_PATH}
    if VAL_PREDS:
        summ["val_count_mae"] = round(float(best_row.count_mae), 3)
    text = f"""# Object detection summary

- Model **{summ['model']}**, input {CFG.IMG_SIZE}px, {summ['n_train_images']} train / {summ['n_val_images']} val images, {len(CLASS_NAMES)} class(es).
- Validation mAP@0.5 **{summ['val_mAP50']}**, mAP@0.5:0.95 **{summ['val_mAP50_95']}**; confidence threshold {CONF_TH}.
- Counting MAE on validation: {summ.get('val_count_mae', '-')}{' (after linear calibration)' if COUNT_MODEL is not None else ''}.

**Draft sentences (paraphrase):**
- "We fine-tuned {summ['model']} on {summ['n_train_images']} annotated images and reached a validation mAP@0.5 of {summ['val_mAP50']}."
- "The confidence threshold was chosen on the validation split to {'minimise counting error' if CFG.TASK_MODE == 'count' else 'maximise F1'}."
"""
    open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8").write(text)
    json.dump(summ, open(os.path.join(OUT_DIR, "run_summary.json"), "w"), indent=2, default=str)
    print(text)
print("✅ Notebook selesai.")

# %% [markdown]
# # 13. Troubleshooting
#
# | Gejala | Penyebab | Solusi |
# |---|---|---|
# | Box di figure contoh bergeser | format koordinat salah | `CSV_BOX_FORMAT="xywh"`/`"xyxy"`; cek normalisasi |
# | `mAP=0` setelah training | label kosong/salah kelas, epoch kurang, training dari nol | cek `labels/*.txt`; `PRETRAINED=True`; tambah epoch |
# | Download `yolo11n.pt` gagal | offline | salin file .pt ke folder kerja dan set `YOLO_MODEL` ke path-nya; atau `PRETRAINED=False` |
# | Windows: script jalan ulang dari awal saat training | DataLoader multiprocessing | `NUM_WORKERS=0` (default Windows) |
# | `CUDA out of memory` | batch/gambar besar | otomatis batch /2; turunkan `IMG_SIZE`/`BATCH_SIZE` |
# | Count selalu kurang | objek berhimpitan / NMS terlalu ketat | naikkan `NMS_IOU` (0.7), `MAX_DET`, `COUNT_CALIBRATION=True` |
# | Banyak false positive | threshold rendah | threshold di-tune otomatis; tambahkan gambar negatif (tanpa objek) ke train |
#
# # 14. Recipes
#
# ```python
# CFG.YOLO_MODEL, CFG.IMG_SIZE, CFG.EPOCHS = "yolo11s.pt", 960, 100     # objek kecil, GPU
# CFG.AUG_FLIPUD, CFG.AUG_MOSAIC = 0.5, 1.0                              # citra udara / mikroskop
# CFG.TASK_MODE, CFG.NMS_IOU, CFG.MAX_DET = "count", 0.7, 1000           # counting padat
# CFG.BACKEND = "torchvision"                                            # tanpa ultralytics
# ```
# Counting **tanpa** box di data train (hanya angka): pakai `04_Computer_Vision/cv_image_classification` dengan
# target angka, atau regresi dari fitur CNN (`feat_*` baseline di template itu).
#
# # 15. Checklist
#
# - [ ] Figure anotasi: box pas dengan objek
# - [ ] Kelas & urutan kelas sama dengan aturan lomba (id kelas di submission!)
# - [ ] Threshold dipilih di val; metrik val dicatat untuk laporan
# - [ ] Format koordinat submission (xyxy/xywh, pixel/normalisasi) sesuai aturan
# - [ ] Jumlah baris/ID submission cocok dengan sample_submission
