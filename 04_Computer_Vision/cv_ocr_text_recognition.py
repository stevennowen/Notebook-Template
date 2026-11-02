# -*- coding: utf-8 -*-
# =====================================================================================================
# cv_ocr_text_recognition.py — versi SCRIPT dari cv_ocr_text_recognition.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python cv_ocr_text_recognition.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb cv_ocr_text_recognition.py
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
# # Computer Vision Competition Template — OCR / Text Recognition
#
# Template untuk lomba yang meminta **membaca teks dari gambar**: deret digit (meteran, nomor rumah, Gemastik XIX 3a
# multi-digit), plat nomor, kode/captcha, kata pendek, aksara (Jawa/Bali/Lontara → setiap simbol = satu "karakter").
# Input: gambar yang sudah berisi **satu baris teks** (crop). Untuk dokumen penuh (banyak baris), deteksi baris dulu
# dengan `cv_object_detection`, lalu crop → template ini.
#
# ---
#
# ## Pipeline besar
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json)
# → Label dari CSV (image, text) / nama file (regex) / file .txt per gambar
# → Charset otomatis dari label train (+ cek karakter test yang mungkin tidak pernah muncul)
# → EDA: panjang teks, frekuensi karakter, ukuran & rasio gambar, contoh gambar + label
# → Preprocessing: grayscale, resize tinggi tetap (lebar mengikuti rasio, padding), normalisasi
# → Augmentasi aman-teks: rotasi kecil, shear, blur, noise, brightness/contrast (TANPA flip)
# → Model CRNN (CNN → BiLSTM → CTC) dari nol; K-fold opsional (ensemble log-prob)
# → Decoding: greedy / beam search + batasan (panjang tetap, regex, charset)
# → Evaluasi: exact match, CER, akurasi per panjang, kesalahan karakter tersering
# → (Opsional) TrOCR pretrained dari transformers sebagai pembanding
# → Submission + validator, ringkasan laporan, troubleshooting, recipes, checklist
# ```
#
# ## Cara pakai cepat
#
# 1. Data: `data/train.csv` (kolom gambar + kolom teks) + folder gambar; `data/test.csv` atau folder test.
# 2. `# >>> USER OVERRIDE`: path, nama kolom (atau biarkan auto), `FIXED_LENGTH` kalau panjang teks selalu sama.
# 3. `RUN_MODE="fast"` → Run All → cek contoh gambar + label dan charset.
# 4. `RUN_MODE="full"` → Run All.
# 5. Submission: `outputs/cv_ocr_text_recognition/submissions/submission.csv`.
#
# **Dua format:** `.ipynb` dan `.py` (isi sama). Headless: `python tools/run_task.py --task ocr --format py`.

# %% [markdown]
# # 1. Config (`CFG`)
#
# **Jika** teks hanya digit **maka** biarkan charset auto (atau `CHARSET="0123456789"`).
# **Jika** panjang selalu sama (mis. 6 digit) **maka** `FIXED_LENGTH=6` → decoding dipaksa 6 karakter (akurasi naik).
# **Jika** gambar lebar & teks panjang **maka** naikkan `IMG_W` (lebar input) supaya tiap karakter dapat ≥ 2 kolom fitur.

# %%
import os


class CFG:
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "cv_ocr_text_recognition"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"

    # ---------------- data
    TRAIN_CSV = "./data/train.csv"
    TRAIN_IMAGE_DIR = "./data/train"
    TEST_CSV = "./data/test.csv"
    TEST_IMAGE_DIR = "./data/test"
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    IMAGE_COL = None             # None = auto (kolom berisi nama file gambar)
    TEXT_COL = None              # None = auto
    LABEL_FROM_FILENAME = None   # tanpa CSV: regex dengan grup (?P<text>...), mis. r"^(?P<text>[0-9]+)_\d+$"
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    OUTPUT_DIR = "./outputs/cv_ocr_text_recognition/"

    # ---------------- teks
    CHARSET = None               # None = semua karakter di label train
    CASE_SENSITIVE = True        # False → semua label di-upper()
    STRIP_SPACES = False         # True → spasi dihapus dari label
    FIXED_LENGTH = None          # mis. 6 → prediksi selalu 6 karakter
    TEXT_REGEX = None            # mis. r"^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$" (plat) → kandidat beam yang cocok diutamakan

    # ---------------- gambar & model
    IMG_H = 32
    IMG_W = None                 # None = auto dari rasio median (kelipatan 4, 64–512)
    GRAYSCALE = True
    CNN_CHANNELS = (64, 128, 256, 256)
    RNN_HIDDEN = 256
    RNN_LAYERS = 2
    DROPOUT = 0.2
    USE_GPU = True
    USE_AMP = True
    BATCH_SIZE = 32              # CTC butuh banyak step untuk keluar dari "plateau blank" → batch kecil membantu
    EPOCHS = 60
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 12
    MIN_EPOCHS = None            # early stopping baru aktif setelah epoch ini (None = sepertiga EPOCHS, min 8)
    N_FOLDS = 1                  # 1 = holdout (VAL_FRAC); ≥2 = K-fold + ensemble
    VAL_FRAC = 0.15
    NUM_WORKERS = 0
    BEAM_WIDTH = 10              # 1 = greedy
    USE_TROCR = False            # True = bandingkan dengan TrOCR pretrained (butuh transformers + internet/cache)
    TROCR_MODEL = "microsoft/trocr-small-printed"

    # ---------------- augmentasi
    AUG_ROTATE = 3.0             # derajat
    AUG_SHEAR = 0.15
    AUG_BLUR = 0.2               # peluang
    AUG_NOISE = 0.2
    AUG_BRIGHTNESS = 0.3

    # ---------------- submission
    SUB_ID_COL = None
    SUB_TEXT_COL = None

    # ---------------- fast
    FAST_SAMPLE = 600
    FAST_EPOCHS = 15
    FIG_DPI = 150

# %% [markdown]
# # 2. USER OVERRIDE

# %%
# >>> USER OVERRIDE
# CFG.RUN_MODE = "fast"
# CFG.FIXED_LENGTH = 6
# CFG.IMG_H, CFG.IMG_W = 48, 256
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
# Inti: numpy, pandas, Pillow, torch. Torch tidak di-install otomatis. `transformers` hanya untuk TrOCR (opsional).

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
                              "matplotlib": "matplotlib"}, CFG.AUTO_INSTALL)
if CFG.USE_TROCR:
    PKG_STATUS.update(ensure_packages({"transformers": "transformers"}, CFG.AUTO_INSTALL))
if importlib.util.find_spec("torch") is None:
    print("❌ PyTorch belum ada → https://pytorch.org/get-started/locally/")

# %% [markdown]
# # 4. Import, device, helper

# %%
import gc
import glob
import json
import math
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
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except Exception:  # noqa
    display = print
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if CFG.USE_GPU and torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | device {DEVICE}")
if CFG.RUN_MODE == "fast":
    CFG.EPOCHS = min(CFG.EPOCHS, CFG.FAST_EPOCHS)
    CFG.N_FOLDS = 1
    CFG.PATIENCE = min(CFG.PATIENCE, 6)
    print(f"[FAST MODE] epochs={CFG.EPOCHS} holdout, sample≤{CFG.FAST_SAMPLE}")
OUT_DIR = CFG.OUTPUT_DIR
FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR = (os.path.join(OUT_DIR, d) for d in ("figures", "tables", "models", "submissions"))
for _d in (OUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR):
    os.makedirs(_d, exist_ok=True)


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


seed_everything(CFG.SEED)
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


def save_table(df, name):
    path = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print("  [table] saved:", path)


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
# # 5. Load data + label
#
# Label teks dibaca sebagai **string** (penting: `"007"` tidak boleh jadi angka 7).
#
# **Jika** `file tidak ditemukan` banyak → cek `TRAIN_IMAGE_DIR` dan apakah kolom gambar berisi ekstensi.

# %%
def index_images(folder):
    idx = {}
    if folder and os.path.isdir(folder):
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(tuple(CFG.IMAGE_EXTS)):
                    p = os.path.join(root, f)
                    idx.setdefault(f, p)
                    idx.setdefault(os.path.splitext(f)[0], p)
    return idx


def resolve(idx, v):
    v = str(v).strip().replace("\\", "/")
    return idx.get(os.path.basename(v)) or idx.get(os.path.splitext(os.path.basename(v))[0]) or (v if os.path.exists(v) else None)


def guess_cols(df):
    img_col = CFG.IMAGE_COL or next((c for c in df.columns if df[c].astype(str).str.lower().str.endswith(tuple(CFG.IMAGE_EXTS)).mean() > .5), None)
    img_col = img_col or next((c for c in df.columns if c.lower() in ("image", "image_id", "filename", "file", "id", "img", "path")), df.columns[0])
    txt_col = CFG.TEXT_COL or next((c for c in df.columns if c != img_col and c.lower() in
                                    ("text", "label", "labels", "transcription", "word", "digits", "number", "plate", "target")), None)
    txt_col = txt_col or next((c for c in df.columns if c != img_col), None)
    return img_col, txt_col


def normalize_text(t):
    t = str(t)
    if not CFG.CASE_SENSITIVE:
        t = t.upper()
    if CFG.STRIP_SPACES:
        t = t.replace(" ", "")
    return t


TRAIN_IDX = index_images(CFG.TRAIN_IMAGE_DIR)
if os.path.exists(CFG.TRAIN_CSV):
    raw = smart_read(CFG.TRAIN_CSV, dtype=str, keep_default_na=False)
    IMG_COL, TXT_COL = guess_cols(raw)
    train_df = pd.DataFrame({"path": raw[IMG_COL].map(lambda v: resolve(TRAIN_IDX, v)), "text": raw[TXT_COL].map(normalize_text)})
    print(f"train.csv: kolom gambar = '{IMG_COL}', kolom teks = '{TXT_COL}'")
elif CFG.LABEL_FROM_FILENAME:
    rx = re.compile(CFG.LABEL_FROM_FILENAME)
    rows = []
    for name, p in TRAIN_IDX.items():
        if "." in name:
            m = rx.match(os.path.splitext(name)[0])
            if m:
                rows.append({"path": p, "text": normalize_text(m.group("text"))})
    train_df = pd.DataFrame(rows)
else:
    rows = []
    for name, p in TRAIN_IDX.items():
        txt = os.path.splitext(p)[0] + ".txt"
        if "." in name and os.path.exists(txt):
            rows.append({"path": p, "text": normalize_text(open(txt, encoding="utf-8").read().strip())})
    train_df = pd.DataFrame(rows)
    if train_df.empty:
        raise FileNotFoundError("Label tidak ditemukan: isi TRAIN_CSV, LABEL_FROM_FILENAME, atau sediakan <gambar>.txt")
missing = train_df["path"].isna().sum()
train_df = train_df.dropna(subset=["path"])
train_df = train_df[train_df["text"].str.len() > 0].reset_index(drop=True)
print(f"Train: {len(train_df)} gambar berlabel | file tidak ditemukan: {missing}")

TEST_IDX = index_images(CFG.TEST_IMAGE_DIR)
if os.path.exists(CFG.TEST_CSV):
    rt = smart_read(CFG.TEST_CSV, dtype=str, keep_default_na=False)
    tcol = guess_cols(rt)[0]
    test_df = pd.DataFrame({"id": rt[tcol], "path": rt[tcol].map(lambda v: resolve(TEST_IDX, v))})
else:
    test_df = pd.DataFrame({"path": sorted({p for k, p in TEST_IDX.items() if "." in k})})
    test_df["id"] = test_df["path"].map(os.path.basename)
print(f"Test: {len(test_df)} | tanpa file: {test_df['path'].isna().sum()}")
SAMPLE_SUB = smart_read(CFG.SAMPLE_SUB_PATH, dtype=str, keep_default_na=False) if os.path.exists(CFG.SAMPLE_SUB_PATH) else None

# %% [markdown]
# # 6. Charset & EDA
#
# - **Charset** = himpunan karakter yang bisa diprediksi. Index 0 dicadangkan untuk *blank* CTC.
# - Teks panjang `L` butuh lebar fitur ≥ `2L+1` kolom (CTC perlu ruang untuk blank di antara huruf kembar).
#   Lebar fitur = `IMG_W / 4` → `IMG_W` otomatis dinaikkan kalau terlalu sempit.

# %%
CHARSET = CFG.CHARSET or "".join(sorted(set("".join(train_df["text"]))))
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
N_OUT = len(CHARSET) + 1
lengths = train_df["text"].str.len()
char_freq = Counter("".join(train_df["text"]))
unknown_chars = set("".join(train_df["text"])) - set(CHARSET)
if unknown_chars:
    print("⚠️ karakter label di luar CHARSET (diabaikan):", unknown_chars)


def img_wh(p):
    with Image.open(p) as im:
        return im.size


wh = np.array([img_wh(p) for p in train_df["path"].sample(min(400, len(train_df)), random_state=0)])
ratio = np.median(wh[:, 0] / wh[:, 1])
if CFG.IMG_W is None:
    CFG.IMG_W = int(np.clip(round(CFG.IMG_H * ratio / 4) * 4, 64, 512))
min_w = 4 * (2 * lengths.max() + 2)
if CFG.IMG_W < min_w:
    print(f"ℹ️ IMG_W {CFG.IMG_W} terlalu sempit untuk teks {lengths.max()} karakter → {min_w}")
    CFG.IMG_W = int(min_w)
print(f"Charset ({len(CHARSET)}): {CHARSET!r}")
print(f"Panjang teks: min {lengths.min()} | median {lengths.median():.0f} | max {lengths.max()} | "
      f"rasio gambar median {ratio:.2f} → input {CFG.IMG_H}×{CFG.IMG_W}")
if CFG.FIXED_LENGTH is None and lengths.nunique() == 1:
    print(f"ℹ️ Semua label panjangnya {lengths.iloc[0]} → pertimbangkan CFG.FIXED_LENGTH = {lengths.iloc[0]}")
save_table(pd.DataFrame(char_freq.most_common(), columns=["char", "count"]), "char_frequency")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.3))
ax[0].hist(lengths, bins=range(lengths.min(), lengths.max() + 2), color="#4C72B0", align="left")
ax[0].set_title("Panjang teks (karakter)")
top = char_freq.most_common(40)
ax[1].bar([c if c != " " else "␣" for c, _ in top], [n for _, n in top], color="#55A868")
ax[1].set_title("Frekuensi karakter (40 teratas)")
save_fig("text_length_char_freq", fig)
k = min(8, len(train_df))
fig, axes = plt.subplots(2, 4, figsize=(12, 3))
for a, (_, r) in zip(axes.ravel(), train_df.sample(k, random_state=1).iterrows()):
    a.imshow(Image.open(r["path"]).convert("RGB"))
    a.set_title(repr(r["text"]), fontsize=9)
    a.axis("off")
fig.suptitle("Contoh gambar + label")
save_fig("sample_images", fig)

# %% [markdown]
# # 7. Dataset, preprocessing, augmentasi
#
# Gambar di-resize ke tinggi `IMG_H` dengan **rasio tetap**, lalu di-pad di kanan sampai `IMG_W` (teks tidak
# melar). Augmentasi tidak memakai flip (teks terbalik = karakter lain).

# %%
RESAMPLE = getattr(Image, "Resampling", Image)


def load_proc(path, train=False, rng=None):
    im = Image.open(path)
    im = im.convert("L" if CFG.GRAYSCALE else "RGB")
    if train and rng is not None:
        if CFG.AUG_ROTATE:
            im = im.rotate(float(rng.uniform(-CFG.AUG_ROTATE, CFG.AUG_ROTATE)), resample=RESAMPLE.BILINEAR, expand=False,
                           fillcolor=int(np.asarray(im).mean()) if CFG.GRAYSCALE else None)
        if CFG.AUG_SHEAR:
            s = float(rng.uniform(-CFG.AUG_SHEAR, CFG.AUG_SHEAR))
            im = im.transform(im.size, Image.AFFINE, (1, s, -s * im.size[1] / 2, 0, 1, 0), resample=RESAMPLE.BILINEAR)
        if rng.random() < CFG.AUG_BLUR:
            im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.0))))
    w, h = im.size
    nw = max(4, min(CFG.IMG_W, int(round(w * CFG.IMG_H / max(1, h)))))
    im = im.resize((nw, CFG.IMG_H), RESAMPLE.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    if a.ndim == 2:
        a = a[None]
    else:
        a = a.transpose(2, 0, 1)
    if train and rng is not None:
        if CFG.AUG_BRIGHTNESS:
            a = a * float(rng.uniform(1 - CFG.AUG_BRIGHTNESS, 1 + CFG.AUG_BRIGHTNESS)) + float(rng.uniform(-.1, .1))
        if rng.random() < CFG.AUG_NOISE:
            a = a + rng.normal(0, 0.04, a.shape).astype(np.float32)
    out = np.full((a.shape[0], CFG.IMG_H, CFG.IMG_W), float(np.median(a[:, :, -1])), np.float32)
    out[:, :, :nw] = a
    return (out - 0.5) / 0.5


def encode_text(t):
    return [CHAR2IDX[c] for c in t if c in CHAR2IDX]


class OCRDataset(Dataset):
    def __init__(self, df, train=False, labels=True):
        self.df, self.train, self.labels = df.reset_index(drop=True), train, labels
        self.rng = np.random.default_rng(CFG.SEED)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = torch.from_numpy(load_proc(r["path"], self.train, self.rng))
        y = encode_text(r["text"]) if self.labels else []
        return x, torch.tensor(y, dtype=torch.long), len(y)


def collate(b):
    xs, ys, ls = zip(*b)
    return torch.stack(xs), torch.cat(ys) if sum(ls) else torch.zeros(0, dtype=torch.long), torch.tensor(ls, dtype=torch.long)

# %% [markdown]
# # 8. Model CRNN + CTC
#
# CNN mengecilkan tinggi menjadi 1 dan lebar menjadi `IMG_W/4` (setiap kolom = "irisan" gambar) → BiLSTM membaca
# irisan dari kiri ke kanan → linear ke `charset + blank`. Loss **CTC** tidak butuh posisi karakter.

# %%
class CRNN(nn.Module):
    def __init__(self, n_out, in_ch=1):
        super().__init__()
        c = CFG.CNN_CHANNELS
        layers, ch = [], in_ch
        pools = [(2, 2), (2, 2), (2, 1), (2, 1)]
        for o, p in zip(c, pools):
            layers += [nn.Conv2d(ch, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                       nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(p, p)]
            ch = o
        self.cnn = nn.Sequential(*layers)
        self.rnn = nn.LSTM(ch, CFG.RNN_HIDDEN, num_layers=CFG.RNN_LAYERS, bidirectional=True, batch_first=True,
                           dropout=CFG.DROPOUT if CFG.RNN_LAYERS > 1 else 0)
        self.drop = nn.Dropout(CFG.DROPOUT)
        self.fc = nn.Linear(2 * CFG.RNN_HIDDEN, n_out)

    def forward(self, x):
        f = self.cnn(x)                     # B, C, H', W'
        f = f.mean(2).permute(0, 2, 1)      # B, T, C   (tinggi dirata-rata)
        f, _ = self.rnn(f)
        return self.fc(self.drop(f))        # B, T, n_out


IN_CH = 1 if CFG.GRAYSCALE else 3
_m = CRNN(N_OUT, IN_CH)
T_STEPS = _m(torch.zeros(1, IN_CH, CFG.IMG_H, CFG.IMG_W)).shape[1]
print(f"CRNN: {sum(p.numel() for p in _m.parameters()) / 1e6:.2f} M parameter | timesteps = {T_STEPS} | output = {N_OUT}")
del _m

# %% [markdown]
# # 9. Decoding (greedy & beam search) + batasan
#
# - **Greedy**: ambil karakter paling mungkin tiap timestep, gabungkan duplikat, buang blank.
# - **Beam search** (prefix beam, tanpa language model): menyimpan `BEAM_WIDTH` kandidat terbaik.
# - **Batasan**: kandidat dengan panjang = `FIXED_LENGTH` dan/atau cocok `TEXT_REGEX` diutamakan.

# %%
def greedy_decode(logp):
    best = logp.argmax(-1)
    out, prev = [], 0
    for k in best:
        if k != prev and k != 0:
            out.append(CHARSET[k - 1])
        prev = k
    return "".join(out)


def beam_decode(logp, width=10):
    """Prefix beam search CTC. logp: (T, C) numpy log-prob. Return list (teks, skor log) terurut."""
    T, C = logp.shape
    beams = {(): (0.0, -np.inf)}          # prefix → (log p blank-ending, log p non-blank-ending)
    lse = np.logaddexp
    top_c = np.argsort(-logp, 1)[:, :min(C, width + 2)]
    for t in range(T):
        nb = {}
        for pre, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            for c in top_c[t]:
                p = logp[t, c]
                if c == 0:
                    b_, nb_ = nb.get(pre, (-np.inf, -np.inf))
                    nb[pre] = (lse(b_, tot + p), nb_)
                    continue
                npre = pre + (c,)
                b_, nb_ = nb.get(npre, (-np.inf, -np.inf))
                if pre and pre[-1] == c:
                    nb[npre] = (b_, lse(nb_, pb + p))
                    b2, nb2 = nb.get(pre, (-np.inf, -np.inf))
                    nb[pre] = (b2, lse(nb2, pnb + p))
                else:
                    nb[npre] = (b_, lse(nb_, tot + p))
        beams = dict(sorted(nb.items(), key=lambda kv: -lse(*kv[1]))[:width])
    return [("".join(CHARSET[i - 1] for i in pre), float(lse(*s))) for pre, s in sorted(beams.items(), key=lambda kv: -lse(*kv[1]))]


RX = re.compile(CFG.TEXT_REGEX) if CFG.TEXT_REGEX else None


def decode(logp):
    if CFG.BEAM_WIDTH <= 1 and not CFG.FIXED_LENGTH and RX is None:
        return greedy_decode(logp)
    cands = beam_decode(logp, max(CFG.BEAM_WIDTH, 2))
    ok = [t for t, _ in cands if (not CFG.FIXED_LENGTH or len(t) == CFG.FIXED_LENGTH) and (RX is None or RX.match(t))]
    return ok[0] if ok else cands[0][0]


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_metrics(pred, true):
    exact = np.mean([p == t for p, t in zip(pred, true)])
    cer = sum(levenshtein(p, t) for p, t in zip(pred, true)) / max(1, sum(len(t) for t in true))
    return {"exact_match": float(exact), "cer": float(cer)}

# %% [markdown]
# # 10. Training
#
# **Cara membaca:** `loss` CTC turun cepat ke ±2.6 lalu **diam beberapa epoch** (model baru memprediksi *blank*) —
# ini normal untuk CTC. Setelah ±150–300 step, loss turun lagi dan `val exact` mulai naik. Karena itu early stopping
# baru aktif setelah `MIN_EPOCHS`. Kalau `val exact` tetap 0 setelah banyak epoch → cek label vs gambar (contoh EDA),
# `IMG_W` terlalu kecil (lihat timesteps), atau data terlalu sedikit (< 500 gambar: tambah augmentasi/epoch).

# %%
def run_epoch_predict(model, loader):
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _, _ in loader:
            with torch.autocast(device_type="cuda", enabled=CFG.USE_AMP and DEVICE == "cuda"):
                lo = model(x.to(DEVICE)).float()
            outs.append(F.log_softmax(lo, -1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, T_STEPS, N_OUT), np.float32)


def train_one(tr, va, fold):
    seed_everything(CFG.SEED + fold)
    model = CRNN(N_OUT, IN_CH).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    tl = DataLoader(OCRDataset(tr, True), CFG.BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=CFG.NUM_WORKERS,
                    drop_last=len(tr) > CFG.BATCH_SIZE)
    vl = DataLoader(OCRDataset(va, False), CFG.BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=CFG.NUM_WORKERS)
    steps = max(1, CFG.EPOCHS * len(tl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG.LR, total_steps=steps, pct_start=0.15)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP and DEVICE == "cuda")
    best, best_ep, hist = (-1.0, -np.inf), 0, []      # (exact match, -CER): CER ikut dihitung saat exact masih 0
    ckpt = os.path.join(MODEL_DIR, f"crnn_fold{fold}.pt")
    for ep in range(CFG.EPOCHS):
        model.train()
        losses = []
        for x, y, ln in tl:
            x = x.to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=CFG.USE_AMP and DEVICE == "cuda"):
                lo = model(x)
            lp = F.log_softmax(lo.float(), -1).permute(1, 0, 2)       # T, B, C
            loss = ctc(lp, y.to(DEVICE), torch.full((x.size(0),), lp.size(0), dtype=torch.long), ln)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < steps - 1:
                sched.step()
            losses.append(loss.item())
        lp_va = run_epoch_predict(model, vl)
        pred = [greedy_decode(l) for l in lp_va]
        m = text_metrics(pred, va["text"].tolist())
        hist.append({"fold": fold, "epoch": ep + 1, "loss": float(np.mean(losses)), **m})
        star = ""
        score = (m["exact_match"], -m["cer"])
        if score > best:
            best, best_ep, star = score, ep, " *"
            torch.save(model.state_dict(), ckpt)
        print(f"  fold {fold} ep {ep + 1:02d} loss {np.mean(losses):.4f} | val exact {m['exact_match']:.4f} cer {m['cer']:.4f}{star}")
        min_ep = CFG.MIN_EPOCHS if CFG.MIN_EPOCHS is not None else max(8, CFG.EPOCHS // 3)
        if ep + 1 >= min_ep and ep - best_ep >= CFG.PATIENCE:
            print("  early stopping")
            break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model, hist, run_epoch_predict(model, vl)


if CFG.RUN_MODE == "fast" and len(train_df) > CFG.FAST_SAMPLE:
    train_df = train_df.sample(CFG.FAST_SAMPLE, random_state=CFG.SEED).reset_index(drop=True)
    print(f"[FAST MODE] train di-subsample ke {len(train_df)}")
if CFG.N_FOLDS > 1:
    splits = list(KFold(CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED).split(train_df))
else:
    tr_i, va_i = train_test_split(np.arange(len(train_df)), test_size=CFG.VAL_FRAC, random_state=CFG.SEED)
    splits = [(tr_i, va_i)]
OOF_LOGP = np.zeros((len(train_df), T_STEPS, N_OUT), np.float32)
OOF_MASK = np.zeros(len(train_df), bool)
MODELS, HIST = [], []
for f, (tr_i, va_i) in enumerate(splits):
    print(f"===== FOLD {f} ({len(tr_i)} train / {len(va_i)} val) =====")
    m, h, lp = train_one(train_df.iloc[tr_i], train_df.iloc[va_i], f)
    MODELS.append(m.cpu())
    HIST += h
    OOF_LOGP[va_i], OOF_MASK[va_i] = lp, True
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
hist = pd.DataFrame(HIST)
save_table(hist, "training_history")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
for f, g in hist.groupby("fold"):
    ax[0].plot(g.epoch, g.loss, label=f"fold {f}")
    ax[1].plot(g.epoch, g.exact_match, label=f"fold {f}")
ax[0].set_title("CTC loss")
ax[1].set_title("Val exact match")
ax[1].legend(fontsize=8)
save_fig("learning_curves", fig)

# %% [markdown]
# # 11. Evaluasi OOF: greedy vs beam (+batasan)
#
# Decoding terbaik dipilih dari OOF. Tabel *kesalahan tersering* menunjukkan pasangan karakter yang tertukar
# (mis. `0↔O`, `1↔7`) → bahan analisis di laporan dan ide augmentasi/aturan post-processing.

# %%
oof_df = train_df[OOF_MASK].copy()
oof_lp = OOF_LOGP[OOF_MASK]
oof_df["pred_greedy"] = [greedy_decode(l) for l in oof_lp]
oof_df["pred_beam"] = [decode(l) for l in oof_lp]
res = {"greedy": text_metrics(oof_df["pred_greedy"], oof_df["text"]), "beam": text_metrics(oof_df["pred_beam"], oof_df["text"])}
display(pd.DataFrame(res).T.round(4))
DECODER = "beam" if (res["beam"]["exact_match"], -res["beam"]["cer"]) > (res["greedy"]["exact_match"], -res["greedy"]["cer"]) else "greedy"
oof_df["pred"] = oof_df[f"pred_{DECODER}"]
print(f"Decoder terpilih: {DECODER} | OOF exact {res[DECODER]['exact_match']:.4f} | CER {res[DECODER]['cer']:.4f}")
oof_df["len"] = oof_df["text"].str.len()
by_len = oof_df.groupby("len").apply(lambda g: pd.Series({"n": len(g), "exact": (g.pred == g.text).mean()})).reset_index()
display(by_len)
save_table(oof_df[["path", "text", "pred", "pred_greedy", "pred_beam"]], "oof_predictions")
subs = Counter()
for p, t in zip(oof_df["pred"], oof_df["text"]):
    if len(p) == len(t):
        subs.update((a, b) for a, b in zip(t, p) if a != b)
err_tab = pd.DataFrame([(f"{a}→{b}", n) for (a, b), n in subs.most_common(15)], columns=["true→pred", "count"])
display(err_tab)
save_table(err_tab, "char_confusions")
wrong = oof_df[oof_df.pred != oof_df.text].head(8)
if len(wrong):
    fig, axes = plt.subplots(2, 4, figsize=(12, 3))
    for a, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        a.imshow(Image.open(r["path"]).convert("RGB"))
        a.set_title(f"GT {r['text']!r}\npred {r['pred']!r}", fontsize=8, color="crimson")
        a.axis("off")
    for a in axes.ravel()[len(wrong):]:
        a.axis("off")
    fig.suptitle("Contoh salah baca (OOF)")
    save_fig("oof_errors", fig)

# %% [markdown]
# ## 11b. (Opsional) TrOCR pretrained sebagai pembanding
#
# `USE_TROCR=True`: menjalankan TrOCR (encoder-decoder transformer, pretrained teks cetak) *zero-shot* di sampel val.
# Kalau jauh lebih baik dari CRNN (mis. teks tulisan/foto alami), pertimbangkan fine-tune TrOCR (lihat Recipes).

# %%
if CFG.USE_TROCR:
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        proc = TrOCRProcessor.from_pretrained(CFG.TROCR_MODEL)
        tro = VisionEncoderDecoderModel.from_pretrained(CFG.TROCR_MODEL).to(DEVICE).eval()
        samp = oof_df.head(min(200, len(oof_df)))
        preds = []
        for i in range(0, len(samp), 16):
            ims = [Image.open(p).convert("RGB") for p in samp["path"].iloc[i:i + 16]]
            with torch.no_grad():
                ids = tro.generate(proc(images=ims, return_tensors="pt").pixel_values.to(DEVICE), max_new_tokens=32)
            preds += [normalize_text(t).strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
        print("TrOCR zero-shot:", text_metrics(preds, samp["text"].tolist()), "vs CRNN:",
              text_metrics(samp["pred"].tolist(), samp["text"].tolist()))
    except Exception as e:  # noqa
        print("TrOCR dilewati:", type(e).__name__, str(e)[:200])

# %% [markdown]
# # 12. Prediksi test + submission
#
# K-fold: log-probabilitas semua fold dirata-rata (sejajar per timestep karena arsitektur sama) lalu di-decode.

# %%
SUB_PATH = None
tst = test_df.dropna(subset=["path"]).reset_index(drop=True)
if len(tst):
    tl = DataLoader(OCRDataset(tst.assign(text=""), False, labels=False), CFG.BATCH_SIZE, shuffle=False, collate_fn=collate)
    acc = 0
    for m in MODELS:
        m.to(DEVICE)
        acc = acc + np.exp(run_epoch_predict(m, tl))
        m.cpu()
    lp_test = np.log(acc / len(MODELS) + 1e-9)
    tst["pred"] = [decode(l) if DECODER == "beam" else greedy_decode(l) for l in lp_test]
    id_col = CFG.SUB_ID_COL or (SAMPLE_SUB.columns[0] if SAMPLE_SUB is not None else "id")
    txt_col = CFG.SUB_TEXT_COL or (SAMPLE_SUB.columns[1] if SAMPLE_SUB is not None and SAMPLE_SUB.shape[1] > 1 else "text")
    pred_by = {}
    for _, r in tst.iterrows():
        for key in (str(r["id"]), os.path.basename(str(r["path"])), os.path.splitext(os.path.basename(str(r["path"])))[0]):
            pred_by.setdefault(key, r["pred"])
    if SAMPLE_SUB is not None:
        sub = SAMPLE_SUB[[id_col]].copy()
        sub[txt_col] = sub[id_col].astype(str).map(lambda v: pred_by.get(v, pred_by.get(os.path.splitext(os.path.basename(v))[0], "")))
    else:
        sub = pd.DataFrame({id_col: tst["id"], txt_col: tst["pred"]})
    SUB_PATH = os.path.join(SUB_DIR, "submission.csv")
    sub.to_csv(SUB_PATH, index=False)
    n_empty = (sub[txt_col] == "").sum()
    print(f"✅ Submission: {SUB_PATH} | {len(sub)} baris | kosong: {n_empty}")
    if SAMPLE_SUB is not None and len(sub) != len(SAMPLE_SUB):
        print("⚠️ jumlah baris beda dengan sample_submission!")
    if n_empty:
        print("⚠️ ada prediksi kosong (id tidak cocok dengan file test, atau model memprediksi blank) — cek!")
    print("ℹ️ Kalau label diawali nol (\"007\"), buka CSV di editor teks — Excel menghapus nol di depan.")
else:
    print("ℹ️ Tidak ada gambar test → submission dilewati.")

# %% [markdown]
# # 13. Ringkasan untuk laporan

# %%
summ = {"model": f"CRNN (CNN {CFG.CNN_CHANNELS} + BiLSTM {CFG.RNN_LAYERS}x{CFG.RNN_HIDDEN}) + CTC", "input": f"{CFG.IMG_H}x{CFG.IMG_W}",
        "charset_size": len(CHARSET), "n_train": len(train_df), "folds": len(splits), "decoder": DECODER,
        "oof_exact_match": round(res[DECODER]["exact_match"], 4), "oof_cer": round(res[DECODER]["cer"], 4), "submission": SUB_PATH}
text = f"""# OCR summary

- {summ['model']}, input {summ['input']}, charset {summ['charset_size']} characters, {summ['folds']} fold(s).
- Out-of-fold exact match **{summ['oof_exact_match']:.4f}**, character error rate **{summ['oof_cer']:.4f}** ({DECODER} decoding).
- Most frequent confusions: {', '.join(err_tab['true→pred'].head(3)) if len(err_tab) else '-'}.

**Draft sentences (paraphrase):**
- "We trained a CRNN with CTC loss from scratch; text images were resized to a fixed height with preserved aspect ratio."
- "On held-out data the model reads {summ['oof_exact_match']*100:.1f}% of strings exactly (CER {summ['oof_cer']:.3f})."
"""
open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8").write(text)
json.dump(summ, open(os.path.join(OUT_DIR, "run_summary.json"), "w"), indent=2, default=str)
print(text)
print("✅ Notebook selesai.")

# %% [markdown]
# # 14. Troubleshooting
#
# | Gejala | Penyebab | Solusi |
# |---|---|---|
# | Loss `inf`/`nan` | label lebih panjang dari timesteps | `IMG_W` lebih besar (otomatis dicek), `zero_infinity` sudah aktif |
# | Val exact 0, prediksi kosong | belum konvergen / LR | tambah epoch, `LR=5e-4`, cek contoh label |
# | Nol di depan hilang (`"007"` → `7`) | CSV dibaca sebagai angka | sudah dibaca sebagai string; jangan buka/simpan CSV dengan Excel |
# | Huruf mirip tertukar (O/0, I/1) | ambigu visual | `TEXT_REGEX` (pola valid), charset terbatas, augmentasi lebih kuat |
# | Gambar berisi banyak baris | bukan tugas recognition | deteksi baris dulu (`cv_object_detection`) → crop |
# | Teks tulisan tangan/foto alami | CRNN dari nol kurang | `USE_TROCR=True` untuk cek; fine-tune TrOCR (recipe) |
#
# # 15. Recipes
#
# ```python
# CFG.FIXED_LENGTH, CFG.CHARSET = 6, "0123456789"            # meteran / kode digit tetap
# CFG.TEXT_REGEX = r"^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$"           # plat nomor Indonesia (tanpa spasi)
# CFG.IMG_H, CFG.IMG_W, CFG.CNN_CHANNELS = 48, 320, (64, 128, 256, 512)   # teks panjang / detail halus
# CFG.N_FOLDS, CFG.EPOCHS = 5, 80                             # final: ensemble 5 fold
# ```
# Fine-tune TrOCR: `VisionEncoderDecoderModel.from_pretrained(CFG.TROCR_MODEL)`, label → `processor.tokenizer(text).input_ids`,
# loss = output `.loss` dari model(pixel_values, labels); LR 5e-5, 5–10 epoch, batch 8–16 (GPU).
#
# # 16. Checklist
#
# - [ ] Contoh gambar + label cocok (label tidak bergeser baris)
# - [ ] Charset lengkap (karakter test yang mungkin muncul ada di train)
# - [ ] Nol di depan & huruf besar/kecil sesuai aturan lomba
# - [ ] Decoder dipilih dari OOF; metrik (exact / CER) sama dengan metrik lomba
# - [ ] Submission: jumlah baris = sample_submission, tidak ada prediksi kosong yang tak terduga
