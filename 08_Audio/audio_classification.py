# -*- coding: utf-8 -*-
# =====================================================================================================
# audio_classification.py — versi SCRIPT dari audio_classification.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python audio_classification.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb audio_classification.py
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
# # Audio Classification Template (+ open-set "Unknown")
#
# Untuk lomba klasifikasi suara: kicau burung (NPC Schematics 2026 — Bondol Jawa / Cendrawasih / Kutilang + *Unknown*),
# suara mesin/alarm, ucapan/perintah, deteksi suara sintetis (IOAI *Synthetic Speech Detector*), genre musik.
#
# ## Pipeline
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json)
# → Label dari CSV (file, label) atau folder per kelas; baca wav/flac/ogg/mp3 (soundfile → librosa → torchaudio → wave)
# → Resample + mono + potong/pad ke durasi tetap
# → EDA: durasi, sample rate asal, jumlah per kelas, contoh waveform & log-mel spectrogram
# → Fitur: log-mel (numpy, tanpa library tambahan) + MFCC & statistik spektral
# → Model A: statistik MFCC + LightGBM / Logistic Regression (cepat, kuat untuk data kecil)
# → Model B: CNN 2D pada log-mel (K-fold, SpecAugment, mixup, crop acak; inferensi sliding window)
# → Ensemble OOF, open-set: prediksi "Unknown" kalau keyakinan < threshold (dikalibrasi di OOF)
# → Evaluasi (F1 macro, confusion matrix), submission + validator, ringkasan laporan
# ```
#
# **GroupKFold**: kalau ada kolom sumber rekaman (lokasi/perekam/pembicara), isi `GROUP_COL` — klip dari rekaman yang
# sama tidak boleh ada di train & valid sekaligus (kalau tidak, skor CV terlalu optimis).
#
# **Dua format:** `audio_classification.ipynb` / `.py`. Headless: `python tools/run_task.py --task audio --format py`.

# %%
import os


class CFG:
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "audio_classification"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"
    TRAIN_CSV = "./data/train.csv"         # kolom file + label; tidak ada → folder per kelas di TRAIN_AUDIO_DIR
    TRAIN_AUDIO_DIR = "./data/train"
    TEST_CSV = None                        # opsional (daftar file test); None = semua file di TEST_AUDIO_DIR
    TEST_AUDIO_DIR = "./data/test"
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    FILE_COL = None                        # None = auto
    LABEL_COL = None                       # None = auto
    GROUP_COL = None                       # sumber rekaman / pembicara (GroupKFold)
    AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif")
    OUTPUT_DIR = "./outputs/audio_classification/"
    SR = 22050
    DURATION = None                        # detik per klip/jendela; None = median durasi (dibatasi 1–10 s)
    N_FFT = 1024
    HOP = 256
    N_MELS = 96
    FMIN = 20
    FMAX = None                            # None = SR/2
    N_MFCC = 20
    METRIC = "f1_macro"                    # "f1_macro" | "accuracy" | "roc_auc" (biner)
    MODELS = ["feat_lgb", "feat_logreg", "cnn"]
    N_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 32
    LR = 1e-3
    MIXUP = 0.2
    SPEC_AUG = True
    USE_GPU = True
    USE_AMP = True
    OPEN_SET = False                       # True = beri label UNKNOWN_LABEL bila keyakinan rendah
    UNKNOWN_LABEL = "Unknown"
    OPEN_SET_KEEP = 0.95                   # threshold = kuantil keyakinan sehingga 95% prediksi benar (OOF) tetap diterima
    SUBMISSION_MODE = "label"              # "label" | "proba"
    NUM_WORKERS = 0
    FAST_SAMPLE = 300
    FAST_EPOCHS = 6
    FAST_FOLDS = 2
    FIG_DPI = 150

# %%
# >>> USER OVERRIDE
# CFG.OPEN_SET = True
# CFG.GROUP_COL = "source"
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
# # 1. Install & import
#
# Fitur audio dihitung dengan numpy/scipy (tidak wajib librosa). `soundfile` dipasang untuk membaca flac/ogg;
# mp3/m4a butuh `librosa` (+ffmpeg) atau `torchaudio`. Torch (CNN) tidak di-install otomatis.

# %%
import importlib.util
import subprocess
import sys

for imp, pip_name in {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
                      "matplotlib": "matplotlib", "soundfile": "soundfile", "lightgbm": "lightgbm"}.items():
    if importlib.util.find_spec(imp) is None and CFG.AUTO_INSTALL:
        try:
            print("Installing", pip_name)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as e:  # noqa
            print("install gagal:", pip_name, e)

import json
import math
import random
import time
import warnings
import wave

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import matplotlib
import matplotlib.pyplot as plt
import scipy.fft
import scipy.signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except Exception:  # noqa
    display = print
HAS = {m: importlib.util.find_spec(m) is not None for m in ("soundfile", "librosa", "torchaudio", "lightgbm", "torch")}
if HAS["lightgbm"]:
    import lightgbm as lgb
if HAS["torch"]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
DEVICE = "cuda" if HAS["torch"] and CFG.USE_GPU and torch.cuda.is_available() else "cpu"
print("Library:", HAS, "| device:", DEVICE)
if CFG.RUN_MODE == "fast":
    CFG.EPOCHS, CFG.N_FOLDS = min(CFG.EPOCHS, CFG.FAST_EPOCHS), min(CFG.N_FOLDS, CFG.FAST_FOLDS)
    print(f"[FAST MODE] epochs {CFG.EPOCHS}, folds {CFG.N_FOLDS}, sample ≤ {CFG.FAST_SAMPLE}")
OUT = CFG.OUTPUT_DIR
FIG_DIR, TAB_DIR, SUB_DIR, MODEL_DIR = (os.path.join(OUT, d) for d in ("figures", "tables", "submissions", "models"))
for d in (OUT, FIG_DIR, TAB_DIR, SUB_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
if HAS["torch"]:
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
# # 2. Baca audio
#
# `load_audio()` mencoba beberapa pembaca berurutan dan selalu mengembalikan **mono float32** pada `CFG.SR`.
# File rusak/tidak terbaca → array kosong + dicatat di `LOAD_ERRORS` (tidak menghentikan notebook).

# %%
LOAD_ERRORS = []


def _read_raw(path):
    ext = os.path.splitext(path)[1].lower()
    if HAS["soundfile"]:
        try:
            import soundfile as sf
            y, sr = sf.read(path, dtype="float32", always_2d=True)
            return y.mean(1), sr
        except Exception:  # noqa
            pass
    if HAS["librosa"]:
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, mono=True)
            return y.astype(np.float32), sr
        except Exception:  # noqa
            pass
    if HAS["torchaudio"]:
        try:
            import torchaudio
            y, sr = torchaudio.load(path)
            return y.mean(0).numpy(), sr
        except Exception:  # noqa
            pass
    if ext == ".wav":
        with wave.open(path, "rb") as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            raw = np.frombuffer(w.readframes(n), dtype={1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
            raw = (raw - 128) / 128 if sw == 1 else raw / float(2 ** (8 * sw - 1))
            return raw.reshape(-1, ch).mean(1), sr
    raise RuntimeError(f"tidak ada pembaca untuk {ext} (pasang soundfile / librosa)")


_CACHE = {}


def load_audio(path):
    if path in _CACHE:
        return _CACHE[path]
    try:
        y, sr = _read_raw(path)
        if sr != CFG.SR and len(y):
            g = math.gcd(int(sr), int(CFG.SR))
            y = scipy.signal.resample_poly(y, CFG.SR // g, sr // g).astype(np.float32)
        y = np.nan_to_num(y)
    except Exception as e:  # noqa
        LOAD_ERRORS.append((path, str(e)[:120]))
        y = np.zeros(0, np.float32)
    if len(_CACHE) < 20000:
        _CACHE[path] = y
    return y

# %% [markdown]
# # 3. Data & label
#
# Kalau `TRAIN_CSV` tidak ada → label = nama subfolder (`data/train/<kelas>/*.wav`).

# %%
def index_files(folder):
    idx = {}
    if folder and os.path.isdir(folder):
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(tuple(CFG.AUDIO_EXTS)):
                    p = os.path.join(root, f)
                    idx.setdefault(f, p)
                    idx.setdefault(os.path.splitext(f)[0], p)
    return idx


TR_IDX = index_files(CFG.TRAIN_AUDIO_DIR)
if CFG.TRAIN_CSV and os.path.exists(CFG.TRAIN_CSV):
    raw = smart_read(CFG.TRAIN_CSV)
    fcol = CFG.FILE_COL or next((c for c in raw.columns if raw[c].astype(str).str.lower().str.endswith(tuple(CFG.AUDIO_EXTS)).mean() > .5),
                                next((c for c in raw.columns if c.lower() in ("filename", "file", "path", "audio", "id", "clip")), raw.columns[0]))
    lcol = CFG.LABEL_COL or next((c for c in raw.columns if c.lower() in ("label", "class", "species", "category", "target", "primary_label")),
                                 [c for c in raw.columns if c != fcol][0])
    train_df = pd.DataFrame({"path": raw[fcol].astype(str).map(lambda v: TR_IDX.get(os.path.basename(v)) or TR_IDX.get(os.path.splitext(os.path.basename(v))[0])),
                             "label": raw[lcol].astype(str)})
    if CFG.GROUP_COL and CFG.GROUP_COL in raw.columns:
        train_df["group"] = raw[CFG.GROUP_COL].astype(str)
    print(f"train.csv: file = '{fcol}', label = '{lcol}'")
else:
    paths = sorted({p for k, p in TR_IDX.items() if "." in k})
    train_df = pd.DataFrame({"path": paths, "label": [os.path.basename(os.path.dirname(p)) for p in paths]})
missing = train_df["path"].isna().sum()
train_df = train_df.dropna(subset=["path"]).reset_index(drop=True)
if CFG.RUN_MODE == "fast" and len(train_df) > CFG.FAST_SAMPLE:          # subsample stratified (min 2 per kelas)
    keep = []
    for _, g in train_df.groupby("label"):
        n = max(2, int(len(g) * CFG.FAST_SAMPLE / len(train_df)))
        keep += g.sample(min(n, len(g)), random_state=CFG.SEED).index.tolist()
    train_df = train_df.loc[sorted(keep)].reset_index(drop=True)
    print(f"[FAST MODE] train di-subsample ke {len(train_df)} klip")
CLASSES = sorted(train_df["label"].unique())
C2I = {c: i for i, c in enumerate(CLASSES)}
train_df["y"] = train_df["label"].map(C2I)
TE_IDX = index_files(CFG.TEST_AUDIO_DIR)
if CFG.TEST_CSV and os.path.exists(CFG.TEST_CSV):
    rt = smart_read(CFG.TEST_CSV)
    tcol = rt.columns[0]
    test_df = pd.DataFrame({"id": rt[tcol].astype(str)})
    test_df["path"] = test_df["id"].map(lambda v: TE_IDX.get(os.path.basename(v)) or TE_IDX.get(os.path.splitext(os.path.basename(v))[0]))
else:
    tp = sorted({p for k, p in TE_IDX.items() if "." in k})
    test_df = pd.DataFrame({"id": [os.path.basename(p) for p in tp], "path": tp})
print(f"Train: {len(train_df)} klip ({missing} file tidak ditemukan) | kelas ({len(CLASSES)}): {CLASSES} | test: {len(test_df)}")

# %% [markdown]
# # 4. EDA
#
# - Durasi sangat bervariasi → model memakai jendela `DURATION` detik; klip panjang diprediksi dengan sliding window.
# - Kelas tidak seimbang → metrik F1 macro & class weight (sudah default).

# %%
durs = train_df["path"].map(lambda p: len(load_audio(p)) / CFG.SR)
train_df["duration"] = durs
if CFG.DURATION is None:
    CFG.DURATION = float(np.clip(np.median(durs[durs > 0]), 1.0, 10.0))
SEG = int(CFG.DURATION * CFG.SR)
FMAX = CFG.FMAX or CFG.SR / 2
print(f"Durasi: median {durs.median():.2f}s | min {durs.min():.2f}s | max {durs.max():.2f}s → jendela {CFG.DURATION:.2f}s")
counts = train_df["label"].value_counts()
display(counts.to_frame("klip"))
save_table(counts.rename_axis("label").reset_index(name="clips"), "class_counts")
if LOAD_ERRORS:
    print(f"⚠️ {len(LOAD_ERRORS)} file gagal dibaca, contoh: {LOAD_ERRORS[:2]}")

# %% [markdown]
# # 5. Fitur: log-mel spectrogram & MFCC (numpy)
#
# **Log-mel**: energi per pita frekuensi skala mel (mirip pendengaran manusia) per frame waktu → "gambar" untuk CNN.
# **MFCC**: DCT dari log-mel (ringkasan bentuk spektrum) → statistik (mean/std/min/max) per klip untuk LightGBM.

# %%
def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        if b > a:
            fb[m - 1, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[m - 1, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


MEL_FB = mel_filterbank(CFG.SR, CFG.N_FFT, CFG.N_MELS, CFG.FMIN, FMAX)
WINDOW = scipy.signal.get_window("hann", CFG.N_FFT).astype(np.float32)


def power_spec(y):
    if len(y) < CFG.N_FFT:
        y = np.pad(y, (0, CFG.N_FFT - len(y)))
    fr = np.lib.stride_tricks.sliding_window_view(y, CFG.N_FFT)[::CFG.HOP] * WINDOW
    return (np.abs(np.fft.rfft(fr, axis=1)) ** 2).T.astype(np.float32)        # (freq, frames)


def logmel(y):
    m = MEL_FB @ power_spec(y)
    return 10 * np.log10(np.maximum(m, 1e-10)).astype(np.float32)               # dB (absolut, tanpa normalisasi per klip)


def fix_length(y, n, rng=None):
    if len(y) >= n:
        s = rng.integers(0, len(y) - n + 1) if rng is not None else (len(y) - n) // 2
        return y[s:s + n]
    return np.pad(y, (0, n - len(y)))


def clip_features(y):
    if len(y) == 0:
        y = np.zeros(CFG.N_FFT, np.float32)
    S = power_spec(y)
    lm = 10 * np.log10(np.maximum(MEL_FB @ S, 1e-10))
    mf = scipy.fft.dct(lm, type=2, axis=0, norm="ortho")[:CFG.N_MFCC]
    d1 = np.diff(mf, axis=1) if mf.shape[1] > 1 else np.zeros_like(mf)
    freqs = np.linspace(0, CFG.SR / 2, S.shape[0])[:, None]
    tot = S.sum(0) + 1e-10
    cent = (freqs * S).sum(0) / tot
    bw = np.sqrt((((freqs - cent) ** 2) * S).sum(0) / tot)
    roll = freqs[np.minimum((np.cumsum(S, 0) < 0.85 * tot).sum(0), S.shape[0] - 1), 0]
    zcr = np.abs(np.diff(np.sign(y))).mean() / 2
    rms = np.sqrt((np.lib.stride_tricks.sliding_window_view(np.pad(y, (0, max(0, CFG.N_FFT - len(y)))), CFG.N_FFT)[::CFG.HOP] ** 2).mean(1))
    stats = lambda a: np.concatenate([a.mean(1), a.std(1), a.min(1), a.max(1)])  # noqa: E731
    return np.concatenate([stats(mf), d1.mean(1), d1.std(1), stats(lm[::8]),
                           [cent.mean(), cent.std(), bw.mean(), bw.std(), roll.mean(), roll.std(), zcr, rms.mean(), rms.std(),
                            rms.max(), len(y) / CFG.SR]]).astype(np.float32)


t0 = time.time()
X_feat = np.stack([clip_features(load_audio(p)) for p in train_df["path"]])
X_feat_te = np.stack([clip_features(load_audio(p)) for p in test_df["path"]]) if len(test_df) else None
print(f"Fitur statistik: {X_feat.shape} ({time.time() - t0:.0f}s)")
k = min(4, len(CLASSES))
fig, axes = plt.subplots(2, k, figsize=(3.4 * k, 5))
axes = np.atleast_2d(axes).reshape(2, k)
for j, c in enumerate(CLASSES[:k]):
    p = train_df[train_df.label == c]["path"].iloc[0]
    y_ = load_audio(p)
    axes[0, j].plot(np.arange(len(y_)) / CFG.SR, y_, lw=.4)
    axes[0, j].set_title(c, fontsize=9)
    axes[1, j].imshow(logmel(y_), aspect="auto", origin="lower", cmap="magma")
    axes[1, j].set_xlabel("frame")
axes[0, 0].set_ylabel("waveform")
axes[1, 0].set_ylabel("mel bin")
save_fig("waveform_logmel_examples", fig)

# %% [markdown]
# # 6. Cross-validation folds
#
# StratifiedGroupKFold kalau `GROUP_COL` ada (rekaman/pembicara sama tidak bocor ke valid), selain itu StratifiedKFold.

# %%
y_all = train_df["y"].values
n_splits = max(2, min(CFG.N_FOLDS, int(np.bincount(y_all).min())))
if "group" in train_df:
    folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=CFG.SEED).split(X_feat, y_all, train_df["group"]))
    print(f"StratifiedGroupKFold ({n_splits}) pada {train_df['group'].nunique()} grup")
else:
    folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=CFG.SEED).split(X_feat, y_all))
    print(f"StratifiedKFold ({n_splits})")
N_CLS = len(CLASSES)
OOF, TEST = {}, {}


def metric(y, p):
    if CFG.METRIC == "accuracy":
        return accuracy_score(y, p.argmax(1))
    if CFG.METRIC == "roc_auc" and N_CLS == 2:
        return roc_auc_score(y, p[:, 1])
    return f1_score(y, p.argmax(1), average="macro")

# %% [markdown]
# # 7. Model A — fitur statistik + LightGBM / Logistic Regression

# %%
def run_feat(kind):
    oof = np.zeros((len(X_feat), N_CLS))
    te = np.zeros((len(X_feat_te), N_CLS)) if X_feat_te is not None else None
    for tr, va in folds:
        if kind == "feat_lgb":
            m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.04, num_leaves=15, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.6, class_weight="balanced", random_state=CFG.SEED, verbose=-1)
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
        m.fit(X_feat[tr], y_all[tr])
        oof[va] = m.predict_proba(X_feat[va])
        if te is not None:
            te += m.predict_proba(X_feat_te) / len(folds)
    return oof, te


for kind in [m for m in CFG.MODELS if m.startswith("feat_")]:
    if kind == "feat_lgb" and not HAS["lightgbm"]:
        print("skip feat_lgb (lightgbm tidak ada)")
        continue
    t0 = time.time()
    OOF[kind], TEST[kind] = run_feat(kind)
    print(f"==> {kind}: OOF {CFG.METRIC} = {metric(y_all, OOF[kind]):.4f} ({time.time() - t0:.0f}s)")

# %% [markdown]
# # 8. Model B — CNN pada log-mel
#
# Training: jendela `DURATION` detik diambil **acak** dari klip (augmentasi gratis), SpecAugment (masking waktu &
# frekuensi), mixup. Inferensi: jendela berurutan (overlap 50%) → rata-rata probabilitas.

# %%
if HAS["torch"]:
    class AudioCNN(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            ch = [1, 32, 64, 128, 256]
            self.blocks = nn.Sequential(*[nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, padding=1, bias=False), nn.BatchNorm2d(ch[i + 1]),
                                                        nn.ReLU(inplace=True), nn.Conv2d(ch[i + 1], ch[i + 1], 3, padding=1, bias=False),
                                                        nn.BatchNorm2d(ch[i + 1]), nn.ReLU(inplace=True), nn.AvgPool2d(2))
                                          for i in range(4)])
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, n_cls))

        def forward(self, x):
            f = self.blocks(x)
            f = f.mean(2)                                   # rata-rata frekuensi
            f = f.max(-1).values + f.mean(-1)               # max + mean waktu
            return self.head(f)

    DB_MEAN, DB_STD = None, None

    def mel_tensor(y):
        m = logmel(y)
        return torch.from_numpy((m - DB_MEAN) / DB_STD)[None]

    def windows(y):
        if len(y) <= SEG:
            return [fix_length(y, SEG)]
        step = SEG // 2
        return [y[s:s + SEG] for s in range(0, len(y) - SEG + 1, step)]

    def spec_augment(x):
        b, _, f, t = x.shape
        for i in range(b):
            for _ in range(2):
                w = random.randint(0, f // 8)
                f0 = random.randint(0, f - w)
                x[i, :, f0:f0 + w] = 0
                w = random.randint(0, t // 8)
                t0_ = random.randint(0, t - w)
                x[i, :, :, t0_:t0_ + w] = 0
        return x

    def predict_clips(model, paths):
        model.eval()
        out = []
        with torch.no_grad():
            for p in paths:
                xs = torch.stack([mel_tensor(w) for w in windows(load_audio(p))]).to(DEVICE)
                with torch.autocast(device_type="cuda", enabled=CFG.USE_AMP and DEVICE == "cuda"):
                    pr = torch.softmax(model(xs).float(), 1).mean(0)
                out.append(pr.cpu().numpy())
        return np.stack(out) if out else np.zeros((0, N_CLS))

    def run_cnn():
        global DB_MEAN, DB_STD
        sample_m = np.concatenate([logmel(fix_length(load_audio(p), SEG)).ravel() for p in train_df["path"].sample(min(200, len(train_df)), random_state=0)])
        DB_MEAN, DB_STD = float(sample_m.mean()), float(sample_m.std() + 1e-6)
        oof = np.zeros((len(train_df), N_CLS))
        te = np.zeros((len(test_df), N_CLS))
        rng = np.random.default_rng(CFG.SEED)
        cw = torch.tensor(len(y_all) / (N_CLS * np.bincount(y_all, minlength=N_CLS).clip(min=1)), dtype=torch.float32, device=DEVICE)
        for f, (tr, va) in enumerate(folds):
            torch.manual_seed(CFG.SEED + f)
            model = AudioCNN(N_CLS).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=1e-3)
            steps = CFG.EPOCHS * math.ceil(len(tr) / CFG.BATCH_SIZE)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG.LR, total_steps=max(1, steps), pct_start=0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=CFG.USE_AMP and DEVICE == "cuda")
            best, best_state = -1, None
            for ep in range(CFG.EPOCHS):
                model.train()
                idx = rng.permutation(tr)
                losses = []
                for i in range(0, len(idx), CFG.BATCH_SIZE):
                    b = idx[i:i + CFG.BATCH_SIZE]
                    x = torch.stack([mel_tensor(fix_length(load_audio(train_df.path[j]), SEG, rng)) for j in b]).to(DEVICE)
                    yb = F.one_hot(torch.tensor(y_all[b], device=DEVICE), N_CLS).float()
                    if CFG.SPEC_AUG:
                        x = spec_augment(x)
                    if CFG.MIXUP and len(b) > 1:
                        lam = float(np.random.beta(CFG.MIXUP, CFG.MIXUP))
                        pi = torch.randperm(len(b), device=DEVICE)
                        x, yb = lam * x + (1 - lam) * x[pi], lam * yb + (1 - lam) * yb[pi]
                    with torch.autocast(device_type="cuda", enabled=CFG.USE_AMP and DEVICE == "cuda"):
                        lo = model(x)
                    loss = -(yb * F.log_softmax(lo.float(), 1) * cw).sum(1).mean()
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    if sched.last_epoch < steps - 1:
                        sched.step()
                    losses.append(loss.item())
                pv = predict_clips(model, train_df.path.values[va])
                s = metric(y_all[va], pv)
                star = s > best
                if star:
                    best, best_state = s, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
                print(f"  fold {f} ep {ep + 1:02d} loss {np.mean(losses):.4f} | val {CFG.METRIC} {s:.4f}{' *' if star else ''}")
            model.load_state_dict(best_state)
            torch.save(best_state, os.path.join(MODEL_DIR, f"cnn_fold{f}.pt"))
            oof[va] = predict_clips(model, train_df.path.values[va])
            if len(test_df):
                te += predict_clips(model, test_df.path.values) / len(folds)
        return oof, te

    if "cnn" in CFG.MODELS:
        t0 = time.time()
        OOF["cnn"], TEST["cnn"] = run_cnn()
        print(f"==> cnn: OOF {CFG.METRIC} = {metric(y_all, OOF['cnn']):.4f} ({time.time() - t0:.0f}s)")
else:
    print("ℹ️ torch tidak ada → CNN dilewati (Model A tetap jalan).")

# %% [markdown]
# # 9. Ensemble, evaluasi, open-set

# %%
board = pd.DataFrame([{"model": k_, CFG.METRIC: metric(y_all, v), "accuracy": accuracy_score(y_all, v.argmax(1))} for k_, v in OOF.items()])
if len(OOF) > 1:
    ens = np.mean(list(OOF.values()), 0)
    board.loc[len(board)] = {"model": "ensemble_mean", CFG.METRIC: metric(y_all, ens), "accuracy": accuracy_score(y_all, ens.argmax(1))}
    OOF["ensemble_mean"] = ens
    if len(test_df):
        TEST["ensemble_mean"] = np.mean([TEST[k_] for k_ in OOF if k_ != "ensemble_mean"], 0)
board = board.sort_values(CFG.METRIC, ascending=False).reset_index(drop=True)
display(board.round(4))
save_table(board, "leaderboard")
FINAL = board["model"].iloc[0]
P_OOF = OOF[FINAL]
print(f"Model final: {FINAL}")
print(classification_report(y_all, P_OOF.argmax(1), target_names=CLASSES, digits=3))
cm = confusion_matrix(y_all, P_OOF.argmax(1))
fig, ax = plt.subplots(figsize=(1.2 + .6 * N_CLS, 1 + .6 * N_CLS))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(N_CLS), CLASSES, rotation=45, fontsize=8)
ax.set_yticks(range(N_CLS), CLASSES, fontsize=8)
for i in range(N_CLS):
    for j in range(N_CLS):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
ax.set_xlabel("prediksi")
ax.set_ylabel("sebenarnya")
ax.set_title(f"Confusion matrix OOF ({FINAL})")
save_fig("confusion_matrix", fig)

OPEN_TH = None
if CFG.OPEN_SET:
    conf = P_OOF.max(1)
    correct = P_OOF.argmax(1) == y_all
    OPEN_TH = float(np.quantile(conf[correct], 1 - CFG.OPEN_SET_KEEP)) if correct.any() else 0.5
    print(f"Open-set: prediksi dengan keyakinan < {OPEN_TH:.4g} → '{CFG.UNKNOWN_LABEL}' "
          f"(di OOF, {(conf < OPEN_TH).mean():.1%} klip akan menjadi Unknown)")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist([conf[correct], conf[~correct]], bins=30, label=["benar", "salah"], density=True)
    ax.axvline(OPEN_TH, c="k", ls="--")
    ax.legend()
    ax.set_title("Keyakinan prediksi OOF (threshold Unknown)")
    save_fig("open_set_threshold", fig)

# %% [markdown]
# # 10. Submission + ringkasan

# %%
SUB_PATH = None
if len(test_df) and FINAL in TEST:
    P = TEST[FINAL]
    pred = np.array(CLASSES, dtype=object)[P.argmax(1)]
    if OPEN_TH is not None:
        pred = np.where(P.max(1) < OPEN_TH, CFG.UNKNOWN_LABEL, pred)
    sample = smart_read(CFG.SAMPLE_SUB_PATH) if os.path.exists(CFG.SAMPLE_SUB_PATH) else None
    if CFG.SUBMISSION_MODE == "proba":
        sub = pd.DataFrame(P, columns=CLASSES)
        sub.insert(0, "id", test_df["id"].values)
    else:
        sub = pd.DataFrame({"id": test_df["id"].values, "label": pred})
    if sample is not None:
        id_name = sample.columns[0]
        key = lambda v: os.path.splitext(os.path.basename(str(v)))[0]  # noqa: E731
        m = dict(zip(test_df["id"].map(key), range(len(test_df))))
        order = sample[id_name].map(lambda v: m.get(key(v)))
        if order.isna().any():
            print(f"⚠️ {order.isna().sum()} id sample_submission tanpa file test yang cocok")
        sub = sub.iloc[order.fillna(0).astype(int).values].reset_index(drop=True)
        sub.iloc[:, 0] = sample[id_name].values
        sub.columns = [id_name] + list(sub.columns[1:]) if CFG.SUBMISSION_MODE == "proba" else list(sample.columns[:2])
    SUB_PATH = os.path.join(SUB_DIR, "submission.csv")
    sub.to_csv(SUB_PATH, index=False)
    print(f"✅ Submission: {SUB_PATH} | {len(sub)} baris | distribusi: {pd.Series(pred).value_counts().to_dict()}")
summ = {"final_model": FINAL, "classes": CLASSES, "n_train": len(train_df), "window_s": CFG.DURATION, "sr": CFG.SR,
        f"oof_{CFG.METRIC}": round(float(board[CFG.METRIC].iloc[0]), 4), "open_set_threshold": OPEN_TH, "submission": SUB_PATH}
text = f"""# Audio classification summary

- {len(train_df)} clips, {N_CLS} classes, {CFG.DURATION:.1f}-second windows at {CFG.SR} Hz; {'grouped ' if 'group' in train_df else ''}{len(folds)}-fold CV.
- Final model **{FINAL}**: OOF {CFG.METRIC} **{summ[f'oof_{CFG.METRIC}']}**. Leaderboard: {', '.join(f"{r.model} {getattr(r, CFG.METRIC):.3f}" for r in board.itertuples())}.
{'- Open-set: clips with confidence below ' + f'{OPEN_TH:.4g}' + ' are labelled ' + CFG.UNKNOWN_LABEL + '.' if OPEN_TH is not None else ''}

**Draft sentences (paraphrase):**
- "Each recording was converted to a log-mel spectrogram; we compared MFCC summary statistics with gradient boosting against a CNN trained on random crops with SpecAugment and mixup."
- "Long recordings were scored with overlapping windows and averaged probabilities."
"""
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(text)
json.dump(summ, open(os.path.join(OUT, "run_summary.json"), "w"), indent=2, default=str)
print(text)
print("✅ Notebook selesai.")

# %% [markdown]
# # 11. Troubleshooting, recipes, checklist
#
# | Gejala | Solusi |
# |---|---|
# | `tidak ada pembaca untuk .mp3` | `pip install librosa` (+ffmpeg) atau konversi ke wav |
# | CV tinggi, leaderboard rendah | klip satu rekaman tersebar di fold → isi `GROUP_COL` |
# | Kelas jarang selalu salah | class weight sudah aktif; tambah mixup/augment, `METRIC="f1_macro"` |
# | Banyak Unknown salah | `OPEN_SET_KEEP=0.98` (threshold lebih rendah) |
# | Lambat | `MODELS=["feat_lgb"]`, kecilkan `N_MELS`, `DURATION` |
#
# ```python
# CFG.SR, CFG.N_MELS, CFG.FMIN = 32000, 128, 50       # burung (frekuensi tinggi)
# CFG.SR, CFG.N_MELS = 16000, 64                       # ucapan / suara mesin
# CFG.OPEN_SET, CFG.UNKNOWN_LABEL = True, "Unknown"   # kelas asing di test
# ```
#
# - [ ] Sample rate & durasi masuk akal (cek contoh spectrogram)
# - [ ] Tidak ada kebocoran rekaman antar fold (GROUP_COL)
# - [ ] Label Unknown & nama kelas persis seperti aturan
# - [ ] Jumlah baris & urutan ID submission = sample_submission
