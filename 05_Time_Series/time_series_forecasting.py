# -*- coding: utf-8 -*-
# =====================================================================================================
# time_series_forecasting.py — versi SCRIPT dari time_series_forecasting.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python time_series_forecasting.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb time_series_forecasting.py
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
# # Time Series Forecasting — Competition Template (Single Series & Panel)
#
# Notebook ini adalah template **lengkap dan mandiri** untuk lomba forecasting deret waktu (time series).
# Semua langkah ada di sini: load data → cleaning → EDA → validasi backtest → baseline → model statistik →
# model ML global (LightGBM/XGBoost/CatBoost/HistGB) → ensemble → prediction interval → submission → bahan laporan.
#
# **Yang didukung:**
# - **Single series** (satu deret, mis. penjualan harian total) dan **multiple series / panel** (mis. 5 toko × 50 item) lewat `CFG.SERIES_ID_COLS`.
# - Frekuensi **hourly / daily / business-day / weekly / monthly / quarterly** — di-infer otomatis (`pd.infer_freq` + fallback).
# - **Exogenous features** (variabel luar yang diketahui di masa depan: promo, harga, hari libur) lewat `CFG.EXOG_COLS`.
# - Dua skenario output: (a) **ada file test** berisi tanggal masa depan (+ ID) → submission dibuat sesuai format;
#   (b) **tidak ada file test** → notebook memprediksi **H langkah ke depan** dari akhir data.
#
# **Cara pakai singkat:**
# 1. Taruh data di `./data/` (`train.csv`, `test.csv`, `sample_submission.csv`) atau ubah path di cell **USER OVERRIDE**.
# 2. Isi minimal: `CFG.TARGET_COL`, `CFG.DATE_COL` (atau biarkan `None` = auto detect), `CFG.SERIES_ID_COLS` (kosong = single series), `CFG.ID_COL`.
# 3. Jalankan dulu dengan `CFG.RUN_MODE = "fast"` (cepat, untuk cek error). Kalau semua jalan → ganti `"full"` untuk hasil final.
# 4. Lihat leaderboard backtest → pilih model/ensemble terbaik → submission otomatis tersimpan di `outputs/.../submissions/`.
#
# **Prinsip utama time series (WAJIB dipahami):**
# - **Jangan pernah random K-Fold.** Data masa depan tidak boleh dipakai untuk memprediksi masa lalu. Kita pakai **backtesting**
#   (train sampai tanggal *cutoff*, prediksi H langkah setelahnya, ulangi untuk beberapa cutoff).
# - **Selalu bandingkan dengan baseline** (naive / seasonal naive). Kalau model canggih kalah dari seasonal naive → ada yang salah.
# - Fitur lag/rolling harus dihitung **hanya dari data masa lalu** (shift dulu baru rolling) supaya tidak leakage.
#
# **Struktur notebook:**
# 0. Config & override → 1. Install & import → 2. Helper → 3. Load data → 4. Data preparation → 5. EDA →
# 6. Desain validasi (backtest) → 7. Metrik → 8. Model (baseline, statistik, ML) → 9. Backtest & leaderboard →
# 10. Ensemble → 11. Plot & diagnostik residual → 12. Feature importance/SHAP → 13. Final forecast + interval →
# 14. Submission + validator → 15. Ringkasan untuk laporan → 16. Troubleshooting → 17. Recipes → 18. Checklist.

# %% [markdown]
# ## 0. Konfigurasi (CFG)
#
# **Tujuan:** semua "knob" ada di satu tempat. Jangan edit angka di tengah notebook — ubah di sini atau di cell **USER OVERRIDE** di bawahnya.
#
# **Cara membaca:** setiap baris punya komentar: apa fungsinya, kapan diubah, nilai umum.
#
# **Jika X maka Y (cepat):**
# - Jika data **harian penjualan toko** → `FILL_METHOD="zero"` (hari tanpa transaksi = 0), `METRIC` sesuai lomba (sering RMSE/MAE/WAPE/RMSLE).
# - Jika data **harga / suhu / stok** (nilai kontinu, tidak boleh 0 sembarangan) → `FILL_METHOD="interpolate"` atau `"ffill"`.
# - Jika target **tidak pernah negatif & skewed** (penjualan) → biarkan `TARGET_TRANSFORM="auto"` (otomatis log1p).
# - Jika ada **trend kuat** dan model tree (LightGBM) under-predict di akhir → coba `TARGET_TRANSFORM="diff"` atau `"log1p_diff"`.
# - Jika data **besar** (ratusan+ series) → `STAT_MAX_SERIES` kecil (mis. 20), andalkan model ML global.
# - Jika lomba memberi file test dengan **tanggal mulai beberapa hari setelah train berakhir** → `GAP` otomatis terdeteksi.

# %%
import os

class CFG:
    # ===================== MODE =====================
    # "fast" = debug cepat (window backtest lebih sedikit, model lebih sedikit, iterasi kecil).
    # "full" = run final untuk submission. Selalu jalankan "fast" dulu sampai tidak ada error.
    RUN_MODE = "full"
    SEED = 42
    # Install otomatis library yang belum ada (pip). Set False kalau tidak ada internet / tidak boleh install.
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"
    # Library opsional berat (prophet, statsforecast). Default False karena install-nya lama/berat. (env TEMPLATE_AUTO_INSTALL=0 → off)
    INSTALL_OPTIONAL = False

    # ===================== PATHS =====================
    NOTEBOOK_NAME = "time_series_forecasting"
    TRAIN_PATH = "./data/train.csv"
    TEST_PATH = "./data/test.csv"                 # boleh tidak ada -> forecast H langkah ke depan
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    OUTPUT_DIR = None                             # None -> ./outputs/<NOTEBOOK_NAME>/

    # ===================== KOLOM =====================
    DATE_COL = None              # None = auto detect (kolom bernama date/ds/tanggal/time/... atau yang bisa di-parse jadi tanggal)
    TARGET_COL = "sales"         # kolom yang diprediksi
    SERIES_ID_COLS = []          # [] = single series. Panel: mis. ["store"] atau ["store", "item"]
    ID_COL = "id"                # kolom ID baris di test/sample_submission (None kalau tidak ada)
    # Exogenous: kolom tambahan yang NILAINYA DIKETAHUI DI MASA DEPAN (ada di test), mis. promo, harga, libur.
    # [] = tidak pakai. "auto" = semua kolom non-target yang ada di train DAN test.
    EXOG_COLS = []
    DATE_FORMAT = None           # mis. "%d/%m/%Y" kalau parsing otomatis salah. None = auto.
    DAYFIRST = False             # True kalau format tanggal Indonesia 31/12/2024 dan DATE_FORMAT None.

    # ===================== FREKUENSI & HORIZON =====================
    FREQ = None                  # None = auto infer. Contoh manual: "D", "W-MON", "MS", "h", "B".
    RESAMPLE_RULE = None         # Ubah level agregasi, mis. data harian -> mingguan "W-MON" / bulanan "MS". None = tidak.
    RESAMPLE_AGG = "sum"         # "sum" (penjualan) | "mean" (suhu/harga) | "last" (stok/saldo)
    HORIZON = None               # None = dari file test, atau default per frekuensi (D:30, W:12, M:12, h:48)
    SEASON_LENGTH = None         # None = auto (D:7, W:52, M:12, h:24, B:5, Q:4)

    # ===================== CLEANING =====================
    DUPLICATE_AGG = "sum"        # timestamp duplikat per series diagregasi: "sum" | "mean" | "last" | "max"
    FILL_METHOD = "zero"         # timestamp hilang: "zero" | "ffill" | "interpolate" | "seasonal" | "mean" | "none"
    REINDEX_TO_GLOBAL_END = True # True: semua series di-reindex sampai tanggal terakhir global (series yang berhenti diisi FILL_METHOD)
    NEGATIVE_POLICY = "keep"     # nilai target negatif: "keep" | "clip0" (jadi 0) | "nan" (dianggap missing lalu diisi)
    OUTLIER_METHOD = None        # None | "iqr" | "zscore" | "hampel" (rolling median, paling aman untuk TS)
    OUTLIER_THRESHOLD = 3.0      # iqr: k*IQR (3 = konservatif), zscore/hampel: jumlah std/MAD
    OUTLIER_ACTION = "clip"      # "clip" (potong ke batas) | "interpolate" (ganti nilai interpolasi)
    PANEL_AGG_FOR_EDA = "sum"    # cara menggabung panel jadi 1 deret untuk EDA: "sum" | "mean"

    # ===================== VALIDASI (BACKTEST) =====================
    N_BACKTEST_WINDOWS = 3       # jumlah cutoff. 3-5 umum. Lebih banyak = estimasi lebih stabil tapi lebih lama.
    BACKTEST_STEP = None         # jarak antar cutoff (langkah). None = HORIZON (window tidak overlap).
    WINDOW_TYPE = "expanding"    # "expanding" (train makin panjang) | "rolling" (panjang train tetap)
    ROLLING_TRAIN_SIZE = None    # dipakai kalau WINDOW_TYPE="rolling", mis. 365 (langkah). None = 3*HORIZON + 2*season.
    GAP = None                   # None = auto (dari jarak akhir train ke awal test). Int = jumlah langkah jeda manual.
    EVAL_EXCLUDE_IMPUTED = True  # True: titik yang hasil imputasi (timestamp hilang) tidak dihitung di metrik

    # ===================== METRIK =====================
    # "mae" | "rmse" | "mape" | "smape" | "wape" | "mase" | "rmsle" | "auto" (=wape). SAMAKAN dengan metrik lomba!
    METRIC = "auto"

    # ===================== MODEL =====================
    # Baseline: naive, snaive, ma, drift
    # Statistik (per series, butuh statsmodels): ses, holt_winters, ets, theta, sarimax
    # ML global (satu model untuk semua series): lgb, xgb, cat, hgb
    # Opsional (kalau terinstall): prophet, sf_autoets, sf_autoarima
    MODELS = ["naive", "snaive", "ma", "drift", "ses", "holt_winters", "ets", "theta", "sarimax",
              "lgb", "xgb", "cat", "hgb", "prophet", "sf_autoets", "sf_autoarima"]
    FAST_MODELS = ["naive", "snaive", "ma", "drift", "holt_winters", "ets", "theta", "sarimax", "lgb", "hgb"]
    MA_WINDOW = None             # None = season length
    HW_SEASONAL = "add"          # Holt-Winters seasonal "add" | "mul" (mul hanya kalau data > 0)
    STAT_MAX_SERIES = 200        # model statistik dijalankan maks untuk N series terbesar; sisanya pakai snaive
    STAT_MAX_TRAIN = 1500        # model statistik hanya pakai N titik terakhir (lebih cepat)
    SARIMA_MAX_P = 2             # grid p in [0..MAX_P]
    SARIMA_MAX_Q = 2             # grid q in [0..MAX_Q]
    SARIMA_MAX_PS = 1            # grid P seasonal
    SARIMA_MAX_QS = 1            # grid Q seasonal
    SARIMA_MAX_SEASON = 24       # season length > ini -> SARIMA tanpa komponen seasonal (terlalu lambat)
    SARIMA_MAX_TRAIN = 500       # SARIMA hanya pakai N titik terakhir
    SARIMA_TIME_LIMIT = 30       # detik per series untuk grid search (guard supaya tidak hang)

    # ----- ML global -----
    ML_STRATEGY = "recursive"    # "recursive" (1 model, prediksi dipakai sbg lag) | "direct" (model per bucket horizon)
    DIRECT_N_BUCKETS = None      # direct: jumlah model. None = min(H, 6). = H -> direct murni (paling akurat, paling lama)
    LAGS = None                  # None = auto dari season length, mis. daily [1..7,14,21,28]
    ROLL_WINDOWS = None          # None = auto, mis. daily [7,14,28]
    USE_EWM = True               # fitur exponential weighted mean
    USE_DATE_FEATURES = True
    USE_HOLIDAY_FEATURES = True  # libur nasional Indonesia + jarak ke Lebaran + Ramadan
    USE_SERIES_STATS = True      # mean/std per series (dari data train window saja) sebagai fitur level
    TARGET_TRANSFORM = "auto"    # "none" | "log1p" | "diff" | "log1p_diff" | "auto" (log1p kalau >=0 & skewed)
    ML_N_ESTIMATORS = 800
    ML_LEARNING_RATE = 0.03
    ML_NUM_LEAVES = 31
    ML_MAX_TRAIN_ROWS = 2_000_000  # kalau lebih, ambil baris paling baru
    ML_N_JOBS = -1

    # ===================== HOLIDAYS =====================
    COUNTRY_HOLIDAYS = "ID"      # "ID" = tabel libur Indonesia bawaan. None = tidak pakai
    CUSTOM_HOLIDAYS = {}         # tambahan: {"2024-02-14": "Pemilu", "2024-11-27": "Pilkada"}

    # ===================== ENSEMBLE & FINAL =====================
    ENSEMBLE_TOP_K = 3           # gabungkan K model terbaik
    FINAL_STRATEGY = "auto"      # "auto" (ensemble kalau lebih baik) | "best" | "ensemble" | nama model, mis. "lgb"

    # ===================== PREDICTION INTERVAL =====================
    PI_LEVEL = 0.90              # 0.8 / 0.9 / 0.95
    PI_METHOD = "both"           # "residual" (dari error backtest) | "quantile_lgb" | "both" | None

    # ===================== SUBMISSION =====================
    SUBMISSION_PRED_COL = None   # None = ambil dari sample_submission (kolom non-ID) atau TARGET_COL
    SUBMISSION_FILE = "submission.csv"
    CLIP_NEGATIVE_PRED = "auto"  # "auto" = clip ke 0 kalau target historis tidak pernah negatif
    ROUND_PRED = False           # True kalau target harus integer (jumlah unit)

    # ===================== FIGURES / EXPLAIN =====================
    FIG_DPI = 200                # 300 untuk laporan
    MAX_SERIES_PLOT = 6
    USE_SHAP = True
    SHAP_SAMPLE = 2000

# %% [markdown]
# ### 0.1 USER OVERRIDE — edit cell ini untuk lomba spesifik
#
# Cell di bawah sengaja hanya berisi komentar. Tulis override di sini (path, kolom, metrik, mode).
# Contoh lengkap:
# ```python
# CFG.TRAIN_PATH = "./data/train.csv"
# CFG.TEST_PATH = "./data/test.csv"
# CFG.SAMPLE_SUB_PATH = "./data/sample_submission.csv"
# CFG.DATE_COL = "date"
# CFG.TARGET_COL = "sales"
# CFG.SERIES_ID_COLS = ["store_id", "product_id"]
# CFG.ID_COL = "id"
# CFG.EXOG_COLS = ["promo", "price"]
# CFG.METRIC = "rmse"
# CFG.RUN_MODE = "fast"
# ```

# %%
# >>> USER OVERRIDE
# Tulis override CFG di bawah ini. Contoh:
# CFG.TARGET_COL = "sales"
# CFG.SERIES_ID_COLS = ["store"]
# CFG.RUN_MODE = "fast"

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
# ## 1. Install & Import
#
# **Tujuan:** memastikan library tersedia. Library opsional (lightgbm, xgboost, catboost, statsmodels, shap, prophet, statsforecast)
# di-import dalam `try/except` → kalau tidak ada, flag `HAS_xxx = False` dan bagian terkait **di-skip dengan pesan**, notebook tetap jalan.
#
# **Cara membaca output:** daftar `✅ ada` / `⚠️ tidak ada`. Yang penting: **statsmodels** (model statistik + EDA STL/ACF/ADF) dan
# **lightgbm** (model ML terbaik biasanya). Kalau keduanya tidak ada, notebook tetap jalan dengan baseline + HistGB (sklearn).
#
# **Jika error install:** set `CFG.AUTO_INSTALL = False`, lalu install manual di terminal: `pip install statsmodels lightgbm`.
# Torch tidak dipakai di notebook ini.

# %%
import sys
import subprocess
import importlib
import importlib.util

PIP_NAME = {"sklearn": "scikit-learn", "statsmodels": "statsmodels", "lightgbm": "lightgbm", "xgboost": "xgboost",
            "catboost": "catboost", "shap": "shap", "seaborn": "seaborn", "scipy": "scipy", "matplotlib": "matplotlib",
            "pandas": "pandas", "numpy": "numpy", "prophet": "prophet", "statsforecast": "statsforecast",
            "joblib": "joblib"}


def ensure_packages(import_names, auto_install=True):
    """Cek library via importlib.util.find_spec; install yang hilang kalau auto_install=True.
    Tidak pernah membuat notebook gagal: semua error install hanya di-print."""
    missing = [n for n in import_names if importlib.util.find_spec(n) is None]
    if not missing:
        print("Semua package inti sudah tersedia.")
        return []
    print("Package belum ada:", missing)
    if not auto_install:
        print("AUTO_INSTALL=False -> skip install. Install manual: pip install " + " ".join(PIP_NAME.get(m, m) for m in missing))
        return missing
    for name in missing:
        pkg = PIP_NAME.get(name, name)
        try:
            print(f"  installing {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *([] if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else ["--user"]), pkg])
        except Exception as e:  # jangan sampai gagal
            print(f"  ⚠️ gagal install {pkg}: {e}")
    importlib.invalidate_caches()
    return [n for n in missing if importlib.util.find_spec(n) is None]


CORE_PACKAGES = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "scipy", "statsmodels", "lightgbm",
                 "xgboost", "catboost", "shap", "joblib"]
OPTIONAL_PACKAGES = ["prophet", "statsforecast"]
_still_missing = ensure_packages(CORE_PACKAGES, auto_install=CFG.AUTO_INSTALL)
if CFG.INSTALL_OPTIONAL:
    ensure_packages(OPTIONAL_PACKAGES, auto_install=CFG.AUTO_INSTALL)

# %%
import gc
import json
import time
import math
import random
import warnings
import traceback
from pathlib import Path
from inspect import signature
from collections import OrderedDict

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass

warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except Exception:
    sns = None

try:
    from IPython.display import display
except Exception:
    display = print

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import scipy.stats as sps
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller, kpss, acf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_SM = True
except Exception as e:
    HAS_SM = False
    print("statsmodels tidak tersedia:", e)

try:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    HAS_ETS = True
except Exception:
    HAS_ETS = False
try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    HAS_THETA = True
except Exception:
    HAS_THETA = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception as e:
    HAS_LGB = False
    print("LightGBM tidak tersedia:", e)
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print("XGBoost tidak tersedia:", e)
try:
    import catboost as cb
    HAS_CAT = True
except Exception as e:
    HAS_CAT = False
    print("CatBoost tidak tersedia:", e)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False
try:
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False
try:
    from statsforecast.models import AutoETS as SF_AutoETS, AutoARIMA as SF_AutoARIMA
    HAS_SF = True
except Exception:
    HAS_SF = False

# statsmodels memasang filter "always" untuk warning-nya sendiri saat di-import -> matikan lagi supaya output bersih
warnings.simplefilter("ignore")

try:
    import sklearn
    SKLEARN_VERSION = sklearn.__version__
except Exception:
    SKLEARN_VERSION = "?"

print("Python", sys.version.split()[0], "| pandas", pd.__version__, "| numpy", np.__version__, "| sklearn", SKLEARN_VERSION)
for _name, _flag in [("statsmodels", HAS_SM), ("ETSModel", HAS_ETS), ("ThetaModel", HAS_THETA), ("lightgbm", HAS_LGB),
                     ("xgboost", HAS_XGB), ("catboost", HAS_CAT), ("shap", HAS_SHAP), ("prophet", HAS_PROPHET),
                     ("statsforecast", HAS_SF)]:
    print(f"  {'✅ ada' if _flag else '⚠️ tidak ada'} : {_name}")

# %% [markdown]
# ### 1.1 Setup: RUN_MODE, folder output, seed
#
# Cell ini menerapkan `RUN_MODE`. Mode **fast** membatasi jumlah window backtest, model, dan iterasi supaya
# seluruh notebook selesai dalam beberapa menit. **Hasil fast mode jangan dipakai untuk submission final.**

# %%
def seed_everything(seed=42):
    """Set seed untuk random, numpy, PYTHONHASHSEED (dan torch kalau ada)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # noqa: F401
        torch.manual_seed(seed)
    except Exception:
        pass


seed_everything(CFG.SEED)
FAST = str(CFG.RUN_MODE).lower() == "fast"
if FAST:
    CFG.N_BACKTEST_WINDOWS = min(CFG.N_BACKTEST_WINDOWS, 2)
    CFG.MODELS = [m for m in CFG.MODELS if m in CFG.FAST_MODELS]
    CFG.ML_N_ESTIMATORS = min(CFG.ML_N_ESTIMATORS, 200)
    CFG.ML_LEARNING_RATE = max(CFG.ML_LEARNING_RATE, 0.08)
    CFG.SARIMA_TIME_LIMIT = min(CFG.SARIMA_TIME_LIMIT, 8)
    CFG.SARIMA_MAX_P = min(CFG.SARIMA_MAX_P, 1)
    CFG.SARIMA_MAX_Q = min(CFG.SARIMA_MAX_Q, 1)
    CFG.STAT_MAX_SERIES = min(CFG.STAT_MAX_SERIES, 10)
    CFG.SHAP_SAMPLE = min(CFG.SHAP_SAMPLE, 500)
    print("RUN_MODE = fast -> setting dipangkas untuk debugging.")
else:
    print("RUN_MODE = full")

OUTPUT_DIR = Path(CFG.OUTPUT_DIR or f"./outputs/{CFG.NOTEBOOK_NAME}")
FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR = [OUTPUT_DIR / s for s in ["figures", "tables", "models", "submissions"]]
for _d in [OUTPUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR]:
    os.makedirs(_d, exist_ok=True)
print("Output dir:", OUTPUT_DIR.resolve())
FAILED_MODELS = {}

# %% [markdown]
# ## 2. Helper functions
#
# Semua fungsi bantu dikumpulkan di sini: baca file (`smart_read`), simpan figure (`save_fig`), metrik, deteksi frekuensi,
# kalender libur Indonesia, dll. **Tidak perlu diubah**, cukup dijalankan. Kalau ada error di cell setelahnya,
# biasanya penyebabnya konfigurasi (nama kolom / path), bukan helper.

# %%
def smart_read(path, **kwargs):
    """Baca csv/tsv/txt/xlsx/xls/parquet/json/feather/pickle dengan fallback separator & encoding."""
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
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
    if ext in [".pkl", ".pickle"]:
        return pd.read_pickle(path, **kwargs)
    sep_default = "\t" if ext == ".tsv" else ","
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            df = pd.read_csv(path, sep=sep_default, encoding=enc, **kwargs)
            if df.shape[1] == 1 and ext not in [".tsv"]:
                # kemungkinan separator lain (; atau |) -> sniff otomatis
                df2 = pd.read_csv(path, sep=None, engine="python", encoding=enc, **kwargs)
                if df2.shape[1] > 1:
                    df = df2
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc, **kwargs)
            except Exception as e2:
                last_err = e2
    raise last_err


_FIG_COUNTER = [0]


def save_fig(name, fig=None, show=True):
    """Simpan figure ke FIG_DIR dengan nomor urut (fig01_<name>.png), tampilkan, lalu close."""
    fig = fig or plt.gcf()
    _FIG_COUNTER[0] += 1
    fname = FIG_DIR / f"fig{_FIG_COUNTER[0]:02d}_{name}.png"
    try:
        fig.savefig(fname, dpi=CFG.FIG_DPI, bbox_inches="tight")
    except Exception as e:  # mis. MemoryError untuk figure sangat besar -> coba DPI rendah
        try:
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            print(f"Info: figure '{name}' disimpan dengan DPI 100 (DPI {CFG.FIG_DPI} gagal: {e})")
        except Exception as e2:
            print("Gagal simpan figure:", e2)
    if show:
        plt.show()
    plt.close(fig)


def save_table(df, name, index=False):
    """Simpan tabel ke TAB_DIR sebagai csv."""
    p = TAB_DIR / f"{name}.csv"
    df.to_csv(p, index=index)


def norm_key(s):
    """Normalisasi kolom ID series jadi string konsisten (1, 1.0, '1' -> '1')."""
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        if vals.notna().all() and np.allclose(vals, np.round(vals)):
            return vals.round().astype("int64").astype(str)
    return s.astype(str).str.strip()


def make_uid(df, cols):
    """Gabungkan SERIES_ID_COLS jadi satu kolom unique_id (string)."""
    if not cols:
        return pd.Series(["__single__"] * len(df), index=df.index)
    parts = [norm_key(df[c]) for c in cols]
    uid = parts[0]
    for p in parts[1:]:
        uid = uid + "|" + p
    return uid.values


def timer():
    t0 = time.time()
    return lambda: time.time() - t0


# ---------------------------------------------------------------- metrics
def _arr(x):
    return np.asarray(x, dtype=float)


def m_mae(y, p, scale=None):
    return float(np.mean(np.abs(_arr(y) - _arr(p))))


def m_rmse(y, p, scale=None):
    return float(np.sqrt(mean_squared_error(_arr(y), _arr(p))))


def m_mape(y, p, scale=None):
    y, p = _arr(y), _arr(p)
    mask = np.abs(y) > 1e-9
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def m_smape(y, p, scale=None):
    y, p = _arr(y), _arr(p)
    denom = np.abs(y) + np.abs(p)
    out = np.where(denom == 0, 0.0, 2 * np.abs(y - p) / np.where(denom == 0, 1, denom))
    return float(np.mean(out) * 100)


def m_wape(y, p, scale=None):
    y, p = _arr(y), _arr(p)
    return float(np.sum(np.abs(y - p)) / max(np.sum(np.abs(y)), 1e-9) * 100)


def m_mase(y, p, scale=None):
    y, p = _arr(y), _arr(p)
    if scale is None:
        return np.nan
    scale = _arr(scale)
    scale = np.where((scale <= 0) | ~np.isfinite(scale), np.nan, scale)
    v = np.abs(y - p) / scale
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def m_rmsle(y, p, scale=None):
    y, p = np.clip(_arr(y), 0, None), np.clip(_arr(p), 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


METRICS = OrderedDict([("mae", m_mae), ("rmse", m_rmse), ("mape", m_mape), ("smape", m_smape),
                       ("wape", m_wape), ("mase", m_mase), ("rmsle", m_rmsle)])
METRIC = "wape" if str(CFG.METRIC).lower() == "auto" else str(CFG.METRIC).lower()
assert METRIC in METRICS, f"METRIC {METRIC} tidak dikenal. Pilihan: {list(METRICS)}"
print("Metrik utama:", METRIC, "(lebih kecil = lebih baik)")


def all_metrics(y, p, scale=None):
    """Hitung semua metrik sekaligus -> dict."""
    out = {}
    for k, f in METRICS.items():
        try:
            out[k] = f(y, p, scale)
        except Exception:
            out[k] = np.nan
    return out


# ---------------------------------------------------------------- frequency helpers
from pandas.tseries.frequencies import to_offset


def infer_offset(dates):
    """Infer frekuensi dari tanggal. pd.infer_freq dulu; kalau gagal (ada gap), pakai modus selisih."""
    d = pd.DatetimeIndex(pd.Series(pd.to_datetime(dates)).dropna().drop_duplicates().sort_values())
    if len(d) < 3:
        return to_offset("D"), "D (default, data terlalu sedikit)"
    try:
        f = pd.infer_freq(d)
        if f is not None:
            return to_offset(f), f"{f} (pd.infer_freq)"
    except Exception:
        pass
    diffs = pd.Series(np.diff(d.asi8) // 10**9)
    mode_sec = int(diffs.mode().iloc[0])
    days = mode_sec / 86400.0
    if 27 <= days <= 31.5:
        if (d.day == 1).mean() > 0.8:
            return pd.offsets.MonthBegin(1), "MS (fallback modus selisih)"
        if d.is_month_end.mean() > 0.8:
            return pd.offsets.MonthEnd(1), "M/ME (fallback modus selisih)"
        return pd.offsets.MonthBegin(1), "MS (fallback, tanggal tidak konsisten)"
    if 88 <= days <= 93:
        if (d.day == 1).mean() > 0.8:
            return pd.offsets.QuarterBegin(startingMonth=1), "QS (fallback)"
        return pd.offsets.QuarterEnd(startingMonth=12), "Q (fallback)"
    if 364 <= days <= 366:
        if (d.dayofyear == 1).mean() > 0.8:
            return pd.offsets.YearBegin(1), "YS (fallback)"
        return pd.offsets.YearEnd(1), "Y (fallback)"
    if mode_sec == 3600:
        return to_offset(pd.Timedelta(hours=1)), "hourly (fallback modus selisih)"
    if mode_sec == 7 * 86400:
        wd = int(pd.Series(d.weekday).mode().iloc[0])
        return pd.offsets.Week(weekday=wd), f"W (weekday={wd}, fallback)"
    if mode_sec == 86400 and np.isin(d.weekday, [5, 6]).any():
        return to_offset("D"), "D (fallback modus selisih; ada tanggal yang hilang)"
    if mode_sec == 86400 and not np.isin(d.weekday, [5, 6]).any() and len(d) > 20:
        return pd.offsets.BDay(1), "B (business day, fallback)"
    return to_offset(pd.Timedelta(seconds=mode_sec)), f"{pd.Timedelta(seconds=mode_sec)} (fallback modus selisih)"


def offset_kind(off):
    """Kategori frekuensi: 'H','D','B','W','M','Q','Y','T' (minute/second)."""
    name = type(off).__name__
    if name in ("Hour",):
        return "H"
    if name in ("Minute", "Second", "Milli", "Micro", "Nano"):
        return "T"
    if name == "Day":
        return "D" if getattr(off, "n", 1) == 1 else "D"
    if name in ("BusinessDay", "CustomBusinessDay"):
        return "B"
    if name == "Week":
        return "W"
    if name.startswith("Month") or name.startswith("BusinessMonth") or name.startswith("SemiMonth"):
        return "M"
    if name.startswith("Quarter") or name.startswith("BQuarter"):
        return "Q"
    if name.startswith("Year") or name.startswith("BYear"):
        return "Y"
    if name == "Tick" or hasattr(off, "nanos"):
        try:
            sec = off.nanos / 1e9
            if sec < 3600:
                return "T"
            if sec < 86400:
                return "H"
            return "D"
        except Exception:
            pass
    return "D"


def default_season_length(off):
    k = offset_kind(off)
    n = max(int(getattr(off, "n", 1) or 1), 1)
    if k == "H":
        return max(24 // n, 1)
    if k == "T":
        try:
            per_day = int(86400 / (off.nanos / 1e9))
            return int(min(per_day, 1440))
        except Exception:
            return 60
    return {"D": 7, "B": 5, "W": 52, "M": 12, "Q": 4, "Y": 1}.get(k, 7)


def default_horizon(off):
    return {"H": 48, "T": 60, "D": 30, "B": 20, "W": 12, "M": 12, "Q": 4, "Y": 3}.get(offset_kind(off), 30)


def steps_between(a, b, off):
    """Jumlah langkah frekuensi dari tanggal a ke b (b > a)."""
    if b <= a:
        return 0
    return len(pd.date_range(a, b, freq=off)) - 1


def future_dates(last, h, off):
    """H tanggal setelah `last`."""
    return pd.date_range(start=last + off, periods=h, freq=off)


# ---------------------------------------------------------------- Indonesian holidays
# Tabel libur nasional Indonesia. Tanggal Lebaran/Idul Adha/Imlek mengikuti kalender pemerintah (SKB),
# ⚠️ VERIFIKASI ulang dengan SKB 3 Menteri / kalender resmi tahun terkait sebelum dipakai di laporan (terutama 2026–2027
# yang bisa bergeser 1 hari tergantung sidang isbat). Tambahkan libur lain (Nyepi, Waisak, Pemilu, cuti bersama) via CFG.CUSTOM_HOLIDAYS.
ID_HOLIDAYS_FIXED = {"01-01": "Tahun Baru Masehi", "05-01": "Hari Buruh", "06-01": "Hari Lahir Pancasila",
                     "08-17": "Hari Kemerdekaan RI", "12-25": "Hari Natal"}
LEBARAN_DATES = {2019: "2019-06-05", 2020: "2020-05-24", 2021: "2021-05-13", 2022: "2022-05-02", 2023: "2023-04-22",
                 2024: "2024-04-10", 2025: "2025-03-31", 2026: "2026-03-21", 2027: "2027-03-10"}
IDUL_ADHA_DATES = {2019: "2019-08-11", 2020: "2020-07-31", 2021: "2021-07-20", 2022: "2022-07-10", 2023: "2023-06-29",
                   2024: "2024-06-17", 2025: "2025-06-06", 2026: "2026-05-27", 2027: "2027-05-17"}
IMLEK_DATES = {2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
               2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17", 2027: "2027-02-06"}


def easter_date(year):
    """Tanggal Paskah (algoritma Gregorian anonim) -> untuk Jumat Agung (-2) dan Kenaikan Isa Almasih (+39)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def get_holiday_table(years):
    """DataFrame [date, holiday] untuk tahun-tahun yang diminta."""
    rows = []
    if CFG.COUNTRY_HOLIDAYS == "ID":
        for y in years:
            for md, nm in ID_HOLIDAYS_FIXED.items():
                rows.append((pd.Timestamp(f"{y}-{md}"), nm))
            if y in LEBARAN_DATES:
                leb = pd.Timestamp(LEBARAN_DATES[y])
                rows += [(leb, "Idul Fitri"), (leb + pd.Timedelta(days=1), "Idul Fitri (hari 2)")]
            if y in IDUL_ADHA_DATES:
                rows.append((pd.Timestamp(IDUL_ADHA_DATES[y]), "Idul Adha"))
            if y in IMLEK_DATES:
                rows.append((pd.Timestamp(IMLEK_DATES[y]), "Imlek"))
            try:
                ea = easter_date(y)
                rows += [(ea - pd.Timedelta(days=2), "Jumat Agung"), (ea + pd.Timedelta(days=39), "Kenaikan Isa Almasih")]
            except Exception:
                pass
    for dt, nm in (CFG.CUSTOM_HOLIDAYS or {}).items():
        rows.append((pd.Timestamp(dt), nm))
    if not rows:
        return pd.DataFrame(columns=["date", "holiday"])
    return pd.DataFrame(rows, columns=["date", "holiday"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def build_calendar_features(ds_values, off):
    """Fitur kalender per timestamp: is_holiday/holiday_count, days_to_lebaran, is_ramadan/ramadan_days, lebaran_window.
    Untuk frekuensi > harian, hari libur diagregasi ke periode (asumsi timestamp = awal periode, kecuali month/quarter-end)."""
    grid = pd.DatetimeIndex(pd.Series(pd.to_datetime(ds_values)).drop_duplicates().sort_values())
    years = list(range(grid.min().year - 1, grid.max().year + 2))
    hol = get_holiday_table(years)
    hol_set = set(pd.DatetimeIndex(hol["date"]).normalize()) if len(hol) else set()
    leb = pd.DatetimeIndex(sorted(pd.Timestamp(v) for v in LEBARAN_DATES.values())) if CFG.COUNTRY_HOLIDAYS == "ID" else pd.DatetimeIndex([])
    daily = pd.date_range(grid.min().normalize() - pd.Timedelta(days=7), grid.max().normalize() + pd.Timedelta(days=400), freq="D")
    cal = pd.DataFrame(index=daily)
    cal["is_holiday"] = np.array([d in hol_set for d in daily], dtype=float)
    if len(leb):
        diff_days = (daily.values[:, None] - leb.values[None, :]).astype("timedelta64[D]").astype(float)
        idx = np.argmin(np.abs(diff_days), axis=1)
        dtl = diff_days[np.arange(len(daily)), idx]
        cal["days_to_lebaran"] = np.clip(dtl, -60, 60)
        cal["is_ramadan"] = ((dtl >= -30) & (dtl <= -1)).astype(float)
        cal["lebaran_window"] = ((dtl >= -7) & (dtl <= 7)).astype(float)
    kind = offset_kind(off)
    if kind in ("D", "B", "H", "T"):
        out = cal.reindex(grid.normalize())
        out.index = grid
        return out
    # Frekuensi lebih kasar: agregasi harian -> periode
    end_labeled = (type(off).__name__ in ("MonthEnd", "QuarterEnd", "YearEnd", "BusinessMonthEnd"))
    side = "left" if end_labeled else "right"
    pos = np.searchsorted(grid.values, daily.values, side=side) - (0 if end_labeled else 1)
    valid = (pos >= 0) & (pos < len(grid))
    tmp = cal[valid].copy()
    tmp["_p"] = pos[valid]
    agg = {"is_holiday": "sum"}
    if "is_ramadan" in tmp:
        agg.update({"is_ramadan": "sum", "lebaran_window": "max"})
    out = tmp.groupby("_p").agg(agg)
    out = out.rename(columns={"is_holiday": "holiday_count", "is_ramadan": "ramadan_days"})
    out = out.reindex(range(len(grid))).fillna(0)
    out.index = grid
    if "days_to_lebaran" in cal:
        out["days_to_lebaran"] = cal["days_to_lebaran"].reindex(grid.normalize()).values
    return out


print("Helper siap.")

# %% [markdown]
# ## 3. Load data
#
# **Tujuan:** membaca train / test / sample_submission, mendeteksi kolom tanggal, dan cek kolom penting ada.
#
# **Cara membaca output:**
# - `shape` train/test, 5 baris pertama, kolom tanggal yang terdeteksi.
# - Kalau `TEST_PATH` tidak ada → notebook tetap jalan dan akan membuat **forecast H langkah ke depan** (bukan submission).
#
# **Jika X maka Y:**
# - Error `TARGET_COL ... tidak ada` → cek ejaan nama kolom (case-sensitive), lihat daftar kolom yang di-print.
# - Kolom tanggal salah terdeteksi → set `CFG.DATE_COL` manual.
# - Tanggal ter-parse aneh (bulan & hari tertukar, mis. 03/04 jadi Maret padahal April) → set `CFG.DAYFIRST=True` atau `CFG.DATE_FORMAT="%d/%m/%Y"`.
# - Separator file `;` (CSV dari Excel Indonesia) → sudah ditangani otomatis oleh `smart_read`.

# %%
assert os.path.exists(CFG.TRAIN_PATH), f"TRAIN_PATH tidak ditemukan: {CFG.TRAIN_PATH}"
train_raw = smart_read(CFG.TRAIN_PATH)
HAS_TEST = bool(CFG.TEST_PATH) and os.path.exists(str(CFG.TEST_PATH))
test_raw = smart_read(CFG.TEST_PATH) if HAS_TEST else None
HAS_SAMPLE = bool(CFG.SAMPLE_SUB_PATH) and os.path.exists(str(CFG.SAMPLE_SUB_PATH))
sample_sub = smart_read(CFG.SAMPLE_SUB_PATH) if HAS_SAMPLE else None

print("Train shape:", train_raw.shape)
print("Test shape :", None if test_raw is None else test_raw.shape)
print("Sample sub :", None if sample_sub is None else sample_sub.shape)
if not HAS_TEST:
    print("⚠️ File test tidak ditemukan -> mode forecast H langkah ke depan (tanpa submission berbasis ID).")
print("Kolom train:", list(train_raw.columns))
display(train_raw.head())
if test_raw is not None:
    display(test_raw.head())

assert CFG.TARGET_COL in train_raw.columns, f"TARGET_COL '{CFG.TARGET_COL}' tidak ada di train. Kolom: {list(train_raw.columns)}"
for c in CFG.SERIES_ID_COLS:
    assert c in train_raw.columns, f"SERIES_ID_COL '{c}' tidak ada di train"
    if test_raw is not None and c not in test_raw.columns:
        print(f"⚠️ SERIES_ID_COL '{c}' tidak ada di test -> merge prediksi ke test akan gagal untuk kolom ini.")

# %%
DATE_NAME_HINTS = ["date", "ds", "tanggal", "tgl", "time", "timestamp", "datetime", "period", "periode", "month", "bulan",
                   "week", "minggu", "day", "waktu"]


def parse_dates(s):
    """Parse kolom tanggal secara robust (format manual, yyyymm/yyyymmdd integer, tahun, string umum, timezone)."""
    s = pd.Series(s)
    if CFG.DATE_FORMAT:
        out = pd.to_datetime(s.astype(str), format=CFG.DATE_FORMAT, errors="coerce")
    elif pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="coerce")
    elif pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        vi = v.dropna().astype("int64")
        if len(vi) and vi.between(1800, 2200).all():
            out = pd.to_datetime(v.astype("Int64").astype(str), format="%Y", errors="coerce")
        elif len(vi) and vi.between(180001, 220012).all():
            out = pd.to_datetime(v.astype("Int64").astype(str), format="%Y%m", errors="coerce")
        elif len(vi) and vi.between(18000101, 22001231).all():
            out = pd.to_datetime(v.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:  # unix timestamp
            unit = "ms" if vi.median() > 1e11 else "s"
            out = pd.to_datetime(v, unit=unit, errors="coerce")
    else:
        try:
            out = pd.to_datetime(s, dayfirst=CFG.DAYFIRST, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            out = pd.to_datetime(s, dayfirst=CFG.DAYFIRST, errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out


def detect_date_col(df):
    """Cari kolom tanggal: nama mirip date/tanggal dulu, lalu kolom object yang >90% bisa di-parse."""
    if CFG.DATE_COL:
        assert CFG.DATE_COL in df.columns, f"DATE_COL '{CFG.DATE_COL}' tidak ada"
        return CFG.DATE_COL
    skip = set([CFG.TARGET_COL] + list(CFG.SERIES_ID_COLS) + ([CFG.ID_COL] if CFG.ID_COL else []))
    cands = [c for c in df.columns if c not in skip]
    named = [c for c in cands if any(h == str(c).lower() or h in str(c).lower() for h in DATE_NAME_HINTS)]
    for c in named + [c for c in cands if c not in named]:
        s = df[c].dropna().head(2000)
        if len(s) == 0:
            continue
        if pd.api.types.is_numeric_dtype(s) and c not in named:
            continue
        try:
            ok = parse_dates(s).notna().mean()
        except Exception:
            ok = 0
        if ok > 0.9:
            return c
    raise ValueError("Kolom tanggal tidak terdeteksi. Set CFG.DATE_COL secara manual.")


DATE_COL = detect_date_col(train_raw)
print("Kolom tanggal:", DATE_COL)
train_raw["_ds"] = parse_dates(train_raw[DATE_COL])
n_bad = int(train_raw["_ds"].isna().sum())
print(f"Tanggal gagal di-parse di train: {n_bad} baris" + (" ⚠️ cek DATE_FORMAT/DAYFIRST" if n_bad else " ✅"))
train_raw = train_raw[train_raw["_ds"].notna()].copy()
print("Rentang tanggal train:", train_raw["_ds"].min(), "->", train_raw["_ds"].max())
if test_raw is not None:
    if DATE_COL not in test_raw.columns:
        print(f"⚠️ Kolom tanggal '{DATE_COL}' tidak ada di test -> test diabaikan, mode forecast H langkah.")
        HAS_TEST = False
        test_raw = None
    else:
        test_raw["_ds"] = parse_dates(test_raw[DATE_COL])
        print("Rentang tanggal test :", test_raw["_ds"].min(), "->", test_raw["_ds"].max(),
              f"| gagal parse: {int(test_raw['_ds'].isna().sum())}")

# %% [markdown]
# ## 4. Data preparation
#
# Urutan langkah (masing-masing dicetak ringkasannya):
# 1. **Target numerik** — string seperti `"1,234"` dikonversi; baris target kosong dibuang (nanti dibuat ulang sebagai timestamp hilang).
# 2. **Exogenous** — validasi kolom ada di test (harus diketahui di masa depan!), encode kategorikal jadi kode angka.
# 3. **Duplikat timestamp per series** — diagregasi (`DUPLICATE_AGG`). Contoh: data transaksi (banyak baris per hari) → dijumlah jadi penjualan harian.
# 4. **Resample (opsional)** — ganti level agregasi (`RESAMPLE_RULE`), mis. harian → mingguan kalau lomba minta mingguan.
# 5. **Frekuensi** — infer otomatis. **Cek outputnya!** Kalau salah, set `CFG.FREQ`.
# 6. **Timestamp hilang** — reindex ke grid lengkap lalu isi (`FILL_METHOD`). Titik yang diisi ditandai `is_imputed`.
# 7. **Nilai negatif** dan **outlier** — dilaporkan, ditangani sesuai `NEGATIVE_POLICY` / `OUTLIER_METHOD`.
#
# **Jika X maka Y:**
# - `% timestamp hilang` > 30% → cek apakah frekuensi benar (mungkin data sebenarnya mingguan/bisnis), atau series memang jarang (intermittent) → pakai `FILL_METHOD="zero"` + metrik MAE/WAPE.
# - Banyak duplikat padahal data harusnya 1 baris per tanggal → mungkin ada level series yang belum dimasukkan ke `SERIES_ID_COLS`.
# - Outlier karena event nyata (promo, Lebaran, lockdown COVID) → **jangan dibuang**, lebih baik jadikan fitur (exog/holiday). Outlier karena error input → clip.
#
# **Untuk laporan:** tulis tabel ringkasan data (jumlah series, rentang tanggal, frekuensi, % missing, cara imputasi) dan alasan pemilihan metode imputasi.

# %%
def to_numeric_clean(s):
    """Konversi ke numerik; tangani string dengan pemisah ribuan/koma desimal sederhana."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    st = s.astype(str).str.strip().str.replace(" ", "", regex=False)
    v = pd.to_numeric(st, errors="coerce")
    if v.notna().mean() < 0.9:
        # coba format Indonesia "1.234,5" -> 1234.5
        v2 = pd.to_numeric(st.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
        v3 = pd.to_numeric(st.str.replace(",", "", regex=False), errors="coerce")
        v = v2 if v2.notna().mean() >= v3.notna().mean() else v3
    return v


train_raw[CFG.TARGET_COL] = to_numeric_clean(train_raw[CFG.TARGET_COL])
n_nan_y = int(train_raw[CFG.TARGET_COL].isna().sum())
print(f"Target NaN/non-numerik: {n_nan_y} baris (dibuang, timestamp-nya nanti diisi ulang sesuai FILL_METHOD)")
train_raw = train_raw[train_raw[CFG.TARGET_COL].notna()].copy()

# ---- Exogenous columns
id_like = set([DATE_COL, "_ds", CFG.TARGET_COL] + list(CFG.SERIES_ID_COLS) + ([CFG.ID_COL] if CFG.ID_COL else []))
if CFG.EXOG_COLS == "auto":
    EXOG = [c for c in train_raw.columns if c not in id_like and test_raw is not None and c in test_raw.columns]
else:
    EXOG = [c for c in (CFG.EXOG_COLS or []) if c in train_raw.columns]
    missing_exog = [c for c in (CFG.EXOG_COLS or []) if c not in train_raw.columns]
    if missing_exog:
        print("⚠️ EXOG tidak ada di train (diabaikan):", missing_exog)
if EXOG and test_raw is None:
    print("⚠️ Tidak ada file test -> nilai exog masa depan tidak diketahui. Exog di-forward-fill dari nilai terakhir "
          "(hanya masuk akal untuk variabel yang jarang berubah). Pertimbangkan EXOG_COLS=[].")
if EXOG and test_raw is not None:
    not_in_test = [c for c in EXOG if c not in test_raw.columns]
    if not_in_test:
        print("⚠️ EXOG tidak ada di test (dibuang karena nilainya tidak diketahui di masa depan):", not_in_test)
        EXOG = [c for c in EXOG if c not in not_in_test]
EXOG_CAT_MAPS = {}
for c in EXOG:
    tr_num = to_numeric_clean(train_raw[c])
    if tr_num.notna().mean() >= 0.9 * max(train_raw[c].notna().mean(), 1e-9):
        train_raw[c] = tr_num
        if test_raw is not None:
            test_raw[c] = to_numeric_clean(test_raw[c])
    else:
        cats = pd.concat([train_raw[c], test_raw[c] if test_raw is not None else pd.Series(dtype=object)]).astype(str).unique()
        mapping = {v: i for i, v in enumerate(sorted(cats))}
        EXOG_CAT_MAPS[c] = mapping
        train_raw[c] = train_raw[c].astype(str).map(mapping).astype(float)
        if test_raw is not None:
            test_raw[c] = test_raw[c].astype(str).map(mapping).astype(float)
print("Exogenous dipakai:", EXOG if EXOG else "(tidak ada)", "| kategorikal:", list(EXOG_CAT_MAPS))

# ---- long format
train_raw["unique_id"] = make_uid(train_raw, CFG.SERIES_ID_COLS)
UID_MAP = (train_raw[["unique_id"] + list(CFG.SERIES_ID_COLS)].drop_duplicates("unique_id").set_index("unique_id")
           if CFG.SERIES_ID_COLS else None)
long = train_raw[["unique_id", "_ds", CFG.TARGET_COL] + EXOG].rename(columns={"_ds": "ds", CFG.TARGET_COL: "y"})

# ---- duplicates per timestamp
dup_count = int(long.duplicated(["unique_id", "ds"]).sum())
print(f"Baris duplikat (unique_id, ds): {dup_count}" + (f" -> diagregasi dengan '{CFG.DUPLICATE_AGG}'" if dup_count else " ✅"))
if dup_count:
    agg = {"y": CFG.DUPLICATE_AGG}
    for c in EXOG:
        agg[c] = "last" if c in EXOG_CAT_MAPS else "mean"
    long = long.groupby(["unique_id", "ds"], as_index=False, sort=False).agg(agg)
long = long.sort_values(["unique_id", "ds"]).reset_index(drop=True)

# ---- optional resample (aggregation level)
if CFG.RESAMPLE_RULE:
    print(f"Resample ke '{CFG.RESAMPLE_RULE}' dengan agg '{CFG.RESAMPLE_AGG}'")
    parts = []
    for uid, g in long.groupby("unique_id", sort=False):
        g = g.set_index("ds")
        agg = {"y": CFG.RESAMPLE_AGG}
        for c in EXOG:
            agg[c] = "last" if c in EXOG_CAT_MAPS else "mean"
        r = g.resample(CFG.RESAMPLE_RULE).agg(agg).reset_index()
        r["unique_id"] = uid
        parts.append(r)
    long = pd.concat(parts, ignore_index=True)[["unique_id", "ds", "y"] + EXOG]
    if test_raw is not None:
        print("⚠️ Tanggal test di-snap ke label periode resample yang sama (pakai aturan label pandas resample).")
        _snap = {}
        for _d in test_raw["_ds"].dropna().unique():
            _snap[_d] = pd.Series([0], index=[pd.Timestamp(_d)]).resample(CFG.RESAMPLE_RULE).sum().index[0]
        test_raw["_ds"] = test_raw["_ds"].map(_snap)
        print("   Catatan: kalau test berisi banyak baris per periode, prediksi periode yang sama akan dipakai untuk semua baris itu.")

# ---- frequency
if CFG.FREQ:
    OFF = to_offset(CFG.FREQ)
    print("Frekuensi (manual):", CFG.FREQ)
else:
    OFF, freq_msg = infer_offset(long["ds"])
    print("Frekuensi terdeteksi:", freq_msg)
KIND = offset_kind(OFF)
SEASON = int(CFG.SEASON_LENGTH or default_season_length(OFF))
print(f"Jenis frekuensi: {KIND} | season length (m): {SEASON}")

# cek keselarasan tanggal dengan grid frekuensi
_u = long["ds"].drop_duplicates().sort_values()
_grid = pd.date_range(_u.min(), _u.max(), freq=OFF)
_on_grid = float(np.isin(_u.values, _grid.values).mean())
print(f"Tanggal yang pas di grid frekuensi: {_on_grid:.1%}" + ("  ⚠️ banyak tanggal tidak sejajar -> cek CFG.FREQ / RESAMPLE_RULE" if _on_grid < 0.95 else "  ✅"))

# %%
def fill_series(y, method, m):
    """Isi NaN pada satu series sesuai metode."""
    y = y.copy()
    if not y.isna().any():
        return y
    if method == "zero":
        return y.fillna(0.0)
    if method == "ffill":
        return y.ffill().bfill()
    if method == "mean":
        return y.fillna(y.mean())
    if method == "seasonal" and m > 1:
        for _ in range(5):
            if not y.isna().any():
                break
            y = y.fillna(y.shift(m))
        return y.interpolate(limit_direction="both")
    # interpolate (default untuk 'none' juga, karena model butuh nilai lengkap)
    return y.interpolate(limit_direction="both").ffill().bfill()


GLOBAL_END = long["ds"].max()
_fill = CFG.FILL_METHOD if CFG.FILL_METHOD in ("zero", "ffill", "interpolate", "seasonal", "mean") else "interpolate"
if _fill != CFG.FILL_METHOD:
    print(f"⚠️ FILL_METHOD '{CFG.FILL_METHOD}' -> pakai 'interpolate' (model butuh deret tanpa lubang)")

# Nilai negatif: tangani sebelum imputasi
neg_count = int((long["y"] < 0).sum())
print(f"Nilai target negatif: {neg_count}" + (f" -> policy '{CFG.NEGATIVE_POLICY}'" if neg_count else " ✅"))
if neg_count and CFG.NEGATIVE_POLICY == "clip0":
    long["y"] = long["y"].clip(lower=0)
elif neg_count and CFG.NEGATIVE_POLICY == "nan":
    long.loc[long["y"] < 0, "y"] = np.nan

parts, miss_rows = [], []
for uid, g in long.groupby("unique_id", sort=False):
    end = GLOBAL_END if CFG.REINDEX_TO_GLOBAL_END else g["ds"].max()
    full_idx = pd.date_range(g["ds"].min(), end, freq=OFF)
    g2 = g.set_index("ds").reindex(full_idx.union(pd.DatetimeIndex(g["ds"])))
    g2 = g2.loc[g2.index.isin(full_idx)]  # buang tanggal di luar grid
    g2.index.name = "ds"
    g2["unique_id"] = uid
    g2["is_imputed"] = g2["y"].isna()
    n_miss = int(g2["is_imputed"].sum())
    miss_rows.append((uid, len(g2), n_miss))
    g2["y"] = fill_series(g2["y"], _fill, SEASON)
    for c in EXOG:
        g2[c] = g2[c].ffill().bfill()
    parts.append(g2.reset_index())
df = pd.concat(parts, ignore_index=True)[["unique_id", "ds", "y", "is_imputed"] + EXOG]
df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
miss_df = pd.DataFrame(miss_rows, columns=["unique_id", "n_points", "n_imputed"])
miss_df["pct_imputed"] = miss_df["n_imputed"] / miss_df["n_points"] * 100
print(f"Timestamp hilang yang diisi ('{_fill}'): {int(miss_df['n_imputed'].sum())} "
      f"({miss_df['n_imputed'].sum() / max(miss_df['n_points'].sum(), 1):.1%} dari grid)")
if (miss_df["pct_imputed"] > 30).any():
    print("⚠️ Ada series dengan >30% titik hasil imputasi:", miss_df.loc[miss_df["pct_imputed"] > 30, "unique_id"].head(10).tolist())

# ---- outliers (per series)
def detect_outliers(y, method, thr, m):
    """Return boolean mask outlier + batas bawah/atas (array)."""
    y = pd.Series(y)
    if method == "iqr":
        q1, q3 = y.quantile(0.25), y.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - thr * iqr, q3 + thr * iqr
        lo, hi = pd.Series(lo, index=y.index), pd.Series(hi, index=y.index)
    elif method == "zscore":
        mu, sd = y.mean(), y.std() + 1e-9
        lo, hi = pd.Series(mu - thr * sd, index=y.index), pd.Series(mu + thr * sd, index=y.index)
    else:  # hampel
        w = max(2 * max(m, 3) + 1, 7)
        med = y.rolling(w, center=True, min_periods=1).median()
        mad = (y - med).abs().rolling(w, center=True, min_periods=1).median() * 1.4826 + 1e-9
        lo, hi = med - thr * mad, med + thr * mad
    mask = (y < lo) | (y > hi)
    return mask.values, lo.values, hi.values


OUTLIER_REPORT = []
if CFG.OUTLIER_METHOD:
    new_y = df["y"].values.copy()
    for uid, idx in df.groupby("unique_id", sort=False).indices.items():
        mask, lo, hi = detect_outliers(df["y"].values[idx], CFG.OUTLIER_METHOD, CFG.OUTLIER_THRESHOLD, SEASON)
        OUTLIER_REPORT.append((uid, int(mask.sum())))
        if mask.any():
            yy = pd.Series(df["y"].values[idx].astype(float))
            if CFG.OUTLIER_ACTION == "interpolate":
                yy[mask] = np.nan
                yy = yy.interpolate(limit_direction="both")
            else:
                yy = pd.Series(np.clip(yy.values, lo, hi))
            new_y[idx] = yy.values
    n_out = sum(n for _, n in OUTLIER_REPORT)
    print(f"Outlier ({CFG.OUTLIER_METHOD}, thr={CFG.OUTLIER_THRESHOLD}): {n_out} titik -> {CFG.OUTLIER_ACTION}")
    df["y_before_outlier"] = df["y"]
    df["y"] = new_y
else:
    # tetap laporkan jumlah kandidat outlier (hampel) tanpa mengubah data
    cnt = 0
    for uid, idx in df.groupby("unique_id", sort=False).indices.items():
        mask, _, _ = detect_outliers(df["y"].values[idx], "hampel", 4.0, SEASON)
        cnt += int(mask.sum())
    print(f"Kandidat outlier (hampel, thr=4, info saja, data TIDAK diubah): {cnt} titik ({cnt / len(df):.2%})")

# ---- summary
SERIES = df["unique_id"].unique().tolist()
N_SERIES = len(SERIES)
IS_PANEL = N_SERIES > 1
Y_NONNEG = bool((df["y"] >= 0).all())
len_per_series = df.groupby("unique_id").size()
summary = pd.DataFrame({
    "item": ["n_series", "n_rows", "date_start", "date_end", "frequency", "season_length", "min_len", "median_len",
             "pct_imputed", "pct_zero", "target_min", "target_mean", "target_max", "nonnegative", "exog"],
    "value": [N_SERIES, len(df), str(df["ds"].min()), str(df["ds"].max()), KIND, SEASON, int(len_per_series.min()),
              int(len_per_series.median()), f"{df['is_imputed'].mean():.2%}", f"{(df['y'] == 0).mean():.2%}",
              f"{df['y'].min():.4g}", f"{df['y'].mean():.4g}", f"{df['y'].max():.4g}", Y_NONNEG, ", ".join(EXOG) or "-"]})
display(summary)
save_table(summary, "data_summary")
if (df["y"] == 0).mean() > 0.3:
    print("⚠️ >30% nilai nol -> intermittent demand. Hindari MAPE; pakai MAE/WAPE/RMSE. Model ML + log1p biasanya bagus.")
print("✅ Data siap: kolom internal = unique_id, ds, y, is_imputed" + (", " + ", ".join(EXOG) if EXOG else ""))

# %% [markdown]
# ### 4.1 Horizon, gap, dan tanggal masa depan
#
# **Logika:**
# - **Ada test:** horizon = jumlah langkah yang dicakup tanggal test. **GAP** = jumlah langkah antara akhir train dan awal test
#   (mis. train sampai 31 Des, test mulai 8 Jan → gap 7 hari). Model akan memprediksi `GAP + H` langkah dan backtest meniru jeda yang sama.
# - **Tidak ada test:** horizon = `CFG.HORIZON` (atau default per frekuensi), gap = `CFG.GAP` atau 0.
#
# **Jika X maka Y:**
# - `⚠️ Tanggal test tidak sejajar grid` → frekuensi test berbeda dengan train (mis. train harian, test mingguan). Samakan dengan `RESAMPLE_RULE`.
# - `⚠️ Ada tanggal test <= akhir train` → test tumpang tindih dengan train (jarang; cek apakah memang lomba "imputation" bukan forecasting).
# - Series di test yang **tidak ada di train** (cold start) → diisi rata-rata prediksi series lain pada tanggal yang sama (lihat pesan saat submission).

# %%
test_long = None
if HAS_TEST:
    test_raw["unique_id"] = make_uid(test_raw, [c for c in CFG.SERIES_ID_COLS if c in test_raw.columns])
    test_long = test_raw.copy()
    t_min, t_max = test_raw["_ds"].min(), test_raw["_ds"].max()
    if t_min <= GLOBAL_END:
        print(f"⚠️ Ada tanggal test <= akhir train ({GLOBAL_END}); baris tsb akan diisi nilai aktual/forecast terdekat.")
    first_future = max(t_min, GLOBAL_END + OFF)
    auto_gap = max(steps_between(GLOBAL_END, first_future, OFF) - 1, 0)
    H = max(steps_between(first_future, t_max, OFF) + 1, 1)
    GAP = int(CFG.GAP) if CFG.GAP is not None else int(auto_gap)
    if CFG.HORIZON and int(CFG.HORIZON) != H:
        print(f"⚠️ CFG.HORIZON={CFG.HORIZON} berbeda dengan horizon dari test ({H}); dipakai horizon test.")
    _tgrid = future_dates(GLOBAL_END, GAP + H, OFF)
    _aligned = float(np.isin(test_raw["_ds"].drop_duplicates().values, _tgrid.values).mean())
    print(f"Tanggal test sejajar grid forecast: {_aligned:.1%}" + ("  ⚠️ Tanggal test tidak sejajar grid" if _aligned < 0.99 else "  ✅"))
    new_series = sorted(set(test_raw["unique_id"]) - set(SERIES))
    if new_series:
        print(f"⚠️ {len(new_series)} series di test tidak ada di train (cold start), contoh: {new_series[:5]}")
else:
    H = int(CFG.HORIZON or default_horizon(OFF))
    GAP = int(CFG.GAP or 0)
H_TOTAL = GAP + H
FUTURE_DATES = future_dates(GLOBAL_END, H_TOTAL, OFF)
print(f"HORIZON H = {H} | GAP = {GAP} | total langkah diprediksi = {H_TOTAL}")
print("Tanggal masa depan:", FUTURE_DATES[0], "->", FUTURE_DATES[-1])

# Fitur kalender untuk seluruh tanggal (historis + masa depan)
CAL_FEATS = None
if CFG.USE_HOLIDAY_FEATURES and CFG.COUNTRY_HOLIDAYS:
    try:
        CAL_FEATS = build_calendar_features(pd.DatetimeIndex(df["ds"].unique()).union(FUTURE_DATES), OFF)
        print("Fitur kalender:", list(CAL_FEATS.columns))
    except Exception as e:
        print("⚠️ Gagal membuat fitur kalender:", e)

# Lag & rolling windows default
med_len = int(len_per_series.median())
if CFG.LAGS:
    LAGS = sorted(set(int(l) for l in CFG.LAGS))
else:
    LAGS = {"D": list(range(1, 8)) + [14, 21, 28], "B": [1, 2, 3, 4, 5, 10, 15, 20],
            "H": list(range(1, 7)) + [12, 24, 48, 168], "W": [1, 2, 3, 4, 8, 13, 26, 52],
            "M": [1, 2, 3, 6, 12, 24], "Q": [1, 2, 3, 4, 8], "Y": [1, 2, 3]}.get(KIND, [1, 2, 3, SEASON, 2 * SEASON])
LAGS = sorted(set([1] + [l for l in LAGS if l <= max(1, int(med_len * 0.5))]))
if CFG.ROLL_WINDOWS:
    ROLL_WINDOWS = sorted(set(int(w) for w in CFG.ROLL_WINDOWS))
else:
    ROLL_WINDOWS = {"D": [7, 14, 28], "B": [5, 10, 20], "H": [6, 24, 168], "W": [4, 8, 13], "M": [3, 6, 12],
                    "Q": [2, 4], "Y": [2, 3]}.get(KIND, [3, 6])
ROLL_WINDOWS = [w for w in ROLL_WINDOWS if w <= max(2, int(med_len * 0.5))] or [2]
print("LAGS:", LAGS, "| ROLL_WINDOWS:", ROLL_WINDOWS)

# %% [markdown]
# ## 5. Exploratory Data Analysis (EDA) untuk time series
#
# **Tujuan:** memahami **trend**, **seasonality** (pola berulang), **level/varian yang berubah**, efek **hari libur**, dan
# **stasioneritas**. Hasil EDA menentukan pilihan model & fitur:
#
# | Temuan EDA | Implikasi |
# |---|---|
# | Seasonality mingguan kuat (harian) | Seasonal naive jadi baseline kuat; fitur `dayofweek`, lag 7/14/28 penting |
# | Trend naik/turun jelas | Model tree (LightGBM) tidak bisa ekstrapolasi → `TARGET_TRANSFORM="diff"` atau andalkan ETS/SARIMA/drift dalam ensemble |
# | Varian membesar seiring level | Pakai `log1p` (multiplicative → additive) |
# | Lonjakan saat Lebaran/akhir tahun | Fitur holiday + `days_to_lebaran` (sudah otomatis) |
# | Banyak nol | Intermittent: hindari MAPE, model ML + log1p, pertimbangkan metrik WAPE/MAE |
#
# Untuk **panel**, EDA dilakukan pada deret **agregat** (jumlah semua series, `PANEL_AGG_FOR_EDA`) + beberapa contoh series.
#
# **Untuk laporan:** figure line plot + STL decomposition + seasonal boxplot hampir selalu layak masuk laporan (1 halaman).
# Tulis kalimat seperti: *"Data menunjukkan seasonality mingguan yang kuat (seasonal strength = 0.82) dan trend naik ~X% per tahun;
# uji ADF/KPSS mengindikasikan deret non-stasioner pada level sehingga dilakukan differencing / model dengan komponen trend."*

# %%
EDA_SERIES = (df.groupby("ds")["y"].sum() if CFG.PANEL_AGG_FOR_EDA == "sum" else df.groupby("ds")["y"].mean()).asfreq(OFF)
EDA_SERIES = EDA_SERIES.interpolate(limit_direction="both")
eda_label = "Total (sum of all series)" if IS_PANEL else "Series"
win = max(SEASON, ROLL_WINDOWS[-1])

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(EDA_SERIES.index, EDA_SERIES.values, lw=0.9, label=CFG.TARGET_COL)
ax.plot(EDA_SERIES.index, EDA_SERIES.rolling(win, min_periods=1).mean(), lw=2, label=f"rolling mean ({win})")
ax.fill_between(EDA_SERIES.index, EDA_SERIES.rolling(win, min_periods=2).mean() - EDA_SERIES.rolling(win, min_periods=2).std(),
                EDA_SERIES.rolling(win, min_periods=2).mean() + EDA_SERIES.rolling(win, min_periods=2).std(), alpha=0.2,
                label="±1 rolling std")
ax.set_title(f"{eda_label}: {CFG.TARGET_COL} over time")
ax.set_xlabel("Date")
ax.set_ylabel(CFG.TARGET_COL)
ax.legend()
save_fig("target_over_time")

# Trend kasar: perubahan rata-rata tahun pertama vs terakhir (kalau data >= 2 season panjang)
if len(EDA_SERIES) >= 2 * max(SEASON, 2):
    first, last = EDA_SERIES.iloc[:max(SEASON, 2) * 2].mean(), EDA_SERIES.iloc[-max(SEASON, 2) * 2:].mean()
    chg = (last - first) / (abs(first) + 1e-9) * 100
    print(f"Perubahan level (awal vs akhir): {chg:+.1f}%" + ("  -> trend jelas, pertimbangkan diff/drift" if abs(chg) > 20 else ""))
rs = EDA_SERIES.rolling(win, min_periods=2).std()
rm = EDA_SERIES.rolling(win, min_periods=2).mean()
if rm.abs().mean() > 0 and np.corrcoef(rm.dropna().values, rs.dropna().values)[0, 1] > 0.6 and Y_NONNEG:
    print("Info: varian ikut membesar dengan level (corr rolling mean vs std > 0.6) -> log1p transform disarankan.")

# %%
if IS_PANEL:
    totals = df.groupby("unique_id")["y"].sum().sort_values(ascending=False)
    rng = np.random.RandomState(CFG.SEED)
    pick = list(totals.index[:max(1, CFG.MAX_SERIES_PLOT // 2)])
    rest = [u for u in totals.index if u not in pick]
    if rest:
        pick += list(rng.choice(rest, size=min(len(rest), CFG.MAX_SERIES_PLOT - len(pick)), replace=False))
    PLOT_SERIES = pick
    n = len(pick)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.3 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, u in zip(axes, pick):
        g = df[df["unique_id"] == u]
        ax.plot(g["ds"], g["y"], lw=0.8)
        imp = g[g["is_imputed"]]
        if len(imp):
            ax.scatter(imp["ds"], imp["y"], s=6, color="red", label="imputed")
        ax.set_title(f"Series {u}", fontsize=10)
        ax.set_ylabel(CFG.TARGET_COL)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Sample series (largest + random)", y=1.0)
    plt.tight_layout()
    save_fig("sample_series")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(np.log10(totals.clip(lower=1e-9) + 1), bins=40)
    axes[0].set_title("Distribution of series total (log10)")
    axes[0].set_xlabel("log10(total + 1)")
    axes[0].set_ylabel("Number of series")
    axes[1].hist(len_per_series.values, bins=40)
    axes[1].set_title("Series length distribution")
    axes[1].set_xlabel("Number of time steps")
    axes[1].set_ylabel("Number of series")
    plt.tight_layout()
    save_fig("panel_distributions")
    print(f"Top 10% series menyumbang {totals.iloc[:max(1, len(totals) // 10)].sum() / max(totals.sum(), 1e-9):.1%} total volume")
else:
    PLOT_SERIES = SERIES[:1]

# %% [markdown]
# ### 5.1 Pola musiman (seasonal boxplot)
#
# **Cara membaca:** kotak per hari-dalam-minggu / bulan / jam. Kalau median antar kotak berbeda jauh → seasonality kuat pada periode itu
# → fitur kalender tsb penting & seasonal naive kuat. Kalau semua kotak mirip → seasonality lemah.
#
# **Jika X maka Y:** weekend jauh lebih tinggi/rendah → pastikan `dayofweek` jadi fitur (otomatis). Bulan Desember/bulan Lebaran melonjak →
# fitur `month` + holiday penting; cek juga plot efek Lebaran di bawah.

# %%
def boxplot_compat(ax, groups, labels):
    """matplotlib>=3.9 pakai tick_labels, versi lama pakai labels."""
    try:
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(groups, labels=labels, showfliers=False)


_s = EDA_SERIES.to_frame("y")
_s["dayofweek"] = _s.index.dayofweek
_s["month"] = _s.index.month
_s["hour"] = _s.index.hour
_s["year"] = _s.index.year
span_days = (EDA_SERIES.index.max() - EDA_SERIES.index.min()).days
panels = []
if KIND in ("D", "B", "H", "T"):
    panels.append(("dayofweek", "Day of week (0=Mon)"))
if KIND in ("H", "T"):
    panels.append(("hour", "Hour of day"))
if span_days >= 360 and KIND not in ("Y",):
    panels.append(("month", "Month"))
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, (col, lab) in zip(axes, panels):
        groups = [g["y"].values for _, g in _s.groupby(col)]
        boxplot_compat(ax, groups, sorted(_s[col].unique()))
        ax.set_title(f"{CFG.TARGET_COL} by {lab}")
        ax.set_xlabel(lab)
        ax.set_ylabel(CFG.TARGET_COL)
    plt.tight_layout()
    save_fig("seasonal_boxplots")
    for col, lab in panels:
        med = _s.groupby(col)["y"].median()
        ratio = med.max() / (abs(med.min()) + 1e-9) if med.min() > 0 else np.nan
        print(f"{lab}: median tertinggi di {med.idxmax()} , terendah di {med.idxmin()}, rasio max/min = {ratio:.2f}")
else:
    print("Seasonal boxplot dilewati (frekuensi terlalu kasar atau data < 1 tahun).")

if span_days >= 2 * 365 and KIND in ("D", "B", "W", "M", "H"):
    ym = _s.groupby(["year", "month"])["y"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for yr in ym.columns:
        ax.plot(ym.index, ym[yr], marker="o", label=str(yr))
    ax.set_title(f"Year-over-year: mean {CFG.TARGET_COL} by month")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Mean {CFG.TARGET_COL}")
    ax.legend(title="Year", ncol=2)
    save_fig("year_over_year")

# %% [markdown]
# ### 5.2 STL decomposition (trend + seasonal + residual)
#
# STL memecah deret menjadi **trend**, **seasonal**, dan **residual** (robust terhadap outlier).
# - **Trend strength** `F_T = max(0, 1 - Var(R)/Var(T+R))` dan **seasonal strength** `F_S = max(0, 1 - Var(R)/Var(S+R))`.
# - Nilai > 0.64 dianggap **kuat** (aturan yang dipakai di literatur forecasting, mis. Hyndman & Athanasopoulos).
#
# **Jika X maka Y:**
# - `F_S` kuat → SARIMA dengan D=1 / Holt-Winters seasonal / fitur lag musiman penting.
# - `F_T` kuat → model dengan trend (Holt, drift, ETS trend damped) atau ML dengan `diff`.
# - Residual masih menunjukkan pola berkala → ada seasonality kedua (mis. tahunan pada data harian) → fitur `month`/`dayofyear` membantu.

# %%
def strength_of(series, period):
    """Return (trend_strength, seasonal_strength, stl_result) atau (nan, nan, None)."""
    if not HAS_SM or period < 2 or len(series) < 2 * period + 1:
        return np.nan, np.nan, None
    try:
        res = STL(series.values, period=period, robust=True).fit()
        r, t, s = res.resid, res.trend, res.seasonal
        ft = max(0.0, 1 - np.var(r) / max(np.var(t + r), 1e-12))
        fs = max(0.0, 1 - np.var(r) / max(np.var(s + r), 1e-12))
        return ft, fs, res
    except Exception as e:
        print("STL gagal:", e)
        return np.nan, np.nan, None


STL_PERIOD = SEASON if SEASON <= 400 else 7
_stl_series = EDA_SERIES.iloc[-5000:]
F_TREND, F_SEAS, _stl = strength_of(_stl_series, STL_PERIOD)
if _stl is not None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    for ax, (nm, v) in zip(axes, [("Observed", _stl_series.values), ("Trend", _stl.trend), ("Seasonal", _stl.seasonal),
                                  ("Residual", _stl.resid)]):
        ax.plot(_stl_series.index, v, lw=0.8)
        ax.set_ylabel(nm)
    axes[0].set_title(f"STL decomposition (period={STL_PERIOD})")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    save_fig("stl_decomposition")
    print(f"Trend strength    F_T = {F_TREND:.3f} -> {'KUAT' if F_TREND > 0.64 else 'lemah/sedang'}")
    print(f"Seasonal strength F_S = {F_SEAS:.3f} -> {'KUAT' if F_SEAS > 0.64 else 'lemah/sedang'}")
else:
    print("STL dilewati (statsmodels tidak ada / data terlalu pendek untuk 2 season).")

# %% [markdown]
# ### 5.3 ACF / PACF
#
# **Cara membaca:**
# - **ACF** (autocorrelation): korelasi deret dengan dirinya sendiri k langkah sebelumnya. Spike di lag m, 2m, ... → seasonality.
#   ACF turun sangat lambat → non-stasioner (ada trend) → perlu differencing.
# - **PACF**: korelasi langsung di lag k setelah efek lag lebih pendek dihilangkan. PACF terpotong setelah lag p → kandidat AR(p).
# - Area biru = interval kepercayaan 95%. Bar di luar area = signifikan.
#
# **Jika X maka Y:** lag yang signifikan di ACF/PACF = kandidat bagus untuk `CFG.LAGS` model ML.

# %%
if HAS_SM:
    nlags = int(min(max(3 * SEASON, 20), len(EDA_SERIES) // 2 - 1, 200))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    try:
        plot_acf(EDA_SERIES.values, lags=nlags, ax=axes[0])
        plot_pacf(EDA_SERIES.values, lags=min(nlags, len(EDA_SERIES) // 2 - 1), ax=axes[1], method="ywm")
        axes[0].set_title("ACF")
        axes[1].set_title("PACF")
        for ax in axes:
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")
        plt.tight_layout()
        save_fig("acf_pacf")
        _acf = acf(EDA_SERIES.values, nlags=nlags, fft=True)
        top = np.argsort(-np.abs(_acf[1:]))[:8] + 1
        print("Lag dengan |ACF| terbesar:", sorted(top.tolist()))
    except Exception as e:
        plt.close("all")
        print("ACF/PACF gagal:", e)
else:
    print("statsmodels tidak ada -> ACF/PACF dilewati.")

# %% [markdown]
# ### 5.4 Uji stasioneritas: ADF + KPSS
#
# - **ADF**: H0 = ada unit root (**non-stasioner**). p < 0.05 → tolak H0 → stasioner.
# - **KPSS**: H0 = **stasioner**. p < 0.05 → tolak H0 → non-stasioner.
#
# **Aturan interpretasi gabungan (pakai keduanya!):**
#
# | ADF | KPSS | Kesimpulan | Tindakan |
# |---|---|---|---|
# | stasioner | stasioner | **Stasioner** | model langsung pada level (d=0) |
# | non-stasioner | non-stasioner | **Non-stasioner** | differencing (d=1) / model dengan trend |
# | non-stasioner | stasioner | **Trend-stationary** | detrend (model dengan trend deterministik) |
# | stasioner | non-stasioner | **Difference-stationary** | differencing |
#
# Catatan: model tree (LightGBM) tidak butuh stasioner, tapi **tidak bisa ekstrapolasi trend** → kalau non-stasioner karena trend, coba `diff`.
#
# **Untuk laporan:** tabel hasil ADF/KPSS (statistik, p-value, kesimpulan) pada level dan setelah differencing.

# %%
def stationarity_tests(x):
    """ADF + KPSS -> dict hasil + kesimpulan."""
    x = pd.Series(x).dropna().values
    out = {"adf_stat": np.nan, "adf_p": np.nan, "kpss_stat": np.nan, "kpss_p": np.nan, "conclusion": "n/a"}
    if not HAS_SM or len(x) < 12 or np.std(x) == 0:
        return out
    try:
        a = adfuller(x, autolag="AIC")
        out["adf_stat"], out["adf_p"] = a[0], a[1]
    except Exception:
        pass
    try:
        k = kpss(x, regression="c", nlags="auto")
        out["kpss_stat"], out["kpss_p"] = k[0], k[1]
    except Exception:
        pass
    adf_st = out["adf_p"] < 0.05 if np.isfinite(out["adf_p"]) else None
    kpss_st = out["kpss_p"] >= 0.05 if np.isfinite(out["kpss_p"]) else None
    if adf_st is None or kpss_st is None:
        out["conclusion"] = "inconclusive"
    elif adf_st and kpss_st:
        out["conclusion"] = "stationary"
    elif not adf_st and not kpss_st:
        out["conclusion"] = "non-stationary -> difference"
    elif not adf_st and kpss_st:
        out["conclusion"] = "trend-stationary -> detrend"
    else:
        out["conclusion"] = "difference-stationary -> difference"
    return out


rows = []
for label, x in [("level", EDA_SERIES), ("diff(1)", EDA_SERIES.diff()),
                 (f"seasonal diff({SEASON})", EDA_SERIES.diff(SEASON) if SEASON > 1 else EDA_SERIES.diff())]:
    r = stationarity_tests(x)
    r["series"] = label
    rows.append(r)
stat_df = pd.DataFrame(rows)[["series", "adf_stat", "adf_p", "kpss_stat", "kpss_p", "conclusion"]]
display(stat_df.round(4))
save_table(stat_df, "stationarity_tests")

if IS_PANEL and HAS_SM:
    per = []
    for u in SERIES[:30]:
        r = stationarity_tests(df.loc[df["unique_id"] == u, "y"])
        r["unique_id"] = u
        per.append(r)
    per = pd.DataFrame(per)
    print("Ringkasan kesimpulan per series (maks 30 series):")
    display(per["conclusion"].value_counts().to_frame("n_series"))

# %% [markdown]
# ### 5.5 Efek hari libur Indonesia & Lebaran
#
# Tabel libur bawaan: Tahun Baru, Imlek, Jumat Agung, Hari Buruh, Kenaikan Isa Almasih, Pancasila, Idul Fitri (2 hari), Idul Adha,
# Kemerdekaan, Natal + tabel tanggal **Lebaran 2019–2027** (⚠️ verifikasi tanggal resmi!). Tambah libur lain lewat `CFG.CUSTOM_HOLIDAYS`.
#
# **Cara membaca:** rata-rata target di hari libur vs bukan libur; plot rata-rata target vs **jarak hari ke Lebaran**
# (negatif = sebelum Lebaran). Retail biasanya naik tajam ~2 minggu sebelum Lebaran lalu anjlok saat hari H.
#
# **Jika X maka Y:** efek besar → biarkan `USE_HOLIDAY_FEATURES=True` (sudah default) & pertimbangkan exog tambahan (mis. cuti bersama).

# %%
if CAL_FEATS is not None:
    cal_hist = CAL_FEATS.reindex(EDA_SERIES.index)
    hcol = "is_holiday" if "is_holiday" in cal_hist else ("holiday_count" if "holiday_count" in cal_hist else None)
    if hcol is not None and cal_hist[hcol].sum() > 0:
        is_h = cal_hist[hcol] > 0
        m_h, m_n = EDA_SERIES[is_h].mean(), EDA_SERIES[~is_h].mean()
        print(f"Rata-rata {CFG.TARGET_COL}: periode libur = {m_h:.4g} | bukan libur = {m_n:.4g} | rasio = {m_h / (m_n + 1e-9):.2f}")
    if "days_to_lebaran" in cal_hist and KIND in ("D", "B") and (cal_hist["days_to_lebaran"].abs() <= 30).sum() > 20:
        tmp = pd.DataFrame({"d": cal_hist["days_to_lebaran"], "y": EDA_SERIES / EDA_SERIES.rolling(56, min_periods=7, center=True).median()})
        tmp = tmp[tmp["d"].abs() <= 30].groupby("d")["y"].mean()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(tmp.index, tmp.values, marker="o", ms=3)
        ax.axvline(0, color="red", ls="--", label="Idul Fitri")
        ax.axhline(1, color="gray", lw=0.8)
        ax.set_title("Lebaran effect: target relative to local median")
        ax.set_xlabel("Days relative to Idul Fitri")
        ax.set_ylabel("Relative level (1 = normal)")
        ax.legend()
        save_fig("lebaran_effect")
    else:
        print("Plot efek Lebaran dilewati (bukan data harian atau rentang data tidak mencakup Lebaran).")
else:
    print("Fitur kalender tidak aktif.")

# %% [markdown]
# ## 6. Desain validasi: time-based backtesting
#
# **Kenapa TIDAK boleh random K-Fold?**
# 1. Random K-Fold mencampur masa depan ke data latih → model "melihat" pola setelah tanggal yang diprediksi → skor CV terlalu optimis (leakage).
# 2. Fitur lag/rolling di baris validasi dihitung dari nilai yang mungkin ada di fold latih → bocor.
# 3. Di dunia nyata (dan di lomba) kita selalu memprediksi **setelah** data terakhir. Validasi harus meniru itu.
#
# **Cara kerja backtest (rolling-origin evaluation):**
# ```
# window 3: [===== train =====][gap][== valid H ==]
# window 2: [======= train =======][gap][== valid H ==]
# window 1: [========= train =========][gap][== valid H ==]   <- paling dekat dengan akhir data
# ```
# - **expanding**: train selalu mulai dari awal data (lebih banyak data). **rolling**: panjang train tetap (`ROLLING_TRAIN_SIZE`),
#   berguna kalau pola lama sudah tidak relevan (mis. sebelum vs sesudah COVID).
# - **gap**: jeda antara akhir train dan awal periode yang dievaluasi, meniru jeda train→test di lomba.
# - Setiap model dilatih ulang di setiap window → skor = rata-rata semua window (lebih stabil daripada satu split).
#
# **Cara membaca output:** tabel cutoff + plot posisi window. Pastikan window terakhir berakhir di tanggal terakhir data.
#
# **Jika X maka Y:**
# - `⚠️ jumlah window dikurangi` → data terlalu pendek untuk N window; kurangi `HORIZON` atau `N_BACKTEST_WINDOWS`.
# - Skor sangat berbeda antar window (std besar) → ada perubahan regime/musim; pertimbangkan `WINDOW_TYPE="rolling"` dan lebih banyak window.
#
# **Untuk laporan:** jelaskan skema validasi: *"Kami menggunakan expanding-window backtesting dengan 3 cutoff, horizon 30 hari,
# sehingga evaluasi meniru kondisi prediksi sebenarnya dan bebas dari kebocoran data masa depan."*

# %%
GRID = pd.date_range(df["ds"].min(), GLOBAL_END, freq=OFF)
STEP = int(CFG.BACKTEST_STEP or H)
MIN_TRAIN = max(2 * SEASON + 1, 3 * H_TOTAL, 20) if len(GRID) > 3 * H_TOTAL + 2 * SEASON else max(SEASON + 1, H_TOTAL, 8)
CUTOFF_IDX = []
for i in range(int(CFG.N_BACKTEST_WINDOWS)):
    c = len(GRID) - 1 - H_TOTAL - i * STEP
    if c + 1 >= MIN_TRAIN:
        CUTOFF_IDX.append(c)
if not CUTOFF_IDX:
    c = len(GRID) - 1 - H_TOTAL
    if c >= 2:
        CUTOFF_IDX = [c]
        print("⚠️ Data sangat pendek: hanya 1 window dengan train minimal.")
    else:
        raise ValueError("Data terlalu pendek untuk backtest dengan horizon ini. Kurangi CFG.HORIZON.")
if len(CUTOFF_IDX) < CFG.N_BACKTEST_WINDOWS:
    print(f"⚠️ jumlah window dikurangi menjadi {len(CUTOFF_IDX)} (data kurang panjang)")
CUTOFF_IDX = sorted(CUTOFF_IDX)
ROLL_SIZE = int(CFG.ROLLING_TRAIN_SIZE or (3 * H_TOTAL + 2 * SEASON))
bt_table = pd.DataFrame([{
    "window": w + 1,
    "train_start": GRID[max(0, c - ROLL_SIZE + 1)] if CFG.WINDOW_TYPE == "rolling" else GRID[0],
    "cutoff (train_end)": GRID[c],
    "valid_start": GRID[c + GAP + 1],
    "valid_end": GRID[min(c + H_TOTAL, len(GRID) - 1)],
} for w, c in enumerate(CUTOFF_IDX)])
display(bt_table)
save_table(bt_table, "backtest_windows")

fig, ax = plt.subplots(figsize=(13, 0.6 * len(CUTOFF_IDX) + 1.5))
for i, r in bt_table.iterrows():
    ax.barh(i, (r["cutoff (train_end)"] - r["train_start"]).days, left=r["train_start"], color="tab:blue", alpha=0.6,
            label="train" if i == 0 else None)
    ax.barh(i, (r["valid_end"] - r["valid_start"]).days + 1, left=r["valid_start"], color="tab:orange",
            label="validation" if i == 0 else None)
ax.set_yticks(range(len(bt_table)))
ax.set_yticklabels([f"window {w}" for w in bt_table["window"]])
ax.set_title(f"Backtest windows ({CFG.WINDOW_TYPE}, H={H}, gap={GAP})")
ax.set_xlabel("Date")
ax.legend(loc="upper left")
save_fig("backtest_windows")

# %% [markdown]
# ## 7. Metrik evaluasi — mana yang dipakai?
#
# | Metrik | Rumus singkat | Kapan dipakai | Catatan |
# |---|---|---|---|
# | **MAE** | mean\|y−ŷ\| | skala sama, robust outlier | satuan sama dengan target |
# | **RMSE** | √mean(y−ŷ)² | kesalahan besar sangat dihukum | sensitif outlier |
# | **MAPE** | mean\|e/y\|×100 | bisnis suka "% error" | **meledak kalau y≈0**, bias ke under-forecast |
# | **sMAPE** | mean 2\|e\|/(\|y\|+\|ŷ\|) | alternatif MAPE | tetap bermasalah kalau y & ŷ ≈ 0 |
# | **WAPE** | Σ\|e\|/Σ\|y\|×100 | **panel / banyak nol** (retail) | default `auto` notebook ini |
# | **MASE** | MAE / MAE seasonal-naive in-sample | bandingkan antar series berbeda skala | <1 = lebih baik dari seasonal naive |
# | **RMSLE** | RMSE pada log1p | target skewed, error relatif | cocok dengan `TARGET_TRANSFORM="log1p"` |
#
# **Aturan:** selalu pakai **metrik resmi lomba** sebagai `CFG.METRIC`. Kalau metrik lomba RMSLE → log1p transform hampir pasti membantu.
# Kalau MAE → median-forecast lebih optimal (pertimbangkan objective `l1`; lihat recipes). Kalau RMSE → mean-forecast (default `l2`).
#
# **Untuk laporan:** laporkan metrik resmi + 1–2 metrik pendukung (mis. WAPE & MASE) supaya pembaca paham besarnya error secara relatif.

# %%
def resolve_transform():
    t = str(CFG.TARGET_TRANSFORM).lower()
    if t == "auto":
        sk = float(pd.Series(df["y"]).skew()) if len(df) > 3 else 0.0
        t = "log1p" if (Y_NONNEG and sk > 1.0) else "none"
        print(f"TARGET_TRANSFORM auto -> '{t}' (skewness target = {sk:.2f}, nonnegative = {Y_NONNEG})")
    if "log1p" in t and not Y_NONNEG:
        print("⚠️ log1p butuh target >= 0 -> log1p dimatikan")
        t = t.replace("log1p_", "").replace("log1p", "none")
    if t not in ("none", "log1p", "diff", "log1p_diff"):
        print(f"⚠️ TARGET_TRANSFORM '{t}' tidak dikenal -> none")
        t = "none"
    return t


TRANSFORM = resolve_transform()
USE_LOG = "log1p" in TRANSFORM
USE_DIFF = "diff" in TRANSFORM
CLIP_ZERO = (Y_NONNEG if CFG.CLIP_NEGATIVE_PRED == "auto" else bool(CFG.CLIP_NEGATIVE_PRED))
SERIES_RANK = df.groupby("unique_id")["y"].sum().sort_values(ascending=False).index.tolist()
UID_CODE = {u: i for i, u in enumerate(sorted(SERIES))}
DS_MIN = df["ds"].min()
print(f"Transform ML = {TRANSFORM} | log untuk model statistik = {USE_LOG} | clip prediksi >= 0: {CLIP_ZERO}")

# %% [markdown]
# ## 8. Model
#
# ### 8.1 Baseline (WAJIB dibandingkan)
# - **naive**: ŷ = nilai terakhir. **seasonal naive (snaive)**: ŷ = nilai di season sebelumnya (mis. Senin depan = Senin lalu).
# - **ma** (moving average): rata-rata `MA_WINDOW` titik terakhir. **drift**: garis dari titik pertama ke terakhir, diperpanjang.
#
# ### 8.2 Model statistik (per series, statsmodels)
# - **ses**: Simple Exponential Smoothing (level saja). **holt_winters**: level + trend (damped) + seasonal.
# - **ets**: ETS state-space; dipilih otomatis dari beberapa konfigurasi (trend none/add/damped × seasonal none/add) berdasarkan **AIC**.
# - **theta**: Theta method (pemenang kompetisi M3), kuat untuk data bulanan/kuartalan.
# - **sarimax**: SARIMA(p,d,q)(P,D,Q)m + exog. `d` dipilih via uji KPSS, `D` via seasonal strength STL (>0.64), lalu **grid search p,q,P,Q
#   berdasarkan AIC** dengan batas waktu `SARIMA_TIME_LIMIT` per series (supaya tidak hang).
#
# ### 8.3 Model ML global (satu model untuk semua series)
# Fitur: lag, rolling mean/std (dihitung dari nilai yang sudah di-*shift* → tanpa leakage), EWM, fitur tanggal, libur & Lebaran,
# exog, kode series + statistik level series. Dua strategi multi-step:
# - **recursive**: 1 model memprediksi 1 langkah, prediksinya dipakai sebagai lag untuk langkah berikutnya. Cepat, tapi error bisa menumpuk.
# - **direct**: model terpisah per kelompok horizon (bucket) yang hanya memakai lag ≥ horizon. Tidak ada akumulasi error, tapi lebih lama.
#
# Transformasi target: `log1p` (varian stabil, cocok RMSLE), `diff` (target = selisih terhadap lag terakhir yang diketahui → membantu model tree
# mengikuti trend), `log1p_diff` (keduanya).
#
# ### 8.4 Opsional: Prophet / statsforecast (hanya kalau terinstall; otomatis di-skip kalau tidak ada).
#
# **Jika X maka Y:**
# - Semua model kalah dari **snaive** → cek frekuensi/season length, cek leakage terbalik (data test bocor ke train?), cek transform.
# - SARIMAX selalu fallback (`series fallback ke snaive`) → data terlalu pendek atau konstan; abaikan model itu.
# - ML jauh lebih baik di window awal tapi buruk di window terakhir → trend/level shift → coba `diff`.

# %%
def fc_naive(y, h, ctx, **kw):
    return np.repeat(float(y[-1]), h)


def fc_snaive(y, h, ctx, **kw):
    m = ctx["m"]
    if m <= 1 or len(y) < m:
        return fc_naive(y, h, ctx)
    last = np.asarray(y[-m:], float)
    return np.array([last[i % m] for i in range(h)])


def fc_ma(y, h, ctx, **kw):
    w = int(CFG.MA_WINDOW or max(ctx["m"], 3))
    return np.repeat(float(np.mean(y[-w:])), h)


def fc_drift(y, h, ctx, **kw):
    n = len(y)
    if n < 2:
        return fc_naive(y, h, ctx)
    slope = (y[-1] - y[0]) / (n - 1)
    return y[-1] + slope * np.arange(1, h + 1)


def _seasonal_ok(y, m, cycles=2):
    return m > 1 and len(y) >= cycles * m + 2


def fc_ses(y, h, ctx, **kw):
    res = SimpleExpSmoothing(np.asarray(y, float), initialization_method="estimated").fit()
    return np.asarray(res.forecast(h))


def fc_holt_winters(y, h, ctx, **kw):
    y = np.asarray(y, float)
    m = ctx["m"]
    seasonal = CFG.HW_SEASONAL if _seasonal_ok(y, m) else None
    if seasonal == "mul" and (y <= 0).any():
        seasonal = "add"
    mod = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=seasonal,
                               seasonal_periods=m if seasonal else None, initialization_method="estimated")
    return np.asarray(mod.fit().forecast(h))


def fc_ets(y, h, ctx, **kw):
    """ETS auto: pilih konfigurasi dengan AIC terkecil."""
    y = np.asarray(y, float)
    m = ctx["m"]
    if not HAS_ETS:
        return fc_holt_winters(y, h, ctx)
    configs = [(None, False, None), ("add", False, None), ("add", True, None)]
    if _seasonal_ok(y, m) and m <= 60:
        configs += [(None, False, "add"), ("add", True, "add")]
    best = None
    for trend, damped, seas in configs:
        try:
            res = ETSModel(y, error="add", trend=trend, damped_trend=damped, seasonal=seas,
                           seasonal_periods=m if seas else None).fit(disp=False, maxiter=200)
            if np.isfinite(res.aic) and (best is None or res.aic < best[0]):
                best = (res.aic, res)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("ETS: tidak ada konfigurasi yang berhasil")
    return np.asarray(best[1].forecast(h))


def fc_theta(y, h, ctx, **kw):
    y = np.asarray(y, float)
    m = ctx["m"]
    if _seasonal_ok(y, m):
        res = ThetaModel(y, period=m, deseasonalize=True, method="auto").fit()
    else:
        res = ThetaModel(y, period=1, deseasonalize=False).fit()
    return np.asarray(res.forecast(h))


def ndiffs_kpss(x, max_d=2):
    """Jumlah differencing (d) berdasarkan uji KPSS berulang (mirip pmdarima.ndiffs)."""
    x = np.asarray(x, float)
    d = 0
    while d < max_d and len(x) > 12:
        try:
            p = kpss(x, regression="c", nlags="auto")[1]
        except Exception:
            break
        if p >= 0.05:
            break
        x = np.diff(x)
        d += 1
    return d


SARIMA_ORDERS = {}


def fc_sarimax(y, h, ctx, ex_tr=None, ex_fu=None, uid=None, **kw):
    """Auto-SARIMAX: grid search kecil berdasarkan AIC dengan batas waktu."""
    y = np.asarray(y, float)[-CFG.SARIMA_MAX_TRAIN:]
    if ex_tr is not None:
        ex_tr = np.nan_to_num(np.asarray(ex_tr, float)[-len(y):])
        ex_fu = np.nan_to_num(np.asarray(ex_fu, float))
    m = ctx["m"]
    seasonal = 1 < m <= CFG.SARIMA_MAX_SEASON and len(y) >= 3 * m
    d = ndiffs_kpss(y)
    D = 0
    if seasonal:
        _, fs, _ = strength_of(pd.Series(y), m)
        D = 1 if (np.isfinite(fs) and fs > 0.64) else 0
    cands = []
    for p in range(CFG.SARIMA_MAX_P + 1):
        for q in range(CFG.SARIMA_MAX_Q + 1):
            for P in (range(CFG.SARIMA_MAX_PS + 1) if seasonal else [0]):
                for Q in (range(CFG.SARIMA_MAX_QS + 1) if seasonal else [0]):
                    cands.append((p, q, P, Q))
    cands.sort(key=lambda c: (sum(c), c))
    t0 = time.time()
    best = None
    for p, q, P, Q in cands:
        if best is not None and time.time() - t0 > CFG.SARIMA_TIME_LIMIT:
            break
        try:
            mod = SARIMAX(y, exog=ex_tr, order=(p, d, q),
                          seasonal_order=(P, D, Q, m) if seasonal else (0, 0, 0, 0),
                          trend="c" if d + D == 0 else "n",
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=50)
            if np.isfinite(res.aic) and (best is None or res.aic < best[0]):
                best = (res.aic, (p, d, q, P, D, Q, m if seasonal else 0), res)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("SARIMAX: semua kandidat gagal")
    SARIMA_ORDERS[uid] = best[1]
    return np.asarray(best[2].forecast(h, exog=ex_fu) if ex_tr is not None else best[2].forecast(h))


def fc_prophet(y, h, ctx, ds_tr=None, ds_fu=None, ex_tr=None, ex_fu=None, **kw):
    dfp = pd.DataFrame({"ds": pd.to_datetime(ds_tr), "y": np.asarray(y, float)})
    hol = get_holiday_table(range(dfp["ds"].min().year - 1, pd.to_datetime(ds_fu).max().year + 2))
    hol = hol.rename(columns={"date": "ds"}) if len(hol) else None
    span = (dfp["ds"].max() - dfp["ds"].min()).days
    mdl = Prophet(holidays=hol, weekly_seasonality=KIND in ("D", "B", "H"), daily_seasonality=KIND == "H",
                  yearly_seasonality=span >= 365 * 2 and KIND != "Y")
    if ex_tr is not None:
        for j, c in enumerate(EXOG):
            dfp[c] = np.asarray(ex_tr, float)[:, j]
            mdl.add_regressor(c)
    mdl.fit(dfp)
    fut = pd.DataFrame({"ds": pd.to_datetime(ds_fu)})
    if ex_fu is not None:
        for j, c in enumerate(EXOG):
            fut[c] = np.nan_to_num(np.asarray(ex_fu, float)[:, j])
    return mdl.predict(fut)["yhat"].values


def fc_sf_autoets(y, h, ctx, **kw):
    return np.asarray(SF_AutoETS(season_length=max(ctx["m"], 1)).forecast(y=np.asarray(y, float), h=h)["mean"])


def fc_sf_autoarima(y, h, ctx, **kw):
    return np.asarray(SF_AutoARIMA(season_length=max(ctx["m"], 1)).forecast(y=np.asarray(y, float), h=h)["mean"])


UNIVARIATE = {"naive": fc_naive, "snaive": fc_snaive, "ma": fc_ma, "drift": fc_drift, "ses": fc_ses,
              "holt_winters": fc_holt_winters, "ets": fc_ets, "theta": fc_theta, "sarimax": fc_sarimax,
              "prophet": fc_prophet, "sf_autoets": fc_sf_autoets, "sf_autoarima": fc_sf_autoarima}
HEAVY = {"ses", "holt_winters", "ets", "theta", "sarimax", "prophet", "sf_autoets", "sf_autoarima"}
LOG_OK = {"ses", "holt_winters", "ets", "theta", "sarimax", "sf_autoets", "sf_autoarima"}
ML_NAMES = ["lgb", "xgb", "cat", "hgb"]
AVAILABLE = {"naive": True, "snaive": True, "ma": True, "drift": True, "ses": HAS_SM, "holt_winters": HAS_SM,
             "ets": HAS_SM, "theta": HAS_THETA, "sarimax": HAS_SM, "prophet": HAS_PROPHET, "sf_autoets": HAS_SF,
             "sf_autoarima": HAS_SF, "lgb": HAS_LGB, "xgb": HAS_XGB, "cat": HAS_CAT, "hgb": True}
ACTIVE_MODELS = []
for _m in list(dict.fromkeys(CFG.MODELS)):  # dict.fromkeys -> buang duplikat, urutan tetap
    if _m not in AVAILABLE:
        print(f"⚠️ model '{_m}' tidak dikenal -> skip")
    elif not AVAILABLE[_m]:
        print(f"⚠️ model '{_m}' di-skip (library tidak tersedia)")
    else:
        ACTIVE_MODELS.append(_m)
if IS_PANEL and N_SERIES > CFG.STAT_MAX_SERIES and any(m in HEAVY for m in ACTIVE_MODELS):
    print(f"⚠️ {N_SERIES} series > STAT_MAX_SERIES={CFG.STAT_MAX_SERIES}: model statistik hanya untuk {CFG.STAT_MAX_SERIES} "
          "series terbesar, sisanya snaive (skor model statistik jadi campuran).")
print("Model aktif:", ACTIVE_MODELS)


def run_univariate(name, train_df, future_df, ctx):
    """Jalankan model univariat per series. Gagal fit -> fallback snaive (dihitung)."""
    fn = UNIVARIATE[name]
    heavy = name in HEAVY
    allowed = set(SERIES_RANK[:CFG.STAT_MAX_SERIES]) if heavy else None
    out = np.full(len(future_df), np.nan)
    fut_groups = future_df.groupby("unique_id", sort=False).indices
    tr_groups = train_df.groupby("unique_id", sort=False).indices
    yv = train_df["y"].values.astype(float)
    dsv = train_df["ds"].values
    hv = future_df["h"].values
    exv_tr = train_df[EXOG].values.astype(float) if EXOG else None
    exv_fu = future_df[EXOG].values.astype(float) if EXOG else None
    n_fb = 0
    use_log = ctx["stat_log"] and name in LOG_OK
    for u, fidx in fut_groups.items():
        tidx = tr_groups.get(u)
        if tidx is None or len(tidx) == 0:
            continue
        hs = hv[fidx]
        order = np.argsort(hs)
        hmax = int(hs.max())
        y = yv[tidx]
        tail = CFG.STAT_MAX_TRAIN if heavy else len(y)
        y = y[-tail:]
        kw = {"uid": u}
        if name in ("sarimax", "prophet") and EXOG:
            kw["ex_tr"] = exv_tr[tidx][-len(y):]
            kw["ex_fu"] = exv_fu[fidx][order]
        if name == "prophet":
            kw["ds_tr"] = dsv[tidx][-len(y):]
            kw["ds_fu"] = future_df["ds"].values[fidx][order]
        f_use = fn if (not heavy or u in allowed) else fc_snaive
        try:
            yy = np.log1p(np.clip(y, 0, None)) if use_log else y
            f = np.asarray(f_use(yy, hmax, ctx, **kw), float)
            if use_log:
                f = np.expm1(np.clip(f, -50, 50))
            if len(f) < hmax or not np.isfinite(f).all():
                raise ValueError("forecast tidak valid")
        except Exception:
            f = fc_snaive(y, hmax, ctx)
            n_fb += 1
        out[fidx] = f[hs - 1]
    if n_fb:
        print(f"    [{name}] {n_fb} series fallback ke snaive (gagal fit)")
    return out

# %%
def lag_set(k):
    """Daftar lag untuk shift minimal k (k=1 untuk recursive)."""
    s = set(k + l - 1 for l in LAGS)
    if SEASON > 1:
        for j in (1, 2, 3):
            if SEASON * j >= k and SEASON * j <= max(LAGS) + k + SEASON:
                s.add(SEASON * j)
    return sorted(s)


def fwd_t(y):
    return np.log1p(np.clip(np.asarray(y, float), 0, None)) if USE_LOG else np.asarray(y, float)


def inv_t(yt):
    return np.expm1(np.clip(np.asarray(yt, float), -50, 50)) if USE_LOG else np.asarray(yt, float)


def date_features(ds):
    """Fitur tanggal sesuai frekuensi."""
    ds = pd.DatetimeIndex(ds)
    f = {}
    f["time_idx"] = (ds - DS_MIN).total_seconds().values / 86400.0
    if KIND in ("H", "T"):
        f["hour"] = ds.hour
    if KIND in ("H", "T", "D", "B"):
        f["dayofweek"] = ds.dayofweek
        f["day"] = ds.day
        f["is_weekend"] = (ds.dayofweek >= 5).astype(int)
        f["dayofyear"] = ds.dayofyear
        f["is_month_start"] = ds.is_month_start.astype(int)
        f["is_month_end"] = ds.is_month_end.astype(int)
    if KIND in ("H", "T", "D", "B", "W"):
        f["weekofyear"] = ds.isocalendar().week.values.astype(int)
    if KIND not in ("Y",):
        f["month"] = ds.month
        f["quarter"] = ds.quarter
    f["year"] = ds.year
    return pd.DataFrame(f, index=range(len(ds)))


def make_ml_features(frame, k, stats):
    """Fitur ML untuk shift minimal k. frame: kolom unique_id, ds, yt (+EXOG), urut per series & tanggal, index 0..n-1."""
    uid = frame["unique_id"]
    g = frame.groupby("unique_id", sort=False)["yt"]
    feats = {}
    for L in lag_set(k):
        feats[f"lag_{L}"] = g.shift(L)
    base = g.shift(k)
    bg = base.groupby(uid, sort=False)
    for w in ROLL_WINDOWS:
        feats[f"rmean_{w}"] = bg.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        feats[f"rstd_{w}"] = bg.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
    if CFG.USE_EWM:
        feats["ewm_a03"] = bg.transform(lambda s: s.ewm(alpha=0.3, ignore_na=True).mean())
    if SEASON > 1 and len([L for L in lag_set(k) if L % SEASON == 0]) >= 2:
        seas_cols = [f"lag_{L}" for L in lag_set(k) if L % SEASON == 0]
        feats["seasonal_lag_mean"] = pd.concat([feats[c] for c in seas_cols], axis=1).mean(axis=1)
    X = pd.DataFrame(feats, index=frame.index)
    if CFG.USE_DATE_FEATURES:
        dfe = date_features(frame["ds"].values)
        dfe.index = frame.index
        X = pd.concat([X, dfe], axis=1)
    if CAL_FEATS is not None:
        cf = CAL_FEATS.reindex(pd.DatetimeIndex(frame["ds"].values))
        cf.index = frame.index
        X = pd.concat([X, cf.add_prefix("cal_")], axis=1)
    for c in EXOG:
        X[f"exog_{c}"] = frame[c].values
    X["series_code"] = uid.map(UID_CODE).astype(float).values
    if CFG.USE_SERIES_STATS and stats is not None:
        X["series_mean"] = uid.map(stats["mean"]).values
        X["series_std"] = uid.map(stats["std"]).values
    return X


def make_ml_model(name, quantile=None):
    """Bangun regressor. quantile=None -> point forecast (L2)."""
    n, lr, seed = int(CFG.ML_N_ESTIMATORS), float(CFG.ML_LEARNING_RATE), CFG.SEED
    if name == "lgb":
        p = dict(n_estimators=n, learning_rate=lr, num_leaves=CFG.ML_NUM_LEAVES, min_child_samples=20, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=CFG.ML_N_JOBS, verbose=-1)
        if quantile is not None:
            p.update(objective="quantile", alpha=quantile)
        return lgb.LGBMRegressor(**p)
    if name == "xgb":
        return xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                min_child_weight=5, tree_method="hist", random_state=seed, n_jobs=CFG.ML_N_JOBS)
    if name == "cat":
        return cb.CatBoostRegressor(iterations=n, learning_rate=lr, depth=6, random_seed=seed, verbose=0,
                                    allow_writing_files=False, thread_count=-1)
    if name == "hgb":
        if quantile is not None:
            return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=n, learning_rate=lr,
                                                 max_leaf_nodes=31, random_state=seed)
        return HistGradientBoostingRegressor(max_iter=n, learning_rate=lr, max_leaf_nodes=31, l2_regularization=1.0,
                                             random_state=seed)
    raise ValueError(name)


def ml_training_matrix(hist, k, stats):
    """X, y untuk melatih model dengan shift minimal k."""
    X = make_ml_features(hist, k, stats)
    y = hist["yt"].values.astype(float)
    base = X[f"lag_{k}"].values.astype(float)
    if USE_DIFF:
        y = y - base
    ok = np.isfinite(y) & np.isfinite(base)
    X, y = X[ok], y[ok]
    if len(X) > CFG.ML_MAX_TRAIN_ROWS:
        keep = np.argsort(hist.loc[X.index, "ds"].values)[-CFG.ML_MAX_TRAIN_ROWS:]
        X, y = X.iloc[keep], y[keep]
    return X, y


def ml_forecast(name, train_df, future_df, ctx, quantiles=None):
    """Forecast global ML (recursive atau direct). Return (yhat, info) dengan yhat sejajar future_df."""
    hist = train_df[["unique_id", "ds", "y"] + EXOG].copy()
    hist["yt"] = fwd_t(hist["y"].values)
    hist["h"] = 0
    hist = hist.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    stats = hist.groupby("unique_id")["yt"].agg(["mean", "std"]) if CFG.USE_SERIES_STATS else None
    fut = future_df[["unique_id", "ds", "h"] + EXOG].copy()
    fut["y"] = np.nan
    fut["yt"] = np.nan
    hmax = int(fut["h"].max())
    if ctx["strategy"] == "direct":
        nb = int(CFG.DIRECT_N_BUCKETS or min(hmax, 6))
        nb = max(1, min(nb, hmax))
        chunks = [c.tolist() for c in np.array_split(np.arange(1, hmax + 1), nb) if len(c)]
        buckets = [(c[-1], c) for c in chunks]
    else:
        buckets = [(1, list(range(1, hmax + 1)))]
    max_k = max(k for k, _ in buckets)
    need = max(lag_set(max_k)) + max(ROLL_WINDOWS) + 60
    frame = pd.concat([hist.groupby("unique_id", sort=False).tail(need), fut], ignore_index=True)
    frame = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pred_t = np.full(len(frame), np.nan)
    q_pred = {q: np.full(len(frame), np.nan) for q in (quantiles or [])}
    info = {"models": []}
    for k, steps in buckets:
        Xtr, ytr = ml_training_matrix(hist, k, stats)
        if len(Xtr) < 10:
            raise RuntimeError(f"data latih ML terlalu sedikit ({len(Xtr)} baris)")
        model = make_ml_model(name).fit(Xtr, ytr)
        qmodels = {}
        for q in (quantiles or []):
            qname = "lgb" if HAS_LGB else "hgb"
            qmodels[q] = make_ml_model(qname, quantile=q).fit(Xtr, ytr)
        cols = list(Xtr.columns)
        info["models"].append((k, model))
        info.setdefault("X_train", Xtr)          # X dari bucket pertama (sejajar dengan models[0])
        info.setdefault("feature_names", cols)
        if ctx["strategy"] == "direct":
            Xf = make_ml_features(frame, k, stats)
            mask = frame["h"].isin(steps).values
            base = Xf.loc[mask, f"lag_{k}"].values if USE_DIFF else 0.0
            pred_t[mask] = model.predict(Xf.loc[mask, cols]) + base
            for q, qm in qmodels.items():
                q_pred[q][mask] = qm.predict(Xf.loc[mask, cols]) + base
        else:
            for s in steps:
                mask = (frame["h"] == s).values
                if not mask.any():
                    continue
                Xf = make_ml_features(frame, k, stats)
                base = Xf.loc[mask, f"lag_{k}"].values if USE_DIFF else 0.0
                p = model.predict(Xf.loc[mask, cols]) + base
                pred_t[mask] = p
                frame.loc[mask, "yt"] = p
                for q, qm in qmodels.items():
                    q_pred[q][mask] = qm.predict(Xf.loc[mask, cols]) + base
    frame["pred"] = inv_t(pred_t)
    for q in q_pred:
        frame[f"q_{q}"] = inv_t(q_pred[q])
    out = future_df[["unique_id", "ds"]].merge(frame[frame["h"] > 0][["unique_id", "ds", "pred"] + [f"q_{q}" for q in q_pred]],
                                               on=["unique_id", "ds"], how="left")
    if q_pred:
        info["quantiles"] = {q: out[f"q_{q}"].values for q in q_pred}
    return out["pred"].values, info


def forecast_with(name, train_df, future_df, ctx, quantiles=None):
    """Dispatcher: ML global atau univariat, + fallback NaN -> snaive, + clip >= 0."""
    if name in ML_NAMES:
        yhat, info = ml_forecast(name, train_df, future_df, ctx, quantiles=quantiles)
    else:
        yhat, info = run_univariate(name, train_df, future_df, ctx), {}
    yhat = np.asarray(yhat, float)
    if not np.isfinite(yhat).all():
        fb = run_univariate("snaive", train_df, future_df, ctx)
        yhat = np.where(np.isfinite(yhat), yhat, fb)
    if CLIP_ZERO:
        yhat = np.clip(yhat, 0, None)
    return yhat, info


CTX = {"m": SEASON, "stat_log": USE_LOG, "strategy": CFG.ML_STRATEGY}
print("Model siap. Strategi ML:", CFG.ML_STRATEGY, "| lag set (k=1):", lag_set(1))

# %% [markdown]
# ## 9. Jalankan backtest semua model + leaderboard
#
# **Tujuan:** setiap model dilatih di setiap window (hanya data ≤ cutoff), lalu memprediksi `GAP + H` langkah. Skor dihitung pada H langkah
# terakhir (setelah gap), pada titik yang **bukan hasil imputasi** (kalau `EVAL_EXCLUDE_IMPUTED=True`).
#
# **Cara membaca leaderboard:**
# - Kolom `<metric>` = skor gabungan semua window (lebih kecil = lebih baik). `<metric>_std` = variasi antar window.
# - `vs_snaive_%` = perbaikan relatif terhadap seasonal naive (positif = lebih baik). **Model yang tidak mengalahkan snaive jangan dipakai.**
# - `time_s` = total waktu (semua window) — penting kalau waktu lomba terbatas.
#
# **Jika X maka Y:**
# - `<metric>_std` > 30% dari rata-rata → model tidak stabil antar periode; prefer model yang sedikit lebih buruk tapi stabil, atau ensemble.
# - Model ML lebih buruk dari ETS di data pendek (< 2 tahun harian / < 5 tahun bulanan) → wajar; data sedikit → model statistik unggul.
# - Ada model di `FAILED_MODELS` → baca pesan error (bagian Debugging), biasanya library/versi atau data terlalu pendek.
#
# **Untuk laporan:** tabel leaderboard (top 5–8 model + baseline) adalah tabel utama bagian "Hasil & Evaluasi".

# %%
def make_future_frame(uids, dates, exog_source=None, hist=None):
    """Grid (unique_id x tanggal) + kolom h (1..len(dates)) + exog (dari sumber, diisi nilai terakhir kalau hilang)."""
    uids = np.asarray(list(uids))
    dates = pd.DatetimeIndex(dates)
    fut = pd.DataFrame({"unique_id": np.repeat(uids, len(dates)), "ds": np.tile(dates.values, len(uids)),
                        "h": np.tile(np.arange(1, len(dates) + 1), len(uids))})
    if EXOG:
        if exog_source is not None and all(c in exog_source.columns for c in EXOG):
            src = exog_source[["unique_id", "ds"] + EXOG].drop_duplicates(["unique_id", "ds"])
            fut = fut.merge(src, on=["unique_id", "ds"], how="left")
        else:
            for c in EXOG:
                fut[c] = np.nan
        if hist is not None and fut[EXOG].isna().any().any():
            last = hist.sort_values("ds").groupby("unique_id")[EXOG].last()
            fut[EXOG] = fut.groupby("unique_id", sort=False)[EXOG].ffill()
            for c in EXOG:
                fut[c] = fut[c].fillna(fut["unique_id"].map(last[c]))
        fut[EXOG] = fut[EXOG].fillna(0)
    return fut


def mase_scale(train_df, m):
    """Skala MASE per series = MAE seasonal-naive in-sample (lag m; fallback lag 1)."""
    out = {}
    for u, g in train_df.groupby("unique_id", sort=False):
        y = g["y"].values.astype(float)
        lag = m if len(y) > m + 1 else 1
        d = np.abs(y[lag:] - y[:-lag]) if len(y) > lag else np.array([np.nan])
        s = np.nanmean(d) if len(d) else np.nan
        out[u] = s if (np.isfinite(s) and s > 0) else np.nan
    return pd.Series(out)


BT_PARTS = []
BT_TIME = {m: 0.0 for m in ACTIVE_MODELS}
for w, c in enumerate(CUTOFF_IDX, 1):
    cutoff = GRID[c]
    tr = df[df["ds"] <= cutoff]
    if CFG.WINDOW_TYPE == "rolling":
        tr = tr[tr["ds"] > GRID[max(0, c - ROLL_SIZE)]]
    uids = tr["unique_id"].unique()
    dates = GRID[c + 1: c + H_TOTAL + 1]
    fut = make_future_frame(uids, dates, df, tr)
    act = fut[["unique_id", "ds", "h"]].merge(df[["unique_id", "ds", "y", "is_imputed"]], on=["unique_id", "ds"], how="left")
    act["scale"] = act["unique_id"].map(mase_scale(tr, SEASON)).values
    evm = (act["h"] > GAP) & act["y"].notna()
    if CFG.EVAL_EXCLUDE_IMPUTED:
        evm &= ~act["is_imputed"].fillna(False).astype(bool)
    print(f"--- Window {w}/{len(CUTOFF_IDX)} | cutoff {cutoff.date() if hasattr(cutoff, 'date') else cutoff} | "
          f"train rows {len(tr)} | eval points {int(evm.sum())}")
    for name in ACTIVE_MODELS:
        t0 = time.time()
        try:
            yhat, _ = forecast_with(name, tr, fut, CTX)
        except Exception as e:
            FAILED_MODELS.setdefault(name, []).append(f"window {w}: {type(e).__name__}: {str(e)[:300]}")
            print(f"    ⚠️ {name} gagal: {type(e).__name__}: {str(e)[:200]}")
            continue
        BT_TIME[name] += time.time() - t0
        r = act.copy()
        r["model"], r["window"], r["yhat"] = name, w, yhat
        BT_PARTS.append(r)
        sc = METRICS[METRIC](r.loc[evm, "y"], r.loc[evm, "yhat"], r.loc[evm, "scale"])
        print(f"    {name:<14s} {METRIC}={sc:10.4f}  ({time.time() - t0:5.1f}s)")

assert BT_PARTS, "Semua model gagal di backtest. Cek FAILED_MODELS dan konfigurasi."
BT = pd.concat(BT_PARTS, ignore_index=True)
EVAL_MASK = (BT["h"] > GAP) & BT["y"].notna()
if CFG.EVAL_EXCLUDE_IMPUTED:
    EVAL_MASK &= ~BT["is_imputed"].fillna(False).astype(bool)
EV = BT[EVAL_MASK].copy()
BT.to_csv(TAB_DIR / "backtest_predictions.csv", index=False)

# %%
def leaderboard_from(ev):
    """Leaderboard: metrik gabungan + mean/std per window."""
    rows = []
    nwin = ev["window"].nunique()
    for name, g in ev.groupby("model", sort=False):
        r = {"model": name}
        r.update(all_metrics(g["y"], g["yhat"], g["scale"]))
        per_w = [METRICS[METRIC](gw["y"], gw["yhat"], gw["scale"]) for _, gw in g.groupby("window")]
        r[f"{METRIC}_mean_window"] = float(np.nanmean(per_w))
        r[f"{METRIC}_std"] = float(np.nanstd(per_w))
        r["n_windows"] = g["window"].nunique()
        r["complete"] = r["n_windows"] == nwin
        r["time_s"] = round(BT_TIME.get(name, 0.0), 1)
        rows.append(r)
    lb = pd.DataFrame(rows).sort_values(["complete", METRIC], ascending=[False, True]).reset_index(drop=True)
    if "snaive" in set(lb["model"]):
        base = float(lb.loc[lb["model"] == "snaive", METRIC].iloc[0])
        lb["vs_snaive_%"] = (base - lb[METRIC]) / (abs(base) + 1e-12) * 100
    return lb


LB = leaderboard_from(EV)
display(LB.round(4))
save_table(LB, "leaderboard_single_models")
best_single = LB.loc[LB["complete"], "model"].iloc[0]
print(f"Model tunggal terbaik: {best_single} ({METRIC} = {LB.loc[LB['model'] == best_single, METRIC].iloc[0]:.4f})")
if "vs_snaive_%" in LB and float(LB.loc[LB["model"] == best_single, "vs_snaive_%"].iloc[0]) <= 0:
    print("⚠️ Tidak ada model yang mengalahkan seasonal naive -> cek frekuensi, season length, transform, atau data leakage.")
_std_ratio = LB.loc[0, f"{METRIC}_std"] / (abs(LB.loc[0, f"{METRIC}_mean_window"]) + 1e-12)
print(f"Variasi antar window model terbaik: {_std_ratio:.1%}" + ("  ⚠️ tidak stabil" if _std_ratio > 0.3 else "  ✅"))

per_window = EV.groupby(["model", "window"]).apply(lambda g: METRICS[METRIC](g["y"], g["yhat"], g["scale"])).unstack("window")
per_window.columns = [f"window_{c}" for c in per_window.columns]
display(per_window.loc[[m for m in LB["model"] if m in per_window.index]].round(4))
save_table(per_window.reset_index(), "score_per_window")

fig, ax = plt.subplots(figsize=(10, max(3.5, 0.4 * len(LB))))
_lb = LB.iloc[::-1]
colors = ["tab:green" if m == best_single else ("tab:gray" if m in ("naive", "snaive", "ma", "drift") else "tab:blue")
          for m in _lb["model"]]
ax.barh(_lb["model"], _lb[METRIC], xerr=_lb[f"{METRIC}_std"], color=colors, alpha=0.85)
ax.set_title(f"Backtest leaderboard ({METRIC.upper()}, lower is better; error bar = std across windows)")
ax.set_xlabel(METRIC.upper())
ax.set_ylabel("Model")
save_fig("leaderboard")

# %% [markdown]
# ### 9.1 Error per horizon step
#
# **Cara membaca:** error biasanya naik untuk langkah yang lebih jauh. Garis datar → model stabil.
# Model recursive yang error-nya naik tajam di horizon jauh → coba `ML_STRATEGY="direct"`.

# %%
top_models = LB.loc[LB["complete"], "model"].head(5).tolist()
if "snaive" in LB["model"].values and "snaive" not in top_models:
    top_models.append("snaive")
hz = (EV[EV["model"].isin(top_models)].groupby(["model", "h"])
      .apply(lambda g: METRICS[METRIC](g["y"], g["yhat"], g["scale"])).unstack("model"))
fig, ax = plt.subplots(figsize=(11, 4.5))
for mname in top_models:
    if mname in hz:
        ax.plot(hz.index, hz[mname], marker="o", ms=3, label=mname)
ax.set_title(f"{METRIC.upper()} by forecast horizon step")
ax.set_xlabel("Horizon step (h)")
ax.set_ylabel(METRIC.upper())
ax.legend()
save_fig("error_by_horizon")
save_table(hz.reset_index(), "score_by_horizon")

# %% [markdown]
# ## 10. Ensemble forecast
#
# **Kenapa ensemble?** Rata-rata beberapa model yang errornya tidak berkorelasi hampir selalu lebih stabil daripada satu model terbaik
# (pelajaran utama kompetisi M4/M5). Metode yang dicoba dari **TOP-K** model (lengkap di semua window):
# - `ens_mean` (rata-rata sederhana — paling robust), `ens_median` (tahan outlier model), `ens_invw` (bobot 1/skor),
#   `ens_nnls` (bobot non-negatif dioptimasi di backtest — paling rawan overfit karena bobot dipilih di data yang sama).
#
# **Jika X maka Y:** ensemble hanya lebih baik < 1% dari model terbaik → pilih yang lebih sederhana (lebih mudah dijelaskan di laporan).
# `ens_nnls` jauh lebih baik dari `ens_mean` tapi windownya sedikit → curigai overfit; prefer `ens_mean`/`ens_invw`.

# %%
ENSEMBLES = {}
complete_models = LB.loc[LB["complete"], "model"].tolist()
K = int(min(CFG.ENSEMBLE_TOP_K, len(complete_models)))
ENS_TOP = complete_models[:K]
if K >= 2:
    piv = EV[EV["model"].isin(ENS_TOP)].pivot_table(index=["window", "unique_id", "ds"], columns="model", values="yhat")
    piv = piv.dropna()
    meta = (EV[EV["model"] == ENS_TOP[0]].set_index(["window", "unique_id", "ds"])[["h", "y", "scale"]]
            .reindex(piv.index))
    scores = LB.set_index("model").loc[ENS_TOP, METRIC].values.astype(float)
    inv = 1.0 / np.maximum(scores, 1e-12)
    ENSEMBLES["ens_mean"] = (ENS_TOP, np.ones(K) / K)
    ENSEMBLES["ens_invw"] = (ENS_TOP, inv / inv.sum())
    ENSEMBLES["ens_median"] = (ENS_TOP, None)
    if HAS_SCIPY:
        try:
            from scipy.optimize import nnls
            wts, _ = nnls(piv[ENS_TOP].values, meta["y"].values)
            if wts.sum() > 0:
                ENSEMBLES["ens_nnls"] = (ENS_TOP, wts / wts.sum())
        except Exception as e:
            print("nnls gagal:", e)
    ens_rows = []
    for en, (members, wts) in ENSEMBLES.items():
        P = piv[members].values
        yh = np.median(P, axis=1) if wts is None else P @ wts
        if CLIP_ZERO:
            yh = np.clip(yh, 0, None)
        r = meta.reset_index().copy()
        r["model"], r["yhat"] = en, yh
        ens_rows.append(r)
        print(f"{en:<11s} members={members} weights={None if wts is None else np.round(wts, 3).tolist()}")
    EV_ALL = pd.concat([EV] + ens_rows, ignore_index=True)
    BT_TIME.update({en: 0.0 for en in ENSEMBLES})
    LB_ALL = leaderboard_from(EV_ALL)
else:
    print("Kurang dari 2 model lengkap -> ensemble dilewati.")
    EV_ALL, LB_ALL = EV, LB
display(LB_ALL.round(4))
save_table(LB_ALL, "leaderboard_all")

# Pilih strategi final
fs = str(CFG.FINAL_STRATEGY)
cands_ok = LB_ALL.loc[LB_ALL["complete"], "model"].tolist()
if fs in cands_ok:
    FINAL_NAME = fs
elif fs == "best":
    FINAL_NAME = best_single
elif fs == "ensemble" and ENSEMBLES:
    FINAL_NAME = min(ENSEMBLES, key=lambda e: float(LB_ALL.loc[LB_ALL["model"] == e, METRIC].iloc[0]))
else:
    # auto: ens_nnls (bobot dioptimasi di backtest yang sama) hanya boleh kalau window >= 3 (risiko overfit)
    _auto = [m for m in cands_ok if not (m == "ens_nnls" and len(CUTOFF_IDX) < 3)]
    FINAL_NAME = _auto[0]
    if fs not in ("auto", "best", "ensemble"):
        print(f"⚠️ FINAL_STRATEGY '{fs}' tidak ada di leaderboard -> pakai auto")
if FINAL_NAME in ENSEMBLES:
    FINAL_MEMBERS, FINAL_WEIGHTS = ENSEMBLES[FINAL_NAME]
else:
    FINAL_MEMBERS, FINAL_WEIGHTS = [FINAL_NAME], np.array([1.0])
FINAL_SCORE = float(LB_ALL.loc[LB_ALL["model"] == FINAL_NAME, METRIC].iloc[0])
print(f"\n==> Pilihan final: {FINAL_NAME} | {METRIC} backtest = {FINAL_SCORE:.4f} | members = {FINAL_MEMBERS}")

# %% [markdown]
# ## 11. Plot forecast vs actual per backtest window & diagnostik residual
#
# **Cara membaca plot:** garis hitam = aktual, biru = model final, abu-abu putus-putus = seasonal naive, garis vertikal = cutoff.
# Cek apakah model mengikuti **level**, **pola musiman**, dan **puncak** (mis. Lebaran). Prediksi terlalu "datar" → model under-fit
# (kurang lag/fitur musiman) atau transform salah.
#
# **Untuk laporan:** 1 figure forecast vs actual (window terakhir) sangat meyakinkan juri bahwa validasi dilakukan dengan benar.

# %%
plot_uids = [u for u in PLOT_SERIES if u in set(EV_ALL["unique_id"])][:4] or EV_ALL["unique_id"].unique()[:1].tolist()
wins = sorted(EV_ALL["window"].unique())
fig, axes = plt.subplots(len(plot_uids), len(wins), figsize=(5.2 * len(wins), 2.8 * len(plot_uids)), squeeze=False)
for i, u in enumerate(plot_uids):
    hist_u = df[df["unique_id"] == u].set_index("ds")["y"]
    for j, w in enumerate(wins):
        ax = axes[i, j]
        c = CUTOFF_IDX[w - 1]
        start = GRID[max(0, c - 2 * H_TOTAL)]
        end = GRID[min(c + H_TOTAL, len(GRID) - 1)]
        hh = hist_u[(hist_u.index >= start) & (hist_u.index <= end)]
        ax.plot(hh.index, hh.values, color="black", lw=1, label="actual")
        for mname, sty in [(FINAL_NAME, dict(color="tab:blue", lw=1.6)), ("snaive", dict(color="gray", ls="--", lw=1))]:
            g = EV_ALL[(EV_ALL["model"] == mname) & (EV_ALL["unique_id"] == u) & (EV_ALL["window"] == w)].sort_values("ds")
            if len(g):
                ax.plot(g["ds"], g["yhat"], label=mname, **sty)
        ax.axvline(GRID[c], color="red", ls=":", lw=1)
        ax.set_title(f"{u} | window {w}", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        if j == 0:
            ax.set_ylabel(CFG.TARGET_COL)
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"Backtest: forecast vs actual ({FINAL_NAME})", y=1.01)
plt.tight_layout()
save_fig("backtest_forecast_vs_actual")

# %% [markdown]
# ### 11.1 Diagnostik residual
#
# - **Bias** = rata-rata (aktual − prediksi). Positif → model **under-forecast** (sering terjadi di data dengan trend naik); negatif → over-forecast.
# - **Histogram residual**: idealnya simetris di sekitar 0.
# - **Ljung-Box test** (H0: residual = white noise / tidak ada autokorelasi). **p < 0.05 → masih ada pola yang belum ditangkap**
#   (tambahkan lag/fitur musiman, atau order SARIMA lebih tinggi). p ≥ 0.05 → residual acak ✅.
#
# **Jika X maka Y:** bias besar dan konsisten di semua window → koreksi level (mis. `diff` transform) atau cek perubahan regime terakhir.

# %%
fin_ev = EV_ALL[EV_ALL["model"] == FINAL_NAME].copy()
fin_ev["resid"] = fin_ev["y"] - fin_ev["yhat"]
bias = fin_ev["resid"].mean()
rel_bias = bias / (fin_ev["y"].abs().mean() + 1e-12) * 100
print(f"Bias model final: {bias:.4g} ({rel_bias:+.2f}% dari rata-rata aktual) -> "
      + ("under-forecast" if rel_bias > 2 else ("over-forecast" if rel_bias < -2 else "✅ kecil")))
display(fin_ev.groupby("window")["resid"].agg(["mean", "std"]).rename(columns={"mean": "bias", "std": "resid_std"}).round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(fin_ev["resid"], bins=50, color="tab:blue", alpha=0.8)
axes[0].axvline(0, color="red")
axes[0].set_title(f"Residual distribution ({FINAL_NAME})")
axes[0].set_xlabel("Actual - Forecast")
axes[0].set_ylabel("Count")
rb = fin_ev.groupby("h")["resid"].mean()
axes[1].bar(rb.index, rb.values, color="tab:orange")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("Mean residual (bias) by horizon step")
axes[1].set_xlabel("Horizon step (h)")
axes[1].set_ylabel("Mean residual")
plt.tight_layout()
save_fig("residual_diagnostics")


def ljung_box_p(resid, lags):
    """p-value Ljung-Box (kompatibel statsmodels lama/baru)."""
    resid = pd.Series(resid).dropna().values
    lags = int(max(1, min(lags, len(resid) // 2 - 1)))
    if len(resid) < 8:
        return np.nan
    try:
        r = acorr_ljungbox(resid, lags=[lags], return_df=True)
        return float(r["lb_pvalue"].iloc[-1])
    except TypeError:
        r = acorr_ljungbox(resid, lags=[lags])
        return float(np.asarray(r[1])[-1])


if HAS_SM:
    rep = SERIES_RANK[0]
    y_rep = df.loc[df["unique_id"] == rep, "y"].values.astype(float)[-CFG.STAT_MAX_TRAIN:]
    try:
        seas = "add" if _seasonal_ok(y_rep, SEASON) else None
        hw = ExponentialSmoothing(y_rep, trend="add", damped_trend=True, seasonal=seas,
                                  seasonal_periods=SEASON if seas else None, initialization_method="estimated").fit()
        resid_in = y_rep - hw.fittedvalues
        lb_lag = max(2 * SEASON, 10)
        p = ljung_box_p(resid_in, lb_lag)
        print(f"Ljung-Box (in-sample Holt-Winters, series '{rep}', lag={min(lb_lag, len(resid_in) // 2 - 1)}): p = {p:.4f} -> "
              + ("⚠️ residual masih berautokorelasi (ada pola tersisa)" if p < 0.05 else "✅ residual ~ white noise"))
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
        axes[0].plot(resid_in, lw=0.7)
        axes[0].set_title(f"In-sample residuals Holt-Winters ({rep})")
        axes[0].set_xlabel("Time index")
        axes[0].set_ylabel("Residual")
        plot_acf(resid_in, lags=int(min(lb_lag, len(resid_in) // 2 - 1)), ax=axes[1])
        axes[1].set_title("ACF of residuals")
        axes[1].set_xlabel("Lag")
        plt.tight_layout()
        save_fig("residual_acf_holt_winters")
    except Exception as e:
        plt.close("all")
        print("Diagnostik Holt-Winters gagal:", e)
    lbp = []
    for (w, u), g in fin_ev.groupby(["window", "unique_id"]):
        if len(g) >= 10:
            lbp.append(ljung_box_p(g.sort_values("ds")["resid"].values, min(10, len(g) // 2 - 1)))
    if lbp:
        lbp = np.array(lbp, float)
        print(f"Ljung-Box residual backtest model final: {np.mean(lbp < 0.05):.1%} dari (window, series) punya p<0.05 "
              "(semakin kecil semakin baik)")
if SARIMA_ORDERS:
    print("Contoh order SARIMA terpilih (p,d,q,P,D,Q,m):", dict(list(SARIMA_ORDERS.items())[:5]))

# %% [markdown]
# ## 12. Final fit (seluruh data) + forecast masa depan
#
# **Tujuan:** model/ensemble terpilih dilatih ulang memakai **seluruh** data train (sampai tanggal terakhir), lalu memprediksi
# `GAP + H` langkah ke depan. Exog masa depan diambil dari file test (kalau ada), sisanya diisi nilai terakhir.
#
# **Jika X maka Y:** salah satu member ensemble gagal di final fit → otomatis dibuang dan bobot dinormalisasi ulang (pesan ⚠️ muncul).

# %%
test_exog_src = None
if HAS_TEST and EXOG:
    test_exog_src = test_long.rename(columns={"_ds": "ds"})[["unique_id", "ds"] + EXOG].drop_duplicates(["unique_id", "ds"])
FULL_FUT = make_future_frame(SERIES, FUTURE_DATES, test_exog_src, df)
ALPHA = 1 - float(CFG.PI_LEVEL)
QS = [round(ALPHA / 2, 4), round(1 - ALPHA / 2, 4)]
WANT_Q = str(CFG.PI_METHOD) in ("quantile_lgb", "both")
WANT_RES = str(CFG.PI_METHOD) in ("residual", "both")
q_member = next((m for m in FINAL_MEMBERS if m in ML_NAMES), None)

FINAL_PREDS, FINAL_INFO = {}, {}
for mname in FINAL_MEMBERS:
    t0 = time.time()
    try:
        yhat, info = forecast_with(mname, df, FULL_FUT, CTX, quantiles=QS if (WANT_Q and mname == q_member) else None)
        FINAL_PREDS[mname], FINAL_INFO[mname] = yhat, info
        print(f"  final fit {mname:<14s} OK ({time.time() - t0:.1f}s)")
    except Exception as e:
        FAILED_MODELS.setdefault(mname, []).append(f"final fit: {type(e).__name__}: {str(e)[:300]}")
        print(f"  ⚠️ final fit {mname} gagal: {e}")
if not FINAL_PREDS:
    print("⚠️ Semua member gagal di final fit -> fallback seasonal naive")
    FINAL_PREDS["snaive"], FINAL_INFO["snaive"] = forecast_with("snaive", df, FULL_FUT, CTX)
    FINAL_MEMBERS, FINAL_WEIGHTS = ["snaive"], np.array([1.0])
ok_members = [m for m in FINAL_MEMBERS if m in FINAL_PREDS]
P = np.column_stack([FINAL_PREDS[m] for m in ok_members])
if FINAL_WEIGHTS is None:
    FINAL_YHAT = np.median(P, axis=1)
else:
    wts = np.array([FINAL_WEIGHTS[FINAL_MEMBERS.index(m)] for m in ok_members], float)
    FINAL_YHAT = P @ (wts / wts.sum())
if CLIP_ZERO:
    FINAL_YHAT = np.clip(FINAL_YHAT, 0, None)

FC = FULL_FUT[["unique_id", "ds", "h"]].copy()
FC["yhat"] = FINAL_YHAT
for mname in ok_members:
    FC[f"yhat_{mname}"] = FINAL_PREDS[mname]
print(f"Forecast final: {len(FC)} baris ({N_SERIES} series x {H_TOTAL} langkah) | NaN: {int(FC['yhat'].isna().sum())}")

# Model ML untuk quantile interval kalau final tidak mengandung model ML
Q_INFO = FINAL_INFO.get(q_member, {}).get("quantiles") if q_member else None
if WANT_Q and Q_INFO is None and (HAS_LGB or True):
    try:
        _qm = "lgb" if HAS_LGB else "hgb"
        _, _qi = forecast_with(_qm, df, FULL_FUT, CTX, quantiles=QS)
        Q_INFO = _qi.get("quantiles")
        print(f"Quantile interval dihitung dengan model {_qm} terpisah.")
    except Exception as e:
        print("⚠️ Quantile model gagal:", e)

# %% [markdown]
# ## 13. Feature importance & SHAP (model ML)
#
# **Cara membaca:**
# - **Gain importance** (LightGBM): seberapa besar kontribusi fitur dalam mengurangi loss. Biasanya `lag_1`, `rmean_7`, `lag_7`, `dayofweek` di atas.
# - **SHAP summary**: tiap titik = 1 baris; posisi horizontal = dampak ke prediksi, warna = nilai fitur (merah tinggi, biru rendah).
#
# **Jika X maka Y:**
# - `time_idx` / `year` sangat dominan → model menghafal waktu; berisiko saat ekstrapolasi → pertimbangkan hapus fitur tsb (`USE_DATE_FEATURES` / edit `date_features`).
# - Fitur exog tidak penting → mungkin tidak berguna atau nilainya tidak informatif; boleh dibuang (lebih sederhana).
# - Fitur `cal_*` (libur/Lebaran) penting → sebut di laporan sebagai insight bisnis ("efek Lebaran signifikan").
#
# **Untuk laporan:** bar chart top-15 feature importance + 1 kalimat interpretasi per 3 fitur teratas.

# %%
IMP_MODEL, IMP_X = None, None
for mname in ok_members:
    if mname in ML_NAMES and FINAL_INFO.get(mname, {}).get("models"):
        IMP_MODEL, IMP_X = FINAL_INFO[mname]["models"][0][1], FINAL_INFO[mname]["X_train"]
        IMP_NAME = mname
        break
if IMP_MODEL is None:
    ml_avail = [m for m in ACTIVE_MODELS if m in ML_NAMES]
    if ml_avail:
        try:
            IMP_NAME = ml_avail[0]
            _h = df[["unique_id", "ds", "y"] + EXOG].sort_values(["unique_id", "ds"]).reset_index(drop=True)
            _h["yt"] = fwd_t(_h["y"].values)
            _stats = _h.groupby("unique_id")["yt"].agg(["mean", "std"])
            IMP_X, _y = ml_training_matrix(_h, 1, _stats)
            IMP_MODEL = make_ml_model(IMP_NAME).fit(IMP_X, _y)
            print(f"Model {IMP_NAME} dilatih khusus untuk analisis feature importance (bukan bagian forecast final).")
        except Exception as e:
            print("Gagal melatih model untuk importance:", e)
if IMP_MODEL is not None:
    try:
        if IMP_NAME == "lgb":
            imp = pd.Series(IMP_MODEL.booster_.feature_importance(importance_type="gain"), index=IMP_X.columns)
        elif hasattr(IMP_MODEL, "feature_importances_"):
            imp = pd.Series(IMP_MODEL.feature_importances_, index=IMP_X.columns)
        else:
            from sklearn.inspection import permutation_importance
            _s = IMP_X.sample(min(3000, len(IMP_X)), random_state=CFG.SEED)
            _pi = permutation_importance(IMP_MODEL, _s, IMP_MODEL.predict(_s), n_repeats=3, random_state=CFG.SEED)
            imp = pd.Series(_pi.importances_mean, index=IMP_X.columns)
        imp = (imp / (imp.sum() + 1e-12)).sort_values(ascending=False)
        save_table(imp.rename("importance").reset_index().rename(columns={"index": "feature"}), "feature_importance")
        top = imp.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
        ax.barh(top.index, top.values, color="tab:blue")
        ax.set_title(f"Feature importance ({IMP_NAME}, normalized)")
        ax.set_xlabel("Relative importance")
        ax.set_ylabel("Feature")
        save_fig("feature_importance")
        print("Top 5 fitur:", imp.head(5).round(3).to_dict())
    except Exception as e:
        plt.close("all")
        print("Feature importance gagal:", e)
    if CFG.USE_SHAP and HAS_SHAP and IMP_NAME in ("lgb", "xgb", "cat"):
        try:
            Xs = IMP_X.sample(min(CFG.SHAP_SAMPLE, len(IMP_X)), random_state=CFG.SEED)
            explainer = shap.TreeExplainer(IMP_MODEL)
            sv = explainer.shap_values(Xs)
            shap.summary_plot(sv, Xs, show=False, max_display=15)
            plt.title(f"SHAP summary ({IMP_NAME})")
            save_fig("shap_summary")
        except Exception as e:
            plt.close("all")
            print("SHAP gagal (tidak fatal):", e)
    elif CFG.USE_SHAP and not HAS_SHAP:
        print("shap tidak terinstall -> SHAP dilewati.")
else:
    print("Tidak ada model ML -> feature importance dilewati.")

# %% [markdown]
# ## 14. Prediction interval
#
# Dua metode:
# 1. **Residual-based (empiris):** distribusi error backtest model final per horizon step, dinormalisasi skala series (skala MASE),
#    lalu kuantil `α/2` dan `1−α/2` ditambahkan ke forecast. Tidak butuh asumsi normal; otomatis melebar untuk horizon jauh.
# 2. **Quantile LightGBM:** model terpisah dengan `objective="quantile"` (α/2 dan 1−α/2) memakai fitur yang sama.
#
# **Cara membaca:** `coverage` = persentase aktual yang jatuh di dalam interval pada backtest. Idealnya ≈ `PI_LEVEL` (mis. 90%).
# Coverage residual dihitung pada data yang sama dengan kalibrasi → sedikit optimis.
#
# **Jika X maka Y:** coverage jauh di bawah target → interval terlalu sempit (tambah window backtest). Interval sangat lebar →
# ketidakpastian tinggi; sebut di laporan sebagai risiko (mis. untuk perencanaan stok gunakan batas atas).

# %%
FC["lo_res"], FC["hi_res"] = np.nan, np.nan
if WANT_RES and len(fin_ev):
    sc_full = mase_scale(df, SEASON)
    fb_scale = df.groupby("unique_id")["y"].apply(lambda s: s.abs().mean() + 1e-9)
    fe = fin_ev.copy()
    fe["sc"] = fe["scale"].fillna(fe["unique_id"].map(fb_scale))
    fe["en"] = fe["resid"] / fe["sc"]
    pooled = fe["en"].quantile(QS).values
    qtab = {}
    for hh in range(1, H_TOTAL + 1):
        v = fe.loc[fe["h"] == hh, "en"]
        qtab[hh] = v.quantile(QS).values if len(v) >= 30 else pooled
    sc_u = FC["unique_id"].map(sc_full).fillna(FC["unique_id"].map(fb_scale)).values
    qlo = np.array([qtab[hh][0] for hh in FC["h"]])
    qhi = np.array([qtab[hh][1] for hh in FC["h"]])
    FC["lo_res"] = FC["yhat"] + qlo * sc_u
    FC["hi_res"] = FC["yhat"] + qhi * sc_u
    if CLIP_ZERO:
        FC["lo_res"] = FC["lo_res"].clip(lower=0)
    lo_b = fe["yhat"] + np.array([qtab[hh][0] for hh in fe["h"]]) * fe["sc"]
    hi_b = fe["yhat"] + np.array([qtab[hh][1] for hh in fe["h"]]) * fe["sc"]
    cov = float(((fe["y"] >= lo_b) & (fe["y"] <= hi_b)).mean())
    print(f"Residual interval {CFG.PI_LEVEL:.0%}: coverage backtest = {cov:.1%}" +
          ("  ✅" if abs(cov - CFG.PI_LEVEL) < 0.07 else "  ⚠️ jauh dari target"))
if Q_INFO is not None:
    FC["lo_q"], FC["hi_q"] = Q_INFO[QS[0]], Q_INFO[QS[1]]
    FC["lo_q"], FC["hi_q"] = np.minimum(FC["lo_q"], FC["hi_q"]), np.maximum(FC["lo_q"], FC["hi_q"])
    if CLIP_ZERO:
        FC["lo_q"] = FC["lo_q"].clip(lower=0)
    print(f"Quantile-LGB interval tersedia (rata-rata lebar = {(FC['hi_q'] - FC['lo_q']).mean():.4g})")
FC.to_csv(TAB_DIR / "final_forecast_all_steps.csv", index=False)

fu = [u for u in PLOT_SERIES if u in set(FC["unique_id"])][:4]
fig, axes = plt.subplots(len(fu), 1, figsize=(13, 3.2 * len(fu)), squeeze=False)
for ax, u in zip(axes[:, 0], fu):
    hist_u = df[df["unique_id"] == u].tail(max(3 * H_TOTAL, 3 * SEASON))
    f = FC[FC["unique_id"] == u]
    ax.plot(hist_u["ds"], hist_u["y"], color="black", lw=1, label="history")
    ax.plot(f["ds"], f["yhat"], color="tab:blue", lw=1.8, label=f"forecast ({FINAL_NAME})")
    if f["lo_res"].notna().any():
        ax.fill_between(f["ds"], f["lo_res"], f["hi_res"], color="tab:blue", alpha=0.2, label=f"{CFG.PI_LEVEL:.0%} PI (residual)")
    if "lo_q" in f and f["lo_q"].notna().any():
        ax.plot(f["ds"], f["lo_q"], color="tab:orange", ls="--", lw=0.9, label="quantile LGB")
        ax.plot(f["ds"], f["hi_q"], color="tab:orange", ls="--", lw=0.9)
    ax.set_title(f"Final forecast: {u}")
    ax.set_xlabel("Date")
    ax.set_ylabel(CFG.TARGET_COL)
axes[0, 0].legend(fontsize=8)
plt.tight_layout()
save_fig("final_forecast")

if HAS_JOBLIB:
    for mname, info in FINAL_INFO.items():
        if info.get("models"):
            try:
                joblib.dump(info["models"], MODEL_DIR / f"final_{mname}.joblib")
            except Exception as e:
                print("Gagal simpan model:", e)

# %% [markdown]
# ## 15. Submission builder + validator
#
# **Logika:**
# - **Ada test:** prediksi di-merge ke baris test berdasarkan (series id, tanggal). Urutan & kolom mengikuti `sample_submission`
#   (kalau ada) berdasarkan `ID_COL`. Baris test yang tidak ketemu (cold start / tanggal di luar grid) diisi rata-rata prediksi
#   series lain di tanggal yang sama, lalu nilai global.
# - **Tidak ada test:** disimpan file forecast H langkah ke depan (kolom series id + tanggal + prediksi + interval).
#
# **Validator mengecek:** jumlah baris = sample/test, nama & urutan kolom = sample, urutan ID sama, tidak ada NaN/inf, tidak negatif
# (kalau target tidak pernah negatif). **Jangan submit kalau ada ❌.**

# %%
PRED_COL = CFG.SUBMISSION_PRED_COL
if PRED_COL is None and sample_sub is not None:
    non_id = [c for c in sample_sub.columns if c != CFG.ID_COL]
    PRED_COL = non_id[0] if non_id else CFG.TARGET_COL
PRED_COL = PRED_COL or CFG.TARGET_COL


def finalize_values(v):
    v = np.asarray(v, float)
    if CLIP_ZERO:
        v = np.clip(v, 0, None)
    if CFG.ROUND_PRED:
        v = np.round(v)
    return v


SUBMISSION = None
if HAS_TEST:
    t = test_long.copy()
    t["ds"] = t["_ds"]
    m = t.merge(FC[["unique_id", "ds", "yhat"]], on=["unique_id", "ds"], how="left")
    n_miss = int(m["yhat"].isna().sum())
    if n_miss:
        # 1) tanggal <= akhir train -> pakai nilai aktual
        act_map = df.set_index(["unique_id", "ds"])["y"]
        idx = m["yhat"].isna()
        keys = list(zip(m.loc[idx, "unique_id"], m.loc[idx, "ds"]))
        m.loc[idx, "yhat"] = [act_map.get(k, np.nan) for k in keys]
        # 2) cold start -> rata-rata prediksi semua series di tanggal sama
        by_date = FC.groupby("ds")["yhat"].mean()
        idx = m["yhat"].isna()
        m.loc[idx, "yhat"] = m.loc[idx, "ds"].map(by_date)
        # 3) sisa -> rata-rata global forecast
        m["yhat"] = m["yhat"].fillna(FC["yhat"].mean())
        print(f"⚠️ {n_miss} baris test tidak punya forecast langsung -> diisi fallback (aktual/rata-rata tanggal/global).")
    m["yhat"] = finalize_values(m["yhat"])
    if sample_sub is not None and CFG.ID_COL and CFG.ID_COL in sample_sub.columns and CFG.ID_COL in m.columns:
        pred_map = pd.Series(m["yhat"].values, index=norm_key(m[CFG.ID_COL]).values)
        pred_map = pred_map[~pred_map.index.duplicated()]
        SUBMISSION = sample_sub.copy()
        SUBMISSION[PRED_COL] = norm_key(SUBMISSION[CFG.ID_COL]).map(pred_map).values
        n_unmatched = int(SUBMISSION[PRED_COL].isna().sum())
        if n_unmatched:
            print(f"⚠️ {n_unmatched} ID sample_submission tidak ditemukan di test -> diisi rata-rata")
            SUBMISSION[PRED_COL] = SUBMISSION[PRED_COL].fillna(float(np.nanmean(m["yhat"])))
    elif CFG.ID_COL and CFG.ID_COL in m.columns:
        SUBMISSION = pd.DataFrame({CFG.ID_COL: m[CFG.ID_COL].values, PRED_COL: m["yhat"].values})
    else:
        keep = [c for c in CFG.SERIES_ID_COLS if c in m.columns] + [DATE_COL]
        SUBMISSION = m[keep].copy()
        SUBMISSION[PRED_COL] = m["yhat"].values
        print("Info: ID_COL tidak ada -> submission memakai kolom series + tanggal.")
    sub_path = SUB_DIR / CFG.SUBMISSION_FILE
else:
    out = FC[FC["h"] > GAP].copy()
    if UID_MAP is not None:
        out = out.merge(UID_MAP.reset_index(), on="unique_id", how="left")
    cols = (list(CFG.SERIES_ID_COLS) if CFG.SERIES_ID_COLS else []) + ["ds", "yhat"] + \
        [c for c in ["lo_res", "hi_res", "lo_q", "hi_q"] if c in out.columns and out[c].notna().any()]
    SUBMISSION = out[cols].rename(columns={"ds": DATE_COL, "yhat": PRED_COL})
    SUBMISSION[PRED_COL] = finalize_values(SUBMISSION[PRED_COL])
    sub_path = SUB_DIR / f"forecast_next_{H}_steps.csv"


def validate_submission(sub, sample=None, test=None):
    """Cek format submission. Return True kalau semua lolos."""
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("✅ " if cond else "❌ ") + msg)
        ok = ok and bool(cond)

    vals = pd.to_numeric(sub[PRED_COL], errors="coerce")
    chk(vals.notna().all() and np.isfinite(vals).all(), f"kolom '{PRED_COL}' tanpa NaN/inf")
    if CLIP_ZERO:
        chk((vals >= 0).all(), "tidak ada prediksi negatif")
    if sample is not None:
        chk(len(sub) == len(sample), f"jumlah baris sama dengan sample ({len(sub)} vs {len(sample)})")
        chk(list(sub.columns) == list(sample.columns), f"kolom sama & urutan sama: {list(sub.columns)}")
        if CFG.ID_COL in sub.columns and CFG.ID_COL in sample.columns:
            chk((norm_key(sub[CFG.ID_COL]).values == norm_key(sample[CFG.ID_COL]).values).all(), "urutan ID sama dengan sample")
            chk(not sub[CFG.ID_COL].duplicated().any(), "ID unik")
    elif test is not None:
        chk(len(sub) == len(test), f"jumlah baris sama dengan test ({len(sub)} vs {len(test)})")
    print(f"Statistik prediksi: min={vals.min():.4g} mean={vals.mean():.4g} max={vals.max():.4g} | "
          f"historis: mean={df['y'].mean():.4g} max={df['y'].max():.4g}")
    if vals.mean() > 3 * df["y"].mean() + 1e-9 or vals.mean() < 0.2 * df["y"].mean():
        print("⚠️ Rata-rata prediksi sangat berbeda dari historis -> cek transform/agregasi/frekuensi")
    return ok


print("=" * 70)
SUB_OK = validate_submission(SUBMISSION, sample_sub if HAS_TEST else None, test_raw if HAS_TEST else None)
SUBMISSION.to_csv(sub_path, index=False)
print("=" * 70)
print(f"{'✅' if SUB_OK else '⚠️'} Disimpan: {sub_path.resolve()}")
display(SUBMISSION.head(10))

# %% [markdown]
# ## 16. Ringkasan untuk laporan
#
# Cell ini mencetak paragraf ringkas yang bisa diadaptasi untuk laporan (ubah bahasa/gaya sesuai kebutuhan) + daftar file figure/tabel.
# **Struktur laporan forecasting (max 10 halaman) yang disarankan:**
# 1. Pendahuluan & tujuan bisnis (kenapa forecast ini penting).
# 2. Data & preprocessing (tabel ringkasan data, imputasi, outlier, agregasi).
# 3. EDA (line plot, STL, seasonal boxplot, efek libur/Lebaran, uji stasioneritas).
# 4. Metodologi (skema backtest, metrik, model yang dicoba, fitur ML, ensemble).
# 5. Hasil (leaderboard, forecast vs actual, error per horizon, feature importance).
# 6. Insight & rekomendasi bisnis (mis. persiapan stok sebelum Lebaran, interval untuk safety stock).
# 7. Keterbatasan & pengembangan.

# %%
lb_print = LB_ALL[["model", METRIC] + (["vs_snaive_%"] if "vs_snaive_%" in LB_ALL else [])].head(8)
base_sn = float(LB_ALL.loc[LB_ALL["model"] == "snaive", METRIC].iloc[0]) if "snaive" in LB_ALL["model"].values else np.nan
impr = (base_sn - FINAL_SCORE) / (abs(base_sn) + 1e-12) * 100 if np.isfinite(base_sn) else np.nan
print("RINGKASAN")
print("-" * 70)
print(f"Data: {N_SERIES} series, frekuensi {KIND}, {df['ds'].min().date()} s/d {df['ds'].max().date()}, {len(df)} observasi; "
      f"{df['is_imputed'].mean():.1%} titik diimputasi ('{_fill}').")
if np.isfinite(F_SEAS):
    print(f"EDA: seasonal strength {F_SEAS:.2f}, trend strength {F_TREND:.2f}; stasioneritas level: {stat_df.loc[0, 'conclusion']}.")
print(f"Validasi: {CFG.WINDOW_TYPE}-window backtesting, {len(CUTOFF_IDX)} cutoff, horizon {H} langkah (gap {GAP}); metrik {METRIC.upper()}.")
print(f"Model final: {FINAL_NAME} (members: {', '.join(ok_members)}), {METRIC.upper()} backtest = {FINAL_SCORE:.4f}"
      + (f", lebih baik {impr:.1f}% dibanding seasonal naive." if np.isfinite(impr) else "."))
display(lb_print.round(4))
print("\nFile figure:")
for p in sorted(FIG_DIR.glob("*.png")):
    print("  ", p.name)
print("File tabel:")
for p in sorted(TAB_DIR.glob("*.csv")):
    print("  ", p.name)

# %% [markdown]
# ## 17. Debugging & sanity checks
#
# Jalankan cell ini setiap kali ada hasil aneh. Ia mengecek: ukuran data, NaN/inf, duplikat, model yang gagal, kolom train vs test,
# heuristik **leakage** (exog yang hampir identik dengan target → mungkin turunan target / informasi masa depan), dan konsistensi forecast.

# %%
print("Shapes: train_raw", train_raw.shape, "| df (long)", df.shape, "| FC", FC.shape, "| submission", SUBMISSION.shape)
print("NaN di df.y:", int(df["y"].isna().sum()), "| inf:", int(np.isinf(df["y"]).sum()))
print("Duplikat (unique_id, ds) di df:", int(df.duplicated(["unique_id", "ds"]).sum()))
print("NaN di forecast:", int(FC["yhat"].isna().sum()))
if HAS_TEST:
    tr_cols, te_cols = set(train_raw.columns) - {"_ds", "unique_id"}, set(test_raw.columns) - {"_ds", "unique_id"}
    print("Kolom hanya di train:", sorted(tr_cols - te_cols))
    print("Kolom hanya di test :", sorted(te_cols - tr_cols))
for c in EXOG:
    try:
        corr = np.corrcoef(df[c].astype(float), df["y"])[0, 1]
        if abs(corr) > 0.98:
            print(f"⚠️ LEAKAGE? exog '{c}' berkorelasi {corr:.3f} dengan target. Pastikan nilainya benar-benar diketahui di masa depan.")
    except Exception:
        pass
if FAILED_MODELS:
    print("\nFAILED_MODELS:")
    for k, v in FAILED_MODELS.items():
        print(f"  {k}: {v[:2]}")
else:
    print("FAILED_MODELS: (kosong) ✅")
_last = df.groupby("unique_id")["y"].apply(lambda s: s.tail(max(SEASON, 7)).mean())
_first_fc = FC[FC["h"] <= max(SEASON, 7)].groupby("unique_id")["yhat"].mean()
_ratio = (_first_fc / (_last + 1e-9)).replace([np.inf, -np.inf], np.nan).dropna()
if len(_ratio):
    print(f"Rasio forecast awal / level terakhir (median antar series): {_ratio.median():.2f}"
          + ("  ⚠️ loncatan level besar di awal forecast" if abs(_ratio.median() - 1) > 0.5 else "  ✅"))

# %% [markdown]
# ## 18. Troubleshooting
#
# | Error / gejala | Penyebab | Solusi |
# |---|---|---|
# | `AssertionError: TARGET_COL ... tidak ada di train` | nama kolom salah / spasi / huruf besar | cek `print(train_raw.columns)`, set `CFG.TARGET_COL` persis |
# | `ValueError: Kolom tanggal tidak terdeteksi` | format tanggal tidak umum | set `CFG.DATE_COL` dan `CFG.DATE_FORMAT` (mis. `"%d/%m/%Y"`) |
# | Tanggal bulan & hari tertukar | format Indonesia dd/mm/yyyy | `CFG.DAYFIRST=True` atau `DATE_FORMAT` |
# | `Frekuensi terdeteksi: ... fallback` tapi salah | data tidak teratur | set `CFG.FREQ` manual ("D", "W-MON", "MS", "h") |
# | `⚠️ banyak tanggal tidak sejajar` | anchor mingguan berbeda (Senin vs Minggu) / data campur | `CFG.FREQ="W-MON"` sesuai hari data, atau `RESAMPLE_RULE` |
# | `ValueError: Data terlalu pendek untuk backtest` | horizon terlalu panjang vs data | kecilkan `CFG.HORIZON` atau `N_BACKTEST_WINDOWS` |
# | `ModuleNotFoundError: statsmodels` | belum terinstall | `pip install statsmodels`; notebook tetap jalan (model statistik di-skip) |
# | `ConvergenceWarning` / SARIMAX lambat | data panjang / order besar | kecilkan `SARIMA_MAX_P/Q`, `SARIMA_MAX_TRAIN`, `SARIMA_TIME_LIMIT` |
# | `[sarimax] N series fallback ke snaive` | fit gagal (data konstan/pendek) | normal untuk series pendek; abaikan atau hapus `sarimax` dari `MODELS` |
# | `LinAlgError: SVD did not converge` | data konstan / NaN / inf | cek series konstan, isi NaN (`FILL_METHOD`) |
# | Holt-Winters `ValueError: ... seasonal_periods` | data < 2 season | otomatis non-seasonal; kalau tetap error hapus `holt_winters` |
# | LightGBM `[Warning] No further splits with positive gain` | data latih kecil | aman; atau kecilkan `ML_NUM_LEAVES` |
# | `data latih ML terlalu sedikit` | series pendek, lag terlalu panjang | kecilkan `CFG.LAGS` / `ROLL_WINDOWS` |
# | XGBoost `feature_names mismatch` | kolom fitur train≠predict (exog beda) | pastikan `EXOG_COLS` ada di train & test dengan nama sama |
# | CatBoost `CatBoostError: ... catboost_info` | permission folder | sudah `allow_writing_files=False`; jalankan dari folder yang bisa ditulis |
# | Forecast datar / terlalu halus | lag/fitur musiman kurang, transform salah | cek `SEASON_LENGTH`, tambah `LAGS` musiman, coba `snaive`/ETS dalam ensemble |
# | Forecast melenceng jauh di akhir (trend) | model tree tidak bisa ekstrapolasi | `TARGET_TRANSFORM="diff"`/`"log1p_diff"`, tambah ETS/drift ke ensemble |
# | MAPE sangat besar / inf | ada nilai aktual ≈ 0 | pakai WAPE/MAE/sMAPE; jangan optimasi MAPE kalau banyak nol |
# | Prediksi negatif | model linear/diff | `CLIP_NEGATIVE_PRED=True` (default auto sudah clip kalau target >= 0) |
# | `❌ urutan ID sama dengan sample` | ID tipe beda (int vs str) | sudah dinormalisasi `norm_key`; cek ID duplikat di test |
# | `⚠️ N baris test tidak punya forecast langsung` | series baru (cold start) / tanggal test di luar grid | cek `SERIES_ID_COLS` sama di test; cek frekuensi test |
# | Kernel mati / MemoryError | panel sangat besar | `RUN_MODE="fast"`, kurangi `ML_MAX_TRAIN_ROWS`, `STAT_MAX_SERIES`, `LAGS` |
# | Prophet error `cmdstan` | instalasi prophet bermasalah | hapus `prophet` dari `MODELS` (opsional saja) |
# | SHAP error / lambat | versi shap vs model | `CFG.USE_SHAP=False`; feature importance tetap ada |
# | Skor backtest bagus, leaderboard lomba jelek | gap/horizon validasi beda dengan test, atau regime berubah | samakan `GAP`/`HORIZON`, pakai lebih banyak window, `WINDOW_TYPE="rolling"` |

# %% [markdown]
# ## 19. Experiment recipes (salin ke cell USER OVERRIDE)
#
# **A. Penjualan harian satu toko, metrik RMSE:**
# ```python
# CFG.TARGET_COL = "sales"; CFG.DATE_COL = "date"; CFG.METRIC = "rmse"; CFG.FILL_METHOD = "zero"
# ```
# **B. Panel toko × item, metrik RMSLE, banyak nol:**
# ```python
# CFG.SERIES_ID_COLS = ["store", "item"]; CFG.METRIC = "rmsle"; CFG.TARGET_TRANSFORM = "log1p"
# CFG.STAT_MAX_SERIES = 20; CFG.MODELS = ["naive", "snaive", "ma", "ets", "lgb", "xgb", "cat"]
# ```
# **C. Data bulanan pendek (< 60 titik), mis. penjualan bulanan / inflasi:**
# ```python
# CFG.MODELS = ["naive", "snaive", "drift", "ses", "holt_winters", "ets", "theta", "sarimax", "lgb"]
# CFG.LAGS = [1, 2, 3, 12]; CFG.ROLL_WINDOWS = [3, 6]; CFG.N_BACKTEST_WINDOWS = 4; CFG.HORIZON = 12
# ```
# **D. Trend kuat (model tree under-predict):**
# ```python
# CFG.TARGET_TRANSFORM = "log1p_diff"   # atau "diff" kalau ada nilai negatif
# ```
# **E. Horizon panjang (mis. 90 hari), error recursive menumpuk:**
# ```python
# CFG.ML_STRATEGY = "direct"; CFG.DIRECT_N_BUCKETS = 6
# ```
# **F. Data per jam (listrik/traffic):**
# ```python
# CFG.FREQ = "h"; CFG.SEASON_LENGTH = 24; CFG.LAGS = [1, 2, 3, 24, 48, 168]; CFG.ROLL_WINDOWS = [24, 168]
# CFG.SARIMA_MAX_SEASON = 24; CFG.STAT_MAX_TRAIN = 24 * 60
# ```
# **G. Data harian tapi lomba minta mingguan:** `CFG.RESAMPLE_RULE = "W-MON"; CFG.RESAMPLE_AGG = "sum"`
# **H. Ada promo & harga di test:** `CFG.EXOG_COLS = ["promo", "price"]` (atau `"auto"`)
# **I. Final run untuk submission:** `CFG.RUN_MODE = "full"; CFG.N_BACKTEST_WINDOWS = 4; CFG.FINAL_STRATEGY = "auto"; CFG.FIG_DPI = 300`
# **J. Metrik MAE (median optimal):** tambahkan `objective="l1"` di `make_ml_model` untuk `lgb` (cari baris `p = dict(...)`).

# %% [markdown]
# ## 20. Checklist sebelum submit
#
# - [ ] `CFG.METRIC` = metrik resmi lomba; `HORIZON`/`GAP` sesuai test (lihat output bagian 4.1).
# - [ ] Frekuensi & season length yang terdeteksi masuk akal.
# - [ ] Persentase imputasi wajar & metode imputasi sesuai arti data (nol vs interpolasi).
# - [ ] Model final mengalahkan **seasonal naive** di backtest (`vs_snaive_%` > 0).
# - [ ] Variasi skor antar window tidak terlalu besar (< 30%).
# - [ ] Plot forecast vs actual & final forecast terlihat masuk akal (level, musiman, tidak loncat).
# - [ ] `RUN_MODE = "full"` untuk run final (bukan fast).
# - [ ] Validator submission semua ✅ (baris, kolom, urutan ID, tanpa NaN, tidak negatif).
# - [ ] Nama file submission sesuai aturan lomba; simpan juga salinan dengan nama versi (mis. `sub_v3_ens.csv`).
# - [ ] Figure untuk laporan disimpan dengan `FIG_DPI=300`; tabel leaderboard & ringkasan data diambil dari folder `tables/`.
# - [ ] Tanggal libur/Lebaran sudah diverifikasi bila efeknya dibahas di laporan.
