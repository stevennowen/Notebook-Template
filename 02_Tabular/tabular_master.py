# -*- coding: utf-8 -*-
# =====================================================================================================
# tabular_master.py — versi SCRIPT dari tabular_master.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python tabular_master.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb tabular_master.py
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
# # Tabular Master Notebook — Binary / Multiclass / Regression
#
# Notebook universal untuk lomba data tabular: **lempar dataset tabular apa saja**, set beberapa variabel di cell
# `# >>> USER OVERRIDE`, lalu jalankan dari atas sampai bawah. Notebook ini mengurus:
#
# - load data yang "berantakan" (separator aneh, encoding aneh, angka tersimpan sebagai teks, tanggal sebagai teks),
# - laporan kualitas data otomatis + EDA visual yang siap dipakai di laporan,
# - deteksi **leakage** dan **drift** train vs test,
# - preprocessing yang **fold-safe** (tidak bocor ke validation),
# - feature engineering yang bisa di-toggle,
# - banyak model (baseline, linear, RandomForest, ExtraTrees, HistGB, LightGBM, XGBoost, CatBoost, KNN/SVM/MLP),
# - hyperparameter tuning dengan Optuna, ensembling (mean, rank, weighted, hill-climbing, stacking),
# - post-processing (threshold, class prior, calibration, clipping), evaluasi mendalam, interpretasi (SHAP, PDP),
# - pembuatan submission yang **divalidasi** terhadap `sample_submission`, plus log eksperimen.
#
# Semua langkah dievaluasi dengan **Out-Of-Fold (OOF) cross-validation**, jadi skor yang kamu lihat adalah estimasi jujur
# untuk skor di leaderboard (selama data test berasal dari distribusi yang sama).
#
# ---
#
# ## Daftar Isi
#
# | No | Section | Isi singkat | Wajib? |
# |---|---|---|---|
# | 0 | Pendahuluan | Daftar isi, cara pakai cepat, peta keputusan | baca sekali |
# | 1 | Install & Import | `ensure_packages`, availability flags, versi library, deteksi GPU | ✅ |
# | 2 | Konfigurasi | `class CFG` + cell `# >>> USER OVERRIDE` + finalisasi config | ✅ |
# | 3 | Load Data | `smart_read`, sanitasi nama kolom, merge multi-file, agregasi tabel anak | ✅ |
# | 4 | Data Quality Report | dtype, missing, konstan, duplikat, ID-like, angka-sebagai-teks, tanggal, outlier, mismatch | ✅ |
# | 5 | Analisis Target | deteksi task, encoding label, class balance, skew target | ✅ |
# | 6 | EDA Visual | distribusi, target vs fitur, korelasi, Cramér's V, MI, leakage detector, drift & adversarial validation | opsional (penting untuk laporan) |
# | 7 | Preprocessing fold-safe | imputasi, rare category, encoder, target encoding, scaling, clipping | ✅ |
# | 8 | Feature Engineering | tanggal, frequency, group agg, rasio, interaksi, binning, teks, row stats, custom hook | ✅ (toggle) |
# | 9 | Strategi CV | Stratified / KFold-binned / Group / StratifiedGroup / TimeSeries | ✅ |
# | 10 | Definisi Model | default params kuat per task, imbalance handling, GPU fallback | ✅ |
# | 11 | Training Loop | OOF + test preds, leaderboard | ✅ |
# | 12 | Tuning Optuna | LightGBM/XGBoost/CatBoost, pruning, time budget | opsional |
# | 13 | Ensembling | mean, rank, weighted, hill-climbing, stacking | sangat disarankan |
# | 14 | Post-processing | threshold, class prior, calibration, clipping/rounding | sesuai metric |
# | 15 | Evaluasi & Debugging | confusion matrix, ROC/PR, residual, learning curve, shuffled-target test, error analysis | penting untuk laporan |
# | 16 | Interpretasi | importance, permutation, SHAP, PDP/ICE, null importance, seleksi fitur | penting untuk laporan |
# | 17 | Final Fit & Submission | strategi final, submission + validator, log eksperimen | ✅ |
# | 18 | Recipes, Checklist, Troubleshooting | resep eksperimen, checklist, tabel error | baca saat butuh |
#
# ---
#
# ## Cara Pakai Cepat (5 langkah)
#
# 1. Taruh file di `./data/` (`train.csv`, `test.csv`, `sample_submission.csv`) **atau** isi path-nya di cell `# >>> USER OVERRIDE`.
# 2. Di cell `# >>> USER OVERRIDE` isi minimal: `CFG.TARGET_COL`, `CFG.ID_COL`, `CFG.METRIC` (sesuai aturan lomba!), dan `CFG.RUN_MODE = "fast"`.
# 3. **Run All**. Mode `fast` memakai subsample + fold sedikit + model sedikit → untuk memastikan semua jalan (biasanya 1–5 menit).
# 4. Baca output section 4–6 (kualitas data, leakage, drift). Perbaiki config bila perlu (DROP_COLS, DATE_COLS, GROUP_COL, dll).
# 5. Ganti `CFG.RUN_MODE = "full"` → Run All lagi. Ambil `outputs/.../submissions/submission.csv`, dan figur/tabel untuk laporan di `figures/` & `tables/`.
#
# > Tips: kalau target / ID tidak kamu isi, notebook mencoba menebak (kolom yang ada di train tapi tidak ada di test = target;
# > kolom pertama `sample_submission` = ID). Selalu cek print-nya!
#
# ## Peta Keputusan — Section mana yang boleh di-skip kalau waktu mepet?
#
# | Sisa waktu | Yang dijalankan | Setting yang disarankan |
# |---|---|---|
# | **< 15 menit** | 1 → 5, 7 → 11, 17 | `RUN_MODE="fast"`, `DO_EDA=False`, `MODELS=["lgb"]` atau `["hgb"]`, `USE_OPTUNA=False`, `DO_INTERPRET=False` |
# | **15–45 menit** | semua kecuali 12 dan bagian berat 16 | `RUN_MODE="full"`, `USE_OPTUNA=False`, `MODELS=["linear","lgb","xgb","cat"]`, `DO_NULL_IMP=False` |
# | **1–3 jam** | semua | `RUN_MODE="full"`, `USE_OPTUNA=True`, `OPTUNA_TIMEOUT=600` per model, ensemble aktif |
# | **> 3 jam** | semua + eksperimen FE | Tambah `N_REPEATS=2–3`, coba `FE_GROUP_AGG=True`, `TE_FOR_TREES=True`, tuning lebih lama, isi `custom_feature_engineering` |
#
# **Prinsip utama lomba tabular (hafalkan):**
#
# - **Percaya CV, bukan public leaderboard.** Public LB biasanya hanya sebagian kecil data test → noisy.
# - **Validasi harus meniru cara test dibuat.** Jika test = data masa depan → TimeSeries CV. Jika test = customer/pasien baru → Group CV.
# - **Fitur > model > tuning.** Feature engineering yang bagus biasanya lebih berpengaruh daripada tuning.
# - **Ensemble model yang berbeda** (GBDT + linear + CatBoost) hampir selalu menaikkan skor sedikit dan menurunkan variance.
# - **Jika skor terlalu bagus untuk jadi kenyataan → curigai leakage** (lihat section 6.8 dan 15.6).

# %% [markdown]
# ---
# # 1. Install & Import Library
#
# **Tujuan:** memastikan semua library tersedia, mengimpor library opsional secara aman (tidak crash kalau tidak ada),
# mencetak versi (penting untuk debugging di laptop panitia), dan mendeteksi GPU.
#
# **Apa yang terjadi di sini:**
#
# 1. `ensure_packages()` mengecek library dengan `importlib.util.find_spec`. Jika belum ada **dan** `AUTO_INSTALL = True`,
#    notebook mencoba `pip install` (butuh internet). Jika gagal, notebook **tetap jalan** — model yang butuh library itu
#    otomatis di-skip.
# 2. Setiap library opsional diimpor dalam `try/except` dan menghasilkan flag, misal `HAS_LGB`, `HAS_XGB`, `HAS_CAT`,
#    `HAS_OPTUNA`, `HAS_SHAP`, `HAS_IMBLEARN`.
# 3. `torch` **tidak pernah** di-install otomatis (ukurannya besar dan versi CUDA harus cocok). Torch hanya dipakai untuk
#    deteksi GPU dan seeding, jadi tidak wajib.
#
# **Cara membaca output:**
#
# - Tabel versi: kolom `available` harus `True` untuk library inti (numpy, pandas, sklearn, scipy, matplotlib).
# - `GPU detected: True/False` → hanya informasi. Penggunaan GPU diatur di `CFG.USE_GPU` (default `False` = CPU, paling aman).
#
# **Jika X maka Y:**
#
# - Jika `lightgbm` / `xgboost` / `catboost` = `False` dan install gagal → jalankan manual di terminal:
#   `pip install lightgbm xgboost catboost`, lalu **Restart Kernel**. Kalau tetap gagal, notebook masih bisa jalan dengan
#   `HistGradientBoosting` (bawaan sklearn) yang performanya mirip LightGBM.
# - Jika muncul error import aneh setelah install → **Restart Kernel** (library baru tidak terbaca sebelum restart).
# - Jika laptop tanpa internet → set `AUTO_INSTALL = False` agar tidak buang waktu menunggu timeout pip.
# - Jika versi sklearn < 1.2 → masih didukung (ada helper kompatibilitas), tapi beberapa fitur (TargetEncoder bawaan
#   sklearn) otomatis diganti versi manual.
#
# **Untuk laporan:** cukup sebutkan library utama dan versinya (tabel versi tersimpan di `tables/library_versions.csv`).

# %%
# ===== Bootstrap config (hanya untuk cell install). Nilai final ada di class CFG (section 2). =====
class CFG:
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"   # False jika laptop tanpa internet / tidak boleh install (env TEMPLATE_AUTO_INSTALL=0 → off)


import importlib
import importlib.util
import subprocess
import sys


def ensure_packages(packages, auto_install=True):
    """Cek package; install yang belum ada (jika auto_install=True). Tidak pernah membuat notebook gagal.

    packages: dict {import_name: pip_name}
    """
    missing = [imp for imp in packages if importlib.util.find_spec(imp) is None]
    if not missing:
        print("✅ Semua package sudah tersedia.")
        return []
    print("Package belum terinstall:", missing)
    if not auto_install:
        print("AUTO_INSTALL=False → skip install. Install manual: pip install " + " ".join(packages[m] for m in missing))
        return missing
    still_missing = []
    for imp in missing:
        pip_name = packages[imp]
        try:
            print(f"  pip install {pip_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *([] if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else ["--user"]), pip_name])
            importlib.invalidate_caches()
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ Gagal install {pip_name}: {e}")
            still_missing.append(imp)
    if still_missing:
        print("⚠️ Masih belum ada:", still_missing, "→ fitur terkait akan di-skip.")
    else:
        print("✅ Install selesai. Jika ada error import, Restart Kernel lalu jalankan ulang.")
    return still_missing


REQUIRED_PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy",
    "matplotlib": "matplotlib", "seaborn": "seaborn",
    "lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost",
    "optuna": "optuna", "shap": "shap", "imblearn": "imbalanced-learn",
    "category_encoders": "category_encoders", "statsmodels": "statsmodels",
    "openpyxl": "openpyxl", "pyarrow": "pyarrow",
}
_missing_pkgs = ensure_packages(REQUIRED_PACKAGES, auto_install=CFG.AUTO_INSTALL)
if importlib.util.find_spec("torch") is None:
    print("ℹ️ torch tidak terinstall (opsional). Tidak di-install otomatis. Jika butuh: lihat https://pytorch.org/get-started/locally/")

# %%
# ===== Import inti =====
import os
import gc
import re
import json
import math
import time
import copy
import random
import shutil
import warnings
import traceback
import platform
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import scipy
from scipy import stats
from scipy.optimize import minimize

import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

try:
    from IPython.display import display
except Exception:  # noqa: BLE001
    display = print

try:
    import seaborn as sns
    HAS_SNS = True
except Exception as e:  # noqa: BLE001
    sns, HAS_SNS = None, False
    print("seaborn tidak tersedia:", e)

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
                                     cross_val_predict)
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:  # noqa: BLE001
    StratifiedGroupKFold, HAS_SGKF = None, False
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss, accuracy_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
                             mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve, brier_score_loss)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
                              ExtraTreesRegressor)
try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    HAS_HGB = True
except Exception:  # noqa: BLE001  (sklearn < 1.0 butuh experimental import)
    try:
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        HAS_HGB = True
    except Exception:  # noqa: BLE001
        HAS_HGB = False
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_sample_weight
try:
    from sklearn.preprocessing import TargetEncoder as SKTargetEncoder
    HAS_SK_TE = True
except Exception:  # noqa: BLE001
    SKTargetEncoder, HAS_SK_TE = None, False

# ===== Import opsional (masing-masing punya flag) =====
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception as e:  # noqa: BLE001
    lgb, HAS_LGB = None, False
    print("LightGBM tidak tersedia:", e)
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:  # noqa: BLE001
    xgb, HAS_XGB = None, False
    print("XGBoost tidak tersedia:", e)
try:
    import catboost as cb
    HAS_CAT = True
except Exception as e:  # noqa: BLE001
    cb, HAS_CAT = None, False
    print("CatBoost tidak tersedia:", e)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception as e:  # noqa: BLE001
    optuna, HAS_OPTUNA = None, False
    print("Optuna tidak tersedia:", e)
try:
    import shap
    HAS_SHAP = True
except Exception as e:  # noqa: BLE001
    shap, HAS_SHAP = None, False
    print("SHAP tidak tersedia (fallback ke native contribution):", e)
try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    HAS_IMBLEARN = True
except Exception as e:  # noqa: BLE001
    SMOTE = SMOTENC = None
    HAS_IMBLEARN = False
    print("imbalanced-learn tidak tersedia (SMOTE nonaktif):", e)
try:
    import category_encoders as ce
    HAS_CE = True
except Exception:  # noqa: BLE001
    ce, HAS_CE = None, False
try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:  # noqa: BLE001
    sm, HAS_SM = None, False
try:
    import torch
    HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch, HAS_TORCH = None, False


def _ver(mod):
    """Ambil versi module dengan aman."""
    try:
        return getattr(mod, "__version__", "?")
    except Exception:  # noqa: BLE001
        return "?"


def version_tuple(v):
    """'1.4.2' -> (1, 4, 2). Dipakai untuk shim kompatibilitas versi."""
    out = []
    for p in str(v).split(".")[:3]:
        m = re.match(r"\d+", p)
        out.append(int(m.group()) if m else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out)


SKLEARN_VERSION = version_tuple(sklearn.__version__)
XGB_VERSION = version_tuple(_ver(xgb)) if HAS_XGB else (0, 0, 0)

VERSIONS = pd.DataFrame([
    ("python", platform.python_version(), True),
    ("numpy", np.__version__, True), ("pandas", pd.__version__, True), ("scikit-learn", sklearn.__version__, True),
    ("scipy", scipy.__version__, True), ("matplotlib", matplotlib.__version__, True),
    ("seaborn", _ver(sns) if HAS_SNS else "-", HAS_SNS),
    ("lightgbm", _ver(lgb) if HAS_LGB else "-", HAS_LGB), ("xgboost", _ver(xgb) if HAS_XGB else "-", HAS_XGB),
    ("catboost", _ver(cb) if HAS_CAT else "-", HAS_CAT), ("optuna", _ver(optuna) if HAS_OPTUNA else "-", HAS_OPTUNA),
    ("shap", _ver(shap) if HAS_SHAP else "-", HAS_SHAP),
    ("imbalanced-learn", "yes" if HAS_IMBLEARN else "-", HAS_IMBLEARN),
    ("category_encoders", _ver(ce) if HAS_CE else "-", HAS_CE),
    ("statsmodels", _ver(sm) if HAS_SM else "-", HAS_SM),
    ("torch", _ver(torch) if HAS_TORCH else "-", HAS_TORCH),
    ("sklearn TargetEncoder", "yes" if HAS_SK_TE else "no (pakai manual K-fold TE)", HAS_SK_TE),
], columns=["library", "version", "available"])
display(VERSIONS)

# %%
# ===== Deteksi GPU (informasi saja; pemakaian diatur oleh CFG.USE_GPU) =====
def detect_gpu():
    """Return dict info GPU. Tidak pernah error."""
    info = {"torch_cuda": False, "nvidia_smi": False, "name": None}
    if HAS_TORCH:
        try:
            info["torch_cuda"] = bool(torch.cuda.is_available())
            if info["torch_cuda"]:
                info["name"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout
            info["nvidia_smi"] = "GPU" in out
            if info["name"] is None and out.strip():
                info["name"] = out.strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
    info["available"] = info["torch_cuda"] or info["nvidia_smi"]
    return info


GPU_INFO = detect_gpu()
GPU_AVAILABLE = GPU_INFO["available"]
print(f"GPU detected: {GPU_AVAILABLE}  | detail: {GPU_INFO}")
print("ℹ️ Default notebook memakai CPU (paling aman). Untuk GPU set CFG.USE_GPU=True di cell USER OVERRIDE;")
print("   jika GPU gagal saat training, notebook otomatis fallback ke CPU.")

# %% [markdown]
# ---
# # 2. Konfigurasi (`class CFG`)
#
# **Tujuan:** semua "kenop" notebook ada di satu tempat. **Jangan edit cell `class CFG` ini untuk lomba tertentu** — edit
# di cell berikutnya (`# >>> USER OVERRIDE`) supaya default tetap utuh dan kamu bisa lihat apa saja yang kamu ubah.
#
# **Kenop terpenting (urut prioritas):**
#
# | Kenop | Arti | Contoh |
# |---|---|---|
# | `TRAIN_PATH`, `TEST_PATH`, `SAMPLE_SUB_PATH` | lokasi file | `"./data/train.csv"` |
# | `TARGET_COL` | nama kolom target | `"Churn"` |
# | `ID_COL` | kolom ID (tidak dipakai sebagai fitur, dipakai di submission) | `"customerID"` |
# | `METRIC` | **harus sama dengan metric lomba** | `"roc_auc"`, `"f1_macro"`, `"rmse"` |
# | `TASK` | `"auto"` / `"binary"` / `"multiclass"` / `"regression"` | `"auto"` |
# | `SUBMISSION_MODE` | binary/multiclass: `"probability"` atau `"label"` (lihat format sample_submission!) | `"auto"` |
# | `RUN_MODE` | `"fast"` untuk debugging, `"full"` untuk final | `"fast"` |
# | `MODELS` | daftar model yang dilatih | `["lgb","xgb","cat"]` |
# | `CV_STRATEGY`, `GROUP_COL`, `TIME_COL` | skema validasi | lihat section 9 |
#
# **Metric yang didukung:** `roc_auc`, `pr_auc` (= `average_precision`), `logloss`, `accuracy`, `f1`, `f1_macro`,
# `f1_weighted`, `balanced_accuracy`, `mcc`, `cohen_kappa`, `qwk` (= `quadratic_weighted_kappa`), `gini`,
# `rmse`, `mae`, `rmsle`, `mape`, `r2`.
#
# **Jika X maka Y (memilih METRIC dan SUBMISSION_MODE):**
#
# - Jika aturan lomba menyebut "AUC" → `METRIC="roc_auc"`, `SUBMISSION_MODE="probability"` (kirim probabilitas, **bukan** label 0/1 — label akan menurunkan AUC drastis).
# - Jika lomba memakai F1 / accuracy → `SUBMISSION_MODE="label"`; notebook otomatis men-tuning threshold (binary) atau class prior (multiclass).
# - Jika `sample_submission` berisi kolom per kelas (prob per kelas) → multiclass `SUBMISSION_MODE="probability"` (biasanya metric logloss).
# - Jika target regresi dan metric RMSLE → `TARGET_TRANSFORM="auto"` otomatis memakai `log1p` (melatih di skala log).
# - Jika target ordinal (rating 1–5) dan metric QWK → bisa `TASK="regression"` + `METRIC="qwk"` (prediksi di-round) atau `TASK="multiclass"`; coba keduanya.
#
# **Untuk laporan:** konfigurasi final tersimpan otomatis di `outputs/.../experiment_*.json` — kutip skema CV, metric, dan model yang dipakai.

# %%
class CFG:
    # ======================================================================================
    # 0. UMUM
    # ======================================================================================
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"          # pip install package yang hilang (section 1). False jika offline. (env TEMPLATE_AUTO_INSTALL=0 → off)
    COMPETITION_NAME = "dac_its_tabular"
    EXPERIMENT_NOTE = ""         # catatan bebas untuk log eksperimen, misal "tambah FE rasio"
    RUN_MODE = "full"            # "fast" = debugging cepat (subsample, 3 fold, model sedikit) | "full" = run final
    SEED = 42                    # seed utama
    SEEDS = None                 # None = [SEED, SEED+1, ...] sebanyak N_REPEATS. Atau isi list, misal [42, 7, 2024]
    N_JOBS = -1                  # jumlah core CPU (-1 = semua). Turunkan ke 2-4 jika laptop panas / memory error.
    VERBOSE = True

    # ======================================================================================
    # 1. PATH
    # ======================================================================================
    TRAIN_PATH = "./data/train.csv"
    TEST_PATH = "./data/test.csv"                 # boleh tidak ada → notebook hanya CV, submission di-skip
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    OUTPUT_DIR = "./outputs/tabular_master/"      # subfolder figures/ tables/ models/ submissions/ dibuat otomatis
    # Tabel tambahan untuk di-merge (multi-file). Contoh:
    # EXTRA_TABLES = [
    #   {"path": "./data/customers.csv", "key": "customer_id", "agg": False, "prefix": "cust_"},       # one-to-one
    #   {"path": "./data/transactions.csv", "key": "customer_id", "agg": True, "prefix": "trx_"},     # one-to-many → agregasi
    # ]
    EXTRA_TABLES = []

    # ======================================================================================
    # 2. KOLOM
    # ======================================================================================
    ID_COL = "id"                # None jika tidak ada ID. Jika tidak ditemukan, dicoba ditebak dari sample_submission.
    TARGET_COL = "target"        # Jika tidak ditemukan, ditebak: kolom yang ada di train tapi tidak di test.
    DROP_COLS = []               # kolom mentah yang dibuang sebelum apa pun (misal kolom bocor / tidak relevan)
    DROP_FEATURES = []           # fitur yang dibuang SETELAH feature engineering (hasil seleksi fitur section 16)
    CAT_COLS = None              # None = auto (object/category/bool). Atau list manual.
    NUM_COLS = None              # None = auto (numerik). Atau list manual.
    NUM_AS_CAT_MAX_UNIQUE = 0    # >0: kolom numerik integer dengan nunique <= nilai ini diperlakukan kategori (misal kode wilayah). 0 = off
    DATE_COLS = None             # None = auto-deteksi string tanggal. Atau list manual, misal ["signup_date"]
    TEXT_COLS = None             # None = auto (string panjang, banyak kata). Atau list manual.
    GROUP_COL = None             # kolom grup untuk GroupKFold (misal customer_id/patient_id) — baris satu grup tidak boleh terpisah
    TIME_COL = None              # kolom waktu untuk TimeSeriesSplit (test = masa depan)
    CLASS_ORDER = None           # urutan label (penting untuk QWK / ordinal), misal ["low","mid","high"]. None = urut alfabet
    POS_LABEL = None             # label positif untuk binary. None = auto (1/Yes/True/kelas minoritas)

    # ======================================================================================
    # 3. TASK & METRIC
    # ======================================================================================
    TASK = "auto"                # "auto" | "binary" | "multiclass" | "regression"
    METRIC = "auto"              # "auto" → binary: roc_auc, multiclass: f1_macro, regression: rmse. SELALU isi sesuai lomba!
    SUBMISSION_MODE = "auto"     # "auto" | "probability" | "label". auto: metric berbasis probabilitas → probability, lainnya → label
    TARGET_TRANSFORM = "auto"    # regresi: None | "log1p" | "auto" (auto = log1p hanya jika metric rmsle)

    # ======================================================================================
    # 4. DATA CLEANING OTOMATIS (section 4)
    # ======================================================================================
    AUTO_CONVERT_NUMERIC_STRINGS = True   # " 1,234.5 ", "$20", "15%" → angka
    AUTO_CONVERT_BOOL_STRINGS = True      # "Yes"/"No", "TRUE"/"FALSE", "Y"/"N" → 1/0
    AUTO_PARSE_DATES = True               # string tanggal → datetime (lalu dipecah jadi fitur di section 8)
    AUTO_DROP_CONSTANT = True             # buang kolom konstan / hampir konstan
    QUASI_CONSTANT_THRESH = 0.999         # kolom dengan 1 nilai >= 99.9% baris dianggap hampir konstan
    AUTO_DROP_DUPLICATE_COLS = True       # buang kolom yang isinya identik dengan kolom lain
    AUTO_DROP_ID_LIKE = True              # buang kolom mirip ID (unik per baris) — biasanya tidak berguna / bocor
    DROP_DUPLICATE_ROWS = False           # True = buang baris duplikat (fitur+target sama persis) di train
    HIGH_CARD_THRESHOLD = 50              # kategori dengan nunique > ini dianggap high-cardinality
    REDUCE_MEMORY = True                  # downcast tipe numerik untuk hemat RAM

    # ======================================================================================
    # 5. EDA (section 6)
    # ======================================================================================
    DO_EDA = True                # False untuk hemat waktu (EDA tidak mempengaruhi model)
    EDA_MAX_PLOTS = 12           # batas jumlah fitur yang diplot per jenis plot (data lebar)
    EDA_SAMPLE_ROWS = 50000      # subsample baris untuk plot/statistik berat
    LEAK_AUC_THRESHOLD = 0.95    # fitur tunggal dengan AUC > ini → dicurigai leakage
    LEAK_R2_THRESHOLD = 0.90     # regresi: fitur tunggal dengan R2 > ini → dicurigai leakage
    DO_ADVERSARIAL = True        # adversarial validation train vs test
    ADV_AUC_WARN = 0.70          # AUC adversarial > ini → ada drift yang perlu diperhatikan

    # ======================================================================================
    # 6. PREPROCESSING (section 7) — semua di-fit di train fold saja
    # ======================================================================================
    NUM_IMPUTE = "median"        # "median" | "mean" | "constant" (untuk model linear/KNN/SVM/RF; GBDT pakai NaN native)
    NUM_IMPUTE_CONSTANT = -999
    ADD_MISSING_INDICATORS = True  # tambah kolom <col>__isna untuk model non-GBDT
    RARE_MIN_COUNT = 10          # kategori dengan frekuensi < ini (di train fold) digabung jadi "__RARE__". 0 = off
    OHE_MAX_CARD = 15            # model linear: one-hot jika nunique <= ini, sisanya target encoding
    TARGET_ENCODING = True       # target encoding (fold-safe) untuk kategori high-card di model linear/KNN/SVM/MLP
    TE_FOR_TREES = False         # True: tambahkan juga kolom target encoding untuk model tree (coba jika banyak kategori high-card)
    TE_MIN_CARD = 10             # target encoding hanya untuk kategori dengan nunique >= ini (untuk TE_FOR_TREES)
    TE_SMOOTHING = 20            # smoothing manual TE (semakin besar → semakin ditarik ke rata-rata global)
    TE_IMPL = "auto"             # "auto" (sklearn TargetEncoder jika ada) | "sklearn" | "manual"
    CLIP_OUTLIERS = False        # winsorize numerik (batas quantile di-fit di train fold)
    CLIP_QUANTILES = (0.005, 0.995)
    LOG_TRANSFORM_LINEAR = True  # model linear: log1p otomatis untuk fitur skew>1 dan >=0
    XGB_NATIVE_CAT = False       # True: XGBoost pakai categorical native (enable_categorical). False: ordinal codes (lebih aman)

    # ======================================================================================
    # 7. FEATURE ENGINEERING (section 8) — tiap toggle bisa dicoba dan dibandingkan CV-nya
    # ======================================================================================
    FE_DATE_PARTS = True         # tahun, bulan, hari, dayofweek, dst dari kolom tanggal
    FE_FREQ_ENCODING = True      # frekuensi kemunculan kategori
    FE_FREQ_FIT_ON = "train+test"  # "train+test" (transduktif, umum di lomba) | "train"
    FE_GROUP_AGG = False         # agregasi numerik per kategori (mean/std/diff) — sering ampuh, tapi menambah banyak kolom
    FE_GROUP_AGG_CATS = None     # None = auto (3 kategori dengan MI tertinggi)
    FE_GROUP_AGG_NUMS = None     # None = auto (5 numerik dengan MI tertinggi)
    FE_RATIO_PAIRS = []          # list pasangan (a, b) → a/b dan a-b. Contoh: [("TotalCharges", "tenure")]
    FE_AUTO_RATIO_TOPK = 0       # >0: buat rasio otomatis antar top-k numerik (k=4 → 6 pasang)
    FE_POLY_TOPK = 0             # >0: interaksi perkalian antar top-k numerik
    FE_BINNING_COLS = []         # kolom numerik yang di-binning (quantile)
    FE_N_BINS = 10
    FE_TEXT_LEN = True           # panjang teks, jumlah kata, dll untuk kolom teks
    DROP_RAW_TEXT = True         # buang kolom teks mentah setelah diekstrak fiturnya
    FE_ROW_STATS = True          # n_missing per baris, statistik numerik per baris
    FE_CUSTOM = True             # jalankan custom_feature_engineering(df)

    # ======================================================================================
    # 8. CROSS-VALIDATION (section 9)
    # ======================================================================================
    CV_STRATEGY = "auto"         # "auto" | "stratified" | "kfold" | "group" | "stratified_group" | "timeseries"
    N_FOLDS = 5
    N_REPEATS = 1                # >1 = repeated CV dengan seed berbeda (lebih stabil, lebih lama)
    REG_STRAT_BINS = 10          # regresi: stratifikasi berdasarkan bin target

    # ======================================================================================
    # 9. MODEL (section 10-11)
    # ======================================================================================
    # Pilihan: "dummy","linear","enet","rf","et","hgb","lgb","xgb","cat","knn","svm","mlp"
    MODELS = ["dummy", "linear", "rf", "et", "hgb", "lgb", "xgb", "cat"]
    FAST_MODELS = ["dummy", "linear", "hgb", "lgb"]   # model yang dipakai saat RUN_MODE="fast"
    MODEL_PARAMS = {}            # override parameter per model, misal {"lgb": {"num_leaves": 63}, "cat": {"depth": 8}}
    GBDT_N_ESTIMATORS = 5000     # batas atas iterasi; early stopping yang menentukan jumlah sebenarnya
    GBDT_LR = 0.03               # learning rate final. Fast mode memakai 0.08
    EARLY_STOPPING = 200
    RF_N_ESTIMATORS = 500
    SMALL_DATA_MAX_ROWS = 20000  # KNN/SVM hanya dijalankan jika jumlah baris <= ini
    IMBALANCE_STRATEGY = "auto"  # "auto" | None | "class_weight" | "smote". auto: class_weight jika imbalance > 3x DAN metric berbasis label
    KEEP_FOLD_MODELS = True      # simpan model tiap fold di memori (dibutuhkan permutation importance/SHAP/PDP)
    USE_GPU = False              # True: coba GPU untuk LightGBM/XGBoost/CatBoost (fallback ke CPU jika gagal)
    LGB_GPU = None               # None = ikut USE_GPU. LightGBM GPU sering tidak ter-compile → default ikut USE_GPU tapi fallback
    XGB_GPU = None
    CAT_GPU = None

    # ======================================================================================
    # 10. TUNING (section 12)
    # ======================================================================================
    USE_OPTUNA = True            # di fast mode otomatis False
    OPTUNA_MODELS = ["lgb", "xgb", "cat"]
    OPTUNA_N_TRIALS = 40
    OPTUNA_TIMEOUT = 600         # detik per model (batas waktu)
    OPTUNA_N_FOLDS = None        # None = semua fold; isi 3 untuk mempercepat

    # ======================================================================================
    # 11. ENSEMBLE & POST-PROCESSING (section 13-14)
    # ======================================================================================
    ENSEMBLE_METHODS = ["mean", "rank", "weighted", "hill", "stack"]
    ENSEMBLE_FORCE = None        # None = pilih terbaik berdasarkan OOF. Atau paksa "mean"/"weighted"/... atau nama model tunggal
    ENSEMBLE_EXCLUDE = ["dummy"] # model yang tidak ikut ensemble
    HILL_MAX_ITERS = 50
    THRESHOLD_TUNING = True      # binary + metric berbasis label → cari threshold terbaik di OOF
    CLASS_PRIOR_ADJUST = True    # multiclass + metric berbasis label → bobot per kelas
    CALIBRATION = "auto"         # "auto" | "isotonic" | "sigmoid" | None. auto = coba, pakai jika metric logloss/brier membaik
    REG_CLIP = True              # regresi: clip prediksi ke range target train
    REG_ROUND = "auto"           # regresi: True/False/"auto" (auto: round jika target integer dan skor membaik)

    # ======================================================================================
    # 12. EVALUASI & INTERPRETASI (section 15-16)
    # ======================================================================================
    DO_EVAL_PLOTS = True
    DO_LEARNING_CURVE = True
    DO_SHUFFLE_TEST = True
    DO_INTERPRET = True          # master switch section 16
    DO_PERM_IMP = True
    PERM_REPEATS = 3
    PERM_MAX_FEATURES = 40
    DO_SHAP = True
    SHAP_SAMPLE = 2000
    SHAP_CLASS_INDEX = None      # multiclass: kelas yang diplot beeswarm/waterfall (None = kelas terakhir)
    DO_PDP = True
    DO_NULL_IMP = True
    NULL_IMP_RUNS = 10
    DO_FEATURE_SELECTION_CHECK = True

    # ======================================================================================
    # 13. FINAL FIT & SUBMISSION (section 17)
    # ======================================================================================
    FINAL_FIT_STRATEGY = "fold_average"  # "fold_average" (rata-rata model tiap fold) | "refit_full" (latih ulang di full train)
    SAVE_SUBMISSION_VARIANTS = True

    # ======================================================================================
    # 14. FAST MODE & FIGURE
    # ======================================================================================
    SAMPLE_FRAC = 0.3            # fast mode: proporsi train yang dipakai
    FAST_MAX_ROWS = 20000        # fast mode: batas atas baris
    FAST_N_FOLDS = 3
    FIG_DPI = 200                # 300 untuk figur laporan final
    SHOW_PLOTS = True            # False = hanya simpan file (lebih cepat untuk data lebar)

# %% [markdown]
# ### 2.1 USER OVERRIDE — edit cell di bawah untuk lomba kamu
#
# Isi hanya yang perlu diubah. Contoh lengkap ada di komentar. Cell ini sengaja hanya berisi komentar secara default
# (tetap valid untuk dijalankan).
#
# **Jika X maka Y:**
#
# - Jika belum yakin semuanya jalan → `CFG.RUN_MODE = "fast"` dulu.
# - Jika data punya kolom grup (customer/pasien/toko berulang di banyak baris dan test berisi grup baru) → `CFG.GROUP_COL = "..."`.
# - Jika test adalah periode waktu setelah train → `CFG.TIME_COL = "tanggal"` (CV otomatis TimeSeriesSplit).
# - Jika ada kolom yang jelas bocor (misal `churn_date` untuk target churn) → masukkan ke `CFG.DROP_COLS`.

# %%
# >>> USER OVERRIDE
# Contoh (hapus tanda # di depan baris yang ingin dipakai):
# CFG.TRAIN_PATH = "./data/train.csv"
# CFG.TEST_PATH = "./data/test.csv"
# CFG.SAMPLE_SUB_PATH = "./data/sample_submission.csv"
# CFG.ID_COL = "id"
# CFG.TARGET_COL = "target"
# CFG.TASK = "auto"
# CFG.METRIC = "roc_auc"
# CFG.SUBMISSION_MODE = "probability"
# CFG.RUN_MODE = "fast"
# CFG.DROP_COLS = []
# CFG.GROUP_COL = None
# CFG.MODELS = ["linear", "lgb", "xgb", "cat"]
# CFG.USE_OPTUNA = False
# CFG.FIG_DPI = 300

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
# ### 2.2 Finalisasi config + helper umum
#
# Cell ini menerjemahkan `CFG` menjadi setting aktif (`RUN.*`) — misal fast mode mengecilkan fold, model, learning
# rate. Juga mendefinisikan helper yang dipakai di seluruh notebook: `seed_everything`, `save_fig`, `save_table`,
# `safe_block` (agar analisis opsional tidak menghentikan notebook), dan **fungsi metric**.
#
# **Cara membaca output:** cek baris `RUN MODE`, `MODELS aktif`, `N_FOLDS`. Jika ada yang tidak sesuai harapan, kembali ke cell override.
#
# **Catatan metric:**
#
# - Metric berbasis **probabilitas** (`roc_auc`, `pr_auc`, `logloss`, `gini`): dihitung langsung dari probabilitas.
# - Metric berbasis **label** (`accuracy`, `f1*`, `mcc`, `balanced_accuracy`, `cohen_kappa`, `qwk`): probabilitas diubah
#   ke label (binary: threshold, default 0.5 → bisa dituning di section 14; multiclass: argmax).
# - `rmsle`: prediksi negatif di-clip ke 0. `mape`: baris dengan target 0 diabaikan agar tidak tak hingga.

# %%
class RUN:
    """Setting aktif hasil turunan CFG (diisi oleh finalize_cfg)."""


METRIC_ALIASES = {
    "auc": "roc_auc", "rocauc": "roc_auc", "roc-auc": "roc_auc", "average_precision": "pr_auc", "ap": "pr_auc",
    "prauc": "pr_auc", "log_loss": "logloss", "cross_entropy": "logloss", "acc": "accuracy",
    "quadratic_weighted_kappa": "qwk", "kappa": "cohen_kappa", "f1_binary": "f1", "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted", "mse_root": "rmse", "root_mean_squared_error": "rmse",
    "mean_absolute_error": "mae", "r2_score": "r2", "mean_absolute_percentage_error": "mape",
}
PROB_METRICS = {"roc_auc", "pr_auc", "logloss", "gini"}
LABEL_METRICS = {"accuracy", "f1", "f1_macro", "f1_weighted", "balanced_accuracy", "mcc", "cohen_kappa", "qwk"}
REG_METRICS = {"rmse", "mae", "rmsle", "mape", "r2"}
GREATER_IS_BETTER = {"roc_auc": True, "pr_auc": True, "logloss": False, "gini": True, "accuracy": True, "f1": True,
                     "f1_macro": True, "f1_weighted": True, "balanced_accuracy": True, "mcc": True,
                     "cohen_kappa": True, "qwk": True, "rmse": False, "mae": False, "rmsle": False,
                     "mape": False, "r2": True}


def normalize_metric(m):
    """Normalisasi nama metric (alias → nama standar)."""
    if m is None:
        return "auto"
    m = str(m).strip().lower()
    return METRIC_ALIASES.get(m, m)


def finalize_cfg():
    """Turunkan setting aktif dari CFG. Aman dijalankan berulang kali (idempotent)."""
    RUN.fast = str(CFG.RUN_MODE).lower() == "fast"
    RUN.n_folds = min(CFG.N_FOLDS, CFG.FAST_N_FOLDS) if RUN.fast else CFG.N_FOLDS
    RUN.n_repeats = 1 if RUN.fast else max(1, int(CFG.N_REPEATS))
    if CFG.SEEDS:
        RUN.seeds = list(CFG.SEEDS)[: RUN.n_repeats] if RUN.fast else list(CFG.SEEDS)
        RUN.n_repeats = len(RUN.seeds)
    else:
        RUN.seeds = [CFG.SEED + i for i in range(RUN.n_repeats)]
    RUN.models = list(CFG.FAST_MODELS) if RUN.fast else list(CFG.MODELS)
    RUN.lr = 0.08 if RUN.fast else CFG.GBDT_LR
    RUN.n_estimators = min(CFG.GBDT_N_ESTIMATORS, 1500) if RUN.fast else CFG.GBDT_N_ESTIMATORS
    RUN.early_stopping = min(CFG.EARLY_STOPPING, 100) if RUN.fast else CFG.EARLY_STOPPING
    RUN.rf_trees = min(CFG.RF_N_ESTIMATORS, 200) if RUN.fast else CFG.RF_N_ESTIMATORS
    RUN.optuna = bool(CFG.USE_OPTUNA) and not RUN.fast
    RUN.null_imp_runs = min(CFG.NULL_IMP_RUNS, 4) if RUN.fast else CFG.NULL_IMP_RUNS
    RUN.perm_repeats = min(CFG.PERM_REPEATS, 2) if RUN.fast else CFG.PERM_REPEATS
    RUN.metric = normalize_metric(CFG.METRIC)
    RUN.use_gpu = {"lgb": CFG.USE_GPU if CFG.LGB_GPU is None else CFG.LGB_GPU,
                   "xgb": CFG.USE_GPU if CFG.XGB_GPU is None else CFG.XGB_GPU,
                   "cat": CFG.USE_GPU if CFG.CAT_GPU is None else CFG.CAT_GPU}
    RUN.out = Path(CFG.OUTPUT_DIR)
    for sub in ["figures", "tables", "models", "submissions", "preds"]:
        os.makedirs(RUN.out / sub, exist_ok=True)


finalize_cfg()
GPU_DISABLED = {"lgb": False, "xgb": False, "cat": False}   # diisi True otomatis jika GPU gagal → fallback CPU


def seed_everything(seed=42):
    """Set seed random, numpy, PYTHONHASHSEED, dan torch (jika ada)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_TORCH:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:  # noqa: BLE001
            pass


seed_everything(CFG.SEED)

# ---------- Figure & table helpers ----------
plt.rcParams.update({"figure.dpi": 100, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                     "axes.grid": True, "grid.alpha": 0.3})
if HAS_SNS:
    sns.set_theme(style="whitegrid", font_scale=0.95)
_FIG_COUNTER = [0]
FIGURES_SAVED = []


def save_fig(name, fig=None):
    """Simpan figure ke figures/figNN_<name>.png (auto-numbering), tampilkan (jika SHOW_PLOTS), lalu close."""
    fig = fig or plt.gcf()
    _FIG_COUNTER[0] += 1
    safe = re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(name)).strip("_")[:80]
    path = RUN.out / "figures" / f"fig{_FIG_COUNTER[0]:02d}_{safe}.png"
    try:
        fig.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight")
        FIGURES_SAVED.append(str(path))
    except Exception as e:  # noqa: BLE001
        print("⚠️ gagal simpan figure:", e)
    if CFG.SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    plt.close("all")
    return path


def save_table(df, name, index=False):
    """Simpan DataFrame ke tables/<name>.csv."""
    path = RUN.out / "tables" / f"{name}.csv"
    try:
        df.to_csv(path, index=index)
    except Exception as e:  # noqa: BLE001
        print("⚠️ gagal simpan tabel:", e)


class safe_block:
    """Context manager: error di dalam blok dicetak ringkas, notebook tetap lanjut.

    Contoh:  with safe_block("EDA korelasi"): ...
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            print(f"⚠️ [{self.name}] gagal: {exc_type.__name__}: {exc}")
            print("   " + " | ".join(traceback.format_exception(exc_type, exc, tb)[-3:]).replace("\n", " ")[:800])
            plt.close("all")
            return True  # tahan error
        return False


def banner(title):
    """Print judul section yang mudah dilihat."""
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


# ---------- Metric helpers ----------
def rmse_score(y, p):
    """RMSE manual (argumen squared= dihapus di sklearn 1.6)."""
    return float(np.sqrt(mean_squared_error(y, p)))


def to_labels(pred, task, threshold=0.5, class_weights=None):
    """Ubah prediksi probabilitas/kontinu menjadi label integer."""
    pred = np.asarray(pred)
    if task == "binary":
        return (pred >= threshold).astype(int)
    if task == "multiclass":
        p = pred * np.asarray(class_weights)[None, :] if class_weights is not None else pred
        return np.argmax(p, axis=1)
    return np.rint(pred).astype(int)


def compute_metric(y_true, pred, metric=None, task=None, threshold=0.5, class_weights=None):
    """Hitung metric standar dari prediksi (prob untuk klasifikasi, nilai untuk regresi)."""
    metric = metric or RUN.metric
    task = task or TASK
    y_true = np.asarray(y_true)
    pred = np.asarray(pred, dtype=float)
    if task == "regression":
        if metric == "rmse":
            return rmse_score(y_true, pred)
        if metric == "mae":
            return float(mean_absolute_error(y_true, pred))
        if metric == "rmsle":
            return rmse_score(np.log1p(np.clip(y_true, 0, None)), np.log1p(np.clip(pred, 0, None)))
        if metric == "mape":
            mask = y_true != 0
            return float(np.mean(np.abs((y_true[mask] - pred[mask]) / y_true[mask])))
        if metric == "r2":
            return float(r2_score(y_true, pred))
        if metric in ("qwk", "cohen_kappa", "accuracy"):
            lab = np.clip(np.rint(pred), np.min(y_true), np.max(y_true)).astype(int)
            if metric == "accuracy":
                return float(accuracy_score(y_true.astype(int), lab))
            w = "quadratic" if metric == "qwk" else None
            return float(cohen_kappa_score(y_true.astype(int), lab, weights=w))
        raise ValueError(f"Metric {metric} tidak cocok untuk regresi")
    # ---- klasifikasi ----
    if task == "binary":
        p = np.clip(pred.reshape(-1), 1e-15, 1 - 1e-15)
        if metric == "roc_auc":
            return float(roc_auc_score(y_true, p))
        if metric == "gini":
            return float(2 * roc_auc_score(y_true, p) - 1)
        if metric == "pr_auc":
            return float(average_precision_score(y_true, p))
        if metric == "logloss":
            return float(log_loss(y_true, p, labels=[0, 1]))
    else:
        K = pred.shape[1]
        p = np.clip(pred, 1e-15, 1)
        p = p / p.sum(axis=1, keepdims=True)
        if metric in ("roc_auc", "gini"):
            present = np.unique(y_true)
            if len(present) < K:   # ada kelas yang tidak muncul → hitung macro hanya pada kelas yang ada
                Y = label_binarize(y_true, classes=list(range(K)))
                aucs = [roc_auc_score(Y[:, k], p[:, k]) for k in present]
                auc = float(np.mean(aucs))
            else:
                auc = float(roc_auc_score(y_true, p, multi_class="ovr", average="macro"))
            return auc if metric == "roc_auc" else 2 * auc - 1
        if metric == "pr_auc":
            Y = label_binarize(y_true, classes=list(range(K)))
            return float(average_precision_score(Y, p, average="macro"))
        if metric == "logloss":
            return float(log_loss(y_true, p, labels=list(range(K))))
    lab = to_labels(pred, task, threshold, class_weights)
    if metric == "accuracy":
        return float(accuracy_score(y_true, lab))
    if metric == "f1":
        return float(f1_score(y_true, lab, average="binary" if task == "binary" else "macro", zero_division=0))
    if metric == "f1_macro":
        return float(f1_score(y_true, lab, average="macro", zero_division=0))
    if metric == "f1_weighted":
        return float(f1_score(y_true, lab, average="weighted", zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, lab))
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, lab))
    if metric == "cohen_kappa":
        return float(cohen_kappa_score(y_true, lab))
    if metric == "qwk":
        return float(cohen_kappa_score(y_true, lab, weights="quadratic"))
    raise ValueError(f"Metric tidak dikenal: {metric}")


def greater_is_better(metric=None):
    """True jika metric semakin besar semakin baik."""
    return GREATER_IS_BETTER.get(metric or RUN.metric, True)


def is_better(a, b, metric=None):
    """Apakah skor a lebih baik dari b?"""
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return True
    return a > b if greater_is_better(metric) else a < b


def metric_loss(y_true, pred, **kw):
    """Versi 'loss' (semakin kecil semakin baik) untuk optimisasi."""
    s = compute_metric(y_true, pred, **kw)
    return -s if greater_is_better(kw.get("metric")) else s


print(f"RUN MODE      : {CFG.RUN_MODE}  (fast={RUN.fast})")
print(f"MODELS aktif  : {RUN.models}")
print(f"N_FOLDS={RUN.n_folds} | N_REPEATS={RUN.n_repeats} | SEEDS={RUN.seeds} | LR={RUN.lr}")
print(f"METRIC (raw)  : {CFG.METRIC} → {RUN.metric}")
print(f"Optuna aktif  : {RUN.optuna} | GPU: {RUN.use_gpu}")
print(f"OUTPUT_DIR    : {RUN.out.resolve()}")

# %% [markdown]
# ---
# # 3. Load Data (smart loading + multi-file merge)
#
# **Tujuan:** membaca train/test/sample_submission dalam format apa pun, menyeragamkan nama kolom, dan (jika ada)
# menggabungkan tabel tambahan.
#
# **Yang dilakukan:**
#
# 1. `smart_read(path)` — mendukung `csv`, `tsv`, `txt`, `xlsx/xls`, `parquet`, `json`, `feather`, `pkl`. Untuk CSV:
#    mencoba encoding `utf-8 → utf-8-sig → latin-1`, dan jika hasilnya cuma 1 kolom (separator bukan koma, misal `;`
#    atau `|`) otomatis mencoba **sniffing separator**.
# 2. Jika file test **tidak ada** → `HAS_TEST=False`. Notebook tetap jalan (CV saja), submission di-skip. Supaya kode
#    tetap seragam, dibuat "pseudo-test" kecil dari train (hanya untuk menjaga alur kode, tidak disimpan).
# 3. **Sanitasi nama kolom**: karakter selain huruf/angka/underscore diganti `_`. Ini mencegah error LightGBM
#    `Do not support special JSON characters in feature name` dan error XGBoost `feature_names must be string, and may not contain [, ] or <`.
#    Nama asli tetap disimpan (`COLMAP`) dan semua nama kolom di CFG ikut diterjemahkan otomatis.
# 4. Deteksi otomatis `TARGET_COL` / `ID_COL` jika yang kamu isi tidak ditemukan.
# 5. **Multi-file**: `safe_merge()` (left merge + cek jumlah baris tidak berubah) dan `aggregate_child()` (agregasi tabel
#    one-to-many: count/mean/sum/min/max/nunique/last per key). Diatur lewat `CFG.EXTRA_TABLES`.
#
# **Cara membaca output:** cek shape train/test, 5 baris pertama, dan baris `TARGET_COL` / `ID_COL` yang dipakai.
#
# **Jika X maka Y:**
#
# - Jika train hanya punya **1 kolom** dengan nama panjang berisi `;` → separator salah; `smart_read` biasanya membetulkan, jika tidak isi `sep` manual: `smart_read(path, sep=";")`.
# - Jika muncul `UnicodeDecodeError` → file bukan UTF-8; `smart_read` mencoba latin-1 otomatis.
# - Jika setelah merge **jumlah baris bertambah** → key di tabel kanan tidak unik (one-to-many). Pakai `"agg": True` agar diagregasi dulu.
# - Jika `TARGET_COL not found` → cek ejaan/kapital. Lihat daftar kolom yang dicetak.
# - Jika train sangat besar (> 2 juta baris) dan RAM kecil → set `CFG.RUN_MODE="fast"` dulu, `REDUCE_MEMORY=True`, atau baca dengan `usecols`.
#
# **Untuk laporan:** tulis sumber data, jumlah baris & kolom train/test, dan (jika ada) bagaimana tabel digabung.

# %%
def smart_read(path, sep=None, **kwargs):
    """Baca file tabular apa pun berdasarkan ekstensi, dengan fallback encoding & separator."""
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path, **kwargs)
    if ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, **kwargs)
    if ext == ".feather":
        return pd.read_feather(path, **kwargs)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(path, **kwargs)
    if ext in (".json", ".jsonl"):
        try:
            return pd.read_json(path, lines=(ext == ".jsonl"), **kwargs)
        except ValueError:
            return pd.read_json(path, lines=True, **kwargs)
    # csv / tsv / txt
    if sep is None and ext == ".tsv":
        sep = "\t"
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            if sep is not None:
                return pd.read_csv(path, sep=sep, encoding=enc, **kwargs)
            df = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
            if df.shape[1] == 1 and any(ch in str(df.columns[0]) for ch in [";", "\t", "|"]):
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc, **kwargs)
                print(f"  ℹ️ separator di-sniff otomatis untuk {Path(path).name}")
            if enc != "utf-8":
                print(f"  ℹ️ {Path(path).name} dibaca dengan encoding {enc}")
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err


def sanitize_name(name):
    """Nama kolom aman untuk LightGBM/XGBoost/CatBoost: hanya [A-Za-z0-9_]."""
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if s == "":
        s = "col"
    if s[0].isdigit():
        s = "f_" + s
    return s


def build_colmap(columns_list):
    """Mapping nama asli → nama aman yang unik (untuk gabungan kolom train+test)."""
    colmap, used = {}, set()
    for c in columns_list:
        if c in colmap:
            continue
        base = sanitize_name(c)
        new, k = base, 1
        while new in used:
            k += 1
            new = f"{base}_{k}"
        colmap[c] = new
        used.add(new)
    return colmap


def unique_names(names):
    """Jadikan list nama unik (duplikat diberi akhiran _2, _3, ...) dengan urutan & panjang tetap."""
    out, used = [], set()
    for n in names:
        base, cand, k = n, n, 1
        while cand in used:
            k += 1
            cand = f"{base}_{k}"
        used.add(cand)
        out.append(cand)
    return out


def safe_merge(left, right, on, how="left", name="right", prefix=""):
    """Left-merge dengan pengecekan: key kanan harus unik, jumlah baris kiri tidak berubah."""
    right = right.copy()
    if prefix:
        right = right.rename(columns={c: prefix + str(c) for c in right.columns if c != on})
    dup = right[on].duplicated().sum()
    if dup > 0:
        raise ValueError(f"Key '{on}' di tabel {name} tidak unik ({dup} duplikat). Gunakan aggregate_child() dulu.")
    n0 = len(left)
    out = left.merge(right, on=on, how=how)
    assert len(out) == n0, f"Jumlah baris berubah setelah merge {name}: {n0} → {len(out)}"
    miss = out[[c for c in right.columns if c != on]].isna().all(axis=1).mean() if right.shape[1] > 1 else 0
    print(f"  merge {name}: +{right.shape[1] - 1} kolom | baris tetap {n0} | {miss:.1%} baris tidak punya pasangan")
    return out


def aggregate_child(child, key, prefix="child_", num_aggs=("count", "mean", "sum", "min", "max", "std"),
                    cat_aggs=("nunique", "last"), sort_col=None):
    """Agregasi tabel anak (one-to-many) menjadi satu baris per key.

    - Numerik: count/mean/sum/min/max/std
    - Kategorikal: nunique/last (last = nilai terakhir; urutkan dulu dengan sort_col, misal tanggal transaksi)
    """
    df = child.copy()
    if sort_col is not None and sort_col in df.columns:
        df = df.sort_values(sort_col)
    num = [c for c in df.columns if c != key and pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if c != key and c not in num]
    parts = [df.groupby(key).size().rename(f"{prefix}n_rows")]
    if num:
        agg_num = df.groupby(key)[num].agg([a for a in num_aggs if a != "count"])
        agg_num.columns = [f"{prefix}{c}_{a}" for c, a in agg_num.columns]
        parts.append(agg_num)
    if cat:
        agg_cat = df.groupby(key)[cat].agg(list(cat_aggs))
        agg_cat.columns = [f"{prefix}{c}_{a}" for c, a in agg_cat.columns]
        parts.append(agg_cat)
    out = pd.concat(parts, axis=1).reset_index()
    print(f"  aggregate_child: {child.shape} → {out.shape} (1 baris per {key})")
    return out

# %%
banner("LOAD DATA")
assert os.path.exists(CFG.TRAIN_PATH), f"File train tidak ditemukan: {CFG.TRAIN_PATH} (cek CFG.TRAIN_PATH)"
train_raw = smart_read(CFG.TRAIN_PATH)
HAS_TEST = bool(CFG.TEST_PATH) and os.path.exists(str(CFG.TEST_PATH))
test_raw = smart_read(CFG.TEST_PATH) if HAS_TEST else None
HAS_SAMPLE = bool(CFG.SAMPLE_SUB_PATH) and os.path.exists(str(CFG.SAMPLE_SUB_PATH))
sample_sub = smart_read(CFG.SAMPLE_SUB_PATH) if HAS_SAMPLE else None
print(f"train: {train_raw.shape} | test: {None if test_raw is None else test_raw.shape} | "
      f"sample_submission: {None if sample_sub is None else sample_sub.shape}")
if not HAS_TEST:
    print("⚠️ File test tidak ditemukan → notebook hanya melakukan CV. Submission akan di-skip.")
if not HAS_SAMPLE:
    print("ℹ️ sample_submission tidak ditemukan → submission dibuat dengan format [ID, TARGET].")

# ---- Merge tabel tambahan (opsional) ----
for spec in CFG.EXTRA_TABLES:
    with safe_block(f"merge {spec.get('path')}"):
        extra = smart_read(spec["path"])
        key = spec["key"]
        if spec.get("agg", False):
            extra = aggregate_child(extra, key, prefix=spec.get("prefix", "x_"), sort_col=spec.get("sort_col"))
            pref = ""
        else:
            pref = spec.get("prefix", "")
        train_raw = safe_merge(train_raw, extra, on=key, name=Path(spec["path"]).name, prefix=pref)
        if HAS_TEST:
            test_raw = safe_merge(test_raw, extra, on=key, name=Path(spec["path"]).name, prefix=pref)

# %%
# ---- Sanitasi nama kolom + resolusi nama kolom di CFG ----
ORIG_TRAIN_COLS = list(train_raw.columns)
_all_cols = ORIG_TRAIN_COLS + ([c for c in test_raw.columns if c not in ORIG_TRAIN_COLS] if HAS_TEST else [])
COLMAP = build_colmap(_all_cols)
INV_COLMAP = {v: k for k, v in COLMAP.items()}
renamed = {k: v for k, v in COLMAP.items() if k != v}
if renamed:
    print(f"ℹ️ {len(renamed)} nama kolom disanitasi, contoh: {dict(list(renamed.items())[:6])}")
train = train_raw.rename(columns=COLMAP)
test = test_raw.rename(columns=COLMAP) if HAS_TEST else None


def col_name(c):
    """Terjemahkan nama kolom dari CFG (nama asli) ke nama tersanitasi."""
    if c is None:
        return None
    if c in COLMAP:
        return COLMAP[c]
    if c in COLMAP.values():
        return c
    return sanitize_name(c)


def col_list(cols):
    """Versi list dari col_name (None tetap None)."""
    return None if cols is None else [col_name(c) for c in cols]


# ---- Resolusi TARGET_COL ----
_n_sub_targets = (sample_sub.shape[1] - 1) if HAS_SAMPLE else 1
if isinstance(CFG.TARGET_COL, (list, tuple)) or str(CFG.TASK).lower() == "multilabel" or (
        _n_sub_targets > 1 and all(col_name(c) in train.columns for c in sample_sub.columns[1:])):
    raise ValueError("Soal ini MULTILABEL (beberapa kolom target sekaligus). Pakai template "
                     "02_Tabular/tabular_multilabel.ipynb (.py) — atau: python tools/new_task.py <folder> --task tabular-multilabel")
TARGET = col_name(CFG.TARGET_COL)
if TARGET not in train.columns:
    cand = [c for c in train.columns if HAS_TEST and c not in test.columns]
    if HAS_SAMPLE and sample_sub.shape[1] >= 2 and col_name(sample_sub.columns[1]) in train.columns:
        TARGET = col_name(sample_sub.columns[1])
    elif len(cand) == 1:
        TARGET = cand[0]
    else:
        raise ValueError(f"TARGET_COL '{CFG.TARGET_COL}' tidak ditemukan. Kandidat (ada di train, tidak di test): {cand}. "
                         f"Kolom train: {list(train.columns)[:50]}")
    print(f"⚠️ TARGET_COL '{CFG.TARGET_COL}' tidak ada → memakai tebakan: '{INV_COLMAP.get(TARGET, TARGET)}'")
TARGET_ORIG = INV_COLMAP.get(TARGET, TARGET)

# ---- Resolusi ID_COL ----
ID = col_name(CFG.ID_COL) if CFG.ID_COL else None
if ID is not None and ID not in train.columns:
    guess = col_name(sample_sub.columns[0]) if HAS_SAMPLE else None
    if guess is not None and guess in train.columns and guess != TARGET:
        print(f"⚠️ ID_COL '{CFG.ID_COL}' tidak ada → memakai kolom pertama sample_submission: '{INV_COLMAP.get(guess, guess)}'")
        ID = guess
    else:
        print(f"ℹ️ ID_COL '{CFG.ID_COL}' tidak ditemukan → dianggap tidak ada ID (index dipakai sebagai ID).")
        ID = None
ID_ORIG = INV_COLMAP.get(ID, ID) if ID else (sample_sub.columns[0] if HAS_SAMPLE else "id")
GROUP = col_name(CFG.GROUP_COL)
TIME = col_name(CFG.TIME_COL)
for nm, c in [("GROUP_COL", GROUP), ("TIME_COL", TIME)]:
    if c is not None and c not in train.columns:
        raise ValueError(f"{nm} '{c}' tidak ada di train.")

# ---- Pseudo-test jika test tidak ada (hanya untuk menjaga alur kode) ----
if not HAS_TEST:
    test = train.drop(columns=[TARGET]).head(min(500, len(train))).copy()

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
print(f"TARGET = '{TARGET_ORIG}' (internal: {TARGET}) | ID = '{ID_ORIG}' (internal: {ID}) | GROUP = {GROUP} | TIME = {TIME}")
print("train shape:", train.shape, "| test shape:", test.shape, "(pseudo)" if not HAS_TEST else "")
display(train.head())

# %% [markdown]
# ---
# # 4. Data Quality Report (otomatis) + Cleaning
#
# **Tujuan:** menemukan masalah data **sebelum** modeling. Data kotor = model jelek, atau lebih buruk lagi: skor CV bagus
# tapi leaderboard hancur.
#
# **Yang dicek (dan diperbaiki otomatis jika toggle di CFG aktif):**
#
# | Cek | Contoh masalah | Aksi otomatis |
# |---|---|---|
# | Overview | dtype, % missing, jumlah unik, contoh nilai | tabel `dq_overview.csv` |
# | Angka tersimpan sebagai teks | `" 1,234.50"`, `"$20"`, `"15%"`, `" "` (Telco `TotalCharges`) | dikonversi ke float (`AUTO_CONVERT_NUMERIC_STRINGS`) |
# | Boolean sebagai teks | `"Yes"/"No"`, `"TRUE"/"FALSE"` | → 1/0 (`AUTO_CONVERT_BOOL_STRINGS`) |
# | Tanggal sebagai teks | `"2024-01-31"`, `"31/01/2024 10:00"` | → datetime, nanti dipecah di section 8 |
# | Nilai `inf` | hasil pembagian nol di data mentah | → NaN |
# | Kolom konstan / hampir konstan | 1 nilai di 99.9% baris | dibuang (`AUTO_DROP_CONSTANT`) |
# | Kolom duplikat | dua kolom isinya sama persis | yang kedua dibuang |
# | Kolom mirip ID | unik per baris, nama mengandung `id` | dibuang (`AUTO_DROP_ID_LIKE`) |
# | Baris duplikat & label konflik | fitur sama, target beda | dilaporkan (opsional dibuang) |
# | High-cardinality | kategori dengan ratusan nilai unik | dilaporkan (ditangani encoder) |
# | Tipe campuran | angka dan string dalam satu kolom | dipaksa jadi string |
# | Nilai negatif mencurigakan | `age=-5`, `price<0` | dilaporkan |
# | Outlier | IQR & z-score | dilaporkan (opsional clipping di section 7) |
# | Kategori test tidak ada di train | kota baru di test | dilaporkan (encoder memetakan ke `__RARE__`) |
# | Kolom train vs test tidak cocok | kolom hanya ada di salah satu | dilaporkan, hanya kolom bersama yang dipakai |
#
# **Cara membaca output:** baris bertanda ⚠️ perlu kamu baca. Tabel lengkap tersimpan di `tables/`.
#
# **Jika X maka Y:**
#
# - Jika ada kolom yang dikonversi ke angka padahal sebenarnya kode (misal kode pos `"00123"`) → masukkan ke `CFG.CAT_COLS` dan matikan konversi (`AUTO_CONVERT_NUMERIC_STRINGS=False`) atau konversi manual.
# - Jika kolom penting ikut dibuang sebagai "ID-like" (misal nominal transaksi yang kebetulan unik) → set `AUTO_DROP_ID_LIKE=False` dan masukkan kolom ID sebenarnya ke `DROP_COLS`.
# - Jika **label konflik** banyak (> 5% baris) → data noisy; model tidak akan bisa mendekati skor sempurna. Pertimbangkan fitur tambahan atau terima batas atas skor.
# - Jika banyak kategori test tidak ada di train (> 10% baris) → gunakan `RARE_MIN_COUNT` lebih besar, frequency encoding, atau target encoding.
# - Jika ada kolom test-only → biasanya kolom bocoran/ID baru; tidak dipakai sebagai fitur.
#
# **Untuk laporan:** ringkas masalah kualitas data (missing %, kolom yang dibuang & alasannya, konversi yang dilakukan) dalam
# satu paragraf atau tabel kecil. Ini menunjukkan ke juri bahwa kamu memahami datamu.

# %%
def overview_table(df, name="train"):
    """Ringkasan per kolom: dtype, missing, unik, top frequency, contoh nilai."""
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        vc = s.value_counts(dropna=False)
        rows.append({
            "column": c, "dtype": str(s.dtype), "n_missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2), "n_unique": int(s.nunique(dropna=True)),
            "unique_pct": round(100 * s.nunique(dropna=True) / max(n, 1), 2),
            "top_freq_pct": round(100 * vc.iloc[0] / max(n, 1), 2) if len(vc) else 0,
            "examples": ", ".join(map(str, s.dropna().unique()[:4]))[:60],
        })
    out = pd.DataFrame(rows)
    save_table(out, f"dq_overview_{name}")
    return out


_MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "?", "--", "nat", "missing", "#n/a"}
_NUM_JUNK = re.compile(r"(\s|\$|€|£|¥|%|Rp\.?|IDR|USD)", flags=re.I)


def to_numeric_series(s):
    """Konversi string angka kotor → float. Return (series_float, n_gagal, contoh_gagal)."""
    st = s.astype(str).str.strip()
    is_missing_tok = st.str.lower().isin(_MISSING_TOKENS) | s.isna()
    st = st.where(~is_missing_tok, np.nan)
    st = st.str.replace(_NUM_JUNK, "", regex=True)
    st = st.str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (123) → -123 (format akuntansi)
    sample = st.dropna()
    dec_comma = len(sample) > 0 and sample.str.match(r"^-?\d+(\.\d{3})*,\d{1,2}$").mean() > 0.5
    if dec_comma:   # format Indonesia/Eropa: 1.234,50
        st = st.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        st = st.str.replace(",", "", regex=False)
    num = pd.to_numeric(st, errors="coerce")
    failed = num.isna() & ~is_missing_tok
    return num, int(failed.sum()), s[failed].astype(str).unique()[:5].tolist()


_BOOL_MAP = {"yes": 1, "no": 0, "true": 1, "false": 0, "y": 1, "n": 0, "t": 1, "f": 0, "ya": 1, "tidak": 0}


def parse_datetime_series(s):
    """Parse string → datetime, kompatibel pandas 1.x dan 2.x."""
    if version_tuple(pd.__version__) >= (2, 0, 0):
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            pass
    return pd.to_datetime(s, errors="coerce")


def looks_like_date(s, min_ratio=0.9):
    """Heuristik: apakah kolom string berisi tanggal?"""
    v = s.dropna().astype(str).str.strip()
    if len(v) == 0:
        return False
    v = v.sample(min(500, len(v)), random_state=0)
    pattern_ok = v.str.contains(r"\d", regex=True) & v.str.contains(r"[-/:.\s]", regex=True) & v.str.len().between(6, 32)
    if pattern_ok.mean() < min_ratio:
        return False
    parsed = parse_datetime_series(v)
    return parsed.notna().mean() >= min_ratio


DQ_LOG = []          # catatan semua aksi cleaning (untuk laporan)
PROTECTED = {c for c in [ID, TARGET, GROUP, TIME] if c is not None}


def dq_note(msg):
    """Catat & print aksi data-quality."""
    DQ_LOG.append(msg)
    print(msg)


banner("4.1 OVERVIEW")
dq_train_overview = overview_table(train, "train")
display(dq_train_overview)
mem_mb = train.memory_usage(deep=True).sum() / 1024 ** 2
print(f"train memory: {mem_mb:.1f} MB | missing cells: {train.isna().mean().mean():.2%}")

# %%
banner("4.2 KONVERSI TIPE (angka-teks, boolean-teks, tanggal-teks, inf)")
# 4.2.0 kolom yang diminta user dibuang lebih dulu
_drop_user = [c for c in col_list(CFG.DROP_COLS) or [] if c in train.columns and c not in PROTECTED]
if _drop_user:
    train = train.drop(columns=_drop_user)
    test = test.drop(columns=[c for c in _drop_user if c in test.columns])
    dq_note(f"🗑️ DROP_COLS dari user dibuang: {_drop_user}")

# 4.2.1 bool dtype → int
for df_ in (train, test):
    for c in df_.columns:
        if df_[c].dtype == bool:
            df_[c] = df_[c].astype("int8")

# 4.2.2 mixed types → string
for c in train.columns:
    if train[c].dtype == object:
        types = train[c].dropna().map(lambda v: type(v).__name__).value_counts()
        if len(types) > 1 and not set(types.index) <= {"str"}:
            dq_note(f"⚠️ Tipe campuran di '{c}': {dict(types)} → dipaksa jadi string")
            for df_ in (train, test):
                if c in df_.columns:
                    df_[c] = df_[c].where(df_[c].isna(), df_[c].astype(str))

# 4.2.3 angka tersimpan sebagai teks
NUMERIC_CONVERTED = []
if CFG.AUTO_CONVERT_NUMERIC_STRINGS:
    for c in [c for c in train.columns if train[c].dtype == object and c not in {ID, GROUP}]:
        num, n_fail, ex = to_numeric_series(train[c])
        n_valid = train[c].notna().sum()
        if n_valid == 0 or num.notna().sum() == 0:
            continue
        ok_ratio = 1 - n_fail / max(n_valid, 1)
        if ok_ratio >= 0.95:
            new_nan = int(num.isna().sum() - train[c].isna().sum())
            train[c] = num
            if c in test.columns:
                test[c] = to_numeric_series(test[c])[0] if test[c].dtype == object else pd.to_numeric(test[c], errors="coerce")
            NUMERIC_CONVERTED.append(c)
            dq_note(f"🔢 '{c}': teks → angka (sukses {ok_ratio:.1%}; {new_nan} nilai kosong/aneh jadi NaN; contoh gagal: {ex})")

# 4.2.4 boolean sebagai teks
BOOL_CONVERTED = []
if CFG.AUTO_CONVERT_BOOL_STRINGS:
    for c in [c for c in train.columns if train[c].dtype == object and c not in PROTECTED]:
        vals = set(train[c].dropna().astype(str).str.strip().str.lower().unique())
        if len(vals) == 2 and vals <= set(_BOOL_MAP):
            for df_ in (train, test):
                if c in df_.columns:
                    df_[c] = df_[c].astype(str).str.strip().str.lower().map(_BOOL_MAP).astype(float)
                    df_.loc[df_[c].isna(), c] = np.nan
            BOOL_CONVERTED.append(c)
    if BOOL_CONVERTED:
        dq_note(f"✅ boolean-teks → 1/0: {BOOL_CONVERTED}")

# 4.2.5 tanggal sebagai teks
DATE_DETECTED = [c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c])]
_date_force = [c for c in (col_list(CFG.DATE_COLS) or []) if c in train.columns]
if TIME is not None and TIME not in _date_force and train[TIME].dtype == object:
    _date_force.append(TIME)
for c in train.columns:
    if c in DATE_DETECTED or c in {ID, TARGET}:
        continue
    if c in _date_force or (CFG.AUTO_PARSE_DATES and CFG.DATE_COLS is None and train[c].dtype == object and looks_like_date(train[c])):
        for df_ in (train, test):
            if c in df_.columns:
                df_[c] = parse_datetime_series(df_[c])
        DATE_DETECTED.append(c)
        dq_note(f"📅 '{c}' diparse sebagai tanggal (NaT: {train[c].isna().mean():.1%})")

# 4.2.6 inf → NaN
for df_, nm in ((train, "train"), (test, "test")):
    num_cols_ = df_.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df_[num_cols_].to_numpy(dtype=float, na_value=np.nan)).sum()) if len(num_cols_) else 0
    if n_inf:
        df_[num_cols_] = df_[num_cols_].replace([np.inf, -np.inf], np.nan)
        dq_note(f"⚠️ {n_inf} nilai inf di {nm} → NaN")
print("✅ Konversi tipe selesai.")

# %%
banner("4.3 KOLOM BERMASALAH (konstan, duplikat, ID-like, high-card)")
import hashlib

feature_like = [c for c in train.columns if c not in PROTECTED]
n_rows = len(train)

# konstan & hampir konstan
CONST_COLS, QUASI_COLS = [], []
for c in feature_like:
    vc = train[c].value_counts(dropna=False, normalize=True)
    if len(vc) <= 1:
        CONST_COLS.append(c)
    elif vc.iloc[0] >= CFG.QUASI_CONSTANT_THRESH:
        QUASI_COLS.append(c)
print(f"Kolom konstan: {CONST_COLS}")
print(f"Kolom hampir konstan (>= {CFG.QUASI_CONSTANT_THRESH:.1%} satu nilai): {QUASI_COLS}")

# kolom duplikat
DUP_COLS, _seen_hash = {}, {}
for c in feature_like:
    try:
        h = hashlib.md5(pd.util.hash_pandas_object(train[c], index=False).values.tobytes()).hexdigest()
    except Exception:  # noqa: BLE001
        continue
    if h in _seen_hash:
        DUP_COLS[c] = _seen_hash[h]
    else:
        _seen_hash[h] = c
print(f"Kolom duplikat (kolom → duplikat dari): {DUP_COLS}")

# kolom mirip ID
ID_LIKE = []
_id_name = re.compile(r"(^id$|_id$|^id_|uuid|guid|^key$|_key$|^no$|nomor|^index$|unnamed)", re.I)
for c in feature_like:
    s = train[c]
    nun = s.nunique(dropna=True)
    ratio = nun / max(n_rows, 1)
    if s.dtype == object and ratio > 0.95 and n_rows > 50:
        ID_LIKE.append(c)
    elif pd.api.types.is_integer_dtype(s) and ratio >= 0.999:
        srt = np.sort(s.dropna().to_numpy())
        sequential = len(srt) > 1 and np.all(np.diff(srt) == 1)
        if sequential or _id_name.search(str(c)):
            ID_LIKE.append(c)
    elif _id_name.search(str(c)) and ratio > 0.9 and not pd.api.types.is_float_dtype(s):
        ID_LIKE.append(c)
print(f"Kolom mirip ID: {ID_LIKE}")

# high-cardinality & mixed
obj_cols = [c for c in feature_like if train[c].dtype == object or str(train[c].dtype) == "category"]
HIGH_CARD = {c: int(train[c].nunique()) for c in obj_cols if train[c].nunique() > CFG.HIGH_CARD_THRESHOLD}
print(f"Kategori high-cardinality (> {CFG.HIGH_CARD_THRESHOLD}): {HIGH_CARD}")

# terapkan drop
_to_drop = []
if CFG.AUTO_DROP_CONSTANT:
    _to_drop += CONST_COLS + QUASI_COLS
if CFG.AUTO_DROP_DUPLICATE_COLS:
    _to_drop += list(DUP_COLS)
if CFG.AUTO_DROP_ID_LIKE:
    _to_drop += [c for c in ID_LIKE if c not in (col_list(CFG.TEXT_COLS) or [])]
_to_drop = [c for c in dict.fromkeys(_to_drop) if c not in PROTECTED]
AUTO_DROPPED = _to_drop
if _to_drop:
    train = train.drop(columns=_to_drop)
    test = test.drop(columns=[c for c in _to_drop if c in test.columns])
    dq_note(f"🗑️ Dibuang otomatis: {_to_drop}")
else:
    print("✅ Tidak ada kolom yang dibuang otomatis.")

# %%
banner("4.4 BARIS DUPLIKAT, NILAI ANEH, OUTLIER, MISMATCH TRAIN-TEST")
feat_now = [c for c in train.columns if c not in PROTECTED]
dup_feat = int(train.duplicated(subset=feat_now).sum()) if feat_now else 0
dup_full = int(train.duplicated(subset=feat_now + [TARGET]).sum()) if feat_now else 0
conflict = 0
if dup_feat > 0:
    try:
        g = train[feat_now + [TARGET]].astype(str).groupby(feat_now)[TARGET].nunique()
        conflict = int((g > 1).sum())
    except Exception:  # noqa: BLE001
        conflict = -1
print(f"Baris duplikat (fitur sama): {dup_feat} | duplikat penuh (fitur+target): {dup_full} | grup label konflik: {conflict}")
if conflict > 0.05 * n_rows:
    print("⚠️ Banyak label konflik → data noisy, batas atas skor tidak bisa sempurna.")
if CFG.DROP_DUPLICATE_ROWS and dup_full > 0:
    train = train.drop_duplicates(subset=feat_now + [TARGET]).reset_index(drop=True)
    dq_note(f"🗑️ {dup_full} baris duplikat penuh dibuang")

# nilai negatif di kolom yang seharusnya positif
_pos_name = re.compile(r"(age|umur|usia|price|harga|amount|jumlah|count|qty|quantity|income|salary|gaji|charge|"
                       r"total|tenure|area|luas|distance|jarak|duration|durasi|size|weight|berat|height|tinggi)", re.I)
num_now = [c for c in feat_now if pd.api.types.is_numeric_dtype(train[c])]
NEG_SUSPECT = {c: float(train[c].min()) for c in num_now if _pos_name.search(c) and train[c].min() < 0}
print(f"Nilai negatif mencurigakan: {NEG_SUSPECT if NEG_SUSPECT else 'tidak ada ✅'}")

# outlier IQR & z-score
out_rows = []
for c in num_now:
    s = train[c].dropna().astype(float)
    if s.nunique() < 3:
        continue
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    n_iqr = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
    sd = s.std()
    n_z = int((np.abs((s - s.mean()) / sd) > 4).sum()) if sd > 0 else 0
    out_rows.append({"column": c, "iqr_outliers": n_iqr, "iqr_pct": round(100 * n_iqr / len(s), 2),
                     "z4_outliers": n_z, "min": s.min(), "max": s.max(), "skew": round(float(s.skew()), 2)})
OUTLIER_TABLE = pd.DataFrame(out_rows).sort_values("iqr_pct", ascending=False) if out_rows else pd.DataFrame()
if len(OUTLIER_TABLE):
    display(OUTLIER_TABLE.head(15))
    save_table(OUTLIER_TABLE, "dq_outliers")
    print("ℹ️ Outlier bukan berarti salah. GBDT robust terhadap outlier; model linear bisa dibantu CLIP_OUTLIERS=True.")

# mismatch kolom & kategori train vs test
if HAS_TEST:
    train_only = [c for c in train.columns if c not in test.columns and c != TARGET]
    test_only = [c for c in test.columns if c not in train.columns]
    print(f"Kolom hanya di train (selain target): {train_only if train_only else 'tidak ada ✅'}")
    print(f"Kolom hanya di test : {test_only if test_only else 'tidak ada ✅'}")
    unseen_rows = []
    for c in [c for c in feat_now if (train[c].dtype == object) and c in test.columns]:
        tr_set = set(train[c].dropna().unique())
        mask = test[c].notna() & ~test[c].isin(tr_set)
        if mask.any():
            unseen_rows.append({"column": c, "test_rows_unseen_pct": round(100 * mask.mean(), 2),
                                "n_unseen_values": int(test.loc[mask, c].nunique()),
                                "examples": ", ".join(map(str, test.loc[mask, c].unique()[:4]))})
    UNSEEN_TABLE = pd.DataFrame(unseen_rows)
    if len(UNSEEN_TABLE):
        display(UNSEEN_TABLE.sort_values("test_rows_unseen_pct", ascending=False))
        save_table(UNSEEN_TABLE, "dq_unseen_categories")
        if UNSEEN_TABLE["test_rows_unseen_pct"].max() > 10:
            print("⚠️ Ada kolom dengan >10% baris test berisi kategori baru → andalkan frequency/target encoding, naikkan RARE_MIN_COUNT.")
    else:
        print("✅ Tidak ada kategori baru di test.")
    # pastikan test punya semua kolom fitur train (isi NaN jika tidak ada)
    for c in train_only:
        test[c] = np.nan
        print(f"⚠️ kolom '{c}' tidak ada di test → diisi NaN di test (pertimbangkan DROP_COLS)")
    if test_only:
        test = test.drop(columns=test_only)


def reduce_mem_usage(df, verbose=True, name="df"):
    """Downcast tipe numerik (int → int terkecil, float64 → float32) untuk hemat RAM."""
    start = df.memory_usage(deep=True).sum() / 1024 ** 2
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_integer_dtype(s) and s.notna().all():
            mn, mx = s.min(), s.max()
            for t in (np.int8, np.int16, np.int32):
                if np.iinfo(t).min <= mn and mx <= np.iinfo(t).max:
                    df[c] = s.astype(t)
                    break
        elif pd.api.types.is_float_dtype(s) and s.dtype != np.float32:
            df[c] = s.astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print(f"reduce_mem_usage({name}): {start:.1f} MB → {end:.1f} MB ({100 * (start - end) / max(start, 1e-9):.0f}% hemat)")
    return df


if CFG.REDUCE_MEMORY:
    keep_cols = [c for c in train.columns if c not in {ID, TARGET}]
    train[keep_cols] = reduce_mem_usage(train[keep_cols].copy(), name="train")
    test_keep = [c for c in keep_cols if c in test.columns]
    test[test_keep] = reduce_mem_usage(test[test_keep].copy(), name="test")

DQ_SUMMARY = {
    "numeric_converted": NUMERIC_CONVERTED, "bool_converted": BOOL_CONVERTED, "date_cols": DATE_DETECTED,
    "constant": CONST_COLS, "quasi_constant": QUASI_COLS, "duplicate_cols": DUP_COLS, "id_like": ID_LIKE,
    "auto_dropped": AUTO_DROPPED, "high_card": HIGH_CARD, "dup_rows": dup_feat, "label_conflicts": conflict,
    "neg_suspect": NEG_SUSPECT,
}
with open(RUN.out / "tables" / "dq_summary.json", "w", encoding="utf-8") as f:
    json.dump(DQ_SUMMARY, f, indent=2, default=str)
print(f"\nShape setelah cleaning → train {train.shape} | test {test.shape}")

# %% [markdown]
# ---
# # 5. Analisis Target
#
# **Tujuan:** menentukan jenis task, meng-encode label, dan memahami distribusi target — ini menentukan metric,
# skema CV, dan strategi imbalance.
#
# **Logika deteksi task (`CFG.TASK="auto"`):**
#
# 1. Jika `METRIC` jelas regresi (`rmse`, `mae`, `rmsle`, `mape`, `r2`) → regression. Jika jelas klasifikasi (`roc_auc`, `f1`, `logloss`, ...) → klasifikasi.
# 2. Jika target berupa teks/boolean → 2 nilai = binary, > 2 nilai = multiclass.
# 3. Jika target numerik: 2 nilai unik → binary; integer dengan ≤ 20 nilai unik → multiclass (⚠️ dicetak peringatan, karena bisa juga regresi ordinal); selain itu → regression.
#
# **Encoding label:** label asli (misal `"Yes"/"No"`, `"bronze"/"gold"`) diubah ke 0..K-1. Mapping disimpan ke
# `tables/label_mapping.json` dan dipakai lagi untuk mengembalikan label asli di submission. Untuk binary, kelas positif
# (index 1) dipilih otomatis: `1`, `Yes`, `True`, `Y`, atau kelas minoritas — bisa dipaksa via `CFG.POS_LABEL`.
#
# **Cara membaca output:**
#
# - Klasifikasi: tabel jumlah & proporsi per kelas, dan **imbalance ratio** (kelas terbanyak / tersedikit).
# - Regresi: statistik, skewness, histogram target asli vs `log1p(target)`, dan jumlah outlier target.
#
# **Jika X maka Y:**
#
# - Jika imbalance ratio > 3 dan metric berbasis label (F1, accuracy, balanced accuracy) → class weight membantu
#   (`IMBALANCE_STRATEGY="auto"` otomatis memakai `class_weight`). Threshold tuning (section 14) biasanya lebih ampuh daripada SMOTE.
# - Jika imbalance ratio > 3 tapi metric AUC/logloss → **jangan** pakai class weight/SMOTE (mengacaukan kalibrasi probabilitas, AUC tidak terbantu).
# - Jika regresi dan skew > 1 dan target ≥ 0 → coba `TARGET_TRANSFORM="log1p"` (bandingkan skor CV). Untuk metric RMSLE ini otomatis.
# - Jika ada target NaN → baris tersebut dibuang (dicetak jumlahnya).
# - Jika ada kelas dengan < `N_FOLDS` sampel → jumlah fold dikurangi otomatis atau gabungkan kelas langka.
#
# **Untuk laporan:** tampilkan figur distribusi target dan sebutkan imbalance ratio / skewness serta strategi yang diambil.

# %%
banner("5. ANALISIS TARGET")
n_na_target = int(train[TARGET].isna().sum())
if n_na_target:
    train = train[train[TARGET].notna()].reset_index(drop=True)
    print(f"⚠️ {n_na_target} baris dengan target NaN dibuang.")

y_raw = train[TARGET]
_metric_cfg = RUN.metric


def detect_task(y, metric):
    """Tebak task dari metric dan nilai target."""
    if metric in REG_METRICS:
        return "regression"
    nun = y.nunique()
    is_num = pd.api.types.is_numeric_dtype(y)
    if metric in PROB_METRICS or metric in (LABEL_METRICS - {"qwk", "cohen_kappa", "accuracy"}):
        return "binary" if nun == 2 else "multiclass"
    if not is_num:
        return "binary" if nun == 2 else "multiclass"
    if nun == 2:
        return "binary"
    yv = y.dropna().to_numpy(dtype=float)
    if np.allclose(yv, np.round(yv)) and nun <= 20:
        print(f"⚠️ Target numerik integer dengan {nun} nilai unik → dianggap multiclass. "
              f"Jika ini sebenarnya nilai kontinu/ordinal, set CFG.TASK='regression'.")
        return "multiclass"
    return "regression"


TASK = CFG.TASK if CFG.TASK in ("binary", "multiclass", "regression") else detect_task(y_raw, _metric_cfg)
if RUN.metric == "auto":
    RUN.metric = {"binary": "roc_auc", "multiclass": "f1_macro", "regression": "rmse"}[TASK]
    print(f"ℹ️ METRIC='auto' → memakai '{RUN.metric}'. ⚠️ Pastikan sesuai aturan lomba!")
if TASK == "regression" and RUN.metric not in REG_METRICS | {"qwk", "cohen_kappa", "accuracy"}:
    raise ValueError(f"Metric '{RUN.metric}' tidak cocok untuk regresi.")
if TASK != "regression" and RUN.metric in REG_METRICS:
    raise ValueError(f"Metric '{RUN.metric}' adalah metric regresi, tapi task = {TASK}.")
if RUN.metric not in GREATER_IS_BETTER:
    raise ValueError(f"Metric '{RUN.metric}' tidak didukung. Pilihan: {sorted(GREATER_IS_BETTER)}")

# ---- encoding label ----
LABEL2ID, ID2LABEL, CLASS_LABELS = {}, {}, []
if TASK in ("binary", "multiclass"):
    uniq = list(pd.Series(y_raw.unique()).dropna())
    if CFG.CLASS_ORDER is not None:
        order = [c for c in CFG.CLASS_ORDER if c in uniq] + [c for c in uniq if c not in CFG.CLASS_ORDER]
    else:
        try:
            order = sorted(uniq)
        except TypeError:
            order = sorted(uniq, key=str)
    if TASK == "binary":
        pos = CFG.POS_LABEL
        if pos is None:
            lower = {str(v).strip().lower(): v for v in order}
            for cand in ["1", "1.0", "yes", "true", "y", "t", "ya", "positive", "pos"]:
                if cand in lower:
                    pos = lower[cand]
                    break
            if pos is None:
                pos = y_raw.value_counts().index[-1]   # kelas minoritas
        order = [v for v in order if v != pos] + [pos]
    CLASS_LABELS = order
    LABEL2ID = {v: i for i, v in enumerate(order)}
    ID2LABEL = {i: v for v, i in LABEL2ID.items()}
    y = y_raw.map(LABEL2ID).to_numpy().astype(int)
    N_CLASSES = len(order)
    if TASK == "binary" and N_CLASSES != 2:
        raise ValueError(f"TASK=binary tapi target punya {N_CLASSES} kelas: {order}")
else:
    y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
    N_CLASSES = 1
    if np.isnan(y).any():
        raise ValueError("Target regresi mengandung nilai non-numerik.")

with open(RUN.out / "tables" / "label_mapping.json", "w", encoding="utf-8") as f:
    json.dump({"task": TASK, "metric": RUN.metric, "label2id": {str(k): v for k, v in LABEL2ID.items()}}, f, indent=2)

if CFG.SUBMISSION_MODE == "auto":
    RUN.sub_mode = "value" if TASK == "regression" else ("probability" if RUN.metric in PROB_METRICS else "label")
else:
    RUN.sub_mode = "value" if TASK == "regression" else CFG.SUBMISSION_MODE
print(f"TASK = {TASK} | METRIC = {RUN.metric} ({'higher' if greater_is_better() else 'lower'} is better) | "
      f"SUBMISSION_MODE = {RUN.sub_mode}")
if TASK != "regression":
    print("Label mapping:", LABEL2ID)

# %%
IMBALANCE_RATIO = 1.0
TARGET_SKEW = 0.0
with safe_block("target analysis plots"):
    if TASK in ("binary", "multiclass"):
        vc = pd.Series(y).map(ID2LABEL).value_counts()
        tbl = pd.DataFrame({"count": vc, "pct": (100 * vc / vc.sum()).round(2)})
        display(tbl)
        save_table(tbl.reset_index().rename(columns={"index": "class"}), "target_distribution")
        IMBALANCE_RATIO = float(vc.max() / max(vc.min(), 1))
        print(f"Imbalance ratio (terbanyak/tersedikit) = {IMBALANCE_RATIO:.2f}")
        if IMBALANCE_RATIO > 10:
            print("⚠️ Sangat imbalanced. Gunakan stratified CV, perhatikan metric (PR-AUC/F1), tuning threshold.")
        elif IMBALANCE_RATIO > 3:
            print("⚠️ Cukup imbalanced. Class weight / threshold tuning berguna untuk metric berbasis label.")
        else:
            print("✅ Kelas relatif seimbang.")
        if vc.min() < RUN.n_folds:
            print(f"⚠️ Kelas terkecil hanya {vc.min()} sampel (< N_FOLDS={RUN.n_folds}). Fold akan dikurangi otomatis.")
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(vc) + 3), 4))
        bars = ax.bar(vc.index.astype(str), vc.values, color=plt.cm.tab10.colors[: len(vc)])
        for b, p in zip(bars, tbl["pct"].values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{p:.1f}%", ha="center", va="bottom")
        ax.set_title(f"Target distribution: {TARGET_ORIG}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        save_fig("target_distribution")
    else:
        s = pd.Series(y)
        TARGET_SKEW = float(s.skew())
        desc = s.describe()
        display(desc.to_frame("target").T)
        q1, q3 = s.quantile([0.25, 0.75])
        n_out = int(((s < q1 - 3 * (q3 - q1)) | (s > q3 + 3 * (q3 - q1))).sum())
        print(f"Skewness = {TARGET_SKEW:.2f} | kurtosis = {s.kurt():.2f} | min = {s.min():.4g} | "
              f"integer? {np.allclose(s, np.round(s))} | outlier ekstrem (3*IQR) = {n_out}")
        if abs(TARGET_SKEW) > 1 and s.min() >= 0:
            print("⚠️ Target skew → coba TARGET_TRANSFORM='log1p' dan bandingkan CV.")
        if n_out > 0.01 * len(s):
            print("⚠️ Banyak outlier target → pertimbangkan metric MAE / objective huber, atau periksa data error.")
        fig, axes = plt.subplots(1, 2 if s.min() >= 0 else 1, figsize=(12, 4), squeeze=False)
        axes[0, 0].hist(s, bins=60, color="steelblue", edgecolor="white")
        axes[0, 0].set_title(f"Target distribution: {TARGET_ORIG}")
        axes[0, 0].set_xlabel(TARGET_ORIG)
        axes[0, 0].set_ylabel("Count")
        if s.min() >= 0:
            axes[0, 1].hist(np.log1p(s), bins=60, color="darkorange", edgecolor="white")
            axes[0, 1].set_title(f"log1p({TARGET_ORIG}) — skew {np.log1p(s).skew():.2f}")
            axes[0, 1].set_xlabel(f"log1p({TARGET_ORIG})")
            axes[0, 1].set_ylabel("Count")
        save_fig("target_distribution")

# transformasi target (regresi)
if TASK == "regression":
    tt = CFG.TARGET_TRANSFORM
    if tt == "auto":
        tt = "log1p" if (RUN.metric == "rmsle" and y.min() >= 0) else None
    if tt == "log1p" and y.min() < 0:
        print("⚠️ log1p tidak bisa untuk target negatif → transform dimatikan.")
        tt = None
    RUN.target_transform = tt
    print(f"TARGET_TRANSFORM aktif: {RUN.target_transform}")
else:
    RUN.target_transform = None


def y_forward(v):
    """Transformasi target sebelum training (regresi)."""
    return np.log1p(v) if RUN.target_transform == "log1p" else v


def y_inverse(v):
    """Kembalikan prediksi ke skala asli."""
    return np.expm1(v) if RUN.target_transform == "log1p" else v


# imbalance strategy
if TASK == "regression":
    RUN.imbalance = None
elif CFG.IMBALANCE_STRATEGY == "auto":
    RUN.imbalance = "class_weight" if (IMBALANCE_RATIO > 3 and RUN.metric in LABEL_METRICS) else None
else:
    RUN.imbalance = CFG.IMBALANCE_STRATEGY
if RUN.imbalance == "smote" and not HAS_IMBLEARN:
    print("⚠️ SMOTE diminta tapi imbalanced-learn tidak ada → pakai class_weight.")
    RUN.imbalance = "class_weight"
print(f"Strategi imbalance aktif: {RUN.imbalance}")

# %% [markdown]
# ### 5.1 Fast mode: subsample train
#
# Di `RUN_MODE="fast"`, train di-subsample (stratified untuk klasifikasi) sebesar `SAMPLE_FRAC` (maks `FAST_MAX_ROWS`)
# supaya seluruh notebook bisa dicek dengan cepat. **Skor di fast mode lebih rendah & lebih noisy** — jangan dipakai
# untuk keputusan final. Jika ada `GROUP_COL`, subsample dilakukan per grup (agar grup tidak terpotong).

# %%
if RUN.fast:
    n_target = int(min(len(train) * CFG.SAMPLE_FRAC, CFG.FAST_MAX_ROWS))
    n_target = max(n_target, min(len(train), 300))
    if n_target < len(train):
        rng = np.random.RandomState(CFG.SEED)
        if GROUP is not None:
            groups_u = train[GROUP].unique()
            keep_g = rng.choice(groups_u, size=max(2, int(len(groups_u) * n_target / len(train))), replace=False)
            idx = np.where(train[GROUP].isin(keep_g))[0]
        elif TASK != "regression":
            idx = (pd.Series(np.arange(len(train))).groupby(y)
                   .apply(lambda s: s.sample(max(min(len(s), RUN.n_folds * 2), int(round(len(s) * n_target / len(train)))),
                                             random_state=CFG.SEED)).to_numpy())
        else:
            idx = rng.choice(len(train), size=n_target, replace=False)
        idx = np.sort(idx)
        train = train.iloc[idx].reset_index(drop=True)
        y = y[idx]
        print(f"⚡ FAST MODE: train di-subsample menjadi {len(train)} baris.")
else:
    print("FULL MODE: memakai seluruh train.")

# %% [markdown]
# ---
# # 6. EDA Visual Otomatis
#
# **Tujuan:** memahami data secara visual, menemukan fitur penting, mendeteksi **leakage** dan **drift** train-test.
# EDA tidak mengubah data (kecuali kamu memutuskan sesuatu berdasarkan hasilnya) — jadi kalau waktu mepet,
# `CFG.DO_EDA=False`. Tapi untuk laporan, section ini adalah sumber figur utama.
#
# Jumlah plot dibatasi `CFG.EDA_MAX_PLOTS` dan baris di-subsample `CFG.EDA_SAMPLE_ROWS` supaya aman untuk data lebar/besar.
#
# **Subsection:**
#
# | No | Analisis | Pertanyaan yang dijawab |
# |---|---|---|
# | 6.0 | Tipe fitur | kolom mana numerik / kategorikal / tanggal / teks? |
# | 6.1 | Missing values | kolom mana yang banyak kosong? polanya acak atau berkelompok? |
# | 6.2 | Distribusi numerik train vs test | apakah distribusi test sama dengan train? |
# | 6.3 | Distribusi kategorikal | kategori dominan? kategori langka? |
# | 6.4 | Target vs fitur | fitur mana yang membedakan target? (rate per kategori, rate per bin numerik) |
# | 6.5 | Korelasi Spearman | fitur redundan (korelasi > 0.95)? |
# | 6.6 | Cramér's V | asosiasi antar kategori & dengan target |
# | 6.7 | Mutual information | ranking kekuatan hubungan non-linear fitur → target |
# | 6.8 | Single-feature CV score | prediktif sendirian? **leakage detector** |
# | 6.9 | Drift: KS, PSI, adversarial validation | apakah test berbeda dari train? fitur mana penyebabnya? |
#
# **Untuk laporan:** pilih 3–5 figur paling informatif: distribusi target, target-vs-fitur untuk 2–3 fitur teratas,
# korelasi, dan (jika relevan) adversarial validation. Tulis insight bisnis, bukan hanya angka (misal: "pelanggan kontrak
# bulanan churn 3x lebih sering").

# %%
def infer_feature_types(df):
    """Tentukan kolom numerik / kategorikal / tanggal / teks dari df (setelah cleaning)."""
    exclude = {c for c in [ID, TARGET, GROUP] if c is not None}
    feats = [c for c in df.columns if c not in exclude]
    date_cols = [c for c in feats if pd.api.types.is_datetime64_any_dtype(df[c])]
    if CFG.TEXT_COLS is not None:
        text_cols = [c for c in col_list(CFG.TEXT_COLS) if c in feats]
    else:
        text_cols = []
        for c in feats:
            if df[c].dtype == object and df[c].nunique() > 50:
                sample = df[c].dropna().astype(str).head(2000)
                if len(sample) and sample.str.split().str.len().mean() >= 3:
                    text_cols.append(c)
    rest = [c for c in feats if c not in date_cols and c not in text_cols]
    if CFG.CAT_COLS is not None:
        cat_cols = [c for c in col_list(CFG.CAT_COLS) if c in rest]
    else:
        cat_cols = [c for c in rest if df[c].dtype == object or str(df[c].dtype) in ("category", "string")]
        if CFG.NUM_AS_CAT_MAX_UNIQUE > 0:
            cat_cols += [c for c in rest if c not in cat_cols and pd.api.types.is_integer_dtype(df[c])
                         and df[c].nunique() <= CFG.NUM_AS_CAT_MAX_UNIQUE]
    if CFG.NUM_COLS is not None:
        num_cols = [c for c in col_list(CFG.NUM_COLS) if c in rest and c not in cat_cols]
    else:
        num_cols = [c for c in rest if c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])]
    leftover = [c for c in rest if c not in cat_cols and c not in num_cols]
    cat_cols += leftover   # tipe tak dikenal → diperlakukan sebagai kategori
    return num_cols, cat_cols, date_cols, text_cols


RAW_NUM, RAW_CAT, RAW_DATE, RAW_TEXT = infer_feature_types(train)
# kolom kategori diseragamkan menjadi string (NaN tetap NaN)
for c in RAW_CAT:
    for df_ in (train, test):
        if c in df_.columns:
            df_[c] = df_[c].astype(object).map(lambda v: v if isinstance(v, str) else (np.nan if pd.isna(v) else str(v)))
print(f"Numerik ({len(RAW_NUM)}): {RAW_NUM[:20]}{' ...' if len(RAW_NUM) > 20 else ''}")
print(f"Kategorikal ({len(RAW_CAT)}): {RAW_CAT[:20]}{' ...' if len(RAW_CAT) > 20 else ''}")
print(f"Tanggal ({len(RAW_DATE)}): {RAW_DATE}")
print(f"Teks ({len(RAW_TEXT)}): {RAW_TEXT}")

# Sample untuk EDA
_eda_n = min(len(train), CFG.EDA_SAMPLE_ROWS)
_eda_idx = np.sort(np.random.RandomState(0).choice(len(train), _eda_n, replace=False))
EDA_DF = train.iloc[_eda_idx].reset_index(drop=True)
EDA_Y = y[_eda_idx]
EDA_TEST = test.sample(min(len(test), CFG.EDA_SAMPLE_ROWS), random_state=0).reset_index(drop=True) if HAS_TEST else None


def quick_encode(df, ref=None, num_cols=None, cat_cols=None, date_cols=None):
    """Encoding cepat untuk analisis (bukan untuk model final): kategori → kode, tanggal → hari."""
    num_cols = RAW_NUM if num_cols is None else num_cols
    cat_cols = RAW_CAT if cat_cols is None else cat_cols
    date_cols = RAW_DATE if date_cols is None else date_cols
    out = pd.DataFrame(index=df.index)
    for c in num_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cat_cols:
        cats = pd.concat([df[c], ref[c]]).astype(str).unique() if ref is not None and c in ref else df[c].astype(str).unique()
        out[c] = pd.Categorical(df[c].astype(str), categories=cats).codes.astype(float)
    for c in date_cols:
        out[c] = (df[c] - pd.Timestamp("1970-01-01")).dt.days.astype(float)
    return out


def _grid(n, ncols=3, w=5.2, h=3.6):
    """Buat grid subplot untuk n plot."""
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for k in range(n, nrows * ncols):
        axes.flat[k].axis("off")
    return fig, axes.flat


MI_RANK = pd.Series(dtype=float)     # diisi di 6.7 (dipakai FE section 8)
LEAK_SUSPECTS = []
DRIFT_FEATURES = []

# %% [markdown]
# ### 6.1 Missing values
#
# **Cara membaca:** bar = % missing per kolom (train vs test). Heatmap = pola missing per baris (garis putih/berwarna
# sejajar di beberapa kolom = missing berkelompok, biasanya karena sumber data sama).
#
# **Jika X maka Y:**
#
# - Missing > 90% → kolom hampir kosong; biasanya buang, **kecuali** indikator missing-nya prediktif (cek 6.4).
# - % missing train sangat beda dengan test → drift / proses pengumpulan berbeda; hati-hati.
# - Missing berkelompok → buat fitur `n_missing` per baris (`FE_ROW_STATS=True` sudah melakukannya).
# - GBDT (LightGBM/XGBoost/CatBoost/HGB) menangani NaN secara native; model linear memakai imputasi + indikator.

# %%
if CFG.DO_EDA:
    with safe_block("6.1 missing"):
        feats_all = RAW_NUM + RAW_CAT + RAW_DATE + RAW_TEXT
        miss_tr = train[feats_all].isna().mean() * 100
        miss_te = test[[c for c in feats_all if c in test.columns]].isna().mean() * 100 if HAS_TEST else None
        miss_df = pd.DataFrame({"train_pct": miss_tr, "test_pct": miss_te}).fillna(0) if HAS_TEST else miss_tr.to_frame("train_pct")
        miss_df = miss_df[miss_df.max(axis=1) > 0].sort_values("train_pct", ascending=False)
        if len(miss_df) == 0:
            print("✅ Tidak ada missing value di fitur.")
        else:
            display(miss_df.round(2).head(30))
            save_table(miss_df.reset_index().rename(columns={"index": "feature"}), "eda_missing")
            top = miss_df.head(30)
            fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(top) + 1)))
            yy = np.arange(len(top))
            ax.barh(yy - 0.2, top["train_pct"], height=0.4, label="train")
            if HAS_TEST:
                ax.barh(yy + 0.2, top["test_pct"], height=0.4, label="test")
            ax.set_yticks(yy)
            ax.set_yticklabels(top.index)
            ax.invert_yaxis()
            ax.set_xlabel("Missing (%)")
            ax.set_title("Missing values per feature")
            ax.legend()
            save_fig("eda_missing_bar")
            cols_hm = list(top.index)
            samp = train[cols_hm].sample(min(len(train), 400), random_state=0).isna()
            fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(cols_hm) + 2), 5))
            ax.imshow(samp.to_numpy().astype(int), aspect="auto", cmap="Greys", interpolation="nearest")
            ax.set_xticks(range(len(cols_hm)))
            ax.set_xticklabels(cols_hm, rotation=90)
            ax.set_ylabel("Sampled rows")
            ax.set_title("Missing pattern (black = missing)")
            ax.grid(False)
            save_fig("eda_missing_heatmap")

# %% [markdown]
# ### 6.2 Distribusi numerik (train vs test) & 6.3 distribusi kategorikal
#
# **Cara membaca:** histogram density train (biru) vs test (oranye). Jika bentuknya menempel → aman. Jika bergeser →
# drift (dicek kuantitatif di 6.9). Plot dipotong di persentil 1–99 agar outlier tidak merusak skala.
#
# **Jika X maka Y:**
#
# - Distribusi sangat skew (ekor panjang ke kanan) → untuk model linear aktifkan `LOG_TRANSFORM_LINEAR` (default on). GBDT tidak terpengaruh.
# - Test punya range di luar train → model tree tidak bisa ekstrapolasi; pertimbangkan fitur rasio/relatif.
# - Kategori dominan > 95% → informasi kecil; kategori langka banyak → `RARE_MIN_COUNT`.

# %%
if CFG.DO_EDA:
    with safe_block("6.2 numeric distributions"):
        cols = RAW_NUM[: CFG.EDA_MAX_PLOTS]
        if cols:
            fig, axes = _grid(len(cols))
            for ax, c in zip(axes, cols):
                s = EDA_DF[c].dropna().astype(float)
                lo, hi = s.quantile([0.01, 0.99]) if len(s) else (0, 1)
                bins = np.linspace(lo, hi, 40) if hi > lo else 20
                ax.hist(s.clip(lo, hi), bins=bins, alpha=0.55, density=True, label="train")
                if HAS_TEST and c in EDA_TEST:
                    ax.hist(EDA_TEST[c].dropna().astype(float).clip(lo, hi), bins=bins, alpha=0.45, density=True, label="test")
                ax.set_title(c)
                ax.set_xlabel(c)
                ax.set_ylabel("Density")
                ax.legend(fontsize=8)
            fig.suptitle("Numeric feature distributions (train vs test)", y=1.01)
            fig.tight_layout()
            save_fig("eda_numeric_distributions")
        else:
            print("Tidak ada fitur numerik.")
    with safe_block("6.3 categorical distributions"):
        cols = RAW_CAT[: CFG.EDA_MAX_PLOTS]
        if cols:
            fig, axes = _grid(len(cols), w=5.5, h=4)
            for ax, c in zip(axes, cols):
                vc_tr = EDA_DF[c].fillna("NaN").astype(str).value_counts(normalize=True).head(12)
                ax.barh(vc_tr.index[::-1], vc_tr.values[::-1], alpha=0.7, label="train")
                if HAS_TEST and c in EDA_TEST:
                    vc_te = EDA_TEST[c].fillna("NaN").astype(str).value_counts(normalize=True).reindex(vc_tr.index).fillna(0)
                    ax.plot(vc_te.values[::-1], np.arange(len(vc_te)), "o", color="darkorange", label="test")
                ax.set_title(f"{c} (n_unique={train[c].nunique()})")
                ax.set_xlabel("Proportion")
                ax.legend(fontsize=8)
            fig.suptitle("Categorical feature distributions (top categories)", y=1.01)
            fig.tight_layout()
            save_fig("eda_categorical_distributions")
        else:
            print("Tidak ada fitur kategorikal.")

# %% [markdown]
# ### 6.4 Target vs fitur
#
# **Cara membaca:**
#
# - Klasifikasi binary: bar = **target rate** (proporsi kelas positif) per kategori / per bin numerik; garis putus-putus =
#   rate global; angka `n=` = jumlah baris (rate dari kategori dengan n kecil tidak bisa dipercaya).
# - Multiclass: proporsi kelas (stacked bar) per kategori/bin.
# - Regresi: rata-rata target per kategori/bin.
#
# **Jika X maka Y:**
#
# - Rate sangat berbeda antar kategori dan n besar → fitur kuat, sebut di laporan sebagai insight.
# - Hubungan numerik non-monoton (naik lalu turun) → model linear kesulitan; GBDT/binning menangkapnya.
# - Kategori dengan rate 0% atau 100% dan n besar → curigai leakage (cek 6.8).

# %%
def _target_stat_by(series, yv):
    """Hitung statistik target per grup (rate / mean / proporsi kelas)."""
    df_ = pd.DataFrame({"g": series.astype(str).values, "y": yv})
    if TASK == "multiclass":
        tab = pd.crosstab(df_["g"], df_["y"], normalize="index")
        tab.columns = [str(ID2LABEL[c]) for c in tab.columns]
        return tab, df_["g"].value_counts()
    agg = df_.groupby("g")["y"].agg(["mean", "count"])
    return agg["mean"], agg["count"]


if CFG.DO_EDA:
    with safe_block("6.4 target vs categorical"):
        cols = RAW_CAT[: CFG.EDA_MAX_PLOTS]
        if cols:
            fig, axes = _grid(len(cols), w=5.5, h=4)
            for ax, c in zip(axes, cols):
                s = EDA_DF[c].fillna("NaN")
                top_cats = s.astype(str).value_counts().head(12).index
                s = s.astype(str).where(s.astype(str).isin(top_cats), "__other__")
                stat, cnt = _target_stat_by(s, EDA_Y)
                if TASK == "multiclass":
                    stat.loc[cnt.index].plot(kind="barh", stacked=True, ax=ax, legend=False, colormap="tab10")
                    ax.set_xlabel("Class proportion")
                else:
                    stat = stat.loc[cnt.sort_values(ascending=False).index]
                    ax.barh(stat.index[::-1].astype(str), stat.values[::-1], color="teal", alpha=0.8)
                    for k, (cat, val) in enumerate(zip(stat.index[::-1], stat.values[::-1])):
                        ax.text(val, k, f" n={cnt[cat]}", va="center", fontsize=7)
                    ax.axvline(np.mean(EDA_Y), ls="--", color="red", lw=1)
                    ax.set_xlabel("Target rate" if TASK == "binary" else f"Mean {TARGET_ORIG}")
                ax.set_title(c)
            if TASK == "multiclass":
                axes[0].legend(fontsize=7, loc="lower right")
            fig.suptitle(f"Target vs categorical features ({TARGET_ORIG})", y=1.01)
            fig.tight_layout()
            save_fig("eda_target_vs_categorical")
    with safe_block("6.4 target vs numeric"):
        cols = RAW_NUM[: CFG.EDA_MAX_PLOTS]
        if cols:
            fig, axes = _grid(len(cols))
            for ax, c in zip(axes, cols):
                s = EDA_DF[c].astype(float)
                try:
                    b = pd.qcut(s, q=10, duplicates="drop")
                except Exception:  # noqa: BLE001
                    b = pd.cut(s, bins=10)
                b = b.astype(str).where(s.notna(), "NaN")
                stat, cnt = _target_stat_by(b, EDA_Y)
                order = [k for k in sorted(cnt.index, key=lambda v: (v == "NaN", float(re.findall(r"-?[\d.]+(?:e-?\d+)?", v)[0]) if re.findall(r"-?[\d.]+(?:e-?\d+)?", v) else 0))]
                if TASK == "multiclass":
                    stat.loc[order].plot(kind="bar", stacked=True, ax=ax, legend=False, colormap="tab10", width=0.9)
                    ax.set_ylabel("Class proportion")
                else:
                    ax.plot(range(len(order)), stat.loc[order].values, "o-", color="teal")
                    ax.axhline(np.mean(EDA_Y), ls="--", color="red", lw=1)
                    ax.set_ylabel("Target rate" if TASK == "binary" else f"Mean {TARGET_ORIG}")
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels([o[:14] for o in order], rotation=60, fontsize=6)
                ax.set_title(c)
                ax.set_xlabel("Quantile bin")
            fig.suptitle(f"Target vs binned numeric features ({TARGET_ORIG})", y=1.01)
            fig.tight_layout()
            save_fig("eda_target_vs_numeric")

# %% [markdown]
# ### 6.5 Korelasi Spearman & 6.6 Cramér's V
#
# **Spearman** (berbasis ranking, tahan outlier, menangkap hubungan monoton). **Cramér's V** (0–1) = kekuatan asosiasi
# antar dua variabel kategorikal.
#
# **Jika X maka Y:**
#
# - Pasangan |rho| > 0.95 → fitur redundan. GBDT tidak terganggu, tapi interpretasi importance terbagi dua; model linear
#   jadi tidak stabil. Pertimbangkan buang salah satu (`DROP_COLS`).
# - Korelasi fitur dengan target sangat tinggi (> 0.9) → curigai leakage.
# - Cramér's V antar kategori ≈ 1 → kategori turunan dari kategori lain (misal `kota` dan `kode_kota`) → buang salah satu.

# %%
def cramers_v(a, b):
    """Cramér's V dengan koreksi bias (Bergsma 2013)."""
    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rc, kc = r - (r - 1) ** 2 / max(n - 1, 1), k - (k - 1) ** 2 / max(n - 1, 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else 0.0


HIGH_CORR_PAIRS = pd.DataFrame()
if CFG.DO_EDA:
    with safe_block("6.5 spearman"):
        cols = RAW_NUM[:30]
        if len(cols) >= 2:
            cdf = EDA_DF[cols].astype(float).copy()
            if TASK != "multiclass":
                cdf["__target__"] = EDA_Y
            corr = cdf.corr(method="spearman")
            fig, ax = plt.subplots(figsize=(min(18, 0.45 * len(corr) + 4), min(16, 0.45 * len(corr) + 3)))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(corr)))
            ax.set_yticklabels(corr.columns, fontsize=7)
            if len(corr) <= 20:
                for i in range(len(corr)):
                    for j in range(len(corr)):
                        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("Spearman correlation (numeric features)")
            ax.grid(False)
            save_fig("eda_spearman_corr")
            pairs = []
            cc = corr.drop(index="__target__", columns="__target__", errors="ignore")
            for i, a in enumerate(cc.columns):
                for b in cc.columns[i + 1:]:
                    if abs(cc.loc[a, b]) > 0.95:
                        pairs.append({"feature_a": a, "feature_b": b, "spearman": round(cc.loc[a, b], 4)})
            HIGH_CORR_PAIRS = pd.DataFrame(pairs)
            if len(HIGH_CORR_PAIRS):
                print("⚠️ Pasangan fitur sangat berkorelasi (|rho| > 0.95):")
                display(HIGH_CORR_PAIRS)
                save_table(HIGH_CORR_PAIRS, "eda_high_corr_pairs")
            else:
                print("✅ Tidak ada pasangan numerik dengan |rho| > 0.95.")
            if "__target__" in corr:
                tcorr = corr["__target__"].drop("__target__").abs().sort_values(ascending=False)
                print("Top |Spearman| dengan target:", tcorr.head(8).round(3).to_dict())
    with safe_block("6.6 cramers v"):
        cols = [c for c in RAW_CAT if EDA_DF[c].nunique() <= 200][:20]
        if len(cols) >= 1:
            allc = cols + (["__target__"] if TASK != "regression" else [])
            dfc = EDA_DF[cols].fillna("NaN").astype(str).copy()
            if TASK != "regression":
                dfc["__target__"] = EDA_Y
            mat = pd.DataFrame(np.eye(len(allc)), index=allc, columns=allc)
            for i, a in enumerate(allc):
                for b in allc[i + 1:]:
                    v = cramers_v(dfc[a], dfc[b])
                    mat.loc[a, b] = mat.loc[b, a] = v
            fig, ax = plt.subplots(figsize=(min(16, 0.5 * len(allc) + 4), min(14, 0.5 * len(allc) + 3)))
            im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(allc)))
            ax.set_xticklabels(allc, rotation=90, fontsize=7)
            ax.set_yticks(range(len(allc)))
            ax.set_yticklabels(allc, fontsize=7)
            if len(allc) <= 16:
                for i in range(len(allc)):
                    for j in range(len(allc)):
                        ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("Cramér's V (categorical association)")
            ax.grid(False)
            save_fig("eda_cramers_v")
            if "__target__" in mat:
                print("Cramér's V dengan target:", mat["__target__"].drop("__target__").sort_values(ascending=False).head(8).round(3).to_dict())

# %% [markdown]
# ### 6.7 Mutual Information & 6.8 Single-feature CV score (leakage detector)
#
# **Mutual Information (MI)**: ukuran ketergantungan (linear maupun non-linear) antara fitur dan target. 0 = independen.
# Ranking MI juga dipakai section 8 untuk memilih fitur bagi group-aggregation / interaksi.
#
# **Single-feature CV score**: untuk tiap fitur, latih decision tree kecil **hanya dengan fitur itu** (3-fold CV), lalu
# ukur AUC (klasifikasi) atau R² (regresi). Ini menunjukkan seberapa prediktif fitur **sendirian**.
#
# **Jika X maka Y (leakage):**
#
# - Jika satu fitur sendirian memberi AUC > `LEAK_AUC_THRESHOLD` (0.95) atau R² > 0.9 → **sangat mungkin leakage**
#   (fitur yang diketahui setelah kejadian target, ID yang terurut sesuai target, dsb). Tanyakan: "apakah informasi ini
#   tersedia saat prediksi dibuat?" Jika tidak → `DROP_COLS`.
# - Jika semua fitur AUC ≈ 0.5 / R² ≈ 0 → sinyal lemah; feature engineering & interaksi sangat penting.
# - Fitur dengan MI ≈ 0 **dan** single-feature score ≈ chance belum tentu tidak berguna (bisa berguna lewat interaksi) — putuskan dengan permutation importance (section 16).

# %%
if CFG.DO_EDA:
    with safe_block("6.7 mutual information"):
        Xq = quick_encode(EDA_DF)
        Xq = Xq.fillna(Xq.median()).fillna(0)
        if Xq.shape[1] > 0:
            n_mi = min(len(Xq), 20000)
            disc = np.array([c in RAW_CAT for c in Xq.columns])
            if TASK == "regression":
                mi = mutual_info_regression(Xq.iloc[:n_mi], EDA_Y[:n_mi], discrete_features=disc, random_state=0)
            else:
                mi = mutual_info_classif(Xq.iloc[:n_mi], EDA_Y[:n_mi], discrete_features=disc, random_state=0)
            MI_RANK = pd.Series(mi, index=Xq.columns).sort_values(ascending=False)
            save_table(MI_RANK.rename("mutual_info").reset_index().rename(columns={"index": "feature"}), "eda_mutual_info")
            top = MI_RANK.head(30)
            fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(top) + 1)))
            ax.barh(top.index[::-1], top.values[::-1], color="slateblue")
            ax.set_xlabel("Mutual information")
            ax.set_title("Mutual information with target (top 30)")
            save_fig("eda_mutual_information")
            print("Top MI:", MI_RANK.head(10).round(4).to_dict())
    with safe_block("6.8 single-feature CV / leakage"):
        Xq = quick_encode(EDA_DF).fillna(-999)
        cand = list(MI_RANK.index[:60]) if len(MI_RANK) else list(Xq.columns[:60])
        n_uni = min(len(Xq), 20000)
        rows = []
        cv3 = (StratifiedKFold(3, shuffle=True, random_state=0) if TASK != "regression" else KFold(3, shuffle=True, random_state=0))
        for c in cand:
            xv = Xq[[c]].iloc[:n_uni].to_numpy()
            yv = EDA_Y[:n_uni]
            leaf = max(10, n_uni // 200)
            if TASK == "regression":
                p = cross_val_predict(DecisionTreeRegressor(max_depth=5, min_samples_leaf=leaf, random_state=0), xv, yv, cv=cv3)
                sc = r2_score(yv, p)
            else:
                p = cross_val_predict(DecisionTreeClassifier(max_depth=5, min_samples_leaf=leaf, random_state=0), xv, yv,
                                      cv=cv3, method="predict_proba")
                sc = roc_auc_score(yv, p[:, 1]) if TASK == "binary" else roc_auc_score(yv, p, multi_class="ovr")
            rows.append({"feature": c, "single_feature_score": round(float(sc), 4)})
        UNI_TABLE = pd.DataFrame(rows).sort_values("single_feature_score", ascending=False)
        save_table(UNI_TABLE, "eda_single_feature_score")
        thr = CFG.LEAK_R2_THRESHOLD if TASK == "regression" else CFG.LEAK_AUC_THRESHOLD
        LEAK_SUSPECTS = UNI_TABLE.loc[UNI_TABLE["single_feature_score"] > thr, "feature"].tolist()
        top = UNI_TABLE.head(25)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(top) + 1)))
        ax.barh(top["feature"][::-1], top["single_feature_score"][::-1], color="seagreen")
        ax.axvline(thr, color="red", ls="--", label=f"leak threshold {thr}")
        if TASK != "regression":
            ax.axvline(0.5, color="grey", ls=":", label="chance (0.5)")
        ax.set_xlabel("Single-feature CV " + ("R2" if TASK == "regression" else "ROC AUC"))
        ax.set_title("Univariate predictive power (top 25)")
        ax.legend(fontsize=8)
        save_fig("eda_single_feature_power")
        display(UNI_TABLE.head(15))
        if LEAK_SUSPECTS:
            print(f"🚨 CURIGA LEAKAGE: {LEAK_SUSPECTS} — fitur tunggal terlalu prediktif. Periksa makna kolom; "
                  f"jika bocor → CFG.DROP_COLS = {[INV_COLMAP.get(c, c) for c in LEAK_SUSPECTS]}")
        else:
            print("✅ Tidak ada fitur tunggal yang mencurigakan (di atas threshold leakage).")

# %% [markdown]
# ### 6.9 Drift train vs test: KS test, PSI, Adversarial Validation
#
# - **KS test** (numerik): statistik 0–1, semakin besar semakin beda distribusinya. p-value kecil dengan data besar hampir selalu signifikan → lihat **besar statistik**, bukan p-value. KS > 0.1 = patut diperhatikan.
# - **PSI** (Population Stability Index): < 0.1 stabil, 0.1–0.25 bergeser sedang, > 0.25 bergeser besar.
# - **Adversarial validation**: latih classifier untuk membedakan baris train vs test. AUC ≈ 0.5 → train & test mirip (bagus).
#   AUC tinggi → ada fitur yang membedakan; feature importance menunjukkan pelakunya.
#
# **Jika X maka Y:**
#
# - AUC adversarial < 0.6 → ✅ aman, CV random bisa dipercaya.
# - AUC 0.6–0.7 → drift ringan; CV masih oke, tapi cek fitur teratas.
# - AUC > 0.7 → ⚠️ drift kuat. Opsi: (1) buang fitur penyebab drift yang tidak penting untuk target (misal ID/tanggal/urutan);
#   (2) ubah fitur absolut menjadi relatif (rasio, rank per periode); (3) gunakan TimeSeries/Group CV jika drift karena waktu/grup;
#   (4) validasi pada subset train yang paling "mirip test" (probabilitas adversarial tinggi).
# - Fitur penyebab drift berupa ID/timestamp → hampir pasti harus dibuang.
#
# **Untuk laporan:** sebutkan AUC adversarial sebagai bukti bahwa skema validasi representatif.

# %%
ADV_AUC = None
if CFG.DO_EDA and HAS_TEST:
    with safe_block("6.9 KS & PSI"):
        def psi(expected, actual, bins=10, categorical=False):
            """Population Stability Index antara distribusi train (expected) dan test (actual)."""
            if categorical:
                e = expected.fillna("NaN").astype(str).value_counts(normalize=True)
                a = actual.fillna("NaN").astype(str).value_counts(normalize=True)
                idx = e.index.union(a.index)
                e, a = e.reindex(idx).fillna(1e-4), a.reindex(idx).fillna(1e-4)
            else:
                e_, a_ = expected.dropna().astype(float), actual.dropna().astype(float)
                if len(e_) == 0 or len(a_) == 0:
                    return np.nan
                qs = np.unique(np.quantile(e_, np.linspace(0, 1, bins + 1)))
                if len(qs) < 3:
                    return 0.0
                qs[0], qs[-1] = -np.inf, np.inf
                e = pd.Series(np.histogram(e_, qs)[0] / len(e_)).clip(lower=1e-4)
                a = pd.Series(np.histogram(a_, qs)[0] / len(a_)).clip(lower=1e-4)
            return float(np.sum((a - e) * np.log(a / e)))

        drift_rows = []
        for c in RAW_NUM:
            if c in EDA_TEST:
                a, b = EDA_DF[c].dropna().astype(float), EDA_TEST[c].dropna().astype(float)
                ks = stats.ks_2samp(a, b) if len(a) and len(b) else None
                drift_rows.append({"feature": c, "type": "num", "ks_stat": round(ks.statistic, 4) if ks else np.nan,
                                   "ks_pvalue": ks.pvalue if ks else np.nan, "psi": round(psi(EDA_DF[c], EDA_TEST[c]), 4)})
        for c in RAW_CAT:
            if c in EDA_TEST:
                drift_rows.append({"feature": c, "type": "cat", "ks_stat": np.nan, "ks_pvalue": np.nan,
                                   "psi": round(psi(EDA_DF[c], EDA_TEST[c], categorical=True), 4)})
        DRIFT_TABLE = pd.DataFrame(drift_rows).sort_values("psi", ascending=False)
        save_table(DRIFT_TABLE, "eda_drift_ks_psi")
        display(DRIFT_TABLE.head(15))
        n_big = int((DRIFT_TABLE["psi"] > 0.25).sum())
        print(f"Fitur PSI > 0.25 (drift besar): {n_big}" + (" ⚠️" if n_big else " ✅"))
    with safe_block("6.9 adversarial validation"):
        if CFG.DO_ADVERSARIAL:
            feats_adv = [c for c in RAW_NUM + RAW_CAT + RAW_DATE if c in EDA_TEST.columns]
            # kode kategori harus konsisten: encode ulang gabungan
            AB = pd.concat([EDA_DF[feats_adv], EDA_TEST[feats_adv]], ignore_index=True)
            ABq = quick_encode(AB, num_cols=[c for c in RAW_NUM if c in feats_adv],
                               cat_cols=[c for c in RAW_CAT if c in feats_adv], date_cols=[c for c in RAW_DATE if c in feats_adv])
            yadv = np.r_[np.zeros(len(EDA_DF)), np.ones(len(EDA_TEST))]
            if HAS_LGB:
                adv_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                               subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=CFG.N_JOBS,
                                               random_state=0)
            else:
                adv_model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
            p_adv = cross_val_predict(adv_model, ABq, yadv, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                      method="predict_proba")[:, 1]
            ADV_AUC = float(roc_auc_score(yadv, p_adv))
            adv_model.fit(ABq, yadv)
            if hasattr(adv_model, "feature_importances_"):
                imp = pd.Series(adv_model.feature_importances_, index=ABq.columns).sort_values(ascending=False)
            else:
                imp = pd.Series(dtype=float)
            print(f"Adversarial validation AUC = {ADV_AUC:.4f}")
            if ADV_AUC > CFG.ADV_AUC_WARN:
                DRIFT_FEATURES = list(imp.index[:10])
                print(f"⚠️ Drift kuat. Fitur paling membedakan train/test: {DRIFT_FEATURES[:10]}")
                print("   Saran: cek apakah fitur tsb ID/waktu/urutan → buang; atau ubah menjadi fitur relatif; "
                      "pertimbangkan Group/TimeSeries CV.")
            elif ADV_AUC > 0.6:
                print("⚠️ Drift ringan. CV masih bisa dipakai, tapi perhatikan fitur teratas di plot.")
            else:
                print("✅ Train dan test mirip (AUC ≈ 0.5). CV acak representatif.")
            if len(imp):
                top = imp.head(20)
                fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(top) + 1)))
                ax.barh(top.index[::-1], top.values[::-1], color="indianred")
                ax.set_xlabel("Importance (split count)")
                ax.set_title(f"Adversarial validation — features separating train/test (AUC={ADV_AUC:.3f})")
                save_fig("eda_adversarial_validation")
elif CFG.DO_EDA:
    print("ℹ️ Test tidak ada → analisis drift di-skip.")

# %% [markdown]
# ---
# # 7. Preprocessing Fold-Safe
#
# **Tujuan:** mengubah data mentah menjadi matriks yang bisa dimakan model **tanpa leakage**. Kuncinya: semua statistik
# (median, kategori yang dikenal, target mean, scaler, batas clipping) **di-fit hanya di train fold**, lalu diterapkan ke
# validation fold dan test. Kalau kamu fit di seluruh train, skor CV jadi terlalu optimis (terutama target encoding!).
#
# Semua logika ada di class `TabularPreprocessor(mode=...)`, dengan 4 mode sesuai kebutuhan model:
#
# | Mode | Dipakai oleh | Numerik | Kategorikal |
# |---|---|---|---|
# | `native` | LightGBM (dan XGBoost jika `XGB_NATIVE_CAT=True`) | NaN dibiarkan | pandas `category` (split kategori native) |
# | `catboost` | CatBoost | NaN dibiarkan | string, `cat_features` (CatBoost punya ordered target statistics sendiri) |
# | `ordinal` | RF, ExtraTrees, HistGB, XGBoost (default), Dummy | imputasi + indikator missing | kode integer (ordinal) |
# | `onehot` | Logistic/Ridge/ElasticNet, KNN, SVM, MLP | log1p (skew), imputasi + indikator, StandardScaler | one-hot (kardinalitas rendah) + target encoding (tinggi) |
#
# **Komponen:**
#
# - **Missing value**: `NUM_IMPUTE` (median/mean/constant) + kolom `<col>__isna` (`ADD_MISSING_INDICATORS`). Kategori NaN → `"__MISSING__"` (jadi kategori tersendiri).
# - **Rare category**: kategori dengan frekuensi < `RARE_MIN_COUNT` di train fold → `"__RARE__"`. Kategori yang tidak pernah dilihat (unseen di valid/test) juga → `"__RARE__"`.
# - **Target encoding (TE)**: rata-rata target per kategori dengan **smoothing** dan **cross-fitting** (K-fold di dalam train fold, supaya baris train tidak "melihat" targetnya sendiri). Implementasi: `sklearn.preprocessing.TargetEncoder` (sklearn ≥ 1.3) jika ada, jika tidak versi manual `ManualKFoldTE`. (category_encoders juga punya TargetEncoder, tapi tidak cross-fitting sehingga rawan bocor di train fold — karena itu tidak dipakai default.)
# - **Scaling**: StandardScaler untuk model berbasis jarak/gradien (linear, KNN, SVM, MLP). Model tree tidak butuh scaling.
# - **Outlier clipping** (`CLIP_OUTLIERS`): winsorize ke quantile train fold.
#
# **Jika X maka Y:**
#
# - Jika model linear jauh lebih buruk dari GBDT → normal (hubungan non-linear). Tetap berguna untuk ensemble.
# - Jika kategori high-card banyak dan GBDT overfit → coba `TE_FOR_TREES=True` atau `RARE_MIN_COUNT` lebih besar (30–100).
# - Jika memory error di mode `onehot` → turunkan `OHE_MAX_CARD` (misal 8).
# - Jika CatBoost error `Invalid type for cat_feature` → artinya ada NaN/float di kolom kategori; mode `catboost` sudah mengisi `"__MISSING__"`, pastikan kolom tsb ada di `CAT_COLS`.
#
# **Untuk laporan:** jelaskan singkat: "Imputasi dan encoding di-fit di dalam setiap fold untuk mencegah data leakage; kategori
# langka digabung; target encoding dengan smoothing dan cross-fitting."

# %%
MISSING_TOKEN = "__MISSING__"
RARE_TOKEN = "__RARE__"


def make_ohe(**kw):
    """OneHotEncoder kompatibel: sparse_output (sklearn>=1.2) vs sparse (lama)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kw)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kw)


class ManualKFoldTE:
    """Target encoding manual dengan smoothing + cross-fitting K-fold (fallback jika sklearn TargetEncoder tidak ada)."""

    def __init__(self, task, n_classes, smoothing=20, n_splits=5, seed=42):
        self.task, self.n_classes, self.m, self.n_splits, self.seed = task, n_classes, smoothing, n_splits, seed

    def _targets(self, y):
        y = np.asarray(y)
        if self.task == "multiclass":
            return np.eye(self.n_classes)[y.astype(int)]
        return y.reshape(-1, 1).astype(float)

    def _fit_map(self, values, T, prior):
        df = pd.DataFrame(T, columns=range(T.shape[1]))
        df["_k"] = values
        g = df.groupby("_k")
        s, n = g.sum(), g.size()
        return (s + prior * self.m).div(n + self.m, axis=0)

    @staticmethod
    def _apply(enc, values, prior):
        out = enc.reindex(values).to_numpy(dtype=float)
        mask = np.isnan(out)
        if mask.any():
            out[mask] = np.broadcast_to(prior, out.shape)[mask]
        return out

    def fit_transform(self, Xc, y):
        T = self._targets(y)
        self.prior_ = T.mean(axis=0)
        self.cols_ = list(Xc.columns)
        kt = T.shape[1]
        out = np.zeros((len(Xc), len(self.cols_) * kt))
        kf = KFold(self.n_splits, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(Xc):
            for j, c in enumerate(self.cols_):
                v = Xc[c].to_numpy()
                enc = self._fit_map(v[tr], T[tr], self.prior_)
                out[va, j * kt:(j + 1) * kt] = self._apply(enc, v[va], self.prior_)
        self.maps_ = {c: self._fit_map(Xc[c].to_numpy(), T, self.prior_) for c in self.cols_}
        self.names_ = [f"{c}__te" if kt == 1 else f"{c}__te{k}" for c in self.cols_ for k in range(kt)]
        return out

    def transform(self, Xc):
        return np.hstack([self._apply(self.maps_[c], Xc[c].to_numpy(), self.prior_) for c in self.cols_])


class SklearnTE:
    """Wrapper sklearn TargetEncoder (cross-fitting bawaan pada fit_transform)."""

    def __init__(self, task, n_classes, seed=42):
        tt = {"binary": "binary", "multiclass": "multiclass", "regression": "continuous"}[task]
        self.enc = SKTargetEncoder(target_type=tt, smooth="auto", cv=5, shuffle=True, random_state=seed)
        self.task, self.n_classes = task, n_classes

    def fit_transform(self, Xc, y):
        out = self.enc.fit_transform(Xc, y)
        kt = out.shape[1] // max(Xc.shape[1], 1)
        self.names_ = [f"{c}__te" if kt == 1 else f"{c}__te{k}" for c in Xc.columns for k in range(kt)]
        return out

    def transform(self, Xc):
        return self.enc.transform(Xc)


def make_target_encoder(task, n_classes, seed=42):
    """Pilih implementasi target encoder sesuai CFG.TE_IMPL & ketersediaan."""
    use_sk = CFG.TE_IMPL in ("auto", "sklearn") and HAS_SK_TE
    if use_sk and task == "multiclass" and SKLEARN_VERSION < (1, 4, 0):
        use_sk = False   # multiclass TE baru ada di sklearn 1.4
    if use_sk:
        return SklearnTE(task, n_classes, seed)
    return ManualKFoldTE(task, n_classes, smoothing=CFG.TE_SMOOTHING, seed=seed)


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing fold-safe untuk data tabular. Fit HANYA pada train fold.

    mode: "native" | "catboost" | "ordinal" | "onehot"
    """

    def __init__(self, mode="native", num_cols=None, cat_cols=None, task="binary", n_classes=2, seed=42):
        self.mode = mode
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.task = task
        self.n_classes = n_classes
        self.seed = seed

    # ---------- helper ----------
    @staticmethod
    def _cat_str(s):
        return s.astype(object).where(s.notna(), MISSING_TOKEN).astype(str)

    def _map_cat(self, X, c):
        s = self._cat_str(X[c]) if c in X.columns else pd.Series(MISSING_TOKEN, index=X.index)
        return s.where(s.isin(self.level_sets_[c]), RARE_TOKEN)

    def _num_block(self, X):
        cols = self.num_cols_
        num = pd.DataFrame({c: pd.to_numeric(X[c], errors="coerce") if c in X.columns else np.nan for c in cols},
                           index=X.index).astype(float)
        num = num.replace([np.inf, -np.inf], np.nan)
        if self.clip_lo_ is not None and len(cols):
            num = num.clip(lower=self.clip_lo_, upper=self.clip_hi_, axis=1)
        for c in self.log_cols_:
            num[c] = np.log1p(num[c].clip(lower=0))
        return num

    # ---------- fit ----------
    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        self.num_cols_ = [c for c in (self.num_cols or []) if c in X.columns]
        self.cat_cols_ = [c for c in (self.cat_cols or []) if c in X.columns]
        self.clip_lo_ = self.clip_hi_ = None
        self.log_cols_ = []
        if CFG.CLIP_OUTLIERS and self.num_cols_:
            q0, q1 = CFG.CLIP_QUANTILES
            raw = X[self.num_cols_].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            self.clip_lo_, self.clip_hi_ = raw.quantile(q0), raw.quantile(q1)
        if self.mode == "onehot" and CFG.LOG_TRANSFORM_LINEAR:
            for c in self.num_cols_:
                s = pd.to_numeric(X[c], errors="coerce")
                if s.notna().sum() > 10 and s.min() >= 0 and s.skew() > 1 and s.nunique() > 10:
                    self.log_cols_.append(c)
        num = self._num_block(X)
        self.impute_ = self.mode in ("ordinal", "onehot")
        if self.impute_:
            self.na_cols_ = [c for c in self.num_cols_ if num[c].isna().any()] if CFG.ADD_MISSING_INDICATORS else []
            if CFG.NUM_IMPUTE == "mean":
                fill = num.mean()
            elif CFG.NUM_IMPUTE == "constant":
                fill = pd.Series(CFG.NUM_IMPUTE_CONSTANT, index=num.columns, dtype=float)
            else:
                fill = num.median()
            self.fill_ = fill.fillna(0.0)
        # kategori: level yang dikenal (rare → __RARE__)
        self.levels_, self.level_sets_ = {}, {}
        for c in self.cat_cols_:
            vc = self._cat_str(X[c]).value_counts()
            keep = list(vc[vc >= CFG.RARE_MIN_COUNT].index) if CFG.RARE_MIN_COUNT > 0 else list(vc.index)
            if not keep:
                keep = list(vc.index[:1])
            self.levels_[c] = keep + ([RARE_TOKEN] if RARE_TOKEN not in keep else [])
            self.level_sets_[c] = set(self.levels_[c])
        mapped = pd.DataFrame({c: self._map_cat(X, c) for c in self.cat_cols_}, index=X.index)
        if self.mode == "onehot":
            self.ohe_cols_ = [c for c in self.cat_cols_ if len(self.levels_[c]) <= CFG.OHE_MAX_CARD + 1]
            hi = [c for c in self.cat_cols_ if c not in self.ohe_cols_]
            self.te_cols_ = hi if (CFG.TARGET_ENCODING and y is not None) else []
            self.freq_cols_ = [c for c in hi if c not in self.te_cols_]
        else:
            self.ohe_cols_, self.freq_cols_ = [], []
            self.te_cols_ = ([c for c in self.cat_cols_ if len(self.levels_[c]) >= CFG.TE_MIN_CARD]
                             if (CFG.TE_FOR_TREES and y is not None) else [])
        te_df = None
        self.te_ = None
        if self.te_cols_:
            try:
                self.te_ = make_target_encoder(self.task, self.n_classes, self.seed)
                arr = self.te_.fit_transform(mapped[self.te_cols_], np.asarray(y))
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ target encoder gagal ({e}) → fallback manual")
                self.te_ = ManualKFoldTE(self.task, self.n_classes, CFG.TE_SMOOTHING, seed=self.seed)
                arr = self.te_.fit_transform(mapped[self.te_cols_], np.asarray(y))
            te_df = pd.DataFrame(arr, columns=self.te_.names_, index=X.index)
        self.freq_maps_ = {c: mapped[c].value_counts(normalize=True) for c in self.freq_cols_}
        if self.ohe_cols_:
            self.ohe_ = make_ohe()
            self.ohe_.fit(mapped[self.ohe_cols_])
        out = self._assemble(num, mapped, te_df, fit=True)
        self.feature_names_out_ = list(out.columns)
        return out

    # ---------- transform ----------
    def transform(self, X):
        num = self._num_block(X)
        mapped = pd.DataFrame({c: self._map_cat(X, c) for c in self.cat_cols_}, index=X.index)
        te_df = None
        if self.te_ is not None:
            te_df = pd.DataFrame(self.te_.transform(mapped[self.te_cols_]), columns=self.te_.names_, index=X.index)
        return self._assemble(num, mapped, te_df, fit=False)

    def _assemble(self, num, mapped, te_df, fit):
        parts = []
        if self.impute_:
            ind = pd.DataFrame({f"{c}__isna": num[c].isna().astype(np.float32) for c in self.na_cols_}, index=num.index)
            num = num.fillna(self.fill_)
            parts += [num, ind]
        else:
            parts.append(num)
        cat_part = None
        if self.mode == "native":
            cat_part = pd.DataFrame({c: pd.Categorical(mapped[c], categories=self.levels_[c]) for c in self.cat_cols_},
                                    index=num.index)
        elif self.mode == "catboost":
            cat_part = mapped.copy()
        elif self.mode == "ordinal":
            cat_part = pd.DataFrame({c: pd.Categorical(mapped[c], categories=self.levels_[c]).codes.astype(np.float32)
                                     for c in self.cat_cols_}, index=num.index)
        elif self.mode == "onehot":
            blocks = []
            if self.ohe_cols_:
                arr = self.ohe_.transform(mapped[self.ohe_cols_])
                if fit:
                    names = []
                    for c, cats in zip(self.ohe_cols_, self.ohe_.categories_):
                        names += [sanitize_name(f"{c}__{v}") for v in cats]
                    self.ohe_names_ = unique_names(names)
                blocks.append(pd.DataFrame(arr.astype(np.float32), columns=self.ohe_names_, index=num.index))
            if self.freq_cols_:
                blocks.append(pd.DataFrame({f"{c}__freq": mapped[c].map(self.freq_maps_[c]).fillna(0).astype(float)
                                            for c in self.freq_cols_}, index=num.index))
            cat_part = pd.concat(blocks, axis=1) if blocks else None
        if cat_part is not None and cat_part.shape[1]:
            parts.append(cat_part)
        if te_df is not None:
            parts.append(te_df)
        out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=num.index)
        if self.mode == "onehot":
            if fit:
                ohe_set = set(getattr(self, "ohe_names_", []))
                self.scale_cols_ = [c for c in out.columns if c not in ohe_set and not c.endswith("__isna")]
                self.scaler_ = StandardScaler().fit(out[self.scale_cols_]) if self.scale_cols_ else None
            if self.scaler_ is not None:
                out[self.scale_cols_] = self.scaler_.transform(out[self.scale_cols_])
        if fit:
            self.cat_out_cols_ = list(self.cat_cols_) if self.mode in ("native", "catboost", "ordinal") else []
            self.cat_cards_ = {c: len(self.levels_[c]) for c in self.cat_out_cols_}
        return out


def make_sample_weight(y_tr):
    """Sample weight 'balanced' untuk imbalance (jika RUN.imbalance == 'class_weight')."""
    if TASK == "regression" or RUN.imbalance != "class_weight":
        return None
    return compute_sample_weight("balanced", y_tr)


def apply_smote(Xt, y_tr, prep, seed):
    """SMOTE / SMOTENC HANYA pada train fold (setelah preprocessing). Return (X, y) baru atau data asli jika tidak bisa."""
    if not HAS_IMBLEARN or prep.mode in ("native", "catboost"):
        return Xt, y_tr, False
    k = int(min(5, np.bincount(y_tr).min() - 1))
    if k < 1:
        return Xt, y_tr, False
    cat_idx = [Xt.columns.get_loc(c) for c in prep.cat_out_cols_] if prep.mode == "ordinal" else []
    sampler = (SMOTENC(categorical_features=cat_idx, k_neighbors=k, random_state=seed) if cat_idx
               else SMOTE(k_neighbors=k, random_state=seed))
    Xr, yr = sampler.fit_resample(Xt, y_tr)
    return pd.DataFrame(Xr, columns=Xt.columns), np.asarray(yr), True

# %% [markdown]
# ### 7.1 Smoke test preprocessing
#
# Uji cepat: fit preprocessor di 80% train, transform 20% sisanya, untuk tiap mode. **Cara membaca:** jumlah kolom output
# per mode dan apakah masih ada NaN. Mode `ordinal`/`onehot` harus **0 NaN**; mode `native`/`catboost` boleh ada NaN
# (ditangani model). Jika cell ini error → ada tipe kolom aneh; cek output section 4 & 6.0.

# %%
with safe_block("7.1 smoke test"):
    _n80 = int(len(train) * 0.8)
    _feat_demo = RAW_NUM + RAW_CAT
    rows = []
    for mode in ["native", "catboost", "ordinal", "onehot"]:
        p = TabularPreprocessor(mode, RAW_NUM, RAW_CAT, TASK, N_CLASSES, CFG.SEED)
        a = p.fit_transform(train[_feat_demo].iloc[:_n80], y[:_n80])
        b = p.transform(train[_feat_demo].iloc[_n80:])
        n_nan = int(b.select_dtypes(include=[np.number]).isna().sum().sum())
        rows.append({"mode": mode, "n_out_cols": a.shape[1], "valid_nan": n_nan, "te_cols": len(p.te_cols_),
                     "ohe_cols": len(p.ohe_cols_), "example_cols": ", ".join(list(a.columns[:5]))})
    display(pd.DataFrame(rows))

# %% [markdown]
# ---
# # 8. Feature Engineering Toolkit
#
# **Tujuan:** membuat fitur baru yang membantu model. Di lomba tabular, **fitur yang baik biasanya memberi kenaikan
# terbesar** — lebih besar dari tuning.
#
# Semua fitur di section ini **tidak memakai target** (label-free), sehingga aman dihitung pada gabungan train+test
# (transduktif — praktik umum di lomba). Fitur yang memakai target (target encoding) **tidak boleh** di sini; itu sudah
# di-handle fold-safe di `TabularPreprocessor`.
#
# | Toggle CFG | Fitur yang dibuat | Kapan berguna |
# |---|---|---|
# | `FE_DATE_PARTS` | year, month, day, dayofweek, dayofyear, week, quarter, is_weekend, hour, sin/cos bulan, elapsed_days, selisih antar tanggal | ada kolom tanggal (musiman, tren) |
# | `FE_FREQ_ENCODING` | `<cat>__count` = frekuensi kategori | kategori high-card; popularitas kategori informatif |
# | `FE_GROUP_AGG` | mean/std numerik per kategori, `num - mean_grup`, `num / mean_grup` | perbandingan relatif terhadap grup (harga vs rata-rata harga kota) |
# | `FE_RATIO_PAIRS`, `FE_AUTO_RATIO_TOPK` | `a/b`, `a-b` | domain knowledge: total/tenure = rata-rata per bulan, debt/income, dll |
# | `FE_POLY_TOPK` | perkalian antar top-k numerik | interaksi untuk model linear |
# | `FE_BINNING_COLS` | bin quantile | hubungan non-monoton untuk model linear |
# | `FE_TEXT_LEN` | panjang karakter, jumlah kata, digit, huruf kapital | kolom teks bebas |
# | `FE_ROW_STATS` | jumlah missing per baris, mean/std z-score numerik per baris, jumlah nol | pola missing informatif |
# | `FE_CUSTOM` | fungsi `custom_feature_engineering(df)` | **fitur domain kamu sendiri** — ini yang membedakan juara |
#
# **Jika X maka Y:**
#
# - Jika menambah fitur membuat CV **turun** → buang (fitur noise membuat model overfit). Selalu bandingkan CV sebelum/sesudah (lihat `experiments.csv`).
# - Jika ada kolom tanggal transaksi & tanggal lahir → buat umur (`(tgl_transaksi - tgl_lahir).dt.days / 365`). Selisih antar tanggal otomatis dibuat.
# - Jika ada kolom "total" dan "jumlah" → rasio total/jumlah = rata-rata per unit (sangat sering ampuh).
# - Jika kategori high-card dan GBDT kesulitan → `FE_FREQ_ENCODING` + `FE_GROUP_AGG`.
# - Jika data punya struktur entitas (customer melakukan banyak transaksi) → agregasi per entitas (`aggregate_child`, section 3).
#
# **Untuk laporan:** daftar fitur baru yang **terbukti** menaikkan CV + alasan domain-nya. Juri suka penjelasan "kenapa".

# %%
def custom_feature_engineering(df):
    """HOOK untuk fitur domain kamu sendiri. df = gabungan train+test (TANPA target).

    Aturan: jangan memakai kolom target; jangan memakai informasi masa depan. Gunakan nama kolom tersanitasi
    (lihat print COLMAP di section 3). Contoh (hapus # untuk mengaktifkan, sesuaikan nama kolom):

        # rata-rata tagihan per bulan (Telco)
        # if {"TotalCharges", "tenure"} <= set(df.columns):
        #     df["avg_charge_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
        # BMI dari tinggi (cm) & berat (kg)
        # df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
        # flag nilai khusus
        # df["is_zero_balance"] = (df["balance"] == 0).astype(int)
        # jumlah layanan yang diambil (kolom Yes/No yang sudah jadi 1/0)
        # svc = [c for c in ["PhoneService", "OnlineSecurity", "OnlineBackup", "TechSupport"] if c in df]
        # df["n_services"] = df[svc].sum(axis=1)
    """
    return df


def fe_date_parts(df, date_cols):
    """Pecah kolom datetime menjadi komponen + selisih antar tanggal; kolom datetime mentah dibuang."""
    new = {}
    for c in date_cols:
        d = df[c]
        new[f"{c}_year"] = d.dt.year
        new[f"{c}_month"] = d.dt.month
        new[f"{c}_day"] = d.dt.day
        new[f"{c}_dayofweek"] = d.dt.dayofweek
        new[f"{c}_dayofyear"] = d.dt.dayofyear
        new[f"{c}_quarter"] = d.dt.quarter
        new[f"{c}_is_weekend"] = (d.dt.dayofweek >= 5).astype(float).where(d.notna())
        try:
            new[f"{c}_week"] = d.dt.isocalendar().week.astype(float)
        except Exception:  # noqa: BLE001
            pass
        if (d.dt.hour.fillna(0) != 0).any():
            new[f"{c}_hour"] = d.dt.hour
        new[f"{c}_month_sin"] = np.sin(2 * np.pi * d.dt.month / 12)
        new[f"{c}_month_cos"] = np.cos(2 * np.pi * d.dt.month / 12)
        new[f"{c}_elapsed_days"] = (d - d.min()).dt.days
    for i, a in enumerate(date_cols[:4]):
        for b in date_cols[i + 1:4]:
            new[f"{a}_minus_{b}_days"] = (df[a] - df[b]).dt.days
    out = pd.concat([df.drop(columns=date_cols), pd.DataFrame(new, index=df.index).astype(float)], axis=1)
    return out


def fe_text(df, text_cols):
    """Statistik sederhana kolom teks."""
    new = {}
    for c in text_cols:
        s = df[c].fillna("").astype(str)
        new[f"{c}_n_chars"] = s.str.len()
        new[f"{c}_n_words"] = s.str.split().str.len()
        new[f"{c}_n_digits"] = s.str.count(r"\d")
        new[f"{c}_n_upper"] = s.str.count(r"[A-Z]")
        new[f"{c}_n_punct"] = s.str.count(r"[^\w\s]")
    return pd.concat([df, pd.DataFrame(new, index=df.index).astype(float)], axis=1)


def fe_freq(df, cat_cols, n_train):
    """Frequency / count encoding (fit pada train+test atau train saja)."""
    new = {}
    for c in cat_cols:
        s = df[c].astype(object).where(df[c].notna(), MISSING_TOKEN).astype(str)
        ref = s if CFG.FE_FREQ_FIT_ON == "train+test" else s.iloc[:n_train]
        new[f"{c}__count"] = s.map(ref.value_counts()).fillna(0).astype(float)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def fe_row_stats(df, num_cols, base_cols, n_train):
    """Statistik per baris: jumlah missing, mean/std z-score numerik, jumlah nol."""
    new = {"row_n_missing": df[base_cols].isna().sum(axis=1).astype(float)}
    if len(num_cols) >= 2:
        z = df[num_cols].astype(float)
        mu, sd = z.iloc[:n_train].mean(), z.iloc[:n_train].std().replace(0, 1)
        z = (z - mu) / sd
        new["row_num_zmean"] = z.mean(axis=1)
        new["row_num_zstd"] = z.std(axis=1)
        new["row_n_zero"] = (df[num_cols] == 0).sum(axis=1).astype(float)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def fe_group_agg(df, cats, nums):
    """Agregasi numerik per kategori: mean, std, selisih dan rasio terhadap mean grup."""
    new = {}
    for g in cats:
        key = df[g].astype(object).where(df[g].notna(), MISSING_TOKEN).astype(str)
        for n in nums:
            v = df[n].astype(float)
            grp = v.groupby(key)
            m = grp.transform("mean")
            new[f"{n}_mean_by_{g}"] = m
            new[f"{n}_std_by_{g}"] = grp.transform("std")
            new[f"{n}_diff_mean_{g}"] = v - m
            new[f"{n}_ratio_mean_{g}"] = v / m.replace(0, np.nan)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def fe_ratios(df, pairs):
    """Rasio a/b dan selisih a-b untuk pasangan numerik."""
    new = {}
    for a, b in pairs:
        if a in df and b in df:
            va, vb = df[a].astype(float), df[b].astype(float)
            new[f"{a}_div_{b}"] = va / vb.replace(0, np.nan)
            new[f"{a}_minus_{b}"] = va - vb
        else:
            print(f"  ⚠️ pasangan rasio ({a}, {b}) dilewati: kolom tidak ada")
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1) if new else df


def fe_poly(df, cols):
    """Interaksi perkalian antar kolom numerik terpilih."""
    new = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            new[f"{a}_x_{b}"] = df[a].astype(float) * df[b].astype(float)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1) if new else df


def fe_binning(df, cols, n_bins, n_train):
    """Binning quantile (batas dari train)."""
    new = {}
    for c in cols:
        if c not in df:
            continue
        v = df[c].astype(float)
        edges = np.unique(np.nanquantile(v.iloc[:n_train], np.linspace(0, 1, n_bins + 1)))
        if len(edges) > 2:
            new[f"{c}_bin"] = pd.Series(np.digitize(v, edges[1:-1]), index=df.index).astype(float).where(v.notna())
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1) if new else df

# %%
banner("8. FEATURE ENGINEERING")
N_TRAIN = len(train)
train_ids = train[ID].to_numpy() if ID else np.arange(N_TRAIN)
test_ids = test[ID].to_numpy() if (ID and ID in test.columns) else np.arange(len(test))
GROUPS = train[GROUP].astype(str).to_numpy() if GROUP else None
TIME_VALUES = train[TIME].to_numpy() if TIME else None

base_feats = RAW_NUM + RAW_CAT + RAW_DATE + RAW_TEXT
full = pd.concat([train[base_feats], test[[c for c in base_feats if c in test.columns]]], axis=0, ignore_index=True)
n_before = full.shape[1]
fe_log = []

if CFG.FE_ROW_STATS:
    full = fe_row_stats(full, RAW_NUM, base_feats, N_TRAIN)
    fe_log.append("row_stats")
if RAW_DATE and CFG.FE_DATE_PARTS:
    full = fe_date_parts(full, RAW_DATE)
    fe_log.append(f"date_parts({len(RAW_DATE)})")
elif RAW_DATE:
    for c in RAW_DATE:   # tanpa date parts: ubah jadi angka (hari sejak 1970)
        full[c] = (full[c] - pd.Timestamp("1970-01-01")).dt.days.astype(float)
if RAW_TEXT and CFG.FE_TEXT_LEN:
    full = fe_text(full, RAW_TEXT)
    fe_log.append(f"text({len(RAW_TEXT)})")
if RAW_TEXT and CFG.DROP_RAW_TEXT:
    full = full.drop(columns=RAW_TEXT)
if CFG.FE_FREQ_ENCODING and RAW_CAT:
    full = fe_freq(full, RAW_CAT, N_TRAIN)
    fe_log.append("freq")
_mi_num = [c for c in MI_RANK.index if c in RAW_NUM] if len(MI_RANK) else list(RAW_NUM)
_mi_cat = [c for c in MI_RANK.index if c in RAW_CAT and train[c].nunique() <= 1000] if len(MI_RANK) else list(RAW_CAT)
if CFG.FE_GROUP_AGG and RAW_CAT and RAW_NUM:
    g_cats = col_list(CFG.FE_GROUP_AGG_CATS) or _mi_cat[:3]
    g_nums = col_list(CFG.FE_GROUP_AGG_NUMS) or _mi_num[:5]
    full = fe_group_agg(full, [c for c in g_cats if c in full], [c for c in g_nums if c in full])
    fe_log.append(f"group_agg({len(g_cats)}x{len(g_nums)})")
pairs = [(col_name(a), col_name(b)) for a, b in CFG.FE_RATIO_PAIRS]
if CFG.FE_AUTO_RATIO_TOPK > 0:
    top = _mi_num[: CFG.FE_AUTO_RATIO_TOPK]
    pairs += [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
if pairs:
    full = fe_ratios(full, pairs)
    fe_log.append(f"ratios({len(pairs)})")
if CFG.FE_POLY_TOPK > 0:
    full = fe_poly(full, [c for c in _mi_num[: CFG.FE_POLY_TOPK] if c in full])
    fe_log.append("poly")
if CFG.FE_BINNING_COLS:
    full = fe_binning(full, col_list(CFG.FE_BINNING_COLS), CFG.FE_N_BINS, N_TRAIN)
    fe_log.append("binning")
if CFG.FE_CUSTOM:
    n0 = full.shape[1]
    full = custom_feature_engineering(full)
    if full.shape[1] != n0:
        fe_log.append(f"custom(+{full.shape[1] - n0})")

# bersihkan: nama aman & unik, inf → NaN, buang DROP_FEATURES, buang kolom konstan baru
full.columns = unique_names([sanitize_name(c) for c in full.columns])
num_like = full.select_dtypes(include=[np.number]).columns
full[num_like] = full[num_like].replace([np.inf, -np.inf], np.nan)
drop_f = [c for c in (col_list(CFG.DROP_FEATURES) or []) if c in full.columns]
const_new = [c for c in full.columns if full[c].iloc[:N_TRAIN].nunique(dropna=False) <= 1]
full = full.drop(columns=list(dict.fromkeys(drop_f + const_new)))
if drop_f:
    print(f"🗑️ DROP_FEATURES dibuang: {drop_f}")
if const_new:
    print(f"🗑️ fitur konstan hasil FE dibuang: {const_new}")

CAT_COLS = [c for c in full.columns if full[c].dtype == object or str(full[c].dtype) == "category"]
NUM_COLS = [c for c in full.columns if c not in CAT_COLS]
for c in NUM_COLS:
    if not pd.api.types.is_numeric_dtype(full[c]):
        full[c] = pd.to_numeric(full[c], errors="coerce")
FEATURES = CAT_COLS + NUM_COLS
X = full.iloc[:N_TRAIN][FEATURES].reset_index(drop=True)
X_test = full.iloc[N_TRAIN:][FEATURES].reset_index(drop=True)
del full
gc.collect()
print(f"FE aktif: {fe_log}")
print(f"Jumlah fitur: {n_before} → {len(FEATURES)} (kategorikal {len(CAT_COLS)}, numerik {len(NUM_COLS)})")
print(f"X: {X.shape} | X_test: {X_test.shape} | y: {y.shape}")
new_cols = [c for c in FEATURES if c not in base_feats]
print(f"Contoh fitur baru: {new_cols[:15]}")

# %% [markdown]
# ---
# # 9. Strategi Cross-Validation
#
# **Tujuan:** membuat skema validasi yang **meniru hubungan train ↔ test**. Ini keputusan paling penting di seluruh
# notebook: CV yang salah → kamu memilih model/fitur yang salah walaupun skornya terlihat bagus.
#
# | `CV_STRATEGY` | Kapan dipakai | Catatan |
# |---|---|---|
# | `stratified` | klasifikasi, baris independen | proporsi kelas sama di tiap fold (default klasifikasi) |
# | `kfold` | regresi (otomatis stratified berdasarkan bin target) | bin = `REG_STRAT_BINS` quantile → distribusi target mirip di tiap fold |
# | `group` | satu entitas punya banyak baris (customer, pasien, toko) **dan** test berisi entitas baru | isi `GROUP_COL`; entitas yang sama tidak pernah ada di train & valid sekaligus |
# | `stratified_group` | seperti `group` + klasifikasi imbalanced | butuh sklearn ≥ 1.0 (fallback ke GroupKFold) |
# | `timeseries` | test = periode setelah train | isi `TIME_COL`; train selalu masa lalu, valid masa depan. Baris paling awal tidak pernah jadi validasi (OOF hanya dihitung di baris yang tervalidasi) |
# | `auto` | default | TIME_COL → timeseries; GROUP_COL → (stratified_)group; klasifikasi → stratified; regresi → kfold-binned |
#
# **Repeated CV (`N_REPEATS > 1`)**: split diulang dengan seed berbeda, prediksi OOF dirata-rata → estimasi lebih stabil &
# test prediction = rata-rata lebih banyak model (biasanya sedikit lebih baik). Biaya waktu × N_REPEATS.
#
# **Cara membaca output:** tabel per fold — ukuran train/valid dan distribusi target di valid harus mirip antar fold.
# Untuk group CV, kolom `group_overlap` harus 0. Untuk timeseries, `valid_time_min` harus > `train_time_max`.
#
# **Jika X maka Y:**
#
# - Jika CV bagus tapi public LB jauh lebih buruk → CV tidak meniru test: cek adversarial validation (6.9), pertimbangkan group/time CV.
# - Jika std skor antar fold besar (> 2–3% dari skor) → data kecil/noisy: naikkan `N_FOLDS` (10) atau `N_REPEATS` (3).
# - Jika data kecil (< 2000 baris) → `N_FOLDS=10`, `N_REPEATS=3`.
# - Jika data sangat besar (> 1 juta) → `N_FOLDS=3` sudah cukup.
#
# **Untuk laporan:** tulis "Validasi: Stratified K-Fold (K=5, seed=42) — skor dilaporkan sebagai mean ± std antar fold dan skor OOF keseluruhan."

# %%
def _resolve_cv_strategy():
    s = CFG.CV_STRATEGY
    if s != "auto":
        return s
    if TIME is not None:
        return "timeseries"
    if GROUPS is not None:
        return "stratified_group" if (TASK != "regression" and HAS_SGKF) else "group"
    return "stratified" if TASK != "regression" else "kfold"


CV_NAME = _resolve_cv_strategy()


def make_splits(seed, strategy=CV_NAME, n_folds=None, y_arr=None):
    """Return list (train_idx, valid_idx) sesuai strategi CV."""
    y_arr = y if y_arr is None else y_arr
    n = len(y_arr)
    k = n_folds or RUN.n_folds
    idx = np.arange(n)
    if TASK != "regression" and strategy in ("stratified", "stratified_group"):
        k = int(max(2, min(k, np.bincount(y_arr).min())))
    if GROUPS is not None and strategy in ("group", "stratified_group"):
        k = int(min(k, len(np.unique(GROUPS))))
    if strategy == "stratified":
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, y_arr))
    if strategy == "kfold":
        if TASK == "regression":
            bins = pd.qcut(pd.Series(y_arr).rank(method="first"), q=min(CFG.REG_STRAT_BINS, max(2, n // (k * 2))),
                           labels=False)
            return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, bins))
        return list(KFold(k, shuffle=True, random_state=seed).split(idx))
    if strategy == "group":
        if seed == RUN.seeds[0]:
            return list(GroupKFold(k).split(idx, y_arr, GROUPS))
        ug = np.random.RandomState(seed).permutation(np.unique(GROUPS))   # repeat: assignment grup diacak
        fold_of = {g: i % k for i, g in enumerate(ug)}
        f = np.array([fold_of[g] for g in GROUPS])
        return [(idx[f != i], idx[f == i]) for i in range(k)]
    if strategy == "stratified_group":
        if HAS_SGKF:
            return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(idx, y_arr, GROUPS))
        return list(GroupKFold(k).split(idx, y_arr, GROUPS))
    if strategy == "timeseries":
        tv = TIME_VALUES
        tnum = pd.to_datetime(pd.Series(tv)).astype("int64").to_numpy() if not np.issubdtype(np.asarray(tv).dtype, np.number) else np.asarray(tv, dtype=float)
        order = np.argsort(tnum, kind="mergesort")
        return [(order[tr], order[va]) for tr, va in TimeSeriesSplit(k).split(order)]
    raise ValueError(f"CV_STRATEGY tidak dikenal: {strategy}")


banner("9. CROSS-VALIDATION")
if CV_NAME == "timeseries" and RUN.n_repeats > 1:
    print("ℹ️ TimeSeries CV: repeats tidak bermakna → N_REPEATS=1")
    RUN.seeds, RUN.n_repeats = RUN.seeds[:1], 1
CV_SPLITS = [make_splits(s) for s in RUN.seeds]
N_FOLDS_EFF = len(CV_SPLITS[0])
OOF_MASK = np.zeros(len(y), dtype=bool)
for tr, va in CV_SPLITS[0]:
    OOF_MASK[va] = True
print(f"Strategi: {CV_NAME} | folds: {N_FOLDS_EFF} | repeats: {RUN.n_repeats} | baris tervalidasi (OOF): {OOF_MASK.mean():.1%}")

rows = []
for f, (tr, va) in enumerate(CV_SPLITS[0]):
    r = {"fold": f, "n_train": len(tr), "n_valid": len(va)}
    if TASK == "regression":
        r["valid_target_mean"] = round(float(np.mean(y[va])), 4)
        r["train_target_mean"] = round(float(np.mean(y[tr])), 4)
    else:
        dist = np.bincount(y[va], minlength=N_CLASSES) / len(va)
        for kc in range(min(N_CLASSES, 6)):
            r[f"valid_pct_{ID2LABEL[kc]}"] = round(100 * dist[kc], 1)
    if GROUPS is not None:
        r["group_overlap"] = len(set(GROUPS[tr]) & set(GROUPS[va]))
    if TIME_VALUES is not None:
        r["train_time_max"] = str(pd.Series(TIME_VALUES[tr]).max())[:19]
        r["valid_time_min"] = str(pd.Series(TIME_VALUES[va]).min())[:19]
    rows.append(r)
FOLD_TABLE = pd.DataFrame(rows)
display(FOLD_TABLE)
save_table(FOLD_TABLE, "cv_folds")
if GROUPS is not None and FOLD_TABLE["group_overlap"].sum() > 0:
    print("🚨 Ada grup yang muncul di train & valid sekaligus → periksa GROUP_COL / strategi CV!")
else:
    print("✅ Fold terbentuk dengan benar.")

# %% [markdown]
# ---
# # 10. Definisi Model
#
# **Tujuan:** mendefinisikan semua model dengan **default parameter yang kuat** per task, plus early stopping, class
# imbalance handling, dan fallback GPU → CPU. Setiap model dibungkus sehingga jika satu gagal, loop tetap jalan
# (error disimpan di `FAILED_MODELS`).
#
# | Nama | Model | Mode preprocessing | Catatan |
# |---|---|---|---|
# | `dummy` | DummyClassifier(prior) / DummyRegressor(mean) | ordinal | **baseline wajib**: model lain harus jauh lebih baik dari ini |
# | `linear` | LogisticRegression / Ridge | onehot | cepat, stabil, bagus untuk ensemble |
# | `enet` | Logistic elastic-net (saga) / ElasticNet | onehot | seleksi fitur implisit (L1) |
# | `rf` | RandomForest | ordinal | robust, jarang overfit parah |
# | `et` | ExtraTrees | ordinal | lebih acak dari RF, sering bagus di ensemble |
# | `hgb` | HistGradientBoosting (sklearn) | ordinal + categorical mask | **selalu tersedia**, fallback jika LightGBM tidak ada |
# | `lgb` | LightGBM | native category | biasanya model terbaik untuk tabular; cepat |
# | `xgb` | XGBoost (hist) | ordinal (atau native) | kuat, sedikit lebih lambat |
# | `cat` | CatBoost | catboost (cat_features) | terbaik untuk banyak kategori; lebih lambat |
# | `knn` | KNeighbors | onehot | hanya data kecil (≤ `SMALL_DATA_MAX_ROWS`) |
# | `svm` | SVC(probability) / SVR | onehot | hanya data kecil; lambat O(n²) |
# | `mlp` | MLP sklearn | onehot | jaringan saraf sederhana, early stopping |
#
# **Early stopping:** LightGBM/XGBoost/CatBoost berhenti jika skor di validation fold tidak membaik selama
# `EARLY_STOPPING` iterasi. (Catatan jujur: memakai validation fold untuk early stopping sedikit optimis; efeknya kecil
# dan praktik ini standar di lomba. HistGB memakai 10% train fold sebagai validasi internal.)
#
# **Class imbalance (`IMBALANCE_STRATEGY`):**
#
# - `class_weight` → `sample_weight="balanced"` dipasang ke semua model yang mendukung (setara `class_weight="balanced"`,
#   `scale_pos_weight` di XGBoost, `is_unbalance` di LightGBM, `auto_class_weights` di CatBoost).
# - `smote` → SMOTE/SMOTENC **hanya pada train fold** setelah preprocessing (tidak pernah pada valid/test!). Hanya untuk
#   model mode ordinal/onehot; LightGBM/CatBoost otomatis memakai class weight.
# - **Kapan JANGAN pakai SMOTE:** metric AUC/logloss (SMOTE merusak kalibrasi, AUC jarang naik), data sudah besar,
#   banyak fitur kategorikal, atau saat GBDT + threshold tuning sudah cukup. Di praktik lomba, **class weight + threshold
#   tuning** hampir selalu ≥ SMOTE.
#
# **Jika X maka Y:**
#
# - Jika model lebih buruk dari `dummy` → ada bug (target salah encode, fitur rusak) atau metric salah arah.
# - Jika `best_iter` selalu = `GBDT_N_ESTIMATORS` → naikkan n_estimators atau learning rate.
# - Jika `best_iter` sangat kecil (< 50) → learning rate terlalu besar atau model langsung overfit; turunkan LR, naikkan regularisasi.
# - Jika GPU error → otomatis fallback CPU (pesan ⚠️ dicetak). LightGBM GPU butuh build khusus; XGBoost/CatBoost GPU biasanya langsung jalan.
# - Ingin ganti parameter → `CFG.MODEL_PARAMS = {"lgb": {"num_leaves": 63, "min_child_samples": 50}}`.

# %%
MODEL_MODES = {"dummy": "ordinal", "linear": "onehot", "enet": "onehot", "rf": "ordinal", "et": "ordinal",
               "hgb": "ordinal", "lgb": "native", "xgb": "native" if CFG.XGB_NATIVE_CAT else "ordinal",
               "cat": "catboost", "knn": "onehot", "svm": "onehot", "mlp": "onehot"}
FAILED_MODELS = {}


def base_model(name):
    """'lgb_tuned' → 'lgb'."""
    return str(name).split("_")[0]


def model_available(name):
    """Return (bool, alasan)."""
    b = base_model(name)
    if b not in MODEL_MODES:
        return False, f"model '{b}' tidak dikenal"
    if b == "lgb" and not HAS_LGB:
        return False, "lightgbm tidak terinstall"
    if b == "xgb" and not HAS_XGB:
        return False, "xgboost tidak terinstall"
    if b == "cat" and not HAS_CAT:
        return False, "catboost tidak terinstall"
    if b == "hgb" and not HAS_HGB:
        return False, "HistGradientBoosting tidak tersedia (sklearn lama)"
    if b in ("knn", "svm") and len(X) > CFG.SMALL_DATA_MAX_ROWS:
        return False, f"data > {CFG.SMALL_DATA_MAX_ROWS} baris (terlalu lambat)"
    return True, "ok"


def default_params(name, seed):
    """Default parameter kuat per model & task."""
    b = base_model(name)
    clf = TASK != "regression"
    small = len(X) < 3000
    m = RUN.metric
    if b == "lgb":
        p = dict(n_estimators=RUN.n_estimators, learning_rate=RUN.lr, num_leaves=15 if small else 31,
                 min_child_samples=10 if small else 20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=CFG.N_JOBS, verbose=-1)
        if TASK == "binary":
            p["objective"] = "binary"
        elif TASK == "multiclass":
            p["objective"] = "multiclass"
        else:
            p["objective"] = {"mae": "regression_l1", "mape": "mape"}.get(m, "regression")
        return p
    if b == "xgb":
        p = dict(n_estimators=RUN.n_estimators, learning_rate=RUN.lr, max_depth=5 if small else 6, min_child_weight=1,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, tree_method="hist",
                 random_state=seed, n_jobs=CFG.N_JOBS)
        if TASK == "binary":
            p.update(objective="binary:logistic", eval_metric="logloss")
        elif TASK == "multiclass":
            p.update(objective="multi:softprob", eval_metric="mlogloss")
        else:
            if m == "mae" and XGB_VERSION >= (1, 7, 0):
                p.update(objective="reg:absoluteerror", eval_metric="mae")
            else:
                p.update(objective="reg:squarederror", eval_metric="rmse")
        return p
    if b == "cat":
        p = dict(iterations=min(RUN.n_estimators, 1000) if RUN.fast else RUN.n_estimators, learning_rate=max(RUN.lr, 0.05), depth=6, l2_leaf_reg=3.0,
                 one_hot_max_size=10,   # kategori kecil → one-hot (jauh lebih cepat dari CTR), besar → CTR
                 random_seed=seed, od_type="Iter", od_wait=RUN.early_stopping, verbose=0,
                 thread_count=CFG.N_JOBS if CFG.N_JOBS and CFG.N_JOBS > 0 else -1, allow_writing_files=False)
        p["loss_function"] = {"binary": "Logloss", "multiclass": "MultiClass"}.get(TASK, "MAE" if m == "mae" else "RMSE")
        return p
    if b == "dummy":
        return dict(strategy="prior") if clf else dict(strategy="median" if m == "mae" else "mean")
    if b == "linear":
        return dict(C=1.0, max_iter=3000) if clf else dict(alpha=1.0)
    if b == "enet":
        return (dict(C=0.5, l1_ratio=0.5, max_iter=3000) if clf else dict(alpha=0.001, l1_ratio=0.5, max_iter=5000))
    if b in ("rf", "et"):
        return dict(n_estimators=RUN.rf_trees, max_features="sqrt" if clf else 0.5, min_samples_leaf=1 if clf else 2,
                    n_jobs=CFG.N_JOBS, random_state=seed)
    if b == "hgb":
        p = dict(max_iter=500 if RUN.fast else 1500, learning_rate=max(RUN.lr, 0.05), max_leaf_nodes=31,
                 min_samples_leaf=20, l2_regularization=0.1, early_stopping=True, validation_fraction=0.1,
                 n_iter_no_change=50, random_state=seed)
        if not clf and m == "mae":
            p["loss"] = "absolute_error"
        return p
    if b == "knn":
        return dict(n_neighbors=25, weights="distance", n_jobs=CFG.N_JOBS)
    if b == "svm":
        return dict(C=1.0, probability=True, random_state=seed) if clf else dict(C=1.0)
    if b == "mlp":
        return dict(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=300,
                    learning_rate_init=1e-3, random_state=seed)
    raise ValueError(b)


def get_params(name, seed, override=None):
    """Default → CFG.MODEL_PARAMS[base] → override (misal hasil Optuna)."""
    p = default_params(name, seed)
    p.update(CFG.MODEL_PARAMS.get(base_model(name), {}))
    p.update(override or {})
    return p


def _build_sklearn(b, params, prep=None):
    """Bangun estimator sklearn sesuai task."""
    clf = TASK != "regression"
    params = dict(params)
    if b == "dummy":
        return DummyClassifier(**params) if clf else DummyRegressor(**params)
    if b == "linear":
        return LogisticRegression(**params) if clf else Ridge(**params)
    if b == "enet":
        if clf:
            try:
                return LogisticRegression(penalty="elasticnet", solver="saga", **params)
            except TypeError:   # sklearn versi baru: elastic-net cukup lewat l1_ratio
                return LogisticRegression(solver="saga", **params)
        return ElasticNet(**params)
    if b == "rf":
        return RandomForestClassifier(**params) if clf else RandomForestRegressor(**params)
    if b == "et":
        return ExtraTreesClassifier(**params) if clf else ExtraTreesRegressor(**params)
    if b == "hgb":
        if prep is not None and prep.mode == "ordinal" and prep.cat_out_cols_:
            mask = np.array([c in prep.cat_out_cols_ and prep.cat_cards_.get(c, 999) <= 250
                             for c in prep.feature_names_out_])
            if mask.any() and "categorical_features" not in params:
                params["categorical_features"] = mask
        return HistGradientBoostingClassifier(**params) if clf else HistGradientBoostingRegressor(**params)
    if b == "knn":
        return KNeighborsClassifier(**params) if clf else KNeighborsRegressor(**params)
    if b == "svm":
        if not clf:
            params.pop("probability", None)
            params.pop("random_state", None)
        return SVC(**params) if clf else SVR(**params)
    if b == "mlp":
        return MLPClassifier(**params) if clf else MLPRegressor(**params)
    raise ValueError(b)


def _fit_lgb(params, Xtr, ytr, Xva, yva, sw):
    cls = lgb.LGBMClassifier if TASK != "regression" else lgb.LGBMRegressor
    use_gpu = RUN.use_gpu["lgb"] and not GPU_DISABLED["lgb"]
    p = dict(params)
    if use_gpu:
        p["device"] = "gpu"

    def _do(pp):
        mdl = cls(**pp)
        if Xva is None:
            mdl.fit(Xtr, ytr, sample_weight=sw)
            return mdl
        try:
            mdl.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)],
                    callbacks=[lgb.early_stopping(RUN.early_stopping, verbose=False), lgb.log_evaluation(0)])
        except (AttributeError, TypeError):   # LightGBM lama
            mdl.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], early_stopping_rounds=RUN.early_stopping,
                    verbose=False)
        return mdl

    try:
        m = _do(p)
    except Exception as e:  # noqa: BLE001
        if not use_gpu:
            raise
        print(f"  ⚠️ LightGBM GPU gagal ({str(e)[:80]}) → fallback CPU")
        GPU_DISABLED["lgb"] = True
        p.pop("device", None)
        m = _do(p)
    best = getattr(m, "best_iteration_", None) or p.get("n_estimators")
    return m, {"best_iter": best}


def _fit_xgb(params, Xtr, ytr, Xva, yva, sw, prep):
    cls = xgb.XGBClassifier if TASK != "regression" else xgb.XGBRegressor
    use_gpu = RUN.use_gpu["xgb"] and not GPU_DISABLED["xgb"]
    p = dict(params)
    if prep.mode == "native":
        p["enable_categorical"] = True
        p["tree_method"] = "hist"
    if use_gpu:
        if XGB_VERSION >= (2, 0, 0):
            p["device"] = "cuda"
        else:
            p["tree_method"] = "gpu_hist"

    def _do(pp):
        if Xva is None:
            pp = {k: v for k, v in pp.items() if k != "early_stopping_rounds"}
            mdl = cls(**pp)
            mdl.fit(Xtr, ytr, sample_weight=sw, verbose=False)
            return mdl
        try:
            mdl = cls(early_stopping_rounds=RUN.early_stopping, **pp)
            mdl.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
        except TypeError:   # XGBoost < 1.6: early_stopping_rounds di fit()
            mdl = cls(**pp)
            mdl.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], early_stopping_rounds=RUN.early_stopping,
                    verbose=False)
        return mdl

    try:
        m = _do(p)
    except Exception as e:  # noqa: BLE001
        if not use_gpu:
            raise
        print(f"  ⚠️ XGBoost GPU gagal ({str(e)[:80]}) → fallback CPU")
        GPU_DISABLED["xgb"] = True
        p.pop("device", None)
        p["tree_method"] = "hist"
        m = _do(p)
    best = getattr(m, "best_iteration", None)
    return m, {"best_iter": (best + 1) if best is not None else p.get("n_estimators")}


def _fit_cat(params, Xtr, ytr, Xva, yva, sw, prep):
    cls = cb.CatBoostClassifier if TASK != "regression" else cb.CatBoostRegressor
    use_gpu = RUN.use_gpu["cat"] and not GPU_DISABLED["cat"]
    p = dict(params)
    if Xva is None:
        p.pop("od_type", None)
        p.pop("od_wait", None)
    if use_gpu:
        p["task_type"] = "GPU"
    cat_feats = [c for c in prep.cat_out_cols_ if c in Xtr.columns]

    def _do(pp):
        mdl = cls(**pp)
        if Xva is None:
            mdl.fit(Xtr, ytr, cat_features=cat_feats, sample_weight=sw, verbose=False)
        else:
            mdl.fit(Xtr, ytr, cat_features=cat_feats, sample_weight=sw, eval_set=(Xva, yva), use_best_model=True,
                    verbose=False)
        return mdl

    try:
        m = _do(p)
    except Exception as e:  # noqa: BLE001
        if not use_gpu:
            raise
        print(f"  ⚠️ CatBoost GPU gagal ({str(e)[:80]}) → fallback CPU")
        GPU_DISABLED["cat"] = True
        p.pop("task_type", None)
        m = _do(p)
    try:
        best = m.get_best_iteration()
    except Exception:  # noqa: BLE001
        best = None
    return m, {"best_iter": (best + 1) if best is not None else p.get("iterations")}


def fit_one(name, params, Xtr, ytr, Xva, yva, sw, prep):
    """Latih satu model pada satu fold. Return (estimator, info)."""
    b = base_model(name)
    if b == "lgb":
        return _fit_lgb(params, Xtr, ytr, Xva, yva, sw)
    if b == "xgb":
        return _fit_xgb(params, Xtr, ytr, Xva, yva, sw, prep)
    if b == "cat":
        return _fit_cat(params, Xtr, ytr, Xva, yva, sw, prep)
    est = _build_sklearn(b, params, prep)
    if sw is not None:
        try:
            est.fit(Xtr, ytr, sample_weight=sw)
        except TypeError:
            est.fit(Xtr, ytr)
    else:
        est.fit(Xtr, ytr)
    info = {"best_iter": getattr(est, "n_iter_", None)}
    if isinstance(info["best_iter"], (list, np.ndarray)):
        info["best_iter"] = int(np.max(info["best_iter"]))
    return est, info


def align_proba(proba, classes, K):
    """Pastikan kolom probabilitas berurutan 0..K-1 (jika ada kelas yang tidak muncul di train fold)."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.c_[1 - proba, proba]
    if proba.shape[1] == K and (classes is None or list(np.asarray(classes).astype(int)) == list(range(K))):
        return proba
    out = np.zeros((proba.shape[0], K))
    for j, c in enumerate(classes):
        out[:, int(c)] = proba[:, j]
    return out


def predict_std(est, Xt):
    """Prediksi standar: binary → prob kelas 1 (n,), multiclass → (n,K), regresi → (n,) (skala transform)."""
    if TASK == "regression":
        return np.asarray(est.predict(Xt), dtype=float).reshape(-1)
    proba = align_proba(est.predict_proba(Xt), getattr(est, "classes_", None), N_CLASSES)
    return proba[:, 1] if TASK == "binary" else proba


def get_importance(est, b, names):
    """Feature importance (gain jika tersedia) → pd.Series."""
    try:
        if b == "lgb":
            return pd.Series(est.booster_.feature_importance("gain"), index=est.booster_.feature_name())
        if b == "xgb":
            sc = est.get_booster().get_score(importance_type="total_gain")
            return pd.Series(sc, dtype=float).reindex(names).fillna(0.0)
        if b == "cat":
            return pd.Series(est.get_feature_importance(), index=names)
        if hasattr(est, "feature_importances_"):
            return pd.Series(est.feature_importances_, index=names)
        if hasattr(est, "coef_"):
            coef = np.abs(np.asarray(est.coef_))
            coef = coef.mean(axis=0) if coef.ndim > 1 else coef
            return pd.Series(coef, index=names)
    except Exception:  # noqa: BLE001
        pass
    return None


# ringkasan model yang akan dijalankan
_rows = []
for m in RUN.models:
    ok, why = model_available(m)
    _rows.append({"model": m, "available": ok, "mode": MODEL_MODES.get(base_model(m), "?"), "note": why})
display(pd.DataFrame(_rows))
print(f"Imbalance strategy: {RUN.imbalance} | GPU: {RUN.use_gpu}")

# %% [markdown]
# ---
# # 11. Training Loop (OOF + Test Predictions) & Leaderboard
#
# **Tujuan:** melatih setiap model di setiap fold (dan setiap repeat), mengumpulkan:
#
# - **OOF prediction**: prediksi untuk baris train yang **tidak** dipakai melatih model tsb → estimasi jujur performa.
# - **Test prediction**: rata-rata prediksi dari semua model fold (fold-averaging).
# - skor per fold, mean ± std, skor train (untuk deteksi overfitting), waktu, best iteration, feature importance.
#
# Semua disimpan di `RESULTS[nama_model]` dan di disk (`preds/oof_<model>.npy`, `preds/test_<model>.npy`) sehingga
# bisa dipakai lagi tanpa melatih ulang.
#
# **Cara membaca leaderboard:**
#
# | Kolom | Arti |
# |---|---|
# | `oof_score` | skor metric dihitung di seluruh OOF — **angka utama untuk membandingkan model** |
# | `fold_mean ± fold_std` | rata-rata dan sebaran skor antar fold — std besar = tidak stabil |
# | `train_score` | skor di data train fold (subsample) |
# | `gap` | seberapa jauh skor train lebih baik dari valid (positif besar = overfitting; ≤ 0 = tidak overfit) |
# | `best_iter` | rata-rata iterasi terbaik (early stopping) |
# | `time_s` | waktu total |
#
# **Jika X maka Y:**
#
# - Jika `gap` sangat besar (misal AUC train 0.99 vs valid 0.80) → overfit: naikkan regularisasi (`min_child_samples`, `reg_lambda`), kurangi `num_leaves`/`max_depth`, kurangi fitur noise.
# - Jika `fold_std` > 0.02 (AUC) / > 5% dari skor → tidak stabil: N_FOLDS/N_REPEATS lebih besar, atau model lebih sederhana.
# - Jika semua model ≈ dummy → fitur tidak informatif / bug target / leakage terbalik. Cek section 5–6.
# - Jika satu model gagal → lihat `FAILED_MODELS[nama]` (traceback) dan tabel Troubleshooting section 18.
# - Jika training terlalu lama → `RUN_MODE="fast"`, kurangi `MODELS`, `N_JOBS`, atau pakai GPU.
#
# **Untuk laporan:** tabel leaderboard (simpan `tables/leaderboard.csv`) + bar chart (figur `leaderboard`). Sebutkan
# baseline (dummy) sebagai pembanding.

# %%
RESULTS = OrderedDict()
KEEP_OBJ_MODELS = {"lgb", "xgb", "cat", "hgb", "linear", "enet"}


def run_cv_model(name, params_override=None, label=None, features=None, y_arr=None, splits=None, store=True,
                 keep_models=None, verbose=True, max_folds=None, trial=None, compute_test=True):
    """Cross-validation lengkap untuk satu model. Return dict hasil (dan simpan di RESULTS jika store=True)."""
    label = label or name
    b = base_model(name)
    feats = list(features) if features is not None else list(FEATURES)
    y_arr = y if y_arr is None else y_arr
    splits_all = CV_SPLITS if splits is None else splits
    mode = MODEL_MODES[b]
    cat_cols = [c for c in CAT_COLS if c in feats]
    num_cols = [c for c in NUM_COLS if c in feats]
    keep = (CFG.KEEP_FOLD_MODELS and (b in KEEP_OBJ_MODELS or len(X) < 50000)) if keep_models is None else keep_models
    n, K = len(y_arr), N_CLASSES
    oof_sum = np.zeros((n, K)) if TASK == "multiclass" else np.zeros(n)
    oof_cnt = np.zeros(n)
    test_sum = np.zeros((len(X_test), K)) if TASK == "multiclass" else np.zeros(len(X_test))
    fold_scores, train_scores, best_iters, imps, fold_objs = [], [], [], [], []
    t0, n_fit = time.time(), 0
    X_test_f = X_test[feats] if compute_test else None
    for r, splits_r in enumerate(splits_all):
        seed = RUN.seeds[r] if r < len(RUN.seeds) else CFG.SEED + r
        params = get_params(name, seed, params_override)
        for f, (tr, va) in enumerate(splits_r):
            if max_folds is not None and f >= max_folds:
                break
            prep = TabularPreprocessor(mode, num_cols, cat_cols, TASK, K, seed)
            ytr, yva = y_arr[tr], y_arr[va]
            Xt_tr = prep.fit_transform(X.iloc[tr][feats], ytr)
            Xt_va = prep.transform(X.iloc[va][feats])
            ytr_fit = y_forward(ytr) if TASK == "regression" else ytr
            yva_fit = y_forward(yva) if TASK == "regression" else yva
            sw = make_sample_weight(ytr)
            Xfit, yfit = Xt_tr, ytr_fit
            if RUN.imbalance == "smote" and TASK != "regression":
                Xfit, yfit, ok = apply_smote(Xt_tr, ytr_fit, prep, seed)
                sw = None if ok else compute_sample_weight("balanced", ytr)
            est, info = fit_one(name, params, Xfit, yfit, Xt_va, yva_fit, sw, prep)
            pva = predict_std(est, Xt_va)
            if TASK == "regression":
                pva = y_inverse(pva)
            oof_sum[va] += pva
            oof_cnt[va] += 1
            if compute_test:
                pte = predict_std(est, prep.transform(X_test_f))
                test_sum += y_inverse(pte) if TASK == "regression" else pte
            n_fit += 1
            fs = compute_metric(yva, pva)
            fold_scores.append(fs)
            ns = min(len(tr), 5000)
            sub = np.random.RandomState(seed).choice(len(tr), ns, replace=False)
            ptr = predict_std(est, Xt_tr.iloc[sub])
            ptr = y_inverse(ptr) if TASK == "regression" else ptr
            try:
                train_scores.append(compute_metric(ytr[sub], ptr))
            except Exception:  # noqa: BLE001
                train_scores.append(np.nan)
            best_iters.append(info.get("best_iter"))
            imp = get_importance(est, b, list(Xt_tr.columns))
            if imp is not None:
                imps.append(imp)
            if keep and r == 0:
                fold_objs.append({"est": est, "prep": prep, "va": va, "tr": tr, "fold": f})
            if verbose:
                bi = f" | best_iter={info.get('best_iter')}" if info.get("best_iter") else ""
                print(f"  [{label}] repeat {r} fold {f}: {RUN.metric}={fs:.5f}{bi}")
            if trial is not None:
                trial.report(float(np.mean(fold_scores)), step=n_fit - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    mask = oof_cnt > 0
    oof = np.full_like(oof_sum, np.nan, dtype=float)
    oof[mask] = (oof_sum[mask].T / oof_cnt[mask]).T
    score = compute_metric(y_arr[mask], oof[mask])
    res = {
        "name": label, "base": b, "mode": mode, "oof": oof, "test": test_sum / max(n_fit, 1) if compute_test else None,
        "score": score, "fold_scores": fold_scores, "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)), "train_mean": float(np.nanmean(train_scores)) if train_scores else np.nan,
        "time": time.time() - t0, "best_iters": best_iters, "params": get_params(name, RUN.seeds[0], params_override),
        "importance": (pd.concat(imps, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False) if imps else None),
        "fold_objs": fold_objs, "features": feats, "mask": mask,
    }
    if store:
        RESULTS[label] = res
        try:
            np.save(RUN.out / "preds" / f"oof_{label}.npy", oof)
            if compute_test:
                np.save(RUN.out / "preds" / f"test_{label}.npy", res["test"])
        except Exception as e:  # noqa: BLE001
            print("⚠️ gagal simpan prediksi:", e)
    if verbose:
        print(f"==> {label}: OOF {RUN.metric} = {score:.5f} | fold {res['fold_mean']:.5f} ± {res['fold_std']:.5f} | "
              f"train {res['train_mean']:.5f} | {res['time']:.1f}s")
    return res


def leaderboard():
    """Tabel ringkas semua model di RESULTS, diurutkan dari yang terbaik."""
    rows = []
    for k, r in RESULTS.items():
        bi = [v for v in r["best_iters"] if v is not None]
        rows.append({"model": k, "oof_score": r["score"], "fold_mean": r["fold_mean"], "fold_std": r["fold_std"],
                     "train_score": r["train_mean"],
                     "gap": (r["train_mean"] - r["fold_mean"]) * (1 if greater_is_better() else -1),
                     "best_iter": int(np.mean(bi)) if bi else None, "time_s": round(r["time"], 1)})
    lb = pd.DataFrame(rows)
    if len(lb):
        lb = lb.sort_values("oof_score", ascending=not greater_is_better()).reset_index(drop=True)
    return lb

# %%
banner("11. TRAINING")
for m in RUN.models:
    ok, why = model_available(m)
    if not ok:
        print(f"⏭️ skip {m}: {why}")
        continue
    print(f"\n--- {m} ---")
    try:
        run_cv_model(m)
    except Exception as e:  # noqa: BLE001
        FAILED_MODELS[m] = traceback.format_exc()
        print(f"❌ {m} gagal: {type(e).__name__}: {str(e)[:300]}")
        print("   Detail: print(FAILED_MODELS['" + m + "'])  | lihat tabel Troubleshooting section 18")
    gc.collect()
if not RESULTS:
    raise RuntimeError("Semua model gagal. Lihat FAILED_MODELS dan tabel Troubleshooting.")

# %%
LB = leaderboard()
display(LB)
save_table(LB, "leaderboard")
if FAILED_MODELS:
    print("❌ Model gagal:", list(FAILED_MODELS))
with safe_block("leaderboard plot"):
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(LB) + 1)))
    ax.barh(LB["model"][::-1], LB["oof_score"][::-1], xerr=LB["fold_std"][::-1], color="steelblue", capsize=3)
    for k, v in enumerate(LB["oof_score"][::-1]):
        ax.text(v, k, f" {v:.4f}", va="center", fontsize=8)
    lo = LB["oof_score"].min() - 2 * LB["fold_std"].max()
    hi = LB["oof_score"].max() + 2 * LB["fold_std"].max()
    pad = (hi - lo) * 0.15 + 1e-9
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(f"OOF {RUN.metric} ({'higher' if greater_is_better() else 'lower'} is better)")
    ax.set_title("Model leaderboard (error bar = fold std)")
    save_fig("leaderboard")
_best = LB.iloc[0]
print(f"🏆 Model tunggal terbaik: {_best['model']} ({RUN.metric}={_best['oof_score']:.5f})")
if "dummy" in RESULTS:
    d = RESULTS["dummy"]["score"]
    print(f"   Baseline dummy: {d:.5f} → peningkatan {abs(_best['oof_score'] - d):.5f}")
for _, r in LB.iterrows():
    if r["model"] != "dummy" and r["fold_std"] > 0.05 * max(abs(r["oof_score"]), 1e-9):
        print(f"⚠️ {r['model']}: fold std tinggi ({r['fold_std']:.4f}) → kurang stabil")

# %% [markdown]
# ---
# # 12. Hyperparameter Tuning (Optuna)
#
# **Tujuan:** mencari parameter LightGBM / XGBoost / CatBoost yang lebih baik dengan **fold CV yang sama** (supaya
# perbandingan adil). Optuna memakai TPE (Bayesian) — lebih efisien dari grid/random search — dan **pruning**
# (MedianPruner): trial yang skor fold awalnya buruk dihentikan lebih cepat.
#
# **Setting:** `USE_OPTUNA` (otomatis mati di fast mode), `OPTUNA_MODELS`, `OPTUNA_N_TRIALS`, `OPTUNA_TIMEOUT` (detik per
# model — batas keras), `OPTUNA_N_FOLDS` (pakai sebagian fold agar cepat). Model hasil tuning dilatih ulang dengan CV
# penuh dan masuk leaderboard sebagai `<model>_tuned`. Jika Optuna tidak ada → fallback `RandomizedSearchCV` (sklearn).
#
# **Cara membaca output:**
#
# - Plot *optimization history*: titik = skor tiap trial, garis = skor terbaik sejauh ini. Jika garis masih naik di akhir → tambah trial/timeout.
# - Plot *parameter importance*: parameter mana yang paling mempengaruhi skor (fokuskan tuning manual ke situ).
# - Bandingkan `<model>_tuned` vs `<model>` di leaderboard.
#
# **Jika X maka Y:**
#
# - Jika `_tuned` lebih buruk dari default → normal jika trial sedikit / data kecil (overfit ke fold). Tetap pakai default, atau naikkan `N_REPEATS`.
# - Jika kenaikan < std antar fold → kemungkinan noise; jangan terlalu percaya. Ensemble (section 13) biasanya lebih bermanfaat.
# - Jika waktu habis → turunkan `OPTUNA_N_FOLDS=3` dan tuning hanya `["lgb"]`.
# - Catatan: `OPTUNA_TIMEOUT` hanya dicek **di antara** trial — trial yang sedang berjalan tidak dipotong, dan training ulang
#   `<model>_tuned` dengan CV penuh dilakukan setelah timeout. CatBoost (terutama multiclass + kategori high-card) bisa sangat lambat.
# - Parameter terbaik tersimpan di `tables/best_params_<model>.json` → bisa dipakai ulang lewat `CFG.MODEL_PARAMS`
#   tanpa tuning lagi: `CFG.MODEL_PARAMS = {"lgb": {...}}`.
#
# **Untuk laporan:** sebutkan ruang pencarian, jumlah trial, dan peningkatan skor CV (default → tuned).

# %%
def suggest_params(trial, b):
    """Ruang pencarian Optuna per model."""
    if b == "lgb":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 8, 256, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True))
    if b == "xgb":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    min_child_weight=trial.suggest_float("min_child_weight", 1e-2, 50.0, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True))
    if b == "cat":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    depth=trial.suggest_int("depth", 4, 8),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                    random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0))
    raise ValueError(b)


def plot_study(study, b):
    """Plot optimization history & parameter importance (matplotlib, tanpa plotly)."""
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]
    if len(df) == 0:
        return
    vals = df["value"].to_numpy()
    best = np.maximum.accumulate(vals) if greater_is_better() else np.minimum.accumulate(vals)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(df["number"], vals, "o", alpha=0.6, label="trial")
    axes[0].plot(df["number"], best, "-", color="red", label="best so far")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel(f"CV {RUN.metric}")
    axes[0].set_title(f"Optuna optimization history — {b}")
    axes[0].legend()
    try:
        imp = optuna.importance.get_param_importances(study)
        axes[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="purple")
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Hyperparameter importance")
    except Exception:  # noqa: BLE001
        axes[1].axis("off")
    fig.tight_layout()
    save_fig(f"optuna_{b}")


TUNED_PARAMS = {}


def tune_optuna(b):
    """Tuning satu model dengan Optuna lalu latih ulang dengan CV penuh."""
    direction = "maximize" if greater_is_better() else "minimize"
    sampler = optuna.samplers.TPESampler(seed=CFG.SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    def objective(trial):
        params = suggest_params(trial, b)
        res = run_cv_model(b, params_override=params, label=f"{b}_trial", splits=CV_SPLITS[:1], store=False,
                           keep_models=False, verbose=False, max_folds=CFG.OPTUNA_N_FOLDS, trial=trial,
                           compute_test=False)
        return res["score"]

    t0 = time.time()
    study.optimize(objective, n_trials=CFG.OPTUNA_N_TRIALS, timeout=CFG.OPTUNA_TIMEOUT, catch=(Exception,),
                   show_progress_bar=False)
    n_done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_pruned = len([t for t in study.trials if t.state.name == "PRUNED"])
    if n_done == 0:
        print(f"⚠️ Tidak ada trial {b} yang selesai.")
        return None
    print(f"{b}: {n_done} trial selesai, {n_pruned} di-prune, {time.time() - t0:.0f}s | best {RUN.metric} = {study.best_value:.5f}")
    print("best params:", study.best_params)
    TUNED_PARAMS[b] = study.best_params
    with open(RUN.out / "tables" / f"best_params_{b}.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    with safe_block("plot optuna"):
        plot_study(study, b)
    return run_cv_model(b, params_override=study.best_params, label=f"{b}_tuned")


def tune_random_search(b):
    """Fallback tanpa Optuna: RandomizedSearchCV dengan pipeline (preprocessor + model), tanpa early stopping."""
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from scipy.stats import loguniform, uniform, randint
    if b == "cat":
        print("ℹ️ Fallback random search tidak mendukung CatBoost (butuh cat_features) → skip.")
        return None
    seed = CFG.SEED
    if b == "lgb":
        est = (lgb.LGBMClassifier if TASK != "regression" else lgb.LGBMRegressor)(
            n_estimators=500, learning_rate=0.05, verbose=-1, random_state=seed, n_jobs=CFG.N_JOBS)
        dists = {"model__num_leaves": randint(8, 128), "model__min_child_samples": randint(5, 100),
                 "model__subsample": uniform(0.5, 0.5), "model__subsample_freq": [1],
                 "model__colsample_bytree": uniform(0.3, 0.7), "model__reg_lambda": loguniform(1e-3, 10)}
    elif b == "xgb":
        est = (xgb.XGBClassifier if TASK != "regression" else xgb.XGBRegressor)(
            n_estimators=500, learning_rate=0.05, tree_method="hist", random_state=seed, n_jobs=CFG.N_JOBS)
        dists = {"model__max_depth": randint(3, 10), "model__min_child_weight": loguniform(0.1, 20),
                 "model__subsample": uniform(0.5, 0.5), "model__colsample_bytree": uniform(0.3, 0.7),
                 "model__reg_lambda": loguniform(1e-3, 10)}
    else:
        return None
    mode = "ordinal"
    pipe = Pipeline([("prep", TabularPreprocessor(mode, NUM_COLS, CAT_COLS, TASK, N_CLASSES, seed)), ("model", est)])

    def scorer(estimator, Xv, yv):
        p = predict_std(estimator, Xv)
        if TASK == "regression":
            p, yv = y_inverse(p), y_inverse(np.asarray(yv))
        s = compute_metric(np.asarray(yv), p)
        return s if greater_is_better() else -s

    y_fit = y_forward(y) if TASK == "regression" else y
    rs = RandomizedSearchCV(pipe, dists, n_iter=min(CFG.OPTUNA_N_TRIALS, 20), cv=CV_SPLITS[0], scoring=scorer,
                            random_state=seed, n_jobs=1, refit=False)
    rs.fit(X[FEATURES], y_fit)
    best = {k.replace("model__", ""): v for k, v in rs.best_params_.items()}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    print(f"{b} random search best: {best} | CV {abs(rs.best_score_):.5f}")
    TUNED_PARAMS[b] = best
    return run_cv_model(b, params_override=best, label=f"{b}_tuned")

# %%
if RUN.optuna:
    banner("12. TUNING")
    for b in CFG.OPTUNA_MODELS:
        ok, why = model_available(b)
        if not ok:
            print(f"⏭️ skip tuning {b}: {why}")
            continue
        try:
            if HAS_OPTUNA:
                tune_optuna(b)
            else:
                print("ℹ️ Optuna tidak tersedia → RandomizedSearchCV")
                tune_random_search(b)
        except Exception as e:  # noqa: BLE001
            FAILED_MODELS[f"{b}_tuned"] = traceback.format_exc()
            print(f"❌ tuning {b} gagal: {type(e).__name__}: {str(e)[:300]}")
    LB = leaderboard()
    display(LB)
    save_table(LB, "leaderboard")
else:
    print("ℹ️ Tuning dimatikan (USE_OPTUNA=False atau fast mode). Untuk mengaktifkan: CFG.USE_OPTUNA=True & RUN_MODE='full'.")

# %% [markdown]
# ---
# # 13. Ensembling
#
# **Tujuan:** menggabungkan prediksi beberapa model. Model yang berbeda membuat kesalahan yang berbeda, sehingga
# rata-ratanya biasanya lebih akurat dan lebih stabil dari model tunggal terbaik.
#
# | Metode | Cara kerja | Kelebihan / risiko |
# |---|---|---|
# | `mean` | rata-rata sederhana | paling robust, hampir tidak bisa overfit |
# | `rank` | rata-rata **ranking** prediksi (binary + metric AUC/Gini/PR-AUC) | cocok untuk AUC karena skala probabilitas antar model berbeda |
# | `weighted` | bobot non-negatif berjumlah 1, dioptimasi di OOF (`scipy.optimize.minimize`: SLSQP + Powell) | bisa lebih baik, tapi bobot bisa overfit OOF |
# | `hill` | hill-climbing greedy (Caruana): mulai dari model terbaik, tambah model (boleh berulang) yang paling menaikkan skor | sering terbaik di lomba, otomatis membuang model jelek |
# | `stack` | meta-model (Logistic/Ridge) dilatih pada prediksi OOF, dengan CV-nya sendiri | menangkap kombinasi, risiko overfit paling besar |
#
# Model yang lebih buruk dari `dummy` dan model di `ENSEMBLE_EXCLUDE` tidak diikutkan. Pemenang dipilih berdasarkan skor
# OOF (atau dipaksa via `ENSEMBLE_FORCE`).
#
# **⚠️ Peringatan overfitting bobot:** bobot `weighted`/`hill`/`stack` dipilih **menggunakan OOF yang sama** dengan yang
# dipakai untuk menilai → skornya sedikit optimis. Notebook menjalankan **cek holdout**: bobot dioptimasi di separuh OOF lalu
# dinilai di separuh lainnya. Jika di separuh lain weighted tidak mengalahkan mean → pilih `mean` (lebih aman).
#
# **Cara membaca output:** tabel metode vs skor OOF, bobot tiap model, dan heatmap **korelasi prediksi antar model**
# (korelasi rendah = lebih beragam = ensemble lebih bermanfaat).
#
# **Jika X maka Y:**
#
# - Jika ensemble terbaik hanya unggul < 0.001 dari model tunggal → keuntungan kecil; pilih yang lebih sederhana jika ragu.
# - Jika korelasi antar model > 0.98 → model terlalu mirip; tambahkan model berbeda (linear, CatBoost, KNN) atau fitur berbeda.
# - Jika stacking jauh lebih bagus dari semua metode lain (lompatan besar) → curigai leakage di OOF.
#
# **Untuk laporan:** tabel perbandingan metode ensemble + bobot akhir. Jelaskan alasan memilih metode final.

# %%
def blend(preds, w):
    """Weighted average prediksi (multiclass: dinormalisasi per baris)."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    out = sum(wi * p for wi, p in zip(w, preds))
    if TASK == "multiclass":
        out = out / out.sum(axis=1, keepdims=True)
    return out


def rank_norm(p):
    """Ranking dinormalisasi 0..1 (untuk rank averaging)."""
    return stats.rankdata(p) / len(p)


def optimize_weights(preds, y_true):
    """Bobot non-negatif berjumlah 1 yang meminimalkan metric_loss. Coba SLSQP & Powell (softplus, untuk metric tidak mulus)."""
    k = len(preds)
    loss = lambda w: metric_loss(y_true, blend(preds, np.clip(w, 0, None) + 1e-12))  # noqa: E731
    best_w, best_l = np.ones(k) / k, loss(np.ones(k) / k)
    try:
        r = minimize(loss, np.ones(k) / k, method="SLSQP", bounds=[(0, 1)] * k,
                     constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1},), options={"maxiter": 200})
        if r.fun < best_l:
            best_w, best_l = np.clip(r.x, 0, None), r.fun
    except Exception:  # noqa: BLE001
        pass
    try:
        r = minimize(lambda z: loss(np.log1p(np.exp(z))), np.ones(k), method="Powell",
                     options={"maxfev": 200 * k, "xtol": 1e-3, "ftol": 1e-7})
        w = np.log1p(np.exp(r.x))
        if r.fun < best_l:
            best_w, best_l = w, r.fun
    except Exception:  # noqa: BLE001
        pass
    best_w = np.clip(best_w, 0, None)
    return best_w / best_w.sum()


def hill_climb(preds, y_true, max_iters=50):
    """Ensemble selection Caruana (dengan pengulangan). Return bobot."""
    k = len(preds)
    scores = [metric_loss(y_true, p) for p in preds]
    counts = np.zeros(k)
    counts[int(np.argmin(scores))] = 1
    cur = metric_loss(y_true, blend(preds, counts))
    for _ in range(max_iters):
        best_j, best_l = None, cur
        for j in range(k):
            c = counts.copy()
            c[j] += 1
            l_ = metric_loss(y_true, blend(preds, c))
            if l_ < best_l - 1e-9:
                best_j, best_l = j, l_
        if best_j is None:
            break
        counts[best_j] += 1
        cur = best_l
    return counts / counts.sum()


def stack_features(preds):
    """Fitur untuk meta-model: logit (binary), log-prob (multiclass), nilai (regresi)."""
    if TASK == "binary":
        P = np.column_stack([np.clip(p, 1e-6, 1 - 1e-6) for p in preds])
        return np.log(P / (1 - P))
    if TASK == "multiclass":
        return np.hstack([np.log(np.clip(p, 1e-6, 1)) for p in preds])
    return np.column_stack(preds)


def make_meta():
    """Meta-model sederhana (regularisasi kuat → tidak mudah overfit)."""
    if TASK == "regression":
        return Ridge(alpha=1.0)
    return LogisticRegression(C=0.5, max_iter=3000)


def meta_predict(meta, F):
    if TASK == "regression":
        return meta.predict(F)
    pr = align_proba(meta.predict_proba(F), meta.classes_, N_CLASSES)
    return pr[:, 1] if TASK == "binary" else pr


def apply_recipe(recipe, test_preds):
    """Terapkan resep ensemble ke dict {model: test_pred}. Dipakai ulang saat refit_full."""
    P = [test_preds[m] for m in recipe["models"]]
    meth = recipe["method"]
    if meth == "single":
        return P[0]
    if meth == "rank":
        return blend([rank_norm(p) for p in P], recipe["weights"])
    if meth == "stack":
        return meta_predict(recipe["meta"], stack_features(P))
    return blend(P, recipe["weights"])


banner("13. ENSEMBLING")
_dummy_score = RESULTS["dummy"]["score"] if "dummy" in RESULTS else None
ENS_MODELS = [k for k, r in RESULTS.items()
              if base_model(k) not in CFG.ENSEMBLE_EXCLUDE and k not in CFG.ENSEMBLE_EXCLUDE
              and r["test"] is not None and not k.endswith("_fs")
              and (_dummy_score is None or is_better(r["score"], _dummy_score))]
MASK = np.all([RESULTS[k]["mask"] for k in ENS_MODELS], axis=0) if ENS_MODELS else OOF_MASK
yM = y[MASK]
ENSEMBLE_ROWS, RECIPES, ENS_OOF, ENS_TEST = [], {}, {}, {}
best_single = LB[LB["model"].isin(ENS_MODELS)].iloc[0]["model"] if ENS_MODELS else LB.iloc[0]["model"]
RECIPES["single:" + best_single] = {"method": "single", "models": [best_single], "weights": [1.0]}
ENS_OOF["single:" + best_single] = RESULTS[best_single]["oof"]
ENS_TEST["single:" + best_single] = RESULTS[best_single]["test"]
ENSEMBLE_ROWS.append({"method": "single:" + best_single, "oof_score": RESULTS[best_single]["score"],
                      "detail": "model tunggal terbaik"})

if len(ENS_MODELS) >= 2:
    print(f"Model untuk ensemble ({len(ENS_MODELS)}): {ENS_MODELS}")
    P_oof = [RESULTS[k]["oof"][MASK] for k in ENS_MODELS]
    P_test = {k: RESULTS[k]["test"] for k in ENS_MODELS}

    def _register(meth, w, oof_m, detail, meta=None):
        full_oof = np.full_like(RESULTS[ENS_MODELS[0]]["oof"], np.nan, dtype=float)
        full_oof[MASK] = oof_m
        RECIPES[meth] = {"method": meth, "models": list(ENS_MODELS), "weights": None if w is None else list(w), "meta": meta}
        ENS_OOF[meth] = full_oof
        ENS_TEST[meth] = apply_recipe(RECIPES[meth], P_test)
        ENSEMBLE_ROWS.append({"method": meth, "oof_score": compute_metric(yM, oof_m), "detail": detail})

    eq = np.ones(len(ENS_MODELS)) / len(ENS_MODELS)
    if "mean" in CFG.ENSEMBLE_METHODS:
        _register("mean", eq, blend(P_oof, eq), "bobot sama")
    if "rank" in CFG.ENSEMBLE_METHODS and TASK == "binary" and RUN.metric in ("roc_auc", "gini", "pr_auc"):
        _register("rank", eq, blend([rank_norm(p) for p in P_oof], eq), "rata-rata ranking")
    if "weighted" in CFG.ENSEMBLE_METHODS:
        with safe_block("weighted"):
            w = optimize_weights(P_oof, yM)
            _register("weighted", w, blend(P_oof, w), ", ".join(f"{m}={v:.2f}" for m, v in zip(ENS_MODELS, w) if v > 0.005))
    if "hill" in CFG.ENSEMBLE_METHODS:
        with safe_block("hill climbing"):
            w = hill_climb(P_oof, yM, CFG.HILL_MAX_ITERS)
            _register("hill", w, blend(P_oof, w), ", ".join(f"{m}={v:.2f}" for m, v in zip(ENS_MODELS, w) if v > 0))
    if "stack" in CFG.ENSEMBLE_METHODS:
        with safe_block("stacking"):
            F = stack_features(P_oof)
            if MASK.all():
                sp = CV_SPLITS[0]
            else:
                sp = list(KFold(5, shuffle=True, random_state=CFG.SEED).split(F))
            oof_meta = np.zeros((len(F), N_CLASSES)) if TASK == "multiclass" else np.zeros(len(F))
            for tr, va in sp:
                mm = make_meta().fit(F[tr], yM[tr])
                oof_meta[va] = meta_predict(mm, F[va])
            meta_full = make_meta().fit(F, yM)
            _register("stack", None, oof_meta, f"meta={type(meta_full).__name__}", meta=meta_full)

ENSEMBLE_TABLE = pd.DataFrame(ENSEMBLE_ROWS).sort_values("oof_score", ascending=not greater_is_better()).reset_index(drop=True)
display(ENSEMBLE_TABLE)
save_table(ENSEMBLE_TABLE, "ensemble_comparison")

# %%
# ---- cek overfitting bobot (holdout separuh OOF) ----
with safe_block("holdout check"):
    if len(ENS_MODELS) >= 2 and "weighted" in RECIPES:
        idxM = np.arange(MASK.sum())
        strat = yM if TASK != "regression" else pd.qcut(pd.Series(yM).rank(method="first"), 5, labels=False)
        a_idx, b_idx = next(StratifiedKFold(2, shuffle=True, random_state=CFG.SEED).split(idxM, strat))
        wA = optimize_weights([p[a_idx] for p in P_oof], yM[a_idx])
        s_w = compute_metric(yM[b_idx], blend([p[b_idx] for p in P_oof], wA))
        s_m = compute_metric(yM[b_idx], blend([p[b_idx] for p in P_oof], eq))
        print(f"Holdout check: weighted (bobot dari separuh A) di separuh B = {s_w:.5f} | mean di separuh B = {s_m:.5f}")
        HOLDOUT_WEIGHTED_OK = not is_better(s_m, s_w)
        print("✅ bobot tergeneralisasi" if HOLDOUT_WEIGHTED_OK else "⚠️ bobot tidak tergeneralisasi → mean lebih aman")

# ---- pilih final ----
if CFG.ENSEMBLE_FORCE:
    key = CFG.ENSEMBLE_FORCE if CFG.ENSEMBLE_FORCE in RECIPES else (
        "single:" + CFG.ENSEMBLE_FORCE if CFG.ENSEMBLE_FORCE in RESULTS else None)
    if key is None:
        print(f"⚠️ ENSEMBLE_FORCE='{CFG.ENSEMBLE_FORCE}' tidak dikenal → pilih terbaik")
        key = ENSEMBLE_TABLE.iloc[0]["method"]
    elif key.startswith("single:") and key not in RECIPES:
        RECIPES[key] = {"method": "single", "models": [CFG.ENSEMBLE_FORCE], "weights": [1.0]}
        ENS_OOF[key], ENS_TEST[key] = RESULTS[CFG.ENSEMBLE_FORCE]["oof"], RESULTS[CFG.ENSEMBLE_FORCE]["test"]
else:
    key = ENSEMBLE_TABLE.iloc[0]["method"]
    if key in ("weighted", "hill", "stack") and "mean" in RECIPES and not globals().get("HOLDOUT_WEIGHTED_OK", True):
        s_key = ENSEMBLE_TABLE.set_index("method").loc[key, "oof_score"]
        s_mean = ENSEMBLE_TABLE.set_index("method").loc["mean", "oof_score"]
        if abs(s_key - s_mean) < 0.002 * max(abs(s_mean), 1e-9):
            print(f"ℹ️ {key} hanya unggul tipis & holdout gagal → memilih 'mean' (lebih robust)")
            key = "mean"
FINAL_NAME = key
FINAL_RECIPE = RECIPES[key]
FINAL_OOF = ENS_OOF[key]
FINAL_TEST = ENS_TEST[key]
FINAL_RAW_SCORE = compute_metric(y[MASK], FINAL_OOF[MASK])
print(f"\n🏁 FINAL (sebelum post-processing): {FINAL_NAME} | OOF {RUN.metric} = {FINAL_RAW_SCORE:.5f}")

with safe_block("ensemble plots"):
    if len(ENS_MODELS) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.4 * len(ENS_MODELS) + 2)))
        et = ENSEMBLE_TABLE.iloc[::-1]
        axes[0].barh(et["method"], et["oof_score"], color=["crimson" if m == FINAL_NAME else "grey" for m in et["method"]])
        rng_ = et["oof_score"].max() - et["oof_score"].min()
        axes[0].set_xlim(et["oof_score"].min() - rng_ * 0.5 - 1e-6, et["oof_score"].max() + rng_ * 0.5 + 1e-6)
        axes[0].set_xlabel(f"OOF {RUN.metric}")
        axes[0].set_title("Ensemble methods (red = selected)")
        cm_in = np.column_stack([p if p.ndim == 1 else p[:, -1] for p in P_oof])
        cc = pd.DataFrame(cm_in, columns=ENS_MODELS).corr(method="spearman")
        im = axes[1].imshow(cc.values, cmap="viridis", vmin=max(0, cc.values.min()), vmax=1)
        axes[1].set_xticks(range(len(cc)))
        axes[1].set_xticklabels(cc.columns, rotation=90)
        axes[1].set_yticks(range(len(cc)))
        axes[1].set_yticklabels(cc.columns)
        for i in range(len(cc)):
            for j in range(len(cc)):
                axes[1].text(j, i, f"{cc.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
        axes[1].set_title("Correlation of OOF predictions (diversity)")
        axes[1].grid(False)
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        fig.tight_layout()
        save_fig("ensemble_comparison")

# %% [markdown]
# ---
# # 14. Post-processing
#
# **Tujuan:** menyesuaikan prediksi final dengan **metric** lomba. Semua dioptimasi di OOF (bukan di test!).
#
# | Task | Teknik | Kapan aktif |
# |---|---|---|
# | Binary | **Threshold optimization** (grid 0.01–0.99) | metric berbasis label (F1, accuracy, MCC, balanced acc, kappa) atau `SUBMISSION_MODE="label"` |
# | Binary | **Calibration** (isotonic / sigmoid-Platt) + reliability diagram + Brier | `CALIBRATION` aktif; diterapkan ke test hanya jika logloss/Brier membaik dan metric = logloss (atau dipaksa `"isotonic"/"sigmoid"`) |
# | Multiclass | **Class prior adjustment** (bobot per kelas pada probabilitas sebelum argmax) | metric berbasis label (F1-macro sering naik banyak!) |
# | Multiclass | **Temperature scaling** | metric logloss |
# | Regresi | **Clipping** ke range target train | `REG_CLIP=True` |
# | Regresi | **Rounding** (target integer) / **cut-point optimization** (metric QWK/kappa/accuracy) | target integer |
#
# **Cara membaca output:** setiap teknik mencetak skor OOF sebelum → sesudah. Teknik hanya dipakai jika skor membaik.
#
# **Jika X maka Y:**
#
# - Jika threshold optimal jauh dari 0.5 (misal 0.25) → wajar untuk data imbalanced dan metric F1.
# - Jika threshold optimal per fold sangat bervariasi (std > 0.1) → threshold tidak stabil; pertimbangkan nilai tengah / jangan terlalu dipercaya.
# - Jika metric AUC → threshold & calibration **tidak** mengubah skor (AUC hanya peduli urutan) → submit probabilitas mentah.
# - Jika kenaikan dari post-processing sangat kecil (< 0.001) → noise, boleh diabaikan.
#
# **Untuk laporan:** sebutkan threshold akhir (dan alasannya), atau bobot kelas, serta kenaikan skor OOF-nya.

# %%
from sklearn.calibration import calibration_curve

banner("14. POST-PROCESSING")
PP = {"threshold": 0.5, "class_weights": None, "calibrator": None, "temperature": None, "clip": None,
      "round": False, "cutpoints": None}
yM = y[MASK]
oofM = FINAL_OOF[MASK]
THR_METRIC = RUN.metric if RUN.metric in LABEL_METRICS else "f1"


def tune_threshold(p, yt, metric=None):
    """Cari threshold terbaik di OOF (binary). Return (thr, score, kurva)."""
    metric = metric or THR_METRIC
    grid = np.linspace(0.01, 0.99, 197)
    sc = np.array([compute_metric(yt, p, metric=metric, threshold=t) for t in grid])
    i = int(np.argmax(sc) if greater_is_better(metric) else np.argmin(sc))
    return float(grid[i]), float(sc[i]), (grid, sc)


def tune_class_weights(P, yt, metric=None, n_rounds=3):
    """Coordinate ascent bobot kelas untuk argmax(P * w) (multiclass)."""
    metric = metric or RUN.metric
    K = P.shape[1]
    w = np.ones(K)
    best = compute_metric(yt, P, metric=metric, class_weights=w)
    grid = np.exp(np.linspace(np.log(0.33), np.log(3.0), 25))
    for _ in range(n_rounds):
        for k in range(K):
            for g in grid:
                w2 = w.copy()
                w2[k] = g
                s = compute_metric(yt, P, metric=metric, class_weights=w2)
                if is_better(s, best, metric) and abs(s - best) > 1e-9:
                    best, w = s, w2
    return w / w.mean(), best


def optimize_cutpoints(pred, yt, metric):
    """Optimasi batas pembulatan (regresi ordinal, metric QWK/kappa/accuracy)."""
    levels = np.sort(np.unique(np.round(yt).astype(int)))
    cuts = (levels[:-1] + 0.5).astype(float)

    def apply(c):
        return levels[np.clip(np.searchsorted(np.sort(c), pred), 0, len(levels) - 1)].astype(float)

    best = compute_metric(yt, apply(cuts), metric=metric)
    for _ in range(2):
        for i in range(len(cuts)):
            lo = cuts[i - 1] if i > 0 else levels[0] - 0.5
            hi = cuts[i + 1] if i < len(cuts) - 1 else levels[-1] + 0.5
            for v in np.linspace(lo, hi, 21)[1:-1]:
                c = cuts.copy()
                c[i] = v
                s = compute_metric(yt, apply(c), metric=metric)
                if is_better(s, best, metric) and abs(s - best) > 1e-9:
                    best, cuts = s, c
    return np.sort(cuts), levels, best


def postprocess(pred, state=PP):
    """Terapkan post-processing (selain threshold/class weight yang dipakai saat konversi label)."""
    p = np.asarray(pred, dtype=float).copy()
    if TASK == "binary" and state["calibrator"] is not None:
        kind, cal = state["calibrator"]
        if kind == "isotonic":
            p = cal.predict(p)
        else:
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            p = cal.predict_proba(z.reshape(-1, 1))[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
    if TASK == "multiclass" and state["temperature"] is not None:
        lg = np.log(np.clip(p, 1e-12, 1)) / state["temperature"]
        lg -= lg.max(axis=1, keepdims=True)
        p = np.exp(lg) / np.exp(lg).sum(axis=1, keepdims=True)
    if TASK == "regression":
        if state["clip"] is not None:
            p = np.clip(p, *state["clip"])
        if state["cutpoints"] is not None:
            cuts, levels = state["cutpoints"]
            p = levels[np.clip(np.searchsorted(cuts, p), 0, len(levels) - 1)].astype(float)
        elif state["round"]:
            p = np.round(p)
    return p


CAL_OOF_CV = None   # OOF terkalibrasi via CV (jujur, bukan in-sample)


def oof_after_pp():
    """OOF setelah post-processing. Jika calibration aktif, pakai versi cross-validated agar skor tidak optimis."""
    if TASK == "binary" and PP["calibrator"] is not None and CAL_OOF_CV is not None:
        return CAL_OOF_CV.copy()
    return postprocess(oofM)


def pp_score(pred_m, state=PP):
    """Skor OOF setelah post-processing."""
    return compute_metric(yM, postprocess(pred_m, state), threshold=state["threshold"], class_weights=state["class_weights"])


score_before = compute_metric(yM, oofM)
print(f"Skor OOF sebelum post-processing: {score_before:.5f}")

# ---------------- BINARY ----------------
if TASK == "binary":
    with safe_block("calibration"):
        if CFG.CALIBRATION:
            z = np.log(np.clip(oofM, 1e-6, 1 - 1e-6) / (1 - np.clip(oofM, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            sp = list(StratifiedKFold(5, shuffle=True, random_state=CFG.SEED).split(z, yM))
            cal_oof = {"isotonic": np.zeros(len(yM)), "sigmoid": np.zeros(len(yM))}
            for tr, va in sp:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oofM[tr], yM[tr])
                cal_oof["isotonic"][va] = iso.predict(oofM[va])
                pl = LogisticRegression(C=100, max_iter=1000).fit(z[tr], yM[tr])
                cal_oof["sigmoid"][va] = pl.predict_proba(z[va])[:, 1]
            rows = [{"method": "raw", "brier": brier_score_loss(yM, oofM), "logloss": log_loss(yM, np.clip(oofM, 1e-6, 1 - 1e-6))}]
            for k, v in cal_oof.items():
                v = np.clip(v, 1e-6, 1 - 1e-6)
                rows.append({"method": k, "brier": brier_score_loss(yM, v), "logloss": log_loss(yM, v)})
            CAL_TABLE = pd.DataFrame(rows)
            display(CAL_TABLE.round(5))
            save_table(CAL_TABLE, "calibration_comparison")
            best_cal = CAL_TABLE.sort_values("logloss").iloc[0]["method"]
            force = CFG.CALIBRATION in ("isotonic", "sigmoid")
            chosen = CFG.CALIBRATION if force else best_cal
            if chosen != "raw" and (force or RUN.metric == "logloss"):
                if chosen == "isotonic":
                    PP["calibrator"] = ("isotonic", IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oofM, yM))
                else:
                    PP["calibrator"] = ("sigmoid", LogisticRegression(C=100, max_iter=1000).fit(z, yM))
                CAL_OOF_CV = np.clip(cal_oof[chosen], 1e-6, 1 - 1e-6)
                print(f"✅ Calibration '{chosen}' diterapkan (OOF calibrated via CV).")
            else:
                print(f"ℹ️ Calibration terbaik: {best_cal}. Tidak diterapkan (metric {RUN.metric} tidak bergantung kalibrasi"
                      f" atau raw sudah terbaik). Set CALIBRATION='isotonic' untuk memaksa.")
            fig, ax = plt.subplots(figsize=(6, 5))
            for nm, pv in [("raw", oofM)] + list(cal_oof.items()):
                fr, mp = calibration_curve(yM, np.clip(pv, 0, 1), n_bins=10, strategy="quantile")
                ax.plot(mp, fr, "o-", label=nm)
            ax.plot([0, 1], [0, 1], "k--", label="perfect")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title("Reliability diagram (OOF)")
            ax.legend()
            save_fig("calibration_reliability")
    with safe_block("threshold tuning"):
        need_thr = CFG.THRESHOLD_TUNING and (RUN.metric in LABEL_METRICS or RUN.sub_mode == "label")
        if need_thr:
            pcal = oof_after_pp()
            pcal_full = np.full(len(y), np.nan)
            pcal_full[MASK] = pcal
            s05 = compute_metric(yM, pcal, metric=THR_METRIC, threshold=0.5)
            thr, s_thr, (grid, sc) = tune_threshold(pcal, yM)
            fold_thr = []
            for tr, va in CV_SPLITS[0]:
                mv = MASK[va]
                if mv.sum() > 10:
                    fold_thr.append(tune_threshold(pcal_full[va][mv], y[va][mv])[0])
            print(f"Threshold 0.5 → {THR_METRIC}={s05:.5f} | threshold {thr:.3f} → {THR_METRIC}={s_thr:.5f}")
            print(f"Threshold optimal per fold: {np.round(fold_thr, 3)} (std {np.std(fold_thr):.3f})")
            if np.std(fold_thr) > 0.1:
                print("⚠️ Threshold tidak stabil antar fold → hati-hati, pertimbangkan median per-fold.")
            if is_better(s_thr, s05, THR_METRIC):
                PP["threshold"] = thr
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(grid, sc)
            ax.axvline(PP["threshold"], color="red", ls="--", label=f"chosen {PP['threshold']:.3f}")
            ax.axvline(0.5, color="grey", ls=":", label="0.5")
            ax.set_xlabel("Threshold")
            ax.set_ylabel(f"OOF {THR_METRIC}")
            ax.set_title("Threshold optimization (OOF)")
            ax.legend()
            save_fig("threshold_optimization")
        else:
            print(f"ℹ️ Threshold tuning tidak diperlukan (metric {RUN.metric}, submission {RUN.sub_mode}).")

# ---------------- MULTICLASS ----------------
if TASK == "multiclass":
    with safe_block("class prior adjust"):
        if CFG.CLASS_PRIOR_ADJUST and RUN.metric in LABEL_METRICS:
            w, s_new = tune_class_weights(oofM, yM)
            print(f"Class weights: {dict(zip([ID2LABEL[k] for k in range(N_CLASSES)], np.round(w, 3)))}")
            print(f"{RUN.metric}: {score_before:.5f} → {s_new:.5f}")
            if is_better(s_new, score_before):
                PP["class_weights"] = w
        else:
            print("ℹ️ Class prior adjust tidak dipakai (metric berbasis probabilitas atau dimatikan).")
    with safe_block("temperature scaling"):
        if CFG.CALIBRATION and RUN.metric == "logloss":
            from scipy.optimize import minimize_scalar
            lg0 = np.log(np.clip(oofM, 1e-12, 1))

            def _ll(T):
                lg = lg0 / T
                lg = lg - lg.max(axis=1, keepdims=True)
                pr = np.exp(lg) / np.exp(lg).sum(axis=1, keepdims=True)
                return log_loss(yM, pr, labels=list(range(N_CLASSES)))

            r = minimize_scalar(_ll, bounds=(0.3, 5.0), method="bounded")
            print(f"Temperature = {r.x:.3f} | logloss {_ll(1.0):.5f} → {r.fun:.5f}")
            if r.fun < _ll(1.0) - 1e-5:
                PP["temperature"] = float(r.x)

# ---------------- REGRESSION ----------------
if TASK == "regression":
    with safe_block("regression post-processing"):
        if CFG.REG_CLIP:
            st = dict(PP, clip=(float(y.min()), float(y.max())))
            s_clip = pp_score(oofM, st)
            print(f"Clip ke [{y.min():.4g}, {y.max():.4g}]: {score_before:.5f} → {s_clip:.5f}")
            if not is_better(score_before, s_clip):
                PP["clip"] = st["clip"]
        is_int = np.allclose(y, np.round(y))
        if is_int and RUN.metric in ("qwk", "cohen_kappa", "accuracy"):
            cuts, levels, s_cut = optimize_cutpoints(postprocess(oofM), yM, RUN.metric)
            s_rnd = pp_score(oofM, dict(PP, round=True))
            print(f"Rounding biasa: {s_rnd:.5f} | cut-point optimal: {s_cut:.5f} (cuts={np.round(cuts, 3)})")
            if is_better(s_cut, s_rnd):
                PP["cutpoints"] = (cuts, levels)
            else:
                PP["round"] = True
        elif is_int and CFG.REG_ROUND in (True, "auto"):
            s_r = pp_score(oofM, dict(PP, round=True))
            print(f"Rounding (target integer): {pp_score(oofM):.5f} → {s_r:.5f}")
            if CFG.REG_ROUND is True or is_better(s_r, pp_score(oofM)):
                PP["round"] = True

FINAL_OOF_PP = np.full_like(FINAL_OOF, np.nan, dtype=float)
FINAL_OOF_PP[MASK] = oof_after_pp()
FINAL_TEST_PP = postprocess(FINAL_TEST)
FINAL_SCORE = compute_metric(yM, FINAL_OOF_PP[MASK], threshold=PP["threshold"], class_weights=PP["class_weights"])
print(f"\n🏁 Skor OOF final: {score_before:.5f} → {FINAL_SCORE:.5f} (setelah post-processing) | state: "
      f"threshold={PP['threshold']}, class_weights={None if PP['class_weights'] is None else np.round(PP['class_weights'], 3)}, "
      f"calibrator={None if PP['calibrator'] is None else PP['calibrator'][0]}, temperature={PP['temperature']}, "
      f"clip={PP['clip']}, round={PP['round']}, cutpoints={'yes' if PP['cutpoints'] is not None else None}")

# %% [markdown]
# ---
# # 15. Evaluasi Mendalam & Debugging
#
# **Tujuan:** memahami **di mana** model benar/salah, mendiagnosis bias vs variance, dan memastikan tidak ada bug/leakage.
# Semua evaluasi memakai prediksi OOF final (setelah post-processing).
#
# | No | Analisis | Pertanyaan |
# |---|---|---|
# | 15.1 | Confusion matrix, classification report, ROC & PR curve, calibration / residual plots, QQ plot | seberapa baik & kesalahan jenis apa? |
# | 15.2 | Learning curve | lebih banyak data membantu? (bias vs variance) |
# | 15.3 | Train-vs-valid gap & fold stability | overfit? stabil? |
# | 15.4 | Shuffled-target sanity test | skor dengan target acak harus ≈ chance — jika tidak, ada **bug/leakage** |
# | 15.5 | OOF vs test prediction distribution | distribusi prediksi test mirip OOF? |
# | 15.6 | Error analysis | baris & segmen mana yang paling sering salah? |
#
# **Jika X maka Y:**
#
# - Confusion matrix: satu kelas sering tertukar dengan kelas lain → buat fitur yang membedakan keduanya / class weight / prior adjust.
# - Recall kelas minoritas rendah → threshold lebih rendah (binary) atau class weight.
# - Residual membentuk pola (corong/kurva) vs prediksi → model kurang menangkap non-linearitas / heteroskedastis → coba transform target (log1p).
# - QQ plot ekor menyimpang jauh → error berat di ekor (outlier) → pertimbangkan objective MAE/huber jika metric MAE.
# - Learning curve: gap train-valid besar dan valid masih naik → **variance** (tambah data/regularisasi/fitur lebih sedikit). Keduanya rendah & datar → **bias** (tambah fitur / model lebih kompleks).
# - Shuffled-target test jauh di atas chance → **BUG**: ada leakage (misal fitur turunan target, duplikat baris lintas fold, preprocessing di-fit di luar fold).
# - Prediksi test bergeser jauh dari OOF → drift (cek 6.9) atau bug preprocessing pada test.
#
# **Untuk laporan:** confusion matrix / ROC (klasifikasi) atau pred-vs-actual + residual (regresi), dan 1 paragraf error analysis
# ("model paling sering salah pada pelanggan dengan ...").

# %%
banner("15.1 METRIK & PLOT EVALUASI (OOF)")
pM = FINAL_OOF_PP[MASK]
yM = y[MASK]
EVAL_TABLES = {}
with safe_block("15.1 evaluation plots"):
    if TASK in ("binary", "multiclass"):
        labs = to_labels(pM, TASK, PP["threshold"], PP["class_weights"])
        names = [str(ID2LABEL[k]) for k in range(N_CLASSES)]
        cm = confusion_matrix(yM, labs, labels=list(range(N_CLASSES)))
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, mat, ttl, fmt in [(axes[0], cm, "Confusion matrix (counts)", "d"), (axes[1], cmn, "Confusion matrix (row-normalized)", ".2f")]:
            ax.imshow(mat, cmap="Blues")
            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                            color="white" if mat[i, j] > mat.max() / 2 else "black")
            ax.set_xticks(range(N_CLASSES))
            ax.set_xticklabels(names, rotation=45)
            ax.set_yticks(range(N_CLASSES))
            ax.set_yticklabels(names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(ttl)
            ax.grid(False)
        fig.tight_layout()
        save_fig("confusion_matrix")
        rep = pd.DataFrame(classification_report(yM, labs, labels=list(range(N_CLASSES)), target_names=names,
                                                 output_dict=True, zero_division=0)).T
        display(rep.round(4))
        save_table(rep.reset_index().rename(columns={"index": "class"}), "classification_report")
        EVAL_TABLES["classification_report"] = rep
        P2 = np.c_[1 - pM, pM] if TASK == "binary" else pM
        Y2 = label_binarize(yM, classes=list(range(N_CLASSES)))
        if TASK == "binary":
            Y2 = np.c_[1 - Y2[:, 0], Y2[:, 0]]
        cls_plot = [1] if TASK == "binary" else list(range(N_CLASSES))
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        for k in cls_plot:
            fpr, tpr, _ = roc_curve(Y2[:, k], P2[:, k])
            pr, rc, _ = precision_recall_curve(Y2[:, k], P2[:, k])
            lab = "final" if TASK == "binary" else names[k]
            axes[0].plot(fpr, tpr, label=f"{lab} (AUC={roc_auc_score(Y2[:, k], P2[:, k]):.3f})")
            axes[1].plot(rc, pr, label=f"{lab} (AP={average_precision_score(Y2[:, k], P2[:, k]):.3f})")
        if TASK == "binary":
            for mname in [m for m in LB["model"] if m in RESULTS and m != "dummy"][:3]:
                po = RESULTS[mname]["oof"][MASK]
                fpr, tpr, _ = roc_curve(yM, po)
                axes[0].plot(fpr, tpr, lw=0.8, alpha=0.6, label=f"{mname} ({roc_auc_score(yM, po):.3f})")
            fr, mp = calibration_curve(yM, np.clip(pM, 0, 1), n_bins=10, strategy="quantile")
            axes[2].plot(mp, fr, "o-", label="final")
            axes[2].set_xlabel("Mean predicted probability")
            axes[2].set_ylabel("Observed frequency")
            axes[2].set_title(f"Calibration (Brier={brier_score_loss(yM, np.clip(pM, 0, 1)):.4f})")
        else:
            conf = P2.max(axis=1)
            correct = (P2.argmax(axis=1) == yM).astype(float)
            bins = np.unique(np.quantile(conf, np.linspace(0, 1, 11)))
            idxb = np.clip(np.digitize(conf, bins[1:-1]), 0, len(bins) - 2)
            mp = [conf[idxb == b].mean() for b in range(len(bins) - 1) if (idxb == b).any()]
            fr = [correct[idxb == b].mean() for b in range(len(bins) - 1) if (idxb == b).any()]
            axes[2].plot(mp, fr, "o-", label="final")
            axes[2].set_xlabel("Confidence (max probability)")
            axes[2].set_ylabel("Accuracy")
            axes[2].set_title("Confidence calibration (multiclass)")
        axes[2].plot([0, 1], [0, 1], "k--")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[0].set_title("ROC curve (OOF)")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall curve (OOF)")
        for ax in axes:
            ax.legend(fontsize=7)
        fig.tight_layout()
        save_fig("roc_pr_calibration")
        if TASK == "binary":
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(pM[yM == 0], bins=40, alpha=0.6, density=True, label=str(ID2LABEL[0]))
            ax.hist(pM[yM == 1], bins=40, alpha=0.6, density=True, label=str(ID2LABEL[1]))
            ax.axvline(PP["threshold"], color="red", ls="--", label=f"threshold {PP['threshold']:.2f}")
            ax.set_xlabel("Predicted probability (positive class)")
            ax.set_ylabel("Density")
            ax.set_title("OOF probability distribution by true class")
            ax.legend()
            save_fig("oof_probability_by_class")
    else:
        res_ = yM - pM
        print(f"RMSE={rmse_score(yM, pM):.5f} | MAE={mean_absolute_error(yM, pM):.5f} | R2={r2_score(yM, pM):.5f} | "
              f"bias (mean residual)={res_.mean():.5f}")
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        lim = [min(yM.min(), pM.min()), max(yM.max(), pM.max())]
        axes[0, 0].scatter(yM, pM, s=4, alpha=0.3)
        axes[0, 0].plot(lim, lim, "r--")
        axes[0, 0].set_xlabel("Actual")
        axes[0, 0].set_ylabel("Predicted")
        axes[0, 0].set_title("Predicted vs actual (OOF)")
        axes[0, 1].scatter(pM, res_, s=4, alpha=0.3)
        axes[0, 1].axhline(0, color="r", ls="--")
        axes[0, 1].set_xlabel("Predicted")
        axes[0, 1].set_ylabel("Residual (actual - predicted)")
        axes[0, 1].set_title("Residuals vs predicted")
        axes[1, 0].hist(res_, bins=60, color="steelblue")
        axes[1, 0].set_xlabel("Residual")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Residual distribution")
        stats.probplot(res_, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("QQ plot of residuals")
        fig.tight_layout()
        save_fig("regression_residuals")
        dec = pd.qcut(pd.Series(yM).rank(method="first"), 10, labels=False)
        seg = pd.DataFrame({"decile": dec, "abs_err": np.abs(res_), "res": res_}).groupby("decile").agg(
            mae=("abs_err", "mean"), bias=("res", "mean"), n=("abs_err", "size"))
        display(seg.round(4))
        save_table(seg.reset_index(), "error_by_target_decile")
        print("ℹ️ bias positif di desil atas & negatif di desil bawah = model 'menarik ke tengah' (normal untuk regresi; "
              "jika parah coba fitur lebih kuat / kurangi regularisasi).")

# %% [markdown]
# ### 15.2 Learning curve, 15.3 train-vs-valid gap & fold stability
#
# Learning curve dihitung dengan model cepat (LightGBM, atau HistGB) pada 10%–100% train fold (3 fold pertama).

# %%
LC_MODEL = "lgb" if HAS_LGB else "hgb"
with safe_block("15.2 learning curve"):
    if CFG.DO_LEARNING_CURVE:
        fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
        lc_rows = []
        for fr in fracs:
            sp = []
            for f, (tr, va) in enumerate(CV_SPLITS[0][:3]):
                n_sub = max(int(len(tr) * fr), 30)
                tr_sub = np.sort(np.random.RandomState(f).choice(tr, min(n_sub, len(tr)), replace=False))
                sp.append((tr_sub, va))
            r = run_cv_model(LC_MODEL, splits=[sp], store=False, keep_models=False, verbose=False, compute_test=False)
            lc_rows.append({"train_frac": fr, "n_train": int(np.mean([len(a) for a, _ in sp])),
                            "train_score": r["train_mean"], "valid_score": r["fold_mean"], "valid_std": r["fold_std"]})
        LC = pd.DataFrame(lc_rows)
        display(LC.round(5))
        save_table(LC, "learning_curve")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(LC["n_train"], LC["train_score"], "o-", label="train")
        ax.plot(LC["n_train"], LC["valid_score"], "o-", label="validation")
        ax.fill_between(LC["n_train"], LC["valid_score"] - LC["valid_std"], LC["valid_score"] + LC["valid_std"], alpha=0.2)
        ax.set_xlabel("Training rows")
        ax.set_ylabel(RUN.metric)
        ax.set_title(f"Learning curve ({LC_MODEL})")
        ax.legend()
        save_fig("learning_curve")
        gap = abs(LC["train_score"].iloc[-1] - LC["valid_score"].iloc[-1])
        slope = LC["valid_score"].iloc[-1] - LC["valid_score"].iloc[-2]
        improving = is_better(LC["valid_score"].iloc[-1], LC["valid_score"].iloc[-2]) and abs(slope) > 0.002 * abs(LC["valid_score"].iloc[-1])
        print(f"Gap train-valid pada data penuh = {gap:.4f} | perubahan valid 75%→100% = {slope:+.4f}")
        if improving:
            print("📈 Valid masih membaik dengan data lebih banyak → variance-limited: data tambahan / regularisasi / augmentasi membantu.")
        else:
            print("📉 Valid sudah datar → menambah data tidak banyak membantu; fokus ke fitur baru / model lebih kuat (bias).")

with safe_block("15.3 gap & stability"):
    lb_ = leaderboard()
    lb_ = lb_[lb_["model"] != "dummy"]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.45 * len(lb_) + 1)))
    yy = np.arange(len(lb_))
    axes[0].barh(yy - 0.2, lb_["train_score"], height=0.4, label="train")
    axes[0].barh(yy + 0.2, lb_["fold_mean"], height=0.4, label="validation")
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(lb_["model"])
    axes[0].invert_yaxis()
    lo_ = min(lb_["train_score"].min(), lb_["fold_mean"].min())
    hi_ = max(lb_["train_score"].max(), lb_["fold_mean"].max())
    axes[0].set_xlim(lo_ - 0.1 * (hi_ - lo_ + 1e-9), hi_ + 0.1 * (hi_ - lo_ + 1e-9))
    axes[0].set_xlabel(RUN.metric)
    axes[0].set_title("Train vs validation score (overfitting check)")
    axes[0].legend()
    data_box = [RESULTS[m]["fold_scores"] for m in lb_["model"]]
    try:
        axes[1].boxplot(data_box, orientation="horizontal")
    except TypeError:
        axes[1].boxplot(data_box, vert=False)
    axes[1].set_yticks(range(1, len(lb_) + 1))
    axes[1].set_yticklabels(lb_["model"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(f"Fold {RUN.metric}")
    axes[1].set_title("Fold score stability")
    fig.tight_layout()
    save_fig("train_valid_gap_and_stability")
    for _, r in lb_.iterrows():
        rel_gap = r["gap"] / max(abs(r["fold_mean"]), 1e-9)   # gap positif = train lebih baik dari valid
        if rel_gap > (0.1 if greater_is_better() else 0.5):
            print(f"⚠️ {r['model']}: gap train-valid besar ({r['train_score']:.4f} vs {r['fold_mean']:.4f})")
    print("ℹ️ Gap selalu ada untuk model tree/GBDT. Yang menentukan adalah skor VALID: jika regularisasi lebih kuat "
          "mengecilkan gap DAN menaikkan valid → model sebelumnya overfit.")

# %% [markdown]
# ### 15.4 Shuffled-target sanity test, 15.5 OOF vs test distribution, 15.6 error analysis

# %%
with safe_block("15.4 shuffled target"):
    if CFG.DO_SHUFFLE_TEST:
        y_shuf = np.random.RandomState(CFG.SEED + 99).permutation(y)
        sp = [make_splits(CFG.SEED, y_arr=y_shuf)]
        ov = {"n_estimators": 300} if LC_MODEL == "lgb" else {"max_iter": 200}
        r = run_cv_model(LC_MODEL, params_override=ov, y_arr=y_shuf, splits=sp, store=False, keep_models=False,
                         verbose=False, compute_test=False)
        ref = RESULTS["dummy"]["score"] if "dummy" in RESULTS else None
        print(f"Skor dengan target diacak = {r['score']:.5f} | dummy = {ref} | model asli terbaik = {LB.iloc[0]['oof_score']:.5f}")
        if RUN.metric in ("roc_auc", "gini", "pr_auc"):
            chance = 0.5 if RUN.metric == "roc_auc" else (0.0 if RUN.metric == "gini" else float(np.mean(y)))
            suspicious = abs(r["score"] - chance) > 0.05
        elif ref is not None:
            span = abs(LB.iloc[0]["oof_score"] - ref)
            suspicious = is_better(r["score"], ref) and abs(r["score"] - ref) > 0.3 * span
        else:
            suspicious = False
        print("🚨 Skor target-acak jauh dari chance → kemungkinan LEAKAGE/BUG!" if suspicious
              else "✅ Sanity check lolos: dengan target acak, model tidak bisa memprediksi (seperti seharusnya).")

with safe_block("15.5 OOF vs test distribution"):
    if HAS_TEST:
        fig, ax = plt.subplots(figsize=(7, 4))
        if TASK == "multiclass":
            lab_o = np.bincount(to_labels(pM, TASK, class_weights=PP["class_weights"]), minlength=N_CLASSES) / len(pM)
            lab_t = np.bincount(to_labels(FINAL_TEST_PP, TASK, class_weights=PP["class_weights"]), minlength=N_CLASSES) / len(FINAL_TEST_PP)
            xx = np.arange(N_CLASSES)
            ax.bar(xx - 0.2, lab_o, 0.4, label="OOF")
            ax.bar(xx + 0.2, lab_t, 0.4, label="test")
            ax.set_xticks(xx)
            ax.set_xticklabels([str(ID2LABEL[k]) for k in xx])
            ax.set_ylabel("Predicted class share")
            print("Proporsi kelas prediksi OOF vs test:", np.round(lab_o, 3), np.round(lab_t, 3))
        else:
            ks = stats.ks_2samp(pM, FINAL_TEST_PP)
            ax.hist(pM, bins=50, alpha=0.6, density=True, label="OOF")
            ax.hist(FINAL_TEST_PP, bins=50, alpha=0.5, density=True, label="test")
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Density")
            print(f"KS statistic OOF vs test = {ks.statistic:.4f} " + ("⚠️ (bergeser)" if ks.statistic > 0.1 else "✅"))
            print(f"mean OOF = {np.mean(pM):.4f} | mean test = {np.mean(FINAL_TEST_PP):.4f}")
        ax.set_title("Prediction distribution: OOF vs test")
        ax.legend()
        save_fig("oof_vs_test_predictions")

with safe_block("15.6 error analysis"):
    rowsM = np.where(MASK)[0]
    if TASK == "binary":
        err = np.abs(yM - pM)
    elif TASK == "multiclass":
        err = 1 - pM[np.arange(len(yM)), yM]
    else:
        err = np.abs(yM - pM)
    show_feats = [c for c in (list(MI_RANK.index[:8]) if len(MI_RANK) else RAW_NUM[:4] + RAW_CAT[:4]) if c in train.columns]
    worst = np.argsort(-err)[:20]
    W = train.iloc[rowsM[worst]][([ID] if ID else []) + show_feats].copy()
    W["y_true"] = [ID2LABEL[v] for v in yM[worst]] if TASK != "regression" else yM[worst]
    W["pred"] = [np.round(pM[i], 4) if TASK != "multiclass" else ID2LABEL[int(np.argmax(pM[i]))] for i in worst]
    W["error"] = np.round(err[worst], 4)
    print("20 prediksi OOF paling salah (cek apakah ada pola / label salah / data aneh):")
    display(W)
    save_table(W, "error_worst_rows")
    seg_cols = [c for c in (list(MI_RANK.index) if len(MI_RANK) else RAW_CAT) if c in RAW_CAT][:3]
    if seg_cols:
        fig, axes = plt.subplots(1, len(seg_cols), figsize=(6 * len(seg_cols), 4), squeeze=False)
        for ax, c in zip(axes[0], seg_cols):
            seg = pd.DataFrame({"g": train.iloc[rowsM][c].fillna("NaN").astype(str).values, "err": err})
            tab = seg.groupby("g")["err"].agg(["mean", "count"]).sort_values("count", ascending=False).head(10)
            ax.barh(tab.index[::-1], tab["mean"][::-1], color="salmon")
            for k, (v, n_) in enumerate(zip(tab["mean"][::-1], tab["count"][::-1])):
                ax.text(v, k, f" n={n_}", va="center", fontsize=7)
            ax.axvline(err.mean(), color="k", ls="--", lw=1)
            ax.set_xlabel("Mean error")
            ax.set_title(f"Error by {c}")
            save_table(tab.reset_index(), f"error_by_{c}")
        fig.tight_layout()
        save_fig("error_by_segment")

# %% [markdown]
# ---
# # 16. Feature Importance & Interpretasi
#
# **Tujuan:** menjelaskan **apa yang dipelajari model** (wajib untuk laporan) dan memilih fitur.
#
# | No | Metode | Arti | Catatan |
# |---|---|---|---|
# | 16.1 | Gain importance (rata-rata semua fold) | seberapa banyak fitur mengurangi loss di split | cepat, tapi bias ke fitur high-card/kontinu |
# | 16.2 | **Permutation importance** pada fold validasi (OOF) | penurunan skor jika kolom diacak | paling jujur; ≤ 0 = fitur tidak berguna / noise |
# | 16.3 | **SHAP** (bar, beeswarm, dependence, waterfall) | kontribusi tiap fitur ke tiap prediksi | binary/multiclass dalam skala log-odds; regresi dalam skala target (atau log jika TARGET_TRANSFORM) |
# | 16.4 | PDP / ICE | efek rata-rata (PDP) & per-baris (ICE) mengubah nilai fitur | menunjukkan arah hubungan (naik/turun/non-linear) |
# | 16.5 | Null importance | gain asli vs gain saat target diacak | fitur yang tidak lebih baik dari noise → kandidat buang |
# | 16.6 | Feature selection check | buang fitur kandidat lalu CV ulang | keputusan berdasarkan CV, bukan feeling |
#
# SHAP dihitung dengan **kontribusi native** model (LightGBM `pred_contrib`, XGBoost `pred_contribs`, CatBoost
# `ShapValues`) — identik dengan TreeSHAP tapi tidak tergantung versi library `shap`. Library `shap` hanya dipakai untuk
# plot (fallback ke plot matplotlib manual jika tidak ada). PDP/ICE dihitung manual (setara `sklearn.inspection.PartialDependenceDisplay`)
# supaya bisa memakai preprocessor fold + model apa pun.
#
# **Cara membaca SHAP beeswarm:** tiap titik = satu baris; posisi x = kontribusi ke prediksi (kanan = menaikkan
# probabilitas/nilai); warna = nilai fitur (merah tinggi, biru rendah). Contoh: titik merah di kanan untuk `tenure` →
# tenure tinggi menaikkan prediksi.
#
# **Jika X maka Y:**
#
# - Fitur teratas tidak masuk akal secara domain (misal ID, nomor urut) → curigai leakage/drift → buang dan cek ulang CV.
# - Gain importance tinggi tapi permutation ≈ 0 → fitur dipakai untuk overfit (biasanya high-cardinality) → pertimbangkan buang/RARE_MIN_COUNT.
# - Banyak fitur dengan permutation ≤ 0 → coba buang (16.6). Jika CV naik/sama → model lebih sederhana lebih baik.
# - PDP datar → fitur tidak berpengaruh (secara marginal). ICE bersilangan → ada interaksi dengan fitur lain.
#
# **Untuk laporan:** SHAP bar + beeswarm (figur paling disukai juri), 1–2 dependence/PDP untuk fitur utama, dan interpretasi
# bisnis tiap fitur teratas (arah pengaruhnya).

# %%
def pick_interp_model():
    """Pilih model terbaik yang punya fold objects (prioritas GBDT untuk SHAP)."""
    ranked = [m for m in leaderboard()["model"] if m in RESULTS and RESULTS[m]["fold_objs"]]
    gbdt = [m for m in ranked if base_model(m) in ("lgb", "xgb", "cat")]
    return (gbdt or ranked or [None])[0]


INTERP = pick_interp_model() if CFG.DO_INTERPRET else None
PERM_TABLE, NULL_TABLE, SHAP_RANK = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)
PI_METRIC = RUN.metric if RUN.metric in PROB_METRICS | REG_METRICS else ("logloss" if TASK != "regression" else "rmse")
print(f"Model untuk interpretasi: {INTERP} | metric permutation importance: {PI_METRIC}")


def fold_predict(obj, Xraw):
    """Prediksi skala asli dari fold object (preprocessor + model) pada data mentah."""
    p = predict_std(obj["est"], obj["prep"].transform(Xraw))
    return y_inverse(p) if TASK == "regression" else p


banner("16.1 GAIN IMPORTANCE")
with safe_block("16.1 importance"):
    imp_tabs = {}
    for m, r in RESULTS.items():
        if r["importance"] is not None and base_model(m) != "dummy":
            s = r["importance"]
            imp_tabs[m] = s / max(s.sum(), 1e-12)
    if imp_tabs:
        IMP_ALL = pd.DataFrame(imp_tabs).fillna(0)
        IMP_ALL["mean_norm"] = IMP_ALL.mean(axis=1)
        IMP_ALL = IMP_ALL.sort_values("mean_norm", ascending=False)
        save_table(IMP_ALL.reset_index().rename(columns={"index": "feature"}), "feature_importance_all_models")
        show_m = INTERP if (INTERP in imp_tabs) else list(imp_tabs)[0]
        top = imp_tabs[show_m].sort_values(ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(top) + 1)))
        ax.barh(top.index[::-1], top.values[::-1], color="darkcyan")
        ax.set_xlabel("Normalized importance (gain)")
        ax.set_title(f"Feature importance — {show_m} (mean over folds)")
        save_fig(f"feature_importance_{show_m}")
        display(IMP_ALL.head(20).round(4))

# %%
banner("16.2 PERMUTATION IMPORTANCE (OOF folds)")
with safe_block("16.2 permutation importance"):
    if CFG.DO_INTERPRET and CFG.DO_PERM_IMP and INTERP:
        objs = RESULTS[INTERP]["fold_objs"][:2]
        feats = RESULTS[INTERP]["features"]
        imp = RESULTS[INTERP]["importance"]
        order = [c for c in (list(imp.index) if imp is not None else []) if c in feats]
        order += [c for c in (list(MI_RANK.index) if len(MI_RANK) else []) if c in feats and c not in order]
        order += [c for c in feats if c not in order]
        cand = order[: CFG.PERM_MAX_FEATURES]
        sign = 1 if greater_is_better(PI_METRIC) else -1
        recs = {c: [] for c in cand}
        rng = np.random.RandomState(CFG.SEED)
        for obj in objs:
            va = obj["va"][:20000]
            Xv = X.iloc[va][feats].reset_index(drop=True)
            yv = y[va]
            base_s = compute_metric(yv, fold_predict(obj, Xv), metric=PI_METRIC)
            for c in cand:
                for _ in range(RUN.perm_repeats):
                    Xp = Xv.copy()
                    Xp[c] = rng.permutation(Xp[c].to_numpy())
                    s = compute_metric(yv, fold_predict(obj, Xp), metric=PI_METRIC)
                    recs[c].append(sign * (base_s - s))
        PERM_TABLE = pd.DataFrame({"feature": cand, "importance_mean": [np.mean(recs[c]) for c in cand],
                                   "importance_std": [np.std(recs[c]) for c in cand]}).sort_values("importance_mean", ascending=False)
        save_table(PERM_TABLE, "permutation_importance")
        top = PERM_TABLE.head(30)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(top) + 1)))
        ax.barh(top["feature"][::-1], top["importance_mean"][::-1], xerr=top["importance_std"][::-1],
                color=["seagreen" if v > 0 else "indianred" for v in top["importance_mean"][::-1]], capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"Score drop when permuted ({PI_METRIC})")
        ax.set_title(f"Permutation importance on OOF folds — {INTERP}")
        save_fig("permutation_importance")
        n_bad = int((PERM_TABLE["importance_mean"] <= 0).sum())
        print(f"Fitur dengan permutation importance ≤ 0: {n_bad} dari {len(PERM_TABLE)} yang dievaluasi")
        display(PERM_TABLE.head(15).round(5))
    else:
        print("ℹ️ Permutation importance di-skip.")

# %%
banner("16.3 SHAP")


def native_shap(obj, Xt):
    """Kontribusi SHAP native. Return (values (n,F) atau (n,K,F), base (n,) atau (n,K))."""
    est, b = obj["est"], base_model(INTERP)
    n, F = Xt.shape
    if b == "lgb":
        c = np.asarray(est.predict(Xt, pred_contrib=True))
        c = c.reshape(n, N_CLASSES, F + 1) if TASK == "multiclass" else c.reshape(n, F + 1)
    elif b == "xgb":
        dm = xgb.DMatrix(Xt, enable_categorical=(obj["prep"].mode == "native"))
        booster = est.get_booster()
        bi = getattr(est, "best_iteration", None)
        try:
            c = booster.predict(dm, pred_contribs=True, iteration_range=(0, bi + 1)) if bi is not None else booster.predict(dm, pred_contribs=True)
        except TypeError:
            c = booster.predict(dm, pred_contribs=True)
        c = np.asarray(c)
    elif b == "cat":
        pool = cb.Pool(Xt, cat_features=[c_ for c_ in obj["prep"].cat_out_cols_ if c_ in Xt.columns])
        c = np.asarray(est.get_feature_importance(pool, type="ShapValues"))
    else:
        if not HAS_SHAP:
            raise RuntimeError("model bukan GBDT dan library shap tidak ada")
        ex = shap.Explainer(est.predict_proba if TASK != "regression" else est.predict, Xt.iloc[:100])
        e = ex(Xt)
        v = np.asarray(e.values)
        bv = np.asarray(e.base_values)
        if TASK == "binary" and v.ndim == 3:
            v, bv = v[:, :, 1], bv[:, 1]
        elif TASK == "multiclass" and v.ndim == 3:
            v, bv = np.transpose(v, (0, 2, 1)), bv
        return v, bv
    if c.ndim == 3:   # (n, K, F+1)
        return c[:, :, :-1], c[:, :, -1]
    return c[:, :-1], c[:, -1]


def display_frame(Xt):
    """Ubah kolom kategori → kode numerik agar bisa diwarnai di plot SHAP."""
    D = Xt.copy()
    for c in D.columns:
        if str(D[c].dtype) == "category":
            D[c] = D[c].cat.codes.astype(float)
        elif D[c].dtype == object:
            D[c] = pd.Categorical(D[c]).codes.astype(float)
    return D.astype(float)


with safe_block("16.3 SHAP"):
    if CFG.DO_INTERPRET and CFG.DO_SHAP and INTERP:
        obj = RESULTS[INTERP]["fold_objs"][0]
        feats = RESULTS[INTERP]["features"]
        va = obj["va"]
        n_shap = CFG.SHAP_SAMPLE if base_model(INTERP) in ("lgb", "xgb", "cat") else min(CFG.SHAP_SAMPLE, 300)
        va_s = np.sort(np.random.RandomState(0).choice(va, min(len(va), n_shap), replace=False))
        Xt = obj["prep"].transform(X.iloc[va_s][feats])
        vals, base_v = native_shap(obj, Xt)
        names = list(Xt.columns)
        Xd = display_frame(Xt)
        cls_idx = (N_CLASSES - 1 if CFG.SHAP_CLASS_INDEX is None else CFG.SHAP_CLASS_INDEX) if TASK == "multiclass" else None
        if vals.ndim == 3:
            mean_abs = np.abs(vals).mean(axis=(0, 1))
            v2, b2 = vals[:, cls_idx, :], base_v[:, cls_idx]
            print(f"Multiclass: bar = rata-rata |SHAP| semua kelas; beeswarm/waterfall untuk kelas '{ID2LABEL[cls_idx]}'")
        else:
            mean_abs = np.abs(vals).mean(axis=0)
            v2, b2 = vals, base_v
        SHAP_RANK = pd.Series(mean_abs, index=names).sort_values(ascending=False)
        save_table(SHAP_RANK.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}), "shap_importance")
        top = SHAP_RANK.head(20)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(top) + 1)))
        ax.barh(top.index[::-1], top.values[::-1], color="darkorange")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(f"SHAP feature importance — {INTERP}")
        save_fig("shap_bar")
        top_idx = [names.index(c) for c in SHAP_RANK.index[:20]]
        drawn = False
        if HAS_SHAP:
            try:
                shap.summary_plot(v2[:, top_idx], Xd.iloc[:, top_idx], feature_names=[names[i] for i in top_idx],
                                  show=False, max_display=20)
                plt.title(f"SHAP beeswarm — {INTERP}")
                save_fig("shap_beeswarm")
                drawn = True
            except Exception as e:  # noqa: BLE001
                print("  shap.summary_plot gagal, pakai plot manual:", str(e)[:100])
                plt.close("all")
        if not drawn:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(top_idx) + 1)))
            for r_, i in enumerate(top_idx[::-1]):
                xv = Xd.iloc[:, i].to_numpy()
                col = (xv - np.nanmin(xv)) / (np.nanmax(xv) - np.nanmin(xv) + 1e-12)
                ax.scatter(v2[:, i], r_ + np.random.uniform(-0.25, 0.25, len(xv)), c=col, cmap="coolwarm", s=5)
            ax.set_yticks(range(len(top_idx)))
            ax.set_yticklabels([names[i] for i in top_idx[::-1]])
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("SHAP value (impact on model output)")
            ax.set_title(f"SHAP beeswarm (manual) — {INTERP}")
            save_fig("shap_beeswarm")
        dep = [c for c in SHAP_RANK.index if c in names][:4]
        fig, axes = plt.subplots(1, len(dep), figsize=(4.5 * len(dep), 3.8), squeeze=False)
        for ax, c in zip(axes[0], dep):
            i = names.index(c)
            ax.scatter(Xd.iloc[:, i], v2[:, i], s=5, alpha=0.5, c=v2[:, i], cmap="coolwarm")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel(c + (" (category code)" if c in obj["prep"].cat_out_cols_ else ""))
            ax.set_ylabel("SHAP value")
            ax.set_title(f"Dependence: {c}")
        fig.tight_layout()
        save_fig("shap_dependence")
        row = int(np.argmax(v2.sum(axis=1)))
        drawn = False
        if HAS_SHAP:
            try:
                ex = shap.Explanation(values=v2[row], base_values=float(b2[row]), data=Xd.iloc[row].to_numpy(),
                                      feature_names=names)
                shap.plots.waterfall(ex, max_display=12, show=False)
                plt.title("SHAP waterfall — highest-scored validation row")
                save_fig("shap_waterfall")
                drawn = True
            except Exception as e:  # noqa: BLE001
                print("  shap waterfall gagal, pakai plot manual:", str(e)[:100])
                plt.close("all")
        if not drawn:
            contrib = pd.Series(v2[row], index=names)
            topc = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:12])
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(topc.index[::-1], topc.values[::-1], color=["crimson" if v > 0 else "steelblue" for v in topc.values[::-1]])
            ax.set_xlabel(f"SHAP contribution (base value {b2[row]:.3f})")
            ax.set_title("SHAP contributions for one row")
            save_fig("shap_waterfall")
        print("Top fitur (SHAP):", SHAP_RANK.head(10).round(4).to_dict())
    else:
        print("ℹ️ SHAP di-skip.")

# %%
banner("16.4 PDP / ICE")
with safe_block("16.4 PDP/ICE"):
    if CFG.DO_INTERPRET and CFG.DO_PDP and INTERP:
        obj = RESULTS[INTERP]["fold_objs"][0]
        feats = RESULTS[INTERP]["features"]
        rank_src = (list(PERM_TABLE["feature"]) if len(PERM_TABLE) else []) + list(SHAP_RANK.index) + list(MI_RANK.index)
        pdp_feats = [c for c in dict.fromkeys(rank_src) if c in NUM_COLS and c in feats and X[c].nunique() > 2][:4]
        if pdp_feats:
            va = obj["va"]
            samp = X.iloc[np.random.RandomState(1).choice(va, min(300, len(va)), replace=False)][feats].reset_index(drop=True)
            cls_idx = (N_CLASSES - 1 if CFG.SHAP_CLASS_INDEX is None else CFG.SHAP_CLASS_INDEX)
            fig, axes = plt.subplots(1, len(pdp_feats), figsize=(4.6 * len(pdp_feats), 3.8), squeeze=False)
            for ax, c in zip(axes[0], pdp_feats):
                grid = np.unique(np.nanquantile(X[c].astype(float), np.linspace(0.05, 0.95, 20)))
                curves = []
                for g in grid:
                    Xs = samp.copy()
                    Xs[c] = g
                    p = fold_predict(obj, Xs)
                    curves.append(p[:, cls_idx] if TASK == "multiclass" else p)
                curves = np.array(curves).T  # (rows, grid)
                for k in range(min(50, len(curves))):
                    ax.plot(grid, curves[k], color="grey", lw=0.4, alpha=0.4)
                ax.plot(grid, curves.mean(axis=0), color="red", lw=2.5, label="PDP (mean)")
                ax.set_xlabel(c)
                ax.set_ylabel("Prediction" + (f" P({ID2LABEL[cls_idx]})" if TASK == "multiclass" else ""))
                ax.set_title(f"PDP + ICE: {c}")
                ax.legend(fontsize=7)
            fig.tight_layout()
            save_fig("pdp_ice")
        else:
            print("Tidak ada fitur numerik untuk PDP.")

# %%
banner("16.5 NULL IMPORTANCE  &  16.6 FEATURE SELECTION CHECK")
NULL_DROP, FS_RESULT = [], None
with safe_block("16.5 null importance"):
    if CFG.DO_INTERPRET and CFG.DO_NULL_IMP and HAS_LGB:
        prep_n = TabularPreprocessor("native", NUM_COLS, CAT_COLS, TASK, N_CLASSES, CFG.SEED)
        Xn = prep_n.fit_transform(X[FEATURES], None)
        yfit = y_forward(y) if TASK == "regression" else y

        def _gain(yy, seed):
            p = get_params("lgb", seed, {"n_estimators": 200, "learning_rate": 0.1, "colsample_bytree": 0.8})
            mdl = (lgb.LGBMClassifier if TASK != "regression" else lgb.LGBMRegressor)(**p)
            mdl.fit(Xn, yy)
            return pd.Series(mdl.booster_.feature_importance("gain"), index=mdl.booster_.feature_name())

        actual = pd.concat([_gain(yfit, CFG.SEED + s) for s in range(2)], axis=1).mean(axis=1)
        nulls = pd.concat([_gain(np.random.RandomState(100 + s).permutation(yfit), CFG.SEED + s)
                           for s in range(RUN.null_imp_runs)], axis=1)
        q75 = nulls.quantile(0.75, axis=1)
        score = np.log(1e-10 + actual / (1 + q75))
        NULL_TABLE = pd.DataFrame({"actual_gain": actual, "null_gain_q75": q75, "null_score": score}).sort_values(
            "null_score", ascending=False)
        save_table(NULL_TABLE.reset_index().rename(columns={"index": "feature"}), "null_importance")
        NULL_DROP = list(NULL_TABLE.index[NULL_TABLE["null_score"] <= 0])
        show = pd.concat([NULL_TABLE.head(15), NULL_TABLE.tail(10)]).drop_duplicates()
        fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(show) + 1)))
        ax.barh(show.index[::-1], show["null_score"][::-1],
                color=["seagreen" if v > 0 else "indianred" for v in show["null_score"][::-1]])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("log(actual gain / (1 + null gain q75))")
        ax.set_title(f"Null importance ({RUN.null_imp_runs} shuffled-target runs)")
        save_fig("null_importance")
        print(f"Fitur tidak lebih baik dari noise (null_score ≤ 0): {len(NULL_DROP)} → {NULL_DROP[:20]}")
    else:
        print("ℹ️ Null importance di-skip (butuh LightGBM & DO_NULL_IMP=True).")

with safe_block("16.6 feature selection check"):
    if CFG.DO_INTERPRET and CFG.DO_FEATURE_SELECTION_CHECK and INTERP:
        if len(PERM_TABLE):
            drop_c = list(PERM_TABLE.loc[PERM_TABLE["importance_mean"] <= 0, "feature"])
            evaluated = set(PERM_TABLE["feature"])
            drop_c += [c for c in NULL_DROP if c not in evaluated]
        else:
            drop_c = list(NULL_DROP)
        drop_c = [c for c in dict.fromkeys(drop_c) if c in FEATURES]
        keep_f = [c for c in FEATURES if c not in drop_c]
        # model cepat untuk uji seleksi fitur (LightGBM jika ada), dibandingkan dengan skor model yang sama
        fs_model = next((m for m in ["lgb", "hgb", "xgb"] if m in RESULTS), base_model(INTERP))
        if drop_c and len(keep_f) >= 1:
            print(f"Uji membuang {len(drop_c)} fitur: {drop_c[:25]}")
            ref_name = fs_model
            params_ref = None
            FS_RESULT = run_cv_model(fs_model, params_override=params_ref, label=f"{fs_model}_fs", features=keep_f, verbose=False)
            s_ref = RESULTS[ref_name]["score"]
            print(f"{ref_name}: semua fitur = {s_ref:.5f} | tanpa kandidat = {FS_RESULT['score']:.5f}")
            if is_better(FS_RESULT["score"], s_ref) or abs(FS_RESULT["score"] - s_ref) < 1e-4:
                print("✅ Membuang fitur tsb tidak merugikan / membantu. Untuk memakainya di run final, tambahkan ke override:")
                print(f"   CFG.DROP_FEATURES = {drop_c}")
            else:
                print("➖ Membuang fitur menurunkan skor → pertahankan semua fitur.")
            save_table(pd.DataFrame({"candidate_drop": drop_c}), "feature_selection_candidates")
        else:
            print("✅ Tidak ada kandidat fitur untuk dibuang.")

# %% [markdown]
# ---
# # 17. Final Fit Strategy & Submission
#
# **Strategi final fit (`FINAL_FIT_STRATEGY`):**
#
# | Strategi | Cara | Kelebihan | Kekurangan |
# |---|---|---|---|
# | `fold_average` (default) | rata-rata prediksi test dari K model fold (sudah dihitung di section 11) | tidak perlu training ulang; ensemble K model → stabil; jumlah iterasi tervalidasi | tiap model hanya melihat (K-1)/K data |
# | `refit_full` | latih ulang di **seluruh** train dengan iterasi = rata-rata best_iter × (1 + 1/K), rata-rata semua SEEDS | memakai 100% data (bagus untuk data kecil) | tidak ada validasi untuk model final; jumlah iterasi hasil tebakan |
#
# Aturan praktis: **data kecil (< 5k baris) → `refit_full` sering sedikit lebih baik; data besar → `fold_average`**. Jika
# ragu, submit keduanya (jika kuota submission memungkinkan) dan pilih final berdasarkan CV, bukan public LB.
#
# **Pembuatan submission:**
#
# 1. Template = `sample_submission` (urutan baris & kolom dipertahankan). Jika tidak ada, dibuat `[ID, TARGET]`.
# 2. ID dicocokkan: jika urutan ID test sama → langsung; jika hanya himpunan sama → di-reindex berdasarkan ID.
# 3. Kolom target: binary probability / label asli (didekode, misal `"Yes"`), multiclass label atau **kolom per kelas**
#    (jika sample_submission punya satu kolom per kelas), regresi nilai. Dtype mengikuti sample_submission.
# 4. **Validator**: jumlah baris, nama & urutan kolom, urutan ID, NaN/inf, range probabilitas, label valid.
# 5. Varian disimpan: `submission.csv` (final), `submission_best_single_<model>.csv`, `submission_mean_blend.csv`, dan untuk
#    binary juga versi label/probability alternatif.
# 6. Log eksperimen: `experiment_<timestamp>.json` (config + skor) dan satu baris ditambahkan ke `experiments.csv`.
#
# **Jika X maka Y:**
#
# - Validator ❌ "ID order differs" → cek `ID_COL`; jika ID di test terduplikasi, pakai urutan baris.
# - Validator ❌ "label tidak dikenal" → mapping label salah; cek `tables/label_mapping.json`.
# - Lomba minta label 0/1 padahal target asli "Yes/No" → ubah manual: `sub[col] = sub[col].map({"No":0,"Yes":1})`.
# - Skor LB jauh di bawah CV → cek format (probability vs label!), urutan ID, dan drift (section 6.9).
#
# **Untuk laporan:** sebutkan model/ensemble final, skor CV final (mean ± std), dan strategi final fit.

# %%
def refit_full_model(name):
    """Latih ulang model di seluruh train (rata-rata semua SEEDS). Return prediksi test."""
    r = RESULTS[name]
    b = base_model(name)
    feats = r["features"]
    bi = [v for v in r["best_iters"] if v]
    preds = []
    for seed in RUN.seeds:
        params = dict(r["params"])
        if bi and b in ("lgb", "xgb"):
            params["n_estimators"] = int(np.mean(bi) * (1 + 1 / N_FOLDS_EFF))
        if bi and b == "cat":
            params["iterations"] = int(np.mean(bi) * (1 + 1 / N_FOLDS_EFF))
        for k in ("random_state", "random_seed"):
            if k in params:
                params[k] = seed
        prep = TabularPreprocessor(MODEL_MODES[b], [c for c in NUM_COLS if c in feats], [c for c in CAT_COLS if c in feats],
                                   TASK, N_CLASSES, seed)
        Xt = prep.fit_transform(X[feats], y)
        yfit = y_forward(y) if TASK == "regression" else y
        sw = make_sample_weight(y)
        est, _ = fit_one(name, params, Xt, yfit, None, None, sw, prep)
        p = predict_std(est, prep.transform(X_test[feats]))
        preds.append(y_inverse(p) if TASK == "regression" else p)
    return np.mean(preds, axis=0)


banner("17. FINAL FIT")
if CFG.FINAL_FIT_STRATEGY == "refit_full":
    with safe_block("refit_full"):
        refit_preds = {}
        for m in FINAL_RECIPE["models"]:
            t0 = time.time()
            refit_preds[m] = refit_full_model(m)
            print(f"  refit {m}: {time.time() - t0:.1f}s")
        FINAL_TEST = apply_recipe(FINAL_RECIPE, refit_preds)
        FINAL_TEST_PP = postprocess(FINAL_TEST)
        print("✅ Test prediction memakai model refit pada full train.")
else:
    print("Strategi: fold_average (prediksi test = rata-rata model tiap fold).")


def decode_labels(lab_int):
    """Kode integer → label asli (dtype dipertahankan)."""
    return np.array([ID2LABEL[int(v)] for v in lab_int], dtype=object)


def build_submission(pred_pp, mode=None, threshold=None, class_weights=None):
    """Buat DataFrame submission mengikuti sample_submission."""
    mode = mode or RUN.sub_mode
    threshold = PP["threshold"] if threshold is None else threshold
    class_weights = PP["class_weights"] if class_weights is None else class_weights
    if HAS_SAMPLE:
        sub = sample_sub.copy()
        id_c = sub.columns[0]
        if ID is not None and ID in test.columns:
            tid = pd.Series(test_ids)
            sid = sub[id_c]
            if len(sid) == len(tid) and (sid.astype(str).values == tid.astype(str).values).all():
                order = np.arange(len(tid))
            elif set(sid.astype(str)) == set(tid.astype(str)) and not tid.duplicated().any():
                pos = pd.Series(np.arange(len(tid)), index=tid.astype(str))
                order = pos.loc[sid.astype(str)].to_numpy()
                print("ℹ️ urutan ID sample_submission berbeda dari test → prediksi di-reindex berdasarkan ID")
            else:
                print("⚠️ ID sample_submission tidak cocok dengan test → memakai urutan baris")
                order = np.arange(len(tid))
        else:
            order = np.arange(len(sub))
        tcols = list(sub.columns[1:])
    else:
        id_c = ID_ORIG
        sub = pd.DataFrame({id_c: test_ids})
        order = np.arange(len(sub))
        tcols = [TARGET_ORIG]
    P = np.asarray(pred_pp)[order]
    if TASK == "regression":
        sub[tcols[0]] = P
        if HAS_SAMPLE and pd.api.types.is_integer_dtype(sample_sub[tcols[0]]):
            sub[tcols[0]] = np.round(P).astype(int)
    elif TASK == "multiclass" and len(tcols) == N_CLASSES and set(map(str, tcols)) == set(map(str, CLASS_LABELS)):
        for k in range(N_CLASSES):
            col = [c for c in tcols if str(c) == str(ID2LABEL[k])][0]
            sub[col] = P[:, k]
    else:
        col = tcols[0]
        if mode == "probability" and TASK == "binary":
            sub[col] = P
        elif mode == "probability" and TASK == "multiclass":
            print("⚠️ multiclass probability butuh satu kolom per kelas di sample_submission → memakai label.")
            sub[col] = decode_labels(to_labels(P, TASK, class_weights=class_weights))
        else:
            lab = decode_labels(to_labels(P, TASK, threshold, class_weights))
            sub[col] = lab
            if HAS_SAMPLE and pd.api.types.is_numeric_dtype(sample_sub[col]):
                try:
                    sub[col] = pd.to_numeric(sub[col])
                    if pd.api.types.is_integer_dtype(sample_sub[col]):
                        sub[col] = sub[col].astype(int)
                except (ValueError, TypeError):
                    print(f"⚠️ sample_submission kolom {col} numerik tapi label asli teks → biarkan teks")
    return sub


def validate_submission(sub, name="submission"):
    """Validasi format submission. Return True jika lolos semua cek kritis."""
    ok = True
    msgs = []
    if HAS_SAMPLE:
        if len(sub) != len(sample_sub):
            ok = False
            msgs.append(f"❌ jumlah baris {len(sub)} ≠ sample {len(sample_sub)}")
        if list(sub.columns) != list(sample_sub.columns):
            ok = False
            msgs.append(f"❌ kolom {list(sub.columns)} ≠ sample {list(sample_sub.columns)}")
        c0 = sample_sub.columns[0]
        if c0 in sub and not (sub[c0].astype(str).values == sample_sub[c0].astype(str).values).all():
            ok = False
            msgs.append("❌ ID order differs from sample_submission")
    tcols = list(sub.columns[1:])
    for c in tcols:
        if sub[c].isna().any():
            ok = False
            msgs.append(f"❌ NaN di kolom {c}")
        if pd.api.types.is_numeric_dtype(sub[c]) and not np.isfinite(sub[c].to_numpy(dtype=float)).all():
            ok = False
            msgs.append(f"❌ inf di kolom {c}")
        if HAS_SAMPLE and c in sample_sub and sub[c].dtype.kind != sample_sub[c].dtype.kind:
            msgs.append(f"⚠️ dtype {c}: {sub[c].dtype} vs sample {sample_sub[c].dtype}")
    if TASK != "regression":
        prob_cols = [c for c in tcols if pd.api.types.is_float_dtype(sub[c])]
        for c in prob_cols:
            if sub[c].min() < 0 or sub[c].max() > 1:
                ok = False
                msgs.append(f"❌ probabilitas di luar [0,1] pada {c}")
        lab_cols = [c for c in tcols if c not in prob_cols]
        for c in lab_cols:
            bad = set(sub[c].astype(str)) - set(map(str, CLASS_LABELS))
            if bad:
                ok = False
                msgs.append(f"❌ label tidak dikenal di {c}: {list(bad)[:5]}")
    if ID is not None and HAS_SAMPLE and sub[sub.columns[0]].duplicated().any():
        msgs.append("⚠️ ID duplikat di submission")
    print(f"[{name}] " + ("✅ VALID" if ok else "❌ TIDAK VALID") + (" | " + " ; ".join(msgs) if msgs else ""))
    return ok

# %%
banner("17. SUBMISSION")
SUBMISSIONS = {}
if not HAS_TEST:
    print("⏭️ Test tidak ada → submission di-skip. (Skor CV tetap tersimpan di log eksperimen.)")
else:
    sub_final = build_submission(FINAL_TEST_PP)
    validate_submission(sub_final, "final")
    p_main = RUN.out / "submissions" / "submission.csv"
    sub_final.to_csv(p_main, index=False)
    SUBMISSIONS["final"] = str(p_main)
    display(sub_final.head())
    print(f"💾 Submission utama: {p_main.resolve()}")
    if CFG.SAVE_SUBMISSION_VARIANTS:
        with safe_block("submission variants"):
            variants = {f"best_single_{best_single}": RESULTS[best_single]}
            if "mean" in ENS_TEST:
                variants["mean_blend"] = {"oof": ENS_OOF["mean"], "test": ENS_TEST["mean"]}
            for vname, v in variants.items():
                st = dict(PP, calibrator=None, temperature=None, cutpoints=None)
                oo, tt = v["oof"][MASK], v["test"]
                if TASK == "binary" and (RUN.metric in LABEL_METRICS or RUN.sub_mode == "label"):
                    st["threshold"] = tune_threshold(oo, y[MASK])[0]
                if TASK == "multiclass" and PP["class_weights"] is not None:
                    st["class_weights"] = tune_class_weights(oo, y[MASK])[0]
                s = build_submission(postprocess(tt, st), threshold=st["threshold"], class_weights=st["class_weights"])
                if validate_submission(s, vname):
                    pth = RUN.out / "submissions" / f"submission_{vname}.csv"
                    s.to_csv(pth, index=False)
                    SUBMISSIONS[vname] = str(pth)
            if TASK == "binary":
                alt = "label" if RUN.sub_mode == "probability" else "probability"
                s = build_submission(FINAL_TEST_PP, mode=alt)
                pth = RUN.out / "submissions" / f"submission_final_{alt}.csv"
                s.to_csv(pth, index=False)
                SUBMISSIONS[f"final_{alt}"] = str(pth)
                print(f"ℹ️ versi alternatif ({alt}) disimpan: {pth.name} — pakai hanya jika format lomba memintanya.")
    print("Semua submission:", {k: Path(v).name for k, v in SUBMISSIONS.items()})

# %%
# ---- Log eksperimen ----
def cfg_dict():
    """Ambil semua atribut CFG yang serializable."""
    out = {}
    for k in dir(CFG):
        if k.startswith("_"):
            continue
        v = getattr(CFG, k)
        if callable(v):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


ts = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP = {
    "timestamp": ts, "run_mode": CFG.RUN_MODE, "task": TASK, "metric": RUN.metric, "cv": CV_NAME,
    "n_folds": N_FOLDS_EFF, "n_repeats": RUN.n_repeats, "n_train": int(len(X)), "n_features": len(FEATURES),
    "best_single": best_single, "best_single_score": float(RESULTS[best_single]["score"]),
    "final_method": FINAL_NAME, "final_score_raw": float(FINAL_RAW_SCORE), "final_score_pp": float(FINAL_SCORE),
    "postprocess": {"threshold": PP["threshold"],
                    "class_weights": None if PP["class_weights"] is None else list(map(float, PP["class_weights"])),
                    "calibrator": None if PP["calibrator"] is None else PP["calibrator"][0],
                    "temperature": PP["temperature"], "clip": PP["clip"], "round": PP["round"]},
    "leaderboard": leaderboard().to_dict(orient="records"), "ensemble": ENSEMBLE_TABLE.to_dict(orient="records"),
    "tuned_params": TUNED_PARAMS, "failed_models": list(FAILED_MODELS), "adv_auc": ADV_AUC,
    "leak_suspects": LEAK_SUSPECTS, "submissions": SUBMISSIONS, "note": CFG.EXPERIMENT_NOTE, "config": cfg_dict(),
}
with open(RUN.out / f"experiment_{ts}.json", "w", encoding="utf-8") as f:
    json.dump(EXP, f, indent=2, default=str)
row = pd.DataFrame([{k: EXP[k] for k in ["timestamp", "run_mode", "task", "metric", "cv", "n_folds", "n_train",
                                         "n_features", "best_single", "best_single_score", "final_method",
                                         "final_score_raw", "final_score_pp", "note"]}])
exp_csv = RUN.out / "experiments.csv"
row.to_csv(exp_csv, mode="a", header=not exp_csv.exists(), index=False)
print(f"📝 Log eksperimen: experiment_{ts}.json + 1 baris di {exp_csv.name}")
display(pd.read_csv(exp_csv).tail(10))

# %% [markdown]
# ### 17.1 Ringkasan otomatis untuk laporan
#
# Cell di bawah mencetak ringkasan yang bisa langsung kamu parafrasekan ke laporan (bagian Metodologi & Hasil), plus daftar
# figur yang tersimpan. **Jangan copy mentah** — tambahkan interpretasi domain kamu.

# %%
banner("RINGKASAN UNTUK LAPORAN")
lb_final = leaderboard()
fs_ = RESULTS[best_single]
top_feats = list(SHAP_RANK.index[:5]) if len(SHAP_RANK) else (list(PERM_TABLE["feature"][:5]) if len(PERM_TABLE) else list(MI_RANK.index[:5]))
print(f"""
Data      : train {len(train_raw)} baris x {train_raw.shape[1]} kolom; test {'tidak ada' if not HAS_TEST else len(test)} baris.
            Setelah cleaning & feature engineering: {len(FEATURES)} fitur ({len(CAT_COLS)} kategorikal, {len(NUM_COLS)} numerik).
Target    : {TARGET_ORIG} ({TASK}{', imbalance ratio %.2f' % IMBALANCE_RATIO if TASK != 'regression' else ', skew %.2f' % TARGET_SKEW}).
Validasi  : {CV_NAME} K={N_FOLDS_EFF}, repeats={RUN.n_repeats}; metric {RUN.metric}.
Cleaning  : {len(DQ_LOG)} aksi (konversi angka-teks: {NUMERIC_CONVERTED}; boolean: {len(BOOL_CONVERTED)} kolom; tanggal: {DATE_DETECTED};
            dibuang: {AUTO_DROPPED}).
Model     : {len(RESULTS)} model dievaluasi. Terbaik tunggal: {best_single} = {fs_['score']:.5f} (fold {fs_['fold_mean']:.5f} ± {fs_['fold_std']:.5f}).
            Baseline dummy: {round(RESULTS['dummy']['score'], 5) if 'dummy' in RESULTS else '-'}.
Ensemble  : {FINAL_NAME} → OOF {FINAL_RAW_SCORE:.5f}; setelah post-processing {FINAL_SCORE:.5f}.
Drift     : adversarial AUC = {ADV_AUC if ADV_AUC is None else round(ADV_AUC, 4)}; leakage suspects = {LEAK_SUSPECTS}.
Top fitur : {top_feats}
""")
print(f"Figur tersimpan ({len(FIGURES_SAVED)}) di {RUN.out / 'figures'}:")
for p_ in FIGURES_SAVED:
    print("  -", Path(p_).name)

# %% [markdown]
# ---
# # 18. Experiment Recipes, Checklist & Troubleshooting
#
# ## 18.1 Experiment Recipes (salin ke cell `# >>> USER OVERRIDE`)
#
# **A. Debug cepat (pertama kali lihat data)**
# ```python
# CFG.RUN_MODE = "fast"; CFG.DO_EDA = True; CFG.DO_INTERPRET = False
# ```
#
# **B. Binary AUC standar (misal churn, fraud ringan)**
# ```python
# CFG.METRIC = "roc_auc"; CFG.SUBMISSION_MODE = "probability"; CFG.RUN_MODE = "full"
# CFG.MODELS = ["linear", "lgb", "xgb", "cat"]; CFG.USE_OPTUNA = True; CFG.OPTUNA_MODELS = ["lgb", "cat"]
# ```
#
# **C. Binary F1 dengan data imbalanced (fraud, penyakit langka)**
# ```python
# CFG.METRIC = "f1"; CFG.SUBMISSION_MODE = "label"; CFG.IMBALANCE_STRATEGY = "class_weight"
# CFG.THRESHOLD_TUNING = True; CFG.N_REPEATS = 2
# ```
#
# **D. Multiclass F1-macro**
# ```python
# CFG.METRIC = "f1_macro"; CFG.SUBMISSION_MODE = "label"; CFG.CLASS_PRIOR_ADJUST = True
# ```
#
# **E. Multiclass logloss (submission probabilitas per kelas)**
# ```python
# CFG.METRIC = "logloss"; CFG.SUBMISSION_MODE = "probability"; CFG.CALIBRATION = "auto"
# ```
#
# **F. Regresi RMSLE / target skew (harga, penjualan)**
# ```python
# CFG.METRIC = "rmsle"; CFG.TARGET_TRANSFORM = "auto"   # otomatis log1p
# # untuk RMSE dengan target skew: CFG.TARGET_TRANSFORM = "log1p" lalu bandingkan CV dengan None
# ```
#
# **G. Regresi ordinal / rating dengan QWK**
# ```python
# CFG.TASK = "regression"; CFG.METRIC = "qwk"     # cut-point otomatis dioptimasi di section 14
# ```
#
# **H. Data dengan entitas berulang (customer/pasien/toko)**
# ```python
# CFG.GROUP_COL = "customer_id"; CFG.CV_STRATEGY = "auto"   # → StratifiedGroupKFold / GroupKFold
# CFG.FE_GROUP_AGG = True
# ```
#
# **I. Data time series / test = masa depan**
# ```python
# CFG.TIME_COL = "date"; CFG.CV_STRATEGY = "timeseries"; CFG.FE_DATE_PARTS = True
# ```
#
# **J. Data kecil (< 2000 baris)**
# ```python
# CFG.N_FOLDS = 10; CFG.N_REPEATS = 3; CFG.MODELS = ["linear", "rf", "et", "lgb", "cat", "svm", "knn"]
# CFG.FINAL_FIT_STRATEGY = "refit_full"; CFG.MODEL_PARAMS = {"lgb": {"num_leaves": 8, "min_child_samples": 10}}
# ```
#
# **K. Data besar (> 500k baris) / laptop lemah**
# ```python
# CFG.N_FOLDS = 3; CFG.MODELS = ["lgb"]; CFG.EDA_SAMPLE_ROWS = 20000; CFG.DO_NULL_IMP = False
# CFG.SHAP_SAMPLE = 500; CFG.N_JOBS = 4; CFG.USE_GPU = True
# ```
#
# **L. Banyak kategori high-cardinality**
# ```python
# CFG.TE_FOR_TREES = True; CFG.RARE_MIN_COUNT = 30; CFG.FE_FREQ_ENCODING = True; CFG.MODELS = ["lgb", "cat"]
# ```
#
# **M. Multi-file (tabel transaksi per customer)**
# ```python
# CFG.EXTRA_TABLES = [{"path": "./data/transactions.csv", "key": "customer_id", "agg": True, "prefix": "trx_", "sort_col": "date"}]
# ```
#
# **N. Final run maksimal (waktu banyak)**
# ```python
# CFG.RUN_MODE = "full"; CFG.N_REPEATS = 3; CFG.USE_OPTUNA = True; CFG.OPTUNA_N_TRIALS = 100; CFG.OPTUNA_TIMEOUT = 1800
# CFG.MODELS = ["linear", "et", "hgb", "lgb", "xgb", "cat"]; CFG.FIG_DPI = 300
# ```
#
# **Cara membandingkan eksperimen:** setiap run menambah 1 baris di `outputs/.../experiments.csv` (isi `CFG.EXPERIMENT_NOTE`).
# Ubah **satu hal per eksperimen** supaya tahu apa yang berpengaruh. Kenaikan lebih kecil dari `fold_std` belum tentu nyata.
#
# ## 18.2 Checklist sebelum submit
#
# - [ ] `METRIC` sama persis dengan aturan lomba (AUC vs F1 vs logloss vs RMSE/RMSLE).
# - [ ] Format sesuai: probability vs label; label asli (`"Yes"`) atau 0/1? Lihat `sample_submission`.
# - [ ] Validator section 17 ✅ VALID (jumlah baris, kolom, urutan ID, tidak ada NaN).
# - [ ] Skor final jauh lebih baik dari `dummy`.
# - [ ] Shuffled-target test ✅ (tidak ada leakage), tidak ada fitur di `LEAK_SUSPECTS` yang tidak masuk akal.
# - [ ] Adversarial AUC wajar (< 0.7) atau sudah ditangani.
# - [ ] Skema CV meniru test (group/time jika perlu).
# - [ ] Run terakhir memakai `RUN_MODE="full"` (bukan fast!).
# - [ ] File yang di-upload adalah `submissions/submission.csv` dari run terbaru (cek timestamp).
# - [ ] Buka CSV di text editor: header benar, separator koma, tidak ada index kolom ekstra.
# - [ ] Simpan notebook + `experiments.csv` + figur untuk laporan; catat skor CV (mean ± std) di laporan.
# - [ ] Pilih submission final berdasarkan **CV**, bukan public leaderboard.
#
# ## 18.3 Troubleshooting
#
# | Error / gejala | Penyebab | Solusi |
# |---|---|---|
# | `LightGBMError: Do not support special JSON characters in feature name` | nama kolom berisi `[ ] { } : , "` | sudah ditangani otomatis oleh `sanitize_name` (section 3). Jika muncul setelah custom FE, pastikan nama kolom baru hanya huruf/angka/underscore |
# | `ValueError: feature_names must be string, and may not contain [, ] or <` (XGBoost) | nama kolom tidak aman | sama: sanitasi otomatis; cek kolom dari `custom_feature_engineering` |
# | XGBoost `DataFrame.dtypes for data must be int, float, bool or category` / `enable_categorical` error | ada kolom object/category tanpa `enable_categorical` | default XGBoost memakai mode `ordinal` (semua numerik). Jika `XGB_NATIVE_CAT=True` dan error → set kembali `False` |
# | CatBoost `Invalid type for cat_feature ... =nan : cat_features must be integer or string` | NaN / float di kolom kategori | mode `catboost` mengisi `"__MISSING__"`; pastikan kolom tsb terdeteksi kategori (cek `CAT_COLS`), atau ubah ke string di custom FE |
# | CatBoost `Bad value for num_feature ... "abc"` | string di kolom yang dianggap numerik | kolom harus masuk `CAT_COLS`; atau konversi angka-teks (section 4) gagal → bersihkan manual |
# | `KeyError: 'target'` / `TARGET_COL not found` | nama target salah (kapital/spasi) | cek print kolom di section 3; nama asli tetap dipakai di CFG (sanitasi otomatis) |
# | `ValueError: n_splits=5 cannot be greater than the number of members in each class` | ada kelas dengan < N_FOLDS sampel | fold dikurangi otomatis; atau gabungkan kelas langka / buang |
# | `ValueError: Input contains NaN, infinity` (sklearn) | NaN/inf masuk ke model non-GBDT | mode `ordinal`/`onehot` sudah impute; jika dari custom FE, pastikan tidak membuat inf (`/0`) — notebook mengganti inf→NaN setelah FE |
# | `MemoryError` / kernel mati | data besar, OHE besar, RF banyak pohon, simpan fold model | `RUN_MODE="fast"`, `REDUCE_MEMORY=True`, kecilkan `OHE_MAX_CARD`, `RF_N_ESTIMATORS`, `KEEP_FOLD_MODELS=False`, buang `rf/et/knn/svm`, `N_JOBS=2` |
# | Shape mismatch saat predict test (`number of features ... does not match`) | test punya kolom berbeda | notebook menyamakan kolom (section 4.4); jika custom FE membuat kolom hanya di train → lakukan FE di `full` (train+test) seperti contoh |
# | `TypeError: unexpected keyword argument 'sparse_output'` | sklearn < 1.2 | sudah ditangani `make_ohe()` |
# | `TypeError: got an unexpected keyword argument 'squared'` | sklearn ≥ 1.6 menghapus `squared=` | notebook memakai `rmse_score()` manual |
# | `TypeError: fit() got an unexpected keyword argument 'early_stopping_rounds'` | XGBoost ≥ 2.0 memindahkan parameter ke constructor | sudah ditangani (`_fit_xgb` mencoba keduanya) |
# | LightGBM `early_stopping() got an unexpected keyword 'verbose'` / callbacks tidak ada | LightGBM lama | sudah ada fallback otomatis; jika tetap error, perbarui LightGBM secara manual lalu restart kernel |
# | LightGBM GPU `No OpenCL device found` / `GPU Tree Learner was not enabled` | LightGBM pip tidak dikompilasi GPU | otomatis fallback CPU; set `LGB_GPU=False` |
# | XGBoost/CatBoost GPU error / CUDA error | driver/CUDA tidak cocok | otomatis fallback CPU; set `USE_GPU=False` |
# | `ModuleNotFoundError: lightgbm` setelah install | kernel belum restart | Kernel → Restart, jalankan ulang dari atas |
# | `UnicodeDecodeError` saat baca CSV | encoding bukan UTF-8 | `smart_read` otomatis mencoba utf-8-sig / latin-1 |
# | Train terbaca 1 kolom | separator `;`, tab, atau pipe | `smart_read` sniffing otomatis; atau `smart_read(path, sep=";")` |
# | `Merge mengubah jumlah baris` (AssertionError) | key tabel tambahan tidak unik | pakai `"agg": True` di `EXTRA_TABLES` |
# | Optuna sangat lama | trial × fold × model | `OPTUNA_N_FOLDS=3`, `OPTUNA_TIMEOUT` lebih kecil, tuning hanya `["lgb"]` |
# | Skor CV ≈ 1.0 / terlalu bagus | leakage | lihat 6.8 (single-feature) & 15.4 (shuffled target); buang fitur bocor via `DROP_COLS` |
# | CV bagus, LB jelek | CV tidak meniru test / drift / format salah | adversarial validation (6.9), group/time CV, cek probability vs label |
# | Submission ❌ ID order differs | ID test & sample berbeda urutan/format (int vs str) | `build_submission` reindex otomatis jika himpunan ID sama; cek `ID_COL` |
# | Label submission berupa angka padahal harus teks (atau sebaliknya) | mapping label | lihat `tables/label_mapping.json`; map manual sebelum `to_csv` |
# | `shap` error / plot kosong | versi shap tidak cocok | SHAP memakai kontribusi native model; plot fallback manual otomatis |
# | Warning `Proactor event loop` / `FutureWarning` | Jupyter di Windows / versi pandas | aman diabaikan |
# | `ConvergenceWarning` LogisticRegression | fitur belum terskala / iterasi kurang | sudah di-scale; abaikan atau naikkan `max_iter` via `MODEL_PARAMS` |
# | CatBoost sangat lambat (multiclass / regresi dengan kategori high-card) | perhitungan CTR target statistics mahal di CPU | `CFG.MODEL_PARAMS = {"cat": {"max_ctr_complexity": 1, "depth": 6}}`, naikkan `RARE_MIN_COUNT`, `USE_GPU=True`, atau keluarkan `cat` dari `MODELS`/`OPTUNA_MODELS` |
# | HistGradientBoosting `categorical feature ... must have at most 255 categories` | kardinalitas > 255 | notebook hanya menandai kategori ≤ 250 sebagai categorical |

# %%
print("✅ Notebook selesai dijalankan sampai akhir.")
print(f"Output: {RUN.out.resolve()}")
if FAILED_MODELS:
    print("Model yang gagal (lihat FAILED_MODELS):", list(FAILED_MODELS))
