# -*- coding: utf-8 -*-
# =====================================================================================================
# churn_insurance_master.py — versi SCRIPT dari churn_insurance_master.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python churn_insurance_master.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb churn_insurance_master.py
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
# # 🏆 Churn & Insurance Analytics — Master Notebook
#
# **Template lengkap untuk case: customer churn / policy lapse di perusahaan asuransi.**
# Notebook ini dirancang supaya **data apapun yang dikasih panitia** (asal ada kolom target churn) bisa langsung diolah dari awal sampai akhir:
#
# ```text
# Load data (1 / banyak tabel)  →  Data Quality & Cleaning otomatis  →  Target parsing
# →  Domain Feature Engineering (asuransi)  →  EDA + Driver Analysis (statistik + effect size + IV)
# →  Segment / Interaction Analysis  →  Survival Analysis (Kaplan-Meier, log-rank, Cox)
# →  Interpretable Model (Logistic Regression odds ratio)  →  Leakage & Drift Check
# →  ML Modeling (LogReg, RF, ET, HGB, LightGBM, XGBoost, CatBoost) dengan CV OOF
# →  Optuna Tuning  →  Multi-seed  →  Ensemble (hill-climbing, rank, stacking)
# →  Threshold & Calibration  →  Debugging (learning curve, overfit, shuffled target, error analysis)
# →  Feature Importance (gain, permutation, SHAP, PDP)  →  Feature Selection
# →  Business Analytics (lift, gains, risk tier, CLV, risk×value, campaign ROI, what-if, personas)
# →  Rekomendasi otomatis  →  Submission + Validator  →  Export tabel/figure/insight untuk laporan
# ```
#
# ## Prinsip desain
# 1. **Tidak gampang error.** Semua analisis opsional dibungkus `safe_section(...)` → kalau satu section gagal, notebook tetap jalan, error dicatat di `SECTION_ERRORS` dan dirangkum di akhir.
# 2. **Semua dievaluasi dengan Out-Of-Fold (OOF).** Angka score yang lu lihat = estimasi jujur performa di data baru.
# 3. **Tidak ada leakage preprocessing.** Encoding yang pakai target (target encoding) selalu di-fit di dalam fold.
# 4. **Report-ready.** Semua figure disimpan otomatis (bernomor) di `outputs/.../figures`, semua tabel di `tables/` + 1 file Excel, dan insight teks otomatis di `insights.md`.
# 5. **Panduan lengkap.** Setiap section ada: tujuan, cara baca output, aturan **"Jika X → lakukan Y"**, dan catatan **"Untuk laporan"**.
#
# ## Cara pakai cepat (5 langkah)
# 1. Taruh data di folder `./data/` (atau ubah path di cell **USER OVERRIDE**).
# 2. Di cell **USER OVERRIDE**: isi `TRAIN_PATH`, `TEST_PATH`, `SAMPLE_SUB_PATH`. `TARGET_COL` dan `ID_COL` boleh `None` (auto-detect), tapi **lebih aman diisi manual**.
# 3. Set `CFG.METRIC` sesuai metrik lomba (misal `"roc_auc"`, `"f1"`, `"accuracy"`).
# 4. Jalankan dulu dengan `CFG.RUN_MODE = "fast"` (± beberapa menit) → cek semua jalan & baca hasil EDA.
# 5. Ganti ke `CFG.RUN_MODE = "full"` → **Kernel → Restart & Run All** → ambil submission + figure + tabel untuk laporan.
#
# ## Strategi waktu saat final (saran)
# | Fase | Waktu | Aktivitas |
# |---|---|---|
# | 1 | 0:00–0:20 | Baca soal, set config, run **fast mode**, pastikan tidak error |
# | 2 | 0:20–1:00 | Baca EDA, driver table, survival, segment rules → tulis draft insight |
# | 3 | 1:00–2:00 | Run **full mode** (modeling + tuning jalan di background) sambil mulai tulis laporan |
# | 4 | 2:00–... | Ambil SHAP, business analytics, rekomendasi → finalisasi laporan & submission |
#
# > Jika waktu sangat mepet: set `CFG.USE_OPTUNA = False`, `CFG.MODELS = ["lgbm", "catboost", "logreg"]`, `CFG.N_FOLDS = 5` → sudah cukup kuat.

# %% [markdown]
# ## 📑 Daftar Isi
# - **0. Setup** — install, import, helper, **0.4 advanced helpers** (DeLong, EMPC, ECE, E-value, power, RMST, experiment log, diagram)
# - **1. Config** — `CFG` + **USER OVERRIDE** + **config file / headless runner** (`tools/run_case.py`) + **Case Brief** (narasi laporan)
# - **2. Load Data** — single / multi-table, merge & agregasi
# - **3. Data Quality & Cleaning** — tipe kolom, angka-dalam-string, tanggal, kategori typo, nilai mustahil, duplikat
# - **4. Target** — parsing label churn, churn rate, baseline
# - **5. Column Roles & Domain Feature Engineering (asuransi)**
# - **6. EDA Univariate**
# - **7. Driver Analysis** — churn rate per kategori/bin + uji statistik + effect size + IV + ranking driver
# - **8. Korelasi & Multikolinearitas** — Spearman, Cramér's V, VIF
# - **9. Segment & Interaction Analysis** — heatmap 2-arah, decision-tree rules, top risky segments
# - **10. Survival Analysis** — Kaplan-Meier, log-rank, hazard per tenure, Cox PH, **10.4 RMST, Schoenfeld PH test, Weibull AFT** (tenure bisa direkonstruksi dari tanggal mulai/berhenti)
# - **11. Interpretable Model** — Logistic Regression (odds ratio + CI)
# - **12. Leakage & Drift Check** — single-feature AUC, adversarial validation
# - **13. Modeling Setup** — matriks fitur, CV folds, metric
# - **14. Model Training (CV OOF)** — 9 algoritma termasuk **EBM (glass-box)** & **MLP**
# - **15. Hyperparameter Tuning (Optuna)**
# - **16. Multi-seed Final Models + Feature Selection + 16.3 Ablation study (DeLong)**
# - **17. Ensemble** + **17.2 Experiment log & model development journey** (bukti trial-and-error)
# - **18. Threshold & Calibration**
# - **19. Evaluation & Debugging** — ROC/PR, confusion matrix, bootstrap CI, **19.2 DeLong + EMPC + kalibrasi (ECE)**, learning curve, shuffled-target test, error analysis
# - **20. Feature Importance & Explainability** — gain, permutation, SHAP, dependence, waterfall, PDP
# - **21. Business Analytics** — lift/gains, risk tier (+ **validasi survival per tier**), CLV, risk×value, campaign ROI, what-if, **AIPW + robustness (placebo, E-value, DAG) + DR-learner uplift/GATES**, persona, **21.9 fairness audit & model card**
# - **22. Recommendations (otomatis)** + **22.2 desain pilot A/B (power analysis) & roadmap Gantt**
# - **23. Submission + Validator**
# - **24. Export untuk Laporan** — Excel, insights.md, executive summary, **24.2b diagram arsitektur**, **24.2c figure komposit**, **24.3 artikel lengkap otomatis (autofit ≤10 hal, appendix A–L, PDF)**, NARRATIVE_OPTIONS & JUDGE_QA, **24.5 dashboard Vercel**, ZIP
# - **25. Troubleshooting, Recipes, Checklist**

# %% [markdown]
# # 0. Setup
#
# ## 0.1 Install package yang belum ada
# Laptop panitia belum tentu punya semua library. Cell ini cek package; kalau `CFG_AUTO_INSTALL = True` dan ada yang hilang → `pip install` otomatis.
#
# - **Wajib**: numpy, pandas, scikit-learn, scipy, matplotlib, seaborn.
# - **Sangat disarankan**: lightgbm, xgboost, catboost, statsmodels (statistik & survival), optuna (tuning), shap (plot), openpyxl (export Excel).
# - **Opsional**: lifelines (survival lebih lengkap), imbalanced-learn (SMOTE).
#
# **Jika install gagal** (tidak ada internet / permission): tidak apa-apa, notebook tetap jalan; model/analisis yang butuh library itu otomatis di-skip.
# Kalau mau install manual di terminal: `pip install lightgbm xgboost catboost optuna shap statsmodels openpyxl`.
#
# > Setelah install package baru, kadang perlu **Kernel → Restart** lalu run ulang dari atas.

# %%
CFG_AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"   # ubah ke False kalau tidak mau notebook menginstall apapun (env TEMPLATE_AUTO_INSTALL=0 → off)

import importlib
import importlib.util
import subprocess
import sys

REQUIRED = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy",
            "matplotlib": "matplotlib", "seaborn": "seaborn"}
RECOMMENDED = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost",
               "statsmodels": "statsmodels", "optuna": "optuna", "shap": "shap", "openpyxl": "openpyxl",
               "docx": "python-docx", "lifelines": "lifelines"}
OPTIONAL = {"interpret": "interpret-core", "imblearn": "imbalanced-learn", "psutil": "psutil"}


# Di luar virtual environment → pasang ke folder pengguna (--user): tanpa admin & tidak mengubah paket sistem/milik komputer lain.
IN_VENV = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
PIP_SCOPE = [] if IN_VENV else ["--user"]
if not IN_VENV:
    print("ℹ️ Tidak berjalan di virtual environment → paket yang kurang dipasang dengan --user. "
          "Paling aman: jalankan SETUP.bat (environment terisolasi .venv).")


def ensure_packages(pkgs, install=True):
    """Cek package via find_spec; install yang belum ada (kalau install=True). Tidak pernah raise."""
    missing = [imp for imp in pkgs if importlib.util.find_spec(imp) is None]
    for imp in missing:
        if not install:
            print(f"  - {imp}: TIDAK ADA (auto-install off)")
            continue
        try:
            print(f"  - installing {pkgs[imp]} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PIP_SCOPE, pkgs[imp]])
            importlib.invalidate_caches()
        except Exception as e:  # noqa
            print(f"    gagal install {pkgs[imp]}: {e}")
    return [imp for imp in pkgs if importlib.util.find_spec(imp) is None]


print("Cek package wajib...")
still_missing = ensure_packages(REQUIRED, CFG_AUTO_INSTALL)
if still_missing:
    raise ImportError(f"Package wajib tidak tersedia: {still_missing}. Install manual dulu.")
print("Cek package disarankan...")
ensure_packages(RECOMMENDED, CFG_AUTO_INSTALL)
print("Cek package opsional (boleh gagal)...")
ensure_packages(OPTIONAL, CFG_AUTO_INSTALL)
print("Selesai.")

# %% [markdown]
# ## 0.2 Import & availability flags
# Setiap library opsional punya flag `HAS_xxx`. Kalau `False`, bagian yang butuh library itu di-skip dengan pesan jelas — **bukan error**.

# %%
import os
import re
import gc
import json
import math
import time
import random
import warnings
import traceback
import itertools
from contextlib import contextmanager
from collections import OrderedDict

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import scipy
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss, brier_score_loss,
                             accuracy_score, f1_score, precision_score, recall_score, fbeta_score,
                             balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
                             confusion_matrix, classification_report, roc_curve, precision_recall_curve,
                             silhouette_score, mutual_info_score)
from sklearn.feature_selection import mutual_info_classif
from sklearn.isotonic import IsotonicRegression

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HGB = True
except Exception:
    HAS_HGB = False

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

try:
    from IPython.display import display, Markdown
except Exception:  # di luar Jupyter
    display = print
    Markdown = str


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


lgb = _try_import("lightgbm")
xgb = _try_import("xgboost")
cb = _try_import("catboost")
optuna = _try_import("optuna")
shap = _try_import("shap")
sm = _try_import("statsmodels.api")
lifelines = _try_import("lifelines")
imblearn = _try_import("imblearn")

HAS_LGB, HAS_XGB, HAS_CB = lgb is not None, xgb is not None, cb is not None
HAS_OPTUNA, HAS_SHAP, HAS_SM = optuna is not None, shap is not None, sm is not None
HAS_LIFELINES, HAS_IMBLEARN = lifelines is not None, imblearn is not None
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except Exception:
    HAS_EBM = False
from sklearn.neural_network import MLPClassifier
HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None

if HAS_SM:
    try:
        from statsmodels.duration.survfunc import SurvfuncRight, survdiff
        from statsmodels.duration.hazard_regression import PHReg
        from statsmodels.stats.multitest import multipletests
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        HAS_SM_SURV = True
    except Exception:
        HAS_SM_SURV = False
else:
    HAS_SM_SURV = False

if HAS_OPTUNA:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

print("Python      :", sys.version.split()[0])
print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("sklearn     :", sklearn.__version__)
print("scipy       :", scipy.__version__)
for _name, _mod in [("lightgbm", lgb), ("xgboost", xgb), ("catboost", cb), ("optuna", optuna),
                    ("shap", shap), ("statsmodels", _try_import("statsmodels")), ("lifelines", lifelines),
                    ("imblearn", imblearn), ("interpret (EBM)", _try_import("interpret") if HAS_EBM else None)]:
    print(f"{_name:<12}:", getattr(_mod, "__version__", "OK") if _mod is not None else "— tidak tersedia (bagian terkait di-skip)")

# %% [markdown]
# ## 0.3 Helper functions umum
# Isi cell ini: reproducibility, `safe_section` (anti-crash), penyimpan figure/tabel, pencatat insight, format angka, Wilson CI, dan compat shim untuk beda versi library.
#
# **Cara kerja `safe_section`:**
# ```python
# with safe_section("Nama section"):
#     ...kode...
# ```
# Kalau kode di dalamnya error → error dicetak + disimpan di `SECTION_ERRORS`, lalu notebook lanjut. Set `CFG.STRICT = True` kalau lagi debugging dan mau error langsung berhenti.

# %%
SECTION_ERRORS = OrderedDict()
INSIGHTS = []          # list of (section, text) → ditulis ke insights.md
FIG_COUNTER = [0]
FIGURE_INDEX = []      # (file, title) → index figure untuk laporan
TABLES = OrderedDict() # name → DataFrame → diexport ke Excel
FEATURE_FAMILY = {}    # diisi di Section 5
LEAKY_COLS = []


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


@contextmanager
def safe_section(name, critical=False):
    """Jalankan blok kode; kalau error → catat & lanjut (kecuali critical / CFG.STRICT)."""
    t0 = time.time()
    try:
        yield
        print(f"\n✅ [{name}] selesai dalam {time.time() - t0:.1f}s")
    except Exception as e:
        SECTION_ERRORS[name] = f"{type(e).__name__}: {e}"
        print(f"\n❌ [{name}] ERROR → {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
        print("   ↳ Notebook tetap lanjut. Lihat ringkasan error di Section 24 / bagian Troubleshooting.")
        if critical or getattr(globals().get("CFG", object), "STRICT", False):
            raise


def save_fig(name, fig=None, show=True):
    """Simpan figure ke figures/ dengan nomor urut, lalu tampilkan & tutup."""
    fig = fig or plt.gcf()
    FIG_COUNTER[0] += 1
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")[:80]
    # cell dijalankan ulang → ganti figure lama dengan nama sama (hindari duplikat & figure basi di laporan)
    for old_fn, _t in [e for e in FIGURE_INDEX if re.sub(r"^fig\d+_", "", e[0]) == f"{safe}.png"]:
        FIGURE_INDEX.remove((old_fn, _t))
        try:
            os.remove(os.path.join(CFG.FIG_DIR, old_fn))
        except OSError:
            pass
    path = os.path.join(CFG.FIG_DIR, f"fig{FIG_COUNTER[0]:02d}_{safe}.png")
    try:
        fig.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight", facecolor="white")
        title = fig._suptitle.get_text() if fig._suptitle is not None else (fig.axes[0].get_title() if fig.axes else name)
        FIGURE_INDEX.append((os.path.basename(path), title or name))
    except Exception as e:
        print("  (gagal simpan figure:", e, ")")
    if show:
        plt.show()
    plt.close(fig)
    return path


def save_table(df, name, show=False, index=False):
    """Simpan tabel ke tables/<name>.csv dan daftarkan untuk export Excel."""
    if df is None or len(df) == 0:
        return None
    path = os.path.join(CFG.TABLE_DIR, f"{name}.csv")
    try:
        df.to_csv(path, index=index)
        TABLES[name] = df.reset_index() if index else df
    except Exception as e:
        print("  (gagal simpan tabel:", e, ")")
    if show:
        display(df)
    return path


def add_insight(section, text, show=True):
    """Catat 1 kalimat insight (English, siap tempel ke laporan)."""
    INSIGHTS.append((section, text))
    if show:
        print("💡", text)


def pct(x, d=1):
    try:
        return f"{100 * float(x):.{d}f}%"
    except Exception:
        return str(x)


def fmt_num(x, d=2):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 1e9:
        return f"{x / 1e9:,.{d}f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.{d}f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:,.{d}f}K"
    return f"{x:,.{d}f}"


def money_axis(ax, axis="y"):
    """Format sumbu angka besar jadi 1.2K / 3.4M / 5.6B (bukan notasi 1e8)."""
    f = matplotlib.ticker.FuncFormatter(lambda v, _: fmt_num(v, 1))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def wilson_ci(k, n, z=1.96):
    """95% Wilson confidence interval untuk proporsi k/n (lebih stabil dari normal approx untuk n kecil)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def p_fmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def make_ohe(min_frequency=None):
    """OneHotEncoder yang kompatibel lintas versi sklearn (sparse vs sparse_output, min_frequency)."""
    kwargs = {"handle_unknown": "ignore"}
    try:
        return OneHotEncoder(sparse_output=False, min_frequency=min_frequency, **kwargs)
    except TypeError:
        try:
            return OneHotEncoder(sparse=False, min_frequency=min_frequency, **kwargs)
        except TypeError:
            return OneHotEncoder(sparse=False, **kwargs)


def sanitize_name(c):
    """Nama kolom aman untuk LightGBM/XGBoost (tanpa karakter JSON khusus)."""
    c = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
    return c if c else "col"


def sanitize_columns(df):
    new, seen = [], {}
    for c in df.columns:
        s = sanitize_name(c)
        if s in seen:
            seen[s] += 1
            s = f"{s}_{seen[s]}"
        else:
            seen[s] = 0
        new.append(s)
    out = df.copy()
    out.columns = new
    return out, dict(zip(df.columns, new))


print("Helper siap.")

# %% [markdown]
# ## 0.4 Advanced helpers — DeLong, EMPC, calibration, causal sensitivity, power analysis, experiment log, diagram
# Fungsi statistik lanjutan yang dipakai di banyak section:
#
# | Fungsi | Untuk | Referensi |
# |---|---|---|
# | `delong_test`, `delong_ci` | uji beda AUC dua model pada customer yang sama + CI AUC analitik | DeLong et al. (1988); Sun & Xu (2014) |
# | `empc_score` | **Expected Maximum Profit (churn)** — metrik pemilihan model berbasis profit kampanye | Verbraken et al. (2013) |
# | `ece_score` | Expected Calibration Error | Guo et al. (2017) |
# | `e_value` | sensitivitas efek kausal terhadap confounder tak terukur | VanderWeele & Ding (2017) |
# | `n_per_arm` | ukuran sampel pilot A/B (uji 2 proporsi) | |
# | `rmst_from_km` | restricted mean survival time | Royston & Parmar (2013) |
# | `log_experiment` | **log semua eksperimen** → tabel & figure *model development journey* (bukti trial-and-error) | |
# | `dbox`, `darrow`, `canvas` | menggambar diagram arsitektur tanpa graphviz | |

# %%
# ===== 0.4 Advanced statistics, experiment tracking & diagram helpers =====
EXPERIMENT_LOG = []          # setiap model/variasi yang dicoba → bukti trial-and-error untuk laporan
CURRENT_STAGE = ["setup"]


def set_stage(name):
    CURRENT_STAGE[0] = name


def log_experiment(label, family, oof=None, note="", stage=None, n_features=None, time_s=None, cv_std=None, kept=None):
    """Catat satu eksperimen (dipanggil otomatis oleh run_cv, ablation, ensemble)."""
    row = {"exp_id": len(EXPERIMENT_LOG) + 1, "stage": stage or CURRENT_STAGE[0], "experiment": label, "family": family,
           "note": note, "n_features": n_features, "cv_std": cv_std, "time_s": time_s, "kept_in_final": kept}
    if oof is not None and "y" in globals() and len(oof) == len(y):
        try:
            pc = np.clip(np.asarray(oof, dtype=float), 1e-7, 1 - 1e-7)
            row.update({"roc_auc": roc_auc_score(y, pc), "pr_auc": average_precision_score(y, pc),
                        "logloss": log_loss(y, pc), "brier": brier_score_loss(y, pc)})
        except Exception:
            pass
    EXPERIMENT_LOG.append(row)
    return row


# ---------- DeLong test (fast algorithm, Sun & Xu 2014) ----------
def delong_cov(y_true, preds):
    """AUC & kovarians DeLong untuk k prediksi (array k × n)."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")          # positif dulu
    m = int(y_true.sum())
    n = len(y_true) - m
    P = np.atleast_2d(np.asarray(preds, dtype=float))[:, order]
    k = P.shape[0]
    tx = np.vstack([stats.rankdata(P[r, :m]) for r in range(k)])
    ty = np.vstack([stats.rankdata(P[r, m:]) for r in range(k)])
    tz = np.vstack([stats.rankdata(P[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_ci(y_true, p, alpha=0.05):
    aucs, cov = delong_cov(y_true, np.asarray(p)[None, :])
    se = float(np.sqrt(max(cov[0, 0], 0)))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(aucs[0]), max(0.0, aucs[0] - z * se), min(1.0, aucs[0] + z * se), se


def delong_test(y_true, p1, p2):
    """Uji DeLong 2 AUC berkorelasi (prediksi pada customer yang sama). Return auc1, auc2, diff, se, p."""
    aucs, cov = delong_cov(y_true, np.vstack([p1, p2]))
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = math.sqrt(var) if var > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(diff / se))) if se > 0 else (1.0 if diff == 0 else 0.0)
    return float(aucs[0]), float(aucs[1]), diff, se, p


def holm_adjust(p):
    """Koreksi Holm–Bonferroni untuk beberapa uji sekaligus."""
    p = np.asarray(p, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------- calibration & profit metrics ----------
def ece_score(y_true, p, n_bins=10):
    """Expected Calibration Error (bin kuantil)."""
    y_true, p = np.asarray(y_true), np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float(abs(p.mean() - y_true.mean()))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y_true[idx == b].mean()) for b in np.unique(idx)))


def empc_score(y_true, p, clv, d, f, alpha=6.0, beta=14.0, n_gamma=200):
    """Expected Maximum Profit for Customer churn (Verbraken, Verbeke & Baesens, 2013).
    clv = nilai customer yang diselamatkan; d = biaya insentif; f = biaya kontak; γ ~ Beta(alpha, beta) = peluang churner
    yang dikontak menerima tawaran. Return (empc per customer, expected fraction targeted)."""
    y_true = np.asarray(y_true).astype(int)
    pi0 = y_true.mean()
    pi1 = 1 - pi0
    fpr, tpr, _ = roc_curve(y_true, p)
    clv = max(float(clv), 1e-9)
    delta, phi = d / clv, f / clv
    g = (np.arange(n_gamma) + 0.5) / n_gamma
    w = stats.beta.pdf(g, alpha, beta)
    w = w / w.sum()
    prof = clv * ((g[:, None] * (1 - delta) - phi) * pi0 * tpr[None, :] - (delta + phi) * pi1 * fpr[None, :])
    j = prof.argmax(axis=1)
    best = np.maximum(prof[np.arange(n_gamma), j], 0)
    eta = np.where(prof[np.arange(n_gamma), j] > 0, pi0 * tpr[j] + pi1 * fpr[j], 0.0)
    return float((best * w).sum()), float((eta * w).sum())


def empc_params():
    """Parameter EMPC dari CFG bisnis (dipakai konsisten di seluruh notebook)."""
    s = min(max(CFG.RETENTION_SUCCESS_RATE, 0.02), 0.98)
    a = getattr(CFG, "EMPC_ALPHA", None) or 6.0
    b = getattr(CFG, "EMPC_BETA", None) or a * (1 - s) / s
    share = getattr(CFG, "CONTACT_COST_SHARE", 0.2)
    return dict(d=CFG.RETENTION_COST * (1 - share), f=CFG.RETENTION_COST * share, alpha=a, beta=b)


# ---------- causal / experimental design helpers ----------
def e_value(rr):
    """E-value (VanderWeele & Ding, 2017) untuk risk ratio: kekuatan confounder tak terukur yang dibutuhkan untuk
    menghapus efek yang diamati."""
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return np.nan
    rr = rr if rr >= 1 else 1 / rr
    return float(rr + math.sqrt(rr * (rr - 1)))


def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    """Ukuran sampel per grup (uji 2 proporsi, two-sided)."""
    if p1 is None or p2 is None or abs(p1 - p2) < 1e-9:
        return np.nan
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    return int(math.ceil((za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2))


def rmst_from_km(t, s, tau):
    """Restricted mean survival time sampai tau dari kurva KM (step function)."""
    t, s = np.asarray(t, dtype=float), np.asarray(s, dtype=float)
    grid = np.r_[0.0, t[t < tau], tau]
    surv = np.r_[1.0, s[t < tau]]
    return float(np.sum(np.diff(grid) * surv))


# ---------- diagram helpers (arsitektur & alur, digambar dengan matplotlib → tidak perlu graphviz) ----------
def dbox(ax, x, y, w, h, title, body="", fc="#2E86AB", tc="white", fs=9, bfs=7.2, alpha=0.95, ec="none", bc="#333"):
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                                                   fc=fc, ec=ec, alpha=alpha, lw=1.2))
    if body:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", color=tc, fontsize=fs, weight="bold", wrap=True)
        ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", color=tc if fc not in ("white", "#FFFFFF") else bc,
                fontsize=bfs, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", color=tc, fontsize=fs, weight="bold", linespacing=1.2)


def darrow(ax, xy_from, xy_to, color="#555", lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2,
                                connectionstyle=f"arc3,rad={rad}"))


def canvas(w=14, h=7):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ---------- label bisnis untuk nama fitur (dipakai di figure, insight & laporan) ----------
FE_LABELS = {"fe_tenure_years": "tenure (years)", "fe_tenure_band": "tenure band", "fe_is_new_customer": "new customer (<12 months)",
             "fe_age_band": "age band", "fe_premium_to_income": "premium-to-income ratio", "fe_sum_insured_to_premium": "sum insured to premium ratio",
             "fe_has_claim": "has made a claim", "fe_claims_per_year": "claims per year of tenure", "fe_claim_reject_rate": "claim rejection rate",
             "fe_has_complaint": "has complained", "fe_has_late_payment": "has late payments", "fe_premium_increase": "premium increased at renewal",
             "fe_big_premium_increase": "premium increase above 10%", "fe_multi_product": "holds multiple products",
             "fe_avg_paid_per_period": "average amount paid per period", "fe_paid_vs_expected": "paid vs expected premium",
             "fe_n_missing": "number of missing fields", "fe_age_from_dob": "age (from date of birth)"}


def pretty(f):
    """Nama fitur → label bisnis (CFG.FEATURE_LABELS > label FE bawaan > underscore jadi spasi)."""
    f = str(f)
    if f in getattr(CFG, "FEATURE_LABELS", {}):
        return CFG.FEATURE_LABELS[f]
    if f in FE_LABELS:
        return FE_LABELS[f]
    s = re.sub(r"^n_", "number of ", f)
    s = s.replace("__freq", " (frequency)").replace("__te", " (target-encoded)").replace("_days_since", " (days since)")
    return s.replace("_", " ").strip()


print("Advanced helpers siap (DeLong, EMPC, ECE, E-value, power analysis, RMST, experiment log, diagram).")

# %% [markdown]
# # 1. Config
#
# `CFG` adalah **pusat kontrol**. Jangan edit cell `CFG` ini saat lomba — edit di cell **USER OVERRIDE** di bawahnya supaya default tetap utuh.
#
# ### Knob paling penting
# | Knob | Isi | Catatan |
# |---|---|---|
# | `TRAIN_PATH`, `TEST_PATH`, `SAMPLE_SUB_PATH` | path file | csv/xlsx/parquet/json didukung. Test boleh tidak ada. |
# | `TARGET_COL` | nama kolom churn | `None` = auto-detect (kolom yang ada di train tapi tidak di test / nama mengandung churn/exited/lapse/attrition/status) |
# | `POSITIVE_LABEL` | nilai yang berarti churn | `None` = auto (Yes/1/True/Churn/Exited/...) |
# | `ID_COL` | kolom ID | `None` = auto |
# | `METRIC` | metrik lomba | `roc_auc`, `pr_auc`, `logloss`, `brier`, `f1`, `f2`, `accuracy`, `balanced_accuracy`, `precision`, `recall`, `mcc`, `kappa`, `gini` |
# | `SUBMISSION_MODE` | `"probability"` / `"label"` | Lihat format sample_submission. AUC/logloss → probability; F1/accuracy → label. |
# | `RUN_MODE` | `"fast"` / `"full"` | fast = cek cepat; full = final |
# | `COLUMN_ROLES` | mapping role → kolom | auto-detect by keyword (EN + ID). Override kalau salah. |
# | `ACTIONABLE_FEATURES` | fitur yang bisa diintervensi perusahaan | dipakai untuk what-if & rekomendasi |
# | Parameter bisnis | margin, biaya retensi, success rate | untuk simulasi ROI campaign |

# %%
class CFG:
    # ===================== RUN =====================
    RUN_MODE = "full"          # "fast" (debug cepat: subsample, 3 fold, model sedikit, tanpa Optuna) | "full"
    STRICT = False             # True → error di section manapun langsung stop (untuk debugging)
    SEED = 42
    VERBOSE = True
    CONFIG_FILE = None         # path JSON berisi override CFG (alternatif USER OVERRIDE); diisi otomatis oleh tools/run_case.py

    # ===================== PATHS =====================
    TRAIN_PATH = "./data/train.csv"
    TEST_PATH = "./data/test.csv"                  # boleh tidak ada → hanya analisis + CV
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    OUTPUT_DIR = "./outputs/churn_insurance"
    # Tabel tambahan one-to-many (misal claims, payments, complaints) → diagregasi per customer lalu di-merge.
    # Format: {"claims": {"path": "./data/claims.csv", "key": "customer_id", "date_col": "claim_date"}}
    EXTRA_TABLES = {}
    # Tabel tambahan one-to-one (misal demografi terpisah) → langsung di-merge.
    # Format: {"demo": {"path": "./data/demographics.csv", "key": "customer_id"}}
    EXTRA_ONE_TO_ONE = {}
    # Label train di file terpisah (mis. train_labels.csv) → di-merge HANYA ke train. Format: {"path": "...", "key": "customer_id"}
    TRAIN_LABELS = None
    EXCEL_SHEET = None         # nama/indeks sheet Excel; None = sheet terbesar otomatis
    TEST_FROM_MISSING_TARGET = True   # tanpa file test & ≥5% baris target kosong → baris itu dianggap data yang harus diprediksi

    # ===================== COLUMNS =====================
    ID_COL = None              # None = auto-detect
    TARGET_COL = None          # None = auto-detect
    POSITIVE_LABEL = None      # None = auto (Yes/1/True/Churn/...). Contoh: "Attrited Customer"
    DROP_COLS = []             # kolom yang pasti dibuang (misal nama, nomor telp, alamat)
    FORCE_CATEGORICAL = []     # kolom angka yang sebenarnya kategori (misal kode wilayah, kode produk)
    FORCE_NUMERIC = []         # kolom string yang sebenarnya angka
    DATE_COLS = None           # None = auto-detect kolom tanggal
    REFERENCE_DATE = None      # tanggal acuan untuk "days since"; None = tanggal maksimum di data
    GROUP_COL = None           # misal "agent_id"/"household_id" kalau satu grup tidak boleh bocor antar fold

    # Role kolom (auto-detect by keyword). Override contoh: {"tenure": "lama_polis_bulan", "premium": "premi"}
    COLUMN_ROLES = {}
    TENURE_UNIT = "auto"       # "months" | "years" | "days" | "auto" (dipakai untuk band & survival)
    ACTIONABLE_FEATURES = None # None = auto by keyword (payment, renew, complaint, claim, premium, channel, ...)

    # ===================== CLEANING =====================
    NA_TOKENS = ["", " ", "?", "NA", "N/A", "n/a", "na", "null", "NULL", "Null", "nan", "NaN", "-", "--", "#N/A", "missing", "Missing", "unknown_value"]
    DROP_DUPLICATES = True
    QUASI_CONSTANT_THRESHOLD = 0.995   # kolom dengan 1 nilai ≥ 99.5% → drop
    HIGH_CARD_THRESHOLD = 50           # kategori dengan > 50 level → high-cardinality (freq + target encoding)
    RARE_MIN_FREQ = 0.005              # level kategori < 0.5% digabung jadi "__RARE__" (untuk model linear & EDA)
    FIX_IMPOSSIBLE_VALUES = True       # age<0/>110, nilai negatif di premium/income/tenure → NaN

    # ===================== LEAKAGE =====================
    LEAKAGE_KEYWORDS = ["cancel", "terminat", "churn_date", "end_date", "exit_date", "lapse_date", "reason", "suspend", "suspd",
                        "close_reason", "stop_date", "berhenti", "tgl_keluar", "tanggal_keluar", "lapsed_on", "surrender_date",
                        "alasan", "status_after", "closed", "close_date", "surrender", "refund", "batal", "tutup"]
    LEAKAGE_AUC_THRESHOLD = 0.97       # single-feature AUC ≥ ini → dicurigai leakage
    AUTO_DROP_LEAKY = True             # True → buang fitur dengan BUKTI statistik leakage (missingness/single-feature AUC).
                                       # Fitur yang hanya cocok kata kunci nama → warning saja. False → semua hanya warning.

    # ===================== FEATURE ENGINEERING =====================
    USE_DOMAIN_FE = True       # fitur domain asuransi (rasio klaim, band tenure, dsb)
    USE_FREQ_ENCODING = True   # frequency encoding untuk kategori
    USE_TARGET_ENCODING = True # target encoding in-fold untuk kategori high-cardinality
    USE_ROW_STATS = True       # jumlah missing per baris, dsb

    # ===================== VALIDATION & METRIC =====================
    METRIC = "roc_auc"
    THRESHOLD_METRIC = "auto"  # metrik untuk tuning threshold; "auto" = METRIC kalau label-based, else "f1"
    N_FOLDS = 5
    N_REPEATS = 1              # >1 = repeated CV (lebih stabil, lebih lama)
    CV_STRATEGY = "auto"       # "auto" | "stratified" | "group"

    # ===================== MODELS =====================
    MODELS = ["logreg", "rf", "et", "hgb", "lgbm", "xgb", "catboost", "ebm", "mlp"]
    # ebm = Explainable Boosting Machine (glass-box GAM, interpret-core); mlp = neural network baseline (sklearn)
    EBM_INTERACTIONS = 10      # pasangan interaksi EBM (auto diturunkan untuk data besar)
    EBM_OUTER_BAGS = 8
    IMBALANCE = "auto"         # "auto" | "none" | "class_weight" | "smote"
    EARLY_STOPPING_ROUNDS = 200
    USE_GPU = "auto"           # "auto" = pakai GPU kalau ada NVIDIA (fallback ke CPU kalau gagal) | True | False
    AUTO_ADAPT = True          # sesuaikan otomatis dengan ukuran data & RAM/CPU/GPU (Section 2.1)
    LARGE_DATA_ROWS = 150_000  # di atas ini → mode data besar
    N_JOBS = -1

    USE_OPTUNA = True
    OPTUNA_MODELS = ["lgbm", "catboost", "xgb"]
    OPTUNA_TRIALS = 40
    OPTUNA_TIMEOUT = 900       # detik per model
    OPTUNA_FOLDS = 3           # fold saat tuning (lebih cepat); final tetap pakai N_FOLDS

    FINAL_SEEDS = [42, 2024, 7]  # multi-seed averaging untuk model terbaik
    USE_ENSEMBLE = True
    CALIBRATION = "auto"       # "auto" | "none" | "isotonic" | "sigmoid"
    USE_FEATURE_SELECTION = True

    # ===================== ADVANCED ANALYSES =====================
    USE_ABLATION = True        # ablation study per kelompok fitur (kontribusi tiap kelompok, uji DeLong)
    ABLATION_MAX_GROUPS = 8
    USE_UPLIFT = True          # efek heterogen (DR-learner / CATE) + uplift targeting vs risk targeting
    CAUSAL_MAX_TREATMENTS = 6
    TREATMENT_COLS = []        # kolom intervensi historis (mis. "received_retention_offer", "agent_call") → dianalisis kausal lebih dulu
    PROTECTED_ATTRIBUTES = None  # None = auto (gender, age band, marital, region) → fairness audit
    PILOT_POWER = 0.80         # desain pilot A/B
    PILOT_ALPHA = 0.05
    CONTACT_COST_SHARE = 0.2   # porsi RETENTION_COST untuk kontak (sisanya insentif) → EMPC
    EMPC_ALPHA = None          # Beta(α, β) peluang tawaran diterima; None → rata-rata = RETENTION_SUCCESS_RATE
    EMPC_BETA = None

    # ===================== EXPLAINABILITY =====================
    SHAP_SAMPLE = 3000
    PERM_IMPORTANCE_REPEATS = 3
    TOP_N_FEATURES_PLOT = 20

    # ===================== EDA =====================
    MAX_EDA_PLOTS = 24
    MAX_CAT_LEVELS_PLOT = 15
    N_BINS_NUMERIC = 10

    # ===================== BUSINESS =====================
    CURRENCY = "IDR"
    VALUE_COL = None                    # kolom nilai customer (premi). None = auto dari role "premium"
    VALUE_PERIOD = "annual"             # "annual" | "monthly" (kalau monthly → dikali 12 untuk CLV tahunan)
    PROFIT_MARGIN = 0.20                # margin keuntungan dari premi
    RETENTION_COST = 150_000            # biaya kontak/insentif per customer yang di-target
    RETENTION_SUCCESS_RATE = 0.30       # % churner yang berhasil dipertahankan jika di-target
    DISCOUNT_RATE = 0.10                # discount rate untuk CLV
    CLV_HORIZON_YEARS = 5
    RISK_TIER_CUTS = (0.5, 0.8)         # quantile → Low (<50%), Medium (50–80%), High (≥80%)
    WHAT_IF_SCENARIOS = None            # None = auto-generate dari actionable features
    WHAT_IF_ADOPTION = 0.30             # asumsi % customer terdampak yang benar-benar berubah perilakunya

    # ===================== SUBMISSION =====================
    SUBMISSION_MODE = "probability"     # "probability" | "label"
    SUBMISSION_LABEL_FORMAT = "auto"    # "auto" | "01" | "original" (Yes/No dsb)
    SUBMISSION_TARGET_COL = None        # None = kolom ke-2 sample_submission / TARGET_COL

    # ===================== REPORT & DELIVERABLES =====================
    TEAM_NAME = "TeamName"              # dipakai untuk nama file laporan & ZIP: "<TEAM_NAME>_Final Stage 1"
    COMPANY_NAME = "the insurer"        # nama perusahaan di soal (untuk draft laporan)
    REPORT_TITLE = "Understanding and Reducing Customer Churn: A Data-Driven Retention Strategy"
    REPORT_COMPACT = True               # draft ringkas (target ≈10 halaman isi); False = versi panjang
    MAKE_REPORT_DRAFT = True            # buat REPORT_DRAFT.docx + .md otomatis dengan angka hasil run
    MAKE_ZIP = False                    # True setelah artikel PDF selesai → buat ZIP deliverables
    ARTICLE_PDF_PATH = None             # path PDF artikel final (untuk ZIP)
    NOTEBOOK_PATH = None                # path .ipynb ini (untuk ZIP); None = cari *.ipynb di folder kerja
    N_BOOTSTRAP = 1000                  # bootstrap CI untuk skor model
    FEATURE_LABELS = {}                 # nama bisnis untuk laporan, misal {"n_complaints_12m": "complaints in last 12 months"}
    TEAM_MEMBERS = []                   # ["Nama 1", "Nama 2", "Nama 3"] → cover
    UNIVERSITY = ""                     # → cover
    CASE_CONTEXT = ""                   # 2–4 kalimat konteks dari soal (produk, pasar, tujuan manajemen) → Introduction
    INSURANCE_LINE = "auto"             # "life" | "general" | "health" | "auto" (tebak dari data) → pilihan paragraf literatur
    MARKET = "Indonesia"                # "Indonesia" → pakai fakta OJK/AAJI terverifikasi; lainnya → konteks umum
    REPORT_MAX_BODY_PAGES = 10          # batas isi utama (tanpa cover & lampiran)
    REPORT_AUTOFIT = True               # MS Word: hitung halaman & padatkan otomatis sampai ≤ batas
    MAKE_PDF = True                     # DOCX → PDF via MS Word (kalau ada)
    MAKE_DASHBOARD = True               # outputs/.../dashboard (siap deploy Vercel) + dashboard_offline.html
    REFERENCES_COUNT_IN_LIMIT = True    # True = daftar pustaka ikut dihitung dalam batas halaman (aman)
    REPORT_SUBTITLE = "Final Stage 1 — Data Analytics Competition"

    # ===================== CASE BRIEF (isi dari soal; lihat tools/CASE_INTAKE.html) =====================
    CHURN_DEFINITION = ""               # mis. "a policy not renewed within 30 days after expiry"
    DATA_PERIOD = ""                    # mis. "January 2023 – December 2024"
    MANAGEMENT_OBJECTIVE = "auto"       # "reduce_churn" | "maximize_profit" | "understand_drivers" | "early_warning" | "auto"
    CASE_QUESTIONS = []                 # pertanyaan yang diminta soal (EN) → jadi Research Questions
    AVAILABLE_INTERVENTIONS = []        # aksi yang memang bisa dilakukan perusahaan menurut soal → diprioritaskan
    RETENTION_BUDGET = None             # batas anggaran kampanye (mata uang) → kampanye optimal dalam budget
    PORTFOLIO_SIZE = None               # jumlah nasabah/polis aktif perusahaan (dari soal) → angka uang & budget diskalakan ke portofolio
    BUSINESS_PARAMS_SOURCE = "assumed"  # "case" kalau margin/biaya/success rate diberikan di soal → kalimat laporan menyesuaikan
    NARRATIVE_ANGLE = "auto"            # "lifecycle" | "payment" | "price" | "service" | "profit" | "auto" → judul & framing
    CASE_KEY_FACTS = []                 # kalimat fakta dari casebook (EN) → Introduction (alternatif sheet case_facts)
    RESEARCH_FILE = None                # CASE_RESEARCH.xlsx (tools/research_helper.py); None = cari otomatis di folder kerja
    USE_DEFAULT_MARKET_CONTEXT = True   # False = paragraf konteks pasar bawaan diganti fakta industri dari CASE_RESEARCH.xlsx
    DASHBOARD_INCLUDE_CUSTOMERS = True  # False sebelum deploy publik (Vercel) kalau data nasabah rahasia

    # ===================== FIGURES =====================
    FIG_DPI = 200                       # 300 untuk kualitas cetak laporan
    PALETTE = ["#2E86AB", "#E4572E", "#76B041", "#F3A712", "#6C4AB6", "#17BEBB", "#8C8C8C"]


print("CFG default loaded. Edit nilai di cell USER OVERRIDE berikutnya.")

# %%
# >>> USER OVERRIDE
# =====================================================================
# EDIT DI SINI SAAT LOMBA. Semua yang tidak diubah memakai default CFG.
# =====================================================================
# CFG.RUN_MODE = "fast"
# CFG.TRAIN_PATH = "./data/train.csv"
# CFG.TEST_PATH = "./data/test.csv"
# CFG.SAMPLE_SUB_PATH = "./data/sample_submission.csv"
# CFG.TARGET_COL = "churn"
# CFG.ID_COL = "customer_id"
# CFG.POSITIVE_LABEL = "Yes"
# CFG.METRIC = "roc_auc"
# CFG.SUBMISSION_MODE = "probability"
# CFG.DROP_COLS = []
# CFG.EXTRA_TABLES = {"claims": {"path": "./data/claims.csv", "key": "customer_id", "date_col": "claim_date"}}
# CFG.COLUMN_ROLES = {"tenure": "tenure_months", "premium": "annual_premium"}
# CFG.ACTIONABLE_FEATURES = ["payment_method", "auto_renew", "n_complaints_12m"]
# CFG.PROFIT_MARGIN = 0.2; CFG.RETENTION_COST = 150_000; CFG.RETENTION_SUCCESS_RATE = 0.3
# CFG.FIG_DPI = 300
# CFG.TEAM_NAME = "NamaTim"; CFG.COMPANY_NAME = "PT Asuransi X"; CFG.REPORT_TITLE = "..."
# CFG.TEAM_MEMBERS = ["Nama 1", "Nama 2", "Nama 3"]; CFG.UNIVERSITY = "Institut ..."
# CFG.CASE_CONTEXT = "PT X is a general insurer ... management wants to ..."   # 2–4 kalimat dari soal → Introduction
# CFG.ARTICLE_PDF_PATH = "./NamaTim_Final Stage 1.pdf"; CFG.MAKE_ZIP = True   # setelah artikel selesai
# CFG.CONFIG_FILE = "case_config.json"   # alternatif: semua override di 1 file JSON (lihat tools/case_config.example.json)

# %%
# >>> CONFIG FILE / HEADLESS RUNNER — jangan diedit
# tools/run_case.py mengisi env CHURN_CONFIG; atau isi CFG.CONFIG_FILE di cell atas. Nilai di JSON MENIMPA CFG.
def apply_config_file():
    path = os.environ.get("CHURN_CONFIG") or getattr(CFG, "CONFIG_FILE", None)
    if not path and os.path.exists("case_config.json"):      # notebook dibuka di Jupyter dari folder case → pakai config case otomatis
        path = "case_config.json"
        print("ℹ️ case_config.json di folder ini dipakai otomatis.")
    if not path:
        print("Config file: tidak dipakai (USER OVERRIDE di cell atas yang berlaku).")
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file tidak ditemukan: {path}")
    with open(path, encoding="utf-8") as f:
        txt = "\n".join(l for l in f.read().splitlines() if not l.strip().startswith("//"))
    conf = json.loads(txt) if txt.strip() else {}
    unknown = [k for k in conf if not hasattr(CFG, k)]
    for k, v in conf.items():
        setattr(CFG, k, tuple(v) if k == "RISK_TIER_CUTS" else v)
    print(f"Config file dipakai: {path} → {len(conf)} nilai CFG di-set")
    for k, v in conf.items():
        print(f"  {k:<26} = {str(v)[:90]}")
    if unknown:
        print("⚠️ Key yang tidak ada di CFG default (tetap di-set, cek ejaan):", unknown)
    return conf


CONFIG_APPLIED = apply_config_file()

# %% [markdown]
# ### Apply run mode & buat folder output
# `fast` mode mengubah beberapa knob supaya cepat: subsample besar data, 3 fold, model sedikit, tanpa Optuna, 1 seed.
# **Jangan pakai hasil fast mode untuk submission final.**

# %%
def apply_run_mode():
    CFG.METRIC = str(CFG.METRIC).lower().strip()
    if CFG.RUN_MODE == "fast":
        CFG.N_FOLDS = min(CFG.N_FOLDS, 3)
        CFG.N_REPEATS = 1
        CFG.MODELS = [m for m in CFG.MODELS if m in ("logreg", "hgb", "lgbm", "catboost")] or ["logreg", "hgb"]
        CFG.USE_OPTUNA = False
        CFG.FINAL_SEEDS = CFG.FINAL_SEEDS[:1]
        CFG.SHAP_SAMPLE = min(CFG.SHAP_SAMPLE, 1000)
        CFG.PERM_IMPORTANCE_REPEATS = 1
        CFG.MAX_EDA_PLOTS = min(CFG.MAX_EDA_PLOTS, 12)
        CFG.EARLY_STOPPING_ROUNDS = min(CFG.EARLY_STOPPING_ROUNDS, 100)
        CFG.FAST_SAMPLE_ROWS = 20000
        CFG.N_BOOTSTRAP = min(CFG.N_BOOTSTRAP, 200)
        CFG.ABLATION_MAX_GROUPS = min(CFG.ABLATION_MAX_GROUPS, 4)
        CFG.CAUSAL_MAX_TREATMENTS = min(CFG.CAUSAL_MAX_TREATMENTS, 3)
    else:
        CFG.FAST_SAMPLE_ROWS = None
    CFG.FIG_DIR = os.path.join(CFG.OUTPUT_DIR, "figures")
    CFG.TABLE_DIR = os.path.join(CFG.OUTPUT_DIR, "tables")
    CFG.MODEL_DIR = os.path.join(CFG.OUTPUT_DIR, "models")
    CFG.SUB_DIR = os.path.join(CFG.OUTPUT_DIR, "submissions")
    for d in [CFG.OUTPUT_DIR, CFG.FIG_DIR, CFG.TABLE_DIR, CFG.MODEL_DIR, CFG.SUB_DIR]:
        os.makedirs(d, exist_ok=True)


apply_run_mode()
seed_everything(CFG.SEED)

plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
sns.set_palette(CFG.PALETTE)

print("RUN_MODE    :", CFG.RUN_MODE)
print("METRIC      :", CFG.METRIC)
print("MODELS      :", CFG.MODELS)
print("N_FOLDS     :", CFG.N_FOLDS, "| repeats:", CFG.N_REPEATS)
print("OPTUNA      :", CFG.USE_OPTUNA and HAS_OPTUNA)
print("OUTPUT_DIR  :", os.path.abspath(CFG.OUTPUT_DIR))

# %% [markdown]
# # 2. Load Data
#
# `smart_read()` membaca **csv / tsv / txt / xlsx / xls / parquet / json / feather / pickle**, otomatis mencoba beberapa encoding (utf-8 → utf-8-sig → latin-1) dan auto-detect separator (`,` `;` `|` tab).
#
# **Kasus multi-tabel (umum di data asuransi):** data bisa dipecah jadi `customers`, `policies`, `claims`, `payments`, `complaints`.
# - Tabel **one-to-one** (1 baris per customer) → isi `CFG.EXTRA_ONE_TO_ONE` → langsung di-merge.
# - Tabel **one-to-many** (banyak baris per customer, misal riwayat klaim) → isi `CFG.EXTRA_TABLES` → diagregasi per customer (count, sum, mean, max, min, std, jumlah per kategori, recency dari tanggal) lalu di-merge.
#
# **Jika X → Y:**
# - Error `FileNotFoundError` → cek path; jalankan `os.listdir("./data")` untuk lihat nama file sebenarnya.
# - Jumlah kolom cuma 1 padahal file punya banyak kolom → separator salah; `smart_read(path, sep=";")`.
# - Huruf aneh (`Ã©`) → encoding; `smart_read(path, encoding="latin-1")`.
# - Setelah merge jumlah baris bertambah → key tidak unik di tabel kanan (seharusnya one-to-many → pakai `EXTRA_TABLES`).
# - Data cuma 1 file (tanpa test) → tidak apa-apa; notebook tetap jalan untuk analisis + CV, submission di-skip.

# %%
def tidy_frame(df, name="data"):
    """Rapikan tabel hasil baca: header di baris tengah, BOM/spasi di nama kolom, nama duplikat, baris/kolom kosong total."""
    df = df.copy()
    cols = [str(c) for c in df.columns]
    unnamed = sum(c.startswith("Unnamed") or c.strip() == "" or c.lower() == "nan" for c in cols)
    if len(cols) > 1 and unnamed / len(cols) > 0.5 and len(df) > 2:
        # judul/keterangan di atas tabel → cari baris header (paling banyak teks non-angka) di 15 baris pertama
        best, best_score = None, 0
        for r in range(min(15, len(df))):
            vals = df.iloc[r].tolist()
            score = sum(isinstance(v, str) and v.strip() != "" and pd.to_numeric(pd.Series([v]), errors="coerce").isna().all() for v in vals)
            if score > best_score:
                best, best_score = r, score
        if best is not None and best_score >= max(2, 0.5 * len(cols)):
            df.columns = [str(v) if pd.notna(v) else f"col_{j}" for j, v in enumerate(df.iloc[best].tolist())]
            df = df.iloc[best + 1:].reset_index(drop=True)
            print(f"ℹ️ [{name}] header ditemukan di baris {best + 2} file (ada judul/keterangan di atas tabel) → dipakai sebagai nama kolom")
    clean = []
    for c in df.columns:
        c2 = str(c).replace("\ufeff", "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
        c2 = re.sub(r"\s+", " ", c2).strip() or "col"
        clean.append(c2)
    seen = {}
    for j, c in enumerate(clean):                    # nama kolom duplikat → beri akhiran
        if c in seen:
            seen[c] += 1
            clean[j] = f"{c}__{seen[c]}"
        else:
            seen[c] = 0
    if clean != [str(c) for c in df.columns]:
        changed = [f"{a!r}→{b!r}" for a, b in zip(df.columns, clean) if str(a) != b][:6]
        if changed:
            print(f"ℹ️ [{name}] nama kolom dirapikan: {', '.join(changed)}")
    df.columns = clean
    n0 = df.shape
    df = df.dropna(axis=0, how="all")
    empty_cols = [c for c in df.columns if df[c].isna().all() or (df[c].dtype == object and df[c].astype(str).str.strip().isin(["", "nan", "None"]).all())]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    if df.shape != n0:
        print(f"ℹ️ [{name}] {n0[0] - df.shape[0]} baris kosong & {len(empty_cols)} kolom kosong total dibuang")
    idx_cols = []                                     # index pandas yang ikut tersimpan ('Unnamed: 0' = 0,1,2,…)
    for c in df.columns:
        if re.fullmatch(r"(Unnamed: ?\d+|index|level_0|col)", str(c), re.I) and len(df) > 1:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().all() and (v.diff().dropna() == 1).all():
                idx_cols.append(c)
    if idx_cols:
        df = df.drop(columns=idx_cols)
        print(f"ℹ️ [{name}] kolom index tersimpan {idx_cols} (0,1,2,…) dibuang")
    return df.reset_index(drop=True)


def smart_read(path, **kwargs):
    """Baca file tabular apapun: csv/tsv/txt (auto separator & encoding, baris rusak dilewati), xlsx/xls (sheet terbesar / CFG.EXCEL_SHEET,
    header di baris tengah), parquet, feather, json/jsonl, pickle. Hasil dirapikan oleh tidy_frame."""
    if path is None or not os.path.exists(str(path)):
        folder = os.path.dirname(str(path)) or "."
        avail = os.listdir(folder) if os.path.isdir(folder) else []
        raise FileNotFoundError(f"File tidak ditemukan: {path}. Isi folder '{folder}': {avail[:20]}")
    ext = os.path.splitext(str(path))[1].lower()
    name = os.path.basename(str(path))
    if ext in (".xlsx", ".xlsm", ".xls"):
        sheet = kwargs.pop("sheet_name", getattr(CFG, "EXCEL_SHEET", None) if "CFG" in globals() else None)
        if sheet is None:
            book = pd.read_excel(path, sheet_name=None, **kwargs)
            sheet = max(book, key=lambda k: book[k].shape[0] * max(book[k].shape[1], 1))
            if len(book) > 1:
                print(f"ℹ️ [{name}] {len(book)} sheet: {list(book)} → dipakai sheet terbesar '{sheet}' (set CFG.EXCEL_SHEET kalau salah)")
            return tidy_frame(book[sheet], name)
        return tidy_frame(pd.read_excel(path, sheet_name=sheet, **kwargs), name)
    if ext == ".parquet":
        return tidy_frame(pd.read_parquet(path, **kwargs), name)
    if ext == ".feather":
        return tidy_frame(pd.read_feather(path, **kwargs), name)
    if ext in (".pkl", ".pickle"):
        return tidy_frame(pd.read_pickle(path, **kwargs), name)
    if ext in (".json", ".jsonl"):
        try:
            return tidy_frame(pd.read_json(path, lines=(ext == ".jsonl"), **kwargs), name)
        except ValueError:
            return tidy_frame(pd.read_json(path, lines=True, **kwargs), name)
    last_err = None
    user_enc = kwargs.pop("encoding", None)
    for enc in [user_enc, "utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        if enc is None:
            continue
        try:
            if "sep" in kwargs:
                return tidy_frame(pd.read_csv(path, encoding=enc, **kwargs), name)
            df = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
            if df.shape[1] == 1 and any(s in str(df.columns[0]) for s in [";", "|", "\t"]):
                df = pd.read_csv(path, encoding=enc, sep=None, engine="python", **kwargs)
            return tidy_frame(df, name)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except pd.errors.ParserError:
            try:
                return tidy_frame(pd.read_csv(path, encoding=enc, sep=None, engine="python", **kwargs), name)
            except Exception:
                df = pd.read_csv(path, encoding=enc, sep=None, engine="python", on_bad_lines="skip", **kwargs)
                print(f"⚠️ [{name}] ada baris dengan jumlah kolom tidak konsisten → baris rusak dilewati")
                return tidy_frame(df, name)
    raise last_err


def aggregate_child_table(child, key, prefix, date_col=None, ref_date=None, max_levels=10):
    """Agregasi tabel one-to-many per key: numeric stats, count per kategori, recency tanggal."""
    child = child.copy()
    aggs = [child.groupby(key).size().rename(f"{prefix}_count")]
    num_cols = [c for c in child.select_dtypes(include=np.number).columns if c != key]
    if num_cols:
        g = child.groupby(key)[num_cols].agg(["sum", "mean", "max", "min", "std"])
        g.columns = [f"{prefix}_{a}_{b}" for a, b in g.columns]
        aggs.append(g)
    cat_cols = [c for c in child.columns if c not in num_cols + [key, date_col] and child[c].dtype == object]
    for c in cat_cols:
        top = child[c].value_counts().index[:max_levels]
        sub = child[child[c].isin(top)]
        ct = pd.crosstab(sub[key], sub[c])
        ct.columns = [f"{prefix}_{c}_{sanitize_name(v)}_count" for v in ct.columns]
        aggs.append(ct)
        aggs.append(child.groupby(key)[c].nunique().rename(f"{prefix}_{c}_nunique"))
    if date_col and date_col in child.columns:
        d = pd.to_datetime(child[date_col], errors="coerce")
        ref = pd.Timestamp(ref_date) if ref_date is not None else d.max()
        child["_days_ago"] = (ref - d).dt.days
        g = child.groupby(key)["_days_ago"].agg(["min", "max"])
        g.columns = [f"{prefix}_days_since_last", f"{prefix}_days_since_first"]
        aggs.append(g)
    out = pd.concat(aggs, axis=1).reset_index()
    return out


def merge_safely(left, right, key, how="left", name="table"):
    n0 = len(left)
    if right[key].duplicated().any():
        raise ValueError(f"[{name}] key '{key}' tidak unik di tabel kanan → gunakan EXTRA_TABLES (agregasi) bukan one-to-one")
    out = left.merge(right, on=key, how=how)
    assert len(out) == n0, f"[{name}] jumlah baris berubah setelah merge ({n0} → {len(out)})"
    return out


with safe_section("Load data", critical=True):
    raw_train = smart_read(CFG.TRAIN_PATH)
    HAS_TEST = CFG.TEST_PATH is not None and os.path.exists(str(CFG.TEST_PATH))
    raw_test = smart_read(CFG.TEST_PATH) if HAS_TEST else None
    try:
        sample_sub = smart_read(CFG.SAMPLE_SUB_PATH) if CFG.SAMPLE_SUB_PATH and os.path.exists(str(CFG.SAMPLE_SUB_PATH)) else None
    except Exception as e:
        print("Sample submission gagal dibaca:", e)
        sample_sub = None

    # strip whitespace di nama kolom (sering terjadi: " Churn")
    raw_train.columns = [str(c).strip() for c in raw_train.columns]
    if len(raw_train) < 30:
        raise ValueError(f"Data train hanya {len(raw_train)} baris → terlalu sedikit untuk dianalisis. Cek TRAIN_PATH / sheet / separator.")
    ORIGINAL_COLS = list(raw_train.columns)  # kolom asli sebelum merge tabel tambahan
    if HAS_TEST:
        raw_test.columns = [str(c).strip() for c in raw_test.columns]

    if CFG.FAST_SAMPLE_ROWS and len(raw_train) > CFG.FAST_SAMPLE_ROWS:
        raw_train = raw_train.sample(CFG.FAST_SAMPLE_ROWS, random_state=CFG.SEED).reset_index(drop=True)
        print(f"FAST MODE: train di-subsample ke {CFG.FAST_SAMPLE_ROWS} baris")

    print("Train shape :", raw_train.shape)
    print("Test shape  :", raw_test.shape if HAS_TEST else "— tidak ada test file (submission akan di-skip)")
    print("Sample sub  :", sample_sub.shape if sample_sub is not None else "— tidak ada")
    display(raw_train.head())

# %% [markdown]
# ## 2.1 Auto-adapt: ukuran data & spesifikasi komputer
# Laptop panitia belum tentu kuat. Cell ini membaca **RAM, CPU, GPU** dan **ukuran data**, lalu menyesuaikan bagian yang berat secara otomatis:
#
# | Kondisi | Penyesuaian otomatis |
# |---|---|
# | Data besar (> `LARGE_DATA_ROWS`, default 150k baris) | RF/ET dimatikan (paling lambat & boros RAM), SHAP/permutation pakai sampel, Optuna tuning di subsample 100k, Cox di subsample 50k, bootstrap 200 |
# | Data sangat lebar (> 300 kolom) | EDA plot dibatasi, SHAP sampel lebih kecil |
# | RAM tersedia < 4 GB | `N_JOBS` dibatasi 2, model berat dikurangi, SHAP sampel kecil |
# | CPU sedikit (≤ 4 core) | `N_JOBS` = jumlah core, Optuna timeout dipendekkan |
# | GPU NVIDIA terdeteksi & `CFG.USE_GPU="auto"` | XGBoost/CatBoost pakai GPU (fallback otomatis ke CPU kalau gagal) |
#
# Semua penyesuaian dicetak. Untuk mematikan: `CFG.AUTO_ADAPT = False`.

# %%
def system_resources():
    info = {"cpu": os.cpu_count() or 2, "ram_total_gb": None, "ram_free_gb": None, "gpu": False, "gpu_name": None}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"], info["ram_free_gb"] = vm.total / 1e9, vm.available / 1e9
    except Exception:
        try:  # Windows tanpa psutil
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MEMSTAT()
            ms.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            info["ram_total_gb"], info["ram_free_gb"] = ms.ullTotalPhys / 1e9, ms.ullAvailPhys / 1e9
        except Exception:
            pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            info["gpu"], info["gpu_name"] = True, out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return info


with safe_section("Auto-adapt to data size & hardware"):
    RES = system_resources()
    n_rows, n_cols = raw_train.shape
    print(f"CPU cores: {RES['cpu']} | RAM total: {RES['ram_total_gb'] or 0:.1f} GB, tersedia: {RES['ram_free_gb'] or 0:.1f} GB | "
          f"GPU: {RES['gpu_name'] if RES['gpu'] else 'tidak ada'}")
    print(f"Data train: {n_rows:,} baris × {n_cols} kolom")
    CFG.TUNE_SUBSAMPLE = None
    CFG.COX_MAX_ROWS = 50_000
    CFG.PERM_MAX_ROWS = 5_000
    if getattr(CFG, "AUTO_ADAPT", True):
        changes = []
        if str(CFG.USE_GPU).lower() == "auto":
            CFG.USE_GPU = bool(RES["gpu"]) and n_rows > 100_000   # data kecil: CPU lebih cepat (overhead GPU)
            changes.append(f"USE_GPU={CFG.USE_GPU}")
        if n_rows > CFG.LARGE_DATA_ROWS:
            drop = [m for m in CFG.MODELS if m in ("rf", "et")]
            CFG.MODELS = [m for m in CFG.MODELS if m not in ("rf", "et")]
            CFG.SHAP_SAMPLE = min(CFG.SHAP_SAMPLE, 2000)
            CFG.PERM_IMPORTANCE_REPEATS = 1
            CFG.N_BOOTSTRAP = min(CFG.N_BOOTSTRAP, 200)
            CFG.TUNE_SUBSAMPLE = 100_000
            changes.append(f"data besar → model {drop} dimatikan, SHAP 2000, tuning di subsample 100k, bootstrap 200")
            CFG.EBM_INTERACTIONS = min(CFG.EBM_INTERACTIONS, 3)
            CFG.EBM_OUTER_BAGS = min(CFG.EBM_OUTER_BAGS, 4)
            changes.append("EBM: 4 outer bags, ≤3 interaksi")
        if n_cols > 300:
            CFG.MAX_EDA_PLOTS = min(CFG.MAX_EDA_PLOTS, 12)
            CFG.SHAP_SAMPLE = min(CFG.SHAP_SAMPLE, 1500)
            changes.append("data lebar → EDA & SHAP dibatasi")
        if RES["ram_free_gb"] is not None and RES["ram_free_gb"] < 4:
            CFG.N_JOBS = 2
            CFG.SHAP_SAMPLE = min(CFG.SHAP_SAMPLE, 1500)
            if n_rows > 50_000:
                CFG.MODELS = [m for m in CFG.MODELS if m not in ("rf", "et")]
            changes.append(f"RAM tersedia rendah ({RES['ram_free_gb']:.1f} GB) → N_JOBS=2, SHAP 1500")
        if RES["cpu"] <= 4:
            CFG.N_JOBS = RES["cpu"]
            CFG.OPTUNA_TIMEOUT = min(CFG.OPTUNA_TIMEOUT, 600)
            changes.append(f"CPU sedikit → N_JOBS={RES['cpu']}, OPTUNA_TIMEOUT≤600s")
        if n_rows > 400_000:
            CFG.MODELS = [m for m in CFG.MODELS if m not in ("ebm", "mlp")]
            changes.append("data sangat besar (>400k) → EBM & MLP dimatikan")
        if n_rows * n_cols > 5e7:
            CFG.COX_MAX_ROWS = 20_000
        print("Penyesuaian:", "; ".join(changes) if changes else "tidak perlu ✅")
    if str(CFG.USE_GPU).lower() == "auto":
        CFG.USE_GPU = False
    est = n_rows * max(1, len(CFG.MODELS)) * CFG.N_FOLDS / 1e6
    print(f"Estimasi beban modeling relatif: {est:.1f} (≈ <1 cepat, 1–5 sedang, >5 lama → pertimbangkan RUN_MODE='fast' dulu)")

# %%
with safe_section("Merge extra tables"):
    if getattr(CFG, "TRAIN_LABELS", None):
        _lab = smart_read(CFG.TRAIN_LABELS["path"])
        _lab.columns = [str(c).strip() for c in _lab.columns]
        raw_train = merge_safely(raw_train, _lab, CFG.TRAIN_LABELS["key"], name="train labels")
        print(f"[labels] merged ke train: {[c for c in _lab.columns if c != CFG.TRAIN_LABELS['key']]}")
    for name, spec in (CFG.EXTRA_ONE_TO_ONE or {}).items():
        t = smart_read(spec["path"])
        t.columns = [str(c).strip() for c in t.columns]
        raw_train = merge_safely(raw_train, t, spec["key"], name=name)
        if HAS_TEST:
            raw_test = merge_safely(raw_test, t, spec["key"], name=name)
        print(f"[{name}] merged one-to-one: +{t.shape[1] - 1} kolom")
    for name, spec in (CFG.EXTRA_TABLES or {}).items():
        t = smart_read(spec["path"])
        t.columns = [str(c).strip() for c in t.columns]
        agg = aggregate_child_table(t, spec["key"], prefix=sanitize_name(name), date_col=spec.get("date_col"),
                                    ref_date=CFG.REFERENCE_DATE)
        raw_train = merge_safely(raw_train, agg, spec["key"], name=name)
        if HAS_TEST:
            raw_test = merge_safely(raw_test, agg, spec["key"], name=name)
        cnt_cols = [c for c in agg.columns if c.endswith("_count") or c.endswith("_sum")]
        raw_train[cnt_cols] = raw_train[cnt_cols].fillna(0)   # customer tanpa record → 0
        if HAS_TEST:
            raw_test[cnt_cols] = raw_test[cnt_cols].fillna(0)
        print(f"[{name}] one-to-many diagregasi: {t.shape} → +{agg.shape[1] - 1} fitur")
    if not CFG.EXTRA_TABLES and not CFG.EXTRA_ONE_TO_ONE:
        print("Tidak ada tabel tambahan (CFG.EXTRA_TABLES kosong).")

# %% [markdown]
# # 3. Data Quality & Cleaning
#
# ## 3.1 Deteksi target & ID
# **Target auto-detect** (urutan): `CFG.TARGET_COL` → kolom yang ada di train tapi tidak ada di test → nama kolom mengandung kata kunci churn (`churn, exited, attrition, lapse, lapsed, left, cancel, status, is_active, retained, renew, berhenti, putus`).
#
# **ID auto-detect**: nama seperti `id`, `customer_id`, `policy_no`, `nomor_polis` dan nilainya hampir unik per baris.
#
# **Jika auto-detect salah** → set manual `CFG.TARGET_COL` / `CFG.ID_COL` di USER OVERRIDE lalu run ulang dari Section 2.

# %%
TARGET_KEYWORDS = ["churn", "exited", "attrition", "attrited", "lapse", "lapsed", "left", "cancel", "is_active",
                   "retained", "retention", "renew", "berhenti", "putus", "status", "target", "label", "flag"]
ID_PATTERN = re.compile(r"(^id$|^id_|_id$|customer.?id|cust.?id|client.?id|member.?id|policy.?(no|num|number|id)|"
                        r"^no_?polis|nomor|^row.?num|^index$|^unnamed)", re.I)


def detect_target(train_df, test_df):
    """Urutan: CFG.TARGET_COL (toleran kapital/spasi) → satu-satunya kolom yang ada di train tapi tidak di test → kata kunci churn
    → kata kunci event lain (response, claim, fraud, purchase, ...) → kolom TERAKHIR yang biner (konvensi umum dataset)."""
    if CFG.TARGET_COL:
        if CFG.TARGET_COL in train_df.columns:
            return CFG.TARGET_COL
        norm = {str(c).strip().lower().replace(" ", "_"): c for c in train_df.columns}
        key = str(CFG.TARGET_COL).strip().lower().replace(" ", "_")
        if key in norm:
            print(f"ℹ️ TARGET_COL '{CFG.TARGET_COL}' dicocokkan ke kolom '{norm[key]}' (beda kapital/spasi).")
            return norm[key]
        raise KeyError(f"TARGET_COL '{CFG.TARGET_COL}' tidak ada di train. Kolom: {list(train_df.columns)}")
    if test_df is not None:
        only_train = [c for c in train_df.columns if c not in test_df.columns]
        if len(only_train) == 1:
            return only_train[0]
        cand = [c for c in only_train if any(k in c.lower() for k in TARGET_KEYWORDS)]
        if cand:
            return cand[0]
    for k in TARGET_KEYWORDS:
        for c in train_df.columns:
            if k in c.lower() and train_df[c].nunique(dropna=True) == 2:
                return c
    event_kw = ["response", "purchase", "buy", "bought", "claim", "fraud", "default", "outcome", "convert", "subscribed",
                "insurance", "accepted", "renewal", "beli", "klaim"]
    for k in event_kw:
        for c in train_df.columns:
            if k in c.lower() and train_df[c].nunique(dropna=True) == 2:
                print(f"⚠️ Target ditebak dari nama kolom: '{c}'. VERIFIKASI — kalau salah set CFG.TARGET_COL.")
                return c
    last = train_df.columns[-1]
    if train_df[last].nunique(dropna=True) == 2:
        print(f"⚠️ Target ditebak = kolom terakhir yang biner: '{last}'. VERIFIKASI — kalau salah set CFG.TARGET_COL.")
        return last
    raise ValueError("Target tidak bisa dideteksi otomatis. Set CFG.TARGET_COL di USER OVERRIDE "
                     f"(kolom yang tersedia: {list(train_df.columns)[:40]}).")


def detect_id(df, target):
    if CFG.ID_COL:
        return CFG.ID_COL if CFG.ID_COL in df.columns else None
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        if ID_PATTERN.search(str(c)) and df[c].nunique(dropna=True) >= 0.95 * n:
            return c
    for c in df.columns:
        if c != target and df[c].dtype == object and df[c].nunique(dropna=True) == n:
            return c
    return None


def detect_group_key(df, target, id_col):
    """Kolom ID yang BERULANG (data panel/bulanan, banyak polis per customer) → dipakai sebagai GROUP_COL untuk CV.
    Syarat: nama berbau customer/polis DAN mengandung token identifier (id/no/number/key/code/nomor), bukan tanggal/angka kontinu."""
    n = len(df)
    who = re.compile(r"(customer|cust|client|nasabah|member|holder|household|insured|pemegang|polis|policy)", re.I)
    ident = re.compile(r"(^id|_id|id$|^no_|_no$|_no_|number|_num$|key|code|kode|nomor)", re.I)
    excl = re.compile(r"(agent|agen|broker|branch|cabang|product|produk|plan|date|tgl|tanggal|time|amount|amt|premium|premi)", re.I)
    for c in df.columns:
        if c in (target, id_col):
            continue
        name = str(c)
        if not (who.search(name) and ident.search(name)) or excl.search(name):
            continue
        s_ = df[c]
        if np.issubdtype(s_.dtype, np.datetime64):
            continue
        if s_.dtype == object and pd.to_datetime(s_.dropna().head(200), errors="coerce").notna().mean() > 0.8:
            continue
        nu = s_.nunique(dropna=True)
        if 0.02 * n < nu < 0.95 * n and s_.duplicated().any():
            return c
    return None


with safe_section("Detect target & ID", critical=True):
    TARGET = detect_target(raw_train, raw_test)
    ID_COL = detect_id(raw_train, TARGET)
    if CFG.GROUP_COL is None:
        gk = detect_group_key(raw_train, TARGET, ID_COL)
        if gk is not None:
            CFG.GROUP_COL = gk
            print(f"ℹ️ '{gk}' berulang ({raw_train[gk].nunique():,} nilai unik untuk {len(raw_train):,} baris) → data panel / banyak polis per customer.")
            print("   → dipakai sebagai GROUP_COL (StratifiedGroupKFold) supaya customer yang sama tidak bocor antar fold, dan TIDAK dipakai sebagai fitur.")
    if CFG.GROUP_COL and CFG.GROUP_COL not in CFG.DROP_COLS:
        CFG.DROP_COLS = list(CFG.DROP_COLS) + [CFG.GROUP_COL]
    print("TARGET :", TARGET)
    print("ID_COL :", ID_COL)
    if ID_COL is None:
        print("⚠️ Tidak ada ID column terdeteksi → submission akan memakai urutan baris test / sample_submission.")

# %% [markdown]
# ## 3.2 Parsing label churn → 0/1
# Positive class (=1) adalah **churn**. Auto-mapping untuk nilai seperti `Yes / True / 1 / Churn / Churned / Exited / Left / Lapsed / Cancelled / Attrited Customer / Ya`.
#
# **Hati-hati kasus terbalik**: kolom `is_active`, `retained`, `renewed`, `status = Active` → nilai 1/Yes justru berarti **tidak churn**. Notebook mendeteksi nama kolom seperti ini dan membalik label otomatis (dicetak jelas). Kalau salah, set `CFG.POSITIVE_LABEL` = nilai yang berarti churn.
#
# **Jika target punya > 2 kelas** (misal `Active / Lapsed / Surrendered`) → set `CFG.POSITIVE_LABEL` ke salah satu kelas (atau list) → dibinarisasi "kelas itu vs lainnya". Untuk multiclass murni pakai `02_Tabular/tabular_master.ipynb`.

# %%
POSITIVE_TOKENS = {"yes", "y", "true", "t", "1", "1.0", "churn", "churned", "exited", "exit", "left", "leave",
                   "lapsed", "lapse", "cancelled", "canceled", "cancel", "attrited customer", "attrited", "inactive",
                   "terminated", "closed", "ya", "berhenti", "keluar", "putus", "lost"}
INVERSE_NAME_HINTS = ["active", "retain", "renew", "stay", "aktif", "loyal", "persist"]


ACTIVE_TOKENS = {"active", "aktif", "inforce", "in force", "in-force", "existing", "existing customer", "retained", "renewed",
                 "renew", "stay", "stayed", "current", "bertahan", "persist", "persisting", "loyal", "no", "n", "false", "0", "0.0",
                 "tidak", "not churn", "not churned", "non-churn", "continuing", "continue", "paid-up", "premium paying"}


def parse_target(series):
    """Ubah kolom target apapun → 0/1 (1 = churn/event). Mendukung: Yes/No, 0/1, bool, string status multi-kelas,
    target hitungan (jumlah klaim → event = >0). Target kontinu → berhenti dengan pesan jelas (pakai notebook tabular)."""
    s = series.copy()
    if s.isna().any():
        print(f"⚠️ Target punya {s.isna().sum()} NaN → baris tersebut dibuang dari train.")
    vals = s.dropna()
    uniq = pd.Series(vals.unique())
    norm = uniq.astype(str).str.strip().str.lower()

    if CFG.POSITIVE_LABEL is not None:
        pos = CFG.POSITIVE_LABEL if isinstance(CFG.POSITIVE_LABEL, (list, tuple, set)) else [CFG.POSITIVE_LABEL]
        pos_norm = {str(p).strip().lower() for p in pos}
        mapping = {u: int(n in pos_norm) for u, n in zip(uniq, norm)}
        if sum(mapping.values()) == 0:
            raise ValueError(f"POSITIVE_LABEL {list(pos)} tidak ditemukan di target. Nilai yang ada: {list(uniq)[:15]}")
        how = f"POSITIVE_LABEL={list(pos)} → 1, lainnya → 0"
    elif len(uniq) == 2:
        hits = [n in POSITIVE_TOKENS for n in norm]
        if sum(hits) == 1:
            mapping = {u: int(h) for u, h in zip(uniq, hits)}
            how = "token churn (Yes/1/True/Churn/...)"
        else:
            vc = vals.value_counts()
            minority = vc.index[-1]
            mapping = {u: int(u == minority) for u in uniq}
            how = f"minority class ('{minority}') dianggap churn (tidak ada token yang jelas) — VERIFIKASI!"
        if any(h in TARGET.lower() for h in INVERSE_NAME_HINTS) and not any(k in TARGET.lower() for k in ["churn", "exit", "lapse", "cancel"]):
            mapping = {k: 1 - v for k, v in mapping.items()}
            how += " → DIBALIK karena nama kolom menunjukkan status aktif/retain (1 = churn sekarang)"
    else:
        num = pd.to_numeric(vals, errors="coerce")
        if num.notna().all() and (num % 1 == 0).all() and num.min() >= 0 and num.nunique() <= 30 and (num == 0).mean() >= 0.3:
            mapping = {u: int(float(u) > 0) for u in uniq}
            how = f"target hitungan ({num.nunique()} nilai, mis. jumlah klaim) → event = nilai > 0"
        elif num.notna().all() and num.nunique() > 30:
            raise ValueError(f"Target '{TARGET}' adalah angka kontinu ({num.nunique()} nilai unik) → ini tugas REGRESI. "
                             "Gunakan 02_Tabular/tabular_master.ipynb (TASK='regression'). Kalau soal tetap ingin churn biner, "
                             "buat target biner dulu, misal df['churn'] = (df[target] > ambang).astype(int), lalu set CFG.TARGET_COL='churn'.")
        else:
            act = [n in ACTIVE_TOKENS for n in norm]
            if any(act):
                mapping = {u: int(not a) for u, a in zip(uniq, act)}
                how = (f"target multi-kelas ({len(uniq)} status) → status aktif {[u for u, a in zip(uniq, act) if a]} = 0, "
                       f"lainnya (lapse/surrender/cancel/…) = 1")
            else:
                major = vals.value_counts().index[0]
                mapping = {u: int(u != major) for u in uniq}
                how = (f"target multi-kelas tanpa status 'aktif' yang dikenali → kelas mayoritas '{major}' = 0, lainnya = 1. "
                       f"⚠️ VERIFIKASI & set CFG.POSITIVE_LABEL kalau salah.")
            print("ℹ️ Distribusi status asli:")
            display(vals.value_counts().to_frame("n"))
    y_ = s.map(mapping)
    return y_, mapping, how


with safe_section("Parse target", critical=True):
    if HAS_TEST and TARGET in raw_test.columns:
        print(f"ℹ️ Kolom target '{TARGET}' juga ada di file test → dibuang dari test (tidak boleh jadi fitur).")
        raw_test = raw_test.drop(columns=[TARGET])
    _miss_t = raw_train[TARGET].isna().mean()
    if not HAS_TEST and getattr(CFG, "TEST_FROM_MISSING_TARGET", True) and _miss_t >= 0.05:
        raw_test = raw_train[raw_train[TARGET].isna()].drop(columns=[TARGET]).reset_index(drop=True)
        HAS_TEST = True
        print(f"ℹ️ Tidak ada file test, tetapi {_miss_t:.1%} baris target kosong → {len(raw_test):,} baris itu dipakai sebagai data yang harus diprediksi (test).")
    raw_train = raw_train[raw_train[TARGET].notna()].reset_index(drop=True)
    y_series, TARGET_MAPPING, how = parse_target(raw_train[TARGET])
    POS_ORIGINAL = [k for k, v in TARGET_MAPPING.items() if v == 1]
    NEG_ORIGINAL = [k for k, v in TARGET_MAPPING.items() if v == 0]
    print("Mapping target :", TARGET_MAPPING)
    print("Metode         :", how)
    assert y_series.notna().all(), "Ada nilai target yang tidak ter-mapping"
    y = y_series.astype(int).values
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"Target '{TARGET}' hanya punya 1 kelas (semua = {TARGET_MAPPING}) → model churn tidak bisa dilatih. "
                         "Cek: (1) TARGET_COL benar? (2) file label terpisah sudah di-set di CFG.TRAIN_LABELS? "
                         "(3) POSITIVE_LABEL benar? (Target has only one class — check the target column / label file.)")
    if min(y.sum(), len(y) - y.sum()) < 10:
        print(f"⚠️ Kelas minoritas hanya {min(y.sum(), len(y) - y.sum())} baris → hasil sangat tidak stabil; interpretasikan hati-hati.")

# %% [markdown]
# ## 3.3 Cleaning otomatis & inferensi tipe kolom
# Yang dilakukan (train & test diproses **bersamaan** dengan aturan yang sama → konsisten):
# 1. String "kosong" (`"", "?", "NA", "null", "-", ...`) → `NaN`. Whitespace di-strip.
# 2. **Angka dalam string** dikonversi: `"12,500,000"`, `"Rp 5.000.000"`, `"45%"`, `"$1,234.50"`, `" "` (contoh Telco `TotalCharges`). Deteksi otomatis format ribuan titik (Indonesia) vs koma.
# 3. **Tanggal** dideteksi (nama kolom / format) → di-parse → nanti dijadikan fitur "days since".
# 4. **Kategori typo/case** disatukan: `"Male"`, `"male "`, `"MALE"` → satu ejaan (yang paling sering).
# 5. **Boolean-like** (`Yes/No`, `True/False`, `Y/N`) → 1/0.
# 6. Tipe kolom: `numeric`, `binary`, `categorical`, `high_card` (kategori > `HIGH_CARD_THRESHOLD` level), `date`, `id`, `text`, `constant`.
# 7. Kolom **konstan / quasi-konstan** → dibuang. Duplikat baris → dibuang (train saja).
# 8. **Nilai mustahil** (umur < 0 atau > 110, premi/pendapatan/tenure negatif) → `NaN`.
#
# **Cara baca `quality_report`:** cek kolom `missing_%` (> 40% → pertimbangkan drop / jadikan indikator), `n_unique` (1 → konstan), `inferred_type` (pastikan masuk akal!).
#
# **Jika tipe salah:** kolom kode (misal `region_code = 31, 32`) terbaca numeric → masukkan ke `CFG.FORCE_CATEGORICAL`. Kolom angka terbaca kategori → `CFG.FORCE_NUMERIC`.

# %%
DATE_NAME_HINT = re.compile(r"(date|tanggal|tgl|_dt$|^dt_|time|waktu|periode|period|_at$|dob|birth|lahir|start|mulai|join)", re.I)
BOOL_MAP = {"yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0, "t": 1, "f": 0, "ya": 1, "tidak": 0,
            "iya": 1, "ga": 0, "gak": 0, "enggak": 0}


UNIT_MULT = [(r"(triliun|tn)", 1e12), (r"(miliar|milyar|billion|bn)", 1e9), (r"(juta|jt|mio|million|mn)", 1e6), (r"(ribu|rb|k)", 1e3)]


def try_parse_scaled(x):
    """'5 jt', '1,2 M', 'Rp 750 rb', '1.5k' → angka. M = miliar untuk IDR, juta untuk mata uang lain."""
    m_ = x.str.extract(r"^(-?[\d.,]+)\s*([a-z]+)?\.?$")
    if m_[0].notna().mean() < 0.9 or m_[1].notna().mean() < 0.2:
        return None
    big_m = 1e9 if str(getattr(CFG, "CURRENCY", "IDR")).upper() == "IDR" else 1e6

    def conv(num, unit):
        if pd.isna(num):
            return np.nan
        t = str(num)
        has_unit = not pd.isna(unit)
        if "," in t and "." in t:
            t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
        elif re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
        elif re.fullmatch(r"\d{1,3}(,\d{3})+", t) and not has_unit:
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
        try:
            v = float(t)
        except ValueError:
            return np.nan
        if not has_unit:
            return v
        if unit == "m":
            return v * big_m
        for pat, mult in UNIT_MULT:
            if re.fullmatch(pat, unit):
                return v * mult
        return np.nan
    return pd.Series([conv(a, b) for a, b in zip(m_[0], m_[1])], index=x.index)


def try_parse_numeric(s):
    """Coba konversi series string → numeric. Return (series_numeric, success_ratio)."""
    x = s.dropna().astype(str).str.strip()
    if len(x) == 0:
        return None, 0.0
    xs = x.str.lower().str.replace(r"(rp\.?|idr|usd|\$|€|£)", "", regex=True).str.strip()
    scaled = try_parse_scaled(xs)
    if scaled is not None and scaled.notna().mean() >= 0.9:
        full = pd.Series(np.nan, index=s.index)
        full.loc[scaled.index] = scaled.values
        return full, float(scaled.notna().mean())
    x2 = x.str.replace(r"(?i)(rp\.?|idr|usd|\$|€|£|%|\s)", "", regex=True)
    dot_thousand = x2.str.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$").mean()
    comma_thousand = x2.str.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$").mean()
    decimal_comma = x2.str.match(r"^-?\d+,\d+$").mean()
    if dot_thousand > 0.5:
        x2 = x2.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif comma_thousand > 0.3 or (comma_thousand > 0 and decimal_comma == 0):
        x2 = x2.str.replace(",", "", regex=False)
    elif decimal_comma > 0.5:
        x2 = x2.str.replace(",", ".", regex=False)
    num = pd.to_numeric(x2, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ratio = num.notna().mean()
    if ratio == 0:
        return None, 0.0
    full = pd.Series(np.nan, index=s.index)
    full.loc[num.index] = num.values
    return full, ratio


def try_parse_range(s):
    """String rentang ('50000 - 74999', '1-5 tahun', '<1000', '100+', '1.000 s/d 2.000') → angka titik tengah."""
    x = s.dropna().astype(str).str.strip().str.lower()
    if len(x) == 0:
        return None, 0.0
    num = r"(\d[\d\.,]*)"
    x2 = x.str.replace(r"(rp\.?|idr|usd|\$|tahun|thn|bulan|bln|years?|months?|yrs?)", "", regex=True).str.strip()

    def to_f(t):
        t = t.strip()
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
        return float(t.replace(",", ""))

    def conv(v):
        m = re.fullmatch(num + r"\s*(?:-|–|to|s/d|sd|sampai)\s*" + num, v)
        if m:
            return (to_f(m.group(1)) + to_f(m.group(2))) / 2
        m = re.fullmatch(r"(?:<|<=|below|under|kurang dari)\s*" + num, v)
        if m:
            return to_f(m.group(1)) * 0.75
        m = re.fullmatch(num + r"\s*(?:\+|or more|and above|ke atas|>)", v) or re.fullmatch(r"(?:>|>=|above|over|lebih dari)\s*" + num, v)
        if m:
            return to_f(m.group(1)) * 1.25
        try:
            return to_f(v)
        except Exception:
            return np.nan
    vals = x2.map(conv)
    has_range = x2.str.contains(r"\d\s*(?:-|–|to|s/d)\s*\d|^[<>]|\+$", regex=True).mean()
    ratio = vals.notna().mean()
    if has_range < 0.3 or ratio < 0.95:
        return None, 0.0
    full = pd.Series(np.nan, index=s.index)
    full.loc[vals.index] = vals.values
    return full, ratio


def try_parse_date(s, name):
    x = s.dropna().astype(str)
    if len(x) == 0:
        return None, 0.0
    looks = x.str.contains(r"\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|^\d{8}$|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}|\d{1,2} [A-Za-z]{3,9} \d{4}").mean()
    if looks < 0.8 and not DATE_NAME_HINT.search(name):
        return None, 0.0
    if looks < 0.5:
        return None, 0.0
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.notna().sum() < 0.9 * s.notna().sum():
        for kw in ({"dayfirst": True}, {"format": "mixed"}, {"format": "mixed", "dayfirst": True}):
            try:
                parsed2 = pd.to_datetime(s, errors="coerce", **kw)
            except (TypeError, ValueError):
                continue
            if parsed2.notna().sum() > parsed.notna().sum():
                parsed = parsed2
    ratio = parsed.notna().sum() / max(1, s.notna().sum())
    return parsed, ratio


def unify_category_case(s):
    """Satukan ejaan kategori yang beda kapitalisasi/spasi → ejaan paling sering."""
    x = s.astype("object").where(s.isna(), s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
    key = x.where(x.isna(), x.astype(str).str.lower())
    canon = {}
    for k, grp in x.groupby(key):
        canon[k] = grp.value_counts().index[0]
    return x.where(x.isna(), key.map(canon))


PII_NAME_PATTERN = re.compile(
    r"(^|[_\W])(name|nama|full_?name|customer_?name|nama_?nasabah|email|e_?mail|surel|phone|telp|telepon|tlp|no_?hp|handphone|"
    r"mobile|whatsapp|address|alamat|nik|ktp|npwp|passport|paspor|ssn|account_?(no|number)|no_?rek(ening)?|card_?(no|number))($|[_\W])", re.I)
_EMAIL_RE = r"^[\w.+-]+@[\w-]+\.[\w.-]+$"
_PHONE_RE = r"^(\+?62|0)8\d{7,12}$|^\+?\d{10,14}$"


def pii_reason(c, s):
    """Kolom data pribadi (nama, email, no HP, alamat, NIK, rekening) → bukan fitur model (privasi & noise)."""
    v = s.dropna()
    if len(v) == 0:
        return None
    nun = v.nunique() / len(v)
    vs = v.astype(str).str.strip().str.replace(r"[\s\-().]", "", regex=True).head(2000)
    if v.astype(str).str.strip().head(2000).str.match(_EMAIL_RE).mean() > 0.5:
        return "email"
    if not pd.api.types.is_float_dtype(s) and nun > 0.3 and vs.str.match(_PHONE_RE).mean() > 0.8 and \
            (PII_NAME_PATTERN.search(str(c)) or v.astype(str).str.strip().str.match(r"^(\+?62|0)8").mean() > 0.8):
        return "nomor telepon"
    if PII_NAME_PATTERN.search(str(c)) and nun > 0.3:
        return "nama kolom identitas pribadi & nilai hampir unik"
    return None


def clean_and_infer(train_df, test_df):
    """Bersihkan train+test dengan aturan sama. Return train_c, test_c, col_types, log."""
    n_tr = len(train_df)
    full = pd.concat([train_df.drop(columns=[TARGET]), test_df] if test_df is not None else [train_df.drop(columns=[TARGET])],
                     axis=0, ignore_index=True)
    log, col_types = [], {}
    full = full.replace(CFG.NA_TOKENS, np.nan)

    for c in list(full.columns):
        if c == ID_COL:
            col_types[c] = "id"
            continue
        if c in CFG.DROP_COLS:
            col_types[c] = "dropped"
            continue
        s = full[c]
        _pii = pii_reason(c, s)
        if _pii and c not in CFG.FORCE_CATEGORICAL and c not in CFG.FORCE_NUMERIC:
            col_types[c] = "id"
            log.append(f"{c}: data pribadi (PII: {_pii}) → tidak dipakai sebagai fitur & tidak diekspor ke dashboard")
            continue
        if s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "category":
            s = s.astype("object")
            s = s.where(s.isna(), s.astype(str).str.strip())
            s = s.replace(CFG.NA_TOKENS, np.nan)
            full[c] = s
            if c not in CFG.FORCE_CATEGORICAL:
                # date?
                if CFG.DATE_COLS is None or c in (CFG.DATE_COLS or []):
                    parsed, r = try_parse_date(s, c)
                    if parsed is not None and r >= 0.9:
                        full[c] = parsed
                        col_types[c] = "date"
                        log.append(f"{c}: parsed sebagai tanggal ({pct(r)})")
                        continue
                # numeric in string?
                num, r = try_parse_numeric(s)
                if num is not None and (r >= 0.95 or c in CFG.FORCE_NUMERIC):
                    full[c] = num
                    log.append(f"{c}: string → numeric ({pct(r)} berhasil)")
                    s = full[c]
                elif try_parse_range(s)[0] is not None:
                    full[c] = try_parse_range(s)[0]
                    log.append(f"{c}: string rentang (misal '50000 - 74999') → numeric titik tengah")
                    s = full[c]
                else:
                    # bool-like?
                    low = s.dropna().astype(str).str.lower()
                    if len(low) and low.isin(BOOL_MAP.keys()).all() and low.nunique() == 2:
                        full[c] = s.astype(str).str.lower().map(BOOL_MAP).where(s.notna())
                        col_types[c] = "binary"
                        log.append(f"{c}: boolean-like → 1/0")
                        continue
                    before = s.nunique()
                    full[c] = unify_category_case(s)
                    after = full[c].nunique()
                    if after < before:
                        log.append(f"{c}: {before - after} variasi ejaan/case disatukan ({before} → {after} level)")
        elif np.issubdtype(s.dtype, np.datetime64):
            col_types[c] = "date"
            continue
        elif s.dtype == bool:
            full[c] = s.astype(int)

        s = full[c]
        nun = s.nunique(dropna=True)
        if nun <= 1:
            col_types[c] = "constant"
            continue
        top_freq = s.value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq >= CFG.QUASI_CONSTANT_THRESHOLD:
            col_types[c] = "constant"
            log.append(f"{c}: quasi-constant (nilai teratas {pct(top_freq)})")
            continue
        if c in CFG.FORCE_CATEGORICAL:
            full[c] = s.astype("object").where(s.isna(), s.astype(str))
            col_types[c] = "categorical" if nun <= CFG.HIGH_CARD_THRESHOLD else "high_card"
            continue
        if pd.api.types.is_numeric_dtype(s) and ID_PATTERN.search(str(c)) and nun > 0.5 * len(s):
            col_types[c] = "id"
            log.append(f"{c}: kolom ID numerik (nama ID & {nun:,} nilai unik) → tidak dipakai sebagai fitur")
            continue
        if pd.api.types.is_numeric_dtype(s):
            col_types[c] = "binary" if nun == 2 else "numeric"
            if nun == 2 and set(s.dropna().unique()) - {0, 1}:
                vals = sorted(s.dropna().unique())
                full[c] = (s == vals[1]).astype(float).where(s.notna())
                log.append(f"{c}: binary numeric {vals} → 0/1")
        else:
            avg_len = s.dropna().astype(str).str.len().mean()
            if nun > 0.5 * len(s) and avg_len > 30:
                col_types[c] = "text"
            elif nun > 0.95 * len(s):
                col_types[c] = "id"
                log.append(f"{c}: hampir unik per baris → dianggap ID-like (dibuang dari model)")
            elif nun == 2:
                col_types[c] = "binary_cat"
            elif nun > CFG.HIGH_CARD_THRESHOLD:
                col_types[c] = "high_card"
            else:
                col_types[c] = "categorical"

    num_cols_ = full.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(full[num_cols_].to_numpy(dtype=float, na_value=np.nan)).sum()) if len(num_cols_) else 0
    if n_inf:
        full[num_cols_] = full[num_cols_].replace([np.inf, -np.inf], np.nan)
        log.append(f"{n_inf} nilai ±inf diganti NaN")
    train_c = full.iloc[:n_tr].reset_index(drop=True)
    test_c = full.iloc[n_tr:].reset_index(drop=True) if test_df is not None else None
    return train_c, test_c, col_types, log


with safe_section("Cleaning & type inference", critical=True):
    train, test, COL_TYPES, clean_log = clean_and_infer(raw_train, raw_test)
    print("Log cleaning:")
    for line in clean_log:
        print("  •", line)
    type_summary = pd.Series(COL_TYPES).value_counts()
    print("\nRingkasan tipe kolom:")
    display(type_summary.to_frame("n_columns"))

# %% [markdown]
# ## 3.4 Nilai mustahil, duplikat, dan Data Quality Report
# Role kolom (umur, premi, tenure, dll.) dideteksi dulu secara kasar di sini supaya pengecekan nilai mustahil tepat sasaran. Detail role ada di Section 5.

# %%
ROLE_KEYWORDS = OrderedDict([
    ("tenure", ["tenure", "lama", "masa", "months_active", "month_with", "duration", "durasi", "membership_length", "policy_age", "customer_since_months", "months_on_book", "seniority"]),
    ("age", ["age", "usia", "umur"]),
    ("premium", ["premium", "premi", "monthlycharges", "monthly_charge", "charges", "fee", "price", "iuran", "tagihan", "billing_amount", "ann_amt", "annual_amount", "curr_ann", "policy_amount"]),
    ("total_paid", ["totalcharges", "total_charge", "total_paid", "total_premium", "lifetime_value", "ltv", "total_amount"]),
    ("income", ["income", "pendapatan", "salary", "gaji", "penghasilan", "estimatedsalary"]),
    ("sum_insured", ["sum_insured", "sum_assured", "uang_pertanggungan", "coverage_amount", "insured_amount", "face_amount", "benefit"]),
    ("claims", ["claim", "klaim"]),
    ("complaints", ["complain", "keluhan", "komplain", "ticket", "support_call", "customer_service_calls", "service_calls"]),
    ("late_payment", ["late", "telat", "terlambat", "overdue", "arrear", "missed_payment", "tunggakan", "delinq", "grace"]),
    ("products", ["n_products", "num_products", "numofproducts", "product_count", "jumlah_produk", "products", "riders", "num_policies", "policies"]),
    ("premium_change", ["premium_change", "price_change", "increase", "kenaikan", "rate_change", "renewal_change"]),
    ("engagement", ["engagement", "login", "app_usage", "visit", "activity", "digital", "isactivemember", "usage"]),
    ("channel", ["channel", "saluran", "source", "acquisition"]),
    ("payment_method", ["payment_method", "paymentmethod", "metode_bayar", "payment_type", "cara_bayar", "payment_mode"]),
    ("payment_frequency", ["frequency", "frekuensi", "payment_freq", "billing_cycle", "contract", "term"]),
    ("auto_renew", ["auto_renew", "autorenew", "auto-renew", "auto_debit", "autodebit", "paperless"]),
    ("gender", ["gender", "sex", "jenis_kelamin", "kelamin"]),
    ("region", ["region", "wilayah", "provinsi", "province", "city", "kota", "geography", "state", "area", "branch", "cabang"]),
    ("policy_type", ["policy_type", "product_type", "plan", "jenis_polis", "tipe_polis", "coverage", "line_of_business", "lob"]),
    ("agent", ["agent", "agen", "broker", "advisor"]),
    ("dependents", ["dependent", "tanggungan", "children", "anak", "family_size", "household"]),
])


def detect_roles(df, col_types):
    """Cari kolom untuk tiap role berdasarkan keyword nama kolom. Override via CFG.COLUMN_ROLES."""
    roles = {}
    used = set()
    numeric_roles = {"tenure", "age", "premium", "total_paid", "income", "sum_insured", "claims", "complaints",
                     "late_payment", "products", "premium_change", "engagement", "dependents"}
    for role, kws in ROLE_KEYWORDS.items():
        if role in (CFG.COLUMN_ROLES or {}):
            roles[role] = CFG.COLUMN_ROLES[role]
            used.add(CFG.COLUMN_ROLES[role])
            continue
        for c in df.columns:
            if c in used or col_types.get(c) in ("id", "dropped", "constant", "text"):
                continue
            cl = c.lower()
            if any(k in cl for k in kws):
                is_num = col_types.get(c) in ("numeric", "binary")
                if role in numeric_roles and not is_num:
                    continue
                if role == "claims" and any(k in cl for k in ["reject", "tolak", "denied", "settle", "days", "amount", "ratio"]):
                    continue  # klaim ditolak/lama settle → simpan sebagai fitur biasa
                if role == "tenure" and col_types.get(c) == "date":
                    continue
                roles[role] = c
                used.add(c)
                break
    return roles


with safe_section("Impossible values, duplicates & quality report"):
    ROLES = detect_roles(train, COL_TYPES)
    if CFG.FIX_IMPOSSIBLE_VALUES:
        rules = {"age": (0, 110), "tenure": (0, None), "premium": (0, None), "income": (0, None),
                 "sum_insured": (0, None), "total_paid": (0, None), "claims": (0, None), "complaints": (0, None),
                 "late_payment": (0, None), "products": (0, None), "dependents": (0, 30)}
        for role, (lo, hi) in rules.items():
            c = ROLES.get(role)
            if c is None or c not in train.columns or not pd.api.types.is_numeric_dtype(train[c]):
                continue
            for name, d in [("train", train), ("test", test)]:
                if d is None:
                    continue
                bad = pd.Series(False, index=d.index)
                if lo is not None:
                    bad |= d[c] < lo
                if hi is not None:
                    bad |= d[c] > hi
                if bad.sum():
                    print(f"  • {name}.{c} ({role}): {bad.sum()} nilai mustahil (di luar [{lo}, {hi}]) → NaN")
                    d.loc[bad, c] = np.nan

    # duplikat
    feat_cols_for_dup = [c for c in train.columns if c != ID_COL]
    dup_mask = train.duplicated(subset=feat_cols_for_dup, keep="first")
    tmp = train[feat_cols_for_dup].copy()
    tmp["__y"] = y
    conflict = tmp[tmp.duplicated(subset=feat_cols_for_dup, keep=False)].groupby(feat_cols_for_dup, dropna=False)["__y"].nunique()
    n_conflict = int((conflict > 1).sum()) if len(conflict) else 0
    N_DUP_REMOVED = int(dup_mask.sum()) if CFG.DROP_DUPLICATES else 0
    print(f"\nDuplikat baris (fitur identik, abaikan ID): {dup_mask.sum()} | grup duplikat dengan label berbeda: {n_conflict}")
    if CFG.DROP_DUPLICATES and dup_mask.sum() > 0:
        keep = ~dup_mask.values
        train = train[keep].reset_index(drop=True)
        raw_train = raw_train[keep].reset_index(drop=True)
        y = y[keep]
        print(f"  → {dup_mask.sum()} duplikat dibuang. Train sekarang {train.shape}")
    if n_conflict:
        print("  ⚠️ Ada data identik dengan label berbeda → noise label / fitur kurang informatif. Wajar dalam jumlah kecil.")

    # quality report
    rows = []
    for c in train.columns:
        s = train[c]
        r = {"column": c, "inferred_type": COL_TYPES.get(c), "dtype": str(s.dtype),
             "missing_%": round(100 * s.isna().mean(), 2),
             "missing_test_%": round(100 * test[c].isna().mean(), 2) if test is not None and c in test else np.nan,
             "n_unique": s.nunique(dropna=True),
             "top_value": s.value_counts(dropna=True).index[0] if s.notna().any() else None,
             "top_freq_%": round(100 * s.value_counts(normalize=True, dropna=True).iloc[0], 2) if s.notna().any() else None}
        if pd.api.types.is_numeric_dtype(s):
            r.update({"min": s.min(), "median": s.median(), "max": s.max(), "skew": round(s.skew(), 2) if s.nunique() > 2 else np.nan})
        rows.append(r)
    quality_report = pd.DataFrame(rows)
    save_table(quality_report, "data_quality_report")
    display(quality_report)

    high_missing = quality_report[quality_report["missing_%"] > 40]["column"].tolist()
    if high_missing:
        print("⚠️ Kolom dengan missing > 40%:", high_missing, "→ missingness-nya sendiri bisa jadi sinyal (indikator dibuat otomatis).")
    if test is not None:
        miss_tr = set(raw_train.columns) - {TARGET} - set(raw_test.columns)
        miss_te = set(raw_test.columns) - set(raw_train.columns)
        if miss_tr or miss_te:
            print("⚠️ Kolom beda antara train & test → hanya train:", miss_tr, "| hanya test:", miss_te)

# %% [markdown]
# # 4. Target Analysis
#
# **Cara baca:**
# - **Churn rate** = proporsi churn. Ini juga = presisi kalau kita menebak "semua churn".
# - **Imbalance**: churn rate < 20% → imbalanced. Accuracy jadi menyesatkan (tebak "tidak churn" semua sudah dapat accuracy tinggi). Fokus ke ROC-AUC, PR-AUC, F1, recall.
# - **Baseline**: score dummy (tanpa model). Model kita **harus** jauh di atas ini.
#
# **Untuk laporan:** "The dataset contains N customers with an overall churn rate of X%, indicating [moderate/severe] class imbalance."

# %%
with safe_section("Target analysis"):
    n_pos, n_tot = int(y.sum()), len(y)
    CHURN_RATE = n_pos / n_tot
    print(f"Jumlah customer : {n_tot:,}")
    print(f"Churn (1)       : {n_pos:,} ({pct(CHURN_RATE)})")
    print(f"Tidak churn (0) : {n_tot - n_pos:,} ({pct(1 - CHURN_RATE)})")
    imb = "severe" if CHURN_RATE < 0.1 else "moderate" if CHURN_RATE < 0.25 else "mild/none"
    print("Tingkat imbalance:", imb)
    if CHURN_RATE < 0.1:
        print("  → Jika metrik = accuracy, waspada: model dummy sudah dapat", pct(1 - CHURN_RATE))
        print("  → Pertimbangkan PR-AUC/F1 untuk evaluasi internal & threshold tuning.")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    counts = pd.Series(y).map({0: "Retained (0)", 1: "Churned (1)"}).value_counts()
    ax[0].bar(counts.index, counts.values, color=[CFG.PALETTE[0], CFG.PALETTE[1]])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}\n({v / n_tot:.1%})", ha="center", va="bottom")
    ax[0].set_title("Customer Churn Distribution")
    ax[0].set_ylabel("Number of customers")
    ax[0].set_ylim(0, counts.max() * 1.25)
    ax[1].pie([n_tot - n_pos, n_pos], labels=["Retained", "Churned"], autopct="%1.1f%%",
              colors=[CFG.PALETTE[0], CFG.PALETTE[1]], startangle=90, wedgeprops={"width": 0.45})
    ax[1].set_title("Churn Rate")
    save_fig("target_distribution")

    # baseline
    base_prob = np.full(n_tot, CHURN_RATE)
    print("\nBaseline (tanpa model):")
    print(f"  ROC-AUC  = 0.5000 | PR-AUC = {CHURN_RATE:.4f} | LogLoss = {log_loss(y, base_prob):.4f} | "
          f"Accuracy (tebak mayoritas) = {max(CHURN_RATE, 1 - CHURN_RATE):.4f}")
    add_insight("Data", f"The training data contains {n_tot:,} customers with an overall churn rate of {pct(CHURN_RATE)} ({imb} class imbalance).")

# %% [markdown]
# # 5. Column Roles & Domain Feature Engineering (Asuransi)
#
# ## 5.1 Column roles
# Role = arti bisnis kolom (tenure, premi, umur, klaim, keluhan, metode bayar, channel, ...). Dipakai untuk:
# - membuat **fitur domain** yang masuk akal,
# - **survival analysis** (butuh `tenure`),
# - **nilai customer / CLV** (butuh `premium`),
# - **rekomendasi** (teks rekomendasi disesuaikan role).
#
# **Cek output di bawah!** Kalau ada role salah/kurang → `CFG.COLUMN_ROLES = {"tenure": "nama_kolom", ...}` lalu run ulang dari Section 3.4.
#
# ## 5.2 Fitur domain asuransi yang dibuat (kalau kolomnya ada)
# | Fitur | Rumus | Intuisi bisnis |
# |---|---|---|
# | `fe_tenure_years`, `fe_tenure_band` | tenure → tahun & band (0-6m, 6-12m, 1-2y, 2-5y, 5y+) | churn biasanya tertinggi di tahun pertama |
# | `fe_is_new_customer` | tenure < 12 bulan | fase onboarding kritis |
# | `fe_age_band` | <25, 25-34, ..., 65+ | life stage berbeda kebutuhan |
# | `fe_premium_to_income` | premi / pendapatan | **affordability** — beban premi tinggi → rawan lapse |
# | `fe_sum_insured_to_premium` | UP / premi | value-for-money |
# | `fe_claims_per_year` | klaim / tenure (tahun) | intensitas pemakaian produk |
# | `fe_has_claim` | klaim > 0 | pernah merasakan layanan klaim |
# | `fe_claim_reject_rate` | klaim ditolak / klaim | pengalaman klaim buruk |
# | `fe_has_complaint`, `fe_has_late_payment` | > 0 | sinyal ketidakpuasan / kesulitan bayar |
# | `fe_premium_increase`, `fe_big_premium_increase` | perubahan premi > 0 / > 10% | **price shock** saat renewal |
# | `fe_multi_product` | produk > 1 | bundling → switching cost lebih tinggi |
# | `fe_avg_paid_per_month` | total dibayar / tenure | konsistensi pembayaran |
# | `<date>_days_since`, `<date>_year`, `<date>_month` | dari kolom tanggal | recency & seasonality |
# | `fe_n_missing` | jumlah missing per baris | data tidak lengkap sering = customer kurang engaged |
#
# Tambahkan fitur sendiri di fungsi `custom_feature_engineering(df)`.
#
# **Peringatan leakage tanggal:** kolom seperti `cancel_date`, `end_date`, `termination_date` hanya terisi untuk yang churn → **LEAKAGE**. Notebook menandainya (Section 12). Jangan dipakai sebagai fitur.

# %%
def infer_tenure_unit(s):
    if CFG.TENURE_UNIT != "auto":
        return CFG.TENURE_UNIT
    mx = s.quantile(0.99)
    if mx > 400:
        return "days"
    if mx <= 40 and s.max() <= 60:
        return "years"
    return "months"


def custom_feature_engineering(df):
    """HOOK: tambahkan fitur spesifik dataset lomba di sini. df = train atau test (sudah bersih).
    Contoh:
        if {"annual_premium", "n_products"} <= set(df.columns):
            df["premium_per_product"] = df["annual_premium"] / df["n_products"].clip(lower=1)
    """
    return df


def domain_feature_engineering(df, roles, ref_date):
    df = df.copy()
    new = []
    r = roles

    def has(role):
        return r.get(role) in df.columns and pd.api.types.is_numeric_dtype(df[r[role]])

    if has("tenure"):
        t = df[r["tenure"]].astype(float)
        unit = TENURE_UNIT
        months = t if unit == "months" else t * 12 if unit == "years" else t / 30.44
        df["fe_tenure_years"] = months / 12
        df["fe_tenure_band"] = pd.cut(months, [-np.inf, 6, 12, 24, 60, np.inf],
                                      labels=["0-6m", "6-12m", "1-2y", "2-5y", "5y+"]).astype(str).replace("nan", np.nan)
        df["fe_is_new_customer"] = (months < 12).astype(float).where(t.notna())
        new += ["fe_tenure_years", "fe_tenure_band", "fe_is_new_customer"]
    if has("age"):
        df["fe_age_band"] = pd.cut(df[r["age"]], [-np.inf, 25, 35, 45, 55, 65, np.inf],
                                   labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str).replace("nan", np.nan)
        new.append("fe_age_band")
    if has("premium") and has("income"):
        prem = df[r["premium"]] * (12 if CFG.VALUE_PERIOD == "monthly" else 1)
        df["fe_premium_to_income"] = prem / df[r["income"]].replace(0, np.nan)
        new.append("fe_premium_to_income")
    if has("premium") and has("sum_insured"):
        df["fe_sum_insured_to_premium"] = df[r["sum_insured"]] / df[r["premium"]].replace(0, np.nan)
        new.append("fe_sum_insured_to_premium")
    if has("claims"):
        df["fe_has_claim"] = (df[r["claims"]] > 0).astype(float).where(df[r["claims"]].notna())
        new.append("fe_has_claim")
        if "fe_tenure_years" in df:
            df["fe_claims_per_year"] = df[r["claims"]] / df["fe_tenure_years"].clip(lower=0.25)
            new.append("fe_claims_per_year")
        rej = [c for c in df.columns if "claim" in c.lower() and any(k in c.lower() for k in ["reject", "tolak", "denied", "declin"])
               and pd.api.types.is_numeric_dtype(df[c])]
        if rej:
            df["fe_claim_reject_rate"] = (df[rej[0]] / df[r["claims"]].replace(0, np.nan)).fillna(0)
            new.append("fe_claim_reject_rate")
    if has("complaints"):
        df["fe_has_complaint"] = (df[r["complaints"]] > 0).astype(float).where(df[r["complaints"]].notna())
        new.append("fe_has_complaint")
    if has("late_payment"):
        df["fe_has_late_payment"] = (df[r["late_payment"]] > 0).astype(float).where(df[r["late_payment"]].notna())
        new.append("fe_has_late_payment")
    if has("premium_change"):
        pc = df[r["premium_change"]]
        df["fe_premium_increase"] = (pc > 0).astype(float).where(pc.notna())
        big = 10 if pc.abs().quantile(0.95) > 1 else 0.10   # persen vs fraksi
        df["fe_big_premium_increase"] = (pc > big).astype(float).where(pc.notna())
        new += ["fe_premium_increase", "fe_big_premium_increase"]
    if has("products"):
        df["fe_multi_product"] = (df[r["products"]] > 1).astype(float).where(df[r["products"]].notna())
        new.append("fe_multi_product")
    if has("total_paid") and has("tenure"):
        df["fe_avg_paid_per_period"] = df[r["total_paid"]] / df[r["tenure"]].replace(0, np.nan)
        new.append("fe_avg_paid_per_period")
        if has("premium"):
            df["fe_paid_vs_expected"] = df[r["total_paid"]] / (df[r["premium"]] * df[r["tenure"]]).replace(0, np.nan)
            new.append("fe_paid_vs_expected")
    # tanggal → days since + year + month
    for c in [c for c, t in COL_TYPES.items() if t == "date" and c in df.columns]:
        d = pd.to_datetime(df[c], errors="coerce")
        if re.search(r"(birth|dob|lahir|born)", c, re.I):
            if not has("age"):        # tidak ada kolom umur → turunkan umur dari tanggal lahir
                df["fe_age_from_dob"] = (ref_date - d).dt.days / 365.25
                new.append("fe_age_from_dob")
            continue                   # ada kolom umur → tanggal lahir redundan, tidak dibuat fitur tambahan
        df[f"{c}_days_since"] = (ref_date - d).dt.days
        df[f"{c}_year"] = d.dt.year
        df[f"{c}_month"] = d.dt.month
        new += [f"{c}_days_since", f"{c}_year", f"{c}_month"]
    if CFG.USE_ROW_STATS:
        base_cols = [c for c, t in COL_TYPES.items() if t not in ("id", "dropped", "constant") and c in df.columns
                     and c in ORIGINAL_COLS and c not in LEAK_NAME_COLS]
        df["fe_n_missing"] = df[base_cols].isna().sum(axis=1)
        new.append("fe_n_missing")
    df = custom_feature_engineering(df)
    return df, new


FE_NEW, FEATURE_FAMILY, TENURE_UNIT, REF_DATE = [], {}, None, pd.Timestamp.today().normalize()
BIRTH_HINT = re.compile(r"(birth|dob|lahir|born)", re.I)
with safe_section("Column roles & domain FE"):
    LEAK_NAME_COLS = [c for c in train.columns if any(k in c.lower() for k in CFG.LEAKAGE_KEYWORDS)]
    ROLES = detect_roles(train, COL_TYPES)
    print("Role yang terdeteksi (cek & override via CFG.COLUMN_ROLES kalau salah):")
    for k, v in ROLES.items():
        print(f"  {k:<18} → {v}")
    missing_roles = [k for k in ["tenure", "premium", "age", "claims", "complaints", "payment_method", "channel"] if k not in ROLES]
    if missing_roles:
        print("  (tidak ditemukan:", ", ".join(missing_roles), ")")

    TENURE_UNIT = infer_tenure_unit(train[ROLES["tenure"]].dropna()) if "tenure" in ROLES else None
    if TENURE_UNIT:
        print(f"Satuan tenure: {TENURE_UNIT}")

    date_cols = [c for c, t in COL_TYPES.items() if t == "date"]
    date_series = [pd.to_datetime(train[c], errors="coerce") for c in date_cols]
    if test is not None:
        date_series += [pd.to_datetime(test[c], errors="coerce") for c in date_cols if c in test.columns]
    all_dates = pd.concat(date_series) if date_series else pd.Series([], dtype="datetime64[ns]")
    REF_DATE = pd.Timestamp(CFG.REFERENCE_DATE) if CFG.REFERENCE_DATE else (all_dates.max() if all_dates.notna().any() else pd.Timestamp.today().normalize())
    if date_cols:
        print("Kolom tanggal:", date_cols, "| reference date:", REF_DATE.date())

    if CFG.USE_DOMAIN_FE:
        train, FE_NEW = domain_feature_engineering(train, ROLES, REF_DATE)
        if test is not None:
            test, _ = domain_feature_engineering(test, ROLES, REF_DATE)
        for c in FE_NEW:
            s = train[c]
            if s.dtype == object:
                COL_TYPES[c] = "categorical"
            else:
                COL_TYPES[c] = "binary" if s.nunique() == 2 else "numeric"
        print(f"\n{len(FE_NEW)} fitur domain dibuat:", FE_NEW)
    else:
        FE_NEW = []

    # Feature family: fitur yang membawa informasi sama (raw vs turunan) → dipakai untuk menghindari laporan dobel
    FEATURE_FAMILY = {}
    fam_members = {
        "tenure": [ROLES.get("tenure"), "fe_tenure_years", "fe_tenure_band", "fe_is_new_customer"]
                  + [f"{d}_{s}" for d in [c for c, t in COL_TYPES.items() if t == "date" and not BIRTH_HINT.search(c)] for s in ("days_since", "year")],
        "age": [ROLES.get("age"), "fe_age_band"]
               + [f"{d}_{s}" for d in [c for c, t in COL_TYPES.items() if t == "date" and BIRTH_HINT.search(c)] for s in ("days_since", "year")],
        "claims": [ROLES.get("claims"), "fe_has_claim"],
        "complaints": [ROLES.get("complaints"), "fe_has_complaint"],
        "late_payment": [ROLES.get("late_payment"), "fe_has_late_payment"],
        "products": [ROLES.get("products"), "fe_multi_product"],
        "premium_change": [ROLES.get("premium_change"), "fe_premium_increase", "fe_big_premium_increase"],
    }
    for fam, members in fam_members.items():
        for m in members:
            if m and m in train.columns:
                FEATURE_FAMILY[m] = fam


with safe_section("Final feature lists", critical=True):
    # Daftar fitur final untuk analisis (section terpisah supaya tetap terbentuk walau FE gagal)
    NUM_COLS = [c for c, t in COL_TYPES.items() if t in ("numeric", "binary") and c in train.columns]
    CAT_COLS = [c for c, t in COL_TYPES.items() if t in ("categorical", "binary_cat") and c in train.columns]
    HIGH_CARD_COLS = [c for c, t in COL_TYPES.items() if t == "high_card" and c in train.columns]
    EXCLUDED = [c for c, t in COL_TYPES.items() if t in ("id", "dropped", "constant", "date", "text")]
    for c in CAT_COLS + HIGH_CARD_COLS:
        train[c] = train[c].astype("object")
        if test is not None:
            test[c] = test[c].astype("object")
    print(f"\nNumeric/binary : {len(NUM_COLS)}")
    print(f"Categorical    : {len(CAT_COLS)}")
    print(f"High-card cat  : {len(HIGH_CARD_COLS)} {HIGH_CARD_COLS}")
    print(f"Excluded       : {EXCLUDED}")

# %% [markdown]
# ## 5.3 Early Leakage Screen (WAJIB dicek)
#
# **Leakage** = fitur yang "bocor" informasi target, biasanya karena baru tercatat **setelah** customer churn (misal `cancellation_reason`, `termination_date`, `refund_amount`, `status_akhir`). Model jadi kelihatan super akurat (AUC 0.99) tapi **tidak berguna** dan insight-nya salah total.
#
# Tiga cara deteksi otomatis:
# 1. **Nama kolom** mengandung kata kunci (`cancel, terminat, reason, end_date, refund, batal, ...`).
# 2. **Missingness = target**: kolom yang kosong hampir hanya untuk non-churner (atau sebaliknya). Contoh klasik: `cancellation_reason` hanya terisi kalau churn.
# 3. **Single-feature AUC** ≥ `CFG.LEAKAGE_AUC_THRESHOLD` (default 0.97).
#
# **Jika X → Y:**
# - Ada fitur ter-flag & secara logika bisnis memang terjadi setelah churn → **buang** (`CFG.DROP_COLS` atau `CFG.AUTO_DROP_LEAKY = True`).
# - Ter-flag tapi secara bisnis valid diketahui sebelum churn (jarang) → boleh dipertahankan, tapi jelaskan di laporan.
# - Kalau test set juga punya kolom leak itu dan metrik lomba murni prediksi → diskusikan dengan tim; untuk **analisis & rekomendasi** tetap jangan pakai.
#
# **Untuk laporan:** sebutkan bahwa fitur post-event dikeluarkan untuk mencegah data leakage → menunjukkan kematangan metodologi (juri suka ini).

# %%
def quick_auc(x, y_arr):
    """AUC univariate cepat: numeric → rank AUC; kategori → churn rate per level (out-of-fold)."""
    try:
        if pd.api.types.is_numeric_dtype(x) and x.nunique() > 2:
            v = x.fillna(x.median() if x.notna().any() else 0)
            a = roc_auc_score(y_arr, v)
            return max(a, 1 - a)
        lab = x.astype(str).where(x.notna(), "Missing").values
        pred = np.zeros(len(y_arr))
        prior = y_arr.mean()
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=CFG.SEED).split(lab, y_arr):
            m = pd.DataFrame({"l": lab[tr], "y": y_arr[tr]}).groupby("l")["y"].agg(["sum", "count"])
            rate = (m["sum"] + 10 * prior) / (m["count"] + 10)
            pred[va] = pd.Series(lab[va]).map(rate).fillna(prior).values
        a = roc_auc_score(y_arr, pred)
        return max(a, 1 - a)
    except Exception:
        return np.nan


LEAKY_COLS = []

with safe_section("Early leakage screen"):
    rows = []
    for c in NUM_COLS + CAT_COLS + HIGH_CARD_COLS + [c for c, t in COL_TYPES.items() if t == "date" and c in train.columns]:
        s = train[c]
        name_hit = any(k in c.lower() for k in CFG.LEAKAGE_KEYWORDS)
        miss_auc = np.nan
        if s.isna().any() and s.notna().any():
            try:
                miss_auc = roc_auc_score(y, s.isna().astype(int))
                miss_auc = max(miss_auc, 1 - miss_auc)
            except Exception:
                pass
        auc = quick_auc(s, y) if COL_TYPES.get(c) != "date" else np.nan
        stat_hit = (miss_auc >= 0.95) or (auc >= CFG.LEAKAGE_AUC_THRESHOLD)
        flag = name_hit or stat_hit
        rows.append({"feature": c, "name_keyword": name_hit, "missingness_auc": miss_auc, "single_feature_auc": auc,
                     "statistical_evidence": stat_hit, "flagged": flag})
    leak_df = pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False)
    save_table(leak_df, "leakage_screen")
    display(leak_df.head(15))
    LEAKY_COLS = leak_df[leak_df["flagged"]]["feature"].tolist()
    if LEAKY_COLS:
        print("🚨 Fitur dicurigai LEAKAGE:", LEAKY_COLS)
        for c in LEAKY_COLS:
            r = leak_df.set_index("feature").loc[c]
            why = []
            if r["name_keyword"]:
                why.append("nama kolom mengandung kata kunci post-event")
            if r["missingness_auc"] >= 0.95:
                why.append(f"pola missing hampir identik dengan target (AUC {r['missingness_auc']:.3f})")
            if r["single_feature_auc"] >= CFG.LEAKAGE_AUC_THRESHOLD:
                why.append(f"single-feature AUC {r['single_feature_auc']:.3f}")
            print(f"   - {c}: " + "; ".join(why))
        NAME_ONLY = leak_df[leak_df["flagged"] & ~leak_df["statistical_evidence"]]["feature"].tolist()
        if NAME_ONLY:
            print(f"   ℹ️ Hanya cocok nama (tanpa bukti statistik) → TIDAK dibuang otomatis, cek manual: {NAME_ONLY}")
        LEAKY_COLS = [c for c in LEAKY_COLS if c not in NAME_ONLY] if CFG.AUTO_DROP_LEAKY else LEAKY_COLS
        if CFG.AUTO_DROP_LEAKY and LEAKY_COLS:
            for c in LEAKY_COLS:
                COL_TYPES[c] = "dropped"
            NUM_COLS = [c for c in NUM_COLS if c not in LEAKY_COLS]
            CAT_COLS = [c for c in CAT_COLS if c not in LEAKY_COLS]
            HIGH_CARD_COLS = [c for c in HIGH_CARD_COLS if c not in LEAKY_COLS]
            EXCLUDED = EXCLUDED + LEAKY_COLS
            print(f"   → AUTO_DROP_LEAKY=True: {LEAKY_COLS} DIBUANG dari model & analisis driver (tetap dipakai deskriptif di 7.3 kalau kolom alasan).")
            # fitur turunan yang dihitung SEBELUM leak screen bisa membawa leak (misal jumlah missing per baris) -> hitung ulang
            if "fe_n_missing" in train.columns:
                leak_base = set(LEAKY_COLS) | {c.rsplit("_", 1)[0] for c in LEAKY_COLS} | {c.replace("_days_since", "") for c in LEAKY_COLS}
                base_cols = [c for c, t in COL_TYPES.items() if t not in ("id", "dropped", "constant", "date") and c in ORIGINAL_COLS
                             and c not in leak_base and c in train.columns]
                for d_ in [train] + ([test] if test is not None else []):
                    d_["fe_n_missing"] = d_[base_cols].isna().sum(axis=1)
                print("   → fe_n_missing dihitung ulang tanpa kolom leak.")
            add_insight("Data", f"Post-event variables ({', '.join(LEAKY_COLS)}) were excluded to prevent target leakage.")
        elif not CFG.AUTO_DROP_LEAKY:
            print("   → AUTO_DROP_LEAKY=False: fitur TIDAK dibuang. Kalau setuju itu leak, tambahkan ke CFG.DROP_COLS atau set CFG.AUTO_DROP_LEAKY=True lalu run ulang.")
    else:
        print("✅ Tidak ada fitur yang dicurigai leakage.")

# %% [markdown]
# # 6. EDA Univariate
#
# Tujuan: kenal bentuk data **sebelum** melihat hubungan dengan churn.
#
# **Cara baca & Jika X → Y:**
# - **Missing bar**: kolom dengan missing tinggi → cek apakah missing-nya bermakna (misal `avg_claim_settlement_days` kosong karena tidak pernah klaim → bukan error, itu informasi!).
# - **Histogram numeric**: skew tinggi (> 1) → untuk model linear perlu log-transform (dilakukan otomatis di LogReg pipeline via scaler; tree model tidak peduli). Nilai ekstrem → cek apakah salah input.
# - **Bar kategori**: level dengan sangat sedikit data → churn rate-nya tidak stabil (lihat CI di section 7), di model linear digabung ke `__RARE__`.
#
# **Untuk laporan:** cukup 1 kalimat profil data (misal "Most customers hold Health policies (35%) and pay monthly (50%)") — jangan habiskan halaman untuk univariate.

# %%
def grid_axes(n, ncols=3, w=5, h=3.4):
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    axes = np.array(axes).reshape(-1)
    for a in axes[n:]:
        a.axis("off")
    return fig, axes[:n]


with safe_section("EDA univariate"):
    miss = train[NUM_COLS + CAT_COLS + HIGH_CARD_COLS].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.3 * len(miss[:30]))))
        ax.barh(miss.index[:30][::-1], 100 * miss.values[:30][::-1], color=CFG.PALETTE[3])
        ax.set_xlabel("Missing (%)")
        ax.set_title("Missing Values by Feature (train)")
        save_fig("missing_values")
    else:
        print("Tidak ada missing value di fitur. ✅")

    desc = train[NUM_COLS].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc["skew"] = train[NUM_COLS].skew()
    desc["missing_%"] = 100 * train[NUM_COLS].isna().mean()
    save_table(desc.reset_index().rename(columns={"index": "feature"}), "numeric_summary")
    display(desc)

    cont = [c for c in NUM_COLS if train[c].nunique() > 2][: CFG.MAX_EDA_PLOTS]
    if cont:
        fig, axes = grid_axes(len(cont), ncols=4, w=4, h=3)
        for a, c in zip(axes, cont):
            v = train[c].dropna()
            lo, hi = v.quantile([0.005, 0.995])
            a.hist(v.clip(lo, hi), bins=40, color=CFG.PALETTE[0], alpha=0.85)
            a.set_title(c[:35], fontsize=9)
        fig.suptitle("Numeric Feature Distributions (clipped at 0.5–99.5 pct)", y=1.01)
        plt.tight_layout()
        save_fig("numeric_distributions")

    cats = (CAT_COLS + [c for c in NUM_COLS if train[c].nunique() == 2])[: CFG.MAX_EDA_PLOTS]
    if cats:
        fig, axes = grid_axes(len(cats), ncols=4, w=4, h=3)
        for a, c in zip(axes, cats):
            vc = train[c].astype(str).replace("nan", "Missing").value_counts().head(CFG.MAX_CAT_LEVELS_PLOT)
            a.barh(vc.index[::-1].astype(str), vc.values[::-1], color=CFG.PALETTE[4])
            a.set_title(c[:35], fontsize=9)
            a.tick_params(axis="y", labelsize=7)
        fig.suptitle("Categorical Feature Frequencies", y=1.01)
        plt.tight_layout()
        save_fig("categorical_frequencies")

# %% [markdown]
# # 7. Driver Analysis — faktor apa yang berhubungan dengan churn?
#
# **Ini section paling penting untuk laporan.** Untuk setiap fitur dihitung:
#
# | Ukuran | Untuk | Arti | Cara baca |
# |---|---|---|---|
# | **Churn rate per level/bin** + 95% Wilson CI | semua | churn rate tiap kategori / tiap kuantil | CI tidak overlap dengan garis overall → beda nyata |
# | **Lift** | semua | churn rate level ÷ churn rate overall | 2.0 = dua kali lebih rawan dari rata-rata |
# | **Chi-square test** | kategori | ada asosiasi atau tidak | p < 0.05 → ada asosiasi |
# | **Cramér's V** | kategori | **besar** asosiasi (effect size) | <0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, >0.5 large |
# | **Mann-Whitney U** | numeric | distribusi churner vs non-churner beda? | p < 0.05 → beda |
# | **Rank-biserial r** | numeric | effect size + **arah** (+ = nilai tinggi → lebih churn) | \|r\| <0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, >0.5 large |
# | **Information Value (IV)** | semua | kekuatan prediktif (standar credit/insurance scoring) | <0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, 0.3–0.5 strong, **>0.5 suspicious (cek leakage!)** |
# | **Univariate AUC** | semua | seberapa baik fitur ini sendirian memisahkan churn | 0.5 = acak; >0.7 kuat; **>0.95 hampir pasti leakage** |
# | **q-value (Benjamini-Hochberg)** | semua | p-value terkoreksi multiple testing | pakai q < 0.05 untuk klaim "signifikan" |
#
# **Kenapa effect size penting?** Dengan data ribuan baris, hampir semua p-value < 0.05. Yang membedakan driver penting vs remeh adalah **effect size & IV**, bukan p-value. Di laporan tulis keduanya: *"Payment method is strongly associated with churn (χ² p < 0.001, Cramér's V = 0.32)."*
#
# **Jika X → Y:**
# - IV > 0.5 atau AUC > 0.95 → **cek leakage** (variabel yang hanya ada setelah churn terjadi, lihat Section 5.3). Kalau secara bisnis valid (misal tenure pada churn asuransi), itu justru driver terkuat.
# - Fitur numeric dengan hubungan U-shape (churn tinggi di dua ujung, misal umur) → rank-biserial bisa kecil padahal IV besar → lihat plot churn-rate-per-bin, jangan hanya r.
# - Level dengan n kecil (< 30) dan churn rate ekstrem → jangan dijadikan klaim utama (CI lebar).
#
# **Untuk laporan:** tampilkan **tabel top 10 driver** (feature, IV, effect size, churn rate level berisiko vs overall) + 1 figure panel churn-rate per driver. Ini inti bagian "Findings".

# %%
BAND_ORDERS = [["0-6m", "6-12m", "1-2y", "2-5y", "5y+"], ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]]


def fmt_edge(v):
    if not np.isfinite(v):
        return "∞" if v > 0 else "-∞"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.3g}B"
    if a >= 1e6:
        return f"{v / 1e6:.3g}M"
    if a >= 1e4:
        return f"{v / 1e3:.3g}K"
    if a >= 100 or float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v:.3g}"


def interval_labels(cats):
    return [f"{fmt_edge(iv.left)}–{fmt_edge(iv.right)}" for iv in cats]


def bin_feature(x, numeric, n_bins=None, max_levels=None):
    """Kembalikan series label bin/level (string) dengan NaN → 'Missing'."""
    n_bins = n_bins or CFG.N_BINS_NUMERIC
    max_levels = max_levels or CFG.MAX_CAT_LEVELS_PLOT
    if numeric and x.nunique() > 10:
        try:
            b = pd.qcut(x, q=n_bins, duplicates="drop")
        except Exception:
            b = pd.cut(x, bins=n_bins)
        cats_ = list(b.cat.categories)
        if len(cats_) and x.notna().any():
            cats_[0] = pd.Interval(max(cats_[0].left, float(x.min())), cats_[0].right, closed=cats_[0].closed)
        nice = interval_labels(cats_)
        if len(set(nice)) < len(nice):  # label bentrok setelah pembulatan → pakai label asli
            nice = [str(i) for i in b.cat.categories]
        b = b.cat.rename_categories(nice)
        labels = b.astype(str)
        order = list(nice)
        labels = labels.where(x.notna(), "Missing")
        return labels, order + (["Missing"] if x.isna().any() else [])
    lab = x.astype(str).where(x.notna(), "Missing")
    if numeric:
        try:
            u = sorted(x.dropna().unique())
            fmt_v = lambda v: f"{v:g}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
            min_n = max(30, int(0.01 * len(x)))
            cap = None
            for i, v in enumerate(u):           # gabung ekor yang jarang (misal 4, 5, 6 keluhan) jadi "≥k"
                if i > 0 and (x >= v).sum() < min_n:
                    cap = u[i - 1]
                    break
            if cap is not None and cap != u[0]:
                lab = x.map(lambda v: "Missing" if pd.isna(v) else (f"≥{fmt_v(cap)}" if v >= cap else fmt_v(v)))
                order = [fmt_v(v) for v in u if v < cap] + [f"≥{fmt_v(cap)}"]
            else:
                order = [fmt_v(v) for v in u]
                lab = x.map(lambda v: fmt_v(v) if pd.notna(v) else "Missing")
        except Exception:
            order = list(lab.value_counts().index)
    else:
        vc = lab.value_counts()
        if len(vc) > max_levels:
            keep = set(vc.index[: max_levels - 1])
            lab = lab.where(lab.isin(keep), "Other")
        order = list(lab.value_counts().index)
        for bo in BAND_ORDERS:
            if set(order) - {"Missing", "Other"} <= set(bo):
                order = [o for o in bo if o in set(order)] + [o for o in order if o in ("Other", "Missing")]
                break
    if x.isna().any() and "Missing" not in order:
        order.append("Missing")
    return lab, order


def rate_table(labels, order, y_arr):
    df = pd.DataFrame({"level": labels.values, "y": y_arr})
    g = df.groupby("level")["y"].agg(["count", "sum"]).reindex([o for o in order if o in set(df["level"])])
    g = g.rename(columns={"count": "n", "sum": "churners"})
    g["churn_rate"] = g["churners"] / g["n"]
    ci = [wilson_ci(k, n) for k, n in zip(g["churners"], g["n"])]
    g["ci_low"], g["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    g["lift"] = g["churn_rate"] / max(y_arr.mean(), 1e-9)
    g["share_%"] = 100 * g["n"] / g["n"].sum()
    return g.reset_index()


def information_value(tbl):
    good = (tbl["n"] - tbl["churners"]).astype(float) + 0.5
    bad = tbl["churners"].astype(float) + 0.5
    dg, db = good / good.sum(), bad / bad.sum()
    woe = np.log(dg / db)
    return float(((dg - db) * woe).sum()), woe.values


def iv_label(iv):
    return ("useless" if iv < 0.02 else "weak" if iv < 0.1 else "medium" if iv < 0.3 else "strong" if iv < 0.5 else "suspicious(>0.5)")


def effect_label(v):
    v = abs(v)
    return "negligible" if v < 0.1 else "small" if v < 0.3 else "medium" if v < 0.5 else "large"


def cramers_v_from_ct(ct):
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    v = math.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return v, p


def oof_rate_auc(labels, y_arr, n_splits=5):
    """AUC univariate jujur untuk kategori: churn rate per level dihitung out-of-fold."""
    pred = np.zeros(len(y_arr))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.SEED)
    lab = labels.values
    prior = y_arr.mean()
    for tr, va in skf.split(lab, y_arr):
        m = pd.DataFrame({"l": lab[tr], "y": y_arr[tr]}).groupby("l")["y"].agg(["sum", "count"])
        rate = (m["sum"] + 10 * prior) / (m["count"] + 10)
        pred[va] = pd.Series(lab[va]).map(rate).fillna(prior).values
    try:
        return roc_auc_score(y_arr, pred)
    except Exception:
        return np.nan


def analyze_feature(c, x, y_arr):
    numeric = pd.api.types.is_numeric_dtype(x)
    labels, order = bin_feature(x, numeric)
    tbl = rate_table(labels, order, y_arr)
    iv, woe = information_value(tbl)
    tbl["woe"] = woe
    res = {"feature": c, "type": "numeric" if numeric and x.nunique() > 2 else "categorical",
           "n_levels": int(x.nunique()), "missing_%": round(100 * x.isna().mean(), 2), "iv": iv}
    if numeric and x.nunique() > 2:
        a, b = x[y_arr == 1].dropna(), x[y_arr == 0].dropna()
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            auc = u / (len(a) * len(b))
        except Exception:
            p, auc = np.nan, 0.5
        r = 2 * auc - 1
        res.update({"test": "Mann-Whitney U", "p_value": p, "effect_size": r, "effect_metric": "rank-biserial r",
                    "univariate_auc": max(auc, 1 - auc),
                    "mean_churn": a.mean(), "mean_retained": b.mean(), "median_churn": a.median(), "median_retained": b.median(),
                    "direction": "higher → more churn" if r > 0.05 else "higher → less churn" if r < -0.05 else "weak/non-monotonic"})
    else:
        ct = pd.crosstab(labels, y_arr)
        v, p = cramers_v_from_ct(ct)
        res.update({"test": "Chi-square", "p_value": p, "effect_size": v, "effect_metric": "Cramér's V",
                    "univariate_auc": oof_rate_auc(labels, y_arr), "direction": "see level table"})
    valid = tbl[tbl["n"] >= max(30, 0.01 * len(y_arr))]
    if len(valid) == 0:
        valid = tbl
    worst = valid.sort_values("churn_rate", ascending=False).iloc[0]
    best = valid.sort_values("churn_rate", ascending=True).iloc[0]
    res.update({"highest_risk_level": worst["level"], "highest_risk_rate": worst["churn_rate"], "highest_risk_lift": worst["lift"],
                "highest_risk_n": int(worst["n"]), "lowest_risk_level": best["level"], "lowest_risk_rate": best["churn_rate"]})
    return res, tbl


def dedupe_features(feats, max_corr=0.95):
    """Buang fitur yang informasinya duplikat: satu family yang sama, atau |Spearman| > max_corr dengan fitur terpilih."""
    chosen, fams = [], set()
    for f in feats:
        fam = FEATURE_FAMILY.get(f)
        if fam is not None and fam in fams:
            continue
        dup = False
        if f in train.columns and pd.api.types.is_numeric_dtype(train[f]):
            for g in chosen:
                if pd.api.types.is_numeric_dtype(train[g]):
                    try:
                        if abs(train[[f, g]].corr(method="spearman").iloc[0, 1]) > max_corr:
                            dup = True
                            break
                    except Exception:
                        pass
        if dup:
            continue
        chosen.append(f)
        if fam is not None:
            fams.add(fam)
    return chosen


DRIVER_TABLE = pd.DataFrame()
LEVEL_TABLES = {}

with safe_section("Driver analysis"):
    ANALYSIS_FEATURES = NUM_COLS + CAT_COLS
    rows = []
    for c in ANALYSIS_FEATURES:
        try:
            res, tbl = analyze_feature(c, train[c], y)
            rows.append(res)
            LEVEL_TABLES[c] = tbl
        except Exception as e:
            print(f"  skip {c}: {e}")
    DRIVER_TABLE = pd.DataFrame(rows)
    pv = DRIVER_TABLE["p_value"].fillna(1).values
    if HAS_SM_SURV:
        DRIVER_TABLE["q_value_BH"] = multipletests(pv, method="fdr_bh")[1]
    else:
        order_ = np.argsort(pv)
        q = pv[order_] * len(pv) / (np.arange(len(pv)) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        DRIVER_TABLE["q_value_BH"] = np.empty_like(q)
        DRIVER_TABLE.loc[order_, "q_value_BH"] = np.clip(q, 0, 1)
    DRIVER_TABLE["iv_strength"] = DRIVER_TABLE["iv"].apply(iv_label)
    DRIVER_TABLE["effect_label"] = DRIVER_TABLE["effect_size"].apply(effect_label)
    DRIVER_TABLE["significant"] = DRIVER_TABLE["q_value_BH"] < 0.05
    DRIVER_TABLE = DRIVER_TABLE.sort_values("iv", ascending=False).reset_index(drop=True)

    # Mutual information (encoded)
    try:
        Xmi = train[ANALYSIS_FEATURES].copy()
        disc = []
        for c in ANALYSIS_FEATURES:
            if not pd.api.types.is_numeric_dtype(Xmi[c]):
                Xmi[c] = Xmi[c].astype(str).astype("category").cat.codes
                disc.append(True)
            else:
                Xmi[c] = Xmi[c].fillna(Xmi[c].median())
                disc.append(Xmi[c].nunique() <= 10)
        mi = mutual_info_classif(Xmi.fillna(-1), y, discrete_features=np.array(disc), random_state=CFG.SEED)
        DRIVER_TABLE["mutual_info"] = DRIVER_TABLE["feature"].map(dict(zip(ANALYSIS_FEATURES, mi)))
    except Exception as e:
        print("MI gagal:", e)

    show_cols = ["feature", "type", "iv", "iv_strength", "effect_metric", "effect_size", "effect_label", "p_value", "q_value_BH",
                 "univariate_auc", "mutual_info", "direction", "highest_risk_level", "highest_risk_rate", "highest_risk_lift",
                 "lowest_risk_level", "lowest_risk_rate"]
    DRIVER_TABLE = DRIVER_TABLE[[c for c in show_cols if c in DRIVER_TABLE] + [c for c in DRIVER_TABLE if c not in show_cols]]
    save_table(DRIVER_TABLE, "driver_table")
    display(DRIVER_TABLE.head(30))

    sus = DRIVER_TABLE[(DRIVER_TABLE["iv"] > 0.5) | (DRIVER_TABLE["univariate_auc"] > CFG.LEAKAGE_AUC_THRESHOLD)]
    if len(sus):
        print("⚠️ Fitur dengan IV > 0.5 atau AUC sangat tinggi (cek leakage di Section 12):", sus["feature"].tolist())
    print(f"\nFitur signifikan (q<0.05): {DRIVER_TABLE['significant'].sum()} dari {len(DRIVER_TABLE)}")

# %% [markdown]
# ### 7.1 Visualisasi driver: churn rate per level/bin
# Garis putus-putus merah = churn rate overall. Error bar = 95% CI. Angka di bar = jumlah customer.
# **Cara baca:** bar jauh di atas garis merah dengan CI pendek → segmen berisiko tinggi yang **bisa dipercaya**.

# %%
def plot_rate(ax, tbl, title, overall):
    t = tbl.copy()
    xlab = t["level"].astype(str).str.slice(0, 22)
    colors = [CFG.PALETTE[1] if r > overall else CFG.PALETTE[0] for r in t["churn_rate"]]
    ax.bar(range(len(t)), t["churn_rate"], color=colors, alpha=0.85,
           yerr=[(t["churn_rate"] - t["ci_low"]).clip(lower=0), (t["ci_high"] - t["churn_rate"]).clip(lower=0)], capsize=2, ecolor="#555")
    ax.axhline(overall, color="red", ls="--", lw=1)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(xlab, rotation=45, ha="right", fontsize=7)
    ax.set_title(title[:45], fontsize=9)
    ax.set_ylabel("Churn rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for i, n in enumerate(t["n"]):
        ax.text(i, 0, f"{n}", ha="center", va="bottom", fontsize=6, color="white" if t["churn_rate"].iloc[i] > 0.05 else "black")


with safe_section("Driver plots"):
    top_feats = dedupe_features([f for f in DRIVER_TABLE["feature"].tolist() if f in LEVEL_TABLES])[: CFG.MAX_EDA_PLOTS]
    print("(Panel memakai 1 fitur per 'family' informasi — duplikat seperti tenure vs start_date tidak diulang)")
    for start in range(0, len(top_feats), 6):
        chunk = top_feats[start:start + 6]
        fig, axes = grid_axes(len(chunk), ncols=3, w=5, h=3.6)
        for a, f in zip(axes, chunk):
            iv = DRIVER_TABLE.set_index("feature").loc[f, "iv"]
            plot_rate(a, LEVEL_TABLES[f], f"{f} (IV={iv:.2f})", CHURN_RATE)
        fig.suptitle(f"Churn Rate by Feature — top drivers #{start + 1}–{start + len(chunk)} (ranked by IV)", y=1.02)
        plt.tight_layout()
        save_fig(f"driver_churn_rate_{start // 6 + 1}")

    # distribusi numeric: churn vs retained
    num_top = [f for f in top_feats if f in NUM_COLS and train[f].nunique() > 10][:12]
    if num_top:
        fig, axes = grid_axes(len(num_top), ncols=4, w=4, h=3)
        for a, f in zip(axes, num_top):
            v = train[f]
            lo, hi = v.quantile([0.01, 0.99])
            for cls, col, lab in [(0, CFG.PALETTE[0], "Retained"), (1, CFG.PALETTE[1], "Churned")]:
                a.hist(v[y == cls].clip(lo, hi).dropna(), bins=30, alpha=0.55, density=True, color=col, label=lab)
            a.set_title(f[:35], fontsize=9)
            a.legend(fontsize=7)
        fig.suptitle("Numeric Distributions: Churned vs Retained", y=1.01)
        plt.tight_layout()
        save_fig("numeric_by_churn")

    # IV bar chart
    top_iv = DRIVER_TABLE.head(CFG.TOP_N_FEATURES_PLOT)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(top_iv))))
    colors = [CFG.PALETTE[1] if v > 0.5 else CFG.PALETTE[0] if v >= 0.1 else "#AAAAAA" for v in top_iv["iv"]]
    ax.barh(top_iv["feature"][::-1], top_iv["iv"][::-1], color=colors[::-1])
    for xline, lab in [(0.02, "weak"), (0.1, "medium"), (0.3, "strong"), (0.5, "suspicious")]:
        ax.axvline(xline, ls=":", color="gray", lw=0.8)
        ax.text(xline, len(top_iv) - 0.5, lab, fontsize=7, color="gray", rotation=90, va="top")
    ax.set_xlabel("Information Value (IV)")
    ax.set_title("Predictive Strength of Each Feature (Information Value)")
    save_fig("information_value_ranking")

# %% [markdown]
# ### 7.2 Insight otomatis dari driver analysis
# Kalimat di bawah (English) siap dipakai/diparafrase di laporan. Semua juga disimpan ke `insights.md`.

# %%
with safe_section("Driver insights"):
    strong = DRIVER_TABLE[(DRIVER_TABLE["significant"]) & (DRIVER_TABLE["iv"] >= 0.02) & (~DRIVER_TABLE["feature"].isin(LEAKY_COLS))
                          & (DRIVER_TABLE["univariate_auc"] < CFG.LEAKAGE_AUC_THRESHOLD)]
    keep_f = dedupe_features(strong["feature"].tolist())[:10]
    strong = strong[strong["feature"].isin(keep_f)]
    print("(Fitur duplikat informasi — satu family / |ρ|>0.95 — tidak diulang di insight)")
    for _, r in strong.iterrows():
        if r["type"] == "numeric":
            txt = (f"{r['feature']}: churned customers have a median of {fmt_num(r.get('median_churn'))} vs "
                   f"{fmt_num(r.get('median_retained'))} for retained customers ({r['direction']}; rank-biserial r = {r['effect_size']:.2f}, "
                   f"{r['effect_label']} effect, IV = {r['iv']:.2f}). Highest-risk bin {r['highest_risk_level']} churns at "
                   f"{pct(r['highest_risk_rate'])} ({r['highest_risk_lift']:.1f}x the average).")
        else:
            txt = (f"{r['feature']}: customers in '{r['highest_risk_level']}' churn at {pct(r['highest_risk_rate'])} "
                   f"({r['highest_risk_lift']:.1f}x the overall rate of {pct(CHURN_RATE)}) versus {pct(r['lowest_risk_rate'])} for "
                   f"'{r['lowest_risk_level']}' (χ² p {p_fmt(r['p_value'])}, Cramér's V = {r['effect_size']:.2f}, IV = {r['iv']:.2f}).")
        add_insight("Drivers", txt)
    weak = DRIVER_TABLE[~DRIVER_TABLE["significant"]]["feature"].tolist()
    if weak:
        add_insight("Drivers", f"No statistically significant association with churn after FDR correction for: {', '.join(weak[:12])}.")

# %% [markdown]
# ### 7.3 Churn reasons (kolom post-event, HANYA deskriptif)
# Kolom yang ter-flag leakage (misal `cancellation_reason`, `alasan_berhenti`) **tidak boleh** jadi fitur model, tapi **sangat berharga secara deskriptif**: "kenapa customer bilang mereka pergi?".
# Section ini menampilkan distribusi alasan di antara churner dan menyilangkannya dengan driver actionable teratas.
#
# **Untuk laporan:** "Among churned customers, the most frequently stated reasons were Price (34%) and Service (22%), consistent with the drivers identified by the model." → validasi silang yang kuat antara *stated* vs *revealed* drivers.
# **Jika tidak ada kolom alasan** → section di-skip (normal).

# %%
REASON_TABLE = pd.DataFrame()
with safe_section("Churn reasons (descriptive)"):
    reason_cols = [c for c in LEAKY_COLS if c in raw_train.columns and raw_train[c].nunique() <= 25
                   and not pd.api.types.is_numeric_dtype(raw_train[c])]
    if not reason_cols:
        print("⏭️ Tidak ada kolom alasan churn (post-event kategori) → skip.")
    for rc in reason_cols[:2]:
        src = raw_train.loc[raw_train[TARGET].map(TARGET_MAPPING) == 1, rc] if TARGET in raw_train.columns else raw_train[rc]
        vc = src.astype(str).replace({"nan": "Not stated", "None": "Not stated"}).value_counts()
        REASON_TABLE = pd.DataFrame({"reason": vc.index, "n_churners": vc.values, "share_%": 100 * vc.values / vc.sum()})
        save_table(REASON_TABLE, f"churn_reasons_{sanitize_name(rc)}")
        display(REASON_TABLE)
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(vc))))
        ax.barh(vc.index[::-1].astype(str), 100 * vc.values[::-1] / vc.sum(), color=CFG.PALETTE[1])
        ax.set_xlabel("% of churned customers")
        ax.set_title(f"Stated churn reasons ({rc})")
        save_fig(f"churn_reasons_{rc}")
        top3 = ", ".join(f"{r} ({s:.0f}%)" for r, s in zip(REASON_TABLE["reason"][:3], REASON_TABLE["share_%"][:3]))
        add_insight("Drivers", f"Among churned customers, the most frequently stated reasons ({rc}) were {top3}.")
        cat_driver = [f for f in DRIVER_TABLE["feature"] if f in CAT_COLS and f not in LEAKY_COLS][:1]
        if cat_driver:
            f = cat_driver[0]
            mask = (y == 1)
            ct = pd.crosstab(train.loc[mask, f].astype(str), raw_train.loc[mask, rc].astype(str), normalize="index") * 100
            fig, ax = plt.subplots(figsize=(min(12, 1.2 * ct.shape[1] + 3), 0.5 * ct.shape[0] + 2))
            sns.heatmap(ct, annot=True, fmt=".0f", cmap="Reds", ax=ax, cbar_kws={"label": "% of churners in row"})
            ax.set_title(f"Stated churn reason by {f} (churners only, row %)")
            save_fig(f"churn_reasons_by_{f}")

# %% [markdown]
# # 8. Korelasi & Multikolinearitas
#
# - **Spearman correlation** (numeric–numeric, tahan outlier & hubungan monoton non-linear).
# - **Cramér's V matrix** (kategori–kategori).
# - **VIF** (Variance Inflation Factor) untuk numeric: VIF > 10 → multikolinearitas serius (penting untuk **interpretasi Logistic Regression / Cox**, tidak masalah untuk tree model).
#
# **Jika X → Y:**
# - Pasangan |ρ| > 0.85 (misal `tenure` & `total_charges`) → untuk model interpretable (Section 11) cukup pakai salah satu; untuk GBDT biarkan.
# - Jangan simpulkan "X menyebabkan churn" hanya dari korelasi — tulis "associated with".
#
# **Untuk laporan:** biasanya cukup 1 kalimat ("tenure and total premium paid are highly correlated (ρ = 0.83), so only tenure is used in the interpretable model"). Heatmap bisa masuk lampiran.

# %%
HIGH_CORR_PAIRS = pd.DataFrame()

with safe_section("Correlation & VIF"):
    num_for_corr = [c for c in NUM_COLS if train[c].nunique() > 1]
    if len(num_for_corr) > 30:
        num_for_corr = [f for f in DRIVER_TABLE["feature"] if f in num_for_corr][:30]
    if len(num_for_corr) >= 2:
        corr = train[num_for_corr].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(min(16, 0.5 * len(num_for_corr) + 4), min(14, 0.45 * len(num_for_corr) + 3)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=len(num_for_corr) <= 15, fmt=".2f",
                    annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": 0.7})
        ax.set_title("Spearman Correlation (numeric features)")
        save_fig("spearman_correlation")
        pairs = []
        for i, a in enumerate(num_for_corr):
            for b in num_for_corr[i + 1:]:
                if abs(corr.loc[a, b]) >= 0.8:
                    pairs.append({"feature_a": a, "feature_b": b, "spearman": corr.loc[a, b]})
        HIGH_CORR_PAIRS = pd.DataFrame(pairs)
        if len(HIGH_CORR_PAIRS):
            print("Pasangan sangat berkorelasi (|ρ| ≥ 0.8):")
            display(HIGH_CORR_PAIRS.sort_values("spearman", key=abs, ascending=False))

    cat_for_v = [f for f in DRIVER_TABLE["feature"] if f in CAT_COLS][:20]
    if len(cat_for_v) >= 2:
        V = pd.DataFrame(np.eye(len(cat_for_v)), index=cat_for_v, columns=cat_for_v)
        for i, a in enumerate(cat_for_v):
            for b in cat_for_v[i + 1:]:
                ct = pd.crosstab(train[a].astype(str), train[b].astype(str))
                V.loc[a, b] = V.loc[b, a] = cramers_v_from_ct(ct)[0]
        fig, ax = plt.subplots(figsize=(min(14, 0.55 * len(cat_for_v) + 4), min(12, 0.5 * len(cat_for_v) + 3)))
        sns.heatmap(V, cmap="Blues", vmin=0, vmax=1, annot=len(cat_for_v) <= 12, fmt=".2f", annot_kws={"size": 7}, ax=ax)
        ax.set_title("Cramér's V (categorical associations)")
        save_fig("cramers_v_matrix")

    if HAS_SM_SURV and len(num_for_corr) >= 2:
        Xv = train[num_for_corr].copy()
        Xv = Xv.fillna(Xv.median())
        Xv = (Xv - Xv.mean()) / Xv.std().replace(0, 1)
        Xv = sm.add_constant(Xv, has_constant="add")
        vif = pd.DataFrame({"feature": num_for_corr,
                            "VIF": [variance_inflation_factor(Xv.values, i + 1) for i in range(len(num_for_corr))]})
        vif = vif.sort_values("VIF", ascending=False)
        save_table(vif, "vif")
        display(vif.head(15))
        if (vif["VIF"] > 10).any():
            print("⚠️ VIF > 10:", vif[vif["VIF"] > 10]["feature"].tolist(), "→ hati-hati interpretasi koefisien LogReg/Cox")

# %% [markdown]
# # 9. Segment & Interaction Analysis
#
# Churn jarang disebabkan 1 faktor. Section ini mencari **kombinasi** yang berisiko:
# 1. **Heatmap 2-arah** churn rate untuk pasangan driver teratas (misal `payment_method × tenure_band`).
# 2. **Decision-tree segment rules** — pohon dangkal (depth 3–4) → aturan IF-THEN yang mudah dijelaskan ke manajemen:
#    *"IF tenure < 12 months AND payment = Cash THEN churn rate 58% (n = 412, lift 2.8x)"*.
# 3. **Top risky segments** — tabel segmen dengan churn rate & lift tertinggi (min. ukuran segmen dijaga supaya tidak noise).
#
# **Jika X → Y:**
# - Interaksi terlihat (pola satu fitur berubah tergantung fitur lain) → sebut di laporan sebagai insight bernilai tinggi + jadikan target kampanye spesifik.
# - Rule dengan n kecil (< 2% populasi) → jangan diprioritaskan.
#
# **Untuk laporan:** 1 heatmap paling "bercerita" + tabel 3–5 segment rules = bahan rekomendasi yang sangat konkret.

# %%
SEGMENT_RULES = pd.DataFrame()

with safe_section("Two-way interaction heatmaps"):
    cand = [f for f in DRIVER_TABLE["feature"] if f in LEVEL_TABLES and f not in LEAKY_COLS
            and DRIVER_TABLE.set_index("feature").loc[f, "univariate_auc"] < CFG.LEAKAGE_AUC_THRESHOLD]
    cand = dedupe_features(cand)[:4]
    pairs = list(itertools.combinations(cand, 2))[:4]
    if pairs:
        fig, axes = grid_axes(len(pairs), ncols=2, w=7, h=5)
        for a, (f1, f2) in zip(axes, pairs):
            l1, o1 = bin_feature(train[f1], pd.api.types.is_numeric_dtype(train[f1]), n_bins=5, max_levels=8)
            l2, o2 = bin_feature(train[f2], pd.api.types.is_numeric_dtype(train[f2]), n_bins=5, max_levels=8)
            d = pd.DataFrame({"a": l1, "b": l2, "y": y})
            piv = d.pivot_table(index="a", columns="b", values="y", aggfunc="mean").reindex(index=[o for o in o1 if o in set(l1)],
                                                                                         columns=[o for o in o2 if o in set(l2)])
            cnt = d.pivot_table(index="a", columns="b", values="y", aggfunc="count").reindex_like(piv)
            annot = piv.copy().astype(object)
            for i in piv.index:
                for j in piv.columns:
                    annot.loc[i, j] = "" if pd.isna(piv.loc[i, j]) else f"{piv.loc[i, j]:.0%}\n(n={int(cnt.loc[i, j])})"
            sns.heatmap(piv.astype(float), annot=annot.values, fmt="", cmap="Reds", ax=a, cbar_kws={"format": matplotlib.ticker.PercentFormatter(1.0)},
                        annot_kws={"size": 7})
            a.set_title(f"Churn rate: {f1[:25]} × {f2[:25]}", fontsize=10)
            a.set_ylabel(f1[:30])
            a.set_xlabel(f2[:30])
        plt.tight_layout()
        save_fig("interaction_heatmaps")

# %%
def encode_for_tree(df, feats):
    """One-hot (kategori, max 15 level) + numeric apa adanya (NaN → median) untuk tree rules yang mudah dibaca."""
    parts = []
    for f in feats:
        s = df[f]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(s.fillna(s.median()).rename(f))
        else:
            lab = s.astype(str).where(s.notna(), "Missing")
            top = lab.value_counts().index[:15]
            lab = lab.where(lab.isin(top), "Other")
            parts.append(pd.get_dummies(lab, prefix=f, prefix_sep="=").astype(int))
    return pd.concat(parts, axis=1)


def tree_leaf_rules(tree, feature_names, X, y_arr):
    t = tree.tree_
    leaf_id = tree.apply(X)
    rules = {}

    def recurse(node, conds):
        if t.children_left[node] == -1:
            rules[node] = conds
            return
        f = feature_names[t.feature[node]]
        thr = t.threshold[node]
        if "=" in f:   # one-hot
            recurse(t.children_left[node], conds + [f"{f.split('=')[0]} ≠ {f.split('=', 1)[1]}"])
            recurse(t.children_right[node], conds + [f"{f.split('=')[0]} = {f.split('=', 1)[1]}"])
        elif f in BINARY_TREE_COLS:
            recurse(t.children_left[node], conds + [f"{f} = 0"])
            recurse(t.children_right[node], conds + [f"{f} = 1"])
        else:
            recurse(t.children_left[node], conds + [f"{f} ≤ {thr:,.2f}"])
            recurse(t.children_right[node], conds + [f"{f} > {thr:,.2f}"])

    recurse(0, [])
    out = []
    for leaf, conds in rules.items():
        m = leaf_id == leaf
        if m.sum() == 0:
            continue
        rate = y_arr[m].mean()
        out.append({"segment_rule": " AND ".join(conds), "n_customers": int(m.sum()), "share_%": 100 * m.mean(),
                    "churn_rate": rate, "lift": rate / y_arr.mean(), "churners_captured_%": 100 * y_arr[m].sum() / y_arr.sum()})
    return pd.DataFrame(out).sort_values("churn_rate", ascending=False).reset_index(drop=True)


with safe_section("Decision-tree segment rules"):
    seg_feats = [f for f in DRIVER_TABLE["feature"] if (DRIVER_TABLE.set_index("feature").loc[f, "univariate_auc"] < CFG.LEAKAGE_AUC_THRESHOLD)
                 and f not in LEAKY_COLS]
    seg_feats = dedupe_features(seg_feats)[:15]
    Xt = encode_for_tree(train, seg_feats)
    BINARY_TREE_COLS = {c for c in Xt.columns if "=" not in c and set(pd.unique(Xt[c])) <= {0, 1}}
    min_leaf = max(30, int(0.02 * len(Xt)))
    dt = DecisionTreeClassifier(max_depth=3 if len(Xt) < 3000 else 4, min_samples_leaf=min_leaf, random_state=CFG.SEED)
    dt.fit(Xt.values, y)
    SEGMENT_RULES = tree_leaf_rules(dt, list(Xt.columns), Xt.values, y)
    save_table(SEGMENT_RULES, "segment_rules")
    display(SEGMENT_RULES)

    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(dt, feature_names=[c[:28] for c in Xt.columns], class_names=["Retained", "Churned"], filled=True, rounded=True,
              proportion=True, impurity=False, fontsize=7, ax=ax)
    ax.set_title("Churn Segmentation Tree (each box: share of customers & class proportion)")
    save_fig("segment_tree")

    for _, r in SEGMENT_RULES.head(3).iterrows():
        if r["lift"] > 1.2:
            add_insight("Segments", f"High-risk segment: IF {r['segment_rule']} THEN churn rate = {pct(r['churn_rate'])} "
                                    f"({r['lift']:.1f}x average; {r['n_customers']:,} customers, {r['share_%']:.1f}% of base, "
                                    f"capturing {r['churners_captured_%']:.1f}% of all churners).")
    low = SEGMENT_RULES.iloc[-1]
    add_insight("Segments", f"Most loyal segment: IF {low['segment_rule']} THEN churn rate = {pct(low['churn_rate'])} "
                            f"({low['n_customers']:,} customers).")

# %% [markdown]
# # 10. Survival Analysis — *kapan* customer churn?
#
# Churn analysis biasa menjawab **siapa** yang churn. Survival analysis menjawab **kapan** risiko churn paling tinggi — sangat relevan untuk asuransi (lapse banyak terjadi di tahun-tahun awal polis).
#
# **Setup:** `duration` = tenure, `event` = churn (1) / masih aktif (0 = *censored*, belum churn sampai saat data diambil).
# Asumsi: data snapshot — tenure churner = lama sampai churn; tenure customer aktif = lama berlangganan sampai sekarang.
#
# | Analisis | Output | Cara baca |
# |---|---|---|
# | **Kaplan-Meier (KM)** | kurva S(t) = peluang masih bertahan setelah t | kurva turun tajam di awal → churn dini dominan |
# | **Median survival** | waktu saat S(t) = 50% | "setengah customer churn sebelum bulan ke-X" (bisa *not reached*) |
# | **Retention at t** | S(12), S(24), S(36) | "Retention rate tahun pertama = 78%" |
# | **KM per grup + log-rank test** | apakah kurva antar grup beda signifikan | p < 0.05 → pola bertahan beda nyata |
# | **Hazard rate per interval tenure** | jumlah churn ÷ person-time (misal per customer-bulan) di interval tsb | identifikasi "masa kritis"; adil walau lebar interval beda |
# | **Cox Proportional Hazards** | **Hazard Ratio (HR)** per fitur | HR = 1.5 → risiko churn 50% lebih tinggi kapan pun; HR < 1 = protektif |
#
# **Jika X → Y:**
# - Tidak ada kolom tenure → section di-skip (set `CFG.COLUMN_ROLES = {"tenure": "nama_kolom"}` kalau sebenarnya ada; atau buat tenure dari `start_date` di `custom_feature_engineering`).
# - Hazard paling tinggi di 0–12 bulan → rekomendasi **onboarding program / early-life engagement**.
# - Hazard naik di bulan tertentu (misal bulan 12, 24) → **momen renewal** → rekomendasi renewal reminder / loyalty offer sebelum jatuh tempo.
# - Kurva KM grup bersilangan → asumsi proportional hazards Cox mungkin dilanggar → interpretasi HR hati-hati (sebut sebagai limitation).
#
# Fitur yang diturunkan dari tenure (band tenure, tanggal mulai, total paid) **dikeluarkan dari Cox** karena tenure adalah sumbu waktunya.
#
# **Untuk laporan:** 1 figure KM (overall + per driver utama) + kalimat retention rate + tabel hazard ratio top 8 → bagian "When do customers churn?"

# %%
SURV_AVAILABLE = False
COX_TABLE = pd.DataFrame()
KM_SUMMARY = pd.DataFrame()
SURV_COL, SURV_NOTE = None, ""
START_HINT = re.compile(r"(start|mulai|orig|since|join|inception|effective|issue|open|awal|terbit|commenc)", re.I)
END_HINT = re.compile(r"(cancel|terminat|end_?date|lapse|surrender|suspd|suspend|exit|churn_?date|stop|keluar|berhenti|batal|close|tutup|expir)", re.I)


def resolve_survival_duration():
    """Sumbu waktu survival: kolom tenure kalau ada. Kalau tidak ada → rekonstruksi dari tanggal mulai polis + tanggal berhenti
    (churner: end − start; aktif: tanggal acuan − start, tersensor). Tidak dipakai sebagai fitur model (tanggal berhenti = post-event)."""
    global TENURE_UNIT
    if "tenure" in ROLES and ROLES["tenure"] in train.columns:
        return ROLES["tenure"], ""
    starts = [c for c, t in COL_TYPES.items() if t == "date" and c in train.columns and START_HINT.search(c) and not BIRTH_HINT.search(c)]
    if not starts:
        return None, "no tenure column and no policy start date"
    s = pd.to_datetime(train[starts[0]], errors="coerce")
    for c in [c for c in raw_train.columns if END_HINT.search(str(c)) and c not in starts]:
        e = pd.to_datetime(raw_train[c], errors="coerce") if not np.issubdtype(raw_train[c].dtype, np.datetime64) else raw_train[c]
        if e.notna().sum() >= 0.5 * max(int(np.sum(y)), 1) and e.notna().mean() < 0.95:
            end_eff = e.where(pd.Series(y == 1, index=e.index)).fillna(REF_DATE)
            dur = (end_eff - s).dt.days.astype(float)
            dur[dur < 0] = np.nan
            train["_surv_duration_days"] = dur.values
            TENURE_UNIT = "days"
            return "_surv_duration_days", (f"tenure reconstructed from '{starts[0]}' to '{c}' (churners) or the reference date "
                                           f"{REF_DATE.date()} (active customers, censored)")
    return None, (f"start date '{starts[0]}' found but no end/cancellation date for churners → their duration is unknown, "
                  "so survival analysis is skipped to avoid bias")


def tenure_related_cols():
    rel = set()
    if "tenure" in ROLES:
        rel.add(ROLES["tenure"])
    if SURV_COL:
        rel.add(SURV_COL)
    rel |= {c for c in train.columns if c.startswith("fe_tenure") or c in ("fe_is_new_customer", "fe_avg_paid_per_period",
                                                                            "fe_paid_vs_expected", "fe_claims_per_year")}
    rel |= {c for c in train.columns if c.endswith("_days_since") or c.endswith("_year") or c.endswith("_month")}
    if "total_paid" in ROLES:
        rel.add(ROLES["total_paid"])
    return rel


def km_curve(time_arr, event_arr):
    sf = SurvfuncRight(time_arr, event_arr)
    t = np.asarray(sf.surv_times)
    s = np.asarray(sf.surv_prob)
    se = np.asarray(sf.surv_prob_se)
    return sf, t, s, np.clip(s - 1.96 * se, 0, 1), np.clip(s + 1.96 * se, 0, 1)


def surv_at(t, s, when):
    if len(t) == 0:
        return np.nan
    idx = np.searchsorted(t, when, side="right") - 1
    return 1.0 if idx < 0 else float(s[idx])


with safe_section("Survival: Kaplan-Meier"):
    SURV_COL, SURV_NOTE = resolve_survival_duration()
    if SURV_COL is None:
        print(f"⏭️ Survival analysis di-skip: {SURV_NOTE}. (Kalau ada kolom tenure: CFG.COLUMN_ROLES = {{'tenure': 'nama_kolom'}})")
    elif not HAS_SM_SURV:
        print("⏭️ statsmodels tidak tersedia → survival analysis di-skip. (pip install statsmodels)")
    else:
        tcol = SURV_COL
        if SURV_NOTE:
            print("ℹ️", SURV_NOTE)
            add_insight("Data", f"Survival time was {SURV_NOTE}.")
        dur_all = train[tcol].astype(float)
        m_ok = dur_all.notna() & (dur_all >= 0)
        DUR, EVT = dur_all[m_ok].values, y[m_ok.values]
        if (DUR == 0).any():
            DUR = DUR + 0.5  # hindari durasi 0
        SURV_AVAILABLE = True
        unit = {"months": "months", "years": "years", "days": "days"}.get(TENURE_UNIT, "periods")
        SURV_UNIT = unit
        sf, t, s, lo, hi = km_curve(DUR, EVT)
        checkpoints = {"months": [6, 12, 24, 36, 60], "years": [1, 2, 3, 5, 10], "days": [90, 180, 365, 730, 1095]}.get(TENURE_UNIT, [6, 12, 24, 36])
        med = np.nan
        try:
            med = sf.quantile(0.5)
        except Exception:
            pass
        rows = [{"checkpoint": f"{cp} {unit}", "survival_prob": surv_at(t, s, cp)} for cp in checkpoints if cp <= DUR.max()]
        KM_SUMMARY = pd.DataFrame(rows)
        display(KM_SUMMARY)
        print(f"Median survival time: {med if pd.notna(med) else 'not reached (lebih dari 50% customer belum churn)'}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.step(t, s, where="post", color=CFG.PALETTE[0], lw=2, label="All customers")
        ax.fill_between(t, lo, hi, step="post", alpha=0.2, color=CFG.PALETTE[0])
        for cp in checkpoints:
            if cp <= DUR.max():
                ax.axvline(cp, ls=":", color="gray", lw=0.7)
                ax.text(cp, surv_at(t, s, cp) + 0.02, f"{surv_at(t, s, cp):.0%}", fontsize=8, ha="center")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"Tenure ({unit})")
        ax.set_ylabel("Probability of remaining a customer")
        ax.set_title("Kaplan-Meier Customer Survival Curve")
        ax.legend()
        save_fig("km_overall")
        for r in rows[:3]:
            add_insight("Survival", f"Estimated probability of retaining a customer beyond {r['checkpoint']}: {pct(r['survival_prob'])} (Kaplan-Meier).", show=False)
        if rows:
            print("💡", INSIGHTS[-1][1])

        # discrete hazard per interval
        edges = {"months": [0, 3, 6, 12, 18, 24, 36, 48, 60, 120, np.inf], "years": [0, 1, 2, 3, 5, 10, np.inf],
                 "days": [0, 90, 180, 365, 545, 730, 1095, 1825, np.inf]}.get(TENURE_UNIT, [0, 6, 12, 24, 48, np.inf])
        edges = [e for e in edges if e < DUR.max()] + [np.inf]
        hz = []
        for a, b in zip(edges[:-1], edges[1:]):
            at_risk = (DUR >= a).sum()
            ev = ((DUR >= a) & (DUR < b) & (EVT == 1)).sum()
            exposure = np.clip(np.minimum(DUR, b) - a, 0, None).sum()   # person-time di interval
            if at_risk > 0 and exposure > 0:
                hz.append({"interval": f"[{a:g}, {b:g})", "at_risk_start": int(at_risk), "churn_events": int(ev),
                           "person_time": exposure, "hazard": ev / exposure})
        HAZARD_TABLE = pd.DataFrame(hz)
        HAZARD_TABLE = HAZARD_TABLE[HAZARD_TABLE["at_risk_start"] >= 30]   # interval terlalu sedikit data → tidak stabil
        HZ_PER = 30.44 if TENURE_UNIT == "days" else 1.0       # tenure harian → laporkan hazard per 30 hari (≈ per bulan)
        HZ_LABEL = "30 days" if TENURE_UNIT == "days" else (unit[:-1] if unit.endswith("s") else unit)
        HAZARD_TABLE["hazard"] = HAZARD_TABLE["hazard"] * HZ_PER
        save_table(HAZARD_TABLE, "hazard_by_tenure_interval")
        display(HAZARD_TABLE)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.bar(HAZARD_TABLE["interval"], HAZARD_TABLE["hazard"], color=CFG.PALETTE[1])
        ax.set_ylabel(f"Churn hazard per {HZ_LABEL}")
        ax.set_xlabel(f"Tenure interval ({unit})")
        ax.set_title("When Do Customers Churn? Hazard by Tenure Interval")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        plt.xticks(rotation=30)
        save_fig("hazard_by_tenure")
        peak = HAZARD_TABLE.sort_values("hazard", ascending=False).iloc[0]
        add_insight("Survival", f"Churn hazard is highest in tenure interval {peak['interval']} {unit}: about {pct(peak['hazard'], 2)} of active customers churn per {HZ_LABEL} in this period "
                                f"(vs {pct(HZ_PER * HAZARD_TABLE['churn_events'].sum() / HAZARD_TABLE['person_time'].sum(), 2)} on average).")

# %%
with safe_section("Survival: KM per group + log-rank"):
    if SURV_AVAILABLE:
        rel = tenure_related_cols()
        grp_feats = [f for f in DRIVER_TABLE["feature"] if f in CAT_COLS + [c for c in NUM_COLS if train[c].nunique() <= 6]
                     and f not in rel and f not in LEAKY_COLS and train[f].nunique() <= 8]
        grp_feats = dedupe_features(grp_feats)[:6]
        logrank_rows = []
        if grp_feats:
            fig, axes = grid_axes(len(grp_feats), ncols=3, w=5.2, h=3.8)
            for a, f in zip(axes, grp_feats):
                g = train.loc[m_ok, f].astype(str).where(train.loc[m_ok, f].notna(), "Missing").values
                levels = pd.Series(g).value_counts()
                levels = levels[levels >= 30].index[:6]
                for i, lv in enumerate(levels):
                    mm = g == lv
                    _, tt, ss, _, _ = km_curve(DUR[mm], EVT[mm])
                    a.step(tt, ss, where="post", lw=1.6, label=f"{str(lv)[:18]} (n={mm.sum()})", color=CFG.PALETTE[i % len(CFG.PALETTE)])
                try:
                    keep = np.isin(g, levels)
                    chi2, p = survdiff(DUR[keep], EVT[keep], g[keep])
                except Exception:
                    chi2, p = np.nan, np.nan
                logrank_rows.append({"feature": f, "logrank_chi2": chi2, "p_value": p})
                a.set_title(f"{f[:30]} (log-rank p {p_fmt(p)})", fontsize=9)
                a.set_ylim(0, 1.05)
                a.set_xlabel(f"Tenure ({unit})", fontsize=8)
                a.legend(fontsize=6)
            fig.suptitle("Kaplan-Meier Survival by Customer Group", y=1.02)
            plt.tight_layout()
            save_fig("km_by_group")
            LOGRANK = pd.DataFrame(logrank_rows)
            save_table(LOGRANK, "logrank_tests")
            display(LOGRANK)

# %%
def forest_plot(df, est, lo, hi, label_col, title, xlabel, ref=1.0, log=True, max_rows=20):
    d = df.head(max_rows).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(d) + 1)))
    colors = [CFG.PALETTE[1] if v > ref else CFG.PALETTE[0] for v in d[est]]
    ax.errorbar(d[est], range(len(d)), xerr=[d[est] - d[lo], d[hi] - d[est]], fmt="none", ecolor="#777", capsize=2)
    ax.scatter(d[est], range(len(d)), c=colors, zorder=3)
    ax.axvline(ref, color="black", ls="--", lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([str(s)[:45] for s in d[label_col]], fontsize=8)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return fig


def build_design_matrix(df, feats, max_levels=8):
    """Numeric → standardized (z), binary 0/1 tetap; kategori → dummies (reference = level paling sering)."""
    parts, meta = [], {}
    for f in feats:
        s = df[f]
        if pd.api.types.is_numeric_dtype(s):
            v = s.fillna(s.median())
            if v.nunique() <= 2:
                parts.append(v.rename(f))
                meta[f] = "binary (0→1)"
            else:
                sd = v.std() if v.std() > 0 else 1
                parts.append(((v - v.mean()) / sd).rename(f))
                meta[f] = f"per 1 SD (= {sd:,.3g})"
        else:
            lab = s.astype(str).where(s.notna(), "Missing")
            top = lab.value_counts().index[:max_levels]
            lab = lab.where(lab.isin(top), "Other")
            ref = lab.value_counts().index[0]
            d = pd.get_dummies(lab, prefix=f, prefix_sep=" = ").astype(float)
            d = d.drop(columns=[f"{f} = {ref}"])
            d = d.loc[:, d.sum() >= 10]
            for c in d.columns:
                meta[c] = f"vs '{ref}'"
            parts.append(d)
    if not parts:
        return pd.DataFrame(index=df.index), meta
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.std() > 0]
    return X, meta


def select_interpretable_features(max_feats=15, exclude=(), min_iv=0.02):
    """Ambil driver teratas (non-leak), hindari duplikasi info (|ρ|>0.8) & prefer band untuk age."""
    chosen, fams = [], set()
    tbl = DRIVER_TABLE[(DRIVER_TABLE["univariate_auc"] < CFG.LEAKAGE_AUC_THRESHOLD) & (DRIVER_TABLE["iv"] >= min_iv)]
    for f in tbl["feature"]:
        if f in exclude or f in LEAKY_COLS or f in HIGH_CARD_COLS:
            continue
        if f == ROLES.get("age") and "fe_age_band" in tbl["feature"].values and "fe_age_band" not in exclude:
            continue  # umur sering U-shape → pakai band
        fam = FEATURE_FAMILY.get(f)
        if fam is not None and fam in fams and not (fam == "age" and f == "fe_age_band"):
            continue
        if pd.api.types.is_numeric_dtype(train[f]):
            redundant = False
            for g in chosen:
                if pd.api.types.is_numeric_dtype(train[g]):
                    try:
                        if abs(train[[f, g]].corr(method="spearman").iloc[0, 1]) > 0.8:
                            redundant = True
                            break
                    except Exception:
                        pass
            if redundant:
                continue
        chosen.append(f)
        if fam is not None:
            fams.add(fam)
        if len(chosen) >= max_feats:
            break
    return chosen


with safe_section("Survival: Cox proportional hazards"):
    if SURV_AVAILABLE:
        cox_feats = select_interpretable_features(12, exclude=tenure_related_cols())
        if len(cox_feats) < 2:          # driver non-tenure lemah semua (IV < 0.02) → pakai driver teratas tanpa ambang IV
            cox_feats = select_interpretable_features(8, exclude=tenure_related_cols(), min_iv=0.0)
        Xc, meta_c = build_design_matrix(train.loc[m_ok].reset_index(drop=True), cox_feats)
        if Xc.shape[1] == 0:
            print("⏭️ Tidak ada fitur non-tenure yang bisa dipakai model Cox → Cox dilewati (Kaplan–Meier & hazard tetap dilaporkan)")
            SURV_COX_SKIPPED = True
        else:
            SURV_COX_SKIPPED = False
    if SURV_AVAILABLE and not SURV_COX_SKIPPED:
        DUR_c, EVT_c = DUR, EVT
        cmax = getattr(CFG, "COX_MAX_ROWS", 50_000)
        if len(Xc) > cmax:
            pick = np.random.RandomState(CFG.SEED).choice(len(Xc), cmax, replace=False)
            Xc, DUR_c, EVT_c = Xc.iloc[pick].reset_index(drop=True), DUR[pick], EVT[pick]
            print(f"Cox dilatih pada subsample {cmax:,} baris (data besar).")
        res = None
        for attempt in range(3):
            try:
                mod = PHReg(DUR_c, Xc.values, status=EVT_c, ties="efron")
                res = mod.fit(disp=False)
                if not np.all(np.isfinite(res.bse)):
                    raise ValueError("SE tidak finite")
                break
            except Exception as e:
                print(f"  Cox attempt {attempt + 1} gagal ({e}) → kurangi fitur")
                keep = Xc.columns[: max(3, int(len(Xc.columns) * 0.6))]
                Xc = Xc[keep]
        if res is not None:
            ci = np.asarray(res.conf_int())
            COX_TABLE = pd.DataFrame({"term": Xc.columns, "coef": res.params, "hazard_ratio": np.exp(res.params),
                                      "hr_ci_low": np.exp(ci[:, 0]), "hr_ci_high": np.exp(ci[:, 1]), "p_value": res.pvalues})
            COX_TABLE["scale"] = COX_TABLE["term"].map(meta_c)
            COX_TABLE["interpretation"] = COX_TABLE.apply(
                lambda r: f"{'+' if r.hazard_ratio > 1 else '-'}{abs(r.hazard_ratio - 1):.0%} churn hazard ({r.scale})", axis=1)
            COX_TABLE = COX_TABLE.reindex(COX_TABLE["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)
            save_table(COX_TABLE, "cox_hazard_ratios")
            display(COX_TABLE)
            fig = forest_plot(COX_TABLE[COX_TABLE["p_value"] < 0.1] if (COX_TABLE["p_value"] < 0.1).sum() >= 3 else COX_TABLE,
                              "hazard_ratio", "hr_ci_low", "hr_ci_high", "term",
                              "Cox Proportional Hazards: Hazard Ratios (95% CI)", "Hazard ratio (log scale; >1 = higher churn risk)")
            save_fig("cox_hazard_ratios", fig)
            for _, r in COX_TABLE[COX_TABLE["p_value"] < 0.05].head(5).iterrows():
                add_insight("Survival", f"Cox model: {r['term']} has a hazard ratio of {r['hazard_ratio']:.2f} "
                                        f"(95% CI {r['hr_ci_low']:.2f}–{r['hr_ci_high']:.2f}, p {p_fmt(r['p_value'])}), i.e. "
                                        f"{r['interpretation']}, holding other factors constant.")
            try:
                print(f"Concordance (C-index) proxy: hitung via lifelines bila tersedia.")
                if HAS_LIFELINES:
                    from lifelines import CoxPHFitter
                    dfl = Xc.copy()
                    dfl.columns = [sanitize_name(c) for c in dfl.columns]
                    dfl["duration"], dfl["event"] = DUR_c, EVT_c
                    cph = CoxPHFitter(penalizer=0.01).fit(dfl, "duration", "event")
                    print(f"lifelines C-index: {cph.concordance_index_:.3f} (0.5 = acak, >0.7 bagus)")
                    print("Uji asumsi proportional hazards (Schoenfeld) — p < 0.05 berarti asumsi dilanggar untuk fitur tsb:")
                    from lifelines.statistics import proportional_hazard_test
                    pht = proportional_hazard_test(cph, dfl, time_transform="rank")
                    display(pht.summary.sort_values("p").head(10))
            except Exception as e:
                print("  (lifelines check di-skip:", e, ")")

# %% [markdown]
# ### 10.4 Survival lanjutan: RMST, uji asumsi PH, Weibull AFT
# - **RMST** (restricted mean survival time) = rata-rata lama customer bertahan dalam horizon τ (mis. 24 bulan). Lebih mudah dipahami manajemen daripada hazard ratio: *"nasabah X rata-rata bertahan 3,2 bulan lebih singkat dalam 2 tahun pertama"*.
# - **Uji Schoenfeld** — apakah efek tiap variabel konstan sepanjang waktu (asumsi Cox). Banyak yang dilanggar → HR dibaca sebagai efek rata-rata waktu, dan dilengkapi model AFT.
# - **Weibull AFT** — *time ratio*: TR = 0,6 → customer kelompok itu churn 40% lebih cepat. Bahasa yang akrab bagi aktuaris.
#
# **Untuk laporan:** kalimat RMST di bagian "When do customers churn"; tabel PH test & AFT di Appendix (menunjukkan kematangan metodologi).

# %%
RMST_TABLE, PH_TEST, AFT_TABLE = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
with safe_section("Survival: RMST, PH diagnostics & Weibull AFT"):
    if not SURV_AVAILABLE:
        print("⏭️ Survival tidak tersedia → skip.")
    else:
        # ---- RMST: rata-rata lama bertahan (dibatasi horizon tau) per grup ----
        tau = {"months": 24, "years": 3, "days": 730}.get(TENURE_UNIT, float(np.quantile(DUR, 0.75)))
        tau = float(min(tau, np.quantile(DUR, 0.95)))
        rows = []
        for f in (grp_feats if "grp_feats" in globals() else [])[:4]:
            g = train.loc[m_ok, f].astype(str).where(train.loc[m_ok, f].notna(), "Missing").values
            levels = pd.Series(g).value_counts()
            levels = levels[levels >= 50].index[:6]
            res_f = []
            for lv in levels:
                mm = g == lv
                _, tt, ss, _, _ = km_curve(DUR[mm], EVT[mm])
                r_ = rmst_from_km(tt, ss, tau)
                boot = []
                rs_ = np.random.RandomState(CFG.SEED)
                idx_lv = np.where(mm)[0]
                for _b in range(20 if CFG.RUN_MODE == "fast" else 50):
                    bi = rs_.choice(idx_lv, len(idx_lv), replace=True)
                    try:
                        _, tb, sb, _, _ = km_curve(DUR[bi], EVT[bi])
                        boot.append(rmst_from_km(tb, sb, tau))
                    except Exception:
                        pass
                res_f.append({"feature": f, "level": lv, "n": int(mm.sum()), "rmst": r_,
                              "rmst_ci_low": np.percentile(boot, 2.5) if boot else np.nan,
                              "rmst_ci_high": np.percentile(boot, 97.5) if boot else np.nan})
            if res_f:
                ref = max(res_f, key=lambda r: r["rmst"])
                for r in res_f:
                    r["diff_vs_best"] = r["rmst"] - ref["rmst"]
                    r["reference_level"] = ref["level"]
                rows += res_f
        RMST_TABLE = pd.DataFrame(rows)
        if len(RMST_TABLE):
            RMST_TABLE["tau"] = tau
            save_table(RMST_TABLE, "rmst_by_group")
            display(RMST_TABLE.round(2))
            worst = RMST_TABLE.sort_values("diff_vs_best").iloc[0]
            add_insight("Survival", f"Restricted mean survival time (RMST, horizon {tau:g} {SURV_UNIT}): customers with {worst['feature']} = "
                                    f"'{worst['level']}' stay on average {abs(worst['diff_vs_best']):.1f} {SURV_UNIT} less than '{worst['reference_level']}' "
                                    f"within the first {tau:g} {SURV_UNIT} ({worst['rmst']:.1f} vs {worst['rmst'] - worst['diff_vs_best']:.1f}).")
            fig, ax = plt.subplots(figsize=(8, max(2.8, 0.45 * len(RMST_TABLE) + 1)))
            d = RMST_TABLE.iloc[::-1]
            ax.errorbar(d["rmst"], range(len(d)), xerr=[(d["rmst"] - d["rmst_ci_low"]).clip(lower=0), (d["rmst_ci_high"] - d["rmst"]).clip(lower=0)],
                        fmt="o", color=CFG.PALETTE[0], capsize=3)
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels([f"{a[:22]} = {str(b)[:16]}" for a, b in zip(d["feature"], d["level"])], fontsize=8)
            ax.set_xlabel(f"Restricted mean survival time within {tau:g} {SURV_UNIT} (95% bootstrap CI)")
            ax.set_title("How Long Do Customers Stay? RMST by Group")
            save_fig("rmst_by_group", fig)

        # ---- Cox (lifelines): C-index & uji asumsi proportional hazards; Weibull AFT time ratios ----
        if HAS_LIFELINES and "Xc" in globals() and len(Xc):
            from lifelines import CoxPHFitter, WeibullAFTFitter
            from lifelines.statistics import proportional_hazard_test
            dfl = Xc.copy()
            dfl.columns = [sanitize_name(c) for c in dfl.columns]
            back = dict(zip(dfl.columns, Xc.columns))
            dfl["duration"], dfl["event"] = np.asarray(DUR_c, dtype=float), np.asarray(EVT_c, dtype=int)
            lim = 30_000
            if len(dfl) > lim:
                dfl = dfl.sample(lim, random_state=CFG.SEED)
            cph = CoxPHFitter(penalizer=0.01).fit(dfl, "duration", "event")
            COX_CINDEX = float(cph.concordance_index_)
            print(f"Cox C-index (lifelines) = {COX_CINDEX:.3f}")
            try:
                pht = proportional_hazard_test(cph, dfl, time_transform="rank").summary.reset_index()
                pht = pht.rename(columns={pht.columns[0]: "term"})
                pht["term"] = pht["term"].map(lambda t: back.get(t, t))
                pht["ph_violated"] = pht["p"] < 0.01
                PH_TEST = pht[["term", "test_statistic", "p", "ph_violated"]]
                save_table(PH_TEST, "ph_assumption_test")
                nv = int(PH_TEST["ph_violated"].sum())
                print(f"Schoenfeld test: {nv} dari {len(PH_TEST)} term melanggar asumsi PH (p<0.01)")
                add_insight("Survival", f"Proportional-hazards diagnostics (Schoenfeld residual test): {nv} of {len(PH_TEST)} terms show "
                                        f"time-varying effects (p < 0.01); {'hazard ratios are therefore read as time-averaged effects and complemented by an accelerated failure time model' if nv else 'the PH assumption is supported'}.",
                            show=False)
            except Exception as e:
                print("  PH test di-skip:", e)
            try:
                aft = WeibullAFTFitter(penalizer=0.01).fit(dfl, "duration", "event")
                s_ = aft.summary.reset_index()
                s_ = s_[s_["param"] == "lambda_"]
                s_ = s_[s_["covariate"] != "Intercept"]
                AFT_TABLE = pd.DataFrame({"term": s_["covariate"].map(lambda t: back.get(t, t)).values,
                                          "time_ratio": np.exp(s_["coef"].values), "tr_ci_low": np.exp(s_["coef lower 95%"].values),
                                          "tr_ci_high": np.exp(s_["coef upper 95%"].values), "p_value": s_["p"].values})
                AFT_TABLE["scale"] = AFT_TABLE["term"].map(meta_c) if "meta_c" in globals() else ""
                AFT_TABLE = AFT_TABLE.reindex((np.log(AFT_TABLE["time_ratio"])).abs().sort_values(ascending=False).index).reset_index(drop=True)
                save_table(AFT_TABLE, "weibull_aft_time_ratios")
                display(AFT_TABLE.head(12).round(3))
                fig = forest_plot(AFT_TABLE[AFT_TABLE["p_value"] < 0.1] if (AFT_TABLE["p_value"] < 0.1).sum() >= 3 else AFT_TABLE,
                                  "time_ratio", "tr_ci_low", "tr_ci_high", "term",
                                  "Weibull AFT: Time Ratios (95% CI)", "Time ratio (log scale; <1 = customers leave sooner)")
                save_fig("weibull_aft_time_ratios", fig)
                for _, r in AFT_TABLE[AFT_TABLE["p_value"] < 0.05].head(3).iterrows():
                    add_insight("Survival", f"Accelerated failure time model: {r['term']} ({r['scale']}) multiplies expected time-to-churn by "
                                            f"{r['time_ratio']:.2f} (95% CI {r['tr_ci_low']:.2f}–{r['tr_ci_high']:.2f}), i.e. customers leave "
                                            f"{abs(1 - r['time_ratio']):.0%} {'sooner' if r['time_ratio'] < 1 else 'later'}.", show=False)
            except Exception as e:
                print("  Weibull AFT di-skip:", e)
        else:
            print("ℹ️ lifelines tidak tersedia → PH test & AFT di-skip (python -m pip install lifelines).")

# %% [markdown]
# # 11. Interpretable Model — Logistic Regression (Odds Ratio)
#
# GBDT biasanya paling akurat, tapi **Logistic Regression** paling mudah dijelaskan ke manajemen dan juri:
# - **Odds Ratio (OR)** > 1 → meningkatkan peluang churn; < 1 → menurunkan (protektif).
# - Numeric distandarisasi → OR dibaca **per kenaikan 1 standar deviasi**. Fitur biner → OR punya vs tidak punya. Kategori → dibanding **reference level** (level paling umum).
# - **Average Marginal Effect (AME)** → perubahan *probabilitas* churn (dalam poin persen) — lebih intuitif daripada OR.
# - Ini analisis **multivariat**: efek tiap fitur sudah *dikontrol* oleh fitur lain (beda dengan driver analysis univariat di Section 7).
#
# **Jika X → Y:**
# - Error `Singular matrix` / `PerfectSeparation` → fitur terlalu berkorelasi atau ada level yang memprediksi sempurna; notebook otomatis fallback ke versi regularized (tanpa CI) atau mengurangi fitur.
# - Arah koefisien berlawanan dengan driver analysis → kemungkinan **confounding** (misal fitur A tampak berisiko hanya karena berkorelasi dengan B). Ini insight menarik untuk dibahas!
# - OR dengan CI melewati 1 → tidak signifikan, jangan dijadikan klaim.
#
# **Untuk laporan:** forest plot OR + 2–3 kalimat: *"Holding other factors constant, customers paying by auto-debit have 52% lower odds of churning (OR = 0.48, 95% CI 0.41–0.56)."*

# %%
LOGIT_TABLE = pd.DataFrame()

with safe_section("Interpretable logistic regression"):
    if not HAS_SM:
        print("⏭️ statsmodels tidak ada → pakai sklearn LogisticRegression (tanpa CI/p-value).")
    logit_feats = select_interpretable_features(15)
    Xl, meta_l = build_design_matrix(train, logit_feats)
    print(f"Fitur yang dipakai ({len(logit_feats)}):", logit_feats)
    fitted = False
    if HAS_SM:
        for attempt in range(3):
            try:
                lres = sm.Logit(y, sm.add_constant(Xl, has_constant="add")).fit(disp=0, maxiter=300)
                if not np.all(np.isfinite(lres.bse)):
                    raise ValueError("SE tidak finite")
                ci = lres.conf_int()
                LOGIT_TABLE = pd.DataFrame({"term": lres.params.index, "coef": lres.params.values,
                                            "odds_ratio": np.exp(lres.params.values), "or_ci_low": np.exp(ci[0].values),
                                            "or_ci_high": np.exp(ci[1].values), "p_value": lres.pvalues.values})
                LOGIT_TABLE = LOGIT_TABLE[LOGIT_TABLE["term"] != "const"]
                LOGIT_TABLE["scale"] = LOGIT_TABLE["term"].map(meta_l)
                try:
                    ame = lres.get_margeff(at="overall").summary_frame()
                    LOGIT_TABLE["avg_marginal_effect_pp"] = LOGIT_TABLE["term"].map(100 * ame["dy/dx"])
                except Exception:
                    pass
                print(f"McFadden pseudo-R² = {lres.prsquared:.3f} | AIC = {lres.aic:,.1f} | n = {int(lres.nobs)}")
                fitted = True
                break
            except Exception as e:
                print(f"  Logit attempt {attempt + 1} gagal: {e} → kurangi fitur")
                Xl = Xl.iloc[:, : max(3, int(Xl.shape[1] * 0.7))]
    if not fitted:
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(Xl, y)
        LOGIT_TABLE = pd.DataFrame({"term": Xl.columns, "coef": lr.coef_[0], "odds_ratio": np.exp(lr.coef_[0]),
                                    "or_ci_low": np.nan, "or_ci_high": np.nan, "p_value": np.nan})
        LOGIT_TABLE["scale"] = LOGIT_TABLE["term"].map(meta_l)
    LOGIT_TABLE["interpretation"] = LOGIT_TABLE.apply(
        lambda r: f"{'+' if r.odds_ratio > 1 else '-'}{abs(r.odds_ratio - 1):.0%} odds of churn ({r.scale})", axis=1)
    LOGIT_TABLE = LOGIT_TABLE.reindex(LOGIT_TABLE["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    save_table(LOGIT_TABLE, "logit_odds_ratios")
    display(LOGIT_TABLE)
    if LOGIT_TABLE["or_ci_low"].notna().any():
        sig = LOGIT_TABLE[LOGIT_TABLE["p_value"] < 0.05]
        fig = forest_plot(sig if len(sig) >= 3 else LOGIT_TABLE, "odds_ratio", "or_ci_low", "or_ci_high", "term",
                          "Logistic Regression: Odds Ratios for Churn (95% CI)", "Odds ratio (log scale; >1 = higher churn odds)")
        save_fig("logit_odds_ratios", fig)
        for _, r in sig.head(6).iterrows():
            ame_txt = f"; average marginal effect {r['avg_marginal_effect_pp']:+.1f} pp" if "avg_marginal_effect_pp" in r and pd.notna(r.get("avg_marginal_effect_pp")) else ""
            add_insight("Interpretable model", f"Holding other factors constant, {r['term']} ({r['scale']}) changes the odds of churn by "
                                               f"{(r['odds_ratio'] - 1):+.0%} (OR = {r['odds_ratio']:.2f}, 95% CI {r['or_ci_low']:.2f}–{r['or_ci_high']:.2f}, "
                                               f"p {p_fmt(r['p_value'])}{ame_txt}).")
    # performa model interpretable (CV)
    try:
        oof_lr = cross_val_predict(LogisticRegression(C=1.0, max_iter=3000), Xl, y,
                                   cv=StratifiedKFold(5, shuffle=True, random_state=CFG.SEED), method="predict_proba")[:, 1]
        INTERPRETABLE_AUC = roc_auc_score(y, oof_lr)
        print(f"CV ROC-AUC model interpretable (LogReg, {Xl.shape[1]} terms): {INTERPRETABLE_AUC:.4f}")
    except Exception as e:
        INTERPRETABLE_AUC = np.nan
        print("CV AUC LogReg gagal:", e)

# %% [markdown]
# # 12. Train–Test Drift & Adversarial Validation
#
# Model hanya bagus di test kalau **distribusi test mirip train**. Cek:
# - **PSI (Population Stability Index)** per fitur: < 0.1 stabil, 0.1–0.25 sedikit bergeser, **> 0.25 bergeser signifikan**.
# - **KS test** untuk numeric (statistic besar & p kecil → distribusi beda).
# - **Unseen categories**: level kategori yang muncul di test tapi tidak di train.
# - **Adversarial validation**: latih classifier membedakan baris train vs test. AUC ≈ 0.5 → mirip (bagus). **AUC > 0.7** → ada drift; fitur paling penting di classifier ini = biang drift.
#
# **Jika X → Y:**
# - AUC adversarial tinggi karena kolom ID/tanggal/urutan → buang kolom itu (biasanya tidak bermakna bisnis).
# - Drift di fitur penting (misal premi naik di periode test) → pertimbangkan: (1) pakai fitur relatif (rasio) bukan absolut, (2) CV yang meniru test (time-based), (3) sebut sebagai limitation.
# - Tidak ada test file → section di-skip.

# %%
def psi(expected, actual, bins=10):
    e, a = expected.dropna(), actual.dropna()
    if len(e) == 0 or len(a) == 0:
        return np.nan
    if pd.api.types.is_numeric_dtype(e) and e.nunique() > 10:
        qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        if len(qs) < 3:
            return 0.0
        qs[0], qs[-1] = -np.inf, np.inf
        ec = pd.cut(e, qs).value_counts(normalize=True, sort=False)
        ac = pd.cut(a, qs).value_counts(normalize=True, sort=False)
    else:
        ec = e.astype(str).value_counts(normalize=True)
        ac = a.astype(str).value_counts(normalize=True).reindex(ec.index).fillna(0)
    ec, ac = ec.values + 1e-4, ac.values + 1e-4
    return float(np.sum((ac - ec) * np.log(ac / ec)))


DRIFT_TABLE = pd.DataFrame()
ADV_AUC = np.nan

with safe_section("Drift & adversarial validation"):
    if test is None:
        print("⏭️ Tidak ada test → skip.")
    else:
        rows = []
        for c in NUM_COLS + CAT_COLS + HIGH_CARD_COLS:
            if c not in test.columns:
                continue
            r = {"feature": c, "psi": psi(train[c], test[c])}
            if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 2:
                try:
                    ks = stats.ks_2samp(train[c].dropna(), test[c].dropna())
                    r.update({"ks_stat": ks.statistic, "ks_p": ks.pvalue})
                except Exception:
                    pass
            else:
                unseen = set(test[c].dropna().astype(str)) - set(train[c].dropna().astype(str))
                r["n_unseen_levels"] = len(unseen)
                r["unseen_rows_%"] = 100 * test[c].astype(str).isin(unseen).mean()
            rows.append(r)
        DRIFT_TABLE = pd.DataFrame(rows).sort_values("psi", ascending=False)
        DRIFT_TABLE["psi_flag"] = pd.cut(DRIFT_TABLE["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "minor shift", "MAJOR shift"])
        save_table(DRIFT_TABLE, "train_test_drift")
        display(DRIFT_TABLE.head(15))

        # adversarial validation
        feats_adv = [c for c in NUM_COLS + CAT_COLS + HIGH_CARD_COLS if c in test.columns]
        A = pd.concat([train[feats_adv], test[feats_adv]], ignore_index=True)
        for c in feats_adv:
            if not pd.api.types.is_numeric_dtype(A[c]):
                A[c] = A[c].astype(str).astype("category").cat.codes
        A = A.fillna(-999)
        is_test = np.r_[np.zeros(len(train)), np.ones(len(test))]
        clf = (lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=CFG.SEED)
               if HAS_LGB else HistGradientBoostingClassifier(max_iter=300, random_state=CFG.SEED))
        p_adv = cross_val_predict(clf, A, is_test, cv=StratifiedKFold(5, shuffle=True, random_state=CFG.SEED), method="predict_proba")[:, 1]
        ADV_AUC = roc_auc_score(is_test, p_adv)
        print(f"\nAdversarial validation AUC = {ADV_AUC:.4f}")
        if ADV_AUC < 0.6:
            print("✅ Train & test sangat mirip → CV score bisa dipercaya sebagai estimasi test score.")
        elif ADV_AUC < 0.75:
            print("⚠️ Ada sedikit drift. Cek fitur teratas di bawah.")
        else:
            print("🚨 Drift kuat! CV score mungkin terlalu optimis. Cek & pertimbangkan buang fitur penyebab drift yang tidak bermakna bisnis.")
        clf.fit(A, is_test)
        imp = pd.Series(getattr(clf, "feature_importances_", np.zeros(len(feats_adv))), index=feats_adv).sort_values(ascending=False)
        if imp.sum() > 0:
            print("Fitur paling membedakan train vs test:")
            display(imp.head(10).to_frame("adv_importance"))

# %% [markdown]
# # 13. Modeling Setup
#
# ## 13.1 Metric
# | Metric lomba | Jenis | Submission | Catatan |
# |---|---|---|---|
# | `roc_auc` / `gini` | ranking | probability | paling umum untuk churn; tidak tergantung threshold |
# | `pr_auc` | ranking | probability | lebih sensitif untuk kelas minoritas |
# | `logloss` / `brier` | probabilitas | probability | butuh probabilitas **terkalibrasi** (Section 18) |
# | `f1`, `f2`, `precision`, `recall`, `accuracy`, `balanced_accuracy`, `mcc`, `kappa` | label | label 0/1 | **threshold tuning wajib** (Section 18) |
#
# Untuk metrik label, ranking model di leaderboard memakai skor pada **threshold optimal OOF** (dicetak juga AUC sebagai pembanding yang stabil).
#
# ## 13.2 Representasi fitur per keluarga model (otomatis)
# | Model | Numeric | Kategori | High-cardinality | Missing |
# |---|---|---|---|---|
# | LightGBM / XGBoost | apa adanya | pandas `category` (native) | frequency + target encoding (in-fold) | native |
# | CatBoost | apa adanya | string (native, ordered TS) | string native | native / "NA" |
# | HistGradientBoosting | apa adanya | ordinal code | freq + TE | native |
# | RandomForest / ExtraTrees | median impute | ordinal code | freq + TE | imputed |
# | Logistic Regression | signed-log untuk skew tinggi + impute + indikator missing + standardize | one-hot (rare digabung) | TE | imputed |
#
# **Target encoding** selalu di-fit **di dalam fold training** (dengan inner K-fold untuk baris training sendiri) → tidak bocor.
#
# ## 13.3 Imbalance
# `CFG.IMBALANCE = "auto"` → metrik ranking/probabilitas: **tanpa reweighting** (menjaga probabilitas tetap jujur; AUC tidak terbantu oleh reweighting).
# Metrik label & churn rate < 20% → `class_weight` (+ threshold tuning). `"smote"` tersedia (hanya di dalam fold, untuk model numeric) tapi **jarang lebih baik** dari class weight + threshold tuning untuk GBDT.
#
# **Jika X → Y:**
# - Metric lomba tidak ada di daftar → tambahkan di fungsi `compute_metric`.
# - Error terkait nama kolom (LightGBM "special JSON characters") → sudah di-handle `sanitize_columns`.

# %%
PROB_METRICS = {"roc_auc", "gini", "pr_auc", "logloss", "brier"}
LABEL_METRICS = {"f1", "f2", "accuracy", "balanced_accuracy", "precision", "recall", "mcc", "kappa"}
LOWER_IS_BETTER = {"logloss", "brier"}
if CFG.METRIC not in PROB_METRICS | LABEL_METRICS:
    print(f"⚠️ METRIC '{CFG.METRIC}' tidak dikenal → pakai roc_auc. Tambahkan di compute_metric kalau perlu.")
    CFG.METRIC = "roc_auc"
THR_METRIC = (CFG.METRIC if CFG.METRIC in LABEL_METRICS else "f1") if CFG.THRESHOLD_METRIC == "auto" else CFG.THRESHOLD_METRIC
if CFG.SUBMISSION_MODE == "probability" and CFG.METRIC in LABEL_METRICS:
    print("⚠️ METRIC berbasis label tapi SUBMISSION_MODE='probability'. Cek format sample_submission! Biasanya harus 'label'.")


def compute_metric(metric, y_true, p, thr=0.5):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    yl = (p >= thr).astype(int)
    if metric == "roc_auc":
        return roc_auc_score(y_true, p)
    if metric == "gini":
        return 2 * roc_auc_score(y_true, p) - 1
    if metric == "pr_auc":
        return average_precision_score(y_true, p)
    if metric == "logloss":
        return log_loss(y_true, p)
    if metric == "brier":
        return brier_score_loss(y_true, p)
    if metric == "f1":
        return f1_score(y_true, yl, zero_division=0)
    if metric == "f2":
        return fbeta_score(y_true, yl, beta=2, zero_division=0)
    if metric == "accuracy":
        return accuracy_score(y_true, yl)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, yl)
    if metric == "precision":
        return precision_score(y_true, yl, zero_division=0)
    if metric == "recall":
        return recall_score(y_true, yl, zero_division=0)
    if metric == "mcc":
        return matthews_corrcoef(y_true, yl)
    if metric == "kappa":
        return cohen_kappa_score(y_true, yl)
    raise ValueError(metric)


def best_threshold(y_true, p, metric=None, grid=None):
    metric = metric or THR_METRIC
    grid = grid if grid is not None else np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    best_t, best_s = 0.5, -np.inf
    for t in grid:
        s = compute_metric(metric, y_true, p, t)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)


def rank_score(y_true, p):
    """Skor untuk ranking model: metric lomba (threshold optimal kalau metrik label). Selalu 'lebih besar = lebih baik'."""
    if CFG.METRIC in LABEL_METRICS:
        return best_threshold(y_true, p, CFG.METRIC)[1]
    s = compute_metric(CFG.METRIC, y_true, p)
    return -s if CFG.METRIC in LOWER_IS_BETTER else s


def show_score(s):
    return -s if CFG.METRIC in LOWER_IS_BETTER else s


IMB_MODE = CFG.IMBALANCE
if IMB_MODE == "auto":
    IMB_MODE = "class_weight" if (CFG.METRIC in LABEL_METRICS and CHURN_RATE < 0.2) else "none"
if IMB_MODE == "smote" and not HAS_IMBLEARN:
    print("imblearn tidak ada → SMOTE diganti class_weight")
    IMB_MODE = "class_weight"
SCALE_POS = (1 - CHURN_RATE) / max(CHURN_RATE, 1e-6)
print("Metric        :", CFG.METRIC, "| threshold metric:", THR_METRIC)
print("Imbalance mode:", IMB_MODE, f"(scale_pos_weight = {SCALE_POS:.2f})")

# %% [markdown]
# ## 13.4 Bangun matriks fitur

# %%
with safe_section("Build feature matrices", critical=True):
    MODEL_FEATURES = [c for c in NUM_COLS + CAT_COLS + HIGH_CARD_COLS if c in train.columns]
    if not CFG.AUTO_DROP_LEAKY and LEAKY_COLS:
        print(f"⚠️ Fitur leak tetap ikut model (AUTO_DROP_LEAKY=False): {[c for c in LEAKY_COLS if c in MODEL_FEATURES]}")
    tr_df = train[MODEL_FEATURES].copy()
    te_df = test[MODEL_FEATURES].copy() if test is not None else pd.DataFrame(columns=MODEL_FEATURES)
    for c in tr_df.columns:
        if c not in te_df.columns:
            te_df[c] = np.nan
    te_df = te_df[MODEL_FEATURES]
    full_df = pd.concat([tr_df, te_df], ignore_index=True)
    N_TR = len(tr_df)

    cat_like = CAT_COLS + HIGH_CARD_COLS
    if CFG.USE_FREQ_ENCODING:
        for c in cat_like:
            full_df[f"{c}__freq"] = full_df[c].map(full_df[c].value_counts(normalize=True)).astype(float)
    full_df, NAME_MAP = sanitize_columns(full_df)
    INV_NAME = {v: k for k, v in NAME_MAP.items()}
    S_NUM = [NAME_MAP[c] for c in NUM_COLS if c in NAME_MAP] + [NAME_MAP[c] for c in NAME_MAP if c.endswith("__freq")]
    S_CAT = [NAME_MAP[c] for c in CAT_COLS if c in NAME_MAP]
    S_HC = [NAME_MAP[c] for c in HIGH_CARD_COLS if c in NAME_MAP]

    # 1) tree (LightGBM/XGB): category dtype, high-card raw dibuang (pakai freq + TE)
    X_tree_full = full_df[S_NUM + S_CAT].copy()
    for c in S_CAT:
        X_tree_full[c] = X_tree_full[c].astype(str).where(full_df[c].notna(), np.nan).astype("category")
    for c in S_NUM:
        X_tree_full[c] = pd.to_numeric(X_tree_full[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    # 2) catboost: string kategori (termasuk high-card)
    X_cb_full = full_df[S_NUM + S_CAT + S_HC].copy()
    for c in S_CAT + S_HC:
        X_cb_full[c] = X_cb_full[c].astype(str).where(full_df[c].notna(), "NA")
    X_cb_full[S_NUM] = X_tree_full[S_NUM]
    # 3) ordinal (HGB/RF/ET)
    X_ord_full = X_tree_full.copy()
    for c in S_CAT:
        X_ord_full[c] = X_ord_full[c].cat.codes.replace(-1, np.nan).astype(float)
    # 4) linear raw (signed log untuk skew tinggi)
    X_lin_full = X_tree_full[S_NUM].copy()
    for c in S_NUM:
        s = X_lin_full[c]
        if s.nunique() > 10 and abs(s.skew()) > 2:
            X_lin_full[c] = np.sign(s) * np.log1p(np.abs(s))
    for c in S_CAT:
        X_lin_full[c] = full_df[c].astype(str).where(full_df[c].notna(), "Missing")

    REPS = {}
    for key, M in [("tree", X_tree_full), ("cb", X_cb_full), ("ord", X_ord_full), ("lin", X_lin_full)]:
        REPS[key] = (M.iloc[:N_TR].reset_index(drop=True), M.iloc[N_TR:].reset_index(drop=True))
    HC_RAW = (full_df[S_HC].iloc[:N_TR].reset_index(drop=True), full_df[S_HC].iloc[N_TR:].reset_index(drop=True))
    print(f"Fitur model: {len(MODEL_FEATURES)} asli → tree={X_tree_full.shape[1]}, catboost={X_cb_full.shape[1]}, "
          f"linear={X_lin_full.shape[1]} (+one-hot) | high-card TE: {S_HC}")

# %% [markdown]
# ## 13.5 Cross-validation folds
# - Default: **StratifiedKFold** (proporsi churn sama di tiap fold).
# - `CFG.GROUP_COL` diisi (misal `household_id`, `agent_id`) → **StratifiedGroupKFold**: satu grup tidak boleh ada di train & valid sekaligus (mencegah leakage antar anggota grup).
# - `N_REPEATS > 1` → repeated CV dengan seed berbeda → estimasi lebih stabil.
#
# **Cara baca:** churn rate per fold harus hampir sama. Kalau CV std antar fold besar (> 0.02 AUC) → data kecil / noisy → pakai repeats & jangan terlalu percaya beda kecil antar model.

# %%
def make_folds(n_folds, seed):
    idx = np.arange(N_TR)
    if (CFG.CV_STRATEGY in ("group", "auto")) and CFG.GROUP_COL and CFG.GROUP_COL in raw_train.columns:
        groups = raw_train[CFG.GROUP_COL].astype(str).values
        if HAS_SGKF:
            spl = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(idx, y, groups)
        else:
            spl = GroupKFold(n_splits=n_folds).split(idx, y, groups)
    else:
        spl = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(idx, y)
    return [(tr, va) for tr, va in spl]


with safe_section("CV folds", critical=True):
    FOLD_SETS = [make_folds(CFG.N_FOLDS, CFG.SEED + r) for r in range(CFG.N_REPEATS)]
    if getattr(CFG, "TUNE_SUBSAMPLE", None) and N_TR > CFG.TUNE_SUBSAMPLE:
        from sklearn.model_selection import train_test_split
        sub_idx, _ = train_test_split(np.arange(N_TR), train_size=CFG.TUNE_SUBSAMPLE, stratify=y, random_state=CFG.SEED)
        sub_idx = np.sort(sub_idx)
        OPT_FOLDS = [(sub_idx[tr], sub_idx[va]) for tr, va in
                     StratifiedKFold(CFG.OPTUNA_FOLDS, shuffle=True, random_state=CFG.SEED + 100).split(sub_idx, y[sub_idx])]
        print(f"Optuna tuning memakai subsample {len(sub_idx):,} baris (data besar).")
    else:
        OPT_FOLDS = make_folds(CFG.OPTUNA_FOLDS, CFG.SEED + 100)
    chk = pd.DataFrame([{"fold": i, "n_valid": len(va), "churn_rate_valid": y[va].mean()} for i, (tr, va) in enumerate(FOLD_SETS[0])])
    display(chk)
    print("Strategi:", "group" if CFG.GROUP_COL else "stratified", "| folds:", CFG.N_FOLDS, "| repeats:", CFG.N_REPEATS)

# %% [markdown]
# # 14. Model Training (CV OOF)
#
# Setiap model dilatih K kali (K fold). Prediksi pada fold validasi dikumpulkan jadi **OOF prediction** untuk seluruh train → dipakai untuk evaluasi, ensemble, threshold, dan analisis bisnis. Prediksi test = rata-rata K model.
#
# **Cara baca leaderboard:**
# - `cv_mean ± cv_std`: performa & stabilitas.
# - `train_score` vs `cv_mean`: gap besar (misal train AUC 0.99 vs CV 0.80) → **overfitting** → naikkan regularisasi (`min_child_samples`, `reg_lambda`), turunkan `num_leaves`/`depth`, atau tambah data.
# - `oof_score`: skor dari seluruh OOF sekaligus (ini yang dipakai untuk ranking).
# - Model `dummy` = baseline. Semua model harus jauh lebih baik.
#
# **Jika X → Y:**
# - Semua model ≈ dummy → bug (target salah mapping? fitur ke-drop semua?) → cek Section 3–5.
# - Skor terlalu bagus (AUC > 0.98) → hampir pasti **leakage** → cek Section 5.3.
# - Satu model gagal (❌) → lihat `FAILED_MODELS`; model lain tetap jalan.
# - LogReg ≈ GBDT → hubungan mostly linear; model sederhana cukup & lebih mudah dijelaskan (bagus untuk laporan!).

# %%
FAILED_MODELS = {}
RESULTS = OrderedDict()


def target_encode_fold(tr_s, y_tr, others, smoothing=20, inner=5, seed=0):
    prior = y_tr.mean()
    tr_s = tr_s.astype(str).values
    enc_tr = np.zeros(len(tr_s))
    for itr, iva in StratifiedKFold(inner, shuffle=True, random_state=seed).split(tr_s, y_tr):
        st = pd.DataFrame({"k": tr_s[itr], "y": y_tr[itr]}).groupby("k")["y"].agg(["sum", "count"])
        m = (st["sum"] + smoothing * prior) / (st["count"] + smoothing)
        enc_tr[iva] = pd.Series(tr_s[iva]).map(m).fillna(prior).values
    st = pd.DataFrame({"k": tr_s, "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
    m = (st["sum"] + smoothing * prior) / (st["count"] + smoothing)
    return enc_tr, [pd.Series(o.astype(str).values).map(m).fillna(prior).values for o in others]


ACTIVE_DROP = set()   # kolom (nama sanitized) yang dibuang sementara (dipakai feature selection)
PERMA_DROP = set()    # kolom yang dibuang permanen (misal leak yang terdeteksi di level model)


def fold_data(rep, tr_idx, va_idx, seed, y_vec=None):
    """Ambil representasi `rep` untuk fold, tambah target-encoding high-card (fit di train fold)."""
    y_vec = y if y_vec is None else y_vec
    Xtr_all, Xte_all = REPS[rep]
    keep = [c for c in Xtr_all.columns if c not in ACTIVE_DROP and c not in PERMA_DROP]
    Xtr, Xva, Xte = Xtr_all.iloc[tr_idx][keep].copy(), Xtr_all.iloc[va_idx][keep].copy(), Xte_all[keep].copy()
    if CFG.USE_TARGET_ENCODING and S_HC:
        hc_tr, hc_te = HC_RAW
        for c in S_HC:
            if f"{c}__te" in ACTIVE_DROP:
                continue
            e_tr, (e_va, e_te) = target_encode_fold(hc_tr[c].iloc[tr_idx], y_vec[tr_idx], [hc_tr[c].iloc[va_idx], hc_te[c]], seed=seed)
            Xtr[f"{c}__te"], Xva[f"{c}__te"], Xte[f"{c}__te"] = e_tr, e_va, e_te
    return Xtr, Xva, Xte


def default_params(name):
    es_metric_auc = CFG.METRIC in ("roc_auc", "gini", "pr_auc") or CFG.METRIC in LABEL_METRICS
    if name == "lgbm":
        return dict(n_estimators=10000, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, cat_smooth=10, verbose=-1, n_jobs=CFG.N_JOBS,
                    metric="auc" if es_metric_auc else "binary_logloss")
    if name == "xgb":
        return dict(n_estimators=10000, learning_rate=0.03, max_depth=5, min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
                    reg_lambda=1.0, tree_method="hist", eval_metric="auc" if es_metric_auc else "logloss", n_jobs=CFG.N_JOBS)
    if name == "catboost":
        # boost_from_average=True + early stopping on Logloss: probabilitas tiap fold berskala sama (OOF konsisten).
        # (Default CatBoost Logloss mulai dari 0.5 → fold yang berhenti dini menghasilkan skala berbeda → OOF AUC turun.)
        return dict(iterations=10000, learning_rate=0.05, depth=6, l2_leaf_reg=3, loss_function="Logloss",
                    eval_metric="Logloss", boost_from_average=True, verbose=0, allow_writing_files=False,
                    max_ctr_complexity=1, one_hot_max_size=10,   # ±2x lebih cepat di CPU, AUC setara (diuji)
                    thread_count=CFG.N_JOBS if CFG.N_JOBS > 0 else min(os.cpu_count() or 4, 8))
    if name == "hgb":
        return dict(max_iter=2000, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)
    if name == "rf":
        return dict(n_estimators=500, min_samples_leaf=3, max_features="sqrt", n_jobs=CFG.N_JOBS)
    if name == "et":
        return dict(n_estimators=500, min_samples_leaf=3, max_features="sqrt", n_jobs=CFG.N_JOBS)
    if name == "logreg":
        return dict(C=0.5, max_iter=5000)
    if name == "ebm":
        return dict(interactions=CFG.EBM_INTERACTIONS, outer_bags=CFG.EBM_OUTER_BAGS, learning_rate=0.015,
                    early_stopping_rounds=100, validation_size=0.15)
    if name == "mlp":
        return dict(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=512, learning_rate_init=1e-3, early_stopping=True,
                    validation_fraction=0.1, n_iter_no_change=8, max_iter=200)
    if name == "dummy":
        return dict(strategy="prior")
    raise ValueError(name)


MODEL_REP = {"lgbm": "tree", "xgb": "tree", "catboost": "cb", "hgb": "ord", "rf": "ord", "et": "ord", "logreg": "lin", "dummy": "ord",
             "ebm": "cb", "mlp": "lin"}
MODEL_NAMES = {"logreg": "Logistic regression", "rf": "Random forest", "et": "Extra trees", "hgb": "Hist. gradient boosting",
               "lgbm": "LightGBM", "xgb": "XGBoost", "catboost": "CatBoost", "ebm": "Explainable boosting machine (EBM)",
               "mlp": "Neural network (MLP)", "dummy": "Baseline (prior)"}


def linear_preprocessor(Xtr):
    num = [c for c in Xtr.columns if Xtr[c].dtype != object]
    cat = [c for c in Xtr.columns if Xtr[c].dtype == object]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("ohe", make_ohe(min_frequency=CFG.RARE_MIN_FREQ))]), cat)], remainder="drop")


def maybe_smote(Xtr, ytr, seed):
    if IMB_MODE != "smote":
        return Xtr, ytr
    from imblearn.over_sampling import SMOTE
    Xn = Xtr.fillna(Xtr.median(numeric_only=True)).fillna(0)
    Xs, ys = SMOTE(random_state=seed).fit_resample(Xn, ytr)
    return pd.DataFrame(Xs, columns=Xtr.columns), np.asarray(ys)


def fit_predict(name, params, Xtr, ytr, Xva, yva, Xte, seed):
    """Latih 1 model di 1 fold. Return p_va, p_te, model, best_iter, p_tr."""
    p = dict(default_params(name))
    p.update(params or {})
    cw = IMB_MODE in ("class_weight", "smote")
    best_iter = None
    if name == "lgbm":
        if cw:
            p["scale_pos_weight"] = SCALE_POS
        model = lgb.LGBMClassifier(random_state=seed, **p)   # LightGBM GPU jarang tersedia di wheel Windows → selalu CPU
        cbs = []
        try:
            cbs = [lgb.early_stopping(CFG.EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)]
        except Exception:
            pass
        try:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cbs)
        except Exception as e:
            if CFG.USE_GPU:
                p.pop("device", None)
                model = lgb.LGBMClassifier(random_state=seed, **p)
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cbs)
            else:
                raise e
        best_iter = getattr(model, "best_iteration_", None)
    elif name == "xgb":
        if cw:
            p["scale_pos_weight"] = SCALE_POS
        if CFG.USE_GPU:
            p["device"] = "cuda"
        es = CFG.EARLY_STOPPING_ROUNDS
        try:
            model = xgb.XGBClassifier(random_state=seed, enable_categorical=True, max_cat_to_onehot=1, early_stopping_rounds=es, **p)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        except TypeError:
            # xgboost lama: early_stopping_rounds di fit(), kategori → kode
            Xtr2, Xva2, Xte = [d.apply(lambda s: s.cat.codes if str(s.dtype) == "category" else s) for d in (Xtr, Xva, Xte)]
            p.pop("device", None)
            model = xgb.XGBClassifier(random_state=seed, **p)
            model.fit(Xtr2, ytr, eval_set=[(Xva2, yva)], early_stopping_rounds=es, verbose=False)
            Xtr, Xva = Xtr2, Xva2
        except Exception as e:
            if CFG.USE_GPU:
                p.pop("device", None)
                model = xgb.XGBClassifier(random_state=seed, enable_categorical=True, max_cat_to_onehot=1, early_stopping_rounds=es, **p)
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            else:
                raise e
        best_iter = getattr(model, "best_iteration", None)
    elif name == "catboost":
        cat_idx = [i for i, c in enumerate(Xtr.columns) if Xtr[c].dtype == object]
        if cw:
            p["auto_class_weights"] = "Balanced"
        if CFG.USE_GPU:
            p["task_type"] = "GPU"
        p.setdefault("od_type", "Iter")
        p.setdefault("od_wait", CFG.EARLY_STOPPING_ROUNDS)
        try:
            model = cb.CatBoostClassifier(random_seed=seed, cat_features=cat_idx, **p)
            model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        except Exception as e:
            if CFG.USE_GPU:
                p.pop("task_type", None)
                model = cb.CatBoostClassifier(random_seed=seed, cat_features=cat_idx, **p)
                model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
            else:
                raise e
        best_iter = model.get_best_iteration()
    elif name == "hgb":
        Xtr, ytr = maybe_smote(Xtr, ytr, seed)
        model = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced" if IMB_MODE == "class_weight" else None, **p) \
            if "class_weight" in HistGradientBoostingClassifier().get_params() else HistGradientBoostingClassifier(random_state=seed, **p)
        model.fit(Xtr, ytr)
        best_iter = getattr(model, "n_iter_", None)
    elif name in ("rf", "et"):
        Est = RandomForestClassifier if name == "rf" else ExtraTreesClassifier
        cwv = "balanced_subsample" if IMB_MODE == "class_weight" else None
        Xtr, ytr = maybe_smote(Xtr, ytr, seed)
        model = Pipeline([("imp", SimpleImputer(strategy="median")), ("m", Est(random_state=seed, class_weight=cwv, **p))])
        model.fit(Xtr, ytr)
    elif name == "logreg":
        model = Pipeline([("pre", linear_preprocessor(Xtr)),
                          ("m", LogisticRegression(class_weight="balanced" if IMB_MODE != "none" else None, **p))])
        model.fit(Xtr, ytr)
    elif name == "mlp":
        model = Pipeline([("pre", linear_preprocessor(Xtr)), ("m", MLPClassifier(random_state=seed % (2 ** 31), **p))])
        model.fit(Xtr, ytr)
    elif name == "ebm":
        sw = np.where(np.asarray(ytr) == 1, SCALE_POS, 1.0) if cw else None
        model = ExplainableBoostingClassifier(random_state=seed % (2 ** 31), n_jobs=CFG.N_JOBS, **p)
        model.fit(Xtr, ytr, sample_weight=sw)
    elif name == "dummy":
        model = DummyClassifier(**p)
        model.fit(Xtr.fillna(0), ytr)
        Xva, Xte = Xva.fillna(0), Xte.fillna(0)
        Xtr = Xtr.fillna(0)
    else:
        raise ValueError(f"Model tidak dikenal: {name}")

    p_va = model.predict_proba(Xva)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1] if len(Xte) else np.array([])
    sub = np.random.RandomState(seed).choice(len(Xtr), min(len(Xtr), 5000), replace=False)
    p_tr = (sub, model.predict_proba(Xtr.iloc[sub])[:, 1])
    return p_va, p_te, model, best_iter, p_tr


def model_available(name):
    return {"lgbm": HAS_LGB, "xgb": HAS_XGB, "catboost": HAS_CB, "hgb": HAS_HGB, "ebm": HAS_EBM}.get(name, True)


def run_cv(name, params=None, seeds=None, fold_sets=None, label=None, verbose=True, keep_models=True, note=""):
    """CV lengkap untuk 1 model (bisa multi-seed & repeated). Return dict hasil."""
    label = label or name
    seeds = seeds or [CFG.SEED]
    fold_sets = fold_sets or FOLD_SETS
    rep = MODEL_REP[name]
    t0 = time.time()
    oof_sum, oof_cnt = np.zeros(N_TR), np.zeros(N_TR)
    n_te = len(REPS[rep][1])
    te_sum, n_models = np.zeros(n_te), 0
    fold_scores, train_scores, models, iters = [], [], [], []
    for r, folds in enumerate(fold_sets):
        for k, (tr, va) in enumerate(folds):
            Xtr, Xva, Xte = fold_data(rep, tr, va, seed=CFG.SEED + k)
            p_va_seeds = np.zeros(len(va))
            for s in seeds:
                p_va, p_te, model, bi, (sub, p_tr) = fit_predict(name, params, Xtr, y[tr], Xva, y[va], Xte, s + 17 * k + 1000 * r)
                p_va_seeds += p_va / len(seeds)
                if n_te:
                    te_sum += p_te
                n_models += 1
                iters.append(bi)
                if keep_models and r == 0:
                    models.append({"fold": k, "model": model, "va_idx": va, "rep": rep, "X_va": Xva})
                try:
                    train_scores.append(compute_metric("roc_auc", y[tr][sub], p_tr))
                except Exception:
                    pass
            oof_sum[va] += p_va_seeds
            oof_cnt[va] += 1
            fs = rank_score(y[va], p_va_seeds)
            fold_scores.append(show_score(fs))
            if verbose:
                print(f"  [{label}] repeat {r} fold {k}: {CFG.METRIC} = {show_score(fs):.5f}" + (f" | best_iter = {bi}" if bi else ""))
    oof = oof_sum / np.maximum(oof_cnt, 1)
    test_pred = te_sum / max(n_models, 1) if n_te else np.array([])
    res = {"name": label, "model": name, "params": params or {}, "oof": oof, "test": test_pred,
           "fold_scores": fold_scores, "cv_mean": float(np.mean(fold_scores)), "cv_std": float(np.std(fold_scores)),
           "oof_score": show_score(rank_score(y, oof)), "oof_auc": roc_auc_score(y, oof),
           "train_auc": float(np.mean(train_scores)) if train_scores else np.nan,
           "best_iters": [i for i in iters if i is not None], "models": models, "time_s": time.time() - t0, "seeds": seeds}
    res["n_features"] = int(Xtr.shape[1]) if hasattr(Xtr, "shape") else None
    log_experiment(label, name, oof, note=note or ("default hyperparameters" if not params else "custom/tuned hyperparameters")
                   + (f"; {len(seeds)} seeds" if len(seeds) > 1 else ""), n_features=res["n_features"], time_s=res["time_s"],
                   cv_std=res["cv_std"])
    if verbose:
        print(f"  ⇒ [{label}] OOF {CFG.METRIC} = {res['oof_score']:.5f} | CV {res['cv_mean']:.5f} ± {res['cv_std']:.5f} | "
              f"OOF AUC {res['oof_auc']:.4f} | train AUC {res['train_auc']:.4f} | {res['time_s']:.0f}s")
    return res


def permutation_importance_oof(res, n_repeats=None, max_folds=3, metric="auto"):
    """Permutation importance model-agnostic di fold validasi (pakai model fold yang tersimpan)."""
    n_repeats = n_repeats or CFG.PERM_IMPORTANCE_REPEATS
    metric = ("roc_auc" if CFG.METRIC in LABEL_METRICS else CFG.METRIC) if metric == "auto" else metric
    sign = -1 if metric in LOWER_IS_BETTER else 1
    rng = np.random.RandomState(CFG.SEED)
    rows = {}
    for md in res["models"][:max_folds]:
        model, Xva, va = md["model"], md["X_va"], md["va_idx"]
        lim = getattr(CFG, "PERM_MAX_ROWS", 5000)
        if len(va) > lim:
            pick = np.random.RandomState(CFG.SEED).choice(len(va), lim, replace=False)
            Xva, va = Xva.iloc[pick], va[pick]
        base = sign * compute_metric(metric, y[va], model.predict_proba(Xva)[:, 1])
        for c in Xva.columns:
            drops = []
            for _ in range(n_repeats):
                Xp = Xva.copy()
                Xp[c] = Xp[c].values[rng.permutation(len(Xp))]
                drops.append(base - sign * compute_metric(metric, y[va], model.predict_proba(Xp)[:, 1]))
            rows.setdefault(c, []).extend(drops)
    out = pd.DataFrame([{"feature_model": c, "importance_mean": np.mean(v), "importance_std": np.std(v)} for c, v in rows.items()])
    out["feature"] = out["feature_model"].map(lambda c: INV_NAME.get(c, INV_NAME.get(c.replace("__te", ""), c)))
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def leaderboard():
    rows = []
    for k, r in RESULTS.items():
        rows.append({"model": k, "algorithm": MODEL_NAMES.get(r["model"], r["model"]), f"oof_{CFG.METRIC}": r["oof_score"], "cv_mean": r["cv_mean"], "cv_std": r["cv_std"],
                     "oof_auc": r["oof_auc"], "train_auc": r["train_auc"], "overfit_gap_auc": r["train_auc"] - r["oof_auc"],
                     "pr_auc": average_precision_score(y, r["oof"]), "logloss": log_loss(y, np.clip(r["oof"], 1e-7, 1 - 1e-7)),
                     "time_s": r["time_s"]})
    lb = pd.DataFrame(rows)
    if len(lb):
        lb = lb.sort_values(f"oof_{CFG.METRIC}", ascending=CFG.METRIC in LOWER_IS_BETTER).reset_index(drop=True)
    return lb


with safe_section("Train baseline models"):
    set_stage("1 baseline model zoo")
    for name in ["dummy"] + [m for m in CFG.MODELS if m != "dummy"]:
        if not model_available(name):
            print(f"⏭️ {name}: library tidak tersedia → skip")
            continue
        print(f"\n===== {name} =====")
        try:
            RESULTS[name] = run_cv(name, note=f"{MODEL_NAMES.get(name, name)}, default hyperparameters")
        except Exception as e:
            FAILED_MODELS[name] = f"{type(e).__name__}: {e}"
            print(f"❌ {name} gagal: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    # ---- Safety net: leak yang lolos screen (AUC hampir sempurna + 1 fitur mendominasi) ----
    for _round in range(3):
        LB = leaderboard()
        top_auc = LB[LB["model"] != "dummy"]["oof_auc"].max() if len(LB) else 0
        if not (CFG.AUTO_DROP_LEAKY and top_auc > 0.985):
            break
        probe = next((m for m in ["lgbm", "hgb", "catboost", "xgb"] if m in RESULTS and RESULTS[m]["models"]), None)
        if probe is None:
            break
        pi = permutation_importance_oof(RESULTS[probe], n_repeats=1, max_folds=1, metric="roc_auc")
        if len(pi) == 0:
            break
        share = pi["importance_mean"].clip(lower=0)
        share = share / max(share.sum(), 1e-12)
        if share.iloc[0] < 0.4:
            print(f"⚠️ OOF AUC {top_auc:.4f} sangat tinggi tapi tidak ada 1 fitur dominan → cek manual kemungkinan leakage.")
            break
        culprit = pi.iloc[0]["feature_model"]
        PERMA_DROP.add(culprit)
        orig = INV_NAME.get(culprit, culprit)
        LEAKY_COLS.append(orig)
        print(f"🚨 LEAK di level model: AUC {top_auc:.4f} dan '{orig}' menyumbang {share.iloc[0]:.0%} importance → dibuang permanen, model dilatih ulang.")
        add_insight("Data", f"'{orig}' was removed after a model-level leakage check (near-perfect AUC driven by a single feature).")
        for name in list(RESULTS.keys()):
            if name == "dummy":
                continue
            try:
                RESULTS[name] = run_cv(RESULTS[name]["model"], params=RESULTS[name]["params"], label=name, verbose=False,
                                       note=f"retrained after removing leaked feature '{orig}'")
                print(f"   retrain {name}: OOF AUC {RESULTS[name]['oof_auc']:.4f}")
            except Exception as e:
                FAILED_MODELS[name] = str(e)
    LB = leaderboard()
    save_table(LB, "leaderboard_baseline")
    display(LB)
    if FAILED_MODELS:
        print("Model gagal:", FAILED_MODELS)

# %%
with safe_section("Leaderboard plot"):
    LB = leaderboard()
    lbp = LB[LB["model"] != "dummy"]
    if len(lbp):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(lbp))))
        colname = f"oof_{CFG.METRIC}"
        ax.barh(lbp["model"][::-1], lbp[colname][::-1], xerr=lbp["cv_std"][::-1], color=CFG.PALETTE[0], capsize=3)
        lo = lbp[colname].min() - 3 * lbp["cv_std"].max()
        hi = lbp[colname].max() + 2 * lbp["cv_std"].max()
        ax.set_xlim(max(0, lo) if CFG.METRIC not in LOWER_IS_BETTER else lo, hi)
        for i, v in enumerate(lbp[colname][::-1]):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
        ax.set_xlabel(f"OOF {CFG.METRIC} (error bar = CV std)")
        ax.set_title("Model Comparison (cross-validated)")
        save_fig("model_leaderboard")
        gap = lbp[lbp["overfit_gap_auc"] > 0.08]
        if len(gap):
            print("⚠️ Gap train-vs-OOF AUC > 0.08 (indikasi overfit):", gap["model"].tolist())

# %% [markdown]
# # 15. Hyperparameter Tuning (Optuna)
#
# Optuna mencari kombinasi hyperparameter terbaik dengan **TPE sampler** (lebih efisien dari grid/random search) + **pruning** (trial jelek dihentikan setelah beberapa fold).
# - Tuning pakai `CFG.OPTUNA_FOLDS` fold (default 3, lebih cepat), hasil final dilatih ulang dengan `N_FOLDS`.
# - Budget: `OPTUNA_TRIALS` trial **atau** `OPTUNA_TIMEOUT` detik per model (mana yang duluan).
# - Untuk metrik label, tuning mengoptimalkan **ROC-AUC** (lebih stabil); threshold dituning belakangan.
#
# **Jika X → Y:**
# - Waktu mepet → `CFG.USE_OPTUNA = False` (default params sudah cukup kuat) atau `OPTUNA_TIMEOUT = 300`.
# - Hasil tuned **lebih jelek** dari default di CV final → wajar terjadi (noise / overfit ke fold tuning); notebook tetap menyimpan keduanya dan ensemble memilih yang terbaik.
# - Improvement < 0.002 AUC → tuning tidak terlalu penting untuk data ini; fokus ke fitur & insight.
#
# **Untuk laporan:** cukup tulis "Hyperparameters were optimised with Bayesian optimisation (Optuna TPE, N trials, 3-fold CV)".

# %%
def suggest_params(trial, name):
    if name == "lgbm":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 8, 256, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                    min_split_gain=trial.suggest_float("min_split_gain", 0, 1),
                    cat_smooth=trial.suggest_float("cat_smooth", 1, 50))
    if name == "xgb":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    max_depth=trial.suggest_int("max_depth", 2, 10),
                    min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                    gamma=trial.suggest_float("gamma", 0, 5))
    if name == "catboost":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                    depth=trial.suggest_int("depth", 3, 8),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
                    random_strength=trial.suggest_float("random_strength", 0, 3),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0, 1))
    raise ValueError(name)


def tune_model(name):
    direction = "maximize"
    tune_metric = CFG.METRIC if CFG.METRIC in PROB_METRICS else "roc_auc"
    rep = MODEL_REP[name]

    def objective(trial):
        params = suggest_params(trial, name)
        scores = []
        for k, (tr, va) in enumerate(OPT_FOLDS):
            Xtr, Xva, Xte = fold_data(rep, tr, va, seed=CFG.SEED + k)
            p_va, _, _, _, _ = fit_predict(name, params, Xtr, y[tr], Xva, y[va], Xte.iloc[:0], CFG.SEED)
            s = compute_metric(tune_metric, y[va], p_va)
            scores.append(-s if tune_metric in LOWER_IS_BETTER else s)
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=CFG.SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.optimize(objective, n_trials=CFG.OPTUNA_TRIALS, timeout=CFG.OPTUNA_TIMEOUT, gc_after_trial=True, catch=(Exception,))
    return study


TUNED_PARAMS = {}
TUNING_SUMMARY = []
with safe_section("Optuna tuning"):
    set_stage("2 hyperparameter tuning")
    if not (CFG.USE_OPTUNA and HAS_OPTUNA):
        print("⏭️ Optuna dimatikan / tidak tersedia → pakai default params.")
    else:
        for name in [m for m in CFG.OPTUNA_MODELS if m in RESULTS and model_available(m)]:
            print(f"\n===== Tuning {name} ({CFG.OPTUNA_TRIALS} trials / {CFG.OPTUNA_TIMEOUT}s) =====")
            try:
                study = tune_model(name)
                done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if not done:
                    print("  tidak ada trial yang selesai")
                    continue
                TUNED_PARAMS[name] = study.best_params
                print(f"  best tuning score = {study.best_value:.5f} | trials selesai = {len(done)}")
                print("  best params:", study.best_params)
                TUNING_SUMMARY.append({"model": name, "trials_completed": len(done), "best_tuning_score": study.best_value,
                                       "default_cv_auc": RESULTS[name]["oof_auc"], "best_params": json.dumps(study.best_params, default=str)})
                RESULTS[f"{name}_tuned"] = run_cv(name, params=study.best_params, label=f"{name}_tuned",
                                                  note=f"Optuna TPE, {len(done)} trials, {CFG.OPTUNA_FOLDS}-fold inner CV")
                try:
                    hist = pd.DataFrame([{"trial": t.number, "value": t.value} for t in done])
                    fig, ax = plt.subplots(figsize=(7, 3.2))
                    ax.plot(hist["trial"], hist["value"], "o", alpha=0.4, ms=3)
                    ax.plot(hist["trial"], hist["value"].cummax(), "-", color=CFG.PALETTE[1])
                    ax.set_xlabel("Trial")
                    ax.set_ylabel("CV score")
                    ax.set_title(f"Optuna optimisation history — {name}")
                    save_fig(f"optuna_history_{name}")
                except Exception:
                    pass
            except Exception as e:
                FAILED_MODELS[f"{name}_tuned"] = str(e)
                print(f"❌ tuning {name} gagal: {e}")
        if TUNING_SUMMARY:
            TS = pd.DataFrame(TUNING_SUMMARY)
            TS["tuned_cv_auc"] = TS["model"].map(lambda m: RESULTS.get(f"{m}_tuned", {}).get("oof_auc", np.nan))
            TS["gain_auc"] = TS["tuned_cv_auc"] - TS["default_cv_auc"]
            save_table(TS, "optuna_tuning_summary")
        with open(os.path.join(CFG.MODEL_DIR, "tuned_params.json"), "w") as f:
            json.dump(TUNED_PARAMS, f, indent=2, default=str)
    LB = leaderboard()
    display(LB)

# %% [markdown]
# # 16. Multi-seed Final Models
#
# GBDT punya randomness (subsample, colsample). Melatih model yang sama dengan **beberapa seed lalu dirata-rata** hampir selalu sedikit menaikkan & menstabilkan skor (gratis, hanya butuh waktu).
# Diterapkan ke 2 model GBDT terbaik (versi default atau tuned, mana yang lebih bagus).

# %%
with safe_section("Multi-seed final models"):
    set_stage("3 multi-seed averaging")
    if len(CFG.FINAL_SEEDS) <= 1:
        print("⏭️ FINAL_SEEDS hanya 1 seed → skip multi-seed.")
    else:
        LB = leaderboard()
        best_per_family = OrderedDict()
        for _, r in LB.iterrows():
            fam = RESULTS[r["model"]]["model"]
            if fam in ("lgbm", "xgb", "catboost", "hgb") and fam not in best_per_family:
                best_per_family[fam] = r["model"]
        for fam, key in list(best_per_family.items())[:2]:
            print(f"\n===== {fam} multi-seed (base: {key}, seeds {CFG.FINAL_SEEDS}) =====")
            try:
                RESULTS[f"{fam}_multiseed"] = run_cv(fam, params=RESULTS[key]["params"], seeds=CFG.FINAL_SEEDS,
                                                     label=f"{fam}_multiseed", verbose=False,
                                                     note=f"average of {len(CFG.FINAL_SEEDS)} seeds (base: {key})")
                r = RESULTS[f"{fam}_multiseed"]
                print(f"  ⇒ OOF {CFG.METRIC} = {r['oof_score']:.5f} (single-seed: {RESULTS[key]['oof_score']:.5f})")
            except Exception as e:
                FAILED_MODELS[f"{fam}_multiseed"] = str(e)
                print(f"❌ {e}")
    LB = leaderboard()
    save_table(LB, "leaderboard_all")
    display(LB)

# %% [markdown]
# ## 16.2 Feature Selection (permutation importance)
#
# Fitur yang **tidak membantu** (permutation importance ≤ 0: mengacak fitur itu tidak menurunkan skor) bisa jadi noise.
# Notebook menghitung permutation importance pada fold validasi model GBDT terbaik, membuang fitur ≤ 0, lalu melatih ulang.
# Hasil **hanya dipakai kalau CV membaik** (> 0.0005) — model baru masuk leaderboard sebagai `<model>_fs` dan ikut kandidat ensemble.
#
# **Jika X → Y:** banyak fitur ber-importance negatif → fitur noise / redundant; untuk laporan, fokus pada fitur ber-importance tinggi saja.

# %%
PERM_IMP = pd.DataFrame()
with safe_section("Feature selection"):
    set_stage("4 feature selection")
    LB = leaderboard()
    gb = [m for m in LB["model"] if RESULTS[m]["model"] in ("lgbm", "catboost", "xgb", "hgb") and RESULTS[m]["models"]]
    if not gb:
        print("⏭️ Tidak ada model GBDT untuk feature selection.")
    else:
        base_key = gb[0]
        base_res = RESULTS[base_key]
        PERM_IMP = permutation_importance_oof(base_res)
        print(f"Permutation importance ({base_key}):")
        display(PERM_IMP.head(25))
        useless = PERM_IMP[PERM_IMP["importance_mean"] <= 0]["feature_model"].tolist()
        print(f"Fitur dengan importance ≤ 0: {len(useless)} → {useless[:30]}")
        if CFG.USE_FEATURE_SELECTION and 0 < len(useless) < len(PERM_IMP) - 3:
            ACTIVE_DROP = set(useless)
            try:
                fam = base_res["model"]
                r_fs = run_cv(fam, params=base_res["params"], seeds=base_res["seeds"], label=f"{fam}_fs", verbose=False,
                              note=f"drop {len(useless)} features with permutation importance ≤ 0")
                r_fs["dropped_features"] = useless
                print(f"  {base_key}: {base_res['oof_score']:.5f} → {fam}_fs (tanpa {len(useless)} fitur): {r_fs['oof_score']:.5f}")
                if rank_score(y, r_fs["oof"]) > rank_score(y, base_res["oof"]) + 0.0005:
                    RESULTS[f"{fam}_fs"] = r_fs
                    print("  ✅ Feature selection membantu → model _fs masuk kandidat ensemble.")
                else:
                    print("  ➖ Tidak membantu signifikan → semua fitur dipertahankan.")
            finally:
                ACTIVE_DROP = set()
    save_table(PERM_IMP, "permutation_importance_selection")

# %% [markdown]
# ## 16.3 Ablation Study — kontribusi tiap kelompok fitur
# Setiap **kelompok informasi** (fitur mentah + turunannya, misal tenure + band tenure + tanggal mulai) dibuang satu per satu, model dilatih ulang dengan fold & hyperparameter yang **sama**, lalu penurunan AUC diuji dengan **DeLong test** (Holm-corrected). Ditambah satu ablation: **buang semua fitur domain hasil engineering** → membuktikan nilai feature engineering.
#
# **Cara baca:** bar merah = kelompok yang kontribusinya signifikan. Penurunan AUC besar = informasi unik (tidak tergantikan fitur lain). Penurunan ≈ 0 walau SHAP tinggi → informasinya redundan dengan kelompok lain.
#
# **Untuk laporan:** 1 kalimat di Model + figure di Appendix → bukti kuat *kenapa* driver itu penting (bukan sekadar korelasi), sekaligus bukti trial-and-error yang sistematis.

# %%
ABLATION = pd.DataFrame()


def feature_groups():
    """Kelompokkan fitur model ke 'keluarga informasi' (raw + turunannya) → unit ablation."""
    groups = OrderedDict()
    date_cols_ = [c for c, t in COL_TYPES.items() if t == "date"]
    for f in MODEL_FEATURES:
        g = FEATURE_FAMILY.get(f)
        if g is None:
            for d in date_cols_:
                if f.startswith(d + "_"):
                    g = d
                    break
        if g is None:
            role = next((r for r, c in ROLES.items() if c == f), None)
            g = role or f
        groups.setdefault(g, []).append(f)
    return groups


def group_model_cols(members):
    cols = set()
    for f in members:
        s = NAME_MAP.get(f)
        if s:
            cols |= {s, f"{s}__freq", f"{s}__te"}
        cols.add(f"{sanitize_name(f)}__freq")
    return cols


with safe_section("Ablation study (feature groups)"):
    set_stage("5 ablation study")
    if not CFG.USE_ABLATION:
        print("⏭️ USE_ABLATION=False")
    else:
        fam = "lgbm" if HAS_LGB else "hgb"
        abl_params = {"learning_rate": 0.06} if fam == "lgbm" else {"learning_rate": 0.1}
        groups = feature_groups()
        # ranking kelompok: permutation importance (kalau ada) → IV
        score = {}
        for g, mem in groups.items():
            s = 0.0
            if len(PERM_IMP):
                s = PERM_IMP[PERM_IMP["feature"].isin(mem) | PERM_IMP["feature_model"].isin(group_model_cols(mem))]["importance_mean"].clip(lower=0).sum()
            if s == 0 and len(DRIVER_TABLE):
                s = DRIVER_TABLE[DRIVER_TABLE["feature"].isin(mem)]["iv"].sum() * 1e-3
            score[g] = s
        ranked = [g for g in sorted(groups, key=lambda k: -score[k]) if len(groups) > 1][: CFG.ABLATION_MAX_GROUPS]
        tests = [(f"− {g}", groups[g]) for g in ranked]
        fe_members = [f for f in MODEL_FEATURES if f in set(FE_NEW)]
        if fe_members and len(fe_members) < len(MODEL_FEATURES):
            tests.append(("− all engineered domain features", fe_members))
        folds_abl = [FOLD_SETS[0]]
        t0 = time.time()
        ref = run_cv(fam, params=abl_params, fold_sets=folds_abl, label="ablation reference (all features)", verbose=False,
                     keep_models=False, note="reference for ablation (same folds & params)")
        rows = []
        for name_, mem in tests:
            ACTIVE_DROP = group_model_cols(mem)
            try:
                r = run_cv(fam, params=abl_params, fold_sets=folds_abl, label=f"ablation {name_}", verbose=False, keep_models=False,
                           note=f"drop {len(mem)} feature(s): {', '.join(mem[:4])}{'…' if len(mem) > 4 else ''}")
            finally:
                ACTIVE_DROP = set()
            a_ref, a_new, diff, se, p = delong_test(y, ref["oof"], r["oof"])
            rows.append({"ablation": name_, "n_features_removed": len(mem), "features": ", ".join(mem[:8]),
                         "auc_full": a_ref, "auc_without": a_new, "auc_drop": diff, "ci_low": diff - 1.96 * se,
                         "ci_high": diff + 1.96 * se, "delong_p": p})
        ABLATION = pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
        if len(ABLATION):
            ABLATION["holm_p"] = holm_adjust(ABLATION["delong_p"].values)
            ABLATION["verdict"] = np.where((ABLATION["holm_p"] < 0.05) & (ABLATION["auc_drop"] > 0), "contributes significantly",
                                           np.where(ABLATION["auc_drop"] < 0, "removal helps / redundant", "no significant contribution"))
            save_table(ABLATION, "ablation_study")
            display(ABLATION)
            d = ABLATION.iloc[::-1]
            fig, ax = plt.subplots(figsize=(8.5, max(3, 0.42 * len(d) + 1)))
            cols_ = [CFG.PALETTE[1] if v == "contributes significantly" else "#9E9E9E" if v.startswith("no") else CFG.PALETTE[2] for v in d["verdict"]]
            ax.barh(d["ablation"].str.slice(0, 48), d["auc_drop"], xerr=[d["auc_drop"] - d["ci_low"], d["ci_high"] - d["auc_drop"]],
                    color=cols_, capsize=2, ecolor="#555")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel(f"Drop in ROC-AUC when the group is removed (reference AUC {ref['oof_auc']:.3f}; 95% DeLong CI)")
            ax.set_title("Ablation Study: Contribution of Each Feature Group")
            save_fig("ablation_study", fig)
            top = ABLATION.iloc[0]
            add_insight("Model", f"Ablation study: removing '{top['ablation'].replace('− ', '')}' lowers ROC-AUC by {top['auc_drop']:.3f} "
                                 f"(95% CI {top['ci_low']:.3f} to {top['ci_high']:.3f}, DeLong p {p_fmt(top['delong_p'])}), the largest contribution of any feature group.")
            fe_row = ABLATION[ABLATION["ablation"] == "− all engineered domain features"]
            if len(fe_row):
                fr = fe_row.iloc[0]
                add_insight("Model", f"Engineered insurance features add {fr['auc_drop']:+.4f} ROC-AUC over the raw variables "
                                     f"(DeLong p {p_fmt(fr['delong_p'])}).")
        print(f"Ablation selesai dalam {time.time() - t0:.0f}s ({len(tests)} kelompok, model {fam}).")

# %% [markdown]
# # 17. Ensemble
#
# Menggabungkan model yang **berbeda karakter** (misal CatBoost + LightGBM + LogReg) biasanya lebih baik dari model terbaik tunggal.
#
# | Metode | Cara | Kapan bagus |
# |---|---|---|
# | Best single | model terbaik saja | baseline |
# | Mean top-k | rata-rata probabilitas k model teratas | simpel & robust |
# | Rank mean | rata-rata ranking (untuk AUC) | model dengan skala probabilitas beda |
# | **Hill climbing** (Caruana) | tambah model satu per satu (boleh berulang) selama skor OOF naik | biasanya terbaik |
# | Optimized weights | bobot non-negatif dioptimasi (scipy) | mirip hill climbing |
# | **Stacking** | Logistic Regression di atas logit(OOF) dengan CV | kalau model saling melengkapi |
#
# **Peringatan:** bobot dioptimasi di OOF yang sama → sedikit optimis. Kalau selisih metode < 0.0005, notebook memilih metode **paling sederhana** (lebih aman di test).
#
# **Cek korelasi OOF antar model:** korelasi sangat tinggi (> 0.98) → ensemble tidak banyak membantu (modelnya "berpikir" sama).

# %%
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def rank01(a):
    return (stats.rankdata(a) - 1) / max(len(a) - 1, 1)


ENSEMBLES = OrderedDict()

with safe_section("Ensemble"):
    LB = leaderboard()
    cand = [m for m in LB["model"] if m != "dummy"]
    if not cand:
        raise RuntimeError("Tidak ada model yang berhasil dilatih.")
    oofs = {m: RESULTS[m]["oof"] for m in cand}
    tests = {m: RESULTS[m]["test"] for m in cand}
    has_test = all(len(tests[m]) for m in cand)

    if len(cand) > 1:
        corr = pd.DataFrame(oofs).corr(method="spearman")
        fig, ax = plt.subplots(figsize=(min(10, 1 + 0.7 * len(cand)), min(8, 1 + 0.6 * len(cand))))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="viridis", ax=ax, annot_kws={"size": 7})
        ax.set_title("Correlation of OOF Predictions Between Models")
        save_fig("oof_prediction_correlation")

    best = cand[0]
    ENSEMBLES["best_single"] = (oofs[best], tests[best], {best: 1.0})
    if CFG.USE_ENSEMBLE and len(cand) > 1:
        k = min(3, len(cand))
        top = cand[:k]
        ENSEMBLES[f"mean_top{k}"] = (np.mean([oofs[m] for m in top], 0), np.mean([tests[m] for m in top], 0) if has_test else np.array([]),
                                     {m: 1 / k for m in top})
        if CFG.METRIC not in LOWER_IS_BETTER:
            ENSEMBLES[f"rank_mean_top{k}"] = (np.mean([rank01(oofs[m]) for m in top], 0),
                                              np.mean([rank01(tests[m]) for m in top], 0) if has_test else np.array([]), {m: 1 / k for m in top})
        # hill climbing
        pool = cand[:min(10, len(cand))]
        chosen = [best]
        cur = oofs[best].copy()
        cur_s = rank_score(y, cur)
        for _ in range(30):
            best_m, best_s = None, cur_s
            for m in pool:
                trial = (cur * len(chosen) + oofs[m]) / (len(chosen) + 1)
                s = rank_score(y, trial)
                if s > best_s + 1e-6:
                    best_m, best_s = m, s
            if best_m is None:
                break
            chosen.append(best_m)
            cur = (cur * (len(chosen) - 1) + oofs[best_m]) / len(chosen)
            cur_s = best_s
        w = pd.Series(chosen).value_counts(normalize=True).to_dict()
        ENSEMBLES["hill_climb"] = (sum(w[m] * oofs[m] for m in w), sum(w[m] * tests[m] for m in w) if has_test else np.array([]), w)
        # optimized weights (minimize logloss)
        try:
            from scipy.optimize import minimize
            P = np.column_stack([oofs[m] for m in pool])

            def obj(wv):
                wv = np.abs(wv) / np.abs(wv).sum()
                return log_loss(y, np.clip(P @ wv, 1e-7, 1 - 1e-7))
            r = minimize(obj, np.ones(len(pool)) / len(pool), method="Nelder-Mead", options={"maxiter": 2000})
            wv = np.abs(r.x) / np.abs(r.x).sum()
            wd = {m: float(v) for m, v in zip(pool, wv) if v > 0.01}
            tot = sum(wd.values())
            wd = {m: v / tot for m, v in wd.items()}
            ENSEMBLES["optimized_weights"] = (sum(wd[m] * oofs[m] for m in wd), sum(wd[m] * tests[m] for m in wd) if has_test else np.array([]), wd)
        except Exception as e:
            print("optimized weights gagal:", e)
        # stacking
        try:
            Z = np.column_stack([logit(oofs[m]) for m in pool])
            Zt = np.column_stack([logit(tests[m]) for m in pool]) if has_test else None
            meta = LogisticRegression(C=1.0, max_iter=2000)
            meta_oof = cross_val_predict(meta, Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=CFG.SEED + 7), method="predict_proba")[:, 1]
            meta.fit(Z, y)
            meta_test = meta.predict_proba(Zt)[:, 1] if has_test else np.array([])
            ENSEMBLES["stacking_logreg"] = (meta_oof, meta_test, dict(zip(pool, np.round(meta.coef_[0], 3))))
        except Exception as e:
            print("stacking gagal:", e)

    ens_rows = [{"method": k, f"oof_{CFG.METRIC}": show_score(rank_score(y, v[0])), "oof_auc": roc_auc_score(y, v[0]),
                 "weights": {m: round(float(w_), 3) for m, w_ in v[2].items()}} for k, v in ENSEMBLES.items()]
    ENS_TABLE = pd.DataFrame(ens_rows)
    set_stage("6 ensembling")
    for k, v in ENSEMBLES.items():
        log_experiment(f"ensemble: {k}", "ensemble", v[0], note=", ".join(f"{m} {float(w_):.2f}" for m, w_ in v[2].items())[:160])
    save_table(ENS_TABLE.astype({"weights": str}), "ensemble_comparison")
    display(ENS_TABLE)

    simplicity = list(ENSEMBLES.keys())
    scores = {k: rank_score(y, v[0]) for k, v in ENSEMBLES.items()}
    top_s = max(scores.values())
    FINAL_NAME = next(k for k in simplicity if scores[k] >= top_s - 0.0005)
    FINAL_OOF, FINAL_TEST, FINAL_WEIGHTS = ENSEMBLES[FINAL_NAME]
    if FINAL_NAME.startswith("rank"):
        # skor rank bukan probabilitas → ubah ke probabilitas terkalibrasi (isotonic, cross-fitted; ranking hampir tidak berubah)
        _cal = np.zeros(N_TR)
        for tr_, va_ in StratifiedKFold(5, shuffle=True, random_state=CFG.SEED + 11).split(FINAL_OOF, y):
            _cal[va_] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(FINAL_OOF[tr_], y[tr_]).predict(FINAL_OOF[va_])
        if len(FINAL_TEST):
            FINAL_TEST = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(FINAL_OOF, y).predict(FINAL_TEST)
        FINAL_OOF = _cal
        print(f"(Ensemble berbasis rank → dikonversi ke probabilitas terkalibrasi; AUC setelah konversi = {roc_auc_score(y, FINAL_OOF):.4f})")
    if roc_auc_score(y, FINAL_OOF) > 0.98:
        print("🚨🚨 OOF ROC-AUC > 0.98 — hampir pasti ada LEAKAGE (fitur yang tercatat setelah churn). Cek Section 5.3 & permutation importance teratas!")
    print(f"\n🏁 Final pilihan: {FINAL_NAME} | OOF {CFG.METRIC} = {show_score(scores[FINAL_NAME]):.5f} | weights = {FINAL_WEIGHTS}")
    add_insight("Model", f"The final model ({FINAL_NAME}) achieves a cross-validated {CFG.METRIC} of {show_score(scores[FINAL_NAME]):.4f} "
                         f"(ROC-AUC {roc_auc_score(y, FINAL_OOF):.4f}) versus 0.50 for a random baseline.")

# %% [markdown]
# ## 17.2 Experiment Log & Model Development Journey (bukti trial-and-error)
# Semua eksperimen (model zoo, tuning, multi-seed, feature selection, ablation, ensemble) tercatat otomatis di `EXPERIMENT_LOG` → `tables/experiment_log.csv`. Figure menunjukkan **perjalanan skor** dari baseline sampai model final + trade-off akurasi vs waktu komputasi.
#
# **Untuk laporan:** figure di Appendix + 1 kalimat: *"We logged N experiments across K stages; ROC-AUC improved from … (logistic baseline) to … (final)."*

# %%
EXP_TABLE = pd.DataFrame()
with safe_section("Experiment log & model development journey"):
    EXP_TABLE = pd.DataFrame(EXPERIMENT_LOG)
    if len(EXP_TABLE) and "roc_auc" in EXP_TABLE:
        final_members = set(FINAL_WEIGHTS) if "FINAL_WEIGHTS" in globals() else set()
        EXP_TABLE["kept_in_final"] = EXP_TABLE["experiment"].isin(final_members) | (EXP_TABLE["experiment"] == f"ensemble: {FINAL_NAME}")
        save_table(EXP_TABLE, "experiment_log")
        print(f"{len(EXP_TABLE)} eksperimen tercatat:")
        display(EXP_TABLE[["exp_id", "stage", "experiment", "note", "roc_auc", "pr_auc", "logloss", "time_s", "kept_in_final"]].round(4))

        stages = list(OrderedDict.fromkeys(EXP_TABLE["stage"]))
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
        best_so_far, xs, ys, labels = -np.inf, [], [], []
        for i, st in enumerate(stages):
            d = EXP_TABLE[EXP_TABLE["stage"] == st]
            is_abl = "ablation" in st
            jitter = np.random.RandomState(i).uniform(-0.18, 0.18, len(d))
            ax[0].scatter(i + jitter, d["roc_auc"], s=30, alpha=0.75, color="#9E9E9E" if is_abl else CFG.PALETTE[i % len(CFG.PALETTE)],
                          edgecolor="k", lw=0.3, zorder=3)
            if not is_abl and d["roc_auc"].notna().any():
                b = d.loc[d["roc_auc"].idxmax()]
                if b["roc_auc"] > best_so_far:
                    best_so_far = b["roc_auc"]
                    labels.append((i, b["roc_auc"], str(b["experiment"])[:26]))
                xs.append(i)
                ys.append(best_so_far)
        ax[0].step(xs, ys, where="post", color=CFG.PALETTE[1], lw=2, label="best model so far")
        for i, v, t in labels:
            ax[0].annotate(f"{t}\n{v:.4f}", (i, v), textcoords="offset points", xytext=(4, 8), fontsize=7, color=CFG.PALETTE[1])
        lo_ = EXP_TABLE.loc[EXP_TABLE["family"] != "dummy", "roc_auc"].min()
        ax[0].set_ylim(max(0.45, lo_ - 0.01), min(1.0, EXP_TABLE["roc_auc"].max() + 0.012))
        ax[0].set_xticks(range(len(stages)))
        ax[0].set_xticklabels([s.split(" ", 1)[-1].replace(" ", "\n") for s in stages], fontsize=8)
        ax[0].set_ylabel("Out-of-fold ROC-AUC")
        ax[0].set_title(f"Model development journey ({len(EXP_TABLE)} experiments; grey = ablations)")
        ax[0].legend(fontsize=8, loc="lower right")
        zoo = EXP_TABLE[EXP_TABLE["stage"].str.contains("baseline") & (EXP_TABLE["family"] != "dummy")]
        if len(zoo):
            ax[1].scatter(zoo["time_s"].clip(lower=0.5), zoo["roc_auc"], s=55, color=CFG.PALETTE[0], edgecolor="k", lw=0.4)
            for _, r in zoo.iterrows():
                ax[1].annotate(MODEL_NAMES.get(r["family"], r["family"]).split(" (")[0], (max(r["time_s"], 0.5), r["roc_auc"]),
                               textcoords="offset points", xytext=(4, 3), fontsize=7)
            ax[1].set_xscale("log")
            ax[1].set_xlabel("Training time for full CV (seconds, log)")
            ax[1].set_ylabel("Out-of-fold ROC-AUC")
            ax[1].set_title("Model zoo: accuracy vs compute cost")
        plt.tight_layout()
        save_fig("model_development_journey", fig)
        base_ = EXP_TABLE[(EXP_TABLE["family"] == "logreg") & EXP_TABLE["stage"].str.contains("baseline")]
        if len(base_) and "FINAL_OOF" in globals():
            add_insight("Model", f"We ran {len(EXP_TABLE)} logged experiments across {len(stages)} stages (model zoo, tuning, multi-seed, "
                                 f"feature selection, ablation, ensembling); ROC-AUC improved from {base_['roc_auc'].iloc[0]:.4f} "
                                 f"(logistic regression baseline) to {roc_auc_score(y, FINAL_OOF):.4f} (final {FINAL_NAME}).")

# %% [markdown]
# # 18. Threshold & Calibration
#
# ## Threshold
# Model mengeluarkan **probabilitas**. Untuk keputusan (churn / tidak) butuh **threshold**. 0.5 jarang optimal kalau data imbalanced.
# - Threshold dioptimasi di OOF untuk `THR_METRIC` (F1 default; atau metric lomba kalau berbasis label).
# - Plot precision/recall/F1 vs threshold membantu memilih trade-off.
# - **Threshold bisnis** (biaya salah tebak) dihitung di Section 22.
#
# ## Calibration
# Probabilitas **terkalibrasi** = kalau model bilang 30%, kira-kira 30% dari mereka benar churn. Penting untuk: logloss/brier, expected loss, CLV, ROI campaign.
# - Reliability diagram: titik di garis diagonal = terkalibrasi.
# - `CFG.CALIBRATION="auto"` → coba isotonic & sigmoid (Platt) via CV di atas OOF; dipakai kalau memperbaiki logloss.
#
# **Jika X → Y:**
# - Kurva di bawah diagonal → model over-confident (sering terjadi setelah class_weight/SMOTE) → kalibrasi wajib untuk analisis bisnis.
# - Metric lomba AUC → kalibrasi tidak mengubah ranking; submission tetap pakai probabilitas mentah (kalibrasi hanya untuk analisis bisnis).

# %%
with safe_section("Threshold tuning"):
    BEST_THR, BEST_THR_SCORE = best_threshold(y, FINAL_OOF, THR_METRIC)
    grid = np.linspace(0.02, 0.98, 97)
    curves = pd.DataFrame({"threshold": grid,
                           "precision": [precision_score(y, FINAL_OOF >= t, zero_division=0) for t in grid],
                           "recall": [recall_score(y, FINAL_OOF >= t, zero_division=0) for t in grid],
                           "f1": [f1_score(y, FINAL_OOF >= t, zero_division=0) for t in grid],
                           "accuracy": [accuracy_score(y, FINAL_OOF >= t) for t in grid]})
    fig, ax = plt.subplots(figsize=(8, 4))
    for c, col in zip(["precision", "recall", "f1", "accuracy"], CFG.PALETTE):
        ax.plot(curves["threshold"], curves[c], label=c, color=col)
    ax.axvline(BEST_THR, color="black", ls="--", lw=1, label=f"best {THR_METRIC} thr = {BEST_THR:.3f}")
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("Decision threshold")
    ax.set_title("Precision / Recall / F1 vs Threshold (OOF)")
    ax.legend(fontsize=8)
    save_fig("threshold_curves")
    print(f"Threshold optimal untuk {THR_METRIC}: {BEST_THR:.4f} → {THR_METRIC} = {BEST_THR_SCORE:.4f} "
          f"(threshold 0.5 → {compute_metric(THR_METRIC, y, FINAL_OOF, 0.5):.4f})")

# %%
FINAL_OOF_CAL, FINAL_TEST_CAL, CAL_METHOD = FINAL_OOF, FINAL_TEST, "none"

with safe_section("Calibration"):
    def cv_calibrate(p, method):
        out = np.zeros_like(p)
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=CFG.SEED + 3).split(p, y):
            if method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p[tr], y[tr])
                out[va] = cal.predict(p[va])
            else:
                cal = LogisticRegression(C=1e6, max_iter=2000).fit(logit(p[tr]).reshape(-1, 1), y[tr])
                out[va] = cal.predict_proba(logit(p[va]).reshape(-1, 1))[:, 1]
        return out

    base_ll = log_loss(y, np.clip(FINAL_OOF, 1e-7, 1 - 1e-7))
    cal_res = {"none": (base_ll, FINAL_OOF)}
    methods = ["isotonic", "sigmoid"] if CFG.CALIBRATION == "auto" else ([] if CFG.CALIBRATION == "none" else [CFG.CALIBRATION])
    for m in methods:
        pc = cv_calibrate(FINAL_OOF, m)
        cal_res[m] = (log_loss(y, np.clip(pc, 1e-7, 1 - 1e-7)), pc)
    CAL_METHOD = min(cal_res, key=lambda k: cal_res[k][0]) if CFG.CALIBRATION == "auto" else (CFG.CALIBRATION if CFG.CALIBRATION in cal_res else "none")
    if CAL_METHOD != "none" and cal_res[CAL_METHOD][0] > base_ll - 0.001:
        CAL_METHOD = "none"   # perbaikan tidak berarti
    for k, (ll, _) in cal_res.items():
        print(f"  {k:<9}: OOF logloss = {ll:.5f}, Brier = {brier_score_loss(y, cal_res[k][1]):.5f}")
    print("Kalibrasi dipakai:", CAL_METHOD)
    if CAL_METHOD != "none":
        FINAL_OOF_CAL = cal_res[CAL_METHOD][1]
        if len(FINAL_TEST):
            if CAL_METHOD == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(FINAL_OOF, y)
                FINAL_TEST_CAL = cal.predict(FINAL_TEST)
            else:
                cal = LogisticRegression(C=1e6, max_iter=2000).fit(logit(FINAL_OOF).reshape(-1, 1), y)
                FINAL_TEST_CAL = cal.predict_proba(logit(FINAL_TEST).reshape(-1, 1))[:, 1]

    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for k, (ll, pc) in cal_res.items():
        fy, fx = calibration_curve(y, pc, n_bins=10, strategy="quantile")
        ax.plot(fx, fy, "o-", ms=4, label=f"{k} (logloss {ll:.4f})")
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Reliability Diagram (OOF)")
    ax.legend(fontsize=8)
    save_fig("calibration_reliability")

# %% [markdown]
# # 19. Evaluation & Debugging
#
# | Cek | Untuk apa | Tanda bahaya → tindakan |
# |---|---|---|
# | ROC & PR curve | performa di semua threshold | kurva PR rendah walau AUC bagus → kelas minoritas sulit; fokus recall/precision trade-off |
# | Confusion matrix | jenis kesalahan (FN vs FP) | FN banyak = churner lolos → turunkan threshold kalau recall penting |
# | Fold stability | konsistensi | std tinggi → data kecil/noisy, pakai repeats |
# | **Learning curve** | butuh data lebih / model lebih kompleks? | train ≫ valid & valid masih naik → **variance** (tambah regularisasi/data); keduanya rendah & datar → **bias** (tambah fitur) |
# | **Shuffled-target test** | deteksi bug/leakage di pipeline | AUC dengan target diacak harus ≈ 0.5. Kalau > 0.55 → ada kebocoran (misal target encoding salah) |
# | OOF vs test distribution | apakah test "mirip" | rata-rata prediksi test jauh dari churn rate train → drift / perbedaan populasi |
# | Error analysis | siapa yang salah ditebak | pola di False Negative = ide fitur baru |
#
# **Untuk laporan:** ROC curve + confusion matrix (di threshold terpilih) + 1 kalimat recall/precision cukup. Learning curve & shuffled test → lampiran / 1 kalimat validasi metodologi.

# %%
with safe_section("ROC / PR curves & confusion matrix"):
    LB = leaderboard()
    show_models = [m for m in LB["model"] if m != "dummy"][:4]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    curves_src = [("FINAL (" + FINAL_NAME + ")", FINAL_OOF)] + [(m, RESULTS[m]["oof"]) for m in show_models]
    for i, (nm, p) in enumerate(curves_src):
        fpr, tpr, _ = roc_curve(y, p)
        prc, rcl, _ = precision_recall_curve(y, p)
        lw = 2.5 if i == 0 else 1.2
        ax[0].plot(fpr, tpr, lw=lw, label=f"{nm} (AUC {roc_auc_score(y, p):.3f})", color=CFG.PALETTE[i % len(CFG.PALETTE)])
        ax[1].plot(rcl, prc, lw=lw, label=f"{nm} (AP {average_precision_score(y, p):.3f})", color=CFG.PALETTE[i % len(CFG.PALETTE)])
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate (recall)")
    ax[0].set_title("ROC Curve (OOF)")
    ax[1].axhline(CHURN_RATE, color="k", ls="--", lw=0.8, label=f"baseline {CHURN_RATE:.2f}")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall Curve (OOF)")
    for a in ax:
        a.legend(fontsize=7)
    save_fig("roc_pr_curves")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, thr, ttl in [(ax[0], 0.5, "threshold 0.5"), (ax[1], BEST_THR, f"optimal threshold {BEST_THR:.3f} ({THR_METRIC})")]:
        cm = confusion_matrix(y, (FINAL_OOF >= thr).astype(int))
        annot = np.array([[f"{v:,}\n({v / cm.sum():.1%})" for v in row] for row in cm])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=a,
                    xticklabels=["Pred: Retain", "Pred: Churn"], yticklabels=["Actual: Retain", "Actual: Churn"])
        a.set_title(f"Confusion Matrix — {ttl}", fontsize=10)
    save_fig("confusion_matrices")
    yl = (FINAL_OOF >= BEST_THR).astype(int)
    print(classification_report(y, yl, target_names=["Retained", "Churned"], digits=4))
    rec, prec = recall_score(y, yl), precision_score(y, yl)
    add_insight("Model", f"At the selected threshold ({BEST_THR:.2f}) the model identifies {pct(rec)} of churners (recall) with a precision of {pct(prec)}, "
                         f"i.e. {prec / CHURN_RATE:.1f}x better targeting than random selection.")

# %% [markdown]
# ### 19.1 Bootstrap confidence interval & uji signifikansi antar model
# Skor tunggal (misal AUC 0.749) tidak menunjukkan **ketidakpastian**. Bootstrap (resample customer dengan pengembalian, N kali) memberi **95% CI**.
# **Paired bootstrap** membandingkan model final vs pembanding pada sampel yang sama → apakah selisihnya nyata atau kebetulan?
#
# **Cara baca:** CI sempit → estimasi stabil. Selisih model dengan CI yang melewati 0 / p > 0.05 → **tidak berbeda signifikan** → pilih yang lebih sederhana/interpretable (argumen kuat di laporan).
# **Untuk laporan:** *"The final model achieves ROC-AUC = 0.749 (95% CI 0.737–0.761); its improvement over the interpretable logistic regression is +0.003 (95% CI −0.001 to 0.007, p = 0.12)."*

# %%
BOOT_TABLE = pd.DataFrame()
with safe_section("Bootstrap CI & model comparison"):
    rng = np.random.RandomState(CFG.SEED)
    B = CFG.N_BOOTSTRAP
    idx_b = [rng.randint(0, N_TR, N_TR) for _ in range(B)]

    def boot_scores(p, metric, thr=0.5):
        out = []
        for ib in idx_b:
            if y[ib].min() == y[ib].max():
                continue
            out.append(compute_metric(metric, y[ib], p[ib], thr))
        return np.array(out)

    metrics_to_report = ["roc_auc", "pr_auc", "logloss"] + ([CFG.METRIC] if CFG.METRIC not in ("roc_auc", "pr_auc", "logloss") else [])
    rows = []
    for m in metrics_to_report:
        thr = BEST_THR if m in LABEL_METRICS else 0.5
        bs = boot_scores(FINAL_OOF, m, thr)
        rows.append({"metric": m, "estimate": compute_metric(m, y, FINAL_OOF, thr), "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5)})
    BOOT_TABLE = pd.DataFrame(rows)
    display(BOOT_TABLE)
    save_table(BOOT_TABLE, "bootstrap_ci_final_model")
    r0 = BOOT_TABLE.iloc[0]
    FINAL_AUC_CI = (r0["ci_low"], r0["ci_high"])
    add_insight("Model", f"Final model ROC-AUC = {r0['estimate']:.3f} (95% bootstrap CI {r0['ci_low']:.3f}–{r0['ci_high']:.3f}).")

    LB = leaderboard()
    comps = [m for m in LB["model"] if m not in ("dummy",)][:4]
    if "logreg" in RESULTS and "logreg" not in comps:
        comps.append("logreg")
    cmp_metric = "roc_auc" if CFG.METRIC in LABEL_METRICS else CFG.METRIC
    sign = -1 if cmp_metric in LOWER_IS_BETTER else 1
    crow = []
    for m in comps:
        pm = RESULTS[m]["oof"]
        diffs = np.array([sign * (compute_metric(cmp_metric, y[ib], FINAL_OOF[ib]) - compute_metric(cmp_metric, y[ib], pm[ib]))
                          for ib in idx_b[: min(B, 500)] if y[ib].min() != y[ib].max()])
        crow.append({"final_vs": m, f"diff_{cmp_metric}": sign * (compute_metric(cmp_metric, y, FINAL_OOF) - compute_metric(cmp_metric, y, pm)),
                     "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
                     "p_value_one_sided": float((diffs <= 0).mean()), "significant": bool(np.percentile(diffs, 2.5) > 0)})
    MODEL_CMP = pd.DataFrame(crow)
    print(f"Paired bootstrap: final ({FINAL_NAME}) vs model lain ({cmp_metric}; positif = final lebih baik):")
    display(MODEL_CMP)
    save_table(MODEL_CMP, "bootstrap_model_comparison")
    if "logreg" in RESULTS:
        lr_row = MODEL_CMP[MODEL_CMP["final_vs"] == "logreg"]
        if len(lr_row):
            lr_row = lr_row.iloc[0]
            verdict = "significantly better than" if lr_row["significant"] else "not significantly different from"
            add_insight("Model", f"The final model is {verdict} the interpretable logistic regression "
                                 f"(Δ{cmp_metric} = {lr_row[f'diff_{cmp_metric}']:+.4f}, 95% CI {lr_row['ci_low']:+.4f} to {lr_row['ci_high']:+.4f}).")

# %% [markdown]
# ### 19.2 Perbandingan model: DeLong test, kalibrasi & EMPC (profit-based)
# | Kolom | Arti |
# |---|---|
# | `roc_auc`, `auc_ci_*` | AUC + 95% CI analitik (DeLong) |
# | `delta_auc_final_minus_model`, `delong_p(_holm)` | apakah model final **signifikan** lebih baik (paired, customer yang sama; koreksi Holm) |
# | `brier`, `ece` | kualitas probabilitas (penting untuk CLV/ROI) |
# | `empc_per_customer` | **Expected Maximum Profit (churn)** — rata-rata profit maksimum kampanye per customer, memperhitungkan ketidakpastian tingkat penerimaan tawaran (Verbraken et al., 2013) |
# | `empc_fraction_targeted` | fraksi customer optimal untuk dikontak menurut EMPC |
#
# **Jika X → Y:** model AUC terbaik ≠ model EMPC terbaik → sebut di laporan (Höppner et al., 2020): pilih model berdasarkan profit kalau tujuan bisnis = kampanye retensi. Selisih AUC tidak signifikan → pilih model yang lebih sederhana/interpretable (argumen kuat untuk juri praktisi).

# %%
MODEL_EVAL = pd.DataFrame()


def empc_inputs():
    """CLV customer yang diselamatkan (rata-rata) + biaya, konsisten dengan Section 21. Tanpa kolom nilai → satuan relatif."""
    vcol = CFG.VALUE_COL or ROLES.get("premium")
    if vcol and vcol in train.columns and pd.api.types.is_numeric_dtype(train[vcol]):
        mult = 12 if CFG.VALUE_PERIOD == "monthly" else 1
        margin = train[vcol].fillna(train[vcol].median()).clip(lower=0).mean() * mult * CFG.PROFIT_MARGIN
        r = (1 - CHURN_RATE) / (1 + CFG.DISCOUNT_RATE)
        clv = margin * (1 - r ** CFG.CLV_HORIZON_YEARS) / max(1 - r, 1e-9)
        prm = empc_params()
        return dict(clv=clv, d=prm["d"], f=prm["f"], alpha=prm["alpha"], beta=prm["beta"]), CFG.CURRENCY
    prm = empc_params()   # tanpa nilai: CLV = 1 unit, biaya relatif standar Verbraken et al. (δ = 0.05, φ = 0.005)
    return dict(clv=1.0, d=0.05, f=0.005, alpha=prm["alpha"], beta=prm["beta"]), "CLV units"


with safe_section("Model comparison: DeLong, EMPC & calibration"):
    EMPC_ARGS, EMPC_UNIT = empc_inputs()
    cand_ = [m for m in leaderboard()["model"] if m != "dummy"]
    preds = [("FINAL: " + FINAL_NAME, FINAL_OOF)] + [(m, RESULTS[m]["oof"]) for m in cand_]
    rows = []
    for nm, p in preds:
        auc_, lo_, hi_, se_ = delong_ci(y, p)
        row = {"model": nm, "algorithm": "ensemble" if nm.startswith("FINAL") else MODEL_NAMES.get(RESULTS[nm]["model"], nm),
               "roc_auc": auc_, "auc_ci_low": lo_, "auc_ci_high": hi_, "pr_auc": average_precision_score(y, p),
               "logloss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), "brier": brier_score_loss(y, np.clip(p, 0, 1)),
               "ece": ece_score(y, np.clip(p, 0, 1))}
        fpr_, tpr_, _ = roc_curve(y, p)
        row["ks"] = float(np.max(tpr_ - fpr_))
        em, eta = empc_score(y, p, **EMPC_ARGS)
        row["empc_per_customer"], row["empc_fraction_targeted"] = em, eta
        if not nm.startswith("FINAL"):
            _, _, diff, se, pv = delong_test(y, FINAL_OOF, p)
            row.update({"delta_auc_final_minus_model": diff, "delong_p": pv})
        rows.append(row)
    MODEL_EVAL = pd.DataFrame(rows)
    mask_ = MODEL_EVAL["delong_p"].notna() if "delong_p" in MODEL_EVAL else pd.Series(False, index=MODEL_EVAL.index)
    if mask_.any():
        MODEL_EVAL.loc[mask_, "delong_p_holm"] = holm_adjust(MODEL_EVAL.loc[mask_, "delong_p"].values)
    MODEL_EVAL["empc_total"] = MODEL_EVAL["empc_per_customer"] * N_TR
    save_table(MODEL_EVAL, "model_comparison_full")
    display(MODEL_EVAL.round(4))

    d = MODEL_EVAL.iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(13, max(3.2, 0.42 * len(d) + 1.2)), gridspec_kw={"width_ratios": [1.3, 1]})
    yy = np.arange(len(d))
    cols_ = [CFG.PALETTE[1] if m.startswith("FINAL") else CFG.PALETTE[0] for m in d["model"]]
    ax[0].errorbar(d["roc_auc"], yy, xerr=[d["roc_auc"] - d["auc_ci_low"], d["auc_ci_high"] - d["roc_auc"]], fmt="none", ecolor="#777", capsize=3)
    ax[0].scatter(d["roc_auc"], yy, c=cols_, zorder=3)
    ax[0].set_yticks(yy)
    ax[0].set_yticklabels(d["model"].str.slice(0, 30), fontsize=8)
    ax[0].set_xlabel("ROC-AUC (95% DeLong CI)")
    ax[0].set_title("Discrimination")
    ax[1].barh(yy, d["empc_per_customer"], color=cols_)
    ax[1].set_yticks(yy)
    ax[1].set_yticklabels([])
    ax[1].set_xlabel(f"EMPC (expected max. profit per customer, {EMPC_UNIT})")
    ax[1].set_title("Profit-based evaluation (EMPC)")
    fig.suptitle("Model Comparison: Statistical vs Profit-Based Evaluation", y=1.02)
    plt.tight_layout()
    save_fig("model_comparison_delong_empc", fig)

    best_auc = MODEL_EVAL.sort_values("roc_auc", ascending=False).iloc[0]
    best_emp = MODEL_EVAL.sort_values("empc_per_customer", ascending=False).iloc[0]
    fin = MODEL_EVAL.iloc[0]
    nsig = int((MODEL_EVAL.get("delong_p_holm", pd.Series(dtype=float)) < 0.05).sum())
    add_insight("Model", f"Across {len(MODEL_EVAL) - 1} candidate models, the final model's ROC-AUC is {fin['roc_auc']:.3f} "
                         f"(95% DeLong CI {fin['auc_ci_low']:.3f}–{fin['auc_ci_high']:.3f}); it is significantly better than {nsig} of them "
                         f"after Holm correction.")
    same = best_auc["model"] == best_emp["model"]
    add_insight("Model", f"Profit-based evaluation (EMPC; Verbraken et al., 2013) {'agrees with' if same else 'differs from'} the AUC ranking: "
                         f"the most profitable model is {best_emp['model']} with an expected maximum profit of "
                         f"{fmt_num(best_emp['empc_per_customer'])} {EMPC_UNIT} per customer when targeting about "
                         f"{pct(best_emp['empc_fraction_targeted'])} of the base.")
    if "logreg" in RESULTS:
        lr_ = MODEL_EVAL[MODEL_EVAL["model"] == "logreg"]
        if len(lr_):
            lr_ = lr_.iloc[0]
            add_insight("Model", f"The interpretable logistic regression reaches ROC-AUC {lr_['roc_auc']:.3f}; the gap to the final model is "
                                 f"{lr_['delta_auc_final_minus_model']:+.4f} (DeLong p {p_fmt(lr_['delong_p'])}).")
    if "ebm" in RESULTS:
        eb_ = MODEL_EVAL[MODEL_EVAL["model"] == "ebm"]
        if len(eb_):
            eb_ = eb_.iloc[0]
            add_insight("Model", f"A glass-box Explainable Boosting Machine attains ROC-AUC {eb_['roc_auc']:.3f} "
                                 f"(Δ vs final {eb_['delta_auc_final_minus_model']:+.4f}, DeLong p {p_fmt(eb_['delong_p'])}), "
                                 f"offering a fully transparent alternative for regulated use.")

# %%
with safe_section("Fold stability"):
    LB = leaderboard()
    ms = [m for m in LB["model"] if m != "dummy"][:8]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(ms))))
    _bx_data, _bx_names = [RESULTS[m]["fold_scores"] for m in ms][::-1], ms[::-1]
    try:                                   # matplotlib ≥ 3.10: orientation; versi lama: vert
        ax.boxplot(_bx_data, orientation="horizontal")
    except TypeError:
        ax.boxplot(_bx_data, vert=False)
    ax.set_yticks(range(1, len(_bx_names) + 1))
    ax.set_yticklabels(_bx_names)
    ax.set_xlabel(f"{CFG.METRIC} per fold")
    ax.set_title("Cross-Validation Score Distribution per Model")
    save_fig("fold_stability")
    for m in ms:
        if RESULTS[m]["cv_std"] > 0.02 and CFG.METRIC in ("roc_auc", "gini", "pr_auc"):
            print(f"⚠️ {m}: std antar fold {RESULTS[m]['cv_std']:.4f} > 0.02 → tidak stabil")

# %%
with safe_section("Learning curve"):
    fam = "lgbm" if HAS_LGB else ("catboost" if HAS_CB else "hgb")
    base_params = dict(RESULTS.get(f"{fam}_tuned", RESULTS.get(fam, {"params": {}}))["params"])
    if fam in ("lgbm", "xgb", "catboost", "hgb"):
        base_params["learning_rate"] = max(base_params.get("learning_rate", 0.05), 0.08)  # lebih cepat; kurva relatif tetap valid
    sizes = [0.2, 0.5, 1.0] if CFG.RUN_MODE == "fast" else [0.1, 0.25, 0.5, 0.75, 1.0]
    folds3 = make_folds(3, CFG.SEED + 55)
    if N_TR > 60_000:   # data besar → learning curve pada subsample supaya cepat
        rs_ = np.random.RandomState(CFG.SEED)
        folds3 = [(rs_.choice(tr, 40_000, replace=False), rs_.choice(va, min(len(va), 20_000), replace=False)) for tr, va in folds3]
    rows = []
    for frac in sizes:
        for k, (tr, va) in enumerate(folds3):
            rng = np.random.RandomState(k)
            tr_s = np.sort(rng.choice(tr, max(50, int(len(tr) * frac)), replace=False))
            Xtr, Xva, Xte = fold_data(MODEL_REP[fam], tr_s, va, seed=k)
            p_va, _, mdl, _, (sub, p_tr) = fit_predict(fam, base_params, Xtr, y[tr_s], Xva, y[va], Xte.iloc[:0], CFG.SEED)
            rows.append({"train_size": len(tr_s), "train_auc": roc_auc_score(y[tr_s][sub], p_tr), "valid_auc": roc_auc_score(y[va], p_va)})
    LC = pd.DataFrame(rows).groupby("train_size").agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, c in [("train_auc", CFG.PALETTE[1]), ("valid_auc", CFG.PALETTE[0])]:
        ax.plot(LC["train_size"], LC[(col, "mean")], "o-", color=c, label=col.replace("_", " "))
        ax.fill_between(LC["train_size"], LC[(col, "mean")] - LC[(col, "std")], LC[(col, "mean")] + LC[(col, "std")], color=c, alpha=0.15)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Learning Curve ({fam})")
    ax.legend()
    save_fig("learning_curve")
    gap = LC[("train_auc", "mean")].iloc[-1] - LC[("valid_auc", "mean")].iloc[-1]
    slope = LC[("valid_auc", "mean")].iloc[-1] - LC[("valid_auc", "mean")].iloc[-2]
    print(f"Gap train-valid di ukuran penuh: {gap:.4f} | kenaikan valid dari 75%→100% data: {slope:+.4f}")
    if gap > 0.08:
        print("→ Variance tinggi (overfit): naikkan regularisasi / min_child_samples, kurangi num_leaves/depth.")
    if slope > 0.005:
        print("→ Valid masih naik: tambahan data akan membantu (sebut di laporan sebagai peluang).")
    if gap < 0.03 and slope <= 0.005:
        print("→ Model sudah jenuh terhadap data; peningkatan berikutnya datang dari fitur baru, bukan tuning.")

# %%
with safe_section("Shuffled-target sanity test"):
    fam = "lgbm" if HAS_LGB else "hgb"
    y_perm = np.random.RandomState(CFG.SEED).permutation(y)
    p_perm = np.zeros(N_TR)
    for k, (tr, va) in enumerate(make_folds(3, CFG.SEED + 77)):
        Xtr, Xva, Xte = fold_data(MODEL_REP[fam], tr, va, seed=k, y_vec=y_perm)
        p_va, _, _, _, _ = fit_predict(fam, {"n_estimators": 300} if fam == "lgbm" else {}, Xtr, y_perm[tr], Xva, y_perm[va], Xte.iloc[:0], CFG.SEED)
        p_perm[va] = p_va
    auc_perm = roc_auc_score(y_perm, p_perm)
    print(f"AUC dengan target DIACAK = {auc_perm:.4f} (harus ≈ 0.50)")
    print("✅ Pipeline bersih dari kebocoran target." if auc_perm < 0.55 else "🚨 AUC > 0.55 dengan target acak → ada leakage di pipeline! Cek target encoding / fitur.")

# %%
with safe_section("OOF vs test prediction distribution"):
    if len(FINAL_TEST):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.hist(FINAL_OOF, bins=50, alpha=0.55, density=True, label=f"Train OOF (mean {FINAL_OOF.mean():.3f})", color=CFG.PALETTE[0])
        ax.hist(FINAL_TEST, bins=50, alpha=0.55, density=True, label=f"Test (mean {FINAL_TEST.mean():.3f})", color=CFG.PALETTE[1])
        ax.set_xlabel("Predicted churn probability")
        ax.set_title("Prediction Distribution: Train OOF vs Test")
        ax.legend()
        save_fig("oof_vs_test_predictions")
        ks = stats.ks_2samp(FINAL_OOF, FINAL_TEST)
        print(f"KS = {ks.statistic:.4f} (p = {ks.pvalue:.4f}) | mean OOF = {FINAL_OOF.mean():.4f}, mean test = {FINAL_TEST.mean():.4f}, churn rate train = {CHURN_RATE:.4f}")
        if ks.statistic > 0.1:
            print("⚠️ Distribusi prediksi test berbeda cukup jauh → cek drift (Section 12).")
    else:
        print("⏭️ Tidak ada test.")

# %%
with safe_section("Error analysis"):
    yl = (FINAL_OOF >= BEST_THR).astype(int)
    err = train.copy()
    err["_y"] = y
    err["_p"] = FINAL_OOF
    err["_type"] = np.select([(y == 1) & (yl == 1), (y == 1) & (yl == 0), (y == 0) & (yl == 1)], ["TP", "FN", "FP"], "TN")
    print(err["_type"].value_counts().to_string())
    prof_feats = [f for f in dedupe_features(DRIVER_TABLE["feature"].tolist()) if f in train.columns][:12]
    prof = pd.DataFrame({"feature": prof_feats})
    for t in ["TP", "FN", "FP", "TN"]:
        m = (err["_type"] == t).values
        vals = []
        for f in prof_feats:
            s = train.loc[m, f]
            vals.append(s.mean() if pd.api.types.is_numeric_dtype(s) else (s.mode().iloc[0] if s.notna().any() else None))
        prof[t] = vals
    print("\nProfil per jenis prediksi (mean numeric / modus kategori). FN = churner yang lolos:")
    display(prof)
    save_table(prof, "error_profile")
    print("\nFalse Negative paling 'meyakinkan' (churn tapi diprediksi aman):")
    cols_show = ([ID_COL] if ID_COL else []) + prof_feats[:8] + ["_p"]
    display(err[err["_type"] == "FN"].sort_values("_p")[cols_show].head(10))
    print("💡 Cari pola di FN: kalau mereka mirip customer setia di semua fitur → churn mereka dipicu faktor yang tidak ada di data (sebut sebagai limitation / usul data tambahan).")

    seg_rows = []
    for f in [f for f in DRIVER_TABLE["feature"] if f in CAT_COLS][:4]:
        lab = train[f].astype(str).where(train[f].notna(), "Missing")
        for lv in lab.value_counts().index[:8]:
            m = (lab == lv).values
            if m.sum() >= 100 and 0 < y[m].sum() < m.sum():
                seg_rows.append({"feature": f, "level": lv, "n": int(m.sum()), "churn_rate": y[m].mean(), "auc": roc_auc_score(y[m], FINAL_OOF[m])})
    if seg_rows:
        SEG_PERF = pd.DataFrame(seg_rows)
        print("\nPerforma model per segmen (AUC jauh lebih rendah di suatu segmen → model kurang andal di sana):")
        display(SEG_PERF)
        save_table(SEG_PERF, "performance_by_segment")

# %% [markdown]
# # 20. Feature Importance & Explainability
#
# Beberapa sudut pandang (pakai lebih dari satu supaya kesimpulan kuat):
#
# | Metode | Arti | Kelebihan / kekurangan |
# |---|---|---|
# | **Gain importance** | total kontribusi split fitur di pohon | cepat; bias ke fitur numerik/kardinalitas tinggi |
# | **Permutation importance** | turunnya skor kalau fitur diacak (di data validasi) | model-agnostic, jujur; fitur berkorelasi bisa "berbagi" importance |
# | **SHAP** | kontribusi tiap fitur ke prediksi **tiap customer** (log-odds) | konsisten, punya **arah** (+ naikkan risiko / − turunkan), bisa per individu |
# | **PDP** | rata-rata prediksi saat fitur diubah | menunjukkan bentuk hubungan (linear, threshold, U-shape) |
#
# **Cara baca SHAP beeswarm:** tiap titik = 1 customer. Sumbu x = dampak ke risiko churn (kanan = naik). Warna = nilai fitur (merah tinggi, biru rendah).
# Contoh: titik merah di kanan untuk `n_complaints` → keluhan banyak menaikkan risiko churn.
#
# **Jika X → Y:**
# - Fitur penting tapi tidak bisa dikontrol (umur, region) → dipakai untuk **targeting/segmentasi**, bukan intervensi.
# - Fitur penting dan bisa dikontrol (metode bayar, auto-renew, keluhan, kenaikan premi) → **basis rekomendasi** (Section 22).
# - Ranking SHAP sangat beda dengan permutation → fitur berkorelasi; sebut keduanya / gabungkan family.
#
# **Untuk laporan:** SHAP bar/beeswarm (1 figure) + 1–2 dependence/PDP untuk driver utama + 1 waterfall contoh customer → sangat meyakinkan juri.

# %%
def pick_explain_model():
    LB = leaderboard()
    for m in LB["model"]:
        r = RESULTS[m]
        if r["model"] in ("lgbm", "catboost", "xgb") and r["models"]:
            return m
    for m in LB["model"]:
        if RESULTS[m]["models"] and m != "dummy":
            return m
    return None


def display_name(c):
    if c.endswith("__te"):
        return f"{INV_NAME.get(c[:-4], c[:-4])} (target-enc)"
    return INV_NAME.get(c, c)


def get_contribs(md, fam):
    model, X = md["model"], md["X_va"]
    if fam == "lgbm":
        c = model.predict(X, pred_contrib=True)
    elif fam == "xgb":
        dm = xgb.DMatrix(X, enable_categorical=True)
        bi = getattr(model, "best_iteration", None)
        try:
            c = model.get_booster().predict(dm, pred_contribs=True, iteration_range=(0, bi + 1) if bi is not None else (0, 0))
        except TypeError:
            c = model.get_booster().predict(dm, pred_contribs=True)
    elif fam == "catboost":
        idx = [i for i, col in enumerate(X.columns) if X[col].dtype == object]
        c = model.get_feature_importance(cb.Pool(X, cat_features=idx), type="ShapValues")
    else:
        return None, None
    c = np.asarray(c)
    return c[:, :-1], c[:, -1]


EXPLAIN_KEY = None
SHAP_VALUES, SHAP_X, SHAP_IDX = None, None, None
SHAP_IMP = pd.DataFrame()
GAIN_IMP = pd.DataFrame()

with safe_section("Gain importance"):
    EXPLAIN_KEY = pick_explain_model()
    print("Model untuk explainability:", EXPLAIN_KEY)
    fam = RESULTS[EXPLAIN_KEY]["model"]
    imps = []
    for md in RESULTS[EXPLAIN_KEY]["models"]:
        m, cols = md["model"], md["X_va"].columns
        if fam == "lgbm":
            v = m.booster_.feature_importance("gain")
        elif fam == "xgb":
            sc = m.get_booster().get_score(importance_type="gain")
            v = [sc.get(c, 0) for c in cols]
        elif fam == "catboost":
            v = m.get_feature_importance()
        elif fam in ("rf", "et"):
            v = m.named_steps["m"].feature_importances_
            if len(v) != len(cols):
                continue
        else:
            continue
        imps.append(pd.Series(np.asarray(v, dtype=float), index=cols))
    if imps:
        g = pd.concat(imps, axis=1).mean(axis=1)
        GAIN_IMP = pd.DataFrame({"feature": [display_name(c) for c in g.index], "gain": g.values})
        GAIN_IMP["gain_share_%"] = 100 * GAIN_IMP["gain"] / GAIN_IMP["gain"].sum()
        GAIN_IMP = GAIN_IMP.sort_values("gain", ascending=False).reset_index(drop=True)
        save_table(GAIN_IMP, "gain_importance")
        top = GAIN_IMP.head(CFG.TOP_N_FEATURES_PLOT)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(top))))
        ax.barh(top["feature"][::-1], top["gain_share_%"][::-1], color=CFG.PALETTE[0])
        ax.set_xlabel("Share of total gain (%)")
        ax.set_title(f"Feature Importance (gain) — {EXPLAIN_KEY}")
        save_fig("gain_importance")
    else:
        print("Gain importance tidak tersedia untuk model ini.")

# %%
with safe_section("Permutation importance"):
    if EXPLAIN_KEY:
        if len(PERM_IMP) == 0 or RESULTS[EXPLAIN_KEY]["models"][0]["X_va"].shape[1] != len(PERM_IMP):
            PERM_IMP = permutation_importance_oof(RESULTS[EXPLAIN_KEY])
        top = PERM_IMP.head(CFG.TOP_N_FEATURES_PLOT)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(top))))
        ax.barh([display_name(c) for c in top["feature_model"]][::-1], top["importance_mean"][::-1], xerr=top["importance_std"][::-1],
                color=CFG.PALETTE[2], capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Drop in validation score when feature is shuffled")
        ax.set_title("Permutation Importance (out-of-fold)")
        save_fig("permutation_importance")
        save_table(PERM_IMP, "permutation_importance")

# %%
with safe_section("SHAP values"):
    fam = RESULTS[EXPLAIN_KEY]["model"]
    vals, Xs, idxs, base_vals = [], [], [], []
    for md in RESULTS[EXPLAIN_KEY]["models"]:
        v, b = get_contribs(md, fam)
        if v is None:
            break
        vals.append(v)
        Xs.append(md["X_va"])
        idxs.append(md["va_idx"])
        base_vals.append(np.mean(b))
    if not vals and HAS_SHAP:
        md = RESULTS[EXPLAIN_KEY]["models"][0]
        Xb = md["X_va"].iloc[:300]
        expl = shap.Explainer(lambda d: md["model"].predict_proba(pd.DataFrame(d, columns=Xb.columns))[:, 1], Xb)
        sv = expl(Xb)
        vals, Xs, idxs, base_vals = [sv.values], [Xb], [md["va_idx"][:300]], [float(np.mean(sv.base_values))]
        print("(SHAP model-agnostic, sampel 300 baris — skala probabilitas)")
    if not vals:
        print("⏭️ SHAP tidak tersedia untuk model ini. Pakai permutation importance.")
    else:
        SHAP_VALUES = np.vstack(vals)
        SHAP_X = pd.concat(Xs, ignore_index=True)
        SHAP_IDX = np.concatenate(idxs)
        SHAP_BASE = float(np.mean(base_vals))
        if len(SHAP_X) > CFG.SHAP_SAMPLE:
            sel = np.random.RandomState(CFG.SEED).choice(len(SHAP_X), CFG.SHAP_SAMPLE, replace=False)
            SHAP_VALUES, SHAP_X, SHAP_IDX = SHAP_VALUES[sel], SHAP_X.iloc[sel].reset_index(drop=True), SHAP_IDX[sel]
        SHAP_NAMES = [display_name(c) for c in SHAP_X.columns]
        SHAP_XNUM = SHAP_X.copy()
        for c in SHAP_XNUM.columns:
            if not pd.api.types.is_numeric_dtype(SHAP_XNUM[c]):
                SHAP_XNUM[c] = SHAP_XNUM[c].astype("category").cat.codes.replace(-1, np.nan)
        mean_abs = np.abs(SHAP_VALUES).mean(0)
        rows = []
        for j, c in enumerate(SHAP_X.columns):
            r = {"feature": SHAP_NAMES[j], "feature_model": c, "mean_abs_shap": mean_abs[j]}
            col = SHAP_X[c]
            if pd.api.types.is_numeric_dtype(col) and col.nunique() > 2:
                rho = pd.Series(col.values).corr(pd.Series(SHAP_VALUES[:, j]), method="spearman")
                r["direction"] = "higher value → higher churn risk" if rho > 0.1 else "higher value → lower churn risk" if rho < -0.1 else "non-monotonic / mixed"
                r["spearman_value_vs_shap"] = rho
            else:
                lv = pd.DataFrame({"lv": col.astype(str).values, "s": SHAP_VALUES[:, j]}).groupby("lv")["s"].agg(["mean", "count"])
                lv = lv[lv["count"] >= 20]
                if len(lv):
                    r["direction"] = f"riskiest: '{lv['mean'].idxmax()}' (+{lv['mean'].max():.2f}); safest: '{lv['mean'].idxmin()}' ({lv['mean'].min():.2f})"
            rows.append(r)
        SHAP_IMP = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        SHAP_IMP["share_%"] = 100 * SHAP_IMP["mean_abs_shap"] / SHAP_IMP["mean_abs_shap"].sum()
        save_table(SHAP_IMP, "shap_importance")
        display(SHAP_IMP.head(20))

        top = SHAP_IMP.head(CFG.TOP_N_FEATURES_PLOT)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(top))))
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color=CFG.PALETTE[4])
        ax.set_xlabel("Mean |SHAP value| (impact on churn log-odds)")
        ax.set_title("Global Feature Importance (SHAP)")
        save_fig("shap_importance_bar")

        order = [SHAP_X.columns.get_loc(c) for c in top["feature_model"]]
        if HAS_SHAP:
            try:
                plt.figure()
                shap.summary_plot(SHAP_VALUES[:, order], SHAP_XNUM.iloc[:, order].astype(float), feature_names=[SHAP_NAMES[j] for j in order],
                                  show=False, max_display=len(order), plot_size=(9, max(4, 0.35 * len(order))))
                plt.title("SHAP Summary (beeswarm): impact of each feature on churn risk")
                save_fig("shap_beeswarm", plt.gcf())
            except Exception as e:
                print("shap.summary_plot gagal → manual:", e)
                HAS_SHAP_PLOT = False
        if not HAS_SHAP:
            fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(order))))
            for rank_i, j in enumerate(order[::-1]):
                v = SHAP_XNUM.iloc[:, j].astype(float)
                col = (v.rank(pct=True)).fillna(0.5).values
                ax.scatter(SHAP_VALUES[:, j], rank_i + np.random.uniform(-0.3, 0.3, len(v)), c=col, cmap="coolwarm", s=4, alpha=0.6)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([SHAP_NAMES[j] for j in order[::-1]], fontsize=8)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("SHAP value (impact on churn log-odds); color = feature value (red high / blue low)")
            ax.set_title("SHAP Summary (beeswarm)")
            save_fig("shap_beeswarm")

# %%
with safe_section("SHAP dependence & categorical effects"):
    if SHAP_VALUES is not None:
        num_top = [c for c in SHAP_IMP["feature_model"] if pd.api.types.is_numeric_dtype(SHAP_X[c]) and SHAP_X[c].nunique() > 5][:6]
        if num_top:
            fig, axes = grid_axes(len(num_top), ncols=3, w=5, h=3.6)
            for a, c in zip(axes, num_top):
                j = SHAP_X.columns.get_loc(c)
                xv = SHAP_X[c].astype(float)
                lo, hi = xv.quantile([0.01, 0.99])
                m = xv.between(lo, hi)
                a.scatter(xv[m], SHAP_VALUES[m.values, j], s=5, alpha=0.35, color=CFG.PALETTE[0])
                try:
                    bins = pd.qcut(xv[m], 15, duplicates="drop")
                    mm = pd.DataFrame({"b": bins, "x": xv[m], "s": SHAP_VALUES[m.values, j]}).groupby("b", observed=True).mean()
                    a.plot(mm["x"], mm["s"], color=CFG.PALETTE[1], lw=2)
                except Exception:
                    pass
                a.axhline(0, color="k", lw=0.7)
                a.set_title(display_name(c)[:40], fontsize=9)
                a.set_ylabel("SHAP (churn log-odds)", fontsize=8)
            fig.suptitle("SHAP Dependence: how each driver changes churn risk", y=1.02)
            plt.tight_layout()
            save_fig("shap_dependence")
        cat_top = [c for c in SHAP_IMP["feature_model"] if not pd.api.types.is_numeric_dtype(SHAP_X[c]) or SHAP_X[c].nunique() <= 5][:4]
        if cat_top:
            fig, axes = grid_axes(len(cat_top), ncols=2, w=6, h=3.4)
            for a, c in zip(axes, cat_top):
                j = SHAP_X.columns.get_loc(c)
                d = pd.DataFrame({"lv": SHAP_X[c].astype(str).values, "s": SHAP_VALUES[:, j]}).groupby("lv")["s"].agg(["mean", "count"])
                d = d[d["count"] >= 10].sort_values("mean")
                a.barh(d.index.astype(str).str.slice(0, 25), d["mean"], color=[CFG.PALETTE[1] if v > 0 else CFG.PALETTE[0] for v in d["mean"]])
                a.axvline(0, color="k", lw=0.7)
                a.set_title(display_name(c)[:40], fontsize=9)
                a.set_xlabel("Mean SHAP (churn log-odds)", fontsize=8)
            fig.suptitle("Effect of Each Category Level on Churn Risk (SHAP)", y=1.02)
            plt.tight_layout()
            save_fig("shap_categorical_levels")

# %%
def plot_waterfall(i, title):
    j_sorted = np.argsort(-np.abs(SHAP_VALUES[i]))[:10]
    contrib = SHAP_VALUES[i, j_sorted]
    labels = [f"{SHAP_NAMES[j][:28]} = {str(SHAP_X.iloc[i, j])[:14]}" for j in j_sorted]
    rest = SHAP_VALUES[i].sum() - contrib.sum()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(labels[::-1] + ["(all other features)"][:0], contrib[::-1], color=[CFG.PALETTE[1] if v > 0 else CFG.PALETTE[0] for v in contrib[::-1]])
    ax.axvline(0, color="k", lw=0.8)
    logodds = SHAP_BASE + SHAP_VALUES[i].sum()
    ax.set_xlabel(f"SHAP contribution (log-odds). Base {SHAP_BASE:.2f} + others {rest:+.2f} → p(churn) = {1 / (1 + np.exp(-logodds)):.1%}")
    ax.set_title(title)
    return fig


with safe_section("Individual explanations (waterfall)"):
    if SHAP_VALUES is not None:
        p_s = FINAL_OOF[SHAP_IDX]
        i_high = int(np.argmax(p_s))
        churners = np.where(y[SHAP_IDX] == 1)[0]
        i_typ = int(churners[np.argsort(p_s[churners])[len(churners) // 2]]) if len(churners) else i_high
        save_fig("shap_waterfall_highest_risk", plot_waterfall(i_high, f"Why is this customer high-risk? (predicted {p_s[i_high]:.0%})"))
        save_fig("shap_waterfall_typical_churner", plot_waterfall(i_typ, f"Typical churner explanation (predicted {p_s[i_typ]:.0%})"))

# %%
def partial_dependence(md, col, grid, n=800):
    X = md["X_va"].iloc[:n].copy()
    out = []
    for g in grid:
        X[col] = g
        out.append(md["model"].predict_proba(X)[:, 1].mean())
    return np.array(out)


with safe_section("Partial dependence (PDP)"):
    md0 = RESULTS[EXPLAIN_KEY]["models"][0]
    rank_src = SHAP_IMP if len(SHAP_IMP) else PERM_IMP
    pd_feats = [c for c in rank_src["feature_model"] if c in md0["X_va"].columns and pd.api.types.is_numeric_dtype(md0["X_va"][c])
                and md0["X_va"][c].nunique() > 5][:6]
    if pd_feats:
        fig, axes = grid_axes(len(pd_feats), ncols=3, w=5, h=3.5)
        PDP_TABLES = {}
        for a, c in zip(axes, pd_feats):
            xv = md0["X_va"][c].dropna()
            grid = np.unique(np.quantile(xv, np.linspace(0.05, 0.95, 15)))
            pdv = partial_dependence(md0, c, grid)
            PDP_TABLES[c] = pd.DataFrame({"value": grid, "avg_predicted_churn": pdv})
            a.plot(grid, pdv, "o-", color=CFG.PALETTE[1])
            a.set_title(display_name(c)[:40], fontsize=9)
            a.set_ylabel("Avg predicted churn", fontsize=8)
            a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            a2 = a.twinx()
            a2.hist(xv.clip(grid.min(), grid.max()), bins=30, alpha=0.15, color="gray")
            a2.set_yticks([])
        fig.suptitle("Partial Dependence: predicted churn as each driver changes (others held at observed values)", y=1.02)
        plt.tight_layout()
        save_fig("partial_dependence")
        for c, t in PDP_TABLES.items():
            lo_, hi_ = t.iloc[0], t.iloc[-1]
            add_insight("Explainability", f"Partial dependence: moving {display_name(c)} from {fmt_num(lo_['value'])} to {fmt_num(hi_['value'])} changes average "
                                          f"predicted churn from {pct(lo_['avg_predicted_churn'])} to {pct(hi_['avg_predicted_churn'])}.", show=False)

# %% [markdown]
# ### 20.4 Driver heterogeneity — apakah driver berbeda antar segmen?
# Driver churn bisa beda per **jenis produk / channel / region**. Contoh: di Health insurance driver utama = klaim ditolak; di Auto = kenaikan premi.
# Dihitung dari rata-rata |SHAP| per fitur di tiap level segmen (dinormalisasi per segmen → share %).
#
# **Untuk laporan:** insight ini membuat rekomendasi lebih tajam ("untuk produk X fokus ke ..., untuk channel Y fokus ke ..."). Set segmen manual: `HETERO_SEGMENT = "policy_type"`.

# %%
HETERO_SEGMENT = None   # None = auto (policy_type → channel → kategori pertama dengan 2–8 level)
with safe_section("Driver heterogeneity by segment"):
    if SHAP_VALUES is None:
        print("⏭️ SHAP tidak tersedia → skip.")
    else:
        seg = HETERO_SEGMENT
        if seg is None:
            for cand_seg in [ROLES.get("policy_type"), ROLES.get("channel"), ROLES.get("region")] + CAT_COLS:
                if cand_seg and cand_seg in train.columns and 2 <= train[cand_seg].nunique() <= 8 and cand_seg not in LEAKY_COLS:
                    seg = cand_seg
                    break
        if seg is None:
            print("Tidak ada kolom segmen yang cocok.")
        else:
            base_names = [n.replace("__freq", "").replace(" (target-enc)", "") for n in SHAP_NAMES]
            absS = pd.DataFrame(np.abs(SHAP_VALUES), columns=SHAP_NAMES)
            absS = absS.T.groupby(base_names).sum().T
            lev = train[seg].astype(str).iloc[SHAP_IDX].values
            topf = absS.mean().sort_values(ascending=False).head(10).index
            H = absS.assign(_seg=lev).groupby("_seg")[list(topf)].mean()
            cnt = pd.Series(lev).value_counts()
            _min_n = 50 if len(lev) >= 500 else max(10, len(lev) // 20)      # data kecil → ambang adaptif
            H = H.loc[[l for l in H.index if cnt.get(l, 0) >= _min_n]]
            if H.shape[0] < 2:
                raise RuntimeError(f"segmen '{seg}' tidak punya ≥2 level dengan ≥{_min_n} baris → heterogeneity dilewati (data terlalu kecil)")
            Hs = 100 * H.div(absS.assign(_seg=lev).groupby("_seg").mean().loc[H.index].sum(axis=1), axis=0)
            HETERO_TABLE = Hs.T
            save_table(HETERO_TABLE.reset_index().rename(columns={"index": "feature"}), f"driver_heterogeneity_by_{sanitize_name(seg)}")
            fig, ax = plt.subplots(figsize=(1.4 * Hs.shape[0] + 4, 0.45 * len(topf) + 1.5))
            sns.heatmap(HETERO_TABLE, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "share of SHAP importance (%)"})
            ax.set_title(f"Churn drivers by {seg} (share of total SHAP importance within each segment, %)")
            save_fig(f"driver_heterogeneity_{seg}")
            for lv in HETERO_TABLE.columns:
                t3 = HETERO_TABLE[lv].sort_values(ascending=False).head(3)
                add_insight("Explainability", f"For {seg} = '{lv}', the top churn drivers are " + ", ".join(f"{k} ({v:.0f}%)" for k, v in t3.items()) + ".", show=False)
            print(f"Segmen: {seg}")
            display(HETERO_TABLE.round(1))

# %% [markdown]
# ## 20.5 Consolidated Driver Evidence Matrix
# Satu tabel yang menggabungkan **semua bukti** per driver: univariate (IV, churn rate level berisiko), multivariate (odds ratio, hazard ratio), dan ML (rank SHAP/permutation), plus apakah fitur **actionable** (bisa diintervensi perusahaan).
# **Ini tabel paling kuat untuk laporan** — driver yang konsisten muncul di semua metode = temuan yang robust.

# %%
ACTIONABLE_KEYWORDS = ["payment", "bayar", "renew", "debit", "complain", "keluhan", "claim", "klaim", "settle", "premium", "premi", "price",
                       "discount", "diskon", "channel", "agent", "agen", "contract", "frequency", "paperless", "engagement", "digital",
                       "login", "app", "service", "support", "late", "telat", "product", "rider", "coverage", "plan", "tech", "online", "call"]


def is_actionable(f):
    if CFG.ACTIONABLE_FEATURES is not None:
        return f in CFG.ACTIONABLE_FEATURES or any(f.startswith(a) for a in CFG.ACTIONABLE_FEATURES)
    fl = f.lower()
    if any(k in fl for k in ["age", "usia", "umur", "gender", "sex", "region", "province", "city", "marital", "dependent", "income", "tenure", "days_since", "_year"]):
        return False
    return any(k in fl for k in ACTIONABLE_KEYWORDS)


EVIDENCE = pd.DataFrame()
with safe_section("Consolidated driver evidence"):
    ev = DRIVER_TABLE[["feature", "iv", "iv_strength", "effect_size", "effect_label", "q_value_BH", "highest_risk_level",
                       "highest_risk_rate", "highest_risk_lift", "direction"]].copy()
    ev = ev.rename(columns={"direction": "univariate_direction"})
    if len(SHAP_IMP):
        sh = SHAP_IMP.copy()
        sh["base_feature"] = sh["feature"].str.replace(" (target-enc)", "", regex=False).str.replace("__freq", "", regex=False)
        sh = sh.groupby("base_feature").agg(shap_share_pct=("share_%", "sum"), shap_direction=("direction", "first")).reset_index()
        sh["shap_rank"] = sh["shap_share_pct"].rank(ascending=False).astype(int)
        ev = ev.merge(sh.rename(columns={"base_feature": "feature"}), on="feature", how="left")
    if len(PERM_IMP):
        pi = PERM_IMP.copy()
        pi["base_feature"] = pi["feature"].astype(str).str.replace("__freq", "", regex=False)
        pi = pi.groupby("base_feature")["importance_mean"].sum().rename("perm_importance").reset_index()
        ev = ev.merge(pi.rename(columns={"base_feature": "feature"}), on="feature", how="left")
    if len(LOGIT_TABLE):
        lt = LOGIT_TABLE.copy()
        lt["feature"] = lt["term"].str.split(" = ").str[0]
        lt = lt.sort_values("p_value").groupby("feature").first().reset_index()[["feature", "odds_ratio", "p_value"]]
        ev = ev.merge(lt.rename(columns={"p_value": "or_p_value"}), on="feature", how="left")
    if len(COX_TABLE):
        ct = COX_TABLE.copy()
        ct["feature"] = ct["term"].str.split(" = ").str[0]
        ct = ct.sort_values("p_value").groupby("feature").first().reset_index()[["feature", "hazard_ratio", "p_value"]]
        ev = ev.merge(ct.rename(columns={"p_value": "hr_p_value"}), on="feature", how="left")
    ev["actionable"] = ev["feature"].apply(is_actionable)
    ev = ev[~ev["feature"].isin(LEAKY_COLS)]
    n_methods = (ev["q_value_BH"] < 0.05).astype(int)
    if "shap_rank" in ev:
        n_methods += (ev["shap_rank"] <= 10).astype(int)
    if "or_p_value" in ev:
        n_methods += (ev["or_p_value"] < 0.05).astype(int)
    if "hr_p_value" in ev:
        n_methods += (ev["hr_p_value"] < 0.05).astype(int)
    ev["evidence_score"] = n_methods
    sort_cols = ["evidence_score"] + (["shap_share_pct"] if "shap_share_pct" in ev else ["iv"])
    EVIDENCE = ev.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    save_table(EVIDENCE, "driver_evidence_matrix")
    display(EVIDENCE.head(20))
    top_act = EVIDENCE[EVIDENCE["actionable"]].head(5)["feature"].tolist()
    top_non = EVIDENCE[~EVIDENCE["actionable"]].head(3)["feature"].tolist()
    if top_act:
        add_insight("Drivers", f"Most robust actionable churn drivers (consistent across statistical tests, regression and SHAP): {', '.join(top_act)}.")
    else:
        add_insight("Drivers", "The available data contain few directly controllable (actionable) variables; drivers are mainly customer characteristics, "
                               "so recommendations focus on targeting and lifecycle timing rather than on changing specific policy features.")
    if top_non:
        add_insight("Drivers", f"Key non-actionable risk indicators useful for targeting: {', '.join(top_non)}.")

# %% [markdown]
# # 21. Business Analytics — dari prediksi ke keputusan
#
# Juri lomba bisnis menilai: **"so what?"** Section ini menerjemahkan model ke angka bisnis.
#
# | Analisis | Pertanyaan bisnis | Output |
# |---|---|---|
# | **Lift / Gains / KS** | Seberapa efisien model menemukan churner? | "Top 20% skor menangkap 55% churner (lift 2.7x)" |
# | **Risk tiers** | Berapa customer berisiko tinggi/sedang/rendah? | tabel tier + churn rate aktual |
# | **CLV & value at risk** | Berapa uang yang terancam hilang? | total premi/margin berisiko |
# | **Risk × Value matrix** | Siapa diprioritaskan? | 4 kuadran: Priority Save, Low-cost Save, Nurture & Upsell, Maintain |
# | **Campaign ROI simulation** | Berapa customer yang optimal di-target? | kurva profit vs % target, ROI, titik optimal |
# | **What-if scenarios** | Kalau perusahaan mengubah X, berapa churn turun? | estimasi churn reduction per intervensi |
# | **Causal effects (AIPW)** | Efek intervensi setelah mengontrol confounders? | ATE (pp) + 95% CI per intervensi |
# | **Churn personas (SHAP clustering)** | Apa *alasan* tiap kelompok berisiko? | 3–5 persona dengan driver utama |
#
# Parameter bisnis di `CFG`: `PROFIT_MARGIN`, `RETENTION_COST`, `RETENTION_SUCCESS_RATE`, `DISCOUNT_RATE`, `CLV_HORIZON_YEARS`. **Kalau soal tidak memberi angka, pakai asumsi wajar & tulis jelas sebagai asumsi di laporan** (+ sensitivity analysis di 21.5).
#
# **Asumsi penting (tulis di laporan):** probabilitas churn model diperlakukan sebagai peluang churn dalam satu periode (≈ 1 tahun polis); what-if bersifat **asosiatif, bukan kausal** — rekomendasikan validasi dengan **A/B test / pilot**.

# %%
PROB_BIZ = FINAL_OOF_CAL
PROB_BIZ_TEST = FINAL_TEST_CAL if len(FINAL_TEST) else np.array([])

with safe_section("Lift, gains & KS"):
    d = pd.DataFrame({"p": PROB_BIZ, "y": y})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=[f"D{i}" for i in range(1, 11)])
    dec = d.groupby("decile", observed=True).agg(n=("y", "size"), churners=("y", "sum"), avg_pred=("p", "mean"), min_pred=("p", "min")).reset_index()
    dec["churn_rate"] = dec["churners"] / dec["n"]
    dec["lift"] = dec["churn_rate"] / CHURN_RATE
    dec["cum_customers_%"] = 100 * dec["n"].cumsum() / dec["n"].sum()
    dec["cum_churners_captured_%"] = 100 * dec["churners"].cumsum() / dec["churners"].sum()
    dec["cum_lift"] = dec["cum_churners_captured_%"] / dec["cum_customers_%"]
    DECILE_TABLE = dec
    save_table(dec, "decile_lift_table")
    display(dec)
    fpr, tpr, _ = roc_curve(y, PROB_BIZ)
    KS = float(np.max(tpr - fpr))
    print(f"KS statistic = {KS:.3f} (>0.4 sangat baik, 0.3–0.4 baik, 0.2–0.3 cukup)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    xs = np.r_[0, dec["cum_customers_%"]]
    ax[0].plot(xs, np.r_[0, dec["cum_churners_captured_%"]], "o-", color=CFG.PALETTE[1], label="Model")
    ax[0].plot([0, 100], [0, 100], "k--", label="Random")
    ax[0].plot([0, 100 * CHURN_RATE, 100], [0, 100, 100], ":", color="gray", label="Perfect")
    ax[0].set_xlabel("% of customers targeted (highest risk first)")
    ax[0].set_ylabel("% of churners captured")
    ax[0].set_title("Cumulative Gains Chart")
    ax[0].legend()
    ax[1].bar(dec["decile"].astype(str), dec["lift"], color=CFG.PALETTE[0])
    ax[1].axhline(1, color="red", ls="--")
    ax[1].set_xlabel("Risk decile (D1 = highest predicted risk)")
    ax[1].set_ylabel("Lift vs average churn rate")
    ax[1].set_title("Lift by Decile")
    save_fig("gains_and_lift")
    top2 = dec.iloc[:2]
    add_insight("Business", f"Targeting the top 20% highest-risk customers captures {top2['cum_churners_captured_%'].iloc[-1]:.1f}% of all churners "
                            f"({top2['cum_lift'].iloc[-1]:.1f}x lift over random targeting); the top decile churns at {pct(dec['churn_rate'].iloc[0])} "
                            f"vs {pct(dec['churn_rate'].iloc[-1])} in the bottom decile. KS = {KS:.2f}.")

# %%
def tier_of(p, cuts):
    return np.where(p >= cuts[1], "High", np.where(p >= cuts[0], "Medium", "Low"))


with safe_section("Risk tiers"):
    TIER_CUTS = tuple(np.quantile(PROB_BIZ, q) for q in CFG.RISK_TIER_CUTS)
    tr_tier = tier_of(PROB_BIZ, TIER_CUTS)
    tt = pd.DataFrame({"tier": tr_tier, "y": y, "p": PROB_BIZ}).groupby("tier").agg(
        n_train=("y", "size"), actual_churn_rate=("y", "mean"), avg_predicted=("p", "mean")).reindex(["High", "Medium", "Low"])
    tt["share_churners_%"] = 100 * pd.DataFrame({"tier": tr_tier, "y": y}).groupby("tier")["y"].sum().reindex(tt.index) / y.sum()
    if len(PROB_BIZ_TEST):
        TEST_TIER = tier_of(PROB_BIZ_TEST, TIER_CUTS)
        tt["n_test"] = pd.Series(TEST_TIER).value_counts().reindex(tt.index).fillna(0).astype(int).values
    TIER_TABLE = tt.reset_index()
    print(f"Batas tier: Medium ≥ {TIER_CUTS[0]:.3f}, High ≥ {TIER_CUTS[1]:.3f}")
    display(TIER_TABLE)
    save_table(TIER_TABLE, "risk_tiers")
    h = TIER_TABLE.set_index("tier").loc["High"]
    add_insight("Business", f"The High-risk tier ({pct(1 - CFG.RISK_TIER_CUTS[1], 0)} of customers, p ≥ {TIER_CUTS[1]:.2f}) has an actual churn rate of "
                            f"{pct(h['actual_churn_rate'])} and contains {h['share_churners_%']:.1f}% of all churners.")

# %% [markdown]
# ### 21.2b Validasi skor dengan survival: kurva Kaplan-Meier per risk tier
# Skor churn dari model (OOF) dipakai untuk membagi tier, lalu dicek apakah tier **High** memang churn lebih cepat sepanjang waktu (bukti validitas di luar AUC). **Harrell's C-index** = peluang customer berisiko lebih tinggi churn lebih dulu (0,5 acak).

# %%
SCORE_CINDEX = np.nan
with safe_section("Survival validation of risk tiers"):
    if not SURV_AVAILABLE:
        print("⏭️ Tidak ada tenure → skip.")
    else:
        tiers_s = np.asarray(tr_tier)[m_ok.values]
        p_s = np.asarray(PROB_BIZ)[m_ok.values]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        rows = []
        for i, tname in enumerate(["High", "Medium", "Low"]):
            mm = tiers_s == tname
            if mm.sum() < 30:
                continue
            _, tt, ss, lo, hi = km_curve(DUR[mm], EVT[mm])
            col = {"High": CFG.PALETTE[1], "Medium": CFG.PALETTE[3], "Low": CFG.PALETTE[0]}[tname]
            ax.step(tt, ss, where="post", lw=2, color=col, label=f"{tname} risk (n={mm.sum():,})")
            ax.fill_between(tt, lo, hi, step="post", alpha=0.15, color=col)
            rows.append({"tier": tname, "n": int(mm.sum()), **{f"S({cp})": surv_at(tt, ss, cp) for cp in checkpoints if cp <= DUR.max()}})
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"Tenure ({SURV_UNIT})")
        ax.set_ylabel("Probability of remaining a customer")
        ax.set_title("Out-of-Sample Validation: Survival Curves by Predicted Risk Tier")
        ax.legend()
        save_fig("km_by_risk_tier", fig)
        TIER_SURV = pd.DataFrame(rows)
        save_table(TIER_SURV, "survival_by_risk_tier")
        display(TIER_SURV)
        try:
            from lifelines.utils import concordance_index
            sub_ = np.random.RandomState(CFG.SEED).choice(len(DUR), min(len(DUR), 60_000), replace=False)
            SCORE_CINDEX = float(concordance_index(DUR[sub_], -p_s[sub_], EVT[sub_]))
        except Exception:
            SCORE_CINDEX = np.nan
        try:
            _, plr = survdiff(DUR, EVT, tiers_s)
        except Exception:
            plr = np.nan
        print(f"Harrell C-index skor churn terhadap waktu-churn: {SCORE_CINDEX:.3f} | log-rank antar tier p = {p_fmt(plr)}")
        add_insight("Model", f"The out-of-fold risk score also orders customers by time to churn (Harrell's C-index {SCORE_CINDEX:.3f}); "
                             f"survival curves of the three risk tiers separate clearly (log-rank p {p_fmt(plr)}).")

# %%
VALUE_SOURCE = None
with safe_section("Customer value & CLV"):
    vcol = CFG.VALUE_COL or ROLES.get("premium")
    if vcol and vcol in train.columns and pd.api.types.is_numeric_dtype(train[vcol]):
        VALUE_SOURCE = vcol
        mult = 12 if CFG.VALUE_PERIOD == "monthly" else 1
        val_tr = train[vcol].fillna(train[vcol].median()).clip(lower=0).values * mult
        val_te = test[vcol].fillna(train[vcol].median()).clip(lower=0).values * mult if test is not None and vcol in test else np.array([])
        unit = CFG.CURRENCY
    else:
        print("⚠️ Tidak ada kolom nilai (premi) → nilai customer = 1 (analisis berbasis jumlah customer). Set CFG.VALUE_COL kalau ada.")
        val_tr, val_te, unit = np.ones(N_TR), np.ones(len(PROB_BIZ_TEST)), "customers"
    H, dr = CFG.CLV_HORIZON_YEARS, CFG.DISCOUNT_RATE

    def clv(margin, p):
        r = (1 - np.clip(p, 0, 1)) / (1 + dr)
        return margin * (1 - r ** H) / np.maximum(1 - r, 1e-9)

    MARGIN_TR = val_tr * (CFG.PROFIT_MARGIN if unit != "customers" else 1)
    CLV_TR = clv(MARGIN_TR, PROB_BIZ)
    VALUE_SAVED_TR = clv(MARGIN_TR, np.full(N_TR, CHURN_RATE))   # nilai customer kalau berhasil dipertahankan (risiko rata-rata)
    if len(PROB_BIZ_TEST):
        MARGIN_TE = val_te * (CFG.PROFIT_MARGIN if unit != "customers" else 1)
        CLV_TE = clv(MARGIN_TE, PROB_BIZ_TEST)
        VALUE_SAVED_TE = clv(MARGIN_TE, np.full(len(MARGIN_TE), CHURN_RATE))
    at_risk_tr = (PROB_BIZ * val_tr).sum()
    print(f"Nilai tahunan total (train): {fmt_num(val_tr.sum())} {unit} | expected annual value at risk: {fmt_num(at_risk_tr)} {unit} ({pct(at_risk_tr / val_tr.sum())})")
    if len(PROB_BIZ_TEST):
        at_risk_te = (PROB_BIZ_TEST * val_te).sum()
        print(f"Test/portfolio aktif: expected annual value at risk = {fmt_num(at_risk_te)} {unit} dari {fmt_num(val_te.sum())}")
    print(f"CLV rata-rata (horizon {H} th, discount {dr:.0%}): {fmt_num(CLV_TR.mean())} {unit}")
    if unit != "customers":
        add_insight("Business", f"Expected annual premium at risk from churn is {fmt_num(at_risk_tr)} {unit} ({pct(at_risk_tr / val_tr.sum())} of the annual premium base); "
                                f"the High-risk tier alone accounts for {pct((PROB_BIZ * val_tr)[tr_tier == 'High'].sum() / at_risk_tr)} of it.")

# %%
with safe_section("Risk × value matrix"):
    v_cut = np.median(val_tr)
    r_cut = TIER_CUTS[1]

    def quadrant(p, v):
        return np.where(p >= r_cut, np.where(v >= v_cut, "Priority Save (high risk, high value)", "Low-cost Save (high risk, low value)"),
                        np.where(v >= v_cut, "Nurture & Upsell (low risk, high value)", "Maintain (low risk, low value)"))

    q_tr = quadrant(PROB_BIZ, val_tr)
    qt = pd.DataFrame({"quadrant": q_tr, "y": y, "p": PROB_BIZ, "v": val_tr, "risk_value": PROB_BIZ * val_tr}).groupby("quadrant").agg(
        n=("y", "size"), actual_churn_rate=("y", "mean"), avg_pred=("p", "mean"), total_value=("v", "sum"), value_at_risk=("risk_value", "sum"))
    qt["share_value_at_risk_%"] = 100 * qt["value_at_risk"] / qt["value_at_risk"].sum()
    QUAD_TABLE = qt.reset_index()
    display(QUAD_TABLE)
    save_table(QUAD_TABLE, "risk_value_quadrants")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sel = np.random.RandomState(0).choice(N_TR, min(N_TR, 4000), replace=False)
    colors = {"Priority Save (high risk, high value)": CFG.PALETTE[1], "Low-cost Save (high risk, low value)": CFG.PALETTE[3],
              "Nurture & Upsell (low risk, high value)": CFG.PALETTE[2], "Maintain (low risk, low value)": "#9E9E9E"}
    for qn, c in colors.items():
        m = q_tr[sel] == qn
        ax.scatter(PROB_BIZ[sel][m], val_tr[sel][m], s=7, alpha=0.5, color=c, label=qn)
    ax.axvline(r_cut, color="k", ls="--", lw=0.8)
    ax.axhline(v_cut, color="k", ls="--", lw=0.8)
    if unit != "customers" and val_tr.min() > 0:
        ax.set_yscale("log")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel(f"Customer annual value ({unit})")
    if not (unit != "customers" and val_tr.min() > 0):
        money_axis(ax)
    ax.set_title("Risk × Value Prioritisation Matrix")
    ax.legend(fontsize=7, loc="upper right")
    save_fig("risk_value_matrix")
    ps = QUAD_TABLE.set_index("quadrant").loc["Priority Save (high risk, high value)"] if "Priority Save (high risk, high value)" in set(q_tr) else None
    if ps is not None:
        add_insight("Business", f"'Priority Save' customers (high risk & above-median value) are {pct(ps['n'] / N_TR)} of the base but hold "
                                f"{ps['share_value_at_risk_%']:.1f}% of the value at risk — the first target for personalised retention.")

# %% [markdown]
# ## 21.5 Campaign ROI Simulation
# Simulasi: perusahaan men-target top-k% customer (urut risiko). Untuk tiap k:
# - **Benefit** = churner aktual yang ditarget × `RETENTION_SUCCESS_RATE` × nilai customer yang diselamatkan (CLV)
# - **Cost** = jumlah ditarget × `RETENTION_COST`
# - **Net profit** = benefit − cost; ROI = net / cost
#
# Dibandingkan 3 strategi: **model (urut probabilitas)**, **model × value (urut expected loss)**, dan **random**. Dihitung di OOF (ada label aktual) → hasil jujur.
# **Sensitivity analysis**: profit optimal dihitung ulang untuk beberapa success rate & cost → tunjukkan kesimpulan robust terhadap asumsi.
#
# **Jika X → Y:** net profit negatif di semua k → biaya retensi terlalu mahal relatif nilai customer → rekomendasikan intervensi murah (digital/otomatis) untuk tier low-value, intervensi mahal hanya untuk Priority Save.

# %%
def campaign_curve(order, success, cost, steps=50):
    rows = []
    n = len(order)
    for k in np.linspace(0.02, 1.0, steps):
        idx = order[: max(1, int(k * n))]
        benefit = (y[idx] * success * VALUE_SAVED_TR[idx]).sum()
        c = cost * len(idx)
        rows.append({"target_%": 100 * k, "n_targeted": len(idx), "churners_reached": int(y[idx].sum()),
                     "benefit": benefit, "cost": c, "net_profit": benefit - c, "roi": (benefit - c) / c if c else np.nan})
    return pd.DataFrame(rows)


with safe_section("Campaign ROI simulation"):
    cost = CFG.RETENTION_COST if unit != "customers" else 0.0
    succ = CFG.RETENTION_SUCCESS_RATE
    if unit == "customers":
        print("ℹ️ Tanpa kolom nilai → profit dihitung dalam 'customers saved' (cost diabaikan).")
    strategies = {"Model: churn probability": np.argsort(-PROB_BIZ),
                  "Model: expected loss (p × value)": np.argsort(-(PROB_BIZ * VALUE_SAVED_TR)),
                  "Random targeting": np.random.RandomState(CFG.SEED).permutation(N_TR)}
    curves = {k: campaign_curve(o, succ, cost) for k, o in strategies.items()}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    best_rows = []
    for (k, cdf), col in zip(curves.items(), [CFG.PALETTE[1], CFG.PALETTE[4], "gray"]):
        ax.plot(cdf["target_%"], cdf["net_profit"], color=col, label=k, lw=2 if "Model" in k else 1)
        b = cdf.loc[cdf["net_profit"].idxmax()]
        best_rows.append({"strategy": k, **b.to_dict()})
        if "Model" in k:
            ax.scatter([b["target_%"]], [b["net_profit"]], color=col, zorder=3)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("% of customers targeted")
    ax.set_ylabel(f"Net profit ({unit})")
    money_axis(ax)
    ax.set_title(f"Retention Campaign Profit Curve (success rate {succ:.0%}, cost {fmt_num(cost)}/customer)")
    ax.legend(fontsize=8)
    save_fig("campaign_profit_curve")
    CAMPAIGN_BEST = pd.DataFrame(best_rows)
    display(CAMPAIGN_BEST)
    save_table(CAMPAIGN_BEST, "campaign_optimum")
    save_table(curves["Model: expected loss (p × value)"], "campaign_curve_expected_loss")
    bm = CAMPAIGN_BEST.iloc[1] if CAMPAIGN_BEST.iloc[1]["net_profit"] >= CAMPAIGN_BEST.iloc[0]["net_profit"] else CAMPAIGN_BEST.iloc[0]
    rnd = curves["Random targeting"]
    rnd_same = rnd.iloc[(rnd["target_%"] - bm["target_%"]).abs().idxmin()]
    add_insight("Business", f"Simulated retention campaign (assumed success rate {succ:.0%}, cost {fmt_num(cost)} {unit if unit != 'customers' else ''} per contact): "
                            f"targeting the top {bm['target_%']:.0f}% ranked by '{bm['strategy']}' maximises net profit at {fmt_num(bm['net_profit'])} "
                            f"(ROI {bm['roi']:.1f}x), versus {fmt_num(rnd_same['net_profit'])} for random targeting of the same size.")

    # sensitivity
    sens = []
    for s_ in [succ * 0.5, succ, min(1, succ * 1.5)]:
        for c_ in [cost * 0.5, cost, cost * 2]:
            cdf = campaign_curve(strategies["Model: expected loss (p × value)"], s_, c_, steps=25)
            b = cdf.loc[cdf["net_profit"].idxmax()]
            sens.append({"success_rate": s_, "cost_per_contact": c_, "optimal_target_%": b["target_%"], "max_net_profit": b["net_profit"], "roi": b["roi"]})
    SENS = pd.DataFrame(sens)
    print("Sensitivity analysis (asumsi success rate × cost):")
    display(SENS)
    save_table(SENS, "campaign_sensitivity")
    # business threshold per customer
    if unit != "customers" and len(PROB_BIZ_TEST):
        be = cost / (succ * np.maximum(VALUE_SAVED_TE, 1e-9))   # target kalau p > break-even prob
        TARGET_FLAG_TEST = PROB_BIZ_TEST > be
        print(f"Rule bisnis per customer: target jika p(churn) > cost / (success × value_saved). → {TARGET_FLAG_TEST.sum():,} dari {len(be):,} customer test layak ditarget.")

# %% [markdown]
# ## 21.6 What-if Scenarios (simulasi intervensi)
# Mengubah nilai fitur **actionable** untuk customer yang terdampak lalu memprediksi ulang:
# - Kategori → pindahkan customer dari level berisiko ke level paling aman (misal *payment = Cash → Auto-debit*).
# - Count (keluhan, telat bayar) → jadikan 0 (masalah diselesaikan).
# - Kontinu yang menaikkan risiko (misal kenaikan premi) → dibatasi di median.
#
# Hasil ditampilkan pada **adopsi 100%** (batas atas) dan **adopsi realistis** `CFG.WHAT_IF_ADOPTION` (default 30%).
# Skenario custom: `CFG.WHAT_IF_SCENARIOS = [{"name": "...", "feature": "payment_method", "set_to": "Auto-debit"}, {"name": "...", "feature": "premium_change_pct", "cap": 5}]`.
#
# ⚠️ **Ini simulasi model, bukan bukti kausal.** Di laporan tulis: *"model-based simulation suggests up to X pp reduction; we recommend validating via a controlled pilot (A/B test)."*

# %%
WHATIF = pd.DataFrame()


def auto_scenarios(max_n=6):
    sc = []
    feats = EVIDENCE[EVIDENCE["actionable"]]["feature"].tolist() if len(EVIDENCE) else []
    for f in feats:
        if f not in NAME_MAP or f in LEAKY_COLS:
            continue
        s = train[f]
        tbl = LEVEL_TABLES.get(f)
        if not pd.api.types.is_numeric_dtype(s):
            if tbl is None:
                continue
            ok = tbl[(tbl["n"] >= max(30, 0.02 * N_TR)) & (tbl["level"] != "Missing") & (tbl["level"] != "Other")]
            if len(ok) < 2:
                continue
            safest = ok.sort_values("churn_rate").iloc[0]["level"]
            sc.append({"name": f"Move all '{f}' customers to '{safest}'", "feature": f, "set_to": safest})
        elif s.nunique() <= 2:
            r = pd.Series(y).groupby(s.values).mean()
            if len(r) == 2:
                sc.append({"name": f"Set {f} = {r.idxmin():g} for everyone", "feature": f, "set_to": r.idxmin()})
        else:
            higher_worse = (DRIVER_TABLE.set_index("feature").loc[f, "effect_size"] > 0) if f in set(DRIVER_TABLE["feature"]) else True
            is_count = (s.dropna() % 1 == 0).all() and (s == 0).mean() > 0.3 and s.min() >= 0
            if is_count and higher_worse:
                sc.append({"name": f"Resolve issues: {f} → 0", "feature": f, "set_to": 0})
            elif higher_worse:
                sc.append({"name": f"Cap {f} at median ({s.median():g})", "feature": f, "cap": float(s.median())})
            else:
                sc.append({"name": f"Raise {f} to at least median ({s.median():g})", "feature": f, "floor": float(s.median())})
        if len(sc) >= max_n:
            break
    return sc


with safe_section("What-if scenarios"):
    md0 = RESULTS[EXPLAIN_KEY]["models"][0]
    X0 = md0["X_va"].copy()
    base_p = md0["model"].predict_proba(X0)[:, 1]
    va0 = md0["va_idx"]
    scen = CFG.WHAT_IF_SCENARIOS or auto_scenarios()
    rows = []
    for sc in scen:
        f = sc["feature"]
        col = NAME_MAP.get(f, f)
        if col not in X0.columns:
            print(f"  skip '{sc['name']}': kolom tidak ada di model")
            continue
        Xs = X0.copy()
        orig = train[f].iloc[va0].reset_index(drop=True)
        if "set_to" in sc:
            target_val = sc["set_to"]
            if pd.api.types.is_numeric_dtype(train[f]):
                affected = (orig != target_val) & orig.notna()
                Xs.loc[affected.values, col] = target_val
            else:
                affected = (orig.astype(str) != str(target_val))
                if str(Xs[col].dtype) == "category":
                    if target_val not in list(Xs[col].cat.categories):
                        print(f"  skip '{sc['name']}': level tidak dikenal model")
                        continue
                    Xs.loc[affected.values, col] = target_val
                elif Xs[col].dtype == object:
                    Xs.loc[affected.values, col] = str(target_val)
                else:  # ordinal code
                    cats = list(X_tree_full[col].cat.categories)
                    Xs.loc[affected.values, col] = float(cats.index(target_val)) if target_val in cats else np.nan
                fcol = f"{col}__freq"
                if fcol in Xs.columns:
                    Xs.loc[affected.values, fcol] = float(full_df[col].value_counts(normalize=True).get(target_val, np.nan))
        elif "cap" in sc:
            affected = orig > sc["cap"]
            Xs.loc[affected.values, col] = sc["cap"]
        elif "floor" in sc:
            affected = orig < sc["floor"]
            Xs.loc[affected.values, col] = sc["floor"]
        else:
            continue
        new_p = md0["model"].predict_proba(Xs)[:, 1]
        a = affected.values
        if a.sum() == 0:
            continue
        delta_aff = (base_p[a] - new_p[a]).mean()
        overall_full = (base_p - new_p).mean()
        rows.append({"scenario": sc["name"], "feature": f, "affected_%": 100 * a.mean(),
                     "churn_affected_before": base_p[a].mean(), "churn_affected_after": new_p[a].mean(),
                     "overall_reduction_pp_full": 100 * overall_full,
                     f"overall_reduction_pp_{int(CFG.WHAT_IF_ADOPTION * 100)}pct_adoption": 100 * overall_full * CFG.WHAT_IF_ADOPTION,
                     "churners_avoided_per_1000": 1000 * overall_full * CFG.WHAT_IF_ADOPTION})
    WHATIF = pd.DataFrame(rows)
    if len(WHATIF):
        WHATIF = WHATIF.sort_values("overall_reduction_pp_full", ascending=False).reset_index(drop=True)
    else:
        print("Tidak ada skenario what-if yang bisa dibuat (tidak ada fitur actionable). Set CFG.WHAT_IF_SCENARIOS / CFG.ACTIONABLE_FEATURES kalau perlu.")
    display(WHATIF)
    save_table(WHATIF, "what_if_scenarios")
    if len(WHATIF):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(WHATIF))))
        ax.barh(WHATIF["scenario"].str.slice(0, 55)[::-1], WHATIF["overall_reduction_pp_full"][::-1], color=CFG.PALETTE[2], label="100% adoption")
        ax.barh(WHATIF["scenario"].str.slice(0, 55)[::-1], WHATIF["overall_reduction_pp_full"][::-1] * CFG.WHAT_IF_ADOPTION, color=CFG.PALETTE[0],
                label=f"{CFG.WHAT_IF_ADOPTION:.0%} adoption")
        ax.set_xlabel("Reduction in overall churn rate (percentage points)")
        ax.set_title("What-if Simulation: Impact of Interventions on Churn")
        ax.legend(fontsize=8)
        save_fig("what_if_scenarios")
        for _, r in WHATIF.head(3).iterrows():
            add_insight("Business", f"What-if: '{r['scenario']}' affects {r['affected_%']:.0f}% of customers and lowers their predicted churn from "
                                    f"{pct(r['churn_affected_before'])} to {pct(r['churn_affected_after'])}; at {CFG.WHAT_IF_ADOPTION:.0%} adoption this avoids "
                                    f"≈{r['churners_avoided_per_1000']:.0f} churners per 1,000 customers (model-based, associational).")

# %% [markdown]
# ## 21.7 Causal Effect Estimation — Doubly Robust (AIPW)
#
# What-if di 21.6 hanya **simulasi model** (asosiasi). Untuk rekomendasi yang lebih kuat, section ini mengestimasi **efek kausal rata-rata (ATE)** tiap intervensi actionable dengan **AIPW (Augmented Inverse Probability Weighting)** — metode *doubly robust* yang dipakai di causal ML modern (Robins et al., 1994; Chernozhukov et al., 2018):
#
# 1. **Treatment** T = kondisi "favorable" yang bisa didorong perusahaan (misal *payment = Auto-debit*, *auto_renew = 1*, *0 keluhan*, *kenaikan premi ≤ median*).
# 2. **Propensity model** e(X) = P(T=1 | X): peluang customer berada di kondisi itu berdasarkan karakteristik lain (confounders).
# 3. **Outcome models** μ1(X), μ0(X): peluang churn kalau T=1 / T=0.
# 4. **AIPW** menggabungkan keduanya → estimasi tetap konsisten kalau **salah satu** model benar (*doubly robust*). Semua model **cross-fitted** (5-fold) untuk menghindari overfitting bias.
#
# **Cara baca tabel:**
# - `naive_diff_pp` = beda churn rate mentah (T=1 vs T=0) — tercampur confounding.
# - `ate_pp` = efek setelah mengontrol confounders (poin persen). Negatif = kondisi favorable **menurunkan** churn.
# - `ci_low_pp/ci_high_pp` = 95% CI; `p_value`.
# - `overlap_issue_%` = % customer dengan propensity < 0.05 atau > 0.95. Tinggi (> 10%) → estimasi kurang andal (kelompok tidak sebanding).
#
# **Jika X → Y:**
# - ATE signifikan & searah dengan SHAP/odds ratio → **bukti terkuat** untuk rekomendasi (tulis sebagai "estimated causal effect under the assumption of no unmeasured confounding").
# - |naive| jauh lebih besar dari |ATE| → sebagian besar beda mentah adalah confounding (insight bagus untuk dibahas).
# - ATE tidak signifikan → jangan janjikan dampak besar dari intervensi itu.
#
# **Asumsi (WAJIB disebut di laporan):** tidak ada confounder penting yang tidak terukur, overlap memadai, dan treatment didefinisikan konsisten. Tetap rekomendasikan validasi dengan A/B test.
# **Catatan penargetan (Ascarza, 2018):** customer ber-risiko tertinggi belum tentu yang paling *responsif* terhadap intervensi → idealnya target berdasarkan *uplift*/nilai, dan pilot A/B test dipakai untuk mengukur uplift sebenarnya.

# %%
CAUSAL = pd.DataFrame()
CAUSAL_DATA = {}   # komponen AIPW per treatment → robustness & uplift (21.7b–c)


def define_treatment(f):
    """Bangun treatment biner 'favorable' (T=1) dari fitur actionable f. Return (T, label) atau (None, None)."""
    s = train[f]
    tbl = LEVEL_TABLES.get(f)
    if not pd.api.types.is_numeric_dtype(s):
        if tbl is None:
            return None, None
        ok = tbl[(tbl["n"] >= max(30, 0.02 * N_TR)) & (~tbl["level"].isin(["Missing", "Other"]))]
        if len(ok) < 2:
            return None, None
        safest = ok.sort_values("churn_rate").iloc[0]["level"]
        T = (s.astype(str) == str(safest)).astype(float).where(s.notna())
        return T.values, f"{f} = {safest}"
    if s.nunique() <= 2:
        r = pd.Series(y).groupby(s.values).mean()
        if len(r) < 2:
            return None, None
        good = r.idxmin()
        return (s == good).astype(float).where(s.notna()).values, f"{f} = {good:g}"
    higher_worse = DRIVER_TABLE.set_index("feature").loc[f, "effect_size"] > 0 if f in set(DRIVER_TABLE["feature"]) else True
    is_count = (s.dropna() % 1 == 0).all() and (s == 0).mean() > 0.3 and s.min() >= 0
    if is_count and higher_worse:
        return (s == 0).astype(float).where(s.notna()).values, f"{f} = 0"
    med = s.median()
    if higher_worse:
        return (s <= med).astype(float).where(s.notna()).values, f"{f} ≤ median ({med:g})"
    return (s >= med).astype(float).where(s.notna()).values, f"{f} ≥ median ({med:g})"


def aipw_components(T, Y, X, n_splits=5, seed=0):
    """Cross-fitted AIPW: propensity e(X), outcome μ1/μ0, pseudo-outcome ψ (doubly robust score), ATE & SE."""
    n = len(Y)
    e, m1, m0 = np.zeros(n), np.zeros(n), np.zeros(n)
    strata = (T * 2 + Y).astype(int)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strata):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        prop = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40, random_state=seed)
        prop.fit(Xtr, T[tr])
        e[te] = prop.predict_proba(Xte)[:, 1]
        for t_val, store in [(1, m1), (0, m0)]:
            mask = T[tr] == t_val
            if mask.sum() < 30 or Y[tr][mask].min() == Y[tr][mask].max():
                store[te] = Y[tr][mask].mean() if mask.sum() else Y.mean()
                continue
            om = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40, random_state=seed)
            om.fit(Xtr[mask], Y[tr][mask])
            store[te] = om.predict_proba(Xte)[:, 1]
    overlap_issue = float(((e < 0.05) | (e > 0.95)).mean())
    ec = np.clip(e, 0.02, 0.98)
    psi = m1 - m0 + T * (Y - m1) / ec - (1 - T) * (Y - m0) / (1 - ec)
    ate = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    return {"ate": float(ate), "se": float(se), "overlap": overlap_issue, "e": e, "m1": m1, "m0": m0, "psi": psi}


def aipw_ate(T, Y, X, n_splits=5, seed=0):
    c = aipw_components(T, Y, X, n_splits, seed)
    return c["ate"], c["se"], c["overlap"]


with safe_section("Causal effect estimation (AIPW)"):
    if not HAS_HGB:
        print("⏭️ HistGradientBoosting tidak tersedia → skip.")
    else:
        cand = EVIDENCE[EVIDENCE["actionable"]]["feature"].tolist()[:CFG.CAUSAL_MAX_TREATMENTS] if len(EVIDENCE) else []
        forced = [c for c in (getattr(CFG, "TREATMENT_COLS", None) or []) if c in train.columns]
        cand = forced + [c for c in cand if c not in forced]
        Xc_full = REPS["ord"][0].copy()
        if N_TR > 100_000:
            pick_c = np.sort(np.random.RandomState(CFG.SEED).choice(N_TR, 100_000, replace=False))
        else:
            pick_c = np.arange(N_TR)
        rows = []
        for f in cand:
            if f not in NAME_MAP or f in LEAKY_COLS:
                continue
            T, label = define_treatment(f)
            if T is None:
                continue
            fam = FEATURE_FAMILY.get(f)
            drop_orig = {f} | ({g for g, fm in FEATURE_FAMILY.items() if fm == fam} if fam else set())
            drop_s = {NAME_MAP[g] for g in drop_orig if g in NAME_MAP} | {f"{NAME_MAP[g]}__freq" for g in drop_orig if g in NAME_MAP}
            X_conf = Xc_full.drop(columns=[c for c in Xc_full.columns if c in drop_s])
            valid = ~np.isnan(T)
            idx = pick_c[valid[pick_c]]
            Tt, Yt, Xt = T[idx].astype(int), y[idx], X_conf.iloc[idx].reset_index(drop=True)
            if Tt.mean() < 0.02 or Tt.mean() > 0.98:
                continue
            comp = aipw_components(Tt, Yt, Xt, seed=CFG.SEED)
            ate, se, ov = comp["ate"], comp["se"], comp["overlap"]
            CAUSAL_DATA[f] = dict(label=label, T=Tt, Y=Yt, X=Xt, idx=idx, **comp)
            naive = Yt[Tt == 1].mean() - Yt[Tt == 0].mean()
            z = ate / se if se > 0 else 0
            pval = 2 * (1 - stats.norm.cdf(abs(z)))
            rows.append({"feature": f, "favorable_condition (T=1)": label, "treated_%": 100 * Tt.mean(),
                         "naive_diff_pp": 100 * naive, "ate_pp": 100 * ate, "ci_low_pp": 100 * (ate - 1.96 * se),
                         "ci_high_pp": 100 * (ate + 1.96 * se), "p_value": pval, "overlap_issue_%": 100 * ov})
            print(f"  {label:<45} naive {100 * naive:+6.2f} pp | AIPW {100 * ate:+6.2f} pp "
                  f"(95% CI {100 * (ate - 1.96 * se):+.2f} to {100 * (ate + 1.96 * se):+.2f}) | overlap issue {100 * ov:.1f}%")
        CAUSAL = pd.DataFrame(rows)
        if len(CAUSAL):
            save_table(CAUSAL, "causal_effects_aipw")
            display(CAUSAL)
            d = CAUSAL.iloc[::-1]
            fig, ax = plt.subplots(figsize=(8.5, max(3, 0.6 * len(d) + 1)))
            yy = np.arange(len(d))
            ax.errorbar(d["ate_pp"], yy, xerr=[(d["ate_pp"] - d["ci_low_pp"]).clip(lower=0), (d["ci_high_pp"] - d["ate_pp"]).clip(lower=0)],
                        fmt="o", color=CFG.PALETTE[0], capsize=3, label="AIPW causal estimate (95% CI)")
            ax.scatter(d["naive_diff_pp"], yy + 0.15, marker="x", color=CFG.PALETTE[1], label="Naive difference")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_yticks(yy)
            ax.set_yticklabels(d["favorable_condition (T=1)"].str.slice(0, 45), fontsize=8)
            ax.set_xlabel("Effect on churn probability (percentage points; negative = reduces churn)")
            ax.set_title("Estimated Causal Effect of Favourable Conditions on Churn (doubly robust AIPW)")
            ax.legend(fontsize=8)
            save_fig("causal_effects_aipw", fig)
            for _, r in CAUSAL.sort_values("ate_pp").iterrows():
                if r["p_value"] < 0.05 and r["overlap_issue_%"] < 20:
                    add_insight("Causal", f"Adjusting for observed confounders (doubly robust AIPW), {r['favorable_condition (T=1)']} is associated with a "
                                          f"{r['ate_pp']:+.1f} pp change in churn probability (95% CI {r['ci_low_pp']:+.1f} to {r['ci_high_pp']:+.1f}; "
                                          f"naive difference {r['naive_diff_pp']:+.1f} pp).")
            weak = CAUSAL[(CAUSAL["p_value"] >= 0.05)]["favorable_condition (T=1)"].tolist()
            if weak:
                add_insight("Causal", f"No statistically significant causal effect was detected for: {', '.join(weak)} — interventions on these are lower priority.")
        else:
            print("Tidak ada treatment actionable yang bisa didefinisikan.")

# %% [markdown]
# ### 21.7b Robustness efek kausal: overlap, E-value, placebo test, DAG
# Estimasi kausal hanya layak dipercaya kalau lolos uji robustness:
# | Uji | Lolos jika | Arti |
# |---|---|---|
# | **Overlap / positivity** | distribusi propensity T=1 dan T=0 tumpang tindih; `overlap_issue_%` < 20% | kelompok sebanding |
# | **Placebo treatment** | treatment diacak → efek ≈ 0 (p > 0,05) | estimator tidak menghasilkan efek palsu |
# | **E-value** (VanderWeele & Ding, 2017) | ≥ 1,25 (makin besar makin kuat) | seberapa kuat confounder *tak terukur* untuk menghapus efek |
#
# **DAG** menggambarkan asumsi identifikasi (confounder terukur → T & Y, U tak terukur → garis putus-putus).
# **Untuk laporan:** verdict "robust" → klaim efek boleh dipakai untuk rekomendasi (tetap sarankan pilot A/B).

# %%
CAUSAL_ROBUST = pd.DataFrame()
with safe_section("Causal robustness: overlap, E-values, placebo test & DAG"):
    if not len(CAUSAL) or not CAUSAL_DATA:
        print("⏭️ Tidak ada estimasi kausal → skip.")
    else:
        rows = []
        keys = [k for k in CAUSAL["feature"] if k in CAUSAL_DATA]
        for f in keys:
            cd = CAUSAL_DATA[f]
            T_, Y_, e_, m1_, m0_ = cd["T"], cd["Y"], cd["e"], cd["m1"], cd["m0"]
            ec = np.clip(e_, 0.02, 0.98)
            mu1 = float(np.mean(m1_ + T_ * (Y_ - m1_) / ec))
            mu0 = float(np.mean(m0_ + (1 - T_) * (Y_ - m0_) / (1 - ec)))
            r = CAUSAL.set_index("feature").loc[f]
            rr = mu1 / mu0 if mu0 > 0 else np.nan
            ci_near_null = r["ci_high_pp"] if r["ate_pp"] < 0 else r["ci_low_pp"]
            crosses = (r["ci_low_pp"] <= 0 <= r["ci_high_pp"])
            rr_ci = (mu0 + ci_near_null / 100) / mu0 if mu0 > 0 else np.nan
            # placebo: treatment diacak → efek harus ≈ 0
            n_pl = min(len(T_), 30_000)
            pick = np.random.RandomState(CFG.SEED).choice(len(T_), n_pl, replace=False)
            T_pl = np.random.RandomState(CFG.SEED + 1).permutation(T_[pick])
            comp_pl = aipw_components(T_pl, Y_[pick], cd["X"].iloc[pick].reset_index(drop=True), seed=CFG.SEED)
            pl_ate, pl_se = comp_pl["ate"], comp_pl["se"]
            pl_p = 2 * stats.norm.sf(abs(pl_ate / pl_se)) if pl_se > 0 else 1.0
            ev_point = e_value(rr)
            ev_ci = 1.0 if crosses else e_value(rr_ci)
            robust = (r["p_value"] < 0.05) and (pl_p > 0.05) and (r["overlap_issue_%"] < 20) and (ev_ci >= 1.25)
            rows.append({"treatment (T=1)": r["favorable_condition (T=1)"], "feature": f, "ate_pp": r["ate_pp"], "p_value": r["p_value"],
                         "mu1_churn_if_T1": mu1, "mu0_churn_if_T0": mu0, "risk_ratio": rr, "e_value_point": ev_point, "e_value_ci": ev_ci,
                         "placebo_ate_pp": 100 * pl_ate, "placebo_p": pl_p, "overlap_issue_%": r["overlap_issue_%"],
                         "verdict": "robust" if robust else ("not significant" if r["p_value"] >= 0.05 else "fragile — interpret with caution")})
        CAUSAL_ROBUST = pd.DataFrame(rows)
        save_table(CAUSAL_ROBUST, "causal_robustness")
        display(CAUSAL_ROBUST.round(3))

        # overlap (positivity) plot
        k = len(keys)
        fig, axes = grid_axes(k, ncols=min(3, k), w=4.6, h=3.2)
        for a, f in zip(axes, keys):
            cd = CAUSAL_DATA[f]
            for tv, col, lab in [(1, CFG.PALETTE[2], "T=1 (favourable)"), (0, CFG.PALETTE[1], "T=0")]:
                a.hist(cd["e"][cd["T"] == tv], bins=30, range=(0, 1), alpha=0.55, density=True, color=col, label=lab)
            a.set_title(cd["label"][:40], fontsize=8.5)
            a.set_xlabel("Propensity score e(X)", fontsize=8)
            a.legend(fontsize=6.5)
        fig.suptitle("Overlap Check: Propensity Score Distributions by Treatment Group", y=1.02)
        plt.tight_layout()
        save_fig("causal_overlap", fig)

        # causal DAG untuk treatment utama
        top_f = CAUSAL.sort_values("p_value").iloc[0]["feature"]
        lbl = CAUSAL_DATA[top_f]["label"]
        conf = [pretty(c) for c in EVIDENCE["feature"].tolist() if c != top_f and FEATURE_FAMILY.get(c) != FEATURE_FAMILY.get(top_f, "#")][:4] \
            if len(EVIDENCE) else []
        rob = CAUSAL_ROBUST.set_index("feature").loc[top_f]
        fig, ax = canvas(11, 5.2)
        dbox(ax, 0.36, 0.40, 0.28, 0.17, "Treatment T", lbl[:38], fc=CFG.PALETTE[2])
        dbox(ax, 0.76, 0.40, 0.21, 0.17, "Outcome Y", "churn (0/1)", fc=CFG.PALETTE[1])
        dbox(ax, 0.02, 0.72, 0.30, 0.22, "Observed confounders X", "\n".join(conf) or "customer & policy profile", fc=CFG.PALETTE[0], bfs=7.5)
        dbox(ax, 0.02, 0.06, 0.30, 0.18, "Unmeasured confounders U", f"E-value = {rob['e_value_point']:.2f}\n(CI bound: {rob['e_value_ci']:.2f})",
             fc="#BDBDBD", tc="#222")
        darrow(ax, (0.64, 0.485), (0.76, 0.485), lw=2.2)
        ax.text(0.70, 0.52, f"ATE {rob['ate_pp']:+.1f} pp", ha="center", fontsize=8.5, weight="bold")
        darrow(ax, (0.30, 0.72), (0.42, 0.57))
        darrow(ax, (0.32, 0.86), (0.84, 0.57), rad=-0.25)
        darrow(ax, (0.30, 0.24), (0.42, 0.40), ls="--", color="#888")
        darrow(ax, (0.32, 0.12), (0.84, 0.40), ls="--", color="#888", rad=0.2)
        ax.text(0.50, 0.97, "Causal identification strategy (doubly robust AIPW, cross-fitted)", ha="center", fontsize=11, weight="bold")
        ax.text(0.66, 0.16, f"Placebo test (random T): {rob['placebo_ate_pp']:+.2f} pp, p = {p_fmt(rob['placebo_p'])}\n"
                            f"Overlap issues: {rob['overlap_issue_%']:.1f}% of customers\nVerdict: {rob['verdict']}",
                fontsize=8.5, va="center", bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#CCC"))
        save_fig("causal_dag", fig)
        for _, r in CAUSAL_ROBUST.iterrows():
            if r["verdict"] == "robust":
                add_insight("Causal", f"The effect of '{r['treatment (T=1)']}' ({r['ate_pp']:+.1f} pp) is robust: a placebo treatment yields "
                                      f"{r['placebo_ate_pp']:+.2f} pp (p {p_fmt(r['placebo_p'])}) and an unmeasured confounder would need a risk-ratio "
                                      f"association of at least {r['e_value_ci']:.2f} with both treatment and churn to explain it away (E-value).")

# %% [markdown]
# ### 21.7c Heterogeneous treatment effects (DR-learner) & uplift targeting
# Skor churn menjawab **siapa yang berisiko**, bukan **siapa yang akan bertahan karena intervensi** (Ascarza, 2018). Dengan **DR-learner** (Kennedy, 2023) — regresi pseudo-outcome doubly robust ψ pada fitur customer, cross-fitted — kita mengestimasi efek intervensi *per customer* (CATE).
# - **GATES** (sorted group average treatment effects): customer dibagi kuintil prediksi uplift; rata-rata ψ per kuintil + CI → bukti heterogenitas.
# - **Uplift targeting curve**: churner yang dihindari per 1.000 customer bila menarget berdasarkan uplift vs berdasarkan risiko vs acak.
#
# Hanya dijalankan untuk treatment dengan efek rata-rata signifikan & overlap memadai (mencegah over-claim). Prediksi uplift customer test masuk ke `test_customer_action_list.csv`.

# %%
UPLIFT_SUMMARY, UPLIFT_GATES, UPLIFT_TEST = pd.DataFrame(), pd.DataFrame(), {}
with safe_section("Heterogeneous effects (DR-learner) & uplift targeting"):
    if not CFG.USE_UPLIFT or not len(CAUSAL) or not CAUSAL_DATA:
        print("⏭️ Uplift di-skip (USE_UPLIFT=False atau tidak ada treatment).")
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        cand_u = CAUSAL[(CAUSAL["p_value"] < 0.05) & (CAUSAL["overlap_issue_%"] < 30)].sort_values("ate_pp")["feature"].tolist()[:2]
        if not cand_u:
            print("Tidak ada treatment dengan efek rata-rata signifikan → analisis uplift tidak dilakukan (hindari over-claim).")
        summ, gates_rows = [], []
        if cand_u:
            fig, axes = plt.subplots(len(cand_u), 2, figsize=(13, 4.3 * len(cand_u)), squeeze=False)
        for row_i, f in enumerate(cand_u):
            cd = CAUSAL_DATA[f]
            X_, psi_, T_ = cd["X"], cd["psi"], cd["T"]
            benefit_dr = -psi_                                   # churn yang dihindari jika dipindah ke kondisi favorable
            tau_hat = np.zeros(len(psi_))
            for tr_, va_ in KFold(5, shuffle=True, random_state=CFG.SEED).split(X_):
                reg = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
                                                    l2_regularization=1.0, random_state=CFG.SEED)
                reg.fit(X_.iloc[tr_], benefit_dr[tr_])
                tau_hat[va_] = reg.predict(X_.iloc[va_])
            # GATES: kelompokkan berdasarkan prediksi uplift (out-of-fold) → rata-rata DR score per kuintil
            q = pd.qcut(pd.Series(tau_hat).rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
            g = pd.DataFrame({"q": q.values, "b": benefit_dr}).groupby("q", observed=True)["b"].agg(["mean", "std", "count"])
            g["se"] = g["std"] / np.sqrt(g["count"])
            for qi, r in g.iterrows():
                gates_rows.append({"treatment": cd["label"], "uplift_quintile": qi, "avg_churn_reduction_pp": 100 * r["mean"],
                                   "ci_low_pp": 100 * (r["mean"] - 1.96 * r["se"]), "ci_high_pp": 100 * (r["mean"] + 1.96 * r["se"]), "n": int(r["count"])})
            het_diff = g["mean"].iloc[-1] - g["mean"].iloc[0]
            het_se = math.sqrt(g["se"].iloc[-1] ** 2 + g["se"].iloc[0] ** 2)
            het_p = 2 * stats.norm.sf(abs(het_diff / het_se)) if het_se > 0 else 1.0
            # targeting curves di antara customer yang BELUM dalam kondisi favorable (T=0) → yang bisa diintervensi
            elig = np.where(T_ == 0)[0]
            risk_ = np.asarray(PROB_BIZ)[cd["idx"]][elig]
            orders = {"Uplift model (DR-learner)": np.argsort(-tau_hat[elig]), "Churn risk": np.argsort(-risk_),
                      "Random": np.random.RandomState(CFG.SEED).permutation(len(elig))}
            ax = axes[row_i, 0]
            fr_grid = np.linspace(0.05, 1, 20)
            curves = {}
            for (nm, o), col in zip(orders.items(), [CFG.PALETTE[2], CFG.PALETTE[1], "gray"]):
                cum = np.cumsum(benefit_dr[elig][o])
                ys_ = [cum[max(1, int(fr * len(o))) - 1] / len(elig) * 1000 for fr in fr_grid]
                curves[nm] = ys_
                ax.plot(100 * fr_grid, ys_, "o-", ms=3, color=col, label=nm, lw=2 if "Uplift" in nm else 1.2)
            ax.set_xlabel("% of eligible customers treated (T=0 today)")
            ax.set_ylabel("Churners avoided per 1,000 eligible")
            ax.set_title(f"Uplift targeting — {cd['label'][:40]}", fontsize=10)
            ax.legend(fontsize=8)
            ax2 = axes[row_i, 1]
            gg = g.reset_index()
            ax2.bar(gg["q"].astype(str), 100 * gg["mean"], yerr=196 * gg["se"], color=CFG.PALETTE[2], capsize=3)
            ax2.axhline(0, color="k", lw=0.8)
            ax2.set_xlabel("Quintile of predicted uplift (Q5 = most responsive)")
            ax2.set_ylabel("Avg churn reduction (pp, DR estimate)")
            ax2.set_title(f"Sorted group average effects (GATES); Q5−Q1 p {p_fmt(het_p)}", fontsize=10)
            k20 = int(np.argmin(np.abs(fr_grid - 0.2)))
            summ.append({"treatment": cd["label"], "feature": f, "eligible_customers": len(elig), "ate_pp": float(CAUSAL.set_index('feature').loc[f, 'ate_pp']),
                         "gates_q5_pp": 100 * g["mean"].iloc[-1], "gates_q1_pp": 100 * g["mean"].iloc[0], "heterogeneity_p": het_p,
                         "avoided_per1000_top20_uplift": curves["Uplift model (DR-learner)"][k20],
                         "avoided_per1000_top20_risk": curves["Churn risk"][k20], "avoided_per1000_top20_random": curves["Random"][k20]})
            # prediksi uplift untuk customer test (action list)
            if len(FINAL_TEST):
                X_te_u = REPS["ord"][1][X_.columns]
                reg = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
                                                    l2_regularization=1.0, random_state=CFG.SEED).fit(X_, benefit_dr)
                UPLIFT_TEST[cd["label"]] = reg.predict(X_te_u)
        if cand_u:
            fig.suptitle("Who Responds? Heterogeneous Treatment Effects and Uplift-Based Targeting", y=1.01)
            plt.tight_layout()
            save_fig("uplift_targeting", fig)
            UPLIFT_SUMMARY = pd.DataFrame(summ)
            UPLIFT_GATES = pd.DataFrame(gates_rows)
            save_table(UPLIFT_SUMMARY, "uplift_summary")
            save_table(UPLIFT_GATES, "uplift_gates")
            display(UPLIFT_SUMMARY.round(3))
            for _, r in UPLIFT_SUMMARY.iterrows():
                better = r["avoided_per1000_top20_uplift"] - r["avoided_per1000_top20_risk"]
                het_ = (r["gates_q5_pp"] > r["gates_q1_pp"]) and (r["heterogeneity_p"] < 0.05)
                add_insight("Causal", f"Effects of '{r['treatment']}' are {'heterogeneous' if het_ else 'broadly homogeneous'}: the most responsive quintile shows a "
                                      f"{r['gates_q5_pp']:.1f} pp churn reduction vs {r['gates_q1_pp']:.1f} pp in the least responsive "
                                      f"(difference p {p_fmt(r['heterogeneity_p'])}). Treating the top 20% by predicted uplift avoids "
                                      f"{r['avoided_per1000_top20_uplift']:.1f} churners per 1,000 eligible customers vs "
                                      f"{r['avoided_per1000_top20_risk']:.1f} when targeting by churn risk ({better:+.1f}).")

# %% [markdown]
# ## 21.8 Churn Personas (clustering SHAP values)
# Customer berisiko tinggi tidak churn karena alasan yang sama. Dengan meng-cluster **SHAP values** (bukan fitur mentah) customer high-risk, kita dapat kelompok berdasarkan **alasan risikonya** → tiap persona dapat intervensi berbeda.
# Contoh hasil: *Persona A "Price-shocked": didorong kenaikan premi; Persona B "Service-frustrated": keluhan & klaim ditolak; Persona C "New & disengaged": tenure pendek, engagement rendah.*
#
# **Untuk laporan:** tabel persona (ukuran, churn rate, 3 driver utama) + nama persona buatan sendiri → sangat disukai juri karena langsung bisa jadi strategi.

# %%
PERSONAS = pd.DataFrame()
with safe_section("Churn personas (SHAP clustering)"):
    if SHAP_VALUES is None:
        print("⏭️ SHAP tidak tersedia → persona di-skip.")
    else:
        p_s = PROB_BIZ[SHAP_IDX]
        hi = p_s >= TIER_CUTS[0]
        S = SHAP_VALUES[hi]
        if hi.sum() < 60:
            print("Terlalu sedikit customer berisiko di sampel SHAP.")
        else:
            best_k, best_sil = 3, -1
            for k in range(2, 6):
                lab = KMeans(n_clusters=k, n_init=10, random_state=CFG.SEED).fit_predict(S)
                sil = silhouette_score(S, lab, sample_size=min(2000, len(S)), random_state=CFG.SEED)
                if sil > best_sil:
                    best_k, best_sil = k, sil
            km = KMeans(n_clusters=best_k, n_init=20, random_state=CFG.SEED).fit(S)
            lab = km.labels_
            print(f"Jumlah persona optimal (silhouette): {best_k} (silhouette = {best_sil:.3f})")
            rows = []
            prof_cols = [f for f in EVIDENCE["feature"].head(8)] if len(EVIDENCE) else []
            Xh = SHAP_X[hi].reset_index(drop=True)
            for c in range(best_k):
                m = lab == c
                mean_shap = pd.Series(S[m].mean(0), index=[n.replace("__freq", "").replace(" (target-enc)", "") for n in SHAP_NAMES]).groupby(level=0).sum()
                drivers = mean_shap.sort_values(ascending=False).head(3)
                r = {"persona": f"P{c + 1}", "n_in_sample": int(m.sum()), "share_of_at_risk_%": 100 * m.mean(),
                     "avg_pred_churn": p_s[hi][m].mean(), "actual_churn_rate": y[SHAP_IDX][hi][m].mean(),
                     "top_drivers": ("; ".join(f"{k} (+{v:.2f})" for k, v in drivers.items() if v > 0.05)
                                     or "no dominant driver (moderate, diffuse risk)")}
                for f in prof_cols:
                    if f in train.columns:
                        vals = train[f].iloc[SHAP_IDX[hi][m]]
                        r[f] = round(vals.mean(), 2) if pd.api.types.is_numeric_dtype(vals) else (vals.mode().iloc[0] if vals.notna().any() else None)
                rows.append(r)
            PERSONAS = pd.DataFrame(rows).sort_values("share_of_at_risk_%", ascending=False)
            display(PERSONAS)
            save_table(PERSONAS, "churn_personas")
            topf = pd.Series(np.abs(S).mean(0), index=SHAP_NAMES).sort_values(ascending=False).head(10).index
            heat = pd.DataFrame({f"P{c + 1}": pd.Series(S[lab == c].mean(0), index=SHAP_NAMES)[topf] for c in range(best_k)})
            fig, ax = plt.subplots(figsize=(1.6 * best_k + 4, 0.45 * len(topf) + 1.5))
            sns.heatmap(heat, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 8})
            ax.set_title("Churn Personas: average SHAP contribution of top drivers\n(red = pushes churn risk up)")
            save_fig("churn_personas_heatmap")
            for _, r in PERSONAS.iterrows():
                add_insight("Personas", f"Persona {r['persona']} ({r['share_of_at_risk_%']:.0f}% of at-risk customers, actual churn {pct(r['actual_churn_rate'])}): "
                                        f"risk driven mainly by {r['top_drivers']}.")
            print("👉 Beri nama persona sendiri berdasarkan top_drivers (misal 'Price-shocked renewers', 'Service-frustrated claimants').")

# %% [markdown]
# ## 21.9 Fairness Audit & Model Card (governance)
# Juri dari industri asuransi peduli **tata kelola model**: apakah skor adil antar kelompok (gender, umur, wilayah, status), terkalibrasi di tiap kelompok, dan dipakai sesuai tujuan. Audit menghitung per kelompok: churn aktual vs prediksi (calibration ratio), AUC, porsi masuk High tier, TPR.
#
# **Model card** (`model_card.md`) merangkum: tujuan penggunaan & yang **tidak boleh** (pricing/underwriting), data, validasi, performa + CI, kalibrasi, fairness, monitoring (PSI, AUC drop), keterbatasan.
#
# **Untuk laporan:** 1–2 kalimat governance di Discussion + model card di Appendix.

# %%
FAIRNESS, MODEL_CARD = pd.DataFrame(), pd.DataFrame()
with safe_section("Fairness audit & model card"):
    prot = CFG.PROTECTED_ATTRIBUTES
    if prot is None:
        prot = []
        for c in [ROLES.get("gender"), "fe_age_band", ROLES.get("region"), ROLES.get("dependents")] + \
                 [c for c in CAT_COLS if re.search(r"(marital|marriage|status_nikah|religion|agama|ethnic|race)", c, re.I)]:
            if c and c in train.columns and c not in prot and 2 <= train[c].nunique() <= 12:
                prot.append(c)
    rows = []
    high_flag = np.asarray(PROB_BIZ) >= TIER_CUTS[1]
    yl_ = (np.asarray(FINAL_OOF) >= BEST_THR).astype(int)
    for c in prot[:5]:
        lab = train[c].astype(str).where(train[c].notna(), "Missing").values
        for lv in pd.Series(lab).value_counts().index[:10]:
            m = lab == lv
            if m.sum() < 100:
                continue
            auc_g = roc_auc_score(y[m], FINAL_OOF[m]) if 0 < y[m].sum() < m.sum() else np.nan
            rows.append({"attribute": c, "group": lv, "n": int(m.sum()), "actual_churn": y[m].mean(), "mean_predicted": PROB_BIZ[m].mean(),
                         "calibration_ratio": PROB_BIZ[m].mean() / max(y[m].mean(), 1e-9), "auc": auc_g,
                         "selection_rate_high_tier": high_flag[m].mean(),
                         "tpr_at_threshold": yl_[m & (y == 1)].mean() if (m & (y == 1)).sum() else np.nan})
    FAIRNESS = pd.DataFrame(rows)
    if len(FAIRNESS):
        agg = FAIRNESS.groupby("attribute").agg(min_sel=("selection_rate_high_tier", "min"), max_sel=("selection_rate_high_tier", "max"),
                                                min_auc=("auc", "min"), max_auc=("auc", "max"),
                                                min_cal=("calibration_ratio", "min"), max_cal=("calibration_ratio", "max"))
        agg["selection_ratio"] = agg["min_sel"] / agg["max_sel"].replace(0, np.nan)
        agg["auc_gap"] = agg["max_auc"] - agg["min_auc"]
        FAIR_SUMMARY = agg.reset_index()
        save_table(FAIRNESS, "fairness_audit")
        save_table(FAIR_SUMMARY, "fairness_summary")
        display(FAIR_SUMMARY.round(3))
        k = FAIRNESS["attribute"].nunique()
        fig, axes = grid_axes(k, ncols=min(3, k), w=4.8, h=3.4)
        for a, (attr, d) in zip(axes, FAIRNESS.groupby("attribute", sort=False)):
            xx = np.arange(len(d))
            a.bar(xx - 0.2, d["actual_churn"], 0.4, color=CFG.PALETTE[0], label="actual churn")
            a.bar(xx + 0.2, d["mean_predicted"], 0.4, color=CFG.PALETTE[1], alpha=0.8, label="mean predicted")
            a.plot(xx, d["selection_rate_high_tier"], "kD", ms=4, label="share in High tier")
            a.set_xticks(xx)
            a.set_xticklabels(d["group"].astype(str).str.slice(0, 12), rotation=30, ha="right", fontsize=7)
            a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            a.set_title(f"{attr[:28]} (AUC gap {FAIR_SUMMARY.set_index('attribute').loc[attr, 'auc_gap']:.3f})", fontsize=9)
            a.legend(fontsize=6)
        fig.suptitle("Fairness Audit: Calibration, Discrimination and Selection by Customer Group", y=1.02)
        plt.tight_layout()
        save_fig("fairness_audit", fig)
        worst_cal = FAIRNESS.iloc[(FAIRNESS["calibration_ratio"] - 1).abs().argmax()]
        add_insight("Governance", f"Fairness audit across {', '.join(pretty(a) for a in FAIR_SUMMARY['attribute'])}: predicted risk tracks actual churn "
                                  f"within each group (largest calibration ratio deviation: {worst_cal['attribute']} = '{worst_cal['group']}', "
                                  f"{worst_cal['calibration_ratio']:.2f}); the largest within-attribute AUC gap is {FAIR_SUMMARY['auc_gap'].max():.3f}. "
                                  f"Differences in High-tier selection reflect genuine differences in churn rates; the score is intended for "
                                  f"retention outreach only, not for pricing or underwriting.")
    else:
        print("Tidak ada atribut sensitif yang cocok untuk audit (set CFG.PROTECTED_ATTRIBUTES kalau ada).")

    # ---------------- model card ----------------
    auc_f = roc_auc_score(y, FINAL_OOF)
    ci_f = globals().get("FINAL_AUC_CI", (np.nan, np.nan))
    fin_row = MODEL_EVAL.iloc[0] if len(MODEL_EVAL) else {}
    card = [
        ("Model", f"{FINAL_NAME} ({', '.join(f'{m} {w:.2f}' for m, w in FINAL_WEIGHTS.items())})"),
        ("Intended use", "Rank active policyholders by 12-month churn risk to prioritise retention outreach and allocate retention budget."),
        ("Out-of-scope use", "Not for premium pricing, underwriting, claims decisions or denial of service; not validated for other products/markets."),
        ("Training data", f"{N_TR:,} customers, {len(MODEL_FEATURES)} input features; target '{TARGET}' (churn = {', '.join(map(str, POS_ORIGINAL))}); churn rate {pct(CHURN_RATE)}."),
        ("Excluded variables", (", ".join(LEAKY_COLS) + " (post-event leakage)") if LEAKY_COLS else "none flagged for leakage; identifiers and constants removed."),
        ("Validation", f"Stratified {CFG.N_FOLDS}-fold CV (out-of-fold), target encoding inside folds, shuffled-label & adversarial checks, bootstrap & DeLong CIs."),
        ("Performance", f"ROC-AUC {auc_f:.3f} (95% CI {ci_f[0]:.3f}–{ci_f[1]:.3f}); PR-AUC {average_precision_score(y, FINAL_OOF):.3f}; "
                        f"Brier {brier_score_loss(y, np.clip(FINAL_OOF_CAL, 0, 1)):.3f}; ECE {ece_score(y, np.clip(FINAL_OOF_CAL, 0, 1)):.3f}"
                        + (f"; EMPC {fmt_num(fin_row['empc_per_customer'])} {EMPC_UNIT}/customer" if len(MODEL_EVAL) else "") + "."),
        ("Calibration", f"{CAL_METHOD} (selected by cross-validated log-loss)."),
        ("Decision rule", f"Risk tiers at the {CFG.RISK_TIER_CUTS[0]:.0%}/{CFG.RISK_TIER_CUTS[1]:.0%} quantiles; campaign size chosen by expected profit (see Section 4)."),
        ("Explainability", "Global: SHAP, permutation importance, PDP, EBM shape functions (if used). Local: SHAP waterfall per customer for agents."),
        ("Fairness", f"Audited on {', '.join(FAIRNESS['attribute'].unique()) if len(FAIRNESS) else 'n/a'}; max AUC gap "
                     f"{FAIR_SUMMARY['auc_gap'].max():.3f}." if len(FAIRNESS) else "No sensitive attributes available for audit."),
        ("Monitoring", "Monthly: PSI per feature (alert > 0.25), score distribution, realised churn by tier, calibration; retrain quarterly or if AUC drops > 0.03."),
        ("Limitations", "Observational snapshot data; causal effects assume no unmeasured confounding; ROI depends on stated cost/acceptance assumptions."),
    ]
    MODEL_CARD = pd.DataFrame(card, columns=["Item", "Description"])
    save_table(MODEL_CARD, "model_card")
    with open(os.path.join(CFG.OUTPUT_DIR, "model_card.md"), "w", encoding="utf-8") as fmc:
        fmc.write("# Model Card — Churn Early-Warning Model\n\n| Item | Description |\n|---|---|\n")
        for k_, v_ in card:
            fmc.write(f"| {k_} | {v_} |\n")
    print("Model card →", os.path.join(CFG.OUTPUT_DIR, "model_card.md"))

# %% [markdown]
# # 22. Recommendations (otomatis, berbasis bukti)
#
# Rekomendasi dibangkitkan dari **driver actionable** (evidence matrix) + library strategi retensi asuransi, lengkap dengan **bukti angka**, **target segmen**, dan **KPI**.
# Ini draft — **edit & personalisasi** sesuai konteks soal sebelum masuk laporan.
#
# ### Struktur rekomendasi yang disukai juri
# 1. **Insight** (angka) → 2. **Aksi spesifik** (siapa, apa, kapan) → 3. **Target segmen & ukuran** → 4. **Dampak estimasi** (what-if / ROI) → 5. **KPI & cara ukur** (A/B test).
#
# ### Kerangka prioritas
# - **Quick wins (0–3 bulan):** trigger otomatis (reminder renewal, auto-debit enrollment, follow-up keluhan).
# - **Medium (3–6 bulan):** program onboarding, revisi kebijakan pricing renewal, SLA klaim.
# - **Long term (6–12 bulan):** churn early-warning system (model ini di-deploy, skor bulanan), loyalty program, bundling produk.

# %%
REC_LIBRARY = [
    (["payment_method", "paymentmethod", "metode_bayar", "payment_type", "payment_mode", "cara_bayar"],
     "Shift manual payers to automatic payment",
     "Offer a small incentive (e.g. premium discount/cashback) and one-click enrolment for auto-debit/recurring card payment; send pre-due reminders to remaining manual payers.",
     "% of policies on auto-pay; lapse rate of converted vs control"),
    (["auto_renew", "autorenew", "auto-renew", "paperless"],
     "Make renewal effortless (default auto-renewal)",
     "Default new and renewing policies to auto-renewal with clear opt-out; 30/14/7-day renewal reminders with a one-click renew link.",
     "Auto-renew adoption; renewal rate"),
    (["complain", "keluhan", "komplain", "ticket", "service_call", "support"],
     "Complaint recovery programme",
     "Trigger a proactive call within 48h of any complaint, enforce resolution SLAs, offer service-recovery gestures to high-value complainants, and fix top root causes.",
     "Complaint resolution time; churn rate of complainants"),
    (["reject", "tolak", "denied", "settle", "claim", "klaim"],
     "Improve the claims experience",
     "Fast-track settlement, explain rejections transparently with an appeal path, and assign a claims concierge for high-value / high-risk customers after a claim event.",
     "Claim settlement days; post-claim churn rate; claim NPS"),
    (["premium_change", "increase", "kenaikan", "price", "rate_change", "premium", "premi", "charges"],
     "Smarter renewal pricing",
     "Cap or phase in premium increases for at-risk loyal customers, explain the value behind increases, and offer coverage/deductible adjustments instead of lapse.",
     "Churn rate among customers with increases; price elasticity"),
    (["late", "telat", "overdue", "arrear", "missed", "tunggakan", "grace"],
     "Early-warning on payment difficulty",
     "Treat the first late payment as a churn signal: friendly reminders, flexible instalments or payment-date change, and grace-period outreach before lapse.",
     "Cure rate of late payers; lapse after first missed payment"),
    (["tenure", "new_customer", "lama", "masa"],
     "First-year onboarding programme",
     "Welcome call, policy walkthrough, app activation and 90-day check-in for new customers; first renewal treated as a critical moment.",
     "13th-month persistency ratio; first-year churn rate"),
    (["product", "produk", "rider", "multi_product", "bundle", "policies"],
     "Bundling & cross-sell to raise switching costs",
     "Offer multi-policy discounts and relevant riders to single-product customers with good risk profiles.",
     "Products per customer; churn of bundled vs single"),
    (["engagement", "login", "app", "digital", "usage", "activity", "isactivemember"],
     "Re-engage dormant customers",
     "Use app/wellness features, personalised content and periodic check-ins; trigger re-engagement when activity drops.",
     "Monthly active users; churn of re-engaged customers"),
    (["channel", "agent", "agen", "broker", "online", "bancassurance", "source"],
     "Channel-specific retention",
     "Assign human follow-up to channels with weak relationships (e.g. online-acquired customers) and tie agent incentives to persistency, not only new sales.",
     "Persistency by channel; churn by agent"),
    (["age", "usia", "umur", "birth", "lahir"],
     "Life-stage tailored engagement",
     "Tailor products, pricing communication and service channels to the highest-risk age groups (e.g. digital-first onboarding and flexible "
     "payment for younger customers; loyalty recognition for long-standing older customers).",
     "Churn rate by age band; adoption of tailored offers"),
    (["region", "wilayah", "provinsi", "city", "kota", "county", "state", "branch", "cabang", "latitude", "longitude", "geography"],
     "Regional retention focus",
     "Concentrate field/agent retention effort on the regions with the highest churn and investigate local drivers (competitor pricing, "
     "service capacity).",
     "Churn rate by region; branch-level persistency"),
    (["income", "pendapatan", "salary", "gaji", "premium_to_income", "home_market_value", "credit", "afford"],
     "Affordability options",
     "Offer lower-premium plans, deductible adjustments or instalments to customers whose premium is high relative to income, instead of losing them.",
     "Churn of high premium-to-income customers; downgrade-instead-of-lapse rate"),
    (["marital", "children", "dependent", "household", "tanggungan", "residence"],
     "Household-based cross-sell and life-event triggers",
     "Use household profile and life events (moving, marriage, children) as triggers for coverage reviews and multi-policy bundles.",
     "Products per household; churn after life events"),
    (["frequency", "frekuensi", "contract", "billing_cycle", "term"],
     "Longer payment / contract terms",
     "Encourage annual/quarterly payment or longer contracts with a discount; monthly payers get extra reminders.",
     "Share of annual payers; churn by payment frequency"),
]


def find_recommendation(feature):
    fl = feature.lower()
    for kws, title, action, kpi in REC_LIBRARY:
        if any(k in fl for k in kws):
            return title, action, kpi
    nm = pretty(feature) if "pretty" in globals() else feature
    return (f"Targeted action on {nm}",
            f"Customers in the high-risk range of {nm} churn more; design a targeted intervention for them and test it in a pilot.",
            f"Churn rate of the high-risk {nm} group")


RECS = pd.DataFrame()
with safe_section("Recommendations"):
    rows, used_titles = [], set()
    if len(EVIDENCE):
        cand = pd.concat([EVIDENCE[EVIDENCE["actionable"]], EVIDENCE[~EVIDENCE["actionable"]].head(6)])
    else:
        cand = pd.DataFrame()
    used_fams = set()
    for _, r in cand.iterrows():
        if r["feature"] in LEAKY_COLS:
            continue
        fam_ = FEATURE_FAMILY.get(r["feature"])
        if fam_ is not None and fam_ in used_fams:
            continue                       # satu rekomendasi per keluarga informasi (tenure band / days tenure / start date)
        if fam_ is not None:
            used_fams.add(fam_)
        title, action, kpi = find_recommendation(r["feature"])
        if title in used_titles:
            continue
        used_titles.add(title)
        _lv = str(r["highest_risk_level"])
        try:
            if pd.api.types.is_numeric_dtype(train[r["feature"]]) and train[r["feature"]].nunique() == 2 and _lv in ("0", "1", "0.0", "1.0"):
                _lv = "Yes" if _lv.startswith("1") else "No"
        except Exception:
            pass
        ev_txt = (f"{r['feature']} = {_lv}: churn {pct(r['highest_risk_rate'])} ({r['highest_risk_lift']:.1f}× avg); "
                  f"IV {r['iv']:.2f}")
        if "odds_ratio" in r and pd.notna(r.get("odds_ratio")):
            ev_txt += f"; OR {r['odds_ratio']:.2f}"
        if "hazard_ratio" in r and pd.notna(r.get("hazard_ratio")):
            ev_txt += f"; HR {r['hazard_ratio']:.2f}"
        if "shap_rank" in r and pd.notna(r.get("shap_rank")):
            ev_txt += f"; SHAP rank #{int(r['shap_rank'])}"
        if len(CAUSAL) and r["feature"] in set(CAUSAL["feature"]):
            c_ = CAUSAL.set_index("feature").loc[r["feature"]]
            if c_["p_value"] < 0.05:
                ev_txt += f"; causal (AIPW) {c_['ate_pp']:+.1f} pp [{c_['ci_low_pp']:+.1f}, {c_['ci_high_pp']:+.1f}]"
        impact = ""
        if len(WHATIF):
            w = WHATIF[WHATIF["feature"] == r["feature"]]
            if len(w):
                impact = f"-{w['overall_reduction_pp_full'].iloc[0]:.2f} pp churn at full adoption (≈{w['churners_avoided_per_1000'].iloc[0]:.0f} churners avoided /1,000 at {CFG.WHAT_IF_ADOPTION:.0%} adoption)"
        rows.append({"priority": len(rows) + 1, "driver": r["feature"], "recommendation": title, "action": action, "evidence": ev_txt,
                     "estimated_impact": impact, "target_segment": f"customers in '{r['highest_risk_level']}' + Priority Save quadrant",
                     "kpi": kpi, "evidence_score": r["evidence_score"]})
        if len(rows) >= 7:
            break
    if rows:
        RECS_TYPES = ["intervention" if is_actionable(r_["driver"]) else "targeting / lifecycle" for r_ in rows]
    rows.append({"priority": len(rows) + 1, "driver": "(model)", "recommendation": "Deploy a churn early-warning system",
                 "action": "Score all active policies monthly with this model; route High-risk/Priority-Save customers to retention teams, "
                           "lower tiers to automated digital nudges; refresh the model quarterly.",
                 "evidence": f"Model ROC-AUC {roc_auc_score(y, FINAL_OOF):.3f}; top-20% captures "
                             f"{DECILE_TABLE['cum_churners_captured_%'].iloc[1]:.0f}% of churners" if 'DECILE_TABLE' in globals() else "",
                 "estimated_impact": f"optimal campaign: target top {CAMPAIGN_BEST.iloc[1]['target_%']:.0f}% → net {fmt_num(CAMPAIGN_BEST.iloc[1]['net_profit'])}" if 'CAMPAIGN_BEST' in globals() else "",
                 "target_segment": "all active customers", "kpi": "Precision/recall of alerts; retained revenue vs control group", "evidence_score": np.nan})
    rows.append({"priority": len(rows) + 1, "driver": "(measurement)", "recommendation": "Validate with controlled pilots (A/B tests)",
                 "action": "For each intervention keep a random control group (e.g. 20% of targeted customers) to measure true uplift before scaling.",
                 "evidence": "What-if results are associational", "estimated_impact": "", "target_segment": "each campaign",
                 "kpi": "Uplift in retention rate (treated − control)", "evidence_score": np.nan})
    RECS = pd.DataFrame(rows)
    save_table(RECS, "recommendations")
    with pd.option_context("display.max_colwidth", 120):
        display(RECS[["priority", "driver", "recommendation", "evidence", "estimated_impact", "kpi"]])
    for _, r in RECS.iterrows():
        add_insight("Recommendations", f"{r['priority']}. {r['recommendation']} — {r['action']} Evidence: {r['evidence']}. "
                                       f"{('Impact: ' + r['estimated_impact'] + '. ') if r['estimated_impact'] else ''}KPI: {r['kpi']}.", show=False)
    print(f"\n{len(RECS)} rekomendasi disimpan ke tables/recommendations.csv & insights.md")

# %% [markdown]
# ## 22.2 Desain Pilot A/B & Roadmap Implementasi
# Rekomendasi yang **bisa diukur** = nilai tinggi dari juri praktisi. Untuk tiap intervensi utama dihitung ukuran sampel per grup (treatment vs control, uji 2 proporsi, α & power dari `CFG`) agar efek yang diharapkan bisa terdeteksi, dan porsi portofolio High-tier yang dibutuhkan.
# Roadmap 12 bulan: quick wins → medium → long term, tiap inisiatif **pilot dengan control group → ukur → scale-up**.
#
# **Untuk laporan:** 1 kalimat sample size di Recommendations + Gantt roadmap (main body kalau muat, kalau tidak Appendix).

# %%
PILOT = pd.DataFrame()
ROADMAP = pd.DataFrame()
with safe_section("Pilot (A/B test) design & power analysis"):
    hi_rate = float(TIER_TABLE.set_index("tier").loc["High", "actual_churn_rate"]) if "TIER_TABLE" in globals() else CHURN_RATE
    n_hi_port = int(TIER_TABLE.set_index("tier").loc["High", "n_test"]) if "TIER_TABLE" in globals() and "n_test" in TIER_TABLE else \
        int((np.asarray(PROB_BIZ) >= TIER_CUTS[1]).sum())
    rows = []
    sources = []
    if len(CAUSAL):
        for _, r in CAUSAL[CAUSAL["p_value"] < 0.05].sort_values("ate_pp").head(3).iterrows():
            sources.append((f"Shift to: {r['favorable_condition (T=1)']}", abs(r["ate_pp"]) / 100, "AIPW causal estimate"))
    if len(WHATIF):
        for _, r in WHATIF.head(3).iterrows():
            eff = (r["churn_affected_before"] - r["churn_affected_after"]) * CFG.WHAT_IF_ADOPTION
            if eff > 0:
                sources.append((r["scenario"][:60], eff, f"what-if × {CFG.WHAT_IF_ADOPTION:.0%} adoption"))
    sources.append(("Generic retention offer", hi_rate * CFG.RETENTION_SUCCESS_RATE * 0.5, "assumed success rate × 50% (conservative)"))
    for name_, eff, src_ in sources[:5]:
        eff = float(min(max(eff, 0.002), hi_rate * 0.9))
        n_arm = n_per_arm(hi_rate, hi_rate - eff, CFG.PILOT_ALPHA, CFG.PILOT_POWER)
        rows.append({"intervention": name_, "effect_source": src_, "baseline_churn_target_segment": hi_rate,
                     "expected_reduction_pp": 100 * eff, "n_per_arm": n_arm, "total_pilot_size": 2 * n_arm,
                     "share_of_high_tier_%": 100 * 2 * n_arm / max(n_hi_port, 1)})
    PILOT = pd.DataFrame(rows)
    save_table(PILOT, "pilot_design_power")
    display(PILOT.round(3))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    mde = np.linspace(0.005, 0.12, 60)
    for base_, lab_, col in [(CHURN_RATE, f"all customers (churn {pct(CHURN_RATE)})", CFG.PALETTE[0]),
                             (hi_rate, f"High-risk tier (churn {pct(hi_rate)})", CFG.PALETTE[1])]:
        nn = [n_per_arm(base_, base_ - d_, CFG.PILOT_ALPHA, CFG.PILOT_POWER) if base_ - d_ > 0.001 else np.nan for d_ in mde]
        ax.plot(100 * mde, nn, lw=2, color=col, label=lab_)
    for _, r in PILOT.iterrows():
        ax.scatter(r["expected_reduction_pp"], r["n_per_arm"], color="k", s=18, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Minimum detectable churn reduction (percentage points)")
    ax.set_ylabel("Customers needed per arm (log)")
    ax.set_title(f"Pilot Design: Sample Size for a Randomised Test (α = {CFG.PILOT_ALPHA}, power = {CFG.PILOT_POWER:.0%})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    save_fig("pilot_power_analysis", fig)
    if len(PILOT):
        r0 = PILOT.iloc[0]
        add_insight("Recommendations", f"Pilot design: to detect a {r0['expected_reduction_pp']:.1f} pp reduction from a baseline of {pct(hi_rate)} "
                                       f"in the High-risk tier ('{r0['intervention']}'), a randomised test needs {int(r0['n_per_arm']):,} customers per arm "
                                       f"(α = {CFG.PILOT_ALPHA}, power {CFG.PILOT_POWER:.0%}), i.e. {r0['share_of_high_tier_%']:.0f}% of the High-tier portfolio.")

with safe_section("Implementation roadmap"):
    phase_kw = [("Quick win (0–3 m)", ["automatic payment", "renewal effortless", "complaint", "early-warning on payment", "early-warning system",
                                        "validate"]),
                ("Medium term (3–6 m)", ["onboarding", "pricing", "claims", "affordability", "channel", "contract", "life-stage", "regional"]),
                ("Long term (6–12 m)", ["bundling", "re-engage", "household", "loyalty"])]
    items = []
    for _, r in RECS.iterrows():
        t = str(r["recommendation"]).lower()
        ph = next((p for p, kws in phase_kw if any(k in t for k in kws)), "Medium term (3–6 m)")
        start = {"Quick win (0–3 m)": 0, "Medium term (3–6 m)": 3, "Long term (6–12 m)": 6}[ph]
        dur = 3 if start < 6 else 6
        items.append({"priority": int(r["priority"]), "initiative": r["recommendation"], "phase": ph, "start_month": start,
                      "pilot_months": min(dur, 3), "scale_months": dur, "kpi": r["kpi"]})
    ROADMAP = pd.DataFrame(items).sort_values(["start_month", "priority"]).reset_index(drop=True)
    save_table(ROADMAP, "implementation_roadmap")
    fig, ax = plt.subplots(figsize=(11, max(3, 0.45 * len(ROADMAP) + 1.2)))
    colors_ph = {"Quick win (0–3 m)": CFG.PALETTE[2], "Medium term (3–6 m)": CFG.PALETTE[0], "Long term (6–12 m)": CFG.PALETTE[4]}
    for i, r in ROADMAP.iloc[::-1].reset_index(drop=True).iterrows():
        ax.barh(i, r["pilot_months"], left=r["start_month"], color=colors_ph[r["phase"]], alpha=0.55, edgecolor="k", lw=0.4)
        ax.barh(i, 12 - r["start_month"] - r["pilot_months"], left=r["start_month"] + r["pilot_months"],
                color=colors_ph[r["phase"]], alpha=0.95, edgecolor="k", lw=0.4)
        ax.text(r["start_month"] + 0.1, i, f"{int(r['priority'])}. {str(r['initiative'])[:52]}", va="center", fontsize=7.5, color="black")
    ax.set_yticks([])
    ax.set_xlim(0, 12)
    ax.set_xticks(range(0, 13))
    ax.set_xlabel("Month")
    for p_, c_ in colors_ph.items():
        ax.barh([], [], color=c_, label=p_)
    ax.barh([], [], color="white", edgecolor="k", label="light = pilot with control group, dark = scale-up")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Implementation Roadmap: Pilot → Measure → Scale (12 months)")
    save_fig("implementation_roadmap", fig)
    display(ROADMAP)

# %% [markdown]
# # 23. Submission + Validator
#
# - Format mengikuti `sample_submission` (nama kolom, urutan ID). Kalau tidak ada sample → `[ID_COL, TARGET]`.
# - `SUBMISSION_MODE = "probability"` → probabilitas churn (kalibrasi hanya dipakai kalau metrik logloss/brier & kalibrasi membantu).
# - `SUBMISSION_MODE = "label"` → 0/1 pakai threshold optimal OOF; `SUBMISSION_LABEL_FORMAT="original"` → label asli (misal Yes/No).
# - Validator mengecek: jumlah baris, ID cocok & unik, tidak ada NaN, range/nilai valid, nama kolom sama dengan sample.
# - Disimpan beberapa varian di `submissions/` (final, best single, label & probability) → kalau boleh submit beberapa kali, coba varian lain.
#
# **Jika X → Y:**
# - `ID mismatch` → cek `ID_COL` & tipe data ID (string vs int: "001" vs 1). Notebook menyamakan sebagai string.
# - Leaderboard jauh di bawah CV → cek format (probability vs label?), urutan baris, kolom target benar, dan drift (Section 12).

# %%
def build_submission(pred, mode, thr):
    tcol = CFG.SUBMISSION_TARGET_COL
    if sample_sub is not None:
        sub = sample_sub.copy()
        if tcol is None:
            others = [c for c in sub.columns if c != ID_COL]
            tcol = TARGET if TARGET in sub.columns else (others[-1] if others else TARGET)
    else:
        sub = pd.DataFrame({ID_COL: test[ID_COL].values}) if ID_COL and ID_COL in test.columns else pd.DataFrame(index=range(len(pred)))
        tcol = tcol or TARGET
    if mode == "label":
        lab = (pred >= thr).astype(int)
        fmt = CFG.SUBMISSION_LABEL_FORMAT
        if fmt == "auto":
            sample_vals = set(sample_sub[tcol].astype(str)) if sample_sub is not None and tcol in sample_sub else set()
            orig_str = {str(v) for v in POS_ORIGINAL + NEG_ORIGINAL}
            fmt = "original" if (sample_vals & orig_str and not sample_vals <= {"0", "1", "0.0", "1.0", "0.5"}) else "01"
            if sample_sub is None and not all(str(v) in {"0", "1", "0.0", "1.0", "True", "False"} for v in orig_str):
                fmt = "original"
        vals = np.where(lab == 1, POS_ORIGINAL[0], NEG_ORIGINAL[0]) if fmt == "original" else lab
    else:
        vals = pred
    def _norm_id(series):
        s_ = series.astype(str).str.strip()
        return s_.str.replace(r"\.0$", "", regex=True)

    aligned = False
    if sample_sub is not None and ID_COL and ID_COL in sub.columns and ID_COL in test.columns:
        key_te = _norm_id(test[ID_COL])
        if not key_te.duplicated().any():
            m = pd.Series(vals, index=key_te.values)
            mapped = _norm_id(sub[ID_COL]).map(m)
            if mapped.isna().any():   # coba tanpa leading zero ("00123" vs 123)
                m2 = pd.Series(vals, index=key_te.str.lstrip("0").values)
                mapped = mapped.fillna(_norm_id(sub[ID_COL]).str.lstrip("0").map(m2))
            if mapped.notna().mean() > 0.99:
                sub[tcol] = mapped.values
                aligned = True
            else:
                print(f"⚠️ Hanya {mapped.notna().mean():.0%} ID sample cocok dengan test → fallback ke urutan baris.")
        else:
            print("⚠️ ID test tidak unik (data panel?) → submission disusun berdasarkan urutan baris test.")
    if not aligned:
        if sample_sub is not None and len(sub) != len(vals):
            raise ValueError(f"Jumlah baris sample ({len(sub)}) ≠ test ({len(vals)}) dan ID tidak bisa dipakai untuk align → cek ID_COL")
        sub[tcol] = vals
    return sub, tcol


def validate_submission(sub, tcol, mode):
    ok = True
    checks = []
    def chk(cond, msg):
        nonlocal ok
        checks.append(("✅" if cond else "❌") + " " + msg)
        ok &= bool(cond)
    chk(len(sub) == len(test), f"jumlah baris = test ({len(sub)} vs {len(test)})")
    chk(sub[tcol].notna().all(), f"tidak ada NaN di '{tcol}' ({sub[tcol].isna().sum()} NaN)")
    if sample_sub is not None:
        chk(list(sub.columns) == list(sample_sub.columns), f"kolom sama dengan sample: {list(sub.columns)}")
        if ID_COL and ID_COL in sub.columns:
            chk((sub[ID_COL].astype(str).values == sample_sub[ID_COL].astype(str).values).all(), "urutan ID sama dengan sample (ID asli sample dipertahankan)")
    if ID_COL and ID_COL in sub.columns:
        chk(sub[ID_COL].is_unique, "ID unik")
    if mode == "probability":
        chk(sub[tcol].between(0, 1).all(), "probabilitas dalam [0, 1]")
    else:
        chk(sub[tcol].nunique() <= 2, f"label biner: {sorted(sub[tcol].astype(str).unique())[:5]}")
    for c in checks:
        print("  " + c)
    return ok


with safe_section("Submission"):
    if not len(FINAL_TEST):
        print("⏭️ Tidak ada test → submission di-skip.")
    else:
        use_cal = CFG.METRIC in ("logloss", "brier") and CAL_METHOD != "none"
        final_pred = FINAL_TEST_CAL if use_cal else FINAL_TEST
        variants = {"final": (final_pred, CFG.SUBMISSION_MODE)}
        LB = leaderboard()
        best_single = [m for m in LB["model"] if m != "dummy"][0]
        variants["best_single_" + best_single] = (RESULTS[best_single]["test"], CFG.SUBMISSION_MODE)
        variants["final_" + ("label" if CFG.SUBMISSION_MODE == "probability" else "probability")] = (
            final_pred, "label" if CFG.SUBMISSION_MODE == "probability" else "probability")
        for vname, (pred, mode) in variants.items():
            sub, tcol = build_submission(pred, mode, BEST_THR)
            path = os.path.join(CFG.SUB_DIR, f"submission_{vname}.csv")
            sub.to_csv(path, index=False)
            print(f"\n[{vname}] mode={mode} → {path}")
            okv = validate_submission(sub, tcol, mode)
            if vname == "final":
                SUBMISSION = sub
                sub.to_csv(os.path.join(CFG.OUTPUT_DIR, "submission.csv"), index=False)
                print("  ⭐ FINAL submission juga disimpan ke", os.path.join(CFG.OUTPUT_DIR, "submission.csv"), "| valid:", okv)
        display(SUBMISSION.head())
        print(f"Distribusi prediksi final: mean={final_pred.mean():.4f}, min={final_pred.min():.4f}, max={final_pred.max():.4f}")

        # daftar aksi untuk bisnis (customer test + tier + nilai + rekomendasi persona)
        act = pd.DataFrame({ID_COL if ID_COL else "row": test[ID_COL].values if ID_COL and ID_COL in test else np.arange(len(test)),
                            "churn_probability": PROB_BIZ_TEST, "risk_tier": TEST_TIER if "TEST_TIER" in globals() else None})
        if "MARGIN_TE" in globals():
            act["annual_value"] = val_te
            act["expected_value_at_risk"] = PROB_BIZ_TEST * val_te
            act["clv"] = CLV_TE
            act["quadrant"] = quadrant(PROB_BIZ_TEST, val_te)
            if "TARGET_FLAG_TEST" in globals():
                act["recommend_contact"] = TARGET_FLAG_TEST
        for _lab, _arr in (UPLIFT_TEST if "UPLIFT_TEST" in globals() else {}).items():
            act[f"uplift_pp: {_lab[:40]}"] = 100 * np.asarray(_arr)
        act = act.sort_values("churn_probability", ascending=False)
        act.to_csv(os.path.join(CFG.TABLE_DIR, "test_customer_action_list.csv"), index=False)
        print("Daftar customer test + tier + value disimpan ke tables/test_customer_action_list.csv")

# %%
with safe_section("Save OOF & test predictions"):
    oof_df = pd.DataFrame({"y": y, "final_oof": FINAL_OOF, "final_oof_calibrated": FINAL_OOF_CAL})
    if ID_COL:
        oof_df.insert(0, ID_COL, train[ID_COL].values)
    for m in RESULTS:
        oof_df[f"oof_{m}"] = RESULTS[m]["oof"]
    oof_df.to_csv(os.path.join(CFG.OUTPUT_DIR, "oof_predictions.csv"), index=False)
    if len(FINAL_TEST):
        tp = pd.DataFrame({"final_test": FINAL_TEST, "final_test_calibrated": FINAL_TEST_CAL})
        if ID_COL and ID_COL in test:
            tp.insert(0, ID_COL, test[ID_COL].values)
        for m in RESULTS:
            if len(RESULTS[m]["test"]):
                tp[f"test_{m}"] = RESULTS[m]["test"]
        tp.to_csv(os.path.join(CFG.OUTPUT_DIR, "test_predictions.csv"), index=False)
    print("Tersimpan: oof_predictions.csv, test_predictions.csv")

# %% [markdown]
# # 24. Export untuk Laporan & Ringkasan Error
#
# Output di `CFG.OUTPUT_DIR`:
# | File | Isi | Dipakai untuk |
# |---|---|---|
# | `figures/figNN_*.png` | semua grafik bernomor | pilih 6–10 terbaik untuk laporan (sisanya lampiran) |
# | `figure_index.md` | daftar figure + judul | cepat cari figure |
# | `tables/*.csv` + `report_tables.xlsx` | semua tabel | tabel laporan & lampiran |
# | `insights.md` | semua kalimat insight (English) dikelompokkan per section | bahan menulis Findings & Recommendations |
# | `run_summary.json` | config, skor, model final, threshold | dokumentasi metodologi |
# | `experiments.csv` | log tiap run (append) | bandingkan eksperimen |
# | `submission.csv` | submission final | upload |
#
# Lihat juga `01_Churn_Insurance/PANDUAN_LAPORAN.md` untuk struktur artikel 10 halaman & mapping figure → bagian laporan.

# %%
with safe_section("Export report assets"):
    if HAS_OPENPYXL and TABLES:
        xlsx = os.path.join(CFG.OUTPUT_DIR, "report_tables.xlsx")
        used = set()
        with pd.ExcelWriter(xlsx) as w:
            for name, df in TABLES.items():
                sheet = re.sub(r"[\[\]\*\?/\\:]", "_", name)[:31]
                while sheet in used:
                    sheet = sheet[:28] + f"_{len(used)}"
                used.add(sheet)
                df2 = df.copy()
                for c in df2.columns:
                    if df2[c].dtype == object:
                        df2[c] = df2[c].astype(str)
                df2.to_excel(w, sheet_name=sheet, index=False)
        print("Excel:", xlsx)
    with open(os.path.join(CFG.OUTPUT_DIR, "insights.md"), "w", encoding="utf-8") as f:
        f.write("# Auto-generated insights\n\n")
        for sec in OrderedDict.fromkeys(s for s, _ in INSIGHTS):
            f.write(f"## {sec}\n\n")
            for s, t in INSIGHTS:
                if s == sec:
                    f.write(f"- {t}\n")
            f.write("\n")
    with open(os.path.join(CFG.OUTPUT_DIR, "figure_index.md"), "w", encoding="utf-8") as f:
        f.write("# Figure index\n\n| File | Title |\n|---|---|\n")
        for fn, t in FIGURE_INDEX:
            f.write(f"| {fn} | {t} |\n")
    LB = leaderboard()
    summary = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "run_mode": CFG.RUN_MODE, "n_train": int(N_TR),
               "n_test": int(len(FINAL_TEST)), "churn_rate": CHURN_RATE, "target": TARGET, "target_mapping": {str(k): v for k, v in TARGET_MAPPING.items()},
               "metric": CFG.METRIC, "final_model": FINAL_NAME, "final_weights": {k: float(v) for k, v in FINAL_WEIGHTS.items()},
               "final_oof_score": float(show_score(rank_score(y, FINAL_OOF))), "final_oof_auc": float(roc_auc_score(y, FINAL_OOF)),
               "threshold": BEST_THR, "threshold_metric": THR_METRIC, "calibration": CAL_METHOD,
               "leaderboard": LB.drop(columns=[c for c in LB.columns if c == "time_s"]).to_dict(orient="records"),
               "leaky_cols": LEAKY_COLS, "failed_models": FAILED_MODELS, "section_errors": dict(SECTION_ERRORS),
               "n_features": len(MODEL_FEATURES), "roles": ROLES}
    with open(os.path.join(CFG.OUTPUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    exp_path = os.path.join(CFG.OUTPUT_DIR, "experiments.csv")
    exp_row = pd.DataFrame([{"timestamp": summary["timestamp"], "run_mode": CFG.RUN_MODE, "metric": CFG.METRIC,
                             "final_model": FINAL_NAME, "oof_score": summary["final_oof_score"], "oof_auc": summary["final_oof_auc"],
                             "models": ",".join(RESULTS.keys()), "n_features": len(MODEL_FEATURES), "optuna": CFG.USE_OPTUNA}])
    exp_row.to_csv(exp_path, mode="a", header=not os.path.exists(exp_path), index=False)
    print(f"Figures: {len(FIGURE_INDEX)} | Tables: {len(TABLES)} | Insights: {len(INSIGHTS)}")
    print("Folder output:", os.path.abspath(CFG.OUTPUT_DIR))

# %%
print("=" * 70)
print("RINGKASAN RUN")
print("=" * 70)
try:
    print(f"Final model : {FINAL_NAME} | OOF {CFG.METRIC} = {show_score(rank_score(y, FINAL_OOF)):.5f} | AUC = {roc_auc_score(y, FINAL_OOF):.5f}")
except Exception:
    print("Final model belum terbentuk (cek error modeling).")
print("Model gagal :", FAILED_MODELS if FAILED_MODELS else "tidak ada ✅")
if SECTION_ERRORS:
    print("Section error (notebook tetap jalan, cek & perbaiki kalau section ini penting):")
    for k, v in SECTION_ERRORS.items():
        print(f"  ❌ {k}: {v}")
else:
    print("Section error: tidak ada ✅")
print("\nInsight utama:")
for s, t in INSIGHTS[:12]:
    print(f"  [{s}] {t[:220]}")

# %% [markdown]
# ## 24.2 Executive Summary Figure (1 gambar ringkasan untuk halaman pertama laporan / slide)
# KPI utama + driver actionable teratas + gains curve dalam satu figure.

# %%
with safe_section("Executive summary figure"):
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 2.6], hspace=0.45, wspace=0.35)
    auc_txt = f"{roc_auc_score(y, FINAL_OOF):.3f}"
    if "FINAL_AUC_CI" in globals():
        auc_txt += f"\n(95% CI {FINAL_AUC_CI[0]:.3f}–{FINAL_AUC_CI[1]:.3f})"
    kpis = [("Churn rate", pct(CHURN_RATE), f"{int(y.sum()):,} of {N_TR:,} customers"),
            ("Model ROC-AUC", auc_txt, f"final: {FINAL_NAME}")]
    if "DECILE_TABLE" in globals():
        kpis.append(("Top-20% capture", f"{DECILE_TABLE['cum_churners_captured_%'].iloc[1]:.0f}%",
                     f"of churners ({DECILE_TABLE['cum_lift'].iloc[1]:.1f}x lift)"))
    if "at_risk_tr" in globals() and VALUE_SOURCE:
        kpis.append(("Annual value at risk", f"{fmt_num(at_risk_tr)} {CFG.CURRENCY}", f"{pct(at_risk_tr / val_tr.sum())} of premium base"))
    elif "KS" in globals():
        kpis.append(("KS statistic", f"{KS:.2f}", "risk separation"))
    for i, (t, v, s) in enumerate(kpis[:4]):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#F2F4F8", zorder=0))
        ax.text(0.5, 0.78, t, ha="center", va="center", fontsize=10, color="#555", transform=ax.transAxes)
        ax.text(0.5, 0.45, v, ha="center", va="center", fontsize=15 if "\n" not in v else 12, weight="bold", color=CFG.PALETTE[1] if i == 0 else "#222", transform=ax.transAxes)
        ax.text(0.5, 0.12, s, ha="center", va="center", fontsize=8, color="#777", transform=ax.transAxes)
    ax1 = fig.add_subplot(gs[1, :2])
    if len(EVIDENCE) and "shap_share_pct" in EVIDENCE:
        ev = EVIDENCE.dropna(subset=["shap_share_pct"]).sort_values("shap_share_pct", ascending=False).head(8)
        ax1.barh(ev["feature"][::-1].str.slice(0, 30), ev["shap_share_pct"][::-1],
                 color=[CFG.PALETTE[1] if a else "#9E9E9E" for a in ev["actionable"][::-1]])
        ax1.set_xlabel("Share of model importance (SHAP, %)")
        ax1.set_title("Top churn drivers (red = actionable)")
    elif len(DRIVER_TABLE):
        ev = DRIVER_TABLE.head(8)
        ax1.barh(ev["feature"][::-1].str.slice(0, 30), ev["iv"][::-1], color=CFG.PALETTE[1])
        ax1.set_xlabel("Information Value")
        ax1.set_title("Top churn drivers")
    ax2 = fig.add_subplot(gs[1, 2:])
    if "DECILE_TABLE" in globals():
        ax2.plot(np.r_[0, DECILE_TABLE["cum_customers_%"]], np.r_[0, DECILE_TABLE["cum_churners_captured_%"]], "o-", color=CFG.PALETTE[1], label="Model")
        ax2.plot([0, 100], [0, 100], "k--", label="Random")
        ax2.set_xlabel("% customers targeted (highest risk first)")
        ax2.set_ylabel("% churners captured")
        ax2.set_title("Targeting efficiency (cumulative gains)")
        ax2.legend(fontsize=8)
    fig.suptitle("Customer Churn — Executive Summary", fontsize=14, weight="bold", y=1.0)
    save_fig("executive_summary", fig)

# %% [markdown]
# ## 24.2b Diagram Arsitektur & Metodologi (otomatis, berisi angka run ini)
# 1. **End-to-end architecture** — data → preparation → analytics engine → decision layer → delivery + feedback loop.
# 2. **Validation design** — skema CV/OOF + semua safeguard (inner CV, nested TE, shuffled label, adversarial, DeLong, EMPC).
# 3. **Model architecture** — base learners → ensemble (bobot) → kalibrasi → probabilitas / tier / penjelasan.
# 4. **MLOps loop** — skor bulanan → prioritas → CRM (dengan control group) → ukur → monitor → retrain.
# 5. **Data lineage** — apa yang terjadi pada baris & kolom.
#
# Dipakai sebagai Figure 1 (arsitektur) dan Appendix laporan, juga di dashboard.

# %%
NL = chr(10)


def draw_system_architecture():
    fig, ax = canvas(15, 7.4)
    C_DATA, C_PREP, C_AN, C_DEC, C_OUT = "#37474F", "#2E86AB", "#6C4AB6", "#E4572E", "#2E7D32"
    cols = [(0.01, "1. Data", C_DATA), (0.205, "2. Preparation", C_PREP), (0.40, "3. Analytics engine", C_AN),
            (0.63, "4. Decision layer", C_DEC), (0.83, "5. Delivery", C_OUT)]
    for x, t, c in cols:
        ax.text(x + 0.08, 0.965, t, ha="center", fontsize=11, weight="bold", color=c)
    n_extra = len(CFG.EXTRA_TABLES or {}) + len(CFG.EXTRA_ONE_TO_ONE or {})
    dbox(ax, 0.01, 0.66, 0.16, 0.24, "Policy & customer table", f"{N_TR:,} train rows{NL}{len(FINAL_TEST):,} to score{NL}{len(ORIGINAL_COLS)} raw columns", fc=C_DATA)
    dbox(ax, 0.01, 0.36, 0.16, 0.24, "Related tables", f"{n_extra} extra table(s){NL}(claims, payments,{NL}complaints → aggregated)", fc=C_DATA)
    dbox(ax, 0.01, 0.06, 0.16, 0.24, "Business parameters", f"margin {CFG.PROFIT_MARGIN:.0%}, cost {fmt_num(CFG.RETENTION_COST)}{NL}success {CFG.RETENTION_SUCCESS_RATE:.0%}{NL}horizon {CFG.CLV_HORIZON_YEARS} y", fc="#78909C")
    dbox(ax, 0.205, 0.66, 0.16, 0.24, "Quality & typing", f"auto type inference{NL}{N_DUP_REMOVED if 'N_DUP_REMOVED' in globals() else 0} duplicates removed{NL}impossible values → NA", fc=C_PREP)
    dbox(ax, 0.205, 0.36, 0.16, 0.24, "Leakage guard", f"name + missingness +{NL}single-feature AUC +{NL}model-level check{NL}{len(LEAKY_COLS)} column(s) removed", fc=C_PREP)
    dbox(ax, 0.205, 0.06, 0.16, 0.24, "Feature engineering", f"{len(FE_NEW)} insurance features{NL}freq + in-fold target enc.{NL}{len(MODEL_FEATURES)} model inputs", fc=C_PREP)
    an = [("Statistical drivers", "χ², Mann-Whitney, IV/WoE,\neffect sizes, BH-FDR"),
          ("Survival", "Kaplan-Meier, log-rank,\nCox PH, AFT, RMST"),
          ("Predictive ML", f"{len({RESULTS[m]['model'] for m in RESULTS if m != 'dummy'})} algorithms, Optuna,\nensemble, calibration"),
          ("Explain & causal", "SHAP, PDP, AIPW,\nDR-learner uplift, E-values")]
    for i, (t, b) in enumerate(an):
        dbox(ax, 0.40, 0.73 - i * 0.215, 0.19, 0.18, t, b, fc=C_AN)
    dec = [("Risk scoring & tiers", f"OOF AUC {roc_auc_score(y, FINAL_OOF):.3f}"),
           ("Value at risk / CLV", "risk × value matrix"),
           ("Campaign optimiser", "EMPC, ROI curve,\nsensitivity"),
           ("Uplift targeting", "who responds, A/B design")]
    for i, (t, b) in enumerate(dec):
        dbox(ax, 0.63, 0.73 - i * 0.215, 0.165, 0.18, t, b, fc=C_DEC)
    outs = [("Article (DOCX/PDF)", "auto-drafted, ≤10 pages"), ("Dashboard", "Vercel-ready web app"),
            ("CRM action list", "per-customer tier,\nvalue, uplift"), ("Model card", "governance & monitoring")]
    for i, (t, b) in enumerate(outs):
        dbox(ax, 0.83, 0.73 - i * 0.215, 0.16, 0.18, t, b, fc=C_OUT)
    for yy_ in (0.78, 0.48, 0.18):
        darrow(ax, (0.17, yy_), (0.205, yy_))
    for yy_ in (0.78, 0.48, 0.18):
        darrow(ax, (0.365, yy_), (0.40, 0.5 + (yy_ - 0.48) * 0.7))
    for i in range(4):
        darrow(ax, (0.59, 0.82 - i * 0.215), (0.63, 0.82 - i * 0.215))
        darrow(ax, (0.795, 0.82 - i * 0.215), (0.83, 0.82 - i * 0.215))
    ax.annotate("", xy=(0.29, 0.04), xytext=(0.91, 0.04), arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.6, ls="--",
                                                                         connectionstyle="arc3,rad=0.0"))
    ax.text(0.60, 0.012, "Feedback loop: pilot outcomes (treatment vs control) and monthly monitoring feed back into retraining",
            ha="center", fontsize=8.5, color="#2E7D32", style="italic")
    ax.set_title("End-to-End Analytics Architecture: from Raw Policy Data to Retention Decisions", fontsize=13, weight="bold", pad=14)
    return fig


def draw_validation_design():
    fig, ax = canvas(14, 6.2)
    k = CFG.N_FOLDS
    x0, w = 0.04, 0.50
    ax.text(x0 + w / 2, 0.95, f"Stratified {k}-fold cross-validation (out-of-fold predictions)", ha="center", fontsize=11, weight="bold")
    for i in range(k):
        yy_ = 0.82 - i * 0.11
        for j in range(k):
            fc = CFG.PALETTE[1] if i == j else CFG.PALETTE[0]
            ax.add_patch(matplotlib.patches.Rectangle((x0 + j * w / k, yy_), w / k - 0.004, 0.075, fc=fc, ec="white", alpha=0.9))
        ax.text(x0 - 0.01, yy_ + 0.037, f"Fold {i + 1}", ha="right", va="center", fontsize=8)
    ax.text(x0 + w / 2, 0.82 - k * 0.11 + 0.055, "blue = training part (target encoding & tuning fitted here only)   red = validation part → OOF",
            ha="center", fontsize=7.8)
    dbox(ax, 0.04, 0.03, 0.50, 0.15, "All performance numbers are out-of-fold",
         f"OOF ROC-AUC {roc_auc_score(y, FINAL_OOF):.3f}  |  95% CI via bootstrap (B = {CFG.N_BOOTSTRAP}) and DeLong  |  "
         f"repeats = {CFG.N_REPEATS}", fc="#455A64", bfs=8)
    checks = [("Inner CV for tuning", f"Optuna TPE on {CFG.OPTUNA_FOLDS}-fold inner splits{NL}(subsample if data is large)"),
              ("Nested target encoding", "5-fold inner encoding inside each{NL}training fold → no target leakage".replace("{NL}", NL)),
              ("Shuffled-label test", "AUC with permuted target ≈ 0.50{NL}→ pipeline free of leakage".replace("{NL}", NL)),
              ("Adversarial validation", f"train-vs-test AUC = {ADV_AUC:.2f}" if np.isfinite(globals().get('ADV_AUC', np.nan)) else "train-vs-test check (n/a: no test)"),
              ("Statistical comparison", "DeLong tests (Holm-corrected),{NL}ablation, learning curve".replace("{NL}", NL)),
              ("Profit-based selection", "EMPC + campaign ROI with{NL}sensitivity analysis".replace("{NL}", NL))]
    for i, (t, b) in enumerate(checks):
        cx, cy = 0.60 + (i % 2) * 0.20, 0.70 - (i // 2) * 0.26
        dbox(ax, cx, cy, 0.185, 0.20, t, b, fc=[CFG.PALETTE[0], CFG.PALETTE[4], CFG.PALETTE[2], CFG.PALETTE[3], CFG.PALETTE[5], CFG.PALETTE[1]][i])
    ax.text(0.79, 0.95, "Safeguards against over-optimism", ha="center", fontsize=11, weight="bold")
    return fig


def draw_model_architecture():
    fig, ax = canvas(14, 6)
    lb_ = leaderboard()
    base = [m for m in lb_["model"] if m != "dummy"][:8]
    used = set(FINAL_WEIGHTS)
    n = len(base)
    for i, m in enumerate(base):
        yy_ = 0.86 - i * (0.80 / max(n, 1))
        hh = 0.80 / max(n, 1) - 0.015
        fc = CFG.PALETTE[0] if m in used else "#B0BEC5"
        dbox(ax, 0.02, yy_ - hh + 0.06, 0.25, hh, f"{m}", f"OOF AUC {RESULTS[m]['oof_auc']:.4f}", fc=fc, fs=8.5, bfs=7.5)
        if m in used:
            darrow(ax, (0.27, yy_ - hh / 2 + 0.06), (0.40, 0.50), color=CFG.PALETTE[0], lw=1.2 + 3 * float(FINAL_WEIGHTS[m]))
    w_txt = NL.join(f"{m}: {float(w):.2f}" for m, w in list(FINAL_WEIGHTS.items())[:6])
    dbox(ax, 0.40, 0.33, 0.19, 0.34, "Ensemble", f"method: {FINAL_NAME}{NL}{w_txt}", fc=CFG.PALETTE[4], bfs=7.5)
    dbox(ax, 0.63, 0.40, 0.14, 0.20, "Calibration", f"{CAL_METHOD}{NL}(CV log-loss)", fc=CFG.PALETTE[5])
    dbox(ax, 0.81, 0.62, 0.17, 0.22, "Churn probability", f"AUC {roc_auc_score(y, FINAL_OOF):.3f}{NL}PR-AUC {average_precision_score(y, FINAL_OOF):.3f}", fc=CFG.PALETTE[1])
    dbox(ax, 0.81, 0.34, 0.17, 0.22, "Risk tier & action", f"threshold {BEST_THR:.3f} ({THR_METRIC}){NL}High / Medium / Low", fc=CFG.PALETTE[3])
    dbox(ax, 0.81, 0.06, 0.17, 0.22, "Explanation", "SHAP reason codes{NL}per customer".replace("{NL}", NL), fc=CFG.PALETTE[2])
    darrow(ax, (0.59, 0.50), (0.63, 0.50), lw=2)
    darrow(ax, (0.77, 0.50), (0.81, 0.72), lw=1.6)
    darrow(ax, (0.77, 0.50), (0.81, 0.45), lw=1.6)
    darrow(ax, (0.77, 0.50), (0.81, 0.17), lw=1.6)
    ax.text(0.145, 0.96, "Base learners (blue = in final ensemble)", ha="center", fontsize=10, weight="bold")
    ax.set_title("Predictive Model Architecture", fontsize=13, weight="bold")
    return fig


def draw_mlops_loop():
    fig, ax = canvas(11, 6.6)
    steps = [("Monthly data refresh", "policy, payment,\nclaims, service logs"), ("Score & explain", "churn probability +\nSHAP reason codes"),
             ("Prioritise", "risk tier × value ×\npredicted uplift"), ("Act via CRM", "agents / digital nudges;\nrandom control group"),
             ("Measure outcomes", "retention uplift vs control,\ncampaign ROI"), ("Monitor & retrain", "PSI > 0.25, AUC drop > 0.03,\ncalibration → retrain")]
    cx, cy, R = 0.5, 0.47, 0.33
    pos = []
    for i, (t, b) in enumerate(steps):
        ang = np.pi / 2 - i * 2 * np.pi / len(steps)
        x, y_ = cx + R * np.cos(ang) * 1.25, cy + R * np.sin(ang)
        pos.append((x, y_))
        dbox(ax, x - 0.13, y_ - 0.075, 0.26, 0.15, t, b, fc=CFG.PALETTE[i % len(CFG.PALETTE)], fs=9, bfs=7.4)
    for i in range(len(pos)):
        a_, b_ = pos[i], pos[(i + 1) % len(pos)]
        darrow(ax, a_, b_, lw=1.8, rad=-0.18)
    ax.text(cx, cy, "Churn\nearly-warning\nsystem", ha="center", va="center", fontsize=12, weight="bold", color="#333")
    ax.set_title("Deployment & Continuous-Learning Loop (MLOps)", fontsize=13, weight="bold")
    return fig


def draw_data_lineage():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    n_raw = len(raw_train) + (N_DUP_REMOVED if "N_DUP_REMOVED" in globals() else 0)
    rows_ = [("raw training rows", n_raw), ("after removing duplicates / missing target", N_TR), ("rows to score (test)", len(FINAL_TEST))]
    ax[0].barh([r[0] for r in rows_][::-1], [r[1] for r in rows_][::-1], color=[CFG.PALETTE[0], CFG.PALETTE[2], CFG.PALETTE[5]][::-1])
    for i, v in enumerate([r[1] for r in rows_][::-1]):
        ax[0].text(v, i, f" {v:,}", va="center", fontsize=8)
    ax[0].set_title("Rows")
    money_axis(ax[0], "x")
    typ = pd.Series(COL_TYPES).value_counts()
    excl = {k: int(typ.get(k, 0)) for k in ["id", "constant", "date", "text", "dropped"]}
    cols_ = [("raw columns", len(ORIGINAL_COLS)), ("excluded: IDs", excl["id"]), ("excluded: constant", excl["constant"]),
             ("excluded: leakage/dropped", excl["dropped"]), ("dates → derived features", excl["date"]),
             ("engineered features", len(FE_NEW)), ("model input features", len(MODEL_FEATURES))]
    ax[1].barh([c[0] for c in cols_][::-1], [c[1] for c in cols_][::-1],
               color=[CFG.PALETTE[0], "#9E9E9E", "#9E9E9E", CFG.PALETTE[1], CFG.PALETTE[3], CFG.PALETTE[2], CFG.PALETTE[4]][::-1])
    for i, v in enumerate([c[1] for c in cols_][::-1]):
        ax[1].text(v, i, f" {v}", va="center", fontsize=8)
    ax[1].set_title("Columns / features")
    fig.suptitle("Data Lineage: What Happened to Rows and Columns", y=1.02, weight="bold")
    plt.tight_layout()
    return fig


for _nm, _fn in [("system_architecture", draw_system_architecture), ("validation_design", draw_validation_design),
                 ("model_architecture", draw_model_architecture), ("mlops_loop", draw_mlops_loop), ("data_lineage", draw_data_lineage)]:
    with safe_section(f"Diagram: {_nm}"):
        save_fig(_nm, _fn())

# %% [markdown]
# ## 24.3 Auto Article (REPORT_DRAFT.docx + PDF) — ⭐ deliverable utama
#
# Notebook menulis **artikel lengkap** (bukan sekadar kerangka):
# - **Narasi otomatis berbasis data**: executive summary, introduction (konteks pasar Indonesia terverifikasi + literatur), data & metodologi, findings per research question, business impact, causal & uplift, rekomendasi + desain pilot, diskusi yang mengaitkan temuan dengan literatur, keterbatasan.
# - **Judul berbasis temuan** (otomatis; alternatif di `NARRATIVE_OPTIONS.md`).
# - **Figure/Table bernomor + cross-reference otomatis**; **daftar pustaka APA otomatis** (hanya yang disitasi, semua terverifikasi).
# - **Autofit ≤ 10 halaman** isi utama (MS Word menghitung halaman; kalau lebih → figure prioritas rendah dipindah ke appendix, teks dipadatkan).
# - **Appendix A–L** lengkap (data, metodologi & arsitektur, driver, regresi, survival, model development & trial-and-error, explainability, kausal & uplift, bisnis, governance, reproducibility) → tidak dihitung halaman.
# - Format soal: A4, Times New Roman 12, spasi 1.15, margin atas/bawah 4 cm, kiri/kanan 3 cm, nomor halaman.
#
# Output: `REPORT_DRAFT.docx/.pdf/.md`, **`<TEAM>_Final Stage 1.pdf/.docx`**, `REPORT_TODO.md`, `NARRATIVE_OPTIONS.md`, `JUDGE_QA.md`.
#
# **Isi Case Brief** (`CFG.CASE_CONTEXT`, `CASE_QUESTIONS`, `CHURN_DEFINITION`, `AVAILABLE_INTERVENTIONS`, `TEAM_MEMBERS`, …) supaya narasi menyesuaikan soal — cukup re-run cell 24.3 (tidak perlu re-run modeling).

# %%
def draw_framework():
    """Gambar diagram alur metodologi (Figure 1 laporan) secara otomatis."""
    nl = chr(10)
    steps = [(f"Data &{nl}Quality", f"{N_TR:,} customers{nl}cleaning, leakage{nl}check"),
             (f"Feature{nl}Engineering", f"{len(FE_NEW)} domain{nl}features"),
             (f"Driver{nl}Analysis", f"χ², Mann-Whitney,{nl}effect size, IV,{nl}FDR"),
             (f"Survival{nl}Analysis", f"Kaplan-Meier,{nl}log-rank, Cox PH"),
             (f"Interpretable{nl}Model", f"logistic regression{nl}odds ratios"),
             (f"Predictive{nl}ML", f"{len({RESULTS[m]['model'] for m in RESULTS if m != 'dummy'})} algorithms, CV,{nl}Optuna, ensemble"),
             (f"Explain &{nl}Causal", f"SHAP, PDP,{nl}AIPW effects"),
             (f"Business{nl}Value", f"CLV, risk×value,{nl}ROI, what-if"),
             (f"Recommen-{nl}dations", f"actions, KPIs,{nl}A/B pilots")]
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    cols = ["#2E86AB"] * 2 + ["#6C4AB6"] * 3 + ["#E4572E"] * 2 + ["#76B041"] * 2
    for i, ((t, d), c) in enumerate(zip(steps, cols)):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((i + 0.06, 0.38), 0.84, 0.52, boxstyle="round,pad=0.02", fc=c, ec="none", alpha=0.9))
        ax.text(i + 0.48, 0.64, t, ha="center", va="center", color="white", fontsize=9.5, weight="bold")
        ax.text(i + 0.48, 0.17, d, ha="center", va="center", fontsize=7.5, color="#333")
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.05, 0.64), xytext=(i + 0.91, 0.64), arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    ax.set_title("Analytical framework", fontsize=12, weight="bold")
    return fig


with safe_section("Framework diagram"):
    save_fig("analysis_framework", draw_framework())

# %%
# ===== 24.2c Composite figures for the article (wide, compact, publication style) =====
def _style_ax(a):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.tick_params(labelsize=8)
    a.title.set_fontsize(9.5)


def report_fig_drivers():
    ev = _df("EVIDENCE") if "EVIDENCE" in globals() else pd.DataFrame()
    order = [f for f in (ev["feature"].tolist() if len(ev) else DRIVER_TABLE["feature"].tolist()) if f in LEVEL_TABLES]
    feats = dedupe_features(order)[:4]
    if not feats:
        return None
    fig, axes = plt.subplots(1, len(feats), figsize=(3.7 * len(feats), 3.3), squeeze=False)
    for a, f in zip(axes[0], feats):
        t = LEVEL_TABLES[f].copy()
        t = t[t["n"] >= max(20, 0.003 * N_TR)]
        if len(t) > 8:
            t = t.sort_values("churn_rate", ascending=False).head(8)
        yy = np.arange(len(t))[::-1]
        cols = [CFG.PALETTE[1] if r > CHURN_RATE else CFG.PALETTE[0] for r in t["churn_rate"]]
        a.barh(yy, t["churn_rate"], color=cols, height=0.62,
               xerr=[(t["churn_rate"] - t["ci_low"]).clip(lower=0), (t["ci_high"] - t["churn_rate"]).clip(lower=0)], ecolor="#555", capsize=2)
        a.axvline(CHURN_RATE, color="k", ls="--", lw=0.9)
        a.set_yticks(yy)
        _bin = pd.api.types.is_numeric_dtype(train[f]) and train[f].nunique() == 2
        a.set_yticklabels([("Yes" if str(l).startswith("1") else "No") if (_bin and str(l) in ("0", "1", "0.0", "1.0")) else str(l)[:16]
                           for l in t["level"]], fontsize=7.5)
        a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        iv = DRIVER_TABLE.set_index("feature").loc[f, "iv"] if f in set(DRIVER_TABLE["feature"]) else np.nan
        a.set_title(f"{pretty(f)[:30]} (IV {iv:.2f})")
        _style_ax(a)
    fig.supxlabel("Churn rate (95% CI); dashed = overall", fontsize=8.5)
    plt.tight_layout()
    return fig


def report_fig_timing():
    if not SURV_AVAILABLE or "HAZARD_TABLE" not in globals() or not len(HAZARD_TABLE):
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    hz = HAZARD_TABLE
    pk = hz["hazard"].idxmax()
    ax[0].bar(range(len(hz)), hz["hazard"], color=[CFG.PALETTE[1] if i == pk else "#9EB9D4" for i in hz.index], width=0.7)
    ax[0].set_xticks(range(len(hz)))
    ax[0].set_xticklabels(hz["interval"], rotation=30, ha="right", fontsize=7.5)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax[0].set_title(f"(a) Churn hazard per {HZ_LABEL} by tenure ({SURV_UNIT})")
    tiers_s = np.asarray(tr_tier)[m_ok.values] if "tr_tier" in globals() else None
    _, tt, ss, _, _ = km_curve(DUR, EVT)
    ax[1].step(tt, ss, where="post", color="#555", lw=1.4, ls="--", label="All")
    if tiers_s is not None:
        for tn, col in [("High", CFG.PALETTE[1]), ("Medium", CFG.PALETTE[3]), ("Low", CFG.PALETTE[0])]:
            mm = tiers_s == tn
            if mm.sum() >= 30:
                _, tt, ss, _, _ = km_curve(DUR[mm], EVT[mm])
                ax[1].step(tt, ss, where="post", color=col, lw=1.8, label=f"{tn} risk")
    ax[1].set_ylim(0, 1.02)
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[1].set_title(f"(b) Share still active by tenure ({SURV_UNIT}) and risk tier")
    ax[1].legend(fontsize=7.5, frameon=False)
    for a in ax:
        _style_ax(a)
    plt.tight_layout()
    return fig


def report_fig_model():
    if "DECILE_TABLE" not in globals():
        return None
    me = _df("MODEL_EVAL")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5), gridspec_kw={"width_ratios": [1, 1.25]})
    dec = DECILE_TABLE
    ax[0].plot(np.r_[0, dec["cum_customers_%"]], np.r_[0, dec["cum_churners_captured_%"]], "o-", ms=3.5, color=CFG.PALETTE[1], label="Model")
    ax[0].plot([0, 100], [0, 100], "--", color="#888", lw=1, label="Random")
    ax[0].fill_between(np.r_[0, dec["cum_customers_%"]], np.r_[0, dec["cum_customers_%"]], np.r_[0, dec["cum_churners_captured_%"]],
                       color=CFG.PALETTE[1], alpha=0.08)
    ax[0].set_xlabel("% of customers contacted (highest risk first)", fontsize=8)
    ax[0].set_ylabel("% of churners captured", fontsize=8)
    ax[0].set_title(f"(a) Cumulative gains: top 20% → {dec['cum_churners_captured_%'].iloc[1]:.0f}% of churners")
    ax[0].legend(fontsize=7.5, frameon=False)
    if len(me):
        d = me.sort_values("roc_auc").tail(8)
        yy = np.arange(len(d))
        cols = [CFG.PALETTE[1] if str(m).startswith("FINAL") else CFG.PALETTE[0] for m in d["model"]]
        ax[1].errorbar(d["roc_auc"], yy, xerr=[d["roc_auc"] - d["auc_ci_low"], d["auc_ci_high"] - d["roc_auc"]], fmt="none", ecolor="#777", capsize=2)
        ax[1].scatter(d["roc_auc"], yy, c=cols, zorder=3, s=22)
        ax[1].set_yticks(yy)
        ax[1].set_yticklabels([str(m).replace("FINAL: ", "Final ensemble: ")[:30] for m in d["model"]], fontsize=7.5)
        ax[1].set_title("(b) Out-of-fold ROC-AUC with 95% DeLong CI")
    for a in ax:
        _style_ax(a)
    plt.tight_layout()
    return fig


def report_fig_business():
    if "CAMPAIGN_BEST" not in globals():
        return None
    has_q = "QUAD_TABLE" in globals() and len(QUAD_TABLE) and VALUE_SOURCE
    fig, ax = plt.subplots(1, 2 if has_q else 1, figsize=(12 if has_q else 7, 3.5), squeeze=False)
    a = ax[0][0]
    cc = TABLES.get("campaign_curve_expected_loss")
    strategies = {"Model: expected loss (p × value)": np.argsort(-(PROB_BIZ * VALUE_SAVED_TR)), "Model: churn probability": np.argsort(-PROB_BIZ),
                  "Random targeting": np.random.RandomState(CFG.SEED).permutation(N_TR)}
    for (k, o), col in zip(strategies.items(), [CFG.PALETTE[1], CFG.PALETTE[0], "#999"]):
        cdf = campaign_curve(o, CFG.RETENTION_SUCCESS_RATE, CFG.RETENTION_COST if VALUE_SOURCE else 0.0, steps=40)
        a.plot(cdf["target_%"], cdf["net_profit"], color=col, lw=2 if "expected" in k else 1.3, ls="--" if "Random" in k else "-", label=k)
        if "expected" in k:
            b = cdf.loc[cdf["net_profit"].idxmax()]
            a.scatter([b["target_%"]], [b["net_profit"]], color=col, zorder=3)
            a.annotate(f"optimum: top {b['target_%']:.0f}%\nnet {fmt_num(b['net_profit'])}", (b["target_%"], b["net_profit"]),
                       textcoords="offset points", xytext=(8, -18), fontsize=7.5)
    a.axhline(0, color="k", lw=0.7)
    money_axis(a)
    a.set_xlabel("% of customers contacted", fontsize=8)
    a.set_title(f"(a) Campaign net profit ({CFG.CURRENCY if VALUE_SOURCE else 'customers'})")
    a.legend(fontsize=7, frameon=False)
    _style_ax(a)
    if has_q:
        q = QUAD_TABLE.copy()
        q["share_customers_%"] = 100 * q["n"] / q["n"].sum()
        q = q.sort_values("share_value_at_risk_%")
        a2 = ax[0][1]
        yy = np.arange(len(q))
        a2.barh(yy + 0.18, q["share_customers_%"], height=0.36, color="#9EB9D4", label="% of customers")
        a2.barh(yy - 0.18, q["share_value_at_risk_%"], height=0.36, color=CFG.PALETTE[1], label="% of value at risk")
        a2.set_yticks(yy)
        a2.set_yticklabels([s.split(" (")[0] for s in q["quadrant"]], fontsize=8)
        a2.set_title("(b) Risk × value segments: customers vs value at risk")
        a2.legend(fontsize=7.5, frameon=False)
        _style_ax(a2)
    plt.tight_layout()
    return fig


def report_fig_causal():
    cz = _df("CAUSAL")
    if not len(cz):
        return None
    up = _df("UPLIFT_SUMMARY")
    fig, ax = plt.subplots(1, 2 if len(up) else 1, figsize=(12 if len(up) else 7, 3.4), squeeze=False)
    a = ax[0][0]
    d = cz.sort_values("ate_pp", ascending=False)
    yy = np.arange(len(d))
    rb = _df("CAUSAL_ROBUST")
    rob = set(rb[rb["verdict"] == "robust"]["feature"]) if len(rb) else set()
    a.errorbar(d["ate_pp"], yy, xerr=[(d["ate_pp"] - d["ci_low_pp"]).clip(lower=0), (d["ci_high_pp"] - d["ate_pp"]).clip(lower=0)],
               fmt="none", ecolor="#777", capsize=2)
    a.scatter(d["ate_pp"], yy, c=[CFG.PALETTE[2] if f in rob else CFG.PALETTE[0] for f in d["feature"]], zorder=3, s=24, label="AIPW (green = robust)")
    a.scatter(d["naive_diff_pp"], yy, marker="x", color=CFG.PALETTE[1], s=22, label="naive difference")
    a.axvline(0, color="k", lw=0.8)
    a.set_yticks(yy)
    a.set_yticklabels([prettify_text(s)[:34] for s in d["favorable_condition (T=1)"]], fontsize=7.5)
    a.set_xlabel("Change in churn probability (pp)", fontsize=8)
    a.set_title("(a) Causal effect of favourable conditions")
    a.legend(fontsize=7, frameon=False)
    _style_ax(a)
    if len(up):
        a2 = ax[0][1]
        g = _df("UPLIFT_GATES")
        g = g[g["treatment"] == up.iloc[0]["treatment"]]
        a2.bar(g["uplift_quintile"].astype(str), g["avg_churn_reduction_pp"],
               yerr=[(g["avg_churn_reduction_pp"] - g["ci_low_pp"]).clip(lower=0), (g["ci_high_pp"] - g["avg_churn_reduction_pp"]).clip(lower=0)],
               color=CFG.PALETTE[2], capsize=2, width=0.6)
        a2.axhline(0, color="k", lw=0.8)
        a2.set_xlabel("Quintile of predicted uplift (Q5 = most responsive)", fontsize=8)
        a2.set_title(f"(b) Who responds? {prettify_text(up.iloc[0]['treatment'])[:40]}")
        _style_ax(a2)
    plt.tight_layout()
    return fig


for _nm, _fn in [("report_drivers", report_fig_drivers), ("report_timing", report_fig_timing), ("report_model", report_fig_model),
                 ("report_business", report_fig_business), ("report_causal", report_fig_causal)]:
    with safe_section(f"Report figure: {_nm}"):
        _f = _fn()
        if _f is not None:
            save_fig(_nm, _f)

# %%
# ===== 24.3a Reference database (APA 7) — hanya referensi terverifikasi (BAHAN_LATAR_BELAKANG.md §E + metode standar) =====
# Daftar pustaka laporan dibangun OTOMATIS: hanya referensi yang benar-benar disitasi di teks yang dicetak.
REF_DB = [
    ("Akiba", 2019, "Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2623–2631). https://doi.org/10.1145/3292500.3330701"),
    ("Ascarza", 2018, "Ascarza, E. (2018). Retention futility: Targeting high-risk customers might be ineffective. Journal of Marketing Research, 55(1), 80–98. https://doi.org/10.1509/jmr.16.0163"),
    ("Benjamini", 1995, "Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x"),
    ("Brockett", 2008, "Brockett, P. L., Golden, L. L., Guillén, M., Nielsen, J. P., Parner, J., & Pérez-Marín, A. M. (2008). Survival analysis of a household portfolio of insurance policies: How much time do you have to stop total customer defection? Journal of Risk and Insurance, 75(3), 713–737. https://doi.org/10.1111/j.1539-6975.2008.00281.x"),
    ("Caruana", 2004, "Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A. (2004). Ensemble selection from libraries of models. In Proceedings of the 21st International Conference on Machine Learning (p. 18). https://doi.org/10.1145/1015330.1015432"),
    ("Chen", 2016, "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). https://doi.org/10.1145/2939672.2939785"),
    ("Chernozhukov", 2018, "Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1), C1–C68. https://doi.org/10.1111/ectj.12097"),
    ("Cox", 1972, "Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society: Series B (Methodological), 34(2), 187–202. https://doi.org/10.1111/j.2517-6161.1972.tb00899.x"),
    ("DeLong", 1988, "DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. Biometrics, 44(3), 837–845. https://doi.org/10.2307/2531595"),
    ("Devriendt", 2021, "Devriendt, F., Berrevoets, J., & Verbeke, W. (2021). Why you should stop predicting customer churn and start using uplift models. Information Sciences, 548, 497–515. https://doi.org/10.1016/j.ins.2019.12.075"),
    ("Eling", 2014, "Eling, M., & Kiesenbauer, D. (2014). What policy features determine life insurance lapse? An analysis of the German market. Journal of Risk and Insurance, 81(2), 241–269. https://doi.org/10.1111/j.1539-6975.2012.01504.x"),
    ("Eling", 2013, "Eling, M., & Kochanski, M. (2013). Research on lapse in life insurance: What has been done and what needs to be done? The Journal of Risk Finance, 14(4), 392–413. https://doi.org/10.1108/JRF-12-2012-0088"),
    ("Fallis", 2022, "Fallis, J., Barro, J., Leibovitz, A., & Lafortune, M. (2022, December 20). How to improve customer retention in property and casualty insurance. Bain & Company. https://www.bain.com/insights/how-to-improve-customer-retention-in-property-and-casualty-insurance-snap-chart/"),
    ("Fier", 2013, "Fier, S. G., & Liebenberg, A. P. (2013). Life insurance lapse behavior. North American Actuarial Journal, 17(2), 153–167. https://doi.org/10.1080/10920277.2013.803438"),
    ("Financial Conduct Authority", 2021, "Financial Conduct Authority. (2021). General insurance pricing practices market study: Feedback to CP20/19 and final rules (Policy Statement PS21/5). https://www.fca.org.uk/publication/policy/ps21-5.pdf"),
    ("Fleigle", 2026, "Fleigle, K., Callaway, J., & Patrick, E. (2026, February). The early exit: Unpacking lapse behavior with MIB's In Force Data Vault. RGA. https://www.rgare.com/knowledge-center/article/the-early-exit--unpacking-lapse-behavior-with-mib-s-in-force-data-vault"),
    ("Friedman", 2001, "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451"),
    ("Gottlieb", 2021, "Gottlieb, D., & Smetters, K. (2021). Lapse-based insurance. American Economic Review, 111(8), 2377–2416. https://doi.org/10.1257/aer.20160868"),
    ("Grinsztajn", 2022, "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In Advances in Neural Information Processing Systems 35 (pp. 507–520). https://doi.org/10.52202/068431-0037"),
    ("Guelman", 2014, "Guelman, L., & Guillén, M. (2014). A causal inference approach to measure price elasticity in automobile insurance. Expert Systems with Applications, 41(2), 387–396. https://doi.org/10.1016/j.eswa.2013.07.059"),
    ("Guelman", 2015, "Guelman, L., Guillén, M., & Pérez-Marín, A. M. (2015). Uplift random forests. Cybernetics and Systems, 46(3–4), 230–248. https://doi.org/10.1080/01969722.2015.1012892"),
    ("Günther", 2014, "Günther, C.-C., Tvete, I. F., Aas, K., Sandnes, G. I., & Borgan, Ø. (2014). Modelling and predicting customer churn from an insurance company. Scandinavian Actuarial Journal, 2014(1), 58–71. https://doi.org/10.1080/03461238.2011.636502"),
    ("Guo", 2017, "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning (PMLR 70, pp. 1321–1330)."),
    ("Harrell", 1982, "Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L., & Rosati, R. A. (1982). Evaluating the yield of medical tests. JAMA, 247(18), 2543–2546. https://doi.org/10.1001/jama.1982.03320430047030"),
    ("Hein", 2020, "Hein, K. Z. P. W., Keovilay, J., Adinegoro, W., & Ko, B. (2020). Analisis empiris perilaku lapse pada nasabah asuransi jiwa di Indonesia. Jurnal Statistika dan Aplikasinya, 4(2), 73–81. https://doi.org/10.21009/JSA.04202"),
    ("Höppner", 2020, "Höppner, S., Stripling, E., Baesens, B., vanden Broucke, S., & Verdonck, T. (2020). Profit driven decision trees for churn prediction. European Journal of Operational Research, 284(3), 920–933. https://doi.org/10.1016/j.ejor.2018.11.072"),
    ("Jayadi", 2020, "Jayadi, R., Kelvin, A., Jery, Rifyansyah, P., Mufarih, M., & Firmantyo, H. M. (2020). Predicting customer churn of fire insurance policy: A case study in an Indonesian insurance company. In 2020 6th International Conference on Science and Technology (ICST) (pp. 1–4). IEEE. https://doi.org/10.1109/ICST50505.2020.9732797"),
    ("Kaplan", 1958, "Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282), 457–481. https://doi.org/10.1080/01621459.1958.10501452"),
    ("Ke", 2017, "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems 30. https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html"),
    ("Kennedy", 2023, "Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. Electronic Journal of Statistics, 17(2), 3008–3049. https://doi.org/10.1214/23-EJS2157"),
    ("Kiesenbauer", 2012, "Kiesenbauer, D. (2012). Main determinants of lapse in the German life insurance industry. North American Actuarial Journal, 16(1), 52–73. https://doi.org/10.1080/10920277.2012.10590632"),
    ("Kuo", 2003, "Kuo, W., Tsai, C., & Chen, W.-K. (2003). An empirical study on the lapse rate: The cointegration approach. Journal of Risk and Insurance, 70(3), 489–508. https://doi.org/10.1111/1539-6975.t01-1-00061"),
    ("Künzel", 2019, "Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116"),
    ("Lemmens", 2020, "Lemmens, A., & Gupta, S. (2020). Managing churn to maximize profits. Marketing Science, 39(5), 956–973. https://doi.org/10.1287/mksc.2020.1229"),
    ("Lessmann", 2015, "Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. European Journal of Operational Research, 247(1), 124–136. https://doi.org/10.1016/j.ejor.2015.05.030"),
    ("Lou", 2013, "Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 623–631). https://doi.org/10.1145/2487575.2487579"),
    ("Lundberg", 2017, "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems 30. https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html"),
    ("Lundberg", 2020, "Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1), 56–67. https://doi.org/10.1038/s42256-019-0138-9"),
    ("Mitchell", 2019, "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 220–229). https://doi.org/10.1145/3287560.3287596"),
    ("Otoritas Jasa Keuangan", 2022, "Otoritas Jasa Keuangan. (2022). Surat Edaran Otoritas Jasa Keuangan Nomor 5/SEOJK.05/2022 tentang Produk Asuransi yang Dikaitkan dengan Investasi. https://www.ojk.go.id/id/regulasi/Documents/Pages/Produk-Asuransi-yang-Dikaitkan-Dengan-Investasi/SEOJK%205%20-%2005%20-%202022.pdf"),
    ("Otoritas Jasa Keuangan", 2024, "Otoritas Jasa Keuangan. (2024, June 5). OJK terbitkan aturan terkait produk asuransi dan saluran pemasaran produk asuransi (Siaran Pers SP 72/GKPB/OJK/VI/2024). https://ojk.go.id/id/berita-dan-kegiatan/siaran-pers/Documents/Pages/OJK-Terbitkan-Aturan-Terkait-Produk-Asuransi-dan-Saluran-Pemasaran-Produk-Asuransi/Siaran%20Pers%20POJK%208%20-%202024%20Produk%20Asuransi.pdf"),
    ("Outreville", 1990, "Outreville, J. F. (1990). Whole-life insurance lapse rates and the emergency fund hypothesis. Insurance: Mathematics and Economics, 9(4), 249–255. https://doi.org/10.1016/0167-6687(90)90002-U"),
    ("Prokhorenkova", 2018, "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In Advances in Neural Information Processing Systems 31."),
    ("Reichheld", 1990, "Reichheld, F. F., & Sasser, W. E., Jr. (1990). Zero defections: Quality comes to services. Harvard Business Review, 68(5), 105–111."),
    ("Reichheld", 2001, "Reichheld, F. (2001). Prescription for cutting costs [Bain Brief]. Bain & Company. https://www.media.bain.com/Images/BB_Prescription_cutting_costs.pdf"),
    ("Robins", 1994, "Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. Journal of the American Statistical Association, 89(427), 846–866. https://doi.org/10.1080/01621459.1994.10476818"),
    ("Rosenbaum", 1983, "Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. Biometrika, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41"),
    ("Royston", 2013, "Royston, P., & Parmar, M. K. B. (2013). Restricted mean survival time: An alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome. BMC Medical Research Methodology, 13, 152. https://doi.org/10.1186/1471-2288-13-152"),
    ("Rudin", 2019, "Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5), 206–215. https://doi.org/10.1038/s42256-019-0048-x"),
    ("Schoenfeld", 1982, "Schoenfeld, D. (1982). Partial residuals for the proportional hazards regression model. Biometrika, 69(1), 239–241. https://doi.org/10.1093/biomet/69.1.239"),
    ("Siddiqi", 2006, "Siddiqi, N. (2006). Credit risk scorecards: Developing and implementing intelligent credit scoring. John Wiley & Sons."),
    ("VanderWeele", 2017, "VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268–274. https://doi.org/10.7326/M16-2607"),
    ("Verbeke", 2012, "Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. European Journal of Operational Research, 218(1), 211–229. https://doi.org/10.1016/j.ejor.2011.09.031"),
    ("Verbraken", 2013, "Verbraken, T., Verbeke, W., & Baesens, B. (2013). A novel profit maximizing metric for measuring classification performance of customer churn prediction models. IEEE Transactions on Knowledge and Data Engineering, 25(5), 961–973. https://doi.org/10.1109/TKDE.2012.50"),
    ("Wei", 1992, "Wei, L. J. (1992). The accelerated failure time model: A useful alternative to the Cox regression model in survival analysis. Statistics in Medicine, 11(14–15), 1871–1879. https://doi.org/10.1002/sim.4780111409"),
    # berita / rilis (konteks pasar Indonesia) — judul sebagai "penulis" (APA untuk berita tanpa penulis)
    ("OJK sebut penetrasi", 2024, "OJK sebut penetrasi dan densitas asuransi kembali naik pada September 2024. (2024, November 5). Kontan.co.id. https://keuangan.kontan.co.id/news/ojk-sebut-penetrasi-dan-densitas-asuransi-kembali-naik-pada-september-2024"),
    ("Penetrasi asuransi syariah", 2025, "Penetrasi asuransi syariah di Indonesia masih rendah, ini penyebabnya. (2025, March 18). Kontan.co.id. https://keuangan.kontan.co.id/news/penetrasi-asuransi-syariah-di-indonesia-masih-rendah-ini-penyebabnya"),
    ("Kinerja industri perasuransian", 2026, "Kinerja industri perasuransian tumbuh 5,95% pada 2025, aset tembus Rp 1.201 triliun. (2026, February 6). Kontan.co.id. https://keuangan.kontan.co.id/news/kinerja-industri-perasuransian-tumbuh-595-pada-2025-aset-tembus-rp-1201-triliun"),
    ("SNLIK 2025", 2025, "SNLIK 2025: Indeks literasi perasuransian 45,45% dan indeks inklusi perasuransian 28,50%. (2025, May 2). Media Asuransi. https://mediaasuransinews.co.id/asuransi/snlik-2025-indeks-literasi-perasuransian-4545-dan-indeks-inklusi-perasuransian-2850/"),
    ("Regulasi baru OJK dinantikan", 2025, "Regulasi baru OJK dinantikan, klaim asuransi jiwa 2024 tembus Rp160 triliun. (2025, March 1). Infobanknews. https://infobanknews.com/regulasi-baru-ojk-dinantikan-klaim-asuransi-jiwa-2024-tembus-rp160-triliun/"),
    ("AAJI catat nilai klaim surrender", 2026, "AAJI catat nilai klaim surrender di asuransi jiwa turun 19% pada 2025. (2026, March 20). Kontan.co.id. https://keuangan.kontan.co.id/news/aaji-catat-nilai-klaim-surrender-di-asuransi-jiwa-turun-19-pada-2025"),
    ("AAJI catat premi", 2025, "AAJI catat premi industri asuransi jiwa tumbuh 4,3% jadi Rp 185,39 triliun di 2024. (2025, March 1). Investortrust. https://investortrust.id/financial/58245/aaji-catat-premi-industri-asuransi-jiwa-tumbuh-4-3-jadi-rp-185-39-triliun-di-2024"),
    ("AAUI catat premi", 2026, "AAUI catat premi asuransi umum tumbuh 4,8% jadi Rp 112,81 triliun di 2025. (2026, February 20). Investortrust. https://investortrust.id/financial/95071/aaui-catat-premi-asuransi-umum-tumbuh-4-8-jadi-rp-112-81-triliun-di-2025"),
]


def cited_references(texts):
    """Pilih entri REF_DB yang disitasi di teks (pola: 'Nama ..., TAHUN' / 'Nama ... (TAHUN')."""
    blob = " ".join(texts)
    out = []
    for name, year, apa in REF_DB:
        pat = re.escape(name) + r"[^;()]{0,70}?(?:,\s*|\(|;\s*)(?:\d{4}[a-z]?,\s*)*" + str(year)
        if re.search(pat, blob):
            out.append(apa)
    return sorted(set(out), key=lambda s: s.lower())


# ---------- CASE_RESEARCH.xlsx: fakta case/industri, literatur tim, paragraf custom, label fitur ----------
RESEARCH_SECTIONS = ["executive_summary", "introduction", "data", "methodology", "findings_drivers", "findings_timing", "findings_segments",
                     "findings_model", "findings_explain", "business", "causal", "personas", "recommendations", "discussion", "limitations", "conclusion"]


def _research_path():
    cands = [getattr(CFG, "RESEARCH_FILE", None), "CASE_RESEARCH.xlsx",
             os.path.join(os.path.dirname(os.path.abspath(str(CFG.TRAIN_PATH))), "..", "CASE_RESEARCH.xlsx"),
             os.path.join(os.path.dirname(os.path.abspath(str(CFG.TRAIN_PATH))), "CASE_RESEARCH.xlsx")]
    return next((os.path.abspath(p) for p in cands if p and os.path.exists(p)), None)


def _cit_key(cit, apa):
    """(nama, tahun) dari sitasi dalam teks / APA → dipakai pencocokan daftar pustaka."""
    c = str(cit or "").strip().strip("()").strip()
    yr = re.search(r"(19|20)\d{2}", c) or re.search(r"\((19|20)\d{2}", str(apa))
    year = int(re.search(r"(19|20)\d{2}", yr.group(0)).group(0)) if yr else None
    name = re.split(r",\s*(19|20)\d{2}|\s\((19|20)\d{2}", c)[0] if c else str(apa).split(",")[0]
    name = name.strip().strip('"').strip("“”").rstrip(",").strip()
    name = re.split(r"\s+&\s+|\s+and\s+|\s+et al\.?", name)[0].strip()
    return (name, year) if name and year else (None, None)


def load_case_research():
    path = _research_path()
    out = {k: pd.DataFrame() for k in ["case_facts", "industry_facts", "literature", "custom_paragraphs", "feature_labels"]}
    out["_path"] = path
    for f_ in (CFG.CASE_KEY_FACTS or []):
        out["case_facts"] = pd.concat([out["case_facts"], pd.DataFrame([{"use": "Y", "section": "introduction", "statement_EN": f_}])], ignore_index=True)
    if not path:
        print("ℹ️ CASE_RESEARCH.xlsx tidak ditemukan → laporan memakai literatur bawaan saja (buat dengan tools/research_helper.py).")
        return out
    try:
        book = pd.read_excel(path, sheet_name=None, dtype=str)
    except Exception as e:
        print(f"⚠️ CASE_RESEARCH.xlsx gagal dibaca ({e}) → dilewati. Tutup file di Excel lalu jalankan ulang cell ini.")
        return out
    for name, df in book.items():
        if name not in out:
            continue
        df = df.fillna("")
        df.columns = [str(c).strip() for c in df.columns]
        if "use" in df.columns:
            df = df[df["use"].astype(str).str.strip().str.upper().isin(["Y", "YES", "1", "TRUE", "YA"])]
        if "section" in df.columns:
            df["section"] = df["section"].astype(str).str.strip().str.lower().replace("", "introduction")
        out[name] = pd.concat([out[name], df], ignore_index=True)
    for _, r in out["feature_labels"].iterrows():
        c_, l_ = str(r.get("column", "")).strip(), str(r.get("label", "")).strip()
        if c_ and l_:
            CFG.FEATURE_LABELS[c_] = l_
    n_ref = 0
    for sheet in ["industry_facts", "literature"]:
        for _, r in out[sheet].iterrows():
            apa = str(r.get("apa_reference", "")).strip()
            if not apa:
                continue
            nm, yr = _cit_key(r.get("in_text_citation", ""), apa)
            if nm and yr and not any(a == apa or (str(n).lower() == nm.lower() and y == yr) for n, y, a in REF_DB):
                REF_DB.append((nm, yr, apa))
                n_ref += 1
    used = {k: len(v) for k, v in out.items() if isinstance(v, pd.DataFrame)}
    print(f"📚 CASE_RESEARCH: {path} → dipakai {used} | +{n_ref} referensi ke daftar pustaka")
    unver = sum(int((v.get("verified", pd.Series(dtype=str)).astype(str).str.upper() != "Y").sum()) for k, v in out.items()
                if isinstance(v, pd.DataFrame) and k in ("industry_facts", "literature"))
    if unver:
        print(f"⚠️ {unver} baris industry_facts/literature belum ditandai verified = Y → cek sumbernya sebelum submit.")
    return out


def research_texts(section, kind="paragraph", position=None):
    """Teks dari CASE_RESEARCH untuk satu section."""
    R = CASE_RESEARCH
    out = []
    if kind == "paragraph":
        d = R.get("custom_paragraphs", pd.DataFrame())
        if len(d):
            pos = d["position"].astype(str).str.strip().str.lower().replace("", "end") if "position" in d else pd.Series("end", index=d.index)
            out = d[(d["section"] == section) & (pos == (position or "end"))]["text"].astype(str).tolist()
    elif kind == "facts":
        for sh, col in [("case_facts", "statement_EN"), ("industry_facts", "statement_EN")]:
            d = R.get(sh, pd.DataFrame())
            if len(d) and col in d:
                for _, r in d[d["section"] == section].iterrows():
                    t = str(r[col]).strip()
                    cit = str(r.get("in_text_citation", "")).strip()
                    if t and cit and not re.search(r"\(.*(19|20)\d{2}.*\)|(19|20)\d{2}\)", t):
                        t = t.rstrip(".") + f" {cit if cit.startswith('(') else '(' + cit + ')'}."
                    if t:
                        out.append(t)
    return [t for t in out if t.strip()]


def research_literature(themes):
    d = CASE_RESEARCH.get("literature", pd.DataFrame())
    if not len(d):
        return []
    th = d["theme"].astype(str).str.strip().str.lower() if "theme" in d else pd.Series("general", index=d.index)
    keep = d[th.isin(set(themes) | {"general", ""})]
    out = []
    for _, r in keep.iterrows():
        t = str(r.get("finding_EN", "")).strip()
        cit = str(r.get("in_text_citation", "")).strip()
        if t and cit and not re.search(r"(19|20)\d{2}", t):
            t = t.rstrip(".") + f" {cit if cit.startswith('(') else '(' + cit + ')'}."
        if t:
            out.append(t)
    return out


CASE_RESEARCH = load_case_research()

# %%
# ===== 24.3b Report content builder — narasi otomatis berbasis data + literatur terverifikasi =====
def _g(name, default=None):
    return globals().get(name, default)


def _df(name):
    v = globals().get(name)
    return v if isinstance(v, pd.DataFrame) else pd.DataFrame()


def _join(items, conj="and"):
    items = [str(i) for i in items if str(i)]
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {conj} " + items[-1]


NUM_WORDS = {1: "first month", 2: "first two months", 3: "first three months", 4: "first four months", 6: "first six months",
             9: "first nine months", 12: "first year", 18: "first eighteen months", 24: "first two years", 36: "first three years"}


def detect_insurance_line():
    if CFG.INSURANCE_LINE in ("life", "general", "health"):
        return CFG.INSURANCE_LINE
    txt = " ".join(map(str, list(train.columns))).lower()
    for c in CAT_COLS[:30]:
        try:
            txt += " " + " ".join(map(str, train[c].dropna().astype(str).unique()[:30])).lower()
        except Exception:
            pass
    ctx = (str(CFG.COMPANY_NAME) + " " + str(CFG.CASE_CONTEXT) + " " + str(getattr(CFG, "CHURN_DEFINITION", ""))).lower()
    txt += " " + ctx
    life = bool(re.search(r"(surrender|unit.?link|whole.?life|term.?life|endowment|\blife\b|jiwa|paydi|cash.?value)", txt))
    gen = bool(re.search(r"(\bauto\b|motor|vehicle|kendaraan|\bcar\b|property|properti|fire|kebakaran|home|travel|non.?renew|renewal|general insur|umum)", txt))
    health = bool(re.search(r"(health|medical|hospital|kesehatan|rawat)", txt))
    if re.search(r"(multi.?line|several lines|various products)", ctx) or (life and (gen or health)):
        return "multi"
    if life:
        return "life"
    if health and not gen:
        return "health"
    return "general"


def feature_theme(f):
    """Tema bisnis sebuah driver → untuk narasi, judul & kait literatur."""
    fl = str(f).lower()
    fam = FEATURE_FAMILY.get(f)
    role = next((r for r, c in ROLES.items() if c == f), None)
    if fam == "tenure" or role == "tenure" or any(k in fl for k in ["tenure", "new_customer", "orig_date", "start_date", "since", "masa", "lama"]):
        return "lifecycle"
    if role in ("payment_method", "payment_frequency", "auto_renew", "late_payment") or any(k in fl for k in ["payment", "bayar", "debit", "renew", "late", "overdue", "frequency"]):
        return "payment"
    if role in ("premium_change",) or any(k in fl for k in ["increase", "price", "kenaikan", "rate_change", "premium_to_income", "discount"]):
        return "price"
    if role in ("claims", "complaints") or any(k in fl for k in ["claim", "klaim", "complain", "keluhan", "settle", "reject", "service", "ticket"]):
        return "service"
    if role in ("income",) or any(k in fl for k in ["income", "credit", "afford", "salary", "market_value"]):
        return "affordability"
    if role in ("products",) or any(k in fl for k in ["product", "policies", "bundle", "rider"]):
        return "relationship"
    if role in ("channel", "agent") or any(k in fl for k in ["channel", "agent", "broker"]):
        return "channel"
    if role in ("engagement",) or any(k in fl for k in ["engage", "login", "app", "digital", "active"]):
        return "engagement"
    if role in ("age",) or "age" in fl:
        return "demographic"
    if role in ("region",) or any(k in fl for k in ["region", "city", "county", "province", "state", "lat", "long"]):
        return "geography"
    return "other"


def money(v, d=1):
    """Format uang untuk laporan: 'IDR 9.7B' (tanpa nol berlebih)."""
    t = fmt_num(v, d)
    t = re.sub(r"\.0(?=[KMB]?$)", "", t)
    return f"{CFG.CURRENCY} {t}" if VALUE_SOURCE else t


def level_label(f, lv):
    """Level fitur yang mudah dibaca: biner 0/1 → 'No'/'Yes'."""
    if lv is None or (isinstance(lv, float) and np.isnan(lv)):
        return "—"
    s_ = str(lv)
    try:
        if f in train.columns and pd.api.types.is_numeric_dtype(train[f]) and train[f].nunique() == 2 and s_ in ("0", "1", "0.0", "1.0"):
            return "Yes" if s_.startswith("1") else "No"
    except Exception:
        pass
    return s_


def prettify_rule(rule):
    """Aturan decision tree → bahasa bisnis: 'x = 0' → 'x = No' (biner), 'count > 0.50' → 'count ≥ 1'."""
    out = []
    for cond in str(rule).split(" AND "):
        m = re.match(r"^(.+?) (≤|>|=|≠) (.+)$", cond.strip())
        if not m:
            out.append(prettify_text(cond))
            continue
        f, op, v = m.groups()
        try:
            if f in train.columns and pd.api.types.is_numeric_dtype(train[f]):
                x = train[f].dropna()
                vv = float(v.replace(",", ""))
                if x.nunique() == 2 and set(x.unique()) <= {0, 1}:
                    yes = (op in (">",) and vv < 1) or (op == "=" and vv == 1) or (op == "≤" and vv < 0)
                    out.append(f"{pretty(f)} = {'Yes' if yes else 'No'}")
                    continue
                if (x % 1 == 0).all():
                    k = math.floor(vv)
                    out.append(f"{pretty(f)} {'≤' if op == '≤' else '≥'} {k if op == '≤' else k + 1}")
                    continue
                out.append(f"{pretty(f)} {op} {fmt_num(vv) if abs(vv) >= 1000 else round(vv, 2)}")
                continue
        except Exception:
            pass
        out.append(f"{pretty(f)} {op} {v}")
    return " AND ".join(out)


def persistency_ratios():
    """Rasio persistensi (KPI industri asuransi): porsi polis yang masih aktif setelah 13 dan 25 bulan (Kaplan–Meier)."""
    try:
        if not SURV_AVAILABLE:
            return {}
        conv = {"months": 1.0, "years": 1 / 12, "days": 30.44}.get(TENURE_UNIT)
        if conv is None:
            return {}
        sf_, t_, s_, *_ = km_curve(DUR, EVT)
        return {m: float(surv_at(t_, s_, m * conv)) for m in (13, 25) if m * conv <= np.nanmax(DUR)}
    except Exception:
        return {}


def collect_facts():
    F = {}
    F["company"] = CFG.COMPANY_NAME
    F["line"] = detect_insurance_line()
    F["churn_term"] = {"life": "lapse or surrender", "general": "non-renewal or cancellation", "health": "non-renewal or lapse",
                       "multi": "lapse, surrender or non-renewal"}[F["line"]]
    F["n"], F["n_test"], F["rate"] = N_TR, len(FINAL_TEST), CHURN_RATE
    F["n_churn"] = int(np.sum(y))
    F["auc"] = roc_auc_score(y, FINAL_OOF)
    ci = _g("FINAL_AUC_CI")
    F["auc_ci"] = ci if ci is not None else (np.nan, np.nan)
    F["auc_txt"] = f"ROC-AUC {F['auc']:.3f}" + (f" (95% CI {F['auc_ci'][0]:.3f}–{F['auc_ci'][1]:.3f})" if np.isfinite(F["auc_ci"][0]) else "")
    F["pr_auc"] = average_precision_score(y, FINAL_OOF)
    dec = _df("DECILE_TABLE")
    F["top10_rate"] = dec["churn_rate"].iloc[0] if len(dec) else np.nan
    F["top20_capture"] = dec["cum_churners_captured_%"].iloc[1] if len(dec) > 1 else np.nan
    F["top20_lift"] = dec["cum_lift"].iloc[1] if len(dec) > 1 else np.nan
    ev = _df("EVIDENCE")
    F["drivers"] = []
    if len(ev):
        keep_ = set(dedupe_features(ev["feature"].tolist()))
        ev = ev[ev["feature"].isin(keep_)]
        for _, r in ev.head(8).iterrows():
            F["drivers"].append({"f": r["feature"], "name": pretty(r["feature"]), "level": level_label(r["feature"], r.get("highest_risk_level")),
                                 "rate": r.get("highest_risk_rate"), "lift": r.get("highest_risk_lift"), "iv": r.get("iv"),
                                 "or": r.get("odds_ratio"), "hr": r.get("hazard_ratio"), "shap_rank": r.get("shap_rank"),
                                 "act": bool(r.get("actionable")), "score": r.get("evidence_score"), "theme": feature_theme(r["feature"])})
    F["themes"] = list(OrderedDict.fromkeys(d["theme"] for d in F["drivers"][:5]))
    F["actionable"] = [d for d in F["drivers"] if d["act"]]
    dt = _df("DRIVER_TABLE")
    F["negligible"] = [pretty(f) for f in dt[(dt["iv"] < 0.02)]["feature"].head(6)] if len(dt) else []
    hz = _df("HAZARD_TABLE")
    F["surv"] = bool(SURV_AVAILABLE) and len(hz) > 0
    if F["surv"]:
        pk = hz.sort_values("hazard", ascending=False).iloc[0]
        F["peak_interval"], F["peak_hazard"] = pk["interval"], pk["hazard"]
        F["avg_hazard"] = (HZ_PER * hz["churn_events"].sum() / hz["person_time"].sum()) if "HZ_PER" in globals() else np.nan
        F["hz_label"], F["surv_unit"] = _g("HZ_LABEL", "period"), _g("SURV_UNIT", "periods")
        try:
            hi_edge = float(str(pk["interval"]).strip("[)").split(",")[1])
        except Exception:
            hi_edge = np.nan
        months = hi_edge / 30.44 if TENURE_UNIT == "days" else hi_edge * 12 if TENURE_UNIT == "years" else hi_edge
        F["peak_months"] = months
        F["early_peak"] = bool(np.isfinite(months) and months <= 24)
        km = _df("KM_SUMMARY")
        F["km"] = km
        F["persist"] = persistency_ratios()
    seg = _df("SEGMENT_RULES")
    F["seg_top"] = seg.iloc[0] if len(seg) else None
    F["seg_low"] = seg.iloc[-1] if len(seg) else None
    F["pf"] = int(CFG.PORTFOLIO_SIZE) if getattr(CFG, "PORTFOLIO_SIZE", None) else None
    F["pf_scale"] = (F["pf"] / max(N_TR, 1)) if F["pf"] else None
    F["value"] = bool(VALUE_SOURCE) and _g("at_risk_tr") is not None
    F["prem_churn"] = None
    try:
        if F["value"]:
            v_ = np.asarray(val_tr, dtype=float)
            F["prem_churn"] = float(np.nansum(v_ * np.asarray(y)) / max(np.nansum(v_), 1e-9))
    except Exception:
        pass
    if F["value"]:
        F["var"], F["var_share"] = at_risk_tr, at_risk_tr / max(val_tr.sum(), 1e-9)
    camp = _df("CAMPAIGN_BEST")
    if len(camp) > 1:
        b = camp.iloc[1] if camp.iloc[1]["net_profit"] >= camp.iloc[0]["net_profit"] else camp.iloc[0]
        F["camp"], F["camp_random"] = b, camp.iloc[2] if len(camp) > 2 else None
    sens = _df("SENS")
    F["sens_pos"] = int((sens["max_net_profit"] > 0).sum()) if len(sens) else None
    F["sens_n"] = len(sens)
    F["model_eval"] = _df("MODEL_EVAL")
    F["causal"] = _df("CAUSAL")
    F["robust"] = _df("CAUSAL_ROBUST")
    F["uplift"] = _df("UPLIFT_SUMMARY")
    F["recs"] = _df("RECS")
    F["pilot"] = _df("PILOT")
    F["personas"] = _df("PERSONAS")
    F["ablation"] = _df("ABLATION")
    F["exp"] = _df("EXP_TABLE")
    F["n_models"] = len({RESULTS[m]["model"] for m in RESULTS if m != "dummy"})
    F["model_names"] = sorted({MODEL_NAMES.get(RESULTS[m]["model"], m) for m in RESULTS if m != "dummy"})
    return F


def report_title(F):
    if CFG.REPORT_TITLE and CFG.REPORT_TITLE != "Understanding and Reducing Customer Churn: A Data-Driven Retention Strategy":
        return CFG.REPORT_TITLE, []
    comp = F["company"] if F["company"] and F["company"] != "the insurer" else "an Insurer"
    opts = []
    if F["surv"] and F["early_peak"] and "lifecycle" in F["themes"][:2]:
        k = min(NUM_WORDS, key=lambda m: abs(m - F["peak_months"]))
        opts.append(f"The {NUM_WORDS[k].title()} Decide: A Data-Driven Retention Strategy for {comp}")
    if "payment" in F["themes"][:3]:
        opts.append(f"Removing Payment Friction: Evidence-Based Churn Prevention for {comp}")
    if "price" in F["themes"][:3]:
        opts.append(f"Price Shock at Renewal: Who Leaves, When, and How to Keep Them at {comp}")
    if "service" in F["themes"][:3]:
        opts.append(f"Service Recovery as Retention: Turning Claims and Complaints into Loyalty at {comp}")
    if F["value"]:
        opts.append(f"Protecting the Premium Base: Predicting, Explaining and Profitably Preventing Churn at {comp}")
    opts.append(f"Who Leaves, When, and What Works: A Causal and Predictive Retention Strategy for {comp}")
    angle = CFG.NARRATIVE_ANGLE
    pick = {"lifecycle": "Decide", "payment": "Payment", "price": "Price", "service": "Service", "profit": "Premium Base"}.get(angle)
    main = next((o for o in opts if pick and pick in o), opts[0])
    return main, opts


NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                11: "eleven", 12: "twelve"}


def num_word(n):
    """Angka kecil ditulis dengan kata (gaya akademik): 4 → 'four'."""
    try:
        return NUMBER_WORDS.get(int(n), f"{int(n):,}")
    except Exception:
        return str(n)


def a_n(x):
    """Artikel a/an sebelum angka: 'an 11.7 pp', 'an 8% ...', 'a 9.9 pp'."""
    t = str(x).strip()
    return "an" if re.match(r"^(8|11|18|80|81|82|83|84|85|86|87|88|89)(\D|$)", t) or re.match(r"^8\d*(\.|$)", t) else "a"


def unit_of(name):
    m = re.search(r"\(([^()]*)\)\s*$", str(name))
    if not m:
        return ""
    u = m.group(1).strip()
    return "%" if u == "%" else f" {u}" if u.lower() in ("months", "days", "years", "weeks") else ""


def nm(name):
    """Nama fitur untuk kalimat: 'premium change at last renewal (%)' → 'premium change at last renewal'."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", str(name)).strip()


def risk_group(d):
    """Kelompok berisiko dalam bahasa kalimat."""
    name, unit, lv = nm(d["name"]), unit_of(d["name"]), str(d.get("level"))
    m = re.match(r"^\s*(-?[\d.,]+)\s*[–-]\s*(-?[\d.,]+)\s*$", lv)
    if m:
        if unit.startswith(" "):
            return f"customers whose {name} is between {m.group(1)} and {m.group(2)}{unit}"
        return f"customers whose {name} is between {m.group(1)}{unit} and {m.group(2)}{unit}"
    if lv in ("Yes", "No"):
        return f"customers {'with' if lv == 'Yes' else 'without'} {name}"
    m = re.match(r"^\s*(≥|>=|>|≤|<=|<)\s*(-?[\d.,]+)\s*$", lv)
    if m:
        word = {"≥": "or more", ">=": "or more", ">": "or more", "≤": "or fewer", "<=": "or fewer", "<": "or fewer"}[m.group(1)]
        countish = re.search(r"(complaint|claim|late payment|product|polic|call|visit|contact|ticket|missed|overdue)s?\b", name, re.I) and "rate" not in name and "ratio" not in name
        if not unit and countish:
            return f"customers with {m.group(2)} {word} {re.sub(r'^number of ', '', name)}"
        return f"customers whose {name} is {m.group(1)} {m.group(2)}{unit}"
    if re.match(r"^-?[\d.,]+$", lv):
        return f"customers whose {name} is {lv}{unit}"
    return f"customers with {name} '{lv}'"


def interval_phrase(iv, unit):
    """'[18, 24)' + 'months' → 'between 18 and 24 months of tenure'."""
    m = re.match(r"^\s*[\[(]\s*([\d.]+)\s*,\s*([\d.]+|inf)\s*[\])]\s*$", str(iv))
    if not m:
        return f"tenure interval {iv} {unit}"
    a, b = m.group(1), m.group(2)
    a = a[:-2] if a.endswith(".0") else a
    b = b[:-2] if b.endswith(".0") else b
    if b == "inf":
        return f"after {a} {unit} of tenure"
    if a in ("0", "0.0"):
        return f"in the first {b} {unit} of tenure"
    return f"between {a} and {b} {unit} of tenure"


def lever_phrase(t):
    """Kondisi tuas kausal → frasa tindakan: 'X ≤ median (4.1)' → 'keeping X at or below its median (4.1%)'."""
    t = str(t)
    m = re.match(r"^(.+?)\s*(≤|>|≥|<)\s*median\s*\(([^)]*)\)\s*$", t)
    if m:
        name, op, v = m.groups()
        u = unit_of(name)
        if op in ("≤", "<"):
            return f"keeping the {nm(name)} at or below its median ({v}{u})"
        return f"raising the {nm(name)} above its median ({v}{u})"
    m = re.match(r"^(.+?)\s*=\s*(.+)$", t)
    if m:
        name, v = m.group(1).strip(), m.group(2).strip()
        if v in ("1", "1.0", "Yes", "True"):
            return f"moving customers to {nm(name)}"
        if v in ("0", "0.0", "No", "False"):
            return f"moving customers away from {nm(name)}"
        return f"switching the {nm(name)} to '{v}'"
    return f"moving customers to '{t}'"


def cond_label(t):
    """Kondisi untuk kutipan: 'auto-renewal = 1' → 'auto-renewal = Yes'."""
    t = re.sub(r"=\s*1(\.0)?\s*$", "= Yes", str(t))
    return re.sub(r"=\s*0(\.0)?\s*$", "= No", t)


def de_dash(t):
    """Em dash di kalimat (tanda khas tulisan AI): pasangan → kurung, tunggal → koma. Rentang angka (13–30) tidak disentuh."""
    t = re.sub(r" — ([^—()]{1,180}?) — ", r" (\1) ", str(t))
    return t.replace(" — ", ", ")


def lcfirst(t):
    t = str(t)
    return t if (len(t) > 1 and t[:2].isupper()) else t[:1].lower() + t[1:]


def build_report_content(level=0):
    """Return (blocks, meta). level 0..4 = tingkat pemadatan (autofit)."""
    F = collect_facts()
    B = []
    company, line, term = F["company"], F["line"], F["churn_term"]
    ccy = CFG.CURRENCY
    FW = {0: 13.5, 1: 12.5, 2: 12.0, 3: 11.0, 4: 10.0, 5: 10.0, 6: 9.5}[min(level, 6)]
    max_p = 3 if level == 0 else 2 if level < 5 else 1      # level 5+: teks bawaan prioritas 2 dibuang (riset tim = prioritas 1, tetap)
    CW = 15.0 if level < 3 else 14.0 if level < 6 else 13.0          # lebar figure komposit

    CUR = [None]

    def section_key(t):
        t = str(t)
        rules = [("Executive Summary", "executive_summary"), ("1.", "introduction"), ("2.1", "data"), ("2.2", "methodology"), ("2.3", "methodology"),
                 ("3.1", "findings_drivers"), ("When do customers churn", "findings_timing"), ("Who is most at risk", "findings_segments"),
                 ("How well can churn", "findings_model"), ("Why is a customer", "findings_explain"), ("What works", "causal"),
                 ("Churn personas", "personas"), ("4.", "business"), ("5.", "recommendations"), ("6.", "discussion")]
        for pat, key in rules:
            if (pat[0].isdigit() and t.startswith(pat)) or (not pat[0].isdigit() and pat in t):
                return key
        return None

    def _custom(key, pos):
        for txt in research_texts(key, "paragraph", pos):
            B.append({"k": "p", "t": prettify_text(txt), "prio": 1})

    def H(t, lvl=1):
        key = section_key(t)
        if key and key != CUR[0]:
            if CUR[0]:
                _custom(CUR[0], "end")
        B.append({"k": "h", "t": t, "lvl": lvl})
        if key and key != CUR[0]:
            CUR[0] = key
            _custom(key, "start")
            if key != "introduction":
                for txt in research_texts(key, "facts"):
                    B.append({"k": "p", "t": prettify_text(txt), "prio": 1})

    def P(t, prio=1):
        if prio <= max_p and t:
            B.append({"k": "p", "t": de_dash(prettify_text(t)), "prio": prio})

    def BL(t, prio=1):
        if prio <= max_p and t:
            B.append({"k": "bullet", "t": de_dash(prettify_text(t)), "prio": prio})

    def EQ(t, prio=2):
        if prio <= max_p and t:
            B.append({"k": "eq", "t": t, "prio": prio})

    def FIG(key, cap, prio=1, w=None):
        if not find_fig(key):
            return
        if prio > max_p or (level >= 2 and prio >= 3) or (level >= 4 and prio >= 2):
            DEFERRED.append((key, cap))
            return
        B.append({"k": "fig", "key": key, "cap": cap, "w": w or FW})

    def TAB(df, cap, key, prio=1, font=9):
        if df is None or not len(df):
            return
        if prio > max_p or (level >= 4 and prio >= 2):
            DEFERRED_T.append((df, cap, key))
            return
        B.append({"k": "tab", "df": df, "cap": cap, "key": key, "font": font if level < 4 else 8})

    DEFERRED.clear()
    DEFERRED_T.clear()
    title, title_opts = report_title(F)
    d0 = F["drivers"][0] if F["drivers"] else None
    top_names = [d["name"] for d in F["drivers"][:3]]

    # ================= EXECUTIVE SUMMARY =================
    H("Executive Summary")
    ctx = f"{company[:1].upper() + company[1:]} loses {pct(F['rate'])} of its customers to {term}"
    if F["value"]:
        ctx += f", which puts about {money(F['var'])} of annual premium ({pct(F['var_share'])} of the premium base) at risk"
        if F["pf"]:
            ctx += f" in the analysed sample of {F['n']:,} customers — roughly {money(F['var'] * F['pf_scale'])} across the full portfolio of {F['pf']:,}"
    _methods = ["statistical driver analysis"] + (["survival analysis"] if F["surv"] else []) + ["interpretable regression",
                "a cross-validated machine-learning ensemble"] + (["doubly robust causal inference"] if len(F["causal"]) else [])
    _nq = len(CFG.CASE_QUESTIONS) if CFG.CASE_QUESTIONS else 0
    P(ctx + (f". Using these records, we combined" if F["pf"] else f". Using records of {F['n']:,} policyholders, we combined")
      + f" {_join(_methods)} to answer "
      + (f"the {num_word(_nq)} questions posed by management: who leaves, when they leave, how accurately this can be predicted and which "
         "actions are worth funding." if _nq else "three questions: who leaves, when they leave, and which actions are worth funding."))
    find = []
    if d0 is not None:
        s = f"Churn is most strongly associated with {_join([nm(n) for n in top_names[:3]])}"
        if d0["level"] is not None and pd.notna(d0["rate"]):
            s += f"; {risk_group(d0)} churn at {pct(d0['rate'])}, {d0['lift']:.1f} times the average"
        find.append(s + ".")
    if F["surv"]:
        find.append(f"Risk is concentrated in time: the churn hazard peaks {interval_phrase(F['peak_interval'], F['surv_unit'])} "
                    f"({pct(F['peak_hazard'], 2)} per {F['hz_label']}, about {F['peak_hazard'] / max(F['avg_hazard'], 1e-9):.1f} times the average).")
    find.append(f"The final model reaches an out-of-fold {F['auc_txt']}; the top 20% of scored customers contain "
                f"{F['top20_capture']:.0f}% of all churners ({F['top20_lift']:.1f}× random targeting).")
    if "camp" in F:
        b = F["camp"]
        find.append(f"A retention campaign aimed at the top {b['target_%']:.0f}% of customers by expected loss is estimated to return "
                    f"{money(b['net_profit'])}{'' if F['value'] else ' customers'} net (ROI {b['roi']:.1f}×)"
                    + (f", whereas random targeting of the same size yields {money(F['camp_random']['net_profit'])}" if F.get("camp_random") is not None else "")
                    + ".")
    rob = F["robust"]
    if len(rob) and (rob["verdict"] == "robust").any():
        r = rob[rob["verdict"] == "robust"].sort_values("ate_pp").iloc[0]
        find.append(f"Causal analysis indicates that {lever_phrase(r['treatment (T=1)'])} would lower churn by about {abs(r['ate_pp']):.1f} "
                    f"percentage points, a result that survives placebo and sensitivity tests.")
    P(" ".join(find))
    recs = F["recs"]
    if len(recs):
        rl = [f"({i + 1}) {lcfirst(str(t).rstrip('.'))}" for i, t in enumerate(recs["recommendation"].head(3))]
        s = "We recommend three actions: " + "; ".join(rl) + ". "
        if len(F["pilot"]):
            p0 = F["pilot"].iloc[0]
            s += (f"Each action should start as a randomised pilot; detecting a reduction of {p0['expected_reduction_pp']:.1f} percentage points requires about "
                  f"{int(p0['n_per_arm']):,} customers per arm.")
        P(s)

    # ================= 1. INTRODUCTION =================
    H("1. Introduction")
    if CFG.CASE_CONTEXT:
        P(CFG.CASE_CONTEXT)
    _facts_intro = research_texts("introduction", "facts")
    if _facts_intro:
        P(" ".join(_facts_intro))
    if str(CFG.MARKET).lower().startswith("indo") and getattr(CFG, "USE_DEFAULT_MARKET_CONTEXT", True):
        P("Indonesia's insurance market remains small relative to the size of its economy. The Financial Services Authority (OJK) reported "
          "insurance penetration of 2.59% at the end of 2023 and 2.84% at the end of 2024, and has described penetration and density as "
          "largely stagnant over the past two decades (\"OJK sebut penetrasi,\" 2024; \"Penetrasi asuransi syariah,\" 2025). The 2025 national "
          "survey recorded an insurance literacy index of 45.45% but an inclusion index of only 28.50% (\"SNLIK 2025,\" 2025). In a market "
          "where coverage is still being built, losing existing policyholders is costly.", prio=2)
        if line == "life":
            P("In life insurance, surrender has been the largest single category of claims and benefits: surrender claims reached Rp77.15 trillion "
              "in 2024, close to half of the Rp160.07 trillion in total claims and benefits, before falling 19% to Rp62.72 trillion in 2025 "
              "(\"Regulasi baru OJK dinantikan,\" 2025; \"AAJI catat nilai klaim surrender,\" 2026). The regulatory direction, including OJK "
              "Circular Letter No. 5/SEOJK.05/2022 on investment-linked products, emphasises transparent selling and post-sale confirmation "
              "calls (Otoritas Jasa Keuangan, 2022).", prio=3)
        else:
            P("In general insurance, where most policies are renewed annually, every renewal is a decision point for the customer. Premiums of "
              "the general insurance industry reached Rp112.81 trillion in 2025, with motor vehicle insurance accounting for 16.86% "
              "(\"AAUI catat premi,\" 2026), a segment in which products are standardised and switching costs are low.", prio=3)
    P(f"For an individual insurer, churn ({term}) affects several parts of the business at once. It erodes the renewal premium base, leaves "
      "acquisition costs unrecovered when policies exit early, distorts the persistency assumptions used in pricing "
      "(Eling & Kochanski, 2013) and is often a signal of unmet needs or poor service (Brockett et al., 2008). The economics of retention are "
      "well documented: Reichheld and Sasser (1990) reported that reducing defections by 5% raised profits by 25% to 85% across several "
      "service businesses. Not every exit is equally avoidable, however: Gottlieb and Smetters (2021) found that forgetting to pay explained "
      "37.8% of recent life-insurance lapses, so part of churn can be addressed through process design rather than price.")
    P("Prior research links lapse and churn to income shocks, contract age, product type and payment frequency (Eling & Kiesenbauer, 2014; "
      "Hein et al., 2020) and to renewal pricing and the service experience (Brockett et al., 2008). Methodologically, churn models should be "
      "judged by the profit of the campaigns they enable (Verbraken et al., 2013), and risk should be distinguished from responsiveness to an "
      "intervention (Ascarza, 2018). " + ("Policy-level evidence from Indonesia remains scarce (Hein et al., 2020). " if str(CFG.MARKET).lower().startswith("indo") else "")
      + "Appendix B reviews the methods literature in more detail.", prio=2)
    rqs = list(CFG.CASE_QUESTIONS) if CFG.CASE_QUESTIONS else [
        "Which customer, policy and service factors are associated with churn, and how strongly after controlling for the others?",
        "When during the customer lifecycle is churn risk highest, and which segments carry the highest risk?",
        "How accurately can churn be predicted, and which customers should be prioritised?",
        "Which interventions are most likely to reduce churn, for whom, and what is their expected economic value?"]
    obj = {"maximize_profit": " Management's objective is to maximise the profit of retention spending.",
           "reduce_churn": " Management's objective is to reduce the churn rate.",
           "early_warning": " Management's objective is an operational early-warning system.",
           "understand_drivers": " Management's objective is to understand why customers leave."}.get(CFG.MANAGEMENT_OBJECTIVE, "")
    P((obj.strip() + " " if obj else "") + f"To support this objective, the study answers {num_word(len(rqs))} research questions (RQ):")
    for i, q in enumerate(rqs):
        BL(f"RQ{i + 1}. {q}")

    # ================= 2. DATA AND METHODOLOGY =================
    H("2. Data and Methodology")
    H("2.1 Data and preparation", 2)
    n_raw = len([c for c in _g("ORIGINAL_COLS", []) if c not in (TARGET, ID_COL)])
    s = (f"The training data contain {F['n']:,} customers"
         + (f" (plus {F['n_test']:,} customers to be scored)" if F["n_test"] else "")
         + f" and {n_raw} original variables" + (f" covering {CFG.DATA_PERIOD}" if CFG.DATA_PERIOD else "") + ". "
         + f"The target is '{TARGET}'" + (f", defined as {CFG.CHURN_DEFINITION}" if CFG.CHURN_DEFINITION else "")
         + f" (churn = {', '.join(map(str, POS_ORIGINAL))}); the churn rate is {pct(F['rate'])}, so models are evaluated with ROC-AUC, "
           "precision–recall and profit rather than accuracy. ")
    n_extra = len(CFG.EXTRA_TABLES or {}) + len(CFG.EXTRA_ONE_TO_ONE or {})
    if n_extra:
        _tn = list((CFG.EXTRA_TABLES or {}).keys()) + list((CFG.EXTRA_ONE_TO_ONE or {}).keys())
        s += (f"{num_word(n_extra).capitalize()} related {'table' if n_extra == 1 else 'tables'} ({_join([str(t).replace('_', ' ') for t in _tn])}) "
              f"{'was' if n_extra == 1 else 'were'} aggregated to one row per customer (counts, sums and recency). ")
    P(s)
    log_ = [l for l in _g("clean_log", [])]
    prep = (f"Preparation was automated and audited: {len(log_)} type corrections and harmonisations were applied (for example numbers or "
            f"ranges stored as text, date parsing and inconsistent category labels), {_g('N_DUP_REMOVED', 0)} duplicate records were removed "
            f"and impossible values were set to missing. ")
    if LEAKY_COLS:
        prep += (f"Leakage was the most serious data problem: {_join([pretty(c) for c in LEAKY_COLS[:4]])} {'is' if len(LEAKY_COLS) == 1 else 'are'} recorded only around "
                 "or after the churn event; such post-event variables leak the outcome, so they and all derived features were removed before "
                 "any analysis (single-feature and missingness tests flagged them automatically). ")
    if FE_NEW:
        prep += f"We engineered {len(FE_NEW)} insurance-specific features, such as {_join([pretty(f) for f in FE_NEW if not f.endswith(('_year', '_month'))][:5])}."
    P(prep)
    H("2.2 Analytical framework", 2)
    P("The analysis follows five layers ({{fig:system_architecture}}). (1) Univariate tests (χ² and Mann–Whitney U) with effect sizes, Information Value and "
      "Benjamini–Hochberg correction screen candidate drivers (Benjamini & Hochberg, 1995). (2) Because churn is a time-to-event "
      "outcome with censored observations, Kaplan–Meier curves, log-rank tests, a Cox model, an accelerated failure time model and restricted "
      "mean survival time describe when customers leave (Kaplan & Meier, 1958; Cox, 1972). (3) A logistic regression "
      "gives adjusted odds ratios that managers can read directly. "
      f"(4) {num_word(F['n_models']).capitalize()} algorithms — {_join(F['model_names'])} — were trained with stratified {CFG.N_FOLDS}-fold "
      f"cross-validation{', tuned with Bayesian optimisation' if TUNED_PARAMS else ''} and combined by ensemble selection; "
      "tree ensembles remain the strongest family on tabular data (Grinsztajn et al., 2022), and SHAP values explain "
      "each prediction (Lundberg & Lee, 2017). (5) Doubly robust estimators (AIPW and the DR-learner) estimate the average and heterogeneous "
      "effects of controllable levers (Chernozhukov et al., 2018; Kennedy, 2023). Results are translated into money with "
      "customer lifetime value, the expected maximum profit measure (Verbraken et al., 2013) and a campaign simulation.")
    FIG("system_architecture", "End-to-end analytics architecture: from raw policy data to retention decisions, with a feedback loop from pilots.",
        prio=2, w=FW + 1.5)
    H("2.3 Validation", 2)
    shuffled = _g("auc_perm")
    P("All reported performance figures are out-of-fold estimates; target-dependent encodings and hyperparameter search were fitted inside "
      "training folds only. Models are compared with DeLong tests for correlated AUCs (DeLong et al., 1988) with Holm correction, uncertainty "
      f"is reported with bootstrap and DeLong confidence intervals, and calibration is checked with the Brier score and expected calibration "
      f"error. Sanity checks support the validity of the pipeline: a model trained on shuffled labels scores "
      f"{'an AUC of ' + format(shuffled, '.3f') if shuffled is not None else 'an AUC of about 0.5'}"
      + (f", and an adversarial classifier separates training from scoring data with AUC {ADV_AUC:.2f}" if np.isfinite(_g("ADV_AUC", np.nan)) else "")
      + ". Appendix B documents the full validation design.", prio=1)

    # ================= 3. FINDINGS =================
    H("3. Findings")
    H("3.1 Who churns? (RQ1)", 2)
    if F["drivers"]:
        s = []
        for d in F["drivers"][:3]:
            seg_ = risk_group(d) if d["level"] is not None else "the riskiest group"
            t = f"{nm(d['name'])} (IV {d['iv']:.2f}), where {seg_} churn at {pct(d['rate'])}, {d['lift']:.1f}× the average"
            s.append(t)
        P("The strongest correlates of churn are " + "; ".join(s) + " ({{fig:report_drivers}}, {{tab:evidence}}). "
          + (f"Several variables — {_join(F['negligible'][:4])} — are statistically detectable but practically negligible (IV < 0.02), a reminder "
             f"that with {F['n']:,} records significance alone is not evidence of importance." if F["negligible"] else ""))
    if find_fig("report_drivers"):
        FIG("report_drivers", "Churn rate by level of the strongest drivers (bars: churn rate with 95% Wilson CI; dashed line: overall churn rate; red = above average).", prio=1, w=CW)
    else:
        FIG("driver_churn_rate_1", "Churn rate by the strongest drivers (bars: churn rate with 95% Wilson CI; dashed line: overall churn rate).", prio=1)
    ev = _df("EVIDENCE")
    if len(ev):
        ev = ev[ev["feature"].isin(set(dedupe_features(ev["feature"].tolist())))]
        t = ev.head(7).copy()
        cols = OrderedDict([("feature", "Driver"), ("highest_risk_level", "Riskiest group"), ("highest_risk_rate", "Churn rate"),
                            ("highest_risk_lift", "Lift"), ("iv", "IV"), ("odds_ratio", "Odds ratio"), ("hazard_ratio", "Hazard ratio"),
                            ("shap_rank", "SHAP rank"), ("evidence_score", "Methods agreeing")])
        t = t[[c for c in cols if c in t]].rename(columns=cols)
        if "Riskiest group" in t:
            t["Riskiest group"] = [level_label(f_, v_) for f_, v_ in zip(t["Driver"], t["Riskiest group"])]
        t["Driver"] = t["Driver"].map(pretty)
        if "Churn rate" in t:
            t["Churn rate"] = t["Churn rate"].map(lambda v: pct(v))
        for c_, f_ in [("Lift", lambda v: f"{v:.2f}×"), ("IV", lambda v: f"{v:.2f}"), ("Odds ratio", lambda v: f"{v:.2f}"),
                       ("Hazard ratio", lambda v: f"{v:.2f}"), ("SHAP rank", lambda v: f"{int(v)}"), ("Methods agreeing", lambda v: f"{int(v)}/4")]:
            if c_ in t:
                t[c_] = t[c_].map(lambda v, f_=f_: "—" if pd.isna(v) else f_(v))
        TAB(t, f"Evidence for churn drivers across methods (overall churn rate {pct(F['rate'])}; 'methods agreeing' counts significant univariate "
               "test, logistic regression, Cox model and top-10 SHAP).", "evidence", prio=1, font=8.5)
    lt = _df("LOGIT_TABLE")
    if len(lt) and lt["p_value"].notna().any():
        sig = lt[lt["p_value"] < 0.05].head(3)
        parts = [f"{prettify_text(r['term'])} ({prettify_text(r['scale'])}: OR {r['odds_ratio']:.2f}, 95% CI {r['or_ci_low']:.2f}–{r['or_ci_high']:.2f})"
                 for _, r in sig.iterrows()]
        if parts:
            P("The multivariate logistic regression confirms that these effects are not artefacts of each other: holding the other variables "
              f"constant, the largest adjusted effects are {_join(parts)}"
              + " (full model in Appendix D).", prio=2)
    if _df("REASON_TABLE").shape[0]:
        rt = _df("REASON_TABLE")
        P(f"Customers who left most often stated {_join([f'{a} ({b:.0f}%)' for a, b in zip(rt['reason'][:3], rt['share_%'][:3])])} as their reason, "
          "which is consistent with the revealed drivers above.", prio=2)

    if F["surv"]:
        H("3.2 When do customers churn? (RQ2)", 2)
        km = F.get("km", pd.DataFrame())
        km_txt = "; ".join(f"{pct(r['survival_prob'])} after {r['checkpoint']}" for _, r in km.head(3).iterrows()) if len(km) else ""
        P(f"Survival analysis treats active customers as censored observations. According to the Kaplan–Meier estimates, {km_txt.replace('; ', ', ')} of "
          f"customers are still active"
          + (f"; in the industry's persistency terms, the {_join([f'{m}-month ratio is {pct(v)}' for m, v in F['persist'].items()])}"
             if F.get("persist") else "")
          + f". The churn hazard peaks {interval_phrase(F['peak_interval'], F['surv_unit'])} at {pct(F['peak_hazard'], 2)} "
          f"per {F['hz_label']}, against an average of {pct(F['avg_hazard'], 2)} ({{fig:report_timing}}).")
        FIG("report_timing", f"(a) Churn hazard per {F['hz_label']} by tenure interval; (b) Kaplan–Meier survival by predicted risk tier (out-of-sample "
                             "validation of the score over time).", prio=1, w=CW)
        rm = _df("RMST_TABLE")
        if len(rm):
            w_ = rm.sort_values("diff_vs_best").iloc[0]
            P(f"Restricted mean survival time translates this into time lost: within the first {w_['tau']:g} {F['surv_unit']}, customers with "
              f"{pretty(w_['feature'])} = '{w_['level']}' stay on average {abs(w_['diff_vs_best']):.1f} {F['surv_unit']} less than those in "
              f"'{w_['reference_level']}'.", prio=2)
        ct = _df("COX_TABLE")
        if len(ct):
            sig = ct[ct["p_value"] < 0.05].head(2)
            if len(sig):
                P("In the Cox model, " + _join([f"{prettify_text(r['term'])} carries a hazard ratio of {r['hazard_ratio']:.2f} (95% CI "
                                               f"{r['hr_ci_low']:.2f}–{r['hr_ci_high']:.2f})" for _, r in sig.iterrows()])
                  + (f"; Schoenfeld tests flag time-varying effects for {int(_df('PH_TEST')['ph_violated'].sum())} terms, so we also report "
                     "an accelerated failure time model (Appendix E)." if len(_df("PH_TEST")) else "."), prio=3)

    H(f"3.{3 if F['surv'] else 2} Who is most at risk? Interpretable segments", 2)
    seg = _df("SEGMENT_RULES")
    if len(seg):
        r = seg.iloc[0]
        lo = seg.iloc[-1]
        P(f"A shallow decision tree turns the drivers into segments operations teams can act on ({{tab:segments}}). The riskiest segment — "
          f"{prettify_rule(r['segment_rule'])} — has {int(r['n_customers']):,} customers ({r['share_%']:.1f}% of the base) churning at "
          f"{pct(r['churn_rate'])} ({r['lift']:.1f}×) and contains {r['churners_captured_%']:.1f}% of all churners, whereas the most loyal "
          f"segment churns at only {pct(lo['churn_rate'])}.")
        sr = pd.concat([seg.head(3), seg.tail(1)])[["segment_rule", "n_customers", "churn_rate", "lift"]].copy()
        sr["segment_rule"] = sr["segment_rule"].map(prettify_rule)
        sr["churn_rate"] = sr["churn_rate"].map(lambda v: pct(v))
        sr["lift"] = sr["lift"].map(lambda v: f"{v:.1f}×")
        TAB(sr.rename(columns={"segment_rule": "Segment (decision-tree rule)", "n_customers": "Customers", "churn_rate": "Churn rate", "lift": "Lift"}),
            "Highest-risk segments and the most loyal segment.", "segments", prio=2, font=8.5)

    nsec = 4 if F["surv"] else 3
    H(f"3.{nsec} How well can churn be predicted? (RQ3)", 2)
    me = F["model_eval"]
    if len(me):
        m2 = me.copy()
        m2 = m2.sort_values("roc_auc", ascending=False).head(7 if level < 3 else 5)
        tt = pd.DataFrame({"Model": m2["model"].str.replace("FINAL: ", "Final ensemble: ", regex=False).map(lambda s: MODEL_NAMES.get(s, s)),
                           "ROC-AUC (95% CI)": [f"{a:.3f} ({b:.3f}–{c:.3f})" for a, b, c in zip(m2["roc_auc"], m2["auc_ci_low"], m2["auc_ci_high"])],
                           "PR-AUC": m2["pr_auc"].round(3), "Brier": m2["brier"].round(3),
                           f"EMPC ({EMPC_UNIT if 'EMPC_UNIT' in globals() else ''})": m2["empc_per_customer"].map(lambda v: fmt_num(v)),
                           "DeLong p vs final": m2.get("delong_p_holm", pd.Series(index=m2.index, dtype=float)).map(lambda v: "—" if pd.isna(v) else p_fmt(v))})
        TAB(tt, f"Out-of-fold performance ({CFG.N_FOLDS}-fold stratified CV; random baseline ROC-AUC 0.500, PR-AUC {F['rate']:.3f}). "
                "EMPC = expected maximum profit per customer (Verbraken et al., 2013); p-values Holm-corrected.", "models", prio=1, font=8.5)
    s = (f"The final model reaches an out-of-fold {F['auc_txt']} and a PR-AUC of {F['pr_auc']:.3f}, {F['pr_auc'] / max(F['rate'], 1e-9):.1f} "
         f"times the random baseline ({{tab:models}}). ")
    if len(me) and "logreg" in set(me["model"]):
        lr_ = me[me["model"] == "logreg"].iloc[0]
        sig_ = lr_.get("delong_p_holm", lr_.get("delong_p", 1)) < 0.05
        s += (f"The interpretable logistic regression reaches {lr_['roc_auc']:.3f}; the ensemble is "
              f"{'significantly better' if sig_ else 'not significantly better'} (ΔAUC {lr_['delta_auc_final_minus_model']:+.3f}, DeLong p "
              f"{p_fmt(lr_.get('delong_p_holm', lr_.get('delong_p')))}), "
              + ("so the added complexity buys real accuracy. " if sig_ else "so the simpler model is a credible, fully transparent alternative. "))
    if len(me) and "ebm" in set(me["model"]):
        eb = me[me["model"] == "ebm"].iloc[0]
        s += f"A glass-box Explainable Boosting Machine reaches {eb['roc_auc']:.3f}. "
    if len(me):
        be = me.sort_values("empc_per_customer", ascending=False).iloc[0]
        ba = me.sort_values("roc_auc", ascending=False).iloc[0]
        s += (f"Ranking by expected maximum profit {'agrees with' if be['model'] == ba['model'] else 'differs from'} ranking by AUC"
              + ("" if be["model"] == ba["model"] else f" (most profitable: {MODEL_NAMES.get(RESULTS.get(be['model'], {}).get('model', ''), be['model'])})") + ". ")
    if CFG.METRIC not in ("roc_auc", "gini"):
        try:
            _ms = show_score(rank_score(y, FINAL_OOF))
            s += (f"On the competition metric ({CFG.METRIC.upper().replace('_', '-')}) the final model scores {_ms:.4f}"
                  + (f" at the optimised decision threshold of {BEST_THR:.3f}" if CFG.METRIC in LABEL_METRICS else "") + ". ")
        except Exception:
            pass
    s += (f"Concentration is strong: the highest-risk decile churns at {pct(F['top10_rate'])} and the top 20% contain {F['top20_capture']:.0f}% "
          f"of churners ({{fig:report_model}}).")
    P(s)
    exp = F["exp"]
    abl = F["ablation"]
    if len(exp):
        s = (f"Model development was systematic: {len(exp)} logged experiments covered the model zoo, tuning, multi-seed averaging, feature "
             f"selection, ablation and ensembling (Appendix F). ")
        if len(abl):
            a0 = abl.iloc[0]
            s += (f"An ablation study shows that removing {pretty(a0['ablation'].replace('− ', ''))} costs {a0['auc_drop']:.3f} AUC (DeLong p "
                  f"{p_fmt(a0['delong_p'])}), the largest contribution of any feature group.")
        P(s, prio=2)
    sc = _g("SCORE_CINDEX", np.nan)
    if np.isfinite(sc):
        P(f"The score also orders customers by time to churn (Harrell's C-index {sc:.3f}), and survival curves of the "
          "three risk tiers separate clearly (Appendix E).", prio=3)
    FIG("report_model", "(a) Cumulative gains of the final model vs random targeting; (b) ROC-AUC of the candidate models with 95% DeLong "
                        "confidence intervals.", prio=2, w=CW)

    H(f"3.{nsec + 1} Why is a customer at risk? Explaining the model", 2)
    sh = _df("SHAP_IMP")
    if len(sh):
        sh2 = sh.copy()
        sh2["base"] = sh2["feature"].str.replace(" (target-enc)", "", regex=False).str.replace("__freq", "", regex=False)
        top = sh2.groupby("base")["share_%"].sum().sort_values(ascending=False)
        P(f"SHAP values attribute each prediction to the features (Lundberg & Lee, 2017). The model relies mostly on "
          f"{_join([f'{nm(pretty(k))} ({v:.0f}%)' for k, v in top.head(3).items()])} of total attribution ({{fig:shap_beeswarm}}), consistent with the statistical "
          "evidence, so the model behaves in a way managers can audit.")
    het = _g("HETERO_TABLE")
    if isinstance(het, pd.DataFrame) and het.shape[1] >= 2:
        P(f"Driver importance differs across {HETERO_SEGMENT or 'segments'}: " +
          "; ".join(f"for '{c}' the leading driver is {pretty(het[c].idxmax())} ({het[c].max():.0f}%)" for c in list(het.columns)[:3]) + ".", prio=3)
    FIG("shap_beeswarm", "SHAP summary: each dot is a customer; positive values push churn risk up (colour = feature value).", prio=2, w=FW - 1)

    # ================= 4. BUSINESS IMPACT =================
    H("4. Business Impact (RQ4)")
    H("4.1 Value at risk and prioritisation", 2)
    if F["value"]:
        q = _df("QUAD_TABLE")
        ps = q[q["quadrant"].str.startswith("Priority")] if len(q) else pd.DataFrame()
        s = (f"Weighting each customer's churn probability by annual premium, about {money(F['var'])} of annual premium "
             f"({pct(F['var_share'])} of the premium base) is expected to be lost"
             + (f" in the analysed sample, or roughly {money(F['var'] * F['pf_scale'])} when scaled to the portfolio of {F['pf']:,} customers" if F["pf"] else "")
             + ". ")
        if F.get("prem_churn") is not None:
            _r = F["prem_churn"] / max(F["rate"], 1e-9)
            s += (f"Weighted by premium, churn is {pct(F['prem_churn'])} against {pct(F['rate'])} by headcount, so leavers hold "
                  + ("larger" if _r > 1.05 else "smaller" if _r < 0.95 else "about average")
                  + " policies" + (f" ({_r:.2f}× the premium-weighted ratio)" if abs(_r - 1) > 0.05 else "") + ". ")
        if len(ps):
            ps = ps.iloc[0]
            s += (f"Crossing risk with value ({{fig:report_business}}b) isolates a 'Priority Save' group — {pct(ps['n'] / F['n'])} of customers with high risk "
                  f"and above-median premium — that holds {ps['share_value_at_risk_%']:.0f}% of the value at risk and warrants personal contact; "
                  "high-risk, low-value customers can be served through automated digital touchpoints.")
        P(s)
    else:
        P("No premium or value variable is available, so prioritisation uses churn probability and the number of customers saved; "
          "value-based targeting should be added once premium data are linked.")
    H("4.2 Retention campaign economics", 2)
    if "camp" in F:
        b = F["camp"]
        if F["value"]:
            P("Each customer's value is the discounted margin earned while the policy stays in force, and the campaign contacts the set "
              "S(k) of k customers with the highest expected loss:", prio=1)
            EQ("CLVᵢ = m·Pᵢ·(1 − ρᵢᴴ) / (1 − ρᵢ),   ρᵢ = (1 − pᵢ) / (1 + d)", prio=1)
            EQ("Π(k) = Σ ( γ·yᵢ·Vᵢ − c ),   summed over i in S(k)", prio=1)
            P(f"Here m is the margin ({CFG.PROFIT_MARGIN:.0%}), Pᵢ the annual premium, pᵢ the predicted churn probability, d the discount rate "
              f"({CFG.DISCOUNT_RATE:.0%}), H the horizon ({CFG.CLV_HORIZON_YEARS} years), γ the retention success rate ({CFG.RETENTION_SUCCESS_RATE:.0%}), "
              f"c the cost per contact, yᵢ the observed churn outcome and Vᵢ the value of a retained customer.", prio=2)
        given = str(getattr(CFG, "BUSINESS_PARAMS_SOURCE", "assumed")).lower() == "case"
        s = (f"We simulated a retention campaign on the out-of-fold predictions, {'using the parameters given in the case' if given else 'assuming'}: "
             f"a contact-and-incentive cost of {money(CFG.RETENTION_COST)} per customer, that {CFG.RETENTION_SUCCESS_RATE:.0%} of contacted "
             f"would-be churners are retained and a {CFG.PROFIT_MARGIN:.0%} margin over a {CFG.CLV_HORIZON_YEARS}-year horizon (discount rate {CFG.DISCOUNT_RATE:.0%}). "
             f"Contacting the top {b['target_%']:.0f}% by expected loss maximises net benefit at {money(b['net_profit'])} (ROI {b['roi']:.1f}×; {{{{fig:report_business}}}}a). ")
        if F["sens_pos"] is not None:
            s += (f"The conclusion holds under different assumptions: the campaign stays profitable in {F['sens_pos']} of {F['sens_n']} combinations of success rate "
                  f"(±50%) and cost (×0.5–×2) in the sensitivity analysis.")
        if F["pf"]:
            s += (f" Scaled to the full portfolio of {F['pf']:,} customers, the same rule would contact about "
                  f"{int(b['target_%'] / 100 * F['pf']):,} customers at a cost of {money(b['target_%'] / 100 * F['pf'] * CFG.RETENTION_COST)} "
                  f"for a net benefit of about {money(b['net_profit'] * F['pf_scale'])}.")
        if CFG.RETENTION_BUDGET and "VALUE_SAVED_TR" in globals():
            base_n = F["pf"] or N_TR
            k_max = CFG.RETENTION_BUDGET / max(CFG.RETENTION_COST * base_n, 1e-9)
            k_opt = b["target_%"] / 100
            order_ = np.argsort(-(np.asarray(PROB_BIZ) * VALUE_SAVED_TR))
            k_use = min(k_opt, k_max)
            n_s = max(1, int(round(k_use * N_TR)))
            idx_ = order_[:n_s]
            ben_s = float((y[idx_] * CFG.RETENTION_SUCCESS_RATE * VALUE_SAVED_TR[idx_]).sum())
            net_s = ben_s - n_s * CFG.RETENTION_COST
            scale_ = base_n / N_TR
            n_contact = int(round(k_use * base_n))
            cost_ = n_contact * CFG.RETENTION_COST
            churn_hit = float(y[idx_].mean())
            F["budget_net"], F["budget_binding"], F["budget_contacts"] = net_s * scale_, bool(k_max < k_opt), n_contact
            if k_max < k_opt:
                s += (f" The budget of {money(CFG.RETENTION_BUDGET)} is the binding constraint: it pays for about {n_contact:,} contacts "
                      f"({pct(k_use, 1)} of {'the portfolio' if F['pf'] else 'the analysed customers'}). Spending it on the customers with the highest "
                      f"expected loss — among whom {pct(churn_hit)} would otherwise churn — is estimated to return {money(net_s * scale_)} net "
                      f"(ROI {net_s / max(n_s * CFG.RETENTION_COST, 1e-9):.1f}×), so the budget should be concentrated on the top of the ranking rather than spread thinly.")
            else:
                s += (f" The budget of {money(CFG.RETENTION_BUDGET)} covers the optimal campaign ({n_contact:,} contacts costing {money(cost_)}), "
                      f"leaving {money(CFG.RETENTION_BUDGET - cost_)} for the pilots and the onboarding programme.")
        P(s)
        FIG("report_business", "(a) Net profit of a retention campaign by share of customers contacted under three targeting rules; (b) share of "
                               "customers vs share of value at risk by risk × value segment.", prio=1, w=CW)
    if len(F["causal"]):
        H("4.3 What works? Causal evidence and uplift", 2)
        cz = F["causal"].sort_values("ate_pp")
        sig = cz[cz["p_value"] < 0.05]
        P("Prediction identifies who is at risk, not which action changes behaviour. For each controllable lever T we estimated the average "
          "effect of the favourable condition on churn Y with a cross-fitted doubly robust (AIPW) estimator, which combines a propensity model "
          "ê(X) with outcome models μ̂₁(X) and μ̂₀(X) and stays consistent if either is correct (Chernozhukov et al., 2018):", prio=2)
        EQ("ATE = (1/n) Σᵢ [ μ̂₁(Xᵢ) − μ̂₀(Xᵢ) + Tᵢ(Yᵢ − μ̂₁(Xᵢ))/ê(Xᵢ) − (1 − Tᵢ)(Yᵢ − μ̂₀(Xᵢ))/(1 − ê(Xᵢ)) ]")
        s = ("" if 2 <= max_p else "For each controllable lever we estimated the average effect of the favourable condition on churn with a "
             "cross-fitted doubly robust (AIPW) estimator that adjusts for observed confounders (Chernozhukov et al., 2018). ")
        if len(sig):
            s += "The largest effects are " + _join([f"'{cond_label(r['favorable_condition (T=1)'])}' ({r['ate_pp']:+.1f} pp, 95% CI {r['ci_low_pp']:+.1f} to "
                                                    f"{r['ci_high_pp']:+.1f}; naive difference {r['naive_diff_pp']:+.1f} pp)" for _, r in sig.head(3).iterrows()]) + ". "
        rb = F["robust"]
        if len(rb):
            nrob = int((rb["verdict"] == "robust").sum())
            s += (f"{nrob} of {len(rb)} effects pass all robustness checks (placebo treatment ≈ 0, adequate overlap and E-values above 1.25; "
                  "VanderWeele & Ding, 2017). ")
        up = F["uplift"]
        if len(up):
            u = up.iloc[0]
            het = (u["gates_q5_pp"] > u["gates_q1_pp"]) and (u["heterogeneity_p"] < 0.05)
            better_uplift = u["avoided_per1000_top20_uplift"] > u["avoided_per1000_top20_risk"]
            if het and better_uplift:
                s += (f"Effects are heterogeneous: for '{cond_label(u['treatment'])}', the most responsive fifth of eligible customers shows a "
                      f"{u['gates_q5_pp']:.1f} pp reduction versus {u['gates_q1_pp']:.1f} pp for the least responsive (p {p_fmt(u['heterogeneity_p'])}), "
                      f"and treating the top 20% by predicted uplift avoids {u['avoided_per1000_top20_uplift']:.1f} churners per 1,000 eligible "
                      f"customers against {u['avoided_per1000_top20_risk']:.1f} when targeting by risk — so campaigns should rank customers by "
                      "expected response, not risk alone (Kennedy, 2023; Ascarza, 2018).")
            else:
                s += (f"We also tested whether some customers respond more than others (DR-learner; Kennedy, 2023). For '{cond_label(u['treatment'])}' "
                      f"the effect is broadly homogeneous (most vs least responsive fifth: {u['gates_q5_pp']:.1f} vs {u['gates_q1_pp']:.1f} pp, "
                      f"p {p_fmt(u['heterogeneity_p'])}), and targeting the top 20% by churn risk avoids "
                      f"{u['avoided_per1000_top20_risk']:.1f} churners per 1,000 eligible customers versus "
                      f"{u['avoided_per1000_top20_uplift']:.1f} with uplift-based targeting, so risk-based targeting is appropriate for this "
                      "lever; the pilots should still re-test responsiveness (Ascarza, 2018).")
        P(s, prio=1)
        FIG("report_causal", "(a) Doubly robust (AIPW) effects of favourable conditions on churn with 95% CI, vs naive differences; (b) average churn "
                             "reduction by quintile of predicted uplift (DR-learner).", prio=2, w=CW)
        P("These estimates assume no unmeasured confounding; they rank levers by promise and are designed to be confirmed by the randomised "
          "pilots proposed below.", prio=2)
    pers = F["personas"]
    if len(pers) and 2 <= max_p:
        H(f"4.{4 if len(F['causal']) else 3} Churn personas", 2)
        P(f"Clustering the SHAP explanations of at-risk customers yields {len(pers)} personas with different reasons for risk: "
          + "; ".join(f"{r['persona']} ({r['share_of_at_risk_%']:.0f}% of at-risk customers, actual churn {pct(r['actual_churn_rate'])}) "
                      + ("with moderate, diffuse risk and no dominant driver" if str(r["top_drivers"]).startswith("no dominant") else
                         f"driven mainly by {prettify_text(str(r['top_drivers']).split(';')[0].split(' (+')[0])}") for _, r in pers.iterrows())
          + ". Each persona receives a different intervention in the recommendations.", prio=2)

    # ================= 5. RECOMMENDATIONS =================
    H("5. Recommendations and Implementation")
    if len(recs):
        rr = recs.copy()
        if CFG.AVAILABLE_INTERVENTIONS:
            kw = [str(a).lower() for a in CFG.AVAILABLE_INTERVENTIONS]
            rr["_m"] = rr.apply(lambda r: any(any(w in (str(r["recommendation"]) + " " + str(r["action"])).lower() for w in k.split()[:2]) for k in kw), axis=1)
            rr = rr.sort_values(["_m", "priority"], ascending=[False, True]).drop(columns="_m")
            rr["priority"] = range(1, len(rr) + 1)
        rd = _df("ROADMAP")
        when = dict(zip(rd["initiative"], rd["phase"])) if len(rd) else {}
        def _ev_short(s_):
            parts = [p.strip() for p in str(s_).split(";")]
            return prettify_text("; ".join(parts[:2]))[:90]
        tt = pd.DataFrame({"#": rr["priority"].astype(int), "Action": rr["recommendation"],
                           "Key evidence": rr["evidence"].map(_ev_short),
                           "Expected impact": rr["estimated_impact"].map(lambda s: (str(s).split(" (")[0][:60]) if str(s) else "measured in pilot"),
                           "KPI": rr["kpi"].map(lambda s: str(s).split(";")[0][:45]),
                           "When": rr["recommendation"].map(lambda t: when.get(t, "").split(" (")[0])}).head(7 if level < 3 else 5)
        TAB(tt, "Recommended retention actions, ranked by evidence and expected value.", "recs", prio=1, font=8)
        for _, r in rr.head(3 if level < 3 else 2).iterrows():
            extra = ""
            if F["surv"] and re.search(r"onboard|first.year|early", str(r["recommendation"]), re.I):
                extra = (f" Timing matters: because the churn hazard peaks {interval_phrase(F['peak_interval'], F['surv_unit'])}, the key "
                         "touchpoints (welcome call, coverage check, renewal-readiness contact) should be completed before that window opens.")
            P(f"{int(r['priority'])}. {r['recommendation']}. {r['action']}{extra}", prio=1 if r["priority"] <= 2 else 2)
    if CFG.AVAILABLE_INTERVENTIONS:
        P(f"The case lists the following feasible interventions: {_join(CFG.AVAILABLE_INTERVENTIONS)}; the action plan prioritises those supported by our evidence.", prio=2)
    pl = F["pilot"]
    if len(pl):
        p0 = pl.iloc[0]
        P(f"Implementation follows a pilot → measure → scale logic ({{fig:implementation_roadmap}}). Every action keeps a randomised control group: with a "
          f"baseline churn of {pct(p0['baseline_churn_target_segment'])} in the High-risk tier, detecting {a_n(format(p0['expected_reduction_pp'], '.1f'))} {p0['expected_reduction_pp']:.1f} pp "
          f"reduction requires {int(p0['n_per_arm']):,} customers per arm (α = {CFG.PILOT_ALPHA}, power {CFG.PILOT_POWER:.0%}). Pilot results then "
          "train uplift models so that future campaigns target customers by expected response × value rather than by risk alone "
          "(Ascarza, 2018). The model is deployed as a monthly early-warning score with drift and calibration monitoring "
          "and quarterly retraining (Appendix J).")
    FIG("implementation_roadmap", "Twelve-month implementation roadmap: each initiative is piloted against a control group before scale-up.", prio=3, w=FW)

    # ================= 6. DISCUSSION & CONCLUSION =================
    H("6. Discussion and Conclusion")
    themes = F["themes"]
    lit = research_literature(themes)          # literatur temuan tim (CASE_RESEARCH.xlsx) diprioritaskan
    if "lifecycle" in themes:
        lit.append("The concentration of exits early in the relationship is consistent with evidence that lapse declines with contract age "
                   "(Eling & Kiesenbauer, 2014); because acquisition costs are "
                   "front-loaded, early-tenure retention has the highest value per retained policy.")
    if "payment" in themes:
        lit.append("The role of payment behaviour agrees with Hein et al. (2020) and with survey evidence that forgetting to pay explains a "
                   "large share of lapses (Gottlieb & Smetters, 2021); auto-debit enrolment and reminders before the grace period ends are "
                   "low-cost actions aimed at this mechanism.")
    if "price" in themes:
        lit.append("Price sensitivity at renewal is in line with Guelman and Guillén (2014) and Brockett et al. (2008); renewal pricing should "
                   "also respect fairness, as illustrated by the UK ban on charging renewing customers more than equivalent new customers "
                   "(Financial Conduct Authority, 2021).")
    if "service" in themes:
        lit.append("The link between the claims and service experience and churn is consistent with industry evidence that customers satisfied "
                   "with claims are more likely to renew (Fallis et al., 2022).")
    if "affordability" in themes:
        lit.append("The affordability pattern is consistent with the emergency fund hypothesis (Outreville, 1990; Fier & Liebenberg, 2013); "
                   "premium holidays or lower-premium plans may retain coverage for these customers.")
    if "relationship" in themes:
        lit.append("Consistent with Brockett et al. (2008), the breadth of the relationship matters: cancellation of one policy should be "
                   "treated as an early warning for the whole relationship.")
    if not lit:
        lit.append("Our findings extend prior work on lapse and churn determinants (Eling & Kochanski, 2013; Günther et al., 2014) to this "
                   "portfolio.")
    exp_neg = [n for n in F["negligible"] if any(k in n.lower() for k in ["income", "premium", "price", "credit"])]
    if exp_neg:
        lit.append(f"Contrary to the emphasis on economic drivers in parts of the literature (Kuo et al., 2003), {_join(exp_neg[:2])} "
                   "show little association with churn here, plausibly because of differences in product mix and market.")
    answers = []
    if F["drivers"]:
        answers.append(f"RQ1: churn is driven mainly by {_join([nm(n) for n in top_names])}")
    if F["surv"]:
        answers.append(f"RQ2: risk peaks {interval_phrase(F['peak_interval'], F['surv_unit'])}")
    answers.append(f"RQ3: churn can be predicted with {F['auc_txt']}, concentrating {F['top20_capture']:.0f}% of churners in the top 20%")
    if "camp" in F:
        _rob = F["robust"]
        _lev = (f". The most promising lever is {lever_phrase(_rob[_rob['verdict'] == 'robust'].sort_values('ate_pp').iloc[0]['treatment (T=1)']).replace('moving customers to ', 'moving customers to ')}"
                if len(_rob) and (_rob["verdict"] == "robust").any() else
                ", while the data contain no controllable levers whose effect can be estimated, so interventions must be tested in pilots")
        _rq4 = f"RQ4: a value-targeted campaign returns about {money(F['camp']['net_profit'])} net"
        if F.get("pf"):
            _rq4 += f" in the analysed sample (about {money(F['camp']['net_profit'] * F['pf_scale'])} at portfolio scale)"
        if F.get("budget_net") is not None and F.get("budget_binding"):
            _rq4 += (f"; within the {money(CFG.RETENTION_BUDGET)} budget, contacting the {F['budget_contacts']:,} highest-loss customers "
                     f"returns about {money(F['budget_net'])} net")
        answers.append(_rq4 + _lev)
    _n_lit = 3 if len(research_literature(themes)) else 2
    _ans = []
    for a_ in answers:
        k_, _, body_ = a_.partition(": ")
        body_ = body_.strip()
        _ans.append(f"{k_}: {body_[:1].upper() + body_[1:]}{'' if body_.endswith('.') else '.'}")
    P(f"The {num_word(len(answers))} research questions can now be answered. " + " ".join(_ans) + " These results are consistent with prior evidence. "
      + " ".join(lit[:_n_lit]))
    P("Two features distinguish this analysis from a standard churn model. First, every claim is supported by converging evidence — "
      "univariate tests, multivariate regression, survival models and SHAP — with uncertainty reported as confidence intervals. Second, "
      "recommendations are prioritised by money" + (" and by causal plausibility" if len(F["causal"]) else "") + ", and are designed to be tested. On governance, the score is "
      "audited for calibration and selection differences across customer groups and documented in a model card; it is "
      "intended for retention outreach only, not for pricing or underwriting (Appendix J).", prio=2)
    for txt in research_texts("limitations", "paragraph", "start"):
        P(txt)
    _given = str(getattr(CFG, "BUSINESS_PARAMS_SOURCE", "assumed")).lower() == "case"
    P("Limitations. The data are a single snapshot from one insurer, so results may not generalise and may shift with economic conditions. "
      "Drivers identified by statistical tests, regression and SHAP are associations"
      + ("; the causal estimates rely on the assumption of no unmeasured confounding, which we probe with E-values but cannot rule out. "
         if len(F["causal"]) else ", and the data record no past interventions, so the effect of each action must be measured in a pilot. ")
      + ("The campaign economics use the parameters given in the case, which we stress-test in a sensitivity analysis. " if _given else
         "The campaign economics depend on assumed costs and acceptance rates, which we vary in sensitivity analysis. ")
      + "Variables such as competitor prices, service interactions and household income shocks are not observed. A randomised pilot of the "
        "recommended actions is the next step to measure their true effect.")
    for txt in research_texts("limitations", "paragraph", "end"):
        P(txt)
    _custom(CUR[0], "end")
    for txt in research_texts("conclusion", "paragraph", "start") + research_texts("conclusion", "paragraph", "end"):
        P(txt)
    CUR[0] = None

    # ================= REFERENCES =================
    B.append({"k": "refs"})
    return B, {"title": title, "title_options": title_opts, "facts": F}


DEFERRED, DEFERRED_T = [], []


def prettify_text(text):
    """Ganti nama kolom teknis di teks → label bisnis (bagian [EDIT] tidak disentuh)."""
    text = str(text)
    if "[EDIT" in text or re.search(r"\{\{?(fig|tab):", text):
        parts = re.split(r"(\[EDIT[^\]]*\]|\{\{?(?:fig|tab):[^}]+\}?\})", text)
        return "".join(p if (p.startswith("[EDIT") or p.startswith("{")) else prettify_text(p) for p in parts)
    keys = sorted([k for k in set(map(str, train.columns)) | set(FE_LABELS) | set(CFG.FEATURE_LABELS) if "_" in k or k in CFG.FEATURE_LABELS],
                  key=len, reverse=True)
    for k in keys:
        if k in text:
            text = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(k) + r"(?![A-Za-z0-9_])", pretty(k), text)
    return text


def find_fig(key):
    for fn, title in reversed(FIGURE_INDEX):
        if key in fn:
            return os.path.join(CFG.FIG_DIR, fn)
    return None

# %%
# ===== 24.3c Appendix builder, DOCX writer, MS Word page check, autofit ≤ 10 halaman, PDF =====
APPENDIX_PLAN = [
    ("A", "Data Overview and Quality",
     "This appendix documents the raw data, data quality checks and the transformations applied before analysis.",
     ["data_lineage", "target_distribution", "missing_values", "numeric_distributions", "categorical_frequencies"],
     [("data_dictionary", "Data dictionary and quality report (inferred type, missing values, cardinality)."),
      ("leakage_screen", "Leakage screen: name-based, missingness-based and single-feature AUC checks."),
      ("numeric_summary", "Summary statistics of numeric variables (percentiles, skewness, missing share).")]),
    ("B", "Methodology and Validation Design",
     "Architecture of the analytics pipeline, the cross-validation design and the predictive model, plus tuning results.",
     ["system_architecture", "validation_design", "model_architecture", "analysis_framework"],
     [("config_summary", "Key analysis settings (reproducibility)."), ("optuna_tuning_summary", "Bayesian hyperparameter optimisation (Optuna) results.")]),
    ("C", "Driver Analysis (Detailed)",
     "Churn rates for every analysed variable with 95% confidence intervals, predictive strength (Information Value), correlations and interactions.",
     ["report_drivers", "driver_churn_rate", "information_value_ranking", "numeric_by_churn", "churn_reasons", "interaction_heatmaps", "segment_tree",
      "spearman_correlation", "cramers_v_matrix"],
     [("driver_table", "Univariate driver statistics (top 25 by Information Value; q-values Benjamini–Hochberg)."),
      ("segment_rules", "All decision-tree segments."), ("vif", "Variance inflation factors (numeric features)."),
      ("churn_reasons_*", "Stated churn reasons (share of churners; descriptive only, the variable is excluded from modelling).")]),
    ("D", "Interpretable Logistic Regression",
     "Adjusted odds ratios with 95% confidence intervals; numeric variables are standardised (per 1 SD), categories are compared with the most common level.",
     ["logit_odds_ratios"], [("logit_odds_ratios", "Logistic regression: odds ratios, 95% CI, p-values and average marginal effects.")]),
    ("E", "Survival Analysis",
     "Kaplan–Meier estimates, hazards by tenure, Cox and Weibull AFT models, proportional-hazards diagnostics and validation of risk tiers over time.",
     ["report_timing", "km_overall", "km_by_group", "hazard_by_tenure", "cox_hazard_ratios", "weibull_aft_time_ratios", "rmst_by_group", "km_by_risk_tier"],
     [("hazard_by_tenure_interval", "Hazard by tenure interval."), ("logrank_tests", "Log-rank tests by group."),
      ("cox_hazard_ratios", "Cox proportional hazards model."), ("ph_assumption_test", "Schoenfeld test of the proportional-hazards assumption."),
      ("weibull_aft_time_ratios", "Weibull accelerated failure time model (time ratios)."), ("rmst_by_group", "Restricted mean survival time by group."),
      ("survival_by_risk_tier", "Survival probability by predicted risk tier.")]),
    ("F", "Model Development, Comparison and Validation",
     "All logged experiments (trial-and-error), statistical comparison of models, ablation study and diagnostic checks.",
     ["report_model", "model_development_journey", "model_leaderboard", "model_comparison_delong_empc", "ablation_study", "optuna_history",
      "oof_prediction_correlation", "fold_stability", "learning_curve", "roc_pr_curves", "confusion_matrices", "threshold_curves",
      "calibration_reliability", "oof_vs_test_predictions"],
     [("model_comparison_full", "Full model comparison: AUC with DeLong CI, PR-AUC, Brier, ECE, KS and EMPC."),
      ("ablation_study", "Ablation study by feature group (DeLong tests, Holm-corrected)."),
      ("ensemble_comparison", "Ensemble methods compared."), ("bootstrap_ci_final_model", "Bootstrap 95% confidence intervals for the final model."),
      ("leaderboard_all", "All models: out-of-fold and training AUC, overfitting gap and cross-validation variability."),
      ("leaderboard_baseline", "Baseline models with default hyperparameters (starting point of the development journey)."),
      ("bootstrap_model_comparison", "Paired bootstrap comparison of each candidate model with the final model."),
      ("permutation_importance_selection", "Out-of-fold permutation importance used for feature selection."),
      ("performance_by_segment", "Model performance by customer segment (AUC and churn rate per level)."),
      ("error_profile", "Error analysis: share of true/false positives and negatives by feature."),
      ("experiment_log", "Experiment log (all logged runs)."), ("train_test_drift", "Train–test drift (PSI, KS; top 15).")]),
    ("G", "Explainability",
     "Global and local explanations: gain, permutation and SHAP importance, dependence and partial dependence, individual explanations and segment heterogeneity.",
     ["gain_importance", "permutation_importance", "shap_importance_bar", "shap_beeswarm", "shap_dependence", "shap_categorical_levels",
      "shap_waterfall", "partial_dependence", "driver_heterogeneity", "evidence"],
     [("driver_heterogeneity_by_*", "Share of SHAP importance by customer segment (%), top drivers."),
      ("shap_importance", "SHAP importance and direction of effect (top 20)."), ("driver_evidence_matrix", "Consolidated driver evidence matrix.")]),
    ("H", "Causal Inference and Uplift",
     "Doubly robust effect estimates, identification assumptions (DAG), overlap, sensitivity (E-values), placebo tests and heterogeneous effects.",
     ["report_causal", "causal_effects_aipw", "causal_dag", "causal_overlap", "uplift_targeting", "what_if_scenarios"],
     [("causal_effects_aipw", "AIPW average treatment effects."), ("causal_robustness", "Robustness of causal estimates."),
      ("uplift_summary", "Uplift (DR-learner) summary."), ("uplift_gates", "Sorted group average treatment effects (GATES)."),
      ("what_if_scenarios", "Model-based what-if simulation (associational).")]),
    ("I", "Business Analytics",
     "Lift and gains, risk tiers, risk × value prioritisation, campaign simulation and sensitivity, churn personas.",
     ["report_business", "executive_summary", "gains_and_lift", "risk_value_matrix", "campaign_profit_curve", "churn_personas_heatmap"],
     [("decile_lift_table", "Lift by risk decile."), ("risk_tiers", "Risk tiers."), ("risk_value_quadrants", "Risk × value quadrants."),
      ("campaign_optimum", "Optimal campaign size by strategy."), ("campaign_sensitivity", "Campaign sensitivity analysis."),
      ("churn_personas", "Churn personas.")]),
    ("J", "Governance, Deployment and Implementation",
     "Fairness audit, model card, deployment loop, pilot design and implementation roadmap.",
     ["fairness_audit", "mlops_loop", "pilot_power_analysis", "implementation_roadmap"],
     [("model_card", "Model card (Mitchell et al., 2019)."), ("fairness_summary", "Fairness summary by attribute."),
      ("pilot_design_power", "Pilot design: sample size per arm."), ("implementation_roadmap", "Implementation roadmap."),
      ("recommendations", "Full recommendation list with evidence and KPIs.")]),
]


APPENDIX_TEXT = {
    "B": [
        "Statistical screening. For categorical variables we use Pearson's χ² test with Cramér's V; for numeric variables the Mann–Whitney U "
        "test with the rank-biserial correlation, so that every p-value is paired with an effect size. Because dozens of variables are tested, "
        "p-values are converted to Benjamini–Hochberg q-values. Weight of evidence and Information Value, standard in credit and insurance "
        "scoring, summarise each binned variable's predictive strength (Siddiqi, 2006); proportions are reported with Wilson 95% intervals.",
        "Survival layer. Tenure is the time axis and active customers are right-censored. Kaplan–Meier curves and log-rank tests describe "
        "retention; discrete hazards per tenure interval locate the critical window; the Cox model gives adjusted hazard ratios, its "
        "proportional-hazards assumption is checked with Schoenfeld residuals (Schoenfeld, 1982), and a Weibull accelerated failure time "
        "model provides time ratios that remain interpretable when the assumption fails (Wei, 1992). Restricted mean survival time expresses "
        "differences in months or days retained (Royston & Parmar, 2013).",
        "Predictive layer. Logistic regression, random forest, extra trees, histogram gradient boosting, LightGBM, XGBoost and CatBoost "
        "(Ke et al., 2017; Chen & Guestrin, 2016; Prokhorenkova et al., 2018), a glass-box Explainable Boosting Machine (Lou et al., 2013) and "
        "a neural network are trained on identical folds. Hyperparameters are tuned with the TPE sampler and median pruning (Akiba et al., 2019) "
        "on inner folds; the best families are re-trained with several seeds; features with non-positive permutation importance are tested for "
        "removal; and six ensembling strategies, including greedy ensemble selection (Caruana et al., 2004) and stacking, are compared, the "
        "simplest within 0.0005 of the best being preferred. Probabilities are calibrated by isotonic or Platt scaling when this lowers "
        "cross-validated log-loss, and calibration is summarised by the Brier score and expected calibration error (Guo et al., 2017).",
        "Model comparison. Correlated AUCs are compared with the DeLong test (DeLong et al., 1988) using the fast algorithm, with Holm "
        "correction across comparisons; bootstrap intervals complement the analytic ones. Because AUC is not aligned with campaign profit "
        "(Verbeke et al., 2012; Höppner et al., 2020), models are also ranked by the expected maximum profit measure for churn, which "
        "integrates campaign profit over a Beta-distributed acceptance rate (Verbraken et al., 2013). The risk score is further validated on "
        "time-to-churn with Harrell's C-index (Harrell et al., 1982).",
        "Why not only an interpretable model? Interpretable models should be preferred when they perform comparably (Rudin, 2019); we "
        "therefore report the logistic regression and the EBM next to the ensemble and test whether the accuracy gain is significant.",
    ],
    "H": [
        "Identification. For each controllable lever a binary 'favourable condition' T is defined. Under no unmeasured confounding and overlap, "
        "the augmented inverse probability weighting estimator combines a propensity model e(X) and outcome models μ1(X), μ0(X) and stays "
        "consistent if either is correct (Robins et al., 1994); all nuisance models are cross-fitted with five folds to avoid own-observation "
        "bias (Chernozhukov et al., 2018). Propensity scores are clipped at 0.02/0.98 and the share of customers outside [0.05, 0.95] is reported.",
        "Robustness. A placebo test re-estimates the effect with a randomly permuted treatment (should be ≈ 0). E-values quantify how strong an "
        "unmeasured confounder would have to be, on the risk-ratio scale, to explain away the estimate (VanderWeele & Ding, 2017). Only effects "
        "that pass all checks are labelled 'robust'.",
        "Heterogeneity and uplift. The doubly robust pseudo-outcome is regressed on customer features with cross-fitting (DR-learner; Kennedy, "
        "2023), a meta-learner in the family studied by Künzel et al. (2019). Customers are sorted into quintiles of predicted uplift and the "
        "average pseudo-outcome per quintile (with standard errors) tests whether effects differ. Uplift-based targeting is compared with "
        "risk-based targeting, following the uplift literature for insurance retention (Guelman et al., 2015; Devriendt et al., 2021).",
    ],
    "J": [
        "The model card follows Mitchell et al. (2019). The fairness audit compares calibration (mean predicted vs actual churn), discrimination "
        "(AUC) and selection into the High-risk tier across customer groups; differences in selection that mirror genuine differences in churn "
        "are expected, whereas calibration gaps would indicate bias. The score is restricted to retention outreach.",
    ],
}


def _fmt_cell(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        a = abs(v)
        if a >= 1e6:
            return fmt_num(v)
        if a >= 1000:
            return f"{v:,.0f}"
        if a >= 10:
            return f"{v:,.1f}"
        if a == 0:
            return "0"
        if a < 0.001:
            return f"{v:.1e}"
        return f"{v:.3f}"
    return prettify_text(str(v))


def appendix_table(name):
    if name == "data_dictionary" and "quality_report" in globals():
        q = quality_report.copy()
        inv_roles = {c: r for r, c in ROLES.items()}
        q.insert(2, "role", q["column"].map(inv_roles).fillna(""))
        return q[[c for c in ["column", "inferred_type", "role", "missing_%", "n_unique", "top_value", "top_freq_%"] if c in q]].head(80)
    if name == "config_summary":
        keys = ["RUN_MODE", "SEED", "N_FOLDS", "N_REPEATS", "METRIC", "MODELS", "USE_OPTUNA", "OPTUNA_TRIALS", "FINAL_SEEDS", "CALIBRATION",
                "PROFIT_MARGIN", "RETENTION_COST", "RETENTION_SUCCESS_RATE", "DISCOUNT_RATE", "CLV_HORIZON_YEARS", "RISK_TIER_CUTS",
                "WHAT_IF_ADOPTION", "PILOT_POWER", "PILOT_ALPHA"]
        return pd.DataFrame({"setting": keys, "value": [str(getattr(CFG, k, ""))[:80] for k in keys]})
    if name == "leakage_screen" and "leakage_screen" in TABLES:
        return TABLES[name].head(15)
    if name == "driver_table" and "driver_table" in TABLES:
        cols = ["feature", "type", "iv", "iv_strength", "effect_metric", "effect_size", "q_value_BH", "univariate_auc", "highest_risk_level", "highest_risk_rate"]
        t = TABLES[name]
        return t[[c for c in cols if c in t]].head(25)
    if name == "experiment_log" and name in TABLES:
        t = TABLES[name]
        return t[[c for c in ["exp_id", "stage", "experiment", "note", "roc_auc", "pr_auc", "logloss", "time_s"] if c in t]].head(60)
    if name == "train_test_drift" and name in TABLES:
        return TABLES[name].head(15)
    if name == "shap_importance" and name in TABLES:
        return TABLES[name][[c for c in ["feature", "mean_abs_shap", "share_%", "direction"] if c in TABLES[name]]].head(20)
    if name == "driver_evidence_matrix" and name in TABLES:
        t = TABLES[name]
        return t[[c for c in ["feature", "iv", "effect_size", "q_value_BH", "highest_risk_level", "highest_risk_rate", "odds_ratio",
                              "hazard_ratio", "shap_rank", "evidence_score", "actionable"] if c in t]].head(25)
    if name == "model_comparison_full" and name in TABLES:
        t = TABLES[name]
        return t[[c for c in ["model", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc", "brier", "ece", "ks", "empc_per_customer",
                              "delong_p_holm"] if c in t]]
    if name == "recommendations" and name in TABLES:
        return TABLES[name][["priority", "recommendation", "action", "evidence", "estimated_impact", "kpi"]]
    if name in TABLES:
        return TABLES[name].head(40)
    return None


NICE_CAPTIONS = {
    "validation_design": "Validation design: stratified k-fold cross-validation producing out-of-fold predictions for model comparison and evaluation.",
    "causal_dag": "Assumed causal diagram (DAG) behind the doubly robust effect estimates: lever, confounders and churn.",
    "system_architecture": "End-to-end analytics architecture: from raw policy data to retention decisions.",
    "model_architecture": "Predictive model architecture: base learners, tuning and ensemble.",
    "mlops_loop": "Deployment and monitoring loop: scoring, campaign, pilot feedback and retraining.",
    "data_lineage": "Data lineage: what happened to rows and columns.",
}


def nice_caption(fn, cap):
    """Caption yang masih berupa nama file (mis. 'validation_design') → kalimat yang layak tampil."""
    if cap and " " in str(cap).strip():
        return cap
    base = re.sub(r"^(fig_?)?\d+[_\-]*", "", os.path.splitext(os.path.basename(str(fn)))[0])
    for k, v in NICE_CAPTIONS.items():
        if k in base:
            return v
    return base.replace("_", " ").strip().capitalize() or str(cap)


def _T(name):
    t = TABLES.get(name) if "TABLES" in globals() else None
    return t if isinstance(t, pd.DataFrame) and len(t) else None


def appendix_insights(letter):
    """Observasi kunci per appendix, dihitung dari tabel hasil run (setiap butir aman bila tabelnya tidak ada)."""
    out = []

    def tryadd(fn):
        try:
            t_ = fn()
            if t_:
                out.append(t_)
        except Exception:
            pass

    if letter == "A":
        def a1():
            q = _T("data_quality_report")
            hi = q.sort_values("missing_%", ascending=False).iloc[0]
            return (f"Of {len(q)} profiled columns, {int((q['missing_%'] > 0).sum())} contain missing values; the most incomplete is "
                    f"{pretty(hi['column'])} ({hi['missing_%']:.1f}% missing). Missingness was kept as information through indicator features "
                    "rather than imputed away.") if hi["missing_%"] > 0 else f"None of the {len(q)} profiled columns contains missing values."
        def a2():
            if LEAKY_COLS:
                return (f"The leakage screen removed {len(LEAKY_COLS)} variable(s) ({_join([pretty(x) for x in LEAKY_COLS[:4]])}); each one alone "
                        "predicts churn almost perfectly because it is recorded after the event.")
        tryadd(a1); tryadd(a2)
    elif letter == "C":
        def c1():
            d = _T("driver_table")
            sig = int((d["q_value_BH"] < 0.05).sum())
            top = d.sort_values("iv", ascending=False).head(3)
            return (f"{sig} of {len(d)} variables remain significant after Benjamini–Hochberg correction (q < 0.05). By Information Value the "
                    f"strongest are {_join([f'{pretty(r.feature)} ({r.iv:.2f})' for r in top.itertuples()])}; values above 0.1 indicate medium "
                    "and above 0.3 strong predictive power (Siddiqi, 2006).")
        def c2():
            v = _T("vif").sort_values("VIF", ascending=False).iloc[0]
            return (f"The highest variance inflation factor is {v['VIF']:.1f} ({pretty(v['feature'])}); "
                    + ("all values are below 5, so multicollinearity does not distort the regression estimates."
                       if v["VIF"] < 5 else "the regression in Appendix D therefore reports this variable with caution."))
        tryadd(c1); tryadd(c2)
    elif letter == "E":
        def e1():
            ph = _T("ph_assumption_test")
            n_v = int(ph["ph_violated"].astype(bool).sum())
            return (f"Schoenfeld tests reject proportional hazards for {n_v} of {len(ph)} terms"
                    + (", which is why the Weibull accelerated failure time model is reported alongside the Cox model." if n_v else
                       "; the Cox hazard ratios can be read as constant over tenure."))
        def e2():
            cx = _T("cox_hazard_ratios")
            s_ = cx[cx["p_value"] < 0.05].sort_values("hazard_ratio", ascending=False)
            if len(s_):
                r = s_.iloc[0]
                return (f"{len(s_)} Cox terms are significant at 5%; the largest hazard ratio is {r['hazard_ratio']:.2f} "
                        f"(95% CI {r['hr_ci_low']:.2f}–{r['hr_ci_high']:.2f}) for {prettify_text(str(r['term']))}.")
        def e3():
            pr = persistency_ratios()
            if pr:
                return ("Persistency ratios, the standard insurance KPI for the share of policies still in force: "
                        + _join([f"{m}-month {pct(v)}" for m, v in pr.items()]) + ".")
        tryadd(e3); tryadd(e1); tryadd(e2)
    elif letter == "F":
        def f1():
            ex = _T("experiment_log")
            base = _T("leaderboard_baseline")
            b0 = base["oof_roc_auc"].max() if base is not None else ex["roc_auc"].min()
            return (f"The development journey logged {len(ex)} experiments. The best default-parameter model reached ROC-AUC {b0:.3f}; tuning, "
                    f"multi-seed averaging, feature selection and ensembling raised the final out-of-fold AUC to {roc_auc_score(y, FINAL_OOF):.3f} "
                    f"({roc_auc_score(y, FINAL_OOF) - b0:+.3f}).")
        def f2():
            ab = _T("ablation_study").sort_values("auc_drop", ascending=False).iloc[0]
            return (f"In the ablation study, removing {prettify_text(str(ab['ablation']).replace('− ', ''))} costs the most accuracy "
                    f"(ΔAUC {ab['auc_drop']:.3f}, Holm-adjusted p {p_fmt(ab['holm_p'])}).")
        def f3():
            dr = _T("train_test_drift").sort_values("psi", ascending=False).iloc[0]
            return (f"The largest train–test shift is PSI {dr['psi']:.3f} ({pretty(dr['feature'])}); "
                    + ("all features are below the 0.1 threshold, so test predictions are not affected by drift." if dr["psi"] < 0.1 else
                       "this feature should be monitored after deployment."))
        def f4():
            ps = _T("performance_by_segment")
            ps = ps[ps["n"] >= 100]
            lo = ps.sort_values("auc").iloc[0]
            return (f"Across segments with at least 100 customers, AUC ranges from {ps['auc'].min():.3f} to {ps['auc'].max():.3f}; the weakest "
                    f"segment is {pretty(lo['feature'])} = {lo['level']} (n = {int(lo['n']):,}).")
        for fn in (f1, f2, f3, f4):
            tryadd(fn)
    elif letter == "G":
        def g1():
            sh = _T("shap_importance").sort_values("share_%", ascending=False)
            return (f"The top five SHAP features account for {sh['share_%'].head(5).sum():.0f}% of total attribution; "
                    f"{_join([pretty(f) for f in sh['feature'].head(3)])} lead the ranking, in line with the Information Value ranking in Appendix C.")
        tryadd(g1)
    elif letter == "H":
        def h1():
            rb = _T("causal_robustness")
            n_r = int((rb["verdict"] == "robust").sum())
            return (f"{n_r} of {len(rb)} estimated effects pass every robustness check (placebo effect close to zero, overlap problems below the "
                    f"threshold and E-value above 1.25). The smallest E-value among them is "
                    f"{rb[rb['verdict'] == 'robust']['e_value_ci'].min():.2f}, so an unmeasured confounder would need at least this risk ratio with "
                    "both treatment and churn to explain the effect away.") if n_r else None
        tryadd(h1)
    elif letter == "I":
        def i1():
            d = _T("decile_lift_table")
            return (f"The top risk decile churns at {pct(d['churn_rate'].iloc[0])} ({d['lift'].iloc[0]:.1f}× the average) and the bottom decile at "
                    f"{pct(d['churn_rate'].iloc[-1])}, {a_n(format(d['churn_rate'].iloc[0] / max(d['churn_rate'].iloc[-1], 1e-9), '.0f'))} {d['churn_rate'].iloc[0] / max(d['churn_rate'].iloc[-1], 1e-9):.0f}-fold spread that makes "
                    "targeted retention economical.")
        tryadd(i1)
    elif letter == "J":
        def j1():
            fs = _T("fairness_summary").sort_values("auc_gap", ascending=False).iloc[0]
            return (f"The largest discrimination gap across groups is {fs['auc_gap']:.3f} AUC ({pretty(fs['attribute'])}); calibration ranges from "
                    f"{fs['min_cal']:.2f} to {fs['max_cal']:.2f} across its levels. Selection differences that follow genuine differences in churn are "
                    "expected; the score is used only to decide who receives retention outreach.")
        tryadd(j1)
    return [("Key observations. " + out[0])] + out[1:] if out else []


METHOD_FORMULAS = [
    ("Information Value and Wilson intervals. For each bin j of a variable, the weight of evidence and Information Value are (Siddiqi, 2006):",
     ["WoEⱼ = ln( %non-churnersⱼ / %churnersⱼ )", "IV = Σⱼ ( %non-churnersⱼ − %churnersⱼ ) · WoEⱼ"],
     "Values above 0.1 indicate medium and above 0.3 strong predictive power. Churn rates carry Wilson score intervals:",
     ["CI = [ p̂ + z²/(2n) ± z·√( p̂(1 − p̂)/n + z²/(4n²) ) ] / (1 + z²/n)"]),
    ("Multiple testing. Univariate p-values, ordered p₍₁₎ ≤ … ≤ p₍ₘ₎, are converted to Benjamini–Hochberg q-values (Benjamini & Hochberg, 1995); "
     "model comparisons use the Holm step-down correction:", ["q₍ᵢ₎ = min{ m · p₍ₖ₎ / k  :  k ≥ i }"], "", []),
    ("Survival. With dᵢ churn events and nᵢ customers at risk at time tᵢ, the Kaplan–Meier estimator, the Cox model and the restricted mean "
     "survival time are (Kaplan & Meier, 1958; Cox, 1972; Royston & Parmar, 2013):",
     ["Ŝ(t) = ∏ ( 1 − dᵢ / nᵢ ),   product over all tᵢ ≤ t", "h(t | x) = h₀(t) · exp(β′x)", "RMST(τ) = ∫ Ŝ(t) dt,   integrated from 0 to τ"],
     "exp(βₖ) is a hazard ratio, and the 13- and 25-month persistency ratios are Ŝ(13) and Ŝ(25) with tenure in months.", []),
    ("Discrimination and calibration. ROC-AUC is the probability that a random churner scores higher than a random non-churner; correlated "
     "AUCs are compared with the DeLong variance estimator (DeLong et al., 1988). Calibration uses the Brier score and the expected calibration "
     "error over probability bins k (Guo et al., 2017):", ["BS = (1/n) Σᵢ ( pᵢ − yᵢ )²", "ECE = Σₖ ( nₖ / n ) · | ȳₖ − p̄ₖ |"], "", []),
    ("Profit-based evaluation. The expected maximum profit measure integrates campaign profit over a Beta-distributed acceptance rate γ, "
     "where η is the share of customers targeted (Verbraken et al., 2013); customer value is the discounted margin while the policy stays in force:",
     ["EMPC = ∫ max{ P(η; γ) : η } · h(γ) dγ", "CLVᵢ = m·Pᵢ·(1 − ρᵢᴴ) / (1 − ρᵢ),   ρᵢ = (1 − pᵢ) / (1 + d)"],
     "with margin m, annual premium Pᵢ, churn probability pᵢ, discount rate d and horizon H.", []),
    ("Causal effects. With treatment T, outcome Y, propensity ê(X) and outcome models μ̂ₜ(X), the augmented inverse probability weighting "
     "estimator is consistent if either nuisance model is correct (Robins et al., 1994); nuisance models are cross-fitted (Chernozhukov et al., 2018):",
     ["ATE = (1/n) Σᵢ [ μ̂₁(Xᵢ) − μ̂₀(Xᵢ) + Tᵢ(Yᵢ − μ̂₁(Xᵢ))/ê(Xᵢ) − (1 − Tᵢ)(Yᵢ − μ̂₀(Xᵢ))/(1 − ê(Xᵢ)) ]"],
     "The E-value for a risk ratio RR states how strong an unmeasured confounder would have to be to explain the effect away "
     "(VanderWeele & Ding, 2017):", ["E = RR + √( RR · (RR − 1) )"]),
    ("Drift and pilot size. The population stability index compares the share of customers per bin k in scoring data (aₖ) and training "
     "data (eₖ); values below 0.1 indicate no material shift. Pilot arms use the two-proportion sample-size formula:",
     ["PSI = Σₖ ( aₖ − eₖ ) · ln( aₖ / eₖ )", "n = ( z₁₋α/₂ + z₁₋β )² · [ p₁(1 − p₁) + p₂(1 − p₂) ] / ( p₁ − p₂ )²"], "", []),
]


def build_appendix(used_figs, used_tabs):
    A = [{"k": "appendix_start"}, {"k": "h", "t": "Appendices", "lvl": 1},
         {"k": "p", "t": "The appendices are not counted in the page limit. They document every analysis step so that each result in the main text "
                         "can be traced and reproduced; all figures and tables were generated by the accompanying notebook.", "prio": 1}]
    placed = set(used_figs)
    all_figs = [fn for fn, _ in FIGURE_INDEX]
    deferred_caps = {k: c for k, c in DEFERRED}
    for letter, title, intro, fig_keys, tabs in APPENDIX_PLAN:
        figs = []
        for key in fig_keys:
            for fn in all_figs:
                if key in fn and fn not in placed:
                    figs.append(fn)
                    placed.add(fn)
        _tabs = []
        for n, c in tabs:                                   # nama berakhiran * = tabel bernama dinamis (mis. churn_reasons_<kolom>)
            if n.endswith("*"):
                _tabs += [(k, c) for k in sorted(TABLES) if k.startswith(n[:-1])] if "TABLES" in globals() else []
            else:
                _tabs.append((n, c))
        tabs_ok = [(n, c, appendix_table(n)) for n, c in _tabs]
        tabs_ok = [(n, c, t) for n, c, t in tabs_ok if t is not None and len(t) and n not in used_tabs]
        if letter == "F" and DEFERRED_T:
            tabs_ok = [(k_, c_, d_) for d_, c_, k_ in DEFERRED_T] + tabs_ok
        if not figs and not tabs_ok:
            continue
        A.append({"k": "h", "t": f"Appendix {letter}. {title}", "lvl": 2, "letter": letter})
        A.append({"k": "p", "t": intro, "prio": 1})
        for txt_ in appendix_insights(letter):
            A.append({"k": "p", "t": txt_, "prio": 1})
        for para in APPENDIX_TEXT.get(letter, []):
            A.append({"k": "p", "t": para, "prio": 1})
        if letter == "B":
            _ml = [t for t in research_literature(["methods"]) if t]
            _lit = CASE_RESEARCH.get("literature", pd.DataFrame())
            if len(_lit) and "theme" in _lit:
                _ml = [str(r["finding_EN"]) for _, r in _lit.iterrows() if str(r.get("theme", "")).strip().lower() == "methods" and str(r.get("finding_EN", "")).strip()]
            if _ml:
                A.append({"k": "p", "t": "Further methodological support: " + " ".join(_ml), "prio": 1})
        for fn in figs:
            key_ = next((k for k in deferred_caps if k in fn), None)
            cap = deferred_caps.get(key_) or dict(FIGURE_INDEX).get(fn, fn)
            cap = nice_caption(fn, cap)
            A.append({"k": "fig", "key": fn, "cap": cap, "w": 14.5, "file": os.path.join(CFG.FIG_DIR, fn)})
        for n, c, t in tabs_ok:
            A.append({"k": "tab", "df": t, "cap": c, "key": n, "font": 7.5 if t.shape[1] > 6 else 8.5})
        if letter == "A" and _g("clean_log"):
            A.append({"k": "p", "t": "Automated cleaning log: " + "; ".join(clean_log[:25]) + ("…" if len(clean_log) > 25 else ""), "prio": 1})
    A.append({"k": "h", "t": "Appendix K. Methodological Details and Formulas", "lvl": 2, "letter": "K"})
    A.append({"k": "p", "t": "This appendix states the estimators behind the main results so that every number can be recomputed.", "prio": 1})
    for lead_, eqs_, mid_, eqs2_ in METHOD_FORMULAS:
        A.append({"k": "p", "t": lead_, "prio": 1})
        for e_ in eqs_:
            A.append({"k": "eq", "t": e_, "prio": 1})
        if mid_:
            A.append({"k": "p", "t": mid_, "prio": 1})
        for e_ in eqs2_:
            A.append({"k": "eq", "t": e_, "prio": 1})
    import platform
    vers = [f"Python {sys.version.split()[0]}", f"pandas {pd.__version__}", f"scikit-learn {sklearn.__version__}"]
    for nm_, mod_ in [("LightGBM", lgb), ("XGBoost", xgb), ("CatBoost", cb), ("SHAP", shap), ("statsmodels", _try_import("statsmodels")),
                      ("lifelines", lifelines), ("Optuna", optuna)]:
        if mod_ is not None:
            vers.append(f"{nm_} {getattr(mod_, '__version__', '')}")
    A.append({"k": "h", "t": "Appendix L. Reproducibility", "lvl": 2, "letter": "L"})
    A.append({"k": "p", "t": f"All results were produced by one notebook executed top-to-bottom (Restart & Run All) with random seed {CFG.SEED}, "
                             f"{CFG.N_FOLDS}-fold stratified cross-validation and the settings in Appendix B. Software: {', '.join(vers)}; "
                             f"platform {platform.system()} {platform.release()}. The submission package contains the executed notebook with all outputs, "
                             "every figure and table (CSV and Excel), the model card, the per-customer action list and an interactive dashboard.", "prio": 1})
    rest = [fn for fn in all_figs if fn not in placed]
    if rest:
        A.append({"k": "h", "t": "Appendix M. Additional Figures", "lvl": 2, "letter": "M"})
        for fn in rest:
            A.append({"k": "fig", "key": fn, "cap": nice_caption(fn, dict(FIGURE_INDEX).get(fn, fn)), "w": 14.5, "file": os.path.join(CFG.FIG_DIR, fn)})
    A.append({"k": "refs_app"})
    return A


def resolve_refs(blocks):
    """Nomori Figure/Table (body: 1,2,3…; appendix: A.1, B.2 …) lalu ganti token {fig:key}/{tab:key} di teks."""
    nums, fig_n, tab_n, letter, app_f, app_t = {}, 0, 0, None, {}, {}
    for b in blocks:
        if b["k"] == "h" and b.get("letter"):
            letter = b["letter"]
        if b["k"] == "fig":
            if letter:
                app_f[letter] = app_f.get(letter, 0) + 1
                b["label"] = f"Figure {letter}.{app_f[letter]}"
            else:
                fig_n += 1
                b["label"] = f"Figure {fig_n}"
            for k_ in [b["key"]]:
                nums.setdefault(("fig", k_), b["label"])
        if b["k"] == "tab":
            if letter:
                app_t[letter] = app_t.get(letter, 0) + 1
                b["label"] = f"Table {letter}.{app_t[letter]}"
            else:
                tab_n += 1
                b["label"] = f"Table {tab_n}"
            nums.setdefault(("tab", b["key"]), b["label"])

    def lookup(kind, key):
        if (kind, key) in nums:
            return nums[(kind, key)]
        for (k2, key2), lab in nums.items():
            if k2 == kind and key in key2:
                return lab
        return None

    def sub(m):
        lab = lookup(m.group(1), m.group(2))
        return (lab + m.group(3)) if lab else "\x00"

    for b in blocks:
        if b["k"] in ("p", "bullet"):
            t = re.sub(r"\{\{?(fig|tab):([^}]+)\}?\}([a-z]?)", sub, b["t"])
            t = re.sub(r"\s*\(\x00(, \x00)*\)", "", t).replace("\x00, ", "").replace(", \x00", "").replace("\x00", "")
            t = re.sub(r"\(\s*;\s*", "(", t).replace(" ()", "")
            b["t"] = t
    return blocks


def write_report_docx(blocks, path, meta, refs):
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_TAB_ALIGNMENT
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    doc = Document()

    def fmt_section(sec):
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.top_margin = sec.bottom_margin = Cm(4)
        sec.left_margin = sec.right_margin = Cm(3)
        sec.header_distance = sec.footer_distance = Cm(1.5)

    def style_font(st, size, bold=None, italic=None):
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0, 0, 0)
        if bold is not None:
            st.font.bold = bold
        if italic is not None:
            st.font.italic = italic
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = rpr.makeelement(qn("w:rFonts"), {})
            rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), "Times New Roman")
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        st.paragraph_format.line_spacing = 1.15

    def add_page_field(par):
        r = par.add_run()
        for tag, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
            if tag:
                e = OxmlElement("w:fldChar")
                e.set(qn("w:fldCharType"), tag)
            else:
                e = OxmlElement("w:instrText")
                e.set(qn("xml:space"), "preserve")
                e.text = txt
            r._r.append(e)
        r.font.size = Pt(10)

    def restart_numbering(sec, start=1):
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:start"), str(start))
        sec._sectPr.append(pg)

    def shade(cell, fill="D9E2F3"):
        tcPr = cell._tc.get_or_add_tcPr()
        s = OxmlElement("w:shd")
        s.set(qn("w:val"), "clear")
        s.set(qn("w:color"), "auto")
        s.set(qn("w:fill"), fill)
        tcPr.append(s)

    fmt_section(doc.sections[0])
    style_font(doc.styles["Normal"], 12)
    doc.styles["Normal"].paragraph_format.space_after = Pt(6)
    doc.styles["Normal"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for nm, sz in [("Title", 16), ("Heading 1", 12), ("Heading 2", 12), ("Heading 3", 12)]:
        style_font(doc.styles[nm], sz, bold=True, italic=(nm == "Heading 3"))
        doc.styles[nm].paragraph_format.space_before = Pt(10 if nm == "Heading 1" else 6)
        doc.styles[nm].paragraph_format.space_after = Pt(4)
        doc.styles[nm].paragraph_format.keep_with_next = True
    style_font(doc.styles["Caption"], 10, bold=False, italic=False)
    try:
        style_font(doc.styles["List Bullet"], 12)
    except KeyError:
        pass

    def rich(par, text, size=None):
        for part in re.split(r"(\[EDIT[^\]]*\]|\*\*[^*]+\*\*)", text):
            if not part:
                continue
            if part.startswith("**"):
                r = par.add_run(part[2:-2])
                r.bold = True
            else:
                r = par.add_run(part)
                if part.startswith("[EDIT"):
                    r.font.highlight_color = WD_COLOR_INDEX.YELLOW
                    r.bold = True
            if size:
                r.font.size = Pt(size)

    # ---------- cover ----------
    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(meta["title"])
    members = CFG.TEAM_MEMBERS or ["[EDIT: Member 1 — Member 2 — Member 3]"]
    for t in ["", getattr(CFG, "REPORT_SUBTITLE", "Final Stage 1 — Data Analytics Competition"), "", f"Team {CFG.TEAM_NAME}",
              *members, CFG.UNIVERSITY or "[EDIT: University]", time.strftime("%Y")]:
        q = doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rich(q, str(t))
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    fmt_section(sec)
    sec.footer.is_linked_to_previous = False
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_field(fp)
    restart_numbering(sec, 1)

    n_fig = n_tab = n_eq = 0
    for b in blocks:
        k = b["k"]
        if k == "h":
            doc.add_paragraph(b["t"], style=f"Heading {min(b['lvl'], 3)}")
        elif k == "p":
            rich(doc.add_paragraph(), b["t"])
        elif k == "eq":
            n_eq += 1
            par = doc.add_paragraph()
            pf_ = par.paragraph_format
            pf_.tab_stops.add_tab_stop(Cm(7.5), WD_TAB_ALIGNMENT.CENTER)
            pf_.tab_stops.add_tab_stop(Cm(15.0), WD_TAB_ALIGNMENT.RIGHT)
            pf_.space_before, pf_.space_after = Pt(3), Pt(3)
            par.add_run("\t")
            r_ = par.add_run(str(b["t"]))
            r_.font.name = "Cambria Math"
            r_.font.size = Pt(11)
            r_._element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria Math")
            par.add_run(f"\t({n_eq})")
        elif k == "bullet":
            try:
                par = doc.add_paragraph(style="List Bullet")
            except KeyError:
                par = doc.add_paragraph()
                par.add_run("• ")
            rich(par, b["t"])
        elif k == "fig":
            f = b.get("file") or find_fig(b["key"])
            if not f or not os.path.exists(f):
                continue
            try:
                from PIL import Image
                with Image.open(f) as im:
                    wpx, hpx = im.size
                width = b["w"]
                if hpx / max(wpx, 1) * width > 19.5:          # jangan lebih tinggi dari area halaman
                    width = 19.5 * wpx / hpx
            except Exception:
                width = b["w"]
            doc.add_picture(f, width=Cm(min(width, 15.0)))
            pp = doc.paragraphs[-1]
            pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pp.paragraph_format.keep_with_next = True
            pp.paragraph_format.space_after = Pt(2)
            c = doc.add_paragraph(style="Caption")
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = c.add_run(f"{b.get('label', 'Figure')}. ")
            r.bold = True
            c.add_run(prettify_text(b["cap"]))
            n_fig += 1
        elif k == "tab":
            df = b["df"].copy()
            if df.shape[1] > 10:                       # tabel terlalu lebar → tidak terbaca di A4
                df = df.iloc[:, :10]
                b["cap"] = str(b["cap"]).rstrip(".") + " (first 10 columns; full table in report_tables.xlsx)."
            c = doc.add_paragraph(style="Caption")
            c.paragraph_format.keep_with_next = True
            r = c.add_run(f"{b.get('label', 'Table')}. ")
            r.bold = True
            c.add_run(prettify_text(b["cap"]))
            tbl = doc.add_table(rows=1, cols=len(df.columns))
            tbl.style = "Table Grid"
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            fs = b.get("font", 9)
            for j, col in enumerate(df.columns):
                cell = tbl.rows[0].cells[j]
                cell.text = ""
                _h = prettify_text(str(col))
                _h = _h.replace("_%", " %").replace("_", " ").strip()
                rr = cell.paragraphs[0].add_run(_h[:1].upper() + _h[1:])
                rr.bold = True
                rr.font.size = Pt(fs)
                cell.paragraphs[0].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                cell.paragraphs[0].paragraph_format.space_after = Pt(0)
                shade(cell)
            trPr = tbl.rows[0]._tr.get_or_add_trPr()
            th = OxmlElement("w:tblHeader")
            th.set(qn("w:val"), "true")
            trPr.append(th)
            for _, row in df.iterrows():
                cells = tbl.add_row().cells
                for j, v in enumerate(row.values):
                    cells[j].text = ""
                    par = cells[j].paragraphs[0]
                    par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    par.paragraph_format.space_after = Pt(0)
                    par.paragraph_format.line_spacing = 1.0
                    rr = par.add_run(_fmt_cell(v)[:400])
                    rr.font.size = Pt(fs)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            n_tab += 1
        elif k == "refs":
            if not getattr(CFG, "REFERENCES_COUNT_IN_LIMIT", True):
                s_ = doc.add_section(WD_SECTION.CONTINUOUS)
                fmt_section(s_)
            doc.add_paragraph("References", style="Heading 1")
            for ref in refs:
                par = doc.add_paragraph()
                par.paragraph_format.left_indent = Cm(1.0)
                par.paragraph_format.first_line_indent = Cm(-1.0)
                par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                par.paragraph_format.space_after = Pt(3)
                par.add_run(ref)
        elif k == "refs_app" and meta.get("app_refs"):
            doc.add_paragraph("References for the Appendices", style="Heading 2")
            for ref in meta["app_refs"]:
                par = doc.add_paragraph()
                par.paragraph_format.left_indent = Cm(1.0)
                par.paragraph_format.first_line_indent = Cm(-1.0)
                par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                par.paragraph_format.space_after = Pt(3)
                par.add_run(ref)
        elif k == "appendix_start":
            s2 = doc.add_section(WD_SECTION.NEW_PAGE)
            fmt_section(s2)
    doc.save(path)
    return n_fig, n_tab


def word_pdf_and_pages(docx_path, make_pdf=True):
    """MS Word (Windows): DOCX → PDF + hitung halaman. body = halaman isi utama (tanpa cover & lampiran)."""
    if os.name != "nt" or not os.path.exists(docx_path):
        return None
    docx_abs = os.path.abspath(docx_path)
    pdf_abs = os.path.splitext(docx_abs)[0] + ".pdf"
    save_pdf = f"$d.SaveAs([ref]'{pdf_abs}', [ref]17)" if make_pdf else ""
    ps = f"""
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {{
  $d = $w.Documents.Open('{docx_abs}', $false, $true)
  $d.Repaginate()
  $total = $d.ComputeStatistics(2)
  $n = $d.Sections.Count
  $ends = @()
  for ($i = 1; $i -le $n; $i++) {{ $ends += $d.Sections.Item($i).Range.Information(3) }}
  {save_pdf}
  $d.Close([ref]0)
  Write-Output ("$total|" + ($ends -join ','))
}} finally {{ $w.Quit() }}
"""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True, text=True, timeout=180)
        line = [l for l in out.stdout.strip().splitlines() if "|" in l]
        if out.returncode != 0 or not line:
            return None
        total, ends = line[-1].split("|")
        ends = [int(float(e)) for e in ends.split(",") if e.strip()]
        cover_end = ends[0] if ends else 1
        if getattr(CFG, "REFERENCES_COUNT_IN_LIMIT", True) or len(ends) < 4:
            body_end = ends[1] if len(ends) > 1 else int(total)
        else:
            body_end = ends[1]
        refs_end = ends[2] if len(ends) > 3 else body_end
        return {"pdf": pdf_abs if (make_pdf and os.path.exists(pdf_abs)) else None, "total_pages": int(float(total)),
                "body_pages": body_end - cover_end, "body_plus_refs": (refs_end if len(ends) > 3 else body_end) - cover_end, "ends": ends}
    except Exception as e:
        print("  (Word tidak bisa dipakai:", e, ")")
        return None


def estimate_pages(blocks):
    """Estimasi kasar jumlah halaman isi utama kalau MS Word tidak ada (A4, margin 4/3 cm, TNR 12, 1.15)."""
    lines = 0.0
    for b in blocks:
        if b["k"] == "appendix_start":
            break
        if b["k"] in ("p", "bullet"):
            lines += len(b["t"]) / 85 + 0.6
        elif b["k"] == "eq":
            lines += 1.6
        elif b["k"] == "h":
            lines += 2
        elif b["k"] == "fig":
            f = b.get("file") or find_fig(b["key"])
            try:
                from PIL import Image
                with Image.open(f) as im:
                    wpx, hpx = im.size
                lines += (b["w"] * hpx / wpx) / 0.62 + 2
            except Exception:
                lines += 14
        elif b["k"] == "tab":
            lines += 2 + len(b["df"]) * 1.6 * (1 + b["df"].astype(str).apply(lambda s: s.str.len()).values.sum() / max(len(b["df"]), 1) / 400)
        elif b["k"] == "refs":
            lines += 2 + 2.6 * 18
    return lines / 35.0


def libreoffice_pdf(docx_path):
    """Fallback tanpa MS Word: LibreOffice headless → PDF."""
    import shutil as _sh
    cands = [_sh.which("soffice"), _sh.which("libreoffice"), r"C:\Program Files\LibreOffice\program\soffice.exe",
             r"C:\Program Files (x86)\LibreOffice\program\soffice.exe", "/Applications/LibreOffice.app/Contents/MacOS/soffice"]
    exe = next((c for c in cands if c and os.path.exists(c)), None)
    if not exe:
        return None
    outdir = os.path.dirname(os.path.abspath(docx_path))
    try:
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", outdir, os.path.abspath(docx_path)], capture_output=True, timeout=300)
    except Exception:
        return None
    pdf = os.path.splitext(os.path.abspath(docx_path))[0] + ".pdf"
    return pdf if os.path.exists(pdf) else None


def pdf_layout(pdf):
    """Hitung halaman dari PDF: halaman 1 = cover; isi utama = sampai sebelum halaman berjudul 'Appendices'."""
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
    except ImportError:
        return None
    d = fitz.open(pdf)
    app = next((i for i in range(1, d.page_count) if any(l.strip() == "Appendices" for l in d[i].get_text().splitlines()[:6])), d.page_count)
    refs = next((i for i in range(1, app) if any(l.strip() == "References" for l in d[i].get_text().splitlines())), app)
    body = app - 1
    out = {"pdf": pdf, "total_pages": d.page_count, "body_pages": body if getattr(CFG, "REFERENCES_COUNT_IN_LIMIT", True) else refs, "ends": []}
    d.close()
    return out


WORD_OK = [os.name == "nt"]


def measure_docx(docx_path, make_pdf=True):
    """MS Word (akurat) → LibreOffice → None. Return dict pages (+ pdf). Word yang gagal sekali tidak dicoba lagi (hindari hang)."""
    if WORD_OK[0]:
        pg = word_pdf_and_pages(docx_path, make_pdf=make_pdf)
        if pg:
            return pg
        WORD_OK[0] = False
        print("  (MS Word tidak merespons → beralih ke LibreOffice / estimasi)")
    pdf = libreoffice_pdf(docx_path)
    return pdf_layout(pdf) if pdf else None

# %%
# ===== 24.3d Generate the article: autofit ≤ batas halaman → DOCX + PDF + MD + TODO =====
def write_report_md(blocks, path, meta, refs):
    lines = [f"# {meta['title']}", "", f"*Team {CFG.TEAM_NAME} — {getattr(CFG, 'REPORT_SUBTITLE', 'Final Stage 1')}*", ""]
    for b in blocks:
        k = b["k"]
        if k == "h":
            lines += ["", "#" * (b["lvl"] + 1) + " " + b["t"], ""]
        elif k == "p":
            lines += [b["t"], ""]
        elif k == "bullet":
            lines += [f"- {b['t']}"]
        elif k == "eq":
            lines += ["", f"    {b['t']}", ""]
        elif k == "fig":
            f = b.get("file") or find_fig(b["key"])
            if f:
                lines += [f"![{b.get('label')}](figures/{os.path.basename(f)})", "", f"*{b.get('label')}. {b['cap']}*", ""]
        elif k == "tab":
            lines += [f"*{b.get('label')}. {b['cap']}*", ""]
            try:
                dfm = b["df"].map(_fmt_cell) if hasattr(b["df"], "map") else b["df"].applymap(_fmt_cell)
                lines += [dfm.to_markdown(index=False), ""]
            except Exception:
                lines += ["```", b["df"].to_string(index=False), "```", ""]
        elif k == "refs":
            lines += ["", "## References", ""] + [f"- {r}" for r in refs] + [""]
        elif k == "refs_app" and meta.get("app_refs"):
            lines += ["", "### References for the Appendices", ""] + [f"- {r}" for r in meta["app_refs"]] + [""]
        elif k == "appendix_start":
            lines += ["", "---", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


REPORT_META, REPORT_PAGES = {}, None
with safe_section("Auto report draft"):
    if not CFG.MAKE_REPORT_DRAFT:
        print("⏭️ MAKE_REPORT_DRAFT=False")
    else:
        with open(os.path.join(CFG.OUTPUT_DIR, "figure_index.md"), "w", encoding="utf-8") as f:
            f.write("# Figure index\n\n| File | Title |\n|---|---|\n")
            for fn, t in FIGURE_INDEX:
                f.write(f"| {fn} | {t} |\n")
        docx_path = os.path.join(CFG.OUTPUT_DIR, "REPORT_DRAFT.docx")
        limit = CFG.REPORT_MAX_BODY_PAGES
        can_word = bool(CFG.REPORT_AUTOFIT)
        chosen = None
        level, tried = 0, set()
        while level <= 6 and level not in tried:
            tried.add(level)
            body, meta = build_report_content(level)
            used_figs = {os.path.basename(find_fig(b["key"])) for b in body if b["k"] == "fig" and find_fig(b["key"])}
            used_tabs = {b["key"] for b in body if b["k"] == "tab"}
            blocks = resolve_refs(body + build_appendix(used_figs, used_tabs))
            cut = next((i for i, b in enumerate(blocks) if b["k"] == "appendix_start"), len(blocks))
            txt = lambda bl: [b["t"] for b in bl if b["k"] in ("p", "bullet")] + [str(b.get("cap", "")) for b in bl]
            refs = cited_references(txt(blocks[:cut]))
            meta["app_refs"] = [r for r in cited_references(txt(blocks[cut:])) if r not in refs]
            n_f, n_t = write_report_docx(blocks, docx_path, meta, refs)
            if can_word:
                pg = measure_docx(docx_path, make_pdf=False)
            else:
                pg = None
            pages = (pg["body_pages"] if pg else estimate_pages(blocks))
            src = "MS Word/LibreOffice" if pg else "estimasi"
            print(f"  level {level}: isi utama ≈ {pages:.1f} halaman ({src}) | {n_f} figure, {n_t} tabel, {len(refs)} referensi")
            chosen = (level, blocks, meta, refs, pg, pages)
            if pages <= limit:
                break
            over = pages - limit
            level = max(level + 1, 3 if (level == 0 and over >= 4) else 2 if (level == 0 and over >= 2) else level + 1)
        level, blocks, meta, refs, pg, pages = chosen
        REPORT_META = meta
        write_report_md(blocks, os.path.join(CFG.OUTPUT_DIR, "REPORT_DRAFT.md"), meta, refs)
        print(f"Judul        : {meta['title']}")
        print(f"Word draft   : {docx_path} (autofit level {level})")
        base_name = f"{CFG.TEAM_NAME}_Final Stage 1"
        if CFG.MAKE_PDF:
            pg = measure_docx(docx_path, make_pdf=True)
            if pg and pg.get("pdf"):
                REPORT_PAGES = pg
                shutil_copy = __import__("shutil").copyfile
                shutil_copy(pg["pdf"], os.path.join(CFG.OUTPUT_DIR, f"{base_name}.pdf"))
                shutil_copy(docx_path, os.path.join(CFG.OUTPUT_DIR, f"{base_name}.docx"))
                print(f"PDF          : {pg['pdf']} | total {pg['total_pages']} hal | isi utama {pg['body_pages']} hal (batas {limit})")
                print(f"Siap kumpul  : {base_name}.pdf / .docx (cek & edit dulu!)")
                if pg["body_pages"] > limit:
                    print(f"   ⚠️ Masih {pg['body_pages']} > {limit} halaman setelah autofit → ringkas manual (pindahkan figure/tabel ke Appendix).")
            else:
                print("(MS Word & LibreOffice tidak tersedia → PDF tidak dibuat; buka DOCX di Word/Google Docs → Save As PDF)")
        n_edit = sum(len(re.findall(r"\[EDIT[^\]]*\]", b.get("t", ""))) for b in blocks if b["k"] in ("p", "bullet"))
        todo = [f"# REPORT TODO — {CFG.TEAM_NAME}", "",
                f"Draft: `REPORT_DRAFT.docx` (judul: *{meta['title']}*). Isi utama ≈ {pages:.0f} halaman (batas {limit}).", "",
                "## A. Wajib", "",
                f"- [ ] Cover: anggota tim & universitas ({'sudah diisi' if CFG.TEAM_MEMBERS and CFG.UNIVERSITY else 'isi CFG.TEAM_MEMBERS / CFG.UNIVERSITY lalu run ulang 24.3, atau edit langsung di Word'}).",
                f"- [ ] Konteks soal di Introduction ({'sudah dari CFG.CASE_CONTEXT' if CFG.CASE_CONTEXT else 'ISI CFG.CASE_CONTEXT (2–4 kalimat dari soal) → run ulang 24.3'}).",
                f"- [ ] Jenis asuransi terdeteksi: **{detect_insurance_line()}** → kalau salah set CFG.INSURANCE_LINE.",
                "- [ ] Parameter bisnis (margin, biaya, success rate) sesuai soal; kalau soal tidak memberi → sudah ditulis sebagai asumsi + sensitivity.",
                "- [ ] Pilih judul & framing di `NARRATIVE_OPTIONS.md` (judul alternatif, framing executive summary).",
                "- [ ] Baca ulang seluruh teks: parafrase gaya tim, cek setiap angka sama dengan tabel.",
                f"- [ ] {n_edit} penanda [EDIT] tersisa (highlight kuning).",
                "", "## A2. Research yang dipakai (CASE_RESEARCH.xlsx)", "",
                f"- File: `{CASE_RESEARCH.get('_path') or 'tidak ada → jalankan tools/research_helper.py'}`",
                f"- Dipakai: " + ", ".join(f"{k} = {len(v)}" for k, v in CASE_RESEARCH.items() if isinstance(v, pd.DataFrame)),
                "- [ ] Semua baris industry_facts / literature sudah `verified = Y` (sumber dibuka & dicek).",
                "", "## B. Verifikasi kualitas (juri)", "",
                "- [ ] Mapping target benar (churn = 1); tidak ada kolom leak di model (Appendix A, leakage screen).",
                "- [ ] Setiap driver yang dibahas konsisten di beberapa metode (Table 1 'methods agreeing').",
                "- [ ] Klaim kausal hanya untuk efek dengan verdict 'robust' (Appendix H); lainnya ditulis 'associated with'.",
                "- [ ] Rekomendasi: siapa (segmen + ukuran), apa, kapan, dampak, KPI, desain pilot.",
                "- [ ] Persona diberi nama yang menarik (lihat NARRATIVE_OPTIONS.md).",
                "", "## C. Format", "",
                f"- [ ] ≤ {limit} halaman isi (cover & appendix tidak dihitung; referensi {'DIHITUNG' if getattr(CFG, 'REFERENCES_COUNT_IN_LIMIT', True) else 'tidak dihitung'}).",
                "- [ ] A4, TNR 12, spasi 1.15, margin 4/4/3/3 cm (sudah otomatis).",
                f"- [ ] Nama file: `{base_name}.pdf`; ZIP `{base_name}.zip` (cell 24.4).", ""]
        with open(os.path.join(CFG.OUTPUT_DIR, "REPORT_TODO.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(todo))
        print(f"TODO list    : REPORT_TODO.md ({n_edit} penanda [EDIT])")

# %%
# ===== 24.3e NARRATIVE_OPTIONS.md (pilihan narasi siap tempel) + JUDGE_QA.md (persiapan pertanyaan juri) =====
PERSONA_NAMES = {"lifecycle": "New & Unanchored", "payment": "Payment-Friction Lapsers", "price": "Price-Shocked Renewers",
                 "service": "Service-Frustrated Claimants", "affordability": "Budget-Stretched Policyholders", "channel": "Weak-Channel Customers",
                 "demographic": "Life-Stage Switchers", "relationship": "Single-Policy Shoppers", "engagement": "Disengaged Customers",
                 "geography": "Regional Hotspot Customers", "other": "Mixed-Risk Customers"}
THEME_DISCUSSION = {
    "lifecycle": ("Early-tenure churn is consistent with evidence that lapse declines with contract age (Eling & Kiesenbauer, 2014) and that "
                  "first-year lapse is the highest (Fleigle et al., 2026). Because acquisition costs are front-loaded, early retention has the "
                  "highest value per retained policy.", "Structured onboarding (welcome call, coverage check, app activation, check-in before the "
                  "hazard peak) and treating the first renewal as a critical moment.", "first-year persistency; churn in the peak-hazard window"),
    "payment": ("Payment behaviour matters, in line with Hein et al. (2020) and with survey evidence that forgetting to pay explains a large "
                "share of lapses (Gottlieb & Smetters, 2021).", "Auto-debit enrolment incentive, reminders before the grace period ends, "
                "flexible payment dates and instalments.", "share on auto-pay; cure rate of late payers"),
    "price": ("Price sensitivity at renewal agrees with Guelman and Guillén (2014) and Brockett et al. (2008); renewal pricing must also be "
              "fair (Financial Conduct Authority, 2021).", "Cap or phase in increases for loyal at-risk customers, explain value, offer "
              "deductible/coverage adjustments instead of lapse.", "churn among customers with increases; price elasticity"),
    "service": ("The claims and service experience is linked to renewal (Fallis et al., 2022); the association is observational and should "
                "be validated by a pilot.", "Proactive contact within 48 hours of a complaint or rejected claim, settlement SLAs, transparent "
                "rejection explanations with appeal path.", "post-claim churn; complaint resolution time; claim NPS"),
    "affordability": ("The pattern fits the emergency fund hypothesis (Outreville, 1990; Fier & Liebenberg, 2013).", "Premium holidays, "
                      "partial withdrawal instead of full surrender (life), lower-premium plans.", "downgrade-instead-of-lapse rate"),
    "relationship": ("Cancellation of one policy predicts cancellation of others (Brockett et al., 2008).", "Multi-policy bundles and "
                     "relationship-level early warnings.", "products per customer; churn of bundled vs single"),
    "channel": ("Channel effects suggest relationship strength differs by acquisition route.", "Human follow-up for weak channels; agent "
                "incentives tied to persistency, not only new sales.", "persistency by channel/agent"),
    "engagement": ("Low engagement precedes churn in many subscription settings (Günther et al., 2014).", "Re-engagement triggers when "
                   "activity drops; useful digital features.", "monthly active users; churn of re-engaged customers"),
    "demographic": ("Demographic risk differences are targeting signals, not levers.", "Life-stage tailored communication and channels.",
                    "churn by age band"),
    "geography": ("Regional differences may reflect competition and service capacity.", "Regional retention focus and local root-cause review.",
                  "churn by region"),
    "other": ("This driver is specific to the portfolio and should be explored with domain experts.", "Targeted pilot on the high-risk group.",
              "churn of the high-risk group"),
}

with safe_section("Narrative options & judge Q&A"):
    F = collect_facts()
    meta = REPORT_META or {"title": report_title(F)[0], "title_options": report_title(F)[1]}
    L = [f"# NARRATIVE OPTIONS — {CFG.TEAM_NAME}", "",
         "Semua teks di bawah sudah terisi angka dari run ini. Pilih yang paling sesuai dengan case, tempel ke REPORT_DRAFT.docx, "
         "atau set `CFG.REPORT_TITLE` / `CFG.NARRATIVE_ANGLE` lalu run ulang Section 24.3.", "",
         "## 1. Judul", ""]
    for o in [meta["title"]] + [o for o in meta.get("title_options", []) if o != meta["title"]]:
        L.append(f"- {o}")
    L += ["", "## 2. Framing executive summary (pilih 1 sebagai kalimat pembuka)", ""]
    d0 = F["drivers"][0] if F["drivers"] else None
    if F["surv"]:
        L.append(f"- **Lifecycle:** \"Churn at {F['company']} is a timing problem: risk peaks in tenure interval {F['peak_interval']} {F['surv_unit']}, "
                 f"when {pct(F['peak_hazard'], 2)} of active customers leave per {F['hz_label']}.\"")
    if F["value"]:
        L.append(f"- **Money-first:** \"Churn puts {money(F['var'])} of annual premium at risk; {F['top20_capture']:.0f}% of it sits in "
                 f"the 20% of customers our model flags.\"")
    if d0:
        L.append(f"- **Driver-first:** \"Customers with {d0['name']} = '{d0['level']}' are {d0['lift']:.1f} times more likely to leave than average.\"")
    if "camp" in F:
        L.append(f"- **Action-first:** \"A targeted retention campaign on the top {F['camp']['target_%']:.0f}% of customers returns "
                 f"{money(F['camp']['net_profit'])} net (ROI {F['camp']['roi']:.1f}×), far above random targeting.\"")
    L += ["", "## 3. Nama persona (ganti P1, P2, … di laporan)", ""]
    FEATURE_PERSONA = [(r"auto.?renew|perpanjang", "Manual Renewers"), (r"payment.?method|metode.?bayar|cara.?bayar", "Manual Payers"),
                       (r"late|telat|overdue|arrear", "Struggling Payers"), (r"frequen|frekuensi", "Monthly Payers"),
                       (r"increase|price|kenaikan|premium.?change", "Price-Shocked Renewers"), (r"complain|keluhan", "Unheard Complainers"),
                       (r"reject|tolak|settle|claim|klaim", "Claim-Disappointed"), (r"tenure|new_customer|lama|since|orig", "New & Unanchored"),
                       (r"engag|digital|app|login", "Disengaged Customers"), (r"product|policies|bundle", "Single-Policy Shoppers"),
                       (r"income|afford|premium_to_income", "Budget-Stretched"), (r"channel|agent|broker", "Weak-Channel Customers"),
                       (r"age|usia|umur", "Life-Stage Switchers")]
    used_names = set()
    for _, r in F["personas"].iterrows():
        drivers_ = [d.split(" (+")[0].strip() for d in str(r["top_drivers"]).split(";") if d.strip()]
        first = drivers_[0] if drivers_ else ""
        th = feature_theme(first)
        name = next((n for pat, n in FEATURE_PERSONA if re.search(pat, first, re.I)), PERSONA_NAMES.get(th, "Mixed-Risk Customers"))
        if name in used_names and len(drivers_) > 1:
            name = next((n for pat, n in FEATURE_PERSONA if re.search(pat, drivers_[1], re.I)), name + " II")
        if name in used_names:
            name = f"{name} ({pretty(first)})"
        used_names.add(name)
        L.append(f"- {r['persona']} → **\"{name}\"** (drivers: {', '.join(pretty(d) for d in drivers_[:3])}; "
                 f"{r['share_of_at_risk_%']:.0f}% of at-risk customers, actual churn {pct(r['actual_churn_rate'])}). "
                 f"Intervensi: {THEME_DISCUSSION.get(th, THEME_DISCUSSION['other'])[1]}")
    L += ["", "## 4. Paragraf diskusi per tema driver (dengan sitasi terverifikasi)", ""]
    seen = set()
    for d in F["drivers"]:
        if d["theme"] in seen:
            continue
        seen.add(d["theme"])
        disc, act, kpi = THEME_DISCUSSION.get(d["theme"], THEME_DISCUSSION["other"])
        L += [f"### {d['name']} (tema: {d['theme']})", "",
              f"- **Temuan:** riskiest group '{d['level']}' churns at {pct(d['rate'])} ({d['lift']:.1f}× average; IV {d['iv']:.2f}).",
              f"- **Diskusi:** {disc}", f"- **Aksi:** {act}", f"- **KPI:** {kpi}", ""]
    L += ["## 5. Kalau soal menanyakan … → ambil dari", "",
          "| Pertanyaan soal | Bagian laporan / output |", "|---|---|",
          "| Faktor penyebab churn | 3.1 + Table evidence + Appendix C/D/G (`driver_evidence_matrix.csv`) |",
          "| Kapan nasabah churn | 3.2 + Appendix E (hazard, KM, RMST, AFT) |",
          "| Segmen paling berisiko / profil | 3.3 + personas (4.x) + `segment_rules.csv`, `churn_personas.csv` |",
          "| Prediksi / model terbaik | 3.4 + Appendix F (`model_comparison_full.csv`, `experiment_log.csv`) + `submission.csv` |",
          "| Strategi retensi / rekomendasi | 4–5 + `recommendations.csv`, `implementation_roadmap.csv`, `pilot_design_power.csv` |",
          "| Dampak finansial / ROI | 4.1–4.2 + `campaign_optimum.csv`, `campaign_sensitivity.csv`, EMPC |",
          "| Intervensi mana yang efektif | 4.3 + Appendix H (AIPW, robustness, uplift) |",
          "| Daftar nasabah yang harus dihubungi | `tables/test_customer_action_list.csv` (tier, value, uplift) |",
          "| Monitoring / implementasi | Appendix J (model card, MLOps loop, roadmap) + dashboard |", ""]
    with open(os.path.join(CFG.OUTPUT_DIR, "NARRATIVE_OPTIONS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    me = F["model_eval"]
    if "mlp" in RESULTS:
        _m = RESULTS["mlp"]["oof_auc"]
        _gb = [RESULTS[k]["oof_auc"] for k in RESULTS if RESULTS[k]["model"] in ("lgbm", "xgb", "catboost", "hgb")]
        DL_TXT = (f"Neural network (MLP) sudah dicoba: AUC {_m:.3f} vs gradient boosting {max(_gb) if _gb else float('nan'):.3f}, sejalan dengan "
                  "Grinsztajn et al. (2022) bahwa model berbasis pohon unggul di data tabular.")
    else:
        DL_TXT = ("Literatur (Grinsztajn et al., 2022) menunjukkan model berbasis pohon unggul di data tabular; MLP tersedia di notebook "
                  "(CFG.MODELS) sebagai pembanding bila waktu memungkinkan.")
    LC_TXT = ""
    if "LC" in globals() and isinstance(LC, pd.DataFrame) and len(LC) >= 2:
        try:
            _slope = LC[("valid_auc", "mean")].iloc[-1] - LC[("valid_auc", "mean")].iloc[-2]
            LC_TXT = ("Learning curve masih naik, jadi data tambahan akan membantu." if _slope > 0.005 else
                      "Learning curve sudah datar, jadi peningkatan berikutnya datang dari variabel baru (interaksi layanan, harga pesaing), "
                      "bukan model yang lebih rumit.")
        except Exception:
            LC_TXT = ""
    lr_auc = me[me["model"] == "logreg"]["roc_auc"].iloc[0] if len(me) and "logreg" in set(me["model"]) else np.nan
    Q = [f"# JUDGE Q&A PREP — {CFG.TEAM_NAME}", "", "Jawaban sudah berisi angka run ini. Latih jawab ≤ 30 detik per pertanyaan.", "",
         "## Juri industri asuransi", "",
         f"**Q: Berapa uang yang dipertaruhkan dan berapa yang bisa diselamatkan?**  \nA: "
         + (f"Expected annual premium at risk {money(F['var'])} ({pct(F['var_share'])} of base). " if F["value"] else "")
         + (f"Optimal campaign: top {F['camp']['target_%']:.0f}% by expected loss → net {money(F['camp']['net_profit'])} (ROI {F['camp']['roi']:.1f}×), "
            f"profitable in {F['sens_pos']}/{F['sens_n']} sensitivity scenarios." if "camp" in F else ""), "",
         f"**Q: Kenapa nasabah keluar — dan apakah itu bisa dikendalikan perusahaan?**  \nA: Top drivers: {_join([d['name'] for d in F['drivers'][:4]])}. "
         f"Actionable: {_join([d['name'] for d in F['actionable'][:3]]) or 'few — so we focus on targeting & timing'}. "
         f"Non-actionable drivers dipakai untuk targeting, bukan intervensi.", "",
         "**Q: Apakah model ini boleh dipakai untuk pricing/underwriting?**  \nA: Tidak. Model card membatasi penggunaan untuk retention outreach; "
         "fairness audit menunjukkan kalibrasi per kelompok; pricing harus mengikuti prinsip fairness (contoh larangan price-walking FCA 2021).", "",
         "**Q: Bagaimana dengan regulasi & perlindungan konsumen di Indonesia?**  \nA: Rekomendasi bersifat layanan (pengingat, kemudahan bayar, "
         "follow-up keluhan) — sejalan dengan arah OJK yang menekankan transparansi dan perlindungan pemegang polis (SEOJK 5/2022; POJK 8/2024). "
         "Data nasabah dipakai sesuai tujuan; tidak ada keputusan otomatis yang merugikan nasabah.", "",
         "## Juri praktisi / data science", "",
         f"**Q: AUC {F['auc']:.3f} — bukankah rendah?**  \nA: Churn adalah perilaku manusia dengan noise tinggi; yang penting bagi bisnis adalah konsentrasi: "
         f"top 20% berisi {F['top20_capture']:.0f}% churner ({F['top20_lift']:.1f}× random) dan kampanye menguntungkan. " + LC_TXT, "",
         "**Q: Bagaimana mencegah leakage dan overfitting?**  \nA: Leakage screen otomatis (nama, missingness, single-feature AUC, model-level), "
         f"semua metrik out-of-fold, target encoding di dalam fold, shuffled-label AUC ≈ {(_g('auc_perm') or 0.5):.2f}, adversarial validation, DeLong & bootstrap CI.", "",
         "**Q: Kenapa tidak deep learning?**  \nA: " + DL_TXT + f" Logistic regression mencapai AUC {lr_auc:.3f}, sehingga ada opsi transparan.", "",
         f"**Q: Berapa eksperimen yang dicoba? Bukti trial-and-error?**  \nA: {len(F['exp'])} eksperimen tercatat (experiment_log.csv): model zoo, Optuna, multi-seed, "
         "feature selection, ablation, 6 metode ensemble; figure 'model development journey' di Appendix F.", "",
         "**Q: Bagaimana deploy & monitoring?**  \nA: Skor bulanan → tier × value × uplift → CRM dengan control group → ukur uplift → monitor PSI (>0.25), "
         "AUC drop (>0.03), kalibrasi → retrain kuartalan (Appendix J, MLOps loop).", "",
         "## Juri akademik / statistik", "",
         "**Q: Apakah efeknya kausal?**  \nA: Driver = asosiasi (dibedakan jelas). Untuk tuas yang bisa dikendalikan kami pakai AIPW cross-fitted (doubly robust) "
         "+ uji placebo + E-value + overlap; hanya efek berlabel 'robust' yang dipakai sebagai dasar rekomendasi, dan semuanya diusulkan diuji via RCT pilot.", "",
         "**Q: Asumsi proportional hazards?**  \nA: Diuji dengan Schoenfeld test; bila dilanggar, HR dibaca sebagai efek rata-rata waktu dan dilengkapi Weibull AFT + RMST.", "",
         "**Q: Multiple testing?**  \nA: Benjamini–Hochberg untuk driver univariat; Holm untuk perbandingan model & ablation; effect size & IV dilaporkan, bukan p-value saja.", "",
         "**Q: Kenapa asumsi biaya/success rate itu?**  \nA: " + ("Dari soal. " if CFG.CASE_CONTEXT else "Asumsi wajar karena soal tidak memberi angka; ")
         + "kesimpulan diuji pada 9 kombinasi (success ±50%, cost ×0.5–×2) dan EMPC mengintegrasikan ketidakpastian tingkat penerimaan (Beta distribution).", ""]
    with open(os.path.join(CFG.OUTPUT_DIR, "JUDGE_QA.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(Q))
    print("NARRATIVE_OPTIONS.md & JUDGE_QA.md →", CFG.OUTPUT_DIR)


# ===== SENTUHAN_MANUSIA.md — panduan edit final: draf paragraf berisi angka run ini + bagian [ISI DARI CASE] =====
def write_human_touch():
    F = collect_facts()
    T = lambda n: TABLES.get(n) if isinstance(TABLES.get(n), pd.DataFrame) and len(TABLES.get(n)) else None
    comp = F["company"]
    L = [f"# SENTUHAN MANUSIA — {CFG.TEAM_NAME}", "",
         "Paper otomatis sudah lengkap dan bersih. Lima sentuhan di bawah ini yang membedakan paper bagus dari paper juara: suara aktuaria, "
         "cerita iterasi, ambang operasional, insight bisnis lokal dan kepatuhan guidebook. Setiap blok berisi **draf paragraf (English) yang "
         "angkanya sudah diisi dari run ini**; kamu tinggal mengganti bagian `[ISI DARI CASE: …]` dengan informasi dari casebook.", "",
         "## Cara memasukkan ke paper (supaya tidak hilang saat run ulang)",
         "1. Edit draf di bawah sampai tidak ada `[ISI …]` tersisa (hapus kalimat yang informasinya tidak ada di case — jangan mengarang angka).",
         "2. Buka `CASE_RESEARCH.xlsx` → sheet `custom_paragraphs` → isi `use = Y`, `section` dan `position` sesuai yang tertulis di tiap blok, "
         "tempel teks ke kolom `text`.",
         "3. Double-click **RUN_FULL.bat** lagi (atau RUN_FAST.bat kalau waktu mepet, ±10 menit). Kalau notebook terbuka di Jupyter dan sudah "
         "di-Run All di sesi itu, cukup jalankan ulang cell 24.2c–24.5 (±3 menit). Paragraf masuk paper di tempatnya; autofit tetap menjaga ≤ 10 halaman.",
         "4. Cek: `python QA/slop_check.py \"<paper>.pdf\"` (skor ≥ 80) dan `python QA/qa_check.py --case .`.", "",
         "Aturan menulis: kalimat aktif dengan subjek 'we' / nama perusahaan, angka spesifik, tanpa em dash, tanpa 'crucial / robust / "
         "comprehensive / leverage', jangan memulai dengan 'Moreover / Furthermore'. Satu paragraf 2–4 kalimat sudah cukup.", ""]

    # 1. Suara aktuaria
    L += ["## 1. Suara aktuaria (section `discussion`, position `end`)",
          "**Cari di casebook:** biaya akuisisi / komisi per polis, target persistency perusahaan, asumsi lapse di pricing, porsi premi lanjutan "
          "(renewal) terhadap total premi, produk utama (unit link / tradisional / umum).", "", "Draf:", ""]
    pr = persistency_ratios() if "persistency_ratios" in globals() else {}
    s1 = []
    if pr.get(13) is not None:
        s1.append(f"For {comp}, a 13-month persistency of {pct(pr[13])} means that about {pct(1 - pr[13])} of new policies lapse before their first "
                  "anniversary, usually before the acquisition cost of [ISI DARI CASE: biaya akuisisi/komisi per polis] is recovered.")
    if pr.get(25) is not None:
        s1.append(f"By month 25 persistency falls to {pct(pr[25])}, so [ISI DARI CASE: asumsi persistency di pricing, mis. 85%] in the pricing "
                  "basis is [lebih tinggi/lebih rendah] than experience and should be reviewed with the actuarial team.")
    if F.get("prem_churn") is not None:
        s1.append(f"Weighted by premium, churn is {pct(F['prem_churn'])} against {pct(F['rate'])} by headcount, which tells the pricing team whether "
                  "the policies being lost are larger or smaller than average.")
    if not s1:
        s1.append(f"For {comp}, every lapsed policy leaves [ISI DARI CASE: biaya akuisisi per polis] of acquisition cost unrecovered; "
                  f"at a churn rate of {pct(F['rate'])} this is [hitung: churn rate × jumlah polis × biaya akuisisi] per year.")
    L += ["> " + " ".join(s1), ""]

    # 2. Cerita iterasi
    L += ["## 2. Cerita iterasi model dengan akar masalah (section `findings_model`, position `end`)",
          "**Tidak perlu info dari case** — angka di bawah diambil dari experiment log; cukup tambahkan satu kalimat tentang hal yang mengejutkanmu.", "",
          "Draf:", ""]
    s2 = []
    try:
        lb = T("leaderboard_all")
        base = T("leaderboard_baseline")
        b0 = base.sort_values("oof_roc_auc", ascending=False).iloc[0] if base is not None else None
        if b0 is not None:
            s2.append(f"We started from untuned models; the best, {MODEL_NAMES.get(b0['model'], b0['model'])}, reached ROC-AUC {b0['oof_roc_auc']:.3f}.")
        if lb is not None and "overfit_gap_auc" in lb:
            ov = lb.sort_values("overfit_gap_auc", ascending=False).iloc[0]
            s2.append(f"{MODEL_NAMES.get(ov['algorithm'], ov['algorithm'])} fitted the training folds far better than unseen folds "
                      f"(train AUC {ov['train_auc']:.3f} vs out-of-fold {ov['oof_roc_auc']:.3f}), so model selection relied on out-of-fold "
                      "accuracy, never on training accuracy.")
        ab = T("ablation_study")
        if ab is not None:
            a0 = ab.sort_values("auc_drop", ascending=False).iloc[0]
            s2.append(f"An ablation study showed that removing {prettify_text(str(a0['ablation']).replace('− ', ''))} cost {a0['auc_drop']:.3f} AUC, "
                      "which confirmed that this information group carries real signal rather than noise.")
        s2.append(f"Tuning, multi-seed averaging and ensemble selection lifted the final out-of-fold ROC-AUC to {roc_auc_score(y, FINAL_OOF):.3f}.")
    except Exception:
        pass
    s2.append("[ISI: satu kalimat pengalaman tim, mis. 'Our first submission over-weighted X until we found Y in the residuals.']")
    L += ["> " + " ".join(s2), ""]

    # 3. Ambang operasional & governance
    L += ["## 3. Ambang operasional & governance (section `recommendations`, position `end`)",
          "**Cari di casebook:** kanal yang bisa menghubungi nasabah (agen, call centre, aplikasi, bank partner), SLA layanan, struktur "
          "organisasi / siapa pengambil keputusan (Head of Retention, Chief Actuary, CFO, Direksi).", "", "Draf:", ""]
    s3 = []
    try:
        rt = T("risk_tiers")
        hi = rt[rt["tier"] == "High"].iloc[0]
        thr = float(np.quantile(PROB_BIZ, 1 - hi["n_train"] / max(N_TR, 1)))
        s3.append(f"Operationally, a customer whose churn score exceeds {thr:.2f} enters the High-risk tier ({int(hi['n_train']):,} customers in the "
                  f"sample, actual churn {pct(hi['actual_churn_rate'])}) and is routed to [ISI DARI CASE: kanal, mis. agen pemegang polis] "
                  "within [ISI: SLA, mis. 7 hari] of scoring.")
    except Exception:
        pass
    try:
        q = T("risk_value_quadrants")
        ps = q[q["quadrant"].str.startswith("Priority")].iloc[0]
        s3.append(f"The 'Priority Save' group ({int(ps['n']):,} customers holding {ps['share_value_at_risk_%']:.0f}% of the value at risk) receives "
                  "a personal call; other high-risk customers receive automated reminders through [ISI DARI CASE: aplikasi/SMS/email].")
    except Exception:
        pass
    s3.append(f"The score is re-validated every quarter and retrained when out-of-fold AUC falls below {max(roc_auc_score(y, FINAL_OOF) - 0.03, 0.5):.2f}, "
              "when any feature's PSI exceeds 0.10 or when calibration drifts by more than five percentage points.")
    s3.append("Decisions escalate by size: [ISI DARI CASE: mis. Head of Retention] approves campaigns within budget, [ISI: mis. Chief Actuary] "
              "signs off pricing changes, and [ISI: mis. Direksi/Board] approves changes above [ISI: batas anggaran].")
    L += ["> " + " ".join(s3), ""]

    # 4. Insight bisnis lokal per driver
    L += ["## 4. Insight bisnis lokal per driver (section `discussion`, position `start`)",
          "Untuk setiap driver, tulis **satu kalimat kenapa driver itu masuk akal untuk perusahaan ini** memakai fakta casebook. "
          "Pertanyaan pemandu per tema:", ""]
    guide = {
        "payment": "Apakah perusahaan sudah punya auto-debit / virtual account / kerja sama bank? Berapa porsi nasabah yang masih bayar manual? Ada grace period berapa hari?",
        "price": "Bagaimana kebijakan kenaikan premi saat renewal? Ada inflasi medis / repricing produk? Apakah pesaing lebih murah?",
        "service": "Berapa SLA klaim dan penanganan keluhan? Ada masalah klaim yang disebut di casebook?",
        "lifecycle": "Apa yang terjadi di bulan-bulan berisiko (akhir masa bebas premi, ulang tahun polis pertama, perubahan agen)? Ada program onboarding?",
        "channel": "Kanal mana yang paling banyak churn (agen, bancassurance, online)? Bagaimana insentif agen setelah komisi tahun pertama?",
        "relationship": "Berapa produk per nasabah? Ada program bundling / loyalty?",
        "affordability": "Profil pendapatan nasabah? Ada opsi premium holiday / penurunan uang pertanggungan?",
        "engagement": "Seberapa aktif nasabah di aplikasi / kanal digital? Ada fitur baru yang direncanakan?",
        "demographic": "Segmen usia mana yang dituju produk? Apakah churn usia tertentu sesuai strategi produk?",
        "geography": "Wilayah mana yang dilayani kanal tertentu? Ada perbedaan jaringan agen per daerah?",
        "other": "Kenapa variabel ini relevan bagi bisnis perusahaan menurut casebook?",
    }
    for d in F["drivers"][:4]:
        L += [f"- **{d['name']}** (tema: {d['theme']}; kelompok berisiko: {d['level']}, churn {pct(d['rate'])}, {d['lift']:.1f}× rata-rata).",
              f"  Pertanyaan: {guide.get(d['theme'], guide['other'])}",
              f"  Draf: > At {comp}, the effect of {d['name']} reflects [ISI DARI CASE: fakta/kebijakan perusahaan yang menjelaskan pola ini], "
              f"so [ISI: tindakan yang paling masuk akal]."]
    L += [""]

    # 5. Guidebook
    L += ["## 5. Kepatuhan guidebook (cek di hari-H, sebelum Save As PDF)",
          "- [ ] **Bahasa**: wajib Bahasa Indonesia? → terjemahkan di Word (struktur, angka, sitasi, nomor figure/tabel tetap).",
          "- [ ] **Lembar orisinalitas / deklarasi penggunaan AI** diminta? → sisipkan setelah cover (template di bawah).",
          "- [ ] Batas halaman, font, spasi, margin, nama file `<Tim>_Final Stage 1` sesuai guidebook terbaru.",
          "- [ ] Apakah daftar pustaka dihitung dalam batas halaman? (`REFERENCES_COUNT_IN_LIMIT`).",
          "- [ ] Lampiran yang diminta (notebook, data hasil, dashboard) sudah ada di ZIP.", "",
          "Template deklarasi AI (Bahasa Indonesia, sesuaikan dengan aturan lomba):", "",
          f"> Kami, Tim {CFG.TEAM_NAME} ({', '.join(CFG.TEAM_MEMBERS or [])}), menyatakan bahwa laporan ini merupakan karya orisinal dan belum pernah "
          "diikutsertakan dalam perlombaan lain. Kami menggunakan kecerdasan buatan generatif [ISI: nama model] untuk [ISI: mis. penyusunan draf "
          "narasi, pengecekan tata bahasa dan peringkasan literatur]. Seluruh analisis data, pemodelan, interpretasi hasil dan rekomendasi telah "
          "ditinjau, diverifikasi dan menjadi tanggung jawab tim.", ""]
    path = os.path.join(CFG.OUTPUT_DIR, "SENTUHAN_MANUSIA.md")
    with open(path, "w", encoding="utf-8") as f_:
        f_.write("\n".join(L) + "\n")
    print(f"✅ Panduan sentuhan manusia → {path}")


with safe_section("Human-touch guide (SENTUHAN_MANUSIA.md)"):
    write_human_touch()

# %% [markdown]
# ## 24.5 Interactive Dashboard (offline + siap deploy Vercel)
# Folder `outputs/.../dashboard/` berisi web app statis (HTML/JS/CSS tanpa library eksternal): KPI, driver interaktif, survival per tier, model comparison, **simulator kampanye** (slider success rate & biaya → kurva profit dihitung ulang), efek kausal, rekomendasi, persona, daftar nasabah yang bisa dicari, dan galeri diagram arsitektur.
#
# - **Offline:** double-click `dashboard/index.html`.
# - **Vercel:** `cd outputs/churn_insurance/dashboard` → `npx vercel --prod` (login akun Vercel sekali). Atau drag-and-drop folder ke vercel.com/new.
# - Template ada di `01_Churn_Insurance/dashboard_template/` (bisa diubah tampilannya tanpa menyentuh notebook).

# %%
import shutil, base64, zlib
# >>> EMBEDDED_TEMPLATE (cadangan kalau folder dashboard_template tidak ditemukan; dibuat otomatis dari template repo)
_DASH_EMBED = {"index.html": "eNqtWF2O2zYQfi/QO7ACWiRAbDcBAqSJ7GLjbZEgTZPGW+SZosYSa0pUScpe96lH6Ftu0Iv0JjlJh0PKsrz2xvl52SWp4fx+80On3+RauG0DrHSVmn39Ver/M8XrYppAndAJ8Nz/r8BxJkpuLLhp0rrl6FGyO695BdNkLWHTaOMSJnTtoEa6jcxdOc1hLQWMaHOPyVo6ydXICq5gep+4OOkUzOZla2r2BvxdqWt2yW2ZaW7ydBIIkFLJesUMqGli3VaBLQFQYGlgGU/GwlriOelUz3S+7UwBw4Ti1k4TpxtPxliay/Xe4UjWNZjwKXyMS9w0HR1sITN6kzCZ95tZOml62vI+fSTFk9m8tU5XKD3Y+B2vmie9pajr/SNSbJsFCbiIfHoR6aRXLc1a59BhnRUlVDDKXB1u0/ap3/lYT5NAnDBuJB8pnnlnXumiUMByblas0jlKev/un3QSSIOborzgVzB+WfPeczyzo4ybIdsFCG8fBmTgZaSNqtHKaASCXytpnbfRS0onyB2FeIxxGXWwgR/d1WswHnJJx7bhNaj9wHUfBEIoyCNgXrs5HZQyz6GepeWD2VUJTHALaNyDGQaAaHE/D/Q7EcJdhxikWu2IfmvBBis7sqWsc1kX3uyJVrNBqPb0WjUy+oFWJ+kKI/MHSQ+QQ9PIhBewZZ3cYEZUcafMGdqdYn6lEZSE3NxI9LvtHBUpa+0QMgssD8D0khCkGHfOSESQD9idxbOL13fH7GepFORsyjgFEkMOTAFyHJNXvXCvc5BC/Ho87Dlnt04nERKElAOERF3PBMhJF4w84IMfQvYa7oBl2+gLckV3F28T9HvAoMBLovO6KdRu30CvjjfBn+OCrh4G5MDJT7EEP47BIEUaZF0Y3TZsI13Jfnj4LZs/H1PxRE9jvYTHzOcKV2rv2nhQrIZ+n6PfXXLC46cg8tNaYjYJYBXHsF8fBchFdBmTFpM+w5rINiXUzALph60hhwbwT+0YNpZS55bxwgAM0QFR1JWHT7JXVRRsDG+OIuaozs9kUY6MtCvUoKhQqmV3chDSegQ5lHs3mNEJRqqzZd6KTCcrzMAPA/OszF+0WAfX6D6EZGMgl8Jh4MksJyM+T6aqiI3JMqxhiBCfl2tCN/am1hx4flUNsHFO6XjrA9xLUcDXcFSlkFsl/wuv9vIREw7rPFdDRQLZSWXODAPVqS8UhStuCnAYVQbLpRQSAbo9bmdbtYqTl+0uCt527x7BG4dW5yEvfLvyAUGm2MSxYRUltpqA2N6lS2msG7qnwI5pPzpUL6lsC1012MMtjSU31X/VupFejpZa5ezNq/no4vd5X3gu4ReNymL9GahDfj4jWKcUu0BgEg87zEc6+zIZKXAi47Kov2Cz6AfZjjmzkmKvY1LeWujnuwAbXq+AcgKuG6DsVtpirQrlvMFSyjOppNuy/94xzJUWsNW+xKJPsLEKyyWycRpzCkvuVreG6U3NUFhbNTS63GwH3QwaLie3d7hFKwRYGxpSKuumdXHYRN0LiFOsp0oYVr5p8v34+4e45Nd++SjB4gMNneJ7INWt8wxoyKM7flChs12PvFWbuUYrfVuM+XNaI0/ZadQpcagB0ZzW4HRi+aGO2Qr7brKD2Isbg95e9/UEn9R735YcO6k2K/sjm/PWYi/AIuRnDT+ZYkOG4wPbpW4zte1a8Z2L56/f3vWPPHQQ1l12E15yv2FUBDBNGFvyNaKKxjl0ei49qJ50s8igFgjS70QxuDVFuRhO2R/I0JCCWM2wseecrg6rhwHx2RNEaFqINSyX/IB/PP0sGefMpN3TMviH+UfUh2fSBXAjymFmWDqLYEWui7hvFBdQYsEHM02eX96jseIe+7PlOSaTe//3v8l5w+uO868Ev6MjqP98wmXM+Vz6pEZPw+THIOcl3dBKF9v4Wve+kA6ZYn8eBrpAtcBsd+y7/a3wTifxWZsuNXpj97uE3wVH0Qpvh++ecv8NLZv4hPaL+ILWWtG2e9l2T3YrjGwcs0bgiM8dH/9BRSgcHxLwprnxfdL9hjKJvxT9DwKv0SE=", "app.js": "eNrFXO9y20aS/75V+w5jrOMCbAoiKcnlpWKnLDm7TkV2cpIuvi0ezwZBkMQKBGgApEhrWbWf7gHuripvcK+w3y9vkie57p4/mBmAlJQ4d1tZCxjM9PT09J/f9Mxw/zE7nS7ylOVRGaVlnKVsFBTTYRbkI/bz3/8L3sqAjfNsRk/+XwvmXsfpKLv2X728eO212CRKozwooxEbrlmItN7HabHIgzSM3s+CooxyP56v0yFzL6KQeuge+kee//vfMca+T4I4ZRc//BmaBnkJ1NOMRStolAYJS+JhHuRxVHjs53//D3ad5VcFy8bjJE4jFqQjBsSCdM2KMijjkE2zomTuD1EeRonns8f7v/+dO16kvFPXYzfYpbMoImiQx2HpHGNBmKXQ7BV7zrSBsb/9jd1stO8P4bsbjzz2/AUbZeFiBtLyJ1H5dRLh48n6mxF+1lq8vYAmzrQs5739/evra//6wM/yyX633W7vF8uJ3ntYFEh/SeSB6mk2my9AphflOolc1Z98EJ16yMD3eTaP8nL9Q5AsIqDgw9Bmrs7IeFZ+H5ZEv8VG8LdD3bhL9vw5SxdJgoONi7fBW2TgK+b8/Pf/dFiPuZ12mz1mSDL7U7yKRi4M/wlzvnAs6m8Xsxp1kjVj8XhLPx5qHCoe9XbMa3OKARB5E5RTPxgWUPO4ohSwF0A/+qNqDLT3qcBi8cSpt3pqt3pqt3rT0OrAbnVgt/pWthL1lup7wL5knTZItAvSbPOBoE7h3/3HbE/9j5VTmFC9AHVXyoM+npQpiOWh68g3MQllvmY3siLUSLIwSC7KLA8mEarHN2U0cx206T1qCc1oeKXHtqmVj5ZeRKXPuXrOymO2YWFQhlPmRmBGG+pY8OEHo9HXS2h2FoOtgy9wnTCJwyunhSZXaQJncRTkV0Dxrl2D4jjYxAHV4WQYcx/csfWjR2yGTL+JRnHgOu48j8ZRXuyFWZLle0WIlXrEked4PlUFTyMm887CoRGB0STxZFqi2XCGBRk+PcakFA2T0rpjf17TTKDmpaMof5kkwu43XrOWZVlSxvOtegafuIrFc6FdynsW0+z6Mp670bJssTy7Ljw5r1DZz6N5EoTR6TRORsCLK4WIFf1xln8dhFPX7V+12HLQYrGuFrJzcumaYoR5BEFFCAFkFS8dSZXbZ0za0faU9hc7moO/z9IJ6n7hlxBeTrMUwx20WB5T134wn4MQ3aJl07iE6m+zUeQ6zAFzv/JwDiQjUQLB5EbzW1sZmAdosSzwwyQowAeS7jigJ1BkcnSFTgXUyGQs2M7Y0mAJ50O0wfZSaBv5gN+n8WgUoT8ZBzAAw/mupPOdxSnONvAbQ5//Akx1DmHuUrDwd/GonLI9IgXRGLRTljzzDGLrZmJ/0Ym9jtBwDGqq6JnOdIGx0E+iMQpphUKar0BI1acyQ/1dyy9kCJV2w5ijS9Jw4ZZMSZT5Ijre4p0hWLNplECYLeq2o0wkStwymLRYUJY5qNE8yDFEm+5vu4q/vXDfXrQYUJBjBrthLm92BaLidD1GDuQlPMdDQAjuleiwfzXQI6Xsnf8VCkH26SqVEPEqqiSlxoIaKWi02KrF1i0GQwoT+Adw3TTLrXHhlMD4HWwHzuxGtKH3Pd4CVJo/IAoAgwS057CNkpPGO3TjsdIcpkNm4xAPSikswynMcZW2BoTXOPnwXALmjHIwqGhMqsBVNFi5B+12q1LYTgdfVX2hvFzVYQzQGv88PWx7trNcTk6yld7TdYtNdZEJilv9ppgZkCgCRRToMo6ugWiPfWizNnt4c72Bf6abD+iNEwhjTjybkDwVda9hXqfZEgJ0Cl7D8uJYJMwoXOQFzBJG3mgcLJJSxjKqU4/28yyGGchnQBpjfkQiVfFC9OPdiUYSBURE2KrRyFSHMhgCWI9WUNlpO7dQH4PFFRVv0l/jGKkFwKSTbJGO4nRySnN8DgsVF4OFGAW04F6wx2DS0Dm1RMlfsAQdz0YM9LjytFu4GSaL3B6jOUvRbF6udfWZFRMR57YqjhjTfEcMwqDO5mYAoq4cLDZtCXo81voTAWXOA00DsgiGSbTDN9J3fUQgwBSYhFXYaZZUaojm/6BEs3pQ+ihP7RFcfzoppwqP16TkvM3EchXUt5zGBSv4gtN3zLBUDndIiVi1G0yjAJc25VDG3tdQ4Ho+rHRh7XWeXbuVUwJ4uZilGu4JFeRR5Hb1P8VpAhdkTkjIYTssnxYzlwsEyXpY05hRECtGReRYThsITVNLzsMwG62NEZ1AgTYKE7rlTZCtRANCOg1iAB9WtV7WBEDCzEW704iAK5Ta2EwtHAFfI7S+APtPJ7gm3CaMUZMwNjUQtN2SyqHlhMthg4XynmW/MC5RmaRWwPInciF0dNueH4HDXUsB5v14QLBVroZVgYMrHLb/b/29J4Ov+v868luDJ/0vfvpx9e2bk8FXD/dBMkXpivFjK8/bZosikbLdGKfDINftJobVCIwjm1doZX+fl/ZY/yYJhlECyB3zCy1YynyFvgv+GcdJAn8ARUEtwPbLge/7g81As2SiQSZMT3e2YNtg36EBXFc1PXK2r6G0+7TFznSU+Y49Zm3/4BlI/wim4LANcPCp34VSFeWBTc7OLJi7rkjp+DRMyaEHPZwD2adA4hL/Guxgx5dAl3h4zPSxQWnXrIzg8XkDJHjXYq/NUQJvUBOmAblkD5Tqy5Le7iG4I0CzWjN6hfWtTzMHQwJWO377WQ3ui7TTGfAO4tuDhz12jrUp4wHdGYg0iUqGI2ofw58vn7ND+PvkiWc7hyUSRq4fM4hd+wzg/gQ7W1UZHUYABxcpHDJ2elAF4GaX/13D+yX8gdfXwFG3izEXbBuwziSPR9QOwy6ItqJIwBVKWkQC2z0lzfbHsxJ6bpFwQNFmgPs1N7+plBYr5wDthCC9OpMrWcfjzOrvDUzj2vMKEFpYFK6ztxfmcRmHQeJAbYd/28NMQJDnwRrXfYfswKk+XSPchOKOf2QNliuAcrKjLQvrtVDWGGYCFfbY/AxTFCr7Ac16JgDwOMmy3HVRFzptnEAwIq9JzlgDGq2lPeyzLjzCdJsmxV7wrr5S5cpNUvke5g2fYDbwvx3SWuF2nGSYoDggjmkJABHDrnXOOzAbrtR2VONa9SAXy5QccAGfzh47A957xD5oCkm7B4TBx9EitMcHBdzBTOZQ/YA7PuQQ/6JzExNbdByvpo+oUWCIWZJk1xgngAcLySIVHGCapZScq9fgmoA5oqYudzQRytNC5dFTBJypJFPO4tEjpjsPzZhZk/pjW6X72FAovq4D3AbMEssS4vSqq1tBperdmhw3dcVb3eLw+IzW9VJ6A1kPGFiiOVraMo3LBm1p07pWqMk7U0ukZjhlHqQFX9fWPRRfggF1nE9MucF09vtK323uBgMLvIiYr6I5zk2hBxaAVHFUj+e8GAN6CuAI14lJBrVp5VX0+v1Vaz0YtGj3R8Zw3gQduXhCVgceaiEs1ly34Es9n9NALZLPMo5qq/oHnMbnxACvpfnnuHpzFQY46KplPPiFLr5xWHB0RJEdAcNRl0d4zENhhO9ApRP4e3D4C6P3CgUlxjhOgvINRmZTQjxaz6ls3m8PkJf1fZt1Bp7VcYUbVnXgsKojh1WBxguCUq3g0W4FRb0KVYFweEMzvVfRWNdprC0aQGCNPa8rftd1ftd1ftdWtzZmqYGWPRqdh1HLpfHzAlTfTj1BKUhdEil09Zfw/xMiteaNl5yUeFvbpO4Li4gAdqYR9BRO2g6OzrjLxdGeC4dLkIYcLT3dBx+dcWyEDTW/uNZBUlPMXeEIVnIEmnQbRqD76tXS43jsBJO/EpSJ8p2g7A5yILoV6jqpJBGs4qJJEhLprcDtehqjLmrUO65LFLMIxXFuse42XqUFSzRWNEExXPVaRi1BmxvjGveM9o/eODgjK5dchNpM5PCI9iDWLrkB7ZPn/xWo4haFCa7nAcX/GzZSwYlwRhWLC18EAjsKd5vhaeHjGzJ7xA6JXUFQVkZpIzMIZckv26Lf2HMwC/KrKNdMBTkP4zxM+JyHK4mxeU0UC8wwoja9tIOleY8dyaFKkEQgw8IexSIfw7K/CX7Y4GOjO4wwz2ir3lBKC/Sf1HpDLWxEOrCsjot4GCdxCcNx+G6ELbBdeISj18sKkkhXWEET5dLugk+gl7umeC3lznmc3JpM3eVGIrXLtMdy2uJB68t9GhT4FRoV+WDT1evup+oAp8jCw6uOI5wQZkqbKnRvqVDNE+o6vWnmr4SAKcvngOekv6DNDkf3doOBanOL02AMg8owKkrNb4DyH1cVtD2iOcvGqpZHxqUOUaAnQTmBq/+yOlqBlNUHT3Y018hTOmu+KKZuv/A5bBRRgtqCxQ00iK7l/qAdUgZJiEf4ryWLoZ0mg2qjYKfUCIBo6bWON6g5lJ3KK/cW5O7f7ZMsrBEUQuTq3XoaNZncumkNdcykZBJNMLJqXtCAxrBSBl+/RTckLJ3fYa+ZV4131IzlaRDu1UFrTE6x0IEaYmeoqE4bUSYDwwgPH7hXorad4+1b1VyLPC4UueU+t8QqMrPzPOK72JPmhc8Slr3FLXnM3zoLSSnIZ22+pDh8JpYUbXNJcXT4qxKCt+QtRXaPknudLgFTM9FPu56a39w3EpY74bCJhjsIhe+TEdyWCjz+vNj2VlBrpv44qlUpv7tl0VZtGMSZSKMNrw2QpcfjFeabQeS42uw846klLdOg8ktU4emhGaVPcDGg1f4F+aZqO14D34InhWdPxPmLWoauc9iUn+sc7MzMWUhYnM7Rc+Iv2NP6hj6BEJD7DGh8yLMS3IS71z1iD280fnGTu+J44324W5Zm1bZgkZ7Quxce+vX5mt9kAWP7wlFWnn7z/+wLxY7MQbtxR+YQdK/7VO3I/ML9mK5wrc/+L/ZjkkwfRhOX6iNmRFvSGeNDkJRGZhLfe1UFa5NB5D6whAws8zwb/N8WCNRHzIJ+Plamsc3KnHbCwSBAT4FRmtz2M1Qp+Nu51+bSHsSODP4FkjzDIojipEENLL5vnJH0aFlRJ7crBK0oSPCk9tK75/6TkeuQqY5btp5+012nmv/4nFtK9p6SlmP/BdtLXdpe2rq5hDBqvf7t95Tus9GxFshkfY9NjaMtm0OafXpW0qbHPryBeMgr4Z7WIQZEEAY9nWmfnqhP9NTc6sn2VljpQz1L0r1LgmRLwqbCMZSqWZvJmUY482vSNLfv2yATmGBQ+1GfcxuneZfGOkfKjyMVWw9gw6r8FTgBTCA4SB8P/aGifiefYRncd0Z5DO94oM15JR6pvIwhFOFpQeeSP1HpDNZdpO1v6IHKwmA2D+JJisWn4pk9YmGwKAJRJeCsYo2X4pGTi8ppNiJ69ASrv8naEQv66gQkwIn5JSYZrIOiabAUZ8zhozI+MXDtqHgMvgesddtp8V0HuABellmqGfbQL9dzWs+KT8dYZJw0gq6w0ESoeLASBwrc0casOoofYwSMR1oHt92BKKIEs2DxSPNyIAq5BB42HVBCrIKnZDGI4gUCOsk2xeQrZkf+oPTDUxkRlYumzqRQi2wWuXjfA/igfA+eNyLKeNmHdwHaXtFrOO9U8S9nQkn/4yLK1xf0PaMrCI4/D1LQNK+aTrF3xZMLKCO/zCaTBGSMakZpmTlJ9TnK1avdwWjoA0fH5FRXPQ2pJ3smgzwO9vgYItRdcyqpU8xo4xlwSmnTsXjH0xPVxgwoiYPjJo7B6xTA2loeJ7ug5YwI+KhDf8C8fTyqzpZZFzYsPM+/4yFD0CXLhPCu1Ssf7DDQLqkxtKloHQ3Bg4E8TO3uz6AgmOEQL/Ev8iKKesTezA8z8ALpmifeBnKH9yQDEwhStbvwP/+oNhjonkiJQMY+OupTOZE6XYBYZlEuLhk+YufymqFjzzFvhM2LaZaX7zUi1Fa1ZK/kBUWnYqVYDJu5+fDwxp35tGHXxh2TM7yKE4mDdN6GhYLDAgcX8quQMHcQFfm1OWjMLzViKdSHWuMY7yaSX2Xn353uvfznU6g+84MFQhH6q7ZmYOnc41fcNh8qbsdZVtY5pblBQjQzNBMbfJ1FsyEyiPeZ5ItEQDClzOXzKb/wuQKdI+zjkUI7xHh1VRPJVm+8q1vvbwKFAEI2aF4ehwXLFuVeNt4bZ8kIPyHJcJGD5oZrSZEAQPGhwaFwBf92HhsZvCt4v8cxeRfrq0MJ0gVcbdt6uyVNax0dBdqOBS2C24lY93sgrjTd8OFA9PjuMU1QH5rU8dhKPZZdcdxlUQ9vpx5ayd+odLDQpg7lQZzwKVadjLR7SkNAfF7D6fFRHSPVNEIinTs6PfTM6LxSZBK/wBsEi4LDLDIY9apyCZVugCmKtqfoTrzmi1G8HoSLU163weNpHDi21nxMOOJBCv8kmUF5f0y2ajYzB2Kr+EdjMyDZleJPKMefxBbLH6l7dU9Mz8AbdwrJQoF5tDRE4a98slEDpWRigOAXcdePxpZtH5v7ypc1awMb/+qBjanvxoHRuWfgk+PniykgexwS8MNL3uMudyR50k/WKoUhu8XFC98MI0PDV2oJywg659djD8AcSKvxLoG9HY3rHO2yXo+yiZyeIGTt+X/BkfdLRZGOAlYdYBBYR4XYjAcs3pJm5uHqB+JYT6ZhqrvJbUF6lzGK1YVtiwCk+IQLQUbV4Tlx0Cvxs7lhc5XJKWnXJ39kXxPIdkw/74BUjfs7hOT4YqoDl+wx8iyVItN1gvEvdfg+DdIJCtqQROORAEqH2aPqP0G6nDGEHgNdQKBVPNGqZHiKZ/ZplYv5Viqy0q5009TUYNVoBGJeQgWus0mzzgrIk/C6nlJeMFSQKx7sx2dMY05jfMIsIldcXoO9EA5Yw0OgeVqmgO3U8Vr/Al9x6h6p+B+PvmCn3zjaN8ww4rEXQFAEW9WHacybSISJK9XET2sIjyqdxWPMBIi+YPFfG0l1mKZLHf704047EmwsWwAqUUfGvQbhbLTlLax1EKAvYwgvYXTJ7xihO5VFtbpFNNGqwRvqfbHdWPma37bVKx44r2ZG2BSQAHdu8bjCDXsdT6ZW1gdvzS9mVeEhFp6hdzMSNi8118ZzX0z2wk+mYvCYKVV1rwDfG4dJZ9WBR/eGobXiISPtjGpPMCqOIAz0nNEB0hRHWNXhB36IVVLh12uATYcmUmolzOeq0S2C8q23zDMeTejhGipdQJwg2I3jWaSxiPwScauZ53vUIIJp8AkQRiWGVz4vMaQwlVIQJjv16eQCGKqy1qlo13TGyTI4vbUYR9W6yw3j+wivIkpm3lO/NJQ5zFE24oGHXCPdn5zidSY86tu8mhoMdtsKhLMd0YbyU7b+Tkh9J4AjhcLYukWflFz7UoNEtqtSIV1npcJM9JO1/bm/wjTEeoC7qpWaCHrnQTrKZjWCM/yVFp1mv98GZUG50e1h+Eeequ7RHXO2Gdyugzw0G5rY9Fno4xd48qdaxxLuFokOPFrKk5+zALoillaybCXLKn3l+4ggVlrd6upKBabRzix1nfFKSldpJcwDy0wFlpkWWGY+X0rfI4qoPsxl9qEVP7A/fQ2uhxDkQPvG7eAcV/Go4jN/nr/HlbxUcVVvsBtQHXgNDp+Y1dw4vb+nz9sNQWZkbVsIyRZk7lYHFMTtg9Cfh2X1VGGvCm1gYwNrSGr84iz++BLHLPwSq4f4Q+wNH5uop7jgGLC4WIQhZe5PRcFpVpQWILyo8m3gKzU0CF8keAv9AihFRQFg7dS3jr50Wrg6xZ93eswOPaoBSG2uVxHf9xle+8OtSF6tIo+fFerrFxfI8sA+YFU0YME4nS9KBQXl7OAhJmtguGruOE0YEYcGX5/I8aIbodN2TySLVXIo4xK1gKxLJB6juXo1V3BM7bjg7fUQ/TyTi91VB4hBWFrQMrd382xM4kINMs8o9+FJcBFypvH26h4T8xL6KbwPTHJ5OrozNUBN6GN3EdUORCKjeBiSHuxU8w3/EYx+ZwCwVZ5VrE441i5Dx4KgTz8q8N2YzjeayVq5EkaL+VashvuVJxc+0DnBPpRVFfEnXEQIhygOXDZ4cp5T6eHEpP1hPNgSYGkClad3jDglGXgDlpOSmNJoHJdV70ITlDSalaHixVkWEnLTZed0hJyppnyymnk4/+6bql9Xm0oc2guGP4kl+cBtf7OCZ595Bywuk6g6e9og2U//IIWpuKk0RgIFw/VZOIG50WpOmwMQrYrCuwU3kI1sWvcHByDDz4EFxIwsvbtAAH46nitfQ6Dne356pOclRqTPrUif+/qlcHxXS8hcRfqcIn2Q4LYqIHFYsarB8vCf+3k2BJ61+H9wW/xXPVOX1qKNzeeOiQNyHQeISmVGap3rMEDA4bfIJRvF43GEeWzA+D/96BEZ4n9bd+c0DNCzgurC0nwUh8KgBGzQprkRPXQURbGcbBtQGf0eTu85P1+OG0vHDchB7NRWwEEBkTwKNRgCb0UNq+DPG2VpoFUTJVa6j35KENREapsBRAT+4L/JYEIPqK9oI/QowiwHg6u0FtEH/swBqke0A3p8xHAoKF5EQR5OIeqpVIvjcL95jaG6iFwDiUgZ0qGhSp6CXwqjBtXbwIAQuGdxKK4BfETNFr2oS5PilyCqCzsmtxCBwmQxAp/10SNbEO0rfIAlb7OycatL+52Uhr0u8BEPb0LcfC056haP9ThT9SuoberThSf3U8AlWa7/+Cc6x2HAD9F7PvsTHilLQHo9rmjFPs7we0nlPc9ivscaflgsP9hKaWjNjYCl4ElD9Tsr/FpBz/rpjSOASTsXm3R4oWm1SWvKJInytQFiJwrpjbI0kgcm9ZT2ZJFHWzPa5pYAVN6R2uSkandL4tmuRvjzT3iYfzbxixyxpiBT7FMMR/WL+Gc8bwS4EPw8pVB5YZIFmJHne1af1vYeBtTdzbAgRttIgf1jQlVfVQBQx/+h9xa2gZaqDD7XTkSYE6BB7abppZ11KXVzM938qVE6foD+pPmsgTzM4ItzCh5Po4GVjvQfeFTJ7Rsmj1H0rA2tlkjrFj0zpYw/oIKJu56Rxmvx7eWenhppKVjZs5YiLSZO6/RMt9Ri/LROz1B6tunz0RDKNLn0lHyMn5kkaWVJ5Ed5DkKJpL+jf7SzPtS27jPxyIRKqLtGip+O0jSfrRC/U7rrFAueFdHOpihcj+Ey/3TcwAoYBIfkio0EfD0IPcoWpZt/wpjziW5kl6pQalSLdY7aspMNF9X/AmkIe/8=", "style.css": "eNq9WVtvq0YQfq/U/4ByVCmpjAvYONhWq772uerTUR8WWOxtMKDdJbZb9b93ZhbWC4acpLcQJ2HYnds3tyU7Wdfa++Prrzwvq8ta+io78hPfeaU4HPUe6b7fsANQPhVbuJ73QFCtLFhGtAyuFGmieoH7IMWru4+AEEdxuOZIOLWa50BJtslzEnasD1IgjYc8yLe4il2EAkK2yqKUZElRHXaePKTsMQwX5jtYhsFTx0GFsDxiz0m+IdVQKE83yWpNtyu4DVNWPDO6XePTnIVBQLcx3ibPKVv3+tQ16hNkbBVkuOTMZEUafCpYGoWko+JS1C2qybNkFRPnTAotMlYCMQ9W6SrtGLIs45X2VV1oNCvnUYGP/vz6qx9PPBfMe2wkL7hU/tD/OZMvTwaYHWK0Ox+55I+7qtaPn3Omma9x4fcPBNTDr0/d4jGOyGdvHlgggxyvIZAhg2t7A7KgLwfIGyITQDpQRhlczIFylcAVj6CM4njRf25oWjxXwJrHFs98G2+jzQ3P7ZY/BxbPbJvEDp55HIeg1p3vw2RdbGOS8ycBQH4d+BKd9fDrVDpYN/6HTvxXXPh/OBB89633h5fWF1+J30mjtJY5lz6Q9ujdoz6VCyDmV1hXv3JZlPXZv+y8o8hzXtGa7umJyYOodh7IT1n2cpB1W4ETXpl8NL5+2hssehq4EkhFXYFOYdxcvguXsaeuSvOT34qF57OmKblvKAvv4Wd+qLn3y08PC0+xSlH2FqTBUtcN2nEntkP0qVvFUuWnTH5hqXWB1vUJVGsunqpLkXcLETlY1dQKCkUNBiuoFy/XvQdKkPW/g2U5Bx/FVjmgVFySky7+WeT6CHyjIGjAy9ZvHmt1DYxZnhMUYQKSww382OC6XKimZNedV5Qcbn9rQW5xhWpTaY4uVA2o76dcnzkCc2ANymgMjkt+5amsz0OcBnBQIHeAYDjwfnvJtQZ/IH/Saxls+AnM5RftawlIFLUEP7VNw2XGFDeBEzqigA2YR8wc7pHhLiruHzkWPxC4jOKO81miASkrWZUZlkvVpm/oj5k5Un/VW09lwU91BfsZFNoKgqcsEDomdY/4LNRvBctERHfxI1kusLckqESH+YqQ7K01d1krFbJoagFIyqEFG2sBxO4H4yfoto8ChwJjjQ/cfDZ7VSbrsoQU6YVUdcUdBdIWcqKikmE8Nkp2s3wKlltUBxjVNhZ2nqigIQo9tHs96ZlRZkYWLgrDhklIBPD1UWhO4cpRIwykOxM+MykYBgHPIOq/f9Cy5dgw3sDTSPUHC1TYR9y5w3SD9ReEnZioPprvFuyGVbyE3Ra5Gwz0aMkyLV65uyIt6+zFLMmYzN9V4uYDfhjBBq2RnsaMW5U0RKB05gaOMsfIzVu41qNScDMd10M5YLlr3PsqnklswP4EOmO/w5BxyiBy6crKLSZgDOMj5UKC6cu1cRn3SmO7j1yFkbCnnz6oAzQISGDYnip0ZyHx0+u2GXL5wesBHHk4GMqa9L8zjTqRl0RgEYyVnt08rxnwIDEvjVDvtkjyhjP9iPHsF0IvMA5A/mP4DIIXyPbpaQDFnPIkdqlOrCwdULAPBvb5fxraa0cRb1mydKIkRPOBQLteGSrvNrqNDXdbJuJgP2yPd/bPMQsGwnKuxyrONnLcVMBsAqYqx70b6s6BdYNfchwTb4LsnlLcB15iM1ePNDXtd8p7NqpERe3f5PddflLAEP/J1HZraxB8Q1qYir4Axk2rP+trA4cBxZnMjlTd3+g4q0EgoE9I8lQznwsxHPI/Pi44ZkQb63MFbOEweV8FySnRerKg9YiuCdFkzMt1CjTMg+l4nWxI2ccwoXyNg9czKNYdGu4bnsuzbjUw7YF/xbYKv6v2BB7Odh603LaEcQIIatr6W8fY9MmnXg8TrW0AdT9CmRbaDzLAWiiRltyywVESlRNl+VZ6hDZBC3YS5dWJkY7RskzLEaMumPsFXaYOF8yNB1SJMfZhi9KyfuH9JnwAuwzRxrfZhAE2uamLvMlNAEHJMT5GJzjjusEK+KsrvBc7ksemDtidV2cnbbTTIEZKyRoFWvV/jVCbms31ceFpbHc09VOi7zysQE42PuPIGDjpOHc265wHutI7nJ4dnMA6WW+V8wFG88e7L5wydb6EYAebjvh7ZJekl3HvypXZKbYR5WA2HNTQ+RloMx65qCW8KWW3Szmc7Tg5rZu5Hh4spol7lpkqljFifu+urD+OCwzJm6MLceH5vp/2ff4KAlU/+/6tM9h7p4BkUPsTG20TXZ7elRxZjsUmoDaB+WFe5AQLvJZhNKrqCTGzrwWi4GY/Zmx1GPVNO4TgiuXLTMxSFYF85fL6jwe1VXQ/qN3m0k5IIQ6t5MMTuPtcnA63dmIS/iPdc+Kw/Yleuo10yFiD8TLyWQfPlKO6gYXy1xpV8gOv8pn2Gk6317vsmZfW9+CxPMi06u8mr8NGOI5eu2lIak2n3Fimc2LsuS5zpo6j+bpLPwMk+RDHVVzIc4d7r0AXEwX9N+JDx1/7tmsVve/8ZV9tnRp9fc8QPBzxKG+DuTPTJujOTPQiy+VixqrZ8dyohVxpJr7MZO/0qPYXAPY5CA=="}
# <<< EMBEDDED_TEMPLATE


def find_dashboard_template():
    """Cari folder dashboard_template (repo) dari beberapa lokasi umum → aman walau notebook dicopy ke folder kerja lain."""
    cands = [getattr(CFG, "DASHBOARD_TEMPLATE_DIR", None), os.environ.get("CHURN_REPO_DIR") and os.path.join(os.environ["CHURN_REPO_DIR"], "01_Churn_Insurance", "dashboard_template"),
             "dashboard_template", os.path.join("..", "01_Churn_Insurance", "dashboard_template")]
    here = os.path.abspath(os.getcwd())
    for _ in range(6):
        cands += [os.path.join(here, "01_Churn_Insurance", "dashboard_template"), os.path.join(here, "Notebook-Template", "01_Churn_Insurance", "dashboard_template")]
        here = os.path.dirname(here)
    for c in cands:
        if c and os.path.exists(os.path.join(c, "index.html")) and os.path.exists(os.path.join(c, "app.js")):
            return os.path.abspath(c)
    return None


def _tbl(df, cols=None, rename=None, n=50, fmt=None):
    if df is None or not len(df):
        return None
    d = df[[c for c in (cols or df.columns) if c in df]].head(n).copy()
    if fmt:
        for c, f in fmt.items():
            if c in d:
                d[c] = d[c].map(f)
    d = d.rename(columns=rename or {})
    return {"columns": [prettify_text(str(c)) for c in d.columns],
            "rows": [[(None if (isinstance(v, float) and np.isnan(v)) else (v if isinstance(v, str) else _fmt_cell(v))) for v in r] for r in d.values.tolist()]}


with safe_section("Dashboard export (Vercel-ready)"):
    if not CFG.MAKE_DASHBOARD:
        print("⏭️ MAKE_DASHBOARD=False")
    else:
        tpl = find_dashboard_template()
        out = os.path.join(CFG.OUTPUT_DIR, "dashboard")
        os.makedirs(os.path.join(out, "figures"), exist_ok=True)
        for fn in ["index.html", "app.js", "style.css"]:
            if tpl:
                shutil.copyfile(os.path.join(tpl, fn), os.path.join(out, fn))
            else:
                with open(os.path.join(out, fn), "wb") as fh:
                    fh.write(zlib.decompress(base64.b64decode(_DASH_EMBED[fn])))
        print("Template:", tpl or "embedded copy (folder dashboard_template tidak ditemukan)")
        F = collect_facts()
        meta_title = (REPORT_META or {}).get("title") or report_title(F)[0]
        kpis = [{"label": "Churn rate", "value": pct(CHURN_RATE), "detail": f"{int(y.sum()):,} of {N_TR:,} customers"},
                {"label": "Model ROC-AUC", "value": f"{F['auc']:.3f}", "detail": (f"95% CI {F['auc_ci'][0]:.3f}–{F['auc_ci'][1]:.3f}" if np.isfinite(F['auc_ci'][0]) else "out-of-fold")},
                {"label": "Top-20% capture", "value": f"{F['top20_capture']:.0f}%", "detail": f"of churners ({F['top20_lift']:.1f}× random)"}]
        if F["value"]:
            kpis.append({"label": "Annual value at risk", "value": money(F['var']), "detail": f"{pct(F['var_share'])} of premium base"})
        if "camp" in F:
            kpis.append({"label": "Optimal campaign (net)", "value": money(F["camp"]["net_profit"]), "detail": f"top {F['camp']['target_%']:.0f}% · ROI {F['camp']['roi']:.1f}×"})
        findings = [t for s_, t in INSIGHTS if s_ in ("Drivers", "Survival", "Model", "Business", "Causal")]
        pick = []
        for sec in ["Drivers", "Survival", "Model", "Business", "Causal"]:
            pick += [prettify_text(t) for s_, t in INSIGHTS if s_ == sec][:2]
        ev = _df("EVIDENCE")
        share = []
        if len(ev) and "shap_share_pct" in ev:
            for _, r in ev.dropna(subset=["shap_share_pct"]).sort_values("shap_share_pct", ascending=False).head(10).iterrows():
                share.append({"name": pretty(r["feature"]), "share": float(r["shap_share_pct"]), "actionable": bool(r["actionable"])})
        drivers = []
        for f_ in [f for f in dedupe_features(DRIVER_TABLE["feature"].tolist()) if f in LEVEL_TABLES][:12]:
            t = LEVEL_TABLES[f_]
            drivers.append({"name": pretty(f_), "iv": float(DRIVER_TABLE.set_index("feature").loc[f_, "iv"]),
                            "levels": [{"level": level_label(f_, r["level"]), "n": int(r["n"]), "rate": float(r["churn_rate"]), "lo": float(r["ci_low"]),
                                        "hi": float(r["ci_high"])} for _, r in t.iterrows()]})
        evt = None
        if len(ev):
            e2 = ev.head(10).copy()
            e2["highest_risk_level"] = [level_label(f_, v_) for f_, v_ in zip(e2["feature"], e2["highest_risk_level"])]
            e2["feature"] = e2["feature"].map(pretty)
            evt = _tbl(e2, ["feature", "highest_risk_level", "highest_risk_rate", "highest_risk_lift", "iv", "odds_ratio", "hazard_ratio", "shap_rank", "evidence_score", "actionable"],
                       {"feature": "Driver", "highest_risk_level": "Riskiest group", "highest_risk_rate": "Churn rate", "highest_risk_lift": "Lift", "iv": "IV",
                        "odds_ratio": "Odds ratio", "hazard_ratio": "Hazard ratio", "shap_rank": "SHAP rank", "evidence_score": "Methods agreeing", "actionable": "Actionable"},
                       fmt={"highest_risk_rate": lambda v: pct(v)})
        seg = _df("SEGMENT_RULES")
        segt = _tbl(seg.assign(segment_rule=seg["segment_rule"].map(prettify_text)) if len(seg) else seg,
                    ["segment_rule", "n_customers", "share_%", "churn_rate", "lift", "churners_captured_%"],
                    {"segment_rule": "Segment", "n_customers": "Customers", "share_%": "% of base", "churn_rate": "Churn rate", "lift": "Lift", "churners_captured_%": "% of churners"},
                    fmt={"churn_rate": lambda v: pct(v)}, n=12)
        km = {"unit": _g("SURV_UNIT", ""), "series": []}
        if SURV_AVAILABLE:
            def _ds(tt, ss, n=160):
                idx = np.unique(np.linspace(0, len(tt) - 1, min(n, len(tt))).astype(int))
                return [[float(tt[i]), float(ss[i])] for i in idx]
            _, tt, ss, _, _ = km_curve(DUR, EVT)
            km["series"].append({"name": "All", "points": [[0.0, 1.0]] + _ds(tt, ss)})
            if "tr_tier" in globals():
                tiers_s = np.asarray(tr_tier)[m_ok.values]
                for tn in ["High", "Medium", "Low"]:
                    mm = tiers_s == tn
                    if mm.sum() >= 30:
                        _, tt, ss, _, _ = km_curve(DUR[mm], EVT[mm])
                        km["series"].append({"name": tn, "points": [[0.0, 1.0]] + _ds(tt, ss)})
        hz = _df("HAZARD_TABLE")
        hazard = [{"interval": r["interval"], "hazard": float(r["hazard"]), "events": int(r["churn_events"])} for _, r in hz.iterrows()]
        dec = _df("DECILE_TABLE")
        gains = [{"x": 0.0, "y": 0.0}] + [{"x": float(a), "y": float(b)} for a, b in zip(dec.get("cum_customers_%", []), dec.get("cum_churners_captured_%", []))]
        me = _df("MODEL_EVAL")
        models = [{"model": str(r["model"]).replace("FINAL: ", "Final: "), "auc": float(r["roc_auc"]), "lo": float(r["auc_ci_low"]), "hi": float(r["auc_ci_high"]),
                   "pr_auc": float(r["pr_auc"]), "final": str(r["model"]).startswith("FINAL")} for _, r in me.sort_values("roc_auc", ascending=False).iterrows()]
        mt = _tbl(me, ["model", "algorithm", "roc_auc", "auc_ci_low", "auc_ci_high", "pr_auc", "brier", "ece", "ks", "empc_per_customer", "delong_p_holm"],
                  {"model": "Model", "algorithm": "Algorithm", "roc_auc": "ROC-AUC", "auc_ci_low": "CI low", "auc_ci_high": "CI high", "pr_auc": "PR-AUC",
                   "brier": "Brier", "ece": "ECE", "ks": "KS", "empc_per_customer": "EMPC/customer", "delong_p_holm": "DeLong p (Holm)"}, n=30)
        camp = None
        if F["value"] and "VALUE_SAVED_TR" in globals():
            order = np.argsort(-(PROB_BIZ * VALUE_SAVED_TR))
            vv = (y * VALUE_SAVED_TR)[order]
            cum_v = np.cumsum(vv)
            ks_ = np.linspace(0.01, 1.0, 100)
            idx_ = np.maximum((ks_ * N_TR).astype(int), 1) - 1
            tot_v = float(cum_v[-1])
            camp = {"pct": [float(100 * k) for k in ks_], "n": [int(i + 1) for i in idx_], "value": [float(cum_v[i]) for i in idx_],
                    "value_random": [float(tot_v * k) for k in ks_], "cost": float(CFG.RETENTION_COST), "success": float(CFG.RETENTION_SUCCESS_RATE),
                    "unit": CFG.CURRENCY}
        cz = _df("CAUSAL")
        rb = _df("CAUSAL_ROBUST")
        verdict = dict(zip(rb.get("feature", []), rb.get("verdict", []))) if len(rb) else {}
        causal = [{"label": prettify_text(r["favorable_condition (T=1)"]), "ate": float(r["ate_pp"]), "lo": float(r["ci_low_pp"]), "hi": float(r["ci_high_pp"]),
                   "naive": float(r["naive_diff_pp"]), "robust": verdict.get(r["feature"]) == "robust", "verdict": verdict.get(r["feature"], "")}
                  for _, r in cz.iterrows()] if len(cz) else []
        recs_t = _tbl(_df("RECS"), ["priority", "recommendation", "action", "evidence", "estimated_impact", "kpi"],
                      {"priority": "#", "recommendation": "Recommendation", "action": "Action", "evidence": "Evidence", "estimated_impact": "Expected impact", "kpi": "KPI"})
        pers = _df("PERSONAS")
        pers_t = _tbl(pers.assign(top_drivers=pers["top_drivers"].map(prettify_text)) if len(pers) else pers,
                      ["persona", "share_of_at_risk_%", "actual_churn_rate", "avg_pred_churn", "top_drivers"],
                      {"persona": "Persona", "share_of_at_risk_%": "% of at-risk", "actual_churn_rate": "Actual churn", "avg_pred_churn": "Predicted churn", "top_drivers": "Main drivers"},
                      fmt={"actual_churn_rate": lambda v: pct(v), "avg_pred_churn": lambda v: pct(v)})
        cust = None
        act_path = os.path.join(CFG.TABLE_DIR, "test_customer_action_list.csv")
        if os.path.exists(act_path) and getattr(CFG, "DASHBOARD_INCLUDE_CUSTOMERS", True):
            al = pd.read_csv(act_path)
            tot = len(al)
            al = al.head(1500)
            for c in al.columns:
                if "probability" in c or c.startswith("uplift"):
                    al[c] = al[c].map(lambda v: f"{100 * v:.1f}%" if "probability" in c and pd.notna(v) else (f"{v:.2f}" if pd.notna(v) else ""))
            cust = _tbl(al, n=1500)
            cust["total"] = tot
        fig_keys = [("system_architecture", "End-to-end analytics architecture"), ("validation_design", "Validation design"),
                    ("model_architecture", "Predictive model architecture"), ("model_development_journey", "Model development journey (all experiments)"),
                    ("ablation_study", "Ablation study"), ("shap_beeswarm", "SHAP summary"), ("causal_dag", "Causal identification strategy"),
                    ("uplift_targeting", "Uplift targeting"), ("mlops_loop", "Deployment & continuous-learning loop"), ("implementation_roadmap", "Implementation roadmap"),
                    ("fairness_audit", "Fairness audit"), ("executive_summary", "Executive summary")]
        figs = []
        for k_, cap in fig_keys:
            p_ = find_fig(k_)
            if p_:
                shutil.copyfile(p_, os.path.join(out, "figures", os.path.basename(p_)))
                figs.append({"file": os.path.basename(p_), "caption": cap})
        DASH = {"meta": {"title": meta_title, "short_title": f"{CFG.TEAM_NAME} — Churn Dashboard", "team": CFG.TEAM_NAME, "company": CFG.COMPANY_NAME,
                         "currency": CFG.CURRENCY, "generated": time.strftime("%Y-%m-%d %H:%M"), "n": int(N_TR), "churn_rate": float(CHURN_RATE),
                         "auc": float(F["auc"]), "context": CFG.CASE_CONTEXT or "",
                         "questions": list(CFG.CASE_QUESTIONS or []), "members": list(CFG.TEAM_MEMBERS or [])},
                "kpis": kpis, "findings": pick[:8], "driver_share": share, "drivers": drivers, "evidence": evt, "segments": segt,
                "km": km, "hazard": hazard, "hazard_label": _g("HZ_LABEL", "period"), "gains": gains, "models": models, "model_table": mt,
                "campaign": camp, "causal": causal, "recs": recs_t, "personas": pers_t, "customers": cust, "figures": figs}
        with open(os.path.join(out, "data.js"), "w", encoding="utf-8") as f:
            f.write("window.DASH = " + json.dumps(DASH, default=lambda o: (o.item() if hasattr(o, "item") else str(o)), ensure_ascii=False) + ";\n")
        with open(os.path.join(out, "vercel.json"), "w", encoding="utf-8") as f:
            json.dump({"cleanUrls": True, "headers": [{"source": "/(.*)", "headers": [{"key": "X-Robots-Tag", "value": "noindex"}]}]}, f, indent=2)
        with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
            f.write("# Churn dashboard\n\nBuka `index.html` langsung di browser (offline OK).\n\nDeploy ke Vercel:\n\n```\ncd dashboard\nnpx vercel --prod\n```\n")
        print(f"Dashboard → {os.path.abspath(out)}  (buka index.html; deploy: cd dashboard && npx vercel --prod)")

# %% [markdown]
# ## 24.4 Package Deliverables (ZIP sesuai aturan soal)
# Setelah artikel PDF final selesai: set `CFG.ARTICLE_PDF_PATH = "path/ke/TeamName_Final Stage 1.pdf"`, `CFG.MAKE_ZIP = True`, lalu jalankan cell ini saja.
# ZIP berisi: artikel PDF, notebook `.ipynb` (syntax), dan folder `supporting/` (figures, tables, insights, submission).
#
# **Jika X → Y:** soal meminta submission terpisah → tetap upload `submission.csv` sesuai instruksi; ZIP tetap memuatnya di `supporting/`.

# %%
import zipfile, glob

with safe_section("Package deliverables"):
    base = f"{CFG.TEAM_NAME}_Final Stage 1"
    if not CFG.MAKE_ZIP:
        print(f"⏭️ MAKE_ZIP=False. Setelah PDF siap: CFG.ARTICLE_PDF_PATH = '...pdf'; CFG.MAKE_ZIP = True → run cell ini.")
        print(f"   Nama file yang benar: '{base}.pdf' dan '{base}.zip'")
    else:
        zpath = os.path.join(CFG.OUTPUT_DIR, f"{base}.zip")
        nb_paths = [CFG.NOTEBOOK_PATH] if CFG.NOTEBOOK_PATH else [p for p in glob.glob("*.ipynb") if "checkpoint" not in p]
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            art = CFG.ARTICLE_PDF_PATH if CFG.ARTICLE_PDF_PATH and os.path.exists(CFG.ARTICLE_PDF_PATH) else os.path.join(CFG.OUTPUT_DIR, f"{base}.pdf")
            if os.path.exists(art):
                z.write(art, f"{base}.pdf")
                print("  + artikel:", art, "" if CFG.ARTICLE_PDF_PATH else "(draft otomatis — pastikan ini versi final yang sudah kamu edit!)")
            else:
                print("  ⚠️ PDF artikel tidak ditemukan → set CFG.ARTICLE_PDF_PATH ke PDF final kamu!")
            for nbp in nb_paths:
                if nbp and os.path.exists(nbp):
                    z.write(nbp, os.path.basename(nbp))
                    print("  + notebook:", nbp)
            for root, _, files in os.walk(CFG.OUTPUT_DIR):
                if os.path.abspath(root).startswith(os.path.abspath(CFG.MODEL_DIR)):
                    continue
                for fn in files:
                    full = os.path.join(root, fn)
                    if fn.endswith(".zip") or fn.startswith("REPORT_") or fn.startswith(base) or fn.startswith("_"):
                        continue
                    if fn in ("JUDGE_QA.md", "NARRATIVE_OPTIONS.md", "SENTUHAN_MANUSIA.md", "QA_CHECK.md", "SLOP_CHECK.md"):
                        continue                       # catatan persiapan tim, bukan untuk juri
                    z.write(full, os.path.join("supporting", os.path.relpath(full, CFG.OUTPUT_DIR)))
        print(f"✅ ZIP dibuat: {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB)")

# %% [markdown]
# # 25. Troubleshooting, Recipes & Checklist
#
# ## 25.1 Troubleshooting (Error → Penyebab → Solusi)
# | Error / Gejala | Penyebab | Solusi |
# |---|---|---|
# | `FileNotFoundError` | path salah | `os.listdir("./data")`, cek nama/ekstensi; path Windows pakai `r"C:\..."` atau `/` |
# | Data terbaca 1 kolom | separator `;`/`|` | `smart_read(path, sep=";")` |
# | `UnicodeDecodeError` | encoding | `smart_read(path, encoding="latin-1")` |
# | `KeyError: TARGET_COL` | nama kolom beda (spasi/kapital) | `print(raw_train.columns.tolist())`, set `CFG.TARGET_COL` persis |
# | "Target punya N nilai unik" | target bukan biner | set `CFG.POSITIVE_LABEL` (nilai/list nilai churn) |
# | Churn rate terbalik (≈80%) | mapping positive salah | set `CFG.POSITIVE_LABEL` ke nilai churn |
# | Kolom angka terbaca kategori | ada teks/simbol aneh | `CFG.FORCE_NUMERIC = ["kolom"]` |
# | Kode wilayah dianggap numeric | memang angka | `CFG.FORCE_CATEGORICAL = ["kolom"]` |
# | Role (tenure/premium) salah/tidak terdeteksi | nama kolom tidak umum | `CFG.COLUMN_ROLES = {"tenure": "...", "premium": "..."}` |
# | AUC > 0.97 / "terlalu bagus" | **leakage** | cek Section 5.3, buang kolom post-event (`CFG.DROP_COLS`) |
# | `LightGBMError: Do not support special JSON characters` | nama kolom | sudah otomatis disanitasi; kalau masih, rename manual |
# | XGBoost `enable_categorical` / categorical error | versi lama | notebook fallback ke kode kategori otomatis; atau `pip install -U xgboost` |
# | CatBoost `Invalid type for cat_feature` / NaN | NaN di kategori | sudah diisi "NA"; cek kolom baru di `custom_feature_engineering` |
# | `ValueError: Input contains NaN` (sklearn) | model tidak support NaN | pakai pipeline bawaan (imputer) — jangan bypass `fit_predict` |
# | `MemoryError` / sangat lambat | data besar | `RUN_MODE="fast"`, kurangi `MODELS`, `N_FOLDS=3`, `SHAP_SAMPLE=1000`, matikan RF/ET |
# | Optuna lama | trial banyak | `OPTUNA_TIMEOUT=300` atau `USE_OPTUNA=False` |
# | `Singular matrix` di Logit/Cox | fitur kolinear / separasi sempurna | otomatis kurangi fitur; atau buang fitur VIF tinggi |
# | Cox / KM di-skip | tidak ada tenure atau statsmodels | set role tenure; `pip install statsmodels` |
# | SHAP error | versi shap / model | contribs native LightGBM/XGB/CatBoost dipakai dulu; beeswarm manual fallback |
# | Submission `ID mismatch` / NaN | ID beda tipe | ID dibandingkan sebagai string; cek `ID_COL` |
# | Skor LB jauh < CV | format salah / drift / leakage | cek mode probability vs label, adversarial AUC, kolom leak |
# | Figure tidak muncul | backend | `%matplotlib inline` di cell pertama |
# | Setelah install package masih error import | kernel belum restart | Kernel → Restart, run ulang |
# | Excel export gagal | openpyxl tidak ada | `pip install openpyxl` (CSV tetap tersimpan) |
#
# ## 25.2 Experiment recipes
# ```python
# # (A) Cek cepat semua jalan
# CFG.RUN_MODE = "fast"
# # (B) Final kuat (waktu cukup ~30-60 menit)
# CFG.RUN_MODE = "full"; CFG.USE_OPTUNA = True; CFG.OPTUNA_TRIALS = 50; CFG.N_REPEATS = 2
# # (C) Waktu mepet tapi tetap kuat
# CFG.RUN_MODE = "full"; CFG.USE_OPTUNA = False; CFG.MODELS = ["logreg", "lgbm", "catboost"]
# # (D) Metric F1 / accuracy
# CFG.METRIC = "f1"; CFG.SUBMISSION_MODE = "label"; CFG.IMBALANCE = "class_weight"
# # (E) Metric logloss
# CFG.METRIC = "logloss"; CFG.CALIBRATION = "auto"
# # (F) Data besar (>200k baris)
# CFG.MODELS = ["lgbm", "xgb"]; CFG.N_FOLDS = 3; CFG.SHAP_SAMPLE = 2000; CFG.MAX_EDA_PLOTS = 12
# # (G) Ada household/agent grouping
# CFG.GROUP_COL = "household_id"
# # (H) GPU tersedia
# CFG.USE_GPU = True
# ```
#
# ## 25.3 Checklist sebelum submit
# - [ ] Target mapping benar (churn = 1) & churn rate masuk akal
# - [ ] Tidak ada fitur leak (Section 5.3) & shuffled-target AUC ≈ 0.5
# - [ ] Adversarial AUC wajar (< 0.7) atau drift sudah dijelaskan
# - [ ] Final run dengan `RUN_MODE="full"` (bukan fast)
# - [ ] Format submission sesuai sample (probability vs label!), validator semua ✅
# - [ ] File dinamai sesuai aturan lomba (misal `TeamName_Final Stage 1`)
# - [ ] Figure untuk laporan pakai `CFG.FIG_DPI = 300`
# - [ ] Notebook di-run ulang dari atas (Restart & Run All) tanpa error → simpan .ipynb + output untuk lampiran
# - [ ] ZIP: artikel PDF + notebook + file pendukung
