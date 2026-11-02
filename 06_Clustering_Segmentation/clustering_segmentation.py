# -*- coding: utf-8 -*-
# =====================================================================================================
# clustering_segmentation.py — versi SCRIPT dari clustering_segmentation.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python clustering_segmentation.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb clustering_segmentation.py
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
# # Clustering & Customer Segmentation — Competition Template
#
# Notebook **mandiri** untuk segmentasi / clustering data tabular apa pun, dengan fokus kuat pada **interpretasi bisnis**
# (mis. segmentasi nasabah asuransi untuk kasus churn). Alur lengkap:
# load data → data quality → pemilihan fitur → preprocessing (missing, skew, outlier, scaling, kategorikal / Gower) →
# reduksi dimensi (PCA, t-SNE, UMAP) → **memilih jumlah cluster K** (elbow+knee, silhouette, CH, DB, gap, BIC) →
# beberapa algoritma (KMeans, MiniBatch, GMM, Agglomerative, DBSCAN, HDBSCAN, K-Prototypes/Gower) → **stabilitas** (ARI) →
# **profil cluster untuk bisnis** (heatmap, snake/radar plot, teks profil otomatis, nama segmen) → **analisis outcome**
# (mis. churn rate per cluster + CI, lift, chi-square, risk × value, rekomendasi aksi) → **penjelasan cluster** (surrogate
# decision tree, permutation importance, SHAP) → export.
#
# **Cara pakai singkat:**
# 1. Taruh data di `./data/train.csv` (atau ubah `CFG.DATA_PATH` di cell **USER OVERRIDE**).
# 2. Set `CFG.ID_COL`, opsional `CFG.FEATURE_COLS` (kosong = otomatis), `CFG.OUTCOME_COL` (mis. `"churn"`) dan `CFG.VALUE_COL` (mis. `"premium"`).
# 3. Jalankan `RUN_MODE="fast"` dulu, cek tidak ada error, lalu `"full"`.
# 4. Baca bagian **Choosing K**, **Stability**, dan **Profiling** — itu inti laporan segmentasi.
#
# **Prinsip penting:**
# - Clustering = **unsupervised**: tidak ada "jawaban benar". Kualitas dinilai dari (a) metrik internal (silhouette dll),
#   (b) **stabilitas** (cluster tetap sama kalau data/seed sedikit berubah), dan (c) **kebermaknaan bisnis** (bisa diberi nama & aksi).
# - **Jangan masukkan outcome (mis. churn) sebagai fitur clustering.** Outcome dipakai SETELAH clustering untuk menilai segmen
#   (supaya segmen menggambarkan profil pelanggan, lalu kita lihat segmen mana yang berisiko).
# - Skala fitur sangat mempengaruhi hasil (KMeans berbasis jarak) → scaling wajib.
#
# **Struktur:** 0 Config → 1 Install/Import → 2 Helper → 3 Load (+RFM) → 4 Data quality → 5 Fitur → 6 EDA → 7 Preprocessing →
# 8 Reduksi dimensi → 9 Memilih K → 10 Algoritma → 11 Stabilitas → 12 Label final → 13 Profiling → 14 Profil teks →
# 15 Outcome & aksi bisnis → 16 Penjelasan cluster → 17 Export & assign data baru → 18 Ringkasan laporan → 19 Debugging →
# 20 Troubleshooting → 21 Recipes → 22 Checklist.

# %% [markdown]
# ## 0. Konfigurasi (CFG)
#
# **Tujuan:** semua pengaturan di satu tempat; komentar menjelaskan fungsi, kapan diubah, dan nilai umum.
#
# **Jika X maka Y (cepat):**
# - Data kebanyakan **numerik** → `CAT_ENCODING="onehot"`, algoritma KMeans/GMM.
# - Data **campuran** numerik + kategorikal penting, ukuran kecil (< 3000 baris) → tambahkan `"kmedoids_gower"` / `"agglo_gower"` (sudah default).
# - Data **besar** (> 50 ribu baris) → `RUN_MODE="fast"` dulu, `METRIC_SAMPLE` kecil, andalkan `kmeans`/`minibatch`.
# - Ada outlier ekstrem (premi sangat besar) → `CLIP_QUANTILES=(0.01, 0.99)` + `LOG_SKEWED=True` + `SCALER="robust"`.
# - Ingin bentuk cluster bebas & deteksi noise → `dbscan` / `hdbscan` (label -1 = noise).

# %%
import os

class CFG:
    # ===================== MODE =====================
    RUN_MODE = "full"             # "fast" (debug: sampel kecil, bootstrap sedikit) | "full"
    SEED = 42
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"           # install package yang BELUM ada saja (tidak upgrade) (env TEMPLATE_AUTO_INSTALL=0 → off)
    INSTALL_OPTIONAL = False      # umap-learn, kmodes (opsional)

    # ===================== PATHS =====================
    NOTEBOOK_NAME = "clustering_segmentation"
    DATA_PATH = "./data/train.csv"   # tabel utama (1 baris = 1 entitas/pelanggan)
    TEST_PATH = "./data/test.csv"    # opsional: data baru yang ingin di-assign ke cluster (boleh tidak ada)
    OUTPUT_DIR = None                # None -> ./outputs/<NOTEBOOK_NAME>/

    # ===================== KOLOM =====================
    ID_COL = "id"                 # None kalau tidak ada; tidak dipakai sebagai fitur
    FEATURE_COLS = None           # None = otomatis (numerik + kategorikal kardinalitas rendah). Atau list kolom.
    EXCLUDE_COLS = []             # kolom yang pasti dibuang dari fitur (mis. nama, alamat, tanggal mentah)
    PROFILE_ONLY_COLS = []        # kolom yang TIDAK dipakai clustering tapi ikut di profil (mis. "gender", "channel")
    OUTCOME_COL = None            # mis. "churn" / "claim_flag" -> dianalisis setelah clustering (BUKAN fitur)
    OUTCOME_POSITIVE = None       # nilai positif outcome (mis. "Yes" / 1). None = auto
    VALUE_COL = None              # mis. "premium" / "monetary" -> untuk matriks risk x value
    MAX_CAT_CARDINALITY = 30      # kategorikal dengan unique > ini dibuang otomatis (mis. nama, kode pos)
    RARE_MIN_FREQ = 0.01          # kategori dengan frekuensi < 1% digabung jadi "Other"
    HIGH_MISSING_DROP = 0.6       # kolom dengan missing > 60% dibuang dari fitur

    # ===================== RFM (opsional, dari tabel transaksi) =====================
    USE_RFM = False
    TRANSACTIONS_PATH = "./data/transactions.csv"
    TX_CUSTOMER_COL = "customer_id"
    TX_DATE_COL = "date"
    TX_AMOUNT_COL = "amount"
    TX_INVOICE_COL = None         # kalau ada nomor invoice: frequency = jumlah invoice unik
    RFM_SNAPSHOT_DATE = None      # None = tanggal transaksi terakhir + 1 hari
    RFM_MERGE = "auto"            # "auto": merge ke DATA_PATH kalau ada, kalau tidak pakai tabel RFM saja | "replace" | "merge"

    # ===================== PREPROCESSING =====================
    NUM_IMPUTE = "median"         # "median" | "mean"
    LOG_SKEWED = True             # log1p untuk fitur skewed (|skew| > SKEW_THRESHOLD)
    SKEW_THRESHOLD = 1.0
    CLIP_QUANTILES = (0.01, 0.99) # winsorize outlier; None = tidak
    SCALER = "standard"           # "standard" | "robust" | "minmax" | "quantile"
    CAT_ENCODING = "onehot"       # "onehot" | "none" (kategorikal hanya untuk profil)
    CAT_WEIGHT = 0.5              # bobot kolom one-hot (0.5 ~ setara gamma K-Prototypes). Naikkan kalau kategori harus lebih menentukan
    DROP_CORRELATED = 0.95        # buang salah satu fitur numerik dengan |corr| > ini. None = tidak

    # ===================== REDUKSI DIMENSI =====================
    USE_PCA_FOR_CLUSTERING = False  # True: cluster di ruang PCA (bagus kalau fitur banyak & berkorelasi)
    PCA_VARIANCE = 0.90           # jumlah komponen = yang menjelaskan 90% varians
    RUN_TSNE = True
    TSNE_SAMPLE = 3000
    RUN_UMAP = True               # hanya jika umap-learn asli terinstall

    # ===================== MEMILIH K =====================
    K_MIN = 2
    K_MAX = 10
    METRIC_SAMPLE = 5000          # sampel untuk silhouette / gap (hemat waktu)
    RUN_GAP = True                # gap statistic (lebih lambat)
    GAP_B = 5                     # jumlah dataset referensi gap
    MIN_CLUSTER_PCT = 0.03        # cluster < 3% data dianggap terlalu kecil untuk aksi bisnis
    N_CLUSTERS = None             # None = pakai rekomendasi otomatis. Isi angka untuk memaksa K.

    # ===================== ALGORITMA =====================
    ALGORITHMS = ["kmeans", "minibatch", "gmm", "agglomerative", "dbscan", "hdbscan", "kprototypes",
                  "kmedoids_gower", "agglo_gower"]
    FINAL_ALGO = "auto"           # "auto" | salah satu nama di atas
    KMEANS_N_INIT = 10            # k-means++ dengan 10 inisialisasi; ambil inertia terbaik
    AGGLO_MAX_N = 8000            # Agglomerative O(n^2): di atas ini fit di sampel lalu assign nearest centroid
    GOWER_MAX_N = 3000            # matriks Gower n x n: di atas ini pakai sampel
    DBSCAN_EPS = None             # None = otomatis dari knee k-distance plot
    DBSCAN_MIN_SAMPLES = None     # None = max(5, 2 x jumlah dimensi) dibatasi 50
    HDBSCAN_MIN_CLUSTER_SIZE = None  # None = max(10, 2% n)

    # ===================== STABILITAS =====================
    STABILITY_BOOTSTRAPS = 20
    STABILITY_SEEDS = 10

    # ===================== PROFILING & BISNIS =====================
    SORT_CLUSTERS_BY = "size"     # "size" (cluster 0 = terbesar) | "outcome" (cluster 0 = outcome rate tertinggi)
    PROFILE_LANG = "id"           # "id" | "en" untuk teks profil otomatis
    TOP_FEATURES_PROFILE = 4      # jumlah fitur pembeda yang disebut di teks profil
    MAX_PLOT_FEATURES = 12

    # ===================== PENJELASAN =====================
    SURROGATE_MAX_DEPTH = 4
    USE_SHAP = True
    SHAP_SAMPLE = 1000

    FIG_DPI = 200                 # 300 untuk laporan

# %% [markdown]
# ### 0.1 USER OVERRIDE — edit cell ini
#
# Contoh untuk kasus **segmentasi nasabah asuransi + churn**:
# ```python
# CFG.DATA_PATH = "./data/customers.csv"
# CFG.ID_COL = "customer_id"
# CFG.FEATURE_COLS = ["age", "premium", "tenure_months", "n_claims", "region", "policy_type"]
# CFG.OUTCOME_COL = "churn"; CFG.OUTCOME_POSITIVE = "Yes"
# CFG.VALUE_COL = "premium"
# CFG.RUN_MODE = "fast"
# ```

# %%
# >>> USER OVERRIDE
# Tulis override CFG di bawah ini. Contoh:
# CFG.DATA_PATH = "./data/customers.csv"
# CFG.OUTCOME_COL = "churn"

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
# Library inti: numpy, pandas, scikit-learn, matplotlib, seaborn, scipy. Opsional: `hdbscan` (fallback ke `sklearn.cluster.HDBSCAN`
# untuk sklearn ≥ 1.3), `umap-learn`, `kmodes` (K-Prototypes), `lightgbm`, `shap`. Semua opsional dicek dengan flag `HAS_xxx`.
#
# ⚠️ Ada package bernama `umap` (bukan `umap-learn`) yang **salah** — `import umap` berhasil tapi tidak punya `umap.UMAP`.
# Notebook mengecek `hasattr(umap, "UMAP")`. Kalau salah package: `pip uninstall umap` lalu `pip install umap-learn`.
#
# `ensure_packages` hanya menginstall package yang **belum ada** (tidak pernah upgrade versi yang sudah terinstall).

# %%
import sys
import subprocess
import importlib
import importlib.util

PIP_NAME = {"sklearn": "scikit-learn", "umap": "umap-learn", "kmodes": "kmodes", "hdbscan": "hdbscan",
            "lightgbm": "lightgbm", "shap": "shap", "seaborn": "seaborn", "scipy": "scipy", "joblib": "joblib",
            "matplotlib": "matplotlib", "pandas": "pandas", "numpy": "numpy"}


def ensure_packages(import_names, auto_install=True):
    """Install HANYA package yang belum ada (find_spec None). Error install hanya di-print."""
    missing = [n for n in import_names if importlib.util.find_spec(n) is None]
    if not missing:
        print("Package yang dicek sudah tersedia:", import_names)
        return []
    print("Package belum ada:", missing)
    if not auto_install:
        print("AUTO_INSTALL=False -> install manual: pip install " + " ".join(PIP_NAME.get(m, m) for m in missing))
        return missing
    for name in missing:
        pkg = PIP_NAME.get(name, name)
        try:
            print(f"  installing {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *([] if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else ["--user"]), pkg])
        except Exception as e:
            print(f"  ⚠️ gagal install {pkg}: {e}")
    importlib.invalidate_caches()
    return [n for n in missing if importlib.util.find_spec(n) is None]


ensure_packages(["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "scipy", "joblib"], auto_install=CFG.AUTO_INSTALL)
if CFG.INSTALL_OPTIONAL:
    ensure_packages(["hdbscan", "kmodes", "lightgbm", "shap"], auto_install=CFG.AUTO_INSTALL)
    # umap: hanya install kalau belum ada modul 'umap' sama sekali (kalau ada package 'umap' yang salah, uninstall manual)
    ensure_packages(["umap"], auto_install=CFG.AUTO_INSTALL)

# %%
import json
import time
import math
import random
import warnings
import traceback
from pathlib import Path
from inspect import signature
from collections import Counter, OrderedDict

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

import sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score,
                             accuracy_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE

try:
    import scipy.stats as sps
    from scipy.cluster.hierarchy import linkage as sch_linkage, dendrogram as sch_dendrogram
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import hdbscan as hdbscan_pkg
    HAS_HDBSCAN_PKG = hasattr(hdbscan_pkg, "HDBSCAN")
except Exception:
    HAS_HDBSCAN_PKG = False
try:
    from sklearn.cluster import HDBSCAN as SkHDBSCAN
    HAS_SK_HDBSCAN = True
except Exception:
    HAS_SK_HDBSCAN = False
try:
    import umap as umap_mod
    HAS_UMAP = hasattr(umap_mod, "UMAP")
    if not HAS_UMAP:
        print("⚠️ Modul 'umap' terinstall tapi bukan umap-learn (tidak ada umap.UMAP) -> UMAP dilewati.")
except Exception:
    HAS_UMAP = False
try:
    from kmodes.kprototypes import KPrototypes
    HAS_KMODES = True
except Exception:
    HAS_KMODES = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
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
warnings.simplefilter("ignore")

print("Python", sys.version.split()[0], "| pandas", pd.__version__, "| numpy", np.__version__, "| sklearn", sklearn.__version__)
for _n, _f in [("scipy", HAS_SCIPY), ("hdbscan (package)", HAS_HDBSCAN_PKG), ("sklearn.cluster.HDBSCAN", HAS_SK_HDBSCAN),
               ("umap-learn", HAS_UMAP), ("kmodes (K-Prototypes)", HAS_KMODES), ("lightgbm", HAS_LGB), ("shap", HAS_SHAP)]:
    print(f"  {'✅ ada' if _f else '⚠️ tidak ada'} : {_n}")

# %% [markdown]
# ### 1.1 Setup: RUN_MODE, folder, seed

# %%
def seed_everything(seed=42):
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
    CFG.METRIC_SAMPLE = min(CFG.METRIC_SAMPLE, 2000)
    CFG.TSNE_SAMPLE = min(CFG.TSNE_SAMPLE, 1000)
    CFG.STABILITY_BOOTSTRAPS = min(CFG.STABILITY_BOOTSTRAPS, 5)
    CFG.STABILITY_SEEDS = min(CFG.STABILITY_SEEDS, 4)
    CFG.GAP_B = min(CFG.GAP_B, 3)
    CFG.KMEANS_N_INIT = min(CFG.KMEANS_N_INIT, 4)
    CFG.K_MAX = min(CFG.K_MAX, 8)
    CFG.GOWER_MAX_N = min(CFG.GOWER_MAX_N, 1500)
    CFG.AGGLO_MAX_N = min(CFG.AGGLO_MAX_N, 3000)
    CFG.SHAP_SAMPLE = min(CFG.SHAP_SAMPLE, 300)
    print("RUN_MODE = fast -> sampel & iterasi dipangkas.")
else:
    print("RUN_MODE = full")
OUTPUT_DIR = Path(CFG.OUTPUT_DIR or f"./outputs/{CFG.NOTEBOOK_NAME}")
FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR = [OUTPUT_DIR / s for s in ["figures", "tables", "models", "submissions"]]
for _d in [OUTPUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR, SUB_DIR]:
    os.makedirs(_d, exist_ok=True)
print("Output dir:", OUTPUT_DIR.resolve())
FAILED_MODELS = {}
RNG = np.random.RandomState(CFG.SEED)

# %% [markdown]
# ## 2. Helper functions
#
# Fungsi bantu: baca file, simpan figure/tabel, shim versi sklearn (OneHotEncoder, Agglomerative, t-SNE), Gower distance,
# K-Medoids sederhana, deteksi knee, Wilson CI, dsb. Tidak perlu diubah.

# %%
def smart_read(path, **kwargs):
    """Baca csv/tsv/xlsx/xls/parquet/json/feather/pickle dengan fallback separator & encoding."""
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
            df_ = pd.read_csv(path, sep=sep_default, encoding=enc, **kwargs)
            if df_.shape[1] == 1 and ext != ".tsv":
                df2 = pd.read_csv(path, sep=None, engine="python", encoding=enc, **kwargs)
                if df2.shape[1] > 1:
                    df_ = df2
            return df_
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            last_err = e
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc, **kwargs)
            except Exception as e2:
                last_err = e2
    raise last_err


_FIG_COUNTER = [0]


def save_fig(name, fig=None, show=True):
    """Simpan figure (figNN_<name>.png) ke FIG_DIR, tampilkan, lalu close."""
    fig = fig or plt.gcf()
    _FIG_COUNTER[0] += 1
    try:
        fig.savefig(FIG_DIR / f"fig{_FIG_COUNTER[0]:02d}_{name}.png", dpi=CFG.FIG_DPI, bbox_inches="tight")
    except Exception as e:  # mis. MemoryError untuk figure sangat besar -> coba DPI rendah
        try:
            fig.savefig(FIG_DIR / f"fig{_FIG_COUNTER[0]:02d}_{name}.png", dpi=100, bbox_inches="tight")
            print(f"Info: figure '{name}' disimpan dengan DPI 100 (DPI {CFG.FIG_DPI} gagal: {e})")
        except Exception as e2:
            print("Gagal simpan figure:", e2)
    if show:
        plt.show()
    plt.close(fig)


def save_table(df_, name, index=False):
    df_.to_csv(TAB_DIR / f"{name}.csv", index=index)


def make_ohe():
    """OneHotEncoder dense: sklearn>=1.2 pakai sparse_output, versi lama pakai sparse."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_agglo(n_clusters, linkage="ward", metric="euclidean"):
    """AgglomerativeClustering: sklearn>=1.2 pakai metric=, versi lama affinity=."""
    try:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=metric)


def make_tsne(n_components=2, perplexity=30, seed=42, n_iter=1000):
    """TSNE kompatibel lintas versi (max_iter vs n_iter, learning_rate='auto')."""
    params = signature(TSNE.__init__).parameters
    kw = dict(n_components=n_components, perplexity=perplexity, random_state=seed, init="pca")
    if "max_iter" in params:
        kw["max_iter"] = n_iter
    elif "n_iter" in params:
        kw["n_iter"] = n_iter
    if "learning_rate" in params:
        kw["learning_rate"] = "auto"
    try:
        return TSNE(**kw)
    except Exception:
        kw.pop("learning_rate", None)
        return TSNE(**kw)


def safe_silhouette(X, labels, metric="euclidean", sample=None, seed=42):
    """Silhouette pada sampel; abaikan noise (-1). NaN kalau < 2 cluster."""
    labels = np.asarray(labels)
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return np.nan
    try:
        if metric == "precomputed":
            idx = np.where(mask)[0]
            return float(silhouette_score(X[np.ix_(idx, idx)], labels[idx], metric="precomputed"))
        Xm, lm = X[mask], labels[mask]
        ss = min(sample or len(lm), len(lm))
        return float(silhouette_score(Xm, lm, sample_size=ss if ss < len(lm) else None, random_state=seed))
    except Exception:
        return np.nan


def internal_scores(X, labels, sample=None):
    """Silhouette, Calinski-Harabasz, Davies-Bouldin (tanpa noise)."""
    labels = np.asarray(labels)
    mask = labels != -1
    out = {"silhouette": safe_silhouette(X, labels, sample=sample, seed=CFG.SEED), "calinski_harabasz": np.nan,
           "davies_bouldin": np.nan}
    if len(np.unique(labels[mask])) >= 2:
        try:
            out["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
            out["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
        except Exception:
            pass
    return out


def knee_point(xs, ys):
    """Knee/elbow otomatis: titik dengan jarak terbesar ke garis lurus antara titik pertama & terakhir (setelah normalisasi)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3:
        return xs[0]
    xn = (xs - xs.min()) / (xs.max() - xs.min() + 1e-12)
    yn = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    v = p2 - p1
    d = np.abs(v[0] * (yn - p1[1]) - v[1] * (xn - p1[0])) / (np.linalg.norm(v) + 1e-12)  # jarak titik ke garis
    return xs[int(np.argmax(d))]


def wilson_ci(k, n, z=1.96):
    """Interval kepercayaan Wilson untuk proporsi (lebih akurat dari normal approx untuk n kecil / p ekstrem)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def gower_matrix(A_num, A_cat, B_num=None, B_cat=None, ranges=None, w_num=1.0, w_cat=1.0):
    """Jarak Gower antara baris A dan B (numpy). Numerik: |a-b|/range; kategorikal: 0 kalau sama, 1 kalau beda.
    NaN numerik diabaikan per pasangan (rata-rata hanya atas fitur yang tersedia)."""
    if B_num is None:
        B_num, B_cat = A_num, A_cat
    nA = len(A_num) if A_num is not None else len(A_cat)
    nB = len(B_num) if B_num is not None else len(B_cat)
    total = np.zeros((nA, nB), dtype=np.float32)
    count = np.zeros((nA, nB), dtype=np.float32)
    if A_num is not None and A_num.shape[1] > 0:
        for j in range(A_num.shape[1]):
            r = ranges[j] if ranges is not None else (np.nanmax(A_num[:, j]) - np.nanmin(A_num[:, j]))
            r = r if r and np.isfinite(r) and r > 0 else 1.0
            d = np.abs(A_num[:, j][:, None].astype(np.float32) - B_num[:, j][None, :].astype(np.float32)) / r
            ok = np.isfinite(d)
            total += np.where(ok, d, 0) * w_num
            count += ok * w_num
    if A_cat is not None and A_cat.shape[1] > 0:
        for j in range(A_cat.shape[1]):
            d = (A_cat[:, j][:, None] != B_cat[:, j][None, :]).astype(np.float32)
            total += d * w_cat
            count += w_cat
    return total / np.maximum(count, 1e-9)


def kmedoids_simple(D, k, n_init=5, max_iter=100, seed=42):
    """K-Medoids (alternating / Voronoi iteration) pada matriks jarak D. Init k-medoids++. Return labels, medoid_idx, cost."""
    rng = np.random.RandomState(seed)
    n = D.shape[0]
    best = None
    for _ in range(n_init):
        med = [rng.randint(n)]
        for _k in range(1, k):
            dmin = D[:, med].min(axis=1) ** 2
            prob = dmin / dmin.sum() if dmin.sum() > 0 else np.ones(n) / n
            med.append(rng.choice(n, p=prob))
        med = np.array(med)
        for _it in range(max_iter):
            lab = np.argmin(D[:, med], axis=1)
            new = med.copy()
            for c in range(k):
                idx = np.where(lab == c)[0]
                if len(idx) == 0:
                    continue
                new[c] = idx[np.argmin(D[np.ix_(idx, idx)].sum(axis=1))]
            if np.array_equal(np.sort(new), np.sort(med)):
                med = new
                break
            med = new
        lab = np.argmin(D[:, med], axis=1)
        cost = D[np.arange(n), med[lab]].sum()
        if best is None or cost < best[2]:
            best = (lab, med, cost)
    return best


def nearest_centroid_assign(X_fit, labels_fit, X_new):
    """Assign titik baru ke centroid cluster terdekat (untuk algoritma tanpa .predict)."""
    labs = [c for c in np.unique(labels_fit) if c != -1]
    cents = np.vstack([X_fit[labels_fit == c].mean(axis=0) for c in labs])
    d = ((X_new[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return np.array(labs)[np.argmin(d, axis=1)], cents


print("Helper siap. sklearn", sklearn.__version__)

# %% [markdown]
# ## 3. Load data (+ RFM builder opsional)
#
# **RFM** (Recency, Frequency, Monetary) adalah fitur klasik segmentasi pelanggan dari **tabel transaksi** (`customer_id, date, amount`):
# - **Recency** = hari sejak transaksi terakhir (kecil = aktif), **Frequency** = jumlah transaksi, **Monetary** = total belanja.
# - **Skor RFM 1–5** per dimensi pakai kuintil (R dibalik: recency kecil = skor 5). Segmen klasik (Champions, Loyal, At Risk, Hibernating, ...)
#   ditentukan dari kombinasi skor R & F.
# - Untuk clustering dipakai nilai R/F/M (di-log & di-scale), sementara skor & segmen klasik dipakai sebagai **pembanding di profil**.
#
# **Jika X maka Y:** tabel utama sudah per pelanggan dan tidak ada transaksi → `USE_RFM=False`. Ada keduanya → `RFM_MERGE="merge"`
# (RFM digabung ke tabel pelanggan lewat `ID_COL`).
#
# **Untuk laporan:** jelaskan definisi R, F, M dan tanggal snapshot; tampilkan tabel distribusi segmen RFM klasik vs cluster.

# %%
RFM_SEGMENT_MAP = OrderedDict([
    (r"[1-2][1-2]", "Hibernating"), (r"[1-2][3-4]", "At Risk"), (r"[1-2]5", "Can't Lose Them"),
    (r"3[1-2]", "About to Sleep"), (r"33", "Need Attention"), (r"[3-4][4-5]", "Loyal Customers"),
    (r"41", "Promising"), (r"51", "New Customers"), (r"[4-5][2-3]", "Potential Loyalists"), (r"5[4-5]", "Champions")])


def score_quintile(s, reverse=False):
    """Skor 1..5 berbasis kuintil rank (aman untuk banyak nilai sama)."""
    n_bins = int(min(5, s.nunique(), len(s)))
    if n_bins < 2:
        return pd.Series(3, index=s.index)
    r = s.rank(method="first")
    sc = pd.qcut(r, n_bins, labels=False) + 1
    sc = (sc * 5 / n_bins).round().clip(1, 5).astype(int)
    return (6 - sc) if reverse else sc


def build_rfm(tx):
    """Bangun tabel RFM per pelanggan dari tabel transaksi."""
    tx = tx.copy()
    tx[CFG.TX_DATE_COL] = pd.to_datetime(tx[CFG.TX_DATE_COL], errors="coerce")
    tx[CFG.TX_AMOUNT_COL] = pd.to_numeric(tx[CFG.TX_AMOUNT_COL], errors="coerce")
    n0 = len(tx)
    tx = tx.dropna(subset=[CFG.TX_CUSTOMER_COL, CFG.TX_DATE_COL, CFG.TX_AMOUNT_COL])
    print(f"Transaksi valid: {len(tx)}/{n0}")
    snap = pd.Timestamp(CFG.RFM_SNAPSHOT_DATE) if CFG.RFM_SNAPSHOT_DATE else tx[CFG.TX_DATE_COL].max() + pd.Timedelta(days=1)
    g = tx.groupby(CFG.TX_CUSTOMER_COL)
    rfm = pd.DataFrame({
        "recency": (snap - g[CFG.TX_DATE_COL].max()).dt.days,
        "frequency": g[CFG.TX_INVOICE_COL].nunique() if CFG.TX_INVOICE_COL else g.size(),
        "monetary": g[CFG.TX_AMOUNT_COL].sum(),
        "tenure_days": (snap - g[CFG.TX_DATE_COL].min()).dt.days,
    })
    rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"].clip(lower=1)
    rfm["R_score"] = score_quintile(rfm["recency"], reverse=True)
    rfm["F_score"] = score_quintile(rfm["frequency"])
    rfm["M_score"] = score_quintile(rfm["monetary"])
    rfm["RFM_score"] = "R" + rfm["R_score"].astype(str) + "F" + rfm["F_score"].astype(str) + "M" + rfm["M_score"].astype(str)
    rf = rfm["R_score"].astype(str) + rfm["F_score"].astype(str)
    seg = pd.Series("Other", index=rfm.index)
    for pat, name in RFM_SEGMENT_MAP.items():
        seg[rf.str.fullmatch(pat)] = name
    rfm["RFM_segment"] = seg
    print(f"Snapshot date: {snap.date()} | pelanggan: {len(rfm)}")
    return rfm.reset_index().rename(columns={CFG.TX_CUSTOMER_COL: CFG.ID_COL or CFG.TX_CUSTOMER_COL})


RFM_COLS = []
data = smart_read(CFG.DATA_PATH) if (CFG.DATA_PATH and os.path.exists(str(CFG.DATA_PATH))) else None
if CFG.USE_RFM:
    assert os.path.exists(str(CFG.TRANSACTIONS_PATH)), f"TRANSACTIONS_PATH tidak ada: {CFG.TRANSACTIONS_PATH}"
    tx = smart_read(CFG.TRANSACTIONS_PATH)
    if CFG.ID_COL is None:
        CFG.ID_COL = CFG.TX_CUSTOMER_COL
    rfm = build_rfm(tx)
    RFM_COLS = ["recency", "frequency", "monetary", "tenure_days", "avg_order_value"]
    display(rfm.head())
    display(rfm["RFM_segment"].value_counts().to_frame("n_customers"))
    save_table(rfm, "rfm_table")
    mode = CFG.RFM_MERGE
    if mode == "auto":
        mode = "merge" if data is not None else "replace"
    if mode == "merge" and data is not None:
        data[CFG.ID_COL] = data[CFG.ID_COL].astype(str)
        rfm[CFG.ID_COL] = rfm[CFG.ID_COL].astype(str)
        data = data.merge(rfm, on=CFG.ID_COL, how="left")
        print(f"RFM di-merge ke data utama. Pelanggan tanpa transaksi: {int(data['recency'].isna().sum())}")
    else:
        data = rfm
    for c in ["R_score", "F_score", "M_score", "RFM_score", "RFM_segment"]:
        if c not in CFG.PROFILE_ONLY_COLS:
            CFG.PROFILE_ONLY_COLS = list(CFG.PROFILE_ONLY_COLS) + [c]
assert data is not None, f"DATA_PATH tidak ditemukan: {CFG.DATA_PATH} (atau aktifkan USE_RFM dengan TRANSACTIONS_PATH)"
data = data.reset_index(drop=True)
HAS_TEST = bool(CFG.TEST_PATH) and os.path.exists(str(CFG.TEST_PATH))
test = smart_read(CFG.TEST_PATH) if HAS_TEST else None
print("Data shape:", data.shape, "| test (assign) shape:", None if test is None else test.shape)
display(data.head())

# %% [markdown]
# ## 4. Data quality report
#
# **Yang dicek:** tipe data, % missing, jumlah unique, kolom konstan, kolom ID-like (semua unik), numerik yang tersimpan sebagai string
# (otomatis dikonversi), tanggal.
#
# **Cara membaca:** kolom `role_guess` = tebakan peran kolom. `numeric`/`categorical` = kandidat fitur; `id-like`/`constant`/`text/high-card`/
# `datetime` = otomatis tidak dipakai sebagai fitur clustering.
#
# **Jika X maka Y:**
# - Missing > 60% → dibuang dari fitur (`HIGH_MISSING_DROP`); 5–60% → diimputasi (median/"Missing").
# - Kolom numerik terbaca sebagai string (mis. `"1.250.000"`) → dikonversi otomatis; cek hasilnya di tabel.
# - Kolom tanggal (mis. `join_date`) → **buat fitur turunan** sendiri (mis. tenure = hari sejak join) di cell USER OVERRIDE/sebelum bagian 5.
#
# **Untuk laporan:** sebutkan jumlah baris/kolom, kolom yang dibuang & alasannya, dan cara imputasi.

# %%
def try_numeric(s):
    """Konversi string numerik (format 1.234,5 / 1,234.5 / 'Rp 10.000') ke float kalau >= 95% nilai non-null berhasil."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s, False
    nn = s.notna()
    if nn.sum() == 0:
        return s, False
    base = s.astype(str).str.strip().str.replace(r"^(Rp\.?|IDR|\$)\s*", "", regex=True).str.replace(" ", "", regex=False)
    best, best_ok = None, -1.0
    for cand in [base, base.str.replace(",", "", regex=False),
                 base.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)]:
        v = pd.to_numeric(cand.where(nn), errors="coerce")
        ok = float(v[nn].notna().mean())
        if ok > best_ok:  # strict '>' -> kandidat pertama (format standar) menang kalau seri
            best, best_ok = v, ok
    if best_ok >= 0.95:
        return best, True
    return s, False


converted = []
for c in data.columns:
    if c == CFG.ID_COL:
        continue
    new, did = try_numeric(data[c])
    if did:
        data[c] = new
        converted.append(c)
        if test is not None and c in test.columns:
            test[c] = try_numeric(test[c])[0]
if converted:
    print("Kolom string -> numerik:", converted)
for c in data.columns:
    if pd.api.types.is_bool_dtype(data[c]):
        data[c] = data[c].astype(int)

rows = []
n = len(data)
for c in data.columns:
    s = data[c]
    nun = s.nunique(dropna=True)
    if c == CFG.ID_COL:
        role = "id"
    elif nun <= 1:
        role = "constant"
    elif pd.api.types.is_datetime64_any_dtype(s):
        role = "datetime"
    elif pd.api.types.is_numeric_dtype(s):
        role = "id-like" if (nun == n and "id" in str(c).lower()) else "numeric"
    else:
        parsed = pd.to_datetime(s.dropna().astype(str).head(200), errors="coerce")
        if parsed.notna().mean() > 0.9 and nun > 20:
            role = "datetime"
        elif nun == n:
            role = "id-like"
        elif nun > CFG.MAX_CAT_CARDINALITY:
            role = "text/high-card"
        else:
            role = "categorical"
    rows.append({"column": c, "dtype": str(s.dtype), "pct_missing": s.isna().mean() * 100, "n_unique": nun,
                 "example": str(s.dropna().iloc[0])[:30] if s.notna().any() else "", "role_guess": role})
DQ = pd.DataFrame(rows)
display(DQ.round(2))
save_table(DQ, "data_quality_report")
print(f"Duplikat baris penuh: {int(data.duplicated().sum())}")
if CFG.ID_COL and CFG.ID_COL in data.columns:
    print(f"Duplikat ID: {int(data[CFG.ID_COL].duplicated().sum())}" + ("  ⚠️ ID tidak unik -> agregasi per ID dulu?" if data[CFG.ID_COL].duplicated().any() else "  ✅"))

# %% [markdown]
# ## 5. Pemilihan fitur
#
# **Aturan otomatis (kalau `FEATURE_COLS=None`):** pakai kolom `numeric` + `categorical` dari data quality report, kecuali ID, `OUTCOME_COL`,
# `PROFILE_ONLY_COLS`, `EXCLUDE_COLS`, dan kolom dengan missing > `HIGH_MISSING_DROP`.
#
# **Tips memilih fitur untuk segmentasi bisnis:**
# - Pilih fitur yang **bisa ditindaklanjuti** dan menggambarkan **perilaku/nilai** (premi, tenure, jumlah klaim, frekuensi transaksi),
#   bukan fitur identitas (ID, nama, kode pos detail).
# - Fitur demografis (gender, region) sering lebih baik sebagai **PROFILE_ONLY** → segmen berbasis perilaku, lalu dilihat komposisi demografinya.
# - Hindari terlalu banyak fitur yang mirip (mis. 5 variasi pendapatan) → satu konsep jadi "berbobot ganda". `DROP_CORRELATED` membantu.
# - **Outcome (churn) bukan fitur** — kalau dimasukkan, cluster akan "dipaksa" memisahkan churn vs non-churn dan profilnya bias.

# %%
excl = set([CFG.ID_COL, CFG.OUTCOME_COL] + list(CFG.EXCLUDE_COLS) + list(CFG.PROFILE_ONLY_COLS)) - {None}
if CFG.FEATURE_COLS:
    FEATURES = [c for c in CFG.FEATURE_COLS if c in data.columns and c not in excl]
    missing_feat = [c for c in CFG.FEATURE_COLS if c not in data.columns]
    if missing_feat:
        print("⚠️ FEATURE_COLS tidak ada di data (diabaikan):", missing_feat)
    if CFG.OUTCOME_COL in (CFG.FEATURE_COLS or []):
        print("⚠️ OUTCOME_COL ada di FEATURE_COLS -> dibuang dari fitur (outcome hanya untuk analisis setelah clustering).")
else:
    FEATURES = DQ.loc[DQ["role_guess"].isin(["numeric", "categorical"]), "column"].tolist()
    FEATURES = [c for c in FEATURES if c not in excl]
dropped_missing = [c for c in FEATURES if data[c].isna().mean() > CFG.HIGH_MISSING_DROP]
FEATURES = [c for c in FEATURES if c not in dropped_missing]
if dropped_missing:
    print("Dibuang karena missing tinggi:", dropped_missing)
NUM_COLS = [c for c in FEATURES if pd.api.types.is_numeric_dtype(data[c])]
CAT_COLS = [c for c in FEATURES if c not in NUM_COLS]
if CFG.CAT_ENCODING == "none":
    print("CAT_ENCODING='none' -> kategorikal hanya untuk profil:", CAT_COLS)
    CFG.PROFILE_ONLY_COLS = list(CFG.PROFILE_ONLY_COLS) + CAT_COLS
    CAT_COLS = []
PROFILE_COLS = [c for c in CFG.PROFILE_ONLY_COLS if c in data.columns]
assert len(NUM_COLS) + len(CAT_COLS) >= 1, "Tidak ada fitur untuk clustering. Set CFG.FEATURE_COLS."
print(f"Fitur numerik ({len(NUM_COLS)}): {NUM_COLS}")
print(f"Fitur kategorikal ({len(CAT_COLS)}): {CAT_COLS}")
print(f"Kolom profil saja: {PROFILE_COLS}")
print(f"Outcome: {CFG.OUTCOME_COL} | Value: {CFG.VALUE_COL}")
if CFG.OUTCOME_COL and CFG.OUTCOME_COL not in data.columns:
    print(f"⚠️ OUTCOME_COL '{CFG.OUTCOME_COL}' tidak ada di data -> analisis outcome dilewati")
    CFG.OUTCOME_COL = None
if CFG.VALUE_COL and CFG.VALUE_COL not in data.columns:
    print(f"⚠️ VALUE_COL '{CFG.VALUE_COL}' tidak ada di data -> analisis risk x value dilewati")
    CFG.VALUE_COL = None
IS_MIXED = len(NUM_COLS) > 0 and len(CAT_COLS) > 0

# %% [markdown]
# ## 6. EDA singkat untuk clustering
#
# **Cara membaca:**
# - **Histogram**: fitur sangat skewed (ekor panjang ke kanan, mis. premi/monetary) → log1p otomatis (`LOG_SKEWED`). Tanpa log, outlier
#   mendominasi jarak dan KMeans membuat cluster "1 orang kaya".
# - **Korelasi**: pasangan |r| > 0.95 → satu dibuang (`DROP_CORRELATED`), supaya satu konsep tidak dihitung dua kali.
#
# **Untuk laporan:** cukup 1 figure distribusi fitur utama + kalimat tentang transformasi yang dilakukan.

# %%
plot_cols = NUM_COLS[:CFG.MAX_PLOT_FEATURES]
if plot_cols:
    ncol = 4
    nrow = int(math.ceil(len(plot_cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), plot_cols):
        v = data[c].dropna()
        ax.hist(v, bins=40, color="tab:blue", alpha=0.8)
        ax.set_title(f"{c} (skew={v.skew():.2f})", fontsize=9)
        ax.set_xlabel(c)
        ax.set_ylabel("Count")
    for ax in axes.ravel()[len(plot_cols):]:
        ax.axis("off")
    plt.suptitle("Numeric feature distributions", y=1.01)
    plt.tight_layout()
    save_fig("numeric_distributions")
if len(NUM_COLS) >= 2:
    corr = data[NUM_COLS].corr()
    fig, ax = plt.subplots(figsize=(min(1 + 0.6 * len(NUM_COLS), 14), min(1 + 0.5 * len(NUM_COLS), 12)))
    if sns is not None:
        sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=len(NUM_COLS) <= 12, fmt=".2f", ax=ax)
    else:
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im)
        ax.set_xticks(range(len(NUM_COLS)))
        ax.set_xticklabels(NUM_COLS, rotation=90)
        ax.set_yticks(range(len(NUM_COLS)))
        ax.set_yticklabels(NUM_COLS)
    ax.set_title("Correlation between numeric features")
    save_fig("correlation_heatmap")
for c in CAT_COLS[:6]:
    vc = data[c].fillna("Missing").astype(str).value_counts()
    print(f"{c}: {len(vc)} kategori | top: {vc.head(5).to_dict()}")

# %% [markdown]
# ## 7. Preprocessing
#
# Urutan (semua parameter di-*fit* pada data utama lalu disimpan, sehingga data baru/test bisa ditransform dengan cara SAMA):
# 1. **Imputasi** numerik (median) & kategorikal (`"Missing"` sebagai kategori sendiri — missing bisa informatif).
# 2. **Clip outlier** ke kuantil 1%–99% (`CLIP_QUANTILES`).
# 3. **log1p** untuk fitur skewed (signed log untuk nilai negatif).
# 4. **Buang fitur numerik yang sangat berkorelasi** (`DROP_CORRELATED`; hanya kalau fitur dipilih otomatis — kalau `FEATURE_COLS` diisi manual cukup diberi peringatan).
# 5. **Scaling**:
#
# | Scaler | Kapan | Catatan |
# |---|---|---|
# | `standard` | default, distribusi kira-kira simetris setelah log | mean 0, std 1 |
# | `robust` | masih banyak outlier | pakai median & IQR |
# | `minmax` | fitur sudah terbatas (skor 1–5, persentase) | sensitif outlier |
# | `quantile` | distribusi sangat aneh/multimodal | mengubah ke distribusi normal; jarak jadi berbasis rank |
#
# 6. **Kategorikal**: one-hot × `CAT_WEIGHT` (supaya 1 variabel kategorikal tidak mendominasi jarak), kategori langka (< 1%) → `"Other"`.
#    Alternatif untuk data campuran kecil: **Gower distance** (dihitung manual di numpy) dipakai algoritma `kmedoids_gower` / `agglo_gower`.
#
# **Jika X maka Y:** cluster yang hasilnya hanya memisahkan satu kategori (mis. region) → turunkan `CAT_WEIGHT` (0.3) atau jadikan kolom itu
# `PROFILE_ONLY_COLS`. Cluster didominasi satu fitur numerik → cek apakah fitur itu masih skewed/berskala besar.

# %%
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    """Preprocessing yang bisa di-fit sekali lalu dipakai untuk data baru (test)."""

    def fit(self, df_):
        self.num_cols, self.cat_cols = list(NUM_COLS), list(CAT_COLS)
        num = df_[self.num_cols].astype(float) if self.num_cols else pd.DataFrame(index=df_.index)
        self.impute = (num.median() if CFG.NUM_IMPUTE == "median" else num.mean()) if self.num_cols else pd.Series(dtype=float)
        num = num.fillna(self.impute)
        if CFG.CLIP_QUANTILES and self.num_cols:
            self.lo, self.hi = num.quantile(CFG.CLIP_QUANTILES[0]), num.quantile(CFG.CLIP_QUANTILES[1])
        else:
            self.lo = self.hi = None
        if self.lo is not None:
            num = num.clip(self.lo, self.hi, axis=1)
        self.log_cols = [c for c in self.num_cols if CFG.LOG_SKEWED and abs(num[c].skew()) > CFG.SKEW_THRESHOLD] if self.num_cols else []
        for c in self.log_cols:
            num[c] = signed_log1p(num[c])
        self.drop_corr = []
        if CFG.DROP_CORRELATED and len(self.num_cols) > 1:
            cm = num.corr().abs()
            upper = cm.where(np.triu(np.ones(cm.shape, dtype=bool), k=1))
            high = [c for c in upper.columns if (upper[c] > CFG.DROP_CORRELATED).any()]
            if CFG.FEATURE_COLS is None:      # fitur otomatis -> buang yang redundan
                self.drop_corr = high
            elif high:                        # fitur dipilih manual -> hormati pilihan user, cukup peringatan
                print(f"⚠️ Fitur sangat berkorelasi (>|{CFG.DROP_CORRELATED}|): {high} -> tidak dibuang karena FEATURE_COLS "
                      "diisi manual; konsep yang sama akan berbobot ganda.")
        self.num_keep = [c for c in self.num_cols if c not in self.drop_corr]
        sc = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler(),
              "quantile": QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(df_)),
                                              random_state=CFG.SEED)}.get(CFG.SCALER, StandardScaler())
        self.scaler = sc.fit(num[self.num_keep].values) if self.num_keep else None
        self.cat_levels = {}
        if self.cat_cols:
            cat = self._cat_clean(df_, fit=True)
            self.ohe = make_ohe().fit(cat)
            try:
                self.ohe_names = list(self.ohe.get_feature_names_out(self.cat_cols))
            except Exception:
                self.ohe_names = [f"{c}_{v}" for c, cats in zip(self.cat_cols, self.ohe.categories_) for v in cats]
        else:
            self.ohe, self.ohe_names = None, []
        self.feature_names = self.num_keep + self.ohe_names
        return self

    def _cat_clean(self, df_, fit=False):
        cat = df_[self.cat_cols].copy()
        for c in self.cat_cols:
            s = cat[c].astype(object).where(cat[c].notna(), "Missing").astype(str)
            if fit:
                freq = s.value_counts(normalize=True)
                self.cat_levels[c] = set(freq[freq >= CFG.RARE_MIN_FREQ].index)
            s = s.where(s.isin(self.cat_levels[c]), "Other")
            cat[c] = s
        return cat

    def num_clean(self, df_, log=True):
        """Numerik setelah imputasi + clip (+log). Dipakai Gower dan profil."""
        num = df_[self.num_cols].apply(pd.to_numeric, errors="coerce").astype(float).fillna(self.impute)
        if self.lo is not None:
            num = num.clip(self.lo, self.hi, axis=1)
        if log:
            for c in self.log_cols:
                num[c] = signed_log1p(num[c])
        return num

    def transform(self, df_, cat_weight=None):
        cw = CFG.CAT_WEIGHT if cat_weight is None else cat_weight
        parts = []
        if self.num_keep:
            parts.append(self.scaler.transform(self.num_clean(df_)[self.num_keep].values))
        if self.cat_cols:
            parts.append(self.ohe.transform(self._cat_clean(df_)) * cw)
        return np.hstack(parts).astype(float)

    def cat_codes(self, df_):
        """Kode integer kategorikal (untuk Gower / K-Prototypes)."""
        cat = self._cat_clean(df_)
        return np.column_stack([pd.Categorical(cat[c], categories=sorted(self.cat_levels[c] | {"Other"})).codes
                                for c in self.cat_cols]) if self.cat_cols else None


PREP = Preprocessor().fit(data)
X = PREP.transform(data)
FEATURE_NAMES = PREP.feature_names
print(f"Matriks fitur X: {X.shape} | numerik dipakai: {len(PREP.num_keep)} | one-hot: {len(PREP.ohe_names)}")
print("log1p diterapkan ke:", PREP.log_cols or "-")
print("Dibuang karena korelasi tinggi:", PREP.drop_corr or "-")
print("NaN di X:", int(np.isnan(X).sum()), "| inf:", int(np.isinf(X).sum()))
assert np.isfinite(X).all(), "X mengandung NaN/inf -> cek preprocessing"
const = [f for f, s in zip(FEATURE_NAMES, X.std(axis=0)) if s < 1e-9]
if const:
    print("⚠️ Fitur konstan setelah preprocessing (tidak berguna):", const[:10])
N = len(X)

# %% [markdown]
# ## 8. Reduksi dimensi: PCA, t-SNE, UMAP
#
# - **PCA** (linear): *explained variance* menunjukkan berapa komponen yang dibutuhkan. **Loadings** = kontribusi fitur ke tiap komponen
#   → membantu memberi arti komponen (mis. PC1 = "nilai pelanggan": premi & tenure tinggi).
# - **t-SNE / UMAP** (non-linear): **hanya untuk visualisasi**. Jarak antar cluster & ukuran cluster di plot t-SNE tidak bermakna;
#   yang dilihat: apakah titik-titik warna sama berkumpul.
#
# **Jika X maka Y:** PC1+PC2 menjelaskan > 60% varians → scatter PCA 2D cukup representatif. Fitur banyak (> 30) & berkorelasi →
# `USE_PCA_FOR_CLUSTERING=True` (cluster di ruang PCA, kurangi noise).
#
# **Untuk laporan:** scatter PCA/t-SNE berwarna cluster (dibuat di bagian 13) + 1 kalimat interpretasi PC1/PC2 dari loadings.

# %%
pca_full = PCA(random_state=CFG.SEED).fit(X)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)
N_PCA = int(np.searchsorted(cum, CFG.PCA_VARIANCE) + 1)
N_PCA = max(2, min(N_PCA, X.shape[1]))
fig, ax = plt.subplots(figsize=(10, 4))
kshow = min(len(evr), 20)
ax.bar(range(1, kshow + 1), evr[:kshow], color="tab:blue", alpha=0.7, label="individual")
ax.plot(range(1, kshow + 1), cum[:kshow], marker="o", color="tab:red", label="cumulative")
ax.axhline(CFG.PCA_VARIANCE, color="gray", ls="--", lw=0.8)
ax.set_title(f"PCA explained variance ({N_PCA} components reach {CFG.PCA_VARIANCE:.0%})")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.legend()
save_fig("pca_explained_variance")
print(f"PC1 = {evr[0]:.1%}, PC2 = {evr[1] if len(evr) > 1 else 0:.1%}, komponen untuk {CFG.PCA_VARIANCE:.0%} varians = {N_PCA}")

n_load = min(5, X.shape[1])
load = pd.DataFrame(pca_full.components_[:n_load].T, index=FEATURE_NAMES, columns=[f"PC{i + 1}" for i in range(n_load)])
top_load = load.loc[load.abs().max(axis=1).sort_values(ascending=False).index[:min(25, len(load))]]
fig, ax = plt.subplots(figsize=(1.6 * n_load + 3, 0.35 * len(top_load) + 1.5))
if sns is not None:
    sns.heatmap(top_load, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
else:
    ax.imshow(top_load.values, cmap="RdBu_r", aspect="auto")
ax.set_title("PCA loadings (top features)")
ax.set_xlabel("Component")
ax.set_ylabel("Feature")
save_fig("pca_loadings")
for pc in load.columns[:2]:
    s = load[pc].sort_values()
    print(f"{pc}: (+) {', '.join(s.index[-3:][::-1])} | (-) {', '.join(s.index[:3])}")
save_table(load.reset_index().rename(columns={"index": "feature"}), "pca_loadings")

X_PCA2 = pca_full.transform(X)[:, :2]
XC = pca_full.transform(X)[:, :N_PCA] if CFG.USE_PCA_FOR_CLUSTERING else X
print("Ruang clustering:", f"PCA {N_PCA} komponen" if CFG.USE_PCA_FOR_CLUSTERING else f"fitur asli ({X.shape[1]} dim)")

EMB = {}
EMB_IDX = RNG.choice(N, size=min(N, CFG.TSNE_SAMPLE), replace=False)
if CFG.RUN_TSNE and N >= 10:
    try:
        t0 = time.time()
        perp = float(min(30, max(5, (len(EMB_IDX) - 1) / 3.5)))
        EMB["t-SNE"] = make_tsne(perplexity=perp, seed=CFG.SEED).fit_transform(XC[EMB_IDX])
        print(f"t-SNE selesai ({len(EMB_IDX)} titik, {time.time() - t0:.1f}s)")
    except Exception as e:
        print("t-SNE gagal:", e)
if CFG.RUN_UMAP and HAS_UMAP:
    try:
        EMB["UMAP"] = umap_mod.UMAP(n_components=2, random_state=CFG.SEED).fit_transform(XC[EMB_IDX])
        print("UMAP selesai")
    except Exception as e:
        print("UMAP gagal:", e)
elif CFG.RUN_UMAP:
    print("UMAP dilewati (umap-learn tidak tersedia).")

# %% [markdown]
# ## 9. Memilih jumlah cluster K
#
# | Metode | Cara membaca | Pilih K |
# |---|---|---|
# | **Elbow (inertia)** | inertia turun saat K naik; cari "siku" | knee otomatis = titik terjauh dari garis lurus awal-akhir |
# | **Silhouette** (−1..1) | > 0.5 struktur kuat, 0.25–0.5 sedang, < 0.25 lemah | maksimum |
# | **Calinski-Harabasz** | rasio separasi antar / di dalam cluster | maksimum |
# | **Davies-Bouldin** | rata-rata kemiripan cluster dengan cluster terdekat | **minimum** |
# | **Gap statistic** | dibanding data acak tanpa struktur | K terkecil dengan gap(K) ≥ gap(K+1) − s(K+1) |
# | **BIC (GMM)** | likelihood − penalti kompleksitas | minimum (atau siku) |
#
# **Logika rekomendasi:** setiap metode memberi 1 "vote"; K dengan vote terbanyak dipilih (seri → silhouette tertinggi), dengan syarat
# cluster terkecil ≥ `MIN_CLUSTER_PCT` (cluster terlalu kecil tidak bisa ditindaklanjuti bisnis).
#
# **Jika X maka Y:**
# - Metrik tidak sepakat (wajar!) → pilih K yang **paling bisa dijelaskan** secara bisnis di antara 2–3 kandidat; cek profil masing-masing.
# - Silhouette < 0.25 untuk semua K → data tidak punya cluster alami yang jelas; segmentasi tetap berguna sebagai "partisi"
#   tapi sebut di laporan & tekankan stabilitas + kebermaknaan bisnis.
# - Ingin memaksa K (mis. bisnis minta 4 segmen) → `CFG.N_CLUSTERS = 4`.
#
# **Untuk laporan:** figure 6 panel di bawah + tabel ringkas, dan kalimat *"K=4 dipilih karena didukung elbow, silhouette, dan DB,
# serta setiap segmen ≥ 5% pelanggan."*

# %%
K_LIST = [k for k in range(max(2, CFG.K_MIN), CFG.K_MAX + 1) if k < N]
SAMPLE_IDX = RNG.choice(N, size=min(N, CFG.METRIC_SAMPLE), replace=False)
XS = XC[SAMPLE_IDX]
GMM_COV = "full" if XC.shape[1] <= 15 else "diag"
krows = []
for k in K_LIST:
    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=CFG.KMEANS_N_INIT, random_state=CFG.SEED).fit(XC)
    sc = internal_scores(XC, km.labels_, sample=CFG.METRIC_SAMPLE)
    sizes = np.bincount(km.labels_, minlength=k) / N
    r = {"k": k, "inertia": km.inertia_, **sc, "min_cluster_pct": sizes.min() * 100}
    try:
        g = GaussianMixture(n_components=k, covariance_type=GMM_COV, random_state=CFG.SEED, n_init=1, reg_covar=1e-4).fit(XS)
        r["gmm_bic"], r["gmm_aic"] = g.bic(XS), g.aic(XS)
    except Exception:
        r["gmm_bic"] = r["gmm_aic"] = np.nan
    r["time_s"] = round(time.time() - t0, 2)
    krows.append(r)
KTAB = pd.DataFrame(krows)


def gap_statistic(Xs, ks, B=5, seed=42):
    """Gap statistic (Tibshirani 2001) dengan referensi uniform di bounding box."""
    rng = np.random.RandomState(seed)
    mins, maxs = Xs.min(axis=0), Xs.max(axis=0)
    gaps, sks = [], []
    for k in ks:
        wk = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(Xs).inertia_
        ref = []
        for b in range(B):
            Xr = rng.uniform(mins, maxs, size=Xs.shape)
            ref.append(np.log(KMeans(n_clusters=k, n_init=2, random_state=seed + b).fit(Xr).inertia_ + 1e-12))
        ref = np.array(ref)
        gaps.append(ref.mean() - np.log(wk + 1e-12))
        sks.append(ref.std() * np.sqrt(1 + 1.0 / B))
    return np.array(gaps), np.array(sks)


if CFG.RUN_GAP and len(K_LIST) >= 2:
    t0 = time.time()
    _gx = XS[:min(len(XS), 2000)]
    KTAB["gap"], KTAB["gap_sk"] = gap_statistic(_gx, K_LIST, B=CFG.GAP_B, seed=CFG.SEED)
    print(f"Gap statistic selesai ({time.time() - t0:.1f}s)")

votes = {}
ok_k = KTAB.loc[KTAB["min_cluster_pct"] >= CFG.MIN_CLUSTER_PCT * 100, "k"].tolist() or KTAB["k"].tolist()
kt = KTAB[KTAB["k"].isin(ok_k)]
votes["elbow_knee"] = int(knee_point(KTAB["k"], KTAB["inertia"]))
if kt["silhouette"].notna().any():
    votes["silhouette"] = int(kt.loc[kt["silhouette"].idxmax(), "k"])
if kt["calinski_harabasz"].notna().any():
    votes["calinski_harabasz"] = int(kt.loc[kt["calinski_harabasz"].idxmax(), "k"])
if kt["davies_bouldin"].notna().any():
    votes["davies_bouldin"] = int(kt.loc[kt["davies_bouldin"].idxmin(), "k"])
if KTAB["gmm_bic"].notna().any():
    bic_knee = int(knee_point(KTAB["k"], KTAB["gmm_bic"])) if KTAB["gmm_bic"].is_monotonic_decreasing else int(KTAB.loc[KTAB["gmm_bic"].idxmin(), "k"])
    votes["gmm_bic"] = bic_knee
if "gap" in KTAB:
    g, s = KTAB["gap"].values, KTAB["gap_sk"].values
    cand = [KTAB["k"].iloc[i] for i in range(len(g) - 1) if g[i] >= g[i + 1] - s[i + 1]]
    votes["gap"] = int(cand[0]) if cand else int(KTAB.loc[KTAB["gap"].idxmax(), "k"])
vc = Counter(votes.values())
top_n = max(vc.values())
tied = [k for k, v in vc.items() if v == top_n]
sil_map = KTAB.set_index("k")["silhouette"]
K_RECOMMENDED = max(tied, key=lambda k: (k in ok_k, sil_map.get(k, -1)))
if K_RECOMMENDED not in ok_k:
    K_RECOMMENDED = int(kt.loc[kt["silhouette"].idxmax(), "k"]) if kt["silhouette"].notna().any() else ok_k[0]
K = int(CFG.N_CLUSTERS or K_RECOMMENDED)
display(KTAB.round(4))
save_table(KTAB, "choose_k_metrics")
print("Votes per metode:", votes)
print(f"==> K rekomendasi = {K_RECOMMENDED} (vote {vc.get(K_RECOMMENDED, 0)}/{len(votes)})"
      + (f" | dipakai K = {K} (CFG.N_CLUSTERS)" if CFG.N_CLUSTERS else ""))
_sil = sil_map.get(K, np.nan)
print(f"Silhouette K={K}: {_sil:.3f} -> " + ("struktur kuat" if _sil > 0.5 else "struktur sedang" if _sil > 0.25 else
                                             "struktur lemah (cluster tidak terpisah tegas; tekankan stabilitas & makna bisnis)"))

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
pan = [("inertia", "Elbow: inertia (WCSS)", votes.get("elbow_knee")), ("silhouette", "Silhouette (higher better)", votes.get("silhouette")),
       ("calinski_harabasz", "Calinski-Harabasz (higher better)", votes.get("calinski_harabasz")),
       ("davies_bouldin", "Davies-Bouldin (lower better)", votes.get("davies_bouldin")),
       ("gmm_bic", "GMM BIC (lower better)", votes.get("gmm_bic")), ("gap", "Gap statistic", votes.get("gap"))]
for ax, (col, title, kv) in zip(axes.ravel(), pan):
    if col in KTAB and KTAB[col].notna().any():
        ax.plot(KTAB["k"], KTAB[col], marker="o")
        if col == "gap":
            ax.errorbar(KTAB["k"], KTAB["gap"], yerr=KTAB["gap_sk"], fmt="none", ecolor="gray")
        if col == "gmm_bic":
            ax.plot(KTAB["k"], KTAB["gmm_aic"], marker="s", ls="--", label="AIC")
            ax.legend()
        if kv is not None:
            ax.axvline(kv, color="red", ls="--", lw=1)
        ax.axvline(K, color="green", ls=":", lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("Number of clusters K")
        ax.set_ylabel(col)
    else:
        ax.axis("off")
plt.suptitle(f"Choosing K (red = metric's choice, green = selected K={K})", y=1.01)
plt.tight_layout()
save_fig("choose_k")

# %% [markdown]
# ## 10. Algoritma clustering & perbandingan
#
# | Algoritma | Ide | Kelebihan | Kekurangan |
# |---|---|---|---|
# | **kmeans** (k-means++, `n_init` kali) | centroid, minimalkan jarak kuadrat | cepat, mudah dijelaskan, bisa `predict` data baru | asumsi cluster bulat & ukuran mirip |
# | **minibatch** | KMeans pakai mini-batch | sangat cepat untuk data besar (>100k) | sedikit kurang akurat |
# | **gmm** | campuran Gaussian (soft clustering) | cluster elips, probabilitas keanggotaan | sensitif inisialisasi, butuh data cukup |
# | **agglomerative** (Ward) | gabung bertahap dari bawah | dendrogram informatif | O(n²) memori → sampel kalau besar |
# | **dbscan** | kepadatan (eps, min_samples) | bentuk bebas, deteksi **noise (-1)** | eps sulit; gagal kalau kepadatan beragam |
# | **hdbscan** | DBSCAN hierarkis | tidak perlu eps, kepadatan beragam | bisa banyak noise; jumlah cluster otomatis |
# | **kprototypes** | KMeans (numerik) + KModes (kategorikal) | data campuran | butuh `kmodes`; fallback = KMeans pada numerik+one-hot berbobot gamma |
# | **kmedoids_gower / agglo_gower** | jarak Gower (campuran numerik+kategorikal) | paling tepat untuk data campuran | O(n²) → sampel `GOWER_MAX_N` |
#
# **Cara membaca tabel perbandingan:** silhouette/CH/DB dihitung di ruang fitur yang sama (`XC`) supaya adil; `noise_pct` (DBSCAN/HDBSCAN)
# = % data yang tidak masuk cluster; `min_cluster_pct` = ukuran cluster terkecil.
#
# **Aturan pemilihan otomatis (`FINAL_ALGO="auto"`):** kandidat harus punya ≥ 2 cluster, noise ≤ 20%, cluster terkecil ≥ `MIN_CLUSTER_PCT`;
# pilih silhouette tertinggi, **tapi KMeans diutamakan kalau selisih silhouette ≤ 0.02** (lebih sederhana, stabil, bisa assign data baru).

# %%
ALGO = OrderedDict()


def register(name, labels, t0, model=None, space="XC", extra=None):
    labels = np.asarray(labels).astype(int)
    ALGO[name] = {"labels": labels, "model": model, "space": space, "time_s": round(time.time() - t0, 2), **(extra or {})}
    k_found = len(set(labels) - {-1})
    print(f"  {name:<16s} clusters={k_found:<3d} noise={np.mean(labels == -1):6.1%}  ({ALGO[name]['time_s']}s)")


def subsample_fit_assign(fit_fn, Xspace, max_n, seed):
    """Fit algoritma O(n^2) di sampel lalu assign sisanya ke centroid terdekat."""
    if len(Xspace) <= max_n:
        return fit_fn(Xspace), None
    idx = np.random.RandomState(seed).choice(len(Xspace), size=max_n, replace=False)
    lab_s = fit_fn(Xspace[idx])
    lab, _ = nearest_centroid_assign(Xspace[idx], lab_s, Xspace)
    lab[idx] = lab_s
    return lab, idx


algos = list(dict.fromkeys(CFG.ALGORITHMS))
print(f"K = {K} | data n = {N} | ruang = {XC.shape}")
for name in algos:
    t0 = time.time()
    try:
        if name == "kmeans":
            m = KMeans(n_clusters=K, n_init=CFG.KMEANS_N_INIT, random_state=CFG.SEED).fit(XC)
            register(name, m.labels_, t0, m)
        elif name == "minibatch":
            m = MiniBatchKMeans(n_clusters=K, batch_size=min(2048, N), n_init=3, random_state=CFG.SEED).fit(XC)
            register(name, m.predict(XC), t0, m)
        elif name == "gmm":
            m = GaussianMixture(n_components=K, covariance_type=GMM_COV, n_init=3, random_state=CFG.SEED, reg_covar=1e-4).fit(XC)
            register(name, m.predict(XC), t0, m, extra={"proba_max": m.predict_proba(XC).max(axis=1)})
        elif name == "agglomerative":
            lab, idx = subsample_fit_assign(lambda Z: make_agglo(K, "ward").fit_predict(Z), XC, CFG.AGGLO_MAX_N, CFG.SEED)
            register(name, lab, t0)
            if HAS_SCIPY:
                di = RNG.choice(N, size=min(N, 1500), replace=False)
                Z = sch_linkage(XC[di], method="ward")
                fig, ax = plt.subplots(figsize=(12, 4.5))
                sch_dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, color_threshold=Z[-(K - 1), 2] if K > 1 else None)
                ax.set_title(f"Ward dendrogram (sample {len(di)}; cut for K={K})")
                ax.set_xlabel("Cluster (size)")
                ax.set_ylabel("Ward distance")
                save_fig("dendrogram")
        elif name == "dbscan":
            ms = int(CFG.DBSCAN_MIN_SAMPLES or min(50, max(5, 2 * XC.shape[1])))
            nn = NearestNeighbors(n_neighbors=ms).fit(XS)
            kd = np.sort(nn.kneighbors(XS)[0][:, -1])
            # eps otomatis = knee kurva k-distance (titik terjauh dari garis lurus ujung-ke-ujung)
            eps = float(CFG.DBSCAN_EPS or kd[int(knee_point(np.arange(len(kd)), kd))])
            eps = max(eps, 1e-6)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(kd)
            ax.axhline(eps, color="red", ls="--", label=f"eps = {eps:.3f}")
            ax.set_title(f"k-distance plot (k = min_samples = {ms})")
            ax.set_xlabel("Points sorted by distance")
            ax.set_ylabel(f"Distance to {ms}-th neighbor")
            ax.legend()
            save_fig("dbscan_k_distance")
            lab, _ = subsample_fit_assign(lambda Z: DBSCAN(eps=eps, min_samples=ms).fit_predict(Z), XC, 20000, CFG.SEED)
            register(name, lab, t0, extra={"eps": eps, "min_samples": ms})
        elif name == "hdbscan":
            mcs = int(CFG.HDBSCAN_MIN_CLUSTER_SIZE or max(10, int(0.02 * N)))
            if HAS_HDBSCAN_PKG:
                fit_fn = lambda Z: hdbscan_pkg.HDBSCAN(min_cluster_size=mcs).fit_predict(Z)
            elif HAS_SK_HDBSCAN:
                fit_fn = lambda Z: SkHDBSCAN(min_cluster_size=mcs).fit_predict(Z)
            else:
                print("  hdbscan: tidak ada package hdbscan maupun sklearn>=1.3 -> skip")
                continue
            lab, _ = subsample_fit_assign(fit_fn, XC, 20000, CFG.SEED)
            register(name, lab, t0, extra={"min_cluster_size": mcs, "impl": "hdbscan" if HAS_HDBSCAN_PKG else "sklearn"})
        elif name == "kprototypes":
            if not CAT_COLS:
                print("  kprototypes: tidak ada fitur kategorikal -> sama dengan kmeans, skip")
                continue
            if HAS_KMODES and PREP.num_keep:
                num_s = PREP.scaler.transform(PREP.num_clean(data)[PREP.num_keep].values)
                mix = np.column_stack([num_s, PREP.cat_codes(data)]).astype(object)
                cat_idx = list(range(num_s.shape[1], mix.shape[1]))
                ns = min(N, 3000 if FAST else 10000)
                idx = RNG.choice(N, size=ns, replace=False)
                kp = KPrototypes(n_clusters=K, init="Cao", n_init=3, random_state=CFG.SEED).fit(mix[idx], categorical=cat_idx)
                register(name, kp.predict(mix, categorical=cat_idx), t0, kp, extra={"impl": "kmodes"})
            else:
                # Fallback: KMeans pada [numerik terskala, one-hot * sqrt(gamma/2)], gamma = 0.5 * rata-rata std numerik (heuristik Huang)
                gamma = 0.5 * (np.std(X[:, :len(PREP.num_keep)], axis=0).mean() if PREP.num_keep else 1.0)
                Xkp = PREP.transform(data, cat_weight=math.sqrt(gamma / 2))
                m = KMeans(n_clusters=K, n_init=CFG.KMEANS_N_INIT, random_state=CFG.SEED).fit(Xkp)
                print("  kprototypes: kmodes tidak terinstall -> FALLBACK KMeans pada numerik + one-hot berbobot gamma")
                register(name, m.labels_, t0, m, space="XKP", extra={"impl": "fallback", "cat_weight": math.sqrt(gamma / 2)})
        elif name in ("kmedoids_gower", "agglo_gower"):
            if not CAT_COLS:
                print(f"  {name}: tidak ada kategorikal -> Gower tidak diperlukan, skip")
                continue
            if "GOWER" not in globals():
                gi = np.sort(RNG.choice(N, size=min(N, CFG.GOWER_MAX_N), replace=False))
                g_num = PREP.num_clean(data).values if PREP.num_cols else None
                g_cat = PREP.cat_codes(data)
                G_RANGES = (np.nanmax(g_num, axis=0) - np.nanmin(g_num, axis=0)) if g_num is not None else None
                GOWER = {"idx": gi, "num": g_num, "cat": g_cat, "ranges": G_RANGES,
                         "D": gower_matrix(g_num[gi] if g_num is not None else None, g_cat[gi], ranges=G_RANGES)}
                print(f"  Gower matrix {GOWER['D'].shape} dihitung ({time.time() - t0:.1f}s)")
            gi, D = GOWER["idx"], GOWER["D"]
            if name == "kmedoids_gower":
                lab_s, med_s, _ = kmedoids_simple(D, K, n_init=3 if FAST else 5, seed=CFG.SEED)
                medoids = gi[med_s]
            else:
                lab_s = make_agglo(K, "average", "precomputed").fit_predict(D)
                medoids = np.array([gi[np.where(lab_s == c)[0][np.argmin(D[np.ix_(lab_s == c, lab_s == c)].sum(axis=1))]]
                                    for c in range(K)])
            gn, gc = GOWER["num"], GOWER["cat"]
            Dm = gower_matrix(gn if gn is not None else None, gc, gn[medoids] if gn is not None else None, gc[medoids],
                              ranges=GOWER["ranges"])
            lab = np.argmin(Dm, axis=1)
            lab[gi] = lab_s
            register(name, lab, t0, space="gower", extra={"medoids": medoids,
                                                          "sil_gower": safe_silhouette(D, lab_s, metric="precomputed")})
        else:
            print(f"  ⚠️ algoritma '{name}' tidak dikenal -> skip")
    except Exception as e:
        FAILED_MODELS[name] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  ⚠️ {name} gagal: {type(e).__name__}: {str(e)[:200]}")

assert ALGO, "Semua algoritma gagal. Lihat FAILED_MODELS."
crow = []
for name, r in ALGO.items():
    lab = r["labels"]
    valid = lab[lab != -1]
    sizes = np.bincount(valid) if len(valid) else np.array([0])
    sizes = sizes[sizes > 0]
    sc = internal_scores(XC, lab, sample=CFG.METRIC_SAMPLE)
    crow.append({"algorithm": name, "n_clusters": len(sizes), "noise_pct": float(np.mean(lab == -1) * 100),
                 "min_cluster_pct": float(sizes.min() / N * 100) if len(sizes) else 0.0, **sc,
                 "silhouette_gower": r.get("sil_gower", np.nan), "time_s": r["time_s"]})
CMP = pd.DataFrame(crow).sort_values("silhouette", ascending=False).reset_index(drop=True)
display(CMP.round(4))
save_table(CMP, "algorithm_comparison")

# %% [markdown]
# ## 11. Stabilitas cluster (Adjusted Rand Index)
#
# **ARI** mengukur kesamaan dua pembagian cluster (1 = identik, 0 = seperti acak). Dua uji:
# 1. **Subsample stability**: fit pada dua subsampel 80% berbeda, bandingkan label pada titik yang sama (diulang `STABILITY_BOOTSTRAPS` kali)
#    untuk K−1, K, K+1 → membantu memastikan pilihan K.
# 2. **Seed stability**: KMeans dengan seed berbeda (inisialisasi berbeda).
#
# **Aturan:** ARI rata-rata > 0.8 = **stabil** ✅, 0.6–0.8 = cukup, < 0.6 = **tidak stabil** ⚠️ (segmen bisa berubah kalau data sedikit berubah →
# kurangi K, kurangi fitur noise, atau pakai `USE_PCA_FOR_CLUSTERING`).
#
# **Untuk laporan:** *"Segmentasi stabil (ARI rata-rata 0.91 pada 20 subsampel), sehingga segmen bukan artefak acak."*

# %%
def refit_labels(name, Z, k, seed):
    """Fit ulang algoritma (yang bekerja di ruang XC) pada Z -> label."""
    if name in ("kmeans", "kprototypes", "kmedoids_gower", "agglo_gower"):
        return KMeans(n_clusters=k, n_init=max(2, CFG.KMEANS_N_INIT // 2), random_state=seed).fit_predict(Z)
    if name == "minibatch":
        return MiniBatchKMeans(n_clusters=k, batch_size=min(2048, len(Z)), n_init=3, random_state=seed).fit_predict(Z)
    if name == "gmm":
        return GaussianMixture(n_components=k, covariance_type=GMM_COV, random_state=seed, reg_covar=1e-4).fit(Z).predict(Z)
    if name == "agglomerative":
        return make_agglo(k, "ward").fit_predict(Z)
    if name == "dbscan":
        return DBSCAN(eps=ALGO["dbscan"]["eps"], min_samples=ALGO["dbscan"]["min_samples"]).fit_predict(Z)
    if name == "hdbscan":
        mcs = max(5, int(ALGO["hdbscan"]["min_cluster_size"] * len(Z) / N))
        return (hdbscan_pkg.HDBSCAN(min_cluster_size=mcs).fit_predict(Z) if HAS_HDBSCAN_PKG
                else SkHDBSCAN(min_cluster_size=mcs).fit_predict(Z))
    raise ValueError(name)


def subsample_stability(name, k, B, frac=0.8, max_n=None, seed=0):
    """Rata-rata & std ARI antar pasangan subsampel (dibandingkan pada titik yang ada di keduanya)."""
    rng = np.random.RandomState(seed)
    pool = SAMPLE_IDX if max_n is None else SAMPLE_IDX[:max_n]
    m = int(frac * len(pool))
    aris = []
    for b in range(B):
        a = rng.choice(pool, size=m, replace=False)
        c = rng.choice(pool, size=m, replace=False)
        la = pd.Series(refit_labels(name, XC[a], k, seed + b), index=a)
        lc = pd.Series(refit_labels(name, XC[c], k, seed + 100 + b), index=c)
        common = np.intersect1d(a, c)
        if len(common) > 10:
            aris.append(adjusted_rand_score(la.loc[common].values, lc.loc[common].values))
    return (float(np.mean(aris)), float(np.std(aris))) if aris else (np.nan, np.nan)


stab_rows = []
for k in sorted(set([max(2, K - 1), K, K + 1])):
    t0 = time.time()
    mu, sd = subsample_stability("kmeans", k, CFG.STABILITY_BOOTSTRAPS, seed=CFG.SEED)
    stab_rows.append({"algorithm": "kmeans", "k": k, "ari_mean": mu, "ari_std": sd, "time_s": round(time.time() - t0, 1)})
seed_labels = [KMeans(n_clusters=K, n_init=CFG.KMEANS_N_INIT, random_state=s).fit_predict(XS) for s in range(CFG.STABILITY_SEEDS)]
seed_ari = [adjusted_rand_score(seed_labels[i], seed_labels[j]) for i in range(len(seed_labels)) for j in range(i + 1, len(seed_labels))]
STAB = pd.DataFrame(stab_rows)
display(STAB.round(3))
print(f"Seed stability KMeans (K={K}, n_init={CFG.KMEANS_N_INIT}): ARI rata-rata = {np.mean(seed_ari):.3f}")
STAB_K = float(STAB.loc[STAB["k"] == K, "ari_mean"].iloc[0])
print(f"Subsample stability KMeans K={K}: ARI = {STAB_K:.3f} -> " +
      ("✅ stabil" if STAB_K > 0.8 else ("cukup" if STAB_K > 0.6 else "⚠️ tidak stabil")))
best_stab_k = int(STAB.loc[STAB["ari_mean"].idxmax(), "k"])
if best_stab_k != K and STAB["ari_mean"].max() - STAB_K > 0.1:
    print(f"Info: K={best_stab_k} jauh lebih stabil (ARI {STAB['ari_mean'].max():.2f}); pertimbangkan CFG.N_CLUSTERS={best_stab_k}.")
save_table(STAB, "stability_subsample")

names = list(ALGO)
if len(names) >= 2:
    A = np.array([[adjusted_rand_score(ALGO[a]["labels"], ALGO[b]["labels"]) for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(names), 0.7 * len(names) + 1))
    if sns is not None:
        sns.heatmap(pd.DataFrame(A, index=names, columns=names), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
    else:
        ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Agreement between algorithms (ARI)")
    save_fig("algorithm_agreement_ari")
    print("Info: ARI tinggi antar algoritma berbeda = struktur cluster nyata (bukan artefak satu algoritma).")

# %% [markdown]
# ## 12. Label cluster final
#
# Algoritma final dipilih (otomatis atau `CFG.FINAL_ALGO`), lalu nomor cluster diurutkan ulang supaya mudah dibaca:
# `SORT_CLUSTERS_BY="size"` → cluster 0 = terbesar; `"outcome"` → cluster 0 = outcome rate (mis. churn) tertinggi. Noise (DBSCAN/HDBSCAN) = -1.

# %%
if CFG.FINAL_ALGO in ALGO:
    FINAL_ALGO = CFG.FINAL_ALGO
else:
    if CFG.FINAL_ALGO not in ("auto", None):
        print(f"⚠️ FINAL_ALGO '{CFG.FINAL_ALGO}' tidak tersedia -> auto")
    elig = CMP[(CMP["n_clusters"] >= 2) & (CMP["noise_pct"] <= 20) & (CMP["min_cluster_pct"] >= CFG.MIN_CLUSTER_PCT * 100)]
    if len(elig) == 0:
        FINAL_ALGO = "kmeans" if "kmeans" in ALGO else CMP["algorithm"].iloc[0]
    else:
        FINAL_ALGO = elig.sort_values("silhouette", ascending=False)["algorithm"].iloc[0]
        if "kmeans" in set(elig["algorithm"]):
            s_best = float(elig["silhouette"].max())
            s_km = float(elig.loc[elig["algorithm"] == "kmeans", "silhouette"].iloc[0])
            if s_km >= s_best - 0.02:
                FINAL_ALGO = "kmeans"
raw_labels = ALGO[FINAL_ALGO]["labels"].copy()

OUTCOME_Y = None
if CFG.OUTCOME_COL:
    oc = data[CFG.OUTCOME_COL]
    uniq = pd.Series(oc.dropna().unique())
    if len(uniq) == 2 or CFG.OUTCOME_POSITIVE is not None:
        if CFG.OUTCOME_POSITIVE is not None:
            pos = CFG.OUTCOME_POSITIVE
        else:
            low = {str(u).strip().lower(): u for u in uniq}
            pos = next((low[k] for k in ["1", "1.0", "yes", "y", "true", "churn", "ya", "positive"] if k in low), None)
            pos = pos if pos is not None else sorted(uniq, key=lambda v: str(v))[-1]
        OUTCOME_Y = (oc.astype(str) == str(pos)).astype(float).where(oc.notna())
        OUTCOME_KIND = "binary"
        print(f"Outcome '{CFG.OUTCOME_COL}' biner; nilai positif = {pos!r}; rate keseluruhan = {OUTCOME_Y.mean():.2%}")
    else:
        OUTCOME_Y = pd.to_numeric(oc, errors="coerce")
        OUTCOME_KIND = "numeric"
        print(f"Outcome '{CFG.OUTCOME_COL}' numerik; mean keseluruhan = {OUTCOME_Y.mean():.4g}")

valid_c = [c for c in np.unique(raw_labels) if c != -1]
if CFG.SORT_CLUSTERS_BY == "outcome" and OUTCOME_Y is not None:
    order = sorted(valid_c, key=lambda c: -np.nanmean(OUTCOME_Y.values[raw_labels == c]))
else:
    order = sorted(valid_c, key=lambda c: -np.sum(raw_labels == c))
remap = {old: new for new, old in enumerate(order)}
remap[-1] = -1
LABELS = np.array([remap[c] for c in raw_labels])
CLUSTERS = sorted(set(LABELS) - {-1})
data["cluster"] = LABELS
sizes = pd.Series(LABELS).value_counts().sort_index()
print(f"==> Algoritma final: {FINAL_ALGO} | jumlah cluster: {len(CLUSTERS)} | noise: {np.mean(LABELS == -1):.1%}")
display(pd.DataFrame({"n": sizes, "pct": (sizes / N * 100).round(2)}))

# %% [markdown]
# ## 13. Profiling cluster (inti untuk bisnis & laporan)
#
# Semua profil memakai **nilai asli** (bukan nilai ter-scale) supaya bisa dibaca bisnis.
# 1. **Ukuran cluster** (n, %).
# 2. **Tabel mean/median** per cluster + baris "Overall".
# 3. **Z-score heatmap**: (mean cluster − mean overall) / std overall. Merah = jauh di atas rata-rata, biru = di bawah.
#    |z| ≥ 0.5 = pembeda yang berarti, |z| ≥ 1 = sangat khas.
# 4. **Komposisi kategorikal** (stacked bar) + **indeks over-representation** (share di cluster / share overall; > 1.3 = khas).
# 5. **Snake plot** & **radar plot** (perbandingan "sidik jari" tiap cluster) dan **boxplot** fitur paling pembeda.
# 6. **Scatter PCA / t-SNE / UMAP** berwarna cluster.
#
# **Untuk laporan:** z-score heatmap + tabel profil + (opsional) radar plot = 1 halaman yang sangat informatif.
# Setiap cluster beri **nama** (lihat bagian 14) dan 1–2 kalimat karakteristik.

# %%
prof_num = [c for c in dict.fromkeys(NUM_COLS + [c for c in PROFILE_COLS if pd.api.types.is_numeric_dtype(data[c])]
                                     + ([CFG.VALUE_COL] if CFG.VALUE_COL else [])) if c in data.columns and c != "cluster"]
prof_cat = [c for c in dict.fromkeys(CAT_COLS + [c for c in PROFILE_COLS if not pd.api.types.is_numeric_dtype(data[c])])
            if c in data.columns]
DV = data[data["cluster"] != -1]
num_vals = data[prof_num].apply(pd.to_numeric, errors="coerce")
overall_mean, overall_std = num_vals.mean(), num_vals.std().replace(0, np.nan)
cl_mean = num_vals[data["cluster"] != -1].groupby(data.loc[data["cluster"] != -1, "cluster"]).mean()
cl_median = num_vals[data["cluster"] != -1].groupby(data.loc[data["cluster"] != -1, "cluster"]).median()
Z = ((cl_mean - overall_mean) / overall_std).fillna(0)

size_tab = pd.DataFrame({"n": DV["cluster"].value_counts().sort_index()})
size_tab["pct"] = size_tab["n"] / N * 100
profile_tab = cl_mean.copy()
profile_tab.loc["Overall"] = overall_mean
profile_tab.insert(0, "pct", list(size_tab["pct"]) + [100.0])
profile_tab.insert(0, "n", list(size_tab["n"]) + [N])
if OUTCOME_Y is not None:
    rates = OUTCOME_Y[data["cluster"] != -1].groupby(DV["cluster"]).mean()
    profile_tab[f"{CFG.OUTCOME_COL}_rate" if OUTCOME_KIND == "binary" else f"{CFG.OUTCOME_COL}_mean"] = list(rates) + [OUTCOME_Y.mean()]
display(profile_tab.round(3))
save_table(profile_tab.reset_index().rename(columns={"index": "cluster"}), "cluster_profile_mean")
save_table(cl_median.reset_index(), "cluster_profile_median")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar([str(c) for c in size_tab.index], size_tab["pct"], color=plt.cm.tab10(np.arange(len(size_tab)) % 10))
for i, (nn, pc) in enumerate(zip(size_tab["n"], size_tab["pct"])):
    ax.text(i, pc, f"{nn}\n({pc:.1f}%)", ha="center", va="bottom", fontsize=8)
ax.set_title("Cluster sizes")
ax.set_xlabel("Cluster")
ax.set_ylabel("% of records")
ax.set_ylim(0, size_tab["pct"].max() * 1.25)
save_fig("cluster_sizes")

if len(prof_num):
    zshow = Z[prof_num[:30]]
    fig, ax = plt.subplots(figsize=(min(1.0 + 0.75 * zshow.shape[1], 18), 0.6 * len(zshow) + 2))
    if sns is not None:
        sns.heatmap(zshow, cmap="RdBu_r", center=0, vmin=-2, vmax=2, annot=True, fmt=".2f", ax=ax,
                    cbar_kws={"label": "z-score vs overall"})
    else:
        im = ax.imshow(zshow.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        plt.colorbar(im)
    ax.set_title("Cluster profile: standardized mean difference vs overall (z)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    save_fig("cluster_zscore_heatmap")

# %%
CAT_INDEX = {}
cat_plot = prof_cat[:6]
if cat_plot:
    fig, axes = plt.subplots(1, len(cat_plot), figsize=(5 * len(cat_plot), 4), squeeze=False)
    for ax, c in zip(axes[0], cat_plot):
        s = data[c].astype(object).where(data[c].notna(), "Missing").astype(str)
        top = s.value_counts().index[:8]
        s = s.where(s.isin(top), "Other")
        ct = pd.crosstab(data["cluster"], s, normalize="index")
        ct = ct.loc[[i for i in ct.index if i != -1]]
        overall_share = s.value_counts(normalize=True)
        CAT_INDEX[c] = (ct, overall_share)
        bottom = np.zeros(len(ct))
        for j, col in enumerate(ct.columns):
            ax.bar(ct.index.astype(str), ct[col], bottom=bottom, label=str(col), color=plt.cm.tab20(j % 20))
            bottom += ct[col].values
        ax.set_title(f"Composition: {c}")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Share")
        ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    save_fig("categorical_composition")
    idx_rows = []
    for c, (ct, osh) in CAT_INDEX.items():
        for cl in ct.index:
            for cat_v in ct.columns:
                idx_rows.append({"feature": c, "category": cat_v, "cluster": cl, "share_cluster": ct.loc[cl, cat_v],
                                 "share_overall": osh.get(cat_v, np.nan), "index": ct.loc[cl, cat_v] / max(osh.get(cat_v, np.nan), 1e-9)})
    CAT_IDX_TAB = pd.DataFrame(idx_rows)
    save_table(CAT_IDX_TAB, "categorical_overrepresentation")
    print("Kategori paling over-represented per cluster (index > 1.3 & share >= 20%):")
    display(CAT_IDX_TAB[(CAT_IDX_TAB["index"] > 1.3) & (CAT_IDX_TAB["share_cluster"] >= 0.2)]
            .sort_values(["cluster", "index"], ascending=[True, False]).groupby("cluster").head(3).round(3))
else:
    CAT_IDX_TAB = pd.DataFrame()

# %%
if len(prof_num) >= 2:
    spread = Z.abs().max(axis=0).sort_values(ascending=False)
    snake_feats = spread.index[:CFG.MAX_PLOT_FEATURES].tolist()
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(snake_feats) + 3), 4.5))
    for cl in Z.index:
        ax.plot(snake_feats, Z.loc[cl, snake_feats], marker="o", label=f"Cluster {cl}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Snake plot: standardized feature means by cluster")
    ax.set_xlabel("Feature")
    ax.set_ylabel("z-score vs overall")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    save_fig("snake_plot")

    radar_feats = snake_feats[:8]
    if len(radar_feats) >= 3:
        mm = cl_mean[radar_feats]
        mm = (mm - mm.min()) / (mm.max() - mm.min()).replace(0, 1)
        ang = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
        ang += ang[:1]
        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)
        for cl in mm.index:
            v = mm.loc[cl].tolist()
            v += v[:1]
            ax.plot(ang, v, label=f"Cluster {cl}")
            ax.fill(ang, v, alpha=0.08)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(radar_feats, fontsize=8)
        ax.set_title("Radar plot: cluster means (min-max scaled across clusters)", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        save_fig("radar_plot")

    # Fitur paling pembeda (Kruskal-Wallis) -> boxplot
    kw = {}
    for c in prof_num:
        groups = [num_vals.loc[data["cluster"] == cl, c].dropna().values for cl in CLUSTERS]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2 and HAS_SCIPY:
            try:
                kw[c] = sps.kruskal(*groups).statistic
            except Exception:
                pass
    if kw:
        KW = pd.Series(kw).sort_values(ascending=False)
        bfeats = KW.index[:6].tolist()
        fig, axes = plt.subplots(1, len(bfeats), figsize=(3.3 * len(bfeats), 4), squeeze=False)
        for ax, c in zip(axes[0], bfeats):
            vals = [num_vals.loc[data["cluster"] == cl, c].dropna().values for cl in CLUSTERS]
            try:
                ax.boxplot(vals, tick_labels=[str(cl) for cl in CLUSTERS], showfliers=False)
            except TypeError:
                ax.boxplot(vals, labels=[str(cl) for cl in CLUSTERS], showfliers=False)
            ax.set_title(c, fontsize=9)
            ax.set_xlabel("Cluster")
            ax.set_ylabel(c)
        plt.suptitle("Most discriminating features (Kruskal-Wallis H)", y=1.02)
        plt.tight_layout()
        save_fig("discriminating_boxplots")
        print("Ranking fitur pembeda (Kruskal-Wallis H):", KW.head(8).round(1).to_dict())

# %%
spaces = [("PCA", X_PCA2[EMB_IDX], LABELS[EMB_IDX])] + [(k, v, LABELS[EMB_IDX]) for k, v in EMB.items()]
fig, axes = plt.subplots(1, len(spaces), figsize=(6 * len(spaces), 5), squeeze=False)
for ax, (nm, emb, lab) in zip(axes[0], spaces):
    for cl in sorted(set(lab)):
        mk = lab == cl
        ax.scatter(emb[mk, 0], emb[mk, 1], s=6, alpha=0.6, label="noise" if cl == -1 else f"Cluster {cl}",
                   color="lightgray" if cl == -1 else plt.cm.tab10(cl % 10))
    ax.set_title(f"{nm} projection colored by cluster")
    ax.set_xlabel(f"{nm} 1")
    ax.set_ylabel(f"{nm} 2")
axes[0, 0].legend(markerscale=3, fontsize=8)
plt.tight_layout()
save_fig("cluster_projection")

# %% [markdown]
# ## 14. Profil teks otomatis & saran nama segmen
#
# Generator menulis kalimat profil per cluster dari: ukuran, fitur dengan |z| terbesar (kata "sangat tinggi/tinggi/rendah/sangat rendah"),
# kategori yang over-represented, dan outcome rate. **Nama segmen adalah saran** — sesuaikan dengan bahasa bisnis
# (mis. "High premium / Short tenure" → "Nasabah Premium Baru — Rawan Churn").
#
# **Untuk laporan:** salin teks profil, perhalus kalimatnya, dan tambahkan interpretasi bisnis (mengapa segmen ini berperilaku demikian).

# %%
WORDS = {"id": {"vh": "sangat tinggi", "h": "tinggi", "l": "rendah", "vl": "sangat rendah", "dom": "didominasi",
                "rec": "data", "rate": "rate", "vs": "vs overall", "noise": "noise"},
         "en": {"vh": "very high", "h": "high", "l": "low", "vl": "very low", "dom": "dominated by",
                "rec": "records", "rate": "rate", "vs": "vs overall", "noise": "noise"}}
W = WORDS.get(CFG.PROFILE_LANG, WORDS["id"])


def level_word(z):
    if z >= 1:
        return W["vh"]
    if z >= 0.5:
        return W["h"]
    if z <= -1:
        return W["vl"]
    if z <= -0.5:
        return W["l"]
    return None


PROFILES = []
for cl in CLUSTERS:
    n_c = int((LABELS == cl).sum())
    parts, name_parts = [], []
    if len(prof_num):
        zc = Z.loc[cl].reindex(prof_num).dropna()
        for f in zc.abs().sort_values(ascending=False).index[:CFG.TOP_FEATURES_PROFILE]:
            w = level_word(zc[f])
            if w is None:
                continue
            parts.append(f"{f} {w} ({cl_mean.loc[cl, f]:.3g} {W['vs']} {overall_mean[f]:.3g})")
            if len(name_parts) < 2:
                name_parts.append(f"{'High' if zc[f] > 0 else 'Low'} {f}")
    if len(CAT_IDX_TAB):
        sub = CAT_IDX_TAB[(CAT_IDX_TAB["cluster"] == cl) & (CAT_IDX_TAB["index"] >= 1.3) & (CAT_IDX_TAB["share_cluster"] >= 0.3)]
        for _, r in sub.sort_values("index", ascending=False).head(2).iterrows():
            parts.append(f"{W['dom']} {r['feature']}={r['category']} ({r['share_cluster']:.0%} {W['vs']} {r['share_overall']:.0%})")
            if len(name_parts) < 3:
                name_parts.append(f"{r['category']}")
    out_txt, rate_c = "", np.nan
    if OUTCOME_Y is not None:
        rate_c = float(np.nanmean(OUTCOME_Y.values[LABELS == cl]))
        ov = float(np.nanmean(OUTCOME_Y.values))
        if OUTCOME_KIND == "binary":
            out_txt = f"; {CFG.OUTCOME_COL} {W['rate']} {rate_c:.1%} (lift {rate_c / max(ov, 1e-9):.2f}x)"
        else:
            out_txt = f"; mean {CFG.OUTCOME_COL} {rate_c:.3g} ({W['vs']} {ov:.3g})"
    name_hint = " / ".join(name_parts) if name_parts else "Average profile"
    text = f"Cluster {cl} ({n_c / N:.1%}, {n_c:,} {W['rec']}): " + ("; ".join(parts) if parts else "profil mendekati rata-rata") + out_txt + "."
    PROFILES.append({"cluster": cl, "n": n_c, "pct": n_c / N * 100, "name_hint": name_hint, "outcome": rate_c, "profile_text": text})
if (LABELS == -1).any():
    PROFILES.append({"cluster": -1, "n": int((LABELS == -1).sum()), "pct": np.mean(LABELS == -1) * 100,
                     "name_hint": "Noise / outlier", "outcome": np.nan,
                     "profile_text": f"Noise ({np.mean(LABELS == -1):.1%}): titik yang tidak masuk cluster mana pun (outlier/unik)."})
PROFILE_TAB = pd.DataFrame(PROFILES)
NAME_MAP = dict(zip(PROFILE_TAB["cluster"], PROFILE_TAB["name_hint"]))
for t in PROFILE_TAB["profile_text"]:
    print("-", t)
print("\nSaran nama segmen:", NAME_MAP)
save_table(PROFILE_TAB, "cluster_text_profiles")
if "RFM_segment" in data.columns:
    print("\nCluster vs segmen RFM klasik (proporsi baris per cluster):")
    display(pd.crosstab(data["cluster"], data["RFM_segment"], normalize="index").round(2))

# %% [markdown]
# ## 15. Analisis outcome per cluster & rekomendasi aksi bisnis
#
# (Jalan hanya kalau `CFG.OUTCOME_COL` diisi, mis. `churn`.)
# - **Outcome rate per cluster** + **95% Wilson CI** (interval kepercayaan untuk proporsi — lebih akurat dari ±1.96·SE untuk n kecil).
# - **Lift** = rate cluster / rate keseluruhan (2.0x = dua kali lebih berisiko).
# - **Chi-square test**: H0 = outcome independen dari cluster. p < 0.05 → rate berbeda signifikan antar cluster. **Cramér's V** = kekuatan
#   asosiasi (0.1 lemah, 0.3 sedang, 0.5 kuat).
# - **Risk × Value** (kalau `VALUE_COL` diisi): sumbu X = outcome rate, Y = rata-rata nilai (mis. premi); ukuran titik = jumlah pelanggan.
#   **Value at risk** = n × rate × rata-rata nilai → urutan prioritas.
#
# | Kuadran | Arti | Aksi umum |
# |---|---|---|
# | Risiko tinggi, nilai tinggi | kehilangan terbesar | **Prioritas retensi**: kontak personal, penawaran loyalitas, review premi |
# | Risiko tinggi, nilai rendah | banyak tapi kecil | retensi **berbiaya rendah**: otomatisasi email/WA, edukasi produk |
# | Risiko rendah, nilai tinggi | aset utama | **pertahankan & upsell/cross-sell**, program VIP |
# | Risiko rendah, nilai rendah | stabil tapi kecil | **growth**: upsell produk entry-level, kanal digital |
#
# **Untuk laporan:** grafik rate per cluster dengan CI + tabel rekomendasi aksi adalah "jembatan" dari analisis ke keputusan bisnis.
# Tulis: *"Segmen 2 (18% nasabah) memiliki churn 32% (lift 1.8x, CI 29–35%) dan premi rata-rata tertinggi → prioritas retensi."*

# %%
ACTIONS = None
if OUTCOME_Y is not None:
    rows = []
    ov = float(np.nanmean(OUTCOME_Y))
    for cl in CLUSTERS:
        yv = OUTCOME_Y.values[LABELS == cl]
        yv = yv[np.isfinite(yv)]
        r = {"cluster": cl, "name_hint": NAME_MAP.get(cl), "n": len(yv)}
        if OUTCOME_KIND == "binary":
            k_pos = int(yv.sum())
            lo, hi = wilson_ci(k_pos, len(yv))
            r.update({"positives": k_pos, "rate": k_pos / max(len(yv), 1), "ci_low": lo, "ci_high": hi})
        else:
            se = yv.std(ddof=1) / math.sqrt(max(len(yv), 1)) if len(yv) > 1 else np.nan
            r.update({"rate": yv.mean(), "ci_low": yv.mean() - 1.96 * se, "ci_high": yv.mean() + 1.96 * se})
        r["lift"] = r["rate"] / ov if ov else np.nan
        if CFG.VALUE_COL:
            vv = pd.to_numeric(data.loc[LABELS == cl, CFG.VALUE_COL], errors="coerce")
            r["mean_value"] = vv.mean()
            r["total_value"] = vv.sum()
            if OUTCOME_KIND == "binary":
                r["value_at_risk"] = r["rate"] * vv.sum()
        rows.append(r)
    OUT_TAB = pd.DataFrame(rows)
    if OUTCOME_KIND == "binary" and HAS_SCIPY:
        ct = pd.crosstab(LABELS[(LABELS != -1) & OUTCOME_Y.notna().values], OUTCOME_Y[(LABELS != -1) & OUTCOME_Y.notna().values])
        chi2, p, dof, _ = sps.chi2_contingency(ct)
        cv = math.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))) if min(ct.shape) > 1 else np.nan
        print(f"Chi-square: chi2={chi2:.2f}, dof={dof}, p={p:.3g} -> " +
              ("✅ outcome rate berbeda signifikan antar cluster" if p < 0.05 else "⚠️ tidak signifikan") + f" | Cramér's V = {cv:.3f}")
    elif OUTCOME_KIND == "numeric" and HAS_SCIPY:
        groups = [OUTCOME_Y.values[(LABELS == cl) & OUTCOME_Y.notna().values] for cl in CLUSTERS]
        hst, p = sps.kruskal(*[g for g in groups if len(g) > 1])
        print(f"Kruskal-Wallis: H={hst:.2f}, p={p:.3g}")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xs = np.arange(len(OUT_TAB))
    ax.bar(xs, OUT_TAB["rate"], color=["tab:red" if l > 1.2 else "tab:blue" for l in OUT_TAB["lift"]], alpha=0.8)
    ax.errorbar(xs, OUT_TAB["rate"], yerr=[OUT_TAB["rate"] - OUT_TAB["ci_low"], OUT_TAB["ci_high"] - OUT_TAB["rate"]],
                fmt="none", ecolor="black", capsize=4)
    ax.axhline(ov, color="gray", ls="--", label=f"overall = {ov:.3g}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"C{c}" for c in OUT_TAB["cluster"]])
    ax.set_title(f"{CFG.OUTCOME_COL} {'rate' if OUTCOME_KIND == 'binary' else 'mean'} by cluster (95% CI)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(f"{CFG.OUTCOME_COL} rate" if OUTCOME_KIND == "binary" else f"Mean {CFG.OUTCOME_COL}")
    ax.legend()
    save_fig("outcome_by_cluster")

    ACTION_TEXT = {
        ("high", "high"): ("Prioritas retensi", "Kontak personal / account manager, penawaran loyalitas, review harga/premi, layanan prioritas"),
        ("high", "low"): ("Retensi biaya rendah", "Otomatisasi pengingat (email/WA), edukasi manfaat produk, kemudahan pembayaran"),
        ("low", "high"): ("Pertahankan & upsell", "Program VIP/loyalty, cross-sell produk pelengkap, referral"),
        ("low", "low"): ("Growth / efisiensi", "Upsell produk entry-level, kanal digital self-service, kampanye massal"),
    }
    rate_thr = ov
    val_thr = float(pd.to_numeric(data[CFG.VALUE_COL], errors="coerce").mean()) if CFG.VALUE_COL else None
    acts = []
    for _, r in OUT_TAB.iterrows():
        risk = "high" if r["rate"] > rate_thr else "low"
        val = ("high" if r["mean_value"] > val_thr else "low") if CFG.VALUE_COL else "high"
        seg, act = ACTION_TEXT[(risk, val)]
        acts.append({"cluster": r["cluster"], "name_hint": r["name_hint"], "n": r["n"], "rate": r["rate"], "lift": r["lift"],
                     "mean_value": r.get("mean_value", np.nan), "value_at_risk": r.get("value_at_risk", np.nan),
                     "risk": risk, "value": val if CFG.VALUE_COL else "-", "strategy": seg, "suggested_actions": act,
                     "kpi": f"{CFG.OUTCOME_COL} rate cluster, retention rate 3/6 bulan, response rate kampanye"})
    ACTIONS = pd.DataFrame(acts)
    sort_col = "value_at_risk" if ("value_at_risk" in ACTIONS and ACTIONS["value_at_risk"].notna().any()) else "rate"
    ACTIONS = ACTIONS.sort_values(sort_col, ascending=False).reset_index(drop=True)
    ACTIONS.insert(0, "priority", np.arange(1, len(ACTIONS) + 1))
    display(OUT_TAB.round(4))
    display(ACTIONS.round(3))
    save_table(OUT_TAB, "outcome_by_cluster")
    save_table(ACTIONS, "recommended_actions")

    if CFG.VALUE_COL:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        sz = OUT_TAB["n"] / OUT_TAB["n"].max() * 1500 + 50
        ax.scatter(OUT_TAB["rate"], OUT_TAB["mean_value"], s=sz, alpha=0.5, c=[plt.cm.tab10(c % 10) for c in OUT_TAB["cluster"]])
        for _, r in OUT_TAB.iterrows():
            ax.annotate(f"C{int(r['cluster'])}", (r["rate"], r["mean_value"]), ha="center", va="center", fontsize=10, weight="bold")
        ax.axvline(rate_thr, color="gray", ls="--")
        ax.axhline(val_thr, color="gray", ls="--")
        ax.set_title(f"Risk x Value matrix (bubble size = n)")
        ax.set_xlabel(f"{CFG.OUTCOME_COL} rate (risk)")
        ax.set_ylabel(f"Mean {CFG.VALUE_COL} (value)")
        xl, yl = ax.get_xlim(), ax.get_ylim()
        for (tx, ty, lab) in [(xl[1], yl[1], "Retain (priority)"), (xl[0], yl[1], "Protect & upsell"),
                              (xl[1], yl[0], "Low-cost retention"), (xl[0], yl[0], "Grow")]:
            ax.text(tx, ty, lab, ha="right" if tx == xl[1] else "left", va="top" if ty == yl[1] else "bottom",
                    fontsize=9, color="dimgray", style="italic")
        save_fig("risk_value_matrix")
else:
    print("OUTCOME_COL tidak diisi -> analisis outcome dilewati. (Isi CFG.OUTCOME_COL, mis. 'churn', untuk analisis risiko per segmen.)")

# %% [markdown]
# ## 16. Menjelaskan cluster: surrogate decision tree, permutation importance, SHAP
#
# - **Surrogate decision tree**: pohon keputusan kecil (depth ≤ 4) dilatih untuk **meniru label cluster** dari fitur asli (skala asli,
#   one-hot kategorikal). Aturan pohon = definisi segmen yang mudah dipahami bisnis (mis. *"premium > 5 jt DAN tenure ≤ 12 bulan → Cluster 2"*).
#   **Fidelity** (akurasi meniru di data holdout) > 0.85 → aturan representatif; < 0.7 → cluster terlalu kompleks untuk dijelaskan dengan aturan sederhana.
# - **Permutation importance** (classifier RandomForest/LightGBM): seberapa turun akurasi prediksi cluster kalau fitur diacak
#   → fitur paling menentukan segmentasi.
# - **SHAP** (opsional): kontribusi rata-rata |SHAP| per fitur untuk tiap cluster.
#
# **Untuk laporan:** tampilkan aturan pohon (teks) atau gambar pohon + bar chart importance.

# %%
interp_num = PREP.num_clean(data, log=False)[PREP.num_cols] if PREP.num_cols else pd.DataFrame(index=data.index)
interp_cat = pd.get_dummies(PREP._cat_clean(data), prefix_sep="=").astype(int) if CAT_COLS else pd.DataFrame(index=data.index)
XI = pd.concat([interp_num, interp_cat], axis=1)
mask_v = LABELS != -1
XI_v, y_v = XI[mask_v], LABELS[mask_v]
strat = y_v if pd.Series(y_v).value_counts().min() >= 2 else None
Xtr, Xte, ytr, yte = train_test_split(XI_v, y_v, test_size=0.25, random_state=CFG.SEED, stratify=strat)
leaf = max(5, int(0.01 * len(Xtr)))
tree = DecisionTreeClassifier(max_depth=CFG.SURROGATE_MAX_DEPTH, min_samples_leaf=leaf, random_state=CFG.SEED).fit(Xtr, ytr)
FIDELITY = accuracy_score(yte, tree.predict(Xte))
print(f"Surrogate tree fidelity (holdout accuracy) = {FIDELITY:.3f} -> " +
      ("✅ aturan representatif" if FIDELITY > 0.85 else ("cukup" if FIDELITY > 0.7 else "⚠️ cluster sulit dijelaskan dengan aturan sederhana")))
RULES = export_text(tree, feature_names=list(XI.columns), decimals=2)
print(RULES[:4000])
with open(TAB_DIR / "surrogate_tree_rules.txt", "w", encoding="utf-8") as f:
    f.write(RULES)
fig, ax = plt.subplots(figsize=(min(4 + 2.2 * 2 ** (CFG.SURROGATE_MAX_DEPTH - 1), 18), 2.0 * CFG.SURROGATE_MAX_DEPTH + 2))
plot_tree(tree, feature_names=list(XI.columns), class_names=[f"C{c}" for c in tree.classes_], filled=True, rounded=True,
          fontsize=7, ax=ax, impurity=False, proportion=True)
ax.set_title(f"Surrogate decision tree (fidelity = {FIDELITY:.2f})")
save_fig("surrogate_tree")

# %%
t0 = time.time()
if HAS_LGB:
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=CFG.SEED, verbose=-1)
    clf_name = "LightGBM"
else:
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=CFG.SEED)
    clf_name = "RandomForest"
clf.fit(Xtr, ytr)
clf_acc = accuracy_score(yte, clf.predict(Xte))
print(f"{clf_name} memprediksi cluster: akurasi holdout = {clf_acc:.3f} ({time.time() - t0:.1f}s)")
Xte_s = Xte.iloc[:min(len(Xte), 3000)]
pi = permutation_importance(clf, Xte_s, yte[:len(Xte_s)], n_repeats=5, random_state=CFG.SEED, n_jobs=1)
PI = pd.Series(pi.importances_mean, index=XI.columns).sort_values(ascending=False)
save_table(PI.rename("perm_importance").reset_index().rename(columns={"index": "feature"}), "cluster_driver_importance")
top = PI.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 0.35 * len(top) + 1.5))
ax.barh(top.index, top.values, color="tab:green")
ax.set_title(f"What drives cluster membership? (permutation importance, {clf_name})")
ax.set_xlabel("Drop in accuracy when feature is shuffled")
ax.set_ylabel("Feature")
save_fig("cluster_driver_importance")
print("Top 5 penentu cluster:", PI.head(5).round(4).to_dict())

if CFG.USE_SHAP and HAS_SHAP:
    try:
        Xsh = Xte.iloc[:min(len(Xte), CFG.SHAP_SAMPLE)].astype(float)
        sv = shap.TreeExplainer(clf).shap_values(Xsh)
        if isinstance(sv, list):
            arr = np.stack([np.abs(s) for s in sv], axis=-1)          # (n, f, classes) dari list per kelas (shap lama)
        else:
            arr = np.abs(np.asarray(sv))
            if arr.ndim == 2:
                arr = arr[:, :, None]
        mean_abs = pd.DataFrame(arr.mean(axis=0), index=Xsh.columns,
                                columns=[f"C{c}" for c in clf.classes_][:arr.shape[2]])
        order_f = mean_abs.sum(axis=1).sort_values(ascending=False).index[:12][::-1]
        fig, ax = plt.subplots(figsize=(9, 0.4 * len(order_f) + 1.5))
        left = np.zeros(len(order_f))
        for j, col in enumerate(mean_abs.columns):
            ax.barh(order_f, mean_abs.loc[order_f, col], left=left, label=col, color=plt.cm.tab10(j % 10))
            left += mean_abs.loc[order_f, col].values
        ax.set_title("Mean |SHAP| per feature, stacked by cluster")
        ax.set_xlabel("mean |SHAP value|")
        ax.set_ylabel("Feature")
        ax.legend(fontsize=8)
        save_fig("shap_cluster_drivers")
    except Exception as e:
        plt.close("all")
        print("SHAP gagal (tidak fatal):", e)
elif CFG.USE_SHAP:
    print("shap tidak terinstall -> dilewati (permutation importance sudah cukup).")

# %% [markdown]
# ## 17. Export label, profil, model + assign data baru
#
# Output utama:
# - `tables/cluster_labels.csv` — ID + cluster + nama segmen (untuk dipakai tim bisnis / di-merge ke data lain, mis. model churn).
# - `tables/cluster_profile_mean.csv`, `cluster_text_profiles.csv`, `recommended_actions.csv`, dll.
# - `models/cluster_pipeline.joblib` — preprocessor + model final (untuk assign data baru).
# - Kalau `TEST_PATH` ada: `submissions/test_cluster_assignments.csv`.
#
# **Assign data baru:** KMeans/MiniBatch/GMM memakai `.predict`; algoritma lain memakai **centroid terdekat** (atau medoid terdekat untuk Gower).
#
# **Tip lomba:** label cluster bisa dijadikan **fitur tambahan** untuk model prediksi churn (notebook tabular) — sering membantu interpretasi.

# %%
lab_out = pd.DataFrame({"cluster": LABELS})
if CFG.ID_COL and CFG.ID_COL in data.columns:
    lab_out.insert(0, CFG.ID_COL, data[CFG.ID_COL].values)
lab_out["segment_name"] = lab_out["cluster"].map(NAME_MAP)
if FINAL_ALGO == "gmm":
    lab_out["membership_prob"] = ALGO["gmm"]["proba_max"]
save_table(lab_out, "cluster_labels")
print("Disimpan:", (TAB_DIR / "cluster_labels.csv").resolve())


def assign_clusters(new_df):
    """Assign data baru ke cluster final memakai preprocessing yang sama. Return label (nomor final)."""
    fa = ALGO[FINAL_ALGO]
    if fa["space"] == "gower":
        gn = PREP.num_clean(new_df).values if PREP.num_cols else None
        gc = PREP.cat_codes(new_df)
        med = fa["medoids"]
        Dm = gower_matrix(gn, gc, GOWER["num"][med] if gn is not None else None, GOWER["cat"][med], ranges=GOWER["ranges"])
        raw = np.argmin(Dm, axis=1)
        raw = ALGO[FINAL_ALGO]["labels"][med][raw]
    else:
        Xn = PREP.transform(new_df, cat_weight=fa.get("cat_weight")) if fa["space"] == "XKP" else PREP.transform(new_df)
        if fa["space"] == "XC" and CFG.USE_PCA_FOR_CLUSTERING:
            Xn = pca_full.transform(Xn)[:, :N_PCA]
        model = fa.get("model")
        if model is not None and hasattr(model, "predict") and FINAL_ALGO in ("kmeans", "minibatch", "gmm") or \
                (FINAL_ALGO == "kprototypes" and fa.get("impl") == "fallback"):
            raw = model.predict(Xn)
        else:
            Xref = XC if fa["space"] == "XC" else PREP.transform(data, cat_weight=fa.get("cat_weight"))
            raw, _ = nearest_centroid_assign(Xref, fa["labels"], Xn)
    return np.array([remap.get(int(c), -1) for c in raw])


chk = assign_clusters(data.iloc[:min(N, 2000)])
agree = float(np.mean(chk == LABELS[:len(chk)]))
print(f"Konsistensi assign ulang data train (2000 baris pertama): {agree:.1%}" + ("  ✅" if agree > 0.9 else "  ⚠️ (wajar untuk algoritma tanpa predict)"))
if HAS_TEST:
    for c in test.columns:
        if c in NUM_COLS:
            test[c] = pd.to_numeric(test[c], errors="coerce")
    miss = [c for c in NUM_COLS + CAT_COLS if c not in test.columns]
    if miss:
        print("⚠️ Kolom fitur tidak ada di test (diisi NaN -> imputasi):", miss)
        for c in miss:
            test[c] = np.nan
    t_lab = assign_clusters(test)
    t_out = pd.DataFrame({"cluster": t_lab})
    if CFG.ID_COL and CFG.ID_COL in test.columns:
        t_out.insert(0, CFG.ID_COL, test[CFG.ID_COL].values)
    t_out["segment_name"] = t_out["cluster"].map(NAME_MAP)
    t_out.to_csv(SUB_DIR / "test_cluster_assignments.csv", index=False)
    print(f"Test di-assign: {len(t_out)} baris -> {SUB_DIR / 'test_cluster_assignments.csv'}")
    display(t_out["cluster"].value_counts().sort_index().to_frame("n_test"))
if HAS_JOBLIB:
    try:
        joblib.dump({"prep": PREP, "final_algo": FINAL_ALGO, "model": ALGO[FINAL_ALGO].get("model"), "remap": remap,
                     "name_map": NAME_MAP, "pca": pca_full if CFG.USE_PCA_FOR_CLUSTERING else None, "n_pca": N_PCA},
                    MODEL_DIR / "cluster_pipeline.joblib")
        print("Model disimpan:", MODEL_DIR / "cluster_pipeline.joblib")
    except Exception as e:
        print("Gagal simpan model (tidak fatal):", e)

# %% [markdown]
# ## 18. Ringkasan untuk laporan
#
# **Struktur laporan segmentasi yang disarankan:** (1) tujuan bisnis segmentasi → (2) data & fitur (alasan pemilihan fitur, preprocessing)
# → (3) metodologi (scaling, algoritma, cara memilih K, uji stabilitas) → (4) hasil: ukuran & profil segmen (heatmap/radar, teks profil, nama)
# → (5) outcome per segmen (churn rate + CI, chi-square, lift) → (6) rekomendasi aksi per segmen (tabel prioritas) → (7) keterbatasan.

# %%
print("RINGKASAN")
print("-" * 70)
print(f"Data: {N} baris; fitur clustering: {len(NUM_COLS)} numerik + {len(CAT_COLS)} kategorikal "
      f"(scaler={CFG.SCALER}, log1p={len(PREP.log_cols)} fitur, clip={CFG.CLIP_QUANTILES}).")
print(f"K dipilih = {K} (votes: {votes}); algoritma final = {FINAL_ALGO}; "
      f"silhouette = {CMP.loc[CMP['algorithm'] == FINAL_ALGO, 'silhouette'].iloc[0]:.3f}; stabilitas ARI KMeans = {STAB_K:.2f}.")
print(f"Surrogate tree fidelity = {FIDELITY:.2f}; classifier accuracy = {clf_acc:.2f}.")
for t in PROFILE_TAB["profile_text"]:
    print("  *", t)
if ACTIONS is not None:
    print("Prioritas aksi:")
    for _, r in ACTIONS.iterrows():
        print(f"  {int(r['priority'])}. Cluster {r['cluster']} ({r['name_hint']}): {r['strategy']} -> {r['suggested_actions']}")
print("\nFigure:", ", ".join(p.name for p in sorted(FIG_DIR.glob("*.png"))))
print("Tabel :", ", ".join(p.name for p in sorted(TAB_DIR.glob("*.csv"))))

# %% [markdown]
# ## 19. Debugging & sanity checks

# %%
print("Shapes: data", data.shape, "| X", X.shape, "| XC", XC.shape)
print("NaN/inf di X:", int((~np.isfinite(X)).sum()))
print("Ukuran cluster:", pd.Series(LABELS).value_counts().sort_index().to_dict())
small = [c for c in CLUSTERS if (LABELS == c).mean() < CFG.MIN_CLUSTER_PCT]
print("Cluster terlalu kecil (< MIN_CLUSTER_PCT):", small if small else "tidak ada ✅")
print("OUTCOME_COL dipakai sebagai fitur?", "⚠️ YA" if CFG.OUTCOME_COL in (NUM_COLS + CAT_COLS) else "tidak ✅")
if OUTCOME_Y is not None and NUM_COLS:
    cors = data[NUM_COLS].apply(pd.to_numeric, errors="coerce").corrwith(OUTCOME_Y).abs().sort_values(ascending=False)
    if len(cors) and cors.iloc[0] > 0.9:
        print(f"⚠️ Fitur '{cors.index[0]}' berkorelasi {cors.iloc[0]:.2f} dengan outcome -> mungkin proxy/bocoran outcome; pertimbangkan exclude.")
if FAILED_MODELS:
    print("FAILED_MODELS:", FAILED_MODELS)
else:
    print("FAILED_MODELS: (kosong) ✅")
if HAS_TEST:
    print("Kolom fitur di test:", all(c in test.columns for c in NUM_COLS + CAT_COLS))

# %% [markdown]
# ## 20. Troubleshooting
#
# | Error / gejala | Penyebab | Solusi |
# |---|---|---|
# | `AssertionError: DATA_PATH tidak ditemukan` | path salah | cek `CFG.DATA_PATH`, gunakan path relatif dari folder notebook |
# | `Tidak ada fitur untuk clustering` | semua kolom terdeteksi ID/teks | set `CFG.FEATURE_COLS` manual |
# | `TypeError: __init__() got an unexpected keyword argument 'sparse_output'` | sklearn < 1.2 | sudah ditangani `make_ohe()`; kalau muncul di kode sendiri pakai helper itu |
# | `TypeError: ... 'affinity'` / `'metric'` di Agglomerative | beda versi sklearn | pakai `make_agglo()` |
# | `TSNE ... unexpected keyword 'n_iter'` | sklearn ≥ 1.5 ganti ke `max_iter` | pakai `make_tsne()` |
# | `AttributeError: module 'umap' has no attribute 'UMAP'` | package `umap` salah | `pip uninstall umap` lalu `pip install umap-learn` |
# | `ModuleNotFoundError: hdbscan` | package tidak ada | otomatis fallback `sklearn.cluster.HDBSCAN` (sklearn ≥ 1.3) |
# | `MemoryError` di agglomerative / Gower | O(n²) | kecilkan `AGGLO_MAX_N` / `GOWER_MAX_N` |
# | Silhouette sangat lama | n besar | kecilkan `METRIC_SAMPLE` (silhouette dihitung di sampel) |
# | DBSCAN: semua noise / 1 cluster | eps terlalu kecil/besar | set `DBSCAN_EPS` manual dari k-distance plot; atau pakai HDBSCAN |
# | HDBSCAN noise > 30% | `min_cluster_size` terlalu besar | turunkan `HDBSCAN_MIN_CLUSTER_SIZE` |
# | Satu cluster berisi 90% data | outlier ekstrem / fitur skewed | `LOG_SKEWED=True`, `CLIP_QUANTILES`, `SCALER="robust"` |
# | Cluster hanya memisahkan satu kategori | bobot one-hot terlalu besar | turunkan `CAT_WEIGHT` atau pindahkan kolom ke `PROFILE_ONLY_COLS` |
# | Hasil berubah tiap run | inisialisasi acak / tidak stabil | `SEED` tetap, `KMEANS_N_INIT` lebih besar, cek bagian Stabilitas |
# | ARI stabilitas < 0.6 | K terlalu besar / tidak ada struktur | kurangi K, kurangi fitur noise, `USE_PCA_FOR_CLUSTERING=True` |
# | `ValueError: Input contains NaN` | kolom test tidak ada / tipe beda | cek nama kolom test sama dengan train |
# | `GaussianMixture ... ill-defined empirical covariance` | fitur kolinear / konstan | sudah `reg_covar=1e-4`; naikkan, atau `GMM` covariance `diag` |
# | KPrototypes sangat lambat | n besar | sudah di-fit pada sampel; atau hapus `kprototypes` dari `ALGORITHMS` |
# | Chi-square warning frekuensi kecil | cluster kecil / outcome jarang | gabungkan cluster kecil atau laporkan dengan hati-hati (CI lebar) |
# | SHAP error bentuk array | beda versi shap | sudah ditangani (list vs array); kalau tetap error `USE_SHAP=False` |
# | Nama segmen kurang bermakna | saran otomatis berbasis z-score | ganti manual: `NAME_MAP = {0: "...", 1: "..."}` lalu jalankan ulang bagian 17 |

# %% [markdown]
# ## 21. Experiment recipes (salin ke USER OVERRIDE)
#
# **A. Segmentasi nasabah asuransi + churn:**
# ```python
# CFG.ID_COL = "customer_id"; CFG.OUTCOME_COL = "churn"; CFG.OUTCOME_POSITIVE = "Yes"; CFG.VALUE_COL = "premium"
# CFG.FEATURE_COLS = ["age", "premium", "tenure_months", "n_claims", "policy_type"]
# CFG.PROFILE_ONLY_COLS = ["region", "gender"]; CFG.SORT_CLUSTERS_BY = "outcome"
# ```
# **B. RFM dari transaksi:**
# ```python
# CFG.USE_RFM = True; CFG.TRANSACTIONS_PATH = "./data/transactions.csv"
# CFG.TX_CUSTOMER_COL = "customer_id"; CFG.TX_DATE_COL = "date"; CFG.TX_AMOUNT_COL = "amount"
# CFG.FEATURE_COLS = ["recency", "frequency", "monetary"]; CFG.VALUE_COL = "monetary"
# ```
# **C. Data besar (> 100k):** `CFG.RUN_MODE="fast"; CFG.ALGORITHMS=["kmeans","minibatch","gmm"]; CFG.METRIC_SAMPLE=3000; CFG.RUN_GAP=False`
# **D. Data campuran kecil:** `CFG.ALGORITHMS=["kmeans","kprototypes","kmedoids_gower","agglo_gower"]; CFG.GOWER_MAX_N=3000`
# **E. Banyak fitur berkorelasi:** `CFG.USE_PCA_FOR_CLUSTERING=True; CFG.PCA_VARIANCE=0.85`
# **F. Bisnis minta tepat 4 segmen:** `CFG.N_CLUSTERS = 4; CFG.FINAL_ALGO = "kmeans"`
# **G. Cari outlier/segmen niche:** `CFG.ALGORITHMS=["hdbscan","dbscan","kmeans"]; CFG.FINAL_ALGO="hdbscan"`
# **H. Final untuk laporan:** `CFG.RUN_MODE="full"; CFG.FIG_DPI=300; CFG.STABILITY_BOOTSTRAPS=30`

# %% [markdown]
# ## 22. Checklist sebelum laporan / submit
#
# - [ ] Fitur clustering masuk akal bisnis; outcome (churn) **tidak** ikut sebagai fitur.
# - [ ] Preprocessing dijelaskan (imputasi, log, clip, scaling, encoding kategorikal).
# - [ ] K dipilih dengan ≥ 3 metode + pertimbangan bisnis; tabel/figure "Choosing K" disimpan.
# - [ ] Stabilitas ARI dilaporkan (idealnya > 0.8).
# - [ ] Tidak ada cluster terlalu kecil (< `MIN_CLUSTER_PCT`) kecuali memang disengaja (niche).
# - [ ] Setiap cluster punya **nama**, profil teks, dan **aksi** yang jelas.
# - [ ] Outcome per cluster dengan CI + uji chi-square (kalau ada outcome).
# - [ ] Aturan surrogate tree (fidelity) / importance untuk menjelaskan segmen.
# - [ ] `RUN_MODE="full"`, `FIG_DPI=300` untuk figure laporan; file `cluster_labels.csv` siap dipakai.
