# -*- coding: utf-8 -*-
# =====================================================================================================
# anomaly_tabular.py — versi SCRIPT dari anomaly_tabular.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python anomaly_tabular.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb anomaly_tabular.py
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
# # Anomaly / Fraud / Outlier Detection — Tabular
#
# Untuk soal "temukan transaksi mencurigakan / klaim fraud / mesin bermasalah / data aneh", baik **tanpa label**
# (unsupervised) maupun dengan **sedikit label** (semi-supervised). Output: skor anomali per baris + penjelasan
# *mengapa* baris itu dianggap aneh (fitur penyumbang), siap untuk laporan.
#
# ## Pipeline
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json) → load → preprocessing (impute, robust scaling, one-hot)
# → EDA: distribusi, missing, (kalau ada label) prevalensi anomali
# → Detektor unsupervised: Isolation Forest, LOF, One-Class SVM, Robust Z (MAD), PCA reconstruction, Autoencoder (torch)
#   → skor dinormalisasi ke rank 0–1 → ensemble rata-rata rank
# → Kalau ada label: AUC / Average Precision / Precision@k per detektor + model supervised (HGB/LGBM, CV)
#   + model supervised dengan skor unsupervised sebagai fitur (stacking)
# → Threshold: dari CONTAMINATION, atau F1 terbaik di data berlabel
# → Penjelasan: fitur penyumbang per baris (robust z), aturan surrogate tree, profil anomali vs normal
# → Submission (skor atau label 0/1) + ringkasan laporan
# ```
#
# **Dua format:** `anomaly_tabular.ipynb` / `.py`. Headless: `python tools/run_task.py --task anomaly-tabular --format py`.

# %%
import os


class CFG:
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "anomaly_tabular"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"
    TRAIN_PATH = "./data/train.csv"
    TEST_PATH = "./data/test.csv"          # boleh tidak ada (skor dihitung untuk train saja)
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    OUTPUT_DIR = "./outputs/anomaly_tabular/"
    ID_COL = "id"
    LABEL_COL = None             # kolom label anomali (1 = anomali) kalau ada; None = unsupervised murni
    DROP_COLS = []
    CONTAMINATION = "auto"       # proporsi anomali yang diharapkan; "auto" = prevalensi label atau 0.05
    METHODS = ["iforest", "lof", "ocsvm", "robust_z", "pca", "autoencoder"]
    FIT_ON = "train"             # "train" | "train+test" (unsupervised: memakai semua data tanpa label)
    FIT_ONLY_NORMAL = True       # ada label: detektor di-fit hanya pada baris normal (lebih tajam)
    SUPERVISED = True            # ada label: latih juga HGB/LGBM (CV) + stacking skor unsupervised
    SUBMISSION_MODE = "auto"     # "auto" | "score" | "label"
    MAX_ONEHOT_CARD = 20
    N_FOLDS = 5
    FAST_MAX_ROWS = 5000
    FIG_DPI = 150

# %%
# >>> USER OVERRIDE
# CFG.LABEL_COL = "is_fraud"
# CFG.CONTAMINATION = 0.02
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
# Semua detektor inti hanya butuh scikit-learn. Autoencoder butuh torch (di-skip kalau tidak ada). LightGBM opsional.

# %%
import importlib.util
import subprocess
import sys

for imp, pip_name in {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "matplotlib": "matplotlib",
                      "lightgbm": "lightgbm"}.items():
    if importlib.util.find_spec(imp) is None and CFG.AUTO_INSTALL:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as e:  # noqa
            print("install gagal:", pip_name, e)

import json
import time
import warnings

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except Exception:  # noqa
    display = print
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:  # noqa
    HAS_LGB = False
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:  # noqa
    HAS_TORCH = False
print(f"LightGBM: {HAS_LGB} | torch: {HAS_TORCH}")
OUT = CFG.OUTPUT_DIR
FIG_DIR, TAB_DIR, SUB_DIR = (os.path.join(OUT, d) for d in ("figures", "tables", "submissions"))
for d in (OUT, FIG_DIR, TAB_DIR, SUB_DIR):
    os.makedirs(d, exist_ok=True)
np.random.seed(CFG.SEED)
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
# # 2. Load data & persiapan fitur
#
# - Kolom ID & label tidak dipakai sebagai fitur. Kolom mirip ID (unik per baris) dibuang otomatis.
# - Teks angka → angka; tanggal → bagian tanggal (jam & hari penting untuk fraud!).
# - **Robust scaling** (median/IQR) dipakai karena anomali sendiri merusak mean/std.

# %%
train = smart_read(CFG.TRAIN_PATH)
test = smart_read(CFG.TEST_PATH) if CFG.TEST_PATH and os.path.exists(CFG.TEST_PATH) else None
sample = smart_read(CFG.SAMPLE_SUB_PATH) if os.path.exists(CFG.SAMPLE_SUB_PATH) else None
LABEL = CFG.LABEL_COL if CFG.LABEL_COL in train.columns else None
y = pd.to_numeric(train[LABEL], errors="coerce").fillna(0).astype(int).clip(0, 1).values if LABEL else None
print(f"train {train.shape} | test {None if test is None else test.shape} | label: {LABEL}")
if CFG.RUN_MODE == "fast" and len(train) > CFG.FAST_MAX_ROWS:
    keep = np.random.RandomState(CFG.SEED).choice(len(train), CFG.FAST_MAX_ROWS, replace=False)
    train = train.iloc[keep].reset_index(drop=True)
    y = y[keep] if y is not None else None
    print(f"[FAST MODE] {len(train)} baris")


def prepare(df):
    df = df.drop(columns=[c for c in [CFG.ID_COL, LABEL] + list(CFG.DROP_COLS) if c and c in df.columns]).copy()
    for c in list(df.columns):
        if df[c].dtype == object:
            s = df[c].astype(str)
            if s.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > .8 or s.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}").mean() > .8:
                dt = pd.to_datetime(df[c], errors="coerce", format="mixed")
                df[c + "_hour"], df[c + "_dow"], df[c + "_day"] = dt.dt.hour, dt.dt.dayofweek, dt.dt.day
                df = df.drop(columns=c)
                continue
            num = pd.to_numeric(s.str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
            if num.notna().mean() > .9 and s.str.contains(r"\d").mean() > .9:
                df[c] = num
    return df


Xtr = prepare(train)
Xte = prepare(test).reindex(columns=Xtr.columns) if test is not None else None
id_like = [c for c in Xtr.columns if Xtr[c].nunique() > 0.95 * len(Xtr) and Xtr[c].dtype == object]
Xtr, Xte = Xtr.drop(columns=id_like), (Xte.drop(columns=id_like) if Xte is not None else None)
NUM = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
CAT = [c for c in Xtr.columns if c not in NUM and Xtr[c].nunique() <= CFG.MAX_ONEHOT_CARD]
print(f"Fitur numerik: {len(NUM)} | kategorikal: {len(CAT)} | dibuang (mirip ID / kardinalitas tinggi): "
      f"{id_like + [c for c in Xtr.columns if c not in NUM + CAT]}")
pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), RobustScaler()), NUM)] +
                        ([("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                OneHotEncoder(handle_unknown="ignore", sparse_output=False)), CAT)] if CAT else []))
FIT_DF = pd.concat([Xtr, Xte]) if (CFG.FIT_ON == "train+test" and Xte is not None) else Xtr
if y is not None and CFG.FIT_ONLY_NORMAL:
    FIT_DF = Xtr[y == 0]
    print(f"Detektor di-fit pada {len(FIT_DF)} baris normal (label = 0)")
pre.fit(FIT_DF.astype({c: str for c in CAT}))
to_mat = lambda d: pre.transform(d.astype({c: str for c in CAT})).astype(np.float32)  # noqa: E731
M_fit, M_tr = to_mat(FIT_DF), to_mat(Xtr)
M_te = to_mat(Xte) if Xte is not None else None
M_tr = np.clip(np.nan_to_num(M_tr), -50, 50)
M_fit = np.clip(np.nan_to_num(M_fit), -50, 50)
M_te = np.clip(np.nan_to_num(M_te), -50, 50) if M_te is not None else None
CONT = (float(y.mean()) if y is not None and y.mean() > 0 else 0.05) if CFG.CONTAMINATION == "auto" else float(CFG.CONTAMINATION)
CONT = float(np.clip(CONT, 0.001, 0.4))
print(f"Matriks fitur: {M_tr.shape[1]} kolom | contamination = {CONT:.4f}")

# %% [markdown]
# # 3. EDA singkat
#
# Kalau ada label: bandingkan distribusi fitur anomali vs normal (median & rasio). Fitur dengan rasio ekstrem adalah
# kandidat "red flag" untuk laporan.

# %%
desc = Xtr[NUM].describe().T
display(desc.round(3).head(20))
if y is not None:
    print(f"Prevalensi anomali (label): {y.mean():.4f} ({y.sum()} dari {len(y)})")
    prof = pd.DataFrame({"median_normal": Xtr.loc[y == 0, NUM].median(), "median_anomali": Xtr.loc[y == 1, NUM].median()})
    prof["rasio"] = prof["median_anomali"] / prof["median_normal"].replace(0, np.nan)
    display(prof.round(3))
    save_table(prof.reset_index(names="feature"), "profile_anomaly_vs_normal")
k = min(6, len(NUM))
if k:
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3))
    for a, c in zip(np.atleast_1d(axes), NUM[:k]):
        v = Xtr[c]
        if y is not None:
            a.hist([v[y == 0].dropna(), v[y == 1].dropna()], bins=30, density=True, label=["normal", "anomali"])
            a.legend(fontsize=7)
        else:
            a.hist(v.dropna(), bins=30)
        a.set_title(c, fontsize=9)
    save_fig("feature_distributions", fig)

# %% [markdown]
# # 4. Detektor unsupervised
#
# | Detektor | Ide | Kuat untuk |
# |---|---|---|
# | Isolation Forest | anomali mudah "diisolasi" dengan sedikit potongan acak | data campuran, skala besar |
# | LOF | kepadatan lokal jauh lebih rendah dari tetangga | anomali lokal / cluster kecil |
# | One-Class SVM | batas di sekitar data normal | data kecil-menengah, dimensi sedang |
# | Robust Z (MAD) | nilai ekstrem per fitur (maks \|z\|) | outlier univariat, mudah dijelaskan |
# | PCA reconstruction | gagal direkonstruksi dari komponen utama | pola korelasi yang dilanggar |
# | Autoencoder | error rekonstruksi jaringan saraf | hubungan non-linear |
#
# Semua skor diubah ke **rank 0–1** (makin besar = makin aneh) supaya bisa dirata-rata.

# %%
def to_rank(s):
    return (rankdata(s) - 1) / max(1, len(s) - 1)


def fit_score(method):
    """Return (skor train, skor test) — makin besar makin anomali."""
    if method == "iforest":
        m = IsolationForest(n_estimators=300, contamination=CONT, random_state=CFG.SEED, n_jobs=-1).fit(M_fit)
        return -m.score_samples(M_tr), (-m.score_samples(M_te) if M_te is not None else None)
    if method == "lof":
        n = min(35, max(5, len(M_fit) // 50))
        m = LocalOutlierFactor(n_neighbors=n, novelty=True, contamination=CONT).fit(M_fit)
        return -m.score_samples(M_tr), (-m.score_samples(M_te) if M_te is not None else None)
    if method == "ocsvm":
        sub = M_fit if len(M_fit) <= 5000 else M_fit[np.random.RandomState(0).choice(len(M_fit), 5000, replace=False)]
        m = OneClassSVM(nu=min(0.5, max(CONT, 0.01)), gamma="scale").fit(sub)
        return -m.score_samples(M_tr), (-m.score_samples(M_te) if M_te is not None else None)
    if method == "robust_z":
        nn_ = len(NUM)
        f = lambda M: np.abs(M[:, :nn_]).max(1) if nn_ else np.zeros(len(M))  # noqa: E731
        return f(M_tr), (f(M_te) if M_te is not None else None)
    if method == "pca":
        k = max(1, min(M_fit.shape[1] - 1, int(np.ceil(M_fit.shape[1] * 0.5)))) if M_fit.shape[1] > 1 else 1
        p = PCA(n_components=k, random_state=CFG.SEED).fit(M_fit)
        f = lambda M: ((M - p.inverse_transform(p.transform(M))) ** 2).sum(1)  # noqa: E731
        return f(M_tr), (f(M_te) if M_te is not None else None)
    if method == "autoencoder":
        if not HAS_TORCH:
            return None, None
        torch.manual_seed(CFG.SEED)
        d = M_fit.shape[1]
        h = max(2, min(64, d // 2))
        net = nn.Sequential(nn.Linear(d, h * 2), nn.ReLU(), nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, h * 2), nn.ReLU(), nn.Linear(h * 2, d))
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
        Xt = torch.tensor(M_fit)
        for ep in range(60 if CFG.RUN_MODE == "full" else 20):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                b = Xt[perm[i:i + 256]]
                loss = ((net(b) - b) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            f = lambda M: ((net(torch.tensor(M)) - torch.tensor(M)) ** 2).mean(1).numpy()  # noqa: E731
            return f(M_tr), (f(M_te) if M_te is not None else None)
    raise ValueError(method)


SCORES_TR, SCORES_TE = {}, {}
for meth in CFG.METHODS:
    t0 = time.time()
    try:
        s_tr, s_te = fit_score(meth)
        if s_tr is None:
            print(f"  {meth}: dilewati (library tidak ada)")
            continue
        SCORES_TR[meth], SCORES_TE[meth] = s_tr, s_te
        msg = f"  {meth:12s} {time.time() - t0:5.1f}s"
        if y is not None and 0 < y.sum() < len(y):
            msg += f" | AUC {roc_auc_score(y, s_tr):.4f} | AP {average_precision_score(y, s_tr):.4f}"
        print(msg)
    except Exception as e:  # noqa
        print(f"  ❌ {meth} gagal: {type(e).__name__}: {str(e)[:150]}")
if not SCORES_TR:
    raise RuntimeError("Semua detektor gagal.")
RANK_TR = pd.DataFrame({k: to_rank(v) for k, v in SCORES_TR.items()})
both = {k: np.concatenate([SCORES_TR[k], SCORES_TE[k]]) for k in SCORES_TR} if M_te is not None else None
RANK_TE = pd.DataFrame({k: to_rank(v)[len(M_tr):] for k, v in both.items()}) if both else None
RANK_TR["ensemble"] = RANK_TR.mean(1)
if RANK_TE is not None:
    RANK_TE["ensemble"] = RANK_TE.mean(1)
corr = RANK_TR.corr(method="spearman")
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(corr, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, fontsize=8)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
ax.set_title("Korelasi rank antar detektor (Spearman)")
save_fig("detector_correlation", fig)

# %% [markdown]
# # 5. Evaluasi (kalau ada label) + model supervised
#
# - **AUC**: seberapa baik skor mengurutkan anomali di atas normal. **AP** (average precision) lebih jujur untuk
#   anomali langka. **Precision@k**: dari k baris teratas, berapa yang benar anomali (k = jumlah anomali sebenarnya).
# - **Supervised** (HGB/LGBM, CV stratified) memakai label langsung; **stacking** menambahkan skor unsupervised
#   sebagai fitur. Pilih yang AP OOF-nya terbaik.

# %%
FINAL_KEY, FINAL_TR, FINAL_TE = "ensemble", RANK_TR["ensemble"].values, (RANK_TE["ensemble"].values if RANK_TE is not None else None)
EVAL = None
if y is not None and 0 < y.sum() < len(y):
    k = int(y.sum())
    rows = []
    for c in RANK_TR.columns:
        s = RANK_TR[c].values
        rows.append({"model": c, "auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
                     "precision_at_k": y[np.argsort(-s)[:k]].mean()})

    def cv_supervised(M, Mt):
        oof, te = np.zeros(len(M)), (np.zeros(len(Mt)) if Mt is not None else None)
        skf = StratifiedKFold(min(CFG.N_FOLDS, int(y.sum())), shuffle=True, random_state=CFG.SEED)
        for tr, va in skf.split(M, y):
            m = (lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, is_unbalance=True, verbose=-1,
                                    random_state=CFG.SEED) if HAS_LGB else
                 HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=CFG.SEED))
            m.fit(M[tr], y[tr])
            oof[va] = m.predict_proba(M[va])[:, 1]
            if te is not None:
                te += m.predict_proba(Mt)[:, 1] / skf.n_splits
        return oof, te

    if CFG.SUPERVISED and y.sum() >= 5:
        sup_oof, sup_te = cv_supervised(M_tr, M_te)
        U_tr = RANK_TR.drop(columns="ensemble").values
        U_te = RANK_TE.drop(columns="ensemble").values if RANK_TE is not None else None
        stk_oof, stk_te = cv_supervised(np.hstack([M_tr, U_tr]), np.hstack([M_te, U_te]) if M_te is not None else None)
        extra = {"supervised": (sup_oof, sup_te), "stacking": (stk_oof, stk_te)}
        for name, (o, t) in extra.items():
            rows.append({"model": name, "auc": roc_auc_score(y, o), "ap": average_precision_score(y, o),
                         "precision_at_k": y[np.argsort(-o)[:k]].mean()})
    EVAL = pd.DataFrame(rows).sort_values("ap", ascending=False).reset_index(drop=True)
    display(EVAL.round(4))
    save_table(EVAL, "model_comparison")
    FINAL_KEY = EVAL["model"].iloc[0]
    if FINAL_KEY in RANK_TR.columns:
        FINAL_TR = RANK_TR[FINAL_KEY].values
        FINAL_TE = RANK_TE[FINAL_KEY].values if RANK_TE is not None else None
    else:
        FINAL_TR, FINAL_TE = extra[FINAL_KEY]
    print(f"Model terpilih (AP terbaik): {FINAL_KEY}")
else:
    print("ℹ️ Tanpa label → model final = ensemble rank semua detektor. Validasi manual: cek tabel top anomali.")

# %% [markdown]
# # 6. Threshold
#
# - Tanpa label: baris dengan skor di atas kuantil `1 − CONTAMINATION` = anomali.
# - Dengan label: threshold dengan F1 terbaik di OOF (dibandingkan dengan threshold contamination).

# %%
TH = float(np.quantile(FINAL_TR, 1 - CONT))
if y is not None and 0 < y.sum() < len(y):
    grid = np.unique(np.append(np.quantile(FINAL_TR, np.linspace(0.5, 0.999, 200)), TH))
    f1s = [f1_score(y, FINAL_TR >= t) for t in grid]
    t_best = float(grid[int(np.argmax(f1s))])
    print(f"F1 @contamination = {f1_score(y, FINAL_TR >= TH):.4f} | F1 @best = {max(f1s):.4f}")
    TH = t_best
    print(f"Precision {precision_score(y, FINAL_TR >= TH):.4f} | Recall {recall_score(y, FINAL_TR >= TH):.4f}")
FLAG_TR = (FINAL_TR >= TH).astype(int)
print(f"Threshold = {TH:.4f} → {FLAG_TR.mean():.2%} baris train ditandai anomali")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
if y is not None:
    ax[0].hist([FINAL_TR[y == 0], FINAL_TR[y == 1]], bins=40, density=True, label=["normal", "anomali"])
    ax[0].legend()
else:
    ax[0].hist(FINAL_TR, bins=40)
ax[0].axvline(TH, c="k", ls="--")
ax[0].set_title(f"Distribusi skor ({FINAL_KEY})")
emb = PCA(2, random_state=CFG.SEED).fit_transform(M_tr)
sc = ax[1].scatter(emb[:, 0], emb[:, 1], c=FINAL_TR, cmap="magma", s=6)
ax[1].scatter(emb[FLAG_TR == 1, 0], emb[FLAG_TR == 1, 1], facecolors="none", edgecolors="cyan", s=20, lw=.6, label="ditandai")
ax[1].legend(fontsize=7)
ax[1].set_title("PCA 2D (warna = skor anomali)")
plt.colorbar(sc, ax=ax[1])
save_fig("score_distribution_pca", fig)

# %% [markdown]
# # 7. Penjelasan: mengapa baris ini anomali?
#
# - **Fitur penyumbang** per baris = fitur numerik dengan |robust z| terbesar (berapa IQR dari median).
# - **Surrogate tree**: pohon keputusan dangkal meniru label "anomali" → aturan IF–THEN yang mudah dibaca juri.

# %%
Z = pd.DataFrame(M_tr[:, :len(NUM)], columns=NUM)
top_idx = np.argsort(-FINAL_TR)[:20]


def reasons(i, k=3):
    z = Z.iloc[i]
    top = z.abs().sort_values(ascending=False).head(k)
    return "; ".join(f"{c}={train.loc[i, c] if c in train.columns else '?'} ({'tinggi' if z[c] > 0 else 'rendah'}, {abs(z[c]):.1f} IQR)"
                     for c in top.index)


top_tab = pd.DataFrame({"row": top_idx, "id": train[CFG.ID_COL].values[top_idx] if CFG.ID_COL in train.columns else top_idx,
                        "score": FINAL_TR[top_idx], "label": y[top_idx] if y is not None else np.nan,
                        "alasan_utama": [reasons(i) for i in top_idx]})
display(top_tab.head(10))
save_table(top_tab, "top_anomalies_explained")
if len(NUM) and FLAG_TR.sum() >= 3:
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(5, len(Xtr) // 200), class_weight="balanced",
                                  random_state=CFG.SEED).fit(Xtr[NUM].fillna(Xtr[NUM].median()), FLAG_TR)
    rules = export_text(tree, feature_names=NUM)
    print("Aturan surrogate (meniru flag anomali):\n" + rules)
    open(os.path.join(TAB_DIR, "surrogate_rules.txt"), "w", encoding="utf-8").write(rules)
    imp = pd.Series(tree.feature_importances_, NUM).sort_values(ascending=False)
    print("Fitur terpenting untuk membedakan anomali:", imp[imp > 0].round(3).to_dict())

# %% [markdown]
# # 8. Submission + ringkasan
#
# `SUBMISSION_MODE="auto"`: kalau sample_submission berisi 0/1 → label; selain itu skor (makin besar makin anomali).

# %%
SUB_PATH = None
if FINAL_TE is not None:
    mode = CFG.SUBMISSION_MODE
    if mode == "auto":
        mode = "label" if (sample is not None and sample.shape[1] > 1 and set(pd.unique(sample.iloc[:, 1].dropna())) <= {0, 1}) else "score"
    vals = (FINAL_TE >= TH).astype(int) if mode == "label" else FINAL_TE
    id_name = sample.columns[0] if sample is not None else (CFG.ID_COL or "id")
    val_name = sample.columns[1] if sample is not None and sample.shape[1] > 1 else ("is_anomaly" if mode == "label" else "anomaly_score")
    ids = test[CFG.ID_COL].values if CFG.ID_COL in test.columns else np.arange(len(test))
    sub = pd.DataFrame({id_name: ids, val_name: vals})
    SUB_PATH = os.path.join(SUB_DIR, "submission.csv")
    sub.to_csv(SUB_PATH, index=False)
    print(f"✅ Submission ({mode}): {SUB_PATH} | {len(sub)} baris | ditandai anomali: {(FINAL_TE >= TH).mean():.2%}")
pd.DataFrame({"id": train[CFG.ID_COL] if CFG.ID_COL in train.columns else np.arange(len(train)), "anomaly_score": FINAL_TR,
              "is_anomaly": FLAG_TR}).to_csv(os.path.join(TAB_DIR, "train_anomaly_scores.csv"), index=False)
summ = {"final_model": FINAL_KEY, "methods": list(SCORES_TR), "contamination": CONT, "threshold": TH, "n_train": len(train),
        "flagged_train_pct": float(FLAG_TR.mean()), "submission": SUB_PATH}
if EVAL is not None:
    best = EVAL.iloc[0]
    summ.update({"auc": round(float(best.auc), 4), "ap": round(float(best.ap), 4), "precision_at_k": round(float(best.precision_at_k), 4)})
text = f"""# Anomaly detection summary

- Detectors: {', '.join(SCORES_TR)}; final score: **{FINAL_KEY}**. Contamination {CONT:.3f}, {FLAG_TR.mean():.1%} of training rows flagged.
{'- Labelled evaluation: AUC ' + str(summ.get('auc')) + ', average precision ' + str(summ.get('ap')) + ', precision@k ' + str(summ.get('precision_at_k')) + '.' if EVAL is not None else '- No labels: detectors combined by rank averaging; top anomalies reviewed manually.'}
- Typical reasons for the top anomaly: {top_tab['alasan_utama'].iloc[0] if len(top_tab) else '-'}.

**Draft sentences (paraphrase):**
- "Because anomalies are rare and heterogeneous, we combined isolation-, density- and reconstruction-based detectors by averaging their rank-normalised scores."
- "For each flagged record we report the features that deviate most from the median, measured in interquartile ranges."
"""
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(text)
json.dump(summ, open(os.path.join(OUT, "run_summary.json"), "w"), indent=2, default=str)
print(text)
print("✅ Notebook selesai.")

# %% [markdown]
# # 9. Troubleshooting, recipes, checklist
#
# | Gejala | Solusi |
# |---|---|
# | Semua detektor AUC ≈ 0.5 | fitur kurang informatif → buat fitur rasio/frekuensi/agregat per entitas (mis. jumlah transaksi per kartu per jam) |
# | Anomali didominasi satu fitur | robust_z terlalu kuat → hapus dari `METHODS` atau transformasi log fitur skewed |
# | LOF/OCSVM lambat | data besar → `METHODS=["iforest","pca","robust_z"]` |
# | Label tersedia tapi sedikit (< 20) | `SUPERVISED=False`, pakai label hanya untuk evaluasi & threshold |
#
# ```python
# CFG.FIT_ON = "train+test"                     # unsupervised murni: pakai semua data
# CFG.METHODS = ["iforest", "lof", "pca"]       # cepat untuk data besar
# ```
#
# - [ ] Kolom ID/label/waktu-kejadian yang bocor tidak masuk fitur
# - [ ] Contamination masuk akal (cek aturan / domain)
# - [ ] Arah skor submission benar (besar = anomali?) dan tipe (skor vs 0/1) sesuai aturan
# - [ ] Contoh top anomali & alasannya masuk laporan
