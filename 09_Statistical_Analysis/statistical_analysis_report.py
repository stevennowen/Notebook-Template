# -*- coding: utf-8 -*-
# =====================================================================================================
# statistical_analysis_report.py — versi SCRIPT dari statistical_analysis_report.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python statistical_analysis_report.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb statistical_analysis_report.py
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
# # Statistical Analysis & Insight Report — tanpa lomba prediksi
#
# Untuk lomba **data analytics** yang dinilai dari *insight* dan laporan, bukan skor leaderboard: survei kepuasan,
# eksperimen A/B (promo, desain aplikasi), evaluasi program, analisis faktor yang berhubungan dengan suatu hasil
# (konversi, pembelian, nilai, kepuasan). Notebook memilih uji statistik yang tepat secara otomatis, menghitung
# **effect size + interval kepercayaan**, mengoreksi uji berganda, lalu menulis kalimat insight & laporan DOCX.
#
# ## Pipeline
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json) → load (csv/xlsx/parquet) → kualitas data (missing, duplikat, tipe, outlier)
# → Statistik deskriptif + grafik distribusi
# → Uji hubungan setiap variabel dengan OUTCOME (pemilihan uji otomatis):
#     numerik~kategori: Welch t / Mann-Whitney / ANOVA / Kruskal-Wallis (+ Cohen d, rank-biserial, eta², epsilon²)
#     kategori~kategori: chi-square / Fisher (+ Cramér's V); numerik~numerik: Pearson & Spearman
#   → koreksi FDR (Benjamini–Hochberg)
# → A/B test (GROUP_COL): selisih + 95% CI, lift relatif, uji, bootstrap, power & sample size, cek SRM
# → Regresi multivariabel: OLS (outcome angka) / Logistic (outcome 0/1) + 95% CI, p-value, odds ratio
# → Analisis segmen (efek per subgrup) & matriks korelasi
# → Insight otomatis (kalimat English siap parafrase) + laporan Markdown & DOCX
# ```
#
# **Dua format:** `statistical_analysis_report.ipynb` / `.py`. Headless: `python tools/run_task.py --task stats --format py`.

# %%
import os


class CFG:
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "statistical_analysis_report"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"
    DATA_PATH = "./data/data.csv"
    SHEET_NAME = 0                  # Excel
    ID_COL = None
    OUTCOME_COL = None              # variabel hasil utama (angka atau 0/1/kategori); None = deskriptif + korelasi saja
    GROUP_COL = None                # variabel perlakuan/kelompok untuk A/B test (mis. "group" berisi A/B)
    CONTROL_LABEL = None            # nilai grup kontrol (None = nilai pertama setelah diurutkan)
    SEGMENT_COLS = None             # kolom segmen untuk efek per subgrup (None = kategorikal ≤ 8 level)
    DROP_COLS = []
    ALPHA = 0.05
    MAX_CATEGORIES = 20             # kategori lebih banyak → digabung "Other"
    N_BOOTSTRAP = 2000
    EXPECTED_SPLIT = None           # A/B: rasio alokasi yang direncanakan, mis. [0.5, 0.5] (cek SRM)
    MDE = None                      # minimum detectable effect untuk power analysis (None = efek yang teramati)
    REPORT_TITLE = "Statistical Analysis Report"
    TEAM_NAME = "Team"
    WRITE_DOCX = True
    OUTPUT_DIR = "./outputs/statistical_analysis_report/"
    FAST_N_BOOTSTRAP = 300
    FIG_DPI = 150

# %%
# >>> USER OVERRIDE
# CFG.OUTCOME_COL = "converted"
# CFG.GROUP_COL = "group"; CFG.CONTROL_LABEL = "A"
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
# Inti: numpy, pandas, scipy, matplotlib. `statsmodels` dipakai kalau ada (regresi lebih lengkap); tanpa statsmodels,
# OLS & logistic + standard error dihitung sendiri (hasil sama untuk model standar). `python-docx` untuk laporan Word.

# %%
import importlib.util
import subprocess
import sys

for imp, pip_name in {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "matplotlib": "matplotlib",
                      "statsmodels": "statsmodels", "docx": "python-docx", "openpyxl": "openpyxl"}.items():
    if importlib.util.find_spec(imp) is None and CFG.AUTO_INSTALL:
        try:
            print("Installing", pip_name)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as e:  # noqa
            print("install gagal:", pip_name, e)

import json
import math
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
try:  # pandas ≥ 3: teks default bertipe "str"; notebook ini ditulis untuk tipe object (perilaku pandas 2)
    pd.set_option("future.infer_string", False)
except Exception:  # noqa
    pass
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except Exception:  # noqa
    display = print
HAS_SM = importlib.util.find_spec("statsmodels") is not None
HAS_DOCX = importlib.util.find_spec("docx") is not None
print(f"statsmodels: {HAS_SM} | python-docx: {HAS_DOCX}")
if CFG.RUN_MODE == "fast":
    CFG.N_BOOTSTRAP = min(CFG.N_BOOTSTRAP, CFG.FAST_N_BOOTSTRAP)
OUT = CFG.OUTPUT_DIR
FIG_DIR, TAB_DIR = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in (OUT, FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)
RNG = np.random.default_rng(CFG.SEED)
FIGS, INSIGHTS = [], []
_FIG = [0]


def save_fig(name, fig=None, caption=""):
    fig = fig or plt.gcf()
    _FIG[0] += 1
    p = os.path.join(FIG_DIR, f"fig{_FIG[0]:02d}_{name}.png")
    fig.savefig(p, dpi=CFG.FIG_DPI, bbox_inches="tight")
    print("  [fig] saved:", p)
    FIGS.append((p, caption or name.replace("_", " ")))
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
        return pd.read_excel(path, sheet_name=CFG.SHEET_NAME, **kw)
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


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

# %% [markdown]
# # 2. Load & kualitas data
#
# - Teks angka (`"Rp 5.000"`, `"12,5"`) → angka kalau ≥ 90% baris bisa dikonversi.
# - Kategori dengan level terlalu banyak → level jarang digabung "Other" (uji chi-square butuh frekuensi cukup).

# %%
df = smart_read(CFG.DATA_PATH)
df = df.drop(columns=[c for c in CFG.DROP_COLS if c in df.columns])
print(f"Data: {df.shape[0]} baris × {df.shape[1]} kolom")
for c in df.columns:
    if df[c].dtype == object:
        s = df[c].astype(str).str.strip()
        num = pd.to_numeric(s.str.replace(r"[^\d,.\-]", "", regex=True).str.replace(",", ".", regex=False), errors="coerce")
        if num.notna().mean() > 0.9 and s.str.contains(r"\d").mean() > 0.9:
            df[c] = num
quality = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_pct": (df.isna().mean() * 100).round(2),
                        "n_unique": df.nunique()}).reset_index(names="column")
display(quality)
save_table(quality, "data_quality")
dups = df.duplicated().sum()
print(f"Baris duplikat: {dups}" + (" ⚠️ (cek: entri ganda atau memang wajar?)" if dups else " ✅"))
ID = CFG.ID_COL if CFG.ID_COL in df.columns else None
OUTCOME = CFG.OUTCOME_COL if CFG.OUTCOME_COL in df.columns else None
GROUP = CFG.GROUP_COL if CFG.GROUP_COL in df.columns else None
NUM = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in (ID,) and df[c].nunique() > 2]
BIN = [c for c in df.columns if c not in (ID,) and df[c].dropna().nunique() == 2]
CAT = [c for c in df.columns if c not in NUM + BIN + [ID] and 2 < df[c].nunique() <= max(CFG.MAX_CATEGORIES * 5, 50)]
for c in CAT:
    vc = df[c].value_counts()
    if len(vc) > CFG.MAX_CATEGORIES:
        df[c] = df[c].where(df[c].isin(vc.index[:CFG.MAX_CATEGORIES - 1]), "Other")
OUT_TYPE = None
if OUTCOME:
    OUT_TYPE = "binary" if OUTCOME in BIN else ("numeric" if OUTCOME in NUM else "categorical")
    if OUT_TYPE == "binary":
        vals = sorted(df[OUTCOME].dropna().unique(), key=str)
        pos = 1 if 1 in vals else vals[-1]
        df["_y"] = (df[OUTCOME] == pos).astype(float).where(df[OUTCOME].notna())
        print(f"Outcome '{OUTCOME}' = biner (positif = {pos}) | rate {df['_y'].mean():.3f}")
    else:
        print(f"Outcome '{OUTCOME}' = {OUT_TYPE}")
print(f"Numerik {len(NUM)} | biner {len(BIN)} | kategorikal {len(CAT)} | grup A/B: {GROUP}")

# %% [markdown]
# # 3. Statistik deskriptif
#
# **Untuk laporan:** laporkan *median (IQR)* untuk variabel miring (|skew| > 1) dan *mean (SD)* untuk yang simetris.

# %%
desc = df[NUM].describe().T
desc["iqr"] = desc["75%"] - desc["25%"]
desc["skew"] = df[NUM].skew()
desc["report_as"] = np.where(desc["skew"].abs() > 1, "median (IQR)", "mean (SD)")
display(desc.round(3))
save_table(desc.reset_index(names="variable"), "descriptive_numeric")
cat_tabs = []
for c in CAT + [b for b in BIN if b != OUTCOME]:
    vc = df[c].value_counts(dropna=False)
    cat_tabs.append(pd.DataFrame({"variable": c, "level": vc.index.astype(str), "n": vc.values, "pct": (100 * vc.values / len(df)).round(1)}))
if cat_tabs:
    save_table(pd.concat(cat_tabs), "descriptive_categorical")
k = min(6, len(NUM))
if k:
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3))
    for a, c in zip(np.atleast_1d(axes), NUM[:k]):
        a.hist(df[c].dropna(), bins=30, color="#4C72B0")
        a.set_title(c, fontsize=9)
    save_fig("numeric_distributions", fig, "Distribution of numeric variables")

# %% [markdown]
# # 4. Uji hubungan dengan OUTCOME (pemilihan uji otomatis)
#
# | Outcome | Prediktor | Uji | Effect size |
# |---|---|---|---|
# | angka | 2 kelompok | Welch t-test (normal) / Mann-Whitney U | Cohen's d / rank-biserial r |
# | angka | > 2 kelompok | Welch ANOVA-ish (one-way) / Kruskal-Wallis | η² / ε² |
# | angka | angka | Pearson / Spearman | r / ρ |
# | biner/kategori | kategori | chi-square (Fisher untuk 2×2 kecil) | Cramér's V |
# | biner | angka | Mann-Whitney U (antar kelas outcome) | rank-biserial r (≈ 2·AUC − 1) |
#
# Normalitas dinilai dengan Shapiro–Wilk (sampel ≤ 5000) **dan** ukuran sampel (n ≥ 30 per grup → t-test cukup robust).
# p-value dikoreksi **Benjamini–Hochberg** (kolom `p_fdr`) karena banyak uji sekaligus.

# %%
def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / max(1, na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0


def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return math.sqrt(chi2 / (n * max(1, min(r, k) - 1))) if n else 0.0


def is_normal(x):
    x = np.asarray(x)
    if len(x) < 8:
        return False
    if len(x) >= 30:
        return abs(stats.skew(x)) < 1
    return stats.shapiro(x[:5000])[1] > 0.05


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


def test_numeric_by_groups(y, g):
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    groups = [v["y"].values for _, v in d.groupby("g") if len(v) >= 3]
    if len(groups) < 2:
        return None
    if len(groups) == 2:
        a, b = groups
        if is_normal(a) and is_normal(b):
            t = stats.ttest_ind(a, b, equal_var=False)
            return {"test": "Welch t-test", "stat": t.statistic, "p": t.pvalue, "effect": cohens_d(a, b), "effect_name": "Cohen's d"}
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"test": "Mann-Whitney U", "stat": u.statistic, "p": u.pvalue,
                "effect": 1 - 2 * u.statistic / (len(a) * len(b)), "effect_name": "rank-biserial r"}
    n = sum(map(len, groups))
    if all(is_normal(x) for x in groups):
        f = stats.f_oneway(*groups)
        ss_b = sum(len(x) * (x.mean() - d["y"].mean()) ** 2 for x in groups)
        ss_t = ((d["y"] - d["y"].mean()) ** 2).sum()
        return {"test": "one-way ANOVA", "stat": f.statistic, "p": f.pvalue, "effect": ss_b / ss_t if ss_t else 0, "effect_name": "eta²"}
    h = stats.kruskal(*groups)
    return {"test": "Kruskal-Wallis", "stat": h.statistic, "p": h.pvalue, "effect": (h.statistic - len(groups) + 1) / (n - len(groups)),
            "effect_name": "epsilon²"}


def test_cat_cat(a, b):
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    if ct.shape == (2, 2) and (ct.values < 5).any():
        o, p = stats.fisher_exact(ct.values)
        return {"test": "Fisher exact", "stat": o, "p": p, "effect": cramers_v(ct), "effect_name": "Cramér's V"}
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    note = " (⚠️ expected < 5)" if (exp < 5).mean() > 0.2 else ""
    return {"test": "chi-square" + note, "stat": chi2, "p": p, "effect": cramers_v(ct), "effect_name": "Cramér's V"}


rows = []
if OUTCOME:
    for c in [x for x in NUM + BIN + CAT if x not in (OUTCOME, "_y")]:
        try:
            if OUT_TYPE == "numeric":
                if c in NUM:
                    d = df[[c, OUTCOME]].dropna()
                    pr, sp = stats.pearsonr(d[c], d[OUTCOME]), stats.spearmanr(d[c], d[OUTCOME])
                    r = {"test": "Spearman", "stat": sp.statistic, "p": sp.pvalue, "effect": sp.statistic, "effect_name": "rho",
                         "pearson_r": pr.statistic}
                else:
                    r = test_numeric_by_groups(df[OUTCOME], df[c])
            else:
                if c in NUM:
                    r = test_numeric_by_groups(df[c], df[OUTCOME])
                    if r and OUT_TYPE == "binary":
                        r["direction"] = "higher in positive" if df.loc[df["_y"] == 1, c].median() > df.loc[df["_y"] == 0, c].median() else "lower in positive"
                else:
                    r = test_cat_cat(df[c], df[OUTCOME])
            if r:
                rows.append({"variable": c, **r, "n": int(df[[c, OUTCOME]].dropna().shape[0])})
        except Exception as e:  # noqa
            print(f"  uji {c} dilewati: {str(e)[:100]}")
assoc = pd.DataFrame(rows)
if len(assoc):
    assoc["p_fdr"] = bh(assoc["p"].values)
    assoc["significant"] = assoc["p_fdr"] < CFG.ALPHA
    assoc["abs_effect"] = assoc["effect"].abs()
    assoc = assoc.sort_values(["significant", "abs_effect"], ascending=[False, False]).reset_index(drop=True)
    display(assoc.drop(columns="abs_effect").round(4))
    save_table(assoc, "association_tests")
    top = assoc.head(12)
    fig, ax = plt.subplots(figsize=(7, .35 * len(top) + 1.2))
    ax.barh(top["variable"][::-1], top["abs_effect"][::-1], color=np.where(top["significant"][::-1], "#C44E52", "#BBBBBB"))
    ax.set_xlabel("|effect size| (merah = signifikan setelah FDR)")
    ax.set_title(f"Kekuatan hubungan dengan '{OUTCOME}'")
    save_fig("association_effect_sizes", fig, f"Effect sizes of variables associated with {OUTCOME} (red: FDR-significant)")
    for r in assoc[assoc.significant].head(5).itertuples():
        INSIGHTS.append(f"**{r.variable}** is associated with {OUTCOME} ({r.test}, {r.effect_name} = {r.effect:.2f}, FDR-adjusted p = {fmt_p(r.p_fdr)}).")

# %% [markdown]
# ## 4b. Outcome per kategori (dengan 95% CI)
#
# Untuk outcome biner: proporsi + **Wilson CI**. Untuk outcome angka: mean + CI t. Grafik ini biasanya yang paling
# mudah dipahami juri.

# %%
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def outcome_by(col):
    g = df.groupby(col)
    if OUT_TYPE == "binary":
        t = g["_y"].agg(["sum", "count"]).reset_index()
        t["value"] = t["sum"] / t["count"]
        ci = t.apply(lambda r: wilson(r["sum"], r["count"]), axis=1)
    else:
        t = g[OUTCOME].agg(["mean", "std", "count"]).reset_index()
        t["value"] = t["mean"]
        ci = t.apply(lambda r: (r["mean"] - stats.t.ppf(.975, max(1, r["count"] - 1)) * r["std"] / math.sqrt(max(1, r["count"])),
                                r["mean"] + stats.t.ppf(.975, max(1, r["count"] - 1)) * r["std"] / math.sqrt(max(1, r["count"]))), axis=1)
    t["ci_low"], t["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    return t


if OUTCOME and OUT_TYPE in ("binary", "numeric"):
    cats = [c for c in (CAT + [b for b in BIN if b != OUTCOME]) if c != OUTCOME][:6]
    if cats:
        fig, axes = plt.subplots(1, len(cats), figsize=(3.4 * len(cats), 3.2))
        for a, c in zip(np.atleast_1d(axes), cats):
            t = outcome_by(c)
            a.bar(t[c].astype(str), t["value"], yerr=[t["value"] - t["ci_low"], t["ci_high"] - t["value"]], capsize=3, color="#4C72B0")
            a.set_title(c, fontsize=9)
            a.tick_params(axis="x", rotation=30, labelsize=7)
        fig.suptitle(f"{'Rate' if OUT_TYPE == 'binary' else 'Mean'} of {OUTCOME} by category (95% CI)")
        save_fig("outcome_by_category", fig, f"{OUTCOME} by category with 95% confidence intervals")

# %% [markdown]
# # 5. A/B test (`GROUP_COL`)
#
# - **Outcome 0/1**: selisih proporsi (B − A) + 95% CI (Newcombe/Wilson), lift relatif, uji z dua proporsi.
# - **Outcome angka**: selisih mean + 95% CI Welch, Mann-Whitney sebagai cek robust, bootstrap CI.
# - **SRM** (*sample ratio mismatch*): kalau alokasi grup jauh dari rencana → hasil eksperimen patut dicurigai.
# - **Power**: peluang mendeteksi efek sebesar `MDE` dengan n yang ada, dan n per grup yang dibutuhkan untuk power 80%.

# %%
AB = None
if GROUP and OUTCOME and OUT_TYPE in ("binary", "numeric"):
    levels = sorted(df[GROUP].dropna().unique(), key=str)
    ctrl = CFG.CONTROL_LABEL if CFG.CONTROL_LABEL in levels else levels[0]
    ab_rows = []
    counts = df[GROUP].value_counts()
    exp = np.array(CFG.EXPECTED_SPLIT if CFG.EXPECTED_SPLIT else [1 / len(levels)] * len(levels)) * counts.sum()
    srm_p = stats.chisquare(counts.reindex(levels).values, exp[:len(levels)])[1]
    print(f"Ukuran grup: {counts.to_dict()} | SRM p = {fmt_p(srm_p)}" + (" ⚠️ alokasi tidak sesuai rencana!" if srm_p < 0.001 else " ✅"))
    for trt in [l for l in levels if l != ctrl]:
        A, B = df[df[GROUP] == ctrl], df[df[GROUP] == trt]
        if OUT_TYPE == "binary":
            ka, na, kb, nb = A["_y"].sum(), A["_y"].count(), B["_y"].sum(), B["_y"].count()
            pa, pb = ka / na, kb / nb
            la, ua = wilson(ka, na)
            lb, ub = wilson(kb, nb)
            diff = pb - pa
            lo, hi = diff - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2), diff + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)
            pp = (ka + kb) / (na + nb)
            z = diff / math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb)) if 0 < pp < 1 else 0
            p = 2 * stats.norm.sf(abs(z))
            boot = [RNG.binomial(nb, pb) / nb - RNG.binomial(na, pa) / na for _ in range(CFG.N_BOOTSTRAP)]
            mde = CFG.MDE or abs(diff) or 0.01
            se1 = math.sqrt(pa * (1 - pa) / na + max(1e-9, (pa + mde) * (1 - pa - mde)) / nb) if 0 < pa + mde < 1 else np.nan
            power = stats.norm.cdf(mde / se1 - 1.96) if se1 == se1 else np.nan
            pbar = pa + mde / 2
            n_need = math.ceil(((1.96 + 0.8416) ** 2 * 2 * pbar * (1 - pbar)) / mde ** 2) if mde else np.nan
            ab_rows.append({"control": ctrl, "treatment": trt, "n_control": na, "n_treatment": nb, "control_value": pa, "treatment_value": pb,
                            "difference": diff, "ci_low": lo, "ci_high": hi, "relative_lift": diff / pa if pa else np.nan, "test": "two-proportion z",
                            "p": p, "boot_ci_low": np.quantile(boot, .025), "boot_ci_high": np.quantile(boot, .975),
                            "power_at_mde": power, "mde": mde, "n_per_group_for_80pct_power": n_need})
        else:
            a, b = A[OUTCOME].dropna().values, B[OUTCOME].dropna().values
            t = stats.ttest_ind(b, a, equal_var=False)
            va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
            diff = b.mean() - a.mean()
            half = stats.t.ppf(.975, dfw) * math.sqrt(va + vb)
            boot = [RNG.choice(b, len(b)).mean() - RNG.choice(a, len(a)).mean() for _ in range(CFG.N_BOOTSTRAP)]
            sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            mde = CFG.MDE or abs(diff) or sd * 0.1
            power = stats.norm.cdf(mde / math.sqrt(va + vb) - 1.96)
            n_need = math.ceil(2 * ((1.96 + 0.8416) * sd / mde) ** 2) if mde else np.nan
            ab_rows.append({"control": ctrl, "treatment": trt, "n_control": len(a), "n_treatment": len(b), "control_value": a.mean(),
                            "treatment_value": b.mean(), "difference": diff, "ci_low": diff - half, "ci_high": diff + half,
                            "relative_lift": diff / a.mean() if a.mean() else np.nan, "test": "Welch t", "p": t.pvalue,
                            "mann_whitney_p": stats.mannwhitneyu(b, a).pvalue, "cohens_d": cohens_d(b, a),
                            "boot_ci_low": np.quantile(boot, .025), "boot_ci_high": np.quantile(boot, .975),
                            "power_at_mde": power, "mde": mde, "n_per_group_for_80pct_power": n_need})
    AB = pd.DataFrame(ab_rows)
    AB["srm_p"] = srm_p
    display(AB.round(4).T)
    save_table(AB, "ab_test")
    fig, ax = plt.subplots(figsize=(6, 2.2 + .4 * len(AB)))
    ax.errorbar(AB["difference"], range(len(AB)), xerr=[AB["difference"] - AB["ci_low"], AB["ci_high"] - AB["difference"]], fmt="o", capsize=4)
    ax.axvline(0, c="k", lw=.8)
    ax.set_yticks(range(len(AB)), [f"{t} vs {c}" for t, c in zip(AB.treatment, AB.control)])
    ax.set_xlabel(f"selisih {OUTCOME} (95% CI)")
    ax.set_title("Efek perlakuan")
    save_fig("ab_effect_ci", fig, f"Treatment effect on {OUTCOME} with 95% confidence interval")
    for r in AB.itertuples():
        sig = "statistically significant" if r.p < CFG.ALPHA else "not statistically significant"
        unit = "percentage points" if OUT_TYPE == "binary" else "units"
        scale = 100 if OUT_TYPE == "binary" else 1
        INSIGHTS.append(f"Group **{r.treatment}** vs {r.control}: {OUTCOME} changes by {scale * r.difference:+.2f} {unit} "
                        f"(95% CI {scale * r.ci_low:+.2f} to {scale * r.ci_high:+.2f}; relative lift {100 * r.relative_lift:+.1f}%; "
                        f"p = {fmt_p(r.p)}), {sig}. Power to detect this effect: {100 * r.power_at_mde:.0f}%.")

# %% [markdown]
# # 6. Regresi multivariabel (efek "setelah mengontrol variabel lain")
#
# OLS untuk outcome angka, logistic untuk outcome 0/1. Koefisien numerik di-standarisasi (per 1 SD) supaya bisa
# dibandingkan. Kategori → dummy (level pertama = referensi). Laporkan **odds ratio** (logistic) dengan 95% CI.

# %%
REG = None


def design_matrix(cols):
    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c in NUM:
            v = df[c].astype(float)
            X[c + " (per SD)"] = (v - v.mean()) / (v.std() or 1)
        else:
            d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)
            X = pd.concat([X, d], axis=1)
    return X


def ols_fit(X, y):
    Xc = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    dof = max(1, len(y) - Xc.shape[1])
    cov = (resid @ resid / dof) * np.linalg.pinv(Xc.T @ Xc)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    p = 2 * stats.t.sf(np.abs(tval), dof)
    q = stats.t.ppf(.975, dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return pd.DataFrame({"term": ["intercept"] + list(X.columns), "coef": beta, "se": se, "p": p,
                         "ci_low": beta - q * se, "ci_high": beta + q * se}), {"R2": r2, "n": len(y)}


def logit_fit(X, y, iters=100):
    Xc = np.column_stack([np.ones(len(X)), X.values])
    beta = np.zeros(Xc.shape[1])
    for _ in range(iters):                       # Newton-Raphson (IRLS)
        p = 1 / (1 + np.exp(-np.clip(Xc @ beta, -30, 30)))
        W = p * (1 - p)
        H = Xc.T @ (Xc * W[:, None]) + 1e-8 * np.eye(Xc.shape[1])
        step = np.linalg.solve(H, Xc.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    ll = (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum()
    ll0 = len(y) * (y.mean() * np.log(y.mean() + 1e-12) + (1 - y.mean()) * np.log(1 - y.mean() + 1e-12))
    out = pd.DataFrame({"term": ["intercept"] + list(X.columns), "coef": beta, "se": se, "p": pv,
                        "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se})
    out["odds_ratio"], out["or_ci_low"], out["or_ci_high"] = np.exp(out["coef"]), np.exp(out["ci_low"]), np.exp(out["ci_high"])
    return out, {"McFadden_R2": 1 - ll / ll0 if ll0 else np.nan, "n": len(y)}


if OUTCOME and OUT_TYPE in ("binary", "numeric"):
    preds = [c for c in NUM + BIN + CAT if c not in (OUTCOME, "_y", ID)]
    X = design_matrix(preds)
    yv = (df["_y"] if OUT_TYPE == "binary" else df[OUTCOME]).astype(float)
    ok = X.notna().all(1) & yv.notna()
    X, yv = X[ok], yv[ok].values
    X = X.loc[:, X.std() > 0]
    try:
        REG, fit_stats = (logit_fit(X, yv) if OUT_TYPE == "binary" else ols_fit(X, yv))
        REG["significant"] = REG["p"] < CFG.ALPHA
        print(f"Model: {'logistic' if OUT_TYPE == 'binary' else 'OLS'} | n = {fit_stats['n']} | "
              + " | ".join(f"{k} = {v:.3f}" for k, v in fit_stats.items() if k != "n"))
        display(REG.round(4))
        save_table(REG, "regression")
        R = REG[REG.term != "intercept"].copy()
        col, lo, hi = ("odds_ratio", "or_ci_low", "or_ci_high") if OUT_TYPE == "binary" else ("coef", "ci_low", "ci_high")
        R = R.reindex(R[col].sub(1 if OUT_TYPE == "binary" else 0).abs().sort_values().index)
        fig, ax = plt.subplots(figsize=(7, .35 * len(R) + 1.3))
        for i_, r_ in enumerate(R.itertuples()):
            v_, l_, h_ = getattr(r_, col), getattr(r_, lo), getattr(r_, hi)
            ax.errorbar([v_], [i_], xerr=[[v_ - l_], [h_ - v_]], fmt="o", capsize=3, color="#C44E52" if r_.significant else "#888888")
        ax.axvline(1 if OUT_TYPE == "binary" else 0, c="k", lw=.8, ls="--")
        ax.set_yticks(range(len(R)), R["term"], fontsize=8)
        if OUT_TYPE == "binary":
            ax.set_xscale("log")
        ax.set_title(f"{'Odds ratio' if OUT_TYPE == 'binary' else 'Coefficient'} (95% CI) — {OUTCOME}")
        save_fig("regression_forest_plot", fig, f"Multivariable {'logistic' if OUT_TYPE == 'binary' else 'linear'} regression for {OUTCOME}")
        for r in R[R.significant].sort_values("p").head(4).itertuples():
            if OUT_TYPE == "binary":
                INSIGHTS.append(f"Holding other variables constant, **{r.term}** multiplies the odds of {OUTCOME} by {r.odds_ratio:.2f} "
                                f"(95% CI {r.or_ci_low:.2f}–{r.or_ci_high:.2f}, p = {fmt_p(r.p)}).")
            else:
                INSIGHTS.append(f"Holding other variables constant, **{r.term}** is associated with a {r.coef:+.3f} change in {OUTCOME} "
                                f"(95% CI {r.ci_low:+.3f} to {r.ci_high:+.3f}, p = {fmt_p(r.p)}).")
        if HAS_SM:
            print("ℹ️ statsmodels tersedia — bisa dipakai untuk model lanjutan (interaksi, robust SE): "
                  "smf.ols/logit('y ~ x1 + C(x2)', data).fit(cov_type='HC3')")
    except Exception as e:  # noqa
        print("Regresi gagal:", type(e).__name__, str(e)[:200])

# %% [markdown]
# # 7. Segmen & korelasi
#
# - **Efek per segmen**: apakah efek A/B berbeda antar segmen (mis. wilayah)? Hati-hati: banyak segmen = banyak uji;
#   perlakukan sebagai hipotesis lanjutan, bukan kesimpulan utama.
# - **Matriks korelasi Spearman** untuk variabel numerik (+ outcome).

# %%
seg_cols = CFG.SEGMENT_COLS or [c for c in CAT + BIN if c not in (OUTCOME, GROUP, "_y") and df[c].nunique() <= 8][:3]
if GROUP and OUTCOME and OUT_TYPE in ("binary", "numeric") and seg_cols and AB is not None:
    ctrl = AB["control"].iloc[0]
    trt = AB["treatment"].iloc[0]
    seg_rows = []
    yc = "_y" if OUT_TYPE == "binary" else OUTCOME
    for c in seg_cols:
        for lvl, g in df.groupby(c):
            a, b = g[g[GROUP] == ctrl][yc].dropna(), g[g[GROUP] == trt][yc].dropna()
            if len(a) >= 10 and len(b) >= 10:
                diff = b.mean() - a.mean()
                se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                seg_rows.append({"segment_col": c, "level": lvl, "n": len(a) + len(b), "effect": diff, "ci_low": diff - 1.96 * se,
                                 "ci_high": diff + 1.96 * se, "p": 2 * stats.norm.sf(abs(diff / se)) if se else np.nan})
    if seg_rows:
        SEG = pd.DataFrame(seg_rows)
        SEG["p_fdr"] = bh(SEG["p"].fillna(1).values)
        display(SEG.round(4))
        save_table(SEG, "segment_effects")
        fig, ax = plt.subplots(figsize=(6.5, .35 * len(SEG) + 1.3))
        ax.errorbar(SEG["effect"], range(len(SEG)), xerr=[SEG["effect"] - SEG["ci_low"], SEG["ci_high"] - SEG["effect"]], fmt="o", capsize=3)
        ax.axvline(0, c="k", lw=.8)
        ax.axvline(AB["difference"].iloc[0], c="#C44E52", ls="--", lw=1, label="efek keseluruhan")
        ax.set_yticks(range(len(SEG)), [f"{r.segment_col} = {r.level}" for r in SEG.itertuples()], fontsize=8)
        ax.legend(fontsize=8)
        ax.set_title(f"Efek {trt} vs {ctrl} per segmen (95% CI)")
        save_fig("segment_effects", fig, "Treatment effect by segment (exploratory)")
if len(NUM) >= 2:
    cols = NUM + (["_y"] if OUT_TYPE == "binary" else [])
    corr = df[cols].corr(method="spearman").rename(index={"_y": OUTCOME}, columns={"_y": OUTCOME})
    fig, ax = plt.subplots(figsize=(1 + .55 * len(corr), .8 + .5 * len(corr)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title("Korelasi Spearman")
    save_fig("correlation_matrix", fig, "Spearman correlation matrix")
    save_table(corr.reset_index(names="variable"), "correlation_spearman")

# %% [markdown]
# # 8. Insight & laporan (Markdown + DOCX)
#
# Kalimat insight disusun dari hasil di atas (English, siap diparafrase). **Jangan** menyalin mentah: tambahkan konteks
# bisnis/domain dan implikasi. Laporan DOCX berisi ringkasan, tabel utama, dan semua figure bernomor.

# %%
if not INSIGHTS:
    INSIGHTS.append("No statistically significant associations were found at the chosen significance level after FDR correction.")
method_notes = [
    f"Significance level α = {CFG.ALPHA}; p-values from multiple association tests were adjusted with the Benjamini–Hochberg procedure.",
    "Test selection was automatic: Welch t-test or Mann–Whitney U for two groups, ANOVA or Kruskal–Wallis for more groups, chi-square/Fisher for categorical pairs, Spearman correlation for numeric pairs.",
    "Effect sizes are reported alongside p-values because statistical significance alone does not indicate practical importance.",
]
md = [f"# {CFG.REPORT_TITLE}", "", f"Team: {CFG.TEAM_NAME} · Data: {os.path.basename(CFG.DATA_PATH)} ({len(df)} rows)", "",
      "## Key insights", *[f"{i + 1}. {s}" for i, s in enumerate(INSIGHTS)], "", "## Methods", *[f"- {m}" for m in method_notes], ""]
if len(assoc):
    md += ["## Association tests (top 10)", "", assoc.head(10)[["variable", "test", "effect_name", "effect", "p", "p_fdr"]].round(4).to_markdown(index=False)
           if importlib.util.find_spec("tabulate") else assoc.head(10).round(4).to_string(), ""]
if AB is not None:
    md += ["## A/B test", "", AB[["treatment", "control", "control_value", "treatment_value", "difference", "ci_low", "ci_high", "p"]].round(4).to_string(index=False), ""]
md += ["## Figures", *[f"- Figure {i + 1}: {c} (`{os.path.relpath(p, OUT)}`)" for i, (p, c) in enumerate(FIGS)]]
REPORT_MD = os.path.join(OUT, "REPORT.md")
open(REPORT_MD, "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md[:20]))
REPORT_DOCX = None
if CFG.WRITE_DOCX and HAS_DOCX:
    from docx import Document
    from docx.shared import Inches, Pt
    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(11)
    doc.add_heading(CFG.REPORT_TITLE, 0)
    doc.add_paragraph(f"Team: {CFG.TEAM_NAME} · Data: {len(df)} rows × {df.shape[1]} columns")
    doc.add_heading("Key insights", 1)
    for s in INSIGHTS:
        doc.add_paragraph(s.replace("**", ""), style="List Number")
    doc.add_heading("Methods", 1)
    for m in method_notes:
        doc.add_paragraph(m, style="List Bullet")

    def add_table(t, title):
        doc.add_heading(title, 2)
        t = t.round(4).astype(str)
        tb = doc.add_table(rows=1, cols=len(t.columns))
        tb.style = "Light Grid Accent 1"
        for j, c in enumerate(t.columns):
            tb.rows[0].cells[j].text = str(c)
        for r in t.itertuples(index=False):
            cells = tb.add_row().cells
            for j, v in enumerate(r):
                cells[j].text = v

    if len(assoc):
        add_table(assoc.head(10)[["variable", "test", "effect_name", "effect", "p_fdr"]], "Association tests")
    if AB is not None:
        add_table(AB[["treatment", "control", "difference", "ci_low", "ci_high", "p"]], "A/B test")
    if REG is not None:
        cols_ = ["term", "odds_ratio", "or_ci_low", "or_ci_high", "p"] if OUT_TYPE == "binary" else ["term", "coef", "ci_low", "ci_high", "p"]
        add_table(REG[cols_], "Multivariable regression")
    doc.add_heading("Figures", 1)
    for i, (p, c) in enumerate(FIGS):
        doc.add_picture(p, width=Inches(6))
        doc.add_paragraph(f"Figure {i + 1}. {c}")
    REPORT_DOCX = os.path.join(OUT, "REPORT.docx")
    doc.save(REPORT_DOCX)
    print("✅ Laporan DOCX:", REPORT_DOCX)
summ = {"rows": len(df), "outcome": OUTCOME, "outcome_type": OUT_TYPE, "group": GROUP, "n_tests": len(assoc),
        "n_significant_fdr": int(assoc["significant"].sum()) if len(assoc) else 0, "insights": INSIGHTS,
        "report_md": REPORT_MD, "report_docx": REPORT_DOCX}
json.dump(summ, open(os.path.join(OUT, "run_summary.json"), "w"), indent=2, default=str)
print("✅ Notebook selesai.")

# %% [markdown]
# # 9. Troubleshooting & checklist
#
# | Gejala | Solusi |
# |---|---|
# | Semua p sangat kecil | n besar → p hampir selalu kecil; fokus ke **effect size & CI** |
# | `expected < 5` di chi-square | gabungkan level jarang (`MAX_CATEGORIES`) atau pakai Fisher (2×2) |
# | SRM p < 0.001 | alokasi grup bermasalah (bug randomisasi/filter data) — sebutkan sebagai keterbatasan |
# | Regresi gagal / CI sangat lebar | kolinearitas (variabel mirip) atau kategori langka → buang/gabungkan variabel |
# | Outcome kategori > 2 level | ubah jadi biner yang relevan (mis. "puas" vs lainnya) atau analisis per level |
#
# - [ ] Outcome & grup kontrol benar (`OUTCOME_COL`, `CONTROL_LABEL`)
# - [ ] Laporkan effect size + 95% CI, bukan hanya p-value
# - [ ] Korelasi ≠ sebab-akibat (kecuali A/B acak) — tulis dengan hati-hati
# - [ ] Analisis segmen disebut eksploratif
