# -*- coding: utf-8 -*-
# =====================================================================================================
# anomaly_image.py — versi SCRIPT dari anomaly_image.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python anomaly_image.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb anomaly_image.py
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
# # Anomaly Detection — Gambar (visual inspection / cacat produk)
#
# Untuk soal "temukan produk cacat" ketika data latih **hanya berisi gambar normal** (atau cacatnya sangat sedikit):
# kacang/hazelnut retak (AICC Round 1 *The Defected Nuts*), PCB, kain, botol, permukaan logam, foto medis normal vs
# abnormal. Metode utama: **PatchCore-lite** — fitur CNN pretrained per patch gambar normal disimpan di *memory bank*;
# patch gambar uji yang jauh dari semua patch normal = cacat. Hasil: skor per gambar + **heatmap lokasi cacat**.
#
# ## Pipeline
#
# ```text
# CFG + USER OVERRIDE (+ task_config.json) → daftar gambar normal (train) + gambar uji (test) [+ label uji opsional]
# → EDA: ukuran, contoh gambar
# → Backbone CNN (torchvision ResNet/WideResNet pretrained) → fitur layer2+layer3 per patch
# → Memory bank patch normal (+ coreset greedy supaya cepat)
# → Skor: jarak patch ke tetangga terdekat → peta anomali → skor gambar (max / rata-rata top-k)
# → Baseline global: kNN fitur rata-rata gambar
# → Threshold: dari gambar normal yang disisihkan (kuantil) atau F1 terbaik kalau ada label
# → Evaluasi (AUROC, AP, F1), heatmap, submission, ringkasan laporan
# ```
#
# **Dua format:** `anomaly_image.ipynb` / `.py`. Headless: `python tools/run_task.py --task anomaly-image --format py`.
# Kalau data latih berisi **banyak** contoh cacat berlabel → ini sebenarnya klasifikasi biasa: pakai `cv_image_classification`.

# %%
import os


class CFG:
    RUN_MODE = "full"
    SEED = 42
    NOTEBOOK_NAME = "anomaly_image"
    AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"
    TRAIN_IMAGE_DIR = "./data/train/good"   # hanya gambar normal
    TEST_IMAGE_DIR = "./data/test"          # gambar campur (normal & cacat); subfolder good/ & <cacat>/ dikenali sebagai label
    TEST_LABELS_PATH = None                 # opsional: csv (id/gambar, label 0/1) untuk evaluasi
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    OUTPUT_DIR = "./outputs/anomaly_image/"
    BACKBONE = "wide_resnet50_2"   # "resnet18" (cepat) | "resnet50" | "wide_resnet50_2" (standar PatchCore)
    PRETRAINED = True              # False = bobot acak (offline; jauh lebih lemah)
    IMG_SIZE = 256
    CENTER_CROP = 224
    CORESET_RATIO = 0.1            # proporsi patch normal yang disimpan (0.01–0.25)
    MAX_BANK = 60000               # batas jumlah patch di memory bank
    TOPK_SCORE = 1                 # skor gambar = rata-rata k patch terburuk (1 = max)
    VAL_FRAC = 0.15                # gambar normal yang disisihkan untuk kalibrasi threshold (tanpa label)
    THRESHOLD_QUANTILE = 0.99      # threshold = kuantil skor gambar normal val
    SUBMISSION_MODE = "auto"       # "auto" | "score" | "label"
    USE_GPU = True
    BATCH_SIZE = 16
    FAST_MAX_TRAIN = 80
    FAST_BACKBONE = "resnet18"
    FIG_DPI = 150

# %%
# >>> USER OVERRIDE
# CFG.TRAIN_IMAGE_DIR = "./data/train/good"
# CFG.BACKBONE = "resnet18"
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
# # 1. Import & cek library
#
# Butuh torch + torchvision (tidak di-install otomatis — lihat https://pytorch.org/get-started/locally/). Tanpa torch,
# notebook memakai baseline statistik pixel + Isolation Forest (lemah, tapi tetap menghasilkan submission).

# %%
import glob
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
from PIL import Image, ImageFile
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    from IPython.display import display
except Exception:  # noqa
    display = print
try:
    import torch
    import torch.nn.functional as F
    import torchvision
    HAS_TORCH = True
except Exception:  # noqa
    HAS_TORCH = False
DEVICE = "cuda" if HAS_TORCH and CFG.USE_GPU and torch.cuda.is_available() else "cpu"
if CFG.RUN_MODE == "fast":
    CFG.BACKBONE = CFG.FAST_BACKBONE
    print(f"[FAST MODE] backbone {CFG.BACKBONE}, train ≤ {CFG.FAST_MAX_TRAIN} gambar")
print(f"torch/torchvision: {HAS_TORCH} | device: {DEVICE}")
OUT = CFG.OUTPUT_DIR
FIG_DIR, TAB_DIR, SUB_DIR = (os.path.join(OUT, d) for d in ("figures", "tables", "submissions"))
for d in (OUT, FIG_DIR, TAB_DIR, SUB_DIR):
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(CFG.SEED)
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


def list_images(folder):
    out = []
    if folder and os.path.isdir(folder):
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(tuple(CFG.IMAGE_EXTS))]
    return out

# %% [markdown]
# # 2. Data
#
# Label test (kalau ada) dibaca dari `TEST_LABELS_PATH`, atau dari nama subfolder test ala MVTec
# (`test/good/` = 0, `test/<jenis_cacat>/` = 1).

# %%
train_paths = list_images(CFG.TRAIN_IMAGE_DIR)
if not train_paths:
    raise FileNotFoundError(f"Tidak ada gambar normal di {CFG.TRAIN_IMAGE_DIR}")
if CFG.RUN_MODE == "fast" and len(train_paths) > CFG.FAST_MAX_TRAIN:
    train_paths = sorted(rng.choice(train_paths, CFG.FAST_MAX_TRAIN, replace=False).tolist())
perm = rng.permutation(len(train_paths))
n_val = max(1, int(len(train_paths) * CFG.VAL_FRAC)) if len(train_paths) >= 10 else 0
val_paths = [train_paths[i] for i in perm[:n_val]]
bank_paths = [train_paths[i] for i in perm[n_val:]]
test_df = pd.DataFrame({"path": list_images(CFG.TEST_IMAGE_DIR)})
test_df["id"] = test_df["path"].map(lambda p: os.path.splitext(os.path.basename(p))[0])
test_df["label"] = np.nan
if CFG.TEST_LABELS_PATH and os.path.exists(CFG.TEST_LABELS_PATH):
    lab = smart_read(CFG.TEST_LABELS_PATH)
    idc = next((c for c in lab.columns if c.lower() in ("id", "image_id", "image", "filename", "file")), lab.columns[0])
    lc = next((c for c in lab.columns if c != idc), None)
    m = dict(zip(lab[idc].astype(str).map(lambda v: os.path.splitext(os.path.basename(v))[0]), lab[lc]))
    test_df["label"] = test_df["id"].map(m)
else:
    rel = test_df["path"].map(lambda p: os.path.relpath(os.path.dirname(p), CFG.TEST_IMAGE_DIR))
    if (rel != ".").any():
        test_df["label"] = rel.map(lambda r: 0 if r.split(os.sep)[0].lower() in ("good", "normal", "ok") else 1)
HAS_LABEL = test_df["label"].notna().any() and test_df["label"].nunique() > 1
print(f"Normal: {len(bank_paths)} memory bank + {len(val_paths)} kalibrasi | test: {len(test_df)} | label test: {HAS_LABEL}")
k = min(6, len(train_paths))
fig, ax = plt.subplots(1, k, figsize=(2.4 * k, 2.6))
for a, p in zip(np.atleast_1d(ax), train_paths[:k]):
    a.imshow(Image.open(p).convert("RGB"))
    a.set_title("normal", fontsize=8)
    a.axis("off")
save_fig("sample_normal_images", fig)

# %% [markdown]
# # 3. Fitur patch (PatchCore-lite)
#
# Fitur `layer2` & `layer3` ResNet (detail lokal + konteks), `layer3` di-upsample ke ukuran `layer2`, lalu
# di-*average pool* 3×3 (setiap patch "melihat" tetangganya). Satu gambar 224² → 28×28 = 784 vektor patch.

# %%
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
RESAMPLE = getattr(Image, "Resampling", Image)


def load_tensor(p):
    im = Image.open(p).convert("RGB").resize((CFG.IMG_SIZE, CFG.IMG_SIZE), RESAMPLE.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    o = (CFG.IMG_SIZE - CFG.CENTER_CROP) // 2
    a = a[o:o + CFG.CENTER_CROP, o:o + CFG.CENTER_CROP]
    return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))


def build_backbone():
    fn = getattr(torchvision.models, CFG.BACKBONE)
    w = None
    if CFG.PRETRAINED:
        try:
            w = "DEFAULT"
            m = fn(weights=w)
            return m.eval().to(DEVICE), "pretrained"
        except Exception as e:  # noqa
            print("⚠️ bobot pretrained gagal dimuat:", str(e)[:120], "→ bobot acak (hasil jauh lebih lemah)")
    return fn(weights=None).eval().to(DEVICE), "random"


FEATS = {}
if HAS_TORCH:
    BACKBONE, W_KIND = build_backbone()
    BACKBONE.layer2.register_forward_hook(lambda m, i, o: FEATS.__setitem__("l2", o))
    BACKBONE.layer3.register_forward_hook(lambda m, i, o: FEATS.__setitem__("l3", o))
    print(f"Backbone: {CFG.BACKBONE} ({W_KIND})")


def patch_features(paths):
    """Return (N, h, w, C) fitur patch dan (N, C) fitur global."""
    pf, gf = [], []
    with torch.no_grad():
        for i in range(0, len(paths), CFG.BATCH_SIZE):
            x = torch.stack([load_tensor(p) for p in paths[i:i + CFG.BATCH_SIZE]]).to(DEVICE)
            BACKBONE(x)
            l2, l3 = FEATS["l2"], FEATS["l3"]
            l3u = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.cat([l2, l3u], 1)
            f = F.avg_pool2d(f, 3, 1, 1)
            gf.append(torch.cat([l2.mean((2, 3)), l3.mean((2, 3))], 1).cpu())
            pf.append(f.permute(0, 2, 3, 1).cpu().half())
    return torch.cat(pf), torch.cat(gf)


def coreset(x, n, seed=0):
    """Greedy k-center pada proyeksi acak (cepat) → index patch representatif."""
    if n >= len(x):
        return torch.arange(len(x))
    g = torch.Generator().manual_seed(seed)
    proj = x.float() @ torch.randn(x.shape[1], min(128, x.shape[1]), generator=g)
    proj = proj.to(DEVICE)
    sel = [int(torch.randint(len(x), (1,), generator=g))]
    dmin = torch.cdist(proj, proj[sel]).squeeze(1)
    for _ in range(n - 1):
        j = int(torch.argmax(dmin))
        sel.append(j)
        dmin = torch.minimum(dmin, torch.cdist(proj, proj[j:j + 1]).squeeze(1))
    return torch.tensor(sel)


def nn_dist(q, bank, chunk=4096):
    out = []
    b = bank.to(DEVICE).float()
    for i in range(0, len(q), chunk):
        out.append(torch.cdist(q[i:i + chunk].to(DEVICE).float(), b).min(1).values.cpu())
    return torch.cat(out)


if HAS_TORCH:
    t0 = time.time()
    PF_bank, GF_bank = patch_features(bank_paths)
    N, H, Wd, C = PF_bank.shape
    flat = PF_bank.reshape(-1, C)
    n_keep = int(min(CFG.MAX_BANK, max(1000, len(flat) * CFG.CORESET_RATIO)))
    BANK = flat[coreset(flat, n_keep, CFG.SEED)]
    print(f"Memory bank: {len(BANK)} patch dari {len(flat)} ({H}×{Wd} patch/gambar, dim {C}) | {time.time() - t0:.0f}s")

# %% [markdown]
# # 4. Skor anomali
#
# - **PatchCore**: skor patch = jarak ke patch normal terdekat; skor gambar = rata-rata `TOPK_SCORE` patch tertinggi.
# - **Global kNN** (baseline): jarak fitur rata-rata gambar ke gambar normal terdekat.
# - Tanpa torch: statistik pixel (histogram, gradien) + Isolation Forest.

# %%
def score_images(paths):
    pf, gf = patch_features(paths)
    n, h, w, c = pf.shape
    d = nn_dist(pf.reshape(-1, c), BANK).reshape(n, h * w)
    k = max(1, CFG.TOPK_SCORE)
    patch_score = d.topk(k, dim=1).values.mean(1).numpy()
    glob_score = nn_dist(gf, GF_bank).numpy()
    return patch_score, glob_score, d.reshape(n, h, w).numpy()


def pixel_stats(p):
    a = np.asarray(Image.open(p).convert("L").resize((128, 128)), np.float32) / 255
    gx, gy = np.abs(np.diff(a, axis=1)), np.abs(np.diff(a, axis=0))
    return np.concatenate([np.histogram(a, 32, (0, 1))[0] / a.size, [a.mean(), a.std(), gx.mean(), gy.mean(), gx.max(), gy.max()],
                           a[::16, ::16].ravel()])


SC = {}
if HAS_TORCH:
    t0 = time.time()
    SC["patchcore"], SC["global_knn"], MAPS = score_images(test_df["path"].tolist())
    VAL_SC = score_images(val_paths) if val_paths else None
    print(f"Skor test dihitung ({time.time() - t0:.0f}s)")
else:
    from sklearn.ensemble import IsolationForest
    Xb = np.array([pixel_stats(p) for p in bank_paths])
    iso = IsolationForest(n_estimators=300, random_state=CFG.SEED).fit(Xb)
    SC["pixel_iforest"] = -iso.score_samples(np.array([pixel_stats(p) for p in test_df["path"]]))
    VAL_SC = (-iso.score_samples(np.array([pixel_stats(p) for p in val_paths])),) if val_paths else None
    MAPS = None
for k_, v in SC.items():
    test_df[f"score_{k_}"] = v

# %% [markdown]
# # 5. Evaluasi & pemilihan metode
#
# Dengan label: AUROC & AP per metode → pilih terbaik. Tanpa label: PatchCore (lebih baik di literatur), threshold
# dari kuantil skor **gambar normal kalibrasi** (tidak pernah masuk memory bank).

# %%
FINAL = "patchcore" if "patchcore" in SC else list(SC)[0]
EVAL = None
y = test_df["label"].values
if HAS_LABEL:
    mask = test_df["label"].notna().values
    EVAL = pd.DataFrame([{"method": k_, "auroc": roc_auc_score(y[mask], v[mask]), "ap": average_precision_score(y[mask], v[mask])}
                         for k_, v in SC.items()]).sort_values("auroc", ascending=False)
    display(EVAL.round(4))
    save_table(EVAL, "method_comparison")
    FINAL = EVAL["method"].iloc[0]
score = SC[FINAL]
val_scores = None
if VAL_SC is not None:
    val_scores = VAL_SC[0] if FINAL in ("patchcore", "pixel_iforest") else VAL_SC[1]
TH = float(np.quantile(val_scores, CFG.THRESHOLD_QUANTILE)) if val_scores is not None and len(val_scores) else float(np.quantile(score, 0.9))
if HAS_LABEL:
    grid = np.unique(np.append(np.quantile(score[mask], np.linspace(0.05, 0.99, 150)), TH))
    f1s = [f1_score(y[mask], score[mask] >= t) for t in grid]
    print(f"F1 @threshold kalibrasi = {f1_score(y[mask], score[mask] >= TH):.4f} | F1 terbaik = {max(f1s):.4f}")
    print("ℹ️ Threshold F1-terbaik memakai label test → hanya untuk analisis; untuk submission dipakai threshold kalibrasi\n"
          "   kecuali label berasal dari data validasi resmi.")
test_df["is_anomaly"] = (score >= TH).astype(int)
print(f"Metode: {FINAL} | threshold {TH:.4f} | ditandai anomali: {test_df.is_anomaly.mean():.1%}")
fig, ax = plt.subplots(figsize=(6, 3.5))
if HAS_LABEL:
    ax.hist([score[y == 0], score[y == 1]], bins=30, label=["normal", "cacat"], density=True)
else:
    ax.hist(score, bins=30, label="test", density=True)
if val_scores is not None:
    ax.hist(val_scores, bins=15, alpha=.5, label="normal (kalibrasi)", density=True)
ax.axvline(TH, c="k", ls="--", label="threshold")
ax.legend(fontsize=8)
ax.set_title(f"Distribusi skor anomali ({FINAL})")
save_fig("score_distribution", fig)

# %% [markdown]
# # 6. Heatmap lokasi cacat
#
# Peta jarak patch di-upsample ke ukuran gambar → area merah = bagian yang paling berbeda dari gambar normal.
# Bagus untuk laporan (menunjukkan model "melihat" cacat, bukan latar).

# %%
if MAPS is not None:
    order = np.argsort(-score)[:min(6, len(score))]
    fig, axes = plt.subplots(2, len(order), figsize=(2.6 * len(order), 5.4))
    axes = np.atleast_2d(axes).reshape(2, len(order))
    for j, i in enumerate(order):
        im = np.asarray(Image.open(test_df.path[i]).convert("RGB").resize((CFG.CENTER_CROP, CFG.CENTER_CROP)))
        hm = torch.from_numpy(MAPS[i])[None, None]
        hm = F.interpolate(hm, size=im.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
        axes[0, j].imshow(im)
        axes[0, j].set_title(f"{test_df.id[i][:12]}\nskor {score[i]:.2f}" + (f" | y={int(y[i])}" if HAS_LABEL else ""), fontsize=7)
        axes[1, j].imshow(im)
        axes[1, j].imshow(hm, cmap="jet", alpha=.45)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle("Gambar test paling anomali + heatmap")
    save_fig("top_anomalies_heatmap", fig)

# %% [markdown]
# # 7. Submission + ringkasan

# %%
sample = smart_read(CFG.SAMPLE_SUB_PATH) if os.path.exists(CFG.SAMPLE_SUB_PATH) else None
mode = CFG.SUBMISSION_MODE
if mode == "auto":
    mode = "label" if (sample is not None and sample.shape[1] > 1 and set(pd.unique(sample.iloc[:, 1].dropna())) <= {0, 1}) else "score"
id_name = sample.columns[0] if sample is not None else "id"
val_name = sample.columns[1] if sample is not None and sample.shape[1] > 1 else ("is_anomaly" if mode == "label" else "anomaly_score")
vals = test_df["is_anomaly"] if mode == "label" else score
sub = pd.DataFrame({id_name: test_df["id"], val_name: vals})
if sample is not None:
    key = sample[id_name].astype(str).map(lambda v: os.path.splitext(os.path.basename(v))[0])
    sub = sample[[id_name]].assign(**{val_name: key.map(dict(zip(test_df["id"], vals))).values})
    if sub[val_name].isna().any():
        print(f"⚠️ {sub[val_name].isna().sum()} id sample_submission tanpa gambar test yang cocok")
SUB_PATH = os.path.join(SUB_DIR, "submission.csv")
sub.to_csv(SUB_PATH, index=False)
print(f"✅ Submission ({mode}): {SUB_PATH} | {len(sub)} baris")
save_table(test_df.drop(columns=["path"]), "test_scores")
summ = {"method": FINAL, "backbone": f"{CFG.BACKBONE} ({W_KIND if HAS_TORCH else '-'})", "n_normal_bank": len(bank_paths),
        "n_test": len(test_df), "threshold": TH, "flagged_pct": float(test_df.is_anomaly.mean()), "submission": SUB_PATH}
if EVAL is not None:
    summ.update({"auroc": round(float(EVAL.auroc.iloc[0]), 4), "ap": round(float(EVAL.ap.iloc[0]), 4)})
text = f"""# Image anomaly detection summary

- Method **{FINAL}** with {summ['backbone']} features; memory bank from {len(bank_paths)} normal images.
- Threshold {TH:.4f} = {CFG.THRESHOLD_QUANTILE:.0%} quantile of held-out normal images; {summ['flagged_pct']:.1%} of test images flagged.
{'- Test AUROC ' + str(summ.get('auroc')) + ', AP ' + str(summ.get('ap')) + '.' if EVAL is not None else ''}

**Draft sentences (paraphrase):**
- "Training data contained only defect-free items, so we modelled normality: patch features from an ImageNet-pretrained CNN form a memory bank, and test patches far from every normal patch are anomalous."
- "Anomaly heatmaps localise the defect, which makes the decision auditable."
"""
open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write(text)
json.dump(summ, open(os.path.join(OUT, "run_summary.json"), "w"), indent=2, default=str)
print(text)
print("✅ Notebook selesai.")

# %% [markdown]
# # 8. Troubleshooting, recipes, checklist
#
# | Gejala | Solusi |
# |---|---|
# | AUROC ≈ 0.5 | `PRETRAINED=False` (bobot acak) → aktifkan internet sekali untuk download, atau salin cache `~/.cache/torch` |
# | Banyak normal ditandai cacat | variasi posisi/pencahayaan besar → tambah gambar normal, `THRESHOLD_QUANTILE=0.995`, `TOPK_SCORE=5` |
# | Cacat kecil tidak terdeteksi | `IMG_SIZE`/`CENTER_CROP` lebih besar, `TOPK_SCORE=1` (max) |
# | Lambat / OOM | `BACKBONE="resnet18"`, `CORESET_RATIO=0.02`, `MAX_BANK=20000` |
# | Background ikut terdeteksi | crop ke objek (`CENTER_CROP`), atau segmentasi objek dulu |
#
# ```python
# CFG.BACKBONE, CFG.CORESET_RATIO = "wide_resnet50_2", 0.1   # standar PatchCore
# CFG.TOPK_SCORE = 3                                          # skor lebih stabil terhadap noise
# ```
#
# - [ ] Train benar-benar hanya gambar normal
# - [ ] Threshold dari gambar normal kalibrasi (bukan dari label test)
# - [ ] Arah skor & tipe submission (skor vs 0/1) sesuai aturan
# - [ ] Heatmap contoh masuk laporan
