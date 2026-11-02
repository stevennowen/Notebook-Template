# -*- coding: utf-8 -*-
# =====================================================================================================
# 00_environment_check.py — versi SCRIPT dari 00_environment_check.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python 00_environment_check.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb 00_environment_check.py
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
# # 00 — Environment Check (jalankan PERTAMA di laptop lomba)
#
# Tujuan: memastikan semua library tersedia **sebelum** mulai mengerjakan soal, sehingga tidak ada kejutan di tengah lomba.
#
# Langkah:
# 1. Jalankan semua cell (**Kernel → Restart & Run All**).
# 2. Lihat tabel status. Package `MISSING` → cell install akan mencoba memasang otomatis (butuh internet).
# 3. Kalau ada yang ter-install, **restart kernel** lalu jalankan notebook ini sekali lagi.
# 4. Cek bagian GPU: kalau tidak ada GPU, template NLP/CV otomatis jalan di CPU (lebih lambat → pakai model kecil / baseline klasik).
#
# **Catatan penting:**
# - **Jangan pakai Google Colab** (dilarang di soal). Jalankan di Jupyter / VS Code lokal.
# - Torch **tidak** di-install otomatis (ukuran besar & harus cocok dengan CUDA). Install manual dari https://pytorch.org/get-started/locally/ hanya kalau butuh NLP transformer / CV.
# - Kalau `pip` gagal karena permission: `pip install --user <package>`.
# - Kalau ada beberapa Python di laptop, pastikan kernel Jupyter = Python yang sama dengan `pip` (cek `sys.executable` di bawah).

# %%
import sys, platform, importlib, importlib.util, subprocess
print("Python     :", sys.version)
print("Executable :", sys.executable, "  ← pip install harus ke Python ini")
print("Platform   :", platform.platform())

# %%
AUTO_INSTALL = __import__("os").environ.get("TEMPLATE_AUTO_INSTALL", "1") != "0"   # False = hanya cek, tidak install (env TEMPLATE_AUTO_INSTALL=0 → off)

PACKAGES = [
    # (import name, pip name, kelompok)
    ("numpy", "numpy", "core"), ("pandas", "pandas", "core"), ("sklearn", "scikit-learn", "core"),
    ("scipy", "scipy", "core"), ("matplotlib", "matplotlib", "core"), ("seaborn", "seaborn", "core"),
    ("openpyxl", "openpyxl", "core"),
    ("lightgbm", "lightgbm", "tabular"), ("xgboost", "xgboost", "tabular"), ("catboost", "catboost", "tabular"),
    ("statsmodels", "statsmodels", "tabular"), ("optuna", "optuna", "tabular"), ("shap", "shap", "tabular"),
    ("lifelines", "lifelines", "tabular"), ("imblearn", "imbalanced-learn", "tabular"),
    ("hdbscan", "hdbscan", "clustering"), ("umap", "umap-learn", "clustering"),
    ("torch", None, "deep-learning"), ("torchvision", None, "deep-learning"), ("timm", "timm", "cv"),
    ("PIL", "Pillow", "cv"), ("transformers", "transformers", "nlp"), ("sentence_transformers", "sentence-transformers", "nlp"),
    ("sentencepiece", "sentencepiece", "nlp"), ("tqdm", "tqdm", "core"),
    # Opsional per template (pip name None = TIDAK di-install di sini; template yang butuh memasangnya sendiri,
    # dan semuanya punya fallback kalau tidak ada):
    ("segmentation_models_pytorch", None, "cv-segmentation"), ("ultralytics", None, "cv-detection"),
    ("cv2", None, "cv-segmentation"), ("soundfile", None, "audio"), ("librosa", None, "audio"),
    ("accelerate", None, "nlp-llm"), ("docx", None, "stats-report"),
]


def status():
    rows = []
    for imp, pipn, grp in PACKAGES:
        ok = importlib.util.find_spec(imp) is not None
        ver = ""
        if ok:
            try:
                m = importlib.import_module(imp)
                ver = getattr(m, "__version__", "")
                if imp == "umap" and not hasattr(m, "UMAP"):
                    ok, ver = False, "salah package (butuh umap-learn)"
            except Exception as e:
                ok, ver = False, f"import error: {e}"[:60]
        rows.append((grp, imp, pipn, "OK" if ok else "MISSING", ver))
    return rows


rows = status()
print(f"{'group':<17}{'import':<30}{'status':<9}version")
for g, i, p, s, v in rows:
    print(f"{g:<17}{i:<30}{s:<9}{v}")

# %%
missing = [(i, p) for g, i, p, s, v in rows if s == "MISSING" and p]
optional_missing = [f"{i} ({g})" for g, i, p, s, v in rows if s == "MISSING" and not p and i not in ("torch", "torchvision")]
if optional_missing:
    print("ℹ️ Opsional, belum ada (template terkait memakai fallback / memasangnya sendiri):", optional_missing)
if not missing:
    print("Semua package (selain yang butuh install manual) sudah tersedia ✅")
elif AUTO_INSTALL:
    for imp, pipn in missing:
        try:
            print("Installing", pipn, "...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *([] if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else ["--user"]), pipn])
        except Exception as e:
            print("  gagal:", e)
    print("\nSelesai. RESTART kernel lalu run ulang notebook ini untuk verifikasi.")
else:
    print("Package belum ada:", [p for _, p in missing])

# %% [markdown]
# ## GPU check

# %%
try:
    import torch
    print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "| VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
except Exception as e:
    print("torch tidak tersedia:", e, "→ template NLP transformer & CV deep learning tidak bisa dipakai; baseline klasik tetap bisa.")
try:
    import os
    print("CPU cores:", os.cpu_count())
    import psutil
    print("RAM:", round(psutil.virtual_memory().total / 1e9, 1), "GB")
except Exception:
    pass

# %% [markdown]
# ## Smoke test cepat (pastikan library inti benar-benar jalan)

# %%
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
X, y = make_classification(1000, 10, random_state=0)
print("LogReg CV AUC:", cross_val_score(LogisticRegression(max_iter=1000), X, y, scoring="roc_auc", cv=3).mean().round(3))
for name in ["lightgbm", "xgboost", "catboost"]:
    try:
        if name == "lightgbm":
            import lightgbm as lgb; m = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
        elif name == "xgboost":
            import xgboost as xgb; m = xgb.XGBClassifier(n_estimators=50)
        else:
            import catboost as cb; m = cb.CatBoostClassifier(iterations=50, verbose=0, allow_writing_files=False)
        print(f"{name:<9} CV AUC:", cross_val_score(m, X, y, scoring="roc_auc", cv=3).mean().round(3), "✅")
    except Exception as e:
        print(f"{name:<9} ❌ {e}")
import matplotlib
matplotlib.use(matplotlib.get_backend())
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1]); plt.title("Plot test"); plt.show()
print("Environment siap. Lanjut buka template sesuai jenis soal (lihat README).")
