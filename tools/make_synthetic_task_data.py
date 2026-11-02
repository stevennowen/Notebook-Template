"""
Buat data latihan kecil (sintetis) untuk setiap task — untuk mencoba template sebelum lomba dan untuk smoke test.
Setiap dataset punya pola yang DIKETAHUI (sinyal ditanam), jadi hasil model bisa dicek masuk akal atau tidak.

    python tools/make_synthetic_task_data.py --task tabular --out ../latihan_tabular
    python tools/make_synthetic_task_data.py --task all --out ../latihan            # satu subfolder per task

Data ditulis ke <out>/data/. Fungsi make(task, folder) mengembalikan dict override CFG yang cocok dengan data itu.
Hanya butuh numpy, pandas, Pillow, scipy (opsional untuk audio: ditulis dengan modul wave bawaan Python).
"""
import argparse
import json
import math
import os
import sys
import wave

import numpy as np
import pandas as pd

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _rng(seed):
    return np.random.default_rng(seed)


def _split_write(df, data, target_cols, id_col="id", test_frac=0.25, seed=0, sub_fill=None):
    rng = _rng(seed)
    idx = rng.permutation(len(df))
    n_te = int(len(df) * test_frac)
    te, tr = df.iloc[idx[:n_te]].reset_index(drop=True), df.iloc[idx[n_te:]].reset_index(drop=True)
    tr.to_csv(os.path.join(data, "train.csv"), index=False)
    te.drop(columns=target_cols).to_csv(os.path.join(data, "test.csv"), index=False)
    sub = te[[id_col]].copy()
    for c in target_cols:
        sub[c] = sub_fill if sub_fill is not None else 0
    sub.to_csv(os.path.join(data, "sample_submission.csv"), index=False)
    te[[id_col] + target_cols].to_csv(os.path.join(data, "_test_labels_hidden.csv"), index=False)
    return tr, te


# ---------------------------------------------------------------------------------------------- tabular
def make_tabular(folder, n=1200, seed=0, multilabel=False):
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    rng = _rng(seed)
    age = rng.integers(18, 70, n)
    income = np.round(rng.lognormal(15.5, 0.5, n), -3)
    tenure = rng.integers(0, 120, n)
    region = rng.choice(["Jawa", "Sumatra", "Kalimantan", "Sulawesi", "Bali"], n, p=[.45, .2, .12, .13, .1])
    channel = rng.choice(["agent", "online", "bank"], n)
    complaints = rng.poisson(0.8, n)
    x5 = rng.normal(0, 1, n)
    signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    logit = -0.5 + 0.9 * complaints - 0.02 * tenure + 0.6 * (channel == "online") + 0.5 * x5 + 0.3 * (region == "Bali")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"id": np.arange(1, n + 1), "age": age, "income": income, "tenure_months": tenure,
                       "region": region, "channel": channel, "complaints": complaints, "x5": np.round(x5, 4),
                       "signup_date": signup.strftime("%Y-%m-%d")})
    df.loc[rng.random(n) < 0.05, "income"] = np.nan
    if multilabel:
        df["promo"] = ((x5 + rng.normal(0, .5, n)) > 0.3).astype(int)
        df["upgrade"] = ((income > np.nanmedian(income)) & (rng.random(n) < .8)).astype(int)
        df["churn"] = y
        _split_write(df, data, ["promo", "upgrade", "churn"], seed=seed)
        return {"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv",
                "SAMPLE_SUB_PATH": "./data/sample_submission.csv", "ID_COL": "id", "TARGET_COL": ["promo", "upgrade", "churn"]}
    df["target"] = y
    _split_write(df, data, ["target"], seed=seed, sub_fill=0.5)
    return {"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "SAMPLE_SUB_PATH": "./data/sample_submission.csv",
            "ID_COL": "id", "TARGET_COL": "target"}


# ---------------------------------------------------------------------------------------------- teks
_POS = ["pelayanan cepat dan ramah", "barang sesuai deskripsi", "pengiriman kilat mantap", "kualitas bagus sekali",
        "harga murah puas banget", "admin responsif terima kasih", "packing rapi aman"]
_NEG = ["pengiriman lama sekali", "barang rusak kecewa", "admin tidak membalas", "kualitas jelek tidak sesuai",
        "refund dipersulit", "harga mahal tidak worth", "packing asal asalan"]
_NEU = ["barang sudah sampai", "pesanan diterima hari ini", "ukuran standar", "warna sesuai foto", "biasa saja"]


def _sentence(rng, parts, k=2):
    return " ".join(rng.choice(parts, k)) + rng.choice(["", ".", "!", " sih", " gan", " 👍"])


def make_text(folder, n=600, seed=0, regression=False):
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    rng = _rng(seed)
    labels = rng.choice(["positif", "negatif", "netral"], n, p=[.45, .35, .2])
    texts, scores = [], []
    for lab in labels:
        pool = {"positif": _POS, "negatif": _NEG, "netral": _NEU}[lab]
        t = _sentence(rng, pool, 2)
        if rng.random() < .3:
            t += " " + _sentence(rng, _NEU, 1)
        texts.append(t)
        scores.append({"positif": 4.5, "negatif": 1.5, "netral": 3.0}[lab] + rng.normal(0, .4))
    df = pd.DataFrame({"id": np.arange(n), "text": texts})
    if regression:
        df["target"] = np.round(np.clip(scores, 1, 5), 2)
        _split_write(df, data, ["target"], seed=seed, sub_fill=3.0)
        return {**{"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv",
                   "SAMPLE_SUB_PATH": "./data/sample_submission.csv"}, "ID_COL": "id", "TEXT_COL": "text", "TARGET_COL": "target"}
    df["label"] = labels
    _split_write(df, data, ["label"], seed=seed, sub_fill="positif")
    return {"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "SAMPLE_SUB_PATH": "./data/sample_submission.csv",
            "ID_COL": "id", "TEXT_COL": "text", "TARGET_COL": "label"}


# ---------------------------------------------------------------------------------------------- gambar
def _shape_image(rng, kind, size=64, bg=None):
    from PIL import Image, ImageDraw
    bg = bg if bg is not None else tuple(int(v) for v in rng.integers(0, 80, 3))
    img = Image.new("RGB", (size, size), bg)
    mask = Image.new("L", (size, size), 0)
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    col = tuple(int(v) for v in rng.integers(150, 256, 3))
    r = int(rng.integers(size // 6, size // 3))
    cx, cy = (int(v) for v in rng.integers(r + 2, size - r - 2, 2))
    box = [cx - r, cy - r, cx + r, cy + r]
    if kind == "circle":
        d.ellipse(box, fill=col); dm.ellipse(box, fill=1)
    elif kind == "square":
        d.rectangle(box, fill=col); dm.rectangle(box, fill=1)
    else:
        pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        d.polygon(pts, fill=col); dm.polygon(pts, fill=1)
    arr = np.asarray(img).astype(np.int16) + rng.integers(-12, 13, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), mask, box


def make_cv_classification(folder, n=240, seed=0):
    data = os.path.join(folder, "data")
    rng = _rng(seed)
    for sub in ("train", "test"):
        os.makedirs(os.path.join(data, sub), exist_ok=True)
    kinds = rng.choice(["circle", "square", "triangle"], n)
    df = pd.DataFrame({"image_id": [f"img_{i:04d}" for i in range(n)], "label": kinds})
    n_te = n // 4
    for i, r in df.iterrows():
        sub = "test" if i < n_te else "train"
        img, _, _ = _shape_image(_rng(seed * 1000 + i), r["label"])
        img.save(os.path.join(data, sub, r["image_id"] + ".png"))
    df.iloc[n_te:].to_csv(os.path.join(data, "train.csv"), index=False)
    df.iloc[:n_te][["image_id"]].to_csv(os.path.join(data, "test.csv"), index=False)
    df.iloc[:n_te][["image_id"]].assign(label="circle").to_csv(os.path.join(data, "sample_submission.csv"), index=False)
    df.iloc[:n_te].to_csv(os.path.join(data, "_test_labels_hidden.csv"), index=False)
    return {"TRAIN_CSV": "./data/train.csv", "TEST_CSV": "./data/test.csv", "SAMPLE_SUB_PATH": "./data/sample_submission.csv",
            "TRAIN_IMAGE_DIR": "./data/train", "TEST_IMAGE_DIR": "./data/test", "IMAGE_COL": "image_id", "TARGET_COL": "label"}


def make_segmentation(folder, n=60, seed=0, size=96):
    """Gambar dengan bentuk; mask: 0 = background, 1 = lingkaran, 2 = kotak (multikelas)."""
    from PIL import Image
    data = os.path.join(folder, "data")
    for p in ("train/images", "train/masks", "test/images"):
        os.makedirs(os.path.join(data, p), exist_ok=True)
    rng = _rng(seed)
    test_ids = []
    for i in range(n):
        sub = "test" if i < n // 5 else "train"
        canvas = np.zeros((size, size, 3), np.int16) + rng.integers(0, 70, 3)
        mask = np.zeros((size, size), np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            kind = rng.choice(["circle", "square"])
            img, m, _ = _shape_image(rng, kind, size, bg=(0, 0, 0))
            m = np.asarray(m) > 0
            canvas[m] = np.asarray(img)[m]
            mask[m] = 1 if kind == "circle" else 2
        canvas = np.clip(canvas + rng.integers(-10, 11, canvas.shape), 0, 255).astype(np.uint8)
        name = f"tile_{i:03d}.png"
        Image.fromarray(canvas).save(os.path.join(data, sub, "images", name))
        if sub == "train":
            Image.fromarray(mask).save(os.path.join(data, "train", "masks", name))
        else:
            test_ids.append(name[:-4])
    return {"TRAIN_IMAGE_DIR": "./data/train/images", "TRAIN_MASK_DIR": "./data/train/masks",
            "TEST_IMAGE_DIR": "./data/test/images"}


def make_detection(folder, n=60, seed=0, size=160):
    """Gambar berisi 1–6 objek (lingkaran = 'coin', kotak = 'box'); label format YOLO txt."""
    from PIL import Image
    data = os.path.join(folder, "data")
    for p in ("train/images", "train/labels", "test/images"):
        os.makedirs(os.path.join(data, p), exist_ok=True)
    rng = _rng(seed)
    counts = []
    for i in range(n):
        sub = "test" if i < n // 5 else "train"
        canvas = np.zeros((size, size, 3), np.int16) + rng.integers(20, 90, 3)
        lines, k = [], int(rng.integers(1, 7))
        for _ in range(k):
            kind = rng.choice(["circle", "square"])
            r = int(rng.integers(8, 18))
            cx, cy = (int(v) for v in rng.integers(r + 1, size - r - 1, 2))
            yy, xx = np.ogrid[:size, :size]
            m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) if kind == "circle" else ((abs(xx - cx) <= r) & (abs(yy - cy) <= r))
            canvas[m] = rng.integers(160, 256, 3)
            cls = 0 if kind == "circle" else 1
            lines.append(f"{cls} {cx / size:.6f} {cy / size:.6f} {2 * r / size:.6f} {2 * r / size:.6f}")
        canvas = np.clip(canvas + rng.integers(-8, 9, canvas.shape), 0, 255).astype(np.uint8)
        name = f"scene_{i:03d}"
        Image.fromarray(canvas).save(os.path.join(data, sub, "images", name + ".jpg"), quality=95)
        if sub == "train":
            with open(os.path.join(data, "train", "labels", name + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        else:
            counts.append({"image_id": name, "count": k})
    with open(os.path.join(data, "classes.txt"), "w") as f:
        f.write("coin\nbox\n")
    pd.DataFrame(counts).to_csv(os.path.join(data, "_test_counts_hidden.csv"), index=False)
    return {"TRAIN_IMAGE_DIR": "./data/train/images", "TRAIN_LABEL_DIR": "./data/train/labels",
            "TEST_IMAGE_DIR": "./data/test/images", "CLASS_NAMES": ["coin", "box"]}


def make_ocr(folder, n=400, seed=0):
    """Gambar berisi 3–6 digit (seperti nomor meter/plat); train.csv: image,text."""
    from PIL import Image, ImageDraw, ImageFont
    data = os.path.join(folder, "data")
    for p in ("train", "test"):
        os.makedirs(os.path.join(data, p), exist_ok=True)
    rng = _rng(seed)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    rows = []
    for i in range(n):
        text = "".join(str(int(d)) for d in rng.integers(0, 10, int(rng.integers(3, 7))))
        img = Image.new("L", (128, 32), int(rng.integers(200, 256)))
        d = ImageDraw.Draw(img)
        d.text((int(rng.integers(2, 20)), int(rng.integers(0, 6))), text, fill=int(rng.integers(0, 60)), font=font)
        arr = np.clip(np.asarray(img).astype(np.int16) + rng.integers(-20, 21, (32, 128)), 0, 255).astype(np.uint8)
        sub = "test" if i < n // 5 else "train"
        name = f"meter_{i:04d}.png"
        Image.fromarray(arr).rotate(float(rng.normal(0, 2)), fillcolor=230).save(os.path.join(data, sub, name))
        rows.append({"image": name, "text": text, "sub": sub})
    df = pd.DataFrame(rows)
    df[df["sub"] == "train"][["image", "text"]].to_csv(os.path.join(data, "train.csv"), index=False)
    te = df[df["sub"] == "test"]
    te[["image"]].to_csv(os.path.join(data, "test.csv"), index=False)
    te[["image"]].assign(text="0").to_csv(os.path.join(data, "sample_submission.csv"), index=False)
    te[["image", "text"]].to_csv(os.path.join(data, "_test_labels_hidden.csv"), index=False)
    return {"TRAIN_CSV": "./data/train.csv", "TRAIN_IMAGE_DIR": "./data/train", "TEST_CSV": "./data/test.csv",
            "TEST_IMAGE_DIR": "./data/test", "SAMPLE_SUB_PATH": "./data/sample_submission.csv"}


def make_anomaly_image(folder, n_good=60, n_test=30, seed=0, size=96):
    """'Produk' = tekstur garis; cacat = goresan/bercak. train/good hanya berisi gambar normal."""
    from PIL import Image, ImageDraw
    data = os.path.join(folder, "data")
    for p in ("train/good", "test"):
        os.makedirs(os.path.join(data, p), exist_ok=True)
    rng = _rng(seed)

    def product(defect):
        x = np.linspace(0, 8 * math.pi, size)
        base = (np.sin(x)[None, :] * 25 + 128 + rng.normal(0, 6, (size, size))).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(np.stack([base] * 3, -1))
        if defect:
            d = ImageDraw.Draw(img)
            if rng.random() < .5:
                x0, y0 = (int(v) for v in rng.integers(10, size - 30, 2))
                d.line([x0, y0, x0 + int(rng.integers(15, 30)), y0 + int(rng.integers(-15, 15))], fill=(20, 20, 20), width=3)
            else:
                cx, cy = (int(v) for v in rng.integers(15, size - 15, 2))
                d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(200, 60, 40))
        return img

    for i in range(n_good):
        product(False).save(os.path.join(data, "train", "good", f"good_{i:03d}.png"))
    rows = []
    for i in range(n_test):
        bad = i % 3 == 0
        name = f"test_{i:03d}.png"
        product(bad).save(os.path.join(data, "test", name))
        rows.append({"image_id": name[:-4], "label": int(bad)})
    pd.DataFrame(rows).to_csv(os.path.join(data, "_test_labels_hidden.csv"), index=False)
    return {"TRAIN_IMAGE_DIR": "./data/train/good", "TEST_IMAGE_DIR": "./data/test",
            "TEST_LABELS_PATH": "./data/_test_labels_hidden.csv"}


# ---------------------------------------------------------------------------------------------- lain
def make_time_series(folder, n_series=4, n_days=420, horizon=28, seed=0):
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    rng = _rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days + horizon, freq="D")
    rows = []
    for s in range(n_series):
        base = 50 + 20 * s
        t = np.arange(len(dates))
        y = base + 0.05 * t + 10 * np.sin(2 * np.pi * t / 7) + 5 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 3, len(t))
        promo = (rng.random(len(t)) < .1).astype(int)
        y = np.maximum(0, y + 15 * promo)
        rows.append(pd.DataFrame({"date": dates, "store": f"S{s + 1}", "promo": promo, "sales": np.round(y, 1)}))
    df = pd.concat(rows, ignore_index=True)
    cut = dates[n_days]
    tr, te = df[df["date"] < cut].copy(), df[df["date"] >= cut].copy()
    te.insert(0, "id", np.arange(len(te)))
    tr["date"] = tr["date"].dt.strftime("%Y-%m-%d")
    te["date"] = te["date"].dt.strftime("%Y-%m-%d")
    tr.to_csv(os.path.join(data, "train.csv"), index=False)
    te.drop(columns=["sales"]).to_csv(os.path.join(data, "test.csv"), index=False)
    te[["id"]].assign(sales=0).to_csv(os.path.join(data, "sample_submission.csv"), index=False)
    te[["id", "sales"]].to_csv(os.path.join(data, "_test_labels_hidden.csv"), index=False)
    return {"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "SAMPLE_SUB_PATH": "./data/sample_submission.csv",
            "DATE_COL": "date", "TARGET_COL": "sales", "SERIES_ID_COLS": ["store"], "ID_COL": "id", "EXOG_COLS": ["promo"]}


def make_clustering(folder, n=600, seed=0):
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    rng = _rng(seed)
    centers = np.array([[25, 3e6, 2, 0.2], [45, 12e6, 10, 0.6], [60, 6e6, 25, 0.9]])
    seg = rng.integers(0, 3, n)
    X = centers[seg] * (1 + rng.normal(0, .15, (n, 4)))
    df = pd.DataFrame({"id": np.arange(n), "age": X[:, 0].round(), "monthly_spend": X[:, 1].round(-3),
                       "transactions": np.maximum(0, X[:, 2]).round(), "app_usage": X[:, 3].clip(0, 1).round(3),
                       "city": rng.choice(["Jakarta", "Surabaya", "Bandung", "Medan"], n),
                       "churned": (rng.random(n) < np.array([.3, .1, .05])[seg]).astype(int)})
    df.to_csv(os.path.join(data, "train.csv"), index=False)
    return {"DATA_PATH": "./data/train.csv", "ID_COL": "id", "OUTCOME_COL": "churned"}


def make_anomaly_tabular(folder, n=1500, seed=0):
    """Transaksi; ~3% fraud ditanam (nominal sangat besar di jam aneh, merchant baru, dsb.)."""
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    rng = _rng(seed)
    amount = rng.lognormal(12, .6, n)
    hour = rng.integers(7, 23, n)
    merchant_age = rng.integers(30, 2000, n)
    n_tx_day = rng.poisson(3, n)
    category = rng.choice(["grocery", "fuel", "online", "travel"], n, p=[.4, .2, .3, .1])
    fraud = rng.random(n) < .03
    amount[fraud] *= rng.uniform(5, 20, fraud.sum())
    hour[fraud] = rng.integers(0, 5, fraud.sum())
    merchant_age[fraud] = rng.integers(0, 10, fraud.sum())
    n_tx_day[fraud] += rng.integers(8, 20, fraud.sum())
    df = pd.DataFrame({"id": np.arange(n), "amount": amount.round(-2), "hour": hour, "merchant_age_days": merchant_age,
                       "n_tx_day": n_tx_day, "category": category, "is_fraud": fraud.astype(int)})
    idx = rng.permutation(n)
    tr, te = df.iloc[idx[: int(n * .7)]], df.iloc[idx[int(n * .7):]]
    tr.to_csv(os.path.join(data, "train.csv"), index=False)
    te.drop(columns=["is_fraud"]).to_csv(os.path.join(data, "test.csv"), index=False)
    te[["id", "is_fraud"]].to_csv(os.path.join(data, "_test_labels_hidden.csv"), index=False)
    return {"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "ID_COL": "id", "LABEL_COL": "is_fraud"}


def _write_wav(path, y, sr):
    y = np.clip(y, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767).astype("<i2").tobytes())


def make_audio(folder, n=150, seed=0, sr=16000):
    """3 'spesies burung' = pola nada berbeda (chirp naik, nada putus-putus, trill); test berisi 1 kelas tak dikenal."""
    data = os.path.join(folder, "data")
    for p in ("train", "test"):
        os.makedirs(os.path.join(data, p), exist_ok=True)
    rng = _rng(seed)

    def clip(kind):
        dur = rng.uniform(1.0, 2.5)
        t = np.arange(int(sr * dur)) / sr
        f0 = rng.uniform(800, 1500)
        if kind == "kutilang":
            y = np.sin(2 * np.pi * (f0 * t + 600 * t ** 2))
        elif kind == "cendrawasih":
            y = np.sin(2 * np.pi * f0 * 1.8 * t) * (np.sin(2 * np.pi * 4 * t) > 0)
        elif kind == "bondol":
            y = np.sin(2 * np.pi * (f0 + 300 * np.sin(2 * np.pi * 18 * t)) * t)
        else:   # kelas asing (open-set): noise pita lebar
            y = rng.normal(0, .5, len(t)) * np.sin(2 * np.pi * 2 * t) ** 2
        return 0.4 * y + rng.normal(0, .05, len(t))

    rows = []
    for i in range(n):
        lab = rng.choice(["kutilang", "cendrawasih", "bondol"])
        name = f"rec_{i:04d}.wav"
        _write_wav(os.path.join(data, "train", name), clip(lab), sr)
        rows.append({"filename": name, "label": lab})
    pd.DataFrame(rows).to_csv(os.path.join(data, "train.csv"), index=False)
    test = []
    for i in range(n // 3):
        lab = rng.choice(["kutilang", "cendrawasih", "bondol", "Unknown"], p=[.3, .3, .3, .1])
        name = f"test_{i:04d}.wav"
        _write_wav(os.path.join(data, "test", name), clip(lab), sr)
        test.append({"filename": name, "label": lab})
    te = pd.DataFrame(test)
    te[["filename"]].assign(label="kutilang").to_csv(os.path.join(data, "sample_submission.csv"), index=False)
    te.to_csv(os.path.join(data, "_test_labels_hidden.csv"), index=False)
    return {"TRAIN_CSV": "./data/train.csv", "TRAIN_AUDIO_DIR": "./data/train", "TEST_AUDIO_DIR": "./data/test",
            "SAMPLE_SUB_PATH": "./data/sample_submission.csv", "OPEN_SET": True}


_PER = ["Budi", "Siti", "Andi", "Rina", "Joko", "Dewi"]
_LOC = ["Jakarta", "Bandung", "Surabaya", "Malang", "Medan"]
_ORG = ["Bank Mandiri", "Universitas Brawijaya", "Pertamina", "Telkom Indonesia"]


def make_ner(folder, n=300, seed=0):
    """Kalimat berlabel BIO (PER, LOC, ORG) dalam format CoNLL (token<TAB>tag, kalimat dipisah baris kosong)."""
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    rng = _rng(seed)
    templates = [["{PER}", "bekerja", "di", "{ORG}", "cabang", "{LOC}"],
                 ["kemarin", "{PER}", "pergi", "ke", "{LOC}", "bersama", "{PER}"],
                 ["{ORG}", "membuka", "kantor", "baru", "di", "{LOC}"],
                 ["saya", "bertemu", "{PER}", "di", "kampus", "{ORG}"]]

    def sent():
        toks, tags = [], []
        for w in templates[int(rng.integers(len(templates)))]:
            if w.startswith("{"):
                typ = w[1:-1]
                ent = str(rng.choice({"PER": _PER, "LOC": _LOC, "ORG": _ORG}[typ])).split()
                toks += ent
                tags += [f"B-{typ}"] + [f"I-{typ}"] * (len(ent) - 1)
            else:
                toks.append(w)
                tags.append("O")
        return toks, tags

    def write(path, k, with_tags=True):
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(k):
                toks, tags = sent()
                for tk, tg in zip(toks, tags):
                    f.write(f"{tk}\t{tg}\n" if with_tags else f"{tk}\n")
                f.write("\n")

    write(os.path.join(data, "train.conll"), n)
    rng = _rng(seed + 1)
    write(os.path.join(data, "test.conll"), n // 4, with_tags=False)
    return {"TRAIN_PATH": "./data/train.conll", "TEST_PATH": "./data/test.conll"}


def make_llm(folder, n=80, seed=0):
    cfg = make_text(folder, n=n, seed=seed)
    return {"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "ID_COL": "id", "TEXT_COL": "text",
            "TARGET_COL": "label", "SAMPLE_SUB_PATH": cfg["SAMPLE_SUB_PATH"], "LABELS": ["positif", "negatif", "netral"]}


def make_stats(folder, n=800, seed=0):
    """Data survei/eksperimen: grup A/B, wilayah, skor kepuasan, konversi."""
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    rng = _rng(seed)
    group = rng.choice(["A", "B"], n)
    region = rng.choice(["Jawa", "Sumatra", "Kalimantan", "Sulawesi"], n)
    age = rng.integers(18, 65, n)
    income = rng.lognormal(15.4, .5, n).round(-3)
    satisfaction = np.clip(3.2 + 0.4 * (group == "B") + 0.01 * (age - 40) + rng.normal(0, .9, n), 1, 5).round(1)
    converted = (rng.random(n) < 0.10 + 0.04 * (group == "B") + 0.02 * (region == "Jawa")).astype(int)
    df = pd.DataFrame({"respondent_id": np.arange(n), "group": group, "region": region, "age": age, "income": income,
                       "satisfaction": satisfaction, "converted": converted})
    df.to_csv(os.path.join(data, "data.csv"), index=False)
    return {"DATA_PATH": "./data/data.csv", "OUTCOME_COL": "converted", "GROUP_COL": "group", "ID_COL": "respondent_id"}


MAKERS = {
    "tabular": make_tabular,
    "tabular-multilabel": lambda f, **k: make_tabular(f, multilabel=True, **k),
    "nlp-classification": make_text,
    "nlp-regression": lambda f, **k: make_text(f, regression=True, **k),
    "cv-classification": make_cv_classification,
    "time-series": make_time_series,
    "clustering": make_clustering,
    "segmentation": make_segmentation,
    "detection": make_detection,
    "ocr": make_ocr,
    "anomaly-tabular": make_anomaly_tabular,
    "anomaly-image": make_anomaly_image,
    "audio": make_audio,
    "ner": make_ner,
    "llm": make_llm,
    "stats": make_stats,
}


def make(task, folder, seed=0):
    os.makedirs(folder, exist_ok=True)
    return MAKERS[task](folder, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="nama task atau 'all': " + ", ".join(MAKERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tasks = list(MAKERS) if a.task == "all" else [a.task]
    for t in tasks:
        folder = os.path.join(a.out, t) if a.task == "all" else a.out
        cfg = make(t, folder, a.seed)
        with open(os.path.join(folder, "synthetic_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"✔ {t:20s} → {os.path.abspath(folder)}  (config: synthetic_config.json)")


if __name__ == "__main__":
    main()
