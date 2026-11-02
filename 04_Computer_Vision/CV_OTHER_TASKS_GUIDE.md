# Panduan Task Computer Vision Lain (Referensi)

Dokumen ini adalah **referensi** untuk task CV selain klasifikasi gambar biasa. Template utama
(`cv_image_classification.ipynb`) sudah menangani klasifikasi binary / multiclass. Di sini ada potongan kode siap
pakai + penjelasan untuk:

1. [Object detection (YOLO / ultralytics)](#1-object-detection-ultralytics-yolo)
2. [Semantic segmentation (segmentation_models_pytorch)](#2-semantic-segmentation-segmentation_models_pytorch)
3. [Image regression](#3-image-regression)
4. [Multilabel image classification](#4-multilabel-image-classification)
5. [OCR (baca teks dari gambar)](#5-ocr-baca-teks-dari-gambar)
6. [Tips umum & troubleshooting lintas task](#6-tips-umum--troubleshooting-lintas-task)

> Kode di sini **tidak dieksekusi otomatis**. Salin ke notebook baru, sesuaikan path & nama kolom, lalu jalankan
> bertahap. Prinsip yang sama dengan template utama tetap berlaku: validasi yang jujur (K-fold / holdout tanpa
> leakage), seed tetap, simpan figure untuk laporan, dan cek format submission terhadap `sample_submission`.

> **Sekarang sudah ada template lengkap (teruji, .ipynb + .py) untuk tiga task paling umum** — pakai ini dulu,
> potongan kode di bawah hanya untuk variasi yang tidak dicakup:
>
> | Task | Template |
> |---|---|
> | Semantic segmentation (mask PNG / RLE) | `cv_segmentation.ipynb` |
> | Object detection & counting (YOLO / Faster R-CNN) | `cv_object_detection.ipynb` |
> | OCR / text recognition (digit, plat, aksara) | `cv_ocr_text_recognition.ipynb` |
> | Anomaly / cacat produk (hanya gambar normal) | `../07_Anomaly_Detection/anomaly_image.ipynb` |
>
> Folder kerja siap pakai: `python tools/new_task.py ../kerja --task segmentation|detection|ocr|anomaly-image --format ipynb|py`.

---

## 0. Cara mengenali task dari soal lomba

| Yang diminta di submission | Task | Bagian |
|---|---|---|
| 1 label per gambar | klasifikasi | template utama |
| Beberapa label per gambar (`"cat dog"`, atau kolom 0/1 per label) | multilabel | §4 |
| Angka kontinu per gambar (umur, berat, skor, jumlah) | image regression | §3 |
| Kotak `x, y, w, h` / `xmin, ymin, xmax, ymax` + kelas + confidence | object detection | §1 |
| Mask per pixel / string RLE (`"1 3 10 5 ..."`) | segmentation | §2 |
| Teks yang tertulis di gambar | OCR | §5 |
| Jumlah objek per gambar | regression (§3) **atau** detection lalu hitung kotak (§1) | §1 / §3 |

**Install yang aman (jangan merusak torch yang sudah ada):** library yang bergantung pada torch sebaiknya di-install
dengan `--no-deps` lalu dependensi ringannya di-install terpisah. Contoh umum:

```python
import subprocess, sys
def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# cek dulu versi torch yang sudah ada — JANGAN sampai ter-upgrade
import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
```

---

## 1. Object detection (ultralytics YOLO)

### 1.1 Install

```python
# ultralytics butuh torch & torchvision. Kalau torch sudah ada, install tanpa mengganti torch:
pip("ultralytics", "--no-deps")
pip("opencv-python-headless", "pyyaml", "matplotlib", "pandas", "scipy", "tqdm", "psutil", "py-cpuinfo",
    "requests", "pillow", "ultralytics-thop")
from ultralytics import YOLO
```

Kalau `import ultralytics` gagal karena library kurang, baca nama modul di error lalu `pip("nama_modul")`.
Kalau torchvision tidak ada: install versi yang cocok (`torch 2.X` ↔ `torchvision 0.(X+15)`), contoh torch 2.5:
`pip install torchvision==0.20.* --no-deps --index-url https://download.pytorch.org/whl/cu121`.

### 1.2 Format label — WAJIB paham

| Format | Koordinat | Isi |
|---|---|---|
| **YOLO** (`.txt` per gambar) | **ternormalisasi 0–1**, pusat kotak | 1 baris per objek: `class_id cx cy w h` |
| **COCO** (1 file `.json`) | **pixel absolut**, pojok kiri atas | `bbox: [x, y, w, h]` + `category_id` + `image_id` |
| **Pascal VOC** (`.xml` per gambar) | pixel absolut, 2 pojok | `xmin ymin xmax ymax` |
| CSV ala Kaggle | biasanya pixel absolut | `image_id, x_min, y_min, x_max, y_max, class` atau `x, y, w, h` |

Struktur folder yang diharapkan YOLO:

```text
yolo_data/
├── images/train/*.jpg     images/val/*.jpg
├── labels/train/*.txt     labels/val/*.txt      (nama file sama dengan gambar, ekstensi .txt)
└── data.yaml
```

Gambar tanpa objek (negatif) → file `.txt` kosong (boleh juga tidak ada file label).
`class_id` mulai dari **0**.

### 1.3 Konversi ke format YOLO

**Dari CSV (xmin, ymin, xmax, ymax dalam pixel):**

```python
import os, shutil
import pandas as pd
from PIL import Image

def csv_to_yolo(df, img_dir, out_dir, split_ids, split_name, class_map,
                id_col="image_id", cols=("x_min", "y_min", "x_max", "y_max"), cls_col="class", ext=".jpg"):
    """df: 1 baris per kotak. split_ids: daftar image_id untuk split ini."""
    os.makedirs(f"{out_dir}/images/{split_name}", exist_ok=True)
    os.makedirs(f"{out_dir}/labels/{split_name}", exist_ok=True)
    groups = df.groupby(id_col)
    for img_id in split_ids:
        src = os.path.join(img_dir, f"{img_id}{ext}")
        W, H = Image.open(src).size
        shutil.copy(src, f"{out_dir}/images/{split_name}/{img_id}{ext}")
        lines = []
        if img_id in groups.groups:
            for _, r in groups.get_group(img_id).iterrows():
                x1, y1, x2, y2 = [float(r[c]) for c in cols]
                x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))      # clip ke dalam gambar
                y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
                if x2 <= x1 or y2 <= y1:
                    continue                                       # kotak rusak → buang
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                w, h = (x2 - x1) / W, (y2 - y1) / H
                lines.append(f"{class_map[r[cls_col]]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(f"{out_dir}/labels/{split_name}/{img_id}.txt", "w") as f:
            f.write("\n".join(lines))
```

Kalau CSV berisi `x, y, w, h` (pojok kiri atas + lebar/tinggi): `x1, y1, x2, y2 = x, y, x + w, y + h`.

**Dari COCO JSON:**

```python
import json
from collections import defaultdict

def coco_to_yolo(coco_json, out_label_dir):
    d = json.load(open(coco_json))
    cats = sorted(c["id"] for c in d["categories"])
    cat2idx = {c: i for i, c in enumerate(cats)}                   # id COCO bisa mulai dari 1 / loncat
    imgs = {im["id"]: im for im in d["images"]}
    anns = defaultdict(list)
    for a in d["annotations"]:
        if a.get("iscrowd", 0):
            continue
        anns[a["image_id"]].append(a)
    os.makedirs(out_label_dir, exist_ok=True)
    for img_id, im in imgs.items():
        W, H = im["width"], im["height"]
        lines = []
        for a in anns[img_id]:
            x, y, w, h = a["bbox"]
            lines.append(f"{cat2idx[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        name = os.path.splitext(os.path.basename(im["file_name"]))[0]
        open(os.path.join(out_label_dir, name + ".txt"), "w").write("\n".join(lines))
    return [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
```

**Dari Pascal VOC XML:**

```python
import xml.etree.ElementTree as ET

def voc_to_yolo(xml_path, class_map):
    root = ET.parse(xml_path).getroot()
    W = float(root.find("size/width").text); H = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        x1, y1, x2, y2 = [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        lines.append(f"{class_map[obj.find('name').text]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines
```

**Split train/val per gambar (bukan per kotak!)**, stratified kasar berdasarkan jumlah objek:

```python
from sklearn.model_selection import train_test_split
img_ids = df["image_id"].unique()
n_obj = df.groupby("image_id").size().reindex(img_ids).fillna(0).clip(upper=5)
tr_ids, va_ids = train_test_split(img_ids, test_size=0.2, random_state=42, stratify=n_obj)
```

Kalau ada grup (pasien, video, lokasi) → split per grup (`GroupShuffleSplit`) supaya frame mirip tidak bocor.

**data.yaml:**

```python
import yaml
names = ["car", "person", "bike"]                    # urutan = class_id 0,1,2
yaml.safe_dump({"path": os.path.abspath("yolo_data"), "train": "images/train", "val": "images/val",
                "names": {i: n for i, n in enumerate(names)}}, open("yolo_data/data.yaml", "w"))
```

**Selalu visualisasikan beberapa label** sebelum training (bug koordinat paling sering terjadi di sini):

```python
import matplotlib.pyplot as plt, matplotlib.patches as patches
def show_yolo(img_path, lbl_path, names):
    im = Image.open(img_path); W, H = im.size
    fig, ax = plt.subplots(figsize=(6, 6)); ax.imshow(im)
    for line in open(lbl_path).read().splitlines():
        c, cx, cy, w, h = map(float, line.split())
        ax.add_patch(patches.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False, color="red"))
        ax.text((cx - w / 2) * W, (cy - h / 2) * H, names[int(c)], color="yellow")
    ax.axis("off"); plt.show()
```

### 1.4 Training

```python
from ultralytics import YOLO

# n (nano) < s < m < l < x : makin besar makin akurat & lambat. CPU → n; GPU 4–8 GB → s/m.
try:
    model = YOLO("yolo11s.pt")        # versi baru
except Exception:
    model = YOLO("yolov8s.pt")        # fallback versi lama
results = model.train(
    data="yolo_data/data.yaml",
    imgsz=640,          # objek kecil → 960/1024 (butuh GPU besar)
    epochs=50,
    batch=16,           # -1 = auto batch sesuai memori GPU
    device=0,           # "cpu" kalau tanpa GPU
    workers=0,          # Windows: 0 lebih aman
    patience=15,        # early stopping
    seed=42,
    project="runs_det", name="yolo11s_640",
    cos_lr=True, close_mosaic=10,     # matikan mosaic di 10 epoch terakhir
    # augmentasi: fliplr=0.5, flipud=0.0 (nyalakan utk citra udara), degrees=0, mosaic=1.0, mixup=0.0
)
best = "runs_det/yolo11s_640/weights/best.pt"
```

**Cara membaca log:** `box_loss`, `cls_loss`, `dfl_loss` harus turun; `mAP50` dan `mAP50-95` di val naik.
Figure siap laporan otomatis tersimpan di folder run: `results.png`, `confusion_matrix.png`, `PR_curve.png`,
`F1_curve.png`, `val_batch0_pred.jpg`.

**Jika X maka Y:**

- mAP val = 0 terus → label tidak terbaca (cek folder `labels/` sejajar `images/`, nama file sama, `class_id` < jumlah `names`).
- mAP50 tinggi tapi mAP50-95 rendah → kotak kurang presisi: naikkan `imgsz`, model lebih besar, epoch lebih banyak.
- Objek kecil sering terlewat → `imgsz` lebih besar atau potong gambar besar jadi tile.
- CUDA OOM → `batch` lebih kecil / `imgsz` lebih kecil / model lebih kecil.

### 1.5 Validasi & mAP

```python
model = YOLO(best)
m = model.val(data="yolo_data/data.yaml", imgsz=640, conf=0.001, iou=0.6, split="val")
print("mAP50-95:", m.box.map, "| mAP50:", m.box.map50, "| mAP75:", m.box.map75)
print("per-class mAP50-95:", dict(zip(names, m.box.maps)))
```

**Konsep mAP (untuk laporan):**

- **IoU** = luas irisan / luas gabungan dua kotak. Prediksi dianggap benar (TP) kalau IoU dengan ground truth ≥ ambang
  (0.5 untuk mAP50) dan kelasnya sama; setiap ground truth hanya boleh dicocokkan sekali.
- Prediksi diurutkan dari confidence tertinggi → kurva precision-recall → **AP** = luas di bawah kurva (interpolasi).
- **mAP50** = rata-rata AP semua kelas pada IoU 0.5. **mAP50-95** (COCO mAP) = rata-rata mAP pada IoU 0.50, 0.55, …, 0.95.
- Karena itu, untuk metric mAP **jangan** buang prediksi confidence rendah (pakai `conf=0.001`): prediksi tambahan
  dengan confidence rendah hampir tidak pernah menurunkan AP tetapi bisa menaikkan recall.

Implementasi sederhana mAP@IoU (berguna kalau lomba memakai format sendiri):

```python
import numpy as np

def iou_xyxy(a, b):
    """a: (N,4), b: (M,4) dalam xyxy → (N,M)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

def average_precision(gt, pred, iou_thr=0.5):
    """gt: {img_id: array (G,4)}; pred: list of (img_id, score, box[4]) untuk SATU kelas."""
    n_gt = sum(len(v) for v in gt.values())
    if n_gt == 0:
        return np.nan
    used = {k: np.zeros(len(v), bool) for k, v in gt.items()}
    pred = sorted(pred, key=lambda p: -p[1])
    tp = np.zeros(len(pred)); fp = np.zeros(len(pred))
    for i, (img, s, box) in enumerate(pred):
        g = gt.get(img, np.zeros((0, 4)))
        if len(g):
            ious = iou_xyxy(np.asarray([box]), g)[0]
            j = int(ious.argmax())
            if ious[j] >= iou_thr and not used[img][j]:
                tp[i] = 1; used[img][j] = True; continue
        fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])                       # envelope
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

# mAP50 = rata-rata average_precision per kelas; mAP50-95 = rata-rata lagi untuk iou_thr di np.arange(0.5, 0.96, 0.05)
```

### 1.6 Prediksi & format submission

```python
model = YOLO(best)
rows = []
for r in model.predict(source="data/test_images", imgsz=640, conf=0.001, iou=0.6, augment=True,   # augment = TTA
                       stream=True, verbose=False, device=0):
    img_id = os.path.splitext(os.path.basename(r.path))[0]
    H, W = r.orig_shape
    boxes = r.boxes.xyxy.cpu().numpy()        # pixel absolut di ukuran ASLI gambar
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        rows.append({"image_id": img_id, "class": names[c], "score": float(s),
                     "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2})
pred_df = pd.DataFrame(rows)
```

Variasi format yang sering diminta:

```python
# (a) 1 baris per kotak dengan x, y, w, h
pred_df["x"], pred_df["y"] = pred_df["x_min"], pred_df["y_min"]
pred_df["w"], pred_df["h"] = pred_df["x_max"] - pred_df["x_min"], pred_df["y_max"] - pred_df["y_min"]

# (b) "PredictionString" per gambar: "class score x1 y1 x2 y2 class score ..."
def to_pred_string(g):
    return " ".join(f"{r['class']} {r['score']:.4f} {r['x_min']:.1f} {r['y_min']:.1f} {r['x_max']:.1f} {r['y_max']:.1f}"
                    for _, r in g.iterrows())
sub = pred_df.groupby("image_id").apply(to_pred_string).rename("PredictionString").reset_index()
ss = pd.read_csv("data/sample_submission.csv")
sub = ss[["image_id"]].merge(sub, on="image_id", how="left").fillna({"PredictionString": ""})  # gambar tanpa deteksi

# (c) COCO results JSON: [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]
# (d) Koordinat ternormalisasi: bagi x dengan W dan y dengan H (simpan W,H per gambar dari r.orig_shape)
```

**Cek wajib:** urutan & jumlah `image_id` sama dengan sample; gambar tanpa deteksi tetap ada barisnya; satuan
koordinat (pixel vs 0–1) dan urutan (`xyxy` vs `xywh`) sesuai dokumentasi lomba.

**Ensemble beberapa model detection:** jangan dirata-rata langsung; pakai Weighted Boxes Fusion
(`pip install ensemble-boxes`, `from ensemble_boxes import weighted_boxes_fusion` — input koordinat harus 0–1).

**Menghitung objek** (kalau target = jumlah objek): `count = (scores >= thr).sum()` per gambar, pilih `thr` yang
meminimalkan MAE di data validasi.

---

## 2. Semantic segmentation (segmentation_models_pytorch)

### 2.1 Install

```python
pip("segmentation-models-pytorch", "--no-deps")
pip("timm", "--no-deps")
pip("huggingface_hub", "safetensors", "pyyaml", "tqdm")
# versi lama smp (<0.4) juga butuh: pip("efficientnet-pytorch", "pretrainedmodels", "--no-deps")
import segmentation_models_pytorch as smp
```

### 2.2 Format mask

| Format | Isi | Cara baca |
|---|---|---|
| PNG mask | nilai pixel = id kelas (0 = background) atau 0/255 untuk binary | `np.array(Image.open(p))` (binary: `> 127`) |
| RLE (Kaggle) | `"start length start length ..."`, pixel 1-indexed, biasanya **column-major** | fungsi `rle_decode` di bawah |
| Polygon (COCO) | daftar titik `[x1, y1, x2, y2, ...]` | `ImageDraw.polygon` |

```python
def rle_decode(rle, shape, order="F"):
    """shape = (H, W). order='F' (column-major, umum di Kaggle) atau 'C' (row-major). Cek di deskripsi lomba!"""
    s = np.asarray(str(rle).split(), dtype=int) if isinstance(rle, str) and rle.strip() else np.array([], int)
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(s[0::2] - 1, s[1::2]):
        mask[start:start + length] = 1
    return mask.reshape(shape, order=order)

def rle_encode(mask, order="F"):
    pixels = np.concatenate([[0], mask.flatten(order=order).astype(np.uint8), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs))

# Uji wajib: rle_encode(rle_decode(r, shape)) == r  untuk beberapa baris train

def polygon_to_mask(polys, shape):
    from PIL import ImageDraw
    m = Image.new("L", (shape[1], shape[0]), 0)
    for p in polys:
        ImageDraw.Draw(m).polygon(list(map(float, p)), fill=1)
    return np.array(m)
```

### 2.3 Dataset + augmentasi yang sama untuk gambar & mask

Augmentasi geometris (flip, crop, rotasi) **harus identik** untuk gambar dan mask. Augmentasi warna hanya ke gambar.
Mask di-resize dengan **NEAREST** (bukan bilinear) supaya id kelas tidak tercampur.

```python
import random, torch
from torch.utils.data import Dataset

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

class SegDataset(Dataset):
    def __init__(self, img_paths, masks=None, size=256, train=False):
        self.img_paths, self.masks, self.size, self.train = img_paths, masks, size, train
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(img, np.float32) / 255.0
        m = None
        if self.masks is not None:
            m = Image.fromarray(self.masks[i]).resize((self.size, self.size), Image.NEAREST)
            m = np.asarray(m, np.int64)
        if self.train:
            if random.random() < 0.5:
                x = x[:, ::-1]; m = m[:, ::-1] if m is not None else m
            if random.random() < 0.5:
                x = x[::-1]; m = m[::-1] if m is not None else m
            x = np.clip(x * random.uniform(0.8, 1.2), 0, 1)       # brightness (gambar saja)
        x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy()).float()
        if m is None:
            return x
        return x, torch.from_numpy(m.copy())
```

`self.masks[i]` di sini adalah array (H, W) asli; untuk dataset besar, simpan path mask dan baca di `__getitem__`.
Ukuran input U-Net harus **kelipatan 32** (256, 320, 384, 512).

### 2.4 Model, loss, training

```python
N_CLASSES = 1          # binary → 1 channel + sigmoid; multiclass K kelas (termasuk background) → K channel + softmax
model = smp.Unet(encoder_name="resnet34",            # atau "efficientnet-b0", "tu-convnext_tiny" (via timm)
                 encoder_weights="imagenet", in_channels=3, classes=N_CLASSES).to(DEVICE)
# Alternatif: smp.UnetPlusPlus, smp.FPN, smp.DeepLabV3Plus (bagus untuk objek besar)

if N_CLASSES == 1:
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    def loss_fn(logits, y):
        y = y.float().unsqueeze(1)
        return 0.5 * dice(logits, y) + 0.5 * bce(logits, y)
else:
    dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    ce = torch.nn.CrossEntropyLoss()
    def loss_fn(logits, y):
        return 0.5 * dice(logits, y) + 0.5 * ce(logits, y)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    # validasi: hitung Dice/IoU di bawah, simpan checkpoint terbaik
```

**Kenapa Dice + BCE/CE?** BCE/CE stabil di awal training; Dice langsung mengoptimasi overlap dan tahan terhadap
ketidakseimbangan foreground (objek kecil). Kombinasi biasanya paling aman.

### 2.5 Metric: Dice & IoU

```python
def dice_iou_binary(prob, target, thr=0.5, eps=1e-7):
    """prob, target: (N,H,W). Return dice & IoU rata-rata per gambar.
    Gambar dengan mask kosong & prediksi kosong → skor 1 (konvensi Kaggle umum; CEK aturan lomba)."""
    pred = (prob >= thr).astype(np.uint8); target = target.astype(np.uint8)
    inter = (pred & target).sum((1, 2)); ps = pred.sum((1, 2)); ts = target.sum((1, 2))
    union = ps + ts - inter
    dice = np.where((ps + ts) == 0, 1.0, 2 * inter / (ps + ts + eps))
    iou = np.where(union == 0, 1.0, inter / (union + eps))
    return dice.mean(), iou.mean()

def iou_multiclass(pred, target, n_classes, ignore_bg=False):
    """pred, target: label map (N,H,W). mIoU = rata-rata IoU per kelas (dari total pixel)."""
    ious = []
    for c in range(1 if ignore_bg else 0, n_classes):
        p, t = pred == c, target == c
        union = (p | t).sum()
        if union:
            ious.append((p & t).sum() / union)
    return float(np.mean(ious)), ious
```

Dice = 2|A∩B| / (|A|+|B|), IoU = |A∩B| / |A∪B|, hubungan: Dice = 2·IoU / (1+IoU). Perhatikan apakah lomba
menghitung metric **per gambar lalu dirata-rata** atau **dari total pixel** — hasilnya bisa sangat beda.

### 2.6 Inference, TTA, post-processing, submission

```python
model.eval(); preds = []
with torch.no_grad():
    for x in test_loader:
        x = x.to(DEVICE)
        p = torch.sigmoid(model(x)) + torch.sigmoid(model(torch.flip(x, [3]))).flip(3)   # TTA hflip
        preds.append((p / 2).cpu().numpy()[:, 0])
preds = np.concatenate(preds)

# 1) Optimasi threshold di data validasi (bukan test): coba np.arange(0.3, 0.71, 0.05)
# 2) Resize kembali ke ukuran ASLI gambar sebelum encode (pakai bilinear pada probabilitas, lalu threshold)
# 3) Buang komponen kecil (noise) — tanpa library tambahan:
from scipy import ndimage
def remove_small(mask, min_area=100):
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, np.where(sizes >= min_area)[0] + 1)
    return keep.astype(np.uint8)

rows = []
for img_path, p in zip(test_paths, preds):
    W, H = Image.open(img_path).size
    p_full = np.asarray(Image.fromarray(p.astype(np.float32)).resize((W, H), Image.BILINEAR))
    m = remove_small(p_full >= BEST_THR)
    rows.append({"id": os.path.splitext(os.path.basename(img_path))[0], "rle": rle_encode(m)})
```

**Fallback tanpa smp** — U-Net mini pure PyTorch (cukup untuk gambar sederhana / dataset kecil):

```python
import torch.nn as nn
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
class MiniUNet(nn.Module):
    def __init__(self, n_classes=1, w=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList(); c = 3
        for o in w:
            self.downs.append(block(c, o)); c = o
        self.ups = nn.ModuleList([nn.ConvTranspose2d(w[i], w[i - 1], 2, stride=2) for i in range(len(w) - 1, 0, -1)])
        self.dec = nn.ModuleList([block(w[i - 1] * 2, w[i - 1]) for i in range(len(w) - 1, 0, -1)])
        self.head = nn.Conv2d(w[0], n_classes, 1)
    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x)
            if i < len(self.downs) - 1:
                skips.append(x); x = nn.functional.max_pool2d(x, 2)
        for up, dec in zip(self.ups, self.dec):
            x = up(x); x = dec(torch.cat([x, skips.pop()], 1))
        return self.head(x)
```

**Untuk laporan:** tampilkan grid `gambar | mask asli | mask prediksi` (beberapa terbaik & terburuk menurut Dice),
tabel Dice/IoU per kelas, dan kurva threshold vs Dice.

---

## 3. Image regression

Target berupa angka (umur, skor estetika, berat, jumlah). Cukup modifikasi template klasifikasi:

| Bagian template | Ganti menjadi |
|---|---|
| `N_CLASSES` | `1` output (`build_model(spec, 1, ...)`) |
| Encode label | tidak ada; simpan `y` float. **Scaling target** (`(y - mean) / std` atau `log1p` kalau skewed) membantu stabilitas |
| Fold | **binned stratified**: `bins = pd.qcut(y, q=10, labels=False, duplicates="drop")` → `StratifiedKFold` pada `bins` |
| Loss | `nn.MSELoss()` (RMSE), `nn.L1Loss()` (MAE), `nn.SmoothL1Loss()` / Huber (tahan outlier) |
| Output | `model(x).squeeze(1)` tanpa softmax; inverse-transform scaling di akhir |
| Mixup | aman (target juga di-mix: `y = lam*y_a + (1-lam)*y_b`); CutMix kurang cocok |
| TTA | rata-rata prediksi |
| Metric | RMSE (hitung manual), MAE, R² |
| Post-process | `np.clip(pred, y_min, y_max)`; bulatkan kalau target integer & metric mengizinkan |
| Baseline frozen features | `Ridge` / `LGBMRegressor` menggantikan LogisticRegression |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))     # jangan pakai squared=False (dihapus di sklearn baru)

y = train_df["target"].values.astype(np.float32)
use_log = (y.min() >= 0) and (pd.Series(y).skew() > 1)  # target miring ke kanan → log1p
y_t = np.log1p(y) if use_log else y
mu, sd = y_t.mean(), y_t.std() + 1e-8
y_scaled = (y_t - mu) / sd
inv = lambda p: (np.expm1(p * sd + mu) if use_log else p * sd + mu)

# Bagian training loop yang berubah:
criterion = nn.SmoothL1Loss(beta=1.0)
out = model(x).squeeze(1)
loss = criterion(out.float(), y_batch.float())

# Baseline cepat dari embedding:
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
oof = np.zeros(len(y))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X_feat, bins):
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X_feat[tr], y_scaled[tr])
    oof[va] = inv(reg.predict(X_feat[va]))
print("RMSE", rmse(y, oof), "MAE", mean_absolute_error(y, oof), "R2", r2_score(y, oof))
```

**Figure laporan:** scatter `true vs predicted` (dengan garis y=x), histogram residual, residual vs true
(pola miring = model "mengecilkan" nilai ekstrem → regression to the mean).

**Kalau target ordinal dengan sedikit nilai (0–4) dan metric QWK:** latih sebagai regresi, lalu cari threshold
pembulatan yang memaksimalkan QWK di OOF:

```python
from sklearn.metrics import cohen_kappa_score
def apply_thr(p, thr):
    return np.digitize(p, thr)
thr = np.array([0.5, 1.5, 2.5, 3.5])
for _ in range(3):                                     # coordinate ascent sederhana
    for i in range(len(thr)):
        best = max(np.arange(thr[i] - 0.5, thr[i] + 0.5, 0.02),
                   key=lambda t: cohen_kappa_score(y, apply_thr(oof, np.r_[thr[:i], t, thr[i + 1:]]), weights="quadratic"))
        thr[i] = best
```

---

## 4. Multilabel image classification

Satu gambar bisa punya banyak label (mis. `"cloudy primary road"`, atau kolom 0/1 per label).

```python
from sklearn.preprocessing import MultiLabelBinarizer
# Format A: string label dipisah spasi
labels = train["tags"].fillna("").str.split()
mlb = MultiLabelBinarizer(); Y = mlb.fit_transform(labels).astype(np.float32); CLASS_NAMES = list(mlb.classes_)
# Format B: kolom 0/1 per label
# CLASS_NAMES = [c for c in train.columns if c not in ("image_id",)]; Y = train[CLASS_NAMES].values.astype(np.float32)
print("label per gambar:", Y.sum(1).mean(), "| frekuensi per label:", dict(zip(CLASS_NAMES, Y.sum(0).astype(int))))
```

**Fold:** `iterstrat` (`pip install iterative-stratification`) → `MultilabelStratifiedKFold`. Fallback tanpa library:
stratifikasi berdasarkan label paling langka di tiap gambar.

```python
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    splits = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(Y)), Y)
except ImportError:
    freq = Y.sum(0)
    rarest = np.where(Y.sum(1) > 0, np.argmin(np.where(Y > 0, freq[None, :], np.inf), axis=1), -1)
    splits = StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(Y)), rarest)
```

**Perubahan model & loss:**

```python
model = build_model(spec, num_classes=len(CLASS_NAMES), pretrained=True)    # sama seperti template
pos = Y.sum(0); pos_weight = torch.tensor((len(Y) - pos) / np.maximum(pos, 1), dtype=torch.float32).clamp(max=20)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))           # pos_weight opsional (label langka)
# Dataset mengembalikan y sebagai vektor float (K,), bukan int
probs = torch.sigmoid(model(x))                                              # BUKAN softmax
```

Label smoothing untuk BCE: `y_smooth = y * (1 - eps) + 0.5 * eps`. Mixup boleh (target di-mix linear).

**Threshold per label** (dioptimasi di OOF, sangat berpengaruh untuk F1):

```python
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
thr = np.full(len(CLASS_NAMES), 0.5)
for k in range(len(CLASS_NAMES)):
    grid = np.arange(0.05, 0.96, 0.05)
    thr[k] = grid[np.argmax([f1_score(Y[:, k], oof[:, k] >= t, zero_division=0) for t in grid])]
pred = (oof >= thr).astype(int)
print("micro-F1:", f1_score(Y, pred, average="micro"), "| macro-F1:", f1_score(Y, pred, average="macro"),
      "| samples-F1:", f1_score(Y, pred, average="samples"), "| mAP:", average_precision_score(Y, oof, average="macro"))
# Pastikan minimal 1 label per gambar kalau aturan lomba mengharuskan:
empty = pred.sum(1) == 0
pred[empty, oof[empty].argmax(1)] = 1
```

Metric: **micro-F1** (didominasi label sering), **macro-F1** (semua label sama penting), **samples-F1**
(rata-rata per gambar), **mAP** = rata-rata average precision per label (tidak butuh threshold), ROC-AUC per label.
Kalau label langka tidak muncul di suatu fold, `roc_auc_score` per label bisa error → hitung per label dalam try/except.

**Submission:**

```python
sub["tags"] = [" ".join(np.array(CLASS_NAMES)[row.astype(bool)]) for row in pred_test]   # format string
# atau kolom 0/1 / probabilitas per label sesuai sample_submission
```

---

## 5. OCR (baca teks dari gambar)

Pilih sesuai kebutuhan:

| Opsi | Kelebihan | Kekurangan |
|---|---|---|
| **EasyOCR** | `pip` saja, banyak bahasa (termasuk `id`), deteksi + rekognisi | lambat di CPU, download model sekali |
| **pytesseract** | ringan, cepat untuk dokumen bersih | butuh install aplikasi Tesseract terpisah; lemah untuk foto/teks miring |
| **PaddleOCR** | akurat untuk scene text & dokumen | instalasi paddle cukup berat |
| **TrOCR** (`transformers`) | sangat akurat untuk 1 baris teks (cetak / tulisan tangan) | hanya rekognisi (butuh crop per baris), berat |
| Model sendiri (CNN + CTC / multi-head) | terbaik untuk teks pendek seragam (captcha, plat, meteran) | perlu data latih berlabel |

```python
# EasyOCR
pip("easyocr", "--no-deps"); pip("opencv-python-headless", "scikit-image", "python-bidi", "pyclipper", "shapely", "ninja")
import easyocr
reader = easyocr.Reader(["id", "en"], gpu=torch.cuda.is_available())
res = reader.readtext("img.jpg", detail=1, paragraph=False)      # [(box, text, conf), ...]
text = " ".join(t for _, t, c in res if c > 0.3)

# pytesseract (Windows: install Tesseract dari https://github.com/UB-Mannheim/tesseract/wiki lalu set path)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
txt = pytesseract.image_to_string(Image.open("img.jpg"), lang="eng", config="--psm 6")   # psm 7 = 1 baris

# TrOCR (1 baris teks)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")       # handwritten: trocr-base-handwritten
ocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
pix = proc(images=Image.open("line.png").convert("RGB"), return_tensors="pt").pixel_values
print(proc.batch_decode(ocr.generate(pix, max_new_tokens=32), skip_special_tokens=True)[0])
```

**Preprocessing yang sering membantu** (terutama untuk Tesseract):

```python
from PIL import ImageOps, ImageFilter
def prep_ocr(img, scale=2):
    g = ImageOps.grayscale(img)
    g = g.resize((g.width * scale, g.height * scale), Image.BICUBIC)          # perbesar teks kecil
    g = ImageOps.autocontrast(g).filter(ImageFilter.MedianFilter(3))          # kontras + hilangkan noise
    arr = np.asarray(g); thr = arr.mean()                                     # binarisasi sederhana
    return Image.fromarray(((arr > thr) * 255).astype(np.uint8))
```

**Post-processing:** normalisasi spasi/kapital, whitelist karakter (mis. hanya digit untuk meteran:
`config="--psm 7 -c tessedit_char_whitelist=0123456789"`), koreksi karakter mirip (`O→0`, `I→1`) kalau format diketahui.

**Evaluasi CER / WER** (tanpa library tambahan):

```python
def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def cer(refs, hyps):
    return sum(levenshtein(r, h) for r, h in zip(refs, hyps)) / max(sum(len(r) for r in refs), 1)

def wer(refs, hyps):
    return sum(levenshtein(r.split(), h.split()) for r, h in zip(refs, hyps)) / max(sum(len(r.split()) for r in refs), 1)

exact_match = np.mean([r == h for r, h in zip(refs, hyps)])
```

**Teks pendek dengan panjang tetap (captcha, kode 5–6 karakter):** perlakukan sebagai klasifikasi multi-head —
backbone dari template + `L` head linear (1 per posisi karakter), loss = jumlah CrossEntropy tiap posisi. Untuk panjang
bervariasi, gunakan CRNN + `nn.CTCLoss`.

---

## 6. Tips umum & troubleshooting lintas task

| Masalah | Solusi |
|---|---|
| `pip install X` ikut meng-upgrade/menurunkan torch | install dengan `--no-deps`, lalu dependensi ringan satu per satu; cek `torch.__version__` setelahnya |
| DataLoader hang / `Can't get attribute` di Windows | `num_workers=0` (atau `workers=0` di ultralytics) |
| CUDA OOM | kecilkan batch / resolusi / model; `torch.cuda.empty_cache()`; restart kernel |
| Loss NaN di GPU GTX 16xx | matikan fp16/AMP (`amp=False` di ultralytics `train`, atau jangan pakai autocast) |
| Koordinat kotak meleset di visualisasi | cek normalisasi (0–1 vs pixel), urutan `xyxy` vs `xywh`, dan EXIF rotation (`ImageOps.exif_transpose`) |
| Mask bergeser / pecah setelah resize | mask harus di-resize NEAREST, gambar BILINEAR; augmentasi geometris harus sama untuk keduanya |
| RLE submission ditolak | cek urutan (column-major `F` vs row-major `C`), indeks mulai 1, dan ukuran mask = ukuran asli gambar |
| Skor validasi bagus, leaderboard jelek | leakage antar split (frame video / pasien sama), threshold dioptimasi di test, beda resolusi train vs test |
| Download pretrained gagal (offline) | jalankan sekali saat online (cache di `~/.cache/torch` / `~/.cache/huggingface`), atau `pretrained=False` |

**Untuk laporan (semua task):** jelaskan format data & preprocessing, skema validasi, arsitektur + hyperparameter,
metric resmi beserta definisinya, tabel hasil validasi, contoh prediksi benar & salah (visual), dan keterbatasan.
