# cv_object_detection — `--task detection`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Object detection & counting (YOLO): bounding box atau jumlah objek

| | |
|---|---|
| Baca | `04_Computer_Vision/cv_object_detection.py` (1,126 baris) — notebook kembarannya: `04_Computer_Vision/cv_object_detection.ipynb` |
| `class CFG` | baris 85 · cell `# >>> USER OVERRIDE` baris 147 |
| Data yang diharapkan | data/train/images + labels (YOLO/COCO/VOC/CSV), data/test/images |
| Contoh `task_config.json` | `{"TRAIN_IMAGE_DIR": "./data/train/images", "ANNOTATION_PATH": null, "TEST_IMAGE_DIR": "./data/test/images", "TASK_MODE": "detect"}` |
| Folder output | `./outputs/cv_object_detection/` |
| Submission ditulis | baris 1050 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Computer Vision Competition Template — Object Detection & Counting | 27 |
| ↳ Pipeline besar | 35 |
| ↳ Cara pakai cepat | 51 |
| ↳ Format anotasi yang dikenali (`DATA_FORMAT="auto"`) | 62 |
| 1. Config (`CFG`) | 75 |
| 2. USER OVERRIDE | 144 |
| 3. Install / cek library | 191 |
| 4. Import, flag, device, helper | 232 |
| 5. Baca anotasi → tabel seragam | 373 |
| 6. EDA | 560 |
| 7. Split train/val + dataset YOLO | 606 |
| 8. Training | 654 |
| 9. Evaluasi di val (implementasi sendiri — sama untuk semua backend) | 777 |
| ↳ 9b. Kalibrasi jumlah (counting) | 924 |
| 10. Visual debugging di val | 962 |
| 11. Prediksi test + submission | 985 |
| 12. Ringkasan untuk laporan | 1070 |
| 13. Troubleshooting | 1096 |
| 14. Recipes | 1108 |
| 15. Checklist | 1119 |

## Semua key CFG (48)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `RUN_MODE` | `"full"` |  | 86 |
| `SEED` | `42` |  | 87 |
| `NOTEBOOK_NAME` | `"cv_object_detection"` |  | 88 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` | pip install ultralytics kalau belum ada (TIDAK pernah torch) | 89 |
| `TASK_MODE` | `"detect"` | "detect" (box + kelas) \| "count" (jumlah objek per gambar) | 90 |
| **data** | | | |
| `DATA_FORMAT` | `"auto"` | "auto" \| "yolo" \| "coco" \| "voc" \| "csv" | 93 |
| `TRAIN_IMAGE_DIR` | `"./data/train/images"` |  | 94 |
| `TRAIN_LABEL_DIR` | `"./data/train/labels"` | yolo txt / voc xml | 95 |
| `ANNOTATION_PATH` | `None` | coco .json atau csv (None = cari otomatis di ./data) | 96 |
| `TEST_IMAGE_DIR` | `"./data/test/images"` |  | 97 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 98 |
| `IMAGE_DIR_CANDIDATES` | `["./data/train/images", "./data/train", "./data/images", "./data/train_images"]` |  | 99 |
| `LABEL_DIR_CANDIDATES` | `["./data/train/labels", "./data/labels", "./data/annotations", "./data/train/annotations", "./data/Annotations"]` |  | 100 |
| `TEST_DIR_CANDIDATES` | `["./data/test/images", "./data/test", "./data/test_images"]` |  | 101 |
| `IMAGE_EXTS` | `(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")` |  | 102 |
| `CLASS_NAMES` | `None` | None = dari classes.txt / categories COCO / label unik CSV | 103 |
| `CSV_IMAGE_COL` | `None` | csv: kolom nama gambar (None = auto) | 104 |
| `CSV_LABEL_COL` | `None` | csv: kolom kelas (None = auto; tidak ada = 1 kelas) | 105 |
| `CSV_BOX_FORMAT` | `"auto"` | "auto" \| "xyxy" \| "xywh" | 106 |
| `OUTPUT_DIR` | `"./outputs/cv_object_detection/"` |  | 107 |
| **model** | | | |
| `BACKEND` | `"auto"` | "auto" \| "ultralytics" \| "torchvision" | 110 |
| `YOLO_MODEL` | `"yolo11n.pt"` | yolo11n/s/m/l/x.pt (pretrained COCO) — n paling ringan | 111 |
| `PRETRAINED` | `True` | False = dari nol (tanpa download weights; jauh lebih lemah) | 112 |
| `IMG_SIZE` | `640` |  | 113 |
| `EPOCHS` | `60` |  | 114 |
| `BATCH_SIZE` | `16` |  | 115 |
| `PATIENCE` | `15` |  | 116 |
| `LR0` | `None` | None = default ultralytics | 117 |
| `AMP` | `True` |  | 118 |
| `USE_GPU` | `True` |  | 119 |
| `NUM_WORKERS` | `0 if os.name == "nt" else 4` | Windows: 0 (mode script aman) | 120 |
| `VAL_FRAC` | `0.2` |  | 121 |
| `AUG_MOSAIC` | `1.0` | 0 kalau gambar sudah padat / objek di tepi penting | 122 |
| `AUG_FLIPUD` | `0.0` | 0.5 untuk citra udara/satelit/mikroskop | 123 |
| `TV_MODEL` | `"fasterrcnn_mobilenet_v3_large_fpn"` | fallback torchvision (atau fasterrcnn_resnet50_fpn_v2) | 124 |
| `TV_LR` | `1e-4` |  | 125 |
| **inference & submission** | | | |
| `CONF_THRESHOLD` | `None` | None = di-tune di val (F1 untuk detect, MAE untuk count) | 128 |
| `NMS_IOU` | `0.6` |  | 129 |
| `MAX_DET` | `300` |  | 130 |
| `TTA` | `False` | ultralytics augment=True saat predict | 131 |
| `COUNT_CALIBRATION` | `True` | count: kalibrasi linear dari fitur deteksi (dicek dengan CV di val) | 132 |
| `SUBMISSION_FORMAT` | `"auto"` | "auto" \| "boxes_csv" \| "prediction_string" \| "count" \| "coco_json" | 133 |
| `SUB_BOX_FORMAT` | `"xyxy"` | boxes_csv: "xyxy" \| "xywh" (pixel) | 134 |
| `SUB_ID_COL` | `None` |  | 135 |
| **fast mode** | | | |
| `FAST_SAMPLE` | `150` |  | 138 |
| `FAST_EPOCHS` | `3` |  | 139 |
| `FAST_IMG_SIZE` | `320` |  | 140 |
| `FIG_DPI` | `150` |  | 141 |
