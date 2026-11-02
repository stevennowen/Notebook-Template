# anomaly_image — `--task anomaly-image`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Anomaly detection gambar (cacat produk/visual inspection, hanya contoh normal)

| | |
|---|---|
| Baca | `07_Anomaly_Detection/anomaly_image.py` (515 baris) — notebook kembarannya: `07_Anomaly_Detection/anomaly_image.ipynb` |
| `class CFG` | baris 54 · cell `# >>> USER OVERRIDE` baris 82 |
| Data yang diharapkan | data/train/good/ (gambar normal), data/test/ (gambar campur) |
| Contoh `task_config.json` | `{"TRAIN_IMAGE_DIR": "./data/train/good", "TEST_IMAGE_DIR": "./data/test"}` |
| Folder output | `./outputs/anomaly_image/` |
| Submission ditulis | baris 473 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Anomaly Detection — Gambar (visual inspection / cacat produk) | 27 |
| ↳ Pipeline | 34 |
| 1. Import & cek library | 126 |
| 2. Data | 221 |
| 3. Fitur patch (PatchCore-lite) | 260 |
| 4. Skor anomali | 349 |
| 5. Evaluasi & pemilihan metode | 390 |
| 6. Heatmap lokasi cacat | 432 |
| 7. Submission + ringkasan | 456 |
| 8. Troubleshooting, recipes, checklist | 496 |

## Semua key CFG (25)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `RUN_MODE` | `"full"` |  | 55 |
| `SEED` | `42` |  | 56 |
| `NOTEBOOK_NAME` | `"anomaly_image"` |  | 57 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` |  | 58 |
| `TRAIN_IMAGE_DIR` | `"./data/train/good"` | hanya gambar normal | 59 |
| `TEST_IMAGE_DIR` | `"./data/test"` | gambar campur (normal & cacat); subfolder good/ & <cacat>/ dikenali sebagai label | 60 |
| `TEST_LABELS_PATH` | `None` | opsional: csv (id/gambar, label 0/1) untuk evaluasi | 61 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 62 |
| `IMAGE_EXTS` | `(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")` |  | 63 |
| `OUTPUT_DIR` | `"./outputs/anomaly_image/"` |  | 64 |
| `BACKBONE` | `"wide_resnet50_2"` | "resnet18" (cepat) \| "resnet50" \| "wide_resnet50_2" (standar PatchCore) | 65 |
| `PRETRAINED` | `True` | False = bobot acak (offline; jauh lebih lemah) | 66 |
| `IMG_SIZE` | `256` |  | 67 |
| `CENTER_CROP` | `224` |  | 68 |
| `CORESET_RATIO` | `0.1` | proporsi patch normal yang disimpan (0.01–0.25) | 69 |
| `MAX_BANK` | `60000` | batas jumlah patch di memory bank | 70 |
| `TOPK_SCORE` | `1` | skor gambar = rata-rata k patch terburuk (1 = max) | 71 |
| `VAL_FRAC` | `0.15` | gambar normal yang disisihkan untuk kalibrasi threshold (tanpa label) | 72 |
| `THRESHOLD_QUANTILE` | `0.99` | threshold = kuantil skor gambar normal val | 73 |
| `SUBMISSION_MODE` | `"auto"` | "auto" \| "score" \| "label" | 74 |
| `USE_GPU` | `True` |  | 75 |
| `BATCH_SIZE` | `16` |  | 76 |
| `FAST_MAX_TRAIN` | `80` |  | 77 |
| `FAST_BACKBONE` | `"resnet18"` |  | 78 |
| `FIG_DPI` | `150` |  | 79 |
