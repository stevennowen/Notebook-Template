# cv_segmentation — `--task segmentation`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Segmentasi gambar (semantic, biner/multikelas): mask PNG atau RLE

| | |
|---|---|
| Baca | `04_Computer_Vision/cv_segmentation.py` (1,439 baris) — notebook kembarannya: `04_Computer_Vision/cv_segmentation.ipynb` |
| `class CFG` | baris 88 · cell `# >>> USER OVERRIDE` baris 184 |
| Data yang diharapkan | data/train/images/, data/train/masks/ (atau train.csv berisi RLE), data/test/images/ |
| Contoh `task_config.json` | `{"TRAIN_IMAGE_DIR": "./data/train/images", "TRAIN_MASK_DIR": "./data/train/masks", "TEST_IMAGE_DIR": "./data/test/images", "SAMPLE_SUB_PATH": "./data/sample_submission.csv"}` |
| Folder output | `./outputs/cv_segmentation/` |
| Submission ditulis | baris 1343 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Computer Vision Competition Template — Image Segmentation (Semantic) | 27 |
| ↳ Pipeline besar | 35 |
| ↳ Cara pakai cepat (5 langkah) | 53 |
| ↳ Layout data yang didukung (`CFG.DATA_LAYOUT = "auto"`) | 65 |
| 1. Config (`CFG`) | 76 |
| 2. USER OVERRIDE (isi untuk lomba spesifik) | 173 |
| 3. Install / cek library | 229 |
| 4. Import, flag ketersediaan, device, helper | 279 |
| 5. Load data — deteksi layout otomatis | 445 |
| 6. Mask → index kelas | 559 |
| 7. EDA | 690 |
| 8. Cross-validation folds | 758 |
| 9. Dataset, augmentasi, DataLoader | 781 |
| 10. Model factory: smp → torchvision → U-Net kecil | 861 |
| 11. Loss & metrik | 955 |
| 12. Training K-fold | 1020 |
| 13. OOF: tuning threshold & min area (post-processing) | 1148 |
| 14. Evaluasi visual & debugging | 1215 |
| 15. Inference test + submission | 1268 |
| 16. Ringkasan untuk laporan | 1375 |
| 17. Troubleshooting | 1405 |
| 18. Experiment recipes (salin ke USER OVERRIDE) | 1418 |
| baseline kuat (GPU) | 1421 |
| objek kecil di gambar besar | 1423 |
| citra satelit / mikroskop (orientasi bebas) | 1425 |
| cepat untuk cek ide | 1427 |
| 19. Checklist sebelum submit | 1431 |

## Semua key CFG (67)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| **mode jalan** | | | |
| `RUN_MODE` | `"full"` | "fast" = debug cepat (subsample, 2 fold, epoch & gambar kecil) \| "full" = final | 90 |
| `SEED` | `42` |  | 91 |
| `NOTEBOOK_NAME` | `"cv_segmentation"` |  | 92 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` | pip install library opsional (TIDAK pernah torch) | 93 |
| `INSTALL_SMP` | `True` | segmentation_models_pytorch (+timm) di-install dengan --no-deps (torch tidak diganti) | 94 |
| **path data** | | | |
| `DATA_LAYOUT` | `"auto"` | "auto" \| "folder" \| "csv_rle" | 97 |
| `TRAIN_IMAGE_DIR` | `"./data/train/images"` |  | 98 |
| `TRAIN_MASK_DIR` | `"./data/train/masks"` |  | 99 |
| `TEST_IMAGE_DIR` | `"./data/test/images"` |  | 100 |
| `TRAIN_CSV` | `"./data/train.csv"` | layout csv_rle | 101 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 102 |
| `IMAGE_DIR_CANDIDATES` | `["./data/train/images", "./data/train_images", "./data/images", "./data/train", "./data/img"]` |  | 103 |
| `MASK_DIR_CANDIDATES` | `["./data/train/masks", "./data/train_masks", "./data/masks", "./data/labels", "./data/mask", "./data/annotations", "./data/gt"]` |  | 104 |
| `TEST_DIR_CANDIDATES` | `["./data/test/images", "./data/test_images", "./data/test"]` |  | 106 |
| `MASK_SUFFIXES` | `["", "_mask", "_masks", "_label", "_labels", "_gt", "_seg", "-mask"]` |  | 107 |
| `IMAGE_EXTS` | `(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")` |  | 108 |
| `OUTPUT_DIR` | `"./outputs/cv_segmentation/"` |  | 109 |
| **mask & kelas** | | | |
| `MASK_MODE` | `"auto"` | "auto" \| "index" (0..K-1) \| "binary255" (0/255) \| "rgb" (warna per kelas) | 112 |
| `N_CLASSES` | `None` | None = auto dari mask (termasuk background) | 113 |
| `CLASS_NAMES` | `None` | mis. ["background", "road", "water"]; None = class_0, class_1, ... | 114 |
| `IGNORE_INDEX` | `None` | nilai mask yang diabaikan saat training (mis. 255 di beberapa dataset) | 115 |
| `RGB_PALETTE` | `None` | layout rgb: {(r,g,b): idx}; None = dibangun otomatis dari warna yang muncul | 116 |
| `RLE_ID_COL` | `None` | csv_rle · None = auto (kolom pertama yang cocok nama gambar) | 118 |
| `RLE_COL` | `None` | None = auto (kolom berisi "1 3 10 5 ...") | 119 |
| `RLE_CLASS_COL` | `None` | multikelas: kolom kelas (1 baris per gambar per kelas) | 120 |
| `RLE_ORDER` | `"F"` | "F" = column-major (Kaggle umum) \| "C" = row-major. Cek dengan figure overlay! | 121 |
| `RLE_ONE_INDEXED` | `True` |  | 122 |
| **model** | | | |
| `ARCH` | `"auto"` | "auto" \| "Unet" \| "UnetPlusPlus" \| "FPN" \| "DeepLabV3Plus" (smp) \| "deeplabv3" \| "fcn" (torchvision) \| "unet_small" | 125 |
| `ENCODER` | `"resnet34"` | encoder smp (resnet18/34/50, efficientnet-b0..b4, mit_b0..) | 126 |
| `PRETRAINED` | `True` | encoder ImageNet (butuh internet sekali; cache di ~/.cache) | 127 |
| `USE_GPU` | `True` |  | 128 |
| `IMG_SIZE` | `320` | sisi gambar saat training (kelipatan 32) | 129 |
| `TILE_SIZE` | `None` | mis. 512 → crop acak ukuran asli (objek kecil) — inference tetap full image (resize) | 130 |
| `BATCH_SIZE` | `8` |  | 131 |
| `EPOCHS` | `25` |  | 132 |
| `LR` | `3e-4` |  | 133 |
| `WEIGHT_DECAY` | `1e-4` |  | 134 |
| `LOSS` | `"dice_ce"` | "dice_ce" \| "ce" \| "dice" \| "focal_dice" | 135 |
| `DICE_WEIGHT` | `0.5` |  | 136 |
| `USE_AMP` | `True` |  | 137 |
| `PATIENCE` | `6` | early stopping (epoch tanpa perbaikan) | 138 |
| `N_FOLDS` | `5` |  | 139 |
| `FOLDS_TO_RUN` | `None` | mis. [0] untuk 1 fold saja (cepat) | 140 |
| `NUM_WORKERS` | `0 if os.name == "nt" else 2` |  | 141 |
| `MAX_MINUTES_PER_FOLD` | `None` |  | 142 |
| **augmentasi** | | | |
| `AUG_HFLIP` | `True` |  | 145 |
| `AUG_VFLIP` | `True` | matikan untuk foto yang orientasinya penting (langit selalu di atas) | 146 |
| `AUG_ROT90` | `True` | citra satelit / mikroskop: True; foto jalan: False | 147 |
| `AUG_SCALE` | `(0.75, 1.0)` | random resized crop (luas relatif) | 148 |
| `AUG_BRIGHTNESS` | `0.2` |  | 149 |
| `AUG_CONTRAST` | `0.2` |  | 150 |
| **post-processing & submission** | | | |
| `TTA` | `["none", "hflip"]` | + "vflip" | 153 |
| `TUNE_THRESHOLD` | `True` | biner: cari threshold terbaik di OOF | 154 |
| `THRESHOLD` | `0.5` |  | 155 |
| `MIN_AREA` | `0` | buang komponen < MIN_AREA pixel (di-tune kalau TUNE_MIN_AREA) | 156 |
| `TUNE_MIN_AREA` | `True` |  | 157 |
| `METRIC` | `"dice"` | "dice" \| "iou" (mIoU) — sesuai aturan lomba | 158 |
| `METRIC_INCLUDE_BG` | `False` | multikelas: rata-rata tanpa background (umum di lomba) | 159 |
| `SUBMISSION_FORMAT` | `"auto"` | "auto" \| "rle" \| "mask_png" | 160 |
| `SUB_ID_COL` | `None` | None = kolom pertama sample_submission | 161 |
| `SUB_RLE_COL` | `None` | None = kolom ke-2 sample_submission | 162 |
| `EMPTY_RLE` | `""` | isi untuk mask kosong ("" atau "-" atau "1 0" — cek aturan) | 163 |
| **fast mode** | | | |
| `FAST_SAMPLE` | `120` |  | 166 |
| `FAST_EPOCHS` | `3` |  | 167 |
| `FAST_FOLDS` | `2` |  | 168 |
| `FAST_IMG_SIZE` | `128` |  | 169 |
| `FIG_DPI` | `150` |  | 170 |
