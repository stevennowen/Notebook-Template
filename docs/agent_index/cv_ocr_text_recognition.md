# cv_ocr_text_recognition — `--task ocr`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

OCR / text recognition: digit, plat, aksara, teks pendek dalam gambar (CRNN-CTC)

| | |
|---|---|
| Baca | `04_Computer_Vision/cv_ocr_text_recognition.py` (886 baris) — notebook kembarannya: `04_Computer_Vision/cv_ocr_text_recognition.ipynb` |
| `class CFG` | baris 73 · cell `# >>> USER OVERRIDE` baris 141 |
| Data yang diharapkan | data/train.csv (image, text) + data/train/, data/test/ |
| Contoh `task_config.json` | `{"TRAIN_CSV": "./data/train.csv", "TRAIN_IMAGE_DIR": "./data/train", "TEST_CSV": "./data/test.csv", "TEST_IMAGE_DIR": "./data/test"}` |
| Folder output | `./outputs/cv_ocr_text_recognition/` |
| Submission ditulis | baris 823 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Computer Vision Competition Template — OCR / Text Recognition | 27 |
| ↳ Pipeline besar | 36 |
| ↳ Cara pakai cepat | 52 |
| 1. Config (`CFG`) | 63 |
| 2. USER OVERRIDE | 138 |
| 3. Install / cek library | 185 |
| 4. Import, device, helper | 225 |
| 5. Load data + label | 323 |
| 6. Charset & EDA | 406 |
| 7. Dataset, preprocessing, augmentasi | 460 |
| 8. Model CRNN + CTC | 523 |
| 9. Decoding (greedy & beam search) + batasan | 562 |
| 10. Training | 634 |
| 11. Evaluasi OOF: greedy vs beam (+batasan) | 732 |
| ↳ 11b. (Opsional) TrOCR pretrained sebagai pembanding | 771 |
| 12. Prediksi test + submission | 795 |
| 13. Ringkasan untuk laporan | 835 |
| 14. Troubleshooting | 857 |
| 15. Recipes | 868 |
| 16. Checklist | 879 |

## Semua key CFG (50)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `RUN_MODE` | `"full"` |  | 74 |
| `SEED` | `42` |  | 75 |
| `NOTEBOOK_NAME` | `"cv_ocr_text_recognition"` |  | 76 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` |  | 77 |
| **data** | | | |
| `TRAIN_CSV` | `"./data/train.csv"` |  | 80 |
| `TRAIN_IMAGE_DIR` | `"./data/train"` |  | 81 |
| `TEST_CSV` | `"./data/test.csv"` |  | 82 |
| `TEST_IMAGE_DIR` | `"./data/test"` |  | 83 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 84 |
| `IMAGE_COL` | `None` | None = auto (kolom berisi nama file gambar) | 85 |
| `TEXT_COL` | `None` | None = auto | 86 |
| `LABEL_FROM_FILENAME` | `None` | tanpa CSV: regex dengan grup (?P<text>...), mis. r"^(?P<text>[0-9]+)_\d+$" | 87 |
| `IMAGE_EXTS` | `(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")` |  | 88 |
| `OUTPUT_DIR` | `"./outputs/cv_ocr_text_recognition/"` |  | 89 |
| **teks** | | | |
| `CHARSET` | `None` | None = semua karakter di label train | 92 |
| `CASE_SENSITIVE` | `True` | False → semua label di-upper() | 93 |
| `STRIP_SPACES` | `False` | True → spasi dihapus dari label | 94 |
| `FIXED_LENGTH` | `None` | mis. 6 → prediksi selalu 6 karakter | 95 |
| `TEXT_REGEX` | `None` | mis. r"^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$" (plat) → kandidat beam yang cocok diutamakan | 96 |
| **gambar & model** | | | |
| `IMG_H` | `32` |  | 99 |
| `IMG_W` | `None` | None = auto dari rasio median (kelipatan 4, 64–512) | 100 |
| `GRAYSCALE` | `True` |  | 101 |
| `CNN_CHANNELS` | `(64, 128, 256, 256)` |  | 102 |
| `RNN_HIDDEN` | `256` |  | 103 |
| `RNN_LAYERS` | `2` |  | 104 |
| `DROPOUT` | `0.2` |  | 105 |
| `USE_GPU` | `True` |  | 106 |
| `USE_AMP` | `True` |  | 107 |
| `BATCH_SIZE` | `32` | CTC butuh banyak step untuk keluar dari "plateau blank" → batch kecil membantu | 108 |
| `EPOCHS` | `60` |  | 109 |
| `LR` | `1e-3` |  | 110 |
| `WEIGHT_DECAY` | `1e-4` |  | 111 |
| `PATIENCE` | `12` |  | 112 |
| `MIN_EPOCHS` | `None` | early stopping baru aktif setelah epoch ini (None = sepertiga EPOCHS, min 8) | 113 |
| `N_FOLDS` | `1` | 1 = holdout (VAL_FRAC); ≥2 = K-fold + ensemble | 114 |
| `VAL_FRAC` | `0.15` |  | 115 |
| `NUM_WORKERS` | `0` |  | 116 |
| `BEAM_WIDTH` | `10` | 1 = greedy | 117 |
| `USE_TROCR` | `False` | True = bandingkan dengan TrOCR pretrained (butuh transformers + internet/cache) | 118 |
| `TROCR_MODEL` | `"microsoft/trocr-small-printed"` |  | 119 |
| **augmentasi** | | | |
| `AUG_ROTATE` | `3.0` | derajat | 122 |
| `AUG_SHEAR` | `0.15` |  | 123 |
| `AUG_BLUR` | `0.2` | peluang | 124 |
| `AUG_NOISE` | `0.2` |  | 125 |
| `AUG_BRIGHTNESS` | `0.3` |  | 126 |
| **submission** | | | |
| `SUB_ID_COL` | `None` |  | 129 |
| `SUB_TEXT_COL` | `None` |  | 130 |
| **fast** | | | |
| `FAST_SAMPLE` | `600` |  | 133 |
| `FAST_EPOCHS` | `15` |  | 134 |
| `FIG_DPI` | `150` |  | 135 |
