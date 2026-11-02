# cv_image_classification — `--task cv-classification`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Gambar → kelas

| | |
|---|---|
| Baca | `04_Computer_Vision/cv_image_classification.py` (3,682 baris) — notebook kembarannya: `04_Computer_Vision/cv_image_classification.ipynb` |
| `class CFG` | baris 107 · cell `# >>> USER OVERRIDE` baris 307 |
| Data yang diharapkan | data/train.csv + data/train/ (gambar) atau data/train/<kelas>/*.jpg, data/test/ |
| Contoh `task_config.json` | `{"TRAIN_CSV": "./data/train.csv", "TEST_CSV": "./data/test.csv", "TRAIN_IMAGE_DIR": "./data/train", "TEST_IMAGE_DIR": "./data/test"}` |
| Folder output | `./outputs/cv_image_classification/` |
| Submission ditulis | baris 3431 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Computer Vision Competition Template — Image Classification | 27 |
| ↳ Pipeline besar | 34 |
| ↳ Cara pakai cepat (5 langkah) | 59 |
| ↳ 4 layout data yang didukung (`CFG.DATA_LAYOUT = "auto"` mendeteksi sendiri) | 68 |
| ↳ Kalau tidak punya GPU | 77 |
| ↳ Kalau `timm` / `torchvision` tidak ada | 83 |
| 1. Config (`CFG`) | 89 |
| 2. USER OVERRIDE (isi untuk lomba spesifik) | 291 |
| 3. Install / cek library | 355 |
| 4. Imports, flag ketersediaan, device, helper umum | 438 |
| ↳ 4b. Terapkan RUN_MODE, buat folder output, helper (seed, save_fig, smart_read) | 570 |
| 5. Load data — deteksi layout otomatis | 706 |
| ↳ 5b. Encode label + FAST subsample | 1070 |
| 6. Image loading helper | 1110 |
| 7. Image EDA | 1173 |
| ↳ 7a. Scan semua gambar: file korup, ukuran, mode channel, hash, brightness, mean/std | 1175 |
| ↳ 7b. Distribusi kelas | 1318 |
| ↳ 7c. Contoh gambar per kelas | 1360 |
| ↳ 7d. Ukuran gambar, aspect ratio, mode channel | 1411 |
| ↳ 7e. Duplikat gambar (average hash) | 1481 |
| ↳ 7f. Brightness & train-vs-test shift | 1589 |
| 8. Cross-validation folds | 1648 |
| 9. Metrics | 1721 |
| 10. Transforms & augmentasi | 1791 |
| ↳ 10b. Preview augmentasi | 2001 |
| 11. Dataset, DataLoader, cache gambar | 2031 |
| 12. Model factory: timm → torchvision → SimpleCNN | 2117 |
| 13. Sanity check: overfit satu batch | 2290 |
| 14. Baseline cepat: frozen features + LogisticRegression / LightGBM | 2349 |
| 15. Training deep learning (K-fold) | 2551 |
| ↳ 15b. Learning curves & ringkasan CV | 2874 |
| 16. Ensemble (hill climbing di OOF) | 2930 |
| 17. Optimasi keputusan: threshold / class weights / temperature | 3003 |
| 18. Evaluasi model final | 3077 |
| ↳ 18a. Metric lengkap, confusion matrix, per-class report | 3079 |
| ↳ 18b. Gambar yang salah diprediksi & confidence histogram | 3156 |
| ↳ 18c. Grad-CAM (interpretability) | 3199 |
| 19. Submission + validator | 3302 |
| 20. Simpan artefak: OOF, test prediction, config, experiment log | 3445 |
| 21. Ringkasan untuk laporan | 3498 |
| 22. Debugging: sanity checks ringkas | 3532 |
| 23. Troubleshooting | 3561 |
| 24. Experiment Recipes | 3591 |
| 25. Checklist sebelum submit | 3669 |

## Semua key CFG (111)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| **MODE JALAN** | | | |
| `RUN_MODE` | `"full"` | "fast" = debug cepat: subsample data, fold & epoch sedikit, gambar kecil, 1 model. "full" = run final. Selalu jalankan "fast" dulu sampai tidak ada error. | 113 |
| `SEED` | `42` |  | 114 |
| `NOTEBOOK_NAME` | `"cv_image_classification"` |  | 115 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` | True: coba pip install library opsional yang hilang (TIDAK pernah install torch) (env TEMPLATE_AUTO_INSTALL=0 → off) | 116 |
| `INSTALL_TIMM` | `True` | timm di-install dengan --no-deps supaya TIDAK mengganti versi torch yang ada | 117 |
| **PATH DATA** | | | |
| `DATA_LAYOUT` | `"auto"` | "auto" \| "csv" \| "folder" \| "pixels"  (lihat tabel layout di atas) | 123 |
| `TRAIN_CSV` | `"./data/train.csv"` |  | 124 |
| `TEST_CSV` | `"./data/test.csv"` | boleh tidak ada | 125 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` | boleh tidak ada (tapi sangat disarankan) | 126 |
| `TRAIN_IMAGE_DIR` | `"./data/train"` | csv: folder gambar train \| folder: root berisi subfolder kelas | 127 |
| `TEST_IMAGE_DIR` | `"./data/test"` | folder gambar test | 128 |
| `IMAGE_DIR_CANDIDATES` | `["./data/train_images", "./data/images", "./data/train", "./data/imgs"]` | Folder lain yang dicoba otomatis kalau gambar tidak ketemu di TRAIN_IMAGE_DIR / TEST_IMAGE_DIR | 130 |
| `TEST_IMAGE_DIR_CANDIDATES` | `["./data/test_images", "./data/test", "./data/images"]` |  | 131 |
| `EXTRA_TRAIN_DIRS` | `[]` | layout folder: folder tambahan berlabel (mis. "./data/val") yang digabung ke train | 132 |
| `IMAGE_EXTS` | `(".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ppm", ".pgm")` |  | 133 |
| `IMAGE_COL` | `None` | kolom nama file / path gambar di train.csv (mis. "image_id", "filename") | 138 |
| `TEST_IMAGE_COL` | `None` | kalau nama kolom gambar di test.csv beda dari train | 139 |
| `TARGET_COL` | `None` | kolom label (mis. "label", "class") | 140 |
| `ID_COL` | `None` | kolom id di sample_submission/test (mis. "id", "image_id"); None = kolom pertama sample_sub | 141 |
| `SUB_TARGET_COL` | `None` | nama kolom target di submission; None = kolom ke-2 sample_sub / TARGET_COL | 142 |
| `GROUP_COL` | `None` | kolom grup (mis. "patient_id") → GroupKFold supaya gambar 1 pasien tidak bocor antar fold | 143 |
| `PIXEL_SHAPE` | `None` | Layout "pixels" (MNIST-like) · None = auto (sqrt jumlah pixel). Contoh: (28, 28) atau (32, 32, 3) | 146 |
| `PIXEL_PREFIX` | `None` | mis. "pixel" → hanya kolom yang diawali "pixel" dianggap pixel | 147 |
| `PIXEL_CHANNEL_FIRST` | `False` | True kalau urutan pixel (C,H,W) seperti CIFAR mentah | 148 |
| **OUTPUT** | | | |
| `OUTPUT_DIR` | `"./outputs/cv_image_classification/"` |  | 153 |
| `FIG_DPI` | `200` | 300 untuk figure laporan final | 154 |
| **TASK & METRIC** | | | |
| `METRIC` | `"f1_macro"` | "accuracy" \| "f1_macro" \| "f1" (binary, kelas positif = kelas ke-2 setelah sort) \| "logloss" \| "roc_auc" \| "qwk" | 160 |
| `SUBMISSION_MODE` | `"label"` | "label" = submit nama kelas \| "proba" = submit probabilitas (binary: 1 kolom P(kelas positif); multiclass: 1 kolom/kelas) | 162 |
| `OPTIMIZE_DECISION` | `True` | optimasi threshold (binary) / class weight (multiclass) di OOF untuk metric label-based | 163 |
| `MIN_DECISION_GAIN` | `0.002` | hanya pakai hasil optimasi kalau naik >= ini (anti-overfit OOF) | 164 |
| `RUN_EDA` | `True` |  | 169 |
| `SCAN_MAX_IMAGES` | `None` | None = scan semua gambar (cek file korup). Isi angka kalau dataset sangat besar. | 170 |
| `STAT_SIZE` | `64` | ukuran thumbnail untuk statistik (mean/std, brightness, hash) | 171 |
| `LOAD_TRUNCATED_IMAGES` | `True` | True: gambar JPEG terpotong tetap bisa dibaca (sisa diisi abu-abu) | 172 |
| `DROP_TRUNCATED` | `False` | True: gambar terpotong dibuang dari train | 173 |
| `NEAR_DUP_HAMMING` | `0` | 0 = hanya duplikat persis (hash sama). 2–4 = near-duplicate (lebih lambat) | 174 |
| `NEAR_DUP_MAX_N` | `6000` | batas jumlah gambar untuk near-duplicate check (O(N^2)) | 175 |
| `DROP_DUPLICATES` | `False` | True: buang duplikat persis di train (sisakan 1); label konflik dibuang | 176 |
| `DUP_AWARE_CV` | `True` | True: gambar duplikat dipaksa di fold yang sama (CV lebih jujur) | 177 |
| `GRID_PER_CLASS` | `6` | jumlah contoh gambar per kelas di sample grid | 178 |
| `MAX_CLASSES_IN_GRID` | `12` |  | 179 |
| **BACKBONE / MODEL** | | | |
| `USE_TIMM` | `True` | False = paksa tidak pakai timm walau terinstall | 184 |
| `USE_TORCHVISION` | `True` | False = paksa tidak pakai torchvision (transform & model) | 185 |
| `USE_GPU` | `True` |  | 186 |
| `MODEL_POOL` | `["efficientnet_b0", "convnext_tiny"]` | Nama model gaya timm. Kalau timm tidak ada → dicari padanan torchvision → kalau tidak ada → "simple_cnn". Pilihan umum (dari ringan ke berat): "mobilenetv3_large_100" (CPU-friendly), "resnet18", "efficientnet_b0", "resne | 191 |
| `PRETRAINED` | `True` | pakai ImageNet weights (download sekali, butuh internet) | 192 |
| `IMG_SIZE` | `224` | 224 standar; 128/160 untuk CPU; 288–384 kalau GPU kuat & detail kecil penting | 193 |
| `NORM` | `"auto"` | "auto" (imagenet kalau pretrained, dataset kalau scratch) \| "imagenet" \| "dataset" \| "none" | 194 |
| `SIMPLE_CNN_WIDTHS` | `(32, 64, 128, 256)` |  | 195 |
| `DROPOUT` | `0.2` | dropout sebelum classifier (SimpleCNN / timm drop_rate) | 196 |
| **AUGMENTASI** | | | |
| `AUG_LEVEL` | `"medium"` | "none" \| "light" \| "medium" \| "strong" | 201 |
| `HFLIP` | `True` | matikan kalau orientasi kiri-kanan bermakna (mis. teks, angka) | 202 |
| `VFLIP` | `False` | nyalakan untuk citra satelit / mikroskop / tampak atas | 203 |
| `ROTATION_DEG` | `15` |  | 204 |
| `COLOR_JITTER` | `0.2` | 0 = mati | 205 |
| `RANDOM_ERASING` | `0.0` | prob random erasing (0.1–0.25 untuk regularisasi) | 206 |
| `MIXUP_ALPHA` | `0.0` | 0.2–0.4 untuk mengaktifkan mixup | 207 |
| `CUTMIX_ALPHA` | `0.0` | 1.0 untuk mengaktifkan cutmix | 208 |
| `MIX_PROB` | `0.5` | peluang sebuah batch di-mix | 209 |
| **VALIDASI** | | | |
| `CV_STRATEGY` | `"kfold"` | "kfold" \| "holdout" | 214 |
| `N_FOLDS` | `5` |  | 215 |
| `HOLDOUT_SIZE` | `0.2` |  | 216 |
| `FOLDS_TO_RUN` | `None` | None = semua fold; contoh [0] untuk cepat (OOF hanya fold 0) | 217 |
| **TRAINING DEEP LEARNING** | | | |
| `RUN_DEEP` | `True` |  | 222 |
| `EPOCHS` | `10` |  | 223 |
| `BATCH_SIZE` | `32` |  | 224 |
| `ACCUM_STEPS` | `1` | gradient accumulation: effective batch = BATCH_SIZE * ACCUM_STEPS | 225 |
| `LR` | `3e-4` | LR backbone pretrained (fine-tuning) | 226 |
| `LR_SCRATCH` | `1e-3` | LR kalau model dilatih dari nol (SimpleCNN / pretrained gagal) | 227 |
| `HEAD_LR_MULT` | `10.0` | LR head = LR * ini (head baru butuh LR lebih besar) | 228 |
| `WEIGHT_DECAY` | `1e-2` |  | 229 |
| `WARMUP_EPOCHS` | `1` |  | 230 |
| `MIN_LR_RATIO` | `0.01` | LR akhir cosine = LR * ini | 231 |
| `LABEL_SMOOTHING` | `0.1` |  | 232 |
| `CLASS_WEIGHT` | `None` | None \| "balanced"  (untuk data imbalanced) | 233 |
| `USE_WEIGHTED_SAMPLER` | `False` | alternatif class weight: oversampling kelas kecil (jangan dipakai bersamaan) | 234 |
| `FREEZE_EPOCHS` | `0` | >0: epoch awal hanya melatih head (linear probe), lalu unfreeze semua | 235 |
| `GRAD_CLIP` | `1.0` | 0 = mati | 236 |
| `PATIENCE` | `3` | early stopping (epoch tanpa perbaikan) | 237 |
| `USE_AMP` | `True` | mixed precision (hanya aktif kalau CUDA) | 238 |
| `AMP_DISABLE_ON_GTX16` | `True` | GTX 16xx (1650/1660) punya bug fp16 di cuDNN → output NaN pada ukuran batch tertentu. True = AMP dimatikan otomatis di GPU tersebut. Selain itu notebook juga punya pengaman: kalau muncul NaN saat AMP, batch dihitung ulan | 242 |
| `NUM_WORKERS` | `0 if os.name == "nt" else 2` | Windows: 0 (lihat penjelasan di section DataLoader) | 243 |
| `CACHE_IMAGES` | `"auto"` | "auto" \| True \| False — simpan gambar ter-resize di RAM (sangat mempercepat di Windows) | 244 |
| `CACHE_MAX_GB` | `2.0` |  | 245 |
| `OOM_RETRIES` | `3` | CUDA OOM → batch size dibagi 2 & accum dikali 2, lalu ulang | 246 |
| `MIN_BATCH_SIZE` | `4` |  | 247 |
| `DETERMINISTIC` | `False` | True = hasil identik tiap run (lebih lambat) | 248 |
| `MAX_MINUTES_PER_MODEL` | `None` | batas waktu per model; fold sisanya di-skip kalau lewat | 249 |
| `SKIP_TRAIN_IF_CKPT` | `False` | True: kalau checkpoint fold sudah ada, load saja (tidak training ulang) | 250 |
| `RUN_OVERFIT_TEST` | `True` | sanity check: model harus bisa menghafal 1 batch | 251 |
| `OVERFIT_STEPS` | `25` |  | 252 |
| **INFERENCE / TTA** | | | |
| `TTA_HFLIP` | `True` |  | 257 |
| `TTA_SCALES` | `[1.0]` | contoh [1.0, 1.15] = multi-scale (resize lebih besar lalu center crop) | 258 |
| `TTA_ON_OOF` | `True` | OOF dihitung dengan TTA yang sama dengan test (lebih konsisten) | 259 |
| `RUN_FEATURE_BASELINE` | `True` |  | 264 |
| `FEATURE_SOURCE` | `"auto"` | "auto" (backbone pretrained kalau ada, else pixel) \| "backbone" \| "pixels" | 265 |
| `FEATURE_BACKBONE` | `None` | None = model pertama di MODEL_POOL | 266 |
| `FEAT_PIXEL_SIZE` | `32` | ukuran resize untuk pixel features | 267 |
| `PCA_COMPONENTS` | `128` |  | 268 |
| `LOGREG_C` | `1.0` |  | 269 |
| `RUN_LGB_ON_FEATURES` | `True` |  | 270 |
| **ENSEMBLE & EVALUASI** | | | |
| `RUN_ENSEMBLE` | `True` |  | 275 |
| `ENSEMBLE_ITERS` | `25` |  | 276 |
| `RUN_GRADCAM` | `True` |  | 277 |
| `N_GRADCAM` | `8` |  | 278 |
| `N_MISCLASSIFIED` | `16` |  | 279 |
| `FAST_SAMPLE` | `800` | maksimal jumlah gambar train | 284 |
| `FAST_EPOCHS` | `2` |  | 285 |
| `FAST_FOLDS` | `2` |  | 286 |
| `FAST_IMG_SIZE` | `96` |  | 287 |
| `FAST_N_MODELS` | `1` |  | 288 |
