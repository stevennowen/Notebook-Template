# tabular_master — `--task tabular`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Tabular: klasifikasi biner/multikelas, regresi

| | |
|---|---|
| Baca | `02_Tabular/tabular_master.py` (5,478 baris) — notebook kembarannya: `02_Tabular/tabular_master.ipynb` |
| `class CFG` | baris 439 · cell `# >>> USER OVERRIDE` baris 649 |
| Data yang diharapkan | data/train.csv, data/test.csv, data/sample_submission.csv |
| Contoh `task_config.json` | `{"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "SAMPLE_SUB_PATH": "./data/sample_submission.csv", "ID_COL": "id", "TARGET_COL": "target", "TASK": "auto", "METRIC": "auto"}` |
| Folder output | `./outputs/tabular_master/` |
| Submission ditulis | baris 5236 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Tabular Master Notebook — Binary / Multiclass / Regression | 27 |
| ↳ Daftar Isi | 47 |
| ↳ Cara Pakai Cepat (5 langkah) | 73 |
| ↳ Peta Keputusan — Section mana yang boleh di-skip kalau waktu mepet? | 84 |
| 1. Install & Import Library | 103 |
| 2. Konfigurasi (`class CFG`) | 405 |
| 3. Load Data (smart loading + multi-file merge) | 976 |
| 4. Data Quality Report (otomatis) + Cleaning | 1233 |
| 5. Analisis Target | 1618 |
| 6. EDA Visual Otomatis | 1858 |
| 7. Preprocessing Fold-Safe | 2426 |
| 8. Feature Engineering Toolkit | 2750 |
| 9. Strategi Cross-Validation | 3002 |
| 10. Definisi Model | 3120 |
| 11. Training Loop (OOF + Test Predictions) & Leaderboard | 3489 |
| 12. Hyperparameter Tuning (Optuna) | 3683 |
| 13. Ensembling | 3869 |
| 14. Post-processing | 4125 |
| 15. Evaluasi Mendalam & Debugging | 4388 |
| 16. Feature Importance & Interpretasi | 4695 |
| 17. Final Fit Strategy & Submission | 5048 |
| 18. Experiment Recipes, Checklist & Troubleshooting | 5339 |
| ↳ 18.1 Experiment Recipes (salin ke cell `# >>> USER OVERRIDE`) | 5341 |
| untuk RMSE dengan target skew: CFG.TARGET_TRANSFORM = "log1p" lalu bandingkan CV dengan No | 5373 |
| ↳ 18.2 Checklist sebelum submit | 5423 |
| ↳ 18.3 Troubleshooting | 5438 |

## Semua key CFG (129)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` | pip install package yang hilang (section 1). False jika offline. (env TEMPLATE_AUTO_INSTALL=0 → off) | 443 |
| `COMPETITION_NAME` | `"dac_its_tabular"` |  | 444 |
| `EXPERIMENT_NOTE` | `""` | catatan bebas untuk log eksperimen, misal "tambah FE rasio" | 445 |
| `RUN_MODE` | `"full"` | "fast" = debugging cepat (subsample, 3 fold, model sedikit) \| "full" = run final | 446 |
| `SEED` | `42` | seed utama | 447 |
| `SEEDS` | `None` | None = [SEED, SEED+1, ...] sebanyak N_REPEATS. Atau isi list, misal [42, 7, 2024] | 448 |
| `N_JOBS` | `-1` | jumlah core CPU (-1 = semua). Turunkan ke 2-4 jika laptop panas / memory error. | 449 |
| `VERBOSE` | `True` |  | 450 |
| `TRAIN_PATH` | `"./data/train.csv"` |  | 455 |
| `TEST_PATH` | `"./data/test.csv"` | boleh tidak ada → notebook hanya CV, submission di-skip | 456 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 457 |
| `OUTPUT_DIR` | `"./outputs/tabular_master/"` | subfolder figures/ tables/ models/ submissions/ dibuat otomatis | 458 |
| `EXTRA_TABLES` | `[]` | Tabel tambahan untuk di-merge (multi-file). Contoh: EXTRA_TABLES = [ {"path": "./data/customers.csv", "key": "customer_id", "agg": False, "prefix": "cust_"},       # one-to-one {"path": "./data/transactions.csv", "key":  | 464 |
| `ID_COL` | `"id"` | None jika tidak ada ID. Jika tidak ditemukan, dicoba ditebak dari sample_submission. | 469 |
| `TARGET_COL` | `"target"` | Jika tidak ditemukan, ditebak: kolom yang ada di train tapi tidak di test. | 470 |
| `DROP_COLS` | `[]` | kolom mentah yang dibuang sebelum apa pun (misal kolom bocor / tidak relevan) | 471 |
| `DROP_FEATURES` | `[]` | fitur yang dibuang SETELAH feature engineering (hasil seleksi fitur section 16) | 472 |
| `CAT_COLS` | `None` | None = auto (object/category/bool). Atau list manual. | 473 |
| `NUM_COLS` | `None` | None = auto (numerik). Atau list manual. | 474 |
| `NUM_AS_CAT_MAX_UNIQUE` | `0` | >0: kolom numerik integer dengan nunique <= nilai ini diperlakukan kategori (misal kode wilayah). 0 = off | 475 |
| `DATE_COLS` | `None` | None = auto-deteksi string tanggal. Atau list manual, misal ["signup_date"] | 476 |
| `TEXT_COLS` | `None` | None = auto (string panjang, banyak kata). Atau list manual. | 477 |
| `GROUP_COL` | `None` | kolom grup untuk GroupKFold (misal customer_id/patient_id) — baris satu grup tidak boleh terpisah | 478 |
| `TIME_COL` | `None` | kolom waktu untuk TimeSeriesSplit (test = masa depan) | 479 |
| `CLASS_ORDER` | `None` | urutan label (penting untuk QWK / ordinal), misal ["low","mid","high"]. None = urut alfabet | 480 |
| `POS_LABEL` | `None` | label positif untuk binary. None = auto (1/Yes/True/kelas minoritas) | 481 |
| `TASK` | `"auto"` | "auto" \| "binary" \| "multiclass" \| "regression" | 486 |
| `METRIC` | `"auto"` | "auto" → binary: roc_auc, multiclass: f1_macro, regression: rmse. SELALU isi sesuai lomba! | 487 |
| `SUBMISSION_MODE` | `"auto"` | "auto" \| "probability" \| "label". auto: metric berbasis probabilitas → probability, lainnya → label | 488 |
| `TARGET_TRANSFORM` | `"auto"` | regresi: None \| "log1p" \| "auto" (auto = log1p hanya jika metric rmsle) | 489 |
| `AUTO_CONVERT_NUMERIC_STRINGS` | `True` | " 1,234.5 ", "$20", "15%" → angka | 494 |
| `AUTO_CONVERT_BOOL_STRINGS` | `True` | "Yes"/"No", "TRUE"/"FALSE", "Y"/"N" → 1/0 | 495 |
| `AUTO_PARSE_DATES` | `True` | string tanggal → datetime (lalu dipecah jadi fitur di section 8) | 496 |
| `AUTO_DROP_CONSTANT` | `True` | buang kolom konstan / hampir konstan | 497 |
| `QUASI_CONSTANT_THRESH` | `0.999` | kolom dengan 1 nilai >= 99.9% baris dianggap hampir konstan | 498 |
| `AUTO_DROP_DUPLICATE_COLS` | `True` | buang kolom yang isinya identik dengan kolom lain | 499 |
| `AUTO_DROP_ID_LIKE` | `True` | buang kolom mirip ID (unik per baris) — biasanya tidak berguna / bocor | 500 |
| `DROP_DUPLICATE_ROWS` | `False` | True = buang baris duplikat (fitur+target sama persis) di train | 501 |
| `HIGH_CARD_THRESHOLD` | `50` | kategori dengan nunique > ini dianggap high-cardinality | 502 |
| `REDUCE_MEMORY` | `True` | downcast tipe numerik untuk hemat RAM | 503 |
| `DO_EDA` | `True` | False untuk hemat waktu (EDA tidak mempengaruhi model) | 508 |
| `EDA_MAX_PLOTS` | `12` | batas jumlah fitur yang diplot per jenis plot (data lebar) | 509 |
| `EDA_SAMPLE_ROWS` | `50000` | subsample baris untuk plot/statistik berat | 510 |
| `LEAK_AUC_THRESHOLD` | `0.95` | fitur tunggal dengan AUC > ini → dicurigai leakage | 511 |
| `LEAK_R2_THRESHOLD` | `0.90` | regresi: fitur tunggal dengan R2 > ini → dicurigai leakage | 512 |
| `DO_ADVERSARIAL` | `True` | adversarial validation train vs test | 513 |
| `ADV_AUC_WARN` | `0.70` | AUC adversarial > ini → ada drift yang perlu diperhatikan | 514 |
| `NUM_IMPUTE` | `"median"` | "median" \| "mean" \| "constant" (untuk model linear/KNN/SVM/RF; GBDT pakai NaN native) | 519 |
| `NUM_IMPUTE_CONSTANT` | `-999` |  | 520 |
| `ADD_MISSING_INDICATORS` | `True` | tambah kolom <col>__isna untuk model non-GBDT | 521 |
| `RARE_MIN_COUNT` | `10` | kategori dengan frekuensi < ini (di train fold) digabung jadi "__RARE__". 0 = off | 522 |
| `OHE_MAX_CARD` | `15` | model linear: one-hot jika nunique <= ini, sisanya target encoding | 523 |
| `TARGET_ENCODING` | `True` | target encoding (fold-safe) untuk kategori high-card di model linear/KNN/SVM/MLP | 524 |
| `TE_FOR_TREES` | `False` | True: tambahkan juga kolom target encoding untuk model tree (coba jika banyak kategori high-card) | 525 |
| `TE_MIN_CARD` | `10` | target encoding hanya untuk kategori dengan nunique >= ini (untuk TE_FOR_TREES) | 526 |
| `TE_SMOOTHING` | `20` | smoothing manual TE (semakin besar → semakin ditarik ke rata-rata global) | 527 |
| `TE_IMPL` | `"auto"` | "auto" (sklearn TargetEncoder jika ada) \| "sklearn" \| "manual" | 528 |
| `CLIP_OUTLIERS` | `False` | winsorize numerik (batas quantile di-fit di train fold) | 529 |
| `CLIP_QUANTILES` | `(0.005, 0.995)` |  | 530 |
| `LOG_TRANSFORM_LINEAR` | `True` | model linear: log1p otomatis untuk fitur skew>1 dan >=0 | 531 |
| `XGB_NATIVE_CAT` | `False` | True: XGBoost pakai categorical native (enable_categorical). False: ordinal codes (lebih aman) | 532 |
| `FE_DATE_PARTS` | `True` | tahun, bulan, hari, dayofweek, dst dari kolom tanggal | 537 |
| `FE_FREQ_ENCODING` | `True` | frekuensi kemunculan kategori | 538 |
| `FE_FREQ_FIT_ON` | `"train+test"` | "train+test" (transduktif, umum di lomba) \| "train" | 539 |
| `FE_GROUP_AGG` | `False` | agregasi numerik per kategori (mean/std/diff) — sering ampuh, tapi menambah banyak kolom | 540 |
| `FE_GROUP_AGG_CATS` | `None` | None = auto (3 kategori dengan MI tertinggi) | 541 |
| `FE_GROUP_AGG_NUMS` | `None` | None = auto (5 numerik dengan MI tertinggi) | 542 |
| `FE_RATIO_PAIRS` | `[]` | list pasangan (a, b) → a/b dan a-b. Contoh: [("TotalCharges", "tenure")] | 543 |
| `FE_AUTO_RATIO_TOPK` | `0` | >0: buat rasio otomatis antar top-k numerik (k=4 → 6 pasang) | 544 |
| `FE_POLY_TOPK` | `0` | >0: interaksi perkalian antar top-k numerik | 545 |
| `FE_BINNING_COLS` | `[]` | kolom numerik yang di-binning (quantile) | 546 |
| `FE_N_BINS` | `10` |  | 547 |
| `FE_TEXT_LEN` | `True` | panjang teks, jumlah kata, dll untuk kolom teks | 548 |
| `DROP_RAW_TEXT` | `True` | buang kolom teks mentah setelah diekstrak fiturnya | 549 |
| `FE_ROW_STATS` | `True` | n_missing per baris, statistik numerik per baris | 550 |
| `FE_CUSTOM` | `True` | jalankan custom_feature_engineering(df) | 551 |
| `CV_STRATEGY` | `"auto"` | "auto" \| "stratified" \| "kfold" \| "group" \| "stratified_group" \| "timeseries" | 556 |
| `N_FOLDS` | `5` |  | 557 |
| `N_REPEATS` | `1` | >1 = repeated CV dengan seed berbeda (lebih stabil, lebih lama) | 558 |
| `REG_STRAT_BINS` | `10` | regresi: stratifikasi berdasarkan bin target | 559 |
| `MODELS` | `["dummy", "linear", "rf", "et", "hgb", "lgb", "xgb", "cat"]` | Pilihan: "dummy","linear","enet","rf","et","hgb","lgb","xgb","cat","knn","svm","mlp" | 565 |
| `FAST_MODELS` | `["dummy", "linear", "hgb", "lgb"]` | model yang dipakai saat RUN_MODE="fast" | 566 |
| `MODEL_PARAMS` | `{}` | override parameter per model, misal {"lgb": {"num_leaves": 63}, "cat": {"depth": 8}} | 567 |
| `GBDT_N_ESTIMATORS` | `5000` | batas atas iterasi; early stopping yang menentukan jumlah sebenarnya | 568 |
| `GBDT_LR` | `0.03` | learning rate final. Fast mode memakai 0.08 | 569 |
| `EARLY_STOPPING` | `200` |  | 570 |
| `RF_N_ESTIMATORS` | `500` |  | 571 |
| `SMALL_DATA_MAX_ROWS` | `20000` | KNN/SVM hanya dijalankan jika jumlah baris <= ini | 572 |
| `IMBALANCE_STRATEGY` | `"auto"` | "auto" \| None \| "class_weight" \| "smote". auto: class_weight jika imbalance > 3x DAN metric berbasis label | 573 |
| `KEEP_FOLD_MODELS` | `True` | simpan model tiap fold di memori (dibutuhkan permutation importance/SHAP/PDP) | 574 |
| `USE_GPU` | `False` | True: coba GPU untuk LightGBM/XGBoost/CatBoost (fallback ke CPU jika gagal) | 575 |
| `LGB_GPU` | `None` | None = ikut USE_GPU. LightGBM GPU sering tidak ter-compile → default ikut USE_GPU tapi fallback | 576 |
| `XGB_GPU` | `None` |  | 577 |
| `CAT_GPU` | `None` |  | 578 |
| `USE_OPTUNA` | `True` | di fast mode otomatis False | 583 |
| `OPTUNA_MODELS` | `["lgb", "xgb", "cat"]` |  | 584 |
| `OPTUNA_N_TRIALS` | `40` |  | 585 |
| `OPTUNA_TIMEOUT` | `600` | detik per model (batas waktu) | 586 |
| `OPTUNA_N_FOLDS` | `None` | None = semua fold; isi 3 untuk mempercepat | 587 |
| `ENSEMBLE_METHODS` | `["mean", "rank", "weighted", "hill", "stack"]` |  | 592 |
| `ENSEMBLE_FORCE` | `None` | None = pilih terbaik berdasarkan OOF. Atau paksa "mean"/"weighted"/... atau nama model tunggal | 593 |
| `ENSEMBLE_EXCLUDE` | `["dummy"]` | model yang tidak ikut ensemble | 594 |
| `HILL_MAX_ITERS` | `50` |  | 595 |
| `THRESHOLD_TUNING` | `True` | binary + metric berbasis label → cari threshold terbaik di OOF | 596 |
| `CLASS_PRIOR_ADJUST` | `True` | multiclass + metric berbasis label → bobot per kelas | 597 |
| `CALIBRATION` | `"auto"` | "auto" \| "isotonic" \| "sigmoid" \| None. auto = coba, pakai jika metric logloss/brier membaik | 598 |
| `REG_CLIP` | `True` | regresi: clip prediksi ke range target train | 599 |
| `REG_ROUND` | `"auto"` | regresi: True/False/"auto" (auto: round jika target integer dan skor membaik) | 600 |
| `DO_EVAL_PLOTS` | `True` |  | 605 |
| `DO_LEARNING_CURVE` | `True` |  | 606 |
| `DO_SHUFFLE_TEST` | `True` |  | 607 |
| `DO_INTERPRET` | `True` | master switch section 16 | 608 |
| `DO_PERM_IMP` | `True` |  | 609 |
| `PERM_REPEATS` | `3` |  | 610 |
| `PERM_MAX_FEATURES` | `40` |  | 611 |
| `DO_SHAP` | `True` |  | 612 |
| `SHAP_SAMPLE` | `2000` |  | 613 |
| `SHAP_CLASS_INDEX` | `None` | multiclass: kelas yang diplot beeswarm/waterfall (None = kelas terakhir) | 614 |
| `DO_PDP` | `True` |  | 615 |
| `DO_NULL_IMP` | `True` |  | 616 |
| `NULL_IMP_RUNS` | `10` |  | 617 |
| `DO_FEATURE_SELECTION_CHECK` | `True` |  | 618 |
| `FINAL_FIT_STRATEGY` | `"fold_average"` | "fold_average" (rata-rata model tiap fold) \| "refit_full" (latih ulang di full train) | 623 |
| `SAVE_SUBMISSION_VARIANTS` | `True` |  | 624 |
| `SAMPLE_FRAC` | `0.3` | fast mode: proporsi train yang dipakai | 629 |
| `FAST_MAX_ROWS` | `20000` | fast mode: batas atas baris | 630 |
| `FAST_N_FOLDS` | `3` |  | 631 |
| `FIG_DPI` | `200` | 300 untuk figur laporan final | 632 |
| `SHOW_PLOTS` | `True` | False = hanya simpan file (lebih cepat untuk data lebar) | 633 |
