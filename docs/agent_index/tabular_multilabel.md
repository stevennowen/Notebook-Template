# tabular_multilabel — `--task tabular-multilabel`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Tabular multilabel: beberapa target 0/1 sekaligus (atau satu kolom berisi daftar label)

| | |
|---|---|
| Baca | `02_Tabular/tabular_multilabel.py` (564 baris) — notebook kembarannya: `02_Tabular/tabular_multilabel.ipynb` |
| `class CFG` | baris 57 · cell `# >>> USER OVERRIDE` baris 82 |
| Data yang diharapkan | data/train.csv (fitur + kolom label), data/test.csv, data/sample_submission.csv |
| Contoh `task_config.json` | `{"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "SAMPLE_SUB_PATH": "./data/sample_submission.csv", "ID_COL": "id", "TARGET_COL": null}` |
| Folder output | `./outputs/tabular_multilabel/` |
| Submission ditulis | baris 515 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Tabular Multilabel Template — beberapa target 0/1 sekaligus | 27 |
| ↳ Pipeline | 39 |
| 1. Install & import | 126 |
| 2. Load data & bentuk target multilabel | 226 |
| 3. EDA label | 267 |
| 4. Preprocessing (fold-safe) & fitur | 293 |
| 5. Training: Binary Relevance (+ Classifier Chain) | 360 |
| 6. Ensemble + threshold per label | 453 |
| 7. Submission + validator | 491 |
| 8. Ringkasan untuk laporan | 524 |
| 9. Troubleshooting, recipes, checklist | 547 |

## Semua key CFG (22)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `RUN_MODE` | `"full"` |  | 58 |
| `SEED` | `42` |  | 59 |
| `NOTEBOOK_NAME` | `"tabular_multilabel"` |  | 60 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` |  | 61 |
| `TRAIN_PATH` | `"./data/train.csv"` |  | 62 |
| `TEST_PATH` | `"./data/test.csv"` |  | 63 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 64 |
| `OUTPUT_DIR` | `"./outputs/tabular_multilabel/"` |  | 65 |
| `ID_COL` | `"id"` |  | 66 |
| `TARGET_COL` | `None` | list kolom 0/1, atau satu kolom berisi daftar label; None = auto dari sample_submission | 67 |
| `LABEL_SEP` | `None` | satu kolom daftar: pemisah ("\|", ",", " ", ";"); None = auto | 68 |
| `DROP_COLS` | `[]` |  | 69 |
| `METRIC` | `"auc_macro"` | "auc_macro" \| "f1_micro" \| "f1_macro" \| "logloss" \| "hamming" — sesuai aturan lomba | 70 |
| `SUBMISSION_MODE` | `"auto"` | "auto" \| "proba" \| "label"  (auto: dari isi sample_submission) | 71 |
| `MODELS` | `["logreg", "hgb", "lgb"]` |  | 72 |
| `USE_CHAIN` | `True` | Classifier Chain (urutan label: prevalensi tertinggi dulu) | 73 |
| `N_FOLDS` | `5` |  | 74 |
| `MAX_ONEHOT_CARD` | `30` | kategorikal ≤ ini → one-hot (logreg); lebih → ordinal | 75 |
| `TUNE_THRESHOLDS` | `True` |  | 76 |
| `FAST_MAX_ROWS` | `3000` |  | 77 |
| `FAST_N_FOLDS` | `3` |  | 78 |
| `FIG_DPI` | `150` |  | 79 |
