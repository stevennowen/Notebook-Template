# anomaly_tabular — `--task anomaly-tabular`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Anomaly / fraud / outlier detection pada data tabular (tanpa label atau sedikit label)

| | |
|---|---|
| Baca | `07_Anomaly_Detection/anomaly_tabular.py` (594 baris) — notebook kembarannya: `07_Anomaly_Detection/anomaly_tabular.ipynb` |
| `class CFG` | baris 53 · cell `# >>> USER OVERRIDE` baris 77 |
| Data yang diharapkan | data/train.csv (+ data/test.csv opsional; kolom label opsional) |
| Contoh `task_config.json` | `{"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "ID_COL": "id", "LABEL_COL": null}` |
| Folder output | `./outputs/anomaly_tabular/` |
| Submission ditulis | baris 551 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Anomaly / Fraud / Outlier Detection — Tabular | 27 |
| ↳ Pipeline | 33 |
| 1. Install & import | 121 |
| 2. Load data & persiapan fitur | 226 |
| 3. EDA singkat | 289 |
| 4. Detektor unsupervised | 317 |
| 5. Evaluasi (kalau ada label) + model supervised | 416 |
| 6. Threshold | 470 |
| 7. Penjelasan: mengapa baris ini anomali? | 503 |
| 8. Submission + ringkasan | 535 |
| 9. Troubleshooting, recipes, checklist | 576 |

## Semua key CFG (21)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `RUN_MODE` | `"full"` |  | 54 |
| `SEED` | `42` |  | 55 |
| `NOTEBOOK_NAME` | `"anomaly_tabular"` |  | 56 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` |  | 57 |
| `TRAIN_PATH` | `"./data/train.csv"` |  | 58 |
| `TEST_PATH` | `"./data/test.csv"` | boleh tidak ada (skor dihitung untuk train saja) | 59 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 60 |
| `OUTPUT_DIR` | `"./outputs/anomaly_tabular/"` |  | 61 |
| `ID_COL` | `"id"` |  | 62 |
| `LABEL_COL` | `None` | kolom label anomali (1 = anomali) kalau ada; None = unsupervised murni | 63 |
| `DROP_COLS` | `[]` |  | 64 |
| `CONTAMINATION` | `"auto"` | proporsi anomali yang diharapkan; "auto" = prevalensi label atau 0.05 | 65 |
| `METHODS` | `["iforest", "lof", "ocsvm", "robust_z", "pca", "autoencoder"]` |  | 66 |
| `FIT_ON` | `"train"` | "train" \| "train+test" (unsupervised: memakai semua data tanpa label) | 67 |
| `FIT_ONLY_NORMAL` | `True` | ada label: detektor di-fit hanya pada baris normal (lebih tajam) | 68 |
| `SUPERVISED` | `True` | ada label: latih juga HGB/LGBM (CV) + stacking skor unsupervised | 69 |
| `SUBMISSION_MODE` | `"auto"` | "auto" \| "score" \| "label" | 70 |
| `MAX_ONEHOT_CARD` | `20` |  | 71 |
| `N_FOLDS` | `5` |  | 72 |
| `FAST_MAX_ROWS` | `5000` |  | 73 |
| `FIG_DPI` | `150` |  | 74 |
