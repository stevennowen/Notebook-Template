# nlp_token_classification — `--task ner`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

NER / slot filling / token classification (BIO)

| | |
|---|---|
| Baca | `03_NLP/nlp_token_classification.py` (674 baris) — notebook kembarannya: `03_NLP/nlp_token_classification.ipynb` |
| `class CFG` | baris 53 · cell `# >>> USER OVERRIDE` baris 82 |
| Data yang diharapkan | data/train.conll|.txt|.jsonl|.csv, data/test.* |
| Contoh `task_config.json` | `{"TRAIN_PATH": "./data/train.conll", "TEST_PATH": "./data/test.conll"}` |
| Folder output | `./outputs/nlp_token_classification/` |
| Submission ditulis | baris 608 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| NLP Token Classification — NER / Slot Filling / POS (label per kata) | 27 |
| ↳ Pipeline | 33 |
| 1. Install & import | 125 |
| 2. Baca data (format otomatis) | 202 |
| 3. EDA | 324 |
| 4. Split & metrik entity-level | 356 |
| 5. Baseline: fitur kata + Logistic Regression | 393 |
| 6. Transformer fine-tuning | 445 |
| 7. Evaluasi & pemilihan model | 562 |
| 8. Prediksi test + submission | 593 |
| 9. Ringkasan untuk laporan | 634 |
| 10. Troubleshooting, recipes, checklist | 657 |

## Semua key CFG (25)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `RUN_MODE` | `"full"` |  | 54 |
| `SEED` | `42` |  | 55 |
| `NOTEBOOK_NAME` | `"nlp_token_classification"` |  | 56 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` |  | 57 |
| `TRAIN_PATH` | `"./data/train.conll"` |  | 58 |
| `TEST_PATH` | `"./data/test.conll"` |  | 59 |
| `VALID_PATH` | `None` | kalau panitia memberi data validasi terpisah | 60 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 61 |
| `FORMAT` | `"auto"` | "auto" \| "conll" \| "jsonl_tokens" \| "jsonl_spans" \| "csv_long" | 62 |
| `OUTPUT_DIR` | `"./outputs/nlp_token_classification/"` |  | 64 |
| `RUN_BASELINE` | `True` |  | 65 |
| `RUN_TRANSFORMER` | `"auto"` | "auto" (kalau transformers + model tersedia) \| True \| False | 66 |
| `TRANSFORMER_MODELS` | `["indobenchmark/indobert-base-p1"]` | Inggris: "bert-base-cased"; multibahasa: "xlm-roberta-base" | 67 |
| `MAX_LEN` | `128` |  | 68 |
| `EPOCHS` | `5` |  | 69 |
| `BATCH_SIZE` | `16` |  | 70 |
| `LR` | `3e-5` |  | 71 |
| `USE_AMP` | `True` |  | 72 |
| `USE_GPU` | `True` |  | 73 |
| `VAL_FRAC` | `0.15` |  | 74 |
| `PATIENCE` | `2` |  | 75 |
| `SUBMISSION_FORMAT` | `"auto"` | "auto" \| "conll" \| "csv_long" \| "jsonl" | 76 |
| `FAST_MAX_SENT` | `600` |  | 77 |
| `FAST_EPOCHS` | `2` |  | 78 |
| `FIG_DPI` | `150` |  | 79 |
