# nlp_llm_zero_shot — `--task llm`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

LLM lokal: zero/few-shot classification & ekstraksi (+ LoRA opsional)

| | |
|---|---|
| Baca | `03_NLP/nlp_llm_zero_shot.py` (574 baris) — notebook kembarannya: `03_NLP/nlp_llm_zero_shot.ipynb` |
| `class CFG` | baris 53 · cell `# >>> USER OVERRIDE` baris 88 |
| Data yang diharapkan | data/test.csv (id, text), data/train.csv opsional (contoh few-shot & evaluasi) |
| Contoh `task_config.json` | `{"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "TEXT_COL": "text", "TARGET_COL": "label", "LABELS": null}` |
| Folder output | `./outputs/nlp_llm_zero_shot/` |
| Submission ditulis | baris 499 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| LLM Lokal — Zero/Few-shot Classification & Ekstraksi | 27 |
| ↳ Pipeline | 33 |
| 1. Install & import | 132 |
| 2. Data | 238 |
| 3. Muat LLM | 259 |
| 4. Prompt & skor label | 279 |
| 5. Evaluasi di data berlabel (kalau ada) | 386 |
| 6. Prediksi test (classify) / ekstraksi (extract) | 445 |
| 7. Ringkasan untuk laporan | 520 |
| 8. Troubleshooting, recipes (LoRA), checklist | 543 |

## Semua key CFG (32)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `RUN_MODE` | `"full"` |  | 54 |
| `SEED` | `42` |  | 55 |
| `NOTEBOOK_NAME` | `"nlp_llm_zero_shot"` |  | 56 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` |  | 57 |
| `TASK` | `"classify"` | "classify" \| "extract" | 58 |
| `TRAIN_PATH` | `"./data/train.csv"` | opsional (contoh few-shot + evaluasi) | 59 |
| `TEST_PATH` | `"./data/test.csv"` |  | 60 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 61 |
| `ID_COL` | `"id"` |  | 62 |
| `TEXT_COL` | `"text"` |  | 63 |
| `TARGET_COL` | `"label"` |  | 64 |
| `LABELS` | `None` | None = label unik di train; atau isi manual, mis. ["positif", "negatif", "netral"] | 65 |
| `LABEL_DESCRIPTIONS` | `{}` | opsional: {"netral": "tidak memuji maupun mengeluh"} | 66 |
| `TASK_INSTRUCTION` | `"Classify the text into exactly one of the categories."` |  | 67 |
| `EXTRACT_FIELDS` | `{"product": "nama produk", "price": "harga dalam rupiah (angka)", "city": "kota"}` | TASK="extract" | 68 |
| `OUTPUT_DIR` | `"./outputs/nlp_llm_zero_shot/"` |  | 69 |
| `LLM_MODEL` | `"Qwen/Qwen2.5-1.5B-Instruct"` | kecil & multibahasa; alternatif: "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-3B-Instruct" | 70 |
| `N_SHOTS` | `2` | contoh per label di prompt (0 = zero-shot) | 71 |
| `SHOT_SELECTION` | `"similar"` | "balanced" \| "similar" (TF-IDF nearest neighbours per teks) | 72 |
| `CALIBRATE` | `True` |  | 73 |
| `MAX_TEXT_CHARS` | `1200` |  | 74 |
| `BATCH_SIZE` | `8` |  | 75 |
| `MAX_NEW_TOKENS` | `128` | mode extract | 76 |
| `USE_GPU` | `True` |  | 77 |
| `DTYPE` | `"auto"` | "auto" \| "float16" \| "bfloat16" \| "float32" | 78 |
| `EVAL_SAMPLE` | `300` | jumlah contoh train berlabel untuk evaluasi LLM | 79 |
| `RUN_BASELINE` | `True` |  | 80 |
| `RUN_EMBED_PROBE` | `True` | embedding rata-rata LLM + Logistic Regression (CV) | 81 |
| `ENSEMBLE` | `True` | rata-rata probabilitas LLM & baseline kalau lebih baik di evaluasi | 82 |
| `FAST_TEST_ROWS` | `40` |  | 83 |
| `FAST_EVAL_SAMPLE` | `40` |  | 84 |
| `FIG_DPI` | `150` |  | 85 |
