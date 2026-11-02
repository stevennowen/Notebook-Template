# nlp_regression — `--task nlp-regression`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Teks → angka (skor, harga, rating)

| | |
|---|---|
| Baca | `03_NLP/nlp_regression.py` (3,427 baris) — notebook kembarannya: `03_NLP/nlp_regression.ipynb` |
| `class CFG` | baris 141 · cell `# >>> USER OVERRIDE` baris 343 |
| Data yang diharapkan | data/train.csv (id, text, target), data/test.csv |
| Contoh `task_config.json` | `{"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "SAMPLE_SUB_PATH": "./data/sample_submission.csv", "ID_COL": "id", "TEXT_COL": "text", "TARGET_COL": "target"}` |
| Folder output | `./outputs/nlp_regression/` |
| Submission ditulis | baris 3273 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| NLP Text Regression — Competition Template | 27 |
| ↳ Pipeline besar | 43 |
| ↳ Cara pakai cepat (5 langkah) | 59 |
| ↳ Pilih model transformer (ringkas) | 67 |
| ↳ 0. Catatan platform & hardware | 80 |
| ↳ 1. CFG — pusat kontrol notebook | 133 |
| ↳ 2. USER OVERRIDE — ubah CFG di sini | 328 |
| ↳ 3. Cek & install library + finalisasi CFG | 393 |
| ↳ 4. Utilities: seed, figure/table saver, smart_read, timer, experiment log | 631 |
| ↳ 5. Load data | 762 |
| ↳ 6. Parsing target & analisis target | 814 |
| ↳ 7. Inspeksi dasar & sanity check | 933 |
| ↳ 8. Text cleaning & normalisasi Bahasa Indonesia | 997 |
| ↳ 9. Text EDA — helper functions | 1216 |
| ↳ 10. Text EDA | 1419 |
| ↳ 11. Metric registry & post-processing dasar | 1601 |
| ↳ 12. Cross-validation folds (binned stratified) | 1691 |
| ↳ 13. Result registry | 1743 |
| ↳ 14. Feature builder TF-IDF (word + char) & helper LightGBM | 1784 |
| ↳ 15. [A] Classical baselines: TF-IDF + model linear | 1865 |
| ↳ 16. [B] Sentence embeddings (frozen) — helper | 1951 |
| ↳ 17. [C] Transformer fine-tuning — engine | 2142 |
| ↳ 18. Pseudo-labeling (opsional, default OFF) | 2754 |
| ↳ 19. Perbandingan model | 2795 |
| ↳ 20. Ensemble | 2832 |
| ↳ 21. Post-processing: clip, rounding ke nilai valid, optimized rounding thresholds | 2957 |
| ↳ 22. Error analysis | 3044 |
| ↳ 23. Interpretability | 3135 |
| ↳ 24. Submission + validator | 3222 |
| ↳ 25. Simpan OOF, prediksi test, konfigurasi & ringkasan untuk laporan | 3283 |
| ↳ 26. Experiment recipes | 3324 |
| atau ekstrak bagian penting dulu (mis. paragraf "menjatuhkan pidana") lewat kolom baru → C | 3355 |
| ↳ 27. Troubleshooting | 3385 |
| ↳ 28. Checklist sebelum submit | 3416 |

## Semua key CFG (121)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| **RUN MODE** | | | |
| `RUN_MODE` | `"full"` | "fast" (debug: subsample, fold & epoch sedikit) \| "full" (final) | 145 |
| `FAST_SAMPLE_ROWS` | `3000` |  | 146 |
| `FAST_N_FOLDS` | `3` |  | 147 |
| `FAST_EPOCHS` | `1` |  | 148 |
| `SEED` | `42` |  | 149 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` | install package yang BELUM ada (tidak pernah upgrade, tidak pernah torch) (env TEMPLATE_AUTO_INSTALL=0 → off) | 150 |
| `INSTALL_OPTIONAL` | `True` |  | 151 |
| **PATHS** | | | |
| `TRAIN_PATH` | `"./data/train.csv"` |  | 156 |
| `TEST_PATH` | `"./data/test.csv"` | tidak ada → CV saja | 157 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 158 |
| `OUTPUT_DIR` | `f"./outputs/{NOTEBOOK_NAME}/"` |  | 159 |
| **KOLOM** | | | |
| `ID_COL` | `"id"` |  | 164 |
| `TEXT_COL` | `"text"` |  | 165 |
| `EXTRA_TEXT_COLS` | `[]` | kolom teks tambahan digabung: "title \| text" | 166 |
| `TARGET_COL` | `"target"` | angka; string angka ("1.234,5", "Rp 10.000") di-parse otomatis | 167 |
| `META_COLS` | `[]` | metadata kategorikal → disisipkan ke teks "[kategori: fashion] ..." | 168 |
| `USE_METADATA_AS_TEXT` | `True` |  | 169 |
| `GROUP_COL` | `None` | mis. "user_id" → GroupKFold | 170 |
| `SUBMISSION_TARGET_COL` | `None` | None = otomatis dari sample_submission / TARGET_COL | 171 |
| **METRIC & TARGET** | | | |
| `METRIC` | `"rmse"` | "rmse", "mae", "r2", "pearson", "spearman", "qwk" (target diskrit ordinal), "rmsle", "mape", "medae" | 177 |
| `TARGET_TRANSFORM` | `"auto"` | Transform target saat training (evaluasi tetap di skala asli): "none" \| "log1p" (target ≥ 0 & skewed) \| "sqrt" \| "auto" (log1p kalau skew > 1 dan min ≥ 0) | 180 |
| `CLIP_PREDICTIONS` | `True` | clip prediksi ke [MIN_TARGET, MAX_TARGET] | 181 |
| `MIN_TARGET` | `None` | None = min target train | 182 |
| `MAX_TARGET` | `None` | None = max target train | 183 |
| `VALID_VALUES` | `None` | nilai valid target, mis. [1, 2, 3, 4, 5]; None = deteksi otomatis kalau diskrit | 184 |
| `ROUND_TO_VALID` | `"auto"` | "auto" = coba rounding & pakai kalau OOF membaik \| True \| False | 185 |
| `OPTIMIZE_ROUNDING` | `True` | cari threshold rounding optimal (berguna untuk QWK / rating) | 186 |
| `MIN_TUNING_GAIN` | `0.002` | post-processing hanya dipakai kalau gain cross-validated ≥ ini (relatif ke skala metric) | 187 |
| **CROSS-VALIDATION** | | | |
| `N_FOLDS` | `5` |  | 192 |
| `CV_STRATEGY` | `"auto"` | "auto" (binned stratified / group) \| "binned" \| "kfold" \| "group" \| "stratified_group" | 193 |
| `TARGET_BINS` | `10` | jumlah bin target untuk stratifikasi | 194 |
| `NORMALIZE_URL` | `True` |  | 199 |
| `NORMALIZE_USER` | `True` |  | 200 |
| `KEEP_HASHTAG_TEXT` | `True` |  | 201 |
| `REMOVE_EMOJI` | `False` |  | 202 |
| `TRANSFORMER_LOWERCASE` | `False` |  | 203 |
| `TRANSFORMER_USE_SLANG` | `False` |  | 204 |
| `CLASSICAL_LOWERCASE` | `True` |  | 205 |
| `USE_SLANG_NORMALIZATION` | `True` |  | 206 |
| `CUSTOM_SLANG_DICT` | `{}` |  | 207 |
| `REDUCE_ELONGATION` | `True` |  | 208 |
| `REMOVE_STOPWORDS_CLASSICAL` | `False` |  | 209 |
| `USE_STEMMING` | `False` |  | 210 |
| `REMOVE_NUMBERS_CLASSICAL` | `False` | HATI-HATI: angka di teks sering sinyal kuat untuk regresi (luas, umur, dll.) | 211 |
| `RUN_EDA` | `True` |  | 216 |
| `EDA_TOKENIZER_MODEL` | `None` |  | 217 |
| `MAX_LEN_PERCENTILE` | `95` |  | 218 |
| `EDA_TARGET_GROUPS` | `3` | target dibagi jadi N kelompok (low/mid/high) untuk perbandingan teks | 219 |
| `TOP_NGRAMS` | `15` |  | 220 |
| `NEAR_DUP_THRESHOLD` | `0.90` |  | 221 |
| `NEAR_DUP_MAX_ROWS` | `20000` |  | 222 |
| `RUN_ADVERSARIAL_VALIDATION` | `True` |  | 223 |
| `MAKE_WORDCLOUD` | `True` |  | 224 |
| **[A] CLASSICAL (TF-IDF)** | | | |
| `RUN_CLASSICAL` | `True` |  | 229 |
| `CLASSICAL_MODELS` | `["ridge", "sgd", "linearsvr", "lgbm_svd"]` | pilihan: ridge, sgd, linearsvr, lgbm_svd | 230 |
| `TFIDF_WORD_NGRAM` | `(1, 2)` |  | 231 |
| `TFIDF_CHAR_NGRAM` | `(2, 5)` |  | 232 |
| `USE_CHAR_TFIDF` | `True` |  | 233 |
| `TFIDF_WORD_MAX_FEATURES` | `200_000` |  | 234 |
| `TFIDF_CHAR_MAX_FEATURES` | `300_000` |  | 235 |
| `TFIDF_MIN_DF` | `2` |  | 236 |
| `TFIDF_MAX_DF` | `0.95` |  | 237 |
| `RIDGE_ALPHA` | `1.0` | makin besar = regularisasi makin kuat (coba 0.3, 1, 3, 10) | 238 |
| `SVD_COMPONENTS` | `256` |  | 239 |
| `LGBM_N_ESTIMATORS` | `2000` |  | 240 |
| **[B] SENTENCE EMBEDDING** | | | |
| `RUN_EMBEDDINGS` | `True` |  | 245 |
| `EMBEDDING_MODELS` | `["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]` |  | 246 |
| `EMBEDDING_PREFIXES` | `{}` |  | 247 |
| `EMBEDDING_HEADS` | `["ridge", "svr", "lgbm"]` | + "xgb", "catboost" kalau terpasang | 248 |
| `EMB_BATCH_SIZE` | `64` |  | 249 |
| `EMB_MAX_LEN` | `256` |  | 250 |
| `CACHE_EMBEDDINGS` | `True` |  | 251 |
| **[C] TRANSFORMER FINE-TUNING** | | | |
| `RUN_TRANSFORMER` | `"auto"` | "auto" = hanya kalau CUDA \| True \| False | 256 |
| `TRAINER_BACKEND` | `"custom_loop"` | "custom_loop" \| "hf_trainer" | 257 |
| `MODEL_POOL` | `{deberta_v3_base, deberta_v3_large, roberta_base, modernbert_base, modernbert_large, indobert_base, indobert_large, indolem_indobert, indobertweet, xlm_roberta_base, xlm_roberta_large, mdeberta_v3_base}` |  | 258 |
| `TRANSFORMER_MODELS` | `["auto"]` | ["auto"] = rekomendasi deteksi bahasa | 272 |
| `MAX_LEN` | `"auto"` |  | 273 |
| `LONG_TEXT_STRATEGY` | `"head_tail"` | "truncate" \| "head_tail" \| "chunk" | 274 |
| `HEAD_RATIO` | `0.5` |  | 275 |
| `CHUNK_OVERLAP` | `64` |  | 276 |
| `CHUNK_POOLING` | `"mean"` |  | 277 |
| `MAX_CHUNKS_PER_DOC` | `8` |  | 278 |
| `BATCH_SIZE` | `16` |  | 279 |
| `EVAL_BATCH_SIZE` | `64` |  | 280 |
| `GRAD_ACCUM_STEPS` | `1` |  | 281 |
| `EPOCHS` | `3` |  | 282 |
| `LR` | `2e-5` |  | 283 |
| `HEAD_LR` | `1e-4` |  | 284 |
| `LLRD` | `1.0` |  | 285 |
| `WEIGHT_DECAY` | `0.01` |  | 286 |
| `WARMUP_RATIO` | `0.1` |  | 287 |
| `SCHEDULER` | `"linear"` |  | 288 |
| `MAX_GRAD_NORM` | `1.0` |  | 289 |
| `USE_AMP` | `True` |  | 290 |
| `AMP_DTYPE` | `"fp16"` |  | 291 |
| `GRADIENT_CHECKPOINTING` | `False` |  | 292 |
| `POOLING` | `"mean"` | mean pooling biasanya lebih stabil untuk regresi | 293 |
| `MULTI_SAMPLE_DROPOUT` | `0` |  | 294 |
| `HEAD_DROPOUT` | `0.0` | regresi: dropout di head sering MERUGIKAN (memperbesar variance output) | 295 |
| `BACKBONE_DROPOUT` | `0.0` | trik populer lomba regresi teks: matikan dropout backbone. None = default model | 296 |
| `REG_LOSS` | `"mse"` | "mse" \| "huber" (tahan outlier) \| "mae" | 297 |
| `EARLY_STOPPING_PATIENCE` | `2` |  | 298 |
| `EVALS_PER_EPOCH` | `1` |  | 299 |
| `OOM_MAX_RETRIES` | `3` |  | 300 |
| `SAVE_FOLD_MODELS` | `False` |  | 301 |
| `FOLDS_TO_RUN` | `None` |  | 302 |
| `NUM_WORKERS` | `0` |  | 303 |
| `DEBUG_SIMULATE_OOM` | `False` |  | 304 |
| **PSEUDO-LABELING** | | | |
| `RUN_PSEUDO_LABEL` | `False` |  | 309 |
| `PSEUDO_MAX_DISAGREEMENT` | `0.5` | ambil test rows dengan std prediksi antar model ≤ kuantil ini (0.5 = 50% paling sepakat) | 310 |
| `PSEUDO_MAX_FRACTION` | `0.5` |  | 311 |
| `RUN_ENSEMBLE` | `True` |  | 316 |
| `ENSEMBLE_METHODS` | `["mean", "weighted", "rank", "stacking"]` |  | 317 |
| `ENSEMBLE_TOP_K` | `6` |  | 318 |
| `MANUAL_WEIGHTS` | `{}` |  | 319 |
| `FINAL_PREDICTION` | `"auto"` |  | 320 |
| `RUN_INTERPRETABILITY` | `True` |  | 321 |
| `N_EXPLAIN_EXAMPLES` | `3` |  | 322 |
| `USE_SHAP` | `True` |  | 323 |
| `FIG_DPI` | `200` |  | 324 |
| `SHOW_FIGS` | `True` |  | 325 |
