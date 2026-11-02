# nlp_classification — `--task nlp-classification`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Teks → kelas (biner/multikelas/multilabel): TF-IDF, embedding, transformer

| | |
|---|---|
| Baca | `03_NLP/nlp_classification.py` (3,878 baris) — notebook kembarannya: `03_NLP/nlp_classification.ipynb` |
| `class CFG` | baris 145 · cell `# >>> USER OVERRIDE` baris 368 |
| Data yang diharapkan | data/train.csv (id, text, label), data/test.csv |
| Contoh `task_config.json` | `{"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "SAMPLE_SUB_PATH": "./data/sample_submission.csv", "ID_COL": "id", "TEXT_COL": "text", "TARGET_COL": "label"}` |
| Folder output | `./outputs/nlp_classification/` |
| Submission ditulis | baris 3670 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| NLP Text Classification — Competition Template (Binary / Multiclass / Multilabel) | 27 |
| ↳ Pipeline besar | 40 |
| ↳ Cara pakai cepat (5 langkah) | 57 |
| ↳ Pilih model transformer (ringkas) | 65 |
| ↳ 0. Catatan platform & hardware | 79 |
| ↳ 1. CFG — pusat kontrol notebook | 134 |
| ↳ 2. USER OVERRIDE — ubah CFG di sini | 350 |
| ↳ 3. Cek & install library + finalisasi CFG | 418 |
| ↳ 4. Utilities: seed, figure/table saver, smart_read, timer, experiment log | 656 |
| ↳ 5. Load data | 787 |
| ↳ 6. Encode target & deteksi task | 850 |
| ↳ 7. Inspeksi dasar & sanity check | 954 |
| ↳ 8. Text cleaning & normalisasi Bahasa Indonesia | 1066 |
| ↳ 9. Text EDA — helper functions | 1285 |
| ↳ 10. Text EDA | 1488 |
| ↳ 11. Metric registry & aturan keputusan (threshold / bias) | 1704 |
| ↳ 12. Cross-validation folds | 1825 |
| ↳ 13. Result registry (semua model disimpan di satu tempat) | 1884 |
| ↳ 14. Feature builder TF-IDF (word + char) & helper LightGBM | 1938 |
| ↳ 15. [A] Classical baselines: TF-IDF + model linear | 2019 |
| ↳ 16. [B] Sentence embeddings (frozen) — helper | 2172 |
| ↳ 17. [C] Transformer fine-tuning — engine | 2394 |
| ↳ 18. Pseudo-labeling (opsional, default OFF) | 3014 |
| ↳ 19. Perbandingan model | 3066 |
| ↳ 20. Ensemble | 3110 |
| ↳ 21. Threshold / bias tuning | 3285 |
| ↳ 22. Error analysis | 3371 |
| ↳ 23. Interpretability | 3490 |
| ↳ 24. Submission + validator | 3597 |
| ↳ 25. Simpan OOF, prediksi test, konfigurasi & ringkasan untuk laporan | 3682 |
| ↳ 26. Catatan multilabel (sudah didukung) | 3740 |
| format A: satu kolom berisi "harga|pengiriman" | 3747 |
| format B: beberapa kolom 0/1 | 3749 |
| ↳ 27. Experiment recipes | 3766 |
| atau model konteks panjang | 3806 |
| ↳ 28. Troubleshooting | 3833 |
| ↳ 29. Checklist sebelum submit | 3866 |

## Semua key CFG (121)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| **RUN MODE** | | | |
| `RUN_MODE` | `"full"` | "fast": debugging cepat → subsample data, fold & epoch sedikit, model classical lebih sedikit. "full": run final untuk submission. Selalu cek "fast" dulu sampai semua cell lolos. | 151 |
| `FAST_SAMPLE_ROWS` | `3000` | jumlah baris train saat fast (stratified subsample) | 152 |
| `FAST_N_FOLDS` | `3` |  | 153 |
| `FAST_EPOCHS` | `1` |  | 154 |
| `SEED` | `42` |  | 155 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` | Install otomatis package yang belum ada (pip). Torch TIDAK pernah di-install otomatis. · (env TEMPLATE_AUTO_INSTALL=0 → off) | 158 |
| `INSTALL_OPTIONAL` | `True` | ikut install yang opsional (lightgbm, sentence-transformers, dll.) | 159 |
| **PATHS** | | | |
| `TRAIN_PATH` | `"./data/train.csv"` |  | 164 |
| `TEST_PATH` | `"./data/test.csv"` | kalau tidak ada → notebook tetap jalan (CV saja) | 165 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 166 |
| `OUTPUT_DIR` | `f"./outputs/{NOTEBOOK_NAME}/"` |  | 167 |
| **KOLOM** | | | |
| `ID_COL` | `"id"` | kalau tidak ada kolom id, biarkan; submission pakai urutan baris | 172 |
| `TEXT_COL` | `"text"` | kolom teks utama | 173 |
| `EXTRA_TEXT_COLS` | `[]` | kolom teks tambahan yang digabung, contoh ["title"] → "title \| text" | 174 |
| `TARGET_COL` | `"label"` | kolom target (string atau angka, akan di-encode otomatis) | 175 |
| `META_COLS` | `[]` | metadata kategorikal (mis. ["platform", "keyword"]) → disisipkan ke teks | 176 |
| `USE_METADATA_AS_TEXT` | `True` | True: "[platform: shopee] teks..." (berguna untuk transformer & TF-IDF) | 177 |
| `GROUP_COL` | `None` | mis. "user_id"/"article_id" → GroupKFold supaya tidak bocor antar fold | 178 |
| `SUBMISSION_TARGET_COL` | `None` | None = otomatis (pakai kolom di sample_submission / TARGET_COL) | 179 |
| **TASK & METRIC** | | | |
| `TASK_TYPE` | `"auto"` | "auto" → binary kalau 2 kelas, multiclass kalau >2, multilabel kalau TARGET_COLS / MULTILABEL_SEP diisi. · "auto" \| "binary" \| "multiclass" \| "multilabel" | 185 |
| `TARGET_COLS` | `[]` | multilabel format kolom 0/1 terpisah, mis. ["harga", "kualitas"] | 186 |
| `MULTILABEL_SEP` | `None` | multilabel format 1 kolom, mis. "harga\|kualitas" → MULTILABEL_SEP = "\|" | 187 |
| `POSITIVE_LABEL` | `None` | binary: label mana yang "positif" (mis. "positif", 1). None = kelas ke-2 urut abjad | 188 |
| `CLASS_ORDER` | `None` | urutan kelas manual (penting untuk QWK / label ordinal), mis. ["low","mid","high"] | 189 |
| `METRIC` | `"auto"` | METRIC sesuai aturan lomba: binary     : "f1", "accuracy", "roc_auc", "log_loss", "mcc", "balanced_accuracy", "pr_auc", "precision", "recall" multiclass : "f1_macro", "f1_micro", "f1_weighted", "accuracy", "log_loss", "r | 196 |
| `SUBMISSION_MODE` | `"label"` | "label" = submit kelas akhir; "probability" = submit probabilitas (biasanya untuk AUC/logloss) | 198 |
| **CROSS-VALIDATION** | | | |
| `N_FOLDS` | `5` | 5 standar; 3 kalau data besar/waktu sempit; 10 kalau data kecil (<2k) | 203 |
| `CV_STRATEGY` | `"auto"` | "auto" \| "stratified" \| "group" \| "stratified_group" \| "kfold" | 204 |
| **TEXT CLEANING** | | | |
| `NORMALIZE_URL` | `True` | Transformer: cleaning MINIMAL (tokenizer sudah pintar, kata slang/emoji bisa jadi sinyal). TF-IDF     : normalisasi lebih banyak membantu (lowercase, slang → baku, elongation, dll.). · "http://..." → "<URL>" | 211 |
| `NORMALIZE_USER` | `True` | "@akun" → "<USER>" | 212 |
| `KEEP_HASHTAG_TEXT` | `True` | "#promo" → "promo" | 213 |
| `REMOVE_EMOJI` | `False` | emoji sering sinyal sentimen → default dibiarkan | 214 |
| `TRANSFORMER_LOWERCASE` | `False` | model uncased (indobertweet, indolem) sudah lowercase sendiri | 215 |
| `TRANSFORMER_USE_SLANG` | `False` | normalisasi slang juga untuk transformer (biasanya tidak perlu) | 216 |
| `CLASSICAL_LOWERCASE` | `True` |  | 217 |
| `USE_SLANG_NORMALIZATION` | `True` | "gak"→"tidak", "yg"→"yang", "bgt"→"banget" (untuk TF-IDF) | 218 |
| `CUSTOM_SLANG_DICT` | `{}` | tambahan kamus slang khusus lomba, mis. {"bgsd": "bagus"} | 219 |
| `REDUCE_ELONGATION` | `True` | "mantappp" → "mantap", "bagus2" → "bagus bagus" | 220 |
| `REMOVE_STOPWORDS_CLASSICAL` | `False` | stopword removal: kadang menurunkan skor (kata "tidak" penting!) | 221 |
| `USE_STEMMING` | `False` | butuh Sastrawi (opsional); lambat; jarang membantu untuk klasifikasi | 222 |
| `REMOVE_NUMBERS_CLASSICAL` | `False` |  | 223 |
| `RUN_EDA` | `True` |  | 228 |
| `EDA_TOKENIZER_MODEL` | `None` | None = tokenizer dari model transformer pertama; bisa nama HF lain | 229 |
| `MAX_LEN_PERCENTILE` | `95` | MAX_LEN rekomendasi = persentil panjang token ini | 230 |
| `TOP_NGRAMS` | `15` |  | 231 |
| `NEAR_DUP_THRESHOLD` | `0.90` | cosine similarity char-TF-IDF ≥ ini dianggap near-duplicate | 232 |
| `NEAR_DUP_MAX_ROWS` | `20000` | batasi baris untuk cek near-duplicate (biar tidak lama) | 233 |
| `RUN_ADVERSARIAL_VALIDATION` | `True` |  | 234 |
| `MAKE_WORDCLOUD` | `True` | hanya jalan kalau package wordcloud terpasang | 235 |
| **[A] CLASSICAL (TF-IDF)** | | | |
| `RUN_CLASSICAL` | `True` |  | 240 |
| `CLASSICAL_MODELS` | `["logreg", "linearsvc", "complement_nb", "sgd", "ridge", "nbsvm", "lgbm_svd"]` | pilihan: "logreg", "linearsvc", "complement_nb", "sgd", "ridge", "nbsvm", "lgbm_svd" | 242 |
| `TFIDF_WORD_NGRAM` | `(1, 2)` |  | 243 |
| `TFIDF_CHAR_NGRAM` | `(2, 5)` |  | 244 |
| `USE_CHAR_TFIDF` | `True` | char n-gram tahan typo/slang → sangat berguna untuk teks Indonesia sosmed | 245 |
| `TFIDF_WORD_MAX_FEATURES` | `200_000` |  | 246 |
| `TFIDF_CHAR_MAX_FEATURES` | `300_000` |  | 247 |
| `TFIDF_MIN_DF` | `2` |  | 248 |
| `TFIDF_MAX_DF` | `0.95` |  | 249 |
| `LOGREG_C` | `4.0` | makin besar = regularisasi makin lemah (coba 1, 4, 10) | 250 |
| `SVD_COMPONENTS` | `256` | untuk lgbm_svd | 251 |
| `LGBM_N_ESTIMATORS` | `2000` | dengan early stopping internal (inner holdout di fold train) | 252 |
| **[B] SENTENCE EMBEDDING (frozen) + model ringan** | | | |
| `RUN_EMBEDDINGS` | `True` |  | 257 |
| `EMBEDDING_MODELS` | `["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]` | Rekomendasi: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (cepat, multibahasa), "intfloat/multilingual-e5-small" / "-base" (kuat; WAJIB prefix "query: " → otomatis ditangani). | 260 |
| `EMBEDDING_PREFIXES` | `{}` | override prefix manual, mis. {"intfloat/multilingual-e5-base": "query: "} | 261 |
| `EMBEDDING_HEADS` | `["logreg", "lgbm"]` |  | 262 |
| `EMB_BATCH_SIZE` | `64` |  | 263 |
| `EMB_MAX_LEN` | `256` |  | 264 |
| `CACHE_EMBEDDINGS` | `True` | simpan ke disk → run berikutnya tidak perlu encode ulang | 265 |
| **[C] TRANSFORMER FINE-TUNING** | | | |
| `RUN_TRANSFORMER` | `"auto"` | "auto" = hanya kalau CUDA tersedia; True = paksa (CPU lambat!); False = skip | 270 |
| `TRAINER_BACKEND` | `"custom_loop"` | "custom_loop" (PyTorch murni, paling tahan beda versi) \| "hf_trainer" | 271 |
| `MODEL_POOL` | `{deberta_v3_base, deberta_v3_large, roberta_base, modernbert_base, modernbert_large, indobert_base, indobert_large, indolem_indobert, indobertweet, xlm_roberta_base, xlm_roberta_large, mdeberta_v3_base}` |  | 272 |
| `TRANSFORMER_MODELS` | `["auto"]` | ["auto"] = dipilih otomatis dari deteksi bahasa di EDA. Bisa key MODEL_POOL atau nama HF/path lokal. | 290 |
| `MAX_LEN` | `"auto"` | "auto" = rekomendasi EDA (persentil token), atau angka: 128 / 256 / 512 | 291 |
| `LONG_TEXT_STRATEGY` | `"head_tail"` | "truncate" \| "head_tail" (awal+akhir) \| "chunk" (potong & rata-rata) | 292 |
| `HEAD_RATIO` | `0.5` | head_tail: porsi token dari awal teks (sisanya dari akhir) | 293 |
| `CHUNK_OVERLAP` | `64` | chunk: token overlap antar potongan | 294 |
| `CHUNK_POOLING` | `"mean"` | chunk: gabung logits potongan dengan "mean" atau "max" | 295 |
| `MAX_CHUNKS_PER_DOC` | `8` |  | 296 |
| `BATCH_SIZE` | `16` |  | 297 |
| `EVAL_BATCH_SIZE` | `64` |  | 298 |
| `GRAD_ACCUM_STEPS` | `1` | batch efektif = BATCH_SIZE * GRAD_ACCUM_STEPS | 299 |
| `EPOCHS` | `3` |  | 300 |
| `LR` | `2e-5` | backbone LR (1e-5 large, 2e-5~3e-5 base) | 301 |
| `HEAD_LR` | `1e-4` | LR untuk classification head (lebih besar dari backbone) | 302 |
| `LLRD` | `1.0` | layer-wise LR decay: 1.0 = off; 0.9 = layer bawah LR lebih kecil | 303 |
| `WEIGHT_DECAY` | `0.01` |  | 304 |
| `WARMUP_RATIO` | `0.1` |  | 305 |
| `SCHEDULER` | `"linear"` | "linear" \| "cosine" | 306 |
| `MAX_GRAD_NORM` | `1.0` | gradient clipping | 307 |
| `USE_AMP` | `True` | mixed precision (hanya aktif jika CUDA). NaN loss? → set False | 308 |
| `AMP_DTYPE` | `"fp16"` | "fp16" \| "bf16" (bf16 hanya GPU Ampere+) | 309 |
| `GRADIENT_CHECKPOINTING` | `False` | hemat VRAM ~30-50%, training ~20-30% lebih lambat | 310 |
| `POOLING` | `"cls"` | "cls" \| "mean" (mean pooling sering lebih stabil) | 311 |
| `MULTI_SAMPLE_DROPOUT` | `0` | 0 = off; 5 = rata-rata 5 dropout head (stabilkan training) | 312 |
| `HEAD_DROPOUT` | `0.1` |  | 313 |
| `LOSS` | `"ce"` | "ce" \| "focal" (focal untuk imbalance berat) | 314 |
| `FOCAL_GAMMA` | `2.0` |  | 315 |
| `LABEL_SMOOTHING` | `0.0` | 0.05–0.1 bisa bantu kalau label noisy | 316 |
| `USE_CLASS_WEIGHTS` | `False` | True untuk imbalance (bobot balanced) | 317 |
| `EARLY_STOPPING_PATIENCE` | `2` | stop kalau skor valid tidak membaik N evaluasi berturut-turut | 318 |
| `EVALS_PER_EPOCH` | `1` | 2 = evaluasi tiap setengah epoch (untuk dataset besar) | 319 |
| `OOM_MAX_RETRIES` | `3` | CUDA OOM → ulang dengan batch size setengah | 320 |
| `SAVE_FOLD_MODELS` | `False` | simpan bobot tiap fold (.pt) — besar! aktifkan kalau perlu inference ulang | 321 |
| `FOLDS_TO_RUN` | `None` | None = semua fold; [0] = hanya fold 0 (eksperimen cepat) | 322 |
| `NUM_WORKERS` | `0` | DataLoader workers (Windows: 0 paling aman) | 323 |
| `DEBUG_SIMULATE_OOM` | `False` | hanya untuk testing alur OOM-retry | 324 |
| `RUN_PSEUDO_LABEL` | `False` |  | 329 |
| `PSEUDO_CONF_THRESHOLD` | `0.95` | ambil prediksi test dengan confidence ≥ ini | 330 |
| `PSEUDO_MAX_FRACTION` | `0.5` | maksimal pseudo rows = fraksi ini × jumlah train | 331 |
| `RUN_ENSEMBLE` | `True` |  | 336 |
| `ENSEMBLE_METHODS` | `["mean", "weighted", "rank", "stacking"]` |  | 337 |
| `ENSEMBLE_TOP_K` | `6` | hanya top-K model terbaik (OOF) yang ikut ensemble | 338 |
| `MANUAL_WEIGHTS` | `{}` | mis. {"tfidf_logreg": 0.3, "tf_indobert_base": 0.7} | 339 |
| `FINAL_PREDICTION` | `"auto"` | "auto" = skor OOF terbaik; atau nama model/ensemble tertentu | 340 |
| `TUNE_THRESHOLD` | `True` | binary: threshold; multiclass: bias per kelas; multilabel: threshold per label | 341 |
| `MIN_TUNING_GAIN` | `0.002` | tuning hanya dipakai kalau gain OOF ≥ ini (hindari overfit) | 342 |
| `RUN_INTERPRETABILITY` | `True` |  | 343 |
| `N_EXPLAIN_EXAMPLES` | `3` |  | 344 |
| `USE_SHAP` | `True` | hanya jika package shap ada | 345 |
| `FIG_DPI` | `200` | 300 untuk figure final laporan | 346 |
| `SHOW_FIGS` | `True` |  | 347 |
