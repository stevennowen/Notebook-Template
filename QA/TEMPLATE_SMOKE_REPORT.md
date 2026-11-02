# Smoke test template — 16/16 PASS

Tanggal: 2026-11-02 · format: **py** · Python 3.11.9 · pip install: TIDAK (library yang hilang harus ditangani fallback)

Setiap task: data sintetis kecil (`tools/make_synthetic_task_data.py`) → `tools/run_task.py --set RUN_MODE=fast`.
PASS = selesai tanpa crash (exit code 0) **dan** file hasil utama (submission/prediksi/laporan) terbentuk.
Kolom *peringatan log* = baris log berisi traceback/❌/gagal yang ditangani template (bukan crash) — cek bila > 0.

| task | template | status | detik | figure | file output | hasil utama | peringatan log |
|---|---|---|---|---|---|---|---|
| tabular | `02_Tabular/tabular_master.ipynb` | PASS | 16 | 30 | 69 | outputs\tabular_master\figures\fig21_oof_vs_test_predictions.png<br>outputs\tabular_master\submissions\submission.csv<br>outputs\tabular_master\submissions\submission_best_single_linear.csv | 1 |
| tabular-multilabel | `02_Tabular/tabular_multilabel.ipynb` | PASS | 8 | 2 | 8 | outputs\tabular_multilabel\submissions\submission.csv | 0 |
| nlp-classification | `03_NLP/nlp_classification.ipynb` | PASS | 27 | 12 | 59 | outputs\nlp_classification\submissions\submission.csv<br>outputs\nlp_classification\submissions\submission_tfidf_logreg_f1_macro1.0000_1102_2206.csv<br>outputs\nlp_classification\tables\classification_report.csv | 0 |
| nlp-regression | `03_NLP/nlp_regression.ipynb` | PASS | 22 | 12 | 56 | outputs\nlp_regression\submissions\submission.csv<br>outputs\nlp_regression\submissions\submission_ens_stacking_rmse0.4326_1102_2207.csv | 2 |
| cv-classification | `04_Computer_Vision/cv_image_classification.ipynb` | PASS | 36 | 11 | 35 | outputs\cv_image_classification\submissions\submission.csv<br>outputs\cv_image_classification\submissions\submission_feat_logreg_f1_macro1.0000_20261102_2207.csv<br>outputs\cv_image_classification\tables\classification_report.csv | 0 |
| time-series | `05_Time_Series/time_series_forecasting.ipynb` | PASS | 14 | 12 | 25 | outputs\time_series_forecasting\submissions\submission.csv<br>outputs\time_series_forecasting\tables\backtest_predictions.csv | 0 |
| clustering | `06_Clustering_Segmentation/clustering_segmentation.ipynb` | PASS | 14 | 18 | 33 | outputs\clustering_segmentation\tables\data_quality_report.csv | 0 |
| segmentation | `04_Computer_Vision/cv_segmentation.ipynb` | PASS | 33 | 18 | 28 | outputs\cv_segmentation\figures\fig04_oof_worst_predictions.png<br>outputs\cv_segmentation\figures\fig06_test_predictions.png<br>outputs\cv_segmentation\submissions\submission_masks.zip | 0 |
| detection | `04_Computer_Vision/cv_object_detection.ipynb` | PASS | 18 | 13 | 131 | outputs\cv_object_detection\figures\fig06_test_predictions.png<br>outputs\cv_object_detection\submissions\predictions_coco.json<br>outputs\cv_object_detection\submissions\submission.csv | 0 |
| ocr | `04_Computer_Vision/cv_ocr_text_recognition.ipynb` | PASS | 20 | 4 | 12 | outputs\cv_ocr_text_recognition\submissions\submission.csv<br>outputs\cv_ocr_text_recognition\tables\oof_predictions.csv | 1 |
| anomaly-tabular | `07_Anomaly_Detection/anomaly_tabular.ipynb` | PASS | 10 | 3 | 11 | outputs\anomaly_tabular\submissions\submission.csv<br>outputs\anomaly_tabular\tables\train_anomaly_scores.csv | 0 |
| anomaly-image | `07_Anomaly_Detection/anomaly_image.ipynb` | PASS | 11 | 3 | 8 | outputs\anomaly_image\submissions\submission.csv | 0 |
| audio | `08_Audio/audio_classification.ipynb` | PASS | 16 | 3 | 10 | outputs\audio_classification\submissions\submission.csv | 0 |
| ner | `03_NLP/nlp_token_classification.ipynb` | PASS | 12 | 2 | 8 | outputs\nlp_token_classification\submissions\submission.conll | 0 |
| llm | `03_NLP/nlp_llm_zero_shot.ipynb` | PASS | 29 | 1 | 6 | outputs\nlp_llm_zero_shot\submissions\submission.csv | 0 |
| stats | `09_Statistical_Analysis/statistical_analysis_report.ipynb` | PASS | 5 | 7 | 18 | outputs\statistical_analysis_report\REPORT.docx<br>outputs\statistical_analysis_report\REPORT.md | 0 |

## Catatan uji
- **nlp-classification**: Transformer & embedding memakai model yang sudah ada di cache (offline).
- **nlp-regression**: Transformer & embedding memakai model yang sudah ada di cache (offline).
- **segmentation**: PRETRAINED=False (tanpa download bobot) → membuktikan alur, bukan akurasi terbaik.
- **detection**: YOLO dari nol (PRETRAINED=False, 3 epoch) → mAP rendah wajar; alur & format submission yang diuji.
- **anomaly-image**: Backbone berbobot acak (PRETRAINED=False).
- **ner**: Transformer memakai model yang sudah ada di cache (offline).
- **llm**: Kalau SMOKE_LLM_MODEL menunjuk LLM mini berbobot acak: yang diuji mekanik (prompt, skor label, kalibrasi, probe, submission), bukan kualitas jawaban LLM.
- Library tidak terpasang saat uji (jalur ini hanya teruji lewat fallback-nya): xgboost, catboost, optuna, shap, statsmodels, lifelines, timm, segmentation_models_pytorch, librosa, soundfile, peft, nbclient.

### tabular — PASS
Peringatan pertama di log:
```text
id  tenure_months  complaints signup_date      region  age channel      income      x5  y_true    pred   error
```

### nlp-regression — PASS
Peringatan pertama di log:
```text
Error per panjang teks:
Worst predictions (abs error terbesar):
```

### ocr — PASS
Peringatan pertama di log:
```text
- Out-of-fold exact match **0.0000**, character error rate **0.8963** (beam decoding).
```
