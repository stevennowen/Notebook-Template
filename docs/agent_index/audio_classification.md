# audio_classification — `--task audio`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Audio → kelas (suara hewan, ucapan, mesin) + open-set 'Unknown'

| | |
|---|---|
| Baca | `08_Audio/audio_classification.py` (754 baris) — notebook kembarannya: `08_Audio/audio_classification.ipynb` |
| `class CFG` | baris 55 · cell `# >>> USER OVERRIDE` baris 99 |
| Data yang diharapkan | data/train.csv (file, label) + data/train/ (wav/flac/mp3) atau data/train/<kelas>/, data/test/ |
| Contoh `task_config.json` | `{"TRAIN_CSV": "./data/train.csv", "TRAIN_AUDIO_DIR": "./data/train", "TEST_AUDIO_DIR": "./data/test", "SAMPLE_SUB_PATH": "./data/sample_submission.csv"}` |
| Folder output | `./outputs/audio_classification/` |
| Submission ditulis | baris 714 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Audio Classification Template (+ open-set "Unknown") | 27 |
| ↳ Pipeline | 32 |
| 1. Install & import | 143 |
| 2. Baca audio | 251 |
| 3. Data & label | 312 |
| 4. EDA | 368 |
| 5. Fitur: log-mel spectrogram & MFCC (numpy) | 388 |
| 6. Cross-validation folds | 471 |
| 7. Model A — fitur statistik + LightGBM / Logistic Regression | 496 |
| 8. Model B — CNN pada log-mel | 524 |
| 9. Ensemble, evaluasi, open-set | 643 |
| 10. Submission + ringkasan | 688 |
| 11. Troubleshooting, recipes, checklist | 734 |

## Semua key CFG (41)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `RUN_MODE` | `"full"` |  | 56 |
| `SEED` | `42` |  | 57 |
| `NOTEBOOK_NAME` | `"audio_classification"` |  | 58 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` |  | 59 |
| `TRAIN_CSV` | `"./data/train.csv"` | kolom file + label; tidak ada → folder per kelas di TRAIN_AUDIO_DIR | 60 |
| `TRAIN_AUDIO_DIR` | `"./data/train"` |  | 61 |
| `TEST_CSV` | `None` | opsional (daftar file test); None = semua file di TEST_AUDIO_DIR | 62 |
| `TEST_AUDIO_DIR` | `"./data/test"` |  | 63 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 64 |
| `FILE_COL` | `None` | None = auto | 65 |
| `LABEL_COL` | `None` | None = auto | 66 |
| `GROUP_COL` | `None` | sumber rekaman / pembicara (GroupKFold) | 67 |
| `AUDIO_EXTS` | `(".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif")` |  | 68 |
| `OUTPUT_DIR` | `"./outputs/audio_classification/"` |  | 69 |
| `SR` | `22050` |  | 70 |
| `DURATION` | `None` | detik per klip/jendela; None = median durasi (dibatasi 1–10 s) | 71 |
| `N_FFT` | `1024` |  | 72 |
| `HOP` | `256` |  | 73 |
| `N_MELS` | `96` |  | 74 |
| `FMIN` | `20` |  | 75 |
| `FMAX` | `None` | None = SR/2 | 76 |
| `N_MFCC` | `20` |  | 77 |
| `METRIC` | `"f1_macro"` | "f1_macro" \| "accuracy" \| "roc_auc" (biner) | 78 |
| `MODELS` | `["feat_lgb", "feat_logreg", "cnn"]` |  | 79 |
| `N_FOLDS` | `5` |  | 80 |
| `EPOCHS` | `30` |  | 81 |
| `BATCH_SIZE` | `32` |  | 82 |
| `LR` | `1e-3` |  | 83 |
| `MIXUP` | `0.2` |  | 84 |
| `SPEC_AUG` | `True` |  | 85 |
| `USE_GPU` | `True` |  | 86 |
| `USE_AMP` | `True` |  | 87 |
| `OPEN_SET` | `False` | True = beri label UNKNOWN_LABEL bila keyakinan rendah | 88 |
| `UNKNOWN_LABEL` | `"Unknown"` |  | 89 |
| `OPEN_SET_KEEP` | `0.95` | threshold = kuantil keyakinan sehingga 95% prediksi benar (OOF) tetap diterima | 90 |
| `SUBMISSION_MODE` | `"label"` | "label" \| "proba" | 91 |
| `NUM_WORKERS` | `0` |  | 92 |
| `FAST_SAMPLE` | `300` |  | 93 |
| `FAST_EPOCHS` | `6` |  | 94 |
| `FAST_FOLDS` | `2` |  | 95 |
| `FIG_DPI` | `150` |  | 96 |
