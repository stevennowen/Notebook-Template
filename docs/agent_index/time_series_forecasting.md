# time_series_forecasting — `--task time-series`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Forecasting deret waktu (single series & panel)

| | |
|---|---|
| Baca | `05_Time_Series/time_series_forecasting.py` (3,028 baris) — notebook kembarannya: `05_Time_Series/time_series_forecasting.ipynb` |
| `class CFG` | baris 76 · cell `# >>> USER OVERRIDE` baris 213 |
| Data yang diharapkan | data/train.csv (tanggal, [id seri], target), data/test.csv opsional |
| Contoh `task_config.json` | `{"TRAIN_PATH": "./data/train.csv", "TEST_PATH": "./data/test.csv", "SAMPLE_SUB_PATH": "./data/sample_submission.csv", "DATE_COL": null, "TARGET_COL": "sales", "SERIES_ID_COLS": [], "HORIZON": null}` |
| Folder output | `./outputs/time_series_forecasting/` |
| Submission ditulis | baris 2874 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Time Series Forecasting — Competition Template (Single Series & Panel) | 27 |
| ↳ 0. Konfigurasi (CFG) | 59 |
| ↳ 1. Install & Import | 257 |
| ↳ 2. Helper functions | 478 |
| ↳ 3. Load data | 851 |
| ↳ 4. Data preparation | 967 |
| ↳ 5. Exploratory Data Analysis (EDA) untuk time series | 1284 |
| ↳ 6. Desain validasi: time-based backtesting | 1619 |
| ↳ 7. Metrik evaluasi — mana yang dipakai? | 1690 |
| ↳ 8. Model | 1733 |
| ↳ 9. Jalankan backtest semua model + leaderboard | 2201 |
| ↳ 10. Ensemble forecast | 2364 |
| ↳ 11. Plot forecast vs actual per backtest window & diagnostik residual | 2439 |
| ↳ 12. Final fit (seluruh data) + forecast masa depan | 2560 |
| ↳ 13. Feature importance & SHAP (model ML) | 2620 |
| ↳ 14. Prediction interval | 2694 |
| ↳ 15. Submission builder + validator | 2768 |
| ↳ 16. Ringkasan untuk laporan | 2880 |
| ↳ 17. Debugging & sanity checks | 2914 |
| ↳ 18. Troubleshooting | 2949 |
| ↳ 19. Experiment recipes (salin ke cell USER OVERRIDE) | 2980 |
| ↳ 20. Checklist sebelum submit | 3015 |

## Semua key CFG (77)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| **MODE** | | | |
| `RUN_MODE` | `"full"` | "fast" = debug cepat (window backtest lebih sedikit, model lebih sedikit, iterasi kecil). "full" = run final untuk submission. Selalu jalankan "fast" dulu sampai tidak ada error. | 80 |
| `SEED` | `42` |  | 81 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` | Install otomatis library yang belum ada (pip). Set False kalau tidak ada internet / tidak boleh install. | 83 |
| `INSTALL_OPTIONAL` | `False` | Library opsional berat (prophet, statsforecast). Default False karena install-nya lama/berat. (env TEMPLATE_AUTO_INSTALL=0 → off) | 85 |
| **PATHS** | | | |
| `NOTEBOOK_NAME` | `"time_series_forecasting"` |  | 88 |
| `TRAIN_PATH` | `"./data/train.csv"` |  | 89 |
| `TEST_PATH` | `"./data/test.csv"` | boleh tidak ada -> forecast H langkah ke depan | 90 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 91 |
| `OUTPUT_DIR` | `None` | None -> ./outputs/<NOTEBOOK_NAME>/ | 92 |
| **KOLOM** | | | |
| `DATE_COL` | `None` | None = auto detect (kolom bernama date/ds/tanggal/time/... atau yang bisa di-parse jadi tanggal) | 95 |
| `TARGET_COL` | `"sales"` | kolom yang diprediksi | 96 |
| `SERIES_ID_COLS` | `[]` | [] = single series. Panel: mis. ["store"] atau ["store", "item"] | 97 |
| `ID_COL` | `"id"` | kolom ID baris di test/sample_submission (None kalau tidak ada) | 98 |
| `EXOG_COLS` | `[]` | Exogenous: kolom tambahan yang NILAINYA DIKETAHUI DI MASA DEPAN (ada di test), mis. promo, harga, libur. [] = tidak pakai. "auto" = semua kolom non-target yang ada di train DAN test. | 101 |
| `DATE_FORMAT` | `None` | mis. "%d/%m/%Y" kalau parsing otomatis salah. None = auto. | 102 |
| `DAYFIRST` | `False` | True kalau format tanggal Indonesia 31/12/2024 dan DATE_FORMAT None. | 103 |
| **FREKUENSI & HORIZON** | | | |
| `FREQ` | `None` | None = auto infer. Contoh manual: "D", "W-MON", "MS", "h", "B". | 106 |
| `RESAMPLE_RULE` | `None` | Ubah level agregasi, mis. data harian -> mingguan "W-MON" / bulanan "MS". None = tidak. | 107 |
| `RESAMPLE_AGG` | `"sum"` | "sum" (penjualan) \| "mean" (suhu/harga) \| "last" (stok/saldo) | 108 |
| `HORIZON` | `None` | None = dari file test, atau default per frekuensi (D:30, W:12, M:12, h:48) | 109 |
| `SEASON_LENGTH` | `None` | None = auto (D:7, W:52, M:12, h:24, B:5, Q:4) | 110 |
| **CLEANING** | | | |
| `DUPLICATE_AGG` | `"sum"` | timestamp duplikat per series diagregasi: "sum" \| "mean" \| "last" \| "max" | 113 |
| `FILL_METHOD` | `"zero"` | timestamp hilang: "zero" \| "ffill" \| "interpolate" \| "seasonal" \| "mean" \| "none" | 114 |
| `REINDEX_TO_GLOBAL_END` | `True` | True: semua series di-reindex sampai tanggal terakhir global (series yang berhenti diisi FILL_METHOD) | 115 |
| `NEGATIVE_POLICY` | `"keep"` | nilai target negatif: "keep" \| "clip0" (jadi 0) \| "nan" (dianggap missing lalu diisi) | 116 |
| `OUTLIER_METHOD` | `None` | None \| "iqr" \| "zscore" \| "hampel" (rolling median, paling aman untuk TS) | 117 |
| `OUTLIER_THRESHOLD` | `3.0` | iqr: k*IQR (3 = konservatif), zscore/hampel: jumlah std/MAD | 118 |
| `OUTLIER_ACTION` | `"clip"` | "clip" (potong ke batas) \| "interpolate" (ganti nilai interpolasi) | 119 |
| `PANEL_AGG_FOR_EDA` | `"sum"` | cara menggabung panel jadi 1 deret untuk EDA: "sum" \| "mean" | 120 |
| **VALIDASI (BACKTEST)** | | | |
| `N_BACKTEST_WINDOWS` | `3` | jumlah cutoff. 3-5 umum. Lebih banyak = estimasi lebih stabil tapi lebih lama. | 123 |
| `BACKTEST_STEP` | `None` | jarak antar cutoff (langkah). None = HORIZON (window tidak overlap). | 124 |
| `WINDOW_TYPE` | `"expanding"` | "expanding" (train makin panjang) \| "rolling" (panjang train tetap) | 125 |
| `ROLLING_TRAIN_SIZE` | `None` | dipakai kalau WINDOW_TYPE="rolling", mis. 365 (langkah). None = 3*HORIZON + 2*season. | 126 |
| `GAP` | `None` | None = auto (dari jarak akhir train ke awal test). Int = jumlah langkah jeda manual. | 127 |
| `EVAL_EXCLUDE_IMPUTED` | `True` | True: titik yang hasil imputasi (timestamp hilang) tidak dihitung di metrik | 128 |
| **METRIK** | | | |
| `METRIC` | `"auto"` | "mae" \| "rmse" \| "mape" \| "smape" \| "wape" \| "mase" \| "rmsle" \| "auto" (=wape). SAMAKAN dengan metrik lomba! | 132 |
| **MODEL** | | | |
| `MODELS` | `["naive", "snaive", "ma", "drift", "ses", "holt_winters", "ets", "theta", "sarimax", "lgb", "xgb", "cat", "hgb", "prophet", "sf_autoets", "sf_autoarima"]` | Baseline: naive, snaive, ma, drift Statistik (per series, butuh statsmodels): ses, holt_winters, ets, theta, sarimax ML global (satu model untuk semua series): lgb, xgb, cat, hgb Opsional (kalau terinstall): prophet, sf_ | 139 |
| `FAST_MODELS` | `["naive", "snaive", "ma", "drift", "holt_winters", "ets", "theta", "sarimax", "lgb", "hgb"]` |  | 141 |
| `MA_WINDOW` | `None` | None = season length | 142 |
| `HW_SEASONAL` | `"add"` | Holt-Winters seasonal "add" \| "mul" (mul hanya kalau data > 0) | 143 |
| `STAT_MAX_SERIES` | `200` | model statistik dijalankan maks untuk N series terbesar; sisanya pakai snaive | 144 |
| `STAT_MAX_TRAIN` | `1500` | model statistik hanya pakai N titik terakhir (lebih cepat) | 145 |
| `SARIMA_MAX_P` | `2` | grid p in [0..MAX_P] | 146 |
| `SARIMA_MAX_Q` | `2` | grid q in [0..MAX_Q] | 147 |
| `SARIMA_MAX_PS` | `1` | grid P seasonal | 148 |
| `SARIMA_MAX_QS` | `1` | grid Q seasonal | 149 |
| `SARIMA_MAX_SEASON` | `24` | season length > ini -> SARIMA tanpa komponen seasonal (terlalu lambat) | 150 |
| `SARIMA_MAX_TRAIN` | `500` | SARIMA hanya pakai N titik terakhir | 151 |
| `SARIMA_TIME_LIMIT` | `30` | detik per series untuk grid search (guard supaya tidak hang) | 152 |
| **ML global** | | | |
| `ML_STRATEGY` | `"recursive"` | "recursive" (1 model, prediksi dipakai sbg lag) \| "direct" (model per bucket horizon) | 155 |
| `DIRECT_N_BUCKETS` | `None` | direct: jumlah model. None = min(H, 6). = H -> direct murni (paling akurat, paling lama) | 156 |
| `LAGS` | `None` | None = auto dari season length, mis. daily [1..7,14,21,28] | 157 |
| `ROLL_WINDOWS` | `None` | None = auto, mis. daily [7,14,28] | 158 |
| `USE_EWM` | `True` | fitur exponential weighted mean | 159 |
| `USE_DATE_FEATURES` | `True` |  | 160 |
| `USE_HOLIDAY_FEATURES` | `True` | libur nasional Indonesia + jarak ke Lebaran + Ramadan | 161 |
| `USE_SERIES_STATS` | `True` | mean/std per series (dari data train window saja) sebagai fitur level | 162 |
| `TARGET_TRANSFORM` | `"auto"` | "none" \| "log1p" \| "diff" \| "log1p_diff" \| "auto" (log1p kalau >=0 & skewed) | 163 |
| `ML_N_ESTIMATORS` | `800` |  | 164 |
| `ML_LEARNING_RATE` | `0.03` |  | 165 |
| `ML_NUM_LEAVES` | `31` |  | 166 |
| `ML_MAX_TRAIN_ROWS` | `2_000_000` | kalau lebih, ambil baris paling baru | 167 |
| `ML_N_JOBS` | `-1` |  | 168 |
| **HOLIDAYS** | | | |
| `COUNTRY_HOLIDAYS` | `"ID"` | "ID" = tabel libur Indonesia bawaan. None = tidak pakai | 171 |
| `CUSTOM_HOLIDAYS` | `{}` | tambahan: {"2024-02-14": "Pemilu", "2024-11-27": "Pilkada"} | 172 |
| **ENSEMBLE & FINAL** | | | |
| `ENSEMBLE_TOP_K` | `3` | gabungkan K model terbaik | 175 |
| `FINAL_STRATEGY` | `"auto"` | "auto" (ensemble kalau lebih baik) \| "best" \| "ensemble" \| nama model, mis. "lgb" | 176 |
| **PREDICTION INTERVAL** | | | |
| `PI_LEVEL` | `0.90` | 0.8 / 0.9 / 0.95 | 179 |
| `PI_METHOD` | `"both"` | "residual" (dari error backtest) \| "quantile_lgb" \| "both" \| None | 180 |
| **SUBMISSION** | | | |
| `SUBMISSION_PRED_COL` | `None` | None = ambil dari sample_submission (kolom non-ID) atau TARGET_COL | 183 |
| `SUBMISSION_FILE` | `"submission.csv"` |  | 184 |
| `CLIP_NEGATIVE_PRED` | `"auto"` | "auto" = clip ke 0 kalau target historis tidak pernah negatif | 185 |
| `ROUND_PRED` | `False` | True kalau target harus integer (jumlah unit) | 186 |
| **FIGURES / EXPLAIN** | | | |
| `FIG_DPI` | `200` | 300 untuk laporan | 189 |
| `MAX_SERIES_PLOT` | `6` |  | 190 |
| `USE_SHAP` | `True` |  | 191 |
| `SHAP_SAMPLE` | `2000` |  | 192 |
