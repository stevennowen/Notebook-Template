# clustering_segmentation — `--task clustering`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Clustering / segmentasi customer tanpa label (+ RFM)

| | |
|---|---|
| Baca | `06_Clustering_Segmentation/clustering_segmentation.py` (2,158 baris) — notebook kembarannya: `06_Clustering_Segmentation/clustering_segmentation.ipynb` |
| `class CFG` | baris 71 · cell `# >>> USER OVERRIDE` baris 174 |
| Data yang diharapkan | data/train.csv |
| Contoh `task_config.json` | `{"DATA_PATH": "./data/train.csv", "ID_COL": "id"}` |
| Folder output | `./outputs/clustering_segmentation/` |
| Submission ditulis | baris 2040 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Clustering & Customer Segmentation — Competition Template | 27 |
| ↳ 0. Konfigurasi (CFG) | 57 |
| ↳ 1. Install & Import | 217 |
| ↳ 2. Helper functions | 402 |
| ↳ 3. Load data (+ RFM builder opsional) | 626 |
| ↳ 4. Data quality report | 720 |
| ↳ 5. Pemilihan fitur | 805 |
| ↳ 6. EDA singkat untuk clustering | 854 |
| ↳ 7. Preprocessing | 899 |
| ↳ 8. Reduksi dimensi: PCA, t-SNE, UMAP | 1022 |
| ↳ 9. Memilih jumlah cluster K | 1093 |
| ↳ 10. Algoritma clustering & perbandingan | 1221 |
| ↳ 11. Stabilitas cluster (Adjusted Rand Index) | 1384 |
| ↳ 12. Label cluster final | 1464 |
| ↳ 13. Profiling cluster (inti untuk bisnis & laporan) | 1521 |
| ↳ 14. Profil teks otomatis & saran nama segmen | 1699 |
| ↳ 15. Analisis outcome per cluster & rekomendasi aksi bisnis | 1772 |
| ↳ 16. Menjelaskan cluster: surrogate decision tree, permutation importance, SHAP | 1888 |
| ↳ 17. Export label, profil, model + assign data baru | 1976 |
| ↳ 18. Ringkasan untuk laporan | 2053 |
| ↳ 19. Debugging & sanity checks | 2077 |
| ↳ 20. Troubleshooting | 2098 |
| ↳ 21. Experiment recipes (salin ke USER OVERRIDE) | 2125 |
| ↳ 22. Checklist sebelum laporan / submit | 2147 |

## Semua key CFG (64)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| **MODE** | | | |
| `RUN_MODE` | `"full"` | "fast" (debug: sampel kecil, bootstrap sedikit) \| "full" | 73 |
| `SEED` | `42` |  | 74 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` | install package yang BELUM ada saja (tidak upgrade) (env TEMPLATE_AUTO_INSTALL=0 → off) | 75 |
| `INSTALL_OPTIONAL` | `False` | umap-learn, kmodes (opsional) | 76 |
| **PATHS** | | | |
| `NOTEBOOK_NAME` | `"clustering_segmentation"` |  | 79 |
| `DATA_PATH` | `"./data/train.csv"` | tabel utama (1 baris = 1 entitas/pelanggan) | 80 |
| `TEST_PATH` | `"./data/test.csv"` | opsional: data baru yang ingin di-assign ke cluster (boleh tidak ada) | 81 |
| `OUTPUT_DIR` | `None` | None -> ./outputs/<NOTEBOOK_NAME>/ | 82 |
| **KOLOM** | | | |
| `ID_COL` | `"id"` | None kalau tidak ada; tidak dipakai sebagai fitur | 85 |
| `FEATURE_COLS` | `None` | None = otomatis (numerik + kategorikal kardinalitas rendah). Atau list kolom. | 86 |
| `EXCLUDE_COLS` | `[]` | kolom yang pasti dibuang dari fitur (mis. nama, alamat, tanggal mentah) | 87 |
| `PROFILE_ONLY_COLS` | `[]` | kolom yang TIDAK dipakai clustering tapi ikut di profil (mis. "gender", "channel") | 88 |
| `OUTCOME_COL` | `None` | mis. "churn" / "claim_flag" -> dianalisis setelah clustering (BUKAN fitur) | 89 |
| `OUTCOME_POSITIVE` | `None` | nilai positif outcome (mis. "Yes" / 1). None = auto | 90 |
| `VALUE_COL` | `None` | mis. "premium" / "monetary" -> untuk matriks risk x value | 91 |
| `MAX_CAT_CARDINALITY` | `30` | kategorikal dengan unique > ini dibuang otomatis (mis. nama, kode pos) | 92 |
| `RARE_MIN_FREQ` | `0.01` | kategori dengan frekuensi < 1% digabung jadi "Other" | 93 |
| `HIGH_MISSING_DROP` | `0.6` | kolom dengan missing > 60% dibuang dari fitur | 94 |
| **RFM (opsional, dari tabel transaksi)** | | | |
| `USE_RFM` | `False` |  | 97 |
| `TRANSACTIONS_PATH` | `"./data/transactions.csv"` |  | 98 |
| `TX_CUSTOMER_COL` | `"customer_id"` |  | 99 |
| `TX_DATE_COL` | `"date"` |  | 100 |
| `TX_AMOUNT_COL` | `"amount"` |  | 101 |
| `TX_INVOICE_COL` | `None` | kalau ada nomor invoice: frequency = jumlah invoice unik | 102 |
| `RFM_SNAPSHOT_DATE` | `None` | None = tanggal transaksi terakhir + 1 hari | 103 |
| `RFM_MERGE` | `"auto"` | "auto": merge ke DATA_PATH kalau ada, kalau tidak pakai tabel RFM saja \| "replace" \| "merge" | 104 |
| **PREPROCESSING** | | | |
| `NUM_IMPUTE` | `"median"` | "median" \| "mean" | 107 |
| `LOG_SKEWED` | `True` | log1p untuk fitur skewed (\|skew\| > SKEW_THRESHOLD) | 108 |
| `SKEW_THRESHOLD` | `1.0` |  | 109 |
| `CLIP_QUANTILES` | `(0.01, 0.99)` | winsorize outlier; None = tidak | 110 |
| `SCALER` | `"standard"` | "standard" \| "robust" \| "minmax" \| "quantile" | 111 |
| `CAT_ENCODING` | `"onehot"` | "onehot" \| "none" (kategorikal hanya untuk profil) | 112 |
| `CAT_WEIGHT` | `0.5` | bobot kolom one-hot (0.5 ~ setara gamma K-Prototypes). Naikkan kalau kategori harus lebih menentukan | 113 |
| `DROP_CORRELATED` | `0.95` | buang salah satu fitur numerik dengan \|corr\| > ini. None = tidak | 114 |
| **REDUKSI DIMENSI** | | | |
| `USE_PCA_FOR_CLUSTERING` | `False` | True: cluster di ruang PCA (bagus kalau fitur banyak & berkorelasi) | 117 |
| `PCA_VARIANCE` | `0.90` | jumlah komponen = yang menjelaskan 90% varians | 118 |
| `RUN_TSNE` | `True` |  | 119 |
| `TSNE_SAMPLE` | `3000` |  | 120 |
| `RUN_UMAP` | `True` | hanya jika umap-learn asli terinstall | 121 |
| **MEMILIH K** | | | |
| `K_MIN` | `2` |  | 124 |
| `K_MAX` | `10` |  | 125 |
| `METRIC_SAMPLE` | `5000` | sampel untuk silhouette / gap (hemat waktu) | 126 |
| `RUN_GAP` | `True` | gap statistic (lebih lambat) | 127 |
| `GAP_B` | `5` | jumlah dataset referensi gap | 128 |
| `MIN_CLUSTER_PCT` | `0.03` | cluster < 3% data dianggap terlalu kecil untuk aksi bisnis | 129 |
| `N_CLUSTERS` | `None` | None = pakai rekomendasi otomatis. Isi angka untuk memaksa K. | 130 |
| **ALGORITMA** | | | |
| `ALGORITHMS` | `["kmeans", "minibatch", "gmm", "agglomerative", "dbscan", "hdbscan", "kprototypes", "kmedoids_gower", "agglo_gower"]` |  | 133 |
| `FINAL_ALGO` | `"auto"` | "auto" \| salah satu nama di atas | 135 |
| `KMEANS_N_INIT` | `10` | k-means++ dengan 10 inisialisasi; ambil inertia terbaik | 136 |
| `AGGLO_MAX_N` | `8000` | Agglomerative O(n^2): di atas ini fit di sampel lalu assign nearest centroid | 137 |
| `GOWER_MAX_N` | `3000` | matriks Gower n x n: di atas ini pakai sampel | 138 |
| `DBSCAN_EPS` | `None` | None = otomatis dari knee k-distance plot | 139 |
| `DBSCAN_MIN_SAMPLES` | `None` | None = max(5, 2 x jumlah dimensi) dibatasi 50 | 140 |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `None` | None = max(10, 2% n) | 141 |
| **STABILITAS** | | | |
| `STABILITY_BOOTSTRAPS` | `20` |  | 144 |
| `STABILITY_SEEDS` | `10` |  | 145 |
| **PROFILING & BISNIS** | | | |
| `SORT_CLUSTERS_BY` | `"size"` | "size" (cluster 0 = terbesar) \| "outcome" (cluster 0 = outcome rate tertinggi) | 148 |
| `PROFILE_LANG` | `"id"` | "id" \| "en" untuk teks profil otomatis | 149 |
| `TOP_FEATURES_PROFILE` | `4` | jumlah fitur pembeda yang disebut di teks profil | 150 |
| `MAX_PLOT_FEATURES` | `12` |  | 151 |
| **PENJELASAN** | | | |
| `SURROGATE_MAX_DEPTH` | `4` |  | 154 |
| `USE_SHAP` | `True` |  | 155 |
| `SHAP_SAMPLE` | `1000` |  | 156 |
| `FIG_DPI` | `200` | 300 untuk laporan | 158 |
