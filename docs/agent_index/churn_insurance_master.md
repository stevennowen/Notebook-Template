# churn_insurance_master — `--task churn`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Churn / retensi / asuransi + artikel, dashboard, ZIP otomatis

> Pakai tools/new_case.py (alur lengkap hari-H) — bukan new_task.py.

| | |
|---|---|
| Baca | `01_Churn_Insurance/churn_insurance_master.py` (10,022 baris) — notebook kembarannya: `01_Churn_Insurance/churn_insurance_master.ipynb` |
| `class CFG` | baris 685 · cell `# >>> USER OVERRIDE` baris 864 |
| Data yang diharapkan | - |
| Contoh `task_config.json` | `-` |
| Folder output | `./outputs/churn_insurance` |
| Submission ditulis | baris 6729 |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| 🏆 Churn & Insurance Analytics — Master Notebook | 27 |
| ↳ Prinsip desain | 45 |
| ↳ Cara pakai cepat (5 langkah) | 52 |
| ↳ Strategi waktu saat final (saran) | 59 |
| ↳ 📑 Daftar Isi | 70 |
| 0. Setup | 99 |
| ↳ 0.1 Install package yang belum ada | 101 |
| ↳ 0.2 Import & availability flags | 164 |
| ↳ 0.3 Helper functions umum | 289 |
| ↳ 0.4 Advanced helpers — DeLong, EMPC, calibration, causal sensitivity, power analysis, expe | 463 |
| 1. Config | 666 |
| 2. Load Data | 965 |
| ↳ 2.1 Auto-adapt: ukuran data & spesifikasi komputer | 1153 |
| 3. Data Quality & Cleaning | 1277 |
| ↳ 3.1 Deteksi target & ID | 1279 |
| ↳ 3.2 Parsing label churn → 0/1 | 1387 |
| ↳ 3.3 Cleaning otomatis & inferensi tipe kolom | 1487 |
| ↳ 3.4 Nilai mustahil, duplikat, dan Data Quality Report | 1792 |
| 4. Target Analysis | 1917 |
| 5. Column Roles & Domain Feature Engineering (Asuransi) | 1960 |
| ↳ 5.1 Column roles | 1962 |
| ↳ 5.2 Fitur domain asuransi yang dibuat (kalau kolomnya ada) | 1971 |
| ↳ 5.3 Early Leakage Screen (WAJIB dicek) | 2169 |
| 6. EDA Univariate | 2268 |
| 7. Driver Analysis — faktor apa yang berhubungan dengan churn? | 2332 |
| 8. Korelasi & Multikolinearitas | 2731 |
| 9. Segment & Interaction Analysis | 2794 |
| 10. Survival Analysis — *kapan* customer churn? | 2915 |
| 11. Interpretable Model — Logistic Regression (Odds Ratio) | 3382 |
| 12. Train–Test Drift & Adversarial Validation | 3461 |
| 13. Modeling Setup | 3547 |
| ↳ 13.1 Metric | 3549 |
| ↳ 13.2 Representasi fitur per keluarga model (otomatis) | 3559 |
| ↳ 13.3 Imbalance | 3570 |
| ↳ 13.4 Bangun matriks fitur | 3656 |
| ↳ 13.5 Cross-validation folds | 3714 |
| 14. Model Training (CV OOF) | 3751 |
| 15. Hyperparameter Tuning (Optuna) | 4138 |
| 16. Multi-seed Final Models | 4253 |
| ↳ 16.2 Feature Selection (permutation importance) | 4286 |
| ↳ 16.3 Ablation Study — kontribusi tiap kelompok fitur | 4328 |
| 17. Ensemble | 4432 |
| ↳ 17.2 Experiment Log & Model Development Journey (bukti trial-and-error) | 4564 |
| 18. Threshold & Calibration | 4625 |
| ↳ Threshold | 4627 |
| ↳ Calibration | 4633 |
| 19. Evaluation & Debugging | 4713 |
| 20. Feature Importance & Explainability | 5056 |
| ↳ 20.5 Consolidated Driver Evidence Matrix | 5405 |
| 21. Business Analytics — dari prediksi ke keputusan | 5475 |
| ↳ 21.5 Campaign ROI Simulation | 5678 |
| ↳ 21.6 What-if Scenarios (simulasi intervensi) | 5755 |
| ↳ 21.7 Causal Effect Estimation — Doubly Robust (AIPW) | 5880 |
| ↳ 21.8 Churn Personas (clustering SHAP values) | 6222 |
| ↳ 21.9 Fairness Audit & Model Card (governance) | 6280 |
| 22. Recommendations (otomatis, berbasis bukti) | 6377 |
| ↳ 22.2 Desain Pilot A/B & Roadmap Implementasi | 6539 |
| 23. Submission + Validator | 6628 |
| 24. Export untuk Laporan & Ringkasan Error | 6774 |
| ↳ 24.2 Executive Summary Figure (1 gambar ringkasan untuk halaman pertama laporan / slide) | 6857 |
| ↳ 24.2b Diagram Arsitektur & Metodologi (otomatis, berisi angka run ini) | 6907 |
| ↳ 24.3 Auto Article (REPORT_DRAFT.docx + PDF) — ⭐ deliverable utama | 7072 |
| ↳ 24.5 Interactive Dashboard (offline + siap deploy Vercel) | 9742 |
| ↳ 24.4 Package Deliverables (ZIP sesuai aturan soal) | 9921 |
| 25. Troubleshooting, Recipes & Checklist | 9962 |
| ↳ 25.1 Troubleshooting (Error → Penyebab → Solusi) | 9964 |
| ↳ 25.2 Experiment recipes | 9992 |
| (A) Cek cepat semua jalan | 9994 |
| (B) Final kuat (waktu cukup ~30-60 menit) | 9996 |
| (C) Waktu mepet tapi tetap kuat | 9998 |
| (D) Metric F1 / accuracy | 10000 |
| (E) Metric logloss | 10002 |
| (F) Data besar (>200k baris) | 10004 |
| (G) Ada household/agent grouping | 10006 |
| (H) GPU tersedia | 10008 |
| ↳ 25.3 Checklist sebelum submit | 10012 |

## Semua key CFG (129)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| **RUN** | | | |
| `RUN_MODE` | `"full"` | "fast" (debug cepat: subsample, 3 fold, model sedikit, tanpa Optuna) \| "full" | 687 |
| `STRICT` | `False` | True → error di section manapun langsung stop (untuk debugging) | 688 |
| `SEED` | `42` |  | 689 |
| `VERBOSE` | `True` |  | 690 |
| `CONFIG_FILE` | `None` | path JSON berisi override CFG (alternatif USER OVERRIDE); diisi otomatis oleh tools/run_case.py | 691 |
| **PATHS** | | | |
| `TRAIN_PATH` | `"./data/train.csv"` |  | 694 |
| `TEST_PATH` | `"./data/test.csv"` | boleh tidak ada → hanya analisis + CV | 695 |
| `SAMPLE_SUB_PATH` | `"./data/sample_submission.csv"` |  | 696 |
| `OUTPUT_DIR` | `"./outputs/churn_insurance"` |  | 697 |
| `EXTRA_TABLES` | `{}` | Tabel tambahan one-to-many (misal claims, payments, complaints) → diagregasi per customer lalu di-merge. Format: {"claims": {"path": "./data/claims.csv", "key": "customer_id", "date_col": "claim_date"}} | 700 |
| `EXTRA_ONE_TO_ONE` | `{}` | Tabel tambahan one-to-one (misal demografi terpisah) → langsung di-merge. Format: {"demo": {"path": "./data/demographics.csv", "key": "customer_id"}} | 703 |
| `TRAIN_LABELS` | `None` | Label train di file terpisah (mis. train_labels.csv) → di-merge HANYA ke train. Format: {"path": "...", "key": "customer_id"} | 705 |
| `EXCEL_SHEET` | `None` | nama/indeks sheet Excel; None = sheet terbesar otomatis | 706 |
| `TEST_FROM_MISSING_TARGET` | `True` | tanpa file test & ≥5% baris target kosong → baris itu dianggap data yang harus diprediksi | 707 |
| **COLUMNS** | | | |
| `ID_COL` | `None` | None = auto-detect | 710 |
| `TARGET_COL` | `None` | None = auto-detect | 711 |
| `POSITIVE_LABEL` | `None` | None = auto (Yes/1/True/Churn/...). Contoh: "Attrited Customer" | 712 |
| `DROP_COLS` | `[]` | kolom yang pasti dibuang (misal nama, nomor telp, alamat) | 713 |
| `FORCE_CATEGORICAL` | `[]` | kolom angka yang sebenarnya kategori (misal kode wilayah, kode produk) | 714 |
| `FORCE_NUMERIC` | `[]` | kolom string yang sebenarnya angka | 715 |
| `DATE_COLS` | `None` | None = auto-detect kolom tanggal | 716 |
| `REFERENCE_DATE` | `None` | tanggal acuan untuk "days since"; None = tanggal maksimum di data | 717 |
| `GROUP_COL` | `None` | misal "agent_id"/"household_id" kalau satu grup tidak boleh bocor antar fold | 718 |
| `COLUMN_ROLES` | `{}` | Role kolom (auto-detect by keyword). Override contoh: {"tenure": "lama_polis_bulan", "premium": "premi"} | 721 |
| `TENURE_UNIT` | `"auto"` | "months" \| "years" \| "days" \| "auto" (dipakai untuk band & survival) | 722 |
| `ACTIONABLE_FEATURES` | `None` | None = auto by keyword (payment, renew, complaint, claim, premium, channel, ...) | 723 |
| **CLEANING** | | | |
| `NA_TOKENS` | `["", " ", "?", "NA", "N/A", "n/a", "na", "null", "NULL", "Null", "nan", "NaN", "-", "--", "#N/A", "missing", "Missing", "unknown_value"]` |  | 726 |
| `DROP_DUPLICATES` | `True` |  | 727 |
| `QUASI_CONSTANT_THRESHOLD` | `0.995` | kolom dengan 1 nilai ≥ 99.5% → drop | 728 |
| `HIGH_CARD_THRESHOLD` | `50` | kategori dengan > 50 level → high-cardinality (freq + target encoding) | 729 |
| `RARE_MIN_FREQ` | `0.005` | level kategori < 0.5% digabung jadi "__RARE__" (untuk model linear & EDA) | 730 |
| `FIX_IMPOSSIBLE_VALUES` | `True` | age<0/>110, nilai negatif di premium/income/tenure → NaN | 731 |
| **LEAKAGE** | | | |
| `LEAKAGE_KEYWORDS` | `["cancel", "terminat", "churn_date", "end_date", "exit_date", "lapse_date", "reason", "suspend", "suspd", "close_reason", "stop_date", "berhenti", "tgl_keluar", "tanggal_keluar", "lapsed_on", "surrender_date", "alasan", "status_after", "closed", "close_date", "surrender", "refund", "batal", "tutup"]` |  | 734 |
| `LEAKAGE_AUC_THRESHOLD` | `0.97` | single-feature AUC ≥ ini → dicurigai leakage | 737 |
| `AUTO_DROP_LEAKY` | `True` | True → buang fitur dengan BUKTI statistik leakage (missingness/single-feature AUC). | 738 |
| **FEATURE ENGINEERING** | | | |
| `USE_DOMAIN_FE` | `True` | fitur domain asuransi (rasio klaim, band tenure, dsb) | 742 |
| `USE_FREQ_ENCODING` | `True` | frequency encoding untuk kategori | 743 |
| `USE_TARGET_ENCODING` | `True` | target encoding in-fold untuk kategori high-cardinality | 744 |
| `USE_ROW_STATS` | `True` | jumlah missing per baris, dsb | 745 |
| **VALIDATION & METRIC** | | | |
| `METRIC` | `"roc_auc"` |  | 748 |
| `THRESHOLD_METRIC` | `"auto"` | metrik untuk tuning threshold; "auto" = METRIC kalau label-based, else "f1" | 749 |
| `N_FOLDS` | `5` |  | 750 |
| `N_REPEATS` | `1` | >1 = repeated CV (lebih stabil, lebih lama) | 751 |
| `CV_STRATEGY` | `"auto"` | "auto" \| "stratified" \| "group" | 752 |
| **MODELS** | | | |
| `MODELS` | `["logreg", "rf", "et", "hgb", "lgbm", "xgb", "catboost", "ebm", "mlp"]` |  | 755 |
| `EBM_INTERACTIONS` | `10` | ebm = Explainable Boosting Machine (glass-box GAM, interpret-core); mlp = neural network baseline (sklearn) · pasangan interaksi EBM (auto diturunkan untuk data besar) | 757 |
| `EBM_OUTER_BAGS` | `8` |  | 758 |
| `IMBALANCE` | `"auto"` | "auto" \| "none" \| "class_weight" \| "smote" | 759 |
| `EARLY_STOPPING_ROUNDS` | `200` |  | 760 |
| `USE_GPU` | `"auto"` | "auto" = pakai GPU kalau ada NVIDIA (fallback ke CPU kalau gagal) \| True \| False | 761 |
| `AUTO_ADAPT` | `True` | sesuaikan otomatis dengan ukuran data & RAM/CPU/GPU (Section 2.1) | 762 |
| `LARGE_DATA_ROWS` | `150_000` | di atas ini → mode data besar | 763 |
| `N_JOBS` | `-1` |  | 764 |
| `USE_OPTUNA` | `True` |  | 766 |
| `OPTUNA_MODELS` | `["lgbm", "catboost", "xgb"]` |  | 767 |
| `OPTUNA_TRIALS` | `40` |  | 768 |
| `OPTUNA_TIMEOUT` | `900` | detik per model | 769 |
| `OPTUNA_FOLDS` | `3` | fold saat tuning (lebih cepat); final tetap pakai N_FOLDS | 770 |
| `FINAL_SEEDS` | `[42, 2024, 7]` | multi-seed averaging untuk model terbaik | 772 |
| `USE_ENSEMBLE` | `True` |  | 773 |
| `CALIBRATION` | `"auto"` | "auto" \| "none" \| "isotonic" \| "sigmoid" | 774 |
| `USE_FEATURE_SELECTION` | `True` |  | 775 |
| **ADVANCED ANALYSES** | | | |
| `USE_ABLATION` | `True` | ablation study per kelompok fitur (kontribusi tiap kelompok, uji DeLong) | 778 |
| `ABLATION_MAX_GROUPS` | `8` |  | 779 |
| `USE_UPLIFT` | `True` | efek heterogen (DR-learner / CATE) + uplift targeting vs risk targeting | 780 |
| `CAUSAL_MAX_TREATMENTS` | `6` |  | 781 |
| `TREATMENT_COLS` | `[]` | kolom intervensi historis (mis. "received_retention_offer", "agent_call") → dianalisis kausal lebih dulu | 782 |
| `PROTECTED_ATTRIBUTES` | `None` | None = auto (gender, age band, marital, region) → fairness audit | 783 |
| `PILOT_POWER` | `0.80` | desain pilot A/B | 784 |
| `PILOT_ALPHA` | `0.05` |  | 785 |
| `CONTACT_COST_SHARE` | `0.2` | porsi RETENTION_COST untuk kontak (sisanya insentif) → EMPC | 786 |
| `EMPC_ALPHA` | `None` | Beta(α, β) peluang tawaran diterima; None → rata-rata = RETENTION_SUCCESS_RATE | 787 |
| `EMPC_BETA` | `None` |  | 788 |
| **EXPLAINABILITY** | | | |
| `SHAP_SAMPLE` | `3000` |  | 791 |
| `PERM_IMPORTANCE_REPEATS` | `3` |  | 792 |
| `TOP_N_FEATURES_PLOT` | `20` |  | 793 |
| **EDA** | | | |
| `MAX_EDA_PLOTS` | `24` |  | 796 |
| `MAX_CAT_LEVELS_PLOT` | `15` |  | 797 |
| `N_BINS_NUMERIC` | `10` |  | 798 |
| **BUSINESS** | | | |
| `CURRENCY` | `"IDR"` |  | 801 |
| `VALUE_COL` | `None` | kolom nilai customer (premi). None = auto dari role "premium" | 802 |
| `VALUE_PERIOD` | `"annual"` | "annual" \| "monthly" (kalau monthly → dikali 12 untuk CLV tahunan) | 803 |
| `PROFIT_MARGIN` | `0.20` | margin keuntungan dari premi | 804 |
| `RETENTION_COST` | `150_000` | biaya kontak/insentif per customer yang di-target | 805 |
| `RETENTION_SUCCESS_RATE` | `0.30` | % churner yang berhasil dipertahankan jika di-target | 806 |
| `DISCOUNT_RATE` | `0.10` | discount rate untuk CLV | 807 |
| `CLV_HORIZON_YEARS` | `5` |  | 808 |
| `RISK_TIER_CUTS` | `(0.5, 0.8)` | quantile → Low (<50%), Medium (50–80%), High (≥80%) | 809 |
| `WHAT_IF_SCENARIOS` | `None` | None = auto-generate dari actionable features | 810 |
| `WHAT_IF_ADOPTION` | `0.30` | asumsi % customer terdampak yang benar-benar berubah perilakunya | 811 |
| **SUBMISSION** | | | |
| `SUBMISSION_MODE` | `"probability"` | "probability" \| "label" | 814 |
| `SUBMISSION_LABEL_FORMAT` | `"auto"` | "auto" \| "01" \| "original" (Yes/No dsb) | 815 |
| `SUBMISSION_TARGET_COL` | `None` | None = kolom ke-2 sample_submission / TARGET_COL | 816 |
| **REPORT & DELIVERABLES** | | | |
| `TEAM_NAME` | `"TeamName"` | dipakai untuk nama file laporan & ZIP: "<TEAM_NAME>_Final Stage 1" | 819 |
| `COMPANY_NAME` | `"the insurer"` | nama perusahaan di soal (untuk draft laporan) | 820 |
| `REPORT_TITLE` | `"Understanding and Reducing Customer Churn: A Data-Driven Retention Strategy"` |  | 821 |
| `REPORT_COMPACT` | `True` | draft ringkas (target ≈10 halaman isi); False = versi panjang | 822 |
| `MAKE_REPORT_DRAFT` | `True` | buat REPORT_DRAFT.docx + .md otomatis dengan angka hasil run | 823 |
| `MAKE_ZIP` | `False` | True setelah artikel PDF selesai → buat ZIP deliverables | 824 |
| `ARTICLE_PDF_PATH` | `None` | path PDF artikel final (untuk ZIP) | 825 |
| `NOTEBOOK_PATH` | `None` | path .ipynb ini (untuk ZIP); None = cari *.ipynb di folder kerja | 826 |
| `N_BOOTSTRAP` | `1000` | bootstrap CI untuk skor model | 827 |
| `FEATURE_LABELS` | `{}` | nama bisnis untuk laporan, misal {"n_complaints_12m": "complaints in last 12 months"} | 828 |
| `TEAM_MEMBERS` | `[]` | ["Nama 1", "Nama 2", "Nama 3"] → cover | 829 |
| `UNIVERSITY` | `""` | → cover | 830 |
| `CASE_CONTEXT` | `""` | 2–4 kalimat konteks dari soal (produk, pasar, tujuan manajemen) → Introduction | 831 |
| `INSURANCE_LINE` | `"auto"` | "life" \| "general" \| "health" \| "auto" (tebak dari data) → pilihan paragraf literatur | 832 |
| `MARKET` | `"Indonesia"` | "Indonesia" → pakai fakta OJK/AAJI terverifikasi; lainnya → konteks umum | 833 |
| `REPORT_MAX_BODY_PAGES` | `10` | batas isi utama (tanpa cover & lampiran) | 834 |
| `REPORT_AUTOFIT` | `True` | MS Word: hitung halaman & padatkan otomatis sampai ≤ batas | 835 |
| `MAKE_PDF` | `True` | DOCX → PDF via MS Word (kalau ada) | 836 |
| `MAKE_DASHBOARD` | `True` | outputs/.../dashboard (siap deploy Vercel) + dashboard_offline.html | 837 |
| `REFERENCES_COUNT_IN_LIMIT` | `True` | True = daftar pustaka ikut dihitung dalam batas halaman (aman) | 838 |
| `REPORT_SUBTITLE` | `"Final Stage 1 — Data Analytics Competition"` |  | 839 |
| **CASE BRIEF (isi dari soal; lihat tools/CASE_INTAKE.html)** | | | |
| `CHURN_DEFINITION` | `""` | mis. "a policy not renewed within 30 days after expiry" | 842 |
| `DATA_PERIOD` | `""` | mis. "January 2023 – December 2024" | 843 |
| `MANAGEMENT_OBJECTIVE` | `"auto"` | "reduce_churn" \| "maximize_profit" \| "understand_drivers" \| "early_warning" \| "auto" | 844 |
| `CASE_QUESTIONS` | `[]` | pertanyaan yang diminta soal (EN) → jadi Research Questions | 845 |
| `AVAILABLE_INTERVENTIONS` | `[]` | aksi yang memang bisa dilakukan perusahaan menurut soal → diprioritaskan | 846 |
| `RETENTION_BUDGET` | `None` | batas anggaran kampanye (mata uang) → kampanye optimal dalam budget | 847 |
| `PORTFOLIO_SIZE` | `None` | jumlah nasabah/polis aktif perusahaan (dari soal) → angka uang & budget diskalakan ke portofolio | 848 |
| `BUSINESS_PARAMS_SOURCE` | `"assumed"` | "case" kalau margin/biaya/success rate diberikan di soal → kalimat laporan menyesuaikan | 849 |
| `NARRATIVE_ANGLE` | `"auto"` | "lifecycle" \| "payment" \| "price" \| "service" \| "profit" \| "auto" → judul & framing | 850 |
| `CASE_KEY_FACTS` | `[]` | kalimat fakta dari casebook (EN) → Introduction (alternatif sheet case_facts) | 851 |
| `RESEARCH_FILE` | `None` | CASE_RESEARCH.xlsx (tools/research_helper.py); None = cari otomatis di folder kerja | 852 |
| `USE_DEFAULT_MARKET_CONTEXT` | `True` | False = paragraf konteks pasar bawaan diganti fakta industri dari CASE_RESEARCH.xlsx | 853 |
| `DASHBOARD_INCLUDE_CUSTOMERS` | `True` | False sebelum deploy publik (Vercel) kalau data nasabah rahasia | 854 |
| **FIGURES** | | | |
| `FIG_DPI` | `200` | 300 untuk kualitas cetak laporan | 857 |
| `PALETTE` | `["#2E86AB", "#E4572E", "#76B041", "#F3A712", "#6C4AB6", "#17BEBB", "#8C8C8C"]` |  | 858 |
