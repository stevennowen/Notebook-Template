# Notebook Template — Data Analytics Competition

Kumpulan template notebook Python (`.ipynb`) siap pakai untuk lomba data analytics/data science, dengan fokus utama **customer churn & data asuransi**. Setiap notebook:

- **End-to-end**: load → cleaning → EDA → preprocessing → modeling (CV OOF) → tuning → ensemble → evaluasi & debugging → explainability → output/submission.
- **Tahan error**: library opsional dicek otomatis (auto-install yang belum ada), setiap model/section dibungkus supaya satu kegagalan tidak menghentikan notebook, kompatibel lintas versi library.
- **Penuh panduan** (Bahasa Indonesia, istilah teknis English): tujuan tiap langkah, cara membaca output, aturan **"Jika X → Y"**, catatan **"Untuk laporan"**, tabel troubleshooting, recipes, dan checklist.
- **Report-ready**: figure bernomor otomatis (DPI bisa 300), tabel CSV/Excel, dan kalimat insight (English) yang siap diparafrase.

> Jalankan di **Jupyter / VS Code lokal** (sesuai aturan lomba: Google Colab tidak diperbolehkan).

> **▶ Hari-H: buka [`MULAI_DI_SINI.md`](MULAI_DI_SINI.md) dulu** — satu panduan berurutan dari download sampai upload.
>
> **Pakai di komputer lain (mis. komputer panitia)?** Baca [`DOWNLOAD_DAN_INSTALASI.md`](DOWNLOAD_DAN_INSTALASI.md): download dari Zenodo/GitHub Release → `python -m pip install -r requirements-churn.txt` → `QA/ALUR_KERJA_HARI_H.md`.
>
> Dibuat oleh **Sains Data UB**. Lisensi MIT (lihat `LICENSE`); cara sitasi di `CITATION.cff`.

---

## Struktur repo

> **AI agent:** mulai dari [`AGENTS.md`](AGENTS.md) — peta ringkas (±3 ribu token) + index per template di `docs/agent_index/`.

```
00_Setup/
  00_environment_check.ipynb        ← JALANKAN PERTAMA di laptop lomba (cek & install package, GPU)
01_Churn_Insurance/                 ← ⭐ UTAMA untuk case churn / asuransi
  00_MULAI_DI_SINI.md               ← RUNBOOK hari H: data masuk → laporan → ZIP (baca ini dulu)
  churn_insurance_master.ipynb      ← analisis + prediksi + rekomendasi bisnis + AUTO DRAFT LAPORAN, end-to-end
  contoh_output/                    ← contoh hasil run pada data sintetis (draft laporan, insights, figure)
  PLAYBOOK_ANALISIS.md              ← "jika X maka Y", cheat-sheet statistik, istilah asuransi, strategi retensi, Q&A juri
  PANDUAN_LAPORAN.md                ← struktur artikel 10 halaman, mapping figure → bab, template kalimat, referensi
  BAHAN_LATAR_BELAKANG.md           ← latar belakang + tinjauan pustaka + rumusan masalah (EN/ID), fakta OJK/AAJI terverifikasi, daftar pustaka
  CONTOH_LAPORAN_FINAL/             ← contoh artikel final lengkap (docx + pdf) dari data asuransi nyata (Kaggle)
  CASE_SCENARIOS.md                 ← semua kemungkinan bentuk case/data → setting & efek ke laporan
  RESEARCH_KIT.md                   ← metode riset latar belakang/"resume": casebook → keyword → sumber → verifikasi → workbook → laporan
  dashboard_template/               ← web app dashboard (diisi otomatis oleh notebook)
  Article_Template_Final_Stage.docx ← template Word sesuai format soal (A4, TNR 12, 1.15, margin 4/3 cm)
02_Tabular/
  tabular_master.ipynb              ← klasifikasi biner/multikelas & regresi tabular umum
  tabular_multilabel.ipynb          ← beberapa target 0/1 sekaligus (binary relevance + classifier chain, threshold per label)
03_NLP/
  nlp_classification.ipynb          ← teks → kelas (TF-IDF, embedding, transformer K-fold)
  nlp_regression.ipynb              ← teks → angka
  nlp_token_classification.ipynb    ← NER / slot filling / label per kata (BIO, entity-level F1)
  nlp_llm_zero_shot.ipynb           ← LLM lokal: zero/few-shot classification (skor label) & ekstraksi JSON
04_Computer_Vision/
  cv_image_classification.ipynb     ← gambar → kelas (timm / torchvision / fallback CNN)
  cv_segmentation.ipynb             ← segmentasi semantik (mask PNG / RLE; smp → torchvision → U-Net)
  cv_object_detection.ipynb         ← deteksi objek & counting (YOLO ultralytics → Faster R-CNN; mAP, kalibrasi jumlah)
  cv_ocr_text_recognition.ipynb     ← OCR teks/digit/aksara (CRNN + CTC, beam search + batasan)
  CV_OTHER_TASKS_GUIDE.md           ← referensi kode (bukan template): image regression, multilabel image, tips lintas task
05_Time_Series/
  time_series_forecasting.ipynb     ← forecasting single/panel (baseline, ETS/SARIMAX, LightGBM global)
06_Clustering_Segmentation/
  clustering_segmentation.ipynb     ← segmentasi customer (KMeans/GMM/HDBSCAN, RFM, profiling bisnis)
07_Anomaly_Detection/
  anomaly_tabular.ipynb             ← fraud/outlier tabular (IForest, LOF, OCSVM, autoencoder, + supervised kalau ada label)
  anomaly_image.ipynb               ← cacat produk dari gambar normal saja (PatchCore-lite + heatmap)
08_Audio/
  audio_classification.ipynb        ← klasifikasi suara (log-mel CNN + MFCC/LightGBM, open-set "Unknown")
09_Statistical_Analysis/
  statistical_analysis_report.ipynb ← uji hipotesis otomatis, A/B test, regresi + CI, insight & laporan DOCX
  (setiap .ipynb punya kembaran .py dengan isi sama — lihat "Notebook atau script?")
Guides/
  CHEATSHEET_METRIK_MODEL_VALIDASI.md
  DEBUGGING_TANPA_AI.md
tools/
  new_task.py                       ← folder kerja untuk task APA PUN: --task <nama> --format ipynb|py (+ RUN_*.bat)
  run_task.py                       ← jalankan template apa pun headless (format ipynb atau py)
  nb_formats.py                     ← .ipynb ⇄ .py (export / to-ipynb / check sinkron)
  make_synthetic_task_data.py       ← data latihan sintetis untuk setiap task (pola yang diketahui)
  smoke_test_templates.py           ← uji semua template dengan data sintetis → QA/TEMPLATE_SMOKE_REPORT.md
  build_agent_index.py              ← buat AGENTS.md + docs/agent_index/ (peta repo untuk AI agent; --check untuk QA)
  new_case.py                       ← buat folder kerja case churn (data/, case_config.json, intake form, RUN_*.bat; --format py)
  CASE_INTAKE.html                  ← form offline: semua info dari soal → case_config.json
  run_case.py                       ← jalankan notebook headless dengan config (output tersimpan di notebook)
  make_zip.py                       ← ZIP pengumpulan sesuai aturan
  stress_test.py                    ← uji 44 bentuk data (22 skenario case + 22 data rusak/dimanipulasi) → STRESS_TEST_REPORT.md
  research_helper.py                ← casebook (pdf/docx/md) → keyword, fakta, link pencarian (RESEARCH_PLAN.html) + workbook riset
  CASE_RESEARCH_TEMPLATE.xlsx       ← workbook riset kosong (case_facts, industry_facts, literature, custom_paragraphs, feature_labels)
  case_config.example.json
  make_synthetic_insurance_churn.py ← generator data latihan asuransi (messy + driver yang diketahui)
examples/
  mock_case_warkab/                 ← contoh case lengkap (casebook fiktif, config, riset terisi) → make_mock_case.py
QA/
  QA_FINAL_REPORT.md                ← hasil QA/QC final (stress test, regresi, demo end-to-end)
  walkthrough_warkab/               ← REKAMAN proses demo: video MP4 + screenshot + STEPS.md
  make_walkthrough.py               ← membuat rekaman walkthrough dari folder case mana pun
requirements.txt
```

---

## Alur hari H (churn/asuransi) — ringkas
```
python tools/new_case.py ../case --casebook soal.pdf → folder kerja + form intake + RUN_*.bat + RESEARCH_PLAN.html + CASE_RESEARCH.xlsx
copy data panitia → case/data/ ; isi case/CASE_INTAKE.html → case_config.json
riset: klik link di RESEARCH_PLAN.html → isi CASE_RESEARCH.xlsx (otomatis masuk Introduction/Discussion/pustaka)
RUN_FAST.bat (cek 5–10 mnt) → RUN_FULL.bat (30–90 mnt)
→ case/outputs/churn_insurance/<Tim>_Final Stage 1.docx/.pdf  (artikel lengkap ≤10 hal + appendix A–L)
→ NARRATIVE_OPTIONS.md (pilih narasi) · JUDGE_QA.md (latihan Q&A) · dashboard/ (npx vercel --prod)
→ edit di Word → Save As PDF → python tools/make_zip.py --case ../case --pdf "<PDF final>"
```
Detail lengkap: `01_Churn_Insurance/00_MULAI_DI_SINI.md` · semua kemungkinan case: `01_Churn_Insurance/CASE_SCENARIOS.md`.

## Yang baru (upgrade final)
- **Headless runner** `tools/run_case.py` + `tools/new_case.py` + form offline `tools/CASE_INTAKE.html` (semua pertanyaan case → `case_config.json`).
- **Artikel ditulis otomatis & lengkap**: narasi berbasis data + literatur terverifikasi, judul berbasis temuan, figure komposit, cross-reference, daftar pustaka hanya yang disitasi, **autofit ≤ 10 halaman via MS Word/LibreOffice**, appendix A–L, PDF otomatis, `NARRATIVE_OPTIONS.md`, `JUDGE_QA.md`.
- **Metodologi tambahan**: EBM (glass-box) & MLP di model zoo, **experiment log + model development journey**, **ablation study (DeLong)**, **DeLong/Holm**, **EMPC (profit-based)**, ECE, RMST, Schoenfeld PH test, Weibull AFT, KM per risk tier + C-index, **AIPW robustness (placebo, E-value, overlap, DAG)**, **DR-learner uplift + GATES**, **fairness audit + model card**, **pilot A/B power analysis**, roadmap Gantt, 5 diagram arsitektur.
- **Robustness**: CatBoost OOF fix, label file terpisah, 1-file dengan target kosong = test, rekonstruksi tenure dari tanggal, figure tidak duplikat saat cell dijalankan ulang, console Windows UTF-8, fallback LibreOffice, template dashboard ter-embed.
- **Dashboard interaktif** (`outputs/.../dashboard/`, tanpa library eksternal) — offline & siap Vercel, dengan simulator kampanye.
- **Stress test** `tools/stress_test.py`: 44 bentuk data (22 skenario case + 22 data rusak: BOM, kolom ganda, "Rp 5 jt", tanggal campur, inf, label kotor, Excel multi-sheet berjudul, jsonl, leakage, target konstan, …) → `STRESS_TEST_REPORT.md`.
- **Research kit**: `research_helper.py` + `RESEARCH_KIT.md` + `CASE_RESEARCH.xlsx` → fakta casebook, fakta industri, literatur tim, paragraf custom & label fitur masuk otomatis ke artikel (dedupe referensi, peringatan sumber belum diverifikasi).
- **Skala portofolio & budget**: `PORTFOLIO_SIZE` → nilai uang dari sampel diskalakan ke portofolio; budget dibandingkan dengan kampanye skala portofolio (binding/tidak).
- **Contoh case + rekaman QA**: `examples/mock_case_warkab/` dan `QA/walkthrough_warkab/` (video proses end-to-end tim Warkab).
- **Dashboard (Vercel)** = hasil analisis case, dibuat ulang tiap run (konteks soal, pertanyaan, tim, KPI, driver, kampanye, rekomendasi); `DASHBOARD_INCLUDE_CUSTOMERS=false` untuk deploy publik tanpa data nasabah.

## Mulai cepat (template lain)

1. **Clone / copy** repo ke laptop lomba → `SETUP.bat` (atau `00_Setup/00_environment_check.ipynb` → Run All).
2. Buat folder kerja (pilih task & format):
   ```
   python tools/new_task.py ../kerja --task segmentation --format ipynb     # notebook Jupyter
   python tools/new_task.py ../kerja --task segmentation --format py        # script Python
   python tools/new_task.py --list                                          # semua nama task
   ```
3. Copy data panitia ke `../kerja/data/`, isi `task_config.json` (path, kolom, metrik).
4. Double-click `RUN_FAST.bat` (cek cepat) → baca output → `RUN_FULL.bat` (final).
   Atau buka file template di Jupyter/VS Code dan edit cell `# >>> USER OVERRIDE` (nilai di `task_config.json` tetap menang).
5. Ambil `outputs/<template>/` (submission, figures, tables, summary.md) untuk laporan & submit.

| Soal | `--task` | Template |
|---|---|---|
| Churn / retensi / lapse / attrition (+ insight, artikel, dashboard) | `churn` (pakai `new_case.py`) | `01_Churn_Insurance/churn_insurance_master` |
| Tabular klasifikasi/regresi | `tabular` | `02_Tabular/tabular_master` |
| Tabular, beberapa target sekaligus | `tabular-multilabel` | `02_Tabular/tabular_multilabel` |
| Teks → kelas / angka | `nlp-classification` / `nlp-regression` | `03_NLP/...` |
| Label per kata (NER, slot) | `ner` | `03_NLP/nlp_token_classification` |
| Label sedikit/tidak ada, ekstraksi info, LLM | `llm` | `03_NLP/nlp_llm_zero_shot` |
| Gambar → kelas | `cv-classification` | `04_Computer_Vision/cv_image_classification` |
| Gambar → mask per pixel | `segmentation` | `04_Computer_Vision/cv_segmentation` |
| Gambar → kotak objek / jumlah objek | `detection` | `04_Computer_Vision/cv_object_detection` |
| Gambar → teks (digit, plat, aksara) | `ocr` | `04_Computer_Vision/cv_ocr_text_recognition` |
| Deret waktu / forecasting | `time-series` | `05_Time_Series/time_series_forecasting` |
| Segmentasi customer tanpa label | `clustering` | `06_Clustering_Segmentation/clustering_segmentation` |
| Fraud / outlier tabular | `anomaly-tabular` | `07_Anomaly_Detection/anomaly_tabular` |
| Cacat produk (hanya gambar normal) | `anomaly-image` | `07_Anomaly_Detection/anomaly_image` |
| Suara → kelas | `audio` | `08_Audio/audio_classification` |
| Analisis statistik / A/B test / laporan insight | `stats` | `09_Statistical_Analysis/statistical_analysis_report` |

## Notebook atau script?

Setiap template tersedia dua-duanya dengan **isi identik**: `<nama>.ipynb` (sumber) dan `<nama>.py` (dibuat otomatis).

| | `.ipynb` (notebook) | `.py` (script) |
|---|---|---|
| Cocok untuk | eksplorasi interaktif, lampiran yang mewajibkan notebook ber-output | jalan sekali dari terminal, server/komputer tanpa Jupyter, diff git yang rapi |
| Cara jalan | Jupyter → Restart & Run All | `python <nama>.py`, atau VS Code → "Run Cell" per `# %%` |
| Headless | `run_task.py --format ipynb` (butuh nbclient) | `run_task.py --format py` (output → `run_log.txt`) |

Pindah format: `python tools/nb_formats.py export <nama>.ipynb` atau `python tools/nb_formats.py to-ipynb <nama>.py`.
Pengembang template: edit `.ipynb` (atau `.py` lalu `to-ipynb`), lalu `python tools/nb_formats.py export` dan
`python tools/nb_formats.py check` (gagal kalau ada .py yang tidak sinkron), lalu
`python tools/build_agent_index.py` supaya `AGENTS.md` (peta untuk AI agent) ikut terbarui.
**Churn**: `new_case.py ... --format py` membuat RUN_*.bat yang menjalankan script; ZIP lalu berisi `.py` + `run_log.txt`
(kalau aturan lomba mewajibkan `.ipynb`, pakai format default).

---

## ⭐ Churn & Insurance master notebook — ringkasan isi

| # | Section | Highlight |
|---|---|---|
| 0–1 | Setup & Config | auto-install, availability flags, `CFG` + `USER OVERRIDE`, fast/full mode, **auto-adapt ke RAM/CPU/GPU & ukuran data** |
| 2 | Load data | csv/xlsx/parquet/json, auto separator/encoding, **multi-tabel** (merge 1-1, agregasi 1-many: claims/payments) |
| 3 | Data quality & cleaning | deteksi target/ID otomatis, parsing label churn (Yes/No, Exited, Attrited, is_active terbalik), angka-dalam-string (`"12,500,000"`, `"Rp 5.000.000"`), tanggal, typo kategori, nilai mustahil, duplikat, quality report |
| 4 | Target | churn rate, imbalance, baseline |
| 5 | Roles & domain FE | deteksi role (tenure, premi, klaim, keluhan, telat bayar, channel, ...) EN+ID; fitur asuransi (premium-to-income, claims per year, reject rate, tenure band, price shock, ...); **leakage screen** |
| 6–7 | EDA & **driver analysis** | churn rate + Wilson CI, chi-square/Mann-Whitney + **effect size**, **Information Value/WoE**, univariate AUC, FDR correction, insight otomatis, **analisis alasan churn** (kolom post-event, deskriptif) |
| 8 | Korelasi | Spearman, Cramér's V, VIF |
| 9 | Segments | heatmap interaksi, **decision-tree segment rules** (IF–THEN + lift) |
| 10 | **Survival** | Kaplan-Meier, retention 6/12/24/36 bln, hazard per interval, log-rank, **Cox hazard ratios** (+ uji asumsi PH bila lifelines ada) |
| 11 | Interpretable model | Logistic regression **odds ratio + 95% CI + marginal effects**, forest plot |
| 12 | Drift | PSI, KS, unseen categories, adversarial validation |
| 13–17 | Modeling | LogReg, RF, ET, HGB, LightGBM, XGBoost, CatBoost (native categorical, in-fold target encoding), Optuna (pruning), multi-seed, feature selection, **ensemble** (mean, rank, hill-climbing, optimized weights, stacking) |
| 18 | Threshold & calibration | threshold optimal per metrik, reliability diagram, isotonic/Platt |
| 19 | Debugging | ROC/PR, confusion matrix, **bootstrap 95% CI & paired model comparison**, fold stability, **learning curve**, **shuffled-target test**, OOF vs test, error analysis, performa per segmen |
| 20 | Explainability | gain, permutation, **SHAP** (bar, beeswarm, dependence, kategori, waterfall), **PDP**, **driver heterogeneity per segmen**, **driver evidence matrix** |
| 21 | **Business analytics** | lift/gains/KS, risk tiers, **CLV & value at risk**, **risk × value matrix**, **campaign ROI simulation + sensitivity**, **what-if scenarios**, **causal effects (doubly robust AIPW)**, **churn personas (SHAP clustering)** |
| 22 | Recommendations | rekomendasi otomatis berbasis bukti (aksi, evidence, impact, KPI) |
| 23 | Submission | validator, beberapa varian submission, action list customer (tier, value, quadrant) |
| 24 | **Deliverables** | Excel semua tabel, `insights.md`, executive summary figure, **5 diagram arsitektur**, **artikel lengkap `<Tim>_Final Stage 1.docx/.pdf` (autofit ≤10 hal, appendix A–L)**, `NARRATIVE_OPTIONS.md`, `JUDGE_QA.md`, **dashboard Vercel**, ZIP |
| 25 | Troubleshooting | 20+ error → solusi, recipes, checklist |

Latihan: `python tools/make_synthetic_insurance_churn.py --out ./data` → jalankan notebook churn (set `CFG.EXTRA_TABLES` untuk `claims.csv`) → bandingkan driver yang ditemukan dengan driver yang ditanam (lihat docstring generator).

---

## Catatan environment
- Python 3.9–3.12 (diuji juga di 3.11 + pandas 3.0: template mematikan `future.infer_string` supaya perilaku tipe teks sama seperti pandas 2). Library inti: numpy, pandas, scikit-learn, scipy, matplotlib, seaborn. Disarankan: lightgbm, xgboost, catboost, statsmodels, optuna, shap, lifelines (lihat `requirements.txt`).
- **Torch tidak di-install otomatis** (harus cocok dengan CUDA) → https://pytorch.org/get-started/locally/
- Kalau di laptop ada beberapa Python, pastikan `pip` menginstall ke Python yang dipakai kernel (`sys.executable` di environment check).
- Cell install hanya memasang package yang **belum ada** (tidak meng-upgrade yang sudah ada). Matikan dengan `CFG_AUTO_INSTALL = False` / `AUTO_INSTALL = False`.
