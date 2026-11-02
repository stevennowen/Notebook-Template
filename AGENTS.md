# AGENTS.md — peta Notebook-Template untuk AI agent

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual; jalankan ulang skrip itu. -->

Repo ini berisi **template lomba data science** (Python, Bahasa Indonesia + istilah teknis English). Setiap template
adalah satu notebook end-to-end: load data → EDA → model + cross-validation → evaluasi → submission → ringkasan laporan.
Tugas agent biasanya: (1) memilih template yang cocok dengan soal, (2) menyiapkan folder kerja & config,
(3) menjalankan dan membaca hasil, (4) menjawab pertanyaan tentang setting/isi template.

## Aturan membaca (hemat token)

1. **Jangan scan seluruh repo.** Mulai dari tabel di bawah, lalu buka `docs/agent_index/<template>.md` untuk detail.
2. **Baca file `.py`, bukan `.ipynb`.** Isinya identik (dibuat `tools/nb_formats.py`), tapi `.ipynb` adalah JSON
   penuh escape → ±2× token. Index memberi **nomor baris di `.py`**: baca rentang baris itu saja.
3. Abaikan folder `outputs/`, `data/`, `QA/demo_warkab/`, `QA/walkthrough_warkab/`, `01_Churn_Insurance/CONTOH_LAPORAN_FINAL/`,
   `01_Churn_Insurance/contoh_output/` kecuali diminta (hasil run contoh, gambar, PDF besar).
4. Untuk mengubah template: edit `.ipynb` **atau** `.py` lalu sinkronkan
   (`python tools/nb_formats.py to-ipynb <file>.py` → `python tools/nb_formats.py export`), lalu
   `python tools/build_agent_index.py` supaya index ikut terbarui.

## Pilih template (soal → `--task`)

| `--task` | Untuk soal | File yang dibaca | Detail (section, baris, semua key CFG) |
|---|---|---|---|
| `churn` | Churn / retensi / asuransi + artikel, dashboard, ZIP otomatis | `01_Churn_Insurance/churn_insurance_master.py` (10,022 baris) | `docs/agent_index/churn_insurance_master.md` |
| `tabular` | Tabular: klasifikasi biner/multikelas, regresi | `02_Tabular/tabular_master.py` (5,478 baris) | `docs/agent_index/tabular_master.md` |
| `tabular-multilabel` | Tabular multilabel: beberapa target 0/1 sekaligus (atau satu kolom berisi daftar label) | `02_Tabular/tabular_multilabel.py` (564 baris) | `docs/agent_index/tabular_multilabel.md` |
| `stats` | Analisis statistik & laporan insight (uji hipotesis, A/B test, regresi + CI) tanpa lomba prediksi | `09_Statistical_Analysis/statistical_analysis_report.py` (775 baris) | `docs/agent_index/statistical_analysis_report.md` |
| `anomaly-tabular` | Anomaly / fraud / outlier detection pada data tabular (tanpa label atau sedikit label) | `07_Anomaly_Detection/anomaly_tabular.py` (594 baris) | `docs/agent_index/anomaly_tabular.md` |
| `time-series` | Forecasting deret waktu (single series & panel) | `05_Time_Series/time_series_forecasting.py` (3,028 baris) | `docs/agent_index/time_series_forecasting.md` |
| `clustering` | Clustering / segmentasi customer tanpa label (+ RFM) | `06_Clustering_Segmentation/clustering_segmentation.py` (2,158 baris) | `docs/agent_index/clustering_segmentation.md` |
| `nlp-classification` | Teks → kelas (biner/multikelas/multilabel): TF-IDF, embedding, transformer | `03_NLP/nlp_classification.py` (3,878 baris) | `docs/agent_index/nlp_classification.md` |
| `nlp-regression` | Teks → angka (skor, harga, rating) | `03_NLP/nlp_regression.py` (3,427 baris) | `docs/agent_index/nlp_regression.md` |
| `ner` | NER / slot filling / token classification (BIO) | `03_NLP/nlp_token_classification.py` (674 baris) | `docs/agent_index/nlp_token_classification.md` |
| `llm` | LLM lokal: zero/few-shot classification & ekstraksi (+ LoRA opsional) | `03_NLP/nlp_llm_zero_shot.py` (574 baris) | `docs/agent_index/nlp_llm_zero_shot.md` |
| `cv-classification` | Gambar → kelas | `04_Computer_Vision/cv_image_classification.py` (3,682 baris) | `docs/agent_index/cv_image_classification.md` |
| `segmentation` | Segmentasi gambar (semantic, biner/multikelas): mask PNG atau RLE | `04_Computer_Vision/cv_segmentation.py` (1,439 baris) | `docs/agent_index/cv_segmentation.md` |
| `detection` | Object detection & counting (YOLO): bounding box atau jumlah objek | `04_Computer_Vision/cv_object_detection.py` (1,126 baris) | `docs/agent_index/cv_object_detection.md` |
| `ocr` | OCR / text recognition: digit, plat, aksara, teks pendek dalam gambar (CRNN-CTC) | `04_Computer_Vision/cv_ocr_text_recognition.py` (886 baris) | `docs/agent_index/cv_ocr_text_recognition.md` |
| `anomaly-image` | Anomaly detection gambar (cacat produk/visual inspection, hanya contoh normal) | `07_Anomaly_Detection/anomaly_image.py` (515 baris) | `docs/agent_index/anomaly_image.md` |
| `audio` | Audio → kelas (suara hewan, ucapan, mesin) + open-set 'Unknown' | `08_Audio/audio_classification.py` (754 baris) | `docs/agent_index/audio_classification.md` |

Belum ada template untuk: topic modeling / clustering teks, association rules (market basket), klasifikasi deret waktu/sinyal, recommender/ranking, geospasial, graph, kemiripan teks/retrieval/RAG, seq2seq (terjemahan/ringkasan), kemiripan gambar/matching, image-to-image, rotated box/keypoint/instance segmentation/video, ASR, multimodal, reinforcement learning. Image regression & multilabel gambar: hanya kode contekan di 04_Computer_Vision/CV_OTHER_TASKS_GUIDE.md. Survival & causal/uplift: hanya di dalam template churn.

## Cara menjalankan (semua template kecuali churn)

```bash
python tools/new_task.py ../kerja --task <task> --format ipynb   # atau --format py / both
# copy data panitia ke ../kerja/data/, isi ../kerja/task_config.json
python tools/run_task.py --task <task> --format py --config ../kerja/task_config.json --set RUN_MODE=fast
python tools/run_task.py --task <task> --format py --config ../kerja/task_config.json --set RUN_MODE=full
```

- Hasil: `../kerja/outputs/<nama template>/` → `submissions/`, `figures/` (bernomor), `tables/`, `summary.md`,
  `run_summary.json`. Format `py` menulis log ke `run_log.txt` (baca bagian akhirnya untuk error).
- Churn/asuransi memakai alur khusus: `python tools/new_case.py <folder> --team <Tim> --casebook <soal.pdf> [--format py]`
  lalu `RUN_FAST.bat` / `RUN_FULL.bat` (lihat `MULAI_DI_SINI.md`).
- Data latihan untuk mencoba: `python tools/make_synthetic_task_data.py --task <task> --out <folder>`.

## Konvensi yang sama di semua template

| Hal | Aturan |
|---|---|
| Setting | Semua di `class CFG` (lihat baris CFG di index). Ubah di cell `# >>> USER OVERRIDE` atau `task_config.json` |
| Prioritas | `task_config.json` / env `TEMPLATE_CONFIG` **menimpa** USER OVERRIDE. Key JSON berawalan `_` diabaikan; list JSON → tuple otomatis kalau default-nya tuple |
| Mode | `RUN_MODE="fast"` (subsample, epoch/fold sedikit — untuk cek) lalu `"full"` (hasil final) |
| Install | Template memasang paket yang hilang (`AUTO_INSTALL`). Env `TEMPLATE_AUTO_INSTALL=0` atau `run_task.py --no-install` mematikannya. Torch tidak pernah di-install otomatis |
| Fallback | Library opsional hilang → jalur itu di-skip dengan pesan, notebook tetap jalan (mis. tanpa smp → torchvision → U-Net kecil) |
| pandas 3 | Semua template memanggil `pd.set_option("future.infer_string", False)` (perilaku teks = pandas 2) |
| Validasi | Selalu cross-validation/holdout; isi `GROUP_COL` kalau ada grup yang tidak boleh bocor antar fold |
| Laporan | `summary.md` berisi angka utama + kalimat draft (English) siap parafrase; figure bernomor `figNN_*.png` |

## Tool

| File | Fungsi |
|---|---|
| `QA/make_walkthrough.py` | Rekam proses end-to-end sebuah case menjadi video walkthrough + folder screenshot (bukti semua fungsi bekerja). |
| `QA/qa_check.py` | Cek otomatis hasil sebuah case (tanpa AI): run sukses, artikel ≤ 10 halaman, isi case & riset tim benar-benar masuk artikel, caption/penanda tidak tertinggal,… |
| `QA/slop_check.py` | Audit tulisan "terasa AI" untuk paper (tanpa AI): kosakata & frasa khas AI, em dash, ekor -ing kosong, adverb pengisi, 'not only X but also Y', pembuka… |
| `tools/build_agent_index.py` | Buat peta repo untuk AI agent: AGENTS.md (root, ringkas, selalu dibaca agent) + docs/agent_index/<template>.md (per template: section + nomor baris di file .py, semua… |
| `tools/make_synthetic_insurance_churn.py` | Generate a realistic, deliberately MESSY synthetic insurance churn dataset for practice. |
| `tools/make_synthetic_task_data.py` | Buat data latihan kecil (sintetis) untuk setiap task — untuk mencoba template sebelum lomba dan untuk smoke test. |
| `tools/make_zip.py` | Buat ZIP pengumpulan sesuai aturan soal: "<TeamName>_Final Stage 1.zip" berisi artikel PDF, notebook (.ipynb) dan file pendukung. |
| `tools/nb_formats.py` | Dua format untuk setiap template: notebook (.ipynb) dan script Python (.py). |
| `tools/new_case.py` | Siapkan folder kerja untuk case baru (hari H) dalam 1 perintah. |
| `tools/new_task.py` | Siapkan folder kerja untuk satu task dalam 1 perintah, dengan format pilihan: notebook (.ipynb) atau script (.py). |
| `tools/research_helper.py` | Research helper — dari casebook → rencana riset + keyword + query siap-klik + workbook CASE_RESEARCH.xlsx. |
| `tools/run_case.py` | Headless runner for 01_Churn_Insurance/churn_insurance_master.ipynb (atau versi script .py-nya). |
| `tools/run_task.py` | Jalankan template apa pun secara headless (tanpa membuka Jupyter), dalam format notebook ATAU script. |
| `tools/setup_env.py` | Setup aman untuk komputer lain (mis. komputer panitia): semua paket dipasang ke virtual environment `.venv` di folder kerja (sebelah folder Notebook-Template). |
| `tools/smoke_test_templates.py` | Smoke test semua template: data sintetis kecil → jalankan template (RUN_MODE=fast) → cek selesai & menghasilkan output. |
| `tools/stress_test.py` | Stress test: run churn_insurance_master.ipynb on many deliberately different insurance-churn datasets and check that every run finishes WITHOUT section errors and… |
| `tools/task_registry.py` | Daftar semua template (task) di repo. |

## Dokumen (buka hanya kalau relevan)

| File | Buka kalau |
|---|---|
| `README.md` | gambaran umum repo, tabel soal → template, format .ipynb vs .py |
| `MULAI_DI_SINI.md` | alur hari-H churn/asuransi langkah demi langkah (download → upload) |
| `DOWNLOAD_DAN_INSTALASI.md` | instalasi di komputer lain / tanpa internet / masalah Python |
| `Guides/CHEATSHEET_METRIK_MODEL_VALIDASI.md` | memilih metrik, model, skema validasi untuk jenis soal apa pun |
| `Guides/DEBUGGING_TANPA_AI.md` | error umum & cara membaca traceback |
| `01_Churn_Insurance/00_MULAI_DI_SINI.md` | runbook detail churn + tabel 'kalau ada masalah' |
| `01_Churn_Insurance/CASE_SCENARIOS.md` | bentuk data churn aneh (label terbalik, file terpisah, data rusak) → setting |
| `01_Churn_Insurance/PLAYBOOK_ANALISIS.md` | aturan 'jika X maka Y', istilah asuransi, pertanyaan juri |
| `01_Churn_Insurance/PANDUAN_LAPORAN.md` | struktur artikel 10 halaman & template kalimat |
| `01_Churn_Insurance/BAHAN_LATAR_BELAKANG.md` | fakta industri (OJK/AAJI) & literatur terverifikasi |
| `01_Churn_Insurance/RESEARCH_KIT.md` | cara riset latar belakang dari casebook |
| `04_Computer_Vision/CV_OTHER_TASKS_GUIDE.md` | kode contekan image regression & multilabel gambar (bukan template) |
| `QA/TEMPLATE_SMOKE_REPORT.md` | bukti setiap template jalan + apa yang belum teruji |
| `QA/QA_FINAL_REPORT.md` | bukti QA churn (stress test 44 varian, demo end-to-end) |

## Jebakan yang sering terjadi

- Kaggle *code competition*: path `/kaggle/input/...`, internet mati (`AUTO_INSTALL=False`, model pretrained dilampirkan
  sebagai Kaggle Dataset lalu isi path lokalnya), dan sering wajib `submission.parquet` → tambahkan cell terakhir
  `pd.read_csv(<folder output>/submissions/submission.csv).to_parquet("submission.parquet", index=False)`
  (baris penulisan submission tiap template ada di index-nya).
- `nlp_classification` default-nya untuk teks Indonesia informal (lowercase, normalisasi slang, buang tanda baca).
  Untuk teks Inggris / gaya tulisan: `CLASSICAL_LOWERCASE=False`, `USE_SLANG_NORMALIZATION=False`,
  `REDUCE_ELONGATION=False`, `KEEP_HASHTAG_TEXT=False`.
- Nilai `null` di `task_config.json` tetap menimpa CFG → hapus key-nya kalau ingin auto-detect / USER OVERRIDE.
- Format `.ipynb` headless butuh `nbclient`; tanpa itu pakai `--format py`.
- Bukti uji & batasan (library yang hanya teruji lewat fallback): `QA/TEMPLATE_SMOKE_REPORT.md`.
