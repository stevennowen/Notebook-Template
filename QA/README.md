# QA — bukti template bekerja (Tim Warkab)

| File / folder | Isi |
|---|---|
| `ALUR_KERJA_HARI_H.md` | **Alur kerja dari soal diterima sampai paper & ZIP terkumpul** (jadwal satu orang ±3 jam, perintah per langkah) |
| `QA_FINAL_REPORT.md` | Hasil QA/QC final: stress test 44 bentuk data, bukti per jenis kerusakan, demo end-to-end, regresi, celah yang diperbaiki |
| `walkthrough_warkab/walkthrough_Warkab.mp4` | **Rekaman proses** (37 langkah, ±2,8 menit): casebook → folder case → intake → research plan → workbook riset → RUN_FULL → ringkasan run → halaman artikel → appendix → 7 tab dashboard → narrative/judge/todo → QA check → stress test → ZIP |
| `walkthrough_warkab/STEPS.md` + `screens/` | Langkah yang sama dalam bentuk screenshot (bisa dibaca tanpa memutar video) |
| `demo_warkab/` | Hasil nyata demo: `Warkab_Final Stage 1.pdf`, dashboard (buka `dashboard/index.html`), `QA_CHECK.md`, `RESEARCH_PLAN.html`, `NARRATIVE_OPTIONS.md`, `JUDGE_QA.md`, `REPORT_TODO.md`, `run_summary.json`, log console |
| `TEMPLATE_SMOKE_REPORT.md` | **Smoke test semua template** (16 task, data sintetis, format `.py`, tanpa pip install): status, waktu, file output, catatan apa yang tidak diuji. Ulangi: `python tools/smoke_test_templates.py` |
| `qa_check.py` | Cek otomatis artikel/case mana pun: `python QA/qa_check.py --case <folder case>` |
| `slop_check.py` | Audit gaya tulisan "terasa AI" pada paper (aturan skill ai-paraphrase, anti-ai-slop-writing, stop-slop untuk ragam akademik): `python QA/slop_check.py "<paper>.pdf"` → skor ≥ 80 |
| `make_walkthrough.py` | Membuat rekaman dari folder case mana pun: `python QA/make_walkthrough.py --case <folder> --out <folder rekaman> --stress tools/STRESS_TEST_REPORT.md` |

Hasil stress test lengkap: `../tools/STRESS_TEST_REPORT.md`. Daftar semua bentuk case & kerusakan: `../01_Churn_Insurance/CASE_SCENARIOS.md`.
