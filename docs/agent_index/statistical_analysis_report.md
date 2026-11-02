# statistical_analysis_report — `--task stats`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

Analisis statistik & laporan insight (uji hipotesis, A/B test, regresi + CI) tanpa lomba prediksi

| | |
|---|---|
| Baca | `09_Statistical_Analysis/statistical_analysis_report.py` (775 baris) — notebook kembarannya: `09_Statistical_Analysis/statistical_analysis_report.ipynb` |
| `class CFG` | baris 55 · cell `# >>> USER OVERRIDE` baris 81 |
| Data yang diharapkan | data/data.csv (satu tabel; boleh xlsx/parquet) |
| Contoh `task_config.json` | `{"DATA_PATH": "./data/data.csv", "OUTCOME_COL": null, "GROUP_COL": null}` |
| Folder output | `./outputs/statistical_analysis_report/` |
| Submission ditulis | - (template ini menghasilkan laporan/tabel, bukan submission) |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
| Statistical Analysis & Insight Report — tanpa lomba prediksi | 27 |
| ↳ Pipeline | 34 |
| 1. Install & import | 125 |
| 2. Load & kualitas data | 221 |
| 3. Statistik deskriptif | 265 |
| 4. Uji hubungan dengan OUTCOME (pemilihan uji otomatis) | 291 |
| ↳ 4b. Outcome per kategori (dengan 95% CI) | 417 |
| 5. A/B test (`GROUP_COL`) | 461 |
| 6. Regresi multivariabel (efek "setelah mengontrol variabel lain") | 538 |
| 7. Segmen & korelasi | 638 |
| 8. Insight & laporan (Markdown + DOCX) | 688 |
| 9. Troubleshooting & checklist | 761 |

## Semua key CFG (23)

| Key | Default | Keterangan | Baris |
|---|---|---|---|
| `RUN_MODE` | `"full"` |  | 56 |
| `SEED` | `42` |  | 57 |
| `NOTEBOOK_NAME` | `"statistical_analysis_report"` |  | 58 |
| `AUTO_INSTALL` | `True (env TEMPLATE_AUTO_INSTALL=0 → False)` |  | 59 |
| `DATA_PATH` | `"./data/data.csv"` |  | 60 |
| `SHEET_NAME` | `0` | Excel | 61 |
| `ID_COL` | `None` |  | 62 |
| `OUTCOME_COL` | `None` | variabel hasil utama (angka atau 0/1/kategori); None = deskriptif + korelasi saja | 63 |
| `GROUP_COL` | `None` | variabel perlakuan/kelompok untuk A/B test (mis. "group" berisi A/B) | 64 |
| `CONTROL_LABEL` | `None` | nilai grup kontrol (None = nilai pertama setelah diurutkan) | 65 |
| `SEGMENT_COLS` | `None` | kolom segmen untuk efek per subgrup (None = kategorikal ≤ 8 level) | 66 |
| `DROP_COLS` | `[]` |  | 67 |
| `ALPHA` | `0.05` |  | 68 |
| `MAX_CATEGORIES` | `20` | kategori lebih banyak → digabung "Other" | 69 |
| `N_BOOTSTRAP` | `2000` |  | 70 |
| `EXPECTED_SPLIT` | `None` | A/B: rasio alokasi yang direncanakan, mis. [0.5, 0.5] (cek SRM) | 71 |
| `MDE` | `None` | minimum detectable effect untuk power analysis (None = efek yang teramati) | 72 |
| `REPORT_TITLE` | `"Statistical Analysis Report"` |  | 73 |
| `TEAM_NAME` | `"Team"` |  | 74 |
| `WRITE_DOCX` | `True` |  | 75 |
| `OUTPUT_DIR` | `"./outputs/statistical_analysis_report/"` |  | 76 |
| `FAST_N_BOOTSTRAP` | `300` |  | 77 |
| `FIG_DPI` | `150` |  | 78 |
