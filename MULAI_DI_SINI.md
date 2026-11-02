# ▶ MULAI DI SINI: dari download sampai upload (±3 jam, satu orang)

File ini satu-satunya panduan yang perlu diikuti, berurutan dari atas ke bawah. File lain hanya dibuka ketika langkah di bawah menyuruh.
Semua perintah diketik di terminal (Command Prompt) yang dibuka di **folder kerja**, misalnya `D:\lomba\`.

---

## A. Persiapan komputer (±15 menit)

**A1. Download & ekstrak.** Buka DOI Zenodo → **Files** → **Download** ZIP versi terbaru. Ekstrak ke `D:\lomba\`, lalu ganti nama folder hasil ekstrak menjadi **`Notebook-Template`**.

**A2. Double-click `Notebook-Template\SETUP.bat`.** Tunggu sampai muncul `✅ SETUP SELESAI` (±8–12 menit, butuh internet).
- SETUP memasang semua paket ke folder `D:\lomba\.venv`.
- Tidak butuh admin dan tidak mengubah Python milik komputer panitia.
- Kalau ada masalah (Python tidak ada, internet diblokir): baca `Notebook-Template\DOWNLOAD_DAN_INSTALASI.md` bagian 4 dan 6.

**A3. Buka terminal di `D:\lomba\`.** Caranya: klik address bar Explorer di folder itu, ketik `cmd`, lalu Enter.

---

## B. Siapkan case (±15 menit)

**B1. Buat folder case.** Simpan dulu soal/casebook dari panitia di `D:\lomba\` (misalnya `casebook.pdf`), lalu jalankan:

```
.venv\Scripts\python Notebook-Template\tools\new_case.py case --casebook "casebook.pdf" --team "NamaTim" --members "Nama 1;Nama 2"
```

Tanpa casebook PDF/DOCX, hapus bagian `--casebook "..."`.
Mau menjalankan versi **script (.py)** alih-alih notebook? Tambahkan `--format py` (RUN_*.bat lalu menjalankan `NamaTim_churn_analysis.py`, output console di `run_log.txt`). Default `.ipynb` paling aman kalau aturan lomba mewajibkan notebook.

**B2. Copy semua file data panitia ke `case\data\`.** Format apa pun bisa: csv, xlsx, json, parquet, data kotor sekalipun.

**B3. Isi form soal.** Buka `case\CASE_INTAKE.html` di browser, isi dari soal (nama kolom target & ID, metrik, nama perusahaan, konteks dalam English, pertanyaan soal, parameter bisnis, budget, jumlah nasabah). Klik **Download JSON**, lalu simpan dan timpa `case\case_config.json`.
- Tidak tahu nama kolom target/ID? Kosongkan saja; notebook mendeteksi otomatis.
- Soal tidak memberi margin/biaya? Kosongkan; laporan menulisnya sebagai asumsi + analisis sensitivitas.

---

## C. Run cepat untuk cek (±10 menit)

**C1. Double-click `case\RUN_FAST.bat`.** Selama berjalan, lanjut ke langkah D1.

**C2. Setelah selesai, cek 3 hal** di `case\outputs\churn_insurance\run_summary.json` atau di tampilan console:
- `target_mapping`: nilai churn harus = 1;
- `leaky_cols`: kolom yang terjadi *setelah* churn (tanggal batal, alasan batal) sudah dibuang;
- `section errors`: kosong.

Kalau ada yang salah, perbaiki `case_config.json` (misalnya `TARGET_COL`, `POSITIVE_LABEL`, `ID_COL`) lalu ulangi C1. Kalau notebook berhenti, pesannya menyebut setting yang harus diubah.

---

## D. Riset & sentuhan manusia (±50 menit)

**D1. Riset latar belakang.** Buka `case\RESEARCH_PLAN.html`: berisi keyword, fakta penting dari casebook, dan link pencarian siap klik. Cari 3–6 sumber (jurnal, OJK/AAJI/AAUI), lalu isi `case\CASE_RESEARCH.xlsx`:
- `case_facts`: angka/fakta dari casebook (English);
- `industry_facts` dan `literature`: temuan + sitasi APA (`use = Y`; `verified = Y` setelah sumbernya kamu buka sendiri);
- `feature_labels`: nama kolom → nama yang enak dibaca di paper.

Contoh workbook yang sudah terisi: `Notebook-Template\examples\mock_case_warkab\CASE_RESEARCH.xlsx`. Metode riset lengkap: `Notebook-Template\01_Churn_Insurance\RESEARCH_KIT.md`.

**D2. Sentuhan manusia.** Buka `case\outputs\churn_insurance\SENTUHAN_MANUSIA.md` (dibuat oleh RUN_FAST). Isinya 5 blok draf paragraf yang angkanya sudah terisi:
1. suara aktuaria;
2. cerita iterasi model;
3. ambang operasional & governance;
4. insight lokal per driver;
5. checklist guidebook.

Ganti setiap `[ISI DARI CASE: …]` dengan fakta casebook. Kalimat yang faktanya tidak ada di case dihapus, jangan dikarang. Tempel paragraf yang sudah jadi ke `CASE_RESEARCH.xlsx` → sheet **`custom_paragraphs`** (`use = Y`, kolom `section` dan `position` sesuai yang tertulis di tiap blok).

Simpan dan **tutup Excel**.

---

## E. Run final (±45–60 menit)

**E1. Double-click `case\RUN_FULL.bat`.** Riset, paragraf tim, dan konteks soal otomatis masuk ke paper.

Selama menunggu:
- buka `NARRATIVE_OPTIONS.md` (pilihan judul/framing) dan `JUDGE_QA.md` (latihan pertanyaan juri) di `case\outputs\churn_insurance\`;
- baca guidebook lomba: wajib Bahasa Indonesia? Perlu lembar orisinalitas / deklarasi AI? (template ada di SENTUHAN_MANUSIA.md blok 5).

Waktu mepet? Lewati E1 dan pakai hasil RUN_FAST, karena paper-nya sudah lengkap. Tapi double-click `RUN_FAST.bat` sekali lagi setelah D2, supaya riset dan paragrafmu ikut masuk.

---

## F. Finalisasi paper (±40 menit)

**F1. Buka `case\outputs\churn_insurance\NamaTim_Final Stage 1.docx` di Word.**
- Parafrase dengan gaya tim, terutama executive summary dan discussion.
- Pastikan angka di teks = angka di tabel.
- Hapus penanda [EDIT] kalau ada. Jangan ubah format halaman.

Edit di Word dikerjakan **paling akhir**. Kalau RUN_FULL/RUN_FAST dijalankan lagi, DOCX ditimpa dan editanmu hilang. Untuk perubahan isi yang besar, tambahkan lewat `custom_paragraphs` lalu run ulang.

**F2. Save As PDF** dengan nama `NamaTim_Final Stage 1.pdf` di folder `case\`.

**F3. Cek gaya tulisan** (skor harus ≥ 80):

```
.venv\Scripts\python Notebook-Template\QA\slop_check.py "case\NamaTim_Final Stage 1.pdf"
```

---

## G. Kumpulkan (±10 menit)

**G1. Buat ZIP pengumpulan.** ZIP berisi PDF + notebook ter-eksekusi + file pendukung. Catatan internal (JUDGE_QA, NARRATIVE_OPTIONS, SENTUHAN_MANUSIA) otomatis tidak ikut.

```
.venv\Scripts\python Notebook-Template\tools\make_zip.py --case case --pdf "case\NamaTim_Final Stage 1.pdf"
```

**G2. Cek akhir.** Harus muncul `✅ semua cek wajib PASS`.

```
.venv\Scripts\python Notebook-Template\QA\qa_check.py --case case
```

**G3. Upload** `case\NamaTim_Final Stage 1.zip` (dan PDF-nya bila diminta terpisah).

---

## Kalau ada masalah

| Masalah | Lihat |
|---|---|
| Install / komputer panitia | `Notebook-Template\DOWNLOAD_DAN_INSTALASI.md` |
| Bentuk data aneh, notebook berhenti, target salah | `Notebook-Template\01_Churn_Insurance\CASE_SCENARIOS.md` dan `00_MULAI_DI_SINI.md` (tabel "Kalau ada masalah") |
| Cara menulis bagian paper | `Notebook-Template\01_Churn_Insurance\PANDUAN_LAPORAN.md` |
| Istilah asuransi, aturan "jika X maka Y", pertanyaan juri | `Notebook-Template\01_Churn_Insurance\PLAYBOOK_ANALISIS.md` |
| Fakta industri & literatur terverifikasi | `Notebook-Template\01_Churn_Insurance\BAHAN_LATAR_BELAKANG.md` |
| Contoh hasil akhir (paper, dashboard, video proses) | `Notebook-Template\QA\demo_warkab\` dan `Notebook-Template\QA\walkthrough_warkab\walkthrough_Warkab.mp4` |

## Ringkasan urutan file yang dibuka

1. `SETUP.bat`
2. terminal → `new_case.py`
3. `case\CASE_INTAKE.html`
4. `case\RUN_FAST.bat`
5. `case\RESEARCH_PLAN.html`
6. `case\CASE_RESEARCH.xlsx`
7. `case\outputs\churn_insurance\SENTUHAN_MANUSIA.md`
8. `case\RUN_FULL.bat`
9. `NARRATIVE_OPTIONS.md` / `JUDGE_QA.md`
10. DOCX di Word → PDF
11. `slop_check` → `make_zip` → `qa_check`
12. upload
