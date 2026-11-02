"""
Siapkan folder kerja untuk case baru (hari H) dalam 1 perintah.

    python tools/new_case.py ../case            # buat folder ../case
    python tools/new_case.py ../case --team Warkab --casebook soal.pdf
    python tools/new_case.py ../case --team Warkab --format py      # RUN_*.bat menjalankan versi script (.py)

Isi folder yang dibuat:
    data/                     ← taruh file dari panitia di sini
    case_config.json          ← isi dari soal (atau pakai CASE_INTAKE.html → Download JSON → timpa file ini)
    CASE_INTAKE.html          ← form pengisian config (buka di browser, offline)
    RUN_FAST.bat / RUN_FULL.bat (Windows) dan run_fast.sh / run_full.sh (Mac/Linux)
    NEXT_STEPS.md
"""
import argparse
import json
import os
import shutil
import sys

for _st in (sys.stdout, sys.stderr):  # console Windows (cp1252) tidak bisa mencetak emoji → paksa UTF-8
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--team", default="Warkab")
    ap.add_argument("--members", default="", help="anggota dipisah titik koma")
    ap.add_argument("--casebook", default=None, help="PDF/DOCX/MD/TXT soal → disalin ke folder case + dibaca research_helper")
    ap.add_argument("--format", choices=["ipynb", "py"], default="ipynb",
                    help="format yang dijalankan RUN_*.bat: ipynb (default, output tersimpan di notebook) atau py (script)")
    a = ap.parse_args()
    d = os.path.abspath(a.folder)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    with open(os.path.join(HERE, "case_config.example.json"), encoding="utf-8") as f:
        txt = "\n".join(l for l in f.read().splitlines() if not l.strip().startswith("//"))
    cfg = json.loads(txt)
    cfg["TEAM_NAME"] = a.team
    cfg["TEAM_MEMBERS"] = [m.strip() for m in a.members.split(";") if m.strip()]
    cfg_path = os.path.join(d, "case_config.json")
    if os.path.exists(cfg_path):
        print("case_config.json sudah ada → tidak ditimpa")
    else:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    shutil.copyfile(os.path.join(HERE, "CASE_INTAKE.html"), os.path.join(d, "CASE_INTAKE.html"))
    casebook = None
    if a.casebook:
        if not os.path.exists(a.casebook):
            print(f"⚠️ casebook tidak ditemukan: {a.casebook} → dilewati (bisa ditambahkan nanti, lihat NEXT_STEPS.md)")
        else:
            casebook = os.path.join(d, "casebook" + os.path.splitext(a.casebook)[1].lower())
            if os.path.abspath(a.casebook) != os.path.abspath(casebook):
                shutil.copyfile(a.casebook, casebook)
            print(f"Casebook disalin → {casebook}")
    try:                                             # research plan + CASE_RESEARCH.xlsx (dibuat ulang dengan --casebook setelah soal ada)
        sys.path.insert(0, HERE)
        import research_helper
        research_helper.run(d, casebook=casebook, quiet=True)
    except Exception as e:
        print("(research_helper dilewati:", e, ")")
    runner = os.path.join(HERE, "run_case.py")
    py = sys.executable
    for mode in ["fast", "full"]:
        with open(os.path.join(d, f"RUN_{mode.upper()}.bat"), "w", encoding="utf-8") as f:
            f.write(f'@echo off\r\nchcp 65001 >nul\r\nset PYTHONIOENCODING=utf-8\r\ncd /d "%~dp0"\r\n"{py}" "{runner}" --config case_config.json'
                    + f" --format {a.format}" + (" --set RUN_MODE=fast" if mode == "fast" else "") + "\r\npause\r\n")
        with open(os.path.join(d, f"run_{mode}.sh"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f'#!/bin/bash\ncd "$(dirname "$0")"\npython "{runner}" --config case_config.json --format {a.format}'
                    + (" --set RUN_MODE=fast" if mode == "fast" else "") + "\n")
    steps = f"""# Next steps — {a.team}

1. Copy semua file data dari panitia ke `data/`.
2. Buka `CASE_INTAKE.html` → isi dari soal → **Download JSON** → simpan sebagai `case_config.json` di folder ini (timpa).
   (Atau edit `case_config.json` langsung.)
3. Simpan PDF/DOCX soal sebagai `casebook.pdf` di folder ini → `python "{os.path.join(HERE, 'research_helper.py')}" --case .`
   → buka **RESEARCH_PLAN.html** (keyword + query siap-klik) & isi **CASE_RESEARCH.xlsx** (panduan: RESEARCH_KIT.md).
4. Double-click **RUN_FAST.bat** (±5–10 menit) → cek `outputs/churn_insurance/run_summary.json` & `REPORT_TODO.md`:
   target/ID benar? churn rate masuk akal? tidak ada section error?
5. Double-click **RUN_FULL.bat** (±30–90 menit; jalan di background).
6. Hasil di `outputs/churn_insurance/`:
   - `{a.team}_Final Stage 1.docx/.pdf` — artikel (≤10 halaman isi, appendix lengkap)
   - `NARRATIVE_OPTIONS.md` — alternatif judul/framing/nama persona/paragraf diskusi
   - `JUDGE_QA.md` — persiapan pertanyaan juri
   - `dashboard/` — buka `index.html`; deploy: `cd outputs/churn_insurance/dashboard` lalu `npx vercel --prod`
   - `submission.csv`, `tables/`, `figures/`, `report_tables.xlsx`, `model_card.md`
   - `{a.team}_churn_analysis.{a.format}` (di folder ini) — {"notebook ter-eksekusi lengkap dengan output" if a.format == "ipynb" else "script yang dijalankan; output console ada di run_log.txt"} (bukti analisis)
7. Setelah CASE_RESEARCH.xlsx (dan paragraf dari SENTUHAN_MANUSIA.md) terisi: double-click RUN_FULL.bat lagi (atau RUN_FAST.bat kalau waktu mepet, ±10 menit). Kalau notebook sedang terbuka di Jupyter dan sudah di-Run All di sesi itu, cukup jalankan ulang cell 24.2c–24.5 (±3 menit) → artikel memuat riset tim.
8. Edit artikel di Word (parafrase, pilih narasi) → Save As PDF `{a.team}_Final Stage 1.pdf`.
9. ZIP pengumpulan: `python "{os.path.join(HERE, 'make_zip.py')}" --case . --pdf "{a.team}_Final Stage 1.pdf"`
   (PDF final hasil edit kamu; tanpa --pdf → pakai draft otomatis).
10. Cek akhir otomatis: `python "{os.path.join(REPO, 'QA', 'qa_check.py')}" --case .` → semua cek wajib PASS (hasil di `QA_CHECK.md`).

Panduan lengkap: {os.path.join(REPO, '01_Churn_Insurance', '00_MULAI_DI_SINI.md')}
"""
    with open(os.path.join(d, "NEXT_STEPS.md"), "w", encoding="utf-8") as f:
        f.write(steps)
    print(f"✅ Folder case siap: {d}")
    print(steps)


if __name__ == "__main__":
    main()
