"""
Buat ZIP pengumpulan sesuai aturan soal: "<TeamName>_Final Stage 1.zip" berisi artikel PDF, notebook (.ipynb) dan file pendukung.

    python tools/make_zip.py --case ../case --pdf "../case/Warkab_Final Stage 1.pdf"
    python tools/make_zip.py --case ../case            # PDF default: outputs/churn_insurance/<Team>_Final Stage 1.pdf

Isi ZIP:
    <Team>_Final Stage 1.pdf          ← artikel final (yang sudah kamu edit!)
    <Team>_churn_analysis.ipynb       ← notebook ter-eksekusi (syntax + output)
      — atau, kalau case dijalankan dengan --format py: <Team>_churn_analysis.py + run_log.txt (output console)
    supporting/                       ← figures, tables, report_tables.xlsx, insights, model card, dashboard, submission, dll.
"""
import argparse
import glob
import json
import os
import sys
import zipfile

for _st in (sys.stdout, sys.stderr):  # console Windows (cp1252) tidak bisa mencetak emoji → paksa UTF-8
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


INTERNAL_FILES = {"JUDGE_QA.md", "NARRATIVE_OPTIONS.md", "REPORT_TODO.md", "SENTUHAN_MANUSIA.md", "QA_CHECK.md", "SLOP_CHECK.md",
                  }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=".", help="folder kerja case (berisi case_config.json)")
    ap.add_argument("--pdf", default=None, help="PDF artikel FINAL")
    ap.add_argument("--team", default=None)
    ap.add_argument("--include-models", action="store_true", help="ikut sertakan folder models/ (besar)")
    a = ap.parse_args()
    case = os.path.abspath(a.case)
    cfg = {}
    cp = os.path.join(case, "case_config.json")
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            cfg = json.loads("\n".join(l for l in f.read().splitlines() if not l.strip().startswith("//")))
    team = a.team or cfg.get("TEAM_NAME", "TeamName")
    out = cfg.get("OUTPUT_DIR", "./outputs/churn_insurance")
    out = out if os.path.isabs(out) else os.path.normpath(os.path.join(case, out))
    base = f"{team}_Final Stage 1"
    pdf = a.pdf or os.path.join(out, f"{base}.pdf")
    if not os.path.exists(pdf):
        raise SystemExit(f"PDF tidak ditemukan: {pdf}  → pakai --pdf <path PDF final>")
    nbs = [p for p in glob.glob(os.path.join(case, "*.ipynb")) if "checkpoint" not in p]
    scripts = []
    if not nbs:                                      # case dijalankan dengan --format py
        scripts = sorted(glob.glob(os.path.join(case, "*_analysis.py")))
        if scripts and os.path.exists(os.path.join(case, "run_log.txt")):
            scripts.append(os.path.join(case, "run_log.txt"))
    zpath = os.path.join(case, f"{base}.zip")
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pdf, f"{base}.pdf")
        for nb in nbs + scripts:
            z.write(nb, os.path.basename(nb))
        for root, _, files in os.walk(out):
            rel_root = os.path.relpath(root, out)
            if rel_root.startswith("models") and not a.include_models:
                continue
            for fn in files:
                if fn.endswith(".zip") or fn.startswith(base) or fn.startswith("_") or fn.startswith("REPORT_DRAFT"):
                    continue
                if fn in INTERNAL_FILES:            # catatan persiapan tim — jangan sampai terbaca juri
                    continue
                z.write(os.path.join(root, fn), os.path.join("supporting", os.path.relpath(os.path.join(root, fn), out)))
                n += 1
    print(f"✅ {zpath}  ({os.path.getsize(zpath) / 1e6:.1f} MB) — artikel: {os.path.basename(pdf)}, notebook: {[os.path.basename(x) for x in nbs]}, {n} file pendukung")
    if scripts:
        print(f"ℹ️ Kode dikumpulkan sebagai script: {[os.path.basename(x) for x in scripts]}. "
              "Kalau aturan lomba WAJIB .ipynb → jalankan ulang dengan format ipynb (RUN_*.bat default) atau "
              "`python tools/nb_formats.py to-ipynb <script>.py` lalu Run All di Jupyter.")
    elif not nbs:
        print("⚠️ Tidak ada .ipynb di folder case — pastikan notebook ikut (aturan: syntax wajib dikumpulkan).")


if __name__ == "__main__":
    main()
