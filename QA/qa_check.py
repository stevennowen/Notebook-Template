"""
Cek otomatis hasil sebuah case (tanpa AI): run sukses, artikel ≤ 10 halaman, isi case & riset tim benar-benar masuk artikel,
caption/penanda tidak tertinggal, dashboard & ZIP lengkap.

    python QA/qa_check.py --case <folder case>            → cetak checklist + tulis <case>/QA_CHECK.md

Exit code 0 = semua cek wajib PASS, 1 = ada FAIL (baca baris FAIL → perbaiki → double-click RUN_FAST/RUN_FULL.bat lagi).
"""
import argparse
import glob
import json
import os
import re
import sys
import zipfile

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_cfg(case):
    p = os.path.join(case, "case_config.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.loads("\n".join(l for l in f.read().splitlines() if not l.strip().startswith("//")) or "{}")


def norm(t):
    return re.sub(r"\s+", " ", str(t)).replace("- ", "-").strip().lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--max-pages", type=int, default=10)
    a = ap.parse_args()
    case = os.path.abspath(a.case)
    cfg = load_cfg(case)
    out = os.path.join(case, cfg.get("OUTPUT_DIR", "outputs/churn_insurance")) if not os.path.isabs(cfg.get("OUTPUT_DIR", "")) else cfg["OUTPUT_DIR"]
    team = str(cfg.get("TEAM_NAME", "TeamName"))
    rows = []

    def chk(name, ok, detail="", required=True):
        rows.append((("PASS" if ok else ("FAIL" if required else "WARN")), name, detail))

    # 1. run
    summ = os.path.join(out, "run_summary.json")
    s = json.load(open(summ, encoding="utf-8")) if os.path.exists(summ) else {}
    chk("Run selesai (run_summary.json)", bool(s), summ if s else "tidak ada → jalankan RUN_FAST/RUN_FULL")
    if s:
        errs = s.get("section_errors") or {}
        chk("Tanpa section error", not errs, "; ".join(f"{k}: {v[:80]}" for k, v in errs.items()) or "0 error")
        chk("Target churn = 1 ter-mapping", 1 in (s.get("target_mapping") or {}).values(), str(s.get("target_mapping")))
        chk("Skor model tercatat", s.get("final_oof_auc") is not None, f"{s.get('final_model')} | OOF AUC {s.get('final_oof_auc', float('nan')):.4f}")

    # 2. article
    # PDF final hasil edit tim (di folder case) diprioritaskan; kalau belum ada → PDF otomatis di outputs
    pdfs = glob.glob(os.path.join(case, "*Final Stage 1.pdf")) or glob.glob(os.path.join(out, "*Final Stage 1.pdf"))
    docx = glob.glob(os.path.join(out, "*Final Stage 1.docx"))
    chk("Artikel DOCX", bool(docx), docx[0] if docx else "")
    chk("Artikel PDF", bool(pdfs), pdfs[0] if pdfs else "PDF tidak ada (Word/LibreOffice tidak tersedia?) → Save As PDF manual", required=False)
    body = full = ""
    if pdfs:
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz
            d = fitz.open(pdfs[0])
            pages = [p.get_text() for p in d]
            app = next((i for i, t in enumerate(pages) if re.search(r"^\s*Appendices\s*$", t, re.M)), len(pages))
            n_body = app - 1                      # halaman 1 = cover (tidak dihitung)
            chk(f"Isi utama ≤ {a.max_pages} halaman (cover & appendix tidak dihitung)", n_body <= a.max_pages, f"{n_body} halaman isi, total PDF {len(pages)}")
            body, full = norm(" ".join(pages[:app])), norm(" ".join(pages))
        except ImportError:
            chk("PyMuPDF untuk cek PDF", False, "python -m pip install pymupdf", required=False)
    if body:
        chk(f"Nama tim '{team}' di cover", norm(team) in norm(full[:3000]))
        for m in cfg.get("TEAM_MEMBERS") or []:
            chk(f"Anggota '{m}' di cover", norm(m) in full[:3000])
        if cfg.get("COMPANY_NAME"):
            chk("Nama perusahaan dari soal dipakai", norm(cfg["COMPANY_NAME"]) in body, cfg["COMPANY_NAME"])
        if cfg.get("CASE_CONTEXT"):
            chk("Konteks soal (CASE_CONTEXT) masuk Introduction", norm(cfg["CASE_CONTEXT"])[:80] in body)
        for q in (cfg.get("CASE_QUESTIONS") or [])[:6]:
            chk(f"Pertanyaan soal dijawab: '{q[:50]}…'", norm(q)[:40] in full, required=False)
        if cfg.get("PORTFOLIO_SIZE"):
            chk("Skala portofolio dipakai", f"{int(cfg['PORTFOLIO_SIZE']):,}" in body, f"{int(cfg['PORTFOLIO_SIZE']):,}")
        if cfg.get("RETENTION_BUDGET"):
            chk("Budget soal dibahas", "budget" in body)
        if str(cfg.get("BUSINESS_PARAMS_SOURCE", "")).lower() == "case":
            chk("Parameter bisnis ditulis 'given in the case'", "given in the case" in full or "parameters given" in full, required=False)
        for f_ in cfg.get("CASE_KEY_FACTS") or []:
            chk(f"Fakta casebook masuk: '{f_[:50]}…'", norm(f_)[:60] in body)
        # research workbook
        rx = cfg.get("RESEARCH_FILE") or os.path.join(case, "CASE_RESEARCH.xlsx")
        rx = rx if os.path.isabs(rx) else os.path.join(case, rx)
        if os.path.exists(rx):
            try:
                import pandas as pd
                book = pd.read_excel(rx, sheet_name=None)
                for sh, col in [("case_facts", "statement_EN"), ("industry_facts", "statement_EN"), ("literature", "finding_EN"), ("custom_paragraphs", "text")]:
                    df = book.get(sh)
                    if df is None or col not in df:
                        continue
                    use = df[df.get("use", "").astype(str).str.strip().str.upper().eq("Y")] if "use" in df else df
                    for _, r in use.iterrows():
                        t = str(r[col]).strip()
                        if t and t.lower() != "nan":
                            hit = norm(t)[:55] in full
                            chk(f"Riset [{sh}] masuk artikel: '{t[:45]}…'", hit, "" if hit else "tidak ditemukan (mungkin dipangkas autofit / tema tidak cocok driver)", required=(sh != "literature"))
                    if sh in ("literature", "industry_facts") and "verified" in use:
                        unv = int((~use["verified"].astype(str).str.strip().str.upper().eq("Y")).sum())
                        chk(f"Semua sumber [{sh}] verified = Y", unv == 0, f"{unv} belum diverifikasi", required=False)
            except Exception as e:
                chk("Baca CASE_RESEARCH.xlsx", False, str(e)[:120], required=False)
        raw_caps = re.findall(r"(?:figure|table) [a-l]?\.?\d+\. ([a-z0-9]+_[a-z0-9_]+)\b", full)
        chk("Tidak ada caption mentah (nama file)", not raw_caps, ", ".join(sorted(set(raw_caps))[:6]))
        chk("Tidak ada penanda [EDIT] tersisa", "[edit]" not in full, "cari [EDIT] di Word (highlight kuning)", required=False)
        chk("Tidak ada 'nan'/'None' di teks isi", not re.search(r"\b(nan|none)\b(?![-\w])", body.replace("none of", "").replace("nonetheless", "")), required=False)
        chk("Daftar pustaka ada", "references" in body)
        try:                                                       # gaya tulisan: indikator tulisan "terasa AI"
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import slop_check
            F_, score_, *_ = slop_check.audit(slop_check.read_text(pdfs[0], False))
            chk("Gaya tulisan tidak terasa AI (slop_check ≥ 80)", score_ >= 80,
                f"skor {score_:.0f}/100, {len(F_)} temuan → python QA/slop_check.py \"{os.path.basename(pdfs[0])}\"", required=False)
        except Exception as e:
            chk("slop_check", False, str(e)[:100], required=False)

    # 3. other deliverables
    _tp = cfg.get("TEST_PATH", "data/test.csv")
    has_test = bool(_tp) and os.path.exists(_tp if os.path.isabs(str(_tp)) else os.path.join(case, str(_tp)))
    if not has_test:
        chk("Tanpa file test → submission tidak dibuat (sesuai soal)", True, "soal tidak memberi data test", required=False)
    for fn, req in [("submission.csv", has_test), ("dashboard/index.html", True), ("NARRATIVE_OPTIONS.md", True), ("JUDGE_QA.md", True), ("REPORT_TODO.md", True)]:
        chk(f"Output {fn}", os.path.exists(os.path.join(out, fn)), required=req)
    dj = os.path.join(out, "dashboard", "data.js")
    if os.path.exists(dj):
        t = open(dj, encoding="utf-8").read()
        chk("Dashboard berisi data case ini", norm(cfg.get("COMPANY_NAME", team))[:20] in norm(t) or team in t)
        if cfg.get("DASHBOARD_INCLUDE_CUSTOMERS") is False:
            chk("Dashboard tanpa daftar nasabah (privasi)", '"customers": [' not in t or '"customers": []' in t)
    zips = glob.glob(os.path.join(case, "*Final Stage 1.zip"))
    chk("ZIP pengumpulan dibuat", bool(zips), zips[0] if zips else "python tools/make_zip.py --case <case> --pdf <pdf>", required=False)
    if zips:
        names = zipfile.ZipFile(zips[0]).namelist()
        has_pdf, has_nb = any(n.endswith(".pdf") for n in names), any(n.endswith(".ipynb") for n in names)
        has_script = any(n.endswith("_analysis.py") for n in names) and "run_log.txt" in names   # case --format py
        chk("ZIP berisi PDF + kode analisis (notebook ter-eksekusi, atau script + run_log)", has_pdf and (has_nb or has_script),
            f"{len(names)} file" + ("" if has_nb or not has_script else " · format script"))
        if has_script and not has_nb:
            chk("Kode dikumpulkan sebagai .ipynb (cek aturan lomba: wajib notebook?)", False,
                "jalankan ulang dengan format ipynb, atau: python tools/nb_formats.py to-ipynb <script>.py", required=False)

    n_fail = sum(r[0] == "FAIL" for r in rows)
    lines = [f"# QA CHECK — {team}", "", f"Case: `{case}`", "",
             f"**{sum(r[0] == 'PASS' for r in rows)} PASS · {n_fail} FAIL · {sum(r[0] == 'WARN' for r in rows)} WARN**", "",
             "| Status | Cek | Detail |", "|---|---|---|"]
    lines += [f"| {st} | {nm} | {str(dt).replace('|', '/')} |" for st, nm, dt in rows]
    with open(os.path.join(case, "QA_CHECK.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for st, nm, dt in rows:
        print(f"[{st}] {nm}" + (f" — {dt}" if dt else ""))
    print(f"\n{'✅ semua cek wajib PASS' if not n_fail else f'❌ {n_fail} cek wajib FAIL'} → {os.path.join(case, 'QA_CHECK.md')}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
