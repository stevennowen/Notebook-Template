"""
Smoke test semua template: data sintetis kecil → jalankan template (RUN_MODE=fast) → cek selesai & menghasilkan output.

    python tools/smoke_test_templates.py                                  # semua task, format py
    python tools/smoke_test_templates.py --tasks tabular segmentation     # sebagian
    python tools/smoke_test_templates.py --format ipynb                   # lewat nbclient (butuh nbclient)
    python tools/smoke_test_templates.py --out D:/tmp/smoke --allow-install

Default: tanpa pip install (TEMPLATE_AUTO_INSTALL=0) dan tanpa download model HuggingFace (HF_HUB_OFFLINE=1),
supaya yang diuji adalah "apa yang terjadi di komputer yang librarynya tidak lengkap". Library yang tidak ada →
template harus memakai fallback, bukan crash.

Hasil: QA/TEMPLATE_SMOKE_REPORT.md (PASS = exit code 0 dan file submission/hasil utama terbentuk).
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_synthetic_task_data as SYN  # noqa: E402
import task_registry as REG  # noqa: E402

# Override khusus smoke test: model kecil/sudah ter-cache, epoch sedikit — HANYA untuk membuktikan alur jalan.
SMOKE_OVERRIDES = {
    "tabular": {},
    "tabular-multilabel": {},
    "nlp-classification": {"EMBEDDING_MODELS": ["sentence-transformers/all-MiniLM-L6-v2"],
                           "TRANSFORMER_MODELS": ["bert-base-uncased"], "MAX_LEN": 64},
    "nlp-regression": {"EMBEDDING_MODELS": ["sentence-transformers/all-MiniLM-L6-v2"],
                       "TRANSFORMER_MODELS": ["bert-base-uncased"], "MAX_LEN": 64},
    "cv-classification": {},
    "time-series": {"HORIZON": 28},
    "clustering": {},
    "segmentation": {"PRETRAINED": False},
    "detection": {"PRETRAINED": False, "AMP": False, "BATCH_SIZE": 8},
    "ocr": {},
    "anomaly-tabular": {},
    "anomaly-image": {"PRETRAINED": False},
    "audio": {},
    "ner": {"TRANSFORMER_MODELS": ["bert-base-uncased"]},
    "llm": {"LLM_MODEL": os.environ.get("SMOKE_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")},   # model lokal: set env SMOKE_LLM_MODEL
    "stats": {},
}
# task smoke → task registry (template yang dijalankan)
TEMPLATE_OF = {}
# Catatan yang ikut ditulis di laporan (apa yang TIDAK dibuktikan oleh smoke test ini)
SMOKE_NOTES = {
    "segmentation": "PRETRAINED=False (tanpa download bobot) → membuktikan alur, bukan akurasi terbaik.",
    "detection": "YOLO dari nol (PRETRAINED=False, 3 epoch) → mAP rendah wajar; alur & format submission yang diuji.",
    "anomaly-image": "Backbone berbobot acak (PRETRAINED=False).",
    "llm": "Kalau SMOKE_LLM_MODEL menunjuk LLM mini berbobot acak: yang diuji mekanik (prompt, skor label, kalibrasi, probe, "
           "submission), bukan kualitas jawaban LLM.",
    "nlp-classification": "Transformer & embedding memakai model yang sudah ada di cache (offline).",
    "nlp-regression": "Transformer & embedding memakai model yang sudah ada di cache (offline).",
    "ner": "Transformer memakai model yang sudah ada di cache (offline).",
}

ERR_PAT = re.compile(r"Traceback \(most recent call last\)|❌|\bERROR\b|section error|gagal:", re.I)


def run_one(task, root, fmt, timeout, allow_install):
    reg_key = TEMPLATE_OF.get(task, task)
    folder = os.path.join(root, task)
    os.makedirs(folder, exist_ok=True)
    for old in glob.glob(os.path.join(folder, "*.ipynb")) + glob.glob(os.path.join(folder, "*.py")):
        os.remove(old)                                   # selalu uji versi terbaru dari repo
    cfg = SYN.make(task, folder)
    cfg.update(SMOKE_OVERRIDES.get(task, {}))
    cfg_path = os.path.join(folder, "smoke_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    cmd = [sys.executable, os.path.join(HERE, "run_task.py"), "--task", reg_key, "--format", fmt, "--config", cfg_path,
           "--workdir", folder, "--set", "RUN_MODE=fast", "--fresh-copy", "--quiet"]
    if not allow_install:
        cmd.append("--no-install")
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=folder, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=None if timeout <= 0 else timeout)
        code, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        code, out = 124, (e.stdout or "") if isinstance(e.stdout, str) else ""
    secs = time.time() - t0
    log = os.path.join(folder, "run_log.txt")
    log_text = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else out
    err_lines = [l.strip() for l in log_text.splitlines() if ERR_PAT.search(l)]
    outputs = [p for p in glob.glob(os.path.join(folder, "outputs", "**", "*"), recursive=True) if os.path.isfile(p)]
    subs = [p for p in outputs if re.search(r"submission|predictions?|anomaly_scores|report", os.path.basename(p), re.I)]
    figs = [p for p in outputs if p.endswith(".png")]
    status = "PASS" if code == 0 and subs else ("TIMEOUT" if code == 124 else "FAIL")
    tail = "\n".join(log_text.strip().splitlines()[-25:]) if status != "PASS" else ""
    return {"task": task, "template": REG.TASKS[reg_key]["path"], "status": status, "exit": code, "seconds": round(secs),
            "n_outputs": len(outputs), "n_figures": len(figs), "main_output": [os.path.relpath(s, folder) for s in subs[:3]],
            "log_warnings": len(err_lines), "first_warnings": err_lines[:4], "tail": tail}


def write_report(rows, path, fmt, allow_install):
    n_pass = sum(r["status"] == "PASS" for r in rows)
    lines = [f"# Smoke test template — {n_pass}/{len(rows)} PASS", "",
             f"Tanggal: {dt.date.today().isoformat()} · format: **{fmt}** · Python {sys.version.split()[0]} · "
             f"pip install: {'boleh' if allow_install else 'TIDAK (library yang hilang harus ditangani fallback)'}",
             "", "Setiap task: data sintetis kecil (`tools/make_synthetic_task_data.py`) → `tools/run_task.py --set RUN_MODE=fast`.",
             "PASS = selesai tanpa crash (exit code 0) **dan** file hasil utama (submission/prediksi/laporan) terbentuk.",
             "Kolom *peringatan log* = baris log berisi traceback/❌/gagal yang ditangani template (bukan crash) — cek bila > 0.",
             "", "| task | template | status | detik | figure | file output | hasil utama | peringatan log |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | `{r['template']}` | {r['status']} | {r['seconds']} | {r['n_figures']} | {r['n_outputs']} | "
                     f"{'<br>'.join(r['main_output']) or '-'} | {r['log_warnings']} |")
    notes = [(r["task"], SMOKE_NOTES[r["task"]]) for r in rows if r["task"] in SMOKE_NOTES]
    try:
        import importlib.util as _u
        missing = [m for m in ("xgboost", "catboost", "optuna", "shap", "statsmodels", "lifelines", "timm",
                               "segmentation_models_pytorch", "librosa", "soundfile", "peft", "nbclient")
                   if _u.find_spec(m) is None]
    except Exception:  # noqa
        missing = []
    if notes or missing:
        lines += ["", "## Catatan uji"]
        lines += [f"- **{t}**: {n}" for t, n in notes]
        if missing:
            lines.append(f"- Library tidak terpasang saat uji (jalur ini hanya teruji lewat fallback-nya): {', '.join(missing)}.")
    for r in rows:
        if r["first_warnings"] or r["tail"]:
            lines += ["", f"### {r['task']} — {r['status']}"]
            if r["first_warnings"]:
                lines += ["Peringatan pertama di log:", "```text", *[w[:220] for w in r["first_warnings"]], "```"]
            if r["tail"]:
                lines += ["Akhir log:", "```text", r["tail"][-3000:], "```"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--format", choices=["py", "ipynb"], default="py")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(REG.REPO), "_smoke_test"))
    ap.add_argument("--timeout", type=int, default=1800, help="per task (detik)")
    ap.add_argument("--allow-install", action="store_true")
    ap.add_argument("--report", default=os.path.join(REG.REPO, "QA", "TEMPLATE_SMOKE_REPORT.md"))
    a = ap.parse_args()
    tasks = a.tasks or [t for t in SYN.MAKERS if REG.available(TEMPLATE_OF.get(t, t))]
    rows = []
    for t in tasks:
        print(f"▶ {t} ...", flush=True)
        r = run_one(t, os.path.abspath(a.out), a.format, a.timeout, a.allow_install)
        rows.append(r)
        print(f"  {r['status']} | {r['seconds']}s | output {r['n_outputs']} | peringatan log {r['log_warnings']}", flush=True)
        if r["status"] != "PASS":
            print("  " + r["tail"].replace("\n", "\n  ")[-2000:])
    if not a.tasks or len(a.tasks) > 1 or a.report != ap.get_default("report"):
        write_report(rows, a.report, a.format, a.allow_install)
        print("📄", a.report)
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
