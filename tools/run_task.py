"""
Jalankan template apa pun secara headless (tanpa membuka Jupyter), dalam format notebook ATAU script.

    python tools/run_task.py --task tabular --config task_config.json                  # format default: ipynb
    python tools/run_task.py --task tabular --config task_config.json --format py      # versi script
    python tools/run_task.py --task segmentation --workdir ../kerja --set RUN_MODE=fast --set EPOCHS=3
    python tools/run_task.py --template ../kerja/tabular_master.py --set RUN_MODE=fast

Aturan file yang dijalankan:
- Kalau folder kerja sudah berisi salinan template (dibuat tools/new_task.py, mungkin sudah kamu edit) → salinan itu dipakai.
- Kalau belum → template dari repo disalin dulu ke folder kerja.
- ipynb: semua output tersimpan di notebook itu (bukti analisis). py: output console tersimpan di run_log.txt.

Config: JSON berisi nama atribut CFG (lihat cell CFG di template). --set KEY=VALUE menimpa isi JSON.
Exit code: 0 = selesai, 1 = berhenti karena error.
"""
import argparse
import glob
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
sys.path.insert(0, HERE)
import _runner_common as R  # noqa: E402
import nb_formats  # noqa: E402
import task_registry as REG  # noqa: E402


def resolve_template(args, workdir):
    """Return (path file yang dijalankan, format)."""
    if args.template:
        src = os.path.abspath(args.template)
        fmt = args.format or ("py" if src.endswith(".py") else "ipynb")
        if fmt == "py" and src.endswith(".ipynb"):
            src_py = os.path.splitext(src)[0] + ".py"
            if not os.path.exists(src_py) or not nb_formats.is_in_sync(src):
                nb_formats.export(src)
            src = src_py
        name = os.path.splitext(os.path.basename(src))[0]
    else:
        if args.task not in REG.TASKS:
            raise SystemExit(f"Task tidak dikenal: {args.task}\nPilihan:\n{REG.describe()}")
        if not REG.available(args.task):
            raise SystemExit(f"Template untuk task '{args.task}' belum ada di repo ({REG.TASKS[args.task]['path']}).")
        fmt = args.format or "ipynb"
        name = REG.template_name(args.task)
        src = REG.template_path(args.task, fmt)
        if fmt == "py" and (not os.path.exists(src) or not nb_formats.is_in_sync(REG.template_path(args.task, "ipynb"))):
            nb_formats.export(REG.template_path(args.task, "ipynb"))
    ext = ".py" if fmt == "py" else ".ipynb"
    local = os.path.join(workdir, name + ext)
    if os.path.abspath(src) == os.path.abspath(local) or (os.path.exists(local) and not args.fresh_copy):
        if os.path.abspath(src) != os.path.abspath(local):
            print(f"ℹ️ memakai salinan di folder kerja: {local}  (--fresh-copy untuk menimpa dengan template repo)")
        return local, fmt
    shutil.copyfile(src, local)
    print(f"ℹ️ template disalin: {os.path.relpath(src, REG.REPO)} → {local}")
    return local, fmt


def summarize_outputs(workdir, since):
    """Cari file yang dihasilkan run ini (submission, figure, tabel) untuk ringkasan."""
    out_root = os.path.join(workdir, "outputs")
    if not os.path.isdir(out_root):
        print("⚠️ folder outputs/ tidak ada")
        return {}
    new = [p for p in glob.glob(os.path.join(out_root, "**", "*"), recursive=True)
           if os.path.isfile(p) and os.path.getmtime(p) >= since - 1]
    subs = [p for p in new if "submission" in os.path.basename(p).lower() and p.lower().endswith((".csv", ".json", ".txt", ".zip"))]
    figs = [p for p in new if p.lower().endswith(".png")]
    info = {"files": len(new), "submissions": [os.path.relpath(p, workdir) for p in subs], "figures": len(figs)}
    print(f"📦 {len(new)} file output baru | {len(figs)} figure | submission: {info['submissions'][:5] or '-'}")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", help="nama task (lihat --list)")
    ap.add_argument("--template", help="path template .ipynb/.py (alternatif --task)")
    ap.add_argument("--format", choices=["ipynb", "py"], default=None, help="ipynb (default) atau py")
    ap.add_argument("--config", help="JSON config (override CFG)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override CFG tambahan (boleh berulang)")
    ap.add_argument("--workdir", default=None, help="folder kerja; default = folder config atau folder saat ini")
    ap.add_argument("--timeout", type=int, default=-1, help="batas waktu (detik), -1 = tanpa batas")
    ap.add_argument("--fresh-copy", action="store_true", help="timpa salinan di folder kerja dengan template repo")
    ap.add_argument("--no-install", action="store_true", help="jangan pip install apa pun (TEMPLATE_AUTO_INSTALL=0)")
    ap.add_argument("--quiet", action="store_true", help="format py: jangan tampilkan output (tetap masuk log)")
    ap.add_argument("--list", action="store_true", help="tampilkan daftar task")
    args = ap.parse_args()
    if args.list:
        print(REG.describe())
        return 0
    if not args.task and not args.template:
        ap.error("isi --task atau --template (lihat --list)")
    if args.task == "churn":
        print("ℹ️ Task churn memakai runner khusus: python tools/run_case.py --config case_config.json [--format py]")
        return 1

    cfg = R.apply_sets(R.read_json_config(args.config), args.set, ap.error)
    workdir = os.path.abspath(args.workdir or (os.path.dirname(os.path.abspath(args.config)) if args.config else os.getcwd()))
    os.makedirs(workdir, exist_ok=True)
    cfg_path = os.path.join(workdir, ".run_task_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str, ensure_ascii=False)

    target, fmt = resolve_template(args, workdir)
    env = {"TEMPLATE_CONFIG": cfg_path, "TEMPLATE_REPO_DIR": REG.REPO}
    if args.no_install:
        env["TEMPLATE_AUTO_INSTALL"] = "0"
    print(f"▶ {os.path.basename(target)} | format {fmt} | folder {workdir}")
    print(f"  config: {json.dumps(cfg, default=str, ensure_ascii=False)[:400]}")
    import time
    t_start = time.time()
    if fmt == "py":
        status, secs = R.execute_script(target, workdir, env, args.timeout, quiet=args.quiet)
        where = os.path.join(workdir, "run_log.txt")
    else:
        status, secs = R.execute_notebook(target, workdir, env, args.timeout)
        where = target
    print(f"\n⏱  {secs:,.0f}s | {'✅ selesai' if status == 0 else '❌ berhenti'} | log/output: {where}")
    summarize_outputs(workdir, t_start)
    return status


if __name__ == "__main__":
    sys.exit(main())
