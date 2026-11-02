"""
Headless runner for 01_Churn_Insurance/churn_insurance_master.ipynb (atau versi script .py-nya).

Executes the notebook top-to-bottom (like "Restart & Run All") with a JSON config, keeps all cell
outputs inside the executed notebook (evidence of analysis for the ZIP), and prints a run summary.
--format py menjalankan churn_insurance_master.py (isi sama) dengan Python biasa; output console → run_log.txt,
salinan script → <TEAM>_churn_analysis.py (ikut ZIP). Cek aturan lomba: kalau wajib .ipynb, pakai format default.

Usage (from anywhere):
    python tools/run_case.py --config case_config.json
    python tools/run_case.py --config case_config.json --format py
    python tools/run_case.py --config case_config.json --set RUN_MODE=fast --set USE_OPTUNA=false
    python tools/run_case.py --workdir final --set TRAIN_PATH=data/train.csv --set TEAM_NAME=Warkab

Config file = JSON object whose keys are CFG attribute names (see the CFG cell), e.g.
    {"RUN_MODE": "full", "TRAIN_PATH": "data/train.csv", "TEST_PATH": "data/test.csv",
     "TARGET_COL": "churn", "TEAM_NAME": "Warkab", "COMPANY_NAME": "PT Asuransi X"}
Lines starting with // are treated as comments.

Exit code: 0 = finished without section errors, 2 = finished with section errors, 1 = crashed.
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
sys.path.insert(0, HERE)
import _runner_common as R  # noqa: E402
import nb_formats  # noqa: E402

DEFAULT_NB = os.path.join(os.path.dirname(HERE), "01_Churn_Insurance", "churn_insurance_master.ipynb")
read_config, parse_value = R.read_json_config, R.parse_value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config (CFG overrides)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra CFG override (repeatable)")
    ap.add_argument("--format", choices=["ipynb", "py"], default="ipynb", help="ipynb (default, output tersimpan di notebook) atau py")
    ap.add_argument("--notebook", default=DEFAULT_NB, help="master notebook to execute")
    ap.add_argument("--workdir", default=None, help="working folder (relative paths in config resolve here); default = config folder or cwd")
    ap.add_argument("--out-notebook", default=None, help="where to save the executed notebook/script (default <workdir>/<TEAM_NAME>_churn_analysis.ipynb|.py)")
    ap.add_argument("--timeout", type=int, default=-1, help="per-cell timeout in seconds (-1 = none)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = R.apply_sets(read_config(args.config), args.set, ap.error)
    workdir = os.path.abspath(args.workdir or (os.path.dirname(os.path.abspath(args.config)) if args.config else os.getcwd()))
    os.makedirs(workdir, exist_ok=True)
    cfg_path = os.path.join(workdir, ".run_case_config.json")

    team = str(cfg.get("TEAM_NAME", "TeamName")).replace(" ", "_")
    ext = ".py" if args.format == "py" else ".ipynb"
    out_nb = os.path.abspath(args.out_notebook or os.path.join(workdir, f"{team}_churn_analysis{ext}"))
    src = args.notebook
    if args.format == "py":
        if src.endswith(".ipynb"):
            if not nb_formats.is_in_sync(src):
                nb_formats.export(src)
            src = os.path.splitext(src)[0] + ".py"
        stale = os.path.splitext(out_nb)[0] + ".ipynb"         # jangan sampai ZIP memuat notebook lama yang tidak dijalankan
        if os.path.exists(stale):
            print(f"ℹ️ {os.path.basename(stale)} (run lama format ipynb) tetap ada — hapus kalau tidak ingin ikut ZIP.")
    if os.path.abspath(src) != out_nb:
        shutil.copyfile(src, out_nb)
    cfg.setdefault("NOTEBOOK_PATH", out_nb)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    env = {"CHURN_CONFIG": cfg_path, "CHURN_REPO_DIR": os.path.dirname(HERE)}   # → dashboard_template & dokumen repo
    print(f"▶ Executing {os.path.basename(out_nb)} in {workdir}")
    print(f"  config: {json.dumps(cfg, default=str)[:400]}")
    if args.format == "py":
        status, secs = R.execute_script(out_nb, workdir, env, args.timeout, log_path=os.path.join(workdir, "run_log.txt"), quiet=args.quiet)
        print(f"\n⏱  {secs:,.0f}s | script → {out_nb} | log → {os.path.join(workdir, 'run_log.txt')}")
    else:
        # stderr level-proses kernel (mis. traceback 'KeyError joblib_memmapping_folder' dari pembersih joblib di Windows — tidak
        # berbahaya) → kernel_stderr.log, supaya console bersih; output analisis tetap tersimpan di notebook.
        status, secs = R.execute_notebook(out_nb, workdir, env, args.timeout, os.path.join(workdir, "kernel_stderr.log"))
        print(f"\n⏱  {secs:,.0f}s | executed notebook saved → {out_nb}")

    out_dir = cfg.get("OUTPUT_DIR", "./outputs/churn_insurance")
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(workdir, out_dir)
    summ = os.path.join(out_dir, "run_summary.json")
    if os.path.exists(summ):
        with open(summ, encoding="utf-8") as f:
            s = json.load(f)
        print(f"✅ final model: {s.get('final_model')} | OOF AUC {s.get('final_oof_auc', float('nan')):.4f} | "
              f"n_train {s.get('n_train')} | churn rate {s.get('churn_rate', float('nan')):.3f}")
        errs = s.get("section_errors") or {}
        if errs:
            print(f"⚠️  {len(errs)} section error(s):")
            for k, v in errs.items():
                print(f"   - {k}: {v[:300]}")
            status = status or 2
        else:
            print("✅ no section errors")
        for fn in ["REPORT_DRAFT.docx", "REPORT_DRAFT.pdf", "dashboard/index.html", "submission.csv"]:
            p = os.path.join(out_dir, fn)
            print(f"   {'✔' if os.path.exists(p) else '✘'} {fn}")
    elif status == 0:
        status = 1
        print("❌ run_summary.json tidak ditemukan")
    return status


if __name__ == "__main__":
    sys.exit(main())
