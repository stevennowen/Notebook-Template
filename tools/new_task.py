"""
Siapkan folder kerja untuk satu task dalam 1 perintah, dengan format pilihan: notebook (.ipynb) atau script (.py).

    python tools/new_task.py --list                                         # daftar task
    python tools/new_task.py ../kerja --task tabular                         # notebook (default)
    python tools/new_task.py ../kerja --task segmentation --format py        # script Python
    python tools/new_task.py ../kerja --task audio --format both             # dua-duanya

Isi folder yang dibuat:
    data/                       ← taruh file dari panitia di sini
    <template>.ipynb / .py      ← salinan template (boleh diedit: cell USER OVERRIDE)
    task_config.json            ← path & nama kolom; nilainya MENIMPA USER OVERRIDE
    RUN_FAST.bat / RUN_FULL.bat (Windows) dan run_fast.sh / run_full.sh (Mac/Linux)
    NEXT_STEPS.md

Task churn/asuransi punya alur lengkap sendiri: python tools/new_case.py (lihat MULAI_DI_SINI.md).
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
import nb_formats  # noqa: E402
import task_registry as REG  # noqa: E402


def write_runners(d, key, fmt):
    runner = os.path.join(HERE, "run_task.py")
    py = sys.executable
    for mode in ["fast", "full"]:
        args = f'--task {key} --format {fmt} --config task_config.json --set RUN_MODE={mode}'
        with open(os.path.join(d, f"RUN_{mode.upper()}.bat"), "w", encoding="utf-8") as f:
            f.write(f'@echo off\r\nchcp 65001 >nul\r\nset PYTHONIOENCODING=utf-8\r\ncd /d "%~dp0"\r\n'
                    f'"{py}" "{runner}" {args}\r\npause\r\n')
        with open(os.path.join(d, f"run_{mode}.sh"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f'#!/bin/bash\ncd "$(dirname "$0")"\npython "{runner}" {args}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--task", help="nama task (lihat --list)")
    ap.add_argument("--format", choices=["ipynb", "py", "both"], default="ipynb",
                    help="ipynb = notebook Jupyter (default), py = script Python, both = keduanya")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.task:
        print("Task yang tersedia:\n" + REG.describe())
        if not a.list:
            print("\nContoh: python tools/new_task.py ../kerja --task tabular --format py")
        return 0
    if a.task not in REG.TASKS:
        raise SystemExit(f"Task tidak dikenal: {a.task}\n{REG.describe()}")
    if a.task == "churn":
        raise SystemExit("Task churn memakai alur lengkap: python tools/new_case.py <folder> --team <Tim> [--format py]\n"
                         "Panduan: MULAI_DI_SINI.md")
    if not REG.available(a.task):
        raise SystemExit(f"Template '{a.task}' belum ada di repo ({REG.TASKS[a.task]['path']}).")
    if not a.folder:
        ap.error("isi nama folder kerja, mis. ../kerja")

    t = REG.TASKS[a.task]
    d = os.path.abspath(a.folder)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    nb_src = REG.template_path(a.task, "ipynb")
    if not nb_formats.is_in_sync(nb_src):
        nb_formats.export(nb_src)
    fmts = ["ipynb", "py"] if a.format == "both" else [a.format]
    copied = []
    for fmt in fmts:
        src = REG.template_path(a.task, fmt)
        dst = os.path.join(d, os.path.basename(src))
        if os.path.exists(dst):
            print(f"{os.path.basename(dst)} sudah ada → tidak ditimpa")
        else:
            shutil.copyfile(src, dst)
        copied.append(os.path.basename(dst))
    cfg_path = os.path.join(d, "task_config.json")
    if os.path.exists(cfg_path):
        print("task_config.json sudah ada → tidak ditimpa")
    else:
        filled = {k: v for k, v in t["config"].items() if v is not None}
        optional = [k for k, v in t["config"].items() if v is None]
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"_catatan": "Nilai di sini MENIMPA cell USER OVERRIDE. Hapus key kalau mau diatur dari notebook. "
                                   "RUN_MODE diatur oleh RUN_FAST.bat / RUN_FULL.bat.",
                       **({"_key_opsional": f"tambahkan bila perlu (default = auto-detect): {', '.join(optional)}"} if optional else {}),
                       **filled},
                      f, indent=2, ensure_ascii=False)
    run_fmt = "py" if a.format == "py" else "ipynb"
    write_runners(d, a.task, run_fmt)
    name = REG.template_name(a.task)
    main_file = f"{name}.{run_fmt}"
    open_hint = (f"Buka `{name}.py` di VS Code → klik **Run Cell** per `# %%`, atau jalankan semua: `python {name}.py`"
                 if run_fmt == "py" else
                 f"Buka `{name}.ipynb` di Jupyter (START_JUPYTER.bat di repo) → Kernel → Restart & Run All")
    steps = f"""# Next steps — {t['title']}

Format: **{', '.join(copied)}**

1. Copy data panitia ke `data/` — yang diharapkan: {t.get('data', '-')}
2. Isi `task_config.json` (path, nama kolom, metrik). Tidak yakin nama kolom? Biarkan `null` → auto-detect.
3. Double-click **RUN_FAST.bat** (cek cepat, data/epoch dikecilkan). Baca output: path benar? target benar? ada error?
4. Double-click **RUN_FULL.bat** untuk hasil final.
5. Hasil ada di `outputs/<nama template>/`: submission, figures/, tables/, ringkasan untuk laporan.

Mau eksplorasi interaktif? {open_hint}.
Perubahan setting bisa di cell `# >>> USER OVERRIDE` — tapi key yang ada di `task_config.json` tetap menang.

Pindah format kapan saja:
- notebook → script : `python "{os.path.join(HERE, 'nb_formats.py')}" export {name}.ipynb`
- script → notebook : `python "{os.path.join(HERE, 'nb_formats.py')}" to-ipynb {name}.py`

File utama yang dijalankan RUN_*.bat: `{main_file}`.
"""
    with open(os.path.join(d, "NEXT_STEPS.md"), "w", encoding="utf-8") as f:
        f.write(steps)
    print(f"✅ Folder kerja siap: {d}")
    print(steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
