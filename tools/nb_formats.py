"""
Dua format untuk setiap template: notebook (.ipynb) dan script Python (.py).

Sumber utama = .ipynb. File .py dibuat otomatis di samping notebook (nama sama) dalam format "percent":
setiap "# %%" = satu cell kode, "# %% [markdown]" = cell penjelasan (ditulis sebagai komentar).
File .py bisa dijalankan langsung (`python nama.py`) dan juga dibuka per cell di VS Code / PyCharm / Spyder.

    python tools/nb_formats.py export                       # semua notebook di repo → .py
    python tools/nb_formats.py export 02_Tabular            # satu folder / satu file .ipynb
    python tools/nb_formats.py check                        # gagal (exit 1) kalau ada .py yang tidak sinkron
    python tools/nb_formats.py to-ipynb kerja/tabular_master.py            # .py (hasil edit) → .ipynb
    python tools/nb_formats.py to-ipynb kerja/tabular_master.py --out x.ipynb

Tidak butuh library apa pun (nbformat/jupytext tidak wajib): notebook dibaca/ditulis sebagai JSON biasa.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

for _st in (sys.stdout, sys.stderr):  # console Windows (cp1252) tidak bisa mencetak emoji → paksa UTF-8
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SKIP_DIRS = {".git", ".ipynb_checkpoints", "outputs", "data", "QA", "examples", "node_modules", "__pycache__"}

MD_MARK = "# %% [markdown]"
CODE_MARK = "# %%"
HEADER_END = "# ---- akhir header ----"

PY_HEADER = '''# -*- coding: utf-8 -*-
# =====================================================================================================
# {name}.py — versi SCRIPT dari {name}.ipynb (dibuat otomatis oleh tools/nb_formats.py)
#
# Jalankan seluruhnya :  python {name}.py            (figure disimpan ke outputs/, tidak muncul di layar)
# Jalankan per cell   :  buka di VS Code → klik "Run Cell" di atas setiap "# %%" (Interactive Window)
# Isinya sama persis dengan notebook. Cell penjelasan (markdown) ditulis sebagai komentar "# ...".
#
# Mau edit di .py lalu kembali ke notebook?  python tools/nb_formats.py to-ipynb {name}.py
# =====================================================================================================
import os as _os
import sys as _sys

_os.environ.setdefault("MPLBACKEND", "Agg")      # mode script: figure disimpan, tidak membuka jendela
for _st in (_sys.stdout, _sys.stderr):           # console Windows (cp1252) tidak bisa mencetak emoji → UTF-8
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
try:                                             # display() hanya ada di Jupyter → di script = print
    from IPython.display import display  # noqa: F401
except Exception:
    display = print
''' + HEADER_END + "\n"


# --------------------------------------------------------------------------------------------------------
# baca / tulis
# --------------------------------------------------------------------------------------------------------
def read_notebook(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def notebook_to_py(nb, name):
    parts = [PY_HEADER.format(name=name)]
    for cell in nb.get("cells", []):
        src = _src(cell).rstrip("\n")
        if cell.get("cell_type") == "markdown":
            lines = [("# " + l) if l.strip() else "#" for l in src.split("\n")]
            parts.append(MD_MARK + "\n" + "\n".join(lines) + "\n")
        elif cell.get("cell_type") == "code":
            parts.append(CODE_MARK + "\n" + src + "\n")
        # raw cell: dilewati
    return "\n".join(parts)


def _cell_id(i, text):
    return hashlib.md5(f"{i}:{text}".encode("utf-8")).hexdigest()[:8]


def _split_lines(text):
    lines = text.split("\n")
    out = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return out


def py_to_notebook(text, base_metadata=None):
    """Parse format percent → dict notebook. Header (sebelum cell pertama) dibuang."""
    lines = text.replace("\r\n", "\n").split("\n")
    if HEADER_END in lines:
        lines = lines[lines.index(HEADER_END) + 1:]
    cells, cur_type, buf = [], None, []

    def flush():
        if cur_type is None:
            return
        body = list(buf)
        while body and not body[-1].strip():
            body.pop()
        if cur_type == "markdown":
            body = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in body]
        text_ = "\n".join(body)
        cell = {"cell_type": cur_type, "id": _cell_id(len(cells), text_), "metadata": {}, "source": _split_lines(text_)}
        if cur_type == "code":
            cell.update({"execution_count": None, "outputs": []})
            cell = {k: cell[k] for k in ["cell_type", "execution_count", "id", "metadata", "outputs", "source"]}
        cells.append(cell)

    for line in lines:
        s = line.rstrip()
        if s == MD_MARK or s.startswith(MD_MARK + " "):
            flush()
            cur_type, buf = "markdown", []
        elif s == CODE_MARK or (s.startswith(CODE_MARK + " ") and not s.startswith(MD_MARK)):
            flush()
            cur_type, buf = "code", []
        elif cur_type is not None:
            buf.append(line)
    flush()
    meta = base_metadata or {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def write_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def py_path_for(nb_path):
    return os.path.splitext(nb_path)[0] + ".py"


def expected_py(nb_path):
    name = os.path.splitext(os.path.basename(nb_path))[0]
    return notebook_to_py(read_notebook(nb_path), name)


# --------------------------------------------------------------------------------------------------------
# perintah
# --------------------------------------------------------------------------------------------------------
def find_notebooks(target=None):
    if target and os.path.isfile(target):
        return [os.path.abspath(target)]
    root = os.path.abspath(target or REPO)
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        out += [os.path.join(d, f) for f in sorted(files) if f.endswith(".ipynb")]
    return out


def export(nb_path, py_path=None):
    py_path = py_path or py_path_for(nb_path)
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(expected_py(nb_path))
    return py_path


def is_in_sync(nb_path):
    p = py_path_for(nb_path)
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n") == expected_py(nb_path)


def _reuse_ids(new_cells, old_cells):
    """Pakai lagi id cell lama (isi sama → id sama; selain itu id di posisi yang sama) supaya diff git kecil."""
    by_src = {}
    for c in old_cells:
        by_src.setdefault((c.get("cell_type"), _src(c).rstrip("\n")), []).append(c.get("id"))
    used = set()
    for c in new_cells:
        ids = by_src.get((c["cell_type"], _src(c).rstrip("\n")))
        if ids:
            c["id"] = ids.pop(0)
            used.add(c["id"])
    old_pos = {c.get("id"): i for i, c in enumerate(old_cells)}
    prev = -1
    for c in new_cells:                      # cell yang diedit: ambil id cell lama berikutnya (urutan tetap)
        if c["id"] in used and c["id"] in old_pos:
            prev = old_pos[c["id"]]
            continue
        j = prev + 1
        if j < len(old_cells) and old_cells[j].get("id") not in used and old_cells[j].get("cell_type") == c["cell_type"]:
            c["id"] = old_cells[j]["id"]
            used.add(c["id"])
            prev = j


def to_ipynb(py_path, out=None):
    out = out or os.path.splitext(py_path)[0] + ".ipynb"
    old = read_notebook(out) if os.path.exists(out) else None
    with open(py_path, encoding="utf-8") as f:
        nb = py_to_notebook(f.read(), old.get("metadata") if old else None)
    if old:
        _reuse_ids(nb["cells"], old.get("cells", []))
    write_notebook(nb, out)
    return out, len(nb["cells"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help=".ipynb → .py")
    e.add_argument("target", nargs="?", default=None)
    c = sub.add_parser("check", help="cek .py sinkron dengan .ipynb")
    c.add_argument("target", nargs="?", default=None)
    t = sub.add_parser("to-ipynb", help=".py → .ipynb")
    t.add_argument("py")
    t.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.cmd == "export":
        nbs = find_notebooks(a.target)
        for nb in nbs:
            print("✔", os.path.relpath(export(nb), REPO))
        print(f"{len(nbs)} notebook diekspor ke .py")
    elif a.cmd == "check":
        bad = [nb for nb in find_notebooks(a.target) if not is_in_sync(nb)]
        if bad:
            for nb in bad:
                print("✘ tidak sinkron:", os.path.relpath(py_path_for(nb), REPO))
            print("→ jalankan: python tools/nb_formats.py export")
            return 1
        print("✅ semua .py sinkron dengan .ipynb")
    elif a.cmd == "to-ipynb":
        out, n = to_ipynb(a.py, a.out)
        print(f"✔ {out} ({n} cell)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
