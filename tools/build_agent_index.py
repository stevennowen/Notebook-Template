"""
Buat peta repo untuk AI agent: AGENTS.md (root, ringkas, selalu dibaca agent) + docs/agent_index/<template>.md
(per template: section + nomor baris di file .py, semua key CFG dengan default & komentar) + docs/agent_index/index.json.

    python tools/build_agent_index.py           # tulis ulang semua file index
    python tools/build_agent_index.py --check   # exit 1 kalau index tidak sinkron dengan template (dipakai QA)

Sumber kebenaran: tools/task_registry.py + isi file .py setiap template (dibuat tools/nb_formats.py export).
Jangan edit AGENTS.md / docs/agent_index/ secara manual — ubah skrip ini lalu jalankan ulang.
"""
import argparse
import ast
import json
import os
import re
import sys

for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import nb_formats  # noqa: E402
import task_registry as REG  # noqa: E402

INDEX_DIR = os.path.join(REPO, "docs", "agent_index")

# Dokumen yang layak dibuka agent, beserta KAPAN dibuka (ditulis manual; judul diambil otomatis).
DOCS = [
    ("README.md", "gambaran umum repo, tabel soal → template, format .ipynb vs .py"),
    ("MULAI_DI_SINI.md", "alur hari-H churn/asuransi langkah demi langkah (download → upload)"),
    ("DOWNLOAD_DAN_INSTALASI.md", "instalasi di komputer lain / tanpa internet / masalah Python"),
    ("Guides/CHEATSHEET_METRIK_MODEL_VALIDASI.md", "memilih metrik, model, skema validasi untuk jenis soal apa pun"),
    ("Guides/DEBUGGING_TANPA_AI.md", "error umum & cara membaca traceback"),
    ("01_Churn_Insurance/00_MULAI_DI_SINI.md", "runbook detail churn + tabel 'kalau ada masalah'"),
    ("01_Churn_Insurance/CASE_SCENARIOS.md", "bentuk data churn aneh (label terbalik, file terpisah, data rusak) → setting"),
    ("01_Churn_Insurance/PLAYBOOK_ANALISIS.md", "aturan 'jika X maka Y', istilah asuransi, pertanyaan juri"),
    ("01_Churn_Insurance/PANDUAN_LAPORAN.md", "struktur artikel 10 halaman & template kalimat"),
    ("01_Churn_Insurance/BAHAN_LATAR_BELAKANG.md", "fakta industri (OJK/AAJI) & literatur terverifikasi"),
    ("01_Churn_Insurance/RESEARCH_KIT.md", "cara riset latar belakang dari casebook"),
    ("04_Computer_Vision/CV_OTHER_TASKS_GUIDE.md", "kode contekan image regression & multilabel gambar (bukan template)"),
    ("QA/TEMPLATE_SMOKE_REPORT.md", "bukti setiap template jalan + apa yang belum teruji"),
    ("QA/QA_FINAL_REPORT.md", "bukti QA churn (stress test 44 varian, demo end-to-end)"),
]

NOT_COVERED = ("topic modeling / clustering teks, association rules (market basket), klasifikasi deret waktu/sinyal, "
               "recommender/ranking, geospasial, graph, kemiripan teks/retrieval/RAG, seq2seq (terjemahan/ringkasan), "
               "kemiripan gambar/matching, image-to-image, rotated box/keypoint/instance segmentation/video, ASR, "
               "multimodal, reinforcement learning. Image regression & multilabel gambar: hanya kode contekan di "
               "04_Computer_Vision/CV_OTHER_TASKS_GUIDE.md. Survival & causal/uplift: hanya di dalam template churn.")


# --------------------------------------------------------------------------------------------------------- parsing
def rel(p):
    return os.path.relpath(p, REPO).replace(os.sep, "/")


def read_py(key):
    path = REG.template_path(key, "py")
    if not os.path.exists(path) or not nb_formats.is_in_sync(REG.template_path(key, "ipynb")):
        nb_formats.export(REG.template_path(key, "ipynb"))
    with open(path, encoding="utf-8") as f:
        return path, f.read().replace("\r\n", "\n").split("\n")


def sections(lines):
    """Heading markdown (# / ##) beserta nomor baris (1-based) di file .py."""
    out, in_md = [], False
    for i, line in enumerate(lines, 1):
        if line.startswith(nb_formats.MD_MARK):
            in_md = True
            continue
        if line.startswith(nb_formats.CODE_MARK):
            in_md = False
            continue
        m = re.match(r"^# (#{1,2}) (.+)$", line) if in_md else None
        if m:
            out.append({"level": len(m.group(1)), "title": m.group(2).strip().rstrip(":"), "line": i})
    return out


def cfg_keys(lines):
    """Key CFG (indentasi 4 spasi) dari blok `class CFG:` utama (yang berisi RUN_MODE)."""
    starts = [i for i, l in enumerate(lines) if re.match(r"^class CFG\b", l)]
    for s in starts:
        block = []
        for j in range(s + 1, len(lines)):
            l = lines[j]
            if l.strip() and not l.startswith((" ", "\t")):
                break
            block.append((j + 1, l))
        if not any(re.match(r"^    RUN_MODE\s*=", l) for _, l in block):
            continue
        keys, group, above, i = [], "", [], 0
        balance = lambda x: sum(x.count(o) for o in "{[(") - sum(x.count(o) for o in "}])")  # noqa: E731
        while i < len(block):
            ln, l = block[i]
            i += 1
            g = re.match(r"^    #\s*[-=]{3,}\s*(.*?)\s*[-=]*\s*$", l) or re.match(r"^    #\s*(\[[A-Z]\].*|[A-Z][A-Z0-9 &/()\-]{3,})\s*$", l)
            if g:
                if g.group(1).strip(" -="):
                    group = g.group(1).strip(" -=")
                above = []
                continue
            c = re.match(r"^    #\s?(.*)$", l)
            if c:                                           # komentar di baris atas key = keterangan key itu
                above.append(c.group(1).strip())
                continue
            m = re.match(r"^    ([A-Z][A-Z0-9_]*)\s*=\s*(.*)$", l)
            if not m:
                above = []
                continue
            name, rest = m.group(1), m.group(2)
            val, comment = rest, ""
            for cut in [k.start() for k in re.finditer(r"\s#", rest)] + [len(rest)]:
                try:                                        # pisahkan komentar tanpa merusak string berisi '#'
                    ast.parse(rest[:cut].strip() or "None", mode="eval")
                    val, comment = rest[:cut].strip(), rest[cut:].strip().lstrip("#").strip()
                    break
                except SyntaxError:
                    continue
            depth = balance(val)
            if depth > 0:                                   # nilai multi-baris (dict/list): gabungkan
                parts = [val]
                while depth > 0 and i < len(block):
                    seg = re.sub(r"\s+#.*$", "", block[i][1]).strip()
                    if not seg.startswith("#"):
                        parts.append(seg)
                        depth += balance(seg)
                    i += 1
                joined = " ".join(p_ for p_ in parts if p_)
                try:
                    obj = ast.literal_eval(joined)
                    val = ("{" + ", ".join(map(str, obj)) + "}") if isinstance(obj, dict) else json.dumps(obj, ensure_ascii=False)
                except Exception:  # noqa
                    val = joined
            if "__import__(\"os\").environ" in val:
                val = "True (env TEMPLATE_AUTO_INSTALL=0 → False)"
            note = " ".join(x for x in above if x)
            comment = (note + (" · " if note and comment else "") + comment).strip()
            keys.append({"key": name, "default": val if len(val) <= 300 else val[:297] + "…",
                         "comment": comment[:220], "group": group, "line": ln})
            above = []
        return s + 1, keys
    return None, []


def first_heading(path):
    try:
        with open(path, encoding="utf-8") as f:
            for l in f:
                if l.startswith("#"):
                    return l.lstrip("# ").strip()
    except OSError:
        pass
    return ""


def tool_summary(path):
    try:
        doc = ast.get_docstring(ast.parse(open(path, encoding="utf-8").read())) or ""
        para = " ".join(doc.strip().split("\n\n")[0].split())          # paragraf pertama, satu baris
        sent = para
        for m in re.finditer(r"\.(?=\s+[A-Z(\"']|$)", para):     # akhir kalimat pertama (bukan singkatan)
            word = para[:m.start()].rsplit(" ", 1)[-1].lower()
            if word not in ("mis", "dll", "dsb", "e.g", "i.e", "vs", "no", "dkk"):
                sent = para[:m.end()]
                break
        return sent if len(sent) <= 170 else sent[:167].rsplit(" ", 1)[0] + "…"
    except Exception:  # noqa
        return ""


def submission_line(lines):
    """Baris kode pertama yang MENULIS file submission (bukan definisi path di CFG, bukan komentar)."""
    for i, l in enumerate(lines, 1):
        code = l.strip()
        if code.startswith("#"):
            continue
        if re.search(r"\.to_(csv|parquet)\(|open\(SUB", code) and re.search(r"(?i)sub", code):
            return i
    return None


def find_line(lines, pattern):
    rx = re.compile(pattern)
    return next((i for i, l in enumerate(lines, 1) if rx.search(l)), None)


# --------------------------------------------------------------------------------------------------------- build
def collect():
    templates = []
    for key, t in REG.TASKS.items():
        if not REG.available(key):
            continue
        path, lines = read_py(key)
        cfg_line, keys = cfg_keys(lines)
        name = os.path.basename(path)[:-3]
        out_dir = next((k["default"] for k in keys if k["key"] == "OUTPUT_DIR"), "None")
        out_dir = re.sub(r"^f?[\"']|[\"']$", "", out_dir).replace("{NOTEBOOK_NAME}", name)
        if out_dir in ("None", ""):
            out_dir = f"./outputs/{name}/"
        templates.append({
            "task": key, "title": t["title"], "py": rel(path), "ipynb": rel(REG.template_path(key, "ipynb")),
            "lines": len(lines), "data": t.get("data", ""), "special": t.get("special", ""),
            "example_config": t.get("config", {}), "cfg_line": cfg_line,
            "override_line": find_line(lines, r"^# >>> USER OVERRIDE"),
            "submission_line": submission_line(lines),
            "output_dir": out_dir, "sections": sections(lines), "cfg": keys,
        })
    tools = [{"file": rel(p), "summary": tool_summary(p)} for p in sorted(
        [os.path.join(REPO, "tools", f) for f in os.listdir(os.path.join(REPO, "tools")) if f.endswith(".py") and not f.startswith("_")]
        + [os.path.join(REPO, "QA", f) for f in os.listdir(os.path.join(REPO, "QA")) if f.endswith(".py")])]
    docs = [{"file": d, "title": first_heading(os.path.join(REPO, d)), "when": w} for d, w in DOCS if os.path.exists(os.path.join(REPO, d))]
    return {"templates": templates, "tools": tools, "docs": docs}


def render_agents(data):
    T = data["templates"]
    rows = "\n".join(
        f"| `{t['task']}` | {t['title']} | `{t['py']}` ({t['lines']:,} baris) | `docs/agent_index/{os.path.basename(t['py'])[:-3]}.md` |"
        for t in T)
    tool_rows = "\n".join(f"| `{x['file']}` | {x['summary']} |" for x in data["tools"] if x["summary"])
    doc_rows = "\n".join(f"| `{d['file']}` | {d['when']} |" for d in data["docs"])
    return f"""# AGENTS.md — peta Notebook-Template untuk AI agent

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual; jalankan ulang skrip itu. -->

Repo ini berisi **template lomba data science** (Python, Bahasa Indonesia + istilah teknis English). Setiap template
adalah satu notebook end-to-end: load data → EDA → model + cross-validation → evaluasi → submission → ringkasan laporan.
Tugas agent biasanya: (1) memilih template yang cocok dengan soal, (2) menyiapkan folder kerja & config,
(3) menjalankan dan membaca hasil, (4) menjawab pertanyaan tentang setting/isi template.

## Aturan membaca (hemat token)

1. **Jangan scan seluruh repo.** Mulai dari tabel di bawah, lalu buka `docs/agent_index/<template>.md` untuk detail.
2. **Baca file `.py`, bukan `.ipynb`.** Isinya identik (dibuat `tools/nb_formats.py`), tapi `.ipynb` adalah JSON
   penuh escape → ±2× token. Index memberi **nomor baris di `.py`**: baca rentang baris itu saja.
3. Abaikan folder `outputs/`, `data/`, `QA/demo_warkab/`, `QA/walkthrough_warkab/`, `01_Churn_Insurance/CONTOH_LAPORAN_FINAL/`,
   `01_Churn_Insurance/contoh_output/` kecuali diminta (hasil run contoh, gambar, PDF besar).
4. Untuk mengubah template: edit `.ipynb` **atau** `.py` lalu sinkronkan
   (`python tools/nb_formats.py to-ipynb <file>.py` → `python tools/nb_formats.py export`), lalu
   `python tools/build_agent_index.py` supaya index ikut terbarui.

## Pilih template (soal → `--task`)

| `--task` | Untuk soal | File yang dibaca | Detail (section, baris, semua key CFG) |
|---|---|---|---|
{rows}

Belum ada template untuk: {NOT_COVERED}

## Cara menjalankan (semua template kecuali churn)

```bash
python tools/new_task.py ../kerja --task <task> --format ipynb   # atau --format py / both
# copy data panitia ke ../kerja/data/, isi ../kerja/task_config.json
python tools/run_task.py --task <task> --format py --config ../kerja/task_config.json --set RUN_MODE=fast
python tools/run_task.py --task <task> --format py --config ../kerja/task_config.json --set RUN_MODE=full
```

- Hasil: `../kerja/outputs/<nama template>/` → `submissions/`, `figures/` (bernomor), `tables/`, `summary.md`,
  `run_summary.json`. Format `py` menulis log ke `run_log.txt` (baca bagian akhirnya untuk error).
- Churn/asuransi memakai alur khusus: `python tools/new_case.py <folder> --team <Tim> --casebook <soal.pdf> [--format py]`
  lalu `RUN_FAST.bat` / `RUN_FULL.bat` (lihat `MULAI_DI_SINI.md`).
- Data latihan untuk mencoba: `python tools/make_synthetic_task_data.py --task <task> --out <folder>`.

## Konvensi yang sama di semua template

| Hal | Aturan |
|---|---|
| Setting | Semua di `class CFG` (lihat baris CFG di index). Ubah di cell `# >>> USER OVERRIDE` atau `task_config.json` |
| Prioritas | `task_config.json` / env `TEMPLATE_CONFIG` **menimpa** USER OVERRIDE. Key JSON berawalan `_` diabaikan; list JSON → tuple otomatis kalau default-nya tuple |
| Mode | `RUN_MODE="fast"` (subsample, epoch/fold sedikit — untuk cek) lalu `"full"` (hasil final) |
| Install | Template memasang paket yang hilang (`AUTO_INSTALL`). Env `TEMPLATE_AUTO_INSTALL=0` atau `run_task.py --no-install` mematikannya. Torch tidak pernah di-install otomatis |
| Fallback | Library opsional hilang → jalur itu di-skip dengan pesan, notebook tetap jalan (mis. tanpa smp → torchvision → U-Net kecil) |
| pandas 3 | Semua template memanggil `pd.set_option("future.infer_string", False)` (perilaku teks = pandas 2) |
| Validasi | Selalu cross-validation/holdout; isi `GROUP_COL` kalau ada grup yang tidak boleh bocor antar fold |
| Laporan | `summary.md` berisi angka utama + kalimat draft (English) siap parafrase; figure bernomor `figNN_*.png` |

## Tool

| File | Fungsi |
|---|---|
{tool_rows}

## Dokumen (buka hanya kalau relevan)

| File | Buka kalau |
|---|---|
{doc_rows}

## Jebakan yang sering terjadi

- Kaggle *code competition*: path `/kaggle/input/...`, internet mati (`AUTO_INSTALL=False`, model pretrained dilampirkan
  sebagai Kaggle Dataset lalu isi path lokalnya), dan sering wajib `submission.parquet` → tambahkan cell terakhir
  `pd.read_csv(<folder output>/submissions/submission.csv).to_parquet("submission.parquet", index=False)`
  (baris penulisan submission tiap template ada di index-nya).
- `nlp_classification` default-nya untuk teks Indonesia informal (lowercase, normalisasi slang, buang tanda baca).
  Untuk teks Inggris / gaya tulisan: `CLASSICAL_LOWERCASE=False`, `USE_SLANG_NORMALIZATION=False`,
  `REDUCE_ELONGATION=False`, `KEEP_HASHTAG_TEXT=False`.
- Nilai `null` di `task_config.json` tetap menimpa CFG → hapus key-nya kalau ingin auto-detect / USER OVERRIDE.
- Format `.ipynb` headless butuh `nbclient`; tanpa itu pakai `--format py`.
- Bukti uji & batasan (library yang hanya teruji lewat fallback): `QA/TEMPLATE_SMOKE_REPORT.md`.
"""


def render_template_index(t):
    sec = "\n".join(f"| {'↳ ' * (s['level'] - 1)}{s['title'][:90]} | {s['line']} |" for s in t["sections"])
    cfg_rows, last = [], None
    for k in t["cfg"]:
        if k["group"] != last and k["group"]:
            cfg_rows.append(f"| **{k['group'][:60]}** | | | |")
            last = k["group"]
        default, comment = k["default"].replace("|", r"\|"), k["comment"].replace("|", r"\|")
        cfg_rows.append(f"| `{k['key']}` | `{default}` | {comment} | {k['line']} |")
    example = json.dumps(t["example_config"], ensure_ascii=False) if t["example_config"] else "-"
    special = f"\n> {t['special']}\n" if t["special"] else ""
    return f"""# {os.path.basename(t['py'])[:-3]} — `--task {t['task']}`

<!-- DIBUAT OTOMATIS oleh tools/build_agent_index.py — jangan edit manual. -->

{t['title']}
{special}
| | |
|---|---|
| Baca | `{t['py']}` ({t['lines']:,} baris) — notebook kembarannya: `{t['ipynb']}` |
| `class CFG` | baris {t['cfg_line']} · cell `# >>> USER OVERRIDE` baris {t['override_line']} |
| Data yang diharapkan | {t['data'] or '-'} |
| Contoh `task_config.json` | `{example}` |
| Folder output | `{t['output_dir'] or '-'}` |
| Submission ditulis | {f"baris {t['submission_line']}" if t['submission_line'] else '- (template ini menghasilkan laporan/tabel, bukan submission)'} |

## Section (nomor baris di file .py)

| Section | Baris |
|---|---|
{sec}

## Semua key CFG ({len(t['cfg'])})

| Key | Default | Keterangan | Baris |
|---|---|---|---|
{chr(10).join(cfg_rows)}
"""


def build():
    data = collect()
    files = {os.path.join(REPO, "AGENTS.md"): render_agents(data)}
    for t in data["templates"]:
        files[os.path.join(INDEX_DIR, os.path.basename(t["py"])[:-3] + ".md")] = render_template_index(t)
    slim = {"templates": [{k: v for k, v in t.items() if k not in ("sections",)} | {"sections": t["sections"]} for t in data["templates"]],
            "tools": data["tools"], "docs": data["docs"], "not_covered": NOT_COVERED}
    files[os.path.join(INDEX_DIR, "index.json")] = json.dumps(slim, indent=1, ensure_ascii=False) + "\n"
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="hanya cek sinkron (exit 1 kalau beda)")
    a = ap.parse_args()
    files = build()
    if a.check:
        stale = []
        for p, content in files.items():
            cur = open(p, encoding="utf-8").read().replace("\r\n", "\n") if os.path.exists(p) else None
            if cur != content:
                stale.append(rel(p))
        if stale:
            print("✘ index agent tidak sinkron:", stale, "\n→ jalankan: python tools/build_agent_index.py")
            return 1
        print(f"✅ index agent sinkron ({len(files)} file)")
        return 0
    os.makedirs(INDEX_DIR, exist_ok=True)
    for p, content in files.items():
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    agents = files[os.path.join(REPO, "AGENTS.md")]
    print(f"✅ {len(files)} file ditulis: AGENTS.md ({len(agents):,} karakter ≈ {len(agents) // 4:,} token) + docs/agent_index/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
