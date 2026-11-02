"""
Eksekusi template secara headless dalam dua format. Dipakai tools/run_task.py dan tools/run_case.py.

- execute_notebook(): seperti "Restart & Run All"; semua output tersimpan di dalam .ipynb (butuh nbclient).
- execute_script():   menjalankan versi .py dengan Python yang sama; output console disalin ke file log.
"""
import json
import os
import subprocess
import sys
import time


def read_json_config(path):
    """JSON dengan baris komentar // (sama seperti case_config.json)."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        text = "\n".join(l for l in f.read().splitlines() if not l.strip().startswith("//"))
    return json.loads(text) if text.strip() else {}


def parse_value(v):
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return json.loads(v)
    except Exception:
        return v


def apply_sets(cfg, sets, error=None):
    for kv in sets or []:
        if "=" not in kv:
            msg = f"--set expects KEY=VALUE, got {kv!r}"
            if error:
                error(msg)
            raise ValueError(msg)
        k, v = kv.split("=", 1)
        cfg[k.strip()] = parse_value(v)
    return cfg


def base_env(extra=None):
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # kernel / subprocess harus memakai interpreter INI (hindari Python lain di PATH)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def execute_notebook(nb_path, workdir, env_extra=None, timeout=-1, stderr_log=None):
    """Jalankan notebook in-place. Return (status, detik). status 0 = selesai, 1 = berhenti karena error."""
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print("❌ nbclient/nbformat tidak ada → python -m pip install nbclient nbformat ipykernel")
        print("   atau jalankan versi script: --format py")
        return 1, 0.0
    if sys.platform.startswith("win"):           # hilangkan warning zmq "Proactor event loop" di Windows
        import asyncio
        import warnings
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        warnings.filterwarnings("ignore", message=".*Proactor event loop.*")
    nb = nbformat.read(nb_path, as_version=4)
    env_backup = dict(os.environ)
    os.environ.clear()
    os.environ.update(base_env(env_extra))
    client = NotebookClient(nb, timeout=None if timeout < 0 else timeout, kernel_name="python3",
                            allow_errors=False, resources={"metadata": {"path": workdir}})
    t0, status = time.time(), 0
    # stderr level-proses kernel (mis. pesan pembersih joblib di Windows — tidak berbahaya) → file log
    err_log = open(stderr_log or os.path.join(workdir, "kernel_stderr.log"), "w", encoding="utf-8", errors="replace")
    try:
        client.execute(stderr=err_log)
    except Exception as e:  # cell gagal → notebook berhenti
        status = 1
        print(f"\n❌ Notebook stopped: {type(e).__name__}: {str(e)[-1500:]}")
    finally:
        err_log.close()
        nbformat.write(nb, nb_path)
        os.environ.clear()
        os.environ.update(env_backup)
    return status, time.time() - t0


def execute_script(py_path, workdir, env_extra=None, timeout=-1, log_path=None, quiet=False):
    """Jalankan script .py, tampilkan output langsung, dan salin ke log. Return (status, detik)."""
    log_path = log_path or os.path.join(workdir, "run_log.txt")
    t0 = time.time()
    cmd = [sys.executable, "-u", os.path.abspath(py_path)]
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=workdir, env=base_env(env_extra), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in proc.stdout:
                log.write(line)
                if not quiet:
                    sys.stdout.write(line)
                if timeout > 0 and time.time() - t0 > timeout:
                    proc.kill()
                    log.write(f"\n❌ timeout {timeout}s\n")
                    print(f"\n❌ timeout {timeout}s")
                    break
            proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            raise
    status = 0 if proc.returncode == 0 else 1
    if status:
        print(f"\n❌ Script berhenti dengan exit code {proc.returncode} — lihat traceback di atas / {log_path}")
    return status, time.time() - t0
