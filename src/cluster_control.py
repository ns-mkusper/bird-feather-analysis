"""Cluster control for notebooks/run_control.ipynb: workers, runs, progress.

Workers: status (Celery view + each node's process, code version, memory),
update the code (fast-forward from GitHub), stop / start / restart. Node
commands go over SSH with FEATHER_CONTROL_SSH_KEY (default: the workspace's
deploy key) as FEATHER_CONTROL_SSH_USER (default FEATHER_SSH_USER).

Runs: one launcher for a sample, a manifest (e.g. the 100-slide slice) or the
full collection. A run is scripts/run_slice.py started as a detached process,
so it keeps going if the notebook or its kernel stops; its log and a small
record live in data/run_control/<run_id>.*. progress() reads the log and the
queue and can be re-run any time; stop_run() stops the driver and, on request,
drops the run's queued jobs.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTROL = REPO / "data" / "run_control"
DEFAULT_KEY = "/root/.ssh/feather_deploy_ed25519"
WORKER_PATTERN = "celery -A src.celery_app worker"
SECONDS_PER_SLIDE_PER_WORKER = 210      # first estimate before a run has its own rate


# ---------------------------------------------------------------- nodes

def _env() -> dict:
    from dotenv import dotenv_values

    vals = {**dotenv_values(REPO / ".env"), **os.environ}
    return vals


def hosts() -> list[str]:
    return [h.strip() for h in _env().get("FEATHER_CLUSTER_HOSTS", "").split(",") if h.strip()]


def remote_repo() -> str:
    return _env().get("FEATHER_REMOTE_REPO_DIR", "~/bird-feather-analysis")


def ssh(host: str, cmd: str, timeout: float = 60) -> subprocess.CompletedProcess:
    env = _env()
    key = env.get("FEATHER_CONTROL_SSH_KEY", DEFAULT_KEY)
    user = env.get("FEATHER_CONTROL_SSH_USER") or env.get("FEATHER_SSH_USER", "")
    args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]
    if key and Path(key).exists():
        args += ["-i", key, "-o", "IdentitiesOnly=yes"]
    return subprocess.run([*args, f"{user}@{host}" if user else host, cmd], capture_output=True, text=True,
                          timeout=timeout)


def _node_status(host: str) -> dict:
    repo = remote_repo()
    cmd = (f"cd {repo} 2>/dev/null && echo code=$(git log -1 --format=%h); "
           f"P=$(pgrep -f '{WORKER_PATTERN}' | head -1); "
           "echo procs=$(pgrep -f '" + WORKER_PATTERN + "' | wc -l | tr -d ' '); "
           "[ -n \"$P\" ] && echo age=$(ps -o etime= -p $P | tr -d ' ') && "
           "echo rss_gb=$(ps -o rss= -p $P | awk '{printf \"%.1f\", $1/1048576}'); "
           "echo free_gb=$(vm_stat | awk '/Pages free|Pages inactive/{gsub(\"\\\\.\",\"\",$3); s+=$3} "
           "END{printf \"%.0f\", s*16384/1073741824}')")
    try:
        p = ssh(host, cmd, timeout=40)
    except subprocess.TimeoutExpired:
        return {"host": host, "ssh": "timeout"}
    if p.returncode != 0 and not p.stdout:
        return {"host": host, "ssh": (p.stderr.strip().splitlines() or ["failed"])[-1][:80]}
    out = {"host": host, "ssh": "ok"}
    for line in p.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def worker_status():
    """(nodes, workers): per node over SSH the code commit on disk, worker
    processes, process age, its memory and free memory; per Celery worker the
    slides running/waiting and whether it has the new code loaded."""
    import pandas as pd

    from src.notebook_remote_runtime import create_celery_client, load_notebook_config_from_env

    app = create_celery_client(load_notebook_config_from_env())
    insp = app.control.inspect(timeout=5)
    act, res, reg = insp.active() or {}, insp.reserved() or {}, insp.registered() or {}
    nodes = pd.DataFrame([_node_status(h) for h in hosts()])
    workers = sorted(set(act) | set(reg))
    celery = pd.DataFrame([{"worker": w, "running": len(act.get(w, [])), "waiting": len(res.get(w, [])),
                            "new_code": "src.celery_tasks.extract_labels" in reg.get(w, []),
                            "slide": ", ".join(Path(t.get("args", [""])[0]).name for t in act.get(w, []))}
                           for w in workers])
    return nodes, celery


def queue_length(queue: str | None = None) -> int:
    from src.notebook_remote_runtime import create_celery_client, load_notebook_config_from_env, task_queue

    app = create_celery_client(load_notebook_config_from_env())
    with app.connection_or_acquire() as conn:
        return int(conn.default_channel.client.llen(queue or task_queue() or "celery"))


def update_code(source: str = "upstream", branch: str = "main", nodes: list[str] | None = None) -> dict:
    """Fast-forward each node's repo: source 'upstream' = the nodes' origin,
    or a GitHub URL (e.g. a fork). Refuses anything but a fast-forward, so local
    changes and untracked run outputs are never touched."""
    src = "origin" if source == "upstream" else shlex.quote(source)
    out = {}
    for h in nodes or hosts():
        p = ssh(h, f"cd {remote_repo()} && git fetch -q {src} {shlex.quote(branch)} && "
                   f"git merge -q --ff-only FETCH_HEAD && git log -1 --format='%h %s'", timeout=120)
        out[h] = (p.stdout.strip() or p.stderr.strip())[:120]
    return out


def stop_workers(nodes: list[str] | None = None, wait: float = 90) -> dict:
    """Asks each worker to finish its current slide and exit. Note: on this
    cluster a watchdog restarts a stopped worker within seconds (with the code
    then on disk), so 'stop' is effectively 'restart'."""
    out = {}
    for h in nodes or hosts():
        p = ssh(h, f"pkill -f '{WORKER_PATTERN}'; for i in $(seq 1 {int(wait // 2)}); do "
                   f"pgrep -f '{WORKER_PATTERN}' >/dev/null || break; sleep 2; done; "
                   f"P=$(pgrep -f '{WORKER_PATTERN}' | head -1); "
                   "[ -n \"$P\" ] && echo running again, age $(ps -o etime= -p $P | tr -d ' ') || echo stopped",
                timeout=wait + 30)
        out[h] = (p.stdout.strip() or p.stderr.strip())[:120]
    return out


def start_workers(nodes: list[str] | None = None) -> dict:
    """Starts a worker on nodes where none runs (never a second one)."""
    out = {}
    for h in nodes or hosts():
        repo = remote_repo()
        p = ssh(h, f"if pgrep -f '{WORKER_PATTERN}' >/dev/null; then echo already running; else cd {repo} && "
                   f"([ -f celery_worker.log ] && mv -f celery_worker.log celery_worker.log.prev); "
                   f"nohup ~/.pixi/bin/pixi run --manifest-path {repo}/pixi.toml worker > celery_worker.log 2>&1 "
                   f"< /dev/null & disown; echo started; fi", timeout=60)
        out[h] = (p.stdout.strip() or p.stderr.strip())[:120]
    return out


def restart_workers(nodes: list[str] | None = None) -> dict:
    stopped = stop_workers(nodes)
    started = start_workers(nodes)
    return {h: f"{stopped.get(h)} / {started.get(h)}" for h in started}


# ---------------------------------------------------------------- runs

def _record(run_id: str) -> Path:
    return CONTROL / f"{run_id}.json"


def all_slides() -> list[str]:
    from src.notebook_remote_runtime import RemoteClusterIO, load_notebook_config_from_env

    return [Path(p).name for p in RemoteClusterIO(load_notebook_config_from_env()).list_remote_images()]


def choose(kind: str, n: int = 10, seed: int = 1, match: list[str] | None = None,
           manifest: str | Path | None = None) -> list[str]:
    """kind: 'sample' (n random, or `match` substrings), 'manifest' (a file of
    file names, e.g. configs/slice_100.txt), 'full' (every slide)."""
    import random

    if kind == "manifest":
        return [l.strip() for l in Path(manifest).read_text().splitlines() if l.strip() and not l.startswith("#")]
    names = all_slides()
    if kind == "full":
        return names
    if match:
        return [s for s in names if any(m in s for m in match)]
    return random.Random(seed).sample(names, min(n, len(names)))


def seed_labels(run_id: str, from_run: str, names: list[str] | None = None) -> dict:
    """Copies an earlier run's label results (labels/<stem>.json, .date.json) into this run's
    output folder on every node, so the workers reuse them instead of reading the labels again
    (src/label_extract.extract_for_pipeline: reused only from the same extractor version and
    without model errors). Refused unless the earlier run used the current label config."""
    import tarfile
    import tempfile

    from src import label_extract as lx

    used = REPO / "data" / "label_runs" / from_run / "config_used.json"
    if not used.exists() or json.loads(used.read_text()) != lx.load_config():
        raise RuntimeError(f"{from_run} used another label config: labels must be read again")
    src = REPO / "data" / "sample_runs" / from_run / "labels"
    stems = {Path(n).stem for n in names} if names else None
    files = [p for p in src.glob("*.json") if stems is None or p.name.split(".")[0] in stems]
    tmp = Path(tempfile.mkstemp(suffix=".tar")[1])
    with tarfile.open(tmp, "w") as tf:
        for p in files:
            tf.add(p, arcname=p.name)
    env = _env()
    key = env.get("FEATHER_CONTROL_SSH_KEY", DEFAULT_KEY)
    user = env.get("FEATHER_CONTROL_SSH_USER") or env.get("FEATHER_SSH_USER", "")
    dest = f"{remote_repo()}/data/runs/{run_id}/processed/labels"
    out = {}
    for h in hosts():
        with open(tmp, "rb") as fh:
            p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-i", key, "-o", "IdentitiesOnly=yes",
                                f"{user}@{h}" if user else h, f"mkdir -p {dest} && tar xf - -C {dest} && ls {dest} | wc -l"],
                               stdin=fh, capture_output=True, text=True, timeout=300)
        out[h] = (p.stdout.strip() or p.stderr.strip())[:80]
    tmp.unlink()
    return {"files": len(files), "nodes": out}


def launch_run(names: list[str], run_id: str, upload: bool = True, timeout_h: float | None = None,
               dry_run: bool = False, reuse_labels_from: str | None = None,
               handwriting: str | Path | None = None) -> dict:
    """Starts scripts/run_slice.py for `names` as a detached process. Returns
    the run record (also written to data/run_control/<run_id>.json). `reuse_labels_from`:
    an earlier run whose label results the workers reuse (seed_labels). `handwriting`: a
    hand-check CSV of handwritten labels for the tables (run_slice --handwriting)."""
    CONTROL.mkdir(parents=True, exist_ok=True)
    if _record(run_id).exists() and is_running(run_id):
        raise RuntimeError(f"{run_id} is already running")
    manifest = CONTROL / f"{run_id}.manifest.txt"
    manifest.write_text("\n".join(names) + "\n")
    log = CONTROL / f"{run_id}.log"
    n_workers = max(1, len(hosts()))
    est_h = len(names) * SECONDS_PER_SLIDE_PER_WORKER / n_workers / 3600
    timeout_s = int((timeout_h or max(2.0, est_h * 3)) * 3600)
    cmd = [sys.executable, "-u", str(REPO / "scripts" / "run_slice.py"), "--manifest", str(manifest),
           "--run-id", run_id, "--timeout", str(timeout_s)] + ([] if upload else ["--no-upload"]) \
        + (["--handwriting", str(Path(handwriting).resolve())] if handwriting else [])
    rec = {"run_id": run_id, "slides": len(names), "log": str(log), "manifest": str(manifest), "cmd": cmd,
           "upload": upload, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "estimate_h": round(est_h, 2)}
    if dry_run:
        return rec
    if reuse_labels_from:
        rec["labels_reused_from"] = {"run": reuse_labels_from, **seed_labels(run_id, reuse_labels_from, names)}
    env = {**os.environ}
    env.setdefault("FEATHER_SSH_KEY", _env().get("FEATHER_CONTROL_SSH_KEY", DEFAULT_KEY))
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, start_new_session=True)
    rec["pid"] = proc.pid
    _record(run_id).write_text(json.dumps(rec, indent=1))
    return rec


def register_run(run_id: str, log: str | Path, slides: int, pid: int | None, started: str) -> dict:
    """Records a run started outside the notebook so progress() can follow it."""
    CONTROL.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": run_id, "slides": slides, "log": str(Path(log).resolve()), "pid": pid, "started": started}
    _record(run_id).write_text(json.dumps(rec, indent=1))
    return rec


def runs():
    import pandas as pd

    rows = []
    for p in sorted(CONTROL.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        r = json.loads(p.read_text())
        pr = parse_log(Path(r["log"]).read_text()) if Path(r["log"]).exists() else {}
        rows.append({"run_id": r["run_id"], "slides": r["slides"], "started": r["started"],
                     "running": is_running(r["run_id"]), "done": pr.get("done", 0), "failed": pr.get("failed", 0),
                     "phase": pr.get("phase")})
    return pd.DataFrame(rows)


def is_running(run_id: str) -> bool:
    if not _record(run_id).exists():
        return False
    pid = json.loads(_record(run_id).read_text()).get("pid")
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
        return "run_slice.py" in cmdline
    except OSError:
        return True


_DONE = re.compile(r"^\s+(ok |FAIL) (.+?\.jpg) on (\S+)", re.M)


def parse_log(text: str) -> dict:
    """Counts from a run_slice / sample_run log."""
    total = re.search(r": (\d+) slides -> workers", text)
    done = _DONE.findall(text)
    phase = ("finished" if re.search(r"^wrote .*labels\.csv", text, re.M) or "copied to" in text
             else "collecting outputs" if "cluster part done" in text
             else "running" if total else "starting")
    if "Traceback" in text and phase != "finished":
        phase = "error"
    by_worker: dict[str, int] = {}
    for _, _, w in done:
        by_worker[w] = by_worker.get(w, 0) + 1
    return {"total": int(total.group(1)) if total else None, "done": len(done),
            "failed": sum(k == "FAIL" for k, _, _ in done), "timeouts": len(re.findall(r"TIMEOUT ", text)),
            "phase": phase, "by_worker": by_worker,
            "failures": [(n, w) for k, n, w in done if k == "FAIL"][:20]}


def progress(run_id: str, live: bool = True) -> dict:
    """Where a run stands: slides done/failed of total, rate, ETA, queue, and
    (live=True) what each worker is doing. Safe to call as often as you like."""
    rec = json.loads(_record(run_id).read_text())
    text = Path(rec["log"]).read_text() if Path(rec["log"]).exists() else ""
    p = parse_log(text)
    total = p["total"] or rec["slides"]
    started = datetime.fromisoformat(rec["started"])
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    rate = p["done"] / elapsed * 3600 if p["done"] and elapsed > 0 else None
    left = total - p["done"]
    if rate:
        eta_h = left / rate
    else:
        eta_h = left * SECONDS_PER_SLIDE_PER_WORKER / max(1, len(hosts())) / 3600
    out = {"run_id": run_id, "phase": p["phase"], "driver_running": is_running(run_id), "total": total,
           "done": p["done"], "failed": p["failed"], "timeouts": p["timeouts"],
           "elapsed_h": round(elapsed / 3600, 2), "slides_per_hour": round(rate, 1) if rate else None,
           "eta_h": round(eta_h, 2) if left else 0.0, "by_worker": p["by_worker"], "failures": p["failures"],
           "log_tail": text.strip().splitlines()[-6:]}
    if live and p["phase"] in ("running", "starting"):
        try:
            _, celery = worker_status_celery_only()
            out["workers_now"] = celery
            out["queued"] = queue_length()
        except Exception as exc:  # noqa: BLE001
            out["workers_now"] = f"unavailable: {exc}"
    return out


def worker_status_celery_only():
    """Celery's view only (fast, no SSH): jobs per worker and current slides."""
    from src.notebook_remote_runtime import create_celery_client, load_notebook_config_from_env

    app = create_celery_client(load_notebook_config_from_env())
    insp = app.control.inspect(timeout=4)
    act = insp.active() or {}
    return None, {w: [Path(t.get("args", [""])[0]).name for t in v] for w, v in act.items()}


def progress_bar(p: dict, width: int = 40) -> str:
    frac = (p["done"] / p["total"]) if p["total"] else 0
    filled = int(round(frac * width))
    return f"[{'#' * filled}{'.' * (width - filled)}] {p['done']}/{p['total']} ({frac:.0%})"


def stop_run(run_id: str, drop_queued: bool = False) -> str:
    """Stops the run's driver (slides already on a worker still finish). With
    drop_queued=True also removes jobs still waiting in the queue, which belong
    to this run only if no other run is going."""
    rec = json.loads(_record(run_id).read_text())
    msg = []
    if is_running(run_id):
        os.killpg(os.getpgid(int(rec["pid"])), signal.SIGTERM)
        msg.append("driver stopped")
    else:
        msg.append("driver not running")
    if drop_queued:
        from src.notebook_remote_runtime import create_celery_client, load_notebook_config_from_env

        n = create_celery_client(load_notebook_config_from_env()).control.purge()
        msg.append(f"{n} queued jobs dropped")
    return "; ".join(msg)


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$")


def verify_drive_copy(run_id: str, remote: str = "gdrive:Feather_Molt_Project/runs") -> tuple[bool, str]:
    """True when every local file of the run (tables and segmentation outputs)
    is on Drive with the same size (rclone check --one-way --size-only)."""
    pairs = [(REPO / "data" / "label_runs" / run_id, f"{remote}/{run_id}"),
             (REPO / "data" / "sample_runs" / run_id, f"{remote}/{run_id}/segmentation")]
    notes = []
    for local, dest in pairs:
        if not local.exists():
            return False, f"no local copy at {local}"
        p = subprocess.run(["rclone", "check", str(local), dest, "--one-way", "--size-only", "-q"],
                           capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            return False, f"{dest}: " + (p.stderr.strip().splitlines() or ["differs"])[-1][:200]
        notes.append(f"{dest} ok")
    return True, "; ".join(notes)


def cleanup_node_outputs(run_id: str, require_drive: bool = True, dry_run: bool = False) -> dict:
    """Deletes data/runs/<run_id>/ on every node, after checking the run's
    results are on Drive (require_drive). Only that run's folder, only a
    plain run id; the slides and everything else on the nodes stay."""
    if not _RUN_ID.match(run_id) or ".." in run_id:
        raise ValueError(f"refusing odd run id {run_id!r}")
    if require_drive:
        ok, why = verify_drive_copy(run_id)
        if not ok:
            return {"deleted": False, "reason": f"Drive copy not verified: {why}"}
    target = f"{remote_repo()}/data/runs/{run_id}"
    out = {}
    for h in hosts():
        cmd = (f"T={target}; [ -d \"$T\" ] || {{ echo 'nothing there'; exit 0; }}; "
               f"du -sh \"$T\" | cut -f1 | tr -d '\\n'; " + ("echo ' (dry run)'" if dry_run else
               "rm -rf -- \"$T\" && echo ' deleted'"))
        p = ssh(h, cmd, timeout=300)
        out[h] = (p.stdout.strip() or p.stderr.strip())[:120]
    return {"deleted": not dry_run, "nodes": out}


def results(run_id: str) -> dict:
    """Where a finished run's outputs are, and its summary."""
    out = {"labels": str(REPO / "data" / "label_runs" / run_id),
           "segmentation": str(REPO / "data" / "sample_runs" / run_id),
           "drive": f"Feather_Molt_Project/runs/{run_id}/ (labels at the top, images under segmentation/)"}
    s = REPO / "data" / "label_runs" / run_id / "summary.json"
    out["summary"] = json.loads(s.read_text()) if s.exists() else None
    return out


def wait_hint(p: dict) -> str:
    if p["phase"] == "finished":
        return "finished"
    if p["eta_h"] is None:
        return "no estimate yet"
    end = datetime.now().astimezone() + timedelta(hours=p["eta_h"])
    return f"about {p['eta_h']:.1f} h left (around {end:%H:%M})"


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(json.dumps(progress(r), indent=1, default=str))
