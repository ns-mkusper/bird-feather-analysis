"""Small segmentation runs on the cluster, results copied back here.

Sends N slides through the workers' normal process_image task (Grounding DINO
+ SAM + crops, with whatever VLM settings the workers run with), waits, then
copies the outputs into data/sample_runs/<run_id>/ for inspection. Run ids
start with "test-", so the Drive run sync ignores them.

    python -m src.sample_run --n 5 --seed 2
    python -m src.sample_run --match "A0012 1996" --match BB17692
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.notebook_remote_runtime import (
    RemoteClusterIO,
    assert_tcp_reachable,
    create_celery_client,
    load_notebook_config_from_env,
)

REPO = Path(__file__).resolve().parent.parent


def run(n: int = 5, seed: int = 0, match: list[str] | None = None, timeout: int = 900,
        run_id: str | None = None, options: dict | None = None) -> Path:
    """`options`: {'segmentation': {...}} overrides for the workers' segmentation
    config in this run (needs workers on the feather-judge code or later)."""
    config = load_notebook_config_from_env()
    assert_tcp_reachable(config.broker_url, "Broker")
    app = create_celery_client(config)
    io = RemoteClusterIO(config)

    images = io.list_remote_images()
    if match:
        chosen = [p for p in images if any(m in Path(p).name for m in match)]
    else:
        chosen = random.Random(seed).sample(images, min(n, len(images)))
    run_id = run_id or "test-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    remote_out = io.remote_runtime_path(f"{config.remote_repo_dir}/data/runs/{run_id}/processed")
    local_out = REPO / "data" / "sample_runs" / run_id
    print(f"{run_id}: {len(chosen)} slides -> workers")

    task_args = (lambda p: [p, remote_out, options]) if options else (lambda p: [p, remote_out])
    pending = {app.send_task("src.celery_tasks.process_image", args=task_args(p)): p for p in chosen}
    deadline = time.time() + timeout
    done: dict[str, dict] = {}
    while pending and time.time() < deadline:
        for res in list(pending):
            if res.ready():
                p = pending.pop(res)
                try:
                    done[p] = res.get(timeout=5) or {}
                except Exception as exc:  # noqa: BLE001
                    done[p] = {"success": False, "reason": f"{type(exc).__name__}: {exc}"}
                info = done[p]
                print(f"  {'ok ' if info.get('success') else 'FAIL'} {Path(p).name} "
                      f"on {info.get('worker', '?')} {info.get('reason') or ''}", flush=True)
        time.sleep(2)
    for p in pending.values():
        print(f"  TIMEOUT {Path(p).name} (still queued or running)")

    # Each worker writes to its own disk; collect from all of them.
    from src.output_names import belongs_to

    slide_names = [Path(p).name for p in chosen]
    got = 0
    for host in config.cluster_hosts:
        cmd = f"find {io._remote_shell_path(remote_out)} -maxdepth 1 -type f -iname '*.jpg' | sort"
        # The folder only exists on hosts that ran one of the tasks.
        proc = subprocess.run([*io._ssh_base_cmd(host), cmd], capture_output=True, text=True)
        listing = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        for f in listing:
            if any(belongs_to(Path(f).name, n) for n in slide_names):
                io.fetch_remote_file(host, f, local_out)
                got += 1
        # per-feather judge panels (new worker code), in the feather_qc/ subfolder
        qc = f"{remote_out}/feather_qc"
        cmd = f"find {io._remote_shell_path(qc)} -maxdepth 1 -type f -iname '*.jpg' | sort"
        proc = subprocess.run([*io._ssh_base_cmd(host), cmd], capture_output=True, text=True)
        for f in (line.strip() for line in proc.stdout.splitlines() if line.strip()):
            if any(belongs_to(Path(f).name.replace("_qc.jpg", ".jpg"), n) for n in slide_names):
                io.fetch_remote_file(host, f, local_out / "feather_qc")
                got += 1
    # what each task returned (success, reason, per-feather records with judge scores, labels)
    local_out.mkdir(parents=True, exist_ok=True)
    import json

    (local_out / "task_results.json").write_text(json.dumps({Path(p).name: done.get(p) for p in chosen},
                                                            indent=1, default=str))
    print(f"copied {got} output files to {local_out}")
    return local_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match", action="append")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the workers")
    ap.add_argument("--seg-options", help="JSON file with segmentation overrides, e.g. {\"sam3\": {\"conf\": 0.3}}")
    args = ap.parse_args(argv)
    import json

    options = {"segmentation": json.loads(Path(args.seg_options).read_text())} if args.seg_options else None
    run(args.n, args.seed, args.match, args.timeout, options=options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
