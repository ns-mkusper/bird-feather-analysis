"""Run a slice of slides through the cluster with the full new pipeline, then
build the tables, mark duplicates, score against the hand-checked answers, and
sync to Drive.

    python scripts/run_slice.py --manifest configs/slice_100.txt --run-id slice100-20261018

Per slide the workers (this branch's code) read the label, segment with SAM 3,
clean up the masks, check for missed feathers, judge every feather, and name
the outputs. Results land in:
    data/sample_runs/<run-id>/      crops, overlays, feather_qc/ panels, task_results.json
    data/label_runs/<run-id>/       labels.csv (per slide), feathers.csv (per feather), review.csv,
                                    feather_duplicates.csv, labels.sqlite3, summary.json
and are copied to gdrive:Feather_Molt_Project/runs/<run-id>/ (tables at the top,
images under segmentation/).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import label_extract as lx  # noqa: E402
from src import prompt_lab as pl  # noqa: E402
from src import sample_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="configs/slice_100.txt")
    ap.add_argument("--run-id", default="slice100-" + time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    ap.add_argument("--gold", default="data/label_gold.csv")
    ap.add_argument("--remote", default="gdrive:Feather_Molt_Project/runs")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--handwriting", help="hand-check CSV of handwritten labels (file_id, handwritten, fields, note, "
                                          "source); wins over the model's check")
    ap.add_argument("--keep-node-outputs", action="store_true",
                    help="keep data/runs/<run-id>/ on the nodes after the Drive upload (default: delete once verified)")
    ap.add_argument("--allow-old-workers", action="store_true", help="run even if the workers lack the new code")
    args = ap.parse_args()

    ok, registered = pl.workers_have_options_support()
    if not ok and not args.allow_old_workers:
        print("The workers do not run this branch's code yet (no extract_labels task):", registered)
        print("Deploy first: FEATHER_REPO_BRANCH=feather-judge ./run_cluster.sh from a machine with "
              "node access (or merge the branch), then rerun this script.")
        return 2
    names = [l.strip() for l in open(args.manifest) if l.strip() and not l.startswith("#")]
    print(f"{args.run_id}: {len(names)} slides -> workers", flush=True)
    t0 = time.time()
    seg_dir = sample_run.run(match=names, timeout=args.timeout, run_id=args.run_id)
    print(f"cluster part done in {(time.time() - t0) / 60:.0f} min", flush=True)

    # tables from what each task returned (labels + per-feather records)
    res = json.loads((seg_dir / "task_results.json").read_text())
    out = Path("data/label_runs") / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.txt").write_text("\n".join(names) + "\n")
    (out / "config_used.json").write_text(json.dumps(lx.load_config(), indent=2))
    failed = []
    with open(out / "labels.jsonl", "w") as fh:
        for n in names:
            item = res.get(n) or {}
            if item.get("labels"):
                r = lx.LabelResult.from_json(item["labels"])
            elif item.get("feather_ids"):
                r = lx.bare_result(item.get("image_path", n), item["feather_ids"])
            else:
                failed.append((n, item.get("reason") or "no result"))
                continue
            if item.get("feather_ids"):
                r.feathers = item["feather_ids"] if not r.feathers else r.feathers
            fh.write(json.dumps(r.to_json()) + "\n")
    try:   # all slide names, for photos of the same bird in the same year
        from src.notebook_remote_runtime import RemoteClusterIO, load_notebook_config_from_env

        all_names = [Path(p).name for p in RemoteClusterIO(load_notebook_config_from_env()).list_remote_images()]
    except Exception as exc:  # noqa: BLE001
        print(f"could not list all slides ({exc}); photos_same_year left empty")
        all_names = None
    summary = lx.finish_run(out, names, lx.load_config(), args.gold, seg_dir, all_names, handwriting=args.handwriting)
    summary["slides_without_result"] = failed
    summary["task_failures"] = [(n, (res.get(n) or {}).get("reason")) for n in names
                                if (res.get(n) or {}).get("success") is False]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    lx.print_summary(summary, out)
    print(f"feathers {summary.get('feathers')} ({summary.get('feathers_usable')} usable), "
          f"fragments {summary.get('fragments')}, duplicates {summary.get('duplicate_feathers')}, "
          f"needing review {summary.get('feathers_needing_review')}, mean judge score {summary.get('feather_score_mean')}")
    if failed:
        print("no result for:", failed[:10])
    if not args.no_upload:
        dest = f"{args.remote}/{args.run_id}"
        subprocess.run(["rclone", "copy", str(out), dest, "-q"], check=True)
        subprocess.run(["rclone", "copy", str(seg_dir), f"{dest}/segmentation", "-q"], check=True)
        print("copied to", dest)
        if not args.keep_node_outputs:
            from src import cluster_control as cc

            res = cc.cleanup_node_outputs(args.run_id, require_drive=True)
            print("node outputs:", res.get("reason") or res.get("nodes"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
