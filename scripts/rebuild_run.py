"""Rebuild a finished run's crops and tables under the current rules, without the cluster or the
model: crops from the saved masks (feather_masks/) with the current glue/card and calamus clean-up,
label statuses re-derived from the saved readings (label_extract.reclassify), plus the slide-level
columns (src/slide_annotations.py). Everything goes to NEW folders; the run's own files are only read.

    python scripts/rebuild_run.py slice100-final-20261020 --to slice100-final-20261020-v2 \\
        --handwriting data/review/label_handwriting_slice100.csv

Slide names for 'photos in the same year' come from the April run's table of all 2,406 slides
(--all-names, default data/before_run/image_segmentation_qc_20260426-075440.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src import label_extract as lx  # noqa: E402
from src.feather_processing import equalized, load_seg_config, make_feather  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--to", required=True, help="name of the new run folder (data/label_runs/, data/sample_runs/)")
    ap.add_argument("--handwriting", help="CSV: file_id, handwritten, fields, note, source")
    ap.add_argument("--all-names", default=str(REPO / "data/before_run/image_segmentation_qc_20260426-075440.csv"))
    ap.add_argument("--slides", default=str(REPO / "data/raw_cache"))
    args = ap.parse_args()

    src_lab, src_seg = REPO / "data/label_runs" / args.run, REPO / "data/sample_runs" / args.run
    out_lab, out_seg = REPO / "data/label_runs" / args.to, REPO / "data/sample_runs" / args.to
    for d in (out_lab, out_seg):
        if d.exists():
            sys.exit(f"{d} exists: choose another --to (this script never overwrites)")
    out_lab.mkdir(parents=True)
    out_seg.mkdir(parents=True)
    # read-only copies of what does not change: masks, check panels, overlays, label records
    for sub in ("feather_masks", "feather_qc", "labels"):
        if (src_seg / sub).exists():
            shutil.copytree(src_seg / sub, out_seg / sub, ignore=shutil.ignore_patterns("evidence"))
    for p in src_seg.glob("*_BoundingBoxes.jpg"):
        shutil.copy2(p, out_seg / p.name)
    for name in ("manifest.txt", "labels.jsonl", "feather_duplicates.csv"):
        if (src_lab / name).exists():
            shutil.copy2(src_lab / name, out_lab / name)

    seg_cfg = load_seg_config()
    trim = float(seg_cfg["crop"].get("trim_bottom_frac", 0.12))
    cleanup = {**(seg_cfg["crop"].get("cleanup") or {}), "enabled": True}
    lab_cfg = lx.load_config()
    (out_lab / "config_used.json").write_text(json.dumps(lab_cfg, indent=2))

    # crops from the saved masks, with the current clean-up
    latest = lx.load_previous(out_lab / "labels.jsonl")
    n = 0
    with open(out_lab / "labels.jsonl", "w") as fh:
        for r in latest.values():
            img = None
            for f in r.feathers:
                if not f.get("crop") or not f.get("mask_image"):
                    continue
                if img is None:
                    img = cv2.imread(str(Path(args.slides) / r.file_id))
                full = cv2.imread(str(out_seg / f["mask_image"]), 0) > 0
                feather = make_feather(img, full, trim, 0, cleanup)
                cv2.imwrite(str(out_seg / f["crop"]), equalized(feather["crop"]))
                f["bbox"], f["mask_cleanup"] = feather["record"]["bbox"], feather["record"]["mask_cleanup"]
                n += 1
            fh.write(json.dumps(r.to_json()) + "\n")
            (out_seg / "labels" / f"{Path(r.image).stem}.json").write_text(json.dumps(r.to_json()))
    print(f"{n} crops rebuilt -> {out_seg}")

    names = [ln.strip() for ln in (out_lab / "manifest.txt").read_text().splitlines() if ln.strip()]
    all_names = [Path(row["image_path"]).name for row in csv.DictReader(open(args.all_names, newline=""))]
    summary = lx.finish_run(out_lab, names, lab_cfg, REPO / "data/label_gold.csv", out_seg, all_names,
                            handwriting=args.handwriting)
    print(json.dumps({k: summary[k] for k in ("slides", "needs_review", "feathers", "feathers_usable",
                                              "slides_with_label_problems", "fields_partly_visible",
                                              "fields_inferred_from_same_bird",
                                              "slides_with_multiple_photos_same_year", "slides_handwritten")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
