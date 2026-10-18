"""Score already-segmented crops (e.g. a sample_run folder from workers on older
code) with the per-feather judge, to calibrate the rubric without redeploying.

The crop's position on its slide is recovered by masked template matching
(the crop is the slide's pixels inside the mask, histogram-equalized), and its
non-black pixels give the mask for the judge's context panel.

    python scripts/judge_existing_crops.py data/sample_runs/test-sample30-seg out.csv [--slides data/raw_cache]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import feather_judge, label_extract as lx  # noqa: E402
from src.feather_processing import load_seg_config  # noqa: E402
from src.crop_locate import crop_mask  # noqa: E402
from src.output_names import find_crops  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("crops_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--slides", default="data/raw_cache")
    ap.add_argument("--manifest", help="slide names to include (default: all crops' slides)")
    args = ap.parse_args()
    cfg = load_seg_config()["feather_judge"]
    client = lx.VLMClient(lx.load_config()["vlm"])
    crops_dir, slides_dir = Path(args.crops_dir), Path(args.slides)
    qc_dir = Path(args.out_csv).with_suffix("")
    qc_dir.mkdir(parents=True, exist_ok=True)
    names = ([l.strip() for l in open(args.manifest) if l.strip() and not l.startswith("#")] if args.manifest
             else sorted({p.name.split("_")[0] + ".jpg" for p in crops_dir.glob("*_BoundingBoxes.jpg")}))
    rows = []
    for n in names:
        slide = cv2.imread(str(slides_dir / n))
        if slide is None:
            print("missing slide", n)
            continue
        for i, cp in enumerate(find_crops(crops_dir, n)):
            crop = cv2.imread(str(cp))
            mask, match = crop_mask(slide, crop)
            res, panel = feather_judge.judge(client, slide, mask, crop, cfg)
            qc = qc_dir / (cp.stem + "_qc.jpg")
            panel.save(qc, quality=85)
            row = {"file_id": n, "seq": lx.seq_letter(i), "crop": cp.name, "match": round(match, 3),
                   "qc_image": str(qc), **{"judge_" + k: v for k, v in res.items() if k != "raw"}}
            rows.append(row)
            print(n, row["seq"], row.get("judge_score"), row.get("judge_status"), row.get("judge_notes", ""),
                  row.get("judge_error", ""), flush=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out_csv, len(rows), "feathers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
