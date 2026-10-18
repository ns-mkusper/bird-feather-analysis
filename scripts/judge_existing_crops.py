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
from src.output_names import find_crops  # noqa: E402


def locate(slide_gray: np.ndarray, crop_bgr: np.ndarray, scale: float = 0.25) -> tuple[int, int, float]:
    """Top-left (x, y) of the crop on the slide and the match score."""
    cg = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    m = (cg > 8).astype(np.uint8)
    s_small = cv2.resize(slide_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    c_small = cv2.resize(cg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m_small = cv2.resize(m, (c_small.shape[1], c_small.shape[0]), interpolation=cv2.INTER_NEAREST)
    # compare edges: equalization changes brightness but keeps edges
    e_s = cv2.Canny(s_small, 30, 90).astype(np.float32)
    e_c = cv2.Canny(c_small, 30, 90).astype(np.float32) * m_small
    res = cv2.matchTemplate(e_s, e_c, cv2.TM_CCORR_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    # refine at full resolution in a small window
    X, Y = int(x / scale), int(y / scale)
    r = int(3 / scale)
    y0, x0 = max(0, Y - r), max(0, X - r)
    win = slide_gray[y0:Y + r + cg.shape[0], x0:X + r + cg.shape[1]]
    if win.shape[0] >= cg.shape[0] and win.shape[1] >= cg.shape[1]:
        res2 = cv2.matchTemplate(cv2.Canny(win, 30, 90).astype(np.float32),
                                 cv2.Canny(cg, 30, 90).astype(np.float32) * m, cv2.TM_CCORR_NORMED)
        _, score, _, (dx, dy) = cv2.minMaxLoc(res2)
        X, Y = x0 + dx, y0 + dy
    return X, Y, float(score)


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
        gray = cv2.cvtColor(slide, cv2.COLOR_BGR2GRAY)
        for i, cp in enumerate(find_crops(crops_dir, n)):
            crop = cv2.imread(str(cp))
            x, y, match = locate(gray, crop)
            mask = np.zeros(gray.shape, bool)
            cm = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 8
            h, w = cm.shape
            mask[y:y + h, x:x + w] = cm[: gray.shape[0] - y, : gray.shape[1] - x]
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
