"""Duplicate feathers across slides of the same bird and day.

Several slides of one bird from the same day often show the same feather
again, in another position, order or rotation (or turned over). Those repeats
are of little use later, so each feather crop gets `duplicate_of`: empty for
the first occurrence, else the feather ID it repeats. "First" = earliest image
number, then left-to-right letter.

Matching is rotation- and position-invariant:
  1. shape: length and width of the feather mask's minimum-area rectangle
     (same scanner, so the same feather has the same size in pixels)
  2. texture: SIFT keypoints on the crop (also against its mirror image),
     ratio test, then a RANSAC similarity transform; the number of inliers
     and the fitted scale (should be ~1) decide.
Only feathers on DIFFERENT slides of the same group are compared, and each
feather matches at most one feather per other slide (best score first).

    python -m src.feather_duplicates <processed_dir>     # uses labels/<stem>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from src.output_names import compact_date, parse_filename

# Calibrated on the 63 same-year slides (see README); raise MIN_INLIERS to be stricter.
MIN_INLIERS = 25
MIN_INLIER_RATIO = 0.25
SIZE_TOLERANCE = 0.12          # length/width may differ by 12 %
SCALE_TOLERANCE = 0.15         # fitted scale must be within 15 % of 1


@dataclass
class Feather:
    feather_id: str
    slide: str                 # file_id
    seq: str
    crop: Path


def _image_number(file_id: str) -> int:
    m = re.search(r"-img?(\d+)", Path(file_id).stem)
    return int(m.group(1)) if m else 0


def _load(path: Path, max_side: int = 900) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        raise OSError(f"cannot read {path}")
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 12).astype(np.uint8) * 255                 # crops sit on black
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return gray, mask


def _shape(mask: np.ndarray) -> tuple[float, float]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return max(w, h), min(w, h)


class _Features:
    _sift = None

    def __init__(self, path: Path):
        gray, mask = _load(path)
        self.length, self.width = _shape(mask)
        if _Features._sift is None:
            _Features._sift = cv2.SIFT_create(nfeatures=1500)
        self.kp, self.des = _Features._sift.detectAndCompute(gray, mask)
        flipped = cv2.flip(gray, 1)
        self.kp_f, self.des_f = _Features._sift.detectAndCompute(flipped, cv2.flip(mask, 1))


def _ransac(kp_a, des_a, kp_b, des_b) -> tuple[int, int, float]:
    """(inliers, good matches, fitted scale) of a similarity transform a -> b."""
    if des_a is None or des_b is None or len(des_a) < 8 or len(des_b) < 8:
        return 0, 0, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 6:
        return 0, len(good), 0.0
    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([kp_b[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
    if M is None or inl is None:
        return 0, len(good), 0.0
    return int(inl.sum()), len(good), float(np.hypot(M[0, 0], M[1, 0]))


def compare(a: _Features, b: _Features) -> dict:
    """Similarity of two feather crops (b also tried mirrored)."""
    size_ok = (a.length > 0 and b.length > 0
               and abs(a.length - b.length) / max(a.length, b.length) <= SIZE_TOLERANCE
               and abs(a.width - b.width) / max(a.width, b.width, 1) <= SIZE_TOLERANCE * 2)
    best = {"inliers": 0, "good": 0, "scale": 0.0, "mirrored": False}
    for mirrored, (kp, des) in ((False, (b.kp, b.des)), (True, (b.kp_f, b.des_f))):
        inl, good, scale = _ransac(a.kp, a.des, kp, des)
        if inl > best["inliers"]:
            best = {"inliers": inl, "good": good, "scale": scale, "mirrored": mirrored}
    ratio = best["inliers"] / best["good"] if best["good"] else 0.0
    scale_ok = abs(best["scale"] - 1) <= SCALE_TOLERANCE
    best.update({"inlier_ratio": round(ratio, 3), "size_ok": size_ok, "scale_ok": scale_ok,
                 "length_a": round(a.length), "length_b": round(b.length),
                 "same": bool(size_ok and scale_ok and best["inliers"] >= MIN_INLIERS
                              and ratio >= MIN_INLIER_RATIO)})
    return best


def group_key(file_id: str, date: str | None) -> tuple[str, str] | None:
    """Same bird (file-name ID) and same day (label date); slides whose date
    is unknown are grouped by year, so they are compared with all siblings."""
    fn = parse_filename(file_id)
    if not fn["id"]:
        return None
    return fn["id"], compact_date(date) or fn["year"]


def find_duplicates(slides: dict[str, list[Feather]], dates: dict[str, str | None]) -> list[dict]:
    """`slides`: file_id -> its feathers (left to right). Returns one row per
    feather in a multi-slide group: feather_id, duplicate_of, match details."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for fid in slides:
        key = group_key(fid, dates.get(fid))
        if key:
            groups[key].append(fid)
    # an unknown date (year key) may be any day of that year: merge into the bird's day groups
    for (bird, period), members in list(groups.items()):
        if len(period) == 4:
            for (b2, p2), m2 in groups.items():
                if b2 == bird and p2 != period and p2.startswith(period):
                    m2.extend(x for x in members if x not in m2)
    rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    parent: dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    order = {f.feather_id: (_image_number(f.slide), f.slide, f.seq) for fs in slides.values() for f in fs}
    details: dict[str, dict] = {}
    for key, members in groups.items():
        if len(members) < 2:
            continue
        feats = {f.feather_id: _Features(f.crop) for fid in members for f in slides[fid]}
        for s1, s2 in combinations(sorted(set(members), key=_image_number), 2):
            if (s1, s2) in seen_pairs:
                continue
            seen_pairs.add((s1, s2))
            scored = []
            for f1 in slides[s1]:
                for f2 in slides[s2]:
                    c = compare(feats[f1.feather_id], feats[f2.feather_id])
                    if c["same"]:
                        scored.append((c["inliers"], f1.feather_id, f2.feather_id, c))
            used1, used2 = set(), set()
            for _, a, b, c in sorted(scored, key=lambda t: -t[0]):
                if a in used1 or b in used2:
                    continue
                used1.add(a)
                used2.add(b)
                ra, rb = find(a), find(b)
                if ra != rb:
                    keep, drop = sorted((ra, rb), key=lambda x: order[x])
                    parent[drop] = keep
                details.setdefault(b, c)
                details.setdefault(a, c)
        for fid in members:
            for f in slides[fid]:
                root = find(f.feather_id)
                c = details.get(f.feather_id, {})
                rows.append({"feather_id": f.feather_id, "file_id": fid, "seq": f.seq, "crop": f.crop.name,
                             "group": f"{key[0]}_{key[1]}",
                             "duplicate_of": "" if root == f.feather_id else root,
                             "match_inliers": c.get("inliers", ""), "match_mirrored": c.get("mirrored", ""),
                             "match_scale": round(c["scale"], 3) if c.get("scale") else ""})
    # a feather can be listed from two merged groups; keep one row each
    uniq = {}
    for r in rows:
        if r["feather_id"] not in uniq or r["duplicate_of"]:
            uniq[r["feather_id"]] = r
    return sorted(uniq.values(), key=lambda r: order[r["feather_id"]])


def apply_to_results(results, crops_dir: str | Path) -> list[dict]:
    """Marks duplicate_of on the feathers of label results (label_extract)."""
    from src.label_extract import result_date

    slides = {r.file_id: [Feather(f["feather_id"], r.file_id, f["seq"], Path(crops_dir) / f["crop"])
                          for f in r.feathers if f.get("crop") and (Path(crops_dir) / f["crop"]).is_file()]
              for r in results if r.feathers}
    rows = find_duplicates(slides, {r.file_id: result_date(r) for r in results})
    dup = {r["feather_id"]: r for r in rows}
    for r in results:
        for f in r.feathers:
            d = dup.get(f.get("feather_id"))
            f["duplicate_of"] = d["duplicate_of"] if d else ""
            f["duplicate_group"] = d["group"] if d else ""
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    cols = ["group", "file_id", "seq", "feather_id", "duplicate_of", "match_inliers", "match_mirrored",
            "match_scale", "crop"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    """Post-run step on a gathered processed folder with labels/<stem>.json."""
    from src.label_extract import LabelResult, save_pipeline_result

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("processed_dir")
    args = ap.parse_args(argv)
    d = Path(args.processed_dir)
    results = []
    for p in sorted((d / "labels").glob("*.json")):
        if p.name.endswith(".date.json"):
            continue
        results.append(LabelResult.from_json(json.loads(p.read_text())))
    rows = apply_to_results(results, d)
    for r in results:
        if r.feathers:
            save_pipeline_result(r, d)
    out = write_csv(rows, d / "feather_duplicates.csv")
    n = sum(bool(r["duplicate_of"]) for r in rows)
    print(f"{len(rows)} feathers in same-day groups, {n} duplicates -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
