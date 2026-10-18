"""Clean up feather masks after SAM 3 (or SAM 2.1): one mask per feather.

SAM sometimes returns a feather as two masks: a thin sliver of one vane next
to the rest, or two pieces where a black board crosses the feather. Box NMS
keeps both because their boxes barely overlap. These rules run on the masks
themselves (thresholds in configs/segmentation.json -> "postprocess"):

  1. contained   a mask lying mostly inside another one is dropped
  2. fragment    a small or narrow mask (vs the slide's typical feather) that
                 touches another mask is merged into it
  3. aligned     two pieces on the same axis, end to end with a small gap
                 (a feather split by a board edge), are merged
Every action is logged, so the per-feather table can say what happened.
Isolated fragments are kept but flagged (a slide may hold one small feather).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULTS = {
    "enabled": True,
    "contained_frac": 0.6,        # rule 1: share of the smaller mask inside the other
    "fragment_area_frac": 0.35,   # rule 2: area below this share of the median feather area ...
    "fragment_width_frac": 0.45,  # ... or width below this share of the median feather width
    "touch_px": 6,                # rule 2: masks closer than this (pixels at slide size) touch
    "align_angle_deg": 12.0,      # rule 3: axes within this angle
    "align_offset_frac": 0.6,     # rule 3: sideways offset below this share of the wider width
    "align_gap_frac": 0.2,        # rule 3: gap along the axis below this share of the joint length
    "min_area_frac": 0.002,       # masks smaller than this share of the slide are noise
}


@dataclass(eq=False)
class Piece:
    mask: np.ndarray              # bool, slide size
    score: float
    sources: list[int] = field(default_factory=list)   # indices of the SAM masks it is made of
    notes: list[str] = field(default_factory=list)


def shape_stats(mask: np.ndarray) -> dict:
    """Area, length/width/angle of the minimum-area rectangle, centre, bbox."""
    m = mask.astype(np.uint8)
    area = int(m.sum())
    ys, xs = np.nonzero(m)
    if area == 0:
        return {"area": 0, "length": 0.0, "width": 0.0, "angle": 0.0, "cx": 0.0, "cy": 0.0, "bbox": (0, 0, 0, 0)}
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), (w, h), _ = cv2.minAreaRect(pts)
    length, width = max(w, h), min(w, h)
    # long-axis direction from the principal component (OpenCV's rectangle angle
    # convention changed between versions)
    cov = np.cov((pts - pts.mean(axis=0)).T)
    vx, vy = np.linalg.eigh(cov)[1][:, -1]
    angle = np.degrees(np.arctan2(vy, vx))
    return {"area": area, "length": float(length), "width": float(width), "angle": float(angle % 180),
            "cx": float(cx), "cy": float(cy),
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)}


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _touch(a: np.ndarray, b: np.ndarray, px: int) -> bool:
    if px <= 0:
        return bool((a & b).any())
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return bool((cv2.dilate(a.astype(np.uint8), k).astype(bool) & b).any())


def _aligned(sa: dict, sb: dict, cfg: dict) -> bool:
    """End-to-end pieces of one feather: same axis, little sideways offset, small gap."""
    if _angle_diff(sa["angle"], sb["angle"]) > cfg["align_angle_deg"]:
        return False
    theta = np.deg2rad((sa["angle"] + sb["angle"]) / 2)
    axis = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-axis[1], axis[0]])
    d = np.array([sb["cx"] - sa["cx"], sb["cy"] - sa["cy"]])
    along, across = abs(d @ axis), abs(d @ normal)
    if across > cfg["align_offset_frac"] * max(sa["width"], sb["width"]):
        return False
    gap = along - (sa["length"] + sb["length"]) / 2
    return gap < cfg["align_gap_frac"] * (sa["length"] + sb["length"])


def clean_masks(masks: list[np.ndarray], scores: list[float], cfg: dict | None = None) -> list[Piece]:
    """`masks`: bool arrays at slide size. Returns one Piece per feather."""
    cfg = {**DEFAULTS, **(cfg or {})}
    pieces = [Piece(m.astype(bool), float(s), [i]) for i, (m, s) in enumerate(zip(masks, scores))]
    if not cfg["enabled"] or not pieces:
        return pieces
    slide_area = pieces[0].mask.size
    pieces = [p for p in pieces if p.mask.sum() >= cfg["min_area_frac"] * slide_area]

    # 1. contained: drop the smaller mask when most of it lies inside another
    pieces.sort(key=lambda p: -p.mask.sum())
    kept: list[Piece] = []
    for p in pieces:
        area = p.mask.sum()
        host = next((k for k in kept if (p.mask & k.mask).sum() >= cfg["contained_frac"] * area), None)
        if host is not None:
            host.notes.append(f"absorbed contained mask {p.sources}")
            host.mask |= p.mask
            host.sources += p.sources
        else:
            kept.append(p)
    pieces = kept

    # 2 + 3: merge fragments into touching masks, and aligned pieces, until stable
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        stats = [shape_stats(p.mask) for p in pieces]
        med_area = float(np.median([s["area"] for s in stats]))
        med_width = float(np.median([s["width"] for s in stats]))
        frag = [s["area"] < cfg["fragment_area_frac"] * med_area or s["width"] < cfg["fragment_width_frac"] * med_width
                for s in stats]
        n = len(pieces)
        best = next(((i, j, "fragment touching") for i in range(n) for j in range(n)
                     if i != j and frag[i] and _touch(pieces[i].mask, pieces[j].mask, cfg["touch_px"])), None)
        if best is None:
            best = next(((i, j, "aligned pieces") for i in range(n) for j in range(i + 1, n)
                         if _aligned(stats[i], stats[j], cfg)
                         and (frag[i] or frag[j] or _touch(pieces[i].mask, pieces[j].mask, cfg["touch_px"] * 8))),
                        None)
        if best:
            i, j, why = best
            host, part = (pieces[j], pieces[i]) if why == "fragment touching" else \
                ((pieces[i], pieces[j]) if stats[i]["area"] >= stats[j]["area"] else (pieces[j], pieces[i]))
            host.mask = host.mask | part.mask
            host.score = max(host.score, part.score)
            host.sources += part.sources
            host.notes += [f"merged {why} {part.sources}"] + part.notes
            pieces.remove(part)
            changed = True

    # flag what is still small or narrow
    if len(pieces) > 1:
        stats = [shape_stats(p.mask) for p in pieces]
        med_area = float(np.median([s["area"] for s in stats]))
        med_width = float(np.median([s["width"] for s in stats]))
        for p, s in zip(pieces, stats):
            if s["area"] < cfg["fragment_area_frac"] * med_area or s["width"] < cfg["fragment_width_frac"] * med_width:
                p.notes.append("possible fragment (small or narrow, not touching another feather)")
    return pieces
