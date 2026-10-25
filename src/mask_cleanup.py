"""Remove glue and card from a feather mask before it is cropped.

Hand verdicts (notebooks/feather_verdicts.ipynb) showed the most common fault of a saved crop
is glue or card at the quill base: the fixed bottom trim (crop.trim_bottom_frac) often leaves
part of the glue dot. Two measurements find it:

  texture  barbs give strongly oriented local texture (structure-tensor coherence high);
           glue and card do not. Patches of the mask without it that touch the mask's edge
           are not feather. In the bottom part of the feather they are cut off with
           everything below them; elsewhere on the edge they are removed.
  neck     going up from the bottom, the mask goes wide (glue dot), narrow (bare quill),
           wide (vane). A narrow quill with a much wider blob below it is cut at its
           narrowest row. This catches glue too translucent for the texture test.
  calamus  the analysis uses the vane only: the bare quill below it is cut on every feather
           (crop.cleanup.calamus).
  vane     on the white card the downy lower vane is see-through, and the quill carries a
  base     translucent collar and glue specks: there the feather is cut where its vane
           narrows below half its typical width (coming down from the middle). On the black
           board nothing is cut this way.

Settings: configs/segmentation.json, crop.cleanup.
"""
from __future__ import annotations

import cv2
import numpy as np


def coherence(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Structure-tensor coherence (0 = no dominant direction, 1 = parallel lines)."""
    g = gray.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    k = int(6 * sigma) | 1
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (k, k), sigma) for a in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy
    return np.where(tr > 1e-3, np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-6), 0)


def textureless_patches(img: np.ndarray, mask: np.ndarray, cfg: dict) -> list[np.ndarray]:
    """Mask patches without barb texture that touch the mask's edge (slide-sized bool masks)."""
    ys, xs = np.nonzero(mask)
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    m = mask[y0:y1, x0:x1]
    coh = coherence(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), float(cfg.get("sigma", 4.0)))
    low = (coh < float(cfg.get("coherence", 0.25))) & m
    low = cv2.morphologyEx(low.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
    edge = m & ~(cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), 8)
    touching = set(np.unique(lbl[edge & low]).tolist()) - {0}
    min_px = float(cfg.get("min_patch_frac", 0.004)) * m.sum()
    out = []
    for i in sorted(touching):
        if st[i, cv2.CC_STAT_AREA] < min_px:
            continue
        p = np.zeros(mask.shape, bool)
        p[y0:y1, x0:x1] = lbl == i
        out.append(p)
    return out


def neck_row(mask: np.ndarray, cfg: dict) -> int | None:
    """Row of a narrow quill neck with a much wider blob (glue dot) below it, or None."""
    ys, _ = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0 + 1
    rows = np.arange(y1 - int(float(cfg.get("neck_search_frac", 0.35)) * h), y1 + 1)
    width = np.convolve(mask[rows].sum(axis=1).astype(float), np.ones(5) / 5, mode="same")
    width[:2] = width[2]; width[-2:] = mask[rows[-2:]].sum(axis=1)       # no smoothing artefact at the ends
    vane = float(np.median(mask[y0 + h // 4:y0 + h // 2].sum(axis=1)))
    blob, neck_max = float(cfg.get("neck_blob_ratio", 2.5)), float(cfg.get("neck_max_frac", 0.12)) * vane
    best = None
    for i in range(3, len(rows) - 3):
        w = width[i]
        if 0 < w <= neck_max and width[i + 1:].max() >= blob * w and (best is None or w < width[best]):
            best = i
    return None if best is None else int(rows[best])


def _above_glue(mask: np.ndarray, top: int, height: int, vane: float, cfg: dict) -> int | None:
    """Where to cut above a glue patch found at `top`: the texture test sees only the glue's
    flattest part, so look up to `glue_search_frac` of the feather's height above it for the
    mask's narrowest row (the bare quill above the glue dot), follow the quill up while it stays
    narrow, and cut there (just below the vane). None if there is no bare quill (no row narrower
    than `quill_max_frac` of the vane): then only the patch itself is removed."""
    lo = max(4, top - int(float(cfg.get("glue_search_frac", 0.15)) * height))
    rows = np.arange(lo, top + 1)
    padded = mask[lo - 4:min(mask.shape[0], top + 5)].sum(axis=1).astype(float)
    width = np.convolve(padded, np.ones(9) / 9, mode="valid")[:len(rows)]   # no shrinking at the ends
    width[width == 0] = np.inf
    i = int(np.argmin(width))
    if width[i] > float(cfg.get("quill_max_frac", 0.6)) * vane:
        return None
    # up the bare quill while it stays narrow, to just below the vane: the glue dot's upper
    # edge often reaches the trim line, so cutting at the narrowest point alone leaves a sliver
    while i > 0 and width[i - 1] <= float(cfg.get("quill_widen", 1.3)) * width[int(np.argmin(width))]:
        i -= 1
    return max(0, int(rows[i]) - int(float(cfg.get("glue_margin_frac", 0.005)) * height))


def vane_base_row(img: np.ndarray, mask: np.ndarray, cfg: dict) -> int | None:
    """Row where the vane, coming down from its middle, narrows below `min_width_frac` of its
    typical width for a sustained stretch, if the slide around the part below is light card.
    None when there is no such row in the bottom `search_frac` of the feather or it lies on the board."""
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0 + 1
    width = mask[y0:y1 + 1].sum(axis=1).astype(float)
    width = np.convolve(np.pad(width, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    vane = float(np.median(width[h // 4:h // 2]))
    run = max(3, int(float(cfg.get("run_frac", 0.015)) * h))
    narrow = width < float(cfg.get("min_width_frac", 0.5)) * vane
    start = max(h // 2, h - int(float(cfg.get("search_frac", 0.4)) * h))
    row = None
    for i in range(start, h - run):
        if narrow[i:i + run].all():
            row = i
            break
    if row is None:
        return None
    # only on the card, never above it: a gradually tapering vane narrows while still on the black
    # board, where nothing shows through. Coming down from the narrowing, find the first rows where
    # the slide just outside the feather is light card and cut no higher than that.
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    near = cv2.dilate(mask.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    ring = near & ~(cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    light_min = float(cfg.get("card_min_lightness", 150))
    xc = int(np.median(xs))
    card_rows = []
    for y in range(y0 + row, y1 + 1):          # card on either side (a board edge often runs diagonally)
        sides = [L[y][:xc][ring[y][:xc]], L[y][xc:][ring[y][xc:]]]
        card_rows.append(any(len(v) >= 10 and float(np.median(v)) >= light_min for v in sides))
    card_rows = np.array(card_rows)
    for i in range(len(card_rows) - run):
        if card_rows[i:i + run].all():
            return int(y0 + row + i)
    return None


def calamus_row(mask: np.ndarray, cfg: dict) -> int | None:
    """Top of the bare quill (calamus) below the vane: coming down from the middle, the first
    sustained stretch where the mask is no wider than the quill itself (`quill_factor` times the
    narrow widths found in the bottom quarter of the mask, at most `max_width_frac` of the vane's
    typical width). The analysis uses the vane only, so everything below is cut on every feather.
    Measuring the quill, not a fixed share of the vane, keeps narrow lower vanes and gradually
    tapering feathers whole and still finds a quill that is wide for its feather."""
    ys, _ = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0 + 1
    width = mask[y0:y1 + 1].sum(axis=1).astype(float)
    width = np.convolve(np.pad(width, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    vane = float(np.median(width[h // 4:h // 2]))
    base = width[int(0.75 * h):]
    quill = float(np.percentile(base[base > 0], 20)) if (base > 0).any() else 0.0
    limit = min(float(cfg.get("quill_factor", 1.6)) * quill, float(cfg.get("max_width_frac", 0.35)) * vane)
    run = max(3, int(float(cfg.get("run_frac", 0.015)) * h))
    narrow = width <= limit
    for i in range(h // 2, h - run):
        if narrow[i:i + run].all():
            return int(y0 + i)
    return None


def clean(img: np.ndarray, mask: np.ndarray, cfg: dict) -> tuple[np.ndarray, int | None, list[str]]:
    """(cleaned mask, row the mask was cut at or None, notes). The cleaned mask never grows."""
    ys, _ = np.nonzero(mask)
    if not len(ys):
        return mask, None, []
    y0, y1 = ys.min(), ys.max()
    base_top = y1 - float(cfg.get("base_frac", 0.35)) * (y1 - y0 + 1)
    out, cut, notes = mask.copy(), None, []
    h = y1 - y0 + 1
    vane = float(np.median(mask[y0 + h // 4:y0 + h // 2].sum(axis=1)))
    if cfg.get("texture", True):
        for p in textureless_patches(img, mask, cfg):
            py, _ = np.nonzero(p)
            row = _above_glue(mask, int(py.min()), y1 - y0 + 1, vane, cfg) if py.mean() >= base_top else None
            if row is not None:                             # glue at the quill base: cut it off with all below
                cut = row if cut is None else min(cut, row)
                notes.append(f"glue/card at the base ({int(p.sum())} px)")
            else:                                           # card along the vane: remove just the patch
                out &= ~p
                notes.append(f"card/background patch removed ({int(p.sum())} px)")
    if cfg.get("neck", True):
        row = neck_row(mask, cfg)
        if row is not None:
            cut = row if cut is None else min(cut, row)
            notes.append("glue dot below the quill neck")
    cal = cfg.get("calamus") or {}
    if cal.get("enabled"):
        row = calamus_row(mask, cal)
        if row is not None and (cut is None or row < cut):
            cut = row
            notes.append("calamus cut (vane only)")
    vb = cfg.get("vane_base") or {}
    if vb.get("enabled", True):
        row = vane_base_row(img, mask, vb)
        if row is not None and (cut is None or row < cut):
            cut = row
            notes.append("cut to the full vane (base on card)")
    if cut is not None:
        out[cut:, :] = False
    if out.sum() < float(cfg.get("min_keep_frac", 0.6)) * mask.sum():     # never cut most of a feather away
        return mask, None, ["cleanup skipped: would remove too much"]
    return out, cut, notes
