"""Per-feather segmentation checks: measured from the mask, and judged by the VLM.

Rules for a good segment (the same for any feather on any slide):
  1. complete outline: no part of this feather lies outside the mask on any
     side (left, right, top, bottom of the image)
  2. no holes or gaps: the mask is one solid piece with nothing of the
     feather missing inside it
  3. exactly one feather, not a fragment, and nothing else inside the mask
     (glue at the quill is asked about on its own: the catch-all question missed it)
  4. the whole feather is visible: a feather partly under the board, card, tape
     or another feather cannot be measured whole (flagged, though its outline may
     be right)

Rule 2 is measured directly on the mask (holes = background enclosed by the
mask; parts = separate pieces of the mask). The VLM checks rule 1 per side,
and rules 2-3 by eye, looking at two panels: LEFT the slide around the
feather with the mask outline, RIGHT exactly the pixels inside the mask. Which
mask is judged: the saved crop's (bottom trimmed, feather_judge.judge_mask
"saved") or the untrimmed segmentation ("full").

Decisions (needs_review, segment_ok, the retry in full_run_sharded) use the
checks themselves. The 1-10 `score` is computed from them with the fixed
deductions in configs/segmentation.json (feather_judge.score), only for
sorting and summaries; the model's own number is kept, unused, as
`model_score`. A failed or garbled answer is status 'model_error' (no score).
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np
from PIL import Image

SIDES = ("left", "right", "top", "bottom")
BOOLS = ("holes_or_gaps", "single_feather", "fragment", "other_material_inside", "other_feather_inside")
# asked by newer prompts only; parsed when present
OPTIONAL_BOOLS = ("glue_inside",)
HIDDEN = ("none", "part", "most")   # how much of the feather is hidden under board, card, tape, another feather
# used when the config has no feather_judge.score (same values as the shipped config)
DEFAULT_SCORE = {"start": 10, "min": 1, "deduct": {"side_cut": 2, "holes_or_gaps": 3, "fragment": 4,
                                                   "not_single_feather": 4, "other_feather_inside": 3,
                                                   "other_material_inside": 2, "glue_inside": 2,
                                                   "hidden_part": 3, "hidden_most": 8}}


# ---------------------------------------------------------------- measured on the mask

def mask_metrics(mask: np.ndarray, min_hole_frac: float = 0.002, min_part_frac: float = 0.01) -> dict:
    """Holes and separate pieces of one feather's mask (rule 2), measured.
    hole_frac: enclosed background / filled feather area; mask_parts: pieces
    larger than min_part_frac of the mask."""
    m = mask.astype(np.uint8)
    area = int(m.sum())
    if area == 0:
        return {"hole_frac": None, "holes": 0, "mask_parts": 0}
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    parts = int(sum(stats[i, cv2.CC_STAT_AREA] >= min_part_frac * area for i in range(1, n)))
    # holes: background regions not connected to the image border
    inv = (1 - m).astype(np.uint8)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = m.shape
    border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])))
    holes = [stats2[i, cv2.CC_STAT_AREA] for i in range(1, n2) if i not in border]
    big = [a for a in holes if a >= min_hole_frac * area]
    return {"hole_frac": round(float(sum(big)) / (area + sum(big)), 4), "holes": len(big), "mask_parts": parts}


def straight_outline(mask: np.ndarray, eps: float = 2.0, min_frac: float = 0.15,
                     across_deg: float = 25.0) -> dict:
    """Ruler-straight stretches of the mask outline, as shares of the feather's length: where the
    edge of the board, a card or another feather lies on it (feather outlines are curved and ragged).
    Measured at half resolution; the saved crop's horizontal bottom trim line is ignored.
    across: total of runs at `across_deg` or more to the feather's long axis (board edges cutting
    across it); along_longest: the longest run along the axis (a covered side; natural straight
    outer vanes and quill sides give shorter ones); total and longest of all runs.
    along_runs: the runs along the axis as [x0, y0, x1, y1, share] in the mask's own pixels, for
    the slide check in cover_edge."""
    m = cv2.resize(mask.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = {"total": 0.0, "longest": 0.0, "across": 0.0, "along_longest": 0.0}
    runs: list[list[float]] = []
    if not cnts:
        return {**out, "along_runs": runs}
    c = max(cnts, key=cv2.contourArea)
    ys, xs = c[:, 0, 1], c[:, 0, 0]
    length = max(int(ys.max() - ys.min()), int(xs.max() - xs.min()), 1)
    py, px = np.nonzero(m)
    axis = np.linalg.eigh(np.cov(np.vstack([px, py]).astype(float)))[1][:, -1] if len(px) > 2 else np.array([0.0, 1.0])
    poly = cv2.approxPolyDP(c, eps, True)[:, 0]
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = float(np.hypot(*(b - a)))
        if n < min_frac * length:
            continue
        if abs(int(a[1]) - int(b[1])) <= 1 and min(a[1], b[1]) >= ys.max() - 2:
            continue
        share = n / length
        ang = float(np.degrees(np.arccos(min(1.0, abs(float((b - a) / n @ axis))))))
        out["total"] += share
        out["longest"] = max(out["longest"], share)
        if ang >= across_deg:
            out["across"] += share
        else:
            out["along_longest"] = max(out["along_longest"], share)
            runs.append([2.0 * float(a[0]), 2.0 * float(a[1]), 2.0 * float(b[0]), 2.0 * float(b[1]), round(share, 3)])
    return {**{k: round(v, 3) for k, v in out.items()}, "along_runs": runs}


def _across_steps(gray: np.ndarray, a: np.ndarray, b: np.ndarray, t0: float, t1: float, d: float,
                  step: float = 4.0) -> np.ndarray:
    """Grey-level steps across the line a->b (pixels `d` to either side), every `step` px from
    t0 to t1 along it (in px from a; negative = before a). Points off the slide are skipped."""
    u = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([-u[1], u[0]])
    h, w = gray.shape
    out = []
    for t in np.arange(t0, t1, step):
        q1, q2 = a + u * t + nrm * d, a + u * t - nrm * d
        if all(0 <= q[0] < w and 0 <= q[1] < h for q in (q1, q2)):
            out.append(float(gray[int(q1[1]), int(q1[0])]) - float(gray[int(q2[1]), int(q2[0])]))
    return np.array(out)


def cover_edge(gray: np.ndarray, run: list[float], length_px: float, h: dict) -> float:
    """How surely a straight run along the feather is the edge of something lying on it, from the
    slide (blurred grey): the edge of a board or card is straight beyond the feather too, while a
    feather's own straight vane edge ends with the feather. Extends the run by `edge_ext_frac` of
    the feather's length past each end and returns the larger share of points there with a grey
    step of `edge_step` or more across the line (`edge_offset_px` to either side). A run on the
    slide's border (the feather runs off the scan) is 1.0."""
    a, b = np.array(run[:2], float), np.array(run[2:4], float)
    hh, ww = gray.shape
    m = 4
    if (min(a[0], b[0]) >= ww - m or max(a[0], b[0]) < m or min(a[1], b[1]) >= hh - m or max(a[1], b[1]) < m):
        return 1.0
    n = float(np.linalg.norm(b - a))
    ext, gap = float(h.get("edge_ext_frac", 0.12)) * length_px, 0.02 * length_px
    d, step_min = float(h.get("edge_offset_px", 6)), float(h.get("edge_step", 40))
    best = 0.0
    for t0, t1 in ((-ext, -gap), (n + gap, n + ext)):
        steps = _across_steps(gray, a, b, t0, t1, d)
        if len(steps) >= 5:
            best = max(best, float(np.mean(np.abs(steps) >= step_min)))
    return round(best, 3)


def hidden_level(straight: dict, rel_area: float | None, cfg: dict, slide_bgr: np.ndarray | None = None) -> str:
    """'none' | 'part' | 'most' from the straight outline (straight_outline) and the feather's area
    relative to the slide's median feather (a mostly covered feather shows only a small piece).
    Hidden: straight runs across the feather add up to `across_frac`, or one run along it reaches
    `along_frac`. With the slide given and `edge_check` on, a run along it counts only when it is
    the edge of something lying on the feather (cover_edge reaches `edge_frac`): short feathers'
    straight outer vanes reach along_frac too (A0028 im1063 D, A0201 im1032 D in full-20261026)."""
    h = cfg.get("hidden") or {}
    across = straight["across"] >= float(h.get("across_frac", 0.2))
    along_frac = float(h.get("along_frac", 0.33))
    along = straight["along_longest"] >= along_frac
    if along and not across and slide_bgr is not None and h.get("edge_check", False):
        gray = slide_bgr if slide_bgr.ndim == 2 else cv2.cvtColor(slide_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        long_runs = [r for r in straight.get("along_runs", []) if r[4] >= along_frac]
        length = max((float(np.hypot(r[2] - r[0], r[3] - r[1])) / r[4] for r in long_runs), default=0.0)
        edges = [cover_edge(gray, r, length, h) for r in long_runs]
        straight["cover_edge"] = max(edges, default=0.0)
        along = straight["cover_edge"] >= float(h.get("edge_frac", 0.25))
    if not (across or along):
        return "none"
    if rel_area is not None and rel_area < float(h.get("most_max_rel_area", 0.6)):
        return "most"
    return "part"


# ---------------------------------------------------------------- what the model sees

def context_panel(slide_bgr: np.ndarray, mask: np.ndarray, cfg: dict) -> Image.Image:
    """LEFT: slide around the feather with the mask outline (and filled holes
    outlined too); RIGHT: the slide pixels inside the mask on black. Same height."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(max(x1 - x0, y1 - y0) * cfg.get("context_margin_frac", 0.15)) + 10
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    left = slide_bgr[y0:y1, x0:x1].copy()
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    cnts, _ = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)   # outer edges and holes
    cv2.drawContours(left, cnts, -1, (0, 255, 0), max(2, int(min(left.shape[:2]) / 300)))
    right = np.where(sub[:, :, None] > 0, slide_bgr[y0:y1, x0:x1], 0).astype(np.uint8)
    if cfg.get("enhance_contrast"):
        # local contrast (CLAHE on lightness), so pale barbs on white card and dark barbs on the
        # board stand out from the background; the same mapping on both panels
        lab = cv2.cvtColor(slide_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=float(cfg.get("contrast_clip", 3.0)), tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        left = enh.copy()
        cv2.drawContours(left, cnts, -1, (0, 255, 0), max(2, int(min(left.shape[:2]) / 300)))
        right = np.where(sub[:, :, None] > 0, enh, 0).astype(np.uint8)
    ph = int(cfg.get("panel_height", 900))

    def fit(img):
        s = ph / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * s)), ph), interpolation=cv2.INTER_AREA)

    both = np.hstack([fit(left), np.full((ph, 16, 3), 255, np.uint8), fit(right)])
    return Image.fromarray(cv2.cvtColor(both, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------- the answer

def _json(text: str) -> dict:
    if re.search(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}", text or ""):
        raise ValueError("escape sequence in answer")
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"no JSON in answer: {(text or '')[:60]!r}")
    return json.loads(m.group(0))


def derive_score(r: dict, score_cfg: dict | None = None) -> int:
    """start minus a fixed deduction per failed check (per cut side), at least min."""
    sc = {**DEFAULT_SCORE, **(score_cfg or {})}
    d = {**DEFAULT_SCORE["deduct"], **sc.get("deduct", {})}
    s = sc["start"]
    s -= d["side_cut"] * len(r.get("sides_cut") or [])
    s -= d["holes_or_gaps"] * bool(r.get("holes_or_gaps"))
    s -= d["fragment"] * bool(r.get("fragment"))
    s -= d["not_single_feather"] * (r.get("single_feather") is False)
    s -= d["other_feather_inside"] * bool(r.get("other_feather_inside"))
    s -= d["other_material_inside"] * bool(r.get("other_material_inside"))
    s -= d["glue_inside"] * bool(r.get("glue_inside"))
    s -= d["hidden_part"] * (r.get("hidden") == "part") + d["hidden_most"] * (r.get("hidden") == "most")
    return max(sc["min"], s)


def parse(text: str, score_cfg: dict | None = None) -> dict:
    """Model answer -> checks, or {'status': 'model_error', 'error': ...}."""
    try:
        d = _json(text)
        outline = d["outline"]
        sides = {s: str(outline[s]).strip().lower() for s in SIDES}
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": "model_error", "error": f"{type(exc).__name__}: {exc}"[:200]}
    if any(v not in ("complete", "cut") for v in sides.values()):
        return {"status": "model_error", "error": f"outline values must be complete|cut: {sides}"}
    out = {"status": "ok", **{f"outline_{s}": v for s, v in sides.items()},
           "sides_cut": [s for s in SIDES if sides[s] == "cut"]}
    for k in BOOLS:
        v = d.get(k)
        if not isinstance(v, bool):
            return {"status": "model_error", "error": f"{k} must be true|false, got {v!r}"}
        out[k] = v
    for k in OPTIONAL_BOOLS:
        if k in d:
            if not isinstance(d[k], bool):
                return {"status": "model_error", "error": f"{k} must be true|false, got {d[k]!r}"}
            out[k] = d[k]
    if "hidden" in d:
        if str(d["hidden"]).strip().lower() not in HIDDEN:
            return {"status": "model_error", "error": f"hidden must be none|part|most, got {d['hidden']!r}"}
        out["hidden"] = str(d["hidden"]).strip().lower()
    ms = d.get("model_score", d.get("score"))
    out["model_score"] = float(ms) if isinstance(ms, (int, float)) and 1 <= ms <= 10 else None
    notes = str(d.get("notes") or "")
    out["notes"] = notes if notes.isascii() else notes.encode("ascii", "ignore").decode()
    _finish(out, score_cfg)
    return out


def _finish(res: dict, score_cfg: dict | None) -> None:
    res["score"] = derive_score(res, score_cfg)
    res["segment_ok"] = not failed_checks(res, prefix="")


_PASS = {"holes_or_gaps": False, "single_feather": True, "fragment": False, "other_feather_inside": False,
         "other_material_inside": False, "glue_inside": False, "hidden": "none"}


def _model_failures(res: dict) -> set[str]:
    """The checks the model's answer fails (not the mask measurements)."""
    out = {f"outline_{s}" for s in SIDES if res.get(f"outline_{s}") == "cut"}
    measured = {"hidden"} if "straight_outline" in res else set()   # not the model's answer
    return out | {k for k, ok in _PASS.items() if k not in measured and res.get(k) is not None and res.get(k) != ok}


def _second_look(client, panel, cfg: dict, res: dict) -> None:
    """Asks again on the same panel at a smaller size when the first answer
    fails a check. A check failed only once is set back to passing and listed
    in `uncertain_checks` (kept for review, not counted as a failure); a check
    failed both times stays failed. If the second request fails, the first
    answer stands: nothing is cleared without a second answer."""
    first = _model_failures(res)
    res["uncertain_checks"] = []
    if not first:
        return
    scale = float((cfg.get("second_look") or {}).get("scale", 0.8))
    small = panel.resize((max(1, int(panel.width * scale)), max(1, int(panel.height * scale))), Image.LANCZOS)
    try:
        raw2 = client.ask(small, cfg["prompt"], max_tokens=cfg.get("max_tokens", 220))
    except Exception as exc:  # noqa: BLE001
        res["second_look"] = f"failed: {type(exc).__name__}"
        return
    res2 = parse(raw2, cfg.get("score"))
    res["second_look"] = (raw2 or "")[:300]
    if res2.get("status") != "ok":
        return
    uncertain = sorted(first - _model_failures(res2))
    for c in uncertain:
        res[c] = "complete" if c.startswith("outline_") else _PASS[c]
    res["sides_cut"] = [s for s in SIDES if res.get(f"outline_{s}") == "cut"]
    res["uncertain_checks"] = uncertain


def judge(client, slide_bgr, mask, crop_bgr, cfg: dict, rel_area: float | None = None,
          raw_mask: np.ndarray | None = None) -> tuple[dict, Image.Image]:
    """Measures and asks about one feather. Returns (result, the panel image). For the measured
    'hidden' check: `raw_mask`, the segmentation before glue/card clean-up (whose cuts add straight
    edges), and `rel_area`, its area relative to the slide's median feather."""
    metrics = mask_metrics(mask, cfg.get("min_hole_frac", 0.002))
    if (cfg.get("hidden") or {}).get("enabled"):
        h = cfg["hidden"]
        st = straight_outline(raw_mask if raw_mask is not None else mask, float(h.get("eps", 2.0)),
                              float(h.get("min_frac", 0.15)), float(h.get("across_deg", 25.0)))
        metrics["hidden"] = hidden_level(st, rel_area, cfg, slide_bgr)
        metrics.update({"straight_outline": st["total"], "straight_longest": st["longest"],
                        "straight_across": st["across"], "straight_along": st["along_longest"]})
        if "cover_edge" in st:
            metrics["straight_cover_edge"] = st["cover_edge"]
    panel = context_panel(slide_bgr, mask, cfg)
    try:
        raw = client.ask(panel, cfg["prompt"], max_tokens=cfg.get("max_tokens", 220))
    except Exception as exc:  # noqa: BLE001
        return {"status": "model_error", "error": f"{type(exc).__name__}: {exc}", **metrics}, panel
    res = parse(raw, cfg.get("score"))
    res.update(metrics)       # measured 'hidden' replaces the model's answer when enabled
    if res["status"] == "ok" and (cfg.get("second_look") or {}).get("enabled", False):
        _second_look(client, panel, cfg, res)
        _finish(res, cfg.get("score"))
    fail_frac = float(cfg.get("fail_hole_frac", 0.02))
    if res["status"] == "ok" and ((metrics["hole_frac"] or 0) >= fail_frac or metrics["mask_parts"] > 1):
        # a mask in pieces, or measured holes of fail_hole_frac or more, count as rule-2
        # failures even if the model missed them; smaller measured holes are mostly natural
        # vane splits (background showing through) and are only recorded
        res["holes_or_gaps"] = True
    if res["status"] == "ok":
        _finish(res, cfg.get("score"))
    res["raw"] = (raw or "")[:400]
    return res, panel


def demote_unsupported(rec: dict, cfg: dict, prefix: str = "judge_") -> None:
    """Model answers that hand verdicts showed to be noise unless something measured backs them
    (feather_judge.needs_support): 'holes or gaps' without a measured hole (fail_hole_frac) or a
    split mask, 'fragment' on a piece the clean-up did not mark small or narrow, and 'not a single
    feather' without another feather inside, a fragment or a split mask. They move
    to the uncertain checks (listed, not failing) and the score is recomputed."""
    if rec.get(prefix + "status") != "ok":
        return
    need = cfg.get("needs_support", [])
    moved = []
    if "holes_or_gaps" in need and rec.get(prefix + "holes_or_gaps") \
            and (rec.get(prefix + "hole_frac") or 0) < float(cfg.get("fail_hole_frac", 0.02)) \
            and (rec.get(prefix + "mask_parts") or 1) <= 1:
        rec[prefix + "holes_or_gaps"] = False
        moved.append("holes_or_gaps")
    if "fragment" in need and rec.get(prefix + "fragment") and "possible fragment" not in str(rec.get("postprocess") or ""):
        rec[prefix + "fragment"] = False
        moved.append("fragment")
    if "single_feather" in need and rec.get(prefix + "single_feather") is False \
            and not rec.get(prefix + "other_feather_inside") and not rec.get(prefix + "fragment") \
            and (rec.get(prefix + "mask_parts") or 1) <= 1 and "possible fragment" not in str(rec.get("postprocess") or ""):
        # 'not a single feather' with nothing else saying so (no other feather inside, not a
        # fragment, one piece): in full-20261026 all 24 such answers were one whole feather, often
        # with the note "ok" (A0596 im1167 A, AX86260 im0002 D)
        rec[prefix + "single_feather"] = True
        moved.append("single_feather")
    if moved:
        rec[prefix + "uncertain_checks"] = sorted(set(rec.get(prefix + "uncertain_checks") or []) | set(moved))
        bare = {k[len(prefix):]: v for k, v in rec.items() if k.startswith(prefix)}
        rec[prefix + "score"] = derive_score(bare, cfg.get("score"))
        rec[prefix + "segment_ok"] = not failed_checks(bare, prefix="")


def add_cut_sides(rec: dict, sides: list[str], cfg: dict, prefix: str = "judge_") -> None:
    """Mark sides as cut from a measurement (feather_repair.regrow_check) and recompute the score."""
    if not sides:
        return
    for side in sides:
        rec[f"{prefix}outline_{side}"] = "cut"
    rec[prefix + "sides_cut"] = [s for s in SIDES if rec.get(f"{prefix}outline_{s}") == "cut"]
    bare = {k[len(prefix):]: v for k, v in rec.items() if k.startswith(prefix)}
    rec[prefix + "score"] = derive_score(bare, cfg.get("score"))
    rec[prefix + "segment_ok"] = not failed_checks(bare, prefix="")


def side_board_share(slide_bgr: np.ndarray, mask: np.ndarray, side: str, cfg: dict) -> float:
    """Share of the slide just outside the `side` ('left'/'right') of the mask that is dark board:
    a band `band_px` [inner, outer] px from the outline, between the feather's top and bottom."""
    m = mask.astype(np.uint8)
    inner, outer = (int(v) for v in cfg.get("band_px", (8, 60)))
    ring = cv2.dilate(m, np.ones((2 * outer + 1,) * 2, np.uint8)).astype(bool) \
        & ~cv2.dilate(m, np.ones((2 * inner + 1,) * 2, np.uint8)).astype(bool)
    ys, xs = np.nonzero(m)
    if not len(xs):
        return 0.0
    xc = (xs.min() + xs.max()) / 2
    cols = np.arange(m.shape[1])[None, :]
    rows = np.arange(m.shape[0])[:, None]
    sel = ring & ((cols < xc) if side == "left" else (cols > xc)) & (rows >= ys.min()) & (rows <= ys.max())
    gray = cv2.GaussianBlur(cv2.cvtColor(slide_bgr, cv2.COLOR_BGR2GRAY), (9, 9), 0)[sel]
    return round(float(np.mean(gray < int(cfg.get("dark", 90)))), 3) if gray.size else 0.0


def demote_board_cuts(rec: dict, slide_bgr: np.ndarray, mask: np.ndarray, cfg: dict, prefix: str = "judge_") -> None:
    """A left/right 'cut' from the model alone (the measured regrow below its min_growth) on a side
    that lies on the black board (side_board_share >= min_board) moves to the uncertain checks:
    on the board the regrow measure sees a missing piece of vane, while the model reads loose
    single barbs or fringe beyond the line as a cut (A0517 im1165 D). On the white card the
    model's answer stands: pale vane there fools the measure (BB17692 im0513 D, hand-checked)."""
    b = cfg.get("board_cut") or {}
    g = cfg.get("regrow") or {}
    if not b.get("enabled") or rec.get(prefix + "status") != "ok":
        return
    moved = []
    for side in ("left", "right"):
        growth = rec.get(f"{prefix}regrow_{side}")
        if rec.get(f"{prefix}outline_{side}") != "cut" or growth is None or growth >= float(g.get("min_growth", 0.15)):
            continue
        share = side_board_share(slide_bgr, mask, side, b)
        rec[f"{prefix}board_{side}"] = share
        if share >= float(b.get("min_board", 0.9)):
            rec[f"{prefix}outline_{side}"] = "complete"
            moved.append(f"outline_{side}")
    if moved:
        rec[prefix + "sides_cut"] = [s for s in SIDES if rec.get(f"{prefix}outline_{s}") == "cut"]
        rec[prefix + "uncertain_checks"] = sorted(set(rec.get(prefix + "uncertain_checks") or []) | set(moved))
        bare = {k[len(prefix):]: v for k, v in rec.items() if k.startswith(prefix)}
        rec[prefix + "score"] = derive_score(bare, cfg.get("score"))
        rec[prefix + "segment_ok"] = not failed_checks(bare, prefix="")


def failed_checks(rec: dict, prefix: str = "judge_") -> list[str]:
    """The checks one feather failed (model answer plus mask measurements);
    keys are judge_* in a feather record, bare in a judge result (prefix='')."""
    g = lambda k: rec.get(prefix + k)  # noqa: E731
    why = []
    sides = g("sides_cut")
    if sides:
        why.append("outline cut: " + ",".join(sides if isinstance(sides, list) else [str(sides)]))
    if g("holes_or_gaps") or (g("mask_parts") or 0) > 1:
        why.append("holes or gaps")
    if g("fragment"):
        why.append("fragment")
    if g("single_feather") is False:
        why.append("not a single feather")
    if g("other_feather_inside"):
        why.append("other feather inside")
    if g("other_material_inside"):
        why.append("non-feather material inside")
    if g("glue_inside"):
        why.append("glue inside")
    if g("hidden") in ("part", "most"):
        why.append(f"{'partly' if g('hidden') == 'part' else 'mostly'} hidden")
    return why


def needs_review(rec: dict) -> str:
    """Why a feather should be looked at ('' if it passed every check). Rows
    that are only review flags (a possible missed feather) keep their reason."""
    if rec.get("segment_status", "segmented") != "segmented":
        return rec.get("needs_review") or str(rec.get("segment_status"))
    why = ["judge failed"] if rec.get("judge_status") == "model_error" else []
    why += failed_checks(rec)
    if "possible fragment" in str(rec.get("postprocess") or ""):
        why.append("small/narrow mask")
    return "; ".join(why)


def is_fragment(rec: dict, prefix: str = "judge_") -> bool:
    """A piece, not a feather: the mask clean-up marked it small or narrow, and
    the model (on two looks) sees a fragment or not a single feather. Such
    pieces get no letter or feather ID; they stay as review rows. Without a
    model answer the piece stays a feather (flagged)."""
    return ("possible fragment" in str(rec.get("postprocess") or "") and rec.get(prefix + "status") == "ok"
            and (rec.get(prefix + "fragment") is True or rec.get(prefix + "single_feather") is False))


def uncertain_review(rec: dict, prefix: str = "judge_") -> str:
    """Checks failed on one look only (second_look): listed for review after
    the confirmed failures, not counted in the score or the retry."""
    names = {"outline_left": "outline cut: left", "outline_right": "outline cut: right",
             "outline_top": "outline cut: top", "outline_bottom": "outline cut: bottom",
             "holes_or_gaps": "holes or gaps", "single_feather": "not a single feather", "fragment": "fragment",
             "other_feather_inside": "other feather inside", "other_material_inside": "non-feather material inside",
             "glue_inside": "glue inside", "hidden": "hidden under something"}
    return "; ".join(names.get(c, c) for c in rec.get(prefix + "uncertain_checks") or [])


def slide_checks(records: list[dict]) -> dict:
    """Slide-level view of the per-feather checks and the missed-feather flags
    (what the retired slide judge used to guess from the overlay)."""
    seg = [r for r in records if r.get("segment_status", "segmented") == "segmented"]
    judged = [r for r in seg if r.get("judge_status") == "ok"]
    inv = next((r.get("inventory_status") for r in records if r.get("inventory_status")), None)
    scores = [r["judge_score"] for r in judged if isinstance(r.get("judge_score"), (int, float))]
    return {
        "feathers_failing": sum(bool(failed_checks(r)) for r in judged),
        "checks_failed": sum(len(failed_checks(r)) for r in judged),
        "judge_errors": sum(r.get("judge_status") == "model_error" for r in seg),
        "possible_missed": sum(r.get("segment_status") == "possible_missed" for r in records),
        "fragments": sum(r.get("segment_status") == "fragment" for r in records),
        "all_feathers_covered": {"complete": True, "missed": False}.get(inv),
        "merged_feathers": (any(r.get("judge_single_feather") is False or r.get("judge_other_feather_inside")
                                for r in judged) if judged else None),
        "other_material": any(bool(r.get("judge_other_material_inside")) for r in judged) if judged else None,
        "score_mean": round(sum(scores) / len(scores), 2) if scores else None,
    }
