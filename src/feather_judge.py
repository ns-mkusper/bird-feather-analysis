"""Per-feather segmentation score from the VLM.

For each kept feather the model sees two panels side by side: the original
slide around the feather with the segmentation outline in green, and the
extracted crop exactly as saved. The rubric (configs/segmentation.json ->
feather_judge.prompt) describes a good segment and the usual failures:
fragments, a missing tip or vane, merged feathers, background included.

The answer is parsed strictly; a failed or garbled answer is status
'model_error' with no score, never a low score.
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np
from PIL import Image

FLAGS = ("complete", "single_feather", "fragment", "background_included", "other_feather_included")
MISSING = {"none", "tip", "vane", "white_parts", "base", "other"}


def context_panel(slide_bgr: np.ndarray, mask: np.ndarray, crop_bgr: np.ndarray, cfg: dict) -> Image.Image:
    """LEFT: slide region around the feather with the mask outline in green;
    RIGHT: the saved crop on black. Both scaled to the same height."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = cfg.get("context_margin_frac", 0.15)
    mx, my = int((x1 - x0) * m) + 10, int((y1 - y0) * m) + 10
    x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    left = slide_bgr[y0:y1, x0:x1].copy()
    cnts, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, cnts, -1, (0, 255, 0), max(2, int(min(left.shape[:2]) / 250)))
    ph = int(cfg.get("panel_height", 900))

    def fit(img):
        s = ph / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * s)), ph), interpolation=cv2.INTER_AREA)

    left, right = fit(left), fit(crop_bgr if crop_bgr.size else np.zeros((10, 10, 3), np.uint8))
    gap = np.full((ph, 16, 3), 255, np.uint8)
    both = np.hstack([left, gap, right])
    return Image.fromarray(cv2.cvtColor(both, cv2.COLOR_BGR2RGB))


def parse(text: str) -> dict:
    """Model answer -> {'status': 'ok', 'score': 7, ...} or {'status': 'model_error', 'error': ...}."""
    if re.search(r"\\u[0-9a-fA-F]{4}", text or ""):
        return {"status": "model_error", "error": "escape sequence in answer"}
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"status": "model_error", "error": f"no JSON in answer: {(text or '')[:60]!r}"}
    try:
        d = json.loads(m.group(0))
        score = float(d["score"])
    except (ValueError, KeyError, TypeError):
        return {"status": "model_error", "error": f"unparseable answer: {(text or '')[:60]!r}"}
    if not 1 <= score <= 10:
        return {"status": "model_error", "error": f"score out of range: {score}"}
    out = {"status": "ok", "score": score}
    for k in FLAGS:
        v = d.get(k)
        out[k] = v if isinstance(v, bool) else None
    mp = str(d.get("missing_part", "")).strip().lower()
    out["missing_part"] = mp if mp in MISSING else None
    notes = str(d.get("notes") or "")
    out["notes"] = notes if notes.isascii() else notes.encode("ascii", "ignore").decode()
    return out


def judge(client, slide_bgr, mask, crop_bgr, cfg: dict) -> tuple[dict, Image.Image]:
    """Ask the model about one feather. Returns (result, the panel image)."""
    panel = context_panel(slide_bgr, mask, crop_bgr, cfg)
    try:
        raw = client.ask(panel, cfg["prompt"])
    except Exception as exc:  # noqa: BLE001
        return {"status": "model_error", "error": f"{type(exc).__name__}: {exc}"}, panel
    res = parse(raw)
    res["raw"] = (raw or "")[:400]
    return res, panel


def needs_review(rec: dict, min_score: float = 7) -> str:
    """Why a feather should be looked at ('' if fine)."""
    why = []
    if rec.get("judge_status") == "model_error":
        why.append("judge failed")
    elif rec.get("judge_score") is not None and rec["judge_score"] < min_score:
        why.append(f"score {rec['judge_score']:g}")
    for k, bad in (("judge_fragment", True), ("judge_single_feather", False), ("judge_complete", False),
                   ("judge_other_feather_included", True)):
        if rec.get(k) is bad:
            why.append(k.replace("judge_", "") + ("" if bad else " = no"))
    if any("possible fragment" in n for n in rec.get("postprocess", "").split("; ")):
        why.append("small/narrow mask")
    return "; ".join(why)
