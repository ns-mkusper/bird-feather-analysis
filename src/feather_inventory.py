"""Missed feathers: which feathers on the slide may have no segmentation at all.

The model sees the slide with every segmented feather outlined and numbered
1..n from left to right, counts all feathers, and lists each one without a
numbered outline by its neighbours ('between': [left number, right number],
null at an end) and its horizontal centre.

Counting overlapping feathers is unreliable, so the answer only ever becomes
review flags: letters A, B, C ..., crop names and feather IDs come from the
segmentation alone and never change because of it. Each possibly missed
feather is one extra row in the feather table (segment_status
'possible_missed', no letter, crop or feather ID; needs_review names its
neighbours by letter), and a model total that does not add up (status
'inconsistent') adds one 'count_unclear' row. A failed request is status
model_error and adds nothing.
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np
from PIL import Image

PALETTE = [(0, 200, 0), (255, 140, 0), (0, 160, 255), (220, 0, 220), (0, 220, 220), (255, 60, 60),
           (140, 90, 255), (255, 210, 0)]


def numbered_overlay(slide_bgr: np.ndarray, masks: list[np.ndarray], max_side: int = 1400) -> Image.Image:
    """Slide with each mask outlined in its own colour and numbered 1..n (left to right)."""
    img = slide_bgr.copy()
    t = max(3, int(max(img.shape[:2]) / 400))
    for i, m in enumerate(masks):
        col = PALETTE[i % len(PALETTE)]
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, col, t)
        ys, xs = np.nonzero(m)
        cx, cy = int(xs.mean()), int(ys.min()) + int(0.08 * (ys.max() - ys.min()))
        label, scale = str(i + 1), max(1.0, img.shape[1] / 900)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
        cv2.rectangle(img, (cx - tw // 2 - 8, cy - th - 8), (cx + tw // 2 + 8, cy + 8), (255, 255, 255), -1)
        cv2.rectangle(img, (cx - tw // 2 - 8, cy - th - 8), (cx + tw // 2 + 8, cy + 8), col, t)
        cv2.putText(img, label, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t)
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def parse(text: str, n_segmented: int) -> dict:
    """{'status': 'ok', 'total': 5, 'missed': [{'between': [2, 3], 'x_center': 0.55, 'partly_hidden': True}]}"""
    if re.search(r"\\u[0-9a-fA-F]{4}", text or ""):
        return {"status": "model_error", "error": "escape sequence in answer"}
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else None
        total = int(d["total_feathers"])
        missed_raw = d.get("missed") or []
        assert isinstance(missed_raw, list)
    except (ValueError, KeyError, TypeError, AssertionError, AttributeError):
        return {"status": "model_error", "error": f"unparseable answer: {(text or '')[:80]!r}"}
    missed = []
    for e in missed_raw:
        try:
            b = e.get("between") or [None, None]
            left = int(b[0]) if b[0] is not None else None
            right = int(b[1]) if b[1] is not None else None
            x = float(e.get("x_center")) if e.get("x_center") is not None else None
        except (ValueError, TypeError, AttributeError, IndexError):
            return {"status": "model_error", "error": f"bad missed entry: {e!r}"[:200]}
        missed.append({"between": [left, right], "x_center": x, "partly_hidden": bool(e.get("partly_hidden"))})
    notes = str(d.get("notes") or "")
    return {"status": "ok", "total": total, "missed": missed,
            "notes": notes if notes.isascii() else notes.encode("ascii", "ignore").decode()}


def merge(seg_centers: list[float], inv: dict, width: int) -> tuple[list[dict], str, str]:
    """Order segmented feathers (by mask centre x) and missed ones into one
    left-to-right sequence. Returns (sequence, inventory_status, note); each
    sequence item is {'kind': 'segmented', 'index': i} or {'kind': 'missed', 'x': px, ...}."""
    order = sorted(range(len(seg_centers)), key=lambda i: seg_centers[i])
    rank = {i: r + 1 for r, i in enumerate(order)}          # 1-based numbers as drawn
    items = [{"kind": "segmented", "index": i, "x": seg_centers[i]} for i in order]
    if inv.get("status") != "ok":
        return items, "model_error", inv.get("error", "")
    notes, conflict = [], False
    by_rank = {rank[i]: seg_centers[i] for i in order}
    index_of = {rank[i]: i for i in order}
    for e in inv["missed"]:
        left, right = e["between"]
        lo = by_rank.get(left, 0.0) if left else 0.0
        hi = by_rank.get(right, float(width)) if right else float(width)
        if (left and left not in by_rank) or (right and right not in by_rank) or lo >= hi:
            conflict = True
            notes.append(f"impossible neighbours {e['between']}")
        x = e["x_center"] * width if e["x_center"] is not None else None
        if x is None or not lo <= x <= hi:
            if x is not None:
                notes.append(f"x_center {e['x_center']:.2f} outside neighbours {e['between']}; placed between them")
            x = (lo + hi) / 2
        items.append({"kind": "missed", "x": x, "between": e["between"], "partly_hidden": e["partly_hidden"],
                      "between_index": [index_of.get(left), index_of.get(right)]})
    items.sort(key=lambda it: it["x"])
    if inv["total"] != len(seg_centers) + len(inv["missed"]):
        conflict = True
        notes.append(f"model counted {inv['total']} feathers but {len(seg_centers)} segmented + "
                     f"{len(inv['missed'])} missed")
    status = "inconsistent" if conflict else ("missed" if inv["missed"] else "complete")
    return items, status, "; ".join(notes + ([inv["notes"]] if inv.get("notes") else []))


def check(client, slide_bgr: np.ndarray, masks: list[np.ndarray], cfg: dict) -> tuple[list[dict], dict, Image.Image]:
    """Runs the inventory for one slide. `masks` in any order; returns the
    merged left-to-right sequence, a summary dict, and the numbered overlay."""
    centers = [float(np.nonzero(m)[1].mean()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: centers[i])
    overlay = numbered_overlay(slide_bgr, [masks[i] for i in order], cfg.get("max_side", 1400))
    try:
        raw = client.ask(overlay, cfg["prompt"])
        inv = parse(raw, len(masks))
    except Exception as exc:  # noqa: BLE001
        raw, inv = "", {"status": "model_error", "error": f"{type(exc).__name__}: {exc}"}
    items, status, note = merge(centers, inv, slide_bgr.shape[1])
    summary = {"inventory_status": status, "inventory_total": inv.get("total"),
               "feathers_segmented": len(masks), "feathers_possibly_missed": sum(it["kind"] == "missed" for it in items),
               "inventory_note": note, "inventory_raw": (raw or "")[:400]}
    return items, summary, overlay


def review_rows(items: list[dict], summary: dict, letters: dict[int, str]) -> list[dict]:
    """Feather-table rows that only flag the slide for review. `letters`: the
    segmented feathers' letters by mask index (the order passed to check)."""
    rows = []
    missed = [it for it in items if it["kind"] == "missed"]
    for k, it in enumerate(missed, start=1):
        left, right = (letters.get(i) if i is not None else None for i in it["between_index"])
        where = (f"between {left} and {right}" if left and right else f"right of {left}" if left
                 else f"left of {right}" if right else "on the slide")
        rows.append({"seq": f"missed_{k}", "crop": "", "segment_status": "possible_missed",
                     "missed_x": round(it["x"]), "missed_between": [left, right],
                     "missed_partly_hidden": it.get("partly_hidden"),
                     "needs_review": f"possible missed feather {where} (model count; check the inventory image)"})
    if summary.get("inventory_status") == "inconsistent":
        rows.append({"seq": "count_check", "crop": "", "segment_status": "count_unclear",
                     "needs_review": f"feather count unclear: {summary.get('inventory_note') or 'model total differs'}"})
    return rows
