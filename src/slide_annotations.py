"""Slide-level facts added after the labels are read (src/label_extract.py), from all slides of a run:

  inferred sex / species   a slide whose own reading is not usable takes the value from the same bird's
                           other slides when all their confident readings agree (status "inferred",
                           source "same_bird"). Age is never inferred: it changes between years.
  photos_same_year         how many slides of the collection have this slide's file-name bird ID and
                           year (the convention the file names follow); multiple_photos_same_year > 1.
  handwritten              whether any label text is handwritten: from the model check during label
                           reading (label_fields.json "handwriting"), or a hand-check CSV, which wins.
  label_problems           every reason to be careful with this slide's label, in words.

Bird ID for grouping is the label's value when it is readable, never a partly visible or
file-name value: an inference must not rest on an uncertain ID.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from src import output_names as on

INFER_FIELDS = ("sex", "species")
UNINFORMATIVE = {None, "", "U"}          # sex 'U' (unknown) says nothing about the bird
PROBLEM_WORDS = {"partly_visible": "partly visible", "needs_review": "needs review", "conflict": "conflicts with file name",
                 "filename_fallback": "from file name only", "model_error": "not read (model error)",
                 "partial": "partial", "not_found": "not on the label"}


def load_handwriting(path: str | Path | None) -> dict[str, dict]:
    """file_id -> {'handwritten': 'yes'|'no', 'fields': '...', 'note': '...', 'source': '...'} from a CSV
    (columns file_id, handwritten, fields, note, source); `note` is any other remark on the label."""
    if not path or not Path(path).exists():
        return {}
    return {r["file_id"]: r for r in csv.DictReader(open(path, newline=""))}


def annotate(results: list, all_names: list[str] | None = None, handwriting: dict | None = None,
             readable: str = "readable") -> None:
    """Adds inferred fields and result.extra (label_problems, photos_same_year,
    multiple_photos_same_year, handwritten, handwritten_fields) to every result."""
    # same bird: confident label IDs only
    groups: dict[str, list] = defaultdict(list)
    for r in results:
        f = r.fields.get("bird_id")
        if f is not None and f.status == readable and f.value:
            groups[f.value].append(r)
    notes: dict[str, list[str]] = defaultdict(list)
    for bird, group in groups.items():
        for name in INFER_FIELDS:
            donors = [r for r in group if (f := r.fields.get(name)) is not None and f.status == readable
                      and f.value not in UNINFORMATIVE]
            values = sorted({r.fields[name].value for r in donors})
            if len(values) > 1:
                for r in group:
                    notes[r.file_id].append(f"{name}: this bird's slides disagree ({'/'.join(values)})")
                continue
            if not values:
                continue
            v = values[0]
            for r in group:
                f = r.fields.get(name)
                if f is None:
                    continue
                if f.status == readable:
                    if f.value not in UNINFORMATIVE and f.value != v:
                        notes[r.file_id].append(f"{name} {f.value} differs from this bird's other slides ({v})")
                    continue
                src = ", ".join(d.file_id for d in donors if d is not r)
                if not src:
                    continue
                f.note = f"inferred from the same bird ({src}); this slide: {f.value!r} [{f.status}]"
                f.value, f.status, f.source, f.confidence = v, "inferred", "same_bird", None
    # photos of the same bird in the same year, over the whole collection (file-name convention)
    counts: dict = defaultdict(int)
    for n in all_names or []:
        k = on._key(n)
        if k:
            counts[k] += 1
    hw = handwriting or {}
    for r in results:
        k = on._key(r.file_id)
        n = counts.get(k) if (k and all_names) else None
        r.extra["photos_same_year"] = n
        r.extra["multiple_photos_same_year"] = (n > 1) if n is not None else None
        h = hw.get(r.file_id)
        if r.extra.get("handwriting_source") == "model":     # keep the model's answer next to a hand check
            r.extra["handwritten_model"] = r.extra.get("handwritten", "")
        if h is not None:      # a hand check overrides the model's handwriting check
            r.extra["handwritten"] = h.get("handwritten") or ""
            r.extra["handwritten_fields"] = h.get("fields") or ""
            r.extra["label_note"] = h.get("note") or ""
            r.extra["handwriting_source"] = h.get("source") or "hand check"
        for k in ("handwritten", "handwritten_fields", "label_note"):
            r.extra.setdefault(k, "")
        problems = []
        for name, f in r.fields.items():
            if f.status in PROBLEM_WORDS and f.status != "not_found":
                problems.append(f"{name} {PROBLEM_WORDS[f.status]}")
            elif f.status == "inferred":
                problems.append(f"{name} inferred from the same bird")
            elif f.status == readable and "partly visible (accepted)" in (f.note or ""):
                problems.append(f"{name} partly visible (accepted)")
        problems += notes.get(r.file_id, [])
        if r.extra["label_note"]:
            problems.append(r.extra["label_note"])
        if r.extra["handwritten"] == "yes":
            problems.append("handwritten" + (f": {r.extra['handwritten_fields']}" if r.extra["handwritten_fields"] else ""))
        r.extra["label_problems"] = "; ".join(problems)
