"""Slide-level facts added after the labels are read (src/label_extract.py), from all slides of a run:

  inferred sex / species   a slide whose own reading is not usable takes the value from the same bird's
                           other slides when all their confident readings agree (status "inferred",
                           source "same_bird"). Age is never inferred: it changes between years.
  photos_same_year         how many slides of the collection have this slide's file-name bird ID and
                           year (the convention the file names follow); multiple_photos_same_year > 1.
  handwritten              whether any label text is handwritten: from the model check during label
                           reading (label_fields.json "handwriting"), or a hand-check CSV, which wins.
  secured age              the age box (src/card_rules.py) goes up by exactly one per calendar year: a
                           slide whose box age is not readable takes it from the same bird's exact box
                           ages in other years when they all agree (status "inferred"); readable box
                           ages that do not follow the years are a problem.
  label_problems           what a person has to look at on this slide's label, in words.
  label_info               everything else worth knowing that needs no one to look: values inferred
                           from the same bird, fields covered or not printed, codes on the card (RC),
                           IDs confirmed by the file name, a species that changes between years (a
                           field call, often a suspected hybrid), supposed sex, handwriting notes.

Bird ID for grouping is the label's value when it is readable (or partly visible and agreeing with
the file name, card_rules.CONFIRMED), never a file-name value alone: an inference must not rest on
an uncertain ID.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from src import card_rules
from src import output_names as on

INFER_FIELDS = ("sex", "species")
UNINFORMATIVE = {None, "", "U"}          # sex 'U' (unknown) says nothing about the bird
PROBLEM_WORDS = {"partly_visible": "partly visible", "needs_review": "needs review", "conflict": "conflicts with file name",
                 "filename_fallback": "from file name only", "model_error": "not read (model error)",
                 "partial": "partial"}
# statuses that are facts about the card, not doubts about the reading
INFO_WORDS = {"not_found": "not found (covered or not printed)", "inferred": "inferred from the same bird",
              card_rules.CONFIRMED: "partly visible, agrees with the file name",
              card_rules.CODE_ON_CARD: "not on the card (code in the box)", card_rules.FROM_COLOUR: "from the box colour",
              "not_on_card": "not on the card (empty box)"}
TRUSTED_ID = (card_rules.READABLE, card_rules.CONFIRMED)
EXACT_AGE = re.compile(r"^(\d{1,2})K$")
CORRECTION = re.compile(r"cross(ed)?[- ]out|correct|overwrit|written over|struck|replaced by hand|changed by hand", re.I)


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
        if f is not None and f.status in TRUSTED_ID and f.value:
            groups[f.value].append(r)
    notes: dict[str, list[str]] = defaultdict(list)
    info: dict[str, list[str]] = defaultdict(list)
    for bird, group in groups.items():
        secured_age(group, notes)
        for name in INFER_FIELDS:
            donors = [r for r in group if (f := r.fields.get(name)) is not None and f.status == readable
                      and f.value not in UNINFORMATIVE]
            values = sorted({r.fields[name].value for r in donors})
            if len(values) > 1:
                for r in group:
                    if name == "species":
                        # the collector's field call; hybrid codes mark suspected hybrids, so a bird's code
                        # can change between years (src/card_rules.py)
                        info[r.file_id].append(f"species differs between this bird's slides ({'/'.join(values)})")
                    elif all(d.extra.get("sex_certainty") == "certain" for d in donors):
                        notes[r.file_id].append(f"{name}: this bird's slides disagree ({'/'.join(values)})")
                    elif any(d.extra.get("sex_certainty") == "supposed" for d in donors):
                        info[r.file_id].append(f"sex differs between this bird's slides ({'/'.join(values)}), "
                                               "one of them supposed (small letter)")
                    else:
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
        problems, facts = [], []
        for name, f in r.fields.items():
            # not_found is still reported (label_info): the model cannot tell a covered field from one
            # that is not printed, and in full-20261026 every sampled species not_found was under the
            # board (BZ10382 2005-im2035)
            if f.status in PROBLEM_WORDS:
                problems.append(f"{name} {PROBLEM_WORDS[f.status]}")
            elif f.status in INFO_WORDS:
                facts.append(f"{name} {INFO_WORDS[f.status]}")
            elif f.status == readable and "partly visible (accepted)" in (f.note or ""):
                facts.append(f"{name} partly visible (accepted)")
            elif f.status == readable and "the known code" in (f.note or ""):
                facts.append(f"{name} {f.note.split('; ')[-1]}")
        if r.extra.get("sex_certainty") == "supposed":
            facts.append("sex supposed (small letter on the card)")
        problems += notes.get(r.file_id, [])
        facts += info.get(r.file_id, [])
        if r.extra["label_note"]:
            # a value corrected, crossed out or written over by hand needs a look; other notes are information
            (problems if CORRECTION.search(r.extra["label_note"]) else facts).append(r.extra["label_note"])
        if r.extra["handwritten"] == "yes":
            written = "handwritten" + (f": {r.extra['handwritten_fields']}" if r.extra["handwritten_fields"] else "")
            # handwriting in the bird ID or date needs a look (the feather IDs rest on them); elsewhere it is a note
            fields = r.extra["handwritten_fields"] or ""
            (problems if re.search(r"bird_id|date", fields) else facts).append(written)
        r.extra["label_problems"] = "; ".join(problems)
        r.extra["label_info"] = "; ".join(facts)


def secured_age(group: list, notes: dict) -> None:
    """The age box of one bird's slides: exact ages ('3K', not '3K+') must go up by one per
    calendar year; a slide without a readable box age takes the one its other years agree on.
    The year is the file name's (a misread label year would break the sequence for no reason);
    when most slides agree, only the ones that break the sequence are flagged."""
    def year(r):
        y = r.filename.get("year") or ""
        if y.isdigit():
            return int(y)
        d = r.fields.get("date")
        y = (d.value or "")[:4] if d is not None and d.value else ""
        return int(y) if y.isdigit() else None

    exact = []
    for r in group:
        f = r.fields.get("age")
        m = EXACT_AGE.match(f.value or "") if f is not None and f.status == card_rules.READABLE else None
        if m and year(r) is not None:
            exact.append((r, year(r) - int(m.group(1)) + 1))      # the bird's first calendar year (1K)
    votes = Counter(b for _, b in exact)
    if not votes:
        return
    (b, top), *rest = votes.most_common()
    if rest and rest[0][1] == top:                 # no majority: every slide with a box age needs a look
        flagged, b = [x for x, _ in exact], None
    else:
        flagged = [x for x, xb in exact if xb != b]
    if flagged:
        ages = ", ".join(f"{x.fields['age'].value} in {year(x)}" for x, _ in sorted(exact, key=lambda e: year(e[0])))
        for r in flagged:
            notes[r.file_id].append(f"age: the box age does not follow this bird's other years ({ages})")
    if b is None:
        return
    for r in group:
        f = r.fields.get("age")
        if f is None or f.status in (card_rules.READABLE, card_rules.CODE_ON_CARD, "model_error") or year(r) is None:
            continue
        age = year(r) - b + 1
        if age < 1 or not card_rules._consistent(f"{age}K", f.value):
            continue
        src = ", ".join(x.file_id for x, _ in exact)
        f.note = f"inferred from the same bird's box ages ({src}); this slide: {f.value!r} [{f.status}]"
        f.value, f.status, f.source, f.confidence = f"{age}K", "inferred", "same_bird", None
