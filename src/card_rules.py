"""What the feather cards mean, from the collector's notes on the feather samples (2026-10-30),
applied to one slide's label reading after the model has answered
(src/label_extract.py). No model calls: everything here works on the stored answers, so a
finished run can be re-derived (scripts/relabel_run.py).

The card (configs/label_fields.json "card"):
  age box      top-left square: the SECURED age from the ringing lists. Its colour is the age
               code: grey/yellowish 1K, lilac 2K, dark blue 3K, light blue 4K, green 5K,
               yellow 6K, red 7K, bright red 8K, pink other. 'RC' (and 'ÅF') in the box means
               the bird was ringed by someone else and its age is held by the ringing centre
               (Ringmärkningscentralen): no secured age on the card, not a misreading.
  field age    right end of the species row (e.g. '4K+'): the collector's ageing from the
               plumage when caught. A different measure; it never replaces the box age.
  species      the collector's field call; hybrid codes (HFXSF, SFXHF) are suspected hybrids, so
               one bird's slides from different years may carry different codes.
  sex          block letters M/F are certain, small letters m/f are 'supposed'.
  ring number  followed by H/R (right tarsus) or V/L (left tarsus); the file names leave the
               letter out.

Rules applied here (each notes what it did in the field's note):
  species      a code one or two letters from exactly one known code is that code (HAFIU ->
               HAFLU); a cut-off code that starts exactly one known code is that code (HAF ->
               HAFLU); the bird ID read in the species crop means the species row is covered
               (not_found).
  age          RC / ÅF in the box -> status 'code_on_card' (information, not review); the box
               colour (label_extract.check_age_box) gives the age where the digits are not
               readable and the colour does not contradict what is visible -> 'from_colour'.
               A purple box with no digit in sight is left alone: RC / ÅF boxes are purple too.
  sex          the case of the model's answer -> extra['sex_certainty'] 'certain' / 'supposed'.
  bird ID, date  a partly visible or uncertain reading whose visible ID (without the tarsus
               letter) or year agrees with the file name -> 'confirmed_by_filename'; the
               feather IDs only need the bird ID and the year. Switch: card.filename_confirms_partial_ids.
               A handwritten YY-MM-DD date read as 20DD-MM-YY is turned round when that makes
               it agree with the file-name year.
"""
from __future__ import annotations

import re

READABLE = "readable"
CONFIRMED = "confirmed_by_filename"
CODE_ON_CARD = "code_on_card"
FROM_COLOUR = "from_colour"
TRUSTED_ID_STATUSES = (READABLE, CONFIRMED)
TARSUS_LETTERS = "HRVL"


def edit_distance(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def nearest_code(value: str, codes: list[str], max_edits: int = 2) -> str | None:
    """The one known code `value` is a misreading of: at most max_edits away, and every other
    code at least two edits further (HAFLU and SVFLU are two apart, so a one-letter slip is
    never ambiguous)."""
    if not value or value in codes:
        return value if value in codes else None
    ranked = sorted((edit_distance(value, c), c) for c in codes)
    best_d, best = ranked[0]
    if best_d > max_edits or (len(ranked) > 1 and ranked[1][0] < best_d + 2):
        return None
    return best


def _bare_id(value: str | None) -> str:
    v = re.sub(r"\s+", "", value or "").upper()
    return v[:-1] if v and v[-1] in TARSUS_LETTERS and re.search(r"\d[A-Z]$", v) else v


def _chosen_raw_value(f) -> str | None:
    """The value as the model wrote it (before upper-casing) in the attempt the field kept."""
    a = next((x for x in f.attempts if x.crop == f.source and not x.error), None)
    if a is None or not a.raw:
        return None
    m = re.search(r'"value"\s*:\s*"([^"]*)"', a.raw)
    return m.group(1).strip() if m else None


def _note(f, text: str) -> None:
    f.note = f"{f.note}; {text}" if f.note else text


def _remember(r, name: str) -> None:
    """Keeps a field's reading as the model left it (extra['card_original']) before a rule changes
    it, so the rules can be run again on a saved result after they change."""
    f = r.fields[name]
    r.extra.setdefault("card_original", {}).setdefault(
        name, {"value": f.value, "status": f.status, "note": f.note, "source": f.source})


def _restore(r) -> None:
    for name, o in (r.extra.get("card_original") or {}).items():
        f = r.fields.get(name)
        if f is not None:
            f.value, f.status, f.note, f.source = o["value"], o["status"], o["note"], o["source"]


def species_rules(r, card: dict) -> None:
    f = r.fields.get("species")
    if f is None or not f.value or f.status in ("model_error", "inferred"):
        return
    codes = card.get("species_codes") or []
    v = f.value
    bird = r.fields.get("bird_id")
    if _bare_id(v) in {_bare_id(bird.value if bird else None), r.filename.get("id")} - {"", None}:
        _remember(r, "species")
        _note(f, f"the species crop showed the bird ID {v!r}: species row covered")
        f.value, f.status = None, "not_found"
        return
    if v in codes:
        return
    code = nearest_code(v, codes)
    if code is None and len(v) >= 3:
        starts = [c for c in codes if c.startswith(v)]
        code = starts[0] if len(starts) == 1 else None
    if code is None:
        return
    _remember(r, "species")
    _note(f, f"read {v!r}, the known code {code}")
    f.value = code
    if f.status == "needs_review" and "format" in (f.note or ""):
        f.status = READABLE


def age_rules(r, card: dict) -> None:
    f = r.fields.get("age")
    if f is None:
        return
    codes = {k.upper(): v for k, v in (card.get("age_box_codes") or {}).items()}
    raw = (f.image_value or f.value or "").strip().upper()
    if f.status != READABLE and raw in codes:
        _remember(r, "age")
        f.status = CODE_ON_CARD
        f.value = raw
        if codes[raw] not in (f.note or ""):
            _note(f, codes[raw])
        return
    box = r.extra.get("age_box") or {}
    if f.status in (READABLE, CODE_ON_CARD, FROM_COLOUR, "model_error"):
        return
    state = (box.get("state") or "").lower()
    if state == "code" and (box.get("text") or "").upper() in codes:
        _remember(r, "age")
        f.value, f.status = box["text"].upper(), CODE_ON_CARD
        _note(f, codes[f.value])
        return
    if state == "empty":
        _remember(r, "age")
        f.status = "not_on_card"
        _note(f, "the age box is empty: no secured age on the card")
        return
    age = colour_age(box.get("colour"), card)
    seen = box.get("text") or f.value
    # the RC / ÅF boxes are purple too: a purple box says 2K only with a digit in sight
    shared = box.get("colour", "").lower() in (card.get("age_colours_shared_with_codes") or [])
    if age and _consistent(age, seen) and not (shared and not re.sub(r"\D", "", seen or "")):
        _remember(r, "age")
        f.value, f.status = age, FROM_COLOUR
        _note(f, f"age from the box colour ({box.get('colour')}); text seen: {box.get('text') or f.value!r}")


def colour_age(colour: str | None, card: dict) -> str | None:
    if not colour:
        return None
    c = colour.strip().lower()
    return (card.get("age_colours") or {}).get(c)


def _consistent(age: str, seen: str | None) -> bool:
    """A colour age agrees with whatever digits are visible (none visible agrees)."""
    digits = re.sub(r"\D", "", seen or "")
    return not digits or digits == re.sub(r"\D", "", age)


def sex_rules(r, card: dict) -> None:
    f = r.fields.get("sex")
    if f is None or f.status in ("model_error", "not_found"):
        return
    raw = _chosen_raw_value(f) if f.attempts else None
    if raw in ("M", "F"):
        r.extra["sex_certainty"] = "certain"
    elif raw in ("m", "f"):
        r.extra["sex_certainty"] = "supposed"
        if "supposed" not in (f.note or ""):
            _note(f, "small letter on the card: supposed sex")


def id_rules(r, card: dict) -> None:
    if not card.get("filename_confirms_partial_ids", True) or r.filename.get("parse") != "standard":
        return
    b = r.fields.get("bird_id")
    if b is not None and b.status in ("partly_visible", "needs_review") and b.value \
            and _bare_id(b.value) == r.filename.get("id"):
        _remember(r, "bird_id")
        b.status = CONFIRMED
        _note(b, f"agrees with the file name {r.filename['id']}")
    d = r.fields.get("date")
    if d is None or not d.value or not r.filename.get("year"):
        return
    fy = r.filename["year"]
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", d.value)
    if d.status == "conflict" and m and r.extra.get("handwritten") == "yes":
        # handwritten '03-06-11' (YY-MM-DD) read as 2011-06-03
        yy, mm, dd = m.group(1)[2:], m.group(2), m.group(3)
        if dd == fy[2:]:
            turned = f"{fy}-{mm}-{yy}"
            _remember(r, "date")
            _note(d, f"handwritten YY-MM-DD: read {d.value}, turned round to {turned} (file-name year {fy})")
            d.value, d.status = turned, CONFIRMED
            return
    if d.status in ("partly_visible", "needs_review") and m and m.group(1) == fy:
        _remember(r, "date")
        d.status = CONFIRMED
        _note(d, f"year agrees with the file name ({fy}); month and day as read")


def apply(r, cfg: dict) -> None:
    """All per-slide card rules; safe to run again on a result they were already applied to."""
    card = cfg.get("card") or {}
    if not card:
        return
    _restore(r)
    species_rules(r, card)
    age_rules(r, card)
    sex_rules(r, card)
    id_rules(r, card)
