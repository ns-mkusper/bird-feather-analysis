"""File names of a slide's segmentation outputs (no heavy imports).

    slide 'CE01168 2008-im2346.jpg' ->
        overlay  'CE01168 2008_BoundingBoxes.jpg'
        crops    'CE01168_2008_Feather_A.jpg', '..._Feather_B.jpg', ... (A = leftmost)

The ID and year come from the file name. Where one bird has several slides
from the same year (63 of the 2406 slides), the year alone would be shared, so
those slides use the full date read from the label instead:
'BB17692 19980612_BoundingBoxes.jpg', 'BB17692_19980612_Feather_A.jpg'. Slides
of the same bird from the same DAY (or whose sibling's date is unknown) also
get the image number: 'A0514_20000521-im1163_Feather_A.jpg'. If the slide's own
date could not be read: 'BB17692_1998-im0513_Feather_A.jpg'. File names that cannot be parsed keep
their stem. The same period (year / YYYYMMDD / year-imNNNN) goes into the
feather IDs, so crop names and IDs line up.

Older runs used '<slide stem>_<id>_<date>_Feather_<n>.jpg' and
'<slide stem>_BoundingBoxes.jpg'; the lookups below find both.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

SEQ_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_FILENAME_STRICT = re.compile(r"^([A-Z]{1,2})\s*(\d{3,5})\s+((?:19|20)\d{2})")
_FILENAME_LENIENT = re.compile(r"^([A-Z]{1,2})(\d{3,6})((?:19|20)\d{2})$")


def parse_filename(name: str) -> dict:
    """'A0012 1996-im1058.jpg' -> {'id': 'A0012', 'year': '1996', 'parse': 'standard'}.
    Also handles the odd names in the collection ('-BD31906 1999-…',
    'AX868161996-img…', 'BX40 368 2004-…') as parse='lenient'."""
    stem = Path(name).stem
    m = _FILENAME_STRICT.match(stem)
    if m:
        return {"id": m.group(1) + m.group(2), "year": m.group(3), "parse": "standard"}
    head = re.split(r"-im", stem.lstrip("-"), maxsplit=1)[0]
    m = _FILENAME_LENIENT.match(re.sub(r"\s+", "", head))
    if m:
        return {"id": m.group(1) + m.group(2), "year": m.group(3), "parse": "lenient"}
    return {"id": None, "year": None, "parse": "unparsed"}


def seq_letter(index: int) -> str:
    """0 -> 'A' (leftmost feather), 1 -> 'B', ..."""
    return SEQ_LETTERS[index] if index < len(SEQ_LETTERS) else f"Z{index - len(SEQ_LETTERS) + 1}"


def _image_tag(name: str) -> str:
    """'A0012 1996-im1058.jpg' -> 'im1058' (the part after the first '-')."""
    stem = Path(name).stem.lstrip("-")
    return stem.split("-", 1)[1] if "-" in stem else stem


def _key(name: str) -> tuple[str, str] | None:
    fn = parse_filename(name)
    return (fn["id"], fn["year"]) if fn["id"] else None


def shares_id_year(name: str, siblings: list[str]) -> bool:
    """Another slide in `siblings` has the same file-name ID and year."""
    key = _key(name)
    return key is not None and any(_key(s) == key for s in siblings if Path(s).name != Path(name).name)


_DATE = re.compile(r"^((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


def compact_date(date: str | None) -> str | None:
    """'1998-06-12' -> '19980612'; None for anything that is not a full date."""
    m = _DATE.match(date or "")
    return "".join(m.groups()) if m else None


def same_year_siblings(name: str, siblings: list[str]) -> list[str]:
    """Other slides with the same file-name ID and year."""
    key = _key(name)
    return [] if key is None else sorted(s for s in siblings
                                         if Path(s).name != Path(name).name and _key(s) == key)


def output_names(name: str, siblings: list[str] | None = None, date: str | None = None,
                 sibling_dates: dict[str, str | None] | None = None) -> dict:
    """{'base': 'CE01168 2008', 'crop_prefix': 'CE01168_2008', 'period': '2008'}.
    `siblings`: the other slide file names (default: the slide's own folder).
    `date`: the label date (YYYY-MM-DD), used when the bird has other slides
    from the same year. `sibling_dates`: label dates of those slides; a shared
    or unknown one adds the image number."""
    name_only = Path(name).name
    if siblings is None:
        folder = os.path.dirname(str(name)) or "."
        siblings = os.listdir(folder) if os.path.isdir(folder) else []
    key = _key(name_only)
    if key is None:
        stem = Path(name_only).stem
        return {"base": stem, "crop_prefix": stem.replace(" ", "_"), "period": None}
    bird, year = key
    period = year
    if shares_id_year(name_only, siblings):
        full = compact_date(date)
        if full is None:
            period = f"{year}-{_image_tag(name_only)}"
        else:
            sib = same_year_siblings(name_only, siblings)
            known = sibling_dates or {}
            same_day = any(compact_date(known.get(Path(s).name)) in (None, full) for s in sib)
            period = f"{full}-{_image_tag(name_only)}" if same_day else full
    return {"base": f"{bird} {period}", "crop_prefix": f"{bird}_{period}", "period": period}


def period_of_crop(crop: str) -> str | None:
    """'BB17692_19980612_Feather_A.jpg' -> '19980612'; None for older names."""
    prefix = Path(crop).name.rsplit("_Feather_", 1)[0]
    if " " in prefix or "_" not in prefix:
        return None
    return prefix.split("_", 1)[1]


def overlay_name(names: dict) -> str:
    return f"{names['base']}_BoundingBoxes.jpg"


def crop_name(names: dict, index: int) -> str:
    """index 0 = leftmost feather -> '<prefix>_Feather_A.jpg'."""
    return f"{names['crop_prefix']}_Feather_{seq_letter(index)}.jpg"


def _seq_key(path: Path) -> tuple[int, str]:
    tail = path.stem.rsplit("_Feather_", 1)[-1]
    return (int(tail), "") if tail.isdigit() else (len(tail), tail)


def _candidates(name: str, date: str | None = None) -> list[dict]:
    """Names this slide may have used: with the label date or the image number
    (when it shares its ID and year with another slide), plain, unparsed."""
    name_only = Path(name).name
    key = _key(name_only)
    if key is None:
        return [output_names(name_only, [])]
    bird, year = key
    tag = _image_tag(name_only)
    out = []
    if compact_date(date):
        full = compact_date(date)
        out.append({"base": f"{bird} {full}-{tag}", "crop_prefix": f"{bird}_{full}-{tag}"})
        out.append({"base": f"{bird} {full}", "crop_prefix": f"{bird}_{full}"})
    out += [{"base": f"{bird} {year}-{tag}", "crop_prefix": f"{bird}_{year}-{tag}"},
            {"base": f"{bird} {year}", "crop_prefix": f"{bird}_{year}"}]
    return out


def _recorded_crops(out_dir: Path, name: str) -> list[Path]:
    """Crop names the pipeline recorded for this slide (labels/<stem>.json)."""
    rec = out_dir / "labels" / f"{Path(name).stem}.json"
    try:
        import json

        feathers = json.loads(rec.read_text()).get("feathers") or []
    except (OSError, ValueError):
        return []
    paths = [out_dir / f["crop"] for f in feathers if f.get("crop")]
    return paths if paths and all(p.exists() for p in paths) else []


def find_crops(out_dir: str | Path, name: str, date: str | None = None) -> list[Path]:
    """The slide's feather crops in `out_dir`, left to right (new or old
    names). Slides that share bird and year with another slide are named by
    their label date: pass `date`, or the pipeline's labels/<stem>.json is used."""
    out_dir = Path(out_dir)
    recorded = _recorded_crops(out_dir, name)
    if recorded:
        return recorded
    for cand in _candidates(name, date):
        found = sorted(out_dir.glob(f"{glob.escape(cand['crop_prefix'])}_Feather_*.jpg"), key=_seq_key)
        if found:
            return found
    stem = Path(name).stem
    old = [p for p in out_dir.glob(f"{glob.escape(stem)}_*_Feather_*.jpg")]
    return sorted(old, key=_seq_key)


def find_overlay(out_dir: str | Path, name: str, date: str | None = None) -> Path | None:
    out_dir = Path(out_dir)
    recorded = _recorded_crops(out_dir, name)
    if recorded:
        p = out_dir / f"{recorded[0].name.rsplit('_Feather_', 1)[0].replace('_', ' ', 1)}_BoundingBoxes.jpg"
        if p.exists():
            return p
    for cand in _candidates(name, date):
        p = out_dir / overlay_name(cand)
        if p.exists():
            return p
    old = out_dir / f"{Path(name).stem}_BoundingBoxes.jpg"
    return old if old.exists() else None


def belongs_to(file_name: str, name: str, date: str | None = None) -> bool:
    """Is `file_name` an overlay or crop of slide `name` (new or old names)?
    Date-named outputs of same-year slides need `date` to be attributed."""
    fname = Path(file_name).name
    if fname.startswith(f"{Path(name).stem}_") and ("_Feather_" in fname or "_BoundingBoxes" in fname):
        return True
    for cand in _candidates(name, date):
        if fname == overlay_name(cand) or (fname.startswith(f"{cand['crop_prefix']}_Feather_")
                                           and fname.endswith(".jpg")):
            return True
    return False


def find_crop(out_dir: str | Path, bird_id: str, year: str, seq: str = "A") -> Path | None:
    """One crop by file-name ID, year (or full YYYYMMDD) and letter; with only
    a year and several slides that year, the earliest date is returned."""
    out_dir = Path(out_dir)
    exact = out_dir / f"{bird_id}_{year}_Feather_{seq}.jpg"
    if exact.exists():
        return exact
    dated = sorted(out_dir.glob(f"{glob.escape(bird_id)}_{year}[0-9][0-9][0-9][0-9]*_Feather_{seq}.jpg"))
    tagged = sorted(out_dir.glob(f"{glob.escape(bird_id)}_{year}-*_Feather_{seq}.jpg"))
    return (dated or tagged or [None])[0]
