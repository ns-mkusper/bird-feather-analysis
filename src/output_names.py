"""File names of a slide's segmentation outputs (no heavy imports).

    slide 'CE01168 2008-im2346.jpg' ->
        overlay  'CE01168 2008_BoundingBoxes.jpg'
        crops    'CE01168_2008_Feather_A.jpg', '..._Feather_B.jpg', ... (A = leftmost)

The ID and year come from the file name (stable, never depend on a model
reading). Where one bird has several slides from the same year (63 of the 2406
slides), the short name would be shared, so those slides keep their image
number: 'BB17692 1998-im0513_BoundingBoxes.jpg', 'BB17692_1998-im0513_Feather_A.jpg'.
File names that cannot be parsed keep their stem.

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


def output_names(name: str, siblings: list[str] | None = None) -> dict:
    """{'base': 'CE01168 2008', 'crop_prefix': 'CE01168_2008'} for a slide.
    `siblings`: the other slide file names (default: the slide's own folder)."""
    name_only = Path(name).name
    if siblings is None:
        folder = os.path.dirname(str(name)) or "."
        siblings = os.listdir(folder) if os.path.isdir(folder) else []
    key = _key(name_only)
    if key is None:
        stem = Path(name_only).stem
        return {"base": stem, "crop_prefix": stem.replace(" ", "_")}
    bird, year = key
    if shares_id_year(name_only, siblings):
        tag = _image_tag(name_only)
        return {"base": f"{bird} {year}-{tag}", "crop_prefix": f"{bird}_{year}-{tag}"}
    return {"base": f"{bird} {year}", "crop_prefix": f"{bird}_{year}"}


def overlay_name(names: dict) -> str:
    return f"{names['base']}_BoundingBoxes.jpg"


def crop_name(names: dict, index: int) -> str:
    """index 0 = leftmost feather -> '<prefix>_Feather_A.jpg'."""
    return f"{names['crop_prefix']}_Feather_{seq_letter(index)}.jpg"


def _seq_key(path: Path) -> tuple[int, str]:
    tail = path.stem.rsplit("_Feather_", 1)[-1]
    return (int(tail), "") if tail.isdigit() else (len(tail), tail)


def _candidates(name: str) -> list[dict]:
    """Names this slide may have used: with image number (when it shares its
    ID and year with another slide), without, and the unparsed fallback."""
    name_only = Path(name).name
    key = _key(name_only)
    if key is None:
        return [output_names(name_only, [])]
    bird, year = key
    tag = _image_tag(name_only)
    return [{"base": f"{bird} {year}-{tag}", "crop_prefix": f"{bird}_{year}-{tag}"},
            {"base": f"{bird} {year}", "crop_prefix": f"{bird}_{year}"}]


def find_crops(out_dir: str | Path, name: str) -> list[Path]:
    """The slide's feather crops in `out_dir`, left to right (new or old names)."""
    out_dir = Path(out_dir)
    for cand in _candidates(name):
        found = sorted(out_dir.glob(f"{glob.escape(cand['crop_prefix'])}_Feather_*.jpg"), key=_seq_key)
        if found:
            return found
    stem = Path(name).stem
    old = [p for p in out_dir.glob(f"{glob.escape(stem)}_*_Feather_*.jpg")]
    return sorted(old, key=_seq_key)


def find_overlay(out_dir: str | Path, name: str) -> Path | None:
    out_dir = Path(out_dir)
    for cand in _candidates(name):
        p = out_dir / overlay_name(cand)
        if p.exists():
            return p
    old = out_dir / f"{Path(name).stem}_BoundingBoxes.jpg"
    return old if old.exists() else None


def belongs_to(file_name: str, name: str) -> bool:
    """Is `file_name` an overlay or crop of slide `name` (new or old names)?"""
    fname = Path(file_name).name
    if fname.startswith(f"{Path(name).stem}_") and ("_Feather_" in fname or "_BoundingBoxes" in fname):
        return True
    for cand in _candidates(name):
        if fname == overlay_name(cand) or (fname.startswith(f"{cand['crop_prefix']}_Feather_")
                                           and fname.endswith(".jpg")):
            return True
    return False


def find_crop(out_dir: str | Path, bird_id: str, year: str, seq: str = "A") -> Path | None:
    """One crop by file-name ID, year and letter (first match if the bird has
    several slides that year)."""
    out_dir = Path(out_dir)
    exact = out_dir / f"{bird_id}_{year}_Feather_{seq}.jpg"
    if exact.exists():
        return exact
    tagged = sorted(out_dir.glob(f"{glob.escape(bird_id)}_{year}-*_Feather_{seq}.jpg"))
    return tagged[0] if tagged else None
