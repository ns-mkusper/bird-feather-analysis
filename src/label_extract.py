"""Specimen label extraction: the second output table.

Reads the label fields (individual ID, species, age, sex, feather line, date)
from the ORIGINAL slide image, not the segmentation overlay. Each field gets
its own crop (configs/label_fields.json) and its own constrained VLM prompt, so
one hard field (the feather line, often under the black board) cannot spoil
the others.

Two identifiers are kept apart:
  file_id         the digital file name, e.g. 'A0012 1996-im1058.jpg'
  individual_id   the code printed on the specimen label, e.g. 'A0012H'
The file name is parsed separately (filename_id 'A0012', filename_year '1996')
and only ever used as a cross-check or a labelled fallback, never as if it had
been read from the image.

Fallback chain per field (stops at the first readable answer):
  1. primary      the field's configured region
  2. expanded     the same region with a margin, for slide-to-slide shift
  3. fallback_N   wider bands from the field's "fallback_regions"
  4. filename     ID/year from the file name, status filename_fallback
Fields marked "verify" are read a second time on a different crop; if the two
readings disagree the field goes to review instead of passing.

Every field ends with one status, so a missing value always says why:
  readable           read, passed the format check (and verification)
  partial            only part of the text is visible (feather line)
  needs_review       read, but fails the format check or the two readings disagree
  conflict           readable on the slide but disagrees with the file name
  filename_fallback  nothing readable on the slide; value comes from the file name
  not_found          every crop was inspected and the text is not there -> N/A
  model_error        the VLM request failed after retries; NOT evidence the
                     label is missing, exported blank (never N/A), rerun it

CLI:
    python -m src.label_extract --sample 10 --seed 1          # random sample
    python -m src.label_extract --match "A0012 1996" --match BB17692
    python -m src.label_extract --manifest configs/label_sample_30.txt --gold data/label_gold.csv
    python -m src.label_extract --all --workers 4 --out data/label_runs/full
Rerunning with the same --out resumes: finished slides are skipped and slides
with a model_error are retried.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env", override=False)  # FEATHER_VLM_ENDPOINTS etc.
DEFAULT_CONFIG = REPO / "configs" / "label_fields.json"
FIELDS_ORDER = ["individual_id", "species", "age", "sex", "feather", "date"]
DATE_PARTS = ["year", "month", "day"]
EXPORT_ORDER = ["individual_id", "species", "age", "sex", "feather", *DATE_PARTS]
EXTRACTOR_VERSION = "label_extract/2"

READABLE = "readable"
REVIEW_STATUSES = {"partial", "needs_review", "conflict", "filename_fallback", "model_error"}
NA = "N/A"


@dataclass
class Attempt:
    crop: str                    # primary / expanded / fallback_1 / verify
    region: list[float]
    value: str | None
    visible: str
    confidence: float | None
    raw: str
    seconds: float
    error: str | None = None
    evidence: str | None = None  # saved crop path


@dataclass
class FieldResult:
    value: str | None
    status: str
    confidence: float | None = None
    raw: str = ""
    seconds: float = 0.0
    source: str = ""             # which crop produced the value, or "filename"
    image_value: str | None = None
    filename_value: str | None = None
    note: str = ""
    evidence: str | None = None
    attempts: list[Attempt] = field(default_factory=list)


@dataclass
class LabelResult:
    image: str
    fields: dict[str, FieldResult] = field(default_factory=dict)
    filename: dict = field(default_factory=dict)
    seconds: float = 0.0
    extracted_at: str = ""

    @property
    def file_id(self) -> str:
        return Path(self.image).name

    @property
    def needs_review(self) -> bool:
        return any(f.status in REVIEW_STATUSES for f in self.fields.values())

    @property
    def has_model_error(self) -> bool:
        return any(f.status == "model_error" for f in self.fields.values())

    def date_parts(self) -> dict[str, tuple[str | None, str]]:
        """year/month/day with the date field's status; a part that is not a
        valid component is reported as needs_review on its own."""
        f = self.fields.get("date")
        if f is None:
            return {}
        parts = split_date(f.value)
        out = {}
        for k in DATE_PARTS:
            v = parts.get(k)
            st = f.status
            if f.value and v is None and st not in ("not_found", "model_error"):
                st = "needs_review"
            out[k] = (v, st)
        return out

    def row(self) -> dict:
        """One flat row: requested values, plus status/source/confidence columns.
        N/A only for not_found; a model_error stays blank so it is never read
        as 'the label has no such field'."""
        out = {"file_id": self.file_id, "image": self.file_id, "needs_review": self.needs_review,
               "filename_id": self.filename.get("id"), "filename_year": self.filename.get("year")}
        for name in FIELDS_ORDER:
            f = self.fields.get(name)
            if f is None:
                continue
            out[name] = export_value(f.value, f.status)
            out[f"{name}_status"] = f.status
            out[f"{name}_source"] = f.source
            out[f"{name}_confidence"] = f.confidence
            if f.note:
                out[f"{name}_note"] = f.note
        for k, (v, st) in self.date_parts().items():
            out[k] = export_value(v, st)
            out[f"{k}_status"] = st
        return out

    def to_json(self) -> dict:
        d = {"file_id": self.file_id, "image": self.image, "extracted_at": self.extracted_at,
             "extractor": EXTRACTOR_VERSION, "seconds": round(self.seconds, 2),
             "filename": self.filename, "needs_review": self.needs_review,
             "fields": {k: asdict(v) for k, v in self.fields.items()}}
        d["date_parts"] = {k: {"value": v, "status": s} for k, (v, s) in self.date_parts().items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "LabelResult":
        r = cls(d["image"], filename=d.get("filename", {}), seconds=d.get("seconds", 0),
                extracted_at=d.get("extracted_at", ""))
        for k, v in d.get("fields", {}).items():
            atts = [Attempt(**a) for a in v.pop("attempts", [])]
            r.fields[k] = FieldResult(**v, attempts=atts)
        return r


def export_value(value: str | None, status: str) -> str:
    if status == "not_found":
        return NA
    if status == "model_error" or value is None:
        return ""
    return value


def split_date(value: str | None) -> dict[str, str | None]:
    out = {k: None for k in DATE_PARTS}
    if not value:
        return out
    m = re.match(r"^((?:19|20)\d{2})(?:-(\d{2}))?(?:-(\d{2}))?", value)
    if not m:
        return out
    out["year"] = m.group(1)
    if m.group(2) and 1 <= int(m.group(2)) <= 12:
        out["month"] = m.group(2)
    if m.group(3) and 1 <= int(m.group(3)) <= 31:
        out["day"] = m.group(3)
    return out


def load_config(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


# ---------------------------------------------------------------- file name

_FILENAME_STRICT = re.compile(r"^([A-Z]{1,2})\s*(\d{3,5})\s+((?:19|20)\d{2})")
_FILENAME_LENIENT = re.compile(r"^([A-Z]{1,2})(\d{3,6})((?:19|20)\d{2})$")


def parse_filename(name: str) -> dict:
    """'A0012 1996-im1058.jpg' -> {'id': 'A0012', 'year': '1996', 'parse': 'standard'}.
    Also handles the odd names in the collection ('-BD31906 1999-…',
    'AX868161996-img…', 'BX40 368 2004-…'); those come back as parse='lenient'
    and are only used for cross-checks, never as a fallback value."""
    stem = Path(name).stem
    m = _FILENAME_STRICT.match(stem)
    if m:
        return {"id": m.group(1) + m.group(2), "year": m.group(3), "parse": "standard"}
    head = re.split(r"-im", stem.lstrip("-"), maxsplit=1)[0]
    m = _FILENAME_LENIENT.match(re.sub(r"\s+", "", head))
    if m:
        return {"id": m.group(1) + m.group(2), "year": m.group(3), "parse": "lenient"}
    return {"id": None, "year": None, "parse": "unparsed"}


# ---------------------------------------------------------------- crops

def _clip(region: list[float]) -> list[float]:
    l, t, r, b = (min(1.0, max(0.0, v)) for v in region)
    return [round(l, 4), round(t, 4), round(r, 4), round(b, 4)]


def crop_plan(spec: dict, cfg: dict) -> list[tuple[str, list[float]]]:
    """Ordered (name, region) list: primary, expanded, then configured fallbacks."""
    l, t, r, b = spec["region"]
    mx, my = cfg.get("expand_margin", [0.06, 0.04])
    plan = [("primary", _clip([l, t, r, b])),
            ("expanded", _clip([l - mx, t - my, r + mx, b + my]))]
    for i, reg in enumerate(spec.get("fallback_regions", []), 1):
        plan.append((f"fallback_{i}", _clip(reg)))
    return plan


def crop_field(image: Image.Image, region: list[float], max_side: int) -> Image.Image:
    w, h = image.size
    l, t, r, b = region
    crop = image.crop((int(l * w), int(t * h), int(r * w), int(b * h))).convert("RGB")
    # Upscale small crops (the model reads small print better), cap big ones.
    scale = max_side / max(crop.size)
    if scale > 1 or max(crop.size) > max_side:
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
    return crop


def _data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class VLMClient:
    """Round-robins requests over the configured endpoints; retries transient
    failures a bounded number of times before giving up."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        env = [e.strip() for e in os.getenv("FEATHER_VLM_ENDPOINTS", "").split(",") if e.strip()]
        self._endpoints = cycle(env or cfg["endpoints"])
        self._lock = Lock()
        self.retries = int(cfg.get("retries", 2))
        self.timeout = float(cfg.get("timeout", 180))

    def _ask_once(self, image: Image.Image, prompt: str) -> str:
        with self._lock:
            base = next(self._endpoints)
        body = {
            "model": self.cfg["model"],
            "max_tokens": self.cfg.get("max_tokens", 120),
            "temperature": self.cfg.get("temperature", 0),
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image)}},
                {"type": "text", "text": prompt},
            ]}],
        }
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def ask(self, image: Image.Image, prompt: str, timeout: float | None = None) -> str:
        if timeout:
            self.timeout = timeout
        last: Exception | None = None
        for i in range(self.retries + 1):
            try:
                return self._ask_once(image, prompt)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if i < self.retries:
                    time.sleep(3 * (i + 1))
        assert last is not None
        raise last


def normalize(value: str | None, how: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip()
    if v.lower() in ("null", "none", "n/a", ""):
        return None
    if how == "remove_spaces_upper":
        v = re.sub(r"\s+", "", v).upper()
    elif how == "date":
        m = re.search(r"(\d{4})\D?(\d{1,2})\D?(\d{1,2})", v)
        if m:
            v = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return v or None


def parse_answer(text: str) -> tuple[str | None, str, float | None]:
    """Returns (value, visible, confidence) from the model's JSON answer."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            conf = d.get("confidence")
            val = d.get("value")
            return (None if val is None else str(val), str(d.get("visible", "yes")).lower(),
                    float(conf) if isinstance(conf, (int, float)) else None)
        except (ValueError, TypeError):
            pass
    # Not JSON: treat the whole reply as the value, confidence unknown.
    t = text.strip().strip("`").strip()
    return (t or None, "yes" if t else "no", None)


def _classify(value: str | None, visible: str, spec: dict) -> str:
    if value is None or visible == "no":
        return "not_found"
    if spec.get("allow_partial") and (visible == "partly" or "..." in value):
        return "partial"
    pattern = spec.get("pattern")
    if pattern and not re.match(pattern, value):
        return "needs_review"
    return READABLE


def _attempt(client: VLMClient, image: Image.Image, crop_name: str, region: list[float],
             spec: dict, cfg: dict, evidence_dir: Path | None, field_name: str,
             prompt_extra: str = "") -> Attempt:
    crop = crop_field(image, region, cfg["vlm"].get("crop_max_side", 1024))
    ev = None
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        ev_path = evidence_dir / f"{field_name}_{crop_name}.jpg"
        small = crop.copy()
        small.thumbnail((900, 900))
        small.save(ev_path, "JPEG", quality=85)
        ev = str(ev_path)
    prompt = spec["prompt"] + prompt_extra + "\n\n" + cfg["answer_rules"]
    t = time.monotonic()
    try:
        raw = client.ask(crop, prompt)
    except Exception as exc:  # noqa: BLE001
        return Attempt(crop_name, region, None, "error", None, "", time.monotonic() - t,
                       error=f"{type(exc).__name__}: {exc}", evidence=ev)
    value, visible, conf = parse_answer(raw)
    return Attempt(crop_name, region, normalize(value, spec.get("normalize")), visible, conf,
                   raw, round(time.monotonic() - t, 2), evidence=ev)


FALLBACK_HINT = ("\nThe text may sit anywhere in this wider crop; look over the whole crop. "
                 "Ignore any other text that does not match this description.")


def read_field(client: VLMClient, image: Image.Image, name: str, spec: dict, cfg: dict,
               evidence_dir: Path | None = None) -> FieldResult:
    """Runs the crop plan until a readable (or partial) answer; then, for fields
    with "verify", reads once more on a different crop and compares."""
    attempts: list[Attempt] = []
    best: tuple[Attempt, str] | None = None
    plan = crop_plan(spec, cfg)
    for crop_name, region in plan:
        a = _attempt(client, image, crop_name, region, spec, cfg, evidence_dir, name,
                     FALLBACK_HINT if crop_name.startswith("fallback") else "")
        attempts.append(a)
        if a.error:
            continue  # technical failure: try the next crop, decide below
        st = _classify(a.value, a.visible, spec)
        if st in (READABLE, "partial"):
            best = (a, st)
            break
        if st == "needs_review" and best is None:
            best = (a, st)  # keep looking for a clean read, remember this one

    secs = round(sum(a.seconds for a in attempts), 2)
    if best is None:
        if all(a.error for a in attempts):
            return FieldResult(None, "model_error", raw=attempts[-1].error or "", seconds=secs,
                               note="all VLM requests failed", attempts=attempts)
        if any(a.error for a in attempts):
            # Some crops said "not there" but others never got an answer: not proof of absence.
            return FieldResult(None, "model_error", raw=attempts[-1].error or "", seconds=secs,
                               note="not seen in answered crops; other crops failed", attempts=attempts)
        return FieldResult(None, "not_found", seconds=secs, note=f"not visible in {len(attempts)} crops",
                           attempts=attempts)

    a, st = best
    res = FieldResult(a.value, st, a.confidence, a.raw, secs, source=a.crop, image_value=a.value,
                      evidence=a.evidence, attempts=attempts)

    if spec.get("verify") and st == READABLE:
        other = next(((n, r) for n, r in plan if n != a.crop), None)
        if other:
            v = _attempt(client, image, "verify_" + other[0], other[1], spec, cfg, evidence_dir, name)
            attempts.append(v)
            res.seconds = round(res.seconds + v.seconds, 2)
            if v.error:
                res.note = "verification request failed"
            elif v.value != a.value:
                res.status = "needs_review"
                res.note = f"readings disagree: {a.value!r} ({a.crop}) vs {v.value!r} ({other[0]})"
    return res


def apply_filename(result: LabelResult, cfg: dict) -> None:
    """Cross-check against the file name; fill from it only where the slide
    gave nothing and the name follows the standard convention."""
    fn = result.filename
    allow = cfg.get("filename_fallback", True) and fn.get("parse") == "standard"

    f = result.fields.get("individual_id")
    if f is not None and fn.get("id"):
        f.filename_value = fn["id"]
        if f.status == READABLE and not (f.value or "").startswith(fn["id"]):
            f.status = "conflict"
            f.note = f"slide {f.value!r} vs file name {fn['id']!r}"
        elif f.status == "not_found" and allow:
            f.value, f.status, f.source = fn["id"], "filename_fallback", "filename"
            f.confidence, f.note = None, "ID from file name; label suffix letter unknown"

    f = result.fields.get("date")
    if f is not None and fn.get("year"):
        f.filename_value = fn["year"]
        if f.status == READABLE and not (f.value or "").startswith(fn["year"]):
            f.status = "conflict"
            f.note = f"slide year {(f.value or '')[:4]!r} vs file name {fn['year']!r}"
        elif f.status == "not_found" and allow:
            f.value, f.status, f.source = fn["year"], "filename_fallback", "filename"
            f.confidence, f.note = None, "year only, from file name"


def check_against_filename(result: LabelResult) -> None:
    """Kept for older callers; extract() now does this itself."""
    if not result.filename:
        result.filename = parse_filename(result.file_id)
    apply_filename(result, {})


def extract(image_path: str | Path, cfg: dict | None = None, client: VLMClient | None = None,
            fields: list[str] | None = None, evidence_dir: str | Path | None = None) -> LabelResult:
    cfg = cfg or load_config()
    client = client or VLMClient(cfg["vlm"])
    t0 = time.monotonic()
    image = Image.open(image_path)
    image.load()
    result = LabelResult(str(image_path), filename=parse_filename(Path(image_path).name),
                         extracted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    ev_dir = Path(evidence_dir) / Path(image_path).stem if evidence_dir else None
    for name in fields or FIELDS_ORDER:
        spec = cfg["fields"].get(name)
        if spec:
            result.fields[name] = read_field(client, image, name, spec, cfg, ev_dir)
    apply_filename(result, cfg)
    result.seconds = time.monotonic() - t0
    return result


def field_crops(image_path: str | Path, cfg: dict | None = None) -> dict[str, Image.Image]:
    """The exact primary crops the model sees, for inspecting/tuning regions."""
    cfg = cfg or load_config()
    image = Image.open(image_path)
    return {n: crop_field(image, s["region"], cfg["vlm"].get("crop_max_side", 1024))
            for n, s in cfg["fields"].items() if not n.startswith("_")}


# ---------------------------------------------------------------- outputs

def write_csv(results: list[LabelResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.row() for r in results]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def write_review_csv(results: list[LabelResult], path: Path) -> Path:
    rows = []
    for r in results:
        for name, f in r.fields.items():
            if f.status in REVIEW_STATUSES:
                rows.append({"file_id": r.file_id, "field": name, "status": f.status, "value": f.value,
                             "source": f.source, "filename_value": f.filename_value, "note": f.note,
                             "evidence": f.evidence or "", "raw": (f.raw or "")[:200]})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_id", "field", "status", "value", "source",
                                           "filename_value", "note", "evidence", "raw"])
        w.writeheader()
        w.writerows(rows)
    return path


SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS specimen_labels (
    run_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    image_path TEXT,
    extracted_at TEXT,
    extractor TEXT,
    model TEXT,
    needs_review INTEGER,
    filename_id TEXT, filename_year TEXT, filename_parse TEXT,
    individual_id TEXT, individual_id_status TEXT, individual_id_source TEXT,
    species TEXT, species_status TEXT, species_source TEXT,
    age TEXT, age_status TEXT, age_source TEXT,
    sex TEXT, sex_status TEXT, sex_source TEXT,
    feather TEXT, feather_status TEXT, feather_source TEXT,
    year TEXT, month TEXT, day TEXT, date_status TEXT, date_source TEXT,
    seconds REAL,
    detail_json TEXT,
    PRIMARY KEY (run_id, file_id)
)"""


def write_sqlite(results: list[LabelResult], path: Path, run_id: str, model: str) -> Path:
    """One row per slide and run; rerunning upserts, never duplicates."""
    conn = sqlite3.connect(path)
    conn.execute(SQL_SCHEMA)
    for r in results:
        row = {"run_id": run_id, "file_id": r.file_id, "image_path": r.image,
               "extracted_at": r.extracted_at, "extractor": EXTRACTOR_VERSION, "model": model,
               "needs_review": int(r.needs_review), "filename_id": r.filename.get("id"),
               "filename_year": r.filename.get("year"), "filename_parse": r.filename.get("parse"),
               "seconds": round(r.seconds, 2), "detail_json": json.dumps(r.to_json())}
        for name in FIELDS_ORDER:
            f = r.fields.get(name)
            if f is None:
                continue
            if name != "date":
                row[name] = export_value(f.value, f.status)
            row[f"{name}_status"] = f.status
            row[f"{name}_source"] = f.source
        for k, (v, st) in r.date_parts().items():
            row[k] = export_value(v, st)
        cols = list(row)
        conn.execute(f"INSERT OR REPLACE INTO specimen_labels ({','.join(cols)}) "
                     f"VALUES ({','.join('?' * len(cols))})", [row[c] for c in cols])
    conn.commit()
    conn.close()
    return path


def summarize(results: list[LabelResult]) -> dict:
    per_field: dict[str, Counter] = defaultdict(Counter)
    sources: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        for name, f in r.fields.items():
            per_field[name][f.status] += 1
            if f.value is not None:
                sources[name][f.source] += 1
    n = len(results)
    secs = [r.seconds for r in results if r.seconds]
    return {
        "slides": n,
        "needs_review": sum(r.needs_review for r in results),
        "slides_with_model_error": sum(r.has_model_error for r in results),
        "mean_seconds_per_slide": round(sum(secs) / len(secs), 1) if secs else None,
        "status_counts": {k: dict(v) for k, v in per_field.items()},
        "readable_rate": {k: round(v[READABLE] / n, 3) if n else None for k, v in per_field.items()},
        "value_source_counts": {k: dict(v) for k, v in sources.items()},
    }


def score_against_gold(results: list[LabelResult], gold_csv: str | Path,
                       cfg: dict | None = None) -> dict[str, dict]:
    """Per-field accuracy against a hand-labelled CSV (image + field columns).
    A blank gold cell is not scored. 'N/A' means the field is truly absent or
    fully covered: not_found (or a file-name fallback) scores.
    'wrong_value' counts readable values that are wrong (false acceptance)."""
    cfg = cfg or load_config()
    gold = {}
    with open(gold_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            gold[row.get("image") or row.get("file_id")] = row
    stats: dict[str, dict] = {}
    for r in results:
        g = gold.get(r.file_id)
        if not g:
            continue
        for name, f in r.fields.items():
            want = (g.get(name) or "").strip()
            if not want:
                continue
            s = stats.setdefault(name, {"n": 0, "correct": 0, "wrong_value": 0, "false_na": 0, "wrong": []})
            s["n"] += 1
            if want.upper() == NA:
                # Truly absent on the slide: not_found is right, and so is a
                # labelled file-name fallback (it never claims to be read).
                ok = f.status in ("not_found", "filename_fallback")
                want_norm = NA
            else:
                want_norm = normalize(want, cfg["fields"][name].get("normalize"))
                ok = (f.value or "") == (want_norm or "") and f.status != "filename_fallback"
                if f.status == "not_found":
                    s["false_na"] += 1
                elif not ok and f.status == READABLE:
                    s["wrong_value"] += 1
            if ok:
                s["correct"] += 1
            else:
                s["wrong"].append((r.file_id, f.value, want_norm, f.status))
    for s in stats.values():
        s["accuracy"] = round(s["correct"] / s["n"], 3) if s["n"] else None
    return stats


# ---------------------------------------------------------------- selection / CLI

def stratified_sample(images: list[str], n: int, seed: int = 0,
                      always: tuple[str, ...] = ("A0012 1996-im1058.jpg",)) -> list[str]:
    """n slides spread over ID prefixes and years, plus odd file names, so a
    small sample still shows the variation of the whole collection."""
    rnd = random.Random(seed)
    by_name = {Path(p).name: p for p in images}
    chosen = [by_name[a] for a in always if a in by_name]
    odd = [p for p in images if parse_filename(Path(p).name)["parse"] != "standard"]
    rnd.shuffle(odd)
    chosen += odd[:2]
    groups: dict[str, list[str]] = defaultdict(list)
    for p in images:
        fn = parse_filename(Path(p).name)
        if fn["parse"] == "standard" and p not in chosen:
            groups[re.match(r"[A-Z]+", fn["id"]).group(0)].append(p)
    for g in groups.values():
        rnd.shuffle(g)
    # Slots per ID prefix in proportion to its share of the collection
    # (largest remainder), so the sample looks like the full run.
    slots = max(0, n - len(chosen))
    total = sum(len(g) for g in groups.values()) or 1
    quota = {k: slots * len(g) / total for k, g in groups.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: -(quota[k] - alloc[k]))[: slots - sum(alloc.values())]:
        alloc[k] += 1
    year_of = lambda p: parse_filename(Path(p).name)["year"]  # noqa: E731
    years_used = Counter(year_of(p) for p in chosen)
    for k in sorted(alloc, key=lambda k: -len(groups[k])):
        g = groups[k]
        for _ in range(min(alloc[k], len(g))):
            # prefer a slide from a year not yet (or least) represented
            pick = min(g, key=lambda p: years_used[year_of(p)])
            g.remove(pick)
            chosen.append(pick)
            years_used[year_of(pick)] += 1
    return chosen[:n]


def _select(images: list[str], args) -> list[str]:
    if args.manifest:
        names = [ln.strip() for ln in Path(args.manifest).read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        by_name = {Path(p).name: p for p in images}
        missing = [nm for nm in names if nm not in by_name]
        if missing:
            print(f"warning: {len(missing)} manifest names not on the head node: {missing[:5]}")
        return [by_name[nm] for nm in names if nm in by_name]
    if args.match:
        return [p for p in images if any(m in Path(p).name for m in args.match)]
    if args.all:
        return images
    if args.stratified:
        return stratified_sample(images, args.stratified, args.seed)
    rnd = random.Random(args.seed)
    return rnd.sample(images, min(args.sample, len(images)))


def load_previous(jsonl: Path) -> dict[str, LabelResult]:
    done: dict[str, LabelResult] = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                r = LabelResult.from_json(json.loads(line))
                done[r.file_id] = r  # last record wins
    return done


def main(argv: list[str] | None = None) -> int:
    from src.notebook_remote_runtime import RemoteClusterIO, load_notebook_config_from_env

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match", action="append", help="substring of the image name (repeatable)")
    ap.add_argument("--manifest", help="text file with one file name per line")
    ap.add_argument("--stratified", type=int, help="pick N slides spread over ID prefixes and years")
    ap.add_argument("--write-manifest", help="only write the selected names to this file and exit")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fields", help="comma-separated subset of fields")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--workers", type=int, default=2, help="images in flight at once")
    ap.add_argument("--out", default=None, help="output dir (default data/label_runs/<timestamp>)")
    ap.add_argument("--gold", default=None, help="hand-labelled CSV to score against")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    client = VLMClient(cfg["vlm"])
    rc = load_notebook_config_from_env()
    io_ = RemoteClusterIO(rc)
    chosen = _select(io_.list_remote_images(), args)
    if args.write_manifest:
        Path(args.write_manifest).write_text("\n".join(Path(p).name for p in chosen) + "\n")
        print(f"wrote {len(chosen)} names to {args.write_manifest}")
        return 0

    out = Path(args.out or REPO / "data" / "label_runs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    run_id = out.name
    jsonl = out / "labels.jsonl"
    previous = load_previous(jsonl)
    todo = [p for p in chosen if Path(p).name not in previous or previous[Path(p).name].has_model_error]
    fields = args.fields.split(",") if args.fields else None
    cache = REPO / "data" / "raw_cache"
    print(f"{len(chosen)} slides -> {out}  ({len(chosen) - len(todo)} already done, {len(todo)} to run)")
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    write_lock = Lock()

    def one(p: str) -> LabelResult:
        local = cache / Path(p).name
        if not local.exists():
            part = cache / ".part"
            io_.fetch_remote_file(rc.head_host, p, part)
            (part / Path(p).name).replace(local)
        r = extract(local, cfg, client, fields, evidence_dir=out / "evidence")
        with write_lock:
            with open(jsonl, "a") as fh:
                fh.write(json.dumps(r.to_json()) + "\n")
        flag = " REVIEW" if r.needs_review else ""
        print(f"  {local.name} ({r.seconds:.0f}s): "
              + ", ".join(f"{k}={v.value!r}/{v.status}" for k, v in r.fields.items()) + flag, flush=True)
        return r

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(one, todo))

    final = load_previous(jsonl)
    results = [final[Path(p).name] for p in chosen if Path(p).name in final]
    write_csv(results, out / "labels.csv")
    write_review_csv(results, out / "review.csv")
    write_sqlite(results, out / "labels.sqlite3", run_id, cfg["vlm"]["model"])
    summary = summarize(results)
    if args.gold:
        score = score_against_gold(results, args.gold, cfg)
        summary["gold"] = {k: {kk: vv for kk, vv in v.items()} for k, v in score.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}/labels.csv, review.csv, labels.jsonl, labels.sqlite3, summary.json  "
          f"({summary['needs_review']} slides need review, "
          f"{summary['slides_with_model_error']} with model errors)")
    for name, counts in summary["status_counts"].items():
        print(f"  {name:14} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.gold:
        print("gold:")
        for name, s in summary["gold"].items():
            print(f"  {name:14} {s['correct']}/{s['n']}  acc={s['accuracy']}  "
                  f"wrong_value={s['wrong_value']}  false_na={s['false_na']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
