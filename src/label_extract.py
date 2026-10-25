"""Specimen label extraction: the second output table.

Reads the label fields (bird ID, species, age, sex, date) from the ORIGINAL
slide image, not the segmentation overlay. Each field gets its own crop
(configs/label_fields.json) and its own constrained VLM prompt, so one
obscured field cannot spoil the others. The red/blue text line under the
label boxes and the short code above the date (e.g. '2B') are ignored.

Identifiers:
  file_id     the digital file name, e.g. 'A0012 1996-im1058.jpg'
  bird_id     the code printed top-center on the label, e.g. 'A0012H';
              backup: the leading ID of the file name ('A0012')
  feather_id  one per segmented feather: <bird_id>_<year>_<letter>, letters
              A, B, C ... from left to right (e.g. 'A0012H_1996_A'); the same
              bird can have slides from several years
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
  partial            only part of the text is visible (fields with allow_partial)
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
import socket
import sqlite3
import sys
import tempfile
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
FIELDS_ORDER = ["bird_id", "species", "age", "sex", "date"]
DATE_PARTS = ["year", "month", "day"]
EXPORT_ORDER = ["bird_id", "species", "age", "sex", *DATE_PARTS]
EXTRACTOR_VERSION = "label_extract/3"
SEQ_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Everything printed on the labels is plain Latin letters, digits and a few
# separators. Anything else in an answer (escape sequences, non-breaking
# spaces, other scripts) means the model misfired: a technical error.
PLAIN_TEXT = re.compile(r"^[A-Za-z0-9 +\-./:;,'()]*$")

READABLE = "readable"
REVIEW_STATUSES = {"partial", "needs_review", "partly_visible", "conflict", "filename_fallback", "model_error"}
# label fields the feather IDs are built from: a feather whose slide has one of these untrusted is not usable
ID_FIELDS = ("bird_id", "date")
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
    node: str = ""
    model: str = ""
    feathers: list[dict] = field(default_factory=list)  # from segmentation, left to right
    # slide-level facts added after reading (annotate_slides): label_problems, photos_same_year,
    # handwriting, inferred fields
    extra: dict = field(default_factory=dict)

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
        seg = segmented(self.feathers)
        out["feather_count"] = len(seg) if self.feathers else None
        out["feather_ids"] = ";".join(f["feather_id"] for f in seg if f.get("feather_id"))
        out["possible_missed"] = sum(f.get("segment_status") == "possible_missed" for f in self.feathers)
        for k in ("label_problems", "label_note", "photos_same_year", "multiple_photos_same_year", "handwritten",
                  "handwritten_fields", "handwritten_model"):
            out[k] = self.extra.get(k)
        return out

    def id_parts(self) -> dict:
        """bird_id and year used for feather IDs, with where each came from:
        the label when it gave a value, else the file name."""
        f = self.fields.get("bird_id")
        if f is not None and f.value and f.status != "model_error":
            bird, bird_src = f.value, ("label" if f.status != "filename_fallback" else "filename")
            bird_status = f.status
        elif self.filename.get("id"):
            bird, bird_src, bird_status = self.filename["id"], "filename", "filename_fallback"
        else:
            bird, bird_src, bird_status = "UNKNOWN", "none", "not_found"
        d = self.fields.get("date")
        year = split_date(d.value)["year"] if d is not None and d.status != "model_error" else None
        if year:
            year_src = "label" if d.status != "filename_fallback" else "filename"
            year_status = d.status
        elif self.filename.get("year"):
            year, year_src, year_status = self.filename["year"], "filename", "filename_fallback"
        else:
            year, year_src, year_status = "UNKNOWN", "none", "not_found"
        return {"bird_id": bird, "bird_id_source": bird_src, "bird_id_status": bird_status,
                "year": year, "year_source": year_src, "year_status": year_status}

    def to_json(self) -> dict:
        d = {"file_id": self.file_id, "image": self.image, "extracted_at": self.extracted_at,
             "extractor": EXTRACTOR_VERSION, "seconds": round(self.seconds, 2),
             "node": self.node, "model": self.model, "feathers": self.feathers,
             "filename": self.filename, "needs_review": self.needs_review,
             "fields": {k: asdict(v) for k, v in self.fields.items()}, "extra": self.extra}
        d["date_parts"] = {k: {"value": v, "status": s} for k, (v, s) in self.date_parts().items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "LabelResult":
        r = cls(d["image"], filename=d.get("filename", {}), seconds=d.get("seconds", 0),
                extracted_at=d.get("extracted_at", ""), node=d.get("node", ""), model=d.get("model", ""),
                feathers=d.get("feathers") or [], extra=d.get("extra") or {})
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


def segmented(feathers: list[dict] | None) -> list[dict]:
    """Feather records with a crop (not the inventory's review-only rows)."""
    return [f for f in feathers or [] if f.get("segment_status", "segmented") == "segmented"]


def feather_id(bird_id: str, period: str, index: int) -> str:
    """'A0012H', '1996', 0 -> 'A0012H_1996_A'; period is the full YYYYMMDD
    when the bird has several slides that year ('BB17692H_19980612_A')."""
    return f"{bird_id}_{period}_{seq_letter(index)}"


def result_date(r: "LabelResult") -> str | None:
    d = r.fields.get("date")
    return d.value if d is not None and d.status not in ("model_error", "not_found") else None


def feather_period(result: "LabelResult", siblings: list[str] | None,
                   sibling_dates: dict[str, str | None] | None = None) -> str:
    """The period in feather IDs and file names: the year; the label date as
    YYYYMMDD when another slide in `siblings` has the same file-name ID and
    year; plus the image number when a sibling has the same (or an unknown)
    date. Same rule as output_names.output_names."""
    from src.output_names import output_names

    parts = result.id_parts()
    if not siblings or not shares_id_year(result.file_id, siblings):
        return parts["year"]
    return output_names(result.file_id, siblings, result_date(result), sibling_dates)["period"]


def slide_date(image_path: str | Path, output_dir: str | Path, client, cfg: dict | None = None) -> str | None:
    """The label date of a slide, for naming its same-year siblings: from the
    slide's pipeline result if there is one, else a date-only reading cached
    as <output_dir>/labels/<stem>.date.json."""
    label_dir = Path(output_dir) / "labels"
    stem = Path(image_path).stem
    for p in (label_dir / f"{stem}.json", label_dir / f"{stem}.date.json"):
        try:
            r = LabelResult.from_json(json.loads(p.read_text()))
            if "date" in r.fields and r.fields["date"].status != "model_error":
                return result_date(r)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        r = extract(image_path, cfg, client, fields=["date"])
    except Exception:  # noqa: BLE001
        return None
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"{stem}.date.json").write_text(json.dumps(r.to_json()))
    return result_date(r)


def assign_feather_ids(result: "LabelResult", crops: list[str], period: str | None = None,
                       seqs: list[str] | None = None) -> list[dict]:
    """Feather IDs for a slide's segmented feathers. `crops` must already be in
    left-to-right order (the segmenter sorts by x). The period comes from the
    crop names when they carry one (year / YYYYMMDD), else the label year.
    Stored on the result."""
    parts = result.id_parts()
    first = next((c for c in crops if c), None)
    period = period or (period_of_crop(first) if first else None) or parts["year"]
    seqs = seqs or [seq_letter(i) for i in range(len(crops))]
    result.feathers = [{"seq": q, "feather_id": f"{parts['bird_id']}_{period}_{q}", "period": period,
                        "crop": Path(c).name if c else "", **parts} for q, c in zip(seqs, crops)]
    return result.feathers


def attach_feathers_from_crops(results: list["LabelResult"], crops_dir: str | Path,
                               siblings: list[str] | None = None) -> int:
    """Feather IDs from a segmentation output folder (crops found under the
    current or the older file names, left to right). `siblings`: all slide
    file names, needed for older crop names to tell same-year slides apart."""
    n = 0
    dates = {r.file_id: result_date(r) for r in results}
    for r in results:
        if r.feathers and any("segment_status" in f for f in r.feathers):
            n += 1      # records written by the pipeline itself (scores, missed feathers): keep them
            continue
        rec = Path(crops_dir) / "labels" / f"{Path(r.image).stem}.json"
        if rec.exists():
            # written by the pipeline itself: full feather records (scores, measurements)
            try:
                feathers = json.loads(rec.read_text()).get("feathers") or []
            except ValueError:
                feathers = []
            if feathers:
                r.feathers = feathers
                n += 1
                continue
        d = r.fields.get("date")
        crops = [str(p) for p in find_crops(crops_dir, r.file_id, d.value if d is not None else None)]
        if crops:
            period = None if period_of_crop(crops[0]) else feather_period(r, siblings, dates)
            assign_feather_ids(r, crops, period)
            n += 1
    return n


def load_config(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_CONFIG).read_text())


# ---------------------------------------------------------------- file name

# File-name parsing lives with the output naming rules (no heavy imports).
from src.output_names import find_crops, parse_filename, period_of_crop, seq_letter, shares_id_year  # noqa: E402
from src.output_names import compact_date  # noqa: E402


# ---------------------------------------------------------------- crops

def _clip(region: list[float]) -> list[float]:
    l, t, r, b = (min(1.0, max(0.0, v)) for v in region)
    return [round(l, 4), round(t, 4), round(r, 4), round(b, 4)]


def crop_plan(spec: dict, cfg: dict) -> list[tuple[str, list[float]]]:
    """Ordered (name, region) list: primary, expanded, then configured fallbacks."""
    l, t, r, b = spec["region"]
    mx, my = spec.get("expand_margin", cfg.get("expand_margin", [0.06, 0.04]))
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
        self.model_name = cfg["model"]
        self.retries = int(cfg.get("retries", 2))
        self.timeout = float(cfg.get("timeout", 180))

    def _ask_once(self, image: Image.Image, prompt: str, max_tokens: int | None = None) -> str:
        with self._lock:
            base = next(self._endpoints)
        body = {
            "model": self.cfg["model"],
            "max_tokens": int(max_tokens or self.cfg.get("max_tokens", 120)),
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

    def ask(self, image: Image.Image, prompt: str, timeout: float | None = None,
            max_tokens: int | None = None) -> str:
        """`max_tokens`: answer length for this call (default: the label config's,
        sized for short field answers)."""
        if timeout:
            self.timeout = timeout
        last: Exception | None = None
        for i in range(self.retries + 1):
            try:
                return self._ask_once(image, prompt, max_tokens)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if i < self.retries:
                    time.sleep(3 * (i + 1))
        assert last is not None
        raise last


class MLXLocalClient:
    """Same interface as VLMClient, but runs on a worker's already-loaded MLX
    model (FeatherProcessor), so a cluster run does not funnel every label
    request through one VLM server and loads no second copy of the model."""

    def __init__(self, generate, model, processor, chat_prompt, model_name: str, max_tokens: int = 120):
        self.generate, self.model, self.processor = generate, model, processor
        self.chat_prompt, self.model_name, self.max_tokens = chat_prompt, model_name, max_tokens
        self._lock = Lock()  # one MLX generation at a time per process

    def ask(self, image: Image.Image, prompt: str, timeout: float | None = None,
            max_tokens: int | None = None) -> str:
        with self._lock, tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name, "JPEG", quality=92)
            out = self.generate(self.model, self.processor, prompt=self.chat_prompt(prompt),
                                image=[tmp.name], max_tokens=int(max_tokens or self.max_tokens))
        text = getattr(out, "text", str(out))
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def normalize(value: str | None, how: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip()
    v = v.replace("\u00a0", " ").strip()
    if v.lower() in ("null", "none", "n/a", ""):
        return None
    if how == "remove_spaces_upper":
        v = re.sub(r"\s+", "", v).upper()
    elif how == "date":
        m = re.search(r"(\d{4})\D?(\d{1,2})\D?(\d{1,2})", v)
        if m:
            v = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return v or None


class MalformedAnswer(ValueError):
    """The model started a JSON answer but it does not parse (often a
    degenerate, truncated reply). A technical failure, not a reading."""


def parse_answer(text: str) -> tuple[str | None, str, float | None]:
    """Returns (value, visible, confidence) from the model's JSON answer."""
    if re.search(r"\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}", text):
        raise MalformedAnswer(f"escape sequence in answer: {text[:60]!r}")
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            conf = d.get("confidence")
            val = d.get("value")
            return (None if val is None else str(val), str(d.get("visible", "yes")).lower(),
                    float(conf) if isinstance(conf, (int, float)) else None)
        except (ValueError, TypeError, AttributeError):
            raise MalformedAnswer(f"unparseable JSON answer: {text[:80]!r}") from None
    if "{" in text:
        raise MalformedAnswer(f"truncated JSON answer: {text[:80]!r}")
    # Not JSON: treat the whole reply as the value, confidence unknown.
    t = text.strip().strip("`").strip()
    return (t or None, "yes" if t else "no", None)


def _classify(a: "Attempt", spec: dict, cfg: dict) -> tuple[str, str]:
    """(status, reason) for one answer."""
    value, visible = a.value, a.visible
    if value is None or visible == "no":
        return "not_found", ""
    if spec.get("allow_partial") and (visible == "partly" or "..." in value):
        return "partial", ""
    pattern = spec.get("pattern")
    if pattern and not re.match(pattern, value):
        return "needs_review", "format"
    if visible == "partly":
        # Characters partly hidden. The model's confidence is not a reliable guard (it reported
        # 0.85-0.95 for a bird ID with its prefix covered), so fields set to partial "review"
        # (bird ID, date: the feather IDs depend on them) are never readable then; others are
        # accepted at partial_min_confidence (short, low-variance text such as sex or species).
        if spec.get("partial") == "review":
            return "partly_visible", f"partly visible, confidence {a.confidence}"
        floor = float(spec.get("partial_min_confidence", cfg.get("partial_min_confidence", 0.8)))
        if a.confidence is None or a.confidence < floor:
            return "partly_visible", f"partly visible, confidence {a.confidence}"
        return READABLE, f"partly visible (accepted), confidence {a.confidence}"
    return READABLE, ""


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
    try:
        value, visible, conf = parse_answer(raw)
        if value is not None and not PLAIN_TEXT.match(str(value)):
            raise MalformedAnswer(f"characters that are not on any label: {str(value)[:40]!r}")
    except MalformedAnswer as exc:
        return Attempt(crop_name, region, None, "error", None, raw, round(time.monotonic() - t, 2),
                       error=str(exc), evidence=ev)
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
    why = ""
    plan = crop_plan(spec, cfg)
    for crop_name, region in plan:
        a = _attempt(client, image, crop_name, region, spec, cfg, evidence_dir, name,
                     FALLBACK_HINT if crop_name.startswith("fallback") else "")
        attempts.append(a)
        if a.error:
            continue  # technical failure: try the next crop, decide below
        st, reason = _classify(a, spec, cfg)
        if st in (READABLE, "partial"):
            best, why = (a, st), reason
            break
        if st == "partly_visible" or (st == "needs_review" and reason != "format"):
            best, why = (a, st), reason  # seen but uncertain: other crops show the same hidden text
            break
        if st == "needs_review" and best is None:
            best, why = (a, st), "value fails the field format"  # keep looking for a clean read

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
                      note=why, evidence=a.evidence, attempts=attempts)

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


def reclassify(result: LabelResult, cfg: dict) -> int:
    """Re-derives each field's status from its stored answers under the current rules
    (_classify), without asking the model again: for results read under older rules.
    Fields resolved otherwise (not_found, model_error, filename, conflict, disagreeing
    readings, partial dates) are left as they are. Returns the number of fields changed."""
    changed = 0
    for name, f in result.fields.items():
        spec = cfg["fields"].get(name, {})
        if f.status not in (READABLE, "needs_review", "partly_visible") or f.note.startswith("readings disagree"):
            continue
        a = next((x for x in f.attempts if x.crop == f.source and not x.error), None)
        if a is None:
            continue
        st, reason = _classify(a, spec, cfg)
        if st == "needs_review" and reason == "format":
            continue
        if (st, reason) != (f.status, f.note):
            f.status, f.note = st, reason
            changed += 1
    return changed


def apply_filename(result: LabelResult, cfg: dict) -> None:
    """Cross-check against the file name; fill from it only where the slide
    gave nothing and the name follows the standard convention."""
    fn = result.filename
    allow = cfg.get("filename_fallback", True) and fn.get("parse") == "standard"

    f = result.fields.get("bird_id")
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


def check_handwriting(client, image: Image.Image, hcfg: dict, cfg: dict) -> dict:
    """Asks whether any label text is handwritten (written, corrected or crossed out by hand),
    on each region in hcfg['regions'] (label strip, date corner). Returns extra columns:
    handwritten 'yes'|'no'|'' (unknown: the request failed), handwritten_fields, label_note."""
    found, fields, notes, failed = False, [], [], False
    for region in hcfg.get("regions", []):
        crop = crop_field(image, region, cfg["vlm"].get("crop_max_side", 1024))
        try:
            text = client.ask(crop, hcfg["prompt"], max_tokens=int(hcfg.get("max_tokens", 120)))
            m = re.search(r"\{.*\}", text or "", re.S)
            d = json.loads(m.group(0)) if m else None
            if not isinstance(d, dict) or not isinstance(d.get("handwritten"), bool):
                raise ValueError("no handwriting answer")
        except Exception:  # noqa: BLE001
            failed = True
            continue
        if d.get("handwritten") is True:
            found = True
            fields += [str(x) for x in d.get("fields") or []]
            if d.get("notes"):
                notes.append(str(d["notes"]))
    return {"handwritten": "yes" if found else ("" if failed else "no"),
            "handwritten_fields": ", ".join(dict.fromkeys(fields)), "label_note": "; ".join(notes),
            "handwriting_source": "model"}


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
    if (cfg.get("handwriting") or {}).get("enabled") and fields is None:
        result.extra.update(check_handwriting(client, image, cfg["handwriting"], cfg))
    result.seconds = time.monotonic() - t0
    result.model = getattr(client, "model_name", "")
    result.node = socket.gethostname()
    return result


def failed_result(image_path: str | Path, exc: BaseException, fields: list[str] | None = None) -> LabelResult:
    """A crash outside the per-field handling (bad image, worker bug): every
    field is model_error, so the slide is retried, never exported as N/A."""
    r = LabelResult(str(image_path), filename=parse_filename(Path(image_path).name),
                    extracted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    node=socket.gethostname())
    for name in fields or FIELDS_ORDER:
        r.fields[name] = FieldResult(None, "model_error", raw=f"{type(exc).__name__}: {exc}",
                                     note="label extraction crashed")
    return r


def extract_for_pipeline(image_path: str | Path, output_dir: str | Path, client,
                         cfg: dict | None = None) -> LabelResult:
    """Pipeline entry point (workers): one result per slide and output dir,
    cached as <output_dir>/labels/<stem>.json with evidence crops next to it.
    A task retry or rerun reuses a finished result instead of asking the model
    again; results with a model_error are redone."""
    label_dir = Path(output_dir) / "labels"
    cache = label_dir / f"{Path(image_path).stem}.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text())
            r = LabelResult.from_json(d)
            if d.get("extractor") == EXTRACTOR_VERSION and not r.has_model_error:
                reclassify(r, cfg or load_config())       # cached under older rules: re-derive statuses
                return r
        except (ValueError, TypeError, KeyError):
            pass
    try:
        r = extract(image_path, cfg, client, evidence_dir=label_dir / "evidence")
    except Exception as exc:  # noqa: BLE001
        r = failed_result(image_path, exc)
    return save_pipeline_result(r, output_dir)


def save_pipeline_result(r: LabelResult, output_dir: str | Path) -> LabelResult:
    """Writes <output_dir>/labels/<stem>.json atomically (also after feather
    IDs are added by segmentation)."""
    label_dir = Path(output_dir) / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    cache = label_dir / f"{Path(r.image).stem}.json"
    tmp = cache.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(r.to_json()))
    tmp.replace(cache)
    return r


def label_metrics(r: LabelResult) -> dict[str, int]:
    """Counters for the run monitors (local state JSON, Redis). Technical
    failures and true absences are separate counters, never one 'missing'."""
    m = {"labels_done": 1, "labels_needs_review": int(r.needs_review),
         "labels_model_error": int(r.has_model_error)}
    for name, f in r.fields.items():
        m[f"labels_{name}_{f.status}"] = 1
    return m


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
    node_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    image_path TEXT,
    extracted_at TEXT,
    extractor TEXT,
    model TEXT,
    needs_review INTEGER,
    filename_id TEXT, filename_year TEXT, filename_parse TEXT,
    bird_id TEXT, bird_id_status TEXT, bird_id_source TEXT,
    species TEXT, species_status TEXT, species_source TEXT,
    age TEXT, age_status TEXT, age_source TEXT,
    sex TEXT, sex_status TEXT, sex_source TEXT,
    year TEXT, month TEXT, day TEXT, date_status TEXT, date_source TEXT,
    feather_count INTEGER, feather_ids TEXT,
    seconds REAL,
    detail_json TEXT,
    PRIMARY KEY (run_id, node_id, file_id)
)"""


# The per-feather output table: one row per kept feather crop, with the
# slide's label fields, the segmentation measurements and the VLM judge.
FEATHER_TABLE_COLS = [
    "run_id", "node_id", "file_id", "feather_id", "seq", "segment_status", "usable", "excluded_because",
    "duplicate_of", "needs_review", "label_needs_review", "label_problems", "label_note", "photos_same_year",
    "multiple_photos_same_year", "handwritten", "handwritten_fields",
    "bird_id", "bird_id_status", "bird_id_source", "species", "species_status", "age", "age_status",
    "sex", "sex_status", "year", "month", "day", "date_status", "year_source", "period",
    "crop", "qc_image", "mask_image", "bbox", "area_px", "length_px", "width_px", "angle_deg", "touches_border",
    "sam_score", "sam_masks", "postprocess", "mask_cleanup",
    # per-feather checks (src/feather_judge.py): score is derived from the checks
    "judge_status", "judge_score", "judge_segment_ok", "judge_sides_cut", "judge_outline_left",
    "judge_outline_right", "judge_outline_top", "judge_outline_bottom", "judge_holes_or_gaps", "judge_holes",
    "judge_hole_frac", "judge_mask_parts", "judge_single_feather", "judge_fragment", "judge_other_feather_inside",
    "judge_other_material_inside", "judge_glue_inside", "judge_hidden", "judge_straight_outline", "judge_straight_longest", "judge_straight_across", "judge_straight_along", "judge_regrow_left", "judge_regrow_right", "judge_model_score", "judge_notes", "judge_error",
    "review_uncertain", "judge_uncertain_checks", "repair", "repair_tried",
    # missed feathers (src/feather_inventory.py): slide-level result repeated per row
    "missed_x", "missed_between", "missed_partly_hidden", "inventory_status", "inventory_total",
    "feathers_segmented", "feathers_possibly_missed", "inventory_note", "inventory_image",
]
FEATHER_SQL = ("CREATE TABLE IF NOT EXISTS specimen_feathers (\n    "
               + ",\n    ".join(f"{c} TEXT" + (" NOT NULL" if c in ("run_id", "node_id", "file_id", "seq") else "")
                                 for c in FEATHER_TABLE_COLS)
               + ",\n    PRIMARY KEY (run_id, node_id, file_id, seq)\n)")


def usable(f: dict) -> tuple[bool, str]:
    """Is the feather usable for analysis, and if not, why: a segmented feather
    with an ID that passed its checks and is not a duplicate. Checks failed on
    one look only (review_uncertain) do not exclude it; label review does not
    either (it concerns the slide's fields, in label_needs_review)."""
    status = f.get("segment_status", "segmented")
    if status != "segmented":
        return False, {"fragment": "fragment", "possible_missed": "review row: possible missed feather",
                       "count_unclear": "review row: feather count unclear"}.get(status, str(status))
    if not f.get("feather_id"):
        return False, "no feather ID"
    if f.get("needs_review"):
        return False, f"check: {f['needs_review']}"
    if f.get("duplicate_of"):
        return False, f"duplicate of {f['duplicate_of']}"
    return True, ""


def feather_table_rows(r: "LabelResult", run_id: str = "", node_id: str = "") -> list[dict]:
    """Rows of the per-feather table for one slide (label fields repeated per feather)."""
    slide = r.row()
    parts = r.id_parts() if r.fields else {}
    rows = []
    for f in r.feathers:
        row = {c: None for c in FEATHER_TABLE_COLS}
        row.update({"run_id": run_id, "node_id": node_id or r.node, "file_id": r.file_id,
                    "label_needs_review": slide.get("needs_review")})
        for c in ("species", "species_status", "age", "age_status", "sex", "sex_status",
                  "year", "month", "day", "date_status"):
            row[c] = slide.get(c)
        row["bird_id"] = f.get("bird_id") or parts.get("bird_id") or slide.get("bird_id")
        row["bird_id_status"] = slide.get("bird_id_status")
        row["bird_id_source"] = f.get("bird_id_source") or parts.get("bird_id_source")
        slide_cols = {"species", "species_status", "age", "age_status", "sex", "sex_status", "year", "month",
                      "day", "date_status", "bird_id", "bird_id_status"}
        for c in FEATHER_TABLE_COLS:
            # label columns come from the slide's current reading, never from the copy a feather
            # record kept when its ID was assigned (statuses can be re-derived later: reclassify)
            if c in f and c not in slide_cols:
                v = f[c]
                row[c] = json.dumps(v) if isinstance(v, (list, dict)) else v
        ok, why = usable(f)
        if ok:   # the feather ID rests on the slide's bird ID and date: both must be read with confidence
            untrusted = [n for n in ID_FIELDS if (fr := r.fields.get(n)) is not None and fr.status not in (READABLE,)]
            if untrusted:
                ok, why = False, "label: " + ", ".join(f"{n} {r.fields[n].status.replace('_', ' ')}" for n in untrusted)
        for k in ("label_problems", "label_note", "photos_same_year", "multiple_photos_same_year", "handwritten",
                  "handwritten_fields"):
            row[k] = r.extra.get(k)
        row["usable"], row["excluded_because"] = ("yes" if ok else "no"), why
        rows.append(row)
    return rows


def _expected_cols(sql: str) -> set[str]:
    conn = sqlite3.connect(":memory:")
    conn.execute(sql)
    table = re.search(r"EXISTS (\w+)", sql).group(1)
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_label_table(conn: sqlite3.Connection) -> None:
    """Creates specimen_labels/specimen_feathers; a table from an older layout
    (different columns or key) is renamed to <table>_old<N> and kept."""
    for table, sql in (("specimen_labels", SQL_SCHEMA), ("specimen_feathers", FEATHER_SQL)):
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if cols and cols != _expected_cols(sql):
            n = 1
            while conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (f"{table}_old{n}",)).fetchone():
                n += 1
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old{n}")
        conn.execute(sql)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_specimen_labels_run ON specimen_labels(run_id)")


def upsert_labels(conn: sqlite3.Connection, results: list[LabelResult], run_id: str,
                  model: str = "", node_id: str | None = None) -> None:
    """One row per (run, node, slide); rerunning replaces the row, never duplicates."""
    ensure_label_table(conn)
    for r in results:
        row = {"run_id": run_id, "node_id": node_id or r.node or socket.gethostname(),
               "file_id": r.file_id, "image_path": r.image,
               "extracted_at": r.extracted_at, "extractor": EXTRACTOR_VERSION, "model": r.model or model,
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
        row["feather_count"] = len(segmented(r.feathers)) if r.feathers else None
        row["feather_ids"] = ";".join(f["feather_id"] for f in segmented(r.feathers) if f.get("feather_id"))
        cols = list(row)
        conn.execute(f"INSERT OR REPLACE INTO specimen_labels ({','.join(cols)}) "
                     f"VALUES ({','.join('?' * len(cols))})", [row[c] for c in cols])
        conn.execute("DELETE FROM specimen_feathers WHERE run_id=? AND node_id=? AND file_id=?",
                     (run_id, row["node_id"], r.file_id))
        for frow in feather_table_rows(r, run_id, row["node_id"]):
            conn.execute(f"INSERT OR REPLACE INTO specimen_feathers ({','.join(FEATHER_TABLE_COLS)}) "
                         f"VALUES ({','.join('?' * len(FEATHER_TABLE_COLS))})",
                         [None if frow[c] is None else str(frow[c]) for c in FEATHER_TABLE_COLS])
    conn.commit()


def write_sqlite(results: list[LabelResult], path: Path, run_id: str, model: str) -> Path:
    conn = sqlite3.connect(path, timeout=30)
    try:
        upsert_labels(conn, results, run_id, model)
    finally:
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


def _append_result(jsonl: Path, r: LabelResult, lock: Lock) -> None:
    with lock, open(jsonl, "a") as fh:
        fh.write(json.dumps(r.to_json()) + "\n")


def _log_result(r: LabelResult, log=print) -> None:
    flag = " REVIEW" if r.needs_review else ""
    log(f"  {r.file_id} ({r.seconds:.0f}s{', ' + r.node if r.node else ''}): "
        + ", ".join(f"{k}={v.value!r}/{v.status}" for k, v in r.fields.items()) + flag)


def pending_images(chosen: list[str], out: Path) -> list[str]:
    """Slides in `chosen` without a finished result in out/labels.jsonl
    (missing, or with a model_error): what a resumed run still has to do."""
    previous = load_previous(out / "labels.jsonl")
    return [p for p in chosen if Path(p).name not in previous or previous[Path(p).name].has_model_error]


def run_local(chosen: list[str], out: Path, cfg: dict, fields: list[str] | None = None,
              workers: int = 2, log=print) -> None:
    """Reads labels from this machine: slides are copied from the head node
    into data/raw_cache, the model is the VLM server (FEATHER_VLM_ENDPOINTS)."""
    from src.notebook_remote_runtime import RemoteClusterIO, load_notebook_config_from_env

    rc = load_notebook_config_from_env()
    io_ = RemoteClusterIO(rc)
    client = VLMClient(cfg["vlm"])
    cache = REPO / "data" / "raw_cache"
    out.mkdir(parents=True, exist_ok=True)
    lock = Lock()

    def one(p: str) -> None:
        local = Path(p) if Path(p).exists() else cache / Path(p).name
        if not local.exists():
            part = cache / ".part"
            io_.fetch_remote_file(rc.head_host, p, part)
            (part / Path(p).name).replace(local)
        try:
            r = extract(local, cfg, client, fields, evidence_dir=out / "evidence")
        except Exception as exc:  # noqa: BLE001
            r = failed_result(local, exc, fields)
        _append_result(out / "labels.jsonl", r, lock)
        _log_result(r, log)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, chosen))


def run_distributed(chosen: list[str], out: Path, remote_output_dir: str | None = None,
                    timeout: float = 3600, log=print, app=None) -> None:
    """Sends one src.celery_tasks.extract_labels task per slide to the cluster
    workers (image paths as seen on the nodes) and collects the results through
    the Celery result backend. Workers keep their evidence crops under
    <remote_output_dir>/labels/evidence/."""
    from src.notebook_remote_runtime import (RemoteClusterIO, assert_tcp_reachable, create_celery_client,
                                             load_notebook_config_from_env)

    if app is None:
        rc = load_notebook_config_from_env()
        assert_tcp_reachable(rc.broker_url, "Broker")
        app = create_celery_client(rc)
        if remote_output_dir is None:
            remote_output_dir = RemoteClusterIO(rc).remote_runtime_path(
                f"{rc.remote_repo_dir}/data/runs/labels-{out.name}/processed")
    out.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    from src.notebook_remote_runtime import task_queue

    pending = {app.send_task("src.celery_tasks.extract_labels", args=[p, remote_output_dir], queue=task_queue()): p
               for p in chosen}
    log(f"sent {len(pending)} label tasks to the workers -> {remote_output_dir}")
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        for res in list(pending):
            if not res.ready():
                continue
            p = pending.pop(res)
            try:
                r = LabelResult.from_json(res.get(timeout=10))
            except Exception as exc:  # noqa: BLE001  (task failed after its retries)
                r = failed_result(p, exc)
            _append_result(out / "labels.jsonl", r, lock)
            _log_result(r, log)
        time.sleep(1)
    for p in pending.values():
        log(f"  TIMEOUT {Path(p).name}: still queued or running; rerun to collect it")


def merge_feather_scores(results: list[LabelResult], scores_csv: str | Path) -> int:
    """Adds judge_* / qc_image columns from a CSV keyed by crop file name (e.g.
    scripts/judge_existing_crops.py output for crops made by older code)."""
    from src.feather_judge import needs_review

    with open(scores_csv, newline="") as fh:
        by_crop = {row["crop"]: row for row in csv.DictReader(fh)}
    n = 0
    for r in results:
        for f in r.feathers:
            row = by_crop.get(f.get("crop"))
            if not row:
                continue
            for k, v in row.items():
                if k.startswith("judge_") or k == "qc_image":
                    if v in ("True", "False"):
                        v = v == "True"
                    elif k == "judge_score" and v:
                        v = float(v)
                    f[k] = v if v != "" else None
            f["needs_review"] = needs_review(f)
            n += 1
    return n


def write_feathers_csv(results: list[LabelResult], path: Path, run_id: str = "",
                       usable_only: bool = False) -> Path:
    """The per-feather output table (FEATHER_TABLE_COLS), one row per feather,
    fragment and review row; `usable_only`: only the feathers usable for analysis."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FEATHER_TABLE_COLS)
        w.writeheader()
        for r in results:
            w.writerows(row for row in feather_table_rows(r, run_id) if not usable_only or row["usable"] == "yes")
    return path


def append_feathers_csv(r: LabelResult, path: str | Path, run_id: str = "", node_id: str = "") -> None:
    """Appends one slide's feathers to a per-feather CSV (header on first write)."""
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FEATHER_TABLE_COLS)
        if new:
            w.writeheader()
        w.writerows(feather_table_rows(r, run_id, node_id))


def bare_result(image_path: str, feathers: list[dict]) -> LabelResult:
    """A slide without a label reading (labels disabled): feathers only."""
    r = LabelResult(str(image_path), filename=parse_filename(Path(image_path).name), node=socket.gethostname())
    seg = segmented(feathers)
    if seg and not all(f.get("feather_id") for f in seg):
        # feather IDs from the file name's ID and year (the label could not be read);
        # fragments and review rows keep no ID
        ids = assign_feather_ids(r, [f["crop"] for f in seg], period_of_crop(seg[0]["crop"]),
                                 seqs=[f["seq"] for f in seg])
        ided = iter([{**f, **i} for f, i in zip(seg, ids)])
        feathers = [next(ided) if f.get("segment_status", "segmented") == "segmented" else f for f in feathers]
    r.feathers = feathers
    return r


def load_gold(gold_csv: str | Path) -> dict[str, dict]:
    with open(gold_csv, newline="") as fh:
        return {(row.get("image") or row.get("file_id")): row for row in csv.DictReader(fh)}


def slide_problems(r: LabelResult, gold_row: dict | None, cfg: dict) -> list[dict]:
    """Everything worth a look on one slide, worst first: wrong values and
    missed fields (vs the gold answers), technical errors, reviews, fallbacks."""
    out = []
    for name, f in r.fields.items():
        want = (gold_row or {}).get(name, "").strip()
        if want:
            if want.upper() == NA:
                if f.status not in ("not_found", "filename_fallback", "model_error"):
                    kind = "wrong" if f.status == READABLE else "flagged_wrong"
                    out.append({"field": name, "kind": kind, "got": f.value, "want": NA, "status": f.status,
                                "note": f.note})
                    continue
            else:
                want_n = normalize(want, cfg["fields"][name].get("normalize"))
                if f.status == "not_found":
                    out.append({"field": name, "kind": "missed", "got": None, "want": want_n, "status": f.status})
                    continue
                if f.value != want_n and f.status != "model_error":
                    kind = "wrong" if f.status == READABLE else "flagged_wrong"
                    out.append({"field": name, "kind": kind, "got": f.value, "want": want_n, "status": f.status})
                    continue
        if f.status == "model_error":
            out.append({"field": name, "kind": "model_error", "got": None, "want": want or None, "status": f.status})
        elif f.status in ("needs_review", "conflict", "filename_fallback", "partial"):
            out.append({"field": name, "kind": f.status, "got": f.value, "want": want or None, "status": f.status,
                        "note": f.note})
    for f in r.feathers or []:
        if f.get("segment_status") in ("possible_missed", "count_unclear"):
            # the inventory's flags: counting overlapping feathers is unreliable, so review only
            out.append({"field": "feathers", "kind": "possible_missed", "got": None, "want": None,
                        "status": f["segment_status"], "note": f.get("needs_review")})
        elif f.get("segment_status") == "fragment":
            out.append({"field": f"feather {f.get('seq')}", "kind": "fragment", "got": None, "want": None,
                        "status": f.get("judge_status"), "note": f.get("needs_review")})
        elif f.get("needs_review"):
            out.append({"field": f"feather {f.get('seq')}", "kind": "feather_check", "got": None, "want": None,
                        "status": f.get("judge_status"), "note": f.get("needs_review")})
        if f.get("review_uncertain"):
            out.append({"field": f"feather {f.get('seq')}", "kind": "feather_uncertain", "got": None, "want": None,
                        "status": f.get("judge_status"), "note": "one look only: " + f["review_uncertain"]})
    want_n = (gold_row or {}).get("feather_count", "").strip()
    if want_n and r.feathers and str(len(segmented(r.feathers))) != want_n:
        out.append({"field": "feathers", "kind": "count", "got": len(segmented(r.feathers)), "want": int(want_n),
                    "status": "segmentation"})
    rank = {"wrong": 0, "missed": 1, "model_error": 2, "count": 3, "flagged_wrong": 4, "conflict": 5,
            "needs_review": 6, "possible_missed": 7, "feather_check": 8, "fragment": 8, "filename_fallback": 9, "partial": 10,
            "feather_uncertain": 11}
    return sorted(out, key=lambda p: rank.get(p["kind"], 9))


def score_feather_counts(results: list[LabelResult], gold_csv: str | Path) -> dict | None:
    gold = load_gold(gold_csv)
    rows = [(r.file_id, len(segmented(r.feathers)), int(gold[r.file_id]["feather_count"])) for r in results
            if r.feathers and (gold.get(r.file_id) or {}).get("feather_count", "").strip()]
    if not rows:
        return None
    return {"n": len(rows), "correct": sum(g == w for _, g, w in rows),
            "wrong": [(f, g, w) for f, g, w in rows if g != w]}


def finish_run(out: Path, chosen: list[str], cfg: dict, gold: str | Path | None = None,
               segmentation_dir: str | Path | None = None, siblings: list[str] | None = None,
               feather_scores: str | Path | None = None, handwriting: str | Path | None = None) -> dict:
    """Writes labels.csv, review.csv, feathers.csv, labels.sqlite3 and
    summary.json from out/labels.jsonl (the latest result per slide), with
    feather IDs from a segmentation output folder when given, and scores
    against gold."""
    final = load_previous(out / "labels.jsonl")
    results = [final[Path(p).name] for p in chosen if Path(p).name in final]
    from src import slide_annotations

    for r in results:
        reclassify(r, cfg)            # the current reading rules, without asking the model again
    slide_annotations.annotate(results, siblings, slide_annotations.load_handwriting(handwriting))
    if segmentation_dir and Path(segmentation_dir).exists():
        attach_feathers_from_crops(results, segmentation_dir, siblings)
        from src import feather_duplicates

        if feather_scores and Path(feather_scores).exists():
            merge_feather_scores(results, feather_scores)
        dup_rows = feather_duplicates.apply_to_results(results, segmentation_dir)
        if dup_rows:
            feather_duplicates.write_csv(dup_rows, out / "feather_duplicates.csv")
    write_feathers_csv(results, out / "feathers.csv", out.name)
    write_feathers_csv(results, out / "feathers_usable.csv", out.name, usable_only=True)
    write_csv(results, out / "labels.csv")
    write_review_csv(results, out / "review.csv")
    write_sqlite(results, out / "labels.sqlite3", out.name, cfg["vlm"]["model"])
    summary = summarize(results)
    summary["slides_with_feather_ids"] = sum(bool(r.feathers) for r in results)
    summary["feathers"] = sum(len(segmented(r.feathers)) for r in results)
    summary["possible_missed_feathers"] = sum(f.get("segment_status") == "possible_missed"
                                              for r in results for f in r.feathers)
    summary["duplicate_feathers"] = sum(bool(f.get("duplicate_of")) for r in results for f in r.feathers)
    scored = [f["judge_score"] for r in results for f in r.feathers
              if isinstance(f.get("judge_score"), (int, float)) and not isinstance(f.get("judge_score"), bool)]
    summary["feathers_scored"] = len(scored)
    summary["feather_score_mean"] = round(sum(scored) / len(scored), 2) if scored else None
    summary["feathers_needing_review"] = sum(bool(f.get("needs_review")) for r in results for f in segmented(r.feathers))
    summary["fragments"] = sum(f.get("segment_status") == "fragment" for r in results for f in r.feathers)
    summary["feathers_usable"] = sum(row["usable"] == "yes" for r in results for row in feather_table_rows(r))
    summary["slides_with_label_problems"] = sum(bool(r.extra.get("label_problems")) for r in results)
    summary["fields_partly_visible"] = {n: sum(r.fields[n].status == "partly_visible" for r in results if n in r.fields)
                                        for n in FIELDS_ORDER}
    summary["fields_inferred_from_same_bird"] = {n: sum(r.fields[n].status == "inferred" for r in results if n in r.fields)
                                                 for n in ("sex", "species")}
    summary["slides_with_multiple_photos_same_year"] = sum(bool(r.extra.get("multiple_photos_same_year")) for r in results)
    summary["slides_handwritten"] = sum(r.extra.get("handwritten") == "yes" for r in results)
    if gold and Path(gold).exists():
        summary["gold"] = score_against_gold(results, gold, cfg)
        summary["gold_feather_count"] = score_feather_counts(results, gold)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_summary(summary: dict, out: Path | None = None, log=print) -> None:
    if out is not None:
        log(f"wrote {out}/labels.csv, review.csv, labels.jsonl, labels.sqlite3, summary.json")
    log(f"{summary['slides']} slides, {summary['needs_review']} need review, "
        f"{summary['slides_with_model_error']} with model errors, "
        f"{summary['mean_seconds_per_slide']} s per slide")
    for name, counts in summary["status_counts"].items():
        log(f"  {name:14} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if "gold" in summary:
        log("gold (exact match):")
        for name, s in summary["gold"].items():
            log(f"  {name:14} {s['correct']}/{s['n']}  acc={s['accuracy']}  "
                f"wrong_value={s['wrong_value']}  false_na={s['false_na']}")
        fc = summary.get("gold_feather_count")
        if fc:
            log(f"  feather count  {fc['correct']}/{fc['n']}  wrong: {fc['wrong']}")


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
    ap.add_argument("--workers", type=int, default=2, help="images in flight at once (local mode)")
    ap.add_argument("--distributed", action="store_true",
                    help="run on the cluster workers (Celery task extract_labels) instead of from here")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds to wait for the workers")
    ap.add_argument("--out", default=None, help="output dir (default data/label_runs/<timestamp>)")
    ap.add_argument("--gold", default=None, help="hand-labelled CSV to score against")
    ap.add_argument("--segmentation-dir", default=None,
                    help="segmentation output folder (e.g. data/sample_runs/<id>) for feather IDs")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    rc = load_notebook_config_from_env()
    chosen = _select(RemoteClusterIO(rc).list_remote_images(), args)
    if args.write_manifest:
        Path(args.write_manifest).write_text("\n".join(Path(p).name for p in chosen) + "\n")
        print(f"wrote {len(chosen)} names to {args.write_manifest}")
        return 0

    out = Path(args.out or REPO / "data" / "label_runs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))
    todo = pending_images(chosen, out)
    print(f"{len(chosen)} slides -> {out}  ({len(chosen) - len(todo)} already done, {len(todo)} to run)",
          flush=True)
    if args.distributed:
        run_distributed(todo, out, timeout=args.timeout)
    else:
        run_local(todo, out, cfg, args.fields.split(",") if args.fields else None, args.workers)
    all_names = [Path(p).name for p in RemoteClusterIO(rc).list_remote_images()] if args.segmentation_dir else None
    print_summary(finish_run(out, chosen, cfg, args.gold, args.segmentation_dir, all_names), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
