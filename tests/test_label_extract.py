"""Offline tests for src/label_extract.py (no VLM server needed)."""
import json
from pathlib import Path

import pytest
from PIL import Image

from src import label_extract as lx

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "raw_cache" / "A0012 1996-im1058.jpg"


@pytest.mark.parametrize("name,want", [
    ("A0012 1996-im1058.jpg", ("A0012", "1996", "standard")),
    ("BB17080 1993-img0372.jpg", ("BB17080", "1993", "standard")),
    ("-BD31906 1999-im0892.jpg", ("BD31906", "1999", "lenient")),
    ("AX868161996-img0194.jpg", ("AX86816", "1996", "lenient")),
    ("BX40 368 2004-im1800.jpg", ("BX40368", "2004", "lenient")),
    ("BX402982003-im1778.jpg", ("BX40298", "2003", "lenient")),
    ("D030813 2003-im2333.jpg", ("D030813", "2003", "lenient")),
    ("scan.jpg", (None, None, "unparsed")),
])
def test_parse_filename(name, want):
    d = lx.parse_filename(name)
    assert (d["id"], d["year"], d["parse"]) == want


def test_split_date_and_export():
    assert lx.split_date("1996-06-04") == {"year": "1996", "month": "06", "day": "04"}
    assert lx.split_date("1996-13-04")["month"] is None
    assert lx.export_value(None, "not_found") == "N/A"
    assert lx.export_value(None, "model_error") == ""   # never N/A for a technical failure


class FakeClient:
    """Answers by crop name order: a list of replies (str) or exceptions."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def ask(self, image, prompt, timeout=None):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _ans(value, visible="yes", conf=0.9):
    return json.dumps({"value": value, "visible": visible, "confidence": conf})


@pytest.fixture
def img():
    return Image.new("RGB", (400, 460), "white")


def test_fallback_to_expanded_crop(img):
    cfg = lx.load_config()
    spec = cfg["fields"]["sex"]
    c = FakeClient([_ans(None, "no"), _ans("F")])
    f = lx.read_field(c, img, "sex", spec, cfg)
    assert (f.value, f.status, f.source) == ("F", "readable", "expanded")


def test_not_found_after_all_crops(img):
    cfg = lx.load_config()
    spec = cfg["fields"]["sex"]
    n = len(lx.crop_plan(spec, cfg))
    f = lx.read_field(FakeClient([_ans(None, "no")] * n), img, "sex", spec, cfg)
    assert f.status == "not_found" and len(f.attempts) == n


def test_model_error_is_not_not_found(img):
    cfg = lx.load_config()
    spec = cfg["fields"]["sex"]
    n = len(lx.crop_plan(spec, cfg))
    f = lx.read_field(FakeClient([TimeoutError("x")] * n), img, "sex", spec, cfg)
    assert f.status == "model_error"
    # mixed: some crops said absent, one failed -> still model_error
    f = lx.read_field(FakeClient([_ans(None, "no")] * (n - 1) + [TimeoutError("x")]), img, "sex", spec, cfg)
    assert f.status == "model_error"


def test_verify_disagreement_goes_to_review(img):
    cfg = lx.load_config()
    spec = cfg["fields"]["date"]
    f = lx.read_field(FakeClient([_ans("1996-06-04"), _ans("1999-06-06")]), img, "date", spec, cfg)
    assert f.status == "needs_review" and "disagree" in f.note
    f = lx.read_field(FakeClient([_ans("1996-06-04"), _ans("1996-06-04")]), img, "date", spec, cfg)
    assert f.status == "readable"


def test_feather_partial_stops_chain(img):
    cfg = lx.load_config()
    f = lx.read_field(FakeClient([_ans("RM-H pga SVFLU-...", "partly")]), img, "feather",
                      cfg["fields"]["feather"], cfg)
    assert f.status == "partial" and len(f.attempts) == 1


def test_filename_conflict_and_fallback():
    cfg = lx.load_config()
    r = lx.LabelResult("x/A0012 1996-im1058.jpg", filename=lx.parse_filename("A0012 1996-im1058.jpg"))
    r.fields["individual_id"] = lx.FieldResult("A0013H", "readable", source="primary")
    r.fields["date"] = lx.FieldResult(None, "not_found")
    lx.apply_filename(r, cfg)
    assert r.fields["individual_id"].status == "conflict"
    d = r.fields["date"]
    assert (d.value, d.status, d.source) == ("1996", "filename_fallback", "filename")
    row = r.row()
    assert row["year"] == "1996" and row["month"] == "" and row["file_id"] == "A0012 1996-im1058.jpg"


def test_lenient_filename_never_used_as_fallback():
    cfg = lx.load_config()
    r = lx.LabelResult("AX868161996-img0194.jpg", filename=lx.parse_filename("AX868161996-img0194.jpg"))
    r.fields["individual_id"] = lx.FieldResult(None, "not_found")
    lx.apply_filename(r, cfg)
    assert r.fields["individual_id"].status == "not_found"
    assert r.row()["individual_id"] == "N/A"


def test_json_roundtrip_and_sqlite_upsert(tmp_path):
    r = lx.LabelResult("A0012 1996-im1058.jpg", filename=lx.parse_filename("A0012 1996-im1058.jpg"))
    r.fields["date"] = lx.FieldResult("1996-06-04", "readable", source="primary",
                                      attempts=[lx.Attempt("primary", [0, 0, 1, 1], "1996-06-04", "yes", 1.0, "{}", 1.0)])
    back = lx.LabelResult.from_json(json.loads(json.dumps(r.to_json())))
    assert back.fields["date"].attempts[0].crop == "primary"
    db = tmp_path / "l.sqlite3"
    lx.write_sqlite([r], db, "run1", "m")
    lx.write_sqlite([r], db, "run1", "m")
    import sqlite3
    rows = sqlite3.connect(db).execute("select year, month, day, date_status from specimen_labels").fetchall()
    assert rows == [("1996", "06", "04", "readable")]


def test_stratified_sample_spreads_prefixes_and_years():
    names = [f"/r/{p}{i:05d} {y} -im{i}.jpg".replace(" -im", "-im")
             for i, (p, y) in enumerate([(p, y) for p in ("A", "BB", "BD", "CE") for y in range(1990, 2000)])]
    names += ["/r/A0012 1996-im1058.jpg", "/r/AX868161996-img0194.jpg"]
    s = lx.stratified_sample(names, 12, seed=0)
    assert len(s) == 12 and s[0].endswith("A0012 1996-im1058.jpg")
    assert "/r/AX868161996-img0194.jpg" in s
    assert len({lx.parse_filename(Path(p).name)["year"] for p in s}) >= 8


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample slide not cached")
def test_crop_plan_regions_on_sample():
    cfg = lx.load_config()
    im = Image.open(SAMPLE)
    for name, spec in cfg["fields"].items():
        for crop_name, reg in lx.crop_plan(spec, cfg):
            assert 0 <= reg[0] < reg[2] <= 1 and 0 <= reg[1] < reg[3] <= 1, (name, crop_name)
            assert lx.crop_field(im, reg, 1024).size[0] > 0


def test_gold_scoring(tmp_path):
    gold = tmp_path / "g.csv"
    gold.write_text("image,individual_id,date\nA0012 1996-im1058.jpg,A0012H,N/A\nB.jpg,,1999-01-01\n")
    r = lx.LabelResult("A0012 1996-im1058.jpg")
    r.fields["individual_id"] = lx.FieldResult("A0012X", "readable")
    r.fields["date"] = lx.FieldResult("1996", "filename_fallback", source="filename")
    r2 = lx.LabelResult("B.jpg")
    r2.fields["date"] = lx.FieldResult(None, "not_found")
    s = lx.score_against_gold([r, r2], gold)
    assert s["individual_id"]["wrong_value"] == 1 and s["individual_id"]["correct"] == 0
    assert s["date"]["n"] == 2 and s["date"]["correct"] == 1 and s["date"]["false_na"] == 1
