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

    def ask(self, image, prompt, timeout=None, max_tokens=None):
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


def test_filename_conflict_and_fallback():
    cfg = lx.load_config()
    r = lx.LabelResult("x/A0012 1996-im1058.jpg", filename=lx.parse_filename("A0012 1996-im1058.jpg"))
    r.fields["bird_id"] = lx.FieldResult("A0013H", "readable", source="primary")
    r.fields["date"] = lx.FieldResult(None, "not_found")
    lx.apply_filename(r, cfg)
    assert r.fields["bird_id"].status == "conflict"
    d = r.fields["date"]
    assert (d.value, d.status, d.source) == ("1996", "filename_fallback", "filename")
    row = r.row()
    assert row["year"] == "1996" and row["month"] == "" and row["file_id"] == "A0012 1996-im1058.jpg"


def test_lenient_filename_never_used_as_fallback():
    cfg = lx.load_config()
    r = lx.LabelResult("AX868161996-img0194.jpg", filename=lx.parse_filename("AX868161996-img0194.jpg"))
    r.fields["bird_id"] = lx.FieldResult(None, "not_found")
    lx.apply_filename(r, cfg)
    assert r.fields["bird_id"].status == "not_found"
    assert r.row()["bird_id"] == "N/A"


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
    gold.write_text("image,bird_id,date\nA0012 1996-im1058.jpg,A0012H,N/A\nB.jpg,,1999-01-01\n")
    r = lx.LabelResult("A0012 1996-im1058.jpg")
    r.fields["bird_id"] = lx.FieldResult("A0012X", "readable")
    r.fields["date"] = lx.FieldResult("1996", "filename_fallback", source="filename")
    r2 = lx.LabelResult("B.jpg")
    r2.fields["date"] = lx.FieldResult(None, "not_found")
    s = lx.score_against_gold([r, r2], gold)
    assert s["bird_id"]["wrong_value"] == 1 and s["bird_id"]["correct"] == 0
    assert s["date"]["n"] == 2 and s["date"]["correct"] == 1 and s["date"]["false_na"] == 1


def test_extract_for_pipeline_caches_and_redoes_errors(tmp_path, monkeypatch):
    img = tmp_path / "A0012 1996-im1058.jpg"
    Image.new("RGB", (400, 460), "white").save(img)
    cfg = lx.load_config()
    n_fields = len(lx.FIELDS_ORDER)

    calls = {"n": 0}
    def fake_extract(path, cfg_, client, fields=None, evidence_dir=None):
        calls["n"] += 1
        r = lx.LabelResult(str(path), filename=lx.parse_filename(Path(path).name))
        status = "model_error" if calls["n"] == 1 else "readable"
        for f in lx.FIELDS_ORDER:
            r.fields[f] = lx.FieldResult("X" if status == "readable" else None, status)
        return r
    monkeypatch.setattr(lx, "extract", fake_extract)
    r1 = lx.extract_for_pipeline(img, tmp_path / "out", client=None, cfg=cfg)
    assert r1.has_model_error and (tmp_path / "out" / "labels" / f"{img.stem}.json").exists()
    r2 = lx.extract_for_pipeline(img, tmp_path / "out", client=None, cfg=cfg)   # error -> redone
    r3 = lx.extract_for_pipeline(img, tmp_path / "out", client=None, cfg=cfg)   # finished -> cached
    assert calls["n"] == 2 and not r2.has_model_error and r3.fields["sex"].value == "X"

    def boom(*a, **k):
        raise OSError("disk")
    monkeypatch.setattr(lx, "extract", boom)
    r4 = lx.extract_for_pipeline(img, tmp_path / "out2", client=None, cfg=cfg)
    assert all(f.status == "model_error" for f in r4.fields.values()) and len(r4.fields) == n_fields
    assert r4.row()["sex"] == ""          # crash is blank, never N/A
    m = lx.label_metrics(r4)
    assert m["labels_model_error"] == 1 and m["labels_sex_model_error"] == 1


def test_mlx_local_client_strips_thinking():
    seen = {}
    def gen(model, processor, prompt, image, max_tokens):
        seen["image"], seen["prompt"] = image[0], prompt
        class Out:
            text = '<think>hmm</think>{"value": "F", "visible": "yes", "confidence": 0.9}'
        return Out()
    c = lx.MLXLocalClient(gen, "m", "p", lambda s: "CHAT:" + s, "qwen")
    out = c.ask(Image.new("RGB", (10, 10)), "read it")
    assert out.strip().startswith("{") and seen["prompt"] == "CHAT:read it" and seen["image"].endswith(".jpg")


def test_sqlite_keeps_one_row_per_node(tmp_path):
    import sqlite3
    r = lx.LabelResult("A0012 1996-im1058.jpg", node="node-1")
    r.fields["sex"] = lx.FieldResult("F", "readable")
    conn = sqlite3.connect(tmp_path / "s.sqlite3")
    lx.upsert_labels(conn, [r], "run1")
    lx.upsert_labels(conn, [r], "run1")
    lx.upsert_labels(conn, [r], "run1", node_id="node-2")
    rows = conn.execute("select node_id, sex from specimen_labels order by node_id").fetchall()
    assert rows == [("node-1", "F"), ("node-2", "F")]


def test_sqlite_migrates_first_version_table(tmp_path):
    import sqlite3
    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE specimen_labels (run_id TEXT, file_id TEXT, sex TEXT, PRIMARY KEY (run_id, file_id))")
    conn.execute("INSERT INTO specimen_labels VALUES ('r', 'a.jpg', 'F')")
    conn.commit()
    r = lx.LabelResult("a.jpg", node="n1")
    r.fields["sex"] = lx.FieldResult("M", "readable")
    lx.upsert_labels(conn, [r], "r")
    assert conn.execute("select node_id, sex from specimen_labels").fetchall() == [("n1", "M")]
    assert conn.execute("select sex from specimen_labels_old1").fetchall() == [("F",)]


def test_malformed_answer_is_a_technical_failure_and_next_crop_is_tried(img):
    cfg = lx.load_config()
    spec = cfg["fields"]["sex"]
    degenerate = '{"value": "na' + " " * 60          # truncated, never closes
    f = lx.read_field(FakeClient([degenerate, _ans("F")]), img, "sex", spec, cfg)
    assert (f.value, f.status, f.source) == ("F", "readable", "expanded")
    assert "truncated JSON" in f.attempts[0].error
    n = len(lx.crop_plan(spec, cfg))
    f = lx.read_field(FakeClient([degenerate] * n), img, "sex", spec, cfg)
    assert f.status == "model_error"


def test_partly_visible_code_needs_confidence(img):
    """Sex and species (short, low-variance text) accept a partly visible reading from 0.6;
    age from 0.8; bird ID and date never (their feather IDs depend on them)."""
    cfg = lx.load_config()
    f = lx.read_field(FakeClient([_ans("F", "partly", 0.5)]), img, "sex", cfg["fields"]["sex"], cfg)
    assert f.status == "partly_visible" and "partly visible" in f.note and len(f.attempts) == 1
    f = lx.read_field(FakeClient([_ans("M", "partly", 0.6)]), img, "sex", cfg["fields"]["sex"], cfg)
    assert f.status == "readable" and "partly visible (accepted)" in f.note
    f = lx.read_field(FakeClient([_ans("2K", "partly", 0.7)]), img, "age", cfg["fields"]["age"], cfg)
    assert f.status == "partly_visible"
    f = lx.read_field(FakeClient([_ans("BZ11933H", "partly", 0.95)]), img, "bird_id", cfg["fields"]["bird_id"], cfg)
    assert f.status == "partly_visible" and f.value == "BZ11933H"     # value kept, not trusted


def test_reclassify_applies_the_current_rules_without_the_model():
    cfg = lx.load_config()
    a = lx.Attempt(crop="primary", region=[0, 0, 1, 1], value="BZ11933H", visible="partly", confidence=0.85,
                   raw="", seconds=1.0)
    r = lx.LabelResult("BZ11933 2007-im2197.jpg")
    r.fields["bird_id"] = lx.FieldResult("BZ11933H", "readable", 0.85, source="primary",
                                         note="partly visible, confidence 0.85", attempts=[a])
    assert lx.reclassify(r, cfg) == 1 and r.fields["bird_id"].status == "partly_visible" and r.needs_review


def test_feather_ids_from_label_and_filename_backup():
    r = lx.LabelResult("x/A0012 1996-im1058.jpg", filename=lx.parse_filename("A0012 1996-im1058.jpg"))
    r.fields["bird_id"] = lx.FieldResult("A0012H", "readable")
    r.fields["date"] = lx.FieldResult("1996-06-04", "readable")
    ids = lx.assign_feather_ids(r, ["s_Feather_1.jpg", "s_Feather_2.jpg", "s_Feather_3.jpg"])
    assert [f["feather_id"] for f in ids] == ["A0012H_1996_A", "A0012H_1996_B", "A0012H_1996_C"]
    assert ids[0]["bird_id_source"] == "label" and r.row()["feather_ids"].count(";") == 2
    # label ID not readable -> leading ID of the file name; covered date -> file name year
    r.fields["bird_id"] = lx.FieldResult(None, "not_found")
    r.fields["date"] = lx.FieldResult(None, "model_error")
    f = lx.assign_feather_ids(r, ["s_Feather_1.jpg"])[0]
    assert f["feather_id"] == "A0012_1996_A" and f["bird_id_source"] == "filename" and f["year_source"] == "filename"


def test_attach_feathers_from_segmentation_folder(tmp_path):
    new, old = tmp_path / "new", tmp_path / "old"
    new.mkdir(); old.mkdir()
    for letter in ("B", "A", "C"):
        (new / f"A0012_1996_Feather_{letter}.jpg").write_bytes(b"x")
    for n in (2, 1, 3):
        (old / f"A0012 1996-im1058_A0012_1996_Feather_{n}.jpg").write_bytes(b"x")
    for folder, first in ((new, "A0012_1996_Feather_A.jpg"), (old, "A0012 1996-im1058_A0012_1996_Feather_1.jpg")):
        r = lx.LabelResult("A0012 1996-im1058.jpg", filename=lx.parse_filename("A0012 1996-im1058.jpg"))
        r.fields["bird_id"] = lx.FieldResult("A0012H", "readable")
        r.fields["date"] = lx.FieldResult("1996-06-04", "readable")
        assert lx.attach_feathers_from_crops([r], folder) == 1
        assert [f["seq"] for f in r.feathers] == ["A", "B", "C"] and r.feathers[0]["crop"] == first


@pytest.mark.parametrize("raw", [
    '{"value": "na\\u00a0\\u00a0", "visible": "yes", "confidence": 1}',   # escape sequence
    '{"value": "na\u00a0\u00a0", "visible": "yes", "confidence": 1}',      # decoded non-breaking spaces
    '{"value": "\u00c5K", "visible": "yes", "confidence": 1}',             # non-Latin letter
])
def test_non_plain_text_is_a_model_error(img, raw):
    cfg = lx.load_config()
    spec = cfg["fields"]["sex"]
    n = len(lx.crop_plan(spec, cfg))
    f = lx.read_field(FakeClient([raw] * n), img, "sex", spec, cfg)
    assert f.status == "model_error" and f.value is None


def test_slide_problems_ranks_wrong_and_missed_first():
    cfg = lx.load_config()
    r = lx.LabelResult("a.jpg")
    r.fields["bird_id"] = lx.FieldResult("A0012H", "readable")
    r.fields["sex"] = lx.FieldResult("F", "readable")            # gold N/A -> wrong
    r.fields["age"] = lx.FieldResult(None, "not_found")          # gold 2K -> missed
    r.fields["species"] = lx.FieldResult("HAF", "needs_review")
    r.feathers = [{"feather_id": "x"}] * 3
    gold = {"bird_id": "A0012H", "sex": "N/A", "age": "2K", "species": "", "feather_count": "4"}
    kinds = [(p["field"], p["kind"]) for p in lx.slide_problems(r, gold, cfg)]
    assert kinds == [("sex", "wrong"), ("age", "missed"), ("feathers", "count"), ("species", "needs_review")]


def test_feather_ids_use_full_date_for_same_year_slides(tmp_path):
    sib = ["BB17692 1998-im0513.jpg", "BB17692 1998-im0514.jpg", "A0012 1996-im1058.jpg"]
    def res(name, date):
        r = lx.LabelResult(name, filename=lx.parse_filename(name))
        r.fields["bird_id"] = lx.FieldResult("BB17692H", "readable")
        r.fields["date"] = lx.FieldResult(date, "readable" if date else "not_found")
        return r
    # new crop names carry the period
    r = res("BB17692 1998-im0513.jpg", "1998-06-12")
    ids = lx.assign_feather_ids(r, ["BB17692_19980612_Feather_A.jpg", "BB17692_19980612_Feather_B.jpg"])
    assert [f["feather_id"] for f in ids] == ["BB17692H_19980612_A", "BB17692H_19980612_B"]
    # older crop names: period from siblings + label date, image number if the date is unknown
    for n in (1, 2):
        (tmp_path / f"BB17692 1998-im0514_BB17692_1998_Feather_{n}.jpg").write_bytes(b"x")
        (tmp_path / f"BB17692 1998-im0513_BB17692_1998_Feather_{n}.jpg").write_bytes(b"x")
    a, b = res("BB17692 1998-im0513.jpg", "1998-06-12"), res("BB17692 1998-im0514.jpg", None)
    lx.attach_feathers_from_crops([a, b], tmp_path, sib)
    # the sibling's date is unknown, so it might be the same day -> image number
    assert a.feathers[0]["feather_id"] == "BB17692H_19980612-im0513_A"
    assert b.feathers[1]["feather_id"] == "BB17692H_1998-im0514_B"


def test_handwriting_check_flags_any_region_with_handwriting(img):
    class Client:
        def __init__(self, answers):
            self.answers = list(answers)

        def ask(self, image, prompt, timeout=None, max_tokens=None):
            return self.answers.pop(0)

    cfg = lx.load_config()
    h = {**cfg["handwriting"], "enabled": True}
    out = lx.check_handwriting(Client(['{"handwritten": false, "fields": [], "notes": "ok"}',
                                       '{"handwritten": true, "fields": ["date"], "notes": "date overwritten"}']),
                               img, h, cfg)
    assert out["handwritten"] == "yes" and out["handwritten_fields"] == "date" and "overwritten" in out["label_note"]
    out = lx.check_handwriting(Client(['{"handwritten": false}', "garbled"]), img, h, cfg)
    assert out["handwritten"] == ""          # one region unanswered: unknown, not 'no'
