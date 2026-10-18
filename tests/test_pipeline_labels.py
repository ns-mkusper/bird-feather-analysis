"""Label table wiring in the distributed pipeline, with a stub processor
(no SAM/MLX models needed)."""
import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from src import label_extract as lx

feather_pkgs = pytest.importorskip("redis")


class StubResult:
    def __init__(self, path, success):
        self.image_path, self.success = path, success
        self.reason = "" if success else "no_boxes"
        self.feathers_saved = 3 if success else 0
        self.vlm_score = None
        self.metadata_source = "filename"
        self.bird_id, self.date = "A0012", "1996"
        self.vlm_all_feathers_covered = None
        self.vlm_background_leakage_detected = None
        self.vlm_green_boxes_grouped_feathers = None
        self.vlm_notes = ""
        self.processing_profile = "default"
        self.feather_ids = []


class StubProcessor:
    """Segmentation fails on purpose: labels must still be persisted."""
    calls = []

    def __init__(self):
        self.vlm_model_id = "stub"

    def extract_labels(self, image_path, output_dir):
        StubProcessor.calls.append(("labels", Path(image_path).name))
        r = lx.LabelResult(image_path, filename=lx.parse_filename(Path(image_path).name), node="stub-node")
        r.fields["bird_id"] = lx.FieldResult("A0012H", "readable", source="primary")
        r.fields["sex"] = lx.FieldResult(None, "model_error")
        return r

    def process_image(self, image_path, output_dir, profile="default", labels=None):
        StubProcessor.calls.append(("segment", Path(image_path).name))
        ok = Path(image_path).name.startswith("BB")   # first slide fails, second gets 2 feathers
        res = StubResult(image_path, success=ok)
        if ok and labels is not None:
            res.feather_ids = lx.assign_feather_ids(labels, ["c_Feather_1.jpg", "c_Feather_2.jpg"])
        return res


def test_sharded_run_persists_labels_when_segmentation_fails(tmp_path, monkeypatch):
    from src import full_run_sharded as frs

    raw = tmp_path / "raw"
    raw.mkdir()
    for n in ("A0012 1996-im1058.jpg", "BB17280 1992-im0412.jpg"):
        Image.new("RGB", (40, 46)).save(raw / n)
    monkeypatch.setattr(frs, "FeatherProcessor", StubProcessor)
    monkeypatch.setattr(frs, "_connect_metrics_redis", lambda: None)
    monkeypatch.setenv("FEATHER_RUN_ID", "t1")
    monkeypatch.setenv("FEATHER_NODE_ID", "node-a")
    monkeypatch.setenv("FEATHER_METRICS_DIR", str(tmp_path / "metrics"))
    frs.run_shard(str(raw), str(tmp_path / "out"), 0, 1, 0, None)

    assert StubProcessor.calls[:2] == [("labels", "A0012 1996-im1058.jpg"), ("segment", "A0012 1996-im1058.jpg")]
    db = sqlite3.connect(tmp_path / "metrics" / "run_stats.sqlite3")
    rows = db.execute("select run_id, node_id, file_id, bird_id, sex, sex_status "
                      "from specimen_labels order by file_id").fetchall()
    assert rows == [("t1", "node-a", "A0012 1996-im1058.jpg", "A0012H", "", "model_error"),
                    ("t1", "node-a", "BB17280 1992-im0412.jpg", "A0012H", "", "model_error")]
    assert db.execute("select count(*) from image_stats where success=0").fetchone()[0] == 1
    feathers = db.execute("select file_id, seq, feather_id, year_source from specimen_feathers order by seq").fetchall()
    assert feathers == [("BB17280 1992-im0412.jpg", "A", "A0012H_1992_A", "filename"),
                        ("BB17280 1992-im0412.jpg", "B", "A0012H_1992_B", "filename")]
    assert db.execute("select feather_ids from specimen_labels where file_id like 'BB%'").fetchone()[0] \
        == "A0012H_1992_A;A0012H_1992_B"
    state = json.loads((tmp_path / "metrics" / "t1_node-a.json").read_text())
    assert state["labels_done"] == 2 and state["labels_model_error"] == 2
    assert state["labels_bird_id_readable"] == 2
    events = (tmp_path / "metrics" / "t1_node-a_events.jsonl").read_text().splitlines()
    assert json.loads(events[0])["labels"]["sex"] == "model_error"


def test_celery_process_image_returns_labels(monkeypatch, tmp_path):
    from src import celery_tasks as ct

    monkeypatch.setattr(ct, "_PROCESSOR", StubProcessor())
    out = ct.process_image.run("/x/A0012 1996-im1058.jpg", str(tmp_path))
    assert out["success"] is False and out["labels"]["fields"]["bird_id"]["value"] == "A0012H"


def test_distributed_driver_writes_label_table(tmp_path):
    from src.full_run_distributed import _write_labels

    r = StubProcessor().extract_labels("/x/A0012 1996-im1058.jpg", None)
    _write_labels([{"success": False, "labels": r.to_json()}, {"success": True, "labels": None}],
                  str(tmp_path / "labels"))
    rows = (tmp_path / "labels" / "labels.csv").read_text().splitlines()
    assert len(rows) == 2 and "A0012H" in rows[1]
    assert (tmp_path / "labels" / "labels.sqlite3").exists()


def test_process_image_writes_new_names_left_to_right(tmp_path):
    """Real FeatherProcessor.process_image with fake masks instead of SAM."""
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    slide = raw / "CE01168 2008-im2346.jpg"
    Image.new("RGB", (300, 200), "white").save(slide)
    (raw / "A0012 1996-im1058.jpg").write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    # leftovers from an earlier attempt (old and new names) must be replaced
    (out / "CE01168 2008-im2346_CE01168_2008_Feather_7.jpg").write_bytes(b"x")
    (out / "CE01168_2008_Feather_E.jpg").write_bytes(b"x")

    def masks_at(xs):
        ms = []
        for x in xs:
            m = np.zeros((200, 300), dtype=np.float32)
            m[20:180, x:x + 30] = 1
            ms.append(m)
        return ms

    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm, p.vlm_scoring_enabled, p.vlm_metadata_enabled = "sam3", False, False, False
    p._segment_sam3 = lambda *a, **k: ([[200, 20, 230, 180], [20, 20, 50, 180], [110, 20, 140, 180]],
                                       masks_at([200, 20, 110]), "")
    labels = lx.LabelResult(str(slide), filename=lx.parse_filename(slide.name))
    labels.fields["bird_id"] = lx.FieldResult("CE01168V", "readable")
    labels.fields["date"] = lx.FieldResult("2008-05-22", "readable")
    res = p.process_image(str(slide), str(out), labels=labels)

    assert res.success and res.feathers_saved == 3
    names = sorted(f.name for f in out.iterdir())
    assert names == ["CE01168 2008_BoundingBoxes.jpg", "CE01168_2008_Feather_A.jpg",
                     "CE01168_2008_Feather_B.jpg", "CE01168_2008_Feather_C.jpg"]
    assert [(f["feather_id"], f["crop"]) for f in res.feather_ids] == [
        ("CE01168V_2008_A", "CE01168_2008_Feather_A.jpg"), ("CE01168V_2008_B", "CE01168_2008_Feather_B.jpg"),
        ("CE01168V_2008_C", "CE01168_2008_Feather_C.jpg")]
    # leftmost mask (x=20) became crop A
    import cv2
    a = cv2.imread(str(out / "CE01168_2008_Feather_A.jpg"))
    assert a is not None and a.shape[1] <= 31


class DateClient:
    """Fake VLM: answers every date crop with a fixed date per slide image size."""
    model_name = "fake"

    def __init__(self, date):
        self.date = date

    def ask(self, image, prompt, timeout=None):
        return json.dumps({"value": self.date, "visible": "yes", "confidence": 0.99})


@pytest.mark.parametrize("sibling_date,expect_prefix", [
    ("1998-06-20", "BB17692_19980612"),           # same year, other day -> date only
    ("1998-06-12", "BB17692_19980612-im0513"),    # same day -> date + image number
])
def test_process_image_names_same_year_slides_by_label_date(tmp_path, sibling_date, expect_prefix):
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    for n in ("BB17692 1998-im0513.jpg", "BB17692 1998-im0514.jpg"):
        Image.new("RGB", (300, 200), "white").save(raw / n)
    out = tmp_path / "out"
    m = np.zeros((200, 300), dtype=np.float32)
    m[20:180, 40:70] = 1
    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm, p.vlm_scoring_enabled, p.vlm_metadata_enabled = "sam3", False, False, False
    p._segment_sam3 = lambda *a, **k: ([[40, 20, 70, 180]], [m], "")
    p._label_client = DateClient(sibling_date)          # the sibling's date reading
    name = "BB17692 1998-im0513.jpg"
    labels = lx.LabelResult(str(raw / name), filename=lx.parse_filename(name))
    labels.fields["bird_id"] = lx.FieldResult("BB17692H", "readable")
    labels.fields["date"] = lx.FieldResult("1998-06-12", "readable")
    res = p.process_image(str(raw / name), str(out), labels=labels)
    assert res.success, res.reason
    assert (out / f"{expect_prefix}_Feather_A.jpg").exists()
    assert res.feather_ids[0]["feather_id"] == f"BB17692H_{expect_prefix.split('_', 1)[1]}_A"
    assert (out / "labels" / "BB17692 1998-im0514.date.json").exists()   # sibling date cached


def test_process_image_unreadable_own_date_uses_image_number(tmp_path):
    import numpy as np
    from src.feather_processing import FeatherProcessor

    raw = tmp_path / "raw"
    raw.mkdir()
    for n in ("BB17692 1998-im0513.jpg", "BB17692 1998-im0514.jpg"):
        Image.new("RGB", (300, 200), "white").save(raw / n)
    m = np.zeros((200, 300), dtype=np.float32)
    m[20:180, 40:70] = 1
    p = object.__new__(FeatherProcessor)
    p.segmenter, p.has_vlm, p.vlm_scoring_enabled, p.vlm_metadata_enabled = "sam3", False, False, False
    p._segment_sam3 = lambda *a, **k: ([[40, 20, 70, 180]], [m], "")
    name = "BB17692 1998-im0514.jpg"
    labels = lx.LabelResult(str(raw / name), filename=lx.parse_filename(name))
    labels.fields["bird_id"] = lx.FieldResult("BB17692H", "readable")
    labels.fields["date"] = lx.FieldResult(None, "not_found")
    res = p.process_image(str(raw / name), str(tmp_path / "out"), labels=labels)
    assert res.feather_ids[0]["feather_id"] == "BB17692H_1998-im0514_A"
    assert (tmp_path / "out" / "BB17692_1998-im0514_Feather_A.jpg").exists()
