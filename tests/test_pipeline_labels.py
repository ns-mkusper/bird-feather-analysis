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


class StubProcessor:
    """Segmentation fails on purpose: labels must still be persisted."""
    calls = []

    def __init__(self):
        self.vlm_model_id = "stub"

    def extract_labels(self, image_path, output_dir):
        StubProcessor.calls.append(("labels", Path(image_path).name))
        r = lx.LabelResult(image_path, filename=lx.parse_filename(Path(image_path).name), node="stub-node")
        r.fields["individual_id"] = lx.FieldResult("A0012H", "readable", source="primary")
        r.fields["sex"] = lx.FieldResult(None, "model_error")
        return r

    def process_image(self, image_path, output_dir, profile="default"):
        StubProcessor.calls.append(("segment", Path(image_path).name))
        return StubResult(image_path, success=False)


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
    rows = db.execute("select run_id, node_id, file_id, individual_id, sex, sex_status "
                      "from specimen_labels order by file_id").fetchall()
    assert rows == [("t1", "node-a", "A0012 1996-im1058.jpg", "A0012H", "", "model_error"),
                    ("t1", "node-a", "BB17280 1992-im0412.jpg", "A0012H", "", "model_error")]
    assert db.execute("select count(*) from image_stats where success=0").fetchone()[0] == 2
    state = json.loads((tmp_path / "metrics" / "t1_node-a.json").read_text())
    assert state["labels_done"] == 2 and state["labels_model_error"] == 2
    assert state["labels_individual_id_readable"] == 2
    events = (tmp_path / "metrics" / "t1_node-a_events.jsonl").read_text().splitlines()
    assert json.loads(events[0])["labels"]["sex"] == "model_error"


def test_celery_process_image_returns_labels(monkeypatch, tmp_path):
    from src import celery_tasks as ct

    monkeypatch.setattr(ct, "_PROCESSOR", StubProcessor())
    out = ct.process_image.run("/x/A0012 1996-im1058.jpg", str(tmp_path))
    assert out["success"] is False and out["labels"]["fields"]["individual_id"]["value"] == "A0012H"


def test_distributed_driver_writes_label_table(tmp_path):
    from src.full_run_distributed import _write_labels

    r = StubProcessor().extract_labels("/x/A0012 1996-im1058.jpg", None)
    _write_labels([{"success": False, "labels": r.to_json()}, {"success": True, "labels": None}],
                  str(tmp_path / "labels"))
    rows = (tmp_path / "labels" / "labels.csv").read_text().splitlines()
    assert len(rows) == 2 and "A0012H" in rows[1]
    assert (tmp_path / "labels" / "labels.sqlite3").exists()
