"""Per-feather checks and the missed-feather inventory (no model)."""
import json

import numpy as np

from src import feather_inventory as fi
from src import feather_judge as fj


def _answer(**kw):
    d = {"outline": {"left": "complete", "right": "complete", "top": "complete", "bottom": "complete"},
         "holes_or_gaps": False, "single_feather": True, "fragment": False, "other_feather_inside": False,
         "other_material_inside": False, "score": 9, "notes": "ok"}
    for k, v in kw.items():
        if k in ("left", "right", "top", "bottom"):
            d["outline"][k] = v
        else:
            d[k] = v
    return json.dumps(d)


def test_score_is_derived_from_the_checks():
    assert fj.parse(_answer())["score"] == 10
    r = fj.parse(_answer(left="cut", right="cut"))
    assert r["score"] == 6 and r["sides_cut"] == ["left", "right"] and not r["segment_ok"]
    assert fj.parse(_answer(holes_or_gaps=True))["score"] == 7
    assert fj.parse(_answer(fragment=True, single_feather=False))["score"] == 2
    assert fj.parse(_answer(score=3))["model_score"] == 3.0       # model's own number kept separately


def test_bad_answers_are_model_errors():
    assert fj.parse("the feather looks fine")["status"] == "model_error"
    assert fj.parse(_answer(left="partial"))["status"] == "model_error"
    assert fj.parse(_answer(holes_or_gaps="maybe"))["status"] == "model_error"
    assert fj.parse('{"outline": {"left": "complete\\u00a0"}}')["status"] == "model_error"


def test_measured_holes_and_split_masks():
    m = np.zeros((200, 100), bool)
    m[20:180, 30:70] = True
    assert fj.mask_metrics(m) == {"hole_frac": 0.0, "holes": 0, "mask_parts": 1}
    holed = m.copy()
    holed[90:110, 45:55] = False                         # a hole in the middle
    r = fj.mask_metrics(holed)
    assert r["holes"] == 1 and r["hole_frac"] > 0.02
    split = m.copy()
    split[95:105, :] = False                             # a gap across the feather
    assert fj.mask_metrics(split)["mask_parts"] == 2


class _Client:
    def __init__(self, text):
        self.text = text

    def ask(self, image, prompt):
        return self.text


def test_measured_hole_overrides_a_clean_model_answer():
    slide = np.full((200, 100, 3), 200, np.uint8)
    m = np.zeros((200, 100), bool)
    m[20:180, 30:70] = True
    m[90:110, 45:55] = False
    res, _ = fj.judge(_Client(_answer()), slide, m, None, {"prompt": "x", "panel_height": 200})
    assert res["holes_or_gaps"] is True and res["score"] == 7
    assert fj.needs_review({f"judge_{k}": v for k, v in res.items()}) == "holes or gaps"


def test_inventory_merge_places_missed_feathers_in_sequence():
    inv = fi.parse(json.dumps({"total_feathers": 4, "missed": [{"between": [2, 3], "x_center": 0.6,
                                                                "partly_hidden": True}]}), 3)
    items, status, note = fi.merge([100.0, 300.0, 700.0], inv, 1000)
    assert [it["kind"] for it in items] == ["segmented", "segmented", "missed", "segmented"]
    assert status == "missed" and note == ""
    # a missed feather at the far right end
    inv = fi.parse(json.dumps({"total_feathers": 3, "missed": [{"between": [2, None], "x_center": 0.9}]}), 2)
    items, status, _ = fi.merge([100.0, 300.0], inv, 1000)
    assert [it["kind"] for it in items] == ["segmented", "segmented", "missed"] and status == "missed"


def test_inventory_contradictions_are_flagged_not_trusted():
    # total does not add up
    inv = fi.parse(json.dumps({"total_feathers": 5, "missed": []}), 3)
    _, status, note = fi.merge([1.0, 2.0, 3.0], inv, 100)
    assert status == "inconsistent" and "counted 5" in note
    # x_center outside the named neighbours: placed between them, noted
    inv = fi.parse(json.dumps({"total_feathers": 3, "missed": [{"between": [1, 2], "x_center": 0.95}]}), 2)
    items, status, note = fi.merge([100.0, 300.0], inv, 1000)
    assert items[1]["kind"] == "missed" and 100 < items[1]["x"] < 300 and "outside neighbours" in note
    # unknown neighbour number
    inv = fi.parse(json.dumps({"total_feathers": 3, "missed": [{"between": [7, None], "x_center": 0.5}]}), 2)
    assert fi.merge([100.0, 300.0], inv, 1000)[1] == "inconsistent"
    # failed request: letters stay as segmented
    items, status, _ = fi.merge([100.0, 300.0], {"status": "model_error", "error": "timeout"}, 1000)
    assert status == "model_error" and [it["kind"] for it in items] == ["segmented", "segmented"]
