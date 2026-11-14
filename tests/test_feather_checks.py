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

    def ask(self, image, prompt, max_tokens=None):
        return self.text


def test_measured_hole_overrides_a_clean_model_answer():
    slide = np.full((200, 100, 3), 200, np.uint8)
    m = np.zeros((200, 100), bool)
    m[20:180, 30:70] = True
    m[90:110, 45:55] = False
    res, _ = fj.judge(_Client(_answer()), slide, m, None, {"prompt": "x", "panel_height": 200})
    assert res["holes_or_gaps"] is True and res["score"] == 7
    assert fj.needs_review({f"judge_{k}": v for k, v in res.items()}) == "holes or gaps"


def test_small_measured_hole_is_recorded_but_not_a_failure():
    """A pocket under fail_hole_frac (natural vane split) is measured, not flagged."""
    slide = np.full((200, 100, 3), 200, np.uint8)
    m = np.zeros((200, 100), bool)
    m[20:180, 30:70] = True                  # 6400 px
    m[100:104, 48:52] = False                # 16 px = 0.25 % of the feather
    res, _ = fj.judge(_Client(_answer()), slide, m, None, {"prompt": "x", "panel_height": 200})
    assert res["holes"] == 1 and 0 < res["hole_frac"] < 0.02
    assert res["holes_or_gaps"] is False and res["score"] == 10
    assert fj.needs_review({f"judge_{k}": v for k, v in res.items()}) == ""
    # the model's own "holes" answer still counts
    res2, _ = fj.judge(_Client(_answer(holes_or_gaps=True)), slide, m, None, {"prompt": "x", "panel_height": 200})
    assert res2["holes_or_gaps"] is True


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


class _SeqClient:
    """Answers in order; an Exception item is raised."""
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def ask(self, image, prompt, max_tokens=None):
        self.calls += 1
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def _mask():
    m = np.zeros((200, 100), bool)
    m[20:180, 30:70] = True
    return m


CFG2 = {"prompt": "x", "panel_height": 200, "second_look": {"enabled": True, "scale": 0.8}}


def test_second_look_confirms_or_moves_failures_to_uncertain():
    slide = np.full((200, 100, 3), 200, np.uint8)
    c = _SeqClient(_answer(bottom="cut", other_material_inside=True), _answer(bottom="cut"))
    res, _ = fj.judge(c, slide, _mask(), None, CFG2)
    rec = {f"judge_{k}": v for k, v in res.items()}
    assert c.calls == 2 and res["sides_cut"] == ["bottom"] and res["other_material_inside"] is False
    assert res["uncertain_checks"] == ["other_material_inside"] and res["score"] == 8
    assert fj.needs_review(rec) == "outline cut: bottom"
    assert fj.uncertain_review(rec) == "non-feather material inside"


def test_second_look_only_when_something_failed_and_never_clears_on_an_error():
    slide = np.full((200, 100, 3), 200, np.uint8)
    c = _SeqClient(_answer())
    res, _ = fj.judge(c, slide, _mask(), None, CFG2)
    assert c.calls == 1 and res["uncertain_checks"] == [] and res["score"] == 10
    c = _SeqClient(_answer(left="cut"), TimeoutError("server"))
    res, _ = fj.judge(c, slide, _mask(), None, CFG2)
    assert res["sides_cut"] == ["left"] and res["uncertain_checks"] == [] and "failed" in res["second_look"]
    # switched off: one look, as before
    c = _SeqClient(_answer(left="cut"))
    res, _ = fj.judge(c, slide, _mask(), None, {"prompt": "x", "panel_height": 200})
    assert c.calls == 1 and res["sides_cut"] == ["left"]


def test_straight_outline_and_hidden_level():
    import numpy as np
    from src import feather_judge as fj

    ragged = np.zeros((400, 200), bool)
    yy, xx = np.mgrid[0:400, 0:200]
    wobble = 6 * np.sin(yy / 4.0)                                                   # barbs: a ragged edge
    ragged[((xx - 100 + wobble) / 60.0) ** 2 + ((yy - 200) / 180.0) ** 2 <= 1] = True
    cut = ragged.copy()
    cut[:, 110:] = False                                                            # a board edge down its length
    s0, s1 = fj.straight_outline(ragged), fj.straight_outline(cut)
    assert s0["total"] < 0.2 and s1["along_longest"] >= 0.33          # a board edge along the whole side
    across = ragged.copy()
    across[:120, :] = False                                           # a board edge across the tip
    s2 = fj.straight_outline(across)
    assert s2["across"] >= 0.2
    cfg = {"hidden": {}}
    assert fj.hidden_level(s0, 1.0, cfg) == "none"
    assert fj.hidden_level(s1, 1.0, cfg) == "part" and fj.hidden_level(s2, 1.0, cfg) == "part"
    assert fj.hidden_level(s1, 0.3, cfg) == "most"
    vane = {"total": 0.5, "longest": 0.29, "across": 0.0, "along_longest": 0.29}   # straight outer vane edges
    assert fj.hidden_level(vane, 0.55, cfg) == "none"

def test_hidden_along_needs_a_cover_edge_on_the_slide():
    """A long straight run along the feather is 'hidden' only when the slide's edge goes on past
    the feather (a board lying on it), not when it is the feather's own straight vane edge."""
    import numpy as np
    from src import feather_judge as fj

    yy, xx = np.mgrid[0:600, 0:400]
    feather = ((xx - 200 + 6 * np.sin(yy / 4.0)) / 60.0) ** 2 + ((yy - 300) / 180.0) ** 2 <= 1
    feather[:, 210:] = False                                  # straight right side, x = 209
    st = fj.straight_outline(feather)
    assert st["along_longest"] >= 0.33 and st["along_runs"]
    cfg = {"hidden": {"edge_check": True}}
    board = np.full((600, 400), 200, np.uint8)               # light card ...
    board[:, 210:] = 30                                      # ... and a dark board over the right side, top to bottom
    vane = np.full((600, 400), 30, np.uint8)                 # the feather on the dark board, nothing on it
    vane[feather] = 120
    assert fj.hidden_level(dict(st), 1.0, cfg, board) == "part"
    assert fj.hidden_level(dict(st), 1.0, cfg, vane) == "none"
    assert fj.hidden_level(dict(st), 1.0, {"hidden": {}}, vane) == "part"     # check off: as before
    on_border = feather[:, :210]                             # the feather runs off the scan
    assert fj.hidden_level(fj.straight_outline(on_border), 1.0, cfg, vane[:, :210]) == "part"


def test_model_only_holes_and_fragments_need_support():
    from src import feather_judge as fj

    cfg = {"needs_support": ["holes_or_gaps", "fragment"], "fail_hole_frac": 0.02}
    rec = {"judge_status": "ok", "judge_sides_cut": [], "judge_holes_or_gaps": True, "judge_hole_frac": 0.0,
           "judge_mask_parts": 1, "judge_single_feather": True, "judge_fragment": True,
           "judge_other_feather_inside": False, "judge_other_material_inside": False, "judge_score": 3}
    fj.demote_unsupported(rec, cfg)
    assert fj.failed_checks(rec) == [] and rec["judge_score"] == 10
    assert rec["judge_uncertain_checks"] == ["fragment", "holes_or_gaps"]
    backed = {**rec, "judge_holes_or_gaps": True, "judge_hole_frac": 0.05, "judge_fragment": True,
              "postprocess": "possible fragment (small or narrow, not touching another feather)"}
    fj.demote_unsupported(backed, cfg)
    assert fj.failed_checks(backed) == ["holes or gaps", "fragment"]


def test_not_a_single_feather_needs_support():
    """'not a single feather' alone (the model's note even says ok) moves to the uncertain checks;
    with another feather inside it still fails."""
    from src import feather_judge as fj
    cfg = {"needs_support": ["single_feather"]}
    base = {"judge_status": "ok", "judge_single_feather": False, "judge_other_feather_inside": False,
            "judge_fragment": False, "judge_mask_parts": 1, "judge_notes": "ok"}
    alone = dict(base)
    fj.demote_unsupported(alone, cfg)
    assert alone["judge_single_feather"] is True and "single_feather" in alone["judge_uncertain_checks"]
    two = {**base, "judge_other_feather_inside": True}
    fj.demote_unsupported(two, cfg)
    assert two["judge_single_feather"] is False and not two.get("judge_uncertain_checks")
    split = {**base, "judge_mask_parts": 2}
    fj.demote_unsupported(split, cfg)
    assert split["judge_single_feather"] is False


def test_model_only_cut_on_the_board_needs_the_regrow_measure():
    """On the black board a model-only cut (regrow below min_growth) moves to the uncertain checks;
    on the white card, or with the regrow measure behind it, the cut stays."""
    import numpy as np
    from src import feather_judge as fj
    mask = np.zeros((400, 300), bool)
    mask[50:350, 100:200] = True
    board = np.full((400, 300, 3), 30, np.uint8)
    card = np.full((400, 300, 3), 210, np.uint8)
    cfg = {"board_cut": {"enabled": True}, "regrow": {"min_growth": 0.15}}
    base = {"judge_status": "ok", "judge_outline_left": "cut", "judge_outline_right": "complete",
            "judge_outline_top": "complete", "judge_outline_bottom": "complete", "judge_sides_cut": ["left"],
            "judge_regrow_left": 0.01, "judge_regrow_right": 0.0}
    on_board = dict(base)
    fj.demote_board_cuts(on_board, board, mask, cfg)
    assert on_board["judge_outline_left"] == "complete" and "outline_left" in on_board["judge_uncertain_checks"]
    on_card = dict(base)
    fj.demote_board_cuts(on_card, card, mask, cfg)
    assert on_card["judge_outline_left"] == "cut"
    measured = {**base, "judge_regrow_left": 0.3}
    fj.demote_board_cuts(measured, board, mask, cfg)
    assert measured["judge_outline_left"] == "cut"
