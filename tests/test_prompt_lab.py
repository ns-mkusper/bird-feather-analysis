"""Prompt lab helpers (no model calls)."""
import pandas as pd

from src import prompt_lab as pl


def _res(**kw):
    """One judge result row (all checks passed unless overridden)."""
    r = {"file_id": "s", "status": "ok", "outline_left": "complete", "outline_right": "complete",
         "outline_top": "complete", "outline_bottom": "complete", "sides_cut": [], "holes_or_gaps": False,
         "holes": 0, "mask_parts": 1, "fragment": False, "single_feather": True, "other_feather_inside": False,
         "other_material_inside": False, "score": 10, "notes": ""}
    r.update(kw)
    return r


def test_compare_uses_the_checks_not_the_score(tmp_path):
    a = pd.DataFrame([_res(seq="A", crop="a"), _res(seq="B", crop="b", score=10),
                      _res(seq="C", crop="c", fragment=True, score=6)])
    b = pd.DataFrame([_res(seq="A", crop="a"), _res(seq="B", crop="b", outline_top="cut", sides_cut=["top"], score=8),
                      _res(seq="C", crop="c", fragment=True, score=6)])
    v = tmp_path / "v.csv"
    v.write_text("file_id,seq,crop,verdict,why\ns,A,a,good,\ns,B,b,bad,tip cut\ns,C,c,bad,\n")
    table, stats = pl.compare(a, b, v)
    assert list(table.decision_B) == ["good", "bad", "bad"] and list(table.delta) == [0, -2, 0]
    s = stats.set_index("prompt")
    assert s.loc["A", "agree"] == 2 and s.loc["A", "bad called good"] == 1
    assert s.loc["B", "agree"] == 3


def test_per_check_agreement(tmp_path):
    res = pd.DataFrame([
        _res(crop="a"),                                                      # model: all pass
        _res(crop="b", outline_top="cut", sides_cut=["top"]),                # model: top cut
        _res(crop="c", other_material_inside=True),                          # model: material inside
    ])
    v = tmp_path / "v.csv"
    v.write_text("file_id,seq,crop,verdict,outline_left,outline_right,outline_top,outline_bottom,holes_or_gaps,"
                 "fragment,single_feather,other_feather_inside,other_material_inside,why\n"
                 "s,A,a,bad,complete,complete,cut,complete,false,false,true,false,false,tip cut\n"
                 "s,B,b,bad,,,cut,,,,,,,\n"
                 "s,C,c,good,,,,,,,,,false,\n")
    t = pl.check_agreement(res, v).set_index("check")
    top = t.loc["outline_top"]
    assert top["feathers you checked"] == 2 and top["agree"] == 1 and top["missed problem"] == 1
    mat = t.loc["other_material_inside"]
    assert mat["feathers you checked"] == 2 and mat["false alarm"] == 1 and mat["missed problem"] == 0
    assert t.loc["fragment", "feathers you checked"] == 1 and t.loc["fragment", "agreement"] == 1.0


def test_verdict_template_keeps_existing_calls(tmp_path):
    p = tmp_path / "v.csv"
    feathers = [{"file_id": "s", "seq": "A", "crop": "a"}, {"file_id": "s", "seq": "B", "crop": "b"}]
    pl.write_verdict_template(feathers, p)
    p.write_text(p.read_text().replace("s,A,a," + "," * 10, "s,A,a,bad,cut,,,,,,,,,cut tip"))
    pl.write_verdict_template(feathers, p)
    rows = pd.read_csv(p)
    assert rows.verdict.tolist()[0] == "bad" and rows.why.tolist()[0] == "cut tip"
    assert rows.outline_left.tolist()[0] == "cut" and list(rows.columns[4:13]) == pl.CHECKS


def test_current_prompts_lists_everything():
    p = pl.current_prompts()
    assert p["sam3_prompts"] and "Outline, side by side" in p["feather_judge_prompt"] and p["inventory_prompt"]
    assert p["feather_score"]["deduct"]["side_cut"] == 2 and "slide_judge_prompt" not in p
