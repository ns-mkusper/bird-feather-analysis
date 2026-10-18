"""Prompt lab helpers (no model calls)."""
import pandas as pd

from src import prompt_lab as pl


def test_compare_and_verdict_agreement(tmp_path):
    a = pd.DataFrame({"file_id": ["s"] * 3, "seq": list("ABC"), "crop": ["a", "b", "c"],
                      "score": [9.0, 8.0, 3.0], "notes": ["", "", "fragment"], "status": ["ok"] * 3})
    b = a.assign(score=[9.0, 5.0, 3.0])
    v = tmp_path / "v.csv"
    v.write_text("file_id,seq,crop,verdict,why\ns,A,a,good,\ns,B,b,bad,vane missing\ns,C,c,bad,\n")
    table, stats = pl.compare(a, b, v, good_from=7)
    assert list(table.delta) == [0.0, -3.0, 0.0]
    s = stats.set_index("prompt")
    assert s.loc["A", "agree"] == 2 and s.loc["A", "bad called good"] == 1
    assert s.loc["B", "agree"] == 3


def test_verdict_template_keeps_existing_calls(tmp_path):
    p = tmp_path / "v.csv"
    feathers = [{"file_id": "s", "seq": "A", "crop": "a"}, {"file_id": "s", "seq": "B", "crop": "b"}]
    pl.write_verdict_template(feathers, p)
    p.write_text(p.read_text().replace("s,A,a,,", "s,A,a,bad,cut tip"))
    pl.write_verdict_template(feathers, p)
    rows = pd.read_csv(p)
    assert rows.verdict.tolist()[0] == "bad" and rows.why.tolist()[0] == "cut tip"


def test_current_prompts_lists_everything():
    p = pl.current_prompts()
    assert p["sam3_prompts"] and "segmented feather" in p["feather_judge_prompt"] and p["slide_judge_prompt"]
