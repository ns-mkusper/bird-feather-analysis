"""What the feather cards mean (src/card_rules.py) and the slide-level rules built on it
(src/slide_annotations.py). Slide names and readings are real cases from full-20261114."""
from PIL import Image

from src import card_rules as cr
from src import label_extract as lx
from src import slide_annotations as sa

CFG = lx.load_config()


def _slide(name, **fields):
    r = lx.LabelResult(name, filename=lx.parse_filename(name))
    base = {"bird_id": (name.split()[0] + "H", "readable"), "species": ("HAFLU", "readable"),
            "age": ("RC", "code_on_card"), "sex": ("F", "readable"), "date": (name.split()[1][:4] + "-06-11", "readable")}
    for k, (v, st) in {**base, **fields}.items():
        r.fields[k] = lx.FieldResult(v, st, image_value=v)
    return r


def _feather(r):
    r.feathers = [{"seq": "A", "feather_id": f"{r.filename['id']}H_{r.filename['year']}_A", "segment_status": "segmented",
                   "needs_review": "", "crop": "x.jpg"}]
    return r


def test_misread_species_codes_become_the_known_code():
    assert cr.nearest_code("HAFIU", CFG["card"]["species_codes"]) == "HAFLU"
    assert cr.nearest_code("SVELL", CFG["card"]["species_codes"]) == "SVFLU"
    assert cr.nearest_code("HEXSF", CFG["card"]["species_codes"]) == "HFXSF"
    assert cr.nearest_code("FLU", CFG["card"]["species_codes"]) is None       # HAFLU or SVFLU: no guess
    r = _slide("BB17575 1994-im0491.jpg", species=("HAFIU", "readable"))
    cr.apply(r, CFG)
    assert r.fields["species"].value == "HAFLU" and "read 'HAFIU'" in r.fields["species"].note
    cut = _slide("A0762 1996-im1176.jpg", species=("HAF", "needs_review"))
    cut.fields["species"].note = "value fails the field format"
    cr.apply(cut, CFG)
    assert (cut.fields["species"].value, cut.fields["species"].status) == ("HAFLU", "readable")


def test_the_bird_id_read_in_the_species_crop_means_the_row_is_covered():
    r = _slide("A0113 1996-im1011.jpg", species=("A0113", "needs_review"))
    cr.apply(r, CFG)
    assert r.fields["species"].status == "not_found" and "species row covered" in r.fields["species"].note


def test_rc_in_the_age_box_is_information_not_a_review():
    r = _slide("AS93827 1990-img0153.jpg", age=("RC", "needs_review"))
    cr.apply(r, CFG)
    sa.annotate([r])
    assert (r.fields["age"].value, r.fields["age"].status) == ("RC", "code_on_card")
    assert not r.needs_review and "age not on the card (code in the box)" in r.extra["label_info"]
    assert "age" not in r.extra["label_problems"]


def test_the_box_colour_gives_a_covered_age_unless_visible_digits_disagree():
    r = _slide("A2534 2003-im1509.jpg", age=(None, "partly_visible"))
    r.extra["age_box"] = {"text": None, "colour": "green", "state": "partly"}
    cr.apply(r, CFG)
    assert (r.fields["age"].value, r.fields["age"].status) == ("5K", "from_colour")
    bad = _slide("A2534 2003-im1509.jpg", age=("3K", "partly_visible"))
    bad.extra["age_box"] = {"text": "3", "colour": "green", "state": "partly"}
    cr.apply(bad, CFG)
    assert bad.fields["age"].status == "partly_visible"
    red = _slide("A2534 2003-im1509.jpg", age=(None, "not_found"))
    red.extra["age_box"] = {"text": None, "colour": "red", "state": "covered"}      # 7K, or the orange 1K?
    cr.apply(red, CFG)
    assert red.fields["age"].status == "not_found"
    empty = _slide("AX87754 1992-img0289.jpg", age=(None, "not_found"))
    empty.extra["age_box"] = {"text": None, "colour": "white", "state": "empty"}
    cr.apply(empty, CFG)
    assert empty.fields["age"].status == "not_on_card"


def test_sex_certainty_comes_from_the_letter_case_the_model_saw():
    r = _slide("A1499 1997-im1329.jpg")
    f = r.fields["sex"]
    f.source = "primary"
    f.attempts = [lx.Attempt("primary", [0, 0, 1, 1], "M", "yes", 0.9, '{"value": "m", "visible": "yes"}', 1.0)]
    cr.apply(r, CFG)
    assert r.extra["sex_certainty"] == "supposed"


def test_a_partly_visible_bird_id_agreeing_with_the_file_name_keeps_its_feathers():
    r = _feather(_slide("BZ11933 2007-im2197.jpg", bird_id=("BZ11933H", "partly_visible")))
    cr.apply(r, CFG)
    sa.annotate([r])
    assert r.fields["bird_id"].status == "confirmed_by_filename"
    assert lx.feather_table_rows(r, "run")[0]["usable"] == "yes"
    other = _feather(_slide("A2349 2000-im1486.jpg", bird_id=("A2340H", "partly_visible")))
    cr.apply(other, CFG)
    assert other.fields["bird_id"].status == "partly_visible"
    assert lx.feather_table_rows(other, "run")[0]["usable"] == "no"
    strict = {**CFG, "card": {**CFG["card"], "filename_confirms_partial_ids": False}}
    off = _feather(_slide("BZ11933 2007-im2197.jpg", bird_id=("BZ11933H", "partly_visible")))
    cr.apply(off, strict)
    assert lx.feather_table_rows(off, "run")[0]["usable"] == "no"


def test_dates_the_file_name_confirms_and_a_handwritten_yy_mm_dd():
    r = _slide("A1108 1999-im1255.jpg", date=("1999-06-22", "partly_visible"))
    cr.apply(r, CFG)
    assert r.fields["date"].status == "confirmed_by_filename"
    guessed = _slide("A0988 2000-im1216.jpg", date=("2006-01-04", "partly_visible"))
    cr.apply(guessed, CFG)
    assert guessed.fields["date"].status == "partly_visible"
    hw = _slide("A1225 2003-im2337.jpg", date=("2011-06-03", "conflict"))
    hw.extra["handwritten"] = "yes"
    cr.apply(hw, CFG)
    assert (hw.fields["date"].value, hw.fields["date"].status) == ("2003-06-11", "confirmed_by_filename")


def test_box_ages_follow_the_years_and_fill_a_missing_one():
    a = _slide("BB17127 1992-im0379.jpg", age=("2K", "readable"), date=("1992-06-10", "readable"))
    b = _slide("BB17127 1993-im0380.jpg", age=(None, "not_found"), date=("1993-06-12", "readable"))
    sa.annotate([a, b])
    assert (b.fields["age"].value, b.fields["age"].status) == ("3K", "inferred")
    c = _slide("A1326 1998-im1300.jpg", age=("2K", "readable"), date=("1998-06-01", "readable"))
    d = _slide("A1326 1999-im1301.jpg", age=("4K", "readable"), date=("1999-06-01", "readable"))
    sa.annotate([c, d])
    assert "does not follow this bird's other years" in c.extra["label_problems"]       # no majority: both
    e = _slide("A2430 2001-im1501.jpg", age=("3K", "readable"))
    f = _slide("A2430 2002-im1502.jpg", age=("4K", "readable"))
    g = _slide("A2430 2004-im1504.jpg", age=("5K", "readable"), date=("2014-06-02", "readable"))
    h = _slide("A2430 2003-im1503.jpg", age=("5K", "readable"))
    sa.annotate([e, f, g, h])
    assert not e.extra["label_problems"] and not h.extra["label_problems"]    # 3K 2001, 4K 2002, 5K 2003
    assert "does not follow" in g.extra["label_problems"]                       # 5K in 2004: should be 6K


def test_species_changing_between_years_is_information_and_supposed_sex_explains_a_disagreement():
    a = _slide("BX40889 2003-im1924.jpg", species=("HAFLU", "readable"))
    b = _slide("BX40889 2005-im1926.jpg", species=("HFXSF", "readable"))
    sa.annotate([a, b])
    assert "species differs between this bird's slides" in a.extra["label_info"] and not a.extra["label_problems"]
    x = _slide("A1503 1997-im1335.jpg", sex=("M", "readable"))
    y = _slide("A1503 1998-im1336.jpg", sex=("F", "readable"))
    x.extra["sex_certainty"], y.extra["sex_certainty"] = "supposed", "certain"
    sa.annotate([x, y])
    assert "one of them supposed" in x.extra["label_info"] and not x.extra["label_problems"]
    x.extra["sex_certainty"] = "certain"
    sa.annotate([x, y])
    assert "sex: this bird's slides disagree (F/M)" in x.extra["label_problems"]


class _FakeClient:
    model_name = "fake"

    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def ask(self, image, prompt, timeout=None, max_tokens=None):
        key = "box" if "coloured square" in prompt else "field_age"
        self.asked.append(key)
        return self.answers[key]


def test_a_cached_reading_gets_the_new_field_and_the_age_box_without_a_full_reread(tmp_path):
    img = tmp_path / "A2534 2003-im1509.jpg"
    Image.new("RGB", (400, 400), "white").save(img)
    r = _slide("A2534 2003-im1509.jpg", age=(None, "partly_visible"))
    client = _FakeClient({"box": '{"text": null, "colour": "green", "state": "partly"}',
                          "field_age": '{"value": "4K+", "visible": "yes", "confidence": 0.9}'})
    assert lx.complete_reading(r, img, client, CFG)
    assert r.fields["field_age"].value == "4K+" and r.extra["age_box"]["colour"] == "green"
    cr.apply(r, CFG)
    assert r.fields["age"].value == "5K"        # the box age, never the field age
    assert not lx.complete_reading(r, img, client, CFG)


def test_a_second_look_must_agree_before_a_partly_visible_field_counts():
    img = Image.new("RGB", (400, 400), "white")
    r = _slide("BD31073 1995-im0683.jpg", species=("HAFLU", "partly_visible"), sex=("F", "partly_visible"))
    agree = _FakeClient({"box": "{}", "field_age": '{"value": "HAFLU", "visible": "yes", "confidence": 0.9}'})
    assert lx.second_look(r, "species", agree, img, CFG)
    assert (r.fields["species"].status, r.fields["species"].source) == ("readable", "second_look")
    differ = _FakeClient({"box": "{}", "field_age": '{"value": "M", "visible": "yes", "confidence": 0.9}'})
    assert not lx.second_look(r, "sex", differ, img, CFG)
    assert r.fields["sex"].status == "partly_visible" and "not accepted" in r.fields["sex"].note


def test_a_covered_purple_box_may_be_rc_and_the_rules_can_run_again():
    r = _slide("BZ11933 2007-im2197.jpg", age=(None, "not_found"))
    r.extra["age_box"] = {"text": None, "colour": "purple", "state": "covered"}
    cr.apply(r, CFG)
    assert r.fields["age"].status == "not_found"            # RC boxes are purple too
    r.extra["age_box"] = {"text": "2", "colour": "purple", "state": "partly"}
    cr.apply(r, CFG)
    assert (r.fields["age"].value, r.fields["age"].status) == ("2K", "from_colour")
    r.extra["age_box"] = {"text": None, "colour": "purple", "state": "covered"}
    cr.apply(r, CFG)                                          # rules changed: back to the model's reading
    assert (r.fields["age"].value, r.fields["age"].status) == (None, "not_found")


def test_a_year_confirmed_by_the_file_name_is_not_enough_with_several_slides_that_year():
    a = _feather(_slide("BB17692 1998-im0513.jpg", date=("1998-06-21", "partly_visible")))
    b = _slide("BB17692 1998-im0515.jpg")
    cr.apply(a, CFG)
    sa.annotate([a, b], ["BB17692 1998-im0513.jpg", "BB17692 1998-im0515.jpg"])
    row = lx.feather_table_rows(a, "run")[0]
    assert a.fields["date"].status == "confirmed_by_filename" and row["usable"] == "no"
    assert "several slides that year" in row["excluded_because"]
    alone = _feather(_slide("A1108 1999-im1255.jpg", date=("1999-06-22", "partly_visible")))
    cr.apply(alone, CFG)
    sa.annotate([alone], ["A1108 1999-im1255.jpg"])
    assert lx.feather_table_rows(alone, "run")[0]["usable"] == "yes"


def test_a_hand_correction_is_a_problem_and_other_notes_are_information():
    r = _slide("BZ10121 2005-im2332.jpg")
    r.extra["label_note"] = "printed bird ID A3839H crossed out by hand, no new ID written"
    note = _slide("A0514 2000-im1163.jpg")
    note.extra["label_note"] = "handwritten note in the date corner (red '1', 'VANO!')"
    sa.annotate([r, note])
    assert "crossed out by hand" in r.extra["label_problems"]
    assert "VANO" in note.extra["label_info"] and not note.extra["label_problems"]


def test_an_unreadable_field_age_is_information_only():
    r = _slide("AS93827 1990-img0153.jpg", field_age=("M", "needs_review"))
    sa.annotate([r])
    assert not r.needs_review and not r.extra["label_problems"]
    assert "field_age needs review" in r.extra["label_info"]
