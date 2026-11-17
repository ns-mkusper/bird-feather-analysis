"""Slide-level facts added after reading the labels (src/slide_annotations.py)."""
from src import label_extract as lx
from src import slide_annotations as sa


def _slide(name, bird=("BZ11933H", "readable"), sex=("M", "readable"), species=("HAFLU", "readable")):
    r = lx.LabelResult(name, filename=lx.parse_filename(name))
    r.fields["bird_id"] = lx.FieldResult(bird[0], bird[1])
    r.fields["sex"] = lx.FieldResult(sex[0], sex[1])
    r.fields["species"] = lx.FieldResult(species[0], species[1])
    r.fields["date"] = lx.FieldResult("2007-06-18", "readable")
    return r


def test_sex_and_species_inferred_from_the_same_birds_confident_slides():
    a = _slide("BZ11933 2007-im2197.jpg", sex=(None, "not_found"), species=("HA", "partly_visible"))
    b = _slide("BZ11933 2007-im2198.jpg")
    c = _slide("BZ11933 2008-im2300.jpg", species=("HAFLU", "readable"))
    sa.annotate([a, b, c], ["BZ11933 2007-im2197.jpg", "BZ11933 2007-im2198.jpg", "BZ11933 2008-im2300.jpg"])
    assert (a.fields["sex"].value, a.fields["sex"].status, a.fields["sex"].source) == ("M", "inferred", "same_bird")
    assert a.fields["species"].status == "inferred" and "this slide: 'HA' [partly_visible]" in a.fields["species"].note
    assert "sex inferred from the same bird" in a.extra["label_info"] and not a.extra["label_problems"]
    assert a.extra["photos_same_year"] == 2 and a.extra["multiple_photos_same_year"] is True
    assert c.extra["photos_same_year"] == 1 and c.extra["multiple_photos_same_year"] is False


def test_no_inference_from_an_uncertain_id_or_disagreeing_slides():
    a = _slide("BZ11933 2007-im2197.jpg", bird=("BZ11933H", "partly_visible"), sex=(None, "not_found"))
    b = _slide("BZ11933 2007-im2198.jpg")
    sa.annotate([a, b])
    assert a.fields["sex"].status == "not_found" and "bird_id partly visible" in a.extra["label_problems"]
    x = _slide("A0012 1996-im1058.jpg", bird=("A0012H", "readable"), sex=("F", "readable"))
    y = _slide("A0012 1996-im1059.jpg", bird=("A0012H", "readable"), sex=("M", "readable"))
    z = _slide("A0012 1997-im1100.jpg", bird=("A0012H", "readable"), sex=(None, "not_found"))
    sa.annotate([x, y, z])
    assert z.fields["sex"].status == "not_found" and "this bird's slides disagree (F/M)" in z.extra["label_problems"]


def test_an_untrusted_bird_id_or_date_makes_the_feathers_unusable():
    r = _slide("BZ11933 2007-im2197.jpg", bird=("BZ11933H", "partly_visible"))
    r.feathers = [{"seq": "A", "feather_id": "BZ11933H_2007_A", "segment_status": "segmented", "needs_review": "",
                   "crop": "BZ11933_2007_Feather_A.jpg"}]
    sa.annotate([r], handwriting={"BZ11933 2007-im2197.jpg": {"handwritten": "yes", "fields": "date"}})
    row = lx.feather_table_rows(r, "run")[0]
    assert row["usable"] == "no" and row["excluded_because"] == "label: bird_id partly visible"
    assert row["handwritten"] == "yes" and "handwritten: date" in row["label_problems"]


def test_feather_rows_show_the_slides_current_label_status():
    r = _slide("BZ11933 2007-im2197.jpg", bird=("BZ11933H", "partly_visible"))
    r.feathers = [{"seq": "A", "feather_id": "BZ11933H_2007_A", "segment_status": "segmented", "needs_review": "",
                   "crop": "c.jpg", "bird_id_status": "readable"}]       # stale copy from ID assignment
    sa.annotate([r])
    assert lx.feather_table_rows(r, "run")[0]["bird_id_status"] == "partly_visible"


def test_a_field_not_found_is_reported():
    """A field the model did not find is reported (it may be under the board), unless inferred: as
    information, since nobody can read a covered field from the image either."""
    a = _slide("BZ10382 2005-im2035.jpg", bird=("BZ10382H", "readable"), species=(None, "not_found"))
    sa.annotate([a])
    assert "species not found (covered or not printed)" in a.extra["label_info"]
    assert "species" not in a.extra["label_problems"]
    b = _slide("A0399 1996-im1148.jpg", bird=("A0399H", "readable"), species=(None, "not_found"))
    c = _slide("A0399 1997-im1149.jpg", bird=("A0399H", "readable"))
    sa.annotate([b, c])
    assert b.extra["label_info"] == "species inferred from the same bird" and b.extra["label_problems"] == ""
