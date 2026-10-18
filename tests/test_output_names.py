"""Segmentation output file names (src/output_names.py)."""
from src import output_names as on

SIBLINGS = ["CE01168 2008-im2346.jpg", "BB17692 1998-im0513.jpg", "BB17692 1998-im0514.jpg",
            "AX868161996-img0194.jpg", "scan.jpg"]


def test_short_names_for_unique_bird_and_year():
    n = on.output_names("CE01168 2008-im2346.jpg", SIBLINGS)
    assert on.overlay_name(n) == "CE01168 2008_BoundingBoxes.jpg"
    assert on.crop_name(n, 3) == "CE01168_2008_Feather_D.jpg"


def test_same_bird_and_year_uses_full_label_date():
    n = on.output_names("BB17692 1998-im0513.jpg", SIBLINGS, date="1998-06-12",
                        sibling_dates={"BB17692 1998-im0514.jpg": "1998-06-20"})
    assert on.overlay_name(n) == "BB17692 19980612_BoundingBoxes.jpg"
    assert on.crop_name(n, 0) == "BB17692_19980612_Feather_A.jpg" and n["period"] == "19980612"
    # a unique bird/year ignores the date
    assert on.output_names("CE01168 2008-im2346.jpg", SIBLINGS, date="2008-05-22")["period"] == "2008"
    assert on.period_of_crop("BB17692_19980612_Feather_A.jpg") == "19980612"
    assert on.period_of_crop("A0012 1996-im1058_A0012_1996_Feather_1.jpg") is None


def test_same_bird_and_year_without_readable_date_keeps_image_number():
    n = on.output_names("BB17692 1998-im0513.jpg", SIBLINGS)
    assert on.overlay_name(n) == "BB17692 1998-im0513_BoundingBoxes.jpg"
    assert on.crop_name(n, 0) == "BB17692_1998-im0513_Feather_A.jpg"
    assert on.output_names("BB17692 1998-im0514.jpg", SIBLINGS)["crop_prefix"] == "BB17692_1998-im0514"


def test_odd_and_unparsed_names():
    assert on.output_names("AX868161996-img0194.jpg", SIBLINGS)["base"] == "AX86816 1996"
    assert on.output_names("scan.jpg", SIBLINGS) == {"base": "scan", "crop_prefix": "scan", "period": None}


def test_default_siblings_come_from_the_slide_folder(tmp_path):
    for n in ("BB17692 1998-im0513.jpg", "BB17692 1998-im0514.jpg", "A0012 1996-im1058.jpg"):
        (tmp_path / n).write_bytes(b"x")
    assert on.output_names(str(tmp_path / "BB17692 1998-im0513.jpg"))["base"] == "BB17692 1998-im0513"
    assert on.output_names(str(tmp_path / "A0012 1996-im1058.jpg"))["base"] == "A0012 1996"


def test_find_crops_and_overlay_new_and_old_names(tmp_path):
    new = tmp_path / "new"
    new.mkdir()
    for f in ("CE01168_2008_Feather_B.jpg", "CE01168_2008_Feather_A.jpg", "CE01168 2008_BoundingBoxes.jpg",
              "BB17692_1998-im0514_Feather_A.jpg", "BB17692 1998-im0514_BoundingBoxes.jpg"):
        (new / f).write_bytes(b"x")
    assert [p.name for p in on.find_crops(new, "CE01168 2008-im2346.jpg")] == \
        ["CE01168_2008_Feather_A.jpg", "CE01168_2008_Feather_B.jpg"]
    assert on.find_crops(new, "BB17692 1998-im0513.jpg") == []          # sibling's crops are not ours
    assert on.find_overlay(new, "BB17692 1998-im0514.jpg").name == "BB17692 1998-im0514_BoundingBoxes.jpg"
    assert on.find_crop(new, "BB17692", "1998", "A").name == "BB17692_1998-im0514_Feather_A.jpg"
    old = tmp_path / "old"
    old.mkdir()
    for k in (10, 2, 1):
        (old / f"A0012 1996-im1058_A0012_1996_Feather_{k}.jpg").write_bytes(b"x")
    (old / "A0012 1996-im1058_BoundingBoxes.jpg").write_bytes(b"x")
    assert [p.name.rsplit("_", 1)[-1] for p in on.find_crops(old, "A0012 1996-im1058.jpg")] == \
        ["1.jpg", "2.jpg", "10.jpg"]
    assert on.find_overlay(old, "A0012 1996-im1058.jpg") is not None


def test_belongs_to():
    assert on.belongs_to("CE01168_2008_Feather_A.jpg", "CE01168 2008-im2346.jpg")
    assert on.belongs_to("CE01168 2008_BoundingBoxes.jpg", "CE01168 2008-im2346.jpg")
    assert on.belongs_to("A0012 1996-im1058_A0012_1996_Feather_1.jpg", "A0012 1996-im1058.jpg")
    assert not on.belongs_to("CE01168_2009_Feather_A.jpg", "CE01168 2008-im2346.jpg")


def test_find_date_named_crops(tmp_path):
    import json
    for f in ("BB17692_19980612_Feather_A.jpg", "BB17692_19980612_Feather_B.jpg", "BB17692 19980612_BoundingBoxes.jpg",
              "BB17692_19980613_Feather_A.jpg"):
        (tmp_path / f).write_bytes(b"x")
    assert len(on.find_crops(tmp_path, "BB17692 1998-im0513.jpg", date="1998-06-12")) == 2
    assert on.find_crops(tmp_path, "BB17692 1998-im0513.jpg") == []       # which sibling? unknown without the date
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "BB17692 1998-im0513.json").write_text(json.dumps(
        {"feathers": [{"crop": "BB17692_19980612_Feather_A.jpg"}, {"crop": "BB17692_19980612_Feather_B.jpg"}]}))
    assert [p.name for p in on.find_crops(tmp_path, "BB17692 1998-im0513.jpg")] == \
        ["BB17692_19980612_Feather_A.jpg", "BB17692_19980612_Feather_B.jpg"]   # recorded by the pipeline
    assert on.find_overlay(tmp_path, "BB17692 1998-im0513.jpg").name == "BB17692 19980612_BoundingBoxes.jpg"
    assert on.find_crop(tmp_path, "BB17692", "1998").name == "BB17692_19980612_Feather_A.jpg"
    assert on.belongs_to("BB17692_19980612_Feather_A.jpg", "BB17692 1998-im0513.jpg", "1998-06-12")


def test_same_day_slides_add_image_number():
    sib = ["A0514 2000-im1163.jpg", "A0514 2000-im1164.jpg", "A0514 2000-im1165.jpg"]
    dates = {"A0514 2000-im1164.jpg": "2000-05-21", "A0514 2000-im1165.jpg": "2000-07-02"}
    n = on.output_names("A0514 2000-im1163.jpg", sib, "2000-05-21", dates)
    assert n["crop_prefix"] == "A0514_20000521-im1163"
    n = on.output_names("A0514 2000-im1165.jpg", sib, "2000-07-02",
                        {"A0514 2000-im1163.jpg": "2000-05-21", "A0514 2000-im1164.jpg": "2000-05-21"})
    assert n["crop_prefix"] == "A0514_20000702"
    # a sibling whose date is unknown might be the same day -> image number
    n = on.output_names("A0514 2000-im1165.jpg", sib, "2000-07-02", {"A0514 2000-im1163.jpg": None})
    assert n["crop_prefix"] == "A0514_20000702-im1165"
