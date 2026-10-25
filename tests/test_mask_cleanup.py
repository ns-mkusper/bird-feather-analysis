"""Glue and card removed from a feather mask before cropping (src/mask_cleanup.py)."""
import numpy as np

from src import mask_cleanup as mc


def _feather(blob: bool, textured_blob: bool = False, background: int = 200):
    """A striped (barb-textured) vane 60 px wide over a narrow quill, and optionally a wide flat
    blob (glue dot) at the bottom. background 200 = the white card, 30 = the black board."""
    img = np.full((400, 200, 3), background, np.uint8)
    yy, xx = np.mgrid[0:400, 0:200]
    stripes = ((xx + yy) // 3) % 2 == 0
    mask = np.zeros((400, 200), bool)
    mask[40:280, 70:130] = True                       # vane
    mask[280:340, 97:103] = True                      # bare quill
    img[mask & stripes] = 60
    if blob:
        mask[330:380, 60:140] = True                  # glue dot, flat grey
        img[330:380, 60:140] = 150
        if textured_blob:
            img[330:380, 60:140][stripes[330:380, 60:140]] = 60
    return img, mask


def test_glue_dot_below_the_quill_is_cut_off():
    img, mask = _feather(blob=True)
    out, cut, notes = mc.clean(img, mask, {})
    assert cut is not None and cut <= 335 and not out[340:].any()
    assert out[40:280].sum() == mask[40:280].sum()    # the vane is untouched
    assert notes


def test_neck_alone_finds_a_glue_dot_with_texture():
    img, mask = _feather(blob=True, textured_blob=True)
    out, cut, notes = mc.clean(img, mask, {"texture": False, "vane_base": {"enabled": False}})
    assert cut is not None and 280 <= cut <= 335 and "neck" in notes[-1]


def test_a_clean_feather_on_the_board_is_left_alone():
    img, mask = _feather(blob=False, background=30)
    out, cut, notes = mc.clean(img, mask, {})
    assert cut is None and (out == mask).all() and not notes


def test_on_the_card_the_feather_is_cut_to_its_full_vane():
    img, mask = _feather(blob=False)
    out, cut, notes = mc.clean(img, mask, {})
    assert cut is not None and 278 <= cut <= 285 and "full vane" in notes[-1]
    assert out[40:278].sum() == mask[40:278].sum() and not out[cut:].any()


def test_never_cuts_most_of_a_feather_away():
    img, mask = _feather(blob=True)
    out, cut, notes = mc.clean(img, mask, {"base_frac": 1.0, "min_keep_frac": 0.99})
    assert cut is None and (out == mask).all() and "skipped" in notes[0]


def test_calamus_cut_on_the_board_keeps_the_vane():
    img, mask = _feather(blob=False, background=30)
    out, cut, notes = mc.clean(img, mask, {"calamus": {"enabled": True}})
    assert cut is not None and 278 <= cut <= 285 and "calamus" in notes[0]
    assert out[40:278].sum() == mask[40:278].sum() and not out[cut:].any()
