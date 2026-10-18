"""Mask clean-up rules (synthetic masks at slide size)."""
import cv2
import numpy as np

from src import segment_postprocess as sp

H, W = 600, 800


def feather(cx, cy, length=400, width=70, angle=0):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (cx, cy), (width // 2, length // 2), angle, 0, 360, 1, -1)
    return m.astype(bool)


def test_four_clean_feathers_untouched():
    ms = [feather(100 + 150 * i, 300) for i in range(4)]
    out = sp.clean_masks(ms, [0.9] * 4)
    assert len(out) == 4 and all(not p.notes for p in out)


def test_sliver_next_to_feather_is_merged():
    """A3818: one vane returned as a thin separate mask beside the feather."""
    full = feather(400, 300, width=70)
    sliver = np.zeros((H, W), bool)
    sliver[150:420, 355:368] = True              # thin strip touching the left edge
    others = [feather(150, 300), feather(650, 300)]
    out = sp.clean_masks([others[0], sliver, full, others[1]], [0.9, 0.5, 0.9, 0.9])
    assert len(out) == 3
    merged = next(p for p in out if 1 in p.sources)
    assert sorted(merged.sources) == [1, 2] and "fragment touching" in merged.notes[0]


def test_contained_duplicate_mask_dropped():
    full = feather(400, 300)
    inner = feather(400, 300, length=200, width=40)
    out = sp.clean_masks([full, inner, feather(150, 300)], [0.9, 0.8, 0.9])
    assert len(out) == 2 and any("contained" in n for p in out for n in p.notes)


def test_feather_cut_by_board_is_joined():
    """Two end-to-end pieces with a gap where a board crosses the feather."""
    top = np.zeros((H, W), bool); top[60:250, 380:440] = True
    bottom = np.zeros((H, W), bool); bottom[290:520, 383:443] = True
    out = sp.clean_masks([top, bottom, feather(150, 300), feather(650, 300)], [0.9] * 4)
    assert len(out) == 3 and any("aligned pieces" in n for p in out for n in p.notes)


def test_two_neighbouring_feathers_stay_separate():
    a, b = feather(380, 300), feather(452, 300)        # side by side, edges touching
    out = sp.clean_masks([a, b, feather(150, 300), feather(650, 300)], [0.9] * 4)
    assert len(out) == 4


def test_single_small_feather_slide_kept():
    out = sp.clean_masks([feather(400, 300, length=200, width=40)], [0.9])
    assert len(out) == 1 and not out[0].notes


def test_shape_stats_long_axis_angle():
    s = sp.shape_stats(feather(400, 300, angle=30))
    assert 395 < s["length"] < 405 and 65 < s["width"] < 75
