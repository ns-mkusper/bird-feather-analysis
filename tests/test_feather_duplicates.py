"""Duplicate feathers across same-day slides (synthetic crops)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import feather_duplicates as fd


def _feather(seed: int, length: int = 420, width: int = 110) -> np.ndarray:
    """Textured ellipse on black, like a segmented crop."""
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (length, width), dtype=np.uint8), (3, 3), 0)
    tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    mask = np.zeros((length, width), np.uint8)
    cv2.ellipse(mask, (width // 2, length // 2), (width // 2 - 2, length // 2 - 2), 0, 0, 360, 255, -1)
    tex[mask == 0] = 0
    tex[mask > 0] = np.maximum(tex[mask > 0], 30)
    return tex


def _save(img, path):
    cv2.imwrite(str(path), img)
    return path


def test_rotated_and_mirrored_repeats_are_duplicates(tmp_path):
    a1, a2 = _feather(1), _feather(2, length=380)
    # slide 2: same two feathers, swapped order, one rotated 90 degrees, one mirrored + tilted
    rot = cv2.rotate(a1, cv2.ROTATE_90_CLOCKWISE)
    m = cv2.getRotationMatrix2D((55, 190), 12, 1.0)
    tilted = cv2.warpAffine(cv2.flip(a2, 1), m, (160, 420))
    other = _feather(3, length=300, width=90)
    s1 = [fd.Feather("AB1234H_20000521-im0001_A", "AB1234 2000-im0001.jpg", "A", _save(a1, tmp_path / "1a.jpg")),
          fd.Feather("AB1234H_20000521-im0001_B", "AB1234 2000-im0001.jpg", "B", _save(a2, tmp_path / "1b.jpg"))]
    s2 = [fd.Feather("AB1234H_20000521-im0002_A", "AB1234 2000-im0002.jpg", "A", _save(tilted, tmp_path / "2a.jpg")),
          fd.Feather("AB1234H_20000521-im0002_B", "AB1234 2000-im0002.jpg", "B", _save(other, tmp_path / "2b.jpg")),
          fd.Feather("AB1234H_20000521-im0002_C", "AB1234 2000-im0002.jpg", "C", _save(rot, tmp_path / "2c.jpg"))]
    slides = {"AB1234 2000-im0001.jpg": s1, "AB1234 2000-im0002.jpg": s2}
    dates = {"AB1234 2000-im0001.jpg": "2000-05-21", "AB1234 2000-im0002.jpg": "2000-05-21"}
    rows = {r["feather_id"]: r for r in fd.find_duplicates(slides, dates)}
    assert rows["AB1234H_20000521-im0002_C"]["duplicate_of"] == "AB1234H_20000521-im0001_A"
    assert rows["AB1234H_20000521-im0002_A"]["duplicate_of"] == "AB1234H_20000521-im0001_B"
    assert rows["AB1234H_20000521-im0002_A"]["match_mirrored"] is True
    assert rows["AB1234H_20000521-im0002_B"]["duplicate_of"] == ""
    assert rows["AB1234H_20000521-im0001_A"]["duplicate_of"] == ""


def test_different_days_are_not_compared(tmp_path):
    a = _feather(1)
    slides = {"AB1234 2000-im0001.jpg": [fd.Feather("a", "AB1234 2000-im0001.jpg", "A", _save(a, tmp_path / "a.jpg"))],
              "AB1234 2000-im0002.jpg": [fd.Feather("b", "AB1234 2000-im0002.jpg", "A", _save(a, tmp_path / "b.jpg"))]}
    rows = fd.find_duplicates(slides, {"AB1234 2000-im0001.jpg": "2000-05-21", "AB1234 2000-im0002.jpg": "2000-06-30"})
    assert rows == []
    # unknown date on one slide: compared with its same-year siblings
    rows = fd.find_duplicates(slides, {"AB1234 2000-im0001.jpg": "2000-05-21", "AB1234 2000-im0002.jpg": None})
    assert {r["feather_id"]: r["duplicate_of"] for r in rows} == {"a": "", "b": "a"}
