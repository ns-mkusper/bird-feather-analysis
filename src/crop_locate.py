"""Find where a saved feather crop sits on its slide (for crops whose mask was
not kept, e.g. outputs of older worker code), and rebuild that mask.

The crop is the slide's pixels inside the feather mask, histogram-equalized,
on black; edges survive the equalization, so masked edge matching finds it.
"""
from __future__ import annotations

import cv2
import numpy as np


def locate(slide_gray: np.ndarray, crop_bgr: np.ndarray, scale: float = 0.25) -> tuple[int, int, float]:
    """Top-left (x, y) of the crop on the slide and the match score (0-1)."""
    cg = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    m = (cg > 8).astype(np.uint8)
    s_small = cv2.resize(slide_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    c_small = cv2.resize(cg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m_small = cv2.resize(m, (c_small.shape[1], c_small.shape[0]), interpolation=cv2.INTER_NEAREST)
    e_s = cv2.Canny(s_small, 30, 90).astype(np.float32)
    e_c = cv2.Canny(c_small, 30, 90).astype(np.float32) * m_small
    res = cv2.matchTemplate(e_s, e_c, cv2.TM_CCORR_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    X, Y = int(x / scale), int(y / scale)
    r = int(3 / scale)
    y0, x0 = max(0, Y - r), max(0, X - r)
    win = slide_gray[y0:Y + r + cg.shape[0], x0:X + r + cg.shape[1]]
    if win.shape[0] >= cg.shape[0] and win.shape[1] >= cg.shape[1]:
        res2 = cv2.matchTemplate(cv2.Canny(win, 30, 90).astype(np.float32),
                                 cv2.Canny(cg, 30, 90).astype(np.float32) * m, cv2.TM_CCORR_NORMED)
        _, score, _, (dx, dy) = cv2.minMaxLoc(res2)
        X, Y = x0 + dx, y0 + dy
    return X, Y, float(score)


def crop_mask(slide_bgr: np.ndarray, crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Slide-sized bool mask of the crop's feather, and the match score."""
    gray = cv2.cvtColor(slide_bgr, cv2.COLOR_BGR2GRAY)
    x, y, score = locate(gray, crop_bgr)
    cm = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) > 8
    mask = np.zeros(gray.shape, bool)
    h, w = min(cm.shape[0], gray.shape[0] - y), min(cm.shape[1], gray.shape[1] - x)
    mask[y:y + h, x:x + w] = cm[:h, :w]
    return mask, score
