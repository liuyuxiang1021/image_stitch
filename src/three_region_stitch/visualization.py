"""Pixel-faithful overlays and same-crop comparison panels."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .regions import RegionMasks


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    result = image.copy()
    colored = np.full_like(image, color)
    blended = cv.addWeighted(image, 1 - alpha, colored, alpha, 0)
    result[mask > 0] = blended[mask > 0]
    return result


def region_overlay(background: np.ndarray, masks: RegionMasks) -> np.ndarray:
    result = background.copy()
    for mask, color, alpha in (
        (masks.global_region, (0, 180, 0), 0.22),
        (masks.raft_region, (255, 0, 0), 0.34),
        (masks.homography_region, (0, 0, 255), 0.34),
    ):
        result = tint(result, mask, color, alpha)
    return result


def focus_bbox(roi: tuple[int, int, int, int], shape: tuple[int, int, int], padding: int = 32):
    x0, y0, x1, y1 = roi
    return (
        max(0, x0 - padding), max(0, y0 - padding),
        min(shape[1], x1 + padding), min(shape[0], y1 + padding),
    )


def compare_same_crop(
    pure: np.ndarray,
    three_region: np.ndarray,
    roi: tuple[int, int, int, int],
    titles: tuple[str, str] = ("Point-line H + feather", "Motion + RAFT + Global"),
) -> np.ndarray:
    if pure.shape != three_region.shape:
        raise ValueError("Comparison images must be on the same Global canvas")
    x0, y0, x1, y1 = focus_bbox(roi, pure.shape)
    bar = 36
    panels = []
    for image, title in zip((pure, three_region), titles):
        crop = image[y0:y1, x0:x1]
        panel = np.zeros((crop.shape[0] + bar, crop.shape[1], 3), dtype=np.uint8)
        panel[bar:] = crop
        cv.putText(panel, title, (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                   0.65, (255, 255, 255), 2, cv.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)
