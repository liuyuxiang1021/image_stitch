"""Point and line H_delta estimation gated by independent motion masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from point_line_stitch.core import collect_matches
from point_line_stitch.optimization import estimate_point_line_homography
from point_line_stitch.types import FeatureMatches, HomographyEstimationError


@dataclass(frozen=True)
class MotionGeometry:
    h_delta: np.ndarray
    h_final: np.ndarray
    points_before_mask: int
    points_after_mask: int
    lines_before_mask: int
    lines_after_mask: int
    point_rmse: float | None
    line_rmse: float | None
    fallback_reason: str | None = None


def _points_inside(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.zeros(0, dtype=bool)
    x = np.rint(points[:, 0]).astype(np.int32)
    y = np.rint(points[:, 1]).astype(np.int32)
    inside = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    result = np.zeros(len(points), dtype=bool)
    result[inside] = mask[y[inside], x[inside]] > 0
    return result


def filter_dual_motion_matches(
    matches: FeatureMatches,
    local_mask: np.ndarray,
    global_mask: np.ndarray,
) -> FeatureMatches:
    """Require each corresponding point or line midpoint in its own mask."""
    points_keep = _points_inside(matches.points_source, local_mask) & _points_inside(
        matches.points_destination, global_mask,
    )
    lines_keep = _points_inside(matches.lines_source.mean(axis=1), local_mask) & _points_inside(
        matches.lines_destination.mean(axis=1), global_mask,
    )
    return FeatureMatches(
        matches.points_source[points_keep],
        matches.points_destination[points_keep],
        matches.point_scores[points_keep],
        matches.lines_source[lines_keep],
        matches.lines_destination[lines_keep],
        matches.line_scores[lines_keep],
    )


def estimate_motion_homography(
    local_frame: np.ndarray,
    global_frame: np.ndarray,
    h_initial: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    local_motion_roi: np.ndarray,
    global_motion_roi: np.ndarray,
    point_matcher,
    line_matcher,
) -> MotionGeometry:
    """Match the H_init projection in ROI coordinates and return H_delta @ H_init."""
    x0, y0, x1, y1 = roi_xyxy
    height, width = global_frame.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Motion ROI is outside the Global frame")
    roi_shape = (y1 - y0, x1 - x0)
    if local_motion_roi.shape != roi_shape or global_motion_roi.shape != roi_shape:
        raise ValueError("Motion masks must match the initial-projection ROI")
    local_warp = cv.warpPerspective(local_frame, h_initial, (width, height))
    local_crop = local_warp[y0:y1, x0:x1]
    global_crop = global_frame[y0:y1, x0:x1]
    matches = collect_matches(local_crop, global_crop, point_matcher, line_matcher)
    local_valid = (cv.cvtColor(local_crop, cv.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    selected = filter_dual_motion_matches(
        matches,
        cv.bitwise_and(local_motion_roi, local_valid),
        global_motion_roi,
    )
    counts = (
        len(matches.points_source), len(selected.points_source),
        len(matches.lines_source), len(selected.lines_source),
    )
    try:
        h_crop, _, diagnostics = estimate_point_line_homography(selected)
    except HomographyEstimationError as error:
        identity = np.eye(3, dtype=np.float64)
        return MotionGeometry(identity, h_initial.copy(), *counts, None, None, str(error))
    global_to_roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    h_delta = np.linalg.inv(global_to_roi) @ h_crop @ global_to_roi
    h_final = h_delta @ h_initial
    if not np.isfinite(h_final).all() or abs(np.linalg.det(h_final)) < 1e-12:
        raise HomographyEstimationError("Motion-masked H_final is singular or non-finite")
    return MotionGeometry(
        h_delta, h_final, *counts, diagnostics.point_rmse, diagnostics.line_rmse,
    )
