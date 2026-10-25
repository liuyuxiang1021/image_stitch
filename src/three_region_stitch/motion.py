"""Independent Local and Global motion masks from short aligned ROI videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    segment_size: int = 100
    diff_threshold: int = 36
    open_kernel_size: int = 5
    close_kernel_size: int = 0
    min_area: int = 240


@dataclass(frozen=True)
class MotionResult:
    mask: np.ndarray
    frame_count: int
    segment_count: int


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    _, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
    keep = stats[:, cv.CC_STAT_AREA] >= min_area
    keep[0] = False
    return (keep[labels] * 255).astype(np.uint8)


def _refine(mask: np.ndarray, config: MotionConfig) -> np.ndarray:
    if config.open_kernel_size > 1:
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (config.open_kernel_size, config.open_kernel_size),
        )
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    if config.close_kernel_size > 1:
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (config.close_kernel_size, config.close_kernel_size),
        )
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return _remove_small_components(mask, config.min_area)


def detect_motion(video_path: str | Path, config: MotionConfig = MotionConfig()) -> MotionResult:
    """Median reference per segment, union differences, then union segments.

    Local video frames must already be projected by H_init into its Global ROI;
    Global video frames must be undistorted and cropped to the same ROI.
    """
    if config.segment_size < 2 or not 0 <= config.diff_threshold <= 255:
        raise ValueError("Invalid segment size or difference threshold")
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open motion ROI video: {video_path}")
    cumulative: np.ndarray | None = None
    frame_count = 0
    segment_count = 0
    try:
        while True:
            grays = []
            for _ in range(config.segment_size):
                success, frame = capture.read()
                if not success:
                    break
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                if cumulative is not None and gray.shape != cumulative.shape:
                    raise ValueError(f"Frame size changed in {video_path}")
                grays.append(gray)
            if not grays:
                break
            reference = np.median(np.stack(grays), axis=0).astype(np.uint8)
            segment_union = np.zeros_like(reference)
            for gray in grays:
                changed = cv.absdiff(gray, reference) >= config.diff_threshold
                segment_union[changed] = 255
            segment_union = _refine(segment_union, config)
            cumulative = (
                segment_union if cumulative is None
                else cv.bitwise_or(cumulative, segment_union)
            )
            frame_count += len(grays)
            segment_count += 1
    finally:
        capture.release()
    if cumulative is None:
        raise ValueError(f"No frames in motion ROI video: {video_path}")
    return MotionResult(cumulative, frame_count, segment_count)
