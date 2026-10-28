"""Select a Local frame against one fixed Global frame in projected coordinates."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


def target_motion_mask(
    target: np.ndarray,
    neighbors: list[np.ndarray],
    valid: np.ndarray,
    *,
    threshold: int = 16,
    dilation: int = 11,
) -> np.ndarray:
    """Mark Global pixels that changed near the anchor frame and have Local coverage."""
    if not neighbors:
        raise ValueError("At least one neighboring Global frame is required")
    gray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    changed = np.zeros(gray.shape, dtype=np.uint8)
    for neighbor in neighbors:
        if neighbor.shape != target.shape:
            raise ValueError("Global neighbor size differs from the target")
        delta = cv.absdiff(gray, cv.cvtColor(neighbor, cv.COLOR_BGR2GRAY))
        changed |= (delta > threshold).astype(np.uint8)
    if dilation > 1:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilation, dilation))
        changed = cv.dilate(changed, kernel)
    valid_interior = cv.erode((valid > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
    return (changed & valid_interior).astype(np.uint8) * 255


def masked_ssim(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    """Mean grayscale SSIM inside a fixed mask; no retained Global pixels enter."""
    if reference.shape != candidate.shape or reference.shape[:2] != mask.shape[:2]:
        raise ValueError("Reference, candidate and mask sizes must agree")
    selected = mask > 0
    if np.count_nonzero(selected) < 100:
        raise ValueError("Too few scored pixels")
    a = cv.cvtColor(reference, cv.COLOR_BGR2GRAY).astype(np.float32)
    b = cv.cvtColor(candidate, cv.COLOR_BGR2GRAY).astype(np.float32)
    ma = cv.GaussianBlur(a, (11, 11), 1.5)
    mb = cv.GaussianBlur(b, (11, 11), 1.5)
    va = cv.GaussianBlur(a * a, (11, 11), 1.5) - ma * ma
    vb = cv.GaussianBlur(b * b, (11, 11), 1.5) - mb * mb
    cov = cv.GaussianBlur(a * b, (11, 11), 1.5) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    values = ((2 * ma * mb + c1) * (2 * cov + c2)) / (
        (ma * ma + mb * mb + c1) * (va + vb + c2) + 1e-8
    )
    return float(values[selected].mean())


def rank_projected_video(
    global_roi: np.ndarray,
    projected_local_video: Path,
    score_mask: np.ndarray,
    start_frame_index: int,
    anchor_global_index: int,
) -> list[dict]:
    """Score every projected Local frame against the same Global ROI and mask."""
    capture = cv.VideoCapture(str(projected_local_video))
    if not capture.isOpened():
        raise FileNotFoundError(projected_local_video)
    scores = []
    try:
        index = start_frame_index
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            score = masked_ssim(global_roi, frame, score_mask)
            scores.append({
                "local_frame_index": index,
                "offset_from_global_index": index - anchor_global_index,
                "masked_ssim": round(score, 7),
            })
            index += 1
    finally:
        capture.release()
    if not scores:
        raise ValueError(f"No frames in {projected_local_video}")
    return scores


def best_candidate(scores: list[dict], anchor_global_index: int) -> dict:
    """Choose the highest score, breaking ties toward the anchor frame."""
    if not scores:
        raise ValueError("No synchronization candidates")
    return max(scores, key=lambda item: (
        item["masked_ssim"],
        -abs(item["local_frame_index"] - anchor_global_index),
        -item["local_frame_index"],
    ))


def select_with_confidence(
    scores: list[dict], anchor_global_index: int, current_local_index: int,
    *, min_gain: float = 0.01,
) -> dict:
    """Apply a clear interior peak; otherwise retain the existing Local frame."""
    best = best_candidate(scores, anchor_global_index)
    current = next((item for item in scores
                    if item["local_frame_index"] == current_local_index), None)
    if current is None:
        raise ValueError("Current Local frame is outside the search window")
    gain = best["masked_ssim"] - current["masked_ssim"]
    at_boundary = best["local_frame_index"] in (
        scores[0]["local_frame_index"], scores[-1]["local_frame_index"],
    )
    if at_boundary:
        reason = "peak_at_search_boundary"
    elif gain < min_gain:
        reason = "gain_below_threshold"
    else:
        reason = "clear_interior_peak"
    return {
        "best_raw": best,
        "selected": best if reason == "clear_interior_peak" else current,
        "current": current,
        "score_gain_over_current": round(gain, 7),
        "minimum_required_gain": min_gain,
        "decision": reason,
    }
