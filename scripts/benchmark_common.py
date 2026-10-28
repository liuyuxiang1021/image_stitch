"""Shared SIFT/OpenCV homography baseline and image metrics for the examples."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity


def read(path: Path, flags: int = cv.IMREAD_COLOR) -> np.ndarray:
    image = cv.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image, [cv.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {path}")


def sift_homography(
    local: np.ndarray,
    global_frame: np.ndarray,
    h_initial: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, int, int]:
    """Refine the same calibration start with SIFT, BFMatcher and OpenCV RANSAC."""
    x0, y0, x1, y1 = roi
    warped = cv.warpPerspective(local, h_initial, (global_frame.shape[1], global_frame.shape[0]))
    source = cv.cvtColor(warped[y0:y1, x0:x1], cv.COLOR_BGR2GRAY)
    target = cv.cvtColor(global_frame[y0:y1, x0:x1], cv.COLOR_BGR2GRAY)
    detector = cv.SIFT_create(nfeatures=4000)
    source_keypoints, source_descriptors = detector.detectAndCompute(source, None)
    target_keypoints, target_descriptors = detector.detectAndCompute(target, None)
    if source_descriptors is None or target_descriptors is None:
        raise RuntimeError("SIFT detected no descriptors in one view")
    matches = cv.BFMatcher(cv.NORM_L2).knnMatch(source_descriptors, target_descriptors, k=2)
    good = [first for first, second in matches if first.distance < 0.75 * second.distance]
    if len(good) < 4:
        raise RuntimeError(f"SIFT has only {len(good)} ratio-test matches")
    source_points = np.float32([source_keypoints[item.queryIdx].pt for item in good])
    target_points = np.float32([target_keypoints[item.trainIdx].pt for item in good])
    h_roi, inliers = cv.findHomography(source_points, target_points, cv.RANSAC, 4.0)
    if h_roi is None or inliers is None or inliers.sum() < 4:
        raise RuntimeError("SIFT/RANSAC could not estimate a homography")
    full_to_roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    h_final = np.linalg.inv(full_to_roi) @ h_roi @ full_to_roi @ h_initial
    return h_final, len(good), int(inliers.sum())


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Metric image shapes differ: {reference.shape} vs {candidate.shape}")
    delta = reference.astype(np.float32) - candidate.astype(np.float32)
    mse = float(np.mean(delta * delta))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
    ssim = float(structural_similarity(
        reference, candidate, channel_axis=2, data_range=255,
    ))
    return {"psnr_db": round(psnr, 3), "ssim": round(ssim, 4)}
