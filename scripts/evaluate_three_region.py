"""Evaluate x6 stitching results over the shared near-4K Local ROI.

The included examples have no ground-truth panorama. These scores compare each
result with the corresponding x6 Global background and are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity

from image_stitch import load_homography
from three_region_stitch.regions import compose_homography_over_background
from three_region_stitch.visualization import compare_same_crop


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


def crop_scaled(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    scale: int,
) -> np.ndarray:
    x0, y0, x1, y1 = roi_xyxy
    return image[y0 * scale:y1 * scale, x0 * scale:x1 * scale]


def error_heatmap(
    reference: np.ndarray,
    candidate: np.ndarray,
    max_error: float = 80.0,
) -> np.ndarray:
    """Return an unframed color map of mean absolute RGB error."""
    if reference.shape != candidate.shape:
        raise ValueError(f"Heatmap image shapes differ: {reference.shape} vs {candidate.shape}")
    error = np.abs(reference.astype(np.float32) - candidate.astype(np.float32)).mean(axis=2)
    normalized = (np.clip(error, 0, max_error) * 255 / max_error).astype(np.uint8)
    return cv.applyColorMap(normalized, cv.COLORMAP_TURBO)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--three-output", type=Path,
                        help="default: EXAMPLE/output/three_region")
    args = parser.parse_args()
    example = args.example.resolve()
    manifest = json.loads((example / "manifest.json").read_text(encoding="utf-8"))
    input_roi = tuple(int(value) for value in manifest["initial_projection_roi_sample_xyxy"])
    global_frame = read(example / "input/global.jpg")
    local = read(example / "input/local.jpg")
    h_initial = load_homography(example / "input/H_init.npz")
    pure_metadata = json.loads((example / "output/pure_artifacts/metadata.json").read_text())
    h_pure = np.asarray(pure_metadata["H_final"], dtype=np.float64)
    h_sift, sift_matches, sift_inliers = sift_homography(local, global_frame, h_initial, input_roi)

    three_output = args.three_output or example / "output/three_region"
    three_summary = json.loads((three_output / "summary.json").read_text())
    scale = int(three_summary["super_resolution"]["scale"])
    feather_radius = int(three_summary.get("feather_radius", 2))
    roi = tuple(int(value) for value in three_summary["local_roi_outputs"]["native_xyxy"])
    x0, y0, x1, y1 = roi
    roi_x6 = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
    roi_width = (x1 - x0) * scale
    roi_height = (y1 - y0) * scale

    # Keep the existing native-resolution diagnostic output, separate from the
    # x6 near-4K metrics and heatmaps below.
    pure_native = compose_homography_over_background(global_frame, local, h_pure, feather_radius)
    sift_native = compose_homography_over_background(global_frame, local, h_sift, feather_radius)
    three_native = read(three_output / "final_result.jpg")
    benchmark_dir = example / "output/benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    write_jpeg(benchmark_dir / "sift_ransac_stitch.jpg", sift_native)
    write_jpeg(
        benchmark_dir / "sift_vs_three_focus.jpg",
        compare_same_crop(
            sift_native, three_native, input_roi,
            ("SIFT + RANSAC H", "Motion + RAFT + Global"),
        ),
    )
    del pure_native, sift_native, three_native

    global_x6 = read(three_output / "global_background_fsrcnn_x6.jpg")
    if global_x6.shape[1] % global_frame.shape[1] != 0:
        raise ValueError("x6 Global background does not match the expected integer scale")
    actual_scale = global_x6.shape[1] // global_frame.shape[1]
    if actual_scale != scale:
        raise ValueError(f"Summary scale {scale} does not match x6 background scale {actual_scale}")
    reference_crop = crop_scaled(global_x6, roi, scale)
    if reference_crop.shape[:2] != (roi_height, roi_width):
        raise ValueError("Computed x6 ROI does not match the Global crop dimensions")

    # SIFT uses the same x6 Global background, ROI and scaled feather width as
    # the saved pure point-line and three-region results.
    h_sift_x6 = np.diag([float(scale), float(scale), 1.0]) @ h_sift
    sift_x6 = compose_homography_over_background(
        global_x6, local, h_sift_x6, feather_radius * scale,
    )
    sift_crop_path = benchmark_dir / "sift_ransac_local_roi_fsrcnn_x6.jpg"
    write_jpeg(sift_crop_path, crop_scaled(sift_x6, roi, scale))
    del sift_x6, global_x6
    sift_crop = read(sift_crop_path)

    pure_x6 = read(three_output / "pure_stitch_global_fsrcnn_x6.jpg")
    pure_crop = crop_scaled(pure_x6, roi, scale)
    del pure_x6
    three_x6 = read(three_output / "final_result_global_fsrcnn_x6.jpg")
    three_crop = crop_scaled(three_x6, roi, scale)
    del three_x6
    if any(crop.shape != reference_crop.shape for crop in (sift_crop, pure_crop, three_crop)):
        raise ValueError("All x6 method crops and the Global reference must have the same dimensions")

    metrics = {
        "scene": manifest["scene"],
        "all_roi_protocol": (
            "Compare the same near-4K crop from each x6 composite against the x6 Global background. "
            "The crop is the Local projection bounding box with 24 native-canvas pixels of margin; "
            "all methods use the same ROI and feather radius."
        ),
        "all_roi_reference": "output/three_region/global_background_fsrcnn_x6.jpg",
        "all_roi_native_xyxy": list(roi),
        "all_roi_x6_xyxy": list(roi_x6),
        "all_roi_width": roi_width,
        "all_roi_height": roi_height,
        "all_roi_pixels": roi_width * roi_height,
        "super_resolution_scale": scale,
        "feather_radius_native_px": feather_radius,
        "all_roi_methods": {
            "SIFT_RANSAC_H": {
                **image_metrics(reference_crop, sift_crop),
                "ratio_test_matches": sift_matches,
                "ransac_inliers": sift_inliers,
                "x6_crop": str(sift_crop_path.relative_to(example)),
            },
            "OmniGlue_LineTR_H": image_metrics(reference_crop, pure_crop),
            "three_region_final": image_metrics(reference_crop, three_crop),
        },
        "interpretation_limit": (
            "PSNR/SSIM measure similarity to the x6 Global reference over this ROI, not ground-truth "
            "stitching quality. Global-retained pixels in the three-region output match that reference "
            "by construction and affect these scores."
        ),
    }

    write_jpeg(
        benchmark_dir / "pure_stitch_error_heatmap_x6.jpg",
        error_heatmap(reference_crop, pure_crop),
    )
    write_jpeg(
        benchmark_dir / "three_region_error_heatmap_x6.jpg",
        error_heatmap(reference_crop, three_crop),
    )
    (benchmark_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
