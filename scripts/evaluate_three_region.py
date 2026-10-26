"""Compare image-stitch, SIFT/RANSAC, and RAFT on the same static pixels.

This is a small diagnostic on the included videos, not an official UDIS-D
benchmark. No ground-truth panorama exists for these cross-view frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity

from image_stitch import load_homography
from three_region_stitch.regions import compose_homography_over_background, prepare_regions
from three_region_stitch.visualization import compare_same_crop


def read(path: Path, flags: int = cv.IMREAD_COLOR) -> np.ndarray:
    image = cv.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def sift_homography(local: np.ndarray, global_frame: np.ndarray,
                    h_initial: np.ndarray, roi: tuple[int, int, int, int]):
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


def warp_with_valid(local: np.ndarray, h: np.ndarray, size: tuple[int, int]):
    warped = cv.warpPerspective(local, h, size)
    valid = cv.warpPerspective(
        np.full(local.shape[:2], 255, np.uint8), h, size, flags=cv.INTER_NEAREST,
    )
    return warped, valid


def feather(local_warp: np.ndarray, global_frame: np.ndarray,
            local_valid: np.ndarray, width: int = 32) -> np.ndarray:
    padded = cv.copyMakeBorder(local_valid, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    distance = cv.distanceTransform(padded, cv.DIST_L2, 5)[1:-1, 1:-1]
    alpha = np.clip(distance / width, 0, 1)[..., None]
    return np.clip(
        local_warp.astype(np.float32) * alpha
        + global_frame.astype(np.float32) * (1 - alpha), 0, 255,
    ).astype(np.uint8)


def masked_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict:
    squared = (reference.astype(np.float32) - candidate.astype(np.float32)) ** 2
    mse = float(np.mean(squared[mask]))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))
    _, map_ssim = structural_similarity(
        reference, candidate, channel_axis=2, data_range=255, full=True,
    )
    ssim = float(np.mean(map_ssim.mean(axis=2)[mask]))
    return {"psnr_db": round(psnr, 3), "ssim": round(ssim, 4)}


def error_heatmap_comparison(
    reference: np.ndarray,
    pure: np.ndarray,
    three_region: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Same-scale RGB error against Global over every pixel in the sample ROI."""
    x0, y0, x1, y1 = roi
    panels = []
    for label, candidate in (("Pure point-line stitch", pure), ("Three-region result", three_region)):
        error = np.abs(reference.astype(np.float32) - candidate.astype(np.float32)).mean(2)
        normalized = (np.clip(error[y0:y1, x0:x1], 0, 80) * 255 / 80).astype(np.uint8)
        colored = cv.applyColorMap(normalized, cv.COLORMAP_TURBO)
        header = np.zeros((40, x1 - x0, 3), np.uint8)
        cv.putText(header, label, (10, 27), cv.FONT_HERSHEY_SIMPLEX,
                   0.75, (255, 255, 255), 2, cv.LINE_AA)
        panels.append(np.vstack((header, colored)))
    footer = np.zeros((30, 2 * (x1 - x0), 3), np.uint8)
    cv.putText(footer, "Mean absolute RGB error vs Global over all ROI pixels: blue=0, red>=80",
               (10, 22), cv.FONT_HERSHEY_SIMPLEX, 0.48,
               (255, 255, 255), 1, cv.LINE_AA)
    return np.vstack((np.hstack(panels), footer))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--three-output", type=Path,
                        help="default: EXAMPLE/output/three_region")
    args = parser.parse_args()
    example = args.example.resolve()
    manifest = json.loads((example / "manifest.json").read_text(encoding="utf-8"))
    roi = tuple(manifest["initial_projection_roi_sample_xyxy"])
    global_frame = read(example / "input/global.jpg")
    local = read(example / "input/local.jpg")
    h_initial = load_homography(example / "input/H_init.npz")
    pure_metadata = json.loads((example / "output/pure_artifacts/metadata.json").read_text())
    h_pure = np.asarray(pure_metadata["H_final"], dtype=np.float64)
    h_sift, sift_matches, sift_inliers = sift_homography(local, global_frame, h_initial, roi)
    size = (global_frame.shape[1], global_frame.shape[0])
    pure_warp, pure_valid = warp_with_valid(local, h_pure, size)
    sift_warp, sift_valid = warp_with_valid(local, h_sift, size)
    three_output = args.three_output or example / "output/three_region"
    three_summary = json.loads((three_output / "summary.json").read_text())
    h_three = np.asarray(three_summary["H_final"], dtype=np.float64)
    local_mask = read(three_output / "local_motion_init_mask.png", cv.IMREAD_GRAYSCALE)
    global_mask = read(three_output / "global_motion_mask.png", cv.IMREAD_GRAYSCALE)
    prepared = prepare_regions(
        global_frame, local, h_initial, h_three, roi, local_mask, global_mask,
        one_sided_to_raft=False,
    )
    raft_warped = read(three_output / "raft_warped.jpg")
    raft_valid = read(three_output / "raft_valid_mask.png", cv.IMREAD_GRAYSCALE)
    common = (
        (prepared.static_candidate > 0) & (pure_valid > 0)
        & (sift_valid > 0) & (raft_valid > 0)
    ).astype(np.uint8)
    common = cv.erode(common, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))) > 0
    if np.count_nonzero(common) < 100:
        raise RuntimeError("Too few jointly valid static pixels for this evaluation")
    metrics = {
        "scene": manifest["scene"],
        "static_common_pixels": int(np.count_nonzero(common)),
        "static_overlap_protocol": "intersection of non-motion Local/Global and valid pixels from all methods, eroded by 3 px",
        "methods": {
            "SIFT_RANSAC_H": {
                **masked_metrics(global_frame, sift_warp, common),
                "ratio_test_matches": sift_matches,
                "ransac_inliers": sift_inliers,
            },
            "OmniGlue_LineTR_H": masked_metrics(global_frame, pure_warp, common),
            "motion_H_plus_RAFT_static": masked_metrics(global_frame, raft_warped, common),
        },
        "interpretation_limit": "Static-overlap alignment only; no ground-truth panorama or motion-region perceptual quality score.",
    }
    benchmark_dir = example / "output/benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    sift_result = feather(sift_warp, global_frame, sift_valid)
    cv.imwrite(str(benchmark_dir / "sift_ransac_stitch.jpg"), sift_result,
               [cv.IMWRITE_JPEG_QUALITY, 95])
    three_result = read(three_output / "final_result.jpg")
    cv.imwrite(str(benchmark_dir / "sift_vs_three_focus.jpg"),
               compare_same_crop(
                   sift_result, three_result, roi,
                   ("SIFT + RANSAC H", "Motion + RAFT + Global"),
               ),
               [cv.IMWRITE_JPEG_QUALITY, 95])
    cv.imwrite(str(benchmark_dir / "static_common_mask.png"), common.astype(np.uint8) * 255)
    pure_result = compose_homography_over_background(
        global_frame, local, h_pure, feather_radius=int(three_summary.get("feather_radius", 2)),
    )
    three_result = read(three_output / "final_result.jpg")
    cv.imwrite(
        str(benchmark_dir / "all_region_error_heatmaps.jpg"),
        error_heatmap_comparison(global_frame, pure_result, three_result, roi),
        [cv.IMWRITE_JPEG_QUALITY, 95],
    )
    (benchmark_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
