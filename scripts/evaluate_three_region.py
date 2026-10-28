"""Evaluate x6 stitching results over the shared near-4K Local ROI.

The included examples have no ground-truth panorama. These scores compare each
result with the corresponding x6 Global background and are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from point_line_stitch import load_homography
from three_region_stitch.regions import compose_homography_over_background
from three_region_stitch.visualization import compare_same_crop
from benchmark_common import image_metrics, read, sift_homography, write_jpeg


def crop_scaled(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    scale: int,
) -> np.ndarray:
    x0, y0, x1, y1 = roi_xyxy
    return image[y0 * scale:y1 * scale, x0 * scale:x1 * scale]


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
    h_pure = np.asarray(
        pure_metadata.get("H_final", pure_metadata["homography_source_to_destination"]),
        dtype=np.float64,
    )
    h_sift, sift_matches, sift_inliers = sift_homography(local, global_frame, h_initial, input_roi)

    three_output = args.three_output or example / "output/three_region"
    three_summary = json.loads((three_output / "summary.json").read_text())
    scale = int(three_summary["super_resolution"]["scale"])
    feather_radius = int(three_summary.get("feather_radius", 6))
    roi = tuple(int(value) for value in three_summary["local_roi_outputs"]["native_xyxy"])
    x0, y0, x1, y1 = roi
    roi_x6 = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
    roi_width = (x1 - x0) * scale
    roi_height = (y1 - y0) * scale

    # Keep native-resolution diagnostic outputs separate from the x6 near-4K metrics.
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

    (benchmark_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
