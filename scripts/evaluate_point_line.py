"""Compare SIFT/OpenCV RANSAC with point-line H on one saved near-4K ROI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_common import image_metrics, read, sift_homography, write_jpeg
from point_line_stitch import compose_local_roi, load_homography


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, type=Path)
    args = parser.parse_args()
    example = args.example.resolve()
    metadata = json.loads((example / "output/metadata.json").read_text(encoding="utf-8"))
    roi_output = metadata["local_roi_output"]
    roi = tuple(int(value) for value in roi_output["global_roi_xyxy"])
    scale = int(roi_output["scale"])
    if scale != 6:
        raise ValueError(f"Expected the saved x6 result, got x{scale}")
    ix, iy, iw, ih = (int(value) for value in metadata["initial_projection_roi_xywh"])
    initial_roi = (ix, iy, ix + iw, iy + ih)
    local = read(example / "input/local.jpg")
    global_frame = read(example / "input/global.jpg")
    h_initial = load_homography(example / "input/H_init.npz")
    h_pure = np.asarray(metadata["homography_source_to_destination"], dtype=np.float64)
    h_sift, match_count, inlier_count = sift_homography(
        local, global_frame, h_initial, initial_roi,
    )
    background = read(example / "output/global_roi_lanczos_x6.jpg")
    pure = read(example / "output/local_roi_x6.jpg")
    sift = compose_local_roi(
        local, global_frame, h_sift,
        scale=scale, roi_xyxy=roi, background_roi=background, feather_width=6.0,
    ).image
    if sift.shape != pure.shape or pure.shape != background.shape:
        raise ValueError("SIFT, point-line and Global ROI dimensions differ")
    benchmark_dir = example / "output/benchmark"
    sift_path = benchmark_dir / "sift_ransac_local_roi_x6.jpg"
    write_jpeg(sift_path, sift)
    sift_saved = read(sift_path)
    result = {
        "sample": example.name,
        "reference": "output/global_roi_lanczos_x6.jpg",
        "shared_roi_native_xyxy": list(roi),
        "roi_size": [background.shape[1], background.shape[0]],
        "scale": scale,
        "feather_radius_native_px": 6,
        "sift_initial_roi_native_xyxy": list(initial_roi),
        "methods": {
            "SIFT_RANSAC_H": {
                **image_metrics(background, sift_saved),
                "ratio_test_matches": match_count,
                "ransac_inliers": inlier_count,
                "x6_crop": str(sift_path.relative_to(example)),
            },
            "OmniGlue_LineTR_H": {
                **image_metrics(background, pure),
                "x6_crop": "output/local_roi_x6.jpg",
            },
        },
    }
    path = benchmark_dir / "metrics.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
