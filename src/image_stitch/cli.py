"""CLI for OmniGlue + LineTR homography image stitching."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2 as cv

from .core import load_homography, stitch_pair
from .matchers import LineTRLineMatcher, OmniGluePointMatcher
from .types import HomographyEstimationError
from .visualization import draw_matches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate a point-line homography and stitch source into destination.")
    parser.add_argument("source", type=Path, help="local/source image")
    parser.add_argument("destination", type=Path, help="global/reference image")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, help="metadata and match visualization")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--linetr-root", type=Path, default=Path("third_party/LineTR"))
    parser.add_argument("--initial-h", type=Path, help="optional H_init JSON or NumPy file")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--gpu", default="0", help="physical GPU index exposed to both frameworks")
    parser.add_argument("--confidence", type=float, default=0.1, help="OmniGlue threshold")
    parser.add_argument("--max-match-width", type=int, default=1200)
    parser.add_argument("--point-weight", type=float, default=0.5)
    parser.add_argument("--line-weight", type=float, default=0.5)
    parser.add_argument("--ransac-threshold", type=float, default=4.0)
    parser.add_argument("--line-distance-threshold", type=float, default=12.0)
    parser.add_argument("--line-angle-threshold", type=float, default=15.0)
    parser.add_argument("--feather-power", type=float, default=1.0)
    return parser


def _read_image(path: Path):
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"error: could not read image: {path}")
    return image


def _metadata(result, args) -> dict:
    estimation = result.estimation
    diagnostics = estimation.diagnostics
    payload = {
        "source": str(args.source),
        "destination": str(args.destination),
        "mode": "refine_initial_h" if args.initial_h else "direct_point_line",
        "homography_source_to_destination": estimation.homography.tolist(),
        "source_to_canvas": result.source_to_canvas.tolist(),
        "destination_to_canvas": result.destination_to_canvas.tolist(),
        "panorama_size": [result.panorama.shape[1], result.panorama.shape[0]],
        "matching": {
            "omniglue_confidence_threshold": args.confidence,
            "points_before_filter": diagnostics.points_before_filter,
            "points_after_filter": diagnostics.points_after_filter,
            "lines_before_filter": diagnostics.lines_before_filter,
            "lines_after_filter": diagnostics.lines_after_filter,
            "point_rmse": diagnostics.point_rmse,
            "line_rmse": diagnostics.line_rmse,
            "optimization_iterations": diagnostics.iterations,
            "point_weight": args.point_weight,
            "line_weight": args.line_weight,
        },
    }
    if estimation.initial_homography is not None:
        payload["H_init"] = estimation.initial_homography.tolist()
        payload["H_delta"] = estimation.delta_homography.tolist()
        payload["H_final"] = estimation.homography.tolist()
    return payload


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Must happen before the lazy TensorFlow/PyTorch imports in the matchers.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = _read_image(args.source)
    destination = _read_image(args.destination)
    initial_h = load_homography(args.initial_h) if args.initial_h else None
    point_matcher = OmniGluePointMatcher(args.model_dir, args.confidence)
    line_matcher = LineTRLineMatcher(args.linetr_root, device=args.device)
    print("Matching points with OmniGlue and lines with LineTR...")
    try:
        result = stitch_pair(
            source, destination, point_matcher, line_matcher,
            initial_homography=initial_h,
            feather_power=args.feather_power,
            max_match_width=args.max_match_width,
            point_weight=args.point_weight,
            line_weight=args.line_weight,
            ransac_threshold=args.ransac_threshold,
            line_distance_threshold=args.line_distance_threshold,
            line_angle_threshold=args.line_angle_threshold,
        )
    except (FileNotFoundError, ImportError, ValueError, HomographyEstimationError) as error:
        raise SystemExit(f"error: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(args.output), result.panorama):
        raise SystemExit(f"error: could not write {args.output}")
    if args.artifacts_dir:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (args.artifacts_dir / "metadata.json").write_text(json.dumps(_metadata(result, args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        match_source = source
        if initial_h is not None:
            match_source = cv.warpPerspective(source, initial_h, (destination.shape[1], destination.shape[0]))
            cv.imwrite(str(args.artifacts_dir / "initial_warp.jpg"), match_source)
        visualization = draw_matches(match_source, destination, result.estimation.filtered_matches)
        cv.imwrite(str(args.artifacts_dir / "point_line_matches.jpg"), visualization)
    diagnostics = result.estimation.diagnostics
    print(f"Wrote {args.output} ({result.panorama.shape[1]}x{result.panorama.shape[0]})")
    print(f"Accepted {diagnostics.points_after_filter}/{diagnostics.points_before_filter} points, {diagnostics.lines_after_filter}/{diagnostics.lines_before_filter} lines; point RMSE={diagnostics.point_rmse!s}, line RMSE={diagnostics.line_rmse!s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
