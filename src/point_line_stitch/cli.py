"""CLI for OmniGlue + LineTR homography image stitching."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2 as cv

from .core import load_homography, stitch_pair
from .high_resolution import compose_local_roi
from .matchers import LineTRLineMatcher, OmniGluePointMatcher
from .types import HomographyEstimationError
from .visualization import (
    draw_final_alignment,
    draw_initial_alignment,
    draw_matches,
    final_alignment_views,
    initial_alignment_views,
    projected_source_bounds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate a point-line homography and stitch source into destination.")
    parser.add_argument("source", type=Path, help="local/source image")
    parser.add_argument("destination", type=Path, help="global/reference image")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, help="metadata and match visualization")
    parser.add_argument("--local-roi-output", type=Path, help="near-4K Local-area result path (default: next to --output)")
    parser.add_argument("--local-roi-scale", type=int, default=6)
    parser.add_argument("--local-roi-margin", type=int, default=24)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--omniglue-root", type=Path, default=Path("third_party/omniglue"))
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
    parser.add_argument("--feather-width", type=float, default=6.0)
    parser.add_argument("--feather-power", type=float, default=1.0)
    parser.add_argument("--no-color-correction", action="store_true", help="disable default HHM Local-to-Global color correction")
    parser.add_argument("--color-correction-coeff", type=float, default=1.0)
    parser.add_argument("--color-correction-threshold", type=float, default=0.05)
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
        "blending": {
            "mode": "local_edge_feather",
            "feather_width_pixels": args.feather_width,
            "feather_power": args.feather_power,
        },
        "color_correction": result.color_correction,
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
        payload["initial_h_input"] = str(args.initial_h)
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
    point_matcher = OmniGluePointMatcher(
        args.model_dir,
        args.confidence,
        omniglue_root=args.omniglue_root,
    )
    line_matcher = LineTRLineMatcher(args.linetr_root, device=args.device)
    print("Matching points with OmniGlue and lines with LineTR...")
    try:
        result = stitch_pair(
            source, destination, point_matcher, line_matcher,
            initial_homography=initial_h,
            feather_width=args.feather_width,
            feather_power=args.feather_power,
            color_correction=not args.no_color_correction,
            color_correction_coeff=args.color_correction_coeff,
            color_correction_threshold=args.color_correction_threshold,
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
    local_roi_output = args.local_roi_output or args.output.parent / f"local_roi_x{args.local_roi_scale}.jpg"
    local_roi_result = compose_local_roi(
        source, destination, result.estimation.homography,
        scale=args.local_roi_scale,
        margin=args.local_roi_margin,
        feather_width=args.feather_width,
        feather_power=args.feather_power,
        color_correction=not args.no_color_correction,
        color_correction_coeff=args.color_correction_coeff,
        color_correction_threshold=args.color_correction_threshold,
    )
    local_roi_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(local_roi_output), local_roi_result.image):
        raise SystemExit(f"error: could not write {local_roi_output}")
    background_output = local_roi_output.parent / f"global_roi_lanczos_x{args.local_roi_scale}.jpg"
    if not cv.imwrite(str(background_output), local_roi_result.background):
        raise SystemExit(f"error: could not write {background_output}")
    if args.artifacts_dir:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        metadata = _metadata(result, args)
        metadata["local_roi_output"] = {
            "path": str(local_roi_output),
            "global_roi_xyxy": list(local_roi_result.global_roi_xyxy),
            "scale": args.local_roi_scale,
            "size": [local_roi_result.image.shape[1], local_roi_result.image.shape[0]],
            "background": "Global Lanczos interpolation",
            "background_path": str(background_output),
            "color_correction": local_roi_result.color_correction,
        }
        match_source = source
        match_region = None
        if initial_h is not None:
            match_source = cv.warpPerspective(source, initial_h, (destination.shape[1], destination.shape[0]))
            cv.imwrite(str(args.artifacts_dir / "initial_warp.jpg"), match_source)
            match_region = projected_source_bounds(
                source, destination, initial_h, padding=32
            )
            metadata["initial_projection_roi_xywh"] = list(match_region)
            initial_alignment = draw_initial_alignment(
                source, destination, initial_h, padding=32
            )
            cv.imwrite(
                str(args.artifacts_dir / "initial_alignment.jpg"),
                initial_alignment,
            )
            initial_overview, initial_zoom = initial_alignment_views(
                source, destination, initial_h, padding=32
            )
            cv.imwrite(str(args.artifacts_dir / "initial_overview.jpg"), initial_overview)
            cv.imwrite(str(args.artifacts_dir / "initial_zoom.jpg"), initial_zoom)
        visualization = draw_matches(
            match_source,
            destination,
            result.estimation.filtered_matches,
            region=match_region,
            max_points=40,
            max_lines=12,
        )
        cv.imwrite(str(args.artifacts_dir / "point_line_matches.jpg"), visualization)
        final_alignment = draw_final_alignment(
            result.panorama, result.source_mask, padding=32
        )
        cv.imwrite(
            str(args.artifacts_dir / "final_alignment.jpg"),
            final_alignment,
        )
        final_overview, final_zoom = final_alignment_views(
            result.panorama, result.source_mask, padding=32
        )
        cv.imwrite(str(args.artifacts_dir / "final_overview.jpg"), final_overview)
        cv.imwrite(str(args.artifacts_dir / "final_zoom.jpg"), final_zoom)
        final_region = projected_source_bounds(
            source,
            result.panorama,
            result.source_to_canvas,
            padding=32,
        )
        metadata["final_projection_roi_canvas_xywh"] = list(final_region)
        (args.artifacts_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    diagnostics = result.estimation.diagnostics
    print(f"Wrote {args.output} ({result.panorama.shape[1]}x{result.panorama.shape[0]})")
    if result.color_correction:
        print(
            "HHM color correction: "
            f"{result.color_correction.get('reason')} "
            f"using {result.color_correction.get('paired_pixels_sampled', 0)} paired overlap pixels"
        )
    print(f"Accepted {diagnostics.points_after_filter}/{diagnostics.points_before_filter} points, {diagnostics.lines_after_filter}/{diagnostics.lines_before_filter} lines; point RMSE={diagnostics.point_rmse!s}, line RMSE={diagnostics.line_rmse!s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
