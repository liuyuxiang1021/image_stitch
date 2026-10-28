"""Run the three-region pipeline on a compact real-video example."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np

from image_stitch import load_homography, projected_local_roi

from .flow import RAFTFlow, align_static_region
from .geometry import estimate_motion_homography
from .motion import MotionConfig, detect_motion
from .regions import (
    DEFAULT_RAFT_MIN_COMPONENT_AREA,
    DEFAULT_RAFT_OPEN_KERNEL_SIZE,
    compose_homography_over_background,
    compose_three_regions,
    compose_with_global_sr,
    prepare_regions,
)
from .super_resolution import FSRCNNX4, FSRCNNX6, sharpen_for_display
from .visualization import region_overlay, tint


def _read_image(path: Path, flags: int = cv.IMREAD_COLOR) -> np.ndarray:
    image = cv.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = [cv.IMWRITE_JPEG_QUALITY, 95] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv.imwrite(str(path), image, options):
        raise RuntimeError(f"Could not write {path}")


def _last_video_frame(path: Path) -> np.ndarray:
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    capture.set(cv.CAP_PROP_POS_FRAMES, max(0, count - 1))
    success, frame = capture.read()
    capture.release()
    if not success:
        raise ValueError(f"Could not read last frame from {path}")
    return frame


def _pure_homography_to_global(path: Path) -> np.ndarray:
    metadata_path = path.parent / "pure_artifacts" / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Pure-stitch metadata is required for comparison: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_to_canvas = np.asarray(metadata["source_to_canvas"], dtype=np.float64)
    destination_to_canvas = np.asarray(metadata["destination_to_canvas"], dtype=np.float64)
    return np.linalg.inv(destination_to_canvas) @ source_to_canvas


def _local_roi_xyxy(
    local_shape: tuple[int, ...],
    homography: np.ndarray,
    global_shape: tuple[int, ...],
    margin: int = 24,
) -> tuple[int, int, int, int]:
    return projected_local_roi(local_shape, homography, global_shape, margin)


def _crop_scaled(image: np.ndarray, roi_xyxy: tuple[int, int, int, int], scale: int) -> np.ndarray:
    x0, y0, x1, y1 = roi_xyxy
    return image[y0 * scale:y1 * scale, x0 * scale:x1 * scale]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True, help="example directory with manifest.json")
    parser.add_argument("--output", type=Path, help="default: EXAMPLE/output/three_region")
    parser.add_argument("--motion-mode", choices=("cached", "recompute"), default="cached")
    parser.add_argument("--geometry-mode", choices=("cached", "recompute"), default="cached")
    parser.add_argument("--flow-mode", choices=("cached", "raft"), default="cached")
    parser.add_argument("--recompute-all", action="store_true", help="recompute masks, point-line H, and RAFT")
    parser.add_argument("--one-sided-to-raft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--feather-radius", type=int, default=6,
        help="shared Gaussian feather radius for three-region and pure-stitch comparison images",
    )
    parser.add_argument("--no-color-correction", action="store_true", help="disable HHM color correction")
    parser.add_argument("--color-correction-coeff", type=float, default=1.0)
    parser.add_argument("--color-correction-threshold", type=float, default=0.05)
    parser.add_argument("--color-correction-max-samples", type=int, default=400_000)
    parser.add_argument(
        "--raft-open-kernel-size", type=int, default=DEFAULT_RAFT_OPEN_KERNEL_SIZE,
        help="morphological opening kernel for cleaning the RAFT ownership mask; 0 disables it",
    )
    parser.add_argument(
        "--raft-min-component-area", type=int, default=DEFAULT_RAFT_MIN_COMPONENT_AREA,
        help="discard RAFT ownership components smaller than this many native-canvas pixels; 0 disables it",
    )
    parser.add_argument(
        "--sr-mode",
        choices=("global-fsrcnn-x6", "global-fsrcnn-x4", "global-fsrcnn", "none"),
        default="global-fsrcnn-x6",
        help="Global x6 (Lanczos x1.5 + FSRCNN x4, default), x4, or no super-resolution",
    )
    parser.add_argument("--sr-model", type=Path, default=Path("models/FSRCNN_x4.pb"))
    parser.add_argument("--sr-tile-size", type=int, default=512)
    parser.add_argument("--sr-tile-overlap", type=int, default=24)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--omniglue-root", type=Path, default=Path("third_party/omniglue"))
    parser.add_argument("--linetr-root", type=Path, default=Path("third_party/LineTR"))
    parser.add_argument("--raft-root", type=Path, default=Path("third_party/RAFT"))
    parser.add_argument("--raft-model", type=Path, default=Path("models/raft-things.pth"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--segment-size", type=int, default=100)
    parser.add_argument("--diff-threshold", type=int, default=36)
    parser.add_argument("--open-kernel", type=int, default=5)
    parser.add_argument("--close-kernel", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=240)
    parser.add_argument(
        "--pure-output", type=Path,
        help="pure image-stitch output with metadata for same-canvas comparison",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.feather_radius < 0:
        raise SystemExit("feather-radius must be non-negative")
    if args.color_correction_max_samples < 0:
        raise SystemExit("color-correction-max-samples must be non-negative")
    if args.raft_open_kernel_size < 0 or args.raft_min_component_area < 0:
        raise SystemExit("RAFT region cleanup parameters must be non-negative")
    if args.sr_tile_size < 16 or args.sr_tile_overlap < 0 or args.sr_tile_overlap * 2 >= args.sr_tile_size:
        raise SystemExit("invalid FSRCNN tile size or overlap")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.recompute_all:
        args.motion_mode, args.geometry_mode, args.flow_mode = "recompute", "recompute", "raft"
    example = args.example.resolve()
    manifest = json.loads((example / "manifest.json").read_text(encoding="utf-8"))
    inputs = {key: example / value for key, value in manifest["inputs"].items()}
    caches = {key: example / value for key, value in manifest.get("cache", {}).items()}
    output = args.output.resolve() if args.output else example / "output" / "three_region"
    output.mkdir(parents=True, exist_ok=True)
    (example / "cache").mkdir(parents=True, exist_ok=True)
    local = _read_image(inputs["local_frame"])
    global_frame = _read_image(inputs["global_frame"])
    h_initial = load_homography(inputs["initial_homography"])
    roi = tuple(int(value) for value in manifest["initial_projection_roi_sample_xyxy"])

    config = MotionConfig(
        args.segment_size, args.diff_threshold, args.open_kernel,
        args.close_kernel, args.min_area,
    )
    if args.motion_mode == "cached":
        local_motion = _read_image(caches["local_motion_mask"], cv.IMREAD_GRAYSCALE)
        global_motion = _read_image(caches["global_motion_mask"], cv.IMREAD_GRAYSCALE)
        motion_frames = manifest["frame_count"]
    else:
        print("Detecting Local motion from the initial-projection ROI video...", flush=True)
        local_result = detect_motion(inputs["local_initial_projection_roi_video"], config)
        print("Detecting Global motion from its independent ROI video...", flush=True)
        global_result = detect_motion(inputs["global_initial_projection_roi_video"], config)
        if local_result.frame_count != global_result.frame_count:
            raise ValueError("Local and Global motion clips have different frame counts")
        local_motion, global_motion = local_result.mask, global_result.mask
        motion_frames = local_result.frame_count
        _write_image(caches["local_motion_mask"], local_motion)
        _write_image(caches["global_motion_mask"], global_motion)

    geometry_info: dict = {"mode": args.geometry_mode}
    if args.geometry_mode == "cached":
        h_final = load_homography(caches["reference_final_homography"])
        geometry_info["source"] = "cached_point_line_motion_homography"
    else:
        from image_stitch import LineTRLineMatcher, OmniGluePointMatcher

        point_matcher = OmniGluePointMatcher(
            args.model_dir, confidence_threshold=0.1, omniglue_root=args.omniglue_root,
        )
        line_matcher = LineTRLineMatcher(args.linetr_root, device=args.device)
        print("Matching OmniGlue points and LineTR lines in both motion masks...", flush=True)
        estimate = estimate_motion_homography(
            local, global_frame, h_initial, roi, local_motion, global_motion,
            point_matcher, line_matcher,
        )
        h_final = estimate.h_final
        np.savez_compressed(caches["reference_final_homography"], H=h_final)
        geometry_info.update({
            "points_before_mask": estimate.points_before_mask,
            "points_after_mask": estimate.points_after_mask,
            "lines_before_mask": estimate.lines_before_mask,
            "lines_after_mask": estimate.lines_after_mask,
            "point_rmse": estimate.point_rmse,
            "line_rmse": estimate.line_rmse,
            "fallback_reason": estimate.fallback_reason,
        })

    prepared = prepare_regions(
        global_frame, local, h_initial, h_final, roi, local_motion,
        global_motion, one_sided_to_raft=args.one_sided_to_raft,
    )
    raft_flow = None
    if args.flow_mode == "cached":
        raft_warped = _read_image(caches["raft_warped"])
        raft_valid = _read_image(caches["raft_valid_mask"], cv.IMREAD_GRAYSCALE)
        flow_path = caches.get("raft_flow")
        if flow_path is not None and not flow_path.is_file():
            flow_path = None
        if flow_path is not None:
            with np.load(flow_path) as cached_flow:
                raft_flow = cached_flow["flow"].astype(np.float32, copy=False)
        elif args.sr_mode in ("global-fsrcnn-x6", "global-fsrcnn-x4", "global-fsrcnn"):
            raise FileNotFoundError(
                "High-resolution Local detail needs a cached RAFT flow field. "
                "Run once with --flow-mode raft to create it."
            )
    else:
        print("Running bidirectional RAFT on the static overlap...", flush=True)
        raft = RAFTFlow(args.raft_root, args.raft_model, device=args.device)
        raft_warped, raft_valid, raft_flow = align_static_region(
            global_frame, prepared.homography_warp, prepared.valid_local,
            prepared.static_candidate, raft,
        )
        _write_image(caches["raft_warped"], raft_warped)
        _write_image(caches["raft_valid_mask"], raft_valid)
        if raft_flow is not None:
            np.savez_compressed(caches["raft_flow"], flow=raft_flow.astype(np.float32, copy=False))
    composed = compose_three_regions(
        global_frame, local, h_initial, h_final, roi,
        local_motion, global_motion, raft_warped, raft_valid,
        one_sided_to_raft=args.one_sided_to_raft,
        feather_radius=args.feather_radius,
        raft_open_kernel_size=args.raft_open_kernel_size,
        raft_min_component_area=args.raft_min_component_area,
        color_correction=not args.no_color_correction,
        color_correction_coeff=args.color_correction_coeff,
        color_correction_threshold=args.color_correction_threshold,
        color_correction_max_samples=args.color_correction_max_samples,
    )

    _write_image(output / "homography_warp.jpg", composed.homography_warp)
    _write_image(output / "region_global_mask.png", composed.regions.global_region)
    _write_image(output / "region_raft_mask.png", composed.regions.raft_region)
    _write_image(output / "region_motion_mask.png", composed.regions.homography_region)
    _write_image(output / "final_result.jpg", composed.image)
    _write_image(
        output / "region_overlay_on_global.jpg",
        region_overlay(global_frame, composed.regions),
    )
    local_last = _last_video_frame(inputs["local_initial_projection_roi_video"])
    global_last = _last_video_frame(inputs["global_initial_projection_roi_video"])
    motion_panel = np.hstack((
        tint(local_last, local_motion, (0, 0, 255), 0.35),
        tint(global_last, global_motion, (0, 0, 255), 0.35),
    ))
    _write_image(output / "local_global_motion_masks.jpg", motion_panel)

    pure_path = args.pure_output or example / "output" / "pure_stitch.jpg"
    pure_homography = _pure_homography_to_global(pure_path) if pure_path.is_file() else None
    comparison_paths = None
    local_roi_outputs = None

    sr_info = {"mode": args.sr_mode}
    if args.sr_mode in ("global-fsrcnn-x6", "global-fsrcnn-x4", "global-fsrcnn"):
        model_path = args.sr_model.resolve()
        use_x6 = args.sr_mode == "global-fsrcnn-x6"
        sr_model = (FSRCNNX6 if use_x6 else FSRCNNX4)(
            model_path, tile_size=args.sr_tile_size, overlap=args.sr_tile_overlap,
        )
        scale = sr_model.scale
        global_sr = sharpen_for_display(sr_model.upsample(global_frame))
        sr_composed = compose_with_global_sr(
            global_sr,
            local,
            h_final,
            raft_warped,
            composed.regions,
            feather_radius=args.feather_radius,
            raft_flow=raft_flow,
            color_correction=not args.no_color_correction,
            color_correction_coeff=args.color_correction_coeff,
            color_correction_threshold=args.color_correction_threshold,
            color_correction_max_samples=args.color_correction_max_samples,
        )
        scale_suffix = f"x{scale}"
        sr_result_path = output / f"final_result_global_fsrcnn_{scale_suffix}.jpg"
        sr_overlay_path = output / f"region_overlay_on_global_fsrcnn_{scale_suffix}.jpg"
        sr_background_path = output / f"global_background_fsrcnn_{scale_suffix}.jpg"
        _write_image(sr_background_path, global_sr)
        _write_image(sr_result_path, sr_composed.image)
        sr_overlay = region_overlay(global_sr, sr_composed.regions)
        _write_image(sr_overlay_path, sr_overlay)
        local_roi = _local_roi_xyxy(local.shape, h_final, global_frame.shape)
        local_result_path = output / f"final_result_local_roi_fsrcnn_{scale_suffix}.jpg"
        local_overlay_path = output / f"region_overlay_local_roi_fsrcnn_{scale_suffix}.jpg"
        _write_image(local_result_path, _crop_scaled(sr_composed.image, local_roi, scale))
        _write_image(local_overlay_path, _crop_scaled(sr_overlay, local_roi, scale))
        local_roi_outputs = {
            "final_result": str(local_result_path.relative_to(example)),
            "region_overlay": str(local_overlay_path.relative_to(example)),
            "native_xyxy": list(local_roi),
        }
        if pure_homography is not None:
            scale_matrix = np.diag([float(scale), float(scale), 1.0])
            pure_result_path = output / f"pure_stitch_global_fsrcnn_{scale_suffix}.jpg"
            pure_highres = compose_homography_over_background(
                global_sr,
                local,
                scale_matrix @ pure_homography,
                args.feather_radius * scale,
                color_correction=not args.no_color_correction,
                color_correction_coeff=args.color_correction_coeff,
                color_correction_threshold=args.color_correction_threshold,
                color_correction_max_samples=args.color_correction_max_samples,
            )
            _write_image(pure_result_path, pure_highres)
            pure_local_path = output / f"pure_stitch_local_roi_fsrcnn_{scale_suffix}.jpg"
            _write_image(pure_local_path, _crop_scaled(pure_highres, local_roi, scale))
            comparison_paths = {
                "pure_stitch": str(pure_result_path.relative_to(example)),
                "three_region": str(sr_result_path.relative_to(example)),
                "pure_stitch_local_roi": str(pure_local_path.relative_to(example)),
                "three_region_local_roi": str(local_result_path.relative_to(example)),
            }
        sr_info.update({
            "model": str(args.sr_model),
            "scale": scale,
            "method": "lanczos_x1.5_then_fsrcnn_x4" if use_x6 else "fsrcnn_x4",
            "raft_high_resolution": raft_flow is not None,
            "tile_size": args.sr_tile_size,
            "tile_overlap": args.sr_tile_overlap,
            "background": str(sr_background_path.relative_to(example)),
            "final_result": str(sr_result_path.relative_to(example)),
            "region_overlay": str(sr_overlay_path.relative_to(example)),
            "local_roi_outputs": local_roi_outputs,
            "output_shape": list(sr_composed.image.shape),
        })
    elif pure_homography is not None:
        pure_result_path = output / "pure_stitch_full.jpg"
        pure_full = compose_homography_over_background(
            global_frame, local, pure_homography, args.feather_radius,
            color_correction=not args.no_color_correction,
            color_correction_coeff=args.color_correction_coeff,
            color_correction_threshold=args.color_correction_threshold,
            color_correction_max_samples=args.color_correction_max_samples,
        )
        _write_image(pure_result_path, pure_full)
        comparison_paths = {
            "pure_stitch": str(pure_result_path.relative_to(example)),
            "three_region": str((output / "final_result.jpg").relative_to(example)),
        }

    summary = {
        "scene": manifest["scene"],
        "camera": manifest["camera"],
        "frame_count": motion_frames,
        "input_shape": list(global_frame.shape),
        "roi_xyxy": list(roi),
        "motion_mode": args.motion_mode,
        "geometry": geometry_info,
        "flow_mode": args.flow_mode,
        "one_sided_to_raft": args.one_sided_to_raft,
        "feather_radius": args.feather_radius,
        "color_correction": {
            "enabled": not args.no_color_correction,
            "method": "HHM",
            "coeff": args.color_correction_coeff,
            "local_straightening_threshold": args.color_correction_threshold,
            "max_paired_pixels": args.color_correction_max_samples,
            "native_canvas": composed.color_correction,
            "x6_output": sr_composed.color_correction if args.sr_mode in ("global-fsrcnn-x6", "global-fsrcnn-x4", "global-fsrcnn") else None,
        },
        "raft_region_cleanup": {
            "morphology": "elliptical opening",
            "open_kernel_size": args.raft_open_kernel_size,
            "min_component_area_native_px": args.raft_min_component_area,
            "removed_pixels_owner": "Global",
            **(composed.regions.raft_cleanup_stats or {}),
        },
        "super_resolution": sr_info,
        "local_roi_outputs": local_roi_outputs,
        "H_init": h_initial.tolist(),
        "H_final": h_final.tolist(),
        "region_pixels": {
            "global": int(np.count_nonzero(composed.regions.global_region)),
            "raft": int(np.count_nonzero(composed.regions.raft_region)),
            "motion": int(np.count_nonzero(composed.regions.homography_region)),
        },
        "comparison_images": comparison_paths,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
