"""Command line entry point for patchwise GMM video background extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from .core import fit_background


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="input video")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--static-threshold", type=float, default=4.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--work-dir", type=Path, help="directory for temporary patch data")
    args = parser.parse_args()

    result = fit_background(
        args.video,
        patch_size=args.patch_size,
        stride=args.stride,
        components=args.components,
        static_threshold=args.static_threshold,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        jobs=args.jobs,
        work_dir=args.work_dir,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    overlay = np.clip(
        result.background.astype(np.float32) * 0.5
        + result.static_mask.astype(np.float32) * 0.5,
        0, 255,
    ).astype(np.uint8)
    masked = cv.bitwise_and(result.background, result.static_mask)
    images = {
        "background_model_gmm.png": result.background,
        "static_region_mask.png": result.static_mask,
        "final_result.png": overlay,
        "final_result_masked.png": masked,
    }
    for name, image in images.items():
        if not cv.imwrite(str(output / name), image):
            raise RuntimeError(f"Could not write {output / name}")
    np.savez_compressed(
        output / "patch_models.npz",
        means=result.patch_means,
        weights=result.patch_weights,
        mean_variances=result.patch_variances,
        rows=np.asarray(result.rows, np.int32),
        columns=np.asarray(result.columns, np.int32),
    )
    summary = {
        "input_video": str(args.video),
        "frame_size": list(result.frame_size),
        "frames_used": result.frames_used,
        "start_frame": args.start_frame,
        "frame_step": args.frame_step,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "gmm_components": args.components,
        "static_overlap_rmse_threshold": args.static_threshold,
        "patch_count": len(result.patch_weights),
        "fallback_patches": result.fallback_count,
        "static_mask_fraction": round(float(np.mean(result.static_mask > 0)), 6),
        "background": "background_model_gmm.png",
        "static_mask": "static_region_mask.png",
        "overlay": "final_result.png",
        "masked_background": "final_result_masked.png",
        "patch_models": "patch_models.npz",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
