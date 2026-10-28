#!/usr/bin/env python3
"""Render a near-4K Local-area result from a saved point-line homography."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from image_stitch import compose_local_roi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--background-output", type=Path)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--feather-width", type=float, default=6.0)
    args = parser.parse_args()

    example = args.example.resolve()
    output = args.output or example / "output/local_roi_x6.jpg"
    background_output = args.background_output or example / "output/global_roi_lanczos_x6.jpg"
    metadata = json.loads((example / "output/metadata.json").read_text(encoding="utf-8"))
    homography = np.asarray(metadata["homography_source_to_destination"], dtype=np.float64)
    local = cv.imread(str(example / "input/local.jpg"))
    global_frame = cv.imread(str(example / "input/global.jpg"))
    if local is None or global_frame is None:
        raise FileNotFoundError(f"Missing input image in {example / 'input'}")
    result = compose_local_roi(
        local, global_frame, homography,
        scale=args.scale, margin=args.margin, feather_width=args.feather_width,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(output), result.image, [cv.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {output}")
    background_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(background_output), result.background, [cv.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {background_output}")
    print(f"{output}: {result.image.shape[1]} × {result.image.shape[0]}; Global ROI {result.global_roi_xyxy}")


if __name__ == "__main__":
    main()
