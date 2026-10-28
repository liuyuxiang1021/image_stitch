"""Prepare compact sync windows from the original dual-camera recordings.

The repository examples retain the projected ROI clip so frame selection can
be reproduced without distributing the full 4K recordings.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np

from point_line_stitch import projected_local_roi
from point_line_stitch.synchronization import (
    rank_projected_video,
    select_with_confidence,
    target_motion_mask,
)


OLD_PURE_ANCHORS = {
    "20201024084037322_cam11": 50,
    "20210313151132256_cam22": 26,
    "20220817171240976_cam35": 50,
}
OLD_PURE_LOCAL_FRAMES = {
    "20201024084037322_cam11": 33,
    "20210313151132256_cam22": 26,
    "20220817171240976_cam35": 50,
}


def read_frame(capture: cv.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Could not decode frame {index}")
    return frame


def prepare(example: Path, video_root: Path, calibration: Path, radius: int) -> dict:
    name = example.name
    scene, camera = name.split("_cam")
    pure = example.parent.name == "point_line_stitch"
    if pure:
        sample = example / "sample.json"
        sample_data = json.loads(sample.read_text()) if sample.exists() else {}
        if not sample_data and name not in OLD_PURE_ANCHORS:
            raise ValueError(f"Missing frame provenance for {name}")
        anchor = int(sample_data.get("global_frame_index_zero_based",
                                     sample_data.get("frame_index_zero_based", OLD_PURE_ANCHORS.get(name))))
        current_local_index = int(sample_data.get(
            "local_frame_index_zero_based", OLD_PURE_LOCAL_FRAMES.get(name, anchor),
        ))
        metadata = json.loads((example / "output/metadata.json").read_text())
        homography = np.asarray(metadata["homography_source_to_destination"], np.float64)
        original_crop = (0, 0)
    else:
        manifest = json.loads((example / "manifest.json").read_text())
        anchor = int(manifest["source_video_frame_range_inclusive"][1])
        current_local_index = int(manifest.get(
            "local_video_frame_range_inclusive", [None, anchor],
        )[1])
        with np.load(example / "cache/H_final_reference.npz") as data:
            homography = data["H"].astype(np.float64)
        original_crop = tuple(manifest["global_display_crop_original_xyxy"][:2])

    local = cv.imread(str(example / "input/local.jpg"))
    global_image = cv.imread(str(example / "input/global.jpg"))
    if local is None or global_image is None:
        raise FileNotFoundError(f"Missing input images in {example}")
    x0, y0, x1, y1 = projected_local_roi(local.shape, homography, global_image.shape, 16)
    width, height = x1 - x0, y1 - y0
    if width % 2:
        x1 = min(global_image.shape[1], x1 + 1)
    if height % 2:
        y1 = min(global_image.shape[0], y1 + 1)
    width, height = x1 - x0, y1 - y0
    if width % 2 or height % 2:
        raise ValueError("Projection ROI needs even video dimensions")
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    h_roi = shift @ homography
    valid = cv.warpPerspective(
        np.ones(local.shape[:2], np.uint8), h_roi, (width, height),
        flags=cv.INTER_NEAREST,
    )

    with calibration.open("rb") as file:
        camera = pickle.load(file)
    K = np.asarray(camera["K"], np.float64)
    distortion = np.asarray(camera["distor"], np.float64)
    new_K, _ = cv.getOptimalNewCameraMatrix(K, distortion, (3840, 2160), 1, (3840, 2160))
    map_x, map_y = cv.initUndistortRectifyMap(
        K, distortion, None, new_K, (3840, 2160), cv.CV_32FC1,
    )
    ox, oy = original_crop
    map_x = map_x[oy + y0:oy + y1, ox + x0:ox + x1]
    map_y = map_y[oy + y0:oy + y1, ox + x0:ox + x1]
    global_video = video_root / scene / "main/100.mp4"
    local_video = video_root / scene / "main" / f"{name.split('_cam')[1]}.mp4"
    global_cap = cv.VideoCapture(str(global_video))
    local_cap = cv.VideoCapture(str(local_video))
    if not global_cap.isOpened() or not local_cap.isOpened():
        raise FileNotFoundError(f"Missing original recordings for {name}")
    try:
        anchor_raw = read_frame(global_cap, anchor)
        neighbors = [read_frame(global_cap, index) for index in (anchor - 10, anchor + 10)
                     if 0 <= index < int(global_cap.get(cv.CAP_PROP_FRAME_COUNT))]
        global_anchor = cv.remap(anchor_raw, map_x, map_y, cv.INTER_LINEAR)
        global_neighbors = [cv.remap(frame, map_x, map_y, cv.INTER_LINEAR) for frame in neighbors]
        mask = target_motion_mask(global_anchor, global_neighbors, valid)
        if np.count_nonzero(mask) < 1000:
            mask = target_motion_mask(global_anchor, global_neighbors, valid,
                                      threshold=10, dilation=17)
        start = max(0, anchor - radius)
        end = min(int(local_cap.get(cv.CAP_PROP_FRAME_COUNT)) - 1, anchor + radius)
        sync_dir = example / "sync"
        sync_dir.mkdir(exist_ok=True)
        cv.imwrite(str(sync_dir / "global_roi.jpg"), global_image[y0:y1, x0:x1],
                   [cv.IMWRITE_JPEG_QUALITY, 95])
        cv.imwrite(str(sync_dir / "score_mask.png"), mask)
        clip = sync_dir / "local_projection.mp4"
        process = subprocess.Popen([
            "/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", "25", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            str(clip),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert process.stdin is not None
        local_cap.set(cv.CAP_PROP_POS_FRAMES, start)
        try:
            for index in range(start, end + 1):
                ok, frame = local_cap.read()
                if not ok:
                    raise ValueError(f"Could not read Local frame {index} from {local_video}")
                warped = cv.warpPerspective(frame, h_roi, (width, height))
                process.stdin.write(warped.tobytes())
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError(process.stderr.read().decode())
        process.stderr.close()
    finally:
        local_cap.release()
        global_cap.release()

    record = {
        "method": "fixed_global_target_motion_masked_ssim_on_projected_local",
        "anchor_global_index": anchor,
        "current_local_index": current_local_index,
        "search_radius_requested": radius,
        "search_start_local_index": start,
        "search_end_local_index": end,
        "global_roi_xyxy_in_sample": [x0, y0, x1, y1],
        "score_mask_pixels": int(np.count_nonzero(mask)),
        "projection_homography_local_to_global_roi": h_roi.tolist(),
        "projection_h_source": "previously_estimated_H_final_fixed_for_search",
        "global_roi": "global_roi.jpg",
        "score_mask": "score_mask.png",
        "local_projection_video": "local_projection.mp4",
    }
    (sync_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    scores = rank_projected_video(
        cv.imread(str(sync_dir / "global_roi.jpg")), clip, mask, start, anchor,
    )
    decision = select_with_confidence(scores, anchor, current_local_index)
    selection = {**record, "best": decision["best_raw"],
                 "selected": decision["selected"], "decision_detail": decision,
                 "candidates": scores}
    (sync_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    selected = decision["selected"]
    print(f"{name}: Global {anchor}, Local {selected['local_frame_index']} "
          f"({selected['offset_from_global_index']:+d}), SSIM {selected['masked_ssim']:.5f}, "
          f"decision {decision['decision']}, mask {record['score_mask_pixels']} px", flush=True)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=Path("examples"))
    parser.add_argument("--video-root", type=Path,
                        default=Path("/mnt/16T/lyx/image_stitching/input/videos"))
    parser.add_argument("--calibration", type=Path,
                        default=Path("/mnt/16T/lyx/image_stitching/input/camera_calibration_initial/100_undistor.pkl"))
    parser.add_argument("--radius", type=int, default=30)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()
    if args.radius < 1:
        raise ValueError("Search radius must be positive")
    roots = [args.examples / "point_line_stitch", args.examples / "three_region_stitch/selected_1000f"]
    for root in roots:
        for example in sorted(root.iterdir()):
            if not example.is_dir() or (args.only and example.name not in args.only):
                continue
            prepare(example, args.video_root, args.calibration, args.radius)


if __name__ == "__main__":
    main()
