"""Export a compact, real-video three-region example from the source project.

The two videos contain the exact initial-projection ROI used by the motion
detector. Their last frames correspond to local.jpg and global.jpg. This
script is for maintainers with access to the original camera recordings.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np


def _video_writer(path: Path, width: int, height: int, fps: int, ffmpeg: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _finish_video(process: subprocess.Popen, path: Path) -> None:
    assert process.stdin is not None
    process.stdin.close()
    status = process.wait()
    if status:
        assert process.stderr is not None
        raise RuntimeError(f"ffmpeg failed for {path}: {process.stderr.read().decode()}")
    assert process.stderr is not None
    process.stderr.close()


def _open_at(path: str, frame_index: int) -> cv.VideoCapture:
    capture = cv.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(path)
    capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    return capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=5801)
    parser.add_argument("--crop", type=int, nargs=4, default=(252, 432, 1322, 1218),
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument(
        "--local-scale", type=float, default=1.0,
        help="scale for the saved Local frame; defaults to its original 3840x2160 resolution",
    )
    parser.add_argument(
        "--ffmpeg", default="/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg",
        help="FFmpeg binary with libx264 support",
    )
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    final_frame = int(metadata["final_frame_idx"])
    if args.start_frame > final_frame:
        raise ValueError("start-frame must not exceed final_frame_idx")
    x0, y0, x1, y1 = args.crop
    bx0, by0, bx1, by1 = (int(v) for v in metadata["bbox"])
    # yuv420p video requires even dimensions; include one extra ROI pixel if needed.
    bx1 += (bx1 - bx0) % 2
    by1 += (by1 - by0) % 2
    if not (x0 <= bx0 < bx1 <= x1 and y0 <= by0 < by1 <= y1):
        raise ValueError("The display crop must contain the full initial-projection ROI")
    if not 0 < args.local_scale <= 1:
        raise ValueError("local-scale must be in (0, 1]")

    with args.calibration.open("rb") as file:
        calibration = pickle.load(file)
    camera_matrix = np.asarray(calibration["K"], dtype=np.float64)
    distortion = np.asarray(calibration["distor"], dtype=np.float64)
    raw_size = (3840, 2160)
    new_matrix, _ = cv.getOptimalNewCameraMatrix(
        camera_matrix, distortion, raw_size, 1, raw_size,
    )
    map_x, map_y = cv.initUndistortRectifyMap(
        camera_matrix, distortion, None, new_matrix, raw_size, cv.CV_32FC1,
    )

    local_video = _open_at(metadata["local_video_path"], args.start_frame)
    global_video = _open_at(metadata["global_video_path"], args.start_frame)
    input_dir = args.output / "input"
    cache_dir = args.output / "cache"
    input_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    roi_width, roi_height = bx1 - bx0, by1 - by0
    local_path = input_dir / "local_init_roi.mp4"
    global_path = input_dir / "global_roi.mp4"
    local_writer = _video_writer(local_path, roi_width, roi_height, 25, args.ffmpeg)
    global_writer = _video_writer(global_path, roi_width, roi_height, 25, args.ffmpeg)

    h_initial = np.asarray(metadata["H_init"], dtype=np.float64)
    initial_to_roi = np.array(
        [[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]], dtype=np.float64,
    ) @ h_initial
    display_translation = np.array(
        [[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64,
    )
    local_to_raw = np.diag([1 / args.local_scale, 1 / args.local_scale, 1.0])
    h_initial_sample = display_translation @ h_initial @ local_to_raw
    h_final_sample = (
        display_translation @ np.asarray(metadata["H_final"], dtype=np.float64)
        @ local_to_raw
    )
    processed = 0
    try:
        for frame_index in range(args.start_frame, final_frame + 1):
            local_ok, local_raw = local_video.read()
            global_ok, global_raw = global_video.read()
            if not local_ok:
                local_video.set(cv.CAP_PROP_POS_FRAMES, frame_index)
                local_ok, local_raw = local_video.read()
            if not global_ok:
                global_video.set(cv.CAP_PROP_POS_FRAMES, frame_index)
                global_ok, global_raw = global_video.read()
            if not local_ok or not global_ok:
                raise RuntimeError(f"Could not decode both videos at frame {frame_index}")
            if local_raw.shape[:2] != (2160, 3840) or global_raw.shape[:2] != (2160, 3840):
                raise ValueError("This export expects 3840x2160 camera recordings")
            local_roi = cv.warpPerspective(
                local_raw, initial_to_roi, (roi_width, roi_height),
                flags=cv.INTER_LINEAR,
            )
            global_roi = cv.remap(
                global_raw, map_x[by0:by1, bx0:bx1], map_y[by0:by1, bx0:bx1],
                cv.INTER_LINEAR,
            )
            assert local_writer.stdin is not None and global_writer.stdin is not None
            local_writer.stdin.write(local_roi.tobytes())
            global_writer.stdin.write(global_roi.tobytes())
            processed += 1
            if frame_index == final_frame:
                local_small = cv.resize(
                    local_raw, None, fx=args.local_scale, fy=args.local_scale,
                    interpolation=cv.INTER_AREA,
                )
                global_display = cv.remap(
                    global_raw, map_x[y0:y1, x0:x1], map_y[y0:y1, x0:x1],
                    cv.INTER_LINEAR,
                )
                cv.imwrite(str(input_dir / "local.jpg"), local_small,
                           [cv.IMWRITE_JPEG_QUALITY, 95])
                cv.imwrite(str(input_dir / "global.jpg"), global_display,
                           [cv.IMWRITE_JPEG_QUALITY, 95])
            if processed % 50 == 0 or frame_index == final_frame:
                print(f"{metadata['scene_name']}: exported {processed} frames", flush=True)
    finally:
        local_video.release()
        global_video.release()
        _finish_video(local_writer, local_path)
        _finish_video(global_writer, global_path)

    np.savez_compressed(input_dir / "H_init.npz", H=h_initial_sample)
    np.savez_compressed(cache_dir / "H_final_reference.npz", H=h_final_sample)
    manifest = {
        "scene": metadata["scene_name"],
        "camera": metadata["camera_name"],
        "source_project": "image_stitching",
        "source_video_frame_range_inclusive": [args.start_frame, final_frame],
        "frame_count": processed,
        "video_fps": 25,
        "local_scale_from_original": args.local_scale,
        "global_display_crop_original_xyxy": [x0, y0, x1, y1],
        "initial_projection_roi_original_xyxy": [bx0, by0, bx1, by1],
        "initial_projection_roi_sample_xyxy": [bx0 - x0, by0 - y0, bx1 - x0, by1 - y0],
        "inputs": {
            "local_frame": "input/local.jpg",
            "global_frame": "input/global.jpg",
            "initial_homography": "input/H_init.npz",
            "local_initial_projection_roi_video": "input/local_init_roi.mp4",
            "global_initial_projection_roi_video": "input/global_roi.mp4",
        },
        "cache": {
            "reference_final_homography": "cache/H_final_reference.npz",
            "local_motion_mask": "cache/local_motion_init_mask.png",
            "global_motion_mask": "cache/global_motion_mask.png",
            "raft_warped": "cache/raft_warped.jpg",
            "raft_valid_mask": "cache/raft_valid_mask.png",
            "raft_flow": "cache/raft_flow.npz",
        },
        "outputs": {
            "pure_stitch": "output/pure_stitch.jpg",
            "pure_diagnostics": "output/pure_artifacts/",
            "three_region": "output/three_region/",
            "benchmark": "output/benchmark/",
        },
        "note": "ROI videos are actual camera frames cropped into H_init coordinates; "
                "they are intended for independent motion-mask computation. Derived masks, "
                "homographies, and RAFT alignment data are stored in cache/.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
