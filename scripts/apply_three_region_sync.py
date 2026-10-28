"""Apply selected Local frame offsets to a three-region example's ROI clips."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--video-root", type=Path,
                        default=Path("/mnt/16T/lyx/image_stitching/input/videos"))
    parser.add_argument("--calibration", type=Path,
                        default=Path("/mnt/16T/lyx/image_stitching/input/camera_calibration_initial/100_undistor.pkl"))
    args = parser.parse_args()
    example = args.example.resolve()
    manifest = json.loads((example / "manifest.json").read_text())
    review_source = manifest.get("selected_review_source")
    choice = json.loads((example / "sync/selection.json").read_text())
    global_index = int(choice["anchor_global_index"])
    local_index = int(choice["selected"]["local_frame_index"])
    offset = local_index - global_index
    scene, camera = manifest["scene"], manifest["camera"]
    start = global_index - manifest["frame_count"] + 1
    if offset:
        x0, y0, _, _ = manifest["global_display_crop_original_xyxy"]
        unshift = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
        with np.load(example / "input/H_init.npz") as data:
            h_initial = unshift @ data["H"]
        with np.load(example / "cache/H_final_reference.npz") as data:
            h_final = unshift @ data["H"]
        source = {
            "scene_name": scene,
            "camera_name": camera,
            "final_frame_idx": global_index,
            "bbox": manifest["initial_projection_roi_original_xyxy"],
            "H_init": h_initial.tolist(),
            "H_final": h_final.tolist(),
            "local_video_path": str(args.video_root / scene / "main" / f"{camera}.mp4"),
            "global_video_path": str(args.video_root / scene / "main/100.mp4"),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(json.dumps(source))
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("export_three_region_sample.py")),
                "--metadata", str(path), "--calibration", str(args.calibration),
                "--output", str(example), "--start-frame", str(start),
                "--local-frame-offset", str(offset), "--crop",
                *(str(v) for v in manifest["global_display_crop_original_xyxy"]),
            ], check=True)
        manifest = json.loads((example / "manifest.json").read_text())
    else:
        manifest["global_video_frame_range_inclusive"] = [start, global_index]
        manifest["local_video_frame_range_inclusive"] = [start, local_index]
        manifest["local_frame_offset_from_global"] = 0
    manifest["sync_selection"] = "sync/selection.json"
    if review_source:
        manifest["selected_review_source"] = review_source
    manifest["selection_note"] = (
        "Global is fixed; Local index selected from projected motion-overlap SSIM. "
        "See sync/selection.json for scores and confidence decision."
    )
    (example / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"{example.name}: Global {global_index}, Local {local_index}, offset {offset:+d}")


if __name__ == "__main__":
    main()
