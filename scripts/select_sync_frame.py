"""Reproduce a sample's fixed-Global, projected-Local frame selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv

from point_line_stitch.synchronization import rank_projected_video, select_with_confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    args = parser.parse_args()
    example = args.example.resolve()
    metadata = json.loads((example / "sync/manifest.json").read_text())
    root = example / "sync"
    global_roi = cv.imread(str(root / metadata["global_roi"]))
    mask = cv.imread(str(root / metadata["score_mask"]), cv.IMREAD_GRAYSCALE)
    if global_roi is None or mask is None:
        raise FileNotFoundError("Synchronization reference or mask is missing")
    scores = rank_projected_video(
        global_roi, root / metadata["local_projection_video"], mask,
        metadata["search_start_local_index"], metadata["anchor_global_index"],
    )
    decision = select_with_confidence(
        scores, metadata["anchor_global_index"], metadata["current_local_index"],
    )
    selected = decision["selected"]
    result = {**metadata, "best": decision["best_raw"], "selected": selected,
              "decision_detail": decision, "candidates": scores}
    path = root / "selection.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"{example.name}: Global {metadata['anchor_global_index']}, "
          f"Local {selected['local_frame_index']} "
          f"({selected['offset_from_global_index']:+d}), "
          f"masked SSIM {selected['masked_ssim']:.5f}")


if __name__ == "__main__":
    main()
