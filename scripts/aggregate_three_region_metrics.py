"""Aggregate the five per-sample x6 benchmark results for the README table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


METHODS = {
    "sift": "SIFT_RANSAC_H",
    "pure": "OmniGlue_LineTR_H",
    "three_region": "three_region_final",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("examples/three_region_stitch/selected_1000f"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "metrics.json"
    existing = json.loads(output.read_text(encoding="utf-8"))
    samples = []
    for item in existing["samples"]:
        name = item["sample"]
        path = root / name / "output/benchmark/metrics.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["super_resolution_scale"] != 6 or result["feather_radius_native_px"] != 6:
            raise ValueError(f"Unexpected scale or feather radius: {name}")
        methods = result["all_roi_methods"]
        sample = {
            "sample": name,
            "roi_size": [result["all_roi_width"], result["all_roi_height"]],
            "frame_count": item["frame_count"],
        }
        for label, source in METHODS.items():
            sample[label] = {
                key: methods[source][key] for key in ("psnr_db", "ssim")
            }
        samples.append(sample)
    if len(samples) != 5:
        raise ValueError(f"Expected five selected samples, found {len(samples)}")
    summary = {
        "reference": "x6 Global background, same Local ROI, JPEG outputs",
        "samples": samples,
        "mean": {
            label: {
                key: round(mean(item[label][key] for item in samples), 3 if key == "psnr_db" else 4)
                for key in ("psnr_db", "ssim")
            }
            for label in METHODS
        },
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["mean"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
