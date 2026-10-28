"""Aggregate five point-line samples' SIFT and point-line benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


METHODS = {"sift": "SIFT_RANSAC_H", "pure": "OmniGlue_LineTR_H"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("examples/point_line_stitch"))
    args = parser.parse_args()
    root = args.root.resolve()
    samples = []
    for example in sorted(root.iterdir()):
        if not example.is_dir():
            continue
        path = example / "output/benchmark/metrics.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["scale"] != 6 or result["feather_radius_native_px"] != 6:
            raise ValueError(f"Unexpected scale or feather radius: {example.name}")
        sample = {"sample": example.name, "roi_size": result["roi_size"]}
        for label, source in METHODS.items():
            sample[label] = {
                key: result["methods"][source][key] for key in ("psnr_db", "ssim")
            }
        samples.append(sample)
    if len(samples) != 5:
        raise ValueError(f"Expected five point-line samples, found {len(samples)}")
    summary = {
        "reference": "x6 Lanczos Global ROI, same Local ROI, JPEG outputs",
        "samples": samples,
        "mean": {
            label: {
                key: round(mean(item[label][key] for item in samples), 3 if key == "psnr_db" else 4)
                for key in ("psnr_db", "ssim")
            }
            for label in METHODS
        },
    }
    output = root / "metrics.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["mean"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
