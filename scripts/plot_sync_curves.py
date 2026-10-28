"""Plot fixed-Global synchronization scores for repository examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=Path("examples"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.examples / "point_line_stitch").glob("*/sync/selection.json"))
    paths += sorted((args.examples / "three_region_stitch/selected_1000f").glob("*/sync/selection.json"))
    if not paths:
        raise FileNotFoundError("No synchronization selections found")
    columns = 2
    rows = (len(paths) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.1 * rows), squeeze=False)
    for ax, path in zip(axes.flat, paths):
        record = json.loads(path.read_text())
        candidates = record["candidates"]
        offsets = [item["offset_from_global_index"] for item in candidates]
        scores = [item["masked_ssim"] for item in candidates]
        best = record["best"]
        selected = record.get("selected", best)
        kind = "point-line" if path.parents[2].name == "point_line_stitch" else "three-region"
        ax.plot(offsets, scores, color="#2563eb", linewidth=1.7)
        ax.axvline(0, color="#64748b", linestyle="--", linewidth=1, label="same index")
        ax.scatter([best["offset_from_global_index"]], [best["masked_ssim"]],
                   color="#dc2626", s=48, zorder=3, label="highest SSIM")
        if selected["local_frame_index"] != best["local_frame_index"]:
            ax.scatter([selected["offset_from_global_index"]], [selected["masked_ssim"]],
                       color="#16a34a", marker="s", s=45, zorder=3,
                       label="retained frame")
        ax.set_title(f"{path.parents[1].name}  [{kind}]  best {best['offset_from_global_index']:+d}",
                     fontsize=10)
        ax.set_xlabel("Local index minus fixed Global index")
        ax.set_ylabel("motion-overlap SSIM")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    for ax in list(axes.flat)[len(paths):]:
        ax.axis("off")
    fig.suptitle("Frame synchronization search: higher SSIM is better", fontsize=15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(args.output)


if __name__ == "__main__":
    main()
