"""Generate deterministic overlapping sample images for the repository."""

from pathlib import Path

import cv2 as cv
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "examples" / "input"


def main() -> None:
    rng = np.random.default_rng(1021)
    height, width = 480, 1200
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = np.clip(35 + 90 * x + 25 * y, 0, 255)
    base[..., 1] = np.clip(70 + 45 * x + 60 * y, 0, 255)
    base[..., 2] = np.clip(120 + 45 * x - 35 * y, 0, 255)

    for index in range(55):
        center = (int(rng.integers(20, width - 20)), int(rng.integers(20, height - 20)))
        color = tuple(int(value) for value in rng.integers(25, 245, size=3))
        if index % 2:
            radius = int(rng.integers(5, 24))
            cv.circle(base, center, radius, color, -1, cv.LINE_AA)
        else:
            size = rng.integers(10, 35, size=2)
            end = (center[0] + int(size[0]), center[1] + int(size[1]))
            cv.rectangle(base, center, end, color, -1, cv.LINE_AA)

    cv.putText(base, "HOMOGRAPHY", (310, 205), cv.FONT_HERSHEY_DUPLEX, 2.0, (250, 250, 250), 4, cv.LINE_AA)
    cv.putText(base, "IMAGE STITCH", (405, 285), cv.FONT_HERSHEY_DUPLEX, 1.45, (20, 35, 55), 3, cv.LINE_AA)
    cv.line(base, (70, 390), (1130, 350), (245, 230, 40), 7, cv.LINE_AA)

    left = base[:, :720]
    right = base[:, 480:]
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(INPUT_DIR / "left.png"), left)
    cv.imwrite(str(INPUT_DIR / "right.png"), right)


if __name__ == "__main__":
    main()
