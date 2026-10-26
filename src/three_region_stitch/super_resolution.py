"""Tiled FSRCNN inference for the Global background."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


class FSRCNNX4:
    """OpenCV DNN FSRCNN x4 with overlapping tiles for large frames."""

    scale = 4

    def __init__(
        self,
        model_path: str | Path,
        tile_size: int = 512,
        overlap: int = 24,
    ) -> None:
        self.model_path = Path(model_path)
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing FSRCNN x4 model: {self.model_path}")
        if not hasattr(cv, "dnn_superres"):
            raise RuntimeError("cv.dnn_superres is unavailable; install opencv-contrib-python")
        if self.tile_size < 16 or self.overlap < 0 or self.overlap * 2 >= self.tile_size:
            raise ValueError("Invalid FSRCNN tile size or overlap")

        self._model = cv.dnn_superres.DnnSuperResImpl_create()
        self._model.readModel(str(self.model_path))
        self._model.setModel("fsrcnn", self.scale)

    def upsample(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 BGR image, got {image.shape}")
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Cannot super-resolve an empty image")

        output = np.empty((height * self.scale, width * self.scale, 3), dtype=np.uint8)
        step = self.tile_size - 2 * self.overlap
        for core_y0 in range(0, height, step):
            core_y1 = min(height, core_y0 + step)
            input_y0 = max(0, core_y0 - self.overlap)
            input_y1 = min(height, core_y1 + self.overlap)
            for core_x0 in range(0, width, step):
                core_x1 = min(width, core_x0 + step)
                input_x0 = max(0, core_x0 - self.overlap)
                input_x1 = min(width, core_x1 + self.overlap)
                tile = self._model.upsample(image[input_y0:input_y1, input_x0:input_x1])
                expected = (
                    (input_y1 - input_y0) * self.scale,
                    (input_x1 - input_x0) * self.scale,
                    3,
                )
                if tile.shape != expected:
                    raise RuntimeError(f"FSRCNN returned {tile.shape}; expected {expected}")

                tile_x0 = (core_x0 - input_x0) * self.scale
                tile_y0 = (core_y0 - input_y0) * self.scale
                tile_x1 = tile_x0 + (core_x1 - core_x0) * self.scale
                tile_y1 = tile_y0 + (core_y1 - core_y0) * self.scale
                output[
                    core_y0 * self.scale:core_y1 * self.scale,
                    core_x0 * self.scale:core_x1 * self.scale,
                ] = tile[tile_y0:tile_y1, tile_x0:tile_x1]
        return output


class FSRCNNX6:
    """Approximate x6 output using Lanczos x1.5 followed by FSRCNN x4."""

    scale = 6
    interpolation_scale = 1.5

    def __init__(
        self,
        model_path: str | Path,
        tile_size: int = 512,
        overlap: int = 24,
    ) -> None:
        self._fsrcnn = FSRCNNX4(model_path, tile_size=tile_size, overlap=overlap)

    def upsample(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 BGR image, got {image.shape}")
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Cannot super-resolve an empty image")

        intermediate_size = (
            int(round(width * self.interpolation_scale)),
            int(round(height * self.interpolation_scale)),
        )
        intermediate = cv.resize(image, intermediate_size, interpolation=cv.INTER_LANCZOS4)
        result = self._fsrcnn.upsample(intermediate)
        output_size = (width * self.scale, height * self.scale)
        if result.shape[1::-1] != output_size:
            result = cv.resize(result, output_size, interpolation=cv.INTER_LANCZOS4)
        return result


def sharpen_for_display(image: np.ndarray, amount: float = 0.22) -> np.ndarray:
    if amount <= 0:
        return image
    blurred = cv.GaussianBlur(image, (0, 0), sigmaX=1.1, sigmaY=1.1)
    return cv.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
