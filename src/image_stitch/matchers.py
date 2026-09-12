"""Lazy adapters for the OmniGlue point matcher and LineTR line matcher."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from .types import HomographyEstimationError


class OmniGluePointMatcher:
    """Point correspondence adapter around the official OmniGlue release."""

    def __init__(
        self,
        model_dir: str | Path,
        confidence_threshold: float = 0.1,
        omniglue_root: str | Path | None = "third_party/omniglue",
    ):
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        self.omniglue_root = Path(omniglue_root).resolve() if omniglue_root else None
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        required = {
            "OmniGlue": self.model_dir / "og_export",
            "SuperPoint": self.model_dir / "sp_v6",
            "DINOv2": self.model_dir / "dinov2_vitb14_pretrain.pth",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model files for {', '.join(missing)} under {self.model_dir}. "
                "Run scripts/setup_models.sh first."
            )
        # TensorFlow and PyTorch share the GPU in OmniGlue. Without memory
        # growth, TensorFlow may reserve the whole device before DINOv2 runs.
        try:
            import tensorflow as tf
            for gpu in tf.config.experimental.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except (ImportError, RuntimeError):
            pass
        # OmniGlue imports DINOv2 as ``third_party.dinov2``. Its current
        # editable-package metadata exposes ``dinov2`` instead, so also make
        # the official repository root importable when a submodule is used.
        if self.omniglue_root and self.omniglue_root.exists():
            root_string = str(self.omniglue_root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)
        try:
            import omniglue
        except ImportError as error:
            raise ImportError(
                "OmniGlue is not installed. Initialize submodules and run "
                "`python -m pip install -e third_party/omniglue`."
            ) from error
        self._model = omniglue.OmniGlue(
            og_export=str(required["OmniGlue"]),
            sp_export=str(required["SuperPoint"]),
            dino_export=str(required["DINOv2"]),
        )
        return self._model

    def match(self, source: np.ndarray, destination: np.ndarray):
        model = self._load()
        source_rgb = cv.cvtColor(source, cv.COLOR_BGR2RGB)
        destination_rgb = cv.cvtColor(destination, cv.COLOR_BGR2RGB)
        points_source, points_destination, scores = model.FindMatches(
            source_rgb, destination_rgb
        )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        keep = scores >= self.confidence_threshold
        return (
            np.asarray(points_source, dtype=np.float32).reshape(-1, 2)[keep],
            np.asarray(points_destination, dtype=np.float32).reshape(-1, 2)[keep],
            scores[keep],
        )


class LineTRLineMatcher:
    """Line correspondence adapter around the official LineTR release."""

    def __init__(
        self,
        linetr_root: str | Path,
        *,
        device: str = "auto",
        max_keylines: int = 200,
        match_threshold: float = 0.9,
    ):
        self.linetr_root = Path(linetr_root).resolve()
        self.requested_device = device
        self.max_keylines = max_keylines
        self.match_threshold = match_threshold
        self._model = None
        self._torch = None

    def _load(self):
        if self._model is not None:
            return self._model
        weights = self.linetr_root / "models" / "weights" / "LineTR_weight.pth"
        if not weights.exists():
            raise FileNotFoundError(
                f"LineTR weights not found at {weights}. Run `git submodule update --init`."
            )
        root_string = str(self.linetr_root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        try:
            import torch
            from models.matching import Matching
        except (ImportError, ModuleNotFoundError) as error:
            raise ImportError(
                "LineTR dependencies are unavailable. Install torch, einops, and "
                "opencv-contrib-python, then initialize third_party/LineTR."
            ) from error

        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.requested_device
        config = {
            "auto_min_length": True,
            "superpoint": {
                "nms_radius": 4,
                "keypoint_threshold": 0.005,
                "max_keypoints": 1024,
                "nn_threshold": 0.9,
            },
            "lsd": {"n_octave": 2},
            "linetransformer": {
                "max_keylines": self.max_keylines,
                "min_length": 16,
                "nn_threshold": self.match_threshold,
            },
        }
        self._model = Matching(config).eval().to(device)
        self._torch = torch
        return self._model

    def match(self, source: np.ndarray, destination: np.ndarray):
        model = self._load()
        torch = self._torch
        device = next(model.parameters()).device

        def prepare(image: np.ndarray):
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
            return torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].to(device)

        with torch.no_grad():
            prediction = model(
                {"image0": prepare(source), "image1": prepare(destination)}
            )
        assignment = prediction["matches_l"][0].detach().cpu().numpy()
        distance = prediction["matching_scores_l"][0].detach().cpu().numpy()
        rows, columns = np.where(assignment > 0)
        lines_source = prediction["klines0"][0].detach().cpu().numpy()[rows]
        lines_destination = prediction["klines1"][0].detach().cpu().numpy()[columns]
        scores = 1.0 / (1.0 + distance[rows, columns])
        return (
            np.asarray(lines_source, dtype=np.float32).reshape(-1, 2, 2),
            np.asarray(lines_destination, dtype=np.float32).reshape(-1, 2, 2),
            np.asarray(scores, dtype=np.float32),
        )


def validate_matcher_inputs(source: np.ndarray, destination: np.ndarray) -> None:
    for name, image in (("source", source), ("destination", destination)):
        if image is None or image.size == 0:
            raise ValueError(f"{name} image is empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} must be a BGR image, got {image.shape}")
    if min(source.shape[:2]) < 16 or min(destination.shape[:2]) < 16:
        raise HomographyEstimationError("Images are too small for point-line matching")
