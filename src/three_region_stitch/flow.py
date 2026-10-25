"""RAFT alignment and bidirectional consistency for the static Local region."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import cv2 as cv
import numpy as np


class RAFTFlow:
    """Lazy adapter around the official RAFT repository and its Things weights."""

    def __init__(
        self,
        root: str | Path,
        weights: str | Path,
        *,
        device: str = "auto",
        iterations: int = 20,
        consistency_threshold: float = 3.0,
    ) -> None:
        self.root = Path(root)
        self.weights = Path(weights)
        self.requested_device = device
        self.iterations = iterations
        self.consistency_threshold = consistency_threshold
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        core = self.root / "core"
        if not (core / "raft.py").is_file():
            raise FileNotFoundError(f"RAFT source missing: {core}. Initialize the RAFT submodule.")
        if not self.weights.is_file():
            raise FileNotFoundError(f"RAFT weights missing: {self.weights}. Run scripts/setup_raft.sh.")
        if str(core.resolve()) not in sys.path:
            sys.path.insert(0, str(core.resolve()))
        import torch
        from raft import RAFT

        device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if self.requested_device == "auto" else self.requested_device
        args = Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
        model = RAFT(args)
        try:
            checkpoint = torch.load(str(self.weights), map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch 1.13 has no weights_only argument.
            checkpoint = torch.load(str(self.weights), map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
        self._model = model.to(device).eval()
        self._device = device
        self._torch = torch

    def compute(self, target: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return target-to-source flow and forward/backward validity."""
        if source.shape != target.shape or target.ndim != 3 or target.shape[2] != 3:
            raise ValueError("RAFT requires same-size BGR target and source images")
        self._load()
        torch = self._torch
        assert torch is not None and self._model is not None
        import torch.nn.functional as functional

        def tensor(image: np.ndarray):
            return torch.from_numpy(image).permute(2, 0, 1).float()[None].to(self._device)

        target_tensor, source_tensor = tensor(target), tensor(source)
        height, width = target.shape[:2]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        target_tensor = functional.pad(target_tensor, padding, mode="replicate")
        source_tensor = functional.pad(source_tensor, padding, mode="replicate")
        with torch.no_grad():
            _, forward = self._model(
                target_tensor, source_tensor, iters=self.iterations, test_mode=True,
            )
            _, reverse = self._model(
                source_tensor, target_tensor, iters=self.iterations, test_mode=True,
            )
        left, right, top, bottom = padding
        forward = forward[0, :, top:top + height, left:left + width]
        reverse = reverse[0, :, top:top + height, left:left + width]
        forward_np = forward.permute(1, 2, 0).cpu().numpy()
        reverse_np = reverse.permute(1, 2, 0).cpu().numpy()
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        map_x = xx + forward_np[..., 0]
        map_y = yy + forward_np[..., 1]
        reverse_at_forward = cv.remap(
            reverse_np, map_x, map_y, cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT, borderValue=(1e6, 1e6),
        )
        error = np.linalg.norm(forward_np + reverse_at_forward, axis=2)
        valid = (
            np.isfinite(error)
            & (error < self.consistency_threshold)
            & (map_x >= 0) & (map_x < width - 1)
            & (map_y >= 0) & (map_y < height - 1)
        )
        return forward_np.astype(np.float32), valid.astype(np.uint8) * 255


def align_static_region(
    global_frame: np.ndarray,
    local_warp: np.ndarray,
    local_valid: np.ndarray,
    static_candidate: np.ndarray,
    raft: RAFTFlow,
) -> tuple[np.ndarray, np.ndarray]:
    """Flow-refine Local in the static candidate bbox and keep valid pixels."""
    if global_frame.shape != local_warp.shape:
        raise ValueError("Global and Local warp must share a canvas")
    rows, columns = np.where(static_candidate > 0)
    refined = local_warp.copy()
    valid_result = np.zeros(global_frame.shape[:2], dtype=np.uint8)
    if not len(rows):
        return refined, valid_result
    x0, x1 = max(0, int(columns.min()) - 8), min(global_frame.shape[1], int(columns.max()) + 9)
    y0, y1 = max(0, int(rows.min()) - 8), min(global_frame.shape[0], int(rows.max()) + 9)
    target = global_frame[y0:y1, x0:x1]
    source = local_warp[y0:y1, x0:x1]
    flow, flow_valid = raft.compute(target, source)
    yy, xx = np.mgrid[:y1-y0, :x1-x0].astype(np.float32)
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    warped = cv.remap(source, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    source_valid = cv.remap(
        local_valid[y0:y1, x0:x1], map_x, map_y, cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
    )
    refined[y0:y1, x0:x1] = warped
    valid_result[y0:y1, x0:x1] = cv.bitwise_and(
        cv.bitwise_and(flow_valid, source_valid), static_candidate[y0:y1, x0:x1],
    )
    return refined, valid_result
