"""Render the Local projection at higher resolution without enlarging Local first."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from .color_correction import apply_hhm_luts, fit_hhm_for_regions
from .core import _feather_pair


@dataclass(frozen=True)
class LocalROIResult:
    image: np.ndarray
    background: np.ndarray
    global_roi_xyxy: tuple[int, int, int, int]
    source_to_roi: np.ndarray
    color_correction: dict


@dataclass(frozen=True)
class HomographyRenderResult:
    image: np.ndarray
    color_correction: dict


def render_homography_on_background(
    background: np.ndarray,
    local: np.ndarray,
    homography: np.ndarray,
    feather_radius: float,
    *,
    feather_power: float = 1.0,
    color_luts: np.ndarray | None = None,
    color_correction: bool = True,
    color_correction_coeff: float = 1.0,
    color_correction_threshold: float = 0.05,
    color_correction_max_samples: int = 400_000,
) -> HomographyRenderResult:
    """Project Local onto an already prepared Global background, then apply HHM and feathering."""
    height, width = background.shape[:2]
    homography = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    warped = cv.warpPerspective(local, homography, (width, height))
    valid = cv.warpPerspective(
        np.full(local.shape[:2], 255, dtype=np.uint8),
        homography, (width, height), flags=cv.INTER_NEAREST,
    )
    correction_stats = {"method": "HHM", "applied": False, "reason": "disabled"}
    if color_correction:
        if color_luts is None:
            color_luts, correction_stats = fit_hhm_for_regions(
                background, [warped], [valid],
                coeff=color_correction_coeff,
                local_straightening_threshold=color_correction_threshold,
                max_samples=color_correction_max_samples,
            )
        else:
            correction_stats = {"method": "HHM", "applied": True, "reason": "supplied_luts"}
        warped = apply_hhm_luts(warped, color_luts, valid)
    image = _feather_pair(
        warped, background, valid,
        np.full(valid.shape, 255, dtype=np.uint8),
        feather_radius, feather_power,
    )
    return HomographyRenderResult(image, correction_stats)


def projected_local_roi(
    local_shape: tuple[int, ...],
    homography: np.ndarray,
    global_shape: tuple[int, ...],
    margin: int = 24,
) -> tuple[int, int, int, int]:
    """Return the Local footprint plus margin, clipped to the Global image."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    local_height, local_width = local_shape[:2]
    global_height, global_width = global_shape[:2]
    corners = np.array(
        [[[0.0, 0.0]], [[local_width - 1.0, 0.0]],
         [[local_width - 1.0, local_height - 1.0]], [[0.0, local_height - 1.0]]],
        dtype=np.float64,
    )
    projected = cv.perspectiveTransform(
        corners, np.asarray(homography, dtype=np.float64).reshape(3, 3),
    ).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        raise ValueError("Local projection has non-finite coordinates")
    x0 = max(0, int(np.floor(projected[:, 0].min())) - margin)
    y0 = max(0, int(np.floor(projected[:, 1].min())) - margin)
    x1 = min(global_width, int(np.ceil(projected[:, 0].max())) + margin + 1)
    y1 = min(global_height, int(np.ceil(projected[:, 1].max())) + margin + 1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("The projected Local image does not overlap the Global canvas")
    return x0, y0, x1, y1


def compose_local_roi(
    local: np.ndarray,
    global_frame: np.ndarray,
    homography: np.ndarray,
    *,
    scale: int = 6,
    margin: int = 24,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    background_roi: np.ndarray | None = None,
    feather_width: float = 6.0,
    feather_power: float = 1.0,
    color_correction: bool = True,
    color_correction_coeff: float = 1.0,
    color_correction_threshold: float = 0.05,
    color_correction_max_samples: int = 400_000,
    max_output_pixels: int = 64_000_000,
) -> LocalROIResult:
    """Blend original Local detail into an enlarged Global ROI.

    ``background_roi`` may supply a previously enlarged Global crop so a
    comparison can use exactly the same background as another method.
    """
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer")
    roi = roi_xyxy or projected_local_roi(local.shape, homography, global_frame.shape, margin)
    x0, y0, x1, y1 = (int(value) for value in roi)
    if not (0 <= x0 < x1 <= global_frame.shape[1] and 0 <= y0 < y1 <= global_frame.shape[0]):
        raise ValueError("ROI must be inside the Global image")
    output_size = ((x1 - x0) * scale, (y1 - y0) * scale)
    if output_size[0] * output_size[1] > max_output_pixels:
        raise ValueError(f"Local ROI output is too large: {output_size}")
    if background_roi is None:
        background = cv.resize(
            global_frame[y0:y1, x0:x1], output_size,
            interpolation=cv.INTER_LANCZOS4,
        )
    else:
        if background_roi.shape[:2] != (output_size[1], output_size[0]):
            raise ValueError("background_roi dimensions do not match the scaled ROI")
        background = background_roi

    shift = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    upsample = np.diag([float(scale), float(scale), 1.0])
    source_to_roi = upsample @ shift @ np.asarray(homography, dtype=np.float64).reshape(3, 3)
    rendered = render_homography_on_background(
        background, local, source_to_roi, feather_width * scale,
        feather_power=feather_power,
        color_correction=color_correction,
        color_correction_coeff=color_correction_coeff,
        color_correction_threshold=color_correction_threshold,
        color_correction_max_samples=color_correction_max_samples,
    )
    return LocalROIResult(
        rendered.image, background, (x0, y0, x1, y1),
        source_to_roi, rendered.color_correction,
    )
