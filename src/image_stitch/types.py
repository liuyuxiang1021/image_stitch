"""Shared result types for point-line homography estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class HomographyEstimationError(RuntimeError):
    """Raised when matching constraints cannot determine a valid homography."""


@dataclass(frozen=True)
class FeatureMatches:
    """Matched point and line features, all in x/y pixel coordinates."""

    points_source: np.ndarray
    points_destination: np.ndarray
    point_scores: np.ndarray
    lines_source: np.ndarray
    lines_destination: np.ndarray
    line_scores: np.ndarray


@dataclass(frozen=True)
class OptimizationDiagnostics:
    """Statistics produced by robust point-line optimization."""

    points_before_filter: int
    points_after_filter: int
    lines_before_filter: int
    lines_after_filter: int
    iterations: int
    initial_homography: np.ndarray
    point_rmse: float | None
    line_rmse: float | None


@dataclass(frozen=True)
class HomographyResult:
    """A local-to-global homography and the evidence used to estimate it."""

    homography: np.ndarray
    matches: FeatureMatches
    filtered_matches: FeatureMatches
    diagnostics: OptimizationDiagnostics
    delta_homography: np.ndarray | None = None
    initial_homography: np.ndarray | None = None


@dataclass(frozen=True)
class StitchResult:
    """Final stitched image, masks, transform, and estimation details."""

    panorama: np.ndarray
    source_mask: np.ndarray
    destination_mask: np.ndarray
    source_to_canvas: np.ndarray
    destination_to_canvas: np.ndarray
    estimation: HomographyResult
