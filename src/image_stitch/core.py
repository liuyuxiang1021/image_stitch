"""End-to-end point-line homography estimation, warping, and blending."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

from .matchers import validate_matcher_inputs
from .optimization import estimate_point_line_homography
from .types import FeatureMatches, HomographyResult, StitchResult


def _resize_for_matching(image: np.ndarray, max_width: int):
    if max_width <= 0 or image.shape[1] <= max_width:
        return image, np.eye(3, dtype=np.float64)
    scale = max_width / image.shape[1]
    resized = cv.resize(
        image,
        (max_width, int(round(image.shape[0] * scale))),
        interpolation=cv.INTER_AREA,
    )
    return resized, np.diag([scale, scale, 1.0])


def collect_matches(source, destination, point_matcher, line_matcher) -> FeatureMatches:
    validate_matcher_inputs(source, destination)
    points_source, points_destination, point_scores = point_matcher.match(source, destination)
    lines_source, lines_destination, line_scores = line_matcher.match(source, destination)
    return FeatureMatches(
        points_source=points_source,
        points_destination=points_destination,
        point_scores=point_scores,
        lines_source=lines_source,
        lines_destination=lines_destination,
        line_scores=line_scores,
    )


def _scale_matches_to_original(
    matches: FeatureMatches,
    source_scale: np.ndarray,
    destination_scale: np.ndarray,
) -> FeatureMatches:
    source_inverse = np.linalg.inv(source_scale)
    destination_inverse = np.linalg.inv(destination_scale)

    def transform(values, matrix, shape):
        if len(values) == 0:
            return np.empty((0, *shape[1:]), dtype=np.float32)
        return cv.perspectiveTransform(values.reshape(-1, 1, 2), matrix).reshape(shape)

    return FeatureMatches(
        transform(matches.points_source, source_inverse, (-1, 2)),
        transform(matches.points_destination, destination_inverse, (-1, 2)),
        matches.point_scores,
        transform(matches.lines_source, source_inverse, (-1, 2, 2)),
        transform(matches.lines_destination, destination_inverse, (-1, 2, 2)),
        matches.line_scores,
    )


def estimate_homography(
    source: np.ndarray,
    destination: np.ndarray,
    point_matcher,
    line_matcher,
    *,
    max_match_width: int = 1200,
    point_weight: float = 0.5,
    line_weight: float = 0.5,
    ransac_threshold: float = 4.0,
    line_distance_threshold: float = 12.0,
    line_angle_threshold: float = 15.0,
) -> HomographyResult:
    """Directly estimate source-to-destination H using point and line matches."""
    source_small, source_scale = _resize_for_matching(source, max_match_width)
    destination_small, destination_scale = _resize_for_matching(destination, max_match_width)
    small_matches = collect_matches(source_small, destination_small, point_matcher, line_matcher)
    h_small, filtered_small, diagnostics = estimate_point_line_homography(
        small_matches,
        point_weight=point_weight,
        line_weight=line_weight,
        ransac_threshold=ransac_threshold,
        line_distance_threshold=line_distance_threshold,
        line_angle_threshold=line_angle_threshold,
    )
    homography = np.linalg.inv(destination_scale) @ h_small @ source_scale
    matches = _scale_matches_to_original(small_matches, source_scale, destination_scale)
    filtered = _scale_matches_to_original(filtered_small, source_scale, destination_scale)
    return HomographyResult(homography, matches, filtered, diagnostics)


def refine_homography(
    source: np.ndarray,
    destination: np.ndarray,
    initial_homography: np.ndarray,
    point_matcher,
    line_matcher,
    **estimation_options,
) -> HomographyResult:
    """Estimate H_delta in the initial-warp overlap and return H_delta @ H_init."""
    height, width = destination.shape[:2]
    initial_homography = np.asarray(initial_homography, dtype=np.float64).reshape(3, 3)
    warped = cv.warpPerspective(source, initial_homography, (width, height))
    source_mask = cv.warpPerspective(
        np.full(source.shape[:2], 255, dtype=np.uint8),
        initial_homography,
        (width, height),
        flags=cv.INTER_NEAREST,
    )
    coordinates = cv.findNonZero(source_mask)
    if coordinates is None:
        raise ValueError("The initial homography maps the source outside the destination")
    x, y, crop_width, crop_height = cv.boundingRect(coordinates)
    source_crop = warped[y:y+crop_height, x:x+crop_width]
    destination_crop = destination[y:y+crop_height, x:x+crop_width]
    delta_crop = estimate_homography(
        source_crop,
        destination_crop,
        point_matcher,
        line_matcher,
        **estimation_options,
    )
    full_to_crop = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=np.float64)
    delta_full = np.linalg.inv(full_to_crop) @ delta_crop.homography @ full_to_crop
    final_h = delta_full @ initial_homography

    offset = np.array([x, y], dtype=np.float32)
    matches = FeatureMatches(
        delta_crop.matches.points_source + offset,
        delta_crop.matches.points_destination + offset,
        delta_crop.matches.point_scores,
        delta_crop.matches.lines_source + offset,
        delta_crop.matches.lines_destination + offset,
        delta_crop.matches.line_scores,
    )
    filtered = FeatureMatches(
        delta_crop.filtered_matches.points_source + offset,
        delta_crop.filtered_matches.points_destination + offset,
        delta_crop.filtered_matches.point_scores,
        delta_crop.filtered_matches.lines_source + offset,
        delta_crop.filtered_matches.lines_destination + offset,
        delta_crop.filtered_matches.line_scores,
    )
    return HomographyResult(
        homography=final_h,
        matches=matches,
        filtered_matches=filtered,
        diagnostics=delta_crop.diagnostics,
        delta_homography=delta_full,
        initial_homography=initial_homography,
    )


def _corners(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)


def _weights(mask: np.ndarray, feather_power: float) -> np.ndarray:
    padded = cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    distance = cv.distanceTransform(padded, cv.DIST_L2, 5)[1:-1, 1:-1]
    output = np.zeros(mask.shape, dtype=np.float32)
    valid = mask > 0
    output[valid] = np.power(distance[valid] + 1.0, feather_power)
    return output


def compose_pair(
    source: np.ndarray,
    destination: np.ndarray,
    estimation: HomographyResult,
    *,
    feather_power: float = 1.0,
    max_canvas_pixels: int = 100_000_000,
) -> StitchResult:
    transformed_source_corners = cv.perspectiveTransform(_corners(source), estimation.homography)
    all_corners = np.concatenate((_corners(destination), transformed_source_corners), axis=0).reshape(-1, 2)
    minimum = np.floor(all_corners.min(axis=0)).astype(int)
    maximum = np.ceil(all_corners.max(axis=0)).astype(int)
    canvas_width, canvas_height = (maximum - minimum).tolist()
    if canvas_width <= 0 or canvas_height <= 0 or canvas_width * canvas_height > max_canvas_pixels:
        raise ValueError(f"Invalid or excessive panorama canvas: {canvas_width}x{canvas_height}")
    translation = np.array([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=np.float64)
    source_transform = translation @ estimation.homography
    destination_transform = translation
    size = (canvas_width, canvas_height)

    warped_source = cv.warpPerspective(source, source_transform, size)
    warped_destination = cv.warpPerspective(destination, destination_transform, size)
    source_mask = cv.warpPerspective(
        np.full(source.shape[:2], 255, dtype=np.uint8), source_transform, size, flags=cv.INTER_NEAREST
    )
    destination_mask = cv.warpPerspective(
        np.full(destination.shape[:2], 255, dtype=np.uint8), destination_transform, size, flags=cv.INTER_NEAREST
    )
    source_weights = _weights(source_mask, feather_power)
    destination_weights = _weights(destination_mask, feather_power)
    weight_sum = source_weights + destination_weights
    color_sum = (
        warped_source.astype(np.float32) * source_weights[..., None]
        + warped_destination.astype(np.float32) * destination_weights[..., None]
    )
    panorama = np.zeros_like(warped_source)
    valid = weight_sum > 0
    panorama[valid] = np.clip(color_sum[valid] / weight_sum[valid, None], 0, 255).astype(np.uint8)
    return StitchResult(
        panorama, source_mask, destination_mask,
        source_transform, destination_transform, estimation,
    )


def stitch_pair(
    source: np.ndarray,
    destination: np.ndarray,
    point_matcher,
    line_matcher,
    *,
    initial_homography: np.ndarray | None = None,
    feather_power: float = 1.0,
    **estimation_options,
) -> StitchResult:
    if initial_homography is None:
        estimation = estimate_homography(
            source, destination, point_matcher, line_matcher, **estimation_options
        )
    else:
        estimation = refine_homography(
            source, destination, initial_homography,
            point_matcher, line_matcher, **estimation_options
        )
    return compose_pair(source, destination, estimation, feather_power=feather_power)


def load_homography(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            for key in ("H_init", "homography", "H"):
                if key in archive:
                    matrix = archive[key].copy()
                    break
            else:
                raise ValueError(f"No H_init, homography, or H matrix in {path}")
    else:
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("homography", payload.get("H_init", payload.get("H")))
        matrix = np.asarray(payload, dtype=np.float64)
    return matrix.reshape(3, 3)
