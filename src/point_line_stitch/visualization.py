"""Deterministic visualizations for initial alignment and accepted matches."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .types import FeatureMatches


def _mask_bounds(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    coordinates = cv.findNonZero(mask)
    if coordinates is None:
        raise ValueError("The projected source is outside the visualization canvas")
    x, y, box_width, box_height = cv.boundingRect(coordinates)
    height, width = mask.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + box_width + padding)
    y1 = min(height, y + box_height + padding)
    return x0, y0, x1 - x0, y1 - y0


def projected_source_bounds(
    source: np.ndarray,
    destination: np.ndarray,
    homography: np.ndarray,
    *,
    padding: int = 0,
) -> tuple[int, int, int, int]:
    """Return the visible projected-source bounding box in destination pixels."""
    height, width = destination.shape[:2]
    mask = cv.warpPerspective(
        np.full(source.shape[:2], 255, dtype=np.uint8),
        np.asarray(homography, dtype=np.float64).reshape(3, 3),
        (width, height),
        flags=cv.INTER_NEAREST,
    )
    return _mask_bounds(mask, padding)


def _crop_matches(
    source: np.ndarray,
    destination: np.ndarray,
    matches: FeatureMatches,
    region: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, FeatureMatches]:
    x, y, width, height = region
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid visualization region: {region}")
    source_crop = source[y:y + height, x:x + width]
    destination_crop = destination[y:y + height, x:x + width]
    if source_crop.size == 0 or destination_crop.size == 0:
        raise ValueError(f"Visualization region lies outside the images: {region}")
    offset = np.array([x, y], dtype=np.float32)
    adjusted = FeatureMatches(
        matches.points_source - offset,
        matches.points_destination - offset,
        matches.point_scores,
        matches.lines_source - offset,
        matches.lines_destination - offset,
        matches.line_scores,
    )
    return source_crop, destination_crop, adjusted


def draw_matches(
    source: np.ndarray,
    destination: np.ndarray,
    matches: FeatureMatches,
    *,
    margin: int = 12,
    region: tuple[int, int, int, int] | None = None,
    max_points: int | None = None,
    max_lines: int | None = None,
) -> np.ndarray:
    """Draw accepted correspondences; optional limits affect the display only."""
    if region is not None:
        source, destination, matches = _crop_matches(
            source, destination, matches, region
        )
    height = max(source.shape[0], destination.shape[0])
    width = source.shape[1] + margin + destination.shape[1]
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas[:source.shape[0], :source.shape[1]] = source
    destination_x = source.shape[1] + margin
    canvas[:destination.shape[0], destination_x:] = destination
    rng = np.random.default_rng(1021)
    point_indices = _display_indices(
        matches.points_source, matches.point_scores, max_points, source.shape[:2]
    )
    line_indices = _display_indices(
        matches.lines_source.mean(axis=1), matches.line_scores,
        max_lines, source.shape[:2],
    )
    for source_point, destination_point in zip(
        matches.points_source[point_indices], matches.points_destination[point_indices]
    ):
        color = tuple(int(value) for value in rng.integers(30, 240, size=3))
        first = tuple(np.rint(source_point).astype(int))
        second = tuple(np.rint(destination_point + [destination_x, 0]).astype(int))
        cv.circle(canvas, first, 3, color, -1, cv.LINE_AA)
        cv.circle(canvas, second, 3, color, -1, cv.LINE_AA)
        cv.line(canvas, first, second, color, 1, cv.LINE_AA)
    for source_line, destination_line in zip(
        matches.lines_source[line_indices], matches.lines_destination[line_indices]
    ):
        color = tuple(int(value) for value in rng.integers(30, 240, size=3))
        source_line = np.rint(source_line).astype(int)
        destination_line = np.rint(destination_line + [destination_x, 0]).astype(int)
        cv.line(canvas, tuple(source_line[0]), tuple(source_line[1]), color, 3, cv.LINE_AA)
        cv.line(
            canvas,
            tuple(destination_line[0]),
            tuple(destination_line[1]),
            color,
            3,
            cv.LINE_AA,
        )
        cv.line(
            canvas,
            tuple(np.rint(source_line.mean(axis=0)).astype(int)),
            tuple(np.rint(destination_line.mean(axis=0)).astype(int)),
            color,
            1,
            cv.LINE_AA,
        )
    return canvas


def _display_indices(
    locations: np.ndarray,
    scores: np.ndarray,
    limit: int | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Select high-score matches across the view without changing estimation."""
    count = len(locations)
    if limit is None or count <= limit:
        return np.arange(count)
    if limit < 0:
        raise ValueError("display match limit must be non-negative")
    height, width = image_shape
    cols, rows = 6, 4
    cell_x = np.clip((locations[:, 0] * cols / max(width, 1)).astype(int), 0, cols - 1)
    cell_y = np.clip((locations[:, 1] * rows / max(height, 1)).astype(int), 0, rows - 1)
    cells = cell_y * cols + cell_x
    order = np.argsort(-scores, kind="stable")
    buckets = [order[cells[order] == cell] for cell in range(rows * cols)]
    selected: list[int] = []
    rank = 0
    while len(selected) < limit:
        added = False
        for bucket in buckets:
            if rank < len(bucket):
                selected.append(int(bucket[rank]))
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        rank += 1
    return np.asarray(selected, dtype=int)


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR,
    )
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def _draw_dashed_contours(
    image: np.ndarray,
    contours: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
    dash_length: float = 14.0,
    gap_length: float = 10.0,
) -> None:
    """Draw closed anti-aliased contours as short dashed segments."""
    period = dash_length + gap_length
    for contour in contours:
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(points) < 2:
            continue
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length <= 0:
                continue
            direction = vector / length
            offset = 0.0
            while offset < length:
                dash_start = start + direction * offset
                dash_end = start + direction * min(offset + dash_length, length)
                cv.line(
                    image,
                    tuple(np.rint(dash_start).astype(int)),
                    tuple(np.rint(dash_end).astype(int)),
                    color,
                    thickness,
                    cv.LINE_AA,
                )
                offset += period


def initial_alignment_views(
    source: np.ndarray,
    destination: np.ndarray,
    homography: np.ndarray,
    *,
    padding: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the projected footprint and magnified initial alignment."""
    height, width = destination.shape[:2]
    homography = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    warped = cv.warpPerspective(source, homography, (width, height))
    mask = cv.warpPerspective(
        np.full(source.shape[:2], 255, dtype=np.uint8),
        homography,
        (width, height),
        flags=cv.INTER_NEAREST,
    )
    x, y, box_width, box_height = projected_source_bounds(
        source, destination, homography, padding=padding
    )

    mixed = cv.addWeighted(destination, 0.45, warped, 0.55, 0)
    overview = destination.copy()
    overview[mask > 0] = mixed[mask > 0]
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(overview, contours, -1, (255, 255, 0), 10, cv.LINE_AA)
    cv.rectangle(
        overview,
        (x, y),
        (x + box_width - 1, y + box_height - 1),
        (0, 215, 255),
        8,
        cv.LINE_AA,
    )

    roi_destination = destination[y:y + box_height, x:x + box_width]
    roi_warped = warped[y:y + box_height, x:x + box_width]
    roi_mask = mask[y:y + box_height, x:x + box_width]
    roi_mixed = cv.addWeighted(roi_destination, 0.45, roi_warped, 0.55, 0)
    zoom = roi_destination.copy()
    zoom[roi_mask > 0] = roi_mixed[roi_mask > 0]
    zoom_contours, _ = cv.findContours(
        roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    _draw_dashed_contours(zoom, zoom_contours)

    return overview, zoom


def draw_initial_alignment(
    source: np.ndarray,
    destination: np.ndarray,
    homography: np.ndarray,
    *,
    panel_width: int = 960,
    panel_height: int = 540,
    padding: int = 32,
) -> np.ndarray:
    """Make a title-free side-by-side preview of the two alignment views."""
    overview, zoom = initial_alignment_views(
        source, destination, homography, padding=padding
    )

    gap = 16
    canvas = np.full(
        (panel_height, panel_width * 2 + gap, 3), 28, dtype=np.uint8
    )
    canvas[:, :panel_width] = _fit_panel(
        overview, panel_width, panel_height
    )
    canvas[:, panel_width + gap:] = _fit_panel(
        zoom, panel_width, panel_height
    )
    return canvas


def final_alignment_views(
    panorama: np.ndarray,
    source_mask: np.ndarray,
    *,
    padding: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the final footprint and magnified feather-blended ROI."""
    if source_mask.shape[:2] != panorama.shape[:2]:
        raise ValueError("source_mask and panorama must have the same dimensions")
    x, y, box_width, box_height = _mask_bounds(source_mask, padding)

    overview = panorama.copy()
    contours, _ = cv.findContours(
        source_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    cv.drawContours(overview, contours, -1, (255, 255, 0), 10, cv.LINE_AA)
    cv.rectangle(
        overview,
        (x, y),
        (x + box_width - 1, y + box_height - 1),
        (0, 215, 255),
        8,
        cv.LINE_AA,
    )

    zoom = panorama[y:y + box_height, x:x + box_width].copy()
    zoom_mask = source_mask[y:y + box_height, x:x + box_width]
    zoom_contours, _ = cv.findContours(
        zoom_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    _draw_dashed_contours(zoom, zoom_contours)

    return overview, zoom


def draw_final_alignment(
    panorama: np.ndarray,
    source_mask: np.ndarray,
    *,
    panel_width: int = 960,
    panel_height: int = 540,
    padding: int = 32,
) -> np.ndarray:
    """Make a title-free side-by-side preview of the final alignment views."""
    overview, zoom = final_alignment_views(
        panorama, source_mask, padding=padding
    )

    gap = 16
    canvas = np.full(
        (panel_height, panel_width * 2 + gap, 3), 28, dtype=np.uint8
    )
    canvas[:, :panel_width] = _fit_panel(
        overview, panel_width, panel_height
    )
    canvas[:, panel_width + gap:] = _fit_panel(
        zoom, panel_width, panel_height
    )
    return canvas
