"""Robust eight-degree-of-freedom homography optimization from points and lines."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .types import FeatureMatches, HomographyEstimationError, OptimizationDiagnostics


def normalize_matches(matches: FeatureMatches) -> FeatureMatches:
    points_source = np.asarray(matches.points_source, dtype=np.float32).reshape(-1, 2)
    points_destination = np.asarray(matches.points_destination, dtype=np.float32).reshape(-1, 2)
    lines_source = np.asarray(matches.lines_source, dtype=np.float32).reshape(-1, 2, 2)
    lines_destination = np.asarray(matches.lines_destination, dtype=np.float32).reshape(-1, 2, 2)
    point_scores = np.asarray(matches.point_scores, dtype=np.float32).reshape(-1)
    line_scores = np.asarray(matches.line_scores, dtype=np.float32).reshape(-1)
    if len(points_source) != len(points_destination) or len(points_source) != len(point_scores):
        raise ValueError("Point match arrays have inconsistent lengths")
    if len(lines_source) != len(lines_destination) or len(lines_source) != len(line_scores):
        raise ValueError("Line match arrays have inconsistent lengths")
    return FeatureMatches(
        points_source, points_destination, point_scores,
        lines_source, lines_destination, line_scores,
    )


def _line_coefficients(lines: np.ndarray) -> np.ndarray:
    direction = lines[:, 1] - lines[:, 0]
    coefficients = np.column_stack(
        (-direction[:, 1], direction[:, 0], np.zeros(len(lines), dtype=np.float32))
    )
    norms = np.linalg.norm(coefficients[:, :2], axis=1)
    valid = norms > 1e-6
    coefficients[valid, :2] /= norms[valid, None]
    coefficients[valid, 2] = -np.sum(coefficients[valid, :2] * lines[valid, 0], axis=1)
    return coefficients


def _initial_from_lines(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    if len(source) < 4:
        raise HomographyEstimationError("At least four line correspondences are required")
    coefficients = _line_coefficients(destination)
    rows, values = [], []
    for source_line, (a, b, c) in zip(source, coefficients):
        for x, y in source_line:
            rows.append([a*x, a*y, a, b*x, b*y, b, c*x, c*y])
            values.append(-c)
    solution, _, rank, _ = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64), rcond=None
    )
    if rank < 8:
        raise HomographyEstimationError("Line configuration is degenerate")
    homography = np.eye(3, dtype=np.float64)
    homography.flat[:8] = solution
    return homography


def robust_initial_homography(
    matches: FeatureMatches,
    ransac_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(matches.points_source) >= 4:
        homography, mask = cv.findHomography(
            matches.points_source,
            matches.points_destination,
            cv.RANSAC,
            ransac_threshold,
        )
        if homography is not None and mask is not None and int(mask.sum()) >= 4:
            return homography / homography[2, 2], mask.ravel().astype(bool)
    homography = _initial_from_lines(matches.lines_source, matches.lines_destination)
    return homography / homography[2, 2], np.ones(len(matches.points_source), dtype=bool)


def _project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)


def _filter_lines(
    source: np.ndarray,
    destination: np.ndarray,
    initial_h: np.ndarray,
    max_distance: float,
    max_angle_degrees: float,
) -> np.ndarray:
    if len(source) == 0:
        return np.zeros(0, dtype=bool)
    projected = _project_points(source.reshape(-1, 2), initial_h).reshape(-1, 2, 2)
    destination_coefficients = _line_coefficients(destination)
    projected_midpoints = projected.mean(axis=1)
    distances = np.abs(
        np.sum(destination_coefficients[:, :2] * projected_midpoints, axis=1)
        + destination_coefficients[:, 2]
    )
    projected_vectors = projected[:, 1] - projected[:, 0]
    destination_vectors = destination[:, 1] - destination[:, 0]
    projected_angles = np.arctan2(projected_vectors[:, 1], projected_vectors[:, 0])
    destination_angles = np.arctan2(destination_vectors[:, 1], destination_vectors[:, 0])
    angle_difference = np.abs(projected_angles - destination_angles)
    angle_difference = np.minimum(angle_difference, 2 * np.pi - angle_difference)
    angle_difference = np.minimum(angle_difference, np.abs(np.pi - angle_difference))
    lengths_source = np.linalg.norm(source[:, 1] - source[:, 0], axis=1)
    lengths_destination = np.linalg.norm(destination_vectors, axis=1)
    return (
        (distances <= max_distance)
        & (angle_difference <= np.deg2rad(max_angle_degrees))
        & (lengths_source >= 10.0)
        & (lengths_destination >= 10.0)
    )


def filter_matches(
    matches: FeatureMatches,
    initial_h: np.ndarray,
    point_inliers: np.ndarray,
    *,
    line_distance_threshold: float = 12.0,
    line_angle_threshold: float = 15.0,
) -> FeatureMatches:
    line_keep = _filter_lines(
        matches.lines_source,
        matches.lines_destination,
        initial_h,
        line_distance_threshold,
        line_angle_threshold,
    )
    return FeatureMatches(
        matches.points_source[point_inliers],
        matches.points_destination[point_inliers],
        matches.point_scores[point_inliers],
        matches.lines_source[line_keep],
        matches.lines_destination[line_keep],
        matches.line_scores[line_keep],
    )


def optimize_homography(
    matches: FeatureMatches,
    initial_h: np.ndarray,
    *,
    point_weight: float = 0.5,
    line_weight: float = 0.5,
    max_iterations: int = 30,
    huber_delta: float = 3.0,
    samples_per_line: int = 5,
) -> tuple[np.ndarray, int]:
    """Refine H with point reprojection and sampled point-to-line residuals."""
    if len(matches.points_source) == 0 and len(matches.lines_source) < 4:
        raise HomographyEstimationError("Not enough point or line constraints")
    if point_weight < 0 or line_weight < 0 or point_weight + line_weight <= 0:
        raise ValueError("Point and line weights must be non-negative and not both zero")

    sample_source, sample_coefficients = [], []
    destination_coefficients = _line_coefficients(matches.lines_destination)
    for source_line, coefficients in zip(matches.lines_source, destination_coefficients):
        for fraction in np.linspace(0.0, 1.0, samples_per_line):
            sample_source.append((1.0 - fraction) * source_line[0] + fraction * source_line[1])
            sample_coefficients.append(coefficients)
    sample_source = np.asarray(sample_source, dtype=np.float64).reshape(-1, 2)
    sample_coefficients = np.asarray(sample_coefficients, dtype=np.float64).reshape(-1, 3)

    normalized_initial = np.asarray(initial_h, dtype=np.float64) / initial_h[2, 2]
    parameters = normalized_initial.flat[:8].copy()

    def residuals_and_jacobian(current: np.ndarray):
        homography = np.eye(3, dtype=np.float64)
        homography.flat[:8] = current
        jacobians, residuals = [], []

        for (x, y), (target_x, target_y), score in zip(
            matches.points_source, matches.points_destination, matches.point_scores
        ):
            denominator = homography[2, 0] * x + homography[2, 1] * y + 1.0
            if abs(denominator) < 1e-9:
                continue
            estimate_x = (homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]) / denominator
            estimate_y = (homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]) / denominator
            confidence_weight = point_weight * float(np.sqrt(max(score, 1e-6)))
            dx = np.array([x/denominator, y/denominator, 1/denominator, 0, 0, 0,
                           -x*estimate_x/denominator, -y*estimate_x/denominator])
            dy = np.array([0, 0, 0, x/denominator, y/denominator, 1/denominator,
                           -x*estimate_y/denominator, -y*estimate_y/denominator])
            residuals.extend(((estimate_x - target_x) * confidence_weight,
                              (estimate_y - target_y) * confidence_weight))
            jacobians.extend((dx * confidence_weight, dy * confidence_weight))

        repeated_line_scores = np.repeat(matches.line_scores, samples_per_line)
        for (x, y), (a, b, c), score in zip(
            sample_source, sample_coefficients, repeated_line_scores
        ):
            denominator = homography[2, 0] * x + homography[2, 1] * y + 1.0
            if abs(denominator) < 1e-9:
                continue
            estimate_x = (homography[0, 0] * x + homography[0, 1] * y + homography[0, 2]) / denominator
            estimate_y = (homography[1, 0] * x + homography[1, 1] * y + homography[1, 2]) / denominator
            dx = np.array([x/denominator, y/denominator, 1/denominator, 0, 0, 0,
                           -x*estimate_x/denominator, -y*estimate_x/denominator])
            dy = np.array([0, 0, 0, x/denominator, y/denominator, 1/denominator,
                           -x*estimate_y/denominator, -y*estimate_y/denominator])
            confidence_weight = line_weight * float(np.sqrt(max(score, 1e-6)))
            residuals.append((a * estimate_x + b * estimate_y + c) * confidence_weight)
            jacobians.append((a * dx + b * dy) * confidence_weight)

        return np.asarray(residuals), np.asarray(jacobians)

    completed_iterations = 0
    for iteration in range(max_iterations):
        residuals, jacobian = residuals_and_jacobian(parameters)
        if len(residuals) < 8:
            raise HomographyEstimationError("Fewer than eight residual constraints remain")
        absolute = np.abs(residuals)
        robust_weights = np.ones_like(residuals)
        outside = absolute > huber_delta
        robust_weights[outside] = huber_delta / absolute[outside]
        weighted_jacobian = jacobian * robust_weights[:, None]
        normal_matrix = jacobian.T @ weighted_jacobian + 1e-6 * np.eye(8)
        right_hand_side = -(weighted_jacobian.T @ residuals)
        try:
            update = np.linalg.solve(normal_matrix, right_hand_side)
        except np.linalg.LinAlgError as error:
            raise HomographyEstimationError("Point-line normal equation is singular") from error
        parameters += update
        completed_iterations = iteration + 1
        if np.linalg.norm(update) < 1e-7:
            break

    homography = np.eye(3, dtype=np.float64)
    homography.flat[:8] = parameters
    if not np.isfinite(homography).all():
        raise HomographyEstimationError("Optimization produced a non-finite homography")
    return homography, completed_iterations


def _errors(matches: FeatureMatches, homography: np.ndarray):
    point_rmse = None
    if len(matches.points_source):
        projected = _project_points(matches.points_source, homography)
        point_rmse = float(np.sqrt(np.mean(np.sum((projected - matches.points_destination) ** 2, axis=1))))
    line_rmse = None
    if len(matches.lines_source):
        projected = _project_points(matches.lines_source.reshape(-1, 2), homography).reshape(-1, 2, 2)
        coefficients = _line_coefficients(matches.lines_destination)
        distances = (
            np.sum(projected * coefficients[:, None, :2], axis=2)
            + coefficients[:, None, 2]
        )
        line_rmse = float(np.sqrt(np.mean(distances ** 2)))
    return point_rmse, line_rmse


def estimate_point_line_homography(
    matches: FeatureMatches,
    *,
    point_weight: float = 0.5,
    line_weight: float = 0.5,
    ransac_threshold: float = 4.0,
    line_distance_threshold: float = 12.0,
    line_angle_threshold: float = 15.0,
    max_iterations: int = 30,
    huber_delta: float = 3.0,
    samples_per_line: int = 5,
):
    matches = normalize_matches(matches)
    if len(matches.points_source) < 4 and len(matches.lines_source) < 4:
        raise HomographyEstimationError(
            f"Need at least four matched points or four matched lines; got "
            f"{len(matches.points_source)} points and {len(matches.lines_source)} lines"
        )
    initial_h, point_inliers = robust_initial_homography(matches, ransac_threshold)
    filtered = filter_matches(
        matches,
        initial_h,
        point_inliers,
        line_distance_threshold=line_distance_threshold,
        line_angle_threshold=line_angle_threshold,
    )
    optimized, iterations = optimize_homography(
        filtered,
        initial_h,
        point_weight=point_weight,
        line_weight=line_weight,
        max_iterations=max_iterations,
        huber_delta=huber_delta,
        samples_per_line=samples_per_line,
    )
    point_rmse, line_rmse = _errors(filtered, optimized)
    diagnostics = OptimizationDiagnostics(
        points_before_filter=len(matches.points_source),
        points_after_filter=len(filtered.points_source),
        lines_before_filter=len(matches.lines_source),
        lines_after_filter=len(filtered.lines_source),
        iterations=iterations,
        initial_homography=initial_h,
        point_rmse=point_rmse,
        line_rmse=line_rmse,
    )
    return optimized, filtered, diagnostics
