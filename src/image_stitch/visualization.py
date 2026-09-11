"""Deterministic visualization of accepted point and line correspondences."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from .types import FeatureMatches


def draw_matches(source: np.ndarray, destination: np.ndarray, matches: FeatureMatches, *, margin: int = 12) -> np.ndarray:
    height = max(source.shape[0], destination.shape[0])
    width = source.shape[1] + margin + destination.shape[1]
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas[:source.shape[0], :source.shape[1]] = source
    destination_x = source.shape[1] + margin
    canvas[:destination.shape[0], destination_x:] = destination
    rng = np.random.default_rng(1021)
    for source_point, destination_point in zip(matches.points_source, matches.points_destination):
        color = tuple(int(value) for value in rng.integers(30, 240, size=3))
        first = tuple(np.rint(source_point).astype(int))
        second = tuple(np.rint(destination_point + [destination_x, 0]).astype(int))
        cv.circle(canvas, first, 3, color, -1, cv.LINE_AA)
        cv.circle(canvas, second, 3, color, -1, cv.LINE_AA)
        cv.line(canvas, first, second, color, 1, cv.LINE_AA)
    for source_line, destination_line in zip(matches.lines_source, matches.lines_destination):
        color = tuple(int(value) for value in rng.integers(30, 240, size=3))
        source_line = np.rint(source_line).astype(int)
        destination_line = np.rint(destination_line + [destination_x, 0]).astype(int)
        cv.line(canvas, tuple(source_line[0]), tuple(source_line[1]), color, 3, cv.LINE_AA)
        cv.line(canvas, tuple(destination_line[0]), tuple(destination_line[1]), color, 3, cv.LINE_AA)
        cv.line(canvas, tuple(np.rint(source_line.mean(axis=0)).astype(int)), tuple(np.rint(destination_line.mean(axis=0)).astype(int)), color, 1, cv.LINE_AA)
    return canvas
