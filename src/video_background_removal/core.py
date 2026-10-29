"""Port of MotionMask's overlapping-patch temporal GMM background method."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class BackgroundResult:
    background: np.ndarray
    static_mask: np.ndarray
    patch_means: np.ndarray
    patch_weights: np.ndarray
    patch_variances: np.ndarray
    fallback_count: int
    frames_used: int
    frame_size: tuple[int, int]
    rows: tuple[int, ...]
    columns: tuple[int, ...]


def patch_positions(length: int, size: int, stride: int) -> tuple[int, ...]:
    """Cover the full axis, including an edge not reached by the regular stride."""
    if size < 1 or stride < 1 or stride > size or length < size:
        raise ValueError("Patch size and stride must be positive and fit the frame")
    values = list(range(0, length - size + 1, stride))
    if values[-1] != length - size:
        values.append(length - size)
    return tuple(values)


def _fit_patch(samples: np.ndarray, components: int) -> tuple[np.ndarray, float, float, bool]:
    """Choose the highest-weight Gaussian component, as in the source project."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    data = samples.astype(np.float64, copy=False)
    if np.var(data) < 1e-5:
        return data.mean(axis=0).astype(np.float32), 1.0, 0.0, False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = GaussianMixture(
                n_components=components,
                covariance_type="diag",
                random_state=0,
                reg_covar=1e-5,
                max_iter=200,
                tol=1e-5,
            ).fit(data)
        index = int(np.argmax(model.weights_))
        return (
            model.means_[index].astype(np.float32),
            float(model.weights_[index]),
            float(np.mean(model.covariances_[index])),
            False,
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return data.mean(axis=0).astype(np.float32), 1.0, 0.0, True


def _fit_batch(
    path: str, shape: tuple[int, int, int], indices: range, components: int,
) -> list[tuple[int, np.ndarray, float, float, bool]]:
    data = np.memmap(path, mode="r", dtype=np.uint8, shape=shape)
    return [(index, *_fit_patch(data[:, index, :], components)) for index in indices]


def _frame_patches(
    frame: np.ndarray, rows: tuple[int, ...], columns: tuple[int, ...], patch_size: int,
) -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    windows = np.lib.stride_tricks.sliding_window_view(gray, (patch_size, patch_size))
    return windows[np.array(rows)[:, None], np.array(columns)[None, :]].reshape(
        len(rows) * len(columns), patch_size * patch_size,
    )


def _static_patch_mask(
    means: np.ndarray,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    patch_size: int,
    threshold: float,
) -> np.ndarray:
    count_rows, count_columns = len(rows), len(columns)
    mean_grid = means.reshape(count_rows, count_columns, patch_size, patch_size)
    start = (count_rows // 2, count_columns // 2)
    accepted = np.zeros((count_rows, count_columns), dtype=bool)
    accepted[start] = True
    queue = deque([start])
    while queue:
        row_index, column_index = queue.popleft()
        for next_row, next_column in (
            (row_index - 1, column_index), (row_index + 1, column_index),
            (row_index, column_index - 1), (row_index, column_index + 1),
        ):
            if not (0 <= next_row < count_rows and 0 <= next_column < count_columns):
                continue
            if accepted[next_row, next_column]:
                continue
            row0 = max(rows[row_index], rows[next_row])
            row1 = min(rows[row_index], rows[next_row]) + patch_size
            col0 = max(columns[column_index], columns[next_column])
            col1 = min(columns[column_index], columns[next_column]) + patch_size
            if row0 >= row1 or col0 >= col1:
                continue
            current = mean_grid[
                row_index, column_index,
                row0 - rows[row_index]:row1 - rows[row_index],
                col0 - columns[column_index]:col1 - columns[column_index],
            ]
            neighbor = mean_grid[
                next_row, next_column,
                row0 - rows[next_row]:row1 - rows[next_row],
                col0 - columns[next_column]:col1 - columns[next_column],
            ]
            rmse = float(np.sqrt(np.mean((current - neighbor) ** 2)))
            if rmse < threshold:
                accepted[next_row, next_column] = True
                queue.append((next_row, next_column))
    return accepted


def fit_background(
    video: str | Path,
    *,
    patch_size: int = 4,
    stride: int = 3,
    components: int = 10,
    static_threshold: float = 3.0,
    start_frame: int = 0,
    max_frames: int | None = None,
    frame_step: int = 1,
    jobs: int = 1,
    work_dir: str | Path | None = None,
) -> BackgroundResult:
    """Fit temporal GMMs to overlapping grayscale patches and rebuild a background."""
    if components < 1 or frame_step < 1 or start_frame < 0 or jobs < 1 or static_threshold <= 0:
        raise ValueError("Invalid component count, frame step, start frame or jobs")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    capture = cv.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    try:
        total = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        if total <= start_frame or width < patch_size or height < patch_size:
            raise ValueError("Video is too short or smaller than the patch")
        indices = range(start_frame, total, frame_step)
        frame_count = min(len(indices), max_frames) if max_frames is not None else len(indices)
        if frame_count < components:
            raise ValueError("Number of selected frames must be at least the GMM component count")
        rows = patch_positions(height, patch_size, stride)
        columns = patch_positions(width, patch_size, stride)
        patch_count = len(rows) * len(columns)
        shape = (frame_count, patch_count, patch_size * patch_size)
        with TemporaryDirectory(prefix="video_background_gmm_", dir=work_dir) as directory:
            path = str(Path(directory) / "patches.uint8.mmap")
            patches = np.memmap(path, mode="w+", dtype=np.uint8, shape=shape)
            capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)
            for frame_index in range(frame_count):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read selected frame {frame_index}")
                patches[frame_index] = _frame_patches(frame, rows, columns, patch_size)
                for _ in range(frame_step - 1):
                    capture.grab()
            patches.flush()
            del patches
            batches = [range(begin, min(begin + 64, patch_count))
                       for begin in range(0, patch_count, 64)]
            if jobs == 1:
                fitted = [_fit_batch(path, shape, batch, components) for batch in batches]
            else:
                from joblib import Parallel, delayed
                fitted = Parallel(n_jobs=jobs, backend="loky")(
                    delayed(_fit_batch)(path, shape, batch, components) for batch in batches
                )
            means = np.empty((patch_count, patch_size * patch_size), dtype=np.float32)
            weights = np.empty(patch_count, dtype=np.float32)
            variances = np.empty(patch_count, dtype=np.float32)
            fallback_count = 0
            for batch in fitted:
                for index, mean_patch, weight, variance, fallback in batch:
                    means[index] = mean_patch
                    weights[index] = weight
                    variances[index] = variance
                    fallback_count += int(fallback)
    finally:
        capture.release()

    background_sum = np.zeros((height, width), dtype=np.float32)
    background_count = np.zeros((height, width), dtype=np.float32)
    for index, (row, column) in enumerate((r, c) for r in rows for c in columns):
        patch = means[index].reshape(patch_size, patch_size)
        background_sum[row:row + patch_size, column:column + patch_size] += patch
        background_count[row:row + patch_size, column:column + patch_size] += 1
    background = np.clip(background_sum / np.maximum(background_count, 1), 0, 255).astype(np.uint8)
    selected = _static_patch_mask(means, rows, columns, patch_size, static_threshold)
    static_mask = np.zeros((height, width), dtype=np.uint8)
    for row_index, column_index in np.argwhere(selected):
        row, column = rows[row_index], columns[column_index]
        static_mask[row:row + patch_size, column:column + patch_size] = 255
    return BackgroundResult(
        background, static_mask, means, weights, variances,
        fallback_count, frame_count, (width, height), rows, columns,
    )
