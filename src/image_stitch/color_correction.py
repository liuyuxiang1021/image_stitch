"""Hybrid histogram matching (HHM) for aligned Local/Global image regions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _histogram(values: np.ndarray) -> np.ndarray:
    return np.bincount(values.astype(np.uint8, copy=False), minlength=256).astype(np.float64)


def _rare_bins(histogram: np.ndarray, threshold: float) -> np.ndarray:
    """Return the least populated bins whose cumulative mass is below threshold."""
    order = np.argsort(histogram, kind="stable")
    cumulative = np.cumsum(histogram[order]) / max(float(histogram.sum()), 1.0)
    return order[cumulative < threshold]


def _channel_mapping(source: np.ndarray, reference: np.ndarray, coeff: float, threshold: float) -> np.ndarray:
    source_hist = _histogram(source)
    reference_hist = _histogram(reference)
    source_cdf = np.cumsum(source_hist) / max(float(source_hist.sum()), 1.0)
    reference_cdf = np.cumsum(reference_hist) / max(float(reference_hist.sum()), 1.0)

    right = np.searchsorted(reference_cdf, source_cdf, side="left").clip(0, 255)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(reference_cdf[left] - source_cdf) <= np.abs(reference_cdf[right] - source_cdf)
    matched = np.where(choose_left, left, right).astype(np.float64)
    identity = np.arange(256, dtype=np.float64)
    mapping = identity + coeff * (matched - identity)

    # Match the original HHM local-straightening behavior: suppress sparse
    # source bins and interpolate across them. The legacy reference-bin loop
    # was a no-op; invalidating those mapped values here collapses broad LUT
    # ranges and creates a gray veil, so retain the source-bin behavior only.
    invalid = np.zeros(256, dtype=bool)
    invalid[_rare_bins(source_hist, threshold)] = True
    expanded = invalid.copy()
    expanded[1:] |= invalid[:-1]
    expanded[:-1] |= invalid[1:]
    invalid = expanded

    mapping[invalid] = np.nan
    valid = np.flatnonzero(np.isfinite(mapping))
    if valid.size == 0:
        mapping = identity
    else:
        mapping = np.interp(np.arange(256), valid, mapping[valid])
        if invalid[0]:
            mapping[0] = 0.0
        if invalid[-1]:
            mapping[-1] = 255.0
    mapping = np.maximum.accumulate(np.clip(mapping, 0, 255))
    return np.rint(mapping).astype(np.uint8)


def fit_hhm_luts(
    paired_samples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    coeff: float = 1.0,
    local_straightening_threshold: float = 0.05,
    max_samples: int = 400_000,
) -> tuple[np.ndarray, dict]:
    """Fit one BGR lookup table from paired aligned source/reference pixels.

    When several source regions are provided, their valid paired pixels are
    pooled before fitting. This is used by three-region stitching so the
    homography and RAFT sources receive exactly the same color transform.
    """
    if not 0.0 <= coeff <= 1.0:
        raise ValueError("HHM coeff must be between 0 and 1")
    if not 0.0 < local_straightening_threshold <= 1.0:
        raise ValueError("HHM local-straightening threshold must be in (0, 1]")
    if max_samples < 0:
        raise ValueError("HHM max_samples must be non-negative")

    collected: list[tuple[np.ndarray, np.ndarray]] = []
    source_counts: list[int] = []
    for source, reference, mask in paired_samples:
        if source.shape != reference.shape or source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("HHM source/reference must be same-shape BGR images")
        if mask.shape != source.shape[:2]:
            raise ValueError("HHM mask must match the source image dimensions")
        valid = mask > 0
        source_pixels = source[valid]
        reference_pixels = reference[valid]
        source_counts.append(int(source_pixels.shape[0]))
        if source_pixels.size:
            collected.append((source_pixels, reference_pixels))

    total_available = sum(source_counts)
    if max_samples and total_available > max_samples:
        # Allocate the cap evenly across contributing regions so a large RAFT
        # area cannot drown out the homography area (or vice versa).
        active = [i for i, (source, _) in enumerate(collected) if source.shape[0] > 0]
        quotas = np.full(len(active), max_samples // len(active), dtype=np.int64)
        quotas[:max_samples % len(active)] += 1
        sampled: list[tuple[np.ndarray, np.ndarray]] = []
        for index, quota in zip(active, quotas):
            source_pixels, reference_pixels = collected[index]
            count = min(int(quota), source_pixels.shape[0])
            if count == source_pixels.shape[0]:
                sampled.append((source_pixels, reference_pixels))
            elif count > 0:
                indices = np.linspace(0, source_pixels.shape[0] - 1, count, dtype=np.int64)
                sampled.append((source_pixels[indices], reference_pixels[indices]))
        collected = sampled

    if collected:
        source_samples = np.concatenate([item[0] for item in collected], axis=0)
        reference_samples = np.concatenate([item[1] for item in collected], axis=0)
    else:
        source_samples = np.empty((0, 3), dtype=np.uint8)
        reference_samples = np.empty((0, 3), dtype=np.uint8)

    if source_samples.shape[0] < 32:
        luts = np.tile(np.arange(256, dtype=np.uint8), (3, 1))
        applied = False
        reason = "fewer_than_32_paired_pixels"
    else:
        luts = np.stack([
            _channel_mapping(
                source_samples[:, channel], reference_samples[:, channel],
                coeff, local_straightening_threshold,
            )
            for channel in range(3)
        ])
        applied = coeff > 0
        reason = "ok" if applied else "zero_coefficient"

    stats = {
        "method": "HHM",
        "applied": applied,
        "reason": reason,
        "coeff": float(coeff),
        "local_straightening_threshold": float(local_straightening_threshold),
        "paired_pixels_available": total_available,
        "paired_pixels_sampled": int(source_samples.shape[0]),
        "paired_pixels_by_source_region": source_counts,
        "mapping_shared_across_source_regions": len(paired_samples) > 1,
    }
    return luts, stats


def fit_hhm_for_regions(
    reference: np.ndarray,
    sources: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    coeff: float = 1.0,
    local_straightening_threshold: float = 0.05,
    max_samples: int = 400_000,
) -> tuple[np.ndarray, dict]:
    """Fit HHM from pixels owned by each source region, paired to reference."""
    if len(sources) != len(masks):
        raise ValueError("Each HHM source needs one ownership mask")
    return fit_hhm_luts(
        [(source, reference, mask) for source, mask in zip(sources, masks)],
        coeff=coeff,
        local_straightening_threshold=local_straightening_threshold,
        max_samples=max_samples,
    )


def apply_hhm_luts(image: np.ndarray, luts: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Apply fitted BGR lookup tables while preserving invalid warp pixels."""
    luts = np.asarray(luts, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or luts.shape != (3, 256):
        raise ValueError("Expected a BGR image and three 256-entry HHM lookup tables")
    corrected = np.stack([luts[channel][image[..., channel]] for channel in range(3)], axis=-1)
    if valid_mask is not None:
        if valid_mask.shape != image.shape[:2]:
            raise ValueError("HHM validity mask must match the image dimensions")
        corrected[valid_mask == 0] = 0
    return corrected
