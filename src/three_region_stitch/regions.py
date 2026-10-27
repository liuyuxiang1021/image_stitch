"""Three-region ownership and edge-limited compositing on a Global canvas."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from image_stitch.color_correction import apply_hhm_luts, fit_hhm_for_regions


DEFAULT_RAFT_OPEN_KERNEL_SIZE = 3
DEFAULT_RAFT_MIN_COMPONENT_AREA = 256


@dataclass(frozen=True)
class RegionMasks:
    local_motion_final: np.ndarray
    mutual_motion: np.ndarray
    static_exclusion: np.ndarray
    global_region: np.ndarray
    raft_region: np.ndarray
    homography_region: np.ndarray
    raft_cleanup_stats: dict[str, int] | None = None


@dataclass(frozen=True)
class CompositeResult:
    image: np.ndarray
    homography_warp: np.ndarray
    regions: RegionMasks
    color_correction: dict | None = None
    color_luts: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedRegions:
    homography_warp: np.ndarray
    valid_local: np.ndarray
    local_motion_final: np.ndarray
    mutual_motion: np.ndarray
    static_exclusion: np.ndarray
    static_candidate: np.ndarray


def embed_roi(mask: np.ndarray, roi_xyxy: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi_xyxy
    width, height = size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI is outside canvas")
    if mask.shape != (y1 - y0, x1 - x0):
        raise ValueError(f"ROI mask shape {mask.shape} does not match {(y1-y0, x1-x0)}")
    result = np.zeros((height, width), np.uint8)
    result[y0:y1, x0:x1] = (mask > 0).astype(np.uint8) * 255
    return result


def prepare_regions(
    global_frame: np.ndarray,
    local_frame: np.ndarray,
    h_initial: np.ndarray,
    h_final: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    local_motion_roi: np.ndarray,
    global_motion_roi: np.ndarray,
    *,
    one_sided_to_raft: bool = True,
) -> PreparedRegions:
    height, width = global_frame.shape[:2]
    h_initial = np.asarray(h_initial, dtype=np.float64)
    h_final = np.asarray(h_final, dtype=np.float64)
    h_delta = h_final @ np.linalg.inv(h_initial)
    local_warp = cv.warpPerspective(local_frame, h_final, (width, height))
    valid_local = cv.warpPerspective(
        np.full(local_frame.shape[:2], 255, np.uint8),
        h_final, (width, height), flags=cv.INTER_NEAREST,
    )
    local_initial = embed_roi(local_motion_roi, roi_xyxy, (width, height))
    global_motion = embed_roi(global_motion_roi, roi_xyxy, (width, height))
    local_final = cv.warpPerspective(
        local_initial, h_delta, (width, height), flags=cv.INTER_NEAREST,
    )
    mutual = cv.bitwise_and(cv.bitwise_and(local_final, global_motion), valid_local)
    exclusion = mutual if one_sided_to_raft else cv.bitwise_or(local_final, global_motion)
    static_candidate = cv.bitwise_and(valid_local, cv.bitwise_not(exclusion))
    return PreparedRegions(
        local_warp, valid_local, local_final, mutual, exclusion, static_candidate,
    )


def feather_sources(
    sources: tuple[np.ndarray, ...],
    masks: tuple[np.ndarray, ...],
    radius: int,
) -> np.ndarray:
    if not sources or len(sources) != len(masks):
        raise ValueError("Each feathered source must have a corresponding ownership mask")
    if any(source.shape != sources[0].shape for source in sources):
        raise ValueError("Feathered sources must have the same canvas shape")
    if any(mask.shape != sources[0].shape[:2] for mask in masks):
        raise ValueError("Feathering masks must match the source canvas")
    if radius <= 0:
        result = sources[0].copy()
        for source, mask in zip(sources[1:], masks[1:]):
            result[mask > 0] = source[mask > 0]
        return result
    kernel = 2 * radius + 1
    weights = [
        cv.GaussianBlur((mask > 0).astype(np.float32), (kernel, kernel),
                        sigmaX=max(radius / 2, 1))
        for mask in masks
    ]
    denominator = np.maximum(sum(weights), 1e-6)
    blended = sum(
        source.astype(np.float32) * (weight / denominator)[..., None]
        for source, weight in zip(sources, weights)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _feather_three_sources(
    global_frame: np.ndarray,
    raft_warped: np.ndarray,
    homography_warped: np.ndarray,
    masks: RegionMasks,
    radius: int,
) -> np.ndarray:
    return feather_sources(
        (global_frame, raft_warped, homography_warped),
        (masks.global_region, masks.raft_region, masks.homography_region),
        radius,
    )


def _clean_raft_region(
    raft_region: np.ndarray,
    *,
    open_kernel_size: int = DEFAULT_RAFT_OPEN_KERNEL_SIZE,
    min_component_area: int = DEFAULT_RAFT_MIN_COMPONENT_AREA,
) -> np.ndarray:
    """Remove speckles from the RAFT ownership mask without growing it.

    A small morphological opening removes pixel-scale noise. Connected
    components smaller than ``min_component_area`` are then discarded; those
    pixels become Global-owned when the complementary region is constructed.
    """
    if open_kernel_size < 0 or min_component_area < 0:
        raise ValueError("RAFT cleanup parameters must be non-negative")
    clean = (raft_region > 0).astype(np.uint8) * 255
    if open_kernel_size > 1:
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size),
        )
        clean = cv.morphologyEx(clean, cv.MORPH_OPEN, kernel)
    if min_component_area > 1 and np.any(clean):
        count, labels, stats, _ = cv.connectedComponentsWithStats(clean, 8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv.CC_STAT_AREA] >= min_component_area
        clean = keep[labels].astype(np.uint8) * 255
    return clean


def compose_homography_over_background(
    background: np.ndarray,
    source: np.ndarray,
    homography: np.ndarray,
    feather_radius: int,
    *,
    color_luts: np.ndarray | None = None,
    color_correction: bool = True,
    color_correction_coeff: float = 1.0,
    color_correction_threshold: float = 0.5,
    color_correction_max_samples: int = 400_000,
) -> np.ndarray:
    """Warp one source onto a full background canvas and feather its boundary."""
    height, width = background.shape[:2]
    homography = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    warped = cv.warpPerspective(source, homography, (width, height))
    source_valid = cv.warpPerspective(
        np.full(source.shape[:2], 255, dtype=np.uint8),
        homography,
        (width, height),
        flags=cv.INTER_NEAREST,
    )
    if color_correction:
        if color_luts is None:
            color_luts, _ = fit_hhm_for_regions(
                background, [warped], [source_valid],
                coeff=color_correction_coeff,
                local_straightening_threshold=color_correction_threshold,
                max_samples=color_correction_max_samples,
            )
        warped = apply_hhm_luts(warped, color_luts, source_valid)
    background_region = cv.bitwise_not(source_valid)
    return feather_sources(
        (background, warped),
        (background_region, source_valid),
        feather_radius,
    )


def compose_three_regions(
    global_frame: np.ndarray,
    local_frame: np.ndarray,
    h_initial: np.ndarray,
    h_final: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    local_motion_roi: np.ndarray,
    global_motion_roi: np.ndarray,
    raft_warped: np.ndarray,
    raft_valid_mask: np.ndarray,
    *,
    one_sided_to_raft: bool = True,
    feather_radius: int = 2,
    raft_open_kernel_size: int = DEFAULT_RAFT_OPEN_KERNEL_SIZE,
    raft_min_component_area: int = DEFAULT_RAFT_MIN_COMPONENT_AREA,
    color_correction: bool = True,
    color_correction_coeff: float = 1.0,
    color_correction_threshold: float = 0.5,
    color_correction_max_samples: int = 400_000,
) -> CompositeResult:
    """Red=both motion, blue=flow-valid Local static, green=Global.

    The Local motion mask is first transformed from H_init coordinates by
    H_delta. A one-sided motion pixel is eligible for RAFT only when its
    forward/backward flow is valid; otherwise Global owns it.
    """
    height, width = global_frame.shape[:2]
    if raft_warped.shape != global_frame.shape or raft_valid_mask.shape != (height, width):
        raise ValueError("RAFT image and validity mask must match the Global canvas")
    prepared = prepare_regions(
        global_frame, local_frame, h_initial, h_final, roi_xyxy,
        local_motion_roi, global_motion_roi, one_sided_to_raft=one_sided_to_raft,
    )
    raft_region = cv.bitwise_and(
        prepared.static_candidate, (raft_valid_mask > 0).astype(np.uint8) * 255,
    )
    raft_region = cv.bitwise_and(raft_region, cv.bitwise_not(prepared.mutual_motion))
    raft_pixels_before = int(np.count_nonzero(raft_region))
    raft_components_before = max(
        0, cv.connectedComponentsWithStats(raft_region, 8)[0] - 1,
    ) if raft_pixels_before else 0
    raft_region = _clean_raft_region(
        raft_region,
        open_kernel_size=raft_open_kernel_size,
        min_component_area=raft_min_component_area,
    )
    raft_pixels_after = int(np.count_nonzero(raft_region))
    raft_components_after = max(
        0, cv.connectedComponentsWithStats(raft_region, 8)[0] - 1,
    ) if raft_pixels_after else 0
    global_region = cv.bitwise_not(cv.bitwise_or(prepared.mutual_motion, raft_region))
    masks = RegionMasks(
        prepared.local_motion_final, prepared.mutual_motion,
        prepared.static_exclusion, global_region, raft_region,
        prepared.mutual_motion,
        {
            "raft_pixels_before_cleanup": raft_pixels_before,
            "raft_pixels_after_cleanup": raft_pixels_after,
            "raft_pixels_removed": raft_pixels_before - raft_pixels_after,
            "connected_components_before_cleanup": raft_components_before,
            "connected_components_after_cleanup": raft_components_after,
        },
    )
    homography_warp = prepared.homography_warp
    correction_stats = {
        "method": "HHM", "applied": False, "reason": "disabled",
        "paired_pixels_available": 0, "paired_pixels_sampled": 0,
        "mapping_shared_across_source_regions": False,
    }
    color_luts = None
    if color_correction:
        color_luts, correction_stats = fit_hhm_for_regions(
            global_frame,
            [raft_warped, homography_warp],
            [masks.raft_region, masks.homography_region],
            coeff=color_correction_coeff,
            local_straightening_threshold=color_correction_threshold,
            max_samples=color_correction_max_samples,
        )
        raft_warped = apply_hhm_luts(raft_warped, color_luts, masks.raft_region)
        homography_warp = apply_hhm_luts(homography_warp, color_luts, prepared.valid_local)
    result = _feather_three_sources(
        global_frame, raft_warped, homography_warp, masks, feather_radius,
    )
    return CompositeResult(result, homography_warp, masks, correction_stats, color_luts)


def compose_with_global_sr(
    global_sr: np.ndarray,
    local_frame: np.ndarray,
    h_final: np.ndarray,
    raft_warped: np.ndarray,
    regions: RegionMasks,
    *,
    feather_radius: int = 2,
    raft_flow: np.ndarray | None = None,
    color_correction: bool = True,
    color_correction_coeff: float = 1.0,
    color_correction_threshold: float = 0.5,
    color_correction_max_samples: int = 400_000,
) -> CompositeResult:
    """Composite Local H/RAFT regions over a super-resolved Global background.

    ``h_final`` maps source Local coordinates to the native Global canvas.
    The Local homography warp is rendered directly at the enlarged output size.
    When the native RAFT flow is supplied, its displacement vectors are scaled
    and applied to this high-resolution Local warp to preserve source detail.
    """
    base_height, base_width = raft_warped.shape[:2]
    out_height, out_width = global_sr.shape[:2]
    if raft_warped.ndim != 3 or global_sr.ndim != 3:
        raise ValueError("Global SR and RAFT inputs must be color images")
    if global_sr.shape[2] != 3 or raft_warped.shape[2] != 3:
        raise ValueError("Global SR and RAFT inputs must have three color channels")
    if out_width % base_width or out_height % base_height:
        raise ValueError("Global SR canvas must be an integer multiple of the native canvas")
    scale_x, scale_y = out_width // base_width, out_height // base_height
    if scale_x != scale_y:
        raise ValueError("Global SR canvas must use the same scale in both dimensions")
    if scale_x < 1:
        raise ValueError("Global SR canvas cannot be smaller than the native canvas")

    size = (out_width, out_height)
    scale_matrix = np.diag([float(scale_x), float(scale_y), 1.0])
    h_highres = scale_matrix @ np.asarray(h_final, dtype=np.float64)
    homography_warp = cv.warpPerspective(local_frame, h_highres, size)

    def enlarge_mask(mask: np.ndarray) -> np.ndarray:
        if mask.shape != (base_height, base_width):
            raise ValueError("Region masks must match the native Global canvas")
        return cv.resize(mask, size, interpolation=cv.INTER_NEAREST)

    highres_regions = RegionMasks(
        local_motion_final=enlarge_mask(regions.local_motion_final),
        mutual_motion=enlarge_mask(regions.mutual_motion),
        static_exclusion=enlarge_mask(regions.static_exclusion),
        global_region=enlarge_mask(regions.global_region),
        raft_region=enlarge_mask(regions.raft_region),
        homography_region=enlarge_mask(regions.homography_region),
        raft_cleanup_stats=regions.raft_cleanup_stats,
    )
    if raft_flow is None:
        raft_highres = cv.resize(raft_warped, size, interpolation=cv.INTER_CUBIC)
    else:
        if raft_flow.shape != (base_height, base_width, 2):
            raise ValueError("RAFT flow must have shape HxWx2 on the native Global canvas")
        flow_highres = cv.resize(raft_flow, size, interpolation=cv.INTER_LINEAR)
        flow_highres *= float(scale_x)
        yy, xx = np.mgrid[:out_height, :out_width].astype(np.float32)
        raft_highres = cv.remap(
            homography_warp,
            xx + flow_highres[..., 0],
            yy + flow_highres[..., 1],
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
        )
    correction_stats = {
        "method": "HHM", "applied": False, "reason": "disabled",
        "paired_pixels_available": 0, "paired_pixels_sampled": 0,
        "mapping_shared_across_source_regions": False,
    }
    color_luts = None
    if color_correction:
        color_luts, correction_stats = fit_hhm_for_regions(
            global_sr,
            [raft_highres, homography_warp],
            [highres_regions.raft_region, highres_regions.homography_region],
            coeff=color_correction_coeff,
            local_straightening_threshold=color_correction_threshold,
            max_samples=color_correction_max_samples,
        )
        raft_highres = apply_hhm_luts(raft_highres, color_luts, highres_regions.raft_region)
        homography_warp = apply_hhm_luts(
            homography_warp, color_luts,
            cv.bitwise_or(highres_regions.raft_region, highres_regions.homography_region),
        )
    highres_result = _feather_three_sources(
        global_sr, raft_highres, homography_warp, highres_regions,
        feather_radius * scale_x,
    )
    return CompositeResult(highres_result, homography_warp, highres_regions, correction_stats, color_luts)
