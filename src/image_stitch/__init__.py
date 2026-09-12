"""Point-line homography image stitching."""

from .core import (
    compose_pair,
    estimate_homography,
    load_homography,
    refine_homography,
    stitch_pair,
)
from .matchers import LineTRLineMatcher, OmniGluePointMatcher
from .optimization import estimate_point_line_homography, optimize_homography
from .types import (
    FeatureMatches,
    HomographyEstimationError,
    HomographyResult,
    OptimizationDiagnostics,
    StitchResult,
)

__all__ = [
    "HomographyEstimationError",
    "FeatureMatches",
    "HomographyResult",
    "OptimizationDiagnostics",
    "StitchResult",
    "OmniGluePointMatcher",
    "LineTRLineMatcher",
    "compose_pair",
    "estimate_homography",
    "load_homography",
    "refine_homography",
    "stitch_pair",
    "estimate_point_line_homography",
    "optimize_homography",
]

__version__ = "0.1.0"
