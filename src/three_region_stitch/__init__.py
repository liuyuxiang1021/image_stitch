"""Motion-aware Local-to-Global stitching with three output regions."""

from .motion import MotionConfig, MotionResult, detect_motion
from .regions import CompositeResult, RegionMasks, compose_three_regions

__all__ = [
    "MotionConfig", "MotionResult", "detect_motion",
    "CompositeResult", "RegionMasks", "compose_three_regions",
]
