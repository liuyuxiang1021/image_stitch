"""Static background reconstruction from video with patchwise temporal GMMs."""

from .core import BackgroundResult, fit_background, patch_positions

__all__ = ["BackgroundResult", "fit_background", "patch_positions"]
