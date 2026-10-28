"""Compatibility imports for the former pure-stitching package name."""

import sys
from importlib import import_module

from point_line_stitch import *  # noqa: F401,F403
from point_line_stitch import __version__

for _module in (
    "cli", "color_correction", "core", "high_resolution", "matchers",
    "optimization", "types", "visualization",
):
    sys.modules[f"{__name__}.{_module}"] = import_module(
        f"point_line_stitch.{_module}"
    )
