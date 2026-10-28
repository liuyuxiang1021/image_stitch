"""Compatibility entry point; use render_point_line_stitch_roi.py for new work."""

from runpy import run_path
from pathlib import Path

run_path(str(Path(__file__).with_name("render_point_line_stitch_roi.py")), run_name="__main__")
