import unittest

import numpy as np

from image_stitch.types import FeatureMatches
from image_stitch.visualization import (
    draw_initial_alignment,
    draw_matches,
    projected_source_bounds,
)


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((80, 100, 3), (40, 120, 220), dtype=np.uint8)
        self.destination = np.full((240, 320, 3), (80, 60, 30), dtype=np.uint8)
        self.homography = np.array(
            [[1.0, 0.0, 120.0], [0.0, 1.0, 80.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def test_initial_alignment_contains_overview_and_zoom(self):
        region = projected_source_bounds(
            self.source, self.destination, self.homography, padding=10
        )
        self.assertEqual(region, (110, 70, 120, 100))
        visualization = draw_initial_alignment(
            self.source,
            self.destination,
            self.homography,
            panel_width=320,
            panel_height=180,
            padding=10,
        )
        self.assertEqual(visualization.shape, (232, 656, 3))

    def test_match_visualization_can_zoom_to_projected_region(self):
        points = np.float32([[130, 90], [180, 120], [210, 145], [150, 135]])
        matches = FeatureMatches(
            points,
            points.copy(),
            np.ones(len(points), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
        region = projected_source_bounds(
            self.source, self.destination, self.homography, padding=10
        )
        visualization = draw_matches(
            self.destination,
            self.destination,
            matches,
            region=region,
        )
        self.assertEqual(visualization.shape, (100, 252, 3))


if __name__ == "__main__":
    unittest.main()
