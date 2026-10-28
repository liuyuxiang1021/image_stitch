import unittest

import numpy as np

from point_line_stitch.types import FeatureMatches
from point_line_stitch.visualization import (
    _display_indices,
    _draw_dashed_contours,
    draw_final_alignment,
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
        self.assertEqual(visualization.shape, (180, 656, 3))

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

    def test_final_alignment_contains_overview_and_zoom(self):
        source_mask = np.zeros(self.destination.shape[:2], dtype=np.uint8)
        source_mask[80:160, 120:220] = 255
        visualization = draw_final_alignment(
            self.destination,
            source_mask,
            panel_width=320,
            panel_height=180,
            padding=10,
        )
        self.assertEqual(visualization.shape, (180, 656, 3))

    def test_display_sampling_limits_only_the_plot(self):
        locations = np.float32([[x * 10, y * 10] for y in range(4) for x in range(6)])
        scores = np.linspace(1.0, 0.1, len(locations), dtype=np.float32)
        chosen = _display_indices(locations, scores, 8, (40, 60))
        self.assertEqual(len(chosen), 8)
        self.assertEqual(len(set(chosen.tolist())), 8)
        np.testing.assert_array_equal(_display_indices(locations, scores, None, (40, 60)), np.arange(24))

    def test_dashed_contour_contains_visible_gaps(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        contour = np.int32([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]])
        _draw_dashed_contours(image, (contour,))
        top_edge = np.any(image[10, 10:91] > 0, axis=1)
        self.assertTrue(np.any(top_edge))
        self.assertTrue(np.any(~top_edge))


if __name__ == "__main__":
    unittest.main()
