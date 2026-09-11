import unittest

import cv2 as cv
import numpy as np

from image_stitch import FeatureMatches, estimate_point_line_homography


class OptimizationTests(unittest.TestCase):
    def test_recovers_homography_from_points_and_lines(self):
        rng = np.random.default_rng(1021)
        expected = np.array(
            [[1.03, 0.025, 38.0], [-0.015, 1.02, 21.0], [0.00008, -0.00005, 1.0]],
            dtype=np.float64,
        )
        points_source = rng.uniform([20, 20], [600, 380], size=(50, 2)).astype(np.float32)
        points_destination = cv.perspectiveTransform(
            points_source.reshape(-1, 1, 2), expected
        ).reshape(-1, 2)
        points_destination += rng.normal(0, 0.15, points_destination.shape)
        points_destination[-5:] = rng.uniform([0, 0], [650, 430], size=(5, 2))

        lines_source = np.array(
            [
                [[30, 50], [560, 50]],
                [[40, 150], [590, 150]],
                [[70, 320], [520, 320]],
                [[100, 20], [100, 360]],
                [[300, 30], [300, 370]],
                [[500, 40], [500, 350]],
            ],
            dtype=np.float32,
        )
        lines_destination = cv.perspectiveTransform(
            lines_source.reshape(-1, 1, 2), expected
        ).reshape(-1, 2, 2)
        matches = FeatureMatches(
            points_source,
            points_destination.astype(np.float32),
            np.ones(len(points_source), dtype=np.float32),
            lines_source,
            lines_destination.astype(np.float32),
            np.ones(len(lines_source), dtype=np.float32),
        )
        estimated, filtered, diagnostics = estimate_point_line_homography(matches)
        probes = np.float32([[[50, 60]], [[300, 200]], [[580, 350]]])
        expected_probes = cv.perspectiveTransform(probes, expected)
        actual_probes = cv.perspectiveTransform(probes, estimated)
        np.testing.assert_allclose(actual_probes, expected_probes, atol=0.8)
        self.assertEqual(len(filtered.lines_source), len(lines_source))
        self.assertLess(diagnostics.point_rmse, 0.5)
        self.assertLess(diagnostics.line_rmse, 0.5)


if __name__ == "__main__":
    unittest.main()
