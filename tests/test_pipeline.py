import unittest

import cv2 as cv
import numpy as np

from image_stitch import refine_homography, stitch_pair


class StaticPointMatcher:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination

    def match(self, _source_image, _destination_image):
        return self.source, self.destination, np.ones(len(self.source), dtype=np.float32)


class StaticLineMatcher:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination

    def match(self, _source_image, _destination_image):
        return self.source, self.destination, np.ones(len(self.source), dtype=np.float32)


class PipelineTests(unittest.TestCase):
    def test_direct_pair_pipeline(self):
        source = np.full((180, 260, 3), (40, 80, 160), dtype=np.uint8)
        destination = np.full((180, 260, 3), (160, 80, 40), dtype=np.uint8)
        points_source = np.float32(
            [[20, 20], [120, 20], [230, 30], [30, 90], [140, 100], [220, 150], [50, 160]]
        )
        homography = np.array([[1, 0, -100], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        points_destination = cv.perspectiveTransform(
            points_source.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        lines_source = np.float32(
            [
                [[20, 30], [230, 30]],
                [[20, 80], [230, 80]],
                [[20, 140], [230, 140]],
                [[40, 10], [40, 170]],
            ]
        )
        lines_destination = cv.perspectiveTransform(
            lines_source.reshape(-1, 1, 2), homography
        ).reshape(-1, 2, 2)
        result = stitch_pair(
            source,
            destination,
            StaticPointMatcher(points_source, points_destination),
            StaticLineMatcher(lines_source, lines_destination),
            max_match_width=0,
        )
        self.assertGreaterEqual(result.panorama.shape[1], 359)
        self.assertLessEqual(result.panorama.shape[1], 361)
        np.testing.assert_allclose(
            result.estimation.homography / result.estimation.homography[2, 2],
            homography,
            atol=1e-4,
        )

    def test_refines_an_existing_initial_homography(self):
        source = np.zeros((180, 260, 3), dtype=np.uint8)
        destination = np.zeros((180, 260, 3), dtype=np.uint8)
        initial_h = np.array([[1, 0, -95], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        points_source = np.float32(
            [[20, 20], [80, 20], [150, 30], [30, 90], [100, 100], [150, 150]]
        )
        delta = np.array([[1, 0, -5], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        points_destination = cv.perspectiveTransform(
            points_source.reshape(-1, 1, 2), delta
        ).reshape(-1, 2)
        lines_source = np.float32(
            [
                [[20, 30], [150, 30]],
                [[20, 80], [150, 80]],
                [[20, 140], [150, 140]],
                [[40, 10], [40, 170]],
            ]
        )
        lines_destination = cv.perspectiveTransform(
            lines_source.reshape(-1, 1, 2), delta
        ).reshape(-1, 2, 2)
        result = refine_homography(
            source,
            destination,
            initial_h,
            StaticPointMatcher(points_source, points_destination),
            StaticLineMatcher(lines_source, lines_destination),
            max_match_width=0,
        )
        expected = delta @ initial_h
        np.testing.assert_allclose(result.homography, expected, atol=1e-4)
        np.testing.assert_allclose(result.delta_homography, delta, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
