import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from video_background_removal import fit_background, patch_positions
from video_background_removal.core import _fit_patch


class VideoBackgroundTests(unittest.TestCase):
    def test_patch_grid_covers_non_divisible_edge(self):
        self.assertEqual(patch_positions(11, 4, 3), (0, 3, 6, 7))
        with self.assertRaises(ValueError):
            patch_positions(11, 4, 5)

    def test_dominant_temporal_component_becomes_background(self):
        samples = np.vstack((
            np.full((30, 16), 40, np.uint8),
            np.full((10, 16), 210, np.uint8),
        ))
        mean, weight, _, fallback = _fit_patch(samples, components=2)
        self.assertTrue(np.allclose(mean, 40, atol=3))
        self.assertGreater(weight, 0.7)
        self.assertFalse(fallback)

    def test_video_reconstruction_keeps_mostly_visible_background(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "motion.avi"
            writer = cv.VideoWriter(str(video), cv.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
            self.assertTrue(writer.isOpened())
            for index in range(20):
                frame = np.full((16, 16, 3), 60, np.uint8)
                frame[5:9, (index % 4) * 3:(index % 4) * 3 + 3] = 220
                writer.write(frame)
            writer.release()
            result = fit_background(video, patch_size=4, stride=3, components=2, jobs=1)
            self.assertEqual(result.background.shape, (16, 16))
            self.assertEqual(result.frames_used, 20)
            self.assertLess(float(np.mean(result.background)), 95)
            self.assertEqual(result.static_mask.shape, (16, 16))


if __name__ == "__main__":
    unittest.main()
