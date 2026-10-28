import unittest

import numpy as np

from point_line_stitch.synchronization import (
    best_candidate,
    masked_ssim,
    select_with_confidence,
    target_motion_mask,
)


class SynchronizationTests(unittest.TestCase):
    def test_moving_object_alignment_beats_unsynchronized_frame(self):
        background = np.full((96, 128, 3), 65, np.uint8)
        target = background.copy()
        target[30:58, 48:75] = (220, 170, 110)
        neighbor = background.copy()
        neighbor[30:58, 18:45] = (220, 170, 110)
        wrong = background.copy()
        wrong[30:58, 78:105] = (220, 170, 110)
        mask = target_motion_mask(target, [neighbor], np.ones((96, 128), np.uint8))
        self.assertGreater(np.count_nonzero(mask), 100)
        self.assertGreater(masked_ssim(target, target, mask), masked_ssim(target, wrong, mask))

    def test_tie_breaking_prefers_anchor(self):
        candidates = [
            {"local_frame_index": 98, "masked_ssim": 0.5},
            {"local_frame_index": 101, "masked_ssim": 0.5},
        ]
        self.assertEqual(best_candidate(candidates, 100)["local_frame_index"], 101)

    def test_flat_or_boundary_peak_keeps_existing_frame(self):
        scores = [
            {"local_frame_index": 9, "masked_ssim": 0.501},
            {"local_frame_index": 10, "masked_ssim": 0.500},
            {"local_frame_index": 11, "masked_ssim": 0.499},
        ]
        result = select_with_confidence(scores, 10, 10)
        self.assertEqual(result["selected"]["local_frame_index"], 10)
        self.assertEqual(result["decision"], "peak_at_search_boundary")


if __name__ == "__main__":
    unittest.main()
