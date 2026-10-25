import unittest

import numpy as np

from three_region_stitch.regions import compose_three_regions, prepare_regions


class ThreeRegionOwnershipTests(unittest.TestCase):
    def test_dual_motion_intersection_and_one_sided_raft(self):
        shape = (12, 12, 3)
        global_frame = np.zeros(shape, np.uint8)
        global_frame[:] = (0, 255, 0)
        local_frame = np.zeros(shape, np.uint8)
        local_frame[:] = (0, 0, 255)
        raft_frame = np.zeros(shape, np.uint8)
        raft_frame[:] = (255, 0, 0)
        local_motion = np.zeros(shape[:2], np.uint8)
        global_motion = np.zeros(shape[:2], np.uint8)
        local_motion[4, 4] = local_motion[5, 5] = 255
        global_motion[4, 4] = global_motion[6, 6] = 255
        raft_valid = np.full(shape[:2], 255, np.uint8)
        raft_valid[6, 6] = 0
        identity = np.eye(3, dtype=np.float64)

        result = compose_three_regions(
            global_frame, local_frame, identity, identity, (0, 0, 12, 12),
            local_motion, global_motion, raft_frame, raft_valid,
            feather_radius=0,
        )
        self.assertEqual(result.regions.homography_region[4, 4], 255)
        self.assertEqual(result.regions.homography_region[5, 5], 0)
        self.assertEqual(result.regions.raft_region[5, 5], 255)
        self.assertEqual(result.regions.global_region[6, 6], 255)
        self.assertEqual(tuple(result.image[4, 4]), (0, 0, 255))
        self.assertEqual(tuple(result.image[5, 5]), (255, 0, 0))
        self.assertEqual(tuple(result.image[6, 6]), (0, 255, 0))
        ownership = sum(
            mask.astype(np.uint16) for mask in (
                result.regions.global_region,
                result.regions.raft_region,
                result.regions.homography_region,
            )
        )
        self.assertTrue(np.all(ownership == 255))

    def test_local_motion_transforms_by_h_delta(self):
        shape = (12, 12, 3)
        frame = np.zeros(shape, np.uint8)
        local_motion = np.zeros(shape[:2], np.uint8)
        global_motion = np.zeros(shape[:2], np.uint8)
        local_motion[4, 4] = 255
        global_motion[4, 6] = 255
        h_final = np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]], np.float64)
        result = prepare_regions(
            frame, frame, np.eye(3), h_final, (0, 0, 12, 12),
            local_motion, global_motion,
        )
        self.assertEqual(result.local_motion_final[4, 6], 255)
        self.assertEqual(result.mutual_motion[4, 6], 255)
        self.assertEqual(result.mutual_motion[4, 4], 0)


if __name__ == "__main__":
    unittest.main()
