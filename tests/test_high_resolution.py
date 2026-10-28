import cv2 as cv
import numpy as np

from image_stitch import compose_local_roi


def test_local_roi_uses_original_local_pixels_and_clipped_global_crop():
    global_frame = np.full((50, 50, 3), (20, 40, 60), np.uint8)
    local = np.full((8, 8, 3), (80, 100, 120), np.uint8)
    local[3, 3] = (220, 10, 30)
    homography = np.array(
        [[0.5, 0, 10], [0, 0.5, 10], [0, 0, 1]], dtype=np.float64,
    )
    result = compose_local_roi(
        local, global_frame, homography,
        scale=4, margin=2, feather_width=0, color_correction=False,
    )

    assert result.global_roi_xyxy == (8, 8, 17, 17)
    assert result.image.shape == (36, 36, 3)
    assert result.background.shape == result.image.shape
    np.testing.assert_array_equal(result.background[0, 0], global_frame[0, 0])
    np.testing.assert_array_equal(result.image[0, 0], global_frame[0, 0])
    target = cv.perspectiveTransform(
        np.array([[[3.0, 3.0]]], np.float64), result.source_to_roi,
    )[0, 0]
    np.testing.assert_array_equal(result.image[round(target[1]), round(target[0])], local[3, 3])
