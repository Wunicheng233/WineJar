from unittest.mock import patch

import cv2
import numpy as np

from xarm_sdk_control.multiview_calibration import find_chessboard_corners, orient_chessboard_corners


def test_mixed_parity_board_order_is_independent_of_detector_direction():
    y, x = np.indices((180, 240))
    gray = (((x // 20 + y // 20) % 2) * 255).astype(np.uint8)
    corners = np.array([[[x * 20., y * 20.]] for y in range(1, 9) for x in range(1, 12)], np.float32)
    np.testing.assert_array_equal(orient_chessboard_corners(gray, corners, (11, 8)), corners)
    np.testing.assert_array_equal(orient_chessboard_corners(gray, corners[::-1].copy(), (11, 8)), corners)


def test_fallback_refinement_window_respects_corner_spacing():
    corners = np.array([[[x * 20.0, y * 8.0]] for y in range(8) for x in range(11)], np.float32)
    with (
        patch.object(cv2, "findChessboardCornersSB", return_value=(False, None)),
        patch.object(cv2, "findChessboardCorners", return_value=(True, corners)),
        patch.object(cv2, "cornerSubPix", return_value=corners) as refine,
    ):
        found, result = find_chessboard_corners(np.zeros((200, 300, 3), np.uint8), (11, 8))
    assert found
    np.testing.assert_array_equal(result, corners)
    assert refine.call_args.args[2] == (3, 3)


def test_sb_result_does_not_need_additional_refinement():
    corners = np.zeros((88, 1, 2), np.float32)
    with (
        patch.object(cv2, "findChessboardCornersSB", return_value=(True, corners)),
        patch.object(cv2, "cornerSubPix") as refine,
    ):
        found, _ = find_chessboard_corners(np.zeros((200, 300, 3), np.uint8), (11, 8))
    assert found
    refine.assert_not_called()
