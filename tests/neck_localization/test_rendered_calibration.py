import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from neck_localization.rendered_calibration import (BoardSettings, order_detected_corners,
    transform_from_rt, world_camera_from_known_board, calibrate_intrinsics, board_pose_plan,
    T_camera_from_board_plan, detect_checkerboard)


class RenderedCalibrationTests(unittest.TestCase):
    def test_actual_image_detection_and_printed_marker_order(self):
        board = BoardSettings()
        rgb = np.full((480, 640, 3), 245, np.uint8)
        square_px, origin = 30, np.array([160, 130])
        for row in range(board.rows+1):
            for col in range(board.columns+1):
                x, y = origin+(np.array([col, row])-1)*square_px
                rgb[y:y+square_px, x:x+square_px] = 20 if (row+col)%2 == 0 else 245
        for point, color in zip(board.marker_points(), [[245, 5, 5], [5, 245, 5]]):
            centre = np.rint(origin+point*square_px/board.square_m).astype(int)
            x, y = centre
            rgb[y-5:y+6, x-5:x+6] = color
        corners, diagnostic = detect_checkerboard(rgb, board)
        self.assertEqual(corners.shape, (54, 1, 2))
        self.assertLess(np.linalg.norm(corners[0, 0]-(origin-.5)), .5)
        self.assertIn('direct subpixel', diagnostic['detector'])

    def test_transform_direction_world_board_camera(self):
        Twc = np.eye(4); Twc[:3, :3] = Rotation.from_euler('xyz', [20, -5, 37], degrees=True).as_matrix()
        Twc[:3, 3] = [.5, -.2, .8]
        Tcb = T_camera_from_board_plan(board_pose_plan()[0])
        rvec = cv2.Rodrigues(Tcb[:3, :3])[0]
        actual = world_camera_from_known_board(Twc@Tcb, rvec, Tcb[:3, 3])
        np.testing.assert_allclose(actual, Twc, atol=1e-14)

    def test_printed_markers_resolve_all_corner_flips_without_camera_truth(self):
        board = BoardSettings()
        points = board.object_points()[:, :2]
        H = np.array([[900., 80, 250.], [-60, 850., 120.], [.3, -.2, 1.]])
        expected = cv2.perspectiveTransform(points[None], H)[0]
        markers = cv2.perspectiveTransform(board.marker_points()[None].astype(np.float32), H)[0]
        grid = expected.reshape(board.rows, board.columns, 2)
        for candidate in [grid, grid[:, ::-1], grid[::-1], grid[::-1, ::-1]]:
            actual, diagnostic = order_detected_corners(candidate.copy(), markers, board)
            np.testing.assert_allclose(actual.reshape(-1, 2), expected, atol=1e-5)
            self.assertLess(diagnostic['marker_reprojection_rms_px'], 1e-3)

    def test_solver_accepts_detected_pixels_only_no_truth_initialization(self):
        board = BoardSettings()
        K = np.array([[605., 0., 424.], [0., 604., 236.], [0., 0., 1.]])
        images=[]
        for item in board_pose_plan():
            if item['split'] != 'train': continue
            T = T_camera_from_board_plan(item, board)
            pixels=cv2.projectPoints(board.object_points(), cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, None)[0]
            images.append(pixels)
        estimated = calibrate_intrinsics(images, (848, 480), board)
        np.testing.assert_allclose(estimated['K'], K, atol=.002)
        np.testing.assert_array_equal(estimated['distortion'], 0.)
        self.assertEqual(estimated['flags'] & cv2.CALIB_USE_INTRINSIC_GUESS, 0)


if __name__ == '__main__':
    unittest.main()
