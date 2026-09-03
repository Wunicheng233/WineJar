from __future__ import annotations

import unittest

import cv2
import numpy as np

from xarm_sdk_control.multiview_calibration import (
    calibrate_relative_camera_pose,
    calibrate_chessboard_intrinsics,
    chessboard_object_points,
    normalized_board_area_ratio,
)
from xarm_sdk_control.multiview_rig import FixedCameraModel, FixedCameraRig


class MultiViewCalibrationTest(unittest.TestCase):
    def test_object_points_follow_inner_corner_grid(self):
        points = chessboard_object_points(3, 2, 0.025)
        self.assertEqual(points.shape, (6, 3))
        np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(points[1], [0.025, 0.0, 0.0])
        np.testing.assert_allclose(points[-1], [0.05, 0.025, 0.0])

    def test_board_area_ratio_uses_convex_hull(self):
        corners = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        self.assertAlmostEqual(normalized_board_area_ratio(corners, (100, 100)), 0.64)

    def test_intrinsic_calibration_recovers_noise_free_projection(self):
        pattern_cols, pattern_rows = 7, 5
        square_size_m = 0.02
        object_points = chessboard_object_points(pattern_cols, pattern_rows, square_size_m)
        expected_matrix = np.array(
            [[820.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros(5, dtype=np.float64)
        poses = (
            ([-0.20, 0.08, 0.03], [-0.06, -0.04, 0.55]),
            ([-0.08, -0.16, 0.12], [-0.02, 0.03, 0.68]),
            ([0.15, 0.11, -0.10], [0.08, -0.06, 0.61]),
            ([0.09, -0.22, 0.18], [0.01, 0.07, 0.74]),
            ([-0.16, 0.19, -0.08], [-0.09, 0.00, 0.58]),
            ([0.22, 0.04, 0.15], [0.06, 0.02, 0.66]),
        )
        observations = []
        for rvec_values, tvec_values in poses:
            corners, _ = cv2.projectPoints(
                object_points,
                np.asarray(rvec_values, dtype=np.float64),
                np.asarray(tvec_values, dtype=np.float64),
                expected_matrix,
                distortion,
            )
            observations.append(corners)

        result = calibrate_chessboard_intrinsics(
            observations,
            (640, 480),
            pattern_cols,
            pattern_rows,
            square_size_m,
        )

        self.assertLess(result.rms_reprojection_error_px, 1e-3)
        np.testing.assert_allclose(result.camera_matrix, expected_matrix, atol=1e-2)
        self.assertEqual(len(result.per_view_reprojection_error_px), len(observations))

    def test_pair_extrinsic_calibration_recovers_known_transform(self):
        pattern_cols, pattern_rows = 7, 5
        object_points = chessboard_object_points(pattern_cols, pattern_rows, 0.02)
        camera_matrix = np.array(
            [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros(5, dtype=np.float64)
        expected_reference_from_target = np.eye(4, dtype=np.float64)
        expected_rotation, _ = cv2.Rodrigues(np.array([0.03, -0.12, 0.08], dtype=np.float64))
        expected_reference_from_target[:3, :3] = expected_rotation
        expected_reference_from_target[:3, 3] = [0.22, -0.04, 0.08]
        reference_observations = []
        target_observations = []
        board_poses = (
            ([-0.18, 0.08, 0.04], [-0.09, -0.04, 0.64]),
            ([-0.06, -0.14, 0.11], [0.04, 0.00, 0.77]),
            ([0.11, 0.15, -0.06], [-0.02, 0.07, 0.71]),
            ([0.16, -0.08, 0.14], [0.08, -0.05, 0.68]),
            ([-0.12, 0.20, -0.12], [-0.06, 0.02, 0.83]),
            ([0.04, 0.06, 0.18], [0.03, -0.07, 0.73]),
        )
        for reference_rvec_values, reference_tvec_values in board_poses:
            reference_rotation, _ = cv2.Rodrigues(np.asarray(reference_rvec_values, dtype=np.float64))
            reference_from_board = np.eye(4, dtype=np.float64)
            reference_from_board[:3, :3] = reference_rotation
            reference_from_board[:3, 3] = reference_tvec_values
            target_from_board = np.linalg.inv(expected_reference_from_target) @ reference_from_board
            target_rvec, _ = cv2.Rodrigues(target_from_board[:3, :3])
            reference_corners, _ = cv2.projectPoints(
                object_points,
                np.asarray(reference_rvec_values, dtype=np.float64),
                np.asarray(reference_tvec_values, dtype=np.float64),
                camera_matrix,
                distortion,
            )
            target_corners, _ = cv2.projectPoints(
                object_points,
                target_rvec,
                target_from_board[:3, 3],
                camera_matrix,
                distortion,
            )
            reference_observations.append(reference_corners)
            target_observations.append(target_corners)

        result = calibrate_relative_camera_pose(
            reference_observations,
            target_observations,
            object_points,
            camera_matrix,
            distortion,
            camera_matrix,
            distortion,
        )

        self.assertEqual(result.accepted_indices, [1, 2, 3, 4, 5, 6])
        np.testing.assert_allclose(
            result.transform_reference_from_target,
            expected_reference_from_target,
            atol=1e-5,
        )

    def test_rig_transforms_and_projects_points_in_reference_frame(self):
        matrix = np.array(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        reference = FixedCameraModel(
            camera_id="overhead",
            serial="reference",
            camera_matrix=matrix,
            distortion_coefficients=np.zeros(5),
            image_size_px=(640, 480),
            transform_reference_from_camera=np.eye(4),
            intrinsics_path=__file__,
        )
        reference_from_side = np.eye(4)
        reference_from_side[:3, 3] = [0.10, 0.0, 0.0]
        side = FixedCameraModel(
            camera_id="side_a",
            serial="side",
            camera_matrix=matrix,
            distortion_coefficients=np.zeros(5),
            image_size_px=(640, 480),
            transform_reference_from_camera=reference_from_side,
            intrinsics_path=__file__,
        )
        rig = FixedCameraRig("overhead", {"overhead": reference, "side_a": side}, ())
        transformed = rig.transform_points_to_reference("side_a", [[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(transformed, [[0.10, 0.0, 1.0]])
        projected = rig.project_reference_points("overhead", [[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(projected, [[320.0, 240.0]])


if __name__ == "__main__":
    unittest.main()
