from __future__ import annotations

import unittest

import cv2
import numpy as np

from xarm_sdk_control.multiview_calibration import (
    calibrate_relative_camera_pose,
    calibrate_relative_camera_pose_from_observations,
    calibrate_chessboard_intrinsics,
    chessboard_object_points,
    create_charuco_board,
    detect_charuco_corners,
    normalized_board_area_ratio,
)
from xarm_sdk_control.multiview_rig import FixedCameraModel, FixedCameraRig


class MultiViewCalibrationTest(unittest.TestCase):
    def test_charuco_board_is_detected_with_identified_corners(self):
        board = create_charuco_board(9, 9, 0.050, 0.037, "DICT_4X4_50")
        gray = board.generateImage((900, 900), marginSize=20)
        detection = detect_charuco_corners(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), board)

        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual(detection.marker_count, 40)
        self.assertEqual(len(detection.corner_ids), 64)
        np.testing.assert_array_equal(detection.corner_ids, np.arange(64))

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

    def test_pair_pose_supports_different_partial_charuco_corners(self):
        board = create_charuco_board(9, 9, 0.050, 0.037, "DICT_4X4_50")
        points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
        camera_matrix = np.array(
            [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.zeros(5, dtype=np.float64)
        expected = np.eye(4, dtype=np.float64)
        expected_rotation, _ = cv2.Rodrigues(np.array([0.04, -0.10, 0.07]))
        expected[:3, :3] = expected_rotation
        expected[:3, 3] = [0.20, -0.05, 0.09]
        reference_objects = []
        reference_images = []
        target_objects = []
        target_images = []
        reference_ids = np.arange(0, 40)
        target_ids = np.arange(24, 64)
        for index in range(6):
            rvec = np.array([-0.15 + 0.04 * index, 0.08 - 0.02 * index, 0.03])
            rotation, _ = cv2.Rodrigues(rvec)
            reference_from_board = np.eye(4)
            reference_from_board[:3, :3] = rotation
            reference_from_board[:3, 3] = [-0.08 + 0.02 * index, -0.05, 0.85 + 0.02 * index]
            target_from_board = np.linalg.inv(expected) @ reference_from_board
            target_rvec, _ = cv2.Rodrigues(target_from_board[:3, :3])
            reference_image, _ = cv2.projectPoints(
                points[reference_ids], rvec, reference_from_board[:3, 3], camera_matrix, distortion
            )
            target_image, _ = cv2.projectPoints(
                points[target_ids], target_rvec, target_from_board[:3, 3], camera_matrix, distortion
            )
            reference_objects.append(points[reference_ids])
            reference_images.append(reference_image)
            target_objects.append(points[target_ids])
            target_images.append(target_image)

        original_matrix, original_distortion = camera_matrix.copy(), distortion.copy()
        for method in ["pose_average", "stereo"]:
            with self.subTest(method=method):
                result = calibrate_relative_camera_pose_from_observations(
                    reference_objects, reference_images, target_objects, target_images,
                    camera_matrix, distortion, camera_matrix, distortion,
                    method=method, image_size_px=(640, 480),
                )
                np.testing.assert_allclose(result.transform_reference_from_target, expected, atol=1e-5)
                self.assertEqual(result.method, method)
                if method == "stereo":
                    self.assertLess(result.stereo_rms_px, 1e-3)
                np.testing.assert_array_equal(camera_matrix, original_matrix)
                np.testing.assert_array_equal(distortion, original_distortion)
        with self.assertRaisesRegex(ValueError, "shared corners"):
            calibrate_relative_camera_pose_from_observations(
                reference_objects, reference_images,
                [v[16:] for v in target_objects], [v[16:] for v in target_images],
                camera_matrix, distortion, camera_matrix, distortion,
                method="stereo", image_size_px=(640, 480),
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
