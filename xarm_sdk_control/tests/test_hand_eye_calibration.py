import cv2
import numpy as np

from xarm_sdk_control.hand_eye_calibration import (
    calibrate_eye_in_hand,
    flange_transform_from_tcp_pose,
    pose_mm_deg_to_transform,
)
from xarm_sdk_control.multiview_calibration import transform_rotation_angle_deg


def transform(rotation_vector, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))[0]
    result[:3, 3] = translation
    return result


def test_tcp_pose_is_reduced_to_flange_pose() -> None:
    base_from_flange = pose_mm_deg_to_transform([100, 200, 300, 0, 0, 90])
    flange_from_tcp = pose_mm_deg_to_transform([10, 0, 100, 0, 0, 0])
    base_from_tcp = base_from_flange @ flange_from_tcp
    # This fixture has yaw=90 degrees, so the flange-relative +X offset maps to base +Y.
    tcp_pose = [100, 210, 400, 0, 0, 90]
    recovered = flange_transform_from_tcp_pose(tcp_pose, [10, 0, 100, 0, 0, 0])
    assert np.allclose(recovered, base_from_flange, atol=1e-10)
    assert np.allclose(base_from_tcp, pose_mm_deg_to_transform(tcp_pose), atol=1e-10)


def test_eye_in_hand_solver_recovers_known_transform() -> None:
    flange_from_camera = transform([0.08, -0.12, 0.04], [0.035, -0.018, 0.072])
    base_from_board = transform([0.02, 0.01, -0.03], [0.42, -0.08, 0.01])
    robot_poses = [
        transform(rotation, translation)
        for rotation, translation in [
            ([0.00, 0.00, 0.00], [0.20, 0.00, 0.40]),
            ([0.20, 0.05, -0.10], [0.24, -0.04, 0.36]),
            ([-0.15, 0.18, 0.08], [0.18, 0.05, 0.34]),
            ([0.10, -0.22, 0.16], [0.27, 0.03, 0.38]),
            ([-0.24, -0.08, -0.12], [0.22, -0.07, 0.32]),
            ([0.18, 0.20, 0.22], [0.16, 0.02, 0.37]),
            ([-0.10, 0.28, -0.18], [0.25, 0.07, 0.35]),
            ([0.27, -0.16, 0.05], [0.19, -0.02, 0.31]),
            ([-0.20, 0.12, 0.25], [0.28, -0.06, 0.33]),
            ([0.14, -0.26, -0.20], [0.21, 0.06, 0.39]),
        ]
    ]
    camera_poses = [
        np.linalg.inv(flange_from_camera) @ np.linalg.inv(robot_pose) @ base_from_board
        for robot_pose in robot_poses
    ]

    result = calibrate_eye_in_hand(robot_poses, camera_poses)

    translation_error = np.linalg.norm(
        result.transform_flange_from_camera[:3, 3] - flange_from_camera[:3, 3]
    )
    rotation_error = transform_rotation_angle_deg(result.transform_flange_from_camera, flange_from_camera)
    assert translation_error < 1e-6
    assert rotation_error < 1e-5
    assert max(result.board_translation_residuals_m) < 1e-6
